// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep

namespace nb = nanobind;

namespace tunix::trajectory_logger_ext {
namespace {

// Wrapper around a pre-formatted array string whose __str__ and __repr__ both
// return the unquoted bracketed array representation (e.g. "[1, 2, 3]").
// This allows both top-level DataFrame columns and arrays nested inside Python
// dicts/lists/tuples to format identically to Python lists in O(1) without
// allocating PyLongObject/PyFloatObject/PyListObject instances.
class SerializedArray {
 public:
  explicit SerializedArray(std::string repr) : repr_(std::move(repr)) {}

  const std::string& str() const { return repr_; }

  bool operator==(const SerializedArray& other) const {
    return repr_ == other.repr_;
  }

  bool operator==(std::string_view other) const { return repr_ == other; }

 private:
  std::string repr_;
};

struct NdArrayDescriptor {
  nb::object owner;
  nb::ndarray<nb::ro, nb::device::cpu> handle;
  const uint8_t* base_ptr = nullptr;
  nb::dlpack::dtype dtype{};
  std::vector<size_t> shape;
  std::vector<int64_t> strides;
  size_t total_elements = 0;
};

bool IsSupportedNumericOrBoolDtype(nb::dlpack::dtype dt) {
  if (dt.lanes != 1) {
    return false;
  }
  const auto code = static_cast<nb::dlpack::dtype_code>(dt.code);
  switch (code) {
    case nb::dlpack::dtype_code::Int:
    case nb::dlpack::dtype_code::UInt:
      return dt.bits == 8 || dt.bits == 16 || dt.bits == 32 || dt.bits == 64;
    case nb::dlpack::dtype_code::Float:
      return dt.bits == 32 || dt.bits == 64;
    case nb::dlpack::dtype_code::Bool:
      return dt.bits == 8;
    default:
      return false;
  }
}

bool TryExtractDescriptor(nb::handle h, NdArrayDescriptor& desc) {
  nb::object owner;
  nb::ndarray<nb::ro, nb::device::cpu> arr;
  if (!nb::try_cast<nb::ndarray<nb::ro, nb::device::cpu>>(h, arr) ||
      !IsSupportedNumericOrBoolDtype(arr.dtype())) {
    // Convert sub-32-bit floats (e.g. float16 / bfloat16) in C-level NumPy
    // without allocating Python float objects or holding the GIL during string
    // formatting.
    if (!nb::hasattr(h, "dtype")) {
      return false;
    }
    nb::object dt = h.attr("dtype");
    const std::string kind = nb::cast<std::string>(nb::str(dt.attr("kind")));
    const std::string dt_name = nb::cast<std::string>(nb::str(dt.attr("name")));
    if (kind == "f" || dt_name == "bfloat16") {
      owner = h.attr("astype")("float32");
      if (!nb::try_cast<nb::ndarray<nb::ro, nb::device::cpu>>(owner, arr) ||
          !IsSupportedNumericOrBoolDtype(arr.dtype())) {
        return false;
      }
    } else {
      return false;
    }
  }
  const size_t ndim = arr.ndim();
  if (ndim == 0) {
    throw nb::value_error("0-D ndarray should be converted via item().");
  }
  desc.owner = std::move(owner);
  desc.dtype = arr.dtype();
  desc.base_ptr = static_cast<const uint8_t*>(arr.data());
  desc.shape.resize(ndim);
  desc.strides.resize(ndim);
  size_t total = 1;
  for (size_t i = 0; i < ndim; ++i) {
    desc.shape[i] = arr.shape(i);
    total *= desc.shape[i];
  }
  desc.total_elements = total;

  if (arr.stride_ptr() != nullptr) {
    for (size_t i = 0; i < ndim; ++i) {
      desc.strides[i] = arr.stride(i);
    }
  } else {
    int64_t stride = 1;
    for (size_t i = ndim; i-- > 0;) {
      desc.strides[i] = stride;
      stride *= static_cast<int64_t>(desc.shape[i]);
    }
  }
  desc.handle = std::move(arr);
  return true;
}

template <typename IntT>
inline void AppendInt(std::string& out, IntT val) {
  char buf[32];
  auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), val);
  if (ec == std::errc()) {
    out.append(buf, ptr);
  }
}

inline void AppendDouble(std::string& out, double val, bool json_mode) {
  if (std::isnan(val)) {
    out.append(json_mode ? "NaN" : "nan");
    return;
  }
  if (std::isinf(val)) {
    if (std::signbit(val)) {
      out.append(json_mode ? "-Infinity" : "-inf");
    } else {
      out.append(json_mode ? "Infinity" : "inf");
    }
    return;
  }
  if (std::signbit(val)) {
    out.push_back('-');
    val = -val;
  }
  char buf[64];
  auto [ptr, ec] =
      std::to_chars(buf, buf + sizeof(buf), val, std::chars_format::scientific);
  if (ec != std::errc()) {
    return;
  }
  std::string_view sv(buf, static_cast<size_t>(ptr - buf));
  const size_t e_pos = sv.find('e');
  if (e_pos == std::string_view::npos) {
    out.append(sv);
    return;
  }
  int exp = 0;
  std::from_chars(sv.data() + e_pos + 1 + (sv[e_pos + 1] == '+' ? 1 : 0),
                  sv.data() + sv.size(), exp);

  // Python's float.__repr__ uses fixed-point for -4 <= exp < 16 and scientific
  // notation otherwise.
  if (exp < -4 || exp >= 16) {
    out.append(sv);
    return;
  }

  // Extract pure significand digits (strip '.' if present).
  char digits[32];
  size_t num_digits = 0;
  for (size_t i = 0; i < e_pos; ++i) {
    if (sv[i] != '.') {
      digits[num_digits++] = sv[i];
    }
  }

  if (exp < 0) {
    out.append("0.");
    out.append(static_cast<size_t>(-exp - 1), '0');
    out.append(digits, num_digits);
  } else if (static_cast<size_t>(exp) >= num_digits - 1) {
    out.append(digits, num_digits);
    out.append(static_cast<size_t>(exp) - (num_digits - 1), '0');
    out.append(".0");
  } else {
    const size_t split = static_cast<size_t>(exp) + 1;
    out.append(digits, split);
    out.push_back('.');
    out.append(digits + split, num_digits - split);
  }
}

struct BoolTag {};

template <typename ScalarT>
inline void AppendScalarValue(std::string& out, ScalarT val, bool json_mode) {
  if constexpr (std::is_same_v<ScalarT, BoolTag>) {
    (void)val;
    (void)json_mode;
  } else if constexpr (std::is_floating_point_v<ScalarT>) {
    AppendDouble(out, static_cast<double>(val), json_mode);
  } else {
    (void)json_mode;
    AppendInt(out, val);
  }
}

inline void AppendBoolValue(std::string& out, uint8_t raw_byte,
                            bool json_mode) {
  if (json_mode) {
    out.append(raw_byte != 0 ? "true" : "false");
  } else {
    out.append(raw_byte != 0 ? "True" : "False");
  }
}

template <typename ScalarT, bool kIsBool = false>
void FormatNdArrayTyped(std::string& out, const ScalarT* base_ptr,
                        const NdArrayDescriptor& desc, size_t dim,
                        int64_t elem_offset, bool json_mode, int indent,
                        int depth) {
  const size_t extent = desc.shape[dim];
  if (extent == 0) {
    out.append("[]");
    return;
  }

  const int64_t stride = desc.strides[dim];
  const bool is_leaf = (dim + 1 == desc.shape.size());

  if (indent < 0) {
    out.push_back('[');
    if (is_leaf) {
      if (stride == 1) {
        const ScalarT* row = base_ptr + elem_offset;
        for (size_t i = 0; i < extent; ++i) {
          if (i > 0) {
            out.append(", ");
          }
          if constexpr (kIsBool) {
            AppendBoolValue(out, static_cast<uint8_t>(row[i]), json_mode);
          } else {
            AppendScalarValue(out, row[i], json_mode);
          }
        }
      } else {
        for (size_t i = 0; i < extent; ++i) {
          if (i > 0) {
            out.append(", ");
          }
          const ScalarT v =
              base_ptr[elem_offset + static_cast<int64_t>(i) * stride];
          if constexpr (kIsBool) {
            AppendBoolValue(out, static_cast<uint8_t>(v), json_mode);
          } else {
            AppendScalarValue(out, v, json_mode);
          }
        }
      }
    } else {
      for (size_t i = 0; i < extent; ++i) {
        if (i > 0) {
          out.append(", ");
        }
        FormatNdArrayTyped<ScalarT, kIsBool>(
            out, base_ptr, desc, dim + 1,
            elem_offset + static_cast<int64_t>(i) * stride, json_mode, indent,
            depth + 1);
      }
    }
    out.push_back(']');
    return;
  }

  // Pretty-printed mode with indentation.
  out.append("[\n");
  const size_t child_spaces = static_cast<size_t>((depth + 1) * indent);
  if (is_leaf) {
    for (size_t i = 0; i < extent; ++i) {
      out.append(child_spaces, ' ');
      const ScalarT v =
          base_ptr[elem_offset + static_cast<int64_t>(i) * stride];
      if constexpr (kIsBool) {
        AppendBoolValue(out, static_cast<uint8_t>(v), json_mode);
      } else {
        AppendScalarValue(out, v, json_mode);
      }
      if (i + 1 < extent) {
        out.append(",\n");
      } else {
        out.push_back('\n');
      }
    }
  } else {
    for (size_t i = 0; i < extent; ++i) {
      out.append(child_spaces, ' ');
      FormatNdArrayTyped<ScalarT, kIsBool>(
          out, base_ptr, desc, dim + 1,
          elem_offset + static_cast<int64_t>(i) * stride, json_mode, indent,
          depth + 1);
      if (i + 1 < extent) {
        out.append(",\n");
      } else {
        out.push_back('\n');
      }
    }
  }
  out.append(static_cast<size_t>(depth * indent), ' ');
  out.push_back(']');
}

void DispatchFormatNdArray(std::string& out, const NdArrayDescriptor& desc,
                           bool json_mode, int indent, int depth) {
  const auto code = static_cast<nb::dlpack::dtype_code>(desc.dtype.code);
  switch (code) {
    case nb::dlpack::dtype_code::Int:
      switch (desc.dtype.bits) {
        case 8:
          FormatNdArrayTyped(out,
                             reinterpret_cast<const int8_t*>(desc.base_ptr),
                             desc, 0, 0, json_mode, indent, depth);
          return;
        case 16:
          FormatNdArrayTyped(out,
                             reinterpret_cast<const int16_t*>(desc.base_ptr),
                             desc, 0, 0, json_mode, indent, depth);
          return;
        case 32:
          FormatNdArrayTyped(out,
                             reinterpret_cast<const int32_t*>(desc.base_ptr),
                             desc, 0, 0, json_mode, indent, depth);
          return;
        case 64:
          FormatNdArrayTyped(out,
                             reinterpret_cast<const int64_t*>(desc.base_ptr),
                             desc, 0, 0, json_mode, indent, depth);
          return;
      }
      break;
    case nb::dlpack::dtype_code::UInt:
      switch (desc.dtype.bits) {
        case 8:
          FormatNdArrayTyped(out,
                             reinterpret_cast<const uint8_t*>(desc.base_ptr),
                             desc, 0, 0, json_mode, indent, depth);
          return;
        case 16:
          FormatNdArrayTyped(out,
                             reinterpret_cast<const uint16_t*>(desc.base_ptr),
                             desc, 0, 0, json_mode, indent, depth);
          return;
        case 32:
          FormatNdArrayTyped(out,
                             reinterpret_cast<const uint32_t*>(desc.base_ptr),
                             desc, 0, 0, json_mode, indent, depth);
          return;
        case 64:
          FormatNdArrayTyped(out,
                             reinterpret_cast<const uint64_t*>(desc.base_ptr),
                             desc, 0, 0, json_mode, indent, depth);
          return;
      }
      break;
    case nb::dlpack::dtype_code::Float:
      if (desc.dtype.bits == 32) {
        FormatNdArrayTyped(out, reinterpret_cast<const float*>(desc.base_ptr),
                           desc, 0, 0, json_mode, indent, depth);
        return;
      }
      if (desc.dtype.bits == 64) {
        FormatNdArrayTyped(out, reinterpret_cast<const double*>(desc.base_ptr),
                           desc, 0, 0, json_mode, indent, depth);
        return;
      }
      break;
    case nb::dlpack::dtype_code::Bool:
      FormatNdArrayTyped<uint8_t, /*kIsBool=*/true>(
          out, reinterpret_cast<const uint8_t*>(desc.base_ptr), desc, 0, 0,
          json_mode, indent, depth);
      return;
    default:
      break;
  }
}

void AppendHex4(std::string& out, uint16_t val) {
  static constexpr char kHex[] = "0123456789abcdef";
  char buf[6] = {'\\',
                 'u',
                 kHex[(val >> 12) & 0xF],
                 kHex[(val >> 8) & 0xF],
                 kHex[(val >> 4) & 0xF],
                 kHex[val & 0xF]};
  out.append(buf, 6);
}

void AppendJsonEscapedString(std::string& out, std::string_view sv) {
  out.push_back('"');
  const size_t n = sv.size();
  size_t i = 0;
  while (i < n) {
    const unsigned char c = static_cast<unsigned char>(sv[i]);
    if (c == '"') {
      out.append("\\\"");
      ++i;
    } else if (c == '\\') {
      out.append("\\\\");
      ++i;
    } else if (c == '\b') {
      out.append("\\b");
      ++i;
    } else if (c == '\f') {
      out.append("\\f");
      ++i;
    } else if (c == '\n') {
      out.append("\\n");
      ++i;
    } else if (c == '\r') {
      out.append("\\r");
      ++i;
    } else if (c == '\t') {
      out.append("\\t");
      ++i;
    } else if (c < 0x20) {
      AppendHex4(out, static_cast<uint16_t>(c));
      ++i;
    } else if (c < 0x80) {
      out.push_back(static_cast<char>(c));
      ++i;
    } else {
      // Decode UTF-8 sequence and emit \uXXXX (or surrogate pair) to match
      // Python's json.dumps(ensure_ascii=True).
      uint32_t cp = 0;
      size_t len = 1;
      if ((c & 0xE0) == 0xC0 && i + 1 < n) {
        cp = (c & 0x1F) << 6;
        cp |= (static_cast<unsigned char>(sv[i + 1]) & 0x3F);
        len = 2;
      } else if ((c & 0xF0) == 0xE0 && i + 2 < n) {
        cp = (c & 0x0F) << 12;
        cp |= (static_cast<unsigned char>(sv[i + 1]) & 0x3F) << 6;
        cp |= (static_cast<unsigned char>(sv[i + 2]) & 0x3F);
        len = 3;
      } else if ((c & 0xF8) == 0xF0 && i + 3 < n) {
        cp = (c & 0x07) << 18;
        cp |= (static_cast<unsigned char>(sv[i + 1]) & 0x3F) << 12;
        cp |= (static_cast<unsigned char>(sv[i + 2]) & 0x3F) << 6;
        cp |= (static_cast<unsigned char>(sv[i + 3]) & 0x3F);
        len = 4;
      } else {
        cp = c;
        len = 1;
      }
      if (cp <= 0xFFFF) {
        AppendHex4(out, static_cast<uint16_t>(cp));
      } else {
        cp -= 0x10000;
        const uint16_t high =
            static_cast<uint16_t>(0xD800 | ((cp >> 10) & 0x3FF));
        const uint16_t low = static_cast<uint16_t>(0xDC00 | (cp & 0x3FF));
        AppendHex4(out, high);
        AppendHex4(out, low);
      }
      i += len;
    }
  }
  out.push_back('"');
}

struct JsonNode {
  enum class Kind {
    kNull,
    kBool,
    kInt64,
    kUInt64,
    kDouble,
    kString,
    kRawJson,
    kNdArray,
    kList,
    kObject,
  };

  Kind kind = Kind::kNull;
  bool bool_val = false;
  int64_t int_val = 0;
  uint64_t uint_val = 0;
  double double_val = 0.0;
  std::string str_val;
  size_t ndarray_idx = 0;
  std::vector<JsonNode> list_val;
  std::vector<std::pair<std::string, JsonNode>> object_val;
};

struct JsonBuildContext {
  std::vector<NdArrayDescriptor> ndarrays;
  size_t estimated_bytes = 64;
  nb::object np_ndarray_type;
  nb::object np_integer_type;
  nb::object np_floating_type;
  nb::object np_bool_type;
  nb::object np_str_type;
  nb::object is_dataclass_fn;
  nb::object asdict_fn;
  nb::object proto_message_type;
  nb::object message_to_dict_fn;
};

constexpr int kMaxJsonDepth = 1000;

JsonNode BuildJsonTree(nb::handle h, JsonBuildContext& ctx, int depth = 0) {
  if (depth > kMaxJsonDepth) {
    throw nb::value_error("Maximum JSON nesting depth exceeded in dumps_json.");
  }
  JsonNode node;
  if (h.is_none()) {
    node.kind = JsonNode::Kind::kNull;
    ctx.estimated_bytes += 4;
    return node;
  }

  if (nb::isinstance<nb::bool_>(h) || nb::isinstance(h, ctx.np_bool_type)) {
    node.kind = JsonNode::Kind::kBool;
    node.bool_val = nb::cast<bool>(nb::bool_(h));
    ctx.estimated_bytes += 5;
    return node;
  }

  if (nb::isinstance<nb::int_>(h) || nb::isinstance(h, ctx.np_integer_type)) {
    nb::int_ py_int(h);
    int64_t v = 0;
    if (nb::try_cast<int64_t>(py_int, v)) {
      node.kind = JsonNode::Kind::kInt64;
      node.int_val = v;
      ctx.estimated_bytes += 12;
      return node;
    }
    uint64_t uv = 0;
    if (nb::try_cast<uint64_t>(py_int, uv)) {
      node.kind = JsonNode::Kind::kUInt64;
      node.uint_val = uv;
      ctx.estimated_bytes += 12;
      return node;
    }
    node.kind = JsonNode::Kind::kRawJson;
    node.str_val = nb::cast<std::string>(nb::str(py_int));
    ctx.estimated_bytes += node.str_val.size();
    return node;
  }

  if (nb::isinstance<nb::float_>(h) ||
      nb::isinstance(h, ctx.np_floating_type)) {
    node.kind = JsonNode::Kind::kDouble;
    node.double_val = nb::cast<double>(nb::float_(h));
    ctx.estimated_bytes += 16;
    return node;
  }

  if (nb::isinstance<nb::str>(h) || nb::isinstance(h, ctx.np_str_type)) {
    node.kind = JsonNode::Kind::kString;
    node.str_val = nb::cast<std::string>(nb::str(h));
    ctx.estimated_bytes += node.str_val.size() + 4;
    return node;
  }

  if (nb::isinstance<SerializedArray>(h)) {
    node.kind = JsonNode::Kind::kRawJson;
    node.str_val = nb::cast<const SerializedArray&>(h).str();
    ctx.estimated_bytes += node.str_val.size();
    return node;
  }

  if (nb::isinstance(h, ctx.np_ndarray_type)) {
    const int ndim = nb::cast<int>(h.attr("ndim"));
    if (ndim == 0) {
      return BuildJsonTree(h.attr("item")(), ctx, depth + 1);
    }
    NdArrayDescriptor desc;
    if (TryExtractDescriptor(h, desc)) {
      ctx.estimated_bytes += desc.total_elements * 6 + 16;
      node.kind = JsonNode::Kind::kNdArray;
      node.ndarray_idx = ctx.ndarrays.size();
      ctx.ndarrays.push_back(std::move(desc));
      return node;
    }
    return BuildJsonTree(h.attr("tolist")(), ctx, depth + 1);
  }

  if (nb::isinstance<nb::dict>(h)) {
    node.kind = JsonNode::Kind::kObject;
    nb::dict d = nb::borrow<nb::dict>(h);
    node.object_val.reserve(d.size());
    for (auto [k, v] : d) {
      std::string key_str = nb::cast<std::string>(nb::str(k));
      ctx.estimated_bytes += key_str.size() + 6;
      node.object_val.emplace_back(std::move(key_str),
                                   BuildJsonTree(v, ctx, depth + 1));
    }
    return node;
  }

  if (nb::isinstance<nb::list>(h) || nb::isinstance<nb::tuple>(h)) {
    node.kind = JsonNode::Kind::kList;
    nb::sequence seq = nb::borrow<nb::sequence>(h);
    const size_t len = nb::len(seq);
    node.list_val.reserve(len);
    for (size_t i = 0; i < len; ++i) {
      node.list_val.push_back(BuildJsonTree(seq[i], ctx, depth + 1));
    }
    return node;
  }

  if (!nb::isinstance<nb::type_object>(h) &&
      nb::cast<bool>(ctx.is_dataclass_fn(h))) {
    return BuildJsonTree(ctx.asdict_fn(h), ctx, depth + 1);
  }

  if (nb::isinstance(h, ctx.proto_message_type)) {
    return BuildJsonTree(ctx.message_to_dict_fn(h), ctx, depth + 1);
  }

  node.kind = JsonNode::Kind::kString;
  node.str_val = nb::cast<std::string>(nb::str(h));
  ctx.estimated_bytes += node.str_val.size() + 4;
  return node;
}

void SerializeJsonNode(std::string& out, const JsonNode& node,
                       const std::vector<NdArrayDescriptor>& ndarrays,
                       int indent, int depth) {
  switch (node.kind) {
    case JsonNode::Kind::kNull:
      out.append("null");
      return;
    case JsonNode::Kind::kBool:
      out.append(node.bool_val ? "true" : "false");
      return;
    case JsonNode::Kind::kInt64:
      AppendInt(out, node.int_val);
      return;
    case JsonNode::Kind::kUInt64:
      AppendInt(out, node.uint_val);
      return;
    case JsonNode::Kind::kDouble:
      AppendDouble(out, node.double_val, /*json_mode=*/true);
      return;
    case JsonNode::Kind::kString:
      AppendJsonEscapedString(out, node.str_val);
      return;
    case JsonNode::Kind::kRawJson:
      out.append(node.str_val);
      return;
    case JsonNode::Kind::kNdArray:
      DispatchFormatNdArray(out, ndarrays[node.ndarray_idx], /*json_mode=*/true,
                            indent, depth);
      return;
    case JsonNode::Kind::kList: {
      if (node.list_val.empty()) {
        out.append("[]");
        return;
      }
      if (indent < 0) {
        out.push_back('[');
        for (size_t i = 0; i < node.list_val.size(); ++i) {
          if (i > 0) {
            out.append(", ");
          }
          SerializeJsonNode(out, node.list_val[i], ndarrays, indent, depth + 1);
        }
        out.push_back(']');
        return;
      }
      out.append("[\n");
      const size_t child_spaces = static_cast<size_t>((depth + 1) * indent);
      for (size_t i = 0; i < node.list_val.size(); ++i) {
        out.append(child_spaces, ' ');
        SerializeJsonNode(out, node.list_val[i], ndarrays, indent, depth + 1);
        if (i + 1 < node.list_val.size()) {
          out.append(",\n");
        } else {
          out.push_back('\n');
        }
      }
      out.append(static_cast<size_t>(depth * indent), ' ');
      out.push_back(']');
      return;
    }
    case JsonNode::Kind::kObject: {
      if (node.object_val.empty()) {
        out.append("{}");
        return;
      }
      if (indent < 0) {
        out.push_back('{');
        for (size_t i = 0; i < node.object_val.size(); ++i) {
          if (i > 0) {
            out.append(", ");
          }
          AppendJsonEscapedString(out, node.object_val[i].first);
          out.append(": ");
          SerializeJsonNode(out, node.object_val[i].second, ndarrays, indent,
                            depth + 1);
        }
        out.push_back('}');
        return;
      }
      out.append("{\n");
      const size_t child_spaces = static_cast<size_t>((depth + 1) * indent);
      for (size_t i = 0; i < node.object_val.size(); ++i) {
        out.append(child_spaces, ' ');
        AppendJsonEscapedString(out, node.object_val[i].first);
        out.append(": ");
        SerializeJsonNode(out, node.object_val[i].second, ndarrays, indent,
                          depth + 1);
        if (i + 1 < node.object_val.size()) {
          out.append(",\n");
        } else {
          out.push_back('\n');
        }
      }
      out.append(static_cast<size_t>(depth * indent), ' ');
      out.push_back('}');
      return;
    }
  }
}

SerializedArray FormatNdArray(nb::handle arr) {
  NdArrayDescriptor desc;
  if (!TryExtractDescriptor(arr, desc)) {
    throw nb::type_error("Unsupported ndarray dtype for C++ fast formatting.");
  }
  std::string out;
  {
    nb::gil_scoped_release release;
    out.reserve(desc.total_elements * 6 + 16);
    DispatchFormatNdArray(out, desc, /*json_mode=*/false, /*indent=*/-1,
                          /*depth=*/0);
  }
  return SerializedArray(std::move(out));
}

std::string DumpsJson(nb::handle obj, int indent) {
  nb::module_ np_mod = nb::module_::import_("numpy");
  nb::module_ dc_mod = nb::module_::import_("dataclasses");
  nb::module_ proto_msg_mod = nb::module_::import_("google.protobuf.message");
  nb::module_ proto_json_mod =
      nb::module_::import_("google.protobuf.json_format");

  JsonBuildContext ctx;
  ctx.np_ndarray_type = np_mod.attr("ndarray");
  ctx.np_integer_type = np_mod.attr("integer");
  ctx.np_floating_type = np_mod.attr("floating");
  ctx.np_bool_type = np_mod.attr("bool_");
  ctx.np_str_type = np_mod.attr("str_");
  ctx.is_dataclass_fn = dc_mod.attr("is_dataclass");
  ctx.asdict_fn = dc_mod.attr("asdict");
  ctx.proto_message_type = proto_msg_mod.attr("Message");
  ctx.message_to_dict_fn = proto_json_mod.attr("MessageToDict");

  JsonNode root = BuildJsonTree(obj, ctx);

  std::string out;
  {
    nb::gil_scoped_release release;
    out.reserve(ctx.estimated_bytes);
    SerializeJsonNode(out, root, ctx.ndarrays, indent, /*depth=*/0);
  }
  return out;
}

}  // namespace

NB_MODULE(_trajectory_logger_ext, m) {
  m.doc() =
      "C++ nanobind GIL-free ndarray and JSON serializer for "
      "trajectory_logger.";

  nb::class_<SerializedArray>(m, "SerializedArray")
      .def(nb::init<std::string>())
      .def("__str__", &SerializedArray::str)
      .def("__repr__", &SerializedArray::str)
      .def("__eq__", [](const SerializedArray& self, nb::handle other) {
        if (nb::isinstance<SerializedArray>(other)) {
          return self == nb::cast<const SerializedArray&>(other);
        }
        if (nb::isinstance<nb::str>(other)) {
          return self == nb::cast<std::string_view>(other);
        }
        return false;
      });

  m.def("format_ndarray", &FormatNdArray, nb::arg("arr"),
        "Formats a numeric or boolean CPU ndarray into a bracketed string "
        "representation with the Python GIL released.");

  m.def("dumps_json", &DumpsJson, nb::arg("obj").none(), nb::arg("indent") = -1,
        "Serializes a trajectory object (dicts, lists, dataclasses, ndarrays) "
        "to a JSON string with the Python GIL released during formatting.");
}

}  // namespace tunix::trajectory_logger_ext
