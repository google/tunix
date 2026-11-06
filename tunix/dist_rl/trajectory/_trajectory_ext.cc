// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep

namespace nb = nanobind;

namespace tunix::trajectory_ext {
namespace {

// RAII wrapper around `Py_buffer` that releases the buffer with the GIL held.
class ScopedPyBuffer {
 public:
  ScopedPyBuffer(PyObject* obj, int flags) {
    if (PyObject_GetBuffer(obj, &view_, flags) != 0) {
      throw nb::python_error();
    }
    valid_ = true;
  }
  ScopedPyBuffer(const ScopedPyBuffer&) = delete;
  ScopedPyBuffer& operator=(const ScopedPyBuffer&) = delete;
  ~ScopedPyBuffer() {
    if (valid_) {
      PyBuffer_Release(&view_);
    }
  }

  const Py_buffer& view() const noexcept { return view_; }

 private:
  Py_buffer view_{};
  bool valid_ = false;
};

// Uses `std::malloc` and `std::free` so buffer management is safe inside
// `nb::gil_scoped_release` across all CPython build configurations.
inline void* RawAllocBytes(size_t nbytes) noexcept {
  return std::malloc(nbytes > 0 ? nbytes : 1);
}

inline void RawFreeBytes(void* ptr) noexcept { std::free(ptr); }

struct RawDeleter {
  void operator()(void* ptr) const noexcept { RawFreeBytes(ptr); }
};

using RawByteBufferPtr = std::unique_ptr<uint8_t[], RawDeleter>;

RawByteBufferPtr AllocRawBytesNoGil(size_t nbytes) noexcept {
  return RawByteBufferPtr(static_cast<uint8_t*>(RawAllocBytes(nbytes)));
}

// Extracts a UTF-8 view preserving exact byte length across embedded NUL bytes.
// `PyUnicode_AsUTF8AndSize` is part of the Python 3.10+ Stable ABI.
std::string_view Utf8StringViewOrThrow(nb::str str_obj) {
  Py_ssize_t py_len = 0;
  const char* data = PyUnicode_AsUTF8AndSize(str_obj.ptr(), &py_len);
  if (data == nullptr) {
    throw nb::python_error();
  }
  return std::string_view(data, static_cast<size_t>(py_len));
}

// =============================================================================
// 1. GIL-Free Base64 Encoding & Decoding Directly to/from NumPy Buffers
// =============================================================================

constexpr char kBase64Alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

constexpr std::array<int8_t, 256> BuildBase64DecodeTable() {
  std::array<int8_t, 256> table{};
  for (int i = 0; i < 256; ++i) {
    table[i] = -1;
  }
  for (int i = 0; i < 64; ++i) {
    table[static_cast<uint8_t>(kBase64Alphabet[i])] = static_cast<int8_t>(i);
  }
  table[static_cast<uint8_t>('=')] = -2;
  return table;
}

constexpr std::array<int8_t, 256> kBase64DecodeTable = BuildBase64DecodeTable();

struct DtypeSpec {
  std::string_view name;
  nb::dlpack::dtype dtype;
  size_t itemsize;
};

constexpr std::array<DtypeSpec, 11> kSupportedDtypes = {{
    {"int8", {static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 8, 1}, 1},
    {"int16", {static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 16, 1}, 2},
    {"int32", {static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 32, 1}, 4},
    {"int64", {static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 64, 1}, 8},
    {"uint8", {static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 8, 1}, 1},
    {"uint16", {static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 16, 1}, 2},
    {"uint32", {static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 32, 1}, 4},
    {"uint64", {static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 64, 1}, 8},
    {"float16",
     {static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 16, 1},
     2},
    {"float32",
     {static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 32, 1},
     4},
    {"float64",
     {static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 64, 1},
     8},
}};

const DtypeSpec* FindDtypeSpec(std::string_view dtype_name) noexcept {
  for (const DtypeSpec& spec : kSupportedDtypes) {
    if (spec.name == dtype_name) {
      return &spec;
    }
  }
  return nullptr;
}

std::string EncodeBase64Bytes(nb::handle raw_obj) {
  ScopedPyBuffer py_buf(raw_obj.ptr(), PyBUF_SIMPLE);
  const Py_buffer& view = py_buf.view();
  const size_t len = static_cast<size_t>(view.len);
  if (len == 0) {
    return std::string();
  }
  if (len > (std::numeric_limits<size_t>::max() / 4) * 3 - 2) {
    throw std::bad_alloc();
  }
  const auto* src = static_cast<const uint8_t*>(view.buf);
  const size_t out_len = 4 * ((len + 2) / 3);
  std::string out;

  {
    nb::gil_scoped_release release;
    out.resize(out_len);
    char* dst = out.data();
    size_t i = 0;
    while (i + 3 <= len) {
      const uint32_t triple = (static_cast<uint32_t>(src[i]) << 16) |
                              (static_cast<uint32_t>(src[i + 1]) << 8) |
                              static_cast<uint32_t>(src[i + 2]);
      dst[0] = kBase64Alphabet[(triple >> 18) & 0x3FU];
      dst[1] = kBase64Alphabet[(triple >> 12) & 0x3FU];
      dst[2] = kBase64Alphabet[(triple >> 6) & 0x3FU];
      dst[3] = kBase64Alphabet[triple & 0x3FU];
      dst += 4;
      i += 3;
    }
    const size_t rem = len - i;
    if (rem == 1) {
      const uint32_t triple = static_cast<uint32_t>(src[i]) << 16;
      dst[0] = kBase64Alphabet[(triple >> 18) & 0x3FU];
      dst[1] = kBase64Alphabet[(triple >> 12) & 0x3FU];
      dst[2] = '=';
      dst[3] = '=';
    } else if (rem == 2) {
      const uint32_t triple = (static_cast<uint32_t>(src[i]) << 16) |
                              (static_cast<uint32_t>(src[i + 1]) << 8);
      dst[0] = kBase64Alphabet[(triple >> 18) & 0x3FU];
      dst[1] = kBase64Alphabet[(triple >> 12) & 0x3FU];
      dst[2] = kBase64Alphabet[(triple >> 6) & 0x3FU];
      dst[3] = '=';
    }
  }

  return out;
}

bool ValidateBase64AndMeasureNoGil(std::string_view b64, size_t* decoded_bytes,
                                   size_t* pad_out) noexcept {
  if (b64.empty()) {
    *decoded_bytes = 0;
    *pad_out = 0;
    return true;
  }
  if ((b64.size() & 3U) != 0) {
    return false;
  }
  const size_t num_quads = b64.size() >> 2;
  size_t pad = 0;
  if (b64.back() == '=') {
    ++pad;
    if (b64[b64.size() - 2] == '=') {
      ++pad;
    }
  }
  const auto* src = reinterpret_cast<const uint8_t*>(b64.data());
  const size_t full_quads = pad > 0 ? (num_quads - 1) : num_quads;

  for (size_t q = 0; q < full_quads; ++q) {
    const int8_t v0 = kBase64DecodeTable[src[4 * q + 0]];
    const int8_t v1 = kBase64DecodeTable[src[4 * q + 1]];
    const int8_t v2 = kBase64DecodeTable[src[4 * q + 2]];
    const int8_t v3 = kBase64DecodeTable[src[4 * q + 3]];
    if ((v0 | v1 | v2 | v3) < 0) {
      return false;
    }
  }
  if (pad == 1) {
    const size_t base = 4 * full_quads;
    if ((kBase64DecodeTable[src[base + 0]] | kBase64DecodeTable[src[base + 1]] |
         kBase64DecodeTable[src[base + 2]]) < 0 ||
        src[base + 3] != '=') {
      return false;
    }
  } else if (pad == 2) {
    const size_t base = 4 * full_quads;
    if ((kBase64DecodeTable[src[base + 0]] |
         kBase64DecodeTable[src[base + 1]]) < 0 ||
        src[base + 2] != '=' || src[base + 3] != '=') {
      return false;
    }
  }

  *decoded_bytes = num_quads * 3 - pad;
  *pad_out = pad;
  return true;
}

void DecodeValidatedBase64NoGil(std::string_view b64, size_t pad,
                                uint8_t* dst) noexcept {
  const size_t num_quads = b64.size() >> 2;
  const size_t full_quads = pad > 0 ? (num_quads - 1) : num_quads;
  const auto* src = reinterpret_cast<const uint8_t*>(b64.data());
  size_t out_idx = 0;

  for (size_t q = 0; q < full_quads; ++q) {
    const uint32_t triple =
        (static_cast<uint32_t>(kBase64DecodeTable[src[0]]) << 18) |
        (static_cast<uint32_t>(kBase64DecodeTable[src[1]]) << 12) |
        (static_cast<uint32_t>(kBase64DecodeTable[src[2]]) << 6) |
        static_cast<uint32_t>(kBase64DecodeTable[src[3]]);
    dst[out_idx + 0] = static_cast<uint8_t>((triple >> 16) & 0xFFU);
    dst[out_idx + 1] = static_cast<uint8_t>((triple >> 8) & 0xFFU);
    dst[out_idx + 2] = static_cast<uint8_t>(triple & 0xFFU);
    src += 4;
    out_idx += 3;
  }

  if (pad == 1) {
    const uint32_t triple =
        (static_cast<uint32_t>(kBase64DecodeTable[src[0]]) << 18) |
        (static_cast<uint32_t>(kBase64DecodeTable[src[1]]) << 12) |
        (static_cast<uint32_t>(kBase64DecodeTable[src[2]]) << 6);
    dst[out_idx + 0] = static_cast<uint8_t>((triple >> 16) & 0xFFU);
    dst[out_idx + 1] = static_cast<uint8_t>((triple >> 8) & 0xFFU);
  } else if (pad == 2) {
    const uint32_t triple =
        (static_cast<uint32_t>(kBase64DecodeTable[src[0]]) << 18) |
        (static_cast<uint32_t>(kBase64DecodeTable[src[1]]) << 12);
    dst[out_idx + 0] = static_cast<uint8_t>((triple >> 16) & 0xFFU);
  }
}

enum class Base64DecodeStatus {
  kOk = 0,
  kInvalidEncoding = 1,
  kLengthMismatch = 2,
  kDimensionExceeded = 3,
  kOutOfMemory = 4,
};

nb::object DecodeBase64Array(nb::str b64_obj, std::string_view dtype_name,
                             nb::sequence shape_seq,
                             std::string_view type_label = "array") {
  const DtypeSpec* spec = FindDtypeSpec(dtype_name);
  if (spec == nullptr) {
    std::string msg = "Unsupported ";
    msg.append(type_label);
    msg.append(" dtype for Base64 decoding: ");
    msg.append(dtype_name);
    msg.push_back('.');
    throw nb::value_error(msg.c_str());
  }

  std::vector<size_t> dims;
  dims.reserve(nb::len(shape_seq));
  size_t num_elements = 1;
  bool has_zero_dim = false;
  bool dim_exceeds_ssize_max = false;
  bool prod_overflow = false;
  for (nb::handle item : shape_seq) {
    if (!PyLong_Check(item.ptr()) || PyBool_Check(item.ptr())) {
      std::string msg = "Invalid ";
      msg.append(type_label);
      msg.append(" shape: dimensions must be non-negative integers.");
      throw nb::value_error(msg.c_str());
    }
    int overflow = 0;
    const int64_t dim_i64 = static_cast<int64_t>(
        PyLong_AsLongLongAndOverflow(item.ptr(), &overflow));
    if (dim_i64 == -1 && PyErr_Occurred() != nullptr) {
      throw nb::python_error();
    }
    if (overflow < 0 || (overflow == 0 && dim_i64 < 0)) {
      std::string msg = "Invalid ";
      msg.append(type_label);
      msg.append(" shape: dimensions must be non-negative integers.");
      throw nb::value_error(msg.c_str());
    }
    if (overflow > 0 ||
        static_cast<uint64_t>(dim_i64) >
            static_cast<uint64_t>(std::numeric_limits<Py_ssize_t>::max())) {
      dim_exceeds_ssize_max = true;
      dims.push_back(0);
      continue;
    }
    const size_t dim = static_cast<size_t>(dim_i64);
    dims.push_back(dim);
    if (dim == 0) {
      has_zero_dim = true;
    } else if (!prod_overflow) {
      if (dim > std::numeric_limits<size_t>::max() / num_elements) {
        prod_overflow = true;
      } else {
        num_elements *= dim;
      }
    }
  }
  if (dims.empty()) {
    std::string msg = "Invalid ";
    msg.append(type_label);
    msg.append(" shape: expected at least 1 dimension.");
    throw nb::value_error(msg.c_str());
  }
  if (has_zero_dim) {
    num_elements = 0;
    prod_overflow = false;
  }

  const std::string_view b64 = Utf8StringViewOrThrow(b64_obj);
  const bool byte_overflow =
      (!has_zero_dim && dim_exceeds_ssize_max) || prod_overflow ||
      (num_elements > std::numeric_limits<size_t>::max() / spec->itemsize);
  const size_t expected_bytes =
      byte_overflow ? 0 : (num_elements * spec->itemsize);

  Base64DecodeStatus status = Base64DecodeStatus::kOk;
  size_t decoded_bytes = 0;
  RawByteBufferPtr out;
  {
    nb::gil_scoped_release release;
    size_t pad = 0;
    if (!ValidateBase64AndMeasureNoGil(b64, &decoded_bytes, &pad)) {
      status = Base64DecodeStatus::kInvalidEncoding;
    } else if (byte_overflow || decoded_bytes != expected_bytes) {
      status = Base64DecodeStatus::kLengthMismatch;
    } else if (dim_exceeds_ssize_max) {
      status = Base64DecodeStatus::kDimensionExceeded;
    } else {
      out = AllocRawBytesNoGil(expected_bytes);
      if (out == nullptr) {
        status = Base64DecodeStatus::kOutOfMemory;
      } else if (decoded_bytes > 0) {
        DecodeValidatedBase64NoGil(b64, pad, out.get());
      }
    }
  }

  if (status == Base64DecodeStatus::kInvalidEncoding) {
    std::string msg = "Failed to decode ";
    msg.append(type_label);
    msg.append(" blob: invalid Base64 encoding.");
    throw nb::value_error(msg.c_str());
  }
  if (status == Base64DecodeStatus::kLengthMismatch) {
    std::string msg = "Decoded ";
    msg.append(type_label);
    msg.append(" byte length ");
    msg.append(std::to_string(decoded_bytes));
    msg.append(" does not match shape");
    if (!byte_overflow) {
      msg.append(" (expected ");
      msg.append(std::to_string(expected_bytes));
      msg.append(" bytes)");
    }
    msg.push_back('.');
    throw nb::value_error(msg.c_str());
  }
  if (status == Base64DecodeStatus::kDimensionExceeded) {
    throw nb::value_error("Maximum allowed dimension exceeded.");
  }
  if (status == Base64DecodeStatus::kOutOfMemory) {
    throw std::bad_alloc();
  }

  uint8_t* raw = out.release();
  nb::capsule deleter(raw, [](void* ptr) noexcept { RawFreeBytes(ptr); });
  return nb::ndarray<nb::numpy>(raw, dims.size(), dims.data(), deleter, nullptr,
                                spec->dtype)
      .cast();
}

// =============================================================================
// 2. GIL-Free JSON NUL-Byte Sanitization (`\x00` and `\u0000`)
// =============================================================================

nb::object StripJsonNulBytes(nb::str json_obj) {
  const std::string_view sv = Utf8StringViewOrThrow(json_obj);

  bool needs_sanitize = false;
  std::string cleaned;
  {
    nb::gil_scoped_release release;
    constexpr std::string_view kEscapedNul = "\\u0000";
    const bool has_raw_nul =
        (std::memchr(sv.data(), '\0', sv.size()) != nullptr);
    const bool has_esc_nul = (sv.find(kEscapedNul) != std::string_view::npos);
    if (has_raw_nul || has_esc_nul) {
      cleaned.reserve(sv.size());
      size_t i = 0;
      bool odd_backslashes = false;
      while (i < sv.size()) {
        const char ch = sv[i];
        if (ch == '\0') {
          odd_backslashes = false;
          ++i;
          continue;
        }
        if (!odd_backslashes && i + kEscapedNul.size() <= sv.size() &&
            std::memcmp(sv.data() + i, kEscapedNul.data(),
                        kEscapedNul.size()) == 0) {
          i += kEscapedNul.size();
          continue;
        }
        cleaned.push_back(ch);
        odd_backslashes = (ch == '\\') ? !odd_backslashes : false;
        ++i;
      }
      needs_sanitize = (cleaned.size() != sv.size());
    }
  }

  if (!needs_sanitize) {
    return json_obj;
  }
  return nb::cast(cleaned);
}

}  // namespace

NB_MODULE(_trajectory_ext, m) {
  m.doc() =
      "C++17 nanobind extension for GIL-free Trajectory Store Base64 codecs "
      "and JSON NUL sanitization.";
  m.def("encode_base64_bytes", &EncodeBase64Bytes, nb::arg("raw"),
        "Encodes a contiguous buffer-protocol object into an ASCII Base64 "
        "string with the GIL released.");
  m.def("decode_base64_array", &DecodeBase64Array, nb::arg("b64"),
        nb::arg("dtype_name"), nb::arg("shape"),
        nb::arg("type_label") = "array",
        "Decodes an ASCII Base64 string directly into a NumPy array with the "
        "GIL released.");
  m.def("strip_json_nul_bytes", &StripJsonNulBytes, nb::arg("json_str"),
        "Strips raw NUL bytes and unescaped \\u0000 sequences from a JSON "
        "string with the GIL released, returning the original str if none "
        "exist.");
}

}  // namespace tunix::trajectory_ext
