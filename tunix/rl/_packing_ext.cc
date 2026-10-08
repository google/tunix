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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/pair.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep

namespace nb = nanobind;

namespace tunix::rl {
namespace {

struct SegmentView {
  const int32_t* prompt_ptr = nullptr;
  int64_t p_len = 0;
  const int32_t* comp_ptr = nullptr;
  int64_t c_len = 0;
  const float* mask_ptr = nullptr;
  const float* adv_ptr = nullptr;
  std::vector<const float*> per_token_ptrs;
  const int16_t* routed_ptr = nullptr;
  int64_t routed_tokens = 0;
  int64_t routed_layers = 0;
  int64_t routed_k = 0;
};

struct RawArrayView {
  const void* ptr = nullptr;
  int64_t size = 0;
};

// Holds active `Py_buffer` exports across `nb::gil_scoped_release` blocks and
// releases them when destroyed (with the GIL held).
class ScopedPyBufferList {
 public:
  ScopedPyBufferList() = default;
  ScopedPyBufferList(const ScopedPyBufferList&) = delete;
  ScopedPyBufferList& operator=(const ScopedPyBufferList&) = delete;
  ~ScopedPyBufferList() {
    for (Py_buffer& view : views_) {
      PyBuffer_Release(&view);
    }
  }

  const Py_buffer& Acquire(PyObject* obj, int flags) {
    Py_buffer view{};
    if (PyObject_GetBuffer(obj, &view, flags) != 0) {
      throw nb::python_error();
    }
    try {
      views_.push_back(view);
    } catch (...) {
      PyBuffer_Release(&view);
      throw;
    }
    return views_.back();
  }

 private:
  std::vector<Py_buffer> views_;
};

inline RawArrayView Extract1DArrayPtr(PyObject* obj, size_t elem_size,
                                      ScopedPyBufferList* buffers) {
  const Py_buffer& view = buffers->Acquire(obj, PyBUF_SIMPLE);
  const int64_t num_elems =
      static_cast<int64_t>(static_cast<size_t>(view.len) / elem_size);
  return RawArrayView{view.buf, num_elems};
}

inline void ExtractRoutedView(PyObject* obj, SegmentView* out,
                              ScopedPyBufferList* buffers) {
  const Py_buffer& view = buffers->Acquire(obj, PyBUF_ND);
  if (view.ndim < 3 || view.shape == nullptr) {
    throw std::invalid_argument(
        "routed_experts array must have at least 3 dimensions "
        "[tokens, layers, k].");
  }
  out->routed_ptr = static_cast<const int16_t*>(view.buf);
  out->routed_tokens = static_cast<int64_t>(view.shape[0]);
  out->routed_layers = static_cast<int64_t>(view.shape[1]);
  out->routed_k = static_cast<int64_t>(view.shape[2]);
}

inline int64_t AlignOffset(int64_t offset, int64_t boundary) {
  if (boundary <= 1 || offset <= 0) {
    return offset;
  }
  const int64_t rem = offset % boundary;
  return rem == 0 ? offset : (offset + boundary - rem);
}

inline void* RawAllocBytes(size_t nbytes) noexcept {
#if defined(Py_LIMITED_API)
  return std::malloc(nbytes);
#else
  return PyMem_RawMalloc(nbytes);
#endif
}

inline void RawFreeBytes(void* p) noexcept {
#if defined(Py_LIMITED_API)
  std::free(p);
#else
  PyMem_RawFree(p);
#endif
}

struct PyRawDeleter {
  void operator()(void* p) const noexcept { RawFreeBytes(p); }
};

template <typename T>
using RawBufferPtr = std::unique_ptr<T[], PyRawDeleter>;

template <typename T>
RawBufferPtr<T> AllocRawBuffer(size_t count) {
  const size_t nbytes = (count > 0 ? count : 1) * sizeof(T);
  void* ptr = RawAllocBytes(nbytes);
  if (ptr == nullptr) {
    throw std::bad_alloc();
  }
  return RawBufferPtr<T>(static_cast<T*>(ptr));
}

template <typename T>
nb::object Wrap1DArray(RawBufferPtr<T> buf, size_t d0) {
  T* raw = buf.release();
  nb::capsule deleter(raw, [](void* p) noexcept { RawFreeBytes(p); });
  size_t shape[1] = {d0};
  return nb::ndarray<nb::numpy, T, nb::ndim<1>>(raw, 1, shape, deleter).cast();
}

template <typename T>
nb::object Wrap2DArray(RawBufferPtr<T> buf, size_t d0, size_t d1) {
  T* raw = buf.release();
  nb::capsule deleter(raw, [](void* p) noexcept { RawFreeBytes(p); });
  size_t shape[2] = {d0, d1};
  return nb::ndarray<nb::numpy, T, nb::ndim<2>>(raw, 2, shape, deleter).cast();
}

template <typename T>
nb::object Wrap4DArray(RawBufferPtr<T> buf, size_t d0, size_t d1, size_t d2,
                       size_t d3) {
  T* raw = buf.release();
  nb::capsule deleter(raw, [](void* p) noexcept { RawFreeBytes(p); });
  size_t shape[4] = {d0, d1, d2, d3};
  return nb::ndarray<nb::numpy, T, nb::ndim<4>>(raw, 4, shape, deleter).cast();
}

struct RawChunkBuffers {
  size_t n_bins = 0;
  size_t budget = 0;
  int64_t routed_layers = 0;
  int64_t routed_k = 0;
  RawBufferPtr<int32_t> ids;
  RawBufferPtr<float> prompt_mask;
  RawBufferPtr<float> completion_mask;
  RawBufferPtr<float> advantages;
  RawBufferPtr<int32_t> segment_ids;
  RawBufferPtr<int32_t> segment_positions;
  std::vector<RawBufferPtr<float>> per_token;
  RawBufferPtr<int16_t> routed_experts;
  std::vector<std::vector<int64_t>> bin_indices;
};

// Allocates uninitialized buffers and populates all rows in a single forward
// pass per row without holding the GIL. Every output byte is written at most
// once (no redundant full-matrix memset before memcpy).
RawChunkBuffers AllocateAndPopulateChunkNogil(
    const std::vector<std::vector<const SegmentView*>>& bins_views,
    size_t n_carried, int64_t budget, int32_t pad_id,
    int64_t segment_alignment_boundary, int64_t routed_layers,
    int64_t routed_k) {
  const size_t n_bins = bins_views.size();
  const size_t total_elems = n_bins * static_cast<size_t>(budget);

  RawChunkBuffers out;
  out.n_bins = n_bins;
  out.budget = static_cast<size_t>(budget);
  out.routed_layers = routed_layers;
  out.routed_k = routed_k;

  out.ids = AllocRawBuffer<int32_t>(total_elems);
  out.prompt_mask = AllocRawBuffer<float>(total_elems);
  out.completion_mask = AllocRawBuffer<float>(total_elems);
  out.advantages = AllocRawBuffer<float>(total_elems);
  out.segment_ids = AllocRawBuffer<int32_t>(total_elems);
  out.segment_positions = AllocRawBuffer<int32_t>(total_elems);
  out.per_token.resize(n_carried);
  for (size_t k = 0; k < n_carried; ++k) {
    out.per_token[k] = AllocRawBuffer<float>(total_elems);
  }

  const int64_t routed_stride = routed_layers * routed_k;
  if (routed_stride > 0) {
    out.routed_experts = AllocRawBuffer<int16_t>(
        total_elems * static_cast<size_t>(routed_stride));
  }

  int32_t* ids_base = out.ids.get();
  float* prompt_mask_base = out.prompt_mask.get();
  float* completion_mask_base = out.completion_mask.get();
  float* advantages_base = out.advantages.get();
  int32_t* segment_ids_base = out.segment_ids.get();
  int32_t* segment_positions_base = out.segment_positions.get();
  int16_t* routed_base = out.routed_experts.get();

  for (size_t b = 0; b < n_bins; ++b) {
    const auto& seg_views = bins_views[b];
    const int64_t row_offset = static_cast<int64_t>(b) * budget;
    int32_t* ids_row = ids_base + row_offset;
    float* pmask_row = prompt_mask_base + row_offset;
    float* cmask_row = completion_mask_base + row_offset;
    float* adv_row = advantages_base + row_offset;
    int32_t* seg_ids_row = segment_ids_base + row_offset;
    int32_t* seg_pos_row = segment_positions_base + row_offset;
    int16_t* routed_row =
        (routed_base != nullptr) ? (routed_base + row_offset * routed_stride)
                                 : nullptr;

    // Shared helper to pad either an alignment gap or the trailing row tail.
    auto zero_pad_span = [&](int64_t start, int64_t len) {
      const size_t len_sz = static_cast<size_t>(len);
      std::fill_n(ids_row + start, len, pad_id);
      std::memset(pmask_row + start, 0, len_sz * sizeof(float));
      std::memset(cmask_row + start, 0, len_sz * sizeof(float));
      std::memset(adv_row + start, 0, len_sz * sizeof(float));
      std::memset(seg_ids_row + start, 0, len_sz * sizeof(int32_t));
      std::memset(seg_pos_row + start, 0, len_sz * sizeof(int32_t));
      for (size_t k = 0; k < n_carried; ++k) {
        std::memset(out.per_token[k].get() + row_offset + start, 0,
                    len_sz * sizeof(float));
      }
      if (routed_row != nullptr) {
        std::fill_n(routed_row + start * routed_stride, len * routed_stride,
                    static_cast<int16_t>(-1));
      }
    };

    int64_t cursor = 0;
    const size_t n_segs = seg_views.size();
    for (size_t s = 0; s < n_segs; ++s) {
      const SegmentView& v = *seg_views[s];
      const int32_t seg_id = static_cast<int32_t>(s + 1);
      const int64_t p = v.p_len;
      const int64_t c = v.c_len;
      const int64_t n = p + c;
      const int64_t aligned =
          (s > 0 && segment_alignment_boundary > 1)
              ? AlignOffset(cursor, segment_alignment_boundary)
              : cursor;
      const int64_t end = aligned + n;
      if (end > budget) {
        throw std::invalid_argument("pack_bin: bin size " +
                                    std::to_string(end) + " exceeds budget " +
                                    std::to_string(budget) + ".");
      }
      if (aligned > cursor) {
        zero_pad_span(cursor, aligned - cursor);
      }
      cursor = aligned;
      const int64_t comp_start = cursor + p;
      const size_t p_sz = static_cast<size_t>(p);
      const size_t c_sz = static_cast<size_t>(c);

      if (p > 0) {
        std::memcpy(ids_row + cursor, v.prompt_ptr, p_sz * sizeof(int32_t));
        std::fill_n(pmask_row + cursor, p, 1.0f);
        std::memset(cmask_row + cursor, 0, p_sz * sizeof(float));
        std::memset(adv_row + cursor, 0, p_sz * sizeof(float));
        for (size_t k = 0; k < n_carried; ++k) {
          std::memset(out.per_token[k].get() + row_offset + cursor, 0,
                      p_sz * sizeof(float));
        }
      }
      if (c > 0) {
        std::memcpy(ids_row + comp_start, v.comp_ptr, c_sz * sizeof(int32_t));
        std::memset(pmask_row + comp_start, 0, c_sz * sizeof(float));
        std::memcpy(cmask_row + comp_start, v.mask_ptr, c_sz * sizeof(float));
        std::memcpy(adv_row + comp_start, v.adv_ptr, c_sz * sizeof(float));
        for (size_t k = 0; k < n_carried; ++k) {
          std::memcpy(out.per_token[k].get() + row_offset + comp_start,
                      v.per_token_ptrs[k], c_sz * sizeof(float));
        }
      }
      std::fill_n(seg_ids_row + cursor, n, seg_id);
      for (int64_t pos = 0; pos < n; ++pos) {
        seg_pos_row[cursor + pos] = static_cast<int32_t>(pos);
      }
      if (routed_row != nullptr) {
        const int64_t m = (v.routed_ptr != nullptr) ? v.routed_tokens : 0;
        if (m > 0) {
          const size_t num_elems = static_cast<size_t>(m * routed_stride);
          std::memcpy(routed_row + cursor * routed_stride, v.routed_ptr,
                      num_elems * sizeof(int16_t));
        }
        if (m < n) {
          std::fill_n(routed_row + (cursor + m) * routed_stride,
                      (n - m) * routed_stride, static_cast<int16_t>(-1));
        }
      }
      cursor = end;
    }

    if (cursor < budget) {
      zero_pad_span(cursor, budget - cursor);
    }
  }
  return out;
}

nb::tuple WrapRawChunkBuffers(RawChunkBuffers&& raw,
                              const std::vector<nb::str>& carried_py_keys,
                              bool include_bin_indices) {
  const size_t n_bins = raw.n_bins;
  const size_t budget = raw.budget;
  const size_t n_carried = carried_py_keys.size();

  nb::object ids_obj = Wrap2DArray(std::move(raw.ids), n_bins, budget);
  nb::object pmask_obj =
      Wrap2DArray(std::move(raw.prompt_mask), n_bins, budget);
  nb::object cmask_obj =
      Wrap2DArray(std::move(raw.completion_mask), n_bins, budget);
  nb::object adv_obj = Wrap2DArray(std::move(raw.advantages), n_bins, budget);
  nb::object seg_ids_obj =
      Wrap2DArray(std::move(raw.segment_ids), n_bins, budget);
  nb::object seg_pos_obj =
      Wrap2DArray(std::move(raw.segment_positions), n_bins, budget);

  nb::dict per_token_dict;
  for (size_t k = 0; k < n_carried; ++k) {
    per_token_dict[carried_py_keys[k]] =
        Wrap2DArray(std::move(raw.per_token[k]), n_bins, budget);
  }

  nb::object routed_obj = nb::none();
  if (raw.routed_experts != nullptr) {
    routed_obj = Wrap4DArray(std::move(raw.routed_experts), n_bins, budget,
                             static_cast<size_t>(raw.routed_layers),
                             static_cast<size_t>(raw.routed_k));
  }

  if (include_bin_indices) {
    return nb::make_tuple(ids_obj, pmask_obj, cmask_obj, adv_obj, seg_ids_obj,
                          seg_pos_obj, per_token_dict, routed_obj,
                          nb::cast(raw.bin_indices));
  }
  return nb::make_tuple(ids_obj, pmask_obj, cmask_obj, adv_obj, seg_ids_obj,
                        seg_pos_obj, per_token_dict, routed_obj);
}

struct InternedAttrs {
  PyObject* prompt_ids = PyUnicode_InternFromString("prompt_ids");
  PyObject* prompt_mask = PyUnicode_InternFromString("prompt_mask");
  PyObject* completion_ids = PyUnicode_InternFromString("completion_ids");
  PyObject* completion_mask = PyUnicode_InternFromString("completion_mask");
  PyObject* advantages = PyUnicode_InternFromString("advantages");
  PyObject* per_token = PyUnicode_InternFromString("per_token");
  PyObject* policy_version = PyUnicode_InternFromString("policy_version");
  PyObject* routed_experts = PyUnicode_InternFromString("routed_experts");
  PyObject* metadata = PyUnicode_InternFromString("metadata");
  PyObject* traj_id = PyUnicode_InternFromString("traj_id");
  static constexpr const char* kPerTokenNames[5] = {
      "ref_per_token_logps", "old_per_token_logps", "returns", "old_values",
      "sampler_is_weights"};
  PyObject* per_token_keys[5] = {
      PyUnicode_InternFromString(kPerTokenNames[0]),
      PyUnicode_InternFromString(kPerTokenNames[1]),
      PyUnicode_InternFromString(kPerTokenNames[2]),
      PyUnicode_InternFromString(kPerTokenNames[3]),
      PyUnicode_InternFromString(kPerTokenNames[4]),
  };
};

inline const InternedAttrs& Attrs() {
  static const InternedAttrs* const kAttrs = new InternedAttrs();
  return *kAttrs;
}

inline nb::object GetAttrChecked(PyObject* obj, PyObject* attr) {
  PyObject* res = PyObject_GetAttr(obj, attr);
  if (res == nullptr) {
    throw nb::python_error();
  }
  return nb::steal<nb::object>(res);
}

inline PyObject* GetDictItemChecked(PyObject* dict, PyObject* key) {
  PyObject* res = PyDict_GetItemWithError(dict, key);
  if (res == nullptr && PyErr_Occurred() != nullptr) {
    throw nb::python_error();
  }
  return res;
}

inline std::vector<nb::str> ToPyStrVector(
    const std::vector<std::string>& names) {
  std::vector<nb::str> out;
  out.reserve(names.size());
  for (const std::string& name : names) {
    out.emplace_back(name.c_str());
  }
  return out;
}

void ExtractSingleItemView(PyObject* item_ptr,
                           const std::vector<nb::str>& carried_py_keys,
                           bool extract_routed, SegmentView* v,
                           ScopedPyBufferList* buffers) {
  const InternedAttrs& a = Attrs();
  nb::object p_obj = GetAttrChecked(item_ptr, a.prompt_ids);
  nb::object c_obj = GetAttrChecked(item_ptr, a.completion_ids);
  nb::object m_obj = GetAttrChecked(item_ptr, a.completion_mask);
  nb::object a_obj = GetAttrChecked(item_ptr, a.advantages);

  RawArrayView p_view =
      Extract1DArrayPtr(p_obj.ptr(), sizeof(int32_t), buffers);
  RawArrayView c_view =
      Extract1DArrayPtr(c_obj.ptr(), sizeof(int32_t), buffers);
  RawArrayView m_view = Extract1DArrayPtr(m_obj.ptr(), sizeof(float), buffers);
  RawArrayView a_view = Extract1DArrayPtr(a_obj.ptr(), sizeof(float), buffers);

  if (m_view.size != c_view.size || a_view.size != c_view.size) {
    throw std::invalid_argument(
        "completion_mask and advantages lengths must match completion_ids.");
  }

  v->prompt_ptr = static_cast<const int32_t*>(p_view.ptr);
  v->p_len = p_view.size;
  v->comp_ptr = static_cast<const int32_t*>(c_view.ptr);
  v->c_len = c_view.size;
  v->mask_ptr = static_cast<const float*>(m_view.ptr);
  v->adv_ptr = static_cast<const float*>(a_view.ptr);

  const size_t n_carried = carried_py_keys.size();
  if (n_carried > 0) {
    nb::object pt_obj = GetAttrChecked(item_ptr, a.per_token);
    PyObject* pt_dict = pt_obj.ptr();
    if (!PyDict_Check(pt_dict)) {
      throw std::invalid_argument("PackItem.per_token must be a dict.");
    }
    v->per_token_ptrs.resize(n_carried);
    for (size_t k = 0; k < n_carried; ++k) {
      PyObject* arr_obj = GetDictItemChecked(pt_dict, carried_py_keys[k].ptr());
      if (arr_obj == nullptr) {
        throw std::invalid_argument(
            "Carried per_token key missing from PackItem.per_token.");
      }
      RawArrayView pt_view = Extract1DArrayPtr(arr_obj, sizeof(float), buffers);
      if (pt_view.size != c_view.size) {
        throw std::invalid_argument(
            "Carried per_token array length must match completion_ids.");
      }
      v->per_token_ptrs[k] = static_cast<const float*>(pt_view.ptr);
    }
  }

  if (extract_routed) {
    nb::object r_obj = GetAttrChecked(item_ptr, a.routed_experts);
    if (!r_obj.is_none()) {
      ExtractRoutedView(r_obj.ptr(), v, buffers);
    }
  }
}

// Shared First-Fit Decreasing (FFD) core for a single chunk over pre-sorted
// `order` indices. Places items into `bin_indices`, records unplaced items in
// `next_order` (preserving descending length order), marks `placed_flags`, and
// returns the total tokens placed in this chunk.
int64_t RunFfdOneChunkNogil(const std::vector<int64_t>& lengths,
                            const std::vector<int64_t>& order, int64_t budget,
                            int64_t pack_size, int64_t max_segments,
                            int64_t segment_alignment_boundary,
                            std::vector<std::vector<int64_t>>* bin_indices,
                            std::vector<int64_t>* next_order,
                            std::vector<uint8_t>* placed_flags) {
  if (pack_size <= 0) {
    bin_indices->clear();
    if (next_order != nullptr) {
      *next_order = order;
    }
    return 0;
  }
  const size_t n_bins = static_cast<size_t>(pack_size);
  bin_indices->assign(n_bins, {});
  if (next_order != nullptr) {
    next_order->clear();
  }
  std::vector<int64_t> next_starts(n_bins, 0);
  std::vector<int64_t> rem_budgets(n_bins, budget);
  std::vector<int64_t> seg_counts(n_bins, 0);
  int64_t max_rem = budget;
  int64_t tokens_placed = 0;
  const bool unaligned = (segment_alignment_boundary == 1);

  for (size_t oi = 0; oi < order.size(); ++oi) {
    const int64_t idx = order[oi];
    const int64_t n = lengths[static_cast<size_t>(idx)];
    if (n > max_rem) {
      if (next_order != nullptr) {
        next_order->push_back(idx);
      }
      continue;
    }
    bool placed = false;
    for (size_t b = 0; b < n_bins; ++b) {
      if (rem_budgets[b] >= n) {
        (*bin_indices)[b].push_back(idx);
        const int64_t end = next_starts[b] + n;
        const int64_t cnt = ++seg_counts[b];
        if (cnt >= max_segments) {
          rem_budgets[b] = -1;
        } else if (unaligned) {
          next_starts[b] = end;
          rem_budgets[b] = budget - end;
        } else {
          const int64_t aligned =
              AlignOffset(end, segment_alignment_boundary);
          next_starts[b] = aligned;
          rem_budgets[b] = budget - aligned;
        }
        (*placed_flags)[static_cast<size_t>(idx)] = 1;
        tokens_placed += n;
        placed = true;
        break;
      }
    }
    if (!placed) {
      if (next_order != nullptr) {
        next_order->push_back(idx);
      }
      max_rem = *std::max_element(rem_budgets.begin(), rem_budgets.end());
      if (max_rem < 0) {
        if (next_order != nullptr) {
          next_order->insert(next_order->end(), order.begin() + oi + 1,
                             order.end());
        }
        break;
      }
    }
  }
  return tokens_placed;
}

// First-Fit Decreasing (FFD) bin packing executed entirely without the GIL.
// Returns `(bin_item_indices, leftover_item_indices)`.
std::pair<std::vector<std::vector<int64_t>>, std::vector<int64_t>>
FillOneChunkFast(const std::vector<int64_t>& lengths, int64_t budget,
                 int64_t pack_size, int64_t max_segments,
                 int64_t segment_alignment_boundary) {
  if (segment_alignment_boundary <= 0) {
    throw std::invalid_argument(
        "segment_alignment_boundary must be positive, got " +
        std::to_string(segment_alignment_boundary) + ".");
  }
  std::vector<std::vector<int64_t>> bins(
      pack_size > 0 ? static_cast<size_t>(pack_size) : 0);
  const size_t num_items = lengths.size();
  if (num_items == 0 || pack_size <= 0 || max_segments <= 0) {
    std::vector<int64_t> all_leftover(num_items);
    std::iota(all_leftover.begin(), all_leftover.end(), 0);
    return {std::move(bins), std::move(all_leftover)};
  }

  std::vector<int64_t> leftover;
  {
    nb::gil_scoped_release release;
    std::vector<int64_t> order(num_items);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&lengths](int64_t a, int64_t b) {
                       return lengths[static_cast<size_t>(a)] >
                              lengths[static_cast<size_t>(b)];
                     });
    std::vector<uint8_t> placed_flags(num_items, 0);
    RunFfdOneChunkNogil(lengths, order, budget, pack_size, max_segments,
                        segment_alignment_boundary, &bins,
                        /*next_order=*/nullptr, &placed_flags);
    leftover.reserve(num_items);
    for (size_t i = 0; i < num_items; ++i) {
      if (!placed_flags[i]) {
        leftover.push_back(static_cast<int64_t>(i));
      }
    }
  }
  return {std::move(bins), std::move(leftover)};
}

// Packs a single pre-binned chunk inside a `nb::gil_scoped_release` block
// (including output buffer allocation and single-pass memory population).
nb::tuple PackChunkFast(nb::sequence bins_seq,
                        const std::vector<std::string>& carried_names,
                        int64_t budget, int32_t pad_id,
                        int64_t segment_alignment_boundary,
                        nb::object routed_shape_obj) {
  const size_t n_bins = nb::len(bins_seq);
  const size_t n_carried = carried_names.size();
  std::vector<nb::str> carried_py_keys = ToPyStrVector(carried_names);

  int64_t routed_layers = 0;
  int64_t routed_k = 0;
  if (!routed_shape_obj.is_none()) {
    auto r_shape = nb::cast<std::pair<int64_t, int64_t>>(routed_shape_obj);
    routed_layers = r_shape.first;
    routed_k = r_shape.second;
  }

  ScopedPyBufferList active_buffers;
  std::vector<std::vector<SegmentView>> owned_views(n_bins);
  std::vector<std::vector<const SegmentView*>> bins_views(n_bins);
  for (size_t b = 0; b < n_bins; ++b) {
    nb::sequence bin_items = nb::borrow<nb::sequence>(bins_seq[b]);
    const size_t n_segs = nb::len(bin_items);
    owned_views[b].resize(n_segs);
    bins_views[b].resize(n_segs);
    for (size_t s = 0; s < n_segs; ++s) {
      ExtractSingleItemView(bin_items[s].ptr(), carried_py_keys,
                            /*extract_routed=*/(routed_layers * routed_k > 0),
                            &owned_views[b][s], &active_buffers);
      bins_views[b][s] = &owned_views[b][s];
    }
  }

  RawChunkBuffers raw;
  {
    nb::gil_scoped_release release;
    raw = AllocateAndPopulateChunkNogil(
        bins_views, n_carried, budget, pad_id, segment_alignment_boundary,
        routed_layers, routed_k);
  }
  return WrapRawChunkBuffers(std::move(raw), carried_py_keys,
                             /*include_bin_indices=*/false);
}

// Packs multiple chunks in a single `nb::gil_scoped_release` block:
// - Extracts `SegmentView` pointers for all `items` once.
// - Releases the GIL once for the entire multi-chunk FFD + buffer allocation +
//   single-pass memory copy loop.
// - Sorts `order` by descending `num_tokens` only once (since removing placed
//   items preserves stable descending order for subsequent chunks).
// Returns `(list_of_packed_chunks, leftover_indices)`.
nb::tuple PackSequenceChunksFast(nb::sequence items_seq,
                                 const std::vector<std::string>& carried_names,
                                 int64_t budget, int64_t pack_size,
                                 int64_t max_segments, int32_t pad_id,
                                 int64_t segment_alignment_boundary,
                                 int64_t min_buffered_tokens) {
  if (segment_alignment_boundary <= 0) {
    throw std::invalid_argument(
        "segment_alignment_boundary must be positive, got " +
        std::to_string(segment_alignment_boundary) + ".");
  }
  if (pack_size <= 0) {
    throw std::invalid_argument("pack_size must be positive, got " +
                                std::to_string(pack_size) + ".");
  }
  const size_t num_items = nb::len(items_seq);
  const size_t n_carried = carried_names.size();
  std::vector<nb::str> carried_py_keys = ToPyStrVector(carried_names);

  ScopedPyBufferList active_buffers;
  std::vector<SegmentView> item_views(num_items);
  std::vector<int64_t> lengths(num_items);
  for (size_t i = 0; i < num_items; ++i) {
    ExtractSingleItemView(items_seq[i].ptr(), carried_py_keys,
                          /*extract_routed=*/true, &item_views[i],
                          &active_buffers);
    lengths[i] = item_views[i].p_len + item_views[i].c_len;
  }

  std::vector<RawChunkBuffers> raw_chunks;
  std::vector<int64_t> leftover_indices;
  {
    nb::gil_scoped_release release;

    int64_t total_remaining_tokens =
        std::accumulate(lengths.begin(), lengths.end(), int64_t{0});

    std::vector<int64_t> order(num_items);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&lengths](int64_t a, int64_t b) {
                       return lengths[static_cast<size_t>(a)] >
                              lengths[static_cast<size_t>(b)];
                     });

    std::vector<uint8_t> placed_globally(num_items, 0);
    std::vector<int64_t> next_order;
    next_order.reserve(num_items);
    const size_t n_bins = static_cast<size_t>(pack_size);

    while (!order.empty()) {
      if (min_buffered_tokens > 0 &&
          total_remaining_tokens < min_buffered_tokens) {
        break;
      }

      std::vector<std::vector<int64_t>> bin_indices;
      const int64_t tokens_placed = RunFfdOneChunkNogil(
          lengths, order, budget, pack_size, max_segments,
          segment_alignment_boundary, &bin_indices, &next_order,
          &placed_globally);

      if (tokens_placed == 0 && next_order.size() == order.size()) {
        throw std::invalid_argument("pack_core: no items placed in any bin.");
      }
      total_remaining_tokens -= tokens_placed;

      std::vector<std::vector<const SegmentView*>> bins_views(n_bins);
      int64_t chunk_routed_layers = 0;
      int64_t chunk_routed_k = 0;
      for (size_t b = 0; b < n_bins; ++b) {
        const auto& b_idxs = bin_indices[b];
        bins_views[b].reserve(b_idxs.size());
        for (int64_t idx : b_idxs) {
          const SegmentView& v = item_views[static_cast<size_t>(idx)];
          bins_views[b].push_back(&v);
          if (v.routed_ptr != nullptr) {
            if (chunk_routed_layers == 0 && chunk_routed_k == 0) {
              chunk_routed_layers = v.routed_layers;
              chunk_routed_k = v.routed_k;
            } else if (chunk_routed_layers != v.routed_layers ||
                       chunk_routed_k != v.routed_k) {
              throw std::invalid_argument(
                  "Items disagree on routed_experts trailing shape.");
            }
          }
        }
      }

      RawChunkBuffers raw = AllocateAndPopulateChunkNogil(
          bins_views, n_carried, budget, pad_id, segment_alignment_boundary,
          chunk_routed_layers, chunk_routed_k);
      raw.bin_indices = std::move(bin_indices);
      raw_chunks.push_back(std::move(raw));

      order.swap(next_order);
    }

    leftover_indices.reserve(order.size());
    for (size_t i = 0; i < num_items; ++i) {
      if (!placed_globally[i]) {
        leftover_indices.push_back(static_cast<int64_t>(i));
      }
    }
  }

  nb::list chunks_out;
  for (auto& raw : raw_chunks) {
    chunks_out.append(WrapRawChunkBuffers(std::move(raw), carried_py_keys,
                                          /*include_bin_indices=*/true));
  }
  return nb::make_tuple(chunks_out, nb::cast(leftover_indices));
}

struct PaddedFieldSource {
  const float* ptr = nullptr;
  int64_t size = 0;
  float scalar_fill = 0.0f;
  bool is_scalar = true;
};

struct PaddedItemView {
  const int32_t* prompt_ptr = nullptr;
  int64_t p_len = 0;
  const float* prompt_mask_ptr = nullptr;
  int64_t p_mask_len = -1;
  const int32_t* comp_ptr = nullptr;
  int64_t c_len = 0;
  bool has_comp_mask = false;
  PaddedFieldSource comp_mask;
  PaddedFieldSource advantages;
  std::vector<PaddedFieldSource> optional_fields;
  const int16_t* routed_ptr = nullptr;
  int64_t routed_tokens = 0;
  int64_t routed_layers = 0;
  int64_t routed_k = 0;
};

inline void PopulateCompletionAlignedRow(const PaddedFieldSource& src,
                                         int64_t p_full_len,
                                         int64_t c_full_len,
                                         int64_t c_trunc_len,
                                         int64_t max_resp, float* dst_row) {
  const size_t max_resp_sz = static_cast<size_t>(max_resp);
  if (src.is_scalar) {
    if (src.scalar_fill == 0.0f) {
      std::memset(dst_row, 0, max_resp_sz * sizeof(float));
    } else {
      std::fill_n(dst_row, c_trunc_len, src.scalar_fill);
      if (c_trunc_len < max_resp) {
        std::memset(dst_row + c_trunc_len, 0,
                    static_cast<size_t>(max_resp - c_trunc_len) *
                        sizeof(float));
      }
    }
    return;
  }
  const float* ptr = src.ptr;
  int64_t avail = src.size;
  if (avail == p_full_len + c_full_len || avail == p_full_len + c_trunc_len) {
    ptr += p_full_len;
    avail -= p_full_len;
  }
  const int64_t copy_len = std::min(avail, c_trunc_len);
  if (copy_len > 0) {
    std::memcpy(dst_row, ptr, static_cast<size_t>(copy_len) * sizeof(float));
  }
  if (copy_len < max_resp) {
    std::memset(dst_row + copy_len, 0,
                static_cast<size_t>(max_resp - copy_len) * sizeof(float));
  }
}

inline nb::object CoerceToContiguousNumpy(PyObject* obj,
                                          const char* dtype_name) {
  static nb::object* const kAsContiguousArray = new nb::object(
      nb::module_::import_("numpy").attr("ascontiguousarray"));
  return (*kAsContiguousArray)(nb::borrow<nb::object>(obj),
                               nb::arg("dtype") = dtype_name);
}

inline PaddedFieldSource ExtractPaddedFieldSource(
    PyObject* obj, float default_fill, std::vector<nb::object>* keep) {
  PaddedFieldSource out;
  if (obj == nullptr || obj == Py_None) {
    out.is_scalar = true;
    out.scalar_fill = default_fill;
    return out;
  }
  if (PyFloat_Check(obj) || PyLong_Check(obj)) {
    out.is_scalar = true;
    out.scalar_fill = static_cast<float>(PyFloat_AsDouble(obj));
    if (PyErr_Occurred()) {
      throw nb::python_error();
    }
    return out;
  }
  nb::ndarray<nb::numpy, const float, nb::c_contig> arr;
  if (!nb::try_cast(nb::borrow<nb::object>(obj), arr, /*convert=*/false)) {
    nb::object cast_obj = CoerceToContiguousNumpy(obj, "float32");
    arr = nb::cast<nb::ndarray<nb::numpy, const float, nb::c_contig>>(cast_obj);
    keep->push_back(std::move(cast_obj));
  }
  if (arr.size() == 1) {
    out.is_scalar = true;
    out.scalar_fill = *arr.data();
  } else {
    out.is_scalar = false;
    out.ptr = arr.data();
    out.size = static_cast<int64_t>(arr.size());
  }
  return out;
}

inline RawArrayView ExtractInt321DView(PyObject* obj,
                                       std::vector<nb::object>* keep) {
  if (obj == nullptr || obj == Py_None) {
    return {nullptr, 0};
  }
  nb::ndarray<nb::numpy, const int32_t, nb::c_contig> arr;
  if (nb::try_cast(nb::borrow<nb::object>(obj), arr, /*convert=*/false)) {
    return {arr.data(), static_cast<int64_t>(arr.size())};
  }
  nb::object cast_obj = CoerceToContiguousNumpy(obj, "int32");
  auto c_arr = nb::cast<nb::ndarray<nb::numpy, const int32_t, nb::c_contig>>(
      cast_obj);
  keep->push_back(std::move(cast_obj));
  return {c_arr.data(), static_cast<int64_t>(c_arr.size())};
}

// Single-pass `nogil` 2D rectangular padding for `PaddedBatchAssembler`.
nb::tuple AssemblePaddedChunkFast(nb::sequence chunk_seq,
                                  const std::vector<std::string>& present_names,
                                  int64_t batch_size, int64_t max_prompt_len,
                                  int64_t max_response_len, int32_t pad_id,
                                  bool replay_routing) {
  const InternedAttrs& a = Attrs();
  const size_t n_items = nb::len(chunk_seq);
  const size_t n_opt = present_names.size();
  std::vector<nb::str> opt_py_keys = ToPyStrVector(present_names);

  std::vector<nb::object> temp_keepalive;
  std::vector<PaddedItemView> views(n_items);
  int64_t routed_layers = 0;
  int64_t routed_k = 0;

  for (size_t i = 0; i < n_items; ++i) {
    PyObject* it = chunk_seq[i].ptr();
    PaddedItemView& v = views[i];

    nb::object p_obj = GetAttrChecked(it, a.prompt_ids);
    RawArrayView p_view = ExtractInt321DView(p_obj.ptr(), &temp_keepalive);
    v.prompt_ptr = static_cast<const int32_t*>(p_view.ptr);
    v.p_len = p_view.size;
    temp_keepalive.push_back(std::move(p_obj));

    nb::object c_obj = GetAttrChecked(it, a.completion_ids);
    RawArrayView c_view = ExtractInt321DView(c_obj.ptr(), &temp_keepalive);
    v.comp_ptr = static_cast<const int32_t*>(c_view.ptr);
    v.c_len = c_view.size;
    temp_keepalive.push_back(std::move(c_obj));

    nb::object pm_obj = GetAttrChecked(it, a.prompt_mask);
    if (!pm_obj.is_none()) {
      nb::ndarray<nb::numpy, const float, nb::c_contig> pm_arr;
      if (!nb::try_cast(pm_obj, pm_arr, /*convert=*/false)) {
        pm_obj = CoerceToContiguousNumpy(pm_obj.ptr(), "float32");
        pm_arr =
            nb::cast<nb::ndarray<nb::numpy, const float, nb::c_contig>>(pm_obj);
      }
      if (static_cast<int64_t>(pm_arr.size()) == v.p_len) {
        v.prompt_mask_ptr = pm_arr.data();
        v.p_mask_len = v.p_len;
      }
      temp_keepalive.push_back(std::move(pm_obj));
    }

    nb::object cm_obj = GetAttrChecked(it, a.completion_mask);
    if (!cm_obj.is_none()) {
      v.has_comp_mask = true;
      v.comp_mask =
          ExtractPaddedFieldSource(cm_obj.ptr(), 0.0f, &temp_keepalive);
      temp_keepalive.push_back(std::move(cm_obj));
    }

    nb::object adv_obj = GetAttrChecked(it, a.advantages);
    v.advantages =
        ExtractPaddedFieldSource(adv_obj.ptr(), 0.0f, &temp_keepalive);
    temp_keepalive.push_back(std::move(adv_obj));

    v.optional_fields.resize(n_opt);
    for (size_t k = 0; k < n_opt; ++k) {
      nb::object opt_obj = GetAttrChecked(it, opt_py_keys[k].ptr());
      v.optional_fields[k] =
          ExtractPaddedFieldSource(opt_obj.ptr(), 0.0f, &temp_keepalive);
      temp_keepalive.push_back(std::move(opt_obj));
    }

    if (replay_routing) {
      nb::object r_obj = GetAttrChecked(it, a.routed_experts);
      if (!r_obj.is_none()) {
        nb::ndarray<nb::numpy, const int16_t, nb::ndim<3>, nb::c_contig> r_arr;
        if (!nb::try_cast(r_obj, r_arr, /*convert=*/false)) {
          r_obj = CoerceToContiguousNumpy(r_obj.ptr(), "int16");
          r_arr = nb::cast<
              nb::ndarray<nb::numpy, const int16_t, nb::ndim<3>, nb::c_contig>>(
              r_obj);
        }
        v.routed_ptr = r_arr.data();
        v.routed_tokens = static_cast<int64_t>(r_arr.shape(0));
        v.routed_layers = static_cast<int64_t>(r_arr.shape(1));
        v.routed_k = static_cast<int64_t>(r_arr.shape(2));
        temp_keepalive.push_back(std::move(r_obj));
        const int64_t c_trunc = std::min(v.c_len, max_response_len);
        const int64_t min_len = std::max<int64_t>(v.p_len + c_trunc - 1, 0);
        if (v.routed_tokens < min_len) {
          throw std::invalid_argument(
              "routed_experts length must be >= " + std::to_string(min_len) +
              " (prompt_len + completion_len - 1 for prompt_len=" +
              std::to_string(v.p_len) +
              ", completion_len=" + std::to_string(c_trunc) + "); got shape (" +
              std::to_string(v.routed_tokens) + ", " +
              std::to_string(v.routed_layers) + ", " +
              std::to_string(v.routed_k) + ")");
        }
        if (routed_layers == 0 && routed_k == 0) {
          routed_layers = v.routed_layers;
          routed_k = v.routed_k;
        }
      }
    }
  }

  const size_t b_sz = static_cast<size_t>(batch_size);
  const size_t p_sz = static_cast<size_t>(max_prompt_len);
  const size_t c_sz = static_cast<size_t>(max_response_len);
  const size_t seq_sz = p_sz + c_sz;

  RawBufferPtr<int32_t> prompt_ids;
  RawBufferPtr<float> prompt_mask;
  RawBufferPtr<int32_t> comp_ids;
  RawBufferPtr<float> comp_mask;
  RawBufferPtr<float> advantages;
  std::vector<RawBufferPtr<float>> opt_bufs(n_opt);
  RawBufferPtr<int16_t> routed_buf;
  RawBufferPtr<int64_t> row_valid_tokens;
  RawBufferPtr<int64_t> row_num_sequences;
  int64_t truncated_prompts = 0;
  int64_t truncated_completions = 0;

  {
    nb::gil_scoped_release release;
    prompt_ids = AllocRawBuffer<int32_t>(b_sz * p_sz);
    prompt_mask = AllocRawBuffer<float>(b_sz * p_sz);
    comp_ids = AllocRawBuffer<int32_t>(b_sz * c_sz);
    comp_mask = AllocRawBuffer<float>(b_sz * c_sz);
    advantages = AllocRawBuffer<float>(b_sz * c_sz);
    for (size_t k = 0; k < n_opt; ++k) {
      opt_bufs[k] = AllocRawBuffer<float>(b_sz * c_sz);
    }
    const int64_t routed_stride = routed_layers * routed_k;
    if (replay_routing && routed_stride > 0) {
      routed_buf = AllocRawBuffer<int16_t>(
          b_sz * seq_sz * static_cast<size_t>(routed_stride));
      std::fill_n(routed_buf.get(),
                  b_sz * seq_sz * static_cast<size_t>(routed_stride),
                  static_cast<int16_t>(-1));
    }
    row_valid_tokens = AllocRawBuffer<int64_t>(b_sz);
    row_num_sequences = AllocRawBuffer<int64_t>(b_sz);
    std::memset(row_valid_tokens.get(), 0, b_sz * sizeof(int64_t));
    std::memset(row_num_sequences.get(), 0, b_sz * sizeof(int64_t));

    for (size_t r = 0; r < b_sz; ++r) {
      int32_t* p_row = prompt_ids.get() + r * p_sz;
      float* pm_row = prompt_mask.get() + r * p_sz;
      int32_t* c_row = comp_ids.get() + r * c_sz;
      float* cm_row = comp_mask.get() + r * c_sz;
      float* adv_row = advantages.get() + r * c_sz;

      if (r >= n_items) {
        std::fill_n(p_row, p_sz, pad_id);
        std::memset(pm_row, 0, p_sz * sizeof(float));
        std::fill_n(c_row, c_sz, pad_id);
        std::memset(cm_row, 0, c_sz * sizeof(float));
        std::memset(adv_row, 0, c_sz * sizeof(float));
        for (size_t k = 0; k < n_opt; ++k) {
          std::memset(opt_bufs[k].get() + r * c_sz, 0, c_sz * sizeof(float));
        }
        continue;
      }

      const PaddedItemView& v = views[r];
      if (v.p_len > max_prompt_len) {
        ++truncated_prompts;
      }
      if (v.c_len > max_response_len) {
        ++truncated_completions;
      }
      const int64_t p_keep = std::min(v.p_len, max_prompt_len);
      const int64_t p_pad = max_prompt_len - p_keep;
      const int64_t p_src_off = v.p_len - p_keep;
      const int64_t c_keep = std::min(v.c_len, max_response_len);
      const int64_t c_pad = max_response_len - c_keep;

      row_valid_tokens[r] = p_keep + c_keep;
      row_num_sequences[r] = 1;

      // Left-pad prompt_ids and prompt_mask.
      if (p_pad > 0) {
        std::fill_n(p_row, p_pad, pad_id);
        std::memset(pm_row, 0, static_cast<size_t>(p_pad) * sizeof(float));
      }
      if (p_keep > 0) {
        std::memcpy(p_row + p_pad, v.prompt_ptr + p_src_off,
                    static_cast<size_t>(p_keep) * sizeof(int32_t));
        if (v.prompt_mask_ptr != nullptr && v.p_mask_len == v.p_len) {
          std::memcpy(pm_row + p_pad, v.prompt_mask_ptr + p_src_off,
                      static_cast<size_t>(p_keep) * sizeof(float));
        } else {
          std::fill_n(pm_row + p_pad, p_keep, 1.0f);
        }
      }

      // Right-pad completion_ids.
      if (c_keep > 0) {
        std::memcpy(c_row, v.comp_ptr,
                    static_cast<size_t>(c_keep) * sizeof(int32_t));
      }
      if (c_pad > 0) {
        std::fill_n(c_row + c_keep, c_pad, pad_id);
      }

      // Completion mask.
      if (!v.has_comp_mask) {
        if (c_keep > 0) {
          std::fill_n(cm_row, c_keep, 1.0f);
        }
        if (c_pad > 0) {
          std::memset(cm_row + c_keep, 0,
                      static_cast<size_t>(c_pad) * sizeof(float));
        }
      } else {
        PopulateCompletionAlignedRow(v.comp_mask, v.p_len, v.c_len, c_keep,
                                     max_response_len, cm_row);
      }

      // Advantages & optional per-token fields.
      PopulateCompletionAlignedRow(v.advantages, v.p_len, v.c_len, c_keep,
                                   max_response_len, adv_row);
      for (size_t k = 0; k < n_opt; ++k) {
        PopulateCompletionAlignedRow(v.optional_fields[k], v.p_len, v.c_len,
                                     c_keep, max_response_len,
                                     opt_bufs[k].get() + r * c_sz);
      }

      // Routed experts alignment.
      if (routed_buf != nullptr && v.routed_ptr != nullptr) {
        int16_t* r_row =
            routed_buf.get() +
            static_cast<int64_t>(r * seq_sz) * routed_stride;
        const int64_t kept_p_start =
            std::max<int64_t>(v.p_len - max_prompt_len, 0);
        const int64_t kept_p_end = std::min(v.p_len, v.routed_tokens);
        const int64_t p_part_len =
            std::max<int64_t>(kept_p_end - kept_p_start, 0);
        const int64_t p_out_start = max_prompt_len - (v.p_len - kept_p_start);
        if (p_part_len > 0) {
          const size_t p_bytes =
              static_cast<size_t>(p_part_len * routed_stride) * sizeof(int16_t);
          std::memcpy(r_row + p_out_start * routed_stride,
                      v.routed_ptr + kept_p_start * routed_stride, p_bytes);
        }
        const int64_t kept_c_end =
            std::min(v.p_len + c_keep, v.routed_tokens);
        const int64_t c_part_len = std::max<int64_t>(kept_c_end - v.p_len, 0);
        if (c_part_len > 0) {
          const size_t c_bytes =
              static_cast<size_t>(c_part_len * routed_stride) * sizeof(int16_t);
          std::memcpy(r_row + max_prompt_len * routed_stride,
                      v.routed_ptr + v.p_len * routed_stride, c_bytes);
        }
      }
    }
  }

  nb::dict opt_dict;
  for (size_t k = 0; k < n_opt; ++k) {
    opt_dict[opt_py_keys[k]] = Wrap2DArray(std::move(opt_bufs[k]), b_sz, c_sz);
  }
  nb::object routed_obj = nb::none();
  if (routed_buf != nullptr) {
    routed_obj = Wrap4DArray(std::move(routed_buf), b_sz, seq_sz,
                             static_cast<size_t>(routed_layers),
                             static_cast<size_t>(routed_k));
  }
  return nb::make_tuple(
      Wrap2DArray(std::move(prompt_ids), b_sz, p_sz),
      Wrap2DArray(std::move(prompt_mask), b_sz, p_sz),
      Wrap2DArray(std::move(comp_ids), b_sz, c_sz),
      Wrap2DArray(std::move(comp_mask), b_sz, c_sz),
      Wrap2DArray(std::move(advantages), b_sz, c_sz), opt_dict, routed_obj,
      Wrap1DArray(std::move(row_valid_tokens), b_sz),
      Wrap1DArray(std::move(row_num_sequences), b_sz), truncated_prompts,
      truncated_completions);
}

// Counts forced router replay tokens on layer 0 (`[B, T, L, K]`) without the
// GIL, returning `(num_forced, num_real)`.
std::pair<int64_t, int64_t> CountRouterReplayForcedTokens(
    nb::ndarray<const int16_t, nb::ndim<4>, nb::c_contig, nb::device::cpu>
        routed_experts,
    nb::ndarray<const int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>
        segment_ids) {
  if (routed_experts.shape(0) != segment_ids.shape(0) ||
      routed_experts.shape(1) != segment_ids.shape(1)) {
    throw std::invalid_argument(
        "routed_experts leading [B, T] shape must match segment_ids shape.");
  }
  const int64_t num_layers = static_cast<int64_t>(routed_experts.shape(2));
  if (num_layers <= 0) {
    throw std::invalid_argument(
        "routed_experts must have at least 1 layer (shape[2] > 0).");
  }
  const int16_t* routed_ptr = routed_experts.data();
  const int32_t* seg_ptr = segment_ids.data();
  const int64_t num_tokens =
      static_cast<int64_t>(segment_ids.shape(0) * segment_ids.shape(1));
  const int64_t top_k = static_cast<int64_t>(routed_experts.shape(3));
  const int64_t token_stride = num_layers * top_k;

  int64_t num_real = 0;
  int64_t num_forced = 0;
  {
    nb::gil_scoped_release release;
    for (int64_t t = 0; t < num_tokens; ++t) {
      if (seg_ptr[t] <= 0) {
        continue;
      }
      ++num_real;
      const int16_t* l0 = routed_ptr + t * token_stride;
      bool valid = true;
      for (int64_t k = 0; k < top_k; ++k) {
        const int16_t e = l0[k];
        if (e < 0) {
          valid = false;
          break;
        }
        for (int64_t j = 0; j < k; ++j) {
          if (l0[j] == e) {
            valid = false;
            break;
          }
        }
        if (!valid) {
          break;
        }
      }
      if (valid) {
        ++num_forced;
      }
    }
  }
  return {num_forced, num_real};
}

inline void CheckUnbatchedRank1D(PyObject* obj, const char* field_name) {
  if (obj == nullptr || obj == Py_None) {
    return;
  }
  Py_buffer view;
  if (PyObject_GetBuffer(obj, &view, PyBUF_ND) == 0) {
    const int ndim = view.ndim;
    PyBuffer_Release(&view);
    if (ndim != 1) {
      throw std::invalid_argument(
          std::string("RLTrainerPayload.") + field_name + " has rank " +
          std::to_string(ndim) +
          "; sequence packing takes UNBATCHED payloads -- pass the unbatched "
          "payloads that produced it.");
    }
    return;
  }
  PyErr_Clear();
  static nb::object* const kNpAsArray =
      new nb::object(nb::module_::import_("numpy").attr("asarray"));
  nb::object arr = (*kNpAsArray)(nb::borrow<nb::object>(obj));
  const int ndim = nb::cast<int>(arr.attr("ndim"));
  if (ndim != 1) {
    throw std::invalid_argument(
        std::string("RLTrainerPayload.") + field_name + " has rank " +
        std::to_string(ndim) +
        "; sequence packing takes UNBATCHED payloads -- pass the unbatched "
        "payloads that produced it.");
  }
}

inline std::pair<nb::object, int64_t> Resolve1DInt32Field(PyObject* obj) {
  if (obj == nullptr || obj == Py_None) {
    return {Wrap1DArray(AllocRawBuffer<int32_t>(0), 0), 0};
  }
  nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> fast_arr;
  if (nb::try_cast(nb::borrow<nb::object>(obj), fast_arr, /*convert=*/false)) {
    return {nb::borrow<nb::object>(obj),
            static_cast<int64_t>(fast_arr.shape(0))};
  }
  nb::object coerced = CoerceToContiguousNumpy(obj, "int32");
  nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> c_arr;
  if (!nb::try_cast(coerced, c_arr, /*convert=*/false)) {
    coerced = coerced.attr("reshape")(-1);
    c_arr = nb::cast<
        nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig>>(
        coerced);
  }
  return {std::move(coerced), static_cast<int64_t>(c_arr.shape(0))};
}

inline nb::object Resolve1DFloatField(PyObject* obj, int64_t p_len,
                                      int64_t c_len, float fill,
                                      const char* field_name) {
  if (obj != nullptr && obj != Py_None) {
    nb::ndarray<nb::numpy, const float, nb::ndim<1>, nb::c_contig> fast_arr;
    if (nb::try_cast(nb::borrow<nb::object>(obj), fast_arr,
                     /*convert=*/false)) {
      const int64_t sz = static_cast<int64_t>(fast_arr.shape(0));
      if (sz == c_len) {
        return nb::borrow<nb::object>(obj);
      }
    }
  }
  std::vector<nb::object> keep;
  PaddedFieldSource src = ExtractPaddedFieldSource(obj, fill, &keep);
  const size_t c_sz = static_cast<size_t>(c_len);
  if (src.is_scalar) {
    RawBufferPtr<float> buf;
    {
      nb::gil_scoped_release release;
      buf = AllocRawBuffer<float>(c_sz);
      if (src.scalar_fill == 0.0f) {
        std::memset(buf.get(), 0, c_sz * sizeof(float));
      } else {
        std::fill_n(buf.get(), c_sz, src.scalar_fill);
      }
    }
    return Wrap1DArray(std::move(buf), c_sz);
  }
  const float* copy_src = nullptr;
  if (src.size == p_len + c_len) {
    copy_src = src.ptr + p_len;
  } else if (src.size == c_len) {
    copy_src = src.ptr;
  } else {
    throw std::invalid_argument(
        std::string("RLTrainerPayload.") + field_name +
        " has unexpected size " + std::to_string(src.size) +
        " which doesn't match either completion length " +
        std::to_string(c_len) + "; or whole sequence length " +
        std::to_string(p_len + c_len) + ".");
  }
  RawBufferPtr<float> buf;
  {
    nb::gil_scoped_release release;
    buf = AllocRawBuffer<float>(c_sz);
    if (c_sz > 0) {
      std::memcpy(buf.get(), copy_src, c_sz * sizeof(float));
    }
  }
  return Wrap1DArray(std::move(buf), c_sz);
}

nb::object ConvertPayloadToPackItem(PyObject* it, nb::handle pack_item_cls,
                                    int64_t* out_num_tokens) {
  const InternedAttrs& a = Attrs();
  nb::object c_raw = GetAttrChecked(it, a.completion_ids);
  if (c_raw.is_none()) {
    throw std::invalid_argument(
        "RLTrainerPayload.completion_ids is required for sequence packing.");
  }
  nb::object p_raw = GetAttrChecked(it, a.prompt_ids);
  nb::object pm_raw = GetAttrChecked(it, a.prompt_mask);
  nb::object cm_raw = GetAttrChecked(it, a.completion_mask);

  CheckUnbatchedRank1D(p_raw.ptr(), "prompt_ids");
  CheckUnbatchedRank1D(pm_raw.ptr(), "prompt_mask");
  CheckUnbatchedRank1D(c_raw.ptr(), "completion_ids");
  CheckUnbatchedRank1D(cm_raw.ptr(), "completion_mask");

  auto [p_arr, p_len] = Resolve1DInt32Field(p_raw.ptr());
  auto [c_arr, c_len] = Resolve1DInt32Field(c_raw.ptr());
  if (out_num_tokens != nullptr) {
    *out_num_tokens = p_len + c_len;
  }

  nb::object cm_arr =
      Resolve1DFloatField(cm_raw.ptr(), p_len, c_len, 1.0f, "completion_mask");
  nb::object adv_raw = GetAttrChecked(it, a.advantages);
  nb::object adv_arr =
      Resolve1DFloatField(adv_raw.ptr(), p_len, c_len, 0.0f, "advantages");

  nb::dict per_token_dict;
  for (size_t k = 0; k < 5; ++k) {
    nb::object f_raw = GetAttrChecked(it, a.per_token_keys[k]);
    if (!f_raw.is_none()) {
      nb::object f_arr = Resolve1DFloatField(f_raw.ptr(), p_len, c_len, 0.0f,
                                             InternedAttrs::kPerTokenNames[k]);
      if (PyDict_SetItem(per_token_dict.ptr(), a.per_token_keys[k],
                         f_arr.ptr()) != 0) {
        throw nb::python_error();
      }
    }
  }

  nb::object routed_out = nb::none();
  nb::object r_obj = GetAttrChecked(it, a.routed_experts);
  if (!r_obj.is_none()) {
    nb::ndarray<nb::numpy, const int16_t, nb::ndim<3>, nb::c_contig> r_arr;
    if (!nb::try_cast(r_obj, r_arr, /*convert=*/false)) {
      r_obj = CoerceToContiguousNumpy(r_obj.ptr(), "int16");
      if (!nb::try_cast(r_obj, r_arr, /*convert=*/false)) {
        throw std::invalid_argument(
            "PackItem.routed_experts must be a 3D numpy array of shape "
            "(p + c - 1 or p + c, num_layers, top_k).");
      }
    }
    const int64_t n = p_len + c_len;
    const int64_t min_len = std::max<int64_t>(n - 1, 0);
    const int64_t r_tokens = static_cast<int64_t>(r_arr.shape(0));
    if (r_tokens < min_len || r_tokens > n) {
      throw std::invalid_argument(
          "PackItem.routed_experts must be a numpy array of shape "
          "(p + c - 1 or p + c, num_layers, top_k) with length in [" +
          std::to_string(min_len) + ", " + std::to_string(n) +
          "], got shape (" + std::to_string(r_tokens) + ", " +
          std::to_string(r_arr.shape(1)) + ", " +
          std::to_string(r_arr.shape(2)) + ").");
    }
    routed_out = std::move(r_obj);
  }

  auto* pack_type = reinterpret_cast<PyTypeObject*>(pack_item_cls.ptr());
#if defined(Py_LIMITED_API)
  auto tp_alloc_fn =
      reinterpret_cast<allocfunc>(PyType_GetSlot(pack_type, Py_tp_alloc));
#else
  allocfunc tp_alloc_fn = pack_type->tp_alloc;
#endif
  nb::object inst = nb::steal<nb::object>(tp_alloc_fn(pack_type, 0));
  if (!inst.is_valid()) {
    throw nb::python_error();
  }
  if (PyObject_GenericSetAttr(inst.ptr(), a.prompt_ids, p_arr.ptr()) != 0 ||
      PyObject_GenericSetAttr(inst.ptr(), a.completion_ids, c_arr.ptr()) != 0 ||
      PyObject_GenericSetAttr(inst.ptr(), a.completion_mask, cm_arr.ptr()) !=
          0 ||
      PyObject_GenericSetAttr(inst.ptr(), a.advantages, adv_arr.ptr()) != 0 ||
      PyObject_GenericSetAttr(inst.ptr(), a.per_token, per_token_dict.ptr()) !=
          0 ||
      PyObject_GenericSetAttr(inst.ptr(), a.policy_version, Py_None) != 0 ||
      PyObject_GenericSetAttr(inst.ptr(), a.routed_experts,
                              routed_out.ptr()) != 0) {
    throw nb::python_error();
  }
  return inst;
}

nb::object ToPackItemFast(nb::handle item, nb::handle pack_item_cls) {
  return ConvertPayloadToPackItem(item.ptr(), pack_item_cls, nullptr);
}

void IngestPayloadsFast(nb::sequence items, int64_t max_packed_len,
                        nb::handle pack_item_cls, nb::list buffer_out) {
  const InternedAttrs& a = Attrs();
  const size_t n_items = nb::len(items);
  for (size_t i = 0; i < n_items; ++i) {
    if ((i & 7) == 7) {
      nb::gil_scoped_release yield_gil;
    }
    nb::object item = items[i];
    int64_t num_tokens = 0;
    nb::object pack_item =
        ConvertPayloadToPackItem(item.ptr(), pack_item_cls, &num_tokens);
    if (num_tokens > max_packed_len) {
      throw std::invalid_argument(
          "Item 0 has " + std::to_string(num_tokens) +
          " tokens, exceeding budget " + std::to_string(max_packed_len) + ".");
    }
    PyObject* raw_meta = PyObject_GetAttr(item.ptr(), a.metadata);
    if (raw_meta == nullptr) {
      if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
        PyErr_Clear();
      } else {
        throw nb::python_error();
      }
    }
    nb::object meta =
        raw_meta != nullptr ? nb::steal<nb::object>(raw_meta) : nb::none();
    nb::object traj_id_str;
    if (!meta.is_none() && PyDict_Check(meta.ptr())) {
      PyObject* raw_tid = GetDictItemChecked(meta.ptr(), a.traj_id);
      if (raw_tid != nullptr && raw_tid != Py_None) {
        PyObject* str_obj = PyObject_Str(raw_tid);
        if (str_obj == nullptr) {
          throw nb::python_error();
        }
        traj_id_str = nb::steal<nb::object>(str_obj);
      }
    } else if (!meta.is_none() && PyMapping_Check(meta.ptr())) {
      PyObject* raw_tid = PyObject_GetItem(meta.ptr(), a.traj_id);
      if (raw_tid != nullptr) {
        nb::object tid_obj = nb::steal<nb::object>(raw_tid);
        if (!tid_obj.is_none()) {
          PyObject* str_obj = PyObject_Str(tid_obj.ptr());
          if (str_obj == nullptr) {
            throw nb::python_error();
          }
          traj_id_str = nb::steal<nb::object>(str_obj);
        }
      } else {
        PyErr_Clear();
      }
    }
    if (!traj_id_str.is_valid()) {
      traj_id_str = nb::str("");
    }
    buffer_out.append(nb::make_tuple(pack_item, traj_id_str, item));
  }
}

}  // namespace

NB_MODULE(_packing_ext, m) {
  m.doc() = "C++ nanobind nogil acceleration for Tunix RL sequence packing.";
  m.def("fill_one_chunk_fast", &FillOneChunkFast, nb::arg("lengths"),
        nb::arg("budget"), nb::arg("pack_size"), nb::arg("max_segments"),
        nb::arg("segment_alignment_boundary"));
  m.def("pack_chunk_fast", &PackChunkFast, nb::arg("bins"),
        nb::arg("carried_names"), nb::arg("budget"), nb::arg("pad_id"),
        nb::arg("segment_alignment_boundary"),
        nb::arg("routed_shape") = nb::none());
  m.def("pack_sequence_chunks_fast", &PackSequenceChunksFast, nb::arg("items"),
        nb::arg("carried_names"), nb::arg("budget"), nb::arg("pack_size"),
        nb::arg("max_segments"), nb::arg("pad_id"),
        nb::arg("segment_alignment_boundary"),
        nb::arg("min_buffered_tokens") = 0);
  m.def("assemble_padded_chunk_fast", &AssemblePaddedChunkFast,
        nb::arg("chunk"), nb::arg("present_names"), nb::arg("batch_size"),
        nb::arg("max_prompt_len"), nb::arg("max_response_len"),
        nb::arg("pad_id"), nb::arg("replay_routing"));
  m.def("to_pack_item_fast", &ToPackItemFast, nb::arg("item"),
        nb::arg("pack_item_cls"));
  m.def("ingest_payloads_fast", &IngestPayloadsFast, nb::arg("items"),
        nb::arg("max_packed_len"), nb::arg("pack_item_cls"),
        nb::arg("buffer_out"));
  m.def("count_router_replay_forced_tokens", &CountRouterReplayForcedTokens,
        nb::arg("routed_experts"), nb::arg("segment_ids"));
}

}  // namespace tunix::rl
