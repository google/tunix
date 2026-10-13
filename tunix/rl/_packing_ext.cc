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

#include <pthread.h>
#if defined(__linux__)
#include <sched.h>
#endif
#include <unistd.h>

#include <algorithm>
#include <condition_variable>  // NOLINT(build/c++11)
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>  // NOLINT(build/c++11)
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

enum class RoutedEncoding : uint8_t {
  kNone = 0,
  kUint8 = 1,  // uint8_t: unbatched [T, L, K], batched [B, T, L, K]
  kUint10 =
      2,       // uint8_t: unbatched [T, L, K//4, 5], batched [B, T, L, K//4, 5]
  kInt16 = 3,  // int16_t: unbatched [T, L, K], batched [B, T, L, K]
};

struct SegmentView {
  const int32_t* prompt_ptr = nullptr;
  int64_t p_len = 0;
  const int32_t* comp_ptr = nullptr;
  int64_t c_len = 0;
  const float* mask_ptr = nullptr;
  const float* adv_ptr = nullptr;
  std::vector<const float*> per_token_ptrs;
  const void* routed_ptr = nullptr;
  int64_t routed_tokens = 0;
  int64_t routed_layers = 0;
  int64_t routed_k = 0;
  RoutedEncoding routed_encoding = RoutedEncoding::kNone;
};

struct RawArrayView {
  const void* ptr = nullptr;
  int64_t size = 0;
};

inline size_t RoutedTokenBytes(RoutedEncoding enc, int64_t layers,
                               int64_t k) noexcept {
  if (layers <= 0 || k <= 0) {
    return 0;
  }
  const size_t lk = static_cast<size_t>(layers * k);
  switch (enc) {
    case RoutedEncoding::kUint8:
      return lk;
    case RoutedEncoding::kUint10:
      return static_cast<size_t>(layers * (k / 4) * 5);
    case RoutedEncoding::kInt16:
      return lk * sizeof(int16_t);
    case RoutedEncoding::kNone:
      return 0;
  }
  return 0;
}

inline void PackFourToUint10(uint16_t v0, uint16_t v1, uint16_t v2, uint16_t v3,
                             uint8_t* dst) noexcept {
  dst[0] = static_cast<uint8_t>(v0 & 0xFF);
  dst[1] = static_cast<uint8_t>(((v0 >> 8) & 0x03) | ((v1 & 0x3F) << 2));
  dst[2] = static_cast<uint8_t>(((v1 >> 6) & 0x0F) | ((v2 & 0x0F) << 4));
  dst[3] = static_cast<uint8_t>(((v2 >> 4) & 0x3F) | ((v3 & 0x03) << 6));
  dst[4] = static_cast<uint8_t>((v3 >> 2) & 0xFF);
}

inline void UnpackUint10ToFour(const uint8_t* src, uint16_t* v0, uint16_t* v1,
                               uint16_t* v2, uint16_t* v3) noexcept {
  const uint16_t b0 = src[0];
  const uint16_t b1 = src[1];
  const uint16_t b2 = src[2];
  const uint16_t b3 = src[3];
  const uint16_t b4 = src[4];
  *v0 = b0 | ((b1 & 0x03) << 8);
  *v1 = (b1 >> 2) | ((b2 & 0x0F) << 6);
  *v2 = (b2 >> 4) | ((b3 & 0x3F) << 4);
  *v3 = (b3 >> 6) | (b4 << 2);
}

inline void UnpackUint10ToFour(const uint8_t* src, uint16_t* dst) noexcept {
  UnpackUint10ToFour(src, &dst[0], &dst[1], &dst[2], &dst[3]);
}

inline void PadUnsetRoutedBytes(uint8_t* dst_base, RoutedEncoding enc,
                                int64_t start_tok, int64_t num_tokens,
                                int64_t layers, int64_t k) noexcept {
  const size_t tok_bytes = RoutedTokenBytes(enc, layers, k);
  if (tok_bytes == 0 || num_tokens <= 0) {
    return;
  }
  uint8_t* ptr = dst_base + static_cast<size_t>(start_tok) * tok_bytes;
  const size_t nbytes = static_cast<size_t>(num_tokens) * tok_bytes;
  // For kUint8 and kUint10, UNSET_ROUTED_EXPERT (-1 + 1 = 0) is 0x00 in every
  // byte. For two's-complement kInt16, -1 is 0xFFFF (0xFF in every byte).
  const int fill_byte = (enc == RoutedEncoding::kInt16) ? 0xFF : 0x00;
  std::memset(ptr, fill_byte, nbytes);
}

inline RoutedEncoding ClassifyUint8RoutedEncoding(
    const nb::ndarray<nb::numpy, const uint8_t, nb::c_contig>& u8_arr,
    bool batched) {
  const size_t ndim = u8_arr.ndim();
  if (batched) {
    if (ndim == 5 && u8_arr.shape(4) == 5) {
      return RoutedEncoding::kUint10;
    }
    if (ndim == 4) {
      return RoutedEncoding::kUint8;
    }
    throw std::invalid_argument(
        "batched uint8 routed_experts must have 4D (B, T, L, K) or 5D "
        "(B, T, L, K // 4, 5) shape, got ndim=" +
        std::to_string(ndim) + ".");
  }
  if (ndim == 4 && u8_arr.shape(3) == 5) {
    return RoutedEncoding::kUint10;
  }
  if (ndim == 3) {
    return RoutedEncoding::kUint8;
  }
  throw std::invalid_argument(
      "unbatched uint8 routed_experts must have 3D (T, L, K) or 4D "
      "(T, L, K // 4, 5) shape, got ndim=" +
      std::to_string(ndim) + ".");
}

inline void CopyOrTranscodeRoutedTokens(const void* src_base,
                                        RoutedEncoding src_enc,
                                        int64_t src_tok_offset,
                                        uint8_t* dst_base,
                                        RoutedEncoding dst_enc,
                                        int64_t dst_tok_offset,
                                        int64_t num_tokens, int64_t layers,
                                        int64_t k) {
  if (num_tokens <= 0 || layers <= 0 || k <= 0) {
    return;
  }
  const size_t src_tok_bytes = RoutedTokenBytes(src_enc, layers, k);
  const size_t dst_tok_bytes = RoutedTokenBytes(dst_enc, layers, k);
  const uint8_t* src_ptr = static_cast<const uint8_t*>(src_base) +
                           static_cast<size_t>(src_tok_offset) * src_tok_bytes;
  uint8_t* dst_ptr =
      dst_base + static_cast<size_t>(dst_tok_offset) * dst_tok_bytes;

  if (src_enc == dst_enc) {
    std::memcpy(dst_ptr, src_ptr,
                static_cast<size_t>(num_tokens) * dst_tok_bytes);
    return;
  }
  if (src_enc == RoutedEncoding::kUint8 && dst_enc == RoutedEncoding::kUint10) {
    if (k % 4 != 0) {
      throw std::invalid_argument(
          "CopyOrTranscodeRoutedTokens: uint10 target requires k % 4 == 0.");
    }
    const size_t num_groups =
        static_cast<size_t>(num_tokens * layers * (k / 4));
    for (size_t g = 0; g < num_groups; ++g) {
      const uint8_t* s = src_ptr + g * 4;
      PackFourToUint10(s[0], s[1], s[2], s[3], dst_ptr + g * 5);
    }
    return;
  }
  if (src_enc == RoutedEncoding::kUint8 && dst_enc == RoutedEncoding::kInt16) {
    const size_t total = static_cast<size_t>(num_tokens * layers * k);
    int16_t* d16 = reinterpret_cast<int16_t*>(dst_ptr);
    for (size_t i = 0; i < total; ++i) {
      d16[i] = static_cast<int16_t>(src_ptr[i]) - 1;
    }
    return;
  }
  if (src_enc == RoutedEncoding::kUint10 && dst_enc == RoutedEncoding::kInt16) {
    if (k % 4 != 0) {
      throw std::invalid_argument(
          "CopyOrTranscodeRoutedTokens: uint10 source requires k % 4 == 0.");
    }
    const size_t num_groups =
        static_cast<size_t>(num_tokens * layers * (k / 4));
    int16_t* d16 = reinterpret_cast<int16_t*>(dst_ptr);
    for (size_t g = 0; g < num_groups; ++g) {
      uint16_t v0, v1, v2, v3;
      UnpackUint10ToFour(src_ptr + g * 5, &v0, &v1, &v2, &v3);
      d16[g * 4 + 0] = static_cast<int16_t>(v0) - 1;
      d16[g * 4 + 1] = static_cast<int16_t>(v1) - 1;
      d16[g * 4 + 2] = static_cast<int16_t>(v2) - 1;
      d16[g * 4 + 3] = static_cast<int16_t>(v3) - 1;
    }
    return;
  }
  throw std::invalid_argument(
      "Unsupported routed_experts transcoding pair in "
      "CopyOrTranscodeRoutedTokens.");
}

inline RoutedEncoding ResolveChunkRoutedEncoding(bool has_u8, bool has_u10,
                                                 bool has_i16, int64_t layers,
                                                 int64_t k) noexcept {
  if (layers <= 0 || k <= 0) {
    return RoutedEncoding::kNone;
  }
  if (has_i16) {
    return RoutedEncoding::kInt16;
  }
  if (has_u10) {
    return (k % 4 == 0) ? RoutedEncoding::kUint10 : RoutedEncoding::kInt16;
  }
  if (has_u8) {
    return RoutedEncoding::kUint8;
  }
  return RoutedEncoding::kInt16;
}

inline RoutedEncoding ResolveChunkRoutedEncoding(
    const std::vector<std::vector<const SegmentView*>>& bins_views,
    int64_t* out_layers, int64_t* out_k) {
  bool has_u8 = false;
  bool has_u10 = false;
  bool has_i16 = false;
  int64_t layers = *out_layers;
  int64_t k = *out_k;
  for (const auto& bin : bins_views) {
    for (const SegmentView* v : bin) {
      if (v->routed_ptr != nullptr &&
          v->routed_encoding != RoutedEncoding::kNone) {
        if (layers == 0 && k == 0) {
          layers = v->routed_layers;
          k = v->routed_k;
        } else if (layers != v->routed_layers || k != v->routed_k) {
          throw std::invalid_argument(
              "Items disagree on routed_experts trailing shape.");
        }
        if (v->routed_encoding == RoutedEncoding::kUint8) {
          has_u8 = true;
        } else if (v->routed_encoding == RoutedEncoding::kUint10) {
          has_u10 = true;
        } else if (v->routed_encoding == RoutedEncoding::kInt16) {
          has_i16 = true;
        }
      }
    }
  }
  *out_layers = layers;
  *out_k = k;
  return ResolveChunkRoutedEncoding(has_u8, has_u10, has_i16, layers, k);
}

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
    views_.emplace_back();
    Py_buffer& view = views_.back();
    if (PyObject_GetBuffer(obj, &view, flags) != 0) {
      views_.pop_back();
      throw nb::python_error();
    }
    return view;
  }

 private:
  std::deque<Py_buffer> views_;
};

inline RawArrayView Extract1DArrayPtr(PyObject* obj, size_t elem_size,
                                      ScopedPyBufferList* buffers) {
  const Py_buffer& view = buffers->Acquire(obj, PyBUF_SIMPLE);
  const int64_t num_elems =
      static_cast<int64_t>(static_cast<size_t>(view.len) / elem_size);
  return RawArrayView{view.buf, num_elems};
}

inline bool IsUint8BufferFormat(const Py_buffer& view) noexcept {
  return view.itemsize == 1 && view.format != nullptr &&
         (view.format[0] == 'B' ||
          (view.format[0] == '@' && view.format[1] == 'B') ||
          (view.format[0] == '=' && view.format[1] == 'B'));
}

inline bool IsInt16BufferFormat(const Py_buffer& view) noexcept {
  return view.itemsize == 2 && view.format != nullptr &&
         (view.format[0] == 'h' ||
          (view.format[0] == '@' && view.format[1] == 'h') ||
          (view.format[0] == '=' && view.format[1] == 'h') ||
          (view.format[0] == '<' && view.format[1] == 'h'));
}

inline void ExtractRoutedView(PyObject* obj, SegmentView* out,
                              ScopedPyBufferList* buffers) {
  const Py_buffer& view =
      buffers->Acquire(obj, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT);
  if (view.ndim < 3 || view.shape == nullptr) {
    throw std::invalid_argument(
        "routed_experts array must have at least 3 dimensions.");
  }
  if (view.ndim == 3 && IsUint8BufferFormat(view)) {
    out->routed_ptr = view.buf;
    out->routed_tokens = static_cast<int64_t>(view.shape[0]);
    out->routed_layers = static_cast<int64_t>(view.shape[1]);
    out->routed_k = static_cast<int64_t>(view.shape[2]);
    out->routed_encoding = RoutedEncoding::kUint8;
    return;
  }
  if (view.ndim == 4 && IsUint8BufferFormat(view) && view.shape[3] == 5) {
    out->routed_ptr = view.buf;
    out->routed_tokens = static_cast<int64_t>(view.shape[0]);
    out->routed_layers = static_cast<int64_t>(view.shape[1]);
    out->routed_k = static_cast<int64_t>(view.shape[2]) * 4;
    out->routed_encoding = RoutedEncoding::kUint10;
    return;
  }
  if (view.ndim == 3 && IsInt16BufferFormat(view)) {
    out->routed_ptr = view.buf;
    out->routed_tokens = static_cast<int64_t>(view.shape[0]);
    out->routed_layers = static_cast<int64_t>(view.shape[1]);
    out->routed_k = static_cast<int64_t>(view.shape[2]);
    out->routed_encoding = RoutedEncoding::kInt16;
    return;
  }
  throw std::invalid_argument(
      "routed_experts array must be C-contiguous int16/uint8 [tokens, layers, "
      "k] or uint8 [tokens, layers, k // 4, 5].");
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
nb::object Wrap3DArray(RawBufferPtr<T> buf, size_t d0, size_t d1, size_t d2) {
  T* raw = buf.release();
  nb::capsule deleter(raw, [](void* p) noexcept { RawFreeBytes(p); });
  size_t shape[3] = {d0, d1, d2};
  return nb::ndarray<nb::numpy, T, nb::ndim<3>>(raw, 3, shape, deleter).cast();
}

template <typename T>
nb::object Wrap4DArray(RawBufferPtr<T> buf, size_t d0, size_t d1, size_t d2,
                       size_t d3) {
  T* raw = buf.release();
  nb::capsule deleter(raw, [](void* p) noexcept { RawFreeBytes(p); });
  size_t shape[4] = {d0, d1, d2, d3};
  return nb::ndarray<nb::numpy, T, nb::ndim<4>>(raw, 4, shape, deleter).cast();
}

template <typename T>
nb::object Wrap5DArray(RawBufferPtr<T> buf, size_t d0, size_t d1, size_t d2,
                       size_t d3, size_t d4) {
  T* raw = buf.release();
  nb::capsule deleter(raw, [](void* p) noexcept { RawFreeBytes(p); });
  size_t shape[5] = {d0, d1, d2, d3, d4};
  return nb::ndarray<nb::numpy, T, nb::ndim<5>>(raw, 5, shape, deleter).cast();
}

inline nb::object WrapBatchedRoutedBuffer(RawBufferPtr<uint8_t> buf,
                                          RoutedEncoding enc, size_t b_sz,
                                          size_t seq_sz, size_t layers,
                                          size_t k) {
  if (buf == nullptr || enc == RoutedEncoding::kNone) {
    return nb::none();
  }
  if (enc == RoutedEncoding::kUint8) {
    return Wrap4DArray<uint8_t>(std::move(buf), b_sz, seq_sz, layers, k);
  }
  if (enc == RoutedEncoding::kUint10) {
    return Wrap5DArray<uint8_t>(std::move(buf), b_sz, seq_sz, layers, k / 4, 5);
  }
  int16_t* raw16 = reinterpret_cast<int16_t*>(buf.release());
  RawBufferPtr<int16_t> buf16(raw16);
  return Wrap4DArray<int16_t>(std::move(buf16), b_sz, seq_sz, layers, k);
}

struct RawChunkBuffers {
  size_t n_bins = 0;
  size_t budget = 0;
  int64_t routed_layers = 0;
  int64_t routed_k = 0;
  RoutedEncoding routed_encoding = RoutedEncoding::kNone;
  RawBufferPtr<int32_t> ids;
  RawBufferPtr<float> prompt_mask;
  RawBufferPtr<float> completion_mask;
  RawBufferPtr<float> advantages;
  RawBufferPtr<int32_t> segment_ids;
  RawBufferPtr<int32_t> segment_positions;
  std::vector<RawBufferPtr<float>> per_token;
  RawBufferPtr<uint8_t> routed_experts_bytes;
  std::vector<std::vector<int64_t>> bin_indices;
};

// Fork-safe persistent worker pool for parallel row population in
// `AllocateAndPopulateChunkNogil` and `AssemblePaddedChunkFast`.
//
// Invariants and concurrency contract:
// 1. Concurrent callers to `ParallelFor` serialize on the process-wide mutex
//    `g_pool_mu_`.
// 2. `fn` must NOT re-enter `ParallelFor` (`g_pool_mu_` is non-recursive and
//    would self-deadlock) and must NOT call the Python C-API (the GIL is
//    released before entering `ParallelFor`).
// 3. If any worker or the calling thread throws an exception inside `fn`, the
//    first exception is captured and rethrown on the calling thread only after
//    all active workers complete the current generation barrier.
// 4. The singleton pool (`g_pool_`) and its worker threads are intentionally
//    leaked at process exit to avoid static destruction order hazards. Across
//    `fork()`, the `pthread_atfork` child handler resets `g_pool_mu_` and
//    `g_pool_` so the child lazily initializes a fresh pool; `pid_ != cur_pid`
//    serves as a secondary fallback if `pthread_atfork` registration fails.
class PackingWorkerPool {
 public:
  static constexpr size_t kMaxThreads = 16;
  static constexpr size_t kMinBytesPerThread = 2 * 1024 * 1024;  // 2 MiB
  static constexpr size_t kMinParallelBytes = 2 * kMinBytesPerThread;  // 4 MiB

  PackingWorkerPool(const PackingWorkerPool&) = delete;
  PackingWorkerPool& operator=(const PackingWorkerPool&) = delete;
  PackingWorkerPool(PackingWorkerPool&&) = delete;
  PackingWorkerPool& operator=(PackingWorkerPool&&) = delete;

  static void ParallelFor(size_t num_rows, size_t total_bytes,
                          const std::function<void(size_t, size_t)>& fn) {
    if (num_rows < 2 || total_bytes < kMinParallelBytes) {
      fn(0, num_rows);
      return;
    }
    EnsureAtforkRegistered();
    std::lock_guard<PoolMutex> pool_lk(*g_pool_mu_);  // NOLINT(build/c++11)
    const pid_t cur_pid = getpid();
    if (g_pool_ == nullptr || g_pool_->pid_ != cur_pid) {
      g_pool_ = new PackingWorkerPool(cur_pid, g_configured_threads_);
    }
    g_pool_->RunLocked(num_rows, total_bytes, fn);
  }

  // Initializes `g_configured_threads_` from CPU affinity / hardware
  // concurrency during `NB_MODULE` initialization.
  static void InitDefaultThreads() noexcept {
    g_configured_threads_ = DetectHardwareThreads();
  }

  static int64_t GetConfiguredThreads() {
    EnsureAtforkRegistered();
    std::lock_guard<PoolMutex> pool_lk(*g_pool_mu_);  // NOLINT(build/c++11)
    return static_cast<int64_t>(g_configured_threads_);
  }

  static void SetConfiguredThreads(int64_t num_threads) {
    if (num_threads < 1 || num_threads > static_cast<int64_t>(kMaxThreads)) {
      throw std::invalid_argument(
          "num_threads must be in [1, " + std::to_string(kMaxThreads) +
          "], got " + std::to_string(num_threads) + ".");
    }
    EnsureAtforkRegistered();
    std::lock_guard<PoolMutex> pool_lk(*g_pool_mu_);  // NOLINT(build/c++11)
    const size_t target = static_cast<size_t>(num_threads);
    g_configured_threads_ = target;
    if (g_pool_ != nullptr) {
      if (g_pool_->pid_ == getpid()) {
        if (g_pool_->num_threads_ != target) {
          g_pool_->ShutdownLaunchedWorkers();
          delete g_pool_;
          g_pool_ = nullptr;
        }
      } else {
        g_pool_ = nullptr;
      }
    }
  }

 private:
  static size_t DetectHardwareThreads() noexcept {
#if defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
      const int affinity_cpus = CPU_COUNT(&cpuset);
      if (affinity_cpus > 0) {
        return std::clamp<size_t>(static_cast<size_t>(affinity_cpus), 1,
                                  kMaxThreads);
      }
    }
#endif
    const unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) {
      return kMaxThreads;
    }
    return std::clamp<size_t>(static_cast<size_t>(hw), 1, kMaxThreads);
  }

  static void EnsureAtforkRegistered() {
    static std::once_flag atfork_once;  // NOLINT(build/c++11)
    std::call_once(atfork_once, []() {  // NOLINT(build/c++11)
      // Best-effort registration; if `pthread_atfork` returns non-zero, the
      // `g_pool_->pid_ != getpid()` check in `ParallelFor` provides a fallback
      // when `g_pool_mu_` was not locked across fork.
      (void)pthread_atfork(nullptr, nullptr, &PackingWorkerPool::OnForkChild);
    });
  }

  static void OnForkChild() {
    g_pool_mu_ = new PoolMutex();
    g_pool_ = nullptr;
  }

  struct WorkerSlot {
    std::condition_variable cv;
    uint64_t gen = 0;
    std::thread thread;
  };

  PackingWorkerPool(pid_t pid, size_t num_threads)
      : pid_(pid), num_threads_(num_threads) {
    if (num_threads_ <= 1) {
      return;
    }
    const size_t num_bg = num_threads_ - 1;
    try {
      for (size_t t = 0; t < num_bg; ++t) {
        workers_.emplace_back();
        workers_.back().thread = std::thread([this, t]() { WorkerLoop(t); });
      }
    } catch (...) {
      ShutdownLaunchedWorkers();
      throw;
    }
  }

  void ShutdownLaunchedWorkers() noexcept {
    {
      std::lock_guard<PoolMutex> lk(mu_);  // NOLINT(build/c++11)
      stop_ = true;
    }
    for (WorkerSlot& slot : workers_) {
      slot.cv.notify_one();
    }
    for (WorkerSlot& slot : workers_) {
      if (slot.thread.joinable()) {
        slot.thread.join();
      }
    }
    workers_.clear();
  }

  void WorkerLoop(size_t slot_idx) {
    WorkerSlot& slot = workers_[slot_idx];
    const size_t worker_idx = slot_idx + 1;
    uint64_t seen_gen = 0;
    while (true) {
      const std::function<void(size_t, size_t)>* fn_ptr = nullptr;
      size_t r_start = 0;
      size_t r_end = 0;
      {
        std::unique_lock<PoolMutex> lk(mu_);  // NOLINT(build/c++11)
        slot.cv.wait(lk, [&]() { return stop_ || slot.gen > seen_gen; });
        if (stop_) {
          return;
        }
        seen_gen = slot.gen;
        fn_ptr = current_fn_;
        r_start = (num_rows_ * worker_idx) / active_threads_;
        r_end = (num_rows_ * (worker_idx + 1)) / active_threads_;
      }
      std::exception_ptr local_exc;
      if (fn_ptr != nullptr && r_start < r_end) {
        try {
          (*fn_ptr)(r_start, r_end);
        } catch (...) {
          local_exc = std::current_exception();
        }
      }
      {
        std::lock_guard<PoolMutex> lk(mu_);  // NOLINT(build/c++11)
        if (local_exc && !worker_exc_) {
          worker_exc_ = local_exc;
        }
        if (--remaining_workers_ == 0) {
          cv_done_.notify_one();
        }
      }
    }
  }

  void RunLocked(size_t num_rows, size_t total_bytes,
                 const std::function<void(size_t, size_t)>& fn) {
    const size_t byte_threads =
        std::max<size_t>(1, total_bytes / kMinBytesPerThread);
    const size_t n_threads = std::min({num_threads_, num_rows, byte_threads});
    if (n_threads <= 1) {
      fn(0, num_rows);
      return;
    }
    const size_t num_active_bg = n_threads - 1;
    {
      std::lock_guard<PoolMutex> lk(mu_);  // NOLINT(build/c++11)
      current_fn_ = &fn;
      num_rows_ = num_rows;
      active_threads_ = n_threads;
      remaining_workers_ = num_active_bg;
      worker_exc_ = nullptr;
      for (size_t t = 0; t < num_active_bg; ++t) {
        ++workers_[t].gen;
      }
    }
    for (size_t t = 0; t < num_active_bg; ++t) {
      workers_[t].cv.notify_one();
    }
    const size_t main_end = num_rows / n_threads;
    std::exception_ptr main_exc;
    if (main_end > 0) {
      try {
        fn(0, main_end);
      } catch (...) {
        main_exc = std::current_exception();
      }
    }
    std::exception_ptr rethrow_exc;
    {
      std::unique_lock<PoolMutex> lk(mu_);  // NOLINT(build/c++11)
      cv_done_.wait(lk, [&]() { return remaining_workers_ == 0; });
      current_fn_ = nullptr;
      rethrow_exc = main_exc ? main_exc : worker_exc_;
      worker_exc_ = nullptr;
    }
    if (rethrow_exc) {
      std::rethrow_exception(rethrow_exc);
    }
  }

  using PoolMutex = std::mutex;  // NOLINT(build/c++11)
  static inline PoolMutex* g_pool_mu_ = new PoolMutex();
  static inline PackingWorkerPool* g_pool_ = nullptr;
  static inline size_t g_configured_threads_ = kMaxThreads;

  pid_t pid_;
  size_t num_threads_;
  PoolMutex mu_;
  std::condition_variable cv_done_;
  std::deque<WorkerSlot> workers_;
  const std::function<void(size_t, size_t)>* current_fn_ = nullptr;
  size_t num_rows_ = 0;
  size_t active_threads_ = 0;
  size_t remaining_workers_ = 0;
  std::exception_ptr worker_exc_;
  bool stop_ = false;
};

// Populates a single packed row `b` inside `AllocateAndPopulateChunkNogil`.
void PopulatePackedRow(size_t b,
                       const std::vector<const SegmentView*>& seg_views,
                       size_t n_carried, int64_t budget, int32_t pad_id,
                       int64_t segment_alignment_boundary,
                       int64_t routed_layers, int64_t routed_k,
                       RoutedEncoding routed_encoding, size_t routed_tok_bytes,
                       RawChunkBuffers* out) {
  const int64_t row_offset = static_cast<int64_t>(b) * budget;
  int32_t* ids_row = out->ids.get() + row_offset;
  float* pmask_row = out->prompt_mask.get() + row_offset;
  float* cmask_row = out->completion_mask.get() + row_offset;
  float* adv_row = out->advantages.get() + row_offset;
  int32_t* seg_ids_row = out->segment_ids.get() + row_offset;
  int32_t* seg_pos_row = out->segment_positions.get() + row_offset;
  uint8_t* routed_base = out->routed_experts_bytes.get();
  uint8_t* routed_row =
      (routed_base != nullptr)
          ? (routed_base + static_cast<size_t>(row_offset) * routed_tok_bytes)
          : nullptr;

  auto zero_pad_span = [&](int64_t start, int64_t len) {
    const size_t len_sz = static_cast<size_t>(len);
    std::fill_n(ids_row + start, len, pad_id);
    std::memset(pmask_row + start, 0, len_sz * sizeof(float));
    std::memset(cmask_row + start, 0, len_sz * sizeof(float));
    std::memset(adv_row + start, 0, len_sz * sizeof(float));
    std::memset(seg_ids_row + start, 0, len_sz * sizeof(int32_t));
    std::memset(seg_pos_row + start, 0, len_sz * sizeof(int32_t));
    for (size_t k = 0; k < n_carried; ++k) {
      std::memset(out->per_token[k].get() + row_offset + start, 0,
                  len_sz * sizeof(float));
    }
    if (routed_row != nullptr) {
      PadUnsetRoutedBytes(routed_row, routed_encoding, start, len,
                          routed_layers, routed_k);
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
      throw std::invalid_argument("pack_bin: bin size " + std::to_string(end) +
                                  " exceeds budget " + std::to_string(budget) +
                                  ".");
    }
    if (v.routed_ptr != nullptr && v.routed_tokens > n) {
      throw std::invalid_argument("PackItem.routed_experts token length " +
                                  std::to_string(v.routed_tokens) +
                                  " exceeds segment token length " +
                                  std::to_string(n) + ".");
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
        std::memset(out->per_token[k].get() + row_offset + cursor, 0,
                    p_sz * sizeof(float));
      }
    }
    if (c > 0) {
      std::memcpy(ids_row + comp_start, v.comp_ptr, c_sz * sizeof(int32_t));
      std::memset(pmask_row + comp_start, 0, c_sz * sizeof(float));
      std::memcpy(cmask_row + comp_start, v.mask_ptr, c_sz * sizeof(float));
      std::memcpy(adv_row + comp_start, v.adv_ptr, c_sz * sizeof(float));
      for (size_t k = 0; k < n_carried; ++k) {
        std::memcpy(out->per_token[k].get() + row_offset + comp_start,
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
        CopyOrTranscodeRoutedTokens(v.routed_ptr, v.routed_encoding,
                                    /*src_tok_offset=*/0, routed_row,
                                    routed_encoding,
                                    /*dst_tok_offset=*/cursor, m, routed_layers,
                                    routed_k);
      }
      if (m < n) {
        PadUnsetRoutedBytes(routed_row, routed_encoding, cursor + m, n - m,
                            routed_layers, routed_k);
      }
    }
    cursor = end;
  }

  if (cursor < budget) {
    zero_pad_span(cursor, budget - cursor);
  }
}

// Allocates uninitialized buffers and populates all rows in parallel across
// `PackingWorkerPool` without holding the GIL. Every output byte is written at
// most once (no redundant full-matrix memset before memcpy), so each worker
// thread faults and populates its own disjoint row pages in parallel.
RawChunkBuffers AllocateAndPopulateChunkNogil(
    const std::vector<std::vector<const SegmentView*>>& bins_views,
    size_t n_carried, int64_t budget, int32_t pad_id,
    int64_t segment_alignment_boundary, int64_t routed_layers, int64_t routed_k,
    RoutedEncoding routed_encoding) {
  const size_t n_bins = bins_views.size();
  const size_t total_elems = n_bins * static_cast<size_t>(budget);

  RawChunkBuffers out;
  out.n_bins = n_bins;
  out.budget = static_cast<size_t>(budget);
  out.routed_layers = routed_layers;
  out.routed_k = routed_k;
  out.routed_encoding = routed_encoding;

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

  const size_t routed_tok_bytes =
      RoutedTokenBytes(routed_encoding, routed_layers, routed_k);
  if (routed_tok_bytes > 0) {
    out.routed_experts_bytes =
        AllocRawBuffer<uint8_t>(total_elems * routed_tok_bytes);
  }

  const size_t bytes_per_token =
      (6 + n_carried) * sizeof(float) + routed_tok_bytes;
  const size_t total_bytes = total_elems * bytes_per_token;

  PackingWorkerPool::ParallelFor(
      n_bins, total_bytes, [&](size_t b_start, size_t b_end) {
        for (size_t b = b_start; b < b_end; ++b) {
          PopulatePackedRow(b, bins_views[b], n_carried, budget, pad_id,
                            segment_alignment_boundary, routed_layers, routed_k,
                            routed_encoding, routed_tok_bytes, &out);
        }
      });

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

  nb::object routed_obj = WrapBatchedRoutedBuffer(
      std::move(raw.routed_experts_bytes), raw.routed_encoding, n_bins, budget,
      static_cast<size_t>(raw.routed_layers),
      static_cast<size_t>(raw.routed_k));

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
    RoutedEncoding routed_encoding = RoutedEncoding::kNone;
    if (routed_layers * routed_k > 0) {
      int64_t resolved_layers = routed_layers;
      int64_t resolved_k = routed_k;
      routed_encoding =
          ResolveChunkRoutedEncoding(bins_views, &resolved_layers, &resolved_k);
      if (routed_encoding == RoutedEncoding::kNone) {
        routed_encoding = RoutedEncoding::kInt16;
      }
    }
    raw = AllocateAndPopulateChunkNogil(
        bins_views, n_carried, budget, pad_id, segment_alignment_boundary,
        routed_layers, routed_k, routed_encoding);
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
      for (size_t b = 0; b < n_bins; ++b) {
        const auto& b_idxs = bin_indices[b];
        bins_views[b].reserve(b_idxs.size());
        for (int64_t idx : b_idxs) {
          const SegmentView& v = item_views[static_cast<size_t>(idx)];
          bins_views[b].push_back(&v);
        }
      }

      int64_t chunk_routed_layers = 0;
      int64_t chunk_routed_k = 0;
      const RoutedEncoding chunk_routed_encoding = ResolveChunkRoutedEncoding(
          bins_views, &chunk_routed_layers, &chunk_routed_k);

      RawChunkBuffers raw = AllocateAndPopulateChunkNogil(
          bins_views, n_carried, budget, pad_id, segment_alignment_boundary,
          chunk_routed_layers, chunk_routed_k, chunk_routed_encoding);
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
  const uint8_t* routed_ptr = nullptr;
  RoutedEncoding routed_encoding = RoutedEncoding::kNone;
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

inline void ExtractPaddedRoutedView(nb::object r_obj, PaddedItemView* v,
                                    std::vector<nb::object>* keep) {
  nb::ndarray<nb::numpy, const uint8_t, nb::ndim<3>, nb::c_contig> u8_3d;
  if (nb::try_cast(r_obj, u8_3d, /*convert=*/false)) {
    v->routed_ptr = u8_3d.data();
    v->routed_encoding = RoutedEncoding::kUint8;
    v->routed_tokens = static_cast<int64_t>(u8_3d.shape(0));
    v->routed_layers = static_cast<int64_t>(u8_3d.shape(1));
    v->routed_k = static_cast<int64_t>(u8_3d.shape(2));
    keep->push_back(std::move(r_obj));
    return;
  }
  nb::ndarray<nb::numpy, const uint8_t, nb::ndim<4>, nb::c_contig> u8_4d;
  if (nb::try_cast(r_obj, u8_4d, /*convert=*/false) && u8_4d.shape(3) == 5) {
    v->routed_ptr = u8_4d.data();
    v->routed_encoding = RoutedEncoding::kUint10;
    v->routed_tokens = static_cast<int64_t>(u8_4d.shape(0));
    v->routed_layers = static_cast<int64_t>(u8_4d.shape(1));
    v->routed_k = static_cast<int64_t>(u8_4d.shape(2)) * 4;
    keep->push_back(std::move(r_obj));
    return;
  }
  nb::ndarray<nb::numpy, const int16_t, nb::ndim<3>, nb::c_contig> i16_3d;
  if (!nb::try_cast(r_obj, i16_3d, /*convert=*/false)) {
    r_obj = CoerceToContiguousNumpy(r_obj.ptr(), "int16");
    i16_3d = nb::cast<
        nb::ndarray<nb::numpy, const int16_t, nb::ndim<3>, nb::c_contig>>(
        r_obj);
  }
  v->routed_ptr = reinterpret_cast<const uint8_t*>(i16_3d.data());
  v->routed_encoding = RoutedEncoding::kInt16;
  v->routed_tokens = static_cast<int64_t>(i16_3d.shape(0));
  v->routed_layers = static_cast<int64_t>(i16_3d.shape(1));
  v->routed_k = static_cast<int64_t>(i16_3d.shape(2));
  keep->push_back(std::move(r_obj));
}

struct PaddedRowCounts {
  int64_t valid_tokens = 0;
  int64_t num_sequences = 0;
  int64_t truncated_prompts = 0;
  int64_t truncated_completions = 0;
};

// Shared batch-level context for `PopulatePaddedRow`.
struct PaddedBatchContext {
  size_t n_items;
  const std::vector<PaddedItemView>* views;
  size_t n_opt;
  size_t p_sz;
  size_t c_sz;
  size_t seq_sz;
  int64_t max_prompt_len;
  int64_t max_response_len;
  int32_t pad_id;
  RoutedEncoding chunk_routed_encoding;
  int64_t routed_layers;
  int64_t routed_k;
  int64_t routed_tok_bytes;
  int32_t* prompt_ids_base;
  float* prompt_mask_base;
  int32_t* comp_ids_base;
  float* comp_mask_base;
  float* advantages_base;
  const std::vector<RawBufferPtr<float>>* opt_bufs;
  uint8_t* routed_base;
};

// Populates a single padded row `r` inside `AssemblePaddedChunkFast`.
PaddedRowCounts PopulatePaddedRow(size_t r, const PaddedBatchContext& ctx) {
  int32_t* p_row = ctx.prompt_ids_base + r * ctx.p_sz;
  float* pm_row = ctx.prompt_mask_base + r * ctx.p_sz;
  int32_t* c_row = ctx.comp_ids_base + r * ctx.c_sz;
  float* cm_row = ctx.comp_mask_base + r * ctx.c_sz;
  float* adv_row = ctx.advantages_base + r * ctx.c_sz;
  uint8_t* r_row = (ctx.routed_base != nullptr)
                       ? (ctx.routed_base +
                          static_cast<int64_t>(r * ctx.seq_sz) *
                              ctx.routed_tok_bytes)
                       : nullptr;

  PaddedRowCounts counts;
  if (r >= ctx.n_items) {
    std::fill_n(p_row, ctx.p_sz, ctx.pad_id);
    std::memset(pm_row, 0, ctx.p_sz * sizeof(float));
    std::fill_n(c_row, ctx.c_sz, ctx.pad_id);
    std::memset(cm_row, 0, ctx.c_sz * sizeof(float));
    std::memset(adv_row, 0, ctx.c_sz * sizeof(float));
    for (size_t k = 0; k < ctx.n_opt; ++k) {
      std::memset((*ctx.opt_bufs)[k].get() + r * ctx.c_sz, 0,
                  ctx.c_sz * sizeof(float));
    }
    if (r_row != nullptr) {
      PadUnsetRoutedBytes(r_row, ctx.chunk_routed_encoding, 0,
                          static_cast<int64_t>(ctx.seq_sz), ctx.routed_layers,
                          ctx.routed_k);
    }
    return counts;
  }

  const PaddedItemView& v = (*ctx.views)[r];
  if (v.p_len > ctx.max_prompt_len) {
    counts.truncated_prompts = 1;
  }
  if (v.c_len > ctx.max_response_len) {
    counts.truncated_completions = 1;
  }
  const int64_t p_keep = std::min(v.p_len, ctx.max_prompt_len);
  const int64_t p_pad = ctx.max_prompt_len - p_keep;
  const int64_t p_src_off = v.p_len - p_keep;
  const int64_t c_keep = std::min(v.c_len, ctx.max_response_len);
  const int64_t c_pad = ctx.max_response_len - c_keep;

  counts.valid_tokens = p_keep + c_keep;
  counts.num_sequences = 1;

  // Left-pad prompt_ids and prompt_mask.
  if (p_pad > 0) {
    std::fill_n(p_row, p_pad, ctx.pad_id);
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
    std::fill_n(c_row + c_keep, c_pad, ctx.pad_id);
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
                                 ctx.max_response_len, cm_row);
  }

  // Advantages & optional per-token fields.
  PopulateCompletionAlignedRow(v.advantages, v.p_len, v.c_len, c_keep,
                               ctx.max_response_len, adv_row);
  for (size_t k = 0; k < ctx.n_opt; ++k) {
    PopulateCompletionAlignedRow(v.optional_fields[k], v.p_len, v.c_len, c_keep,
                                 ctx.max_response_len,
                                 (*ctx.opt_bufs)[k].get() + r * ctx.c_sz);
  }

  // Single-pass routed experts alignment + padding (no upfront full-buffer
  // memset).
  if (r_row != nullptr) {
    if (v.routed_ptr == nullptr) {
      PadUnsetRoutedBytes(r_row, ctx.chunk_routed_encoding, 0,
                          static_cast<int64_t>(ctx.seq_sz), ctx.routed_layers,
                          ctx.routed_k);
    } else {
      const int64_t kept_p_start =
          std::max<int64_t>(v.p_len - ctx.max_prompt_len, 0);
      const int64_t kept_p_end = std::min(v.p_len, v.routed_tokens);
      const int64_t p_part_len =
          std::max<int64_t>(kept_p_end - kept_p_start, 0);
      const int64_t p_out_start = ctx.max_prompt_len - (v.p_len - kept_p_start);
      if (p_out_start > 0) {
        PadUnsetRoutedBytes(r_row, ctx.chunk_routed_encoding, 0, p_out_start,
                            ctx.routed_layers, ctx.routed_k);
      }
      if (p_part_len > 0) {
        CopyOrTranscodeRoutedTokens(
            v.routed_ptr, v.routed_encoding, kept_p_start, r_row,
            ctx.chunk_routed_encoding, p_out_start, p_part_len,
            ctx.routed_layers, ctx.routed_k);
      }
      const int64_t p_written_end = p_out_start + p_part_len;
      if (p_written_end < ctx.max_prompt_len) {
        PadUnsetRoutedBytes(r_row, ctx.chunk_routed_encoding, p_written_end,
                            ctx.max_prompt_len - p_written_end,
                            ctx.routed_layers, ctx.routed_k);
      }
      const int64_t kept_c_end = std::min(v.p_len + c_keep, v.routed_tokens);
      const int64_t c_part_len = std::max<int64_t>(kept_c_end - v.p_len, 0);
      if (c_part_len > 0) {
        CopyOrTranscodeRoutedTokens(
            v.routed_ptr, v.routed_encoding, v.p_len, r_row,
            ctx.chunk_routed_encoding, ctx.max_prompt_len, c_part_len,
            ctx.routed_layers, ctx.routed_k);
      }
      const int64_t c_written_end = ctx.max_prompt_len + c_part_len;
      const int64_t seq_len_i64 = static_cast<int64_t>(ctx.seq_sz);
      if (c_written_end < seq_len_i64) {
        PadUnsetRoutedBytes(r_row, ctx.chunk_routed_encoding, c_written_end,
                            seq_len_i64 - c_written_end, ctx.routed_layers,
                            ctx.routed_k);
      }
    }
  }
  return counts;
}
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
  RoutedEncoding chunk_routed_encoding = RoutedEncoding::kNone;

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
        ExtractPaddedRoutedView(std::move(r_obj), &v, &temp_keepalive);
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
        } else if (routed_layers != v.routed_layers || routed_k != v.routed_k) {
          throw std::invalid_argument(
              "Items disagree on routed_experts trailing shape.");
        }
        if (static_cast<int>(v.routed_encoding) >
            static_cast<int>(chunk_routed_encoding)) {
          chunk_routed_encoding = v.routed_encoding;
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
  RawBufferPtr<uint8_t> routed_buf;
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
    const int64_t routed_tok_bytes =
        RoutedTokenBytes(chunk_routed_encoding, routed_layers, routed_k);
    if (replay_routing && routed_tok_bytes > 0) {
      routed_buf = AllocRawBuffer<uint8_t>(
          b_sz * seq_sz * static_cast<size_t>(routed_tok_bytes));
    }
    row_valid_tokens = AllocRawBuffer<int64_t>(b_sz);
    row_num_sequences = AllocRawBuffer<int64_t>(b_sz);
    std::vector<int64_t> row_trunc_p(b_sz, 0);
    std::vector<int64_t> row_trunc_c(b_sz, 0);

    const PaddedBatchContext pad_ctx{
        n_items,
        &views,
        n_opt,
        p_sz,
        c_sz,
        seq_sz,
        max_prompt_len,
        max_response_len,
        pad_id,
        chunk_routed_encoding,
        routed_layers,
        routed_k,
        routed_tok_bytes,
        prompt_ids.get(),
        prompt_mask.get(),
        comp_ids.get(),
        comp_mask.get(),
        advantages.get(),
        &opt_bufs,
        routed_buf.get(),
    };
    const size_t bytes_per_row =
        p_sz * (sizeof(int32_t) + sizeof(float)) +
        c_sz * (sizeof(int32_t) + (2 + n_opt) * sizeof(float)) +
        (routed_buf ? seq_sz * static_cast<size_t>(routed_tok_bytes) : 0);
    const size_t total_bytes = b_sz * bytes_per_row;
    PackingWorkerPool::ParallelFor(
        b_sz, total_bytes, [&](size_t r_start, size_t r_end) {
          for (size_t r = r_start; r < r_end; ++r) {
            const PaddedRowCounts counts = PopulatePaddedRow(r, pad_ctx);
            row_valid_tokens[r] = counts.valid_tokens;
            row_num_sequences[r] = counts.num_sequences;
            row_trunc_p[r] = counts.truncated_prompts;
            row_trunc_c[r] = counts.truncated_completions;
          }
        });
    truncated_prompts =
        std::accumulate(row_trunc_p.begin(), row_trunc_p.end(), int64_t{0});
    truncated_completions =
        std::accumulate(row_trunc_c.begin(), row_trunc_c.end(), int64_t{0});
  }

  nb::dict opt_dict;
  for (size_t k = 0; k < n_opt; ++k) {
    opt_dict[opt_py_keys[k]] = Wrap2DArray(std::move(opt_bufs[k]), b_sz, c_sz);
  }
  nb::object routed_obj = WrapBatchedRoutedBuffer(
      std::move(routed_buf), chunk_routed_encoding, b_sz, seq_sz,
      static_cast<size_t>(routed_layers), static_cast<size_t>(routed_k));
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
    int64_t r_tokens = -1;
    int64_t r_layers = -1;
    int64_t r_k = -1;
    nb::ndarray<nb::numpy, const uint8_t, nb::ndim<3>, nb::c_contig> u8_3d;
    nb::ndarray<nb::numpy, const uint8_t, nb::ndim<4>, nb::c_contig> u8_4d;
    nb::ndarray<nb::numpy, const int16_t, nb::ndim<3>, nb::c_contig> r_arr;
    if (nb::try_cast(r_obj, u8_3d, /*convert=*/false)) {
      r_tokens = static_cast<int64_t>(u8_3d.shape(0));
      r_layers = static_cast<int64_t>(u8_3d.shape(1));
      r_k = static_cast<int64_t>(u8_3d.shape(2));
    } else if (nb::try_cast(r_obj, u8_4d, /*convert=*/false) &&
               u8_4d.shape(3) == 5) {
      r_tokens = static_cast<int64_t>(u8_4d.shape(0));
      r_layers = static_cast<int64_t>(u8_4d.shape(1));
      r_k = static_cast<int64_t>(u8_4d.shape(2)) * 4;
    } else if (nb::try_cast(r_obj, r_arr, /*convert=*/false)) {
      r_tokens = static_cast<int64_t>(r_arr.shape(0));
      r_layers = static_cast<int64_t>(r_arr.shape(1));
      r_k = static_cast<int64_t>(r_arr.shape(2));
    } else {
      r_obj = CoerceToContiguousNumpy(r_obj.ptr(), "int16");
      if (!nb::try_cast(r_obj, r_arr, /*convert=*/false)) {
        throw std::invalid_argument(
            "PackItem.routed_experts must be a 3D numpy array of shape "
            "(p + c - 1 or p + c, num_layers, top_k) or 4D uint10 array of "
            "shape (p + c - 1 or p + c, num_layers, top_k // 4, 5).");
      }
      r_tokens = static_cast<int64_t>(r_arr.shape(0));
      r_layers = static_cast<int64_t>(r_arr.shape(1));
      r_k = static_cast<int64_t>(r_arr.shape(2));
    }
    const int64_t n = p_len + c_len;
    const int64_t min_len = std::max<int64_t>(n - 1, 0);
    if (r_tokens < min_len || r_tokens > n) {
      throw std::invalid_argument(
          "PackItem.routed_experts must be a numpy array of shape "
          "(p + c - 1 or p + c, num_layers, top_k) with length in [" +
          std::to_string(min_len) + ", " + std::to_string(n) +
          "], got shape (" + std::to_string(r_tokens) + ", " +
          std::to_string(r_layers) + ", " + std::to_string(r_k) + ").");
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

nb::object DecodeRoutedExpertsFast(nb::object encoded_obj,
                                   bool batched = false);

// Fast C++ `nogil` encoder for `routed_experts` (`int16` 3D `(T, L, K)` or 4D
// `(B, T, L, K)` -> compact `uint8` / `uint10` / `int16`).
nb::object EncodeRoutedExpertsFast(nb::object routed_obj,
                                   const std::string& mode,
                                   nb::object num_experts_obj,
                                   bool batched = false) {
  if (routed_obj.is_none()) {
    return nb::none();
  }
  const bool has_num_experts = !num_experts_obj.is_none();
  if (has_num_experts && PyBool_Check(num_experts_obj.ptr())) {
    throw std::invalid_argument(
        "num_experts must be a positive int or None, got bool.");
  }
  const int64_t num_experts =
      has_num_experts ? nb::cast<int64_t>(num_experts_obj) : -1;
  if (has_num_experts && num_experts <= 0) {
    throw std::invalid_argument(
        "num_experts must be a positive int or None, got " +
        std::to_string(num_experts) + ".");
  }

  // If already encoded as C-contiguous uint8 3D/4D/5D, classify using
  // `batched` so batched `uint8` with `K == 5` (`(B, T, L, 5)`) never collides
  // with unbatched `uint10` (`(T, L, K // 4, 5)`).
  nb::ndarray<nb::numpy, const uint8_t, nb::c_contig> u8_any;
  if (nb::try_cast(routed_obj, u8_any, /*convert=*/false)) {
    const RoutedEncoding src_enc =
        ClassifyUint8RoutedEncoding(u8_any, batched);
    if (src_enc == RoutedEncoding::kUint8) {
      if (mode == "uint8" ||
          (mode == "auto" && (!has_num_experts || num_experts <= 255))) {
        if (has_num_experts && num_experts <= 255) {
          const uint8_t* p = u8_any.data();
          const size_t n = u8_any.size();
          uint8_t max_u8 = 0;
          for (size_t i = 0; i < n; ++i) {
            if (p[i] > max_u8) max_u8 = p[i];
          }
          const int64_t max_val = static_cast<int64_t>(max_u8) - 1;
          if (max_val >= num_experts) {
            throw std::invalid_argument(
                "routed_experts contains expert ID " + std::to_string(max_val) +
                " >= num_experts (" + std::to_string(num_experts) + ").");
          }
        }
        return routed_obj;
      }
      routed_obj = DecodeRoutedExpertsFast(routed_obj, /*batched=*/batched);
    } else if (src_enc == RoutedEncoding::kUint10) {
      if (mode == "uint10" ||
          (mode == "auto" &&
           (!has_num_experts || (num_experts >= 256 && num_experts <= 1023)))) {
        if (has_num_experts && num_experts <= 1022) {
          const uint8_t* p = u8_any.data();
          const size_t n_groups = u8_any.size() / 5;
          uint16_t max_u10 = 0;
          for (size_t g = 0; g < n_groups; ++g) {
            uint16_t v[4];
            UnpackUint10ToFour(p + g * 5, v);
            for (int k = 0; k < 4; ++k) {
              if (v[k] > max_u10) max_u10 = v[k];
            }
          }
          const int64_t max_val = static_cast<int64_t>(max_u10) - 1;
          if (max_val >= num_experts) {
            throw std::invalid_argument(
                "routed_experts contains expert ID " + std::to_string(max_val) +
                " >= num_experts (" + std::to_string(num_experts) + ").");
          }
        }
        return routed_obj;
      }
      routed_obj = DecodeRoutedExpertsFast(routed_obj, /*batched=*/batched);
    }
  }

  nb::ndarray<nb::numpy, const int16_t, nb::c_contig> arr;
  nb::object keepalive;
  if (!nb::try_cast(routed_obj, arr, /*convert=*/false)) {
    static nb::object* const kNpAsArray =
        new nb::object(nb::module_::import_("numpy").attr("asarray"));
    nb::object raw_np = (*kNpAsArray)(routed_obj);
    nb::ndarray<nb::numpy, const uint8_t> u8_noncontig;
    if (nb::try_cast(raw_np, u8_noncontig, /*convert=*/false)) {
      static nb::object* const kNpAsContig = new nb::object(
          nb::module_::import_("numpy").attr("ascontiguousarray"));
      return EncodeRoutedExpertsFast((*kNpAsContig)(raw_np), mode,
                                     num_experts_obj, batched);
    }
    keepalive = CoerceToContiguousNumpy(routed_obj.ptr(), "int16");
    arr = nb::cast<nb::ndarray<nb::numpy, const int16_t, nb::c_contig>>(
        keepalive);
  }
  const size_t ndim = arr.ndim();
  if ((batched && ndim != 4) || (!batched && ndim != 3 && ndim != 4)) {
    throw std::invalid_argument(
        "encode_routed_experts expects a 3D (T, L, K) or 4D (B, T, L, K) "
        "routed_experts array, got ndim=" +
        std::to_string(ndim) + ".");
  }

  const size_t top_k = arr.shape(ndim - 1);
  const size_t num_layers = arr.shape(ndim - 2);
  const size_t total_elems = arr.size();
  const int16_t* src = arr.data();

  RoutedEncoding target_enc = RoutedEncoding::kInt16;
  if (mode == "int16") {
    target_enc = RoutedEncoding::kInt16;
  } else if (mode == "uint8") {
    target_enc = RoutedEncoding::kUint8;
  } else if (mode == "uint10") {
    if (top_k % 4 != 0) {
      throw std::invalid_argument(
          "uint10 encoding requires top_k divisible by 4, got top_k=" +
          std::to_string(top_k) + ".");
    }
    target_enc = RoutedEncoding::kUint10;
  } else if (mode != "auto") {
    throw std::invalid_argument("Unsupported routed_experts encoding mode: " +
                                mode);
  }

  RawBufferPtr<uint8_t> out_buf;
  {
    nb::gil_scoped_release release;
    int16_t min_val = 0;
    int16_t max_val = -1;
    for (size_t i = 0; i < total_elems; ++i) {
      const int16_t v = src[i];
      if (v < min_val) min_val = v;
      if (v > max_val) max_val = v;
    }
    if (min_val < -1) {
      throw std::invalid_argument(
          "routed_experts contains invalid expert ID < -1: " +
          std::to_string(min_val));
    }
    if (has_num_experts && max_val >= num_experts) {
      throw std::invalid_argument(
          "routed_experts contains expert ID " + std::to_string(max_val) +
          " >= num_experts (" + std::to_string(num_experts) + ").");
    }
    const int64_t effective_max =
        has_num_experts ? std::max<int64_t>(max_val, num_experts - 1)
                        : static_cast<int64_t>(max_val);

    if (mode == "auto") {
      if (effective_max <= 254) {
        target_enc = RoutedEncoding::kUint8;
      } else if (effective_max <= 1022 && (top_k % 4 == 0)) {
        target_enc = RoutedEncoding::kUint10;
      } else {
        target_enc = RoutedEncoding::kInt16;
      }
    } else if (target_enc == RoutedEncoding::kUint8 && max_val > 254) {
      throw std::invalid_argument(
          "uint8 encoding requires expert IDs <= 254, got max=" +
          std::to_string(max_val));
    } else if (target_enc == RoutedEncoding::kUint10 && max_val > 1022) {
      throw std::invalid_argument(
          "uint10 encoding requires expert IDs <= 1022, got max=" +
          std::to_string(max_val));
    }

    if (target_enc == RoutedEncoding::kUint8) {
      out_buf = AllocRawBuffer<uint8_t>(total_elems);
      uint8_t* dst = out_buf.get();
      for (size_t i = 0; i < total_elems; ++i) {
        dst[i] = static_cast<uint8_t>(src[i] + 1);
      }
    } else if (target_enc == RoutedEncoding::kUint10) {
      const size_t num_groups = total_elems / 4;
      out_buf = AllocRawBuffer<uint8_t>(num_groups * 5);
      uint8_t* dst = out_buf.get();
      for (size_t g = 0; g < num_groups; ++g) {
        const int16_t* s = src + g * 4;
        PackFourToUint10(static_cast<uint16_t>(s[0] + 1),
                         static_cast<uint16_t>(s[1] + 1),
                         static_cast<uint16_t>(s[2] + 1),
                         static_cast<uint16_t>(s[3] + 1), dst + g * 5);
      }
    }
  }

  if (target_enc == RoutedEncoding::kInt16) {
    return keepalive.is_valid() ? std::move(keepalive) : std::move(routed_obj);
  }
  if (ndim == 3) {
    const size_t t_dim = arr.shape(0);
    if (target_enc == RoutedEncoding::kUint8) {
      return Wrap3DArray(std::move(out_buf), t_dim, num_layers, top_k);
    }
    return Wrap4DArray(std::move(out_buf), t_dim, num_layers, top_k / 4, 5);
  }
  const size_t b_dim = arr.shape(0);
  const size_t t_dim = arr.shape(1);
  return WrapBatchedRoutedBuffer(std::move(out_buf), target_enc, b_dim, t_dim,
                                 num_layers, top_k);
}

// Fast C++ `nogil` decoder for `routed_experts` (`uint8` / `uint10` / `int16`
// -> `int16` 3D `(T, L, K)` when `batched=false` or 4D `(B, T, L, K)` when
// `batched=true`).
//
// Contract: `batched` explicitly selects between unbatched per-trajectory
// layouts (`3D` `uint8`/`int16` `(T, L, K)` or `4D` `uint10` `(T, L, K//4, 5)`)
// and batched microbatch layouts (`4D` `uint8`/`int16` `(B, T, L, K)` or `5D`
// `uint10` `(B, T, L, K//4, 5)`), so batched `uint8` with `K == 5` never
// collides with unbatched `uint10`.
nb::object DecodeRoutedExpertsFast(nb::object encoded_obj, bool batched) {
  if (encoded_obj.is_none()) {
    return nb::none();
  }
  const size_t expected_raw_ndim = batched ? 4 : 3;
  const auto check_raw_ndim = [batched, expected_raw_ndim](size_t actual_ndim) {
    if (actual_ndim != expected_raw_ndim) {
      throw std::invalid_argument(
          "decode_routed_experts with batched=" +
          std::string(batched ? "True" : "False") + " expects a " +
          std::to_string(expected_raw_ndim) +
          "D int16 array, got ndim=" + std::to_string(actual_ndim) + ".");
    }
  };
  nb::ndarray<nb::numpy, const int16_t, nb::c_contig> i16_arr;
  if (nb::try_cast(encoded_obj, i16_arr, /*convert=*/false)) {
    check_raw_ndim(i16_arr.ndim());
    return encoded_obj;
  }

  nb::ndarray<nb::numpy, const uint8_t, nb::c_contig> u8_arr;
  nb::object keepalive;
  if (!nb::try_cast(encoded_obj, u8_arr, /*convert=*/false)) {
    static nb::object* const kNpAsContig =
        new nb::object(nb::module_::import_("numpy").attr("ascontiguousarray"));
    keepalive = (*kNpAsContig)(encoded_obj);
    if (nb::try_cast(keepalive, i16_arr, /*convert=*/false)) {
      check_raw_ndim(i16_arr.ndim());
      return keepalive;
    }
    if (!nb::try_cast(keepalive, u8_arr, /*convert=*/false)) {
      nb::object i16_coerced =
          CoerceToContiguousNumpy(encoded_obj.ptr(), "int16");
      i16_arr = nb::cast<nb::ndarray<nb::numpy, const int16_t, nb::c_contig>>(
          i16_coerced);
      check_raw_ndim(i16_arr.ndim());
      return i16_coerced;
    }
  }

  const RoutedEncoding src_enc = ClassifyUint8RoutedEncoding(u8_arr, batched);
  const uint8_t* src = u8_arr.data();
  if (batched) {
    if (src_enc == RoutedEncoding::kUint10) {
      // Batched uint10: (B, T, L, K//4, 5)
      const size_t b_dim = u8_arr.shape(0);
      const size_t t_dim = u8_arr.shape(1);
      const size_t l_dim = u8_arr.shape(2);
      const size_t g_dim = u8_arr.shape(3);
      const size_t k_dim = g_dim * 4;
      const size_t num_groups = b_dim * t_dim * l_dim * g_dim;
      RawBufferPtr<int16_t> out;
      {
        nb::gil_scoped_release release;
        out = AllocRawBuffer<int16_t>(num_groups * 4);
        int16_t* dst = out.get();
        for (size_t g = 0; g < num_groups; ++g) {
          uint16_t v[4];
          UnpackUint10ToFour(src + g * 5, v);
          dst[g * 4 + 0] = static_cast<int16_t>(v[0]) - 1;
          dst[g * 4 + 1] = static_cast<int16_t>(v[1]) - 1;
          dst[g * 4 + 2] = static_cast<int16_t>(v[2]) - 1;
          dst[g * 4 + 3] = static_cast<int16_t>(v[3]) - 1;
        }
      }
      return Wrap4DArray(std::move(out), b_dim, t_dim, l_dim, k_dim);
    }
    // Batched uint8: (B, T, L, K) (including when K == 5)
    const size_t b_dim = u8_arr.shape(0);
    const size_t t_dim = u8_arr.shape(1);
    const size_t l_dim = u8_arr.shape(2);
    const size_t k_dim = u8_arr.shape(3);
    const size_t total = b_dim * t_dim * l_dim * k_dim;
    RawBufferPtr<int16_t> out;
    {
      nb::gil_scoped_release release;
      out = AllocRawBuffer<int16_t>(total);
      int16_t* dst = out.get();
      for (size_t i = 0; i < total; ++i) {
        dst[i] = static_cast<int16_t>(src[i]) - 1;
      }
    }
    return Wrap4DArray(std::move(out), b_dim, t_dim, l_dim, k_dim);
  }

  if (src_enc == RoutedEncoding::kUint10) {
    // Unbatched uint10: (T, L, K//4, 5)
    const size_t t_dim = u8_arr.shape(0);
    const size_t l_dim = u8_arr.shape(1);
    const size_t g_dim = u8_arr.shape(2);
    const size_t k_dim = g_dim * 4;
    const size_t num_groups = t_dim * l_dim * g_dim;
    RawBufferPtr<int16_t> out;
    {
      nb::gil_scoped_release release;
      out = AllocRawBuffer<int16_t>(num_groups * 4);
      int16_t* dst = out.get();
      for (size_t g = 0; g < num_groups; ++g) {
        uint16_t v[4];
        UnpackUint10ToFour(src + g * 5, v);
        dst[g * 4 + 0] = static_cast<int16_t>(v[0]) - 1;
        dst[g * 4 + 1] = static_cast<int16_t>(v[1]) - 1;
        dst[g * 4 + 2] = static_cast<int16_t>(v[2]) - 1;
        dst[g * 4 + 3] = static_cast<int16_t>(v[3]) - 1;
      }
    }
    return Wrap3DArray(std::move(out), t_dim, l_dim, k_dim);
  }
  // Unbatched uint8: (T, L, K) (including when K == 5)
  const size_t t_dim = u8_arr.shape(0);
  const size_t l_dim = u8_arr.shape(1);
  const size_t k_dim = u8_arr.shape(2);
  const size_t total = t_dim * l_dim * k_dim;
  RawBufferPtr<int16_t> out;
  {
    nb::gil_scoped_release release;
    out = AllocRawBuffer<int16_t>(total);
    int16_t* dst = out.get();
    for (size_t i = 0; i < total; ++i) {
      dst[i] = static_cast<int16_t>(src[i]) - 1;
    }
  }
  return Wrap3DArray(std::move(out), t_dim, l_dim, k_dim);
}

#if defined(__SANITIZE_THREAD__)
constexpr bool kThreadSanitizer = true;
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
constexpr bool kThreadSanitizer = true;
#else
constexpr bool kThreadSanitizer = false;
#endif
#else
constexpr bool kThreadSanitizer = false;
#endif

}  // namespace

NB_MODULE(_packing_ext, m) {
  m.doc() = "C++ nanobind nogil acceleration for Tunix RL sequence packing.";
  PackingWorkerPool::InitDefaultThreads();
  m.attr("MIN_BYTES_PER_THREAD") =
      static_cast<int64_t>(PackingWorkerPool::kMinBytesPerThread);
  m.attr("MIN_PARALLEL_BYTES") =
      static_cast<int64_t>(PackingWorkerPool::kMinParallelBytes);
  m.attr("MAX_PACKING_THREADS") =
      static_cast<int64_t>(PackingWorkerPool::kMaxThreads);
  m.attr("THREAD_SANITIZER") = kThreadSanitizer;
  m.def("get_packing_threads", &PackingWorkerPool::GetConfiguredThreads);
  m.def("set_packing_threads", &PackingWorkerPool::SetConfiguredThreads,
        nb::arg("num_threads"));
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
  m.def("encode_routed_experts_fast", &EncodeRoutedExpertsFast,
        nb::arg("routed").none(), nb::arg("mode") = "auto",
        nb::arg("num_experts") = nb::none(), nb::arg("batched") = false);
  m.def("decode_routed_experts_fast", &DecodeRoutedExpertsFast,
        nb::arg("encoded").none(), nb::arg("batched") = false);
}

}  // namespace tunix::rl
