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
#include <unistd.h>

#include <algorithm>
#include <atomic>
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
#include <stdexcept>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"

namespace nb = nanobind;

namespace tunix::worker {
namespace {

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

// =============================================================================
// Out-of-Band Bulk TCP Framed Multi-Block LZ4 Codec (`nogil`)
// =============================================================================

constexpr uint8_t kLz4FrameMagic[4] = {'T', 'L', 'Z', '4'};
constexpr size_t kLz4FrameHeaderBytes =
    20;                             // magic(4) + raw(8) + blk(4) + n(4)
constexpr size_t kLz4HashLog = 12;  // 4096-entry hash table (16 KiB L1)
constexpr size_t kLz4HashSize = size_t{1} << kLz4HashLog;
constexpr uint32_t kLz4Prime = 2654435761U;
constexpr size_t kLz4MinMatch = 4;
constexpr size_t kLz4LastLiterals = 5;
constexpr size_t kLz4MfLimit = 12;
constexpr size_t kLz4MaxOffset = 65535;
constexpr size_t kLz4SampleProbeSize = 4096;
constexpr size_t kLz4SampleProbeTriggerBytes = 65536;

inline uint32_t ReadU32LE(const uint8_t* p) noexcept {
  uint32_t v;
  std::memcpy(&v, p, sizeof(uint32_t));
  return v;
}

inline uint64_t ReadU64LE(const uint8_t* p) noexcept {
  uint64_t v;
  std::memcpy(&v, p, sizeof(uint64_t));
  return v;
}

inline void WriteU16LE(uint8_t* p, uint16_t v) noexcept {
  std::memcpy(p, &v, sizeof(uint16_t));
}

inline void WriteU32LE(uint8_t* p, uint32_t v) noexcept {
  std::memcpy(p, &v, sizeof(uint32_t));
}

inline void WriteU64LE(uint8_t* p, uint64_t v) noexcept {
  std::memcpy(p, &v, sizeof(uint64_t));
}

inline uint32_t Lz4Hash4(uint32_t seq) noexcept {
  return (seq * kLz4Prime) >> (32 - kLz4HashLog);
}

// Compresses a single block `src[0..src_len)` into `dst[0..max_dst_len)` using
// standard LZ4 block format. Returns compressed byte count, or `0` if the
// compressed representation would exceed `max_dst_len`.
size_t CompressLz4BlockBounded(const uint8_t* src, size_t src_len, uint8_t* dst,
                               size_t max_dst_len) noexcept {
  if (src_len == 0 || max_dst_len == 0) {
    return 0;
  }
  uint32_t hash_table[kLz4HashSize] = {0};
  const uint8_t* ip = src;
  const uint8_t* anchor = src;
  const uint8_t* const iend = src + src_len;
  const uint8_t* const mflimit =
      (src_len > kLz4MfLimit) ? (iend - kLz4MfLimit) : src;
  const uint8_t* const matchlimit =
      (src_len > kLz4LastLiterals) ? (iend - kLz4LastLiterals) : src;

  uint8_t* op = dst;
  uint8_t* const oend = dst + max_dst_len;

  if (src_len >= kLz4MfLimit) {
    hash_table[Lz4Hash4(ReadU32LE(ip))] = 0;
    ++ip;
    uint32_t step = 1;
    uint32_t search_match_nb = 1U << 5;

    while (ip < mflimit) {
      const uint8_t* ref = nullptr;
      // Find a 4-byte match within 64 KiB window.
      while (true) {
        const uint32_t seq = ReadU32LE(ip);
        const uint32_t h = Lz4Hash4(seq);
        const uint32_t ref_idx = hash_table[h];
        hash_table[h] = static_cast<uint32_t>(ip - src);
        ref = src + ref_idx;
        if ((ip - ref) <= static_cast<ptrdiff_t>(kLz4MaxOffset) &&
            ReadU32LE(ref) == seq) {
          break;
        }
        ip += step;
        if (ip >= mflimit) {
          break;
        }
        step = (search_match_nb++ >> 5);
      }
      if (ip >= mflimit) {
        break;
      }

      // Catch up backward if match extends before `ip` and `ref`.
      while (ip > anchor && ref > src && ip[-1] == ref[-1]) {
        --ip;
        --ref;
      }

      const size_t lit_len = static_cast<size_t>(ip - anchor);
      // Fast shift bound avoids 64-bit integer division by 255 in the inner
      // loop (`(lit_len >> 7)` >= `lit_len / 255`).
      const size_t needed_lit =
          (lit_len < 15) ? (lit_len + 4) : (5 + (lit_len >> 7) + lit_len);
      if (static_cast<size_t>(oend - op) < needed_lit) {
        return 0;
      }
      uint8_t* token_ptr = op++;
      if (lit_len >= 15) {
        *token_ptr = 15 << 4;
        size_t rem = lit_len - 15;
        while (rem >= 255) {
          *op++ = 255;
          rem -= 255;
        }
        *op++ = static_cast<uint8_t>(rem);
      } else {
        *token_ptr = static_cast<uint8_t>(lit_len << 4);
      }
      if (lit_len > 0) {
        std::memcpy(op, anchor, lit_len);
        op += lit_len;
      }

      // Encode 16-bit little-endian backward offset.
      const uint16_t offset = static_cast<uint16_t>(ip - ref);
      WriteU16LE(op, offset);
      op += 2;

      // Count match length beyond initial 4 bytes.
      ip += kLz4MinMatch;
      ref += kLz4MinMatch;
      const uint8_t* const match_start = ip;
      while (true) {
        if (ip + 8 <= matchlimit) {
          const uint64_t diff = ReadU64LE(ip) ^ ReadU64LE(ref);
          if (diff != 0) {
            ip += (__builtin_ctzll(diff) >> 3);
            break;
          }
          ip += 8;
          ref += 8;
        } else {
          while (ip < matchlimit && *ip == *ref) {
            ++ip;
            ++ref;
          }
          break;
        }
      }
      const size_t match_len = static_cast<size_t>(ip - match_start);
      if (match_len >= 15) {
        const size_t needed_match = 1 + (match_len >> 7);
        if (static_cast<size_t>(oend - op) < needed_match) {
          return 0;
        }
        *token_ptr |= 15;
        size_t rem = match_len - 15;
        while (rem >= 255) {
          *op++ = 255;
          rem -= 255;
        }
        *op++ = static_cast<uint8_t>(rem);
      } else {
        *token_ptr |= static_cast<uint8_t>(match_len);
      }

      anchor = ip;
      if (ip >= mflimit) {
        break;
      }
      hash_table[Lz4Hash4(ReadU32LE(ip - 2))] =
          static_cast<uint32_t>((ip - 2) - src);
      step = 1;
      search_match_nb = 1U << 5;
    }
  }

  // Emit trailing literals.
  const size_t last_lit = static_cast<size_t>(iend - anchor);
  const size_t needed_tail =
      (last_lit < 15) ? (1 + last_lit) : (2 + (last_lit >> 7) + last_lit);
  if (static_cast<size_t>(oend - op) < needed_tail) {
    return 0;
  }
  if (last_lit >= 15) {
    *op++ = 15 << 4;
    size_t rem = last_lit - 15;
    while (rem >= 255) {
      *op++ = 255;
      rem -= 255;
    }
    *op++ = static_cast<uint8_t>(rem);
  } else {
    *op++ = static_cast<uint8_t>(last_lit << 4);
  }
  if (last_lit > 0) {
    std::memcpy(op, anchor, last_lit);
    op += last_lit;
  }
  return static_cast<size_t>(op - dst);
}

// Decompresses a single LZ4 block `src[0..src_len)` into `dst[0..dst_cap)`.
// Returns the exact number of bytes written to `dst`, or throws
// `std::invalid_argument` if the stream is corrupted or overflows `dst_cap`.
size_t DecompressLz4BlockSafe(const uint8_t* src, size_t src_len, uint8_t* dst,
                              size_t dst_cap) {
  const uint8_t* ip = src;
  const uint8_t* const iend = src + src_len;
  uint8_t* op = dst;
  uint8_t* const oend = dst + dst_cap;

  while (ip < iend) {
    const uint8_t token = *ip++;
    size_t lit_len = token >> 4;
    if (lit_len == 15) {
      while (true) {
        if (ip >= iend) {
          throw std::invalid_argument(
              "Corrupted LZ4 block: truncated literal length.");
        }
        const uint8_t s = *ip++;
        lit_len += s;
        if (s != 255) {
          break;
        }
      }
    }
    if (static_cast<size_t>(iend - ip) < lit_len ||
        static_cast<size_t>(oend - op) < lit_len) {
      throw std::invalid_argument(
          "Corrupted LZ4 block: literal length out of bounds.");
    }
    if (lit_len > 0) {
      std::memcpy(op, ip, lit_len);
      ip += lit_len;
      op += lit_len;
    }
    if (ip == iend) {
      break;
    }
    if (static_cast<size_t>(iend - ip) < 2) {
      throw std::invalid_argument(
          "Corrupted LZ4 block: truncated match offset.");
    }
    const uint16_t offset =
        static_cast<uint16_t>(ip[0]) | (static_cast<uint16_t>(ip[1]) << 8);
    ip += 2;
    if (offset == 0 || static_cast<size_t>(op - dst) < offset) {
      throw std::invalid_argument(
          "Corrupted LZ4 block: invalid backward match offset.");
    }
    size_t match_len = (token & 0x0F) + kLz4MinMatch;
    if ((token & 0x0F) == 15) {
      while (true) {
        if (ip >= iend) {
          throw std::invalid_argument(
              "Corrupted LZ4 block: truncated match length.");
        }
        const uint8_t s = *ip++;
        match_len += s;
        if (s != 255) {
          break;
        }
      }
    }
    if (static_cast<size_t>(oend - op) < match_len) {
      throw std::invalid_argument(
          "Corrupted LZ4 block: match length exceeds output buffer.");
    }
    const uint8_t* match = op - offset;
    if (offset >= match_len) {
      // Non-overlapping backward reference: single vectorized memcpy.
      std::memcpy(op, match, match_len);
      op += match_len;
    } else if (offset == 1) {
      // 1-byte RLE repeat (e.g., 0x00 unset expert padding): hardware memset.
      std::memset(op, *match, match_len);
      op += match_len;
    } else {
      // Overlapping periodic match (1 < offset < match_len): byte-by-byte copy
      // preserves the exact period-`offset` repeating pattern.
      for (size_t i = 0; i < match_len; ++i) {
        *op++ = *match++;
      }
    }
  }
  return static_cast<size_t>(op - dst);
}

// Fork-safe persistent worker pool for parallel multi-block LZ4 compression
// and decompression without per-call OS thread creation overhead.
class Lz4WorkerPool {
 public:
  static constexpr size_t kNumThreads = 8;

  static void ParallelFor(size_t num_blocks,
                          const std::function<void(size_t, size_t)>& fn) {
    if (num_blocks < 4) {
      fn(0, num_blocks);
      return;
    }
    static std::once_flag atfork_once;  // NOLINT(build/c++11)
    std::call_once(atfork_once, []() {  // NOLINT(build/c++11)
      pthread_atfork(nullptr, nullptr, &Lz4WorkerPool::OnForkChild);
    });
    std::lock_guard<std::mutex> call_lock(*g_pool_mu_);  // NOLINT(build/c++11)
    const pid_t cur_pid = getpid();
    if (g_pool_ == nullptr || g_pool_->pid_ != cur_pid) {
      // If forked, abandon the inherited descriptor without joining dead
      // threads from the parent process and allocate a fresh pool.
      g_pool_ = new Lz4WorkerPool(cur_pid);
    }
    g_pool_->RunLocked(num_blocks, fn);
  }

 private:
  static void OnForkChild() {
    // Reinitialize the pool mutex and pool pointer in the child process so a
    // fork occurring concurrently while another thread holds `g_pool_mu_`
    // cannot leave the child process with a permanently locked mutex.
    g_pool_mu_ = new std::mutex();  // NOLINT(build/c++11)
    g_pool_ = nullptr;
  }

  explicit Lz4WorkerPool(pid_t pid) : pid_(pid) {
    workers_.reserve(kNumThreads - 1);
    for (size_t t = 0; t < kNumThreads - 1; ++t) {
      workers_.emplace_back([this, t]() { WorkerLoop(t + 1); });
      workers_.back().detach();
    }
  }

  void WorkerLoop(size_t worker_idx) {
    uint64_t seen_gen = 0;
    while (true) {
      const std::function<void(size_t, size_t)>* fn_ptr = nullptr;
      size_t b_start = 0;
      size_t b_end = 0;
      {
        std::unique_lock<std::mutex> lk(mu_);  // NOLINT(build/c++11)
        cv_work_.wait(lk, [&]() { return gen_ > seen_gen; });
        seen_gen = gen_;
        if (worker_idx < active_threads_) {
          fn_ptr = current_fn_;
          b_start = (num_blocks_ * worker_idx) / active_threads_;
          b_end = (num_blocks_ * (worker_idx + 1)) / active_threads_;
        }
      }
      std::exception_ptr local_exc;
      if (fn_ptr != nullptr && b_start < b_end) {
        try {
          (*fn_ptr)(b_start, b_end);
        } catch (...) {
          local_exc = std::current_exception();
        }
      }
      {
        std::lock_guard<std::mutex> lk(mu_);  // NOLINT(build/c++11)
        if (local_exc && !worker_exc_) {
          worker_exc_ = local_exc;
        }
        if (--remaining_workers_ == 0) {
          cv_done_.notify_one();
        }
      }
    }
  }

  void RunLocked(size_t num_blocks,
                 const std::function<void(size_t, size_t)>& fn) {
    const size_t n_threads = std::min(kNumThreads, num_blocks);
    {
      std::lock_guard<std::mutex> lk(mu_);  // NOLINT(build/c++11)
      current_fn_ = &fn;
      num_blocks_ = num_blocks;
      active_threads_ = n_threads;
      remaining_workers_ = kNumThreads - 1;
      worker_exc_ = nullptr;
      ++gen_;
    }
    cv_work_.notify_all();
    const size_t main_end = num_blocks / n_threads;
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
      std::unique_lock<std::mutex> lk(mu_);  // NOLINT(build/c++11)
      cv_done_.wait(lk, [&]() { return remaining_workers_ == 0; });
      current_fn_ = nullptr;
      rethrow_exc = main_exc ? main_exc : worker_exc_;
      worker_exc_ = nullptr;
    }
    if (rethrow_exc) {
      std::rethrow_exception(rethrow_exc);
    }
  }

  // Process-wide mutex serializing `ParallelFor` invocations across callers
  // (e.g. concurrent Python `_SERDE_EXECUTOR` threads) so the fixed 8-thread
  // worker pool never oversubscribes host CPU cores.
  static inline std::mutex* g_pool_mu_ =  // NOLINT(build/c++11)
      new std::mutex();                   // NOLINT(build/c++11)
  static inline Lz4WorkerPool* g_pool_ = nullptr;

  pid_t pid_;
  std::mutex mu_;  // NOLINT(build/c++11)
  std::condition_variable cv_work_;
  std::condition_variable cv_done_;
  std::vector<std::thread> workers_;
  const std::function<void(size_t, size_t)>* current_fn_ = nullptr;
  size_t num_blocks_ = 0;
  size_t active_threads_ = 0;
  size_t remaining_workers_ = 0;
  std::exception_ptr worker_exc_;
  uint64_t gen_ = 0;
};

// Compresses a contiguous buffer into a `TLZ4` multi-block frame if the
// compressed size is at most `raw_size * max_ratio`, or returns `None`.
//
// Memory & concurrency notes:
// - During compression, allocates a transient `raw_size` scratch buffer
//   (`1.0x`) for parallel per-block LZ4 output plus the compacted frame buffer
//   (`<= max_ratio * raw_size`), so peak transient memory while inside the
//   `nogil` block is `~(1 + max_ratio) * raw_size` (`~1.85x` at default
//   `max_ratio=0.85`) before `scratch_buf` is freed on scope exit.
// - Multi-block work is dispatched to the 8-thread `Lz4WorkerPool`, whose
//   `ParallelFor` calls are serialized via `g_pool_mu_`.
nb::object TryCompressBulkBufferFast(nb::object src_obj,
                                     double max_ratio = 0.85,
                                     int64_t block_size = 262144) {
  if (block_size <= 0 || block_size > (64 * 1024 * 1024)) {
    throw std::invalid_argument("block_size must be in (0, 64 MiB].");
  }
  if (max_ratio <= 0.0 || max_ratio >= 1.0) {
    return nb::none();
  }
  ScopedPyBufferList buffers;
  const Py_buffer& view = buffers.Acquire(src_obj.ptr(), PyBUF_SIMPLE);
  const size_t raw_size = static_cast<size_t>(view.len);
  const uint8_t* src = static_cast<const uint8_t*>(view.buf);
  if (raw_size < 64 || src == nullptr) {
    return nb::none();
  }
  const size_t target_max_total =
      static_cast<size_t>(static_cast<double>(raw_size) * max_ratio);
  const size_t blk_sz = static_cast<size_t>(block_size);
  const size_t num_blocks = (raw_size + blk_sz - 1) / blk_sz;
  const size_t header_and_dir_bytes =
      kLz4FrameHeaderBytes + num_blocks * sizeof(uint32_t);
  if (target_max_total <= header_and_dir_bytes) {
    return nb::none();
  }

  RawBufferPtr<uint8_t> out_buf;
  size_t total_written = 0;
  bool rejected = false;
  {
    nb::gil_scoped_release release;
    // Fast 4x4 KiB odd-eighth sample probe (at 1/8, 3/8, 5/8, 7/8 of the
    // buffer) to accurately estimate compressibility across batched `(B, T,
    // ...)` tensors while rejecting high-entropy float32 buffers in < 2 us
    // before allocating output memory.
    if (raw_size >= kLz4SampleProbeTriggerBytes) {
      uint8_t probe_dst[kLz4SampleProbeSize];
      size_t probe_total_out = 0;
      for (size_t k = 1; k <= 7; k += 2) {
        size_t probe_off = ((raw_size * k) / 8) & ~size_t{63};
        if (probe_off + kLz4SampleProbeSize > raw_size) {
          probe_off = raw_size - kLz4SampleProbeSize;
        }
        const size_t csize =
            CompressLz4BlockBounded(src + probe_off, kLz4SampleProbeSize,
                                    probe_dst, kLz4SampleProbeSize - 1);
        probe_total_out += (csize > 0 && csize < kLz4SampleProbeSize)
                               ? csize
                               : kLz4SampleProbeSize;
      }
      const size_t max_probe_out =
          static_cast<size_t>((4 * kLz4SampleProbeSize) * max_ratio);
      if (probe_total_out > max_probe_out) {
        rejected = true;
      }
    }

    if (!rejected) {
      RawBufferPtr<uint8_t> scratch_buf = AllocRawBuffer<uint8_t>(raw_size);
      uint8_t* scratch = scratch_buf.get();
      std::vector<uint32_t> dir_entries(num_blocks);

      Lz4WorkerPool::ParallelFor(num_blocks, [&](size_t b_start, size_t b_end) {
        for (size_t b = b_start; b < b_end; ++b) {
          const size_t blk_off = b * blk_sz;
          const size_t cur_raw_len = std::min(blk_sz, raw_size - blk_off);
          const size_t csize =
              (cur_raw_len > 1)
                  ? CompressLz4BlockBounded(src + blk_off, cur_raw_len,
                                            scratch + blk_off, cur_raw_len - 1)
                  : 0;
          if (csize > 0 && csize < cur_raw_len) {
            dir_entries[b] = static_cast<uint32_t>(csize);
          } else {
            dir_entries[b] = 0x80000000U | static_cast<uint32_t>(cur_raw_len);
          }
        }
      });

      std::vector<size_t> dst_offsets(num_blocks);
      size_t payload_cursor = header_and_dir_bytes;
      for (size_t b = 0; b < num_blocks; ++b) {
        dst_offsets[b] = payload_cursor;
        payload_cursor += static_cast<size_t>(dir_entries[b] & 0x7FFFFFFFU);
      }
      if (payload_cursor > target_max_total) {
        rejected = true;
      } else {
        out_buf = AllocRawBuffer<uint8_t>(payload_cursor);
        uint8_t* dst = out_buf.get();
        std::memcpy(dst, kLz4FrameMagic, 4);
        WriteU64LE(dst + 4, static_cast<uint64_t>(raw_size));
        WriteU32LE(dst + 12, static_cast<uint32_t>(blk_sz));
        WriteU32LE(dst + 16, static_cast<uint32_t>(num_blocks));
        uint8_t* dir_ptr = dst + kLz4FrameHeaderBytes;
        for (size_t b = 0; b < num_blocks; ++b) {
          WriteU32LE(dir_ptr + b * 4, dir_entries[b]);
        }

        Lz4WorkerPool::ParallelFor(
            num_blocks, [&](size_t b_start, size_t b_end) {
              for (size_t b = b_start; b < b_end; ++b) {
                const uint32_t entry = dir_entries[b];
                const bool is_raw = (entry & 0x80000000U) != 0;
                const size_t csize = static_cast<size_t>(entry & 0x7FFFFFFFU);
                const size_t blk_off = b * blk_sz;
                const uint8_t* block_src =
                    is_raw ? (src + blk_off) : (scratch + blk_off);
                std::memcpy(dst + dst_offsets[b], block_src, csize);
              }
            });
        total_written = payload_cursor;
      }
    }
  }
  if (rejected) {
    return nb::none();
  }
  return Wrap1DArray<uint8_t>(std::move(out_buf), total_written);
}

void DecompressBulkBufferFast(nb::object src_obj, nb::object dst_obj) {
  ScopedPyBufferList buffers;
  const Py_buffer& src_view = buffers.Acquire(src_obj.ptr(), PyBUF_SIMPLE);
  const Py_buffer& dst_view = buffers.Acquire(dst_obj.ptr(), PyBUF_WRITABLE);
  const uint8_t* src = static_cast<const uint8_t*>(src_view.buf);
  const size_t src_len = static_cast<size_t>(src_view.len);
  uint8_t* dst = static_cast<uint8_t*>(dst_view.buf);
  const size_t dst_len = static_cast<size_t>(dst_view.len);

  {
    nb::gil_scoped_release release;
    if (src == nullptr || dst == nullptr || src_len < kLz4FrameHeaderBytes) {
      throw std::invalid_argument(
          "Invalid LZ4 frame: buffer smaller than 20-byte header.");
    }
    if (std::memcmp(src, kLz4FrameMagic, 4) != 0) {
      throw std::invalid_argument("Invalid LZ4 frame: magic mismatch.");
    }
    const uint64_t raw_size = ReadU64LE(src + 4);
    const uint32_t block_size = ReadU32LE(src + 12);
    const uint32_t num_blocks = ReadU32LE(src + 16);
    if (raw_size != dst_len) {
      throw std::invalid_argument("LZ4 frame raw_size (" +
                                  std::to_string(raw_size) +
                                  ") does not match target buffer length (" +
                                  std::to_string(dst_len) + ").");
    }
    if (block_size == 0 || block_size > (64 * 1024 * 1024)) {
      throw std::invalid_argument("Invalid LZ4 frame block_size.");
    }
    const size_t blk_sz = static_cast<size_t>(block_size);
    const size_t expected_blocks = (dst_len + blk_sz - 1) / blk_sz;
    if (static_cast<size_t>(num_blocks) != expected_blocks) {
      throw std::invalid_argument("Invalid LZ4 frame num_blocks.");
    }
    const size_t header_and_dir =
        kLz4FrameHeaderBytes + static_cast<size_t>(num_blocks) * 4;
    if (src_len < header_and_dir) {
      throw std::invalid_argument(
          "Invalid LZ4 frame: truncated block directory.");
    }

    const uint8_t* dir_ptr = src + kLz4FrameHeaderBytes;
    std::vector<size_t> src_offsets(num_blocks);
    size_t src_cursor = header_and_dir;
    for (uint32_t b = 0; b < num_blocks; ++b) {
      const uint32_t entry = ReadU32LE(dir_ptr + static_cast<size_t>(b) * 4);
      const bool is_raw = (entry & 0x80000000U) != 0;
      const size_t csize = static_cast<size_t>(entry & 0x7FFFFFFFU);
      const size_t dst_cursor = static_cast<size_t>(b) * blk_sz;
      const size_t expected_raw = std::min(blk_sz, dst_len - dst_cursor);
      if (csize == 0 || src_cursor + csize > src_len) {
        throw std::invalid_argument(
            "Invalid LZ4 frame: block payload out of bounds.");
      }
      if (is_raw && csize != expected_raw) {
        throw std::invalid_argument(
            "Invalid LZ4 frame: raw block size mismatch.");
      }
      src_offsets[b] = src_cursor;
      src_cursor += csize;
    }
    if (src_cursor != src_len) {
      throw std::invalid_argument(
          "Invalid LZ4 frame: trailing bytes or incomplete output.");
    }

    std::atomic<bool> failed{false};
    std::mutex err_mu;  // NOLINT(build/c++11)
    std::string first_err;
    Lz4WorkerPool::ParallelFor(num_blocks, [&](size_t b_start, size_t b_end) {
      for (size_t b = b_start; b < b_end; ++b) {
        if (failed.load(std::memory_order_relaxed)) {
          return;
        }
        const uint32_t entry = ReadU32LE(dir_ptr + b * 4);
        const bool is_raw = (entry & 0x80000000U) != 0;
        const size_t csize = static_cast<size_t>(entry & 0x7FFFFFFFU);
        const size_t dst_cursor = b * blk_sz;
        const size_t expected_raw = std::min(blk_sz, dst_len - dst_cursor);
        const uint8_t* blk_src = src + src_offsets[b];
        if (is_raw) {
          std::memcpy(dst + dst_cursor, blk_src, csize);
        } else {
          try {
            const size_t written = DecompressLz4BlockSafe(
                blk_src, csize, dst + dst_cursor, expected_raw);
            if (written != expected_raw) {
              throw std::invalid_argument(
                  "Invalid LZ4 frame: decompressed block size mismatch.");
            }
          } catch (const std::exception& e) {
            if (!failed.exchange(true, std::memory_order_relaxed)) {
              std::lock_guard<std::mutex> lk(err_mu);  // NOLINT(build/c++11)
              first_err = e.what();
            }
            return;
          }
        }
      }
    });
    if (failed.load(std::memory_order_relaxed)) {
      throw std::invalid_argument(first_err);
    }
  }
}

}  // namespace

NB_MODULE(_compression_ext, m) {
  m.doc() =
      "C++ nanobind nogil multi-block LZ4 codec for Tunix bulk TCP transport.";
  m.def("try_compress_bulk_buffer", &TryCompressBulkBufferFast, nb::arg("src"),
        nb::arg("max_ratio") = 0.85, nb::arg("block_size") = 262144);
  m.def("decompress_bulk_buffer", &DecompressBulkBufferFast, nb::arg("src"),
        nb::arg("dst"));
}

}  // namespace tunix::worker
