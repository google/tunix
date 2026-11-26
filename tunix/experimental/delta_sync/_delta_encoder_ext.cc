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

// Host-side encoder for the trainer half of delta weight sync.
//
// After an RL step only a few percent of the bf16 weights change. The trainer
// keeps the previously synced weights in host memory, compares them with the
// new weights, and ships only the changed words; the rollout engine patches
// its own copy in place.
//
// Payload layout. A 2D bf16 tensor is cut into blocks of `block_rows` rows.
// Within a block, one 32-bit word holds rows 2p and 2p + 1 of the same column,
// where p = local_row / 2. For block b and slot i:
//
//   offsets[b, i] = p * cols + col, or PAD_OFFSET for padding
//   values[b, i]  = new_bits[2p + 1, col] << 16 | new_bits[2p, col], or 0 for
//                   padding
//
// A word is sent when either of its rows changed and always carries both new
// halves, so applying a payload twice is a no-op. Within a block the words are
// sorted by offset and unique, padding comes last, and every block is padded
// to the same `capacity`.
//
// Arrays are used in place: nothing is copied or converted, and the GIL is
// released while scanning. Each worker thread holds scratch space for one
// whole block, 8 bytes per row-pair word (128 KiB for a 32K-element block).

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>  // NOLINT(build/c++11)
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <utility>
#include <vector>

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define TUNIX_DELTA_HAVE_AVX2 1
#else
#define TUNIX_DELTA_HAVE_AVX2 0
#endif

namespace nb = nanobind;

namespace tunix::delta_sync {
namespace {

// Marks an unused slot in the offsets of a block.
constexpr int32_t kPadOffset = -1;
// The capacity of a block must be a multiple of this.
constexpr int64_t kCapacityAlign = 128;
// Rows in one bf16 tile; a block must cover whole tiles.
constexpr int64_t kTileRows = 16;
// Columns in one tile.
constexpr int64_t kTileCols = 128;
// Weight bytes one worker claims at a time.
constexpr int64_t kChunkBytes = 4 << 20;
// Margin after the scratch buffers. A vector store writes 8 words at the
// current count, which never passes the end of the block.
constexpr int64_t kScratchSlack = 8;

using BitsArray = nb::ndarray<nb::numpy, const uint16_t, nb::ndim<2>,
                              nb::c_contig, nb::device::cpu>;
using PatchArray =
    nb::ndarray<nb::numpy, int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <typename... Args>
std::string StrCat(const Args&... args) {
  std::ostringstream out;
  (out << ... << args);
  return out.str();
}

// Scans one block ([block_rows, cols], row-major) and writes its changed
// row-pair words, in row-major order, to `offsets` / `values`. Both must have
// room for block_rows / 2 * cols + kScratchSlack words. Returns the number of
// changed words.
using EncodeBlockFn = int64_t (*)(const uint16_t* old_bits,
                                  const uint16_t* new_bits, int64_t cols,
                                  int64_t block_rows, int32_t* offsets,
                                  int32_t* values);

// Returns the number of changed row-pair words in one block.
using CountBlockFn = int64_t (*)(const uint16_t* old_bits,
                                 const uint16_t* new_bits, int64_t cols,
                                 int64_t block_rows);

int64_t EncodeBlockScalar(const uint16_t* old_bits, const uint16_t* new_bits,
                          int64_t cols, int64_t block_rows, int32_t* offsets,
                          int32_t* values) {
  int64_t count = 0;
  for (int64_t pair = 0; pair < block_rows / 2; ++pair) {
    const uint16_t* old_even = old_bits + 2 * pair * cols;
    const uint16_t* old_odd = old_even + cols;
    const uint16_t* new_even = new_bits + 2 * pair * cols;
    const uint16_t* new_odd = new_even + cols;
    for (int64_t col = 0; col < cols; ++col) {
      if (old_even[col] == new_even[col] && old_odd[col] == new_odd[col]) {
        continue;
      }
      offsets[count] = static_cast<int32_t>(pair * cols + col);
      values[count] = static_cast<int32_t>(
          static_cast<uint32_t>(new_odd[col]) << 16 | new_even[col]);
      ++count;
    }
  }
  return count;
}

int64_t CountBlockScalar(const uint16_t* old_bits, const uint16_t* new_bits,
                         int64_t cols, int64_t block_rows) {
  int64_t count = 0;
  for (int64_t pair = 0; pair < block_rows / 2; ++pair) {
    const uint16_t* old_even = old_bits + 2 * pair * cols;
    const uint16_t* old_odd = old_even + cols;
    const uint16_t* new_even = new_bits + 2 * pair * cols;
    const uint16_t* new_odd = new_even + cols;
    for (int64_t col = 0; col < cols; ++col) {
      count +=
          (old_even[col] != new_even[col]) | (old_odd[col] != new_odd[col]);
    }
  }
  return count;
}

#if TUNIX_DELTA_HAVE_AVX2

// lanes[mask] lists the lanes whose bit is set in `mask`, lowest first. Each
// row is one aligned 32-byte vector.
struct CompressLut {
  alignas(32) int32_t lanes[256][8];
};

constexpr CompressLut MakeCompressLut() {
  CompressLut lut = {};
  for (int mask = 0; mask < 256; ++mask) {
    int next = 0;
    for (int lane = 0; lane < 8; ++lane) {
      if (mask >> lane & 1) lut.lanes[mask][next++] = lane;
    }
  }
  return lut;
}
constexpr CompressLut kCompressLut = MakeCompressLut();

__attribute__((target("avx2,popcnt"))) inline __m256i LoadU16x16(
    const uint16_t* p) {
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}

// Compares 16 columns at a time and skips the group when nothing changed;
// otherwise packs the changed words of each 8-column half to the front with a
// permutation looked up from the half's change mask.
__attribute__((target("avx2,popcnt"))) int64_t EncodeBlockAvx2(
    const uint16_t* old_bits, const uint16_t* new_bits, int64_t cols,
    int64_t block_rows, int32_t* offsets, int32_t* values) {
  int64_t count = 0;
  const __m256i iota = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  const __m256i zero = _mm256_setzero_si256();
  for (int64_t pair = 0; pair < block_rows / 2; ++pair) {
    const uint16_t* old_even = old_bits + 2 * pair * cols;
    const uint16_t* old_odd = old_even + cols;
    const uint16_t* new_even = new_bits + 2 * pair * cols;
    const uint16_t* new_odd = new_even + cols;
    for (int64_t col = 0; col < cols; col += 16) {
      const __m256i even = LoadU16x16(new_even + col);
      const __m256i odd = LoadU16x16(new_odd + col);
      const __m256i diff =
          _mm256_or_si256(_mm256_xor_si256(even, LoadU16x16(old_even + col)),
                          _mm256_xor_si256(odd, LoadU16x16(old_odd + col)));
      if (_mm256_testz_si256(diff, diff)) continue;
      // 0xFFFF in every 16-bit lane whose word is unchanged.
      const __m256i same = _mm256_cmpeq_epi16(diff, zero);
      const __m128i same_halves[2] = {_mm256_castsi256_si128(same),
                                      _mm256_extracti128_si256(same, 1)};
      const __m128i even_halves[2] = {_mm256_castsi256_si128(even),
                                      _mm256_extracti128_si256(even, 1)};
      const __m128i odd_halves[2] = {_mm256_castsi256_si128(odd),
                                     _mm256_extracti128_si256(odd, 1)};
      // A half without changes is not skipped: it stores 8 words past `count`
      // that the next store overwrites, which is cheaper than a branch that
      // is taken about half the time.
      for (int half = 0; half < 2; ++half) {
        const unsigned mask =
            ~static_cast<unsigned>(_mm_movemask_epi8(
                _mm_packs_epi16(same_halves[half], same_halves[half]))) &
            0xFFu;
        const __m256i words = _mm256_or_si256(
            _mm256_cvtepu16_epi32(even_halves[half]),
            _mm256_slli_epi32(_mm256_cvtepu16_epi32(odd_halves[half]), 16));
        const __m256i word_offsets = _mm256_add_epi32(
            _mm256_set1_epi32(
                static_cast<int32_t>(pair * cols + col + 8 * half)),
            iota);
        const __m256i lanes = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(kCompressLut.lanes[mask]));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(values + count),
                            _mm256_permutevar8x32_epi32(words, lanes));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(offsets + count),
                            _mm256_permutevar8x32_epi32(word_offsets, lanes));
        count += __builtin_popcount(mask);
      }
    }
  }
  return count;
}

__attribute__((target("avx2,popcnt"))) int64_t
CountBlockAvx2(const uint16_t* old_bits, const uint16_t* new_bits, int64_t cols,
               int64_t block_rows) {
  int64_t count = 0;
  const __m256i zero = _mm256_setzero_si256();
  for (int64_t pair = 0; pair < block_rows / 2; ++pair) {
    const uint16_t* old_even = old_bits + 2 * pair * cols;
    const uint16_t* old_odd = old_even + cols;
    const uint16_t* new_even = new_bits + 2 * pair * cols;
    const uint16_t* new_odd = new_even + cols;
    for (int64_t col = 0; col < cols; col += 16) {
      const __m256i diff =
          _mm256_or_si256(_mm256_xor_si256(LoadU16x16(new_even + col),
                                           LoadU16x16(old_even + col)),
                          _mm256_xor_si256(LoadU16x16(new_odd + col),
                                           LoadU16x16(old_odd + col)));
      // Two mask bits per 16-bit lane.
      const unsigned same = static_cast<unsigned>(
          _mm256_movemask_epi8(_mm256_cmpeq_epi16(diff, zero)));
      count += __builtin_popcount(~same) / 2;
    }
  }
  return count;
}

bool CpuHasAvx2() {
  static const bool has_avx2 = [] {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("popcnt");
  }();
  return has_avx2;
}

EncodeBlockFn SelectEncodeBlock(bool force_scalar) {
  return !force_scalar && CpuHasAvx2() ? EncodeBlockAvx2 : EncodeBlockScalar;
}

CountBlockFn SelectCountBlock(bool force_scalar) {
  return !force_scalar && CpuHasAvx2() ? CountBlockAvx2 : CountBlockScalar;
}

#else  // !TUNIX_DELTA_HAVE_AVX2

bool CpuHasAvx2() { return false; }

EncodeBlockFn SelectEncodeBlock(bool) { return EncodeBlockScalar; }

CountBlockFn SelectCountBlock(bool) { return CountBlockScalar; }

#endif  // TUNIX_DELTA_HAVE_AVX2

// One [rows, cols] tensor and, when encoding, where its patches go.
struct Tensor {
  const uint16_t* old_bits = nullptr;
  const uint16_t* new_bits = nullptr;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t block_rows = 0;
  // Both [NumBlocks(), capacity], row-major.
  int32_t* offsets = nullptr;
  int32_t* values = nullptr;
  int64_t capacity = 0;

  int64_t NumBlocks() const { return rows / block_rows; }
  int64_t BlockElements() const { return block_rows * cols; }
};

// The address range [begin, end) of an array. Addresses are integers so that
// ranges of unrelated arrays can be ordered.
using ByteRange = std::pair<uintptr_t, uintptr_t>;

template <typename Array>
ByteRange RangeOf(const Array& array) {
  const uintptr_t begin = reinterpret_cast<uintptr_t>(array.data());
  return {begin, begin + array.nbytes()};
}

// Throws unless `array` is aligned for its element type. NumPy can produce
// misaligned arrays, e.g. from a buffer at an odd offset.
template <typename Array, typename T>
void CheckAligned(const Array& array, const char* name, size_t index) {
  if (reinterpret_cast<uintptr_t>(array.data()) % alignof(T) != 0) {
    throw std::invalid_argument(
        StrCat(name, "[", index, "] is not aligned to its element size"));
  }
}

// Validates the weights and returns a view of each tensor.
std::vector<Tensor> MakeTensors(const std::vector<BitsArray>& old_bits,
                                const std::vector<BitsArray>& new_bits,
                                const std::vector<int64_t>& block_rows,
                                int num_threads) {
  if (num_threads < 1) {
    throw std::invalid_argument(
        StrCat("num_threads=", num_threads, " must be at least 1"));
  }
  if (old_bits.size() != new_bits.size() ||
      old_bits.size() != block_rows.size()) {
    throw std::invalid_argument(
        StrCat("got ", old_bits.size(), " old_bits, ", new_bits.size(),
               " new_bits and ", block_rows.size(),
               " block_rows; expected one of each per tensor"));
  }
  std::vector<Tensor> tensors;
  tensors.reserve(old_bits.size());
  for (size_t i = 0; i < old_bits.size(); ++i) {
    Tensor t;
    t.rows = static_cast<int64_t>(old_bits[i].shape(0));
    t.cols = static_cast<int64_t>(old_bits[i].shape(1));
    t.block_rows = block_rows[i];
    if (new_bits[i].shape(0) != old_bits[i].shape(0) ||
        new_bits[i].shape(1) != old_bits[i].shape(1)) {
      throw std::invalid_argument(
          StrCat("new_bits[", i, "] has shape [", new_bits[i].shape(0), ", ",
                 new_bits[i].shape(1), "] but old_bits[", i, "] has shape [",
                 t.rows, ", ", t.cols, "]"));
    }
    if (t.block_rows <= 0 || t.block_rows % kTileRows != 0) {
      throw std::invalid_argument(
          StrCat("tensor ", i, ": block_rows=", t.block_rows,
                 " must be a positive multiple of ", kTileRows));
    }
    if (t.rows <= 0 || t.rows % t.block_rows != 0) {
      throw std::invalid_argument(
          StrCat("tensor ", i, ": rows=", t.rows,
                 " must be a positive multiple of block_rows=", t.block_rows));
    }
    if (t.cols < kTileCols || (t.cols & (t.cols - 1)) != 0) {
      throw std::invalid_argument(
          StrCat("tensor ", i, ": cols=", t.cols,
                 " must be a power of two >= ", kTileCols));
    }
    // Divide rather than multiply so that absurd dimensions cannot wrap.
    if (t.cols > std::numeric_limits<int32_t>::max() / (t.block_rows / 2)) {
      throw std::invalid_argument(StrCat("tensor ", i, ": a block of ",
                                         t.block_rows, " rows x ", t.cols,
                                         " cols overflows int32 offsets"));
    }
    CheckAligned<BitsArray, uint16_t>(old_bits[i], "old_bits", i);
    CheckAligned<BitsArray, uint16_t>(new_bits[i], "new_bits", i);
    t.old_bits = old_bits[i].data();
    t.new_bits = new_bits[i].data();
    tensors.push_back(t);
  }
  return tensors;
}

// Validates the patch arrays and points each tensor at its pair.
void AttachPatches(const std::vector<BitsArray>& old_bits,
                   const std::vector<BitsArray>& new_bits,
                   const std::vector<PatchArray>& offsets,
                   const std::vector<PatchArray>& values,
                   std::vector<Tensor>& tensors) {
  if (offsets.size() != tensors.size() || values.size() != tensors.size()) {
    throw std::invalid_argument(StrCat("got ", tensors.size(), " tensors but ",
                                       offsets.size(), " offsets and ",
                                       values.size(), " values arrays"));
  }
  std::vector<ByteRange> outputs;
  outputs.reserve(2 * tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    Tensor& t = tensors[i];
    t.capacity = static_cast<int64_t>(offsets[i].shape(1));
    if (values[i].shape(0) != offsets[i].shape(0) ||
        values[i].shape(1) != offsets[i].shape(1)) {
      throw std::invalid_argument(
          StrCat("values[", i, "] has shape [", values[i].shape(0), ", ",
                 values[i].shape(1), "] but offsets[", i, "] has shape [",
                 offsets[i].shape(0), ", ", t.capacity, "]"));
    }
    if (t.capacity <= 0 || t.capacity % kCapacityAlign != 0) {
      throw std::invalid_argument(
          StrCat("tensor ", i, ": capacity=", t.capacity,
                 " must be a positive multiple of ", kCapacityAlign));
    }
    if (static_cast<int64_t>(offsets[i].shape(0)) != t.NumBlocks()) {
      throw std::invalid_argument(
          StrCat("offsets[", i, "] has ", offsets[i].shape(0),
                 " rows but the tensor has ", t.NumBlocks(), " blocks"));
    }
    CheckAligned<PatchArray, int32_t>(offsets[i], "offsets", i);
    CheckAligned<PatchArray, int32_t>(values[i], "values", i);
    t.offsets = offsets[i].data();
    t.values = values[i].data();
    outputs.push_back(RangeOf(offsets[i]));
    outputs.push_back(RangeOf(values[i]));
  }
  // Threads write the outputs concurrently, so they must not share memory
  // with each other or with the weights. The weights may share memory among
  // themselves: they are only read. Once the outputs are known to be pairwise
  // disjoint, sorting them by begin also sorts them by end.
  std::sort(outputs.begin(), outputs.end());
  for (size_t i = 1; i < outputs.size(); ++i) {
    if (outputs[i].first < outputs[i - 1].second) {
      throw std::invalid_argument(
          "the offsets and values arrays must not share memory");
    }
  }
  for (const std::vector<BitsArray>* inputs : {&old_bits, &new_bits}) {
    for (const BitsArray& input : *inputs) {
      const ByteRange range = RangeOf(input);
      // The first output that ends after the input begins.
      const auto it =
          std::upper_bound(outputs.begin(), outputs.end(), range.first,
                           [](uintptr_t begin, const ByteRange& output) {
                             return begin < output.second;
                           });
      if (it != outputs.end() && it->first < range.second) {
        throw std::invalid_argument(
            "the offsets and values arrays must not share memory with the"
            " weights");
      }
    }
  }
}

// A run of blocks of one tensor that a worker processes in one go.
struct Chunk {
  size_t tensor;
  int64_t block_begin;
  int64_t block_end;
};

// Chunks in ascending (tensor, block) order.
std::vector<Chunk> MakeChunks(const std::vector<Tensor>& tensors) {
  std::vector<Chunk> chunks;
  for (size_t i = 0; i < tensors.size(); ++i) {
    const Tensor& t = tensors[i];
    const int64_t block_bytes =
        t.BlockElements() * static_cast<int64_t>(sizeof(uint16_t));
    const int64_t step = std::max<int64_t>(1, kChunkBytes / block_bytes);
    for (int64_t b = 0; b < t.NumBlocks(); b += step) {
      chunks.push_back({i, b, std::min(b + step, t.NumBlocks())});
    }
  }
  return chunks;
}

// Hands out chunk indices in ascending order to concurrent workers.
class ChunkQueue {
 public:
  explicit ChunkQueue(size_t num_chunks) : limit_(num_chunks) {}

  // Claims the next chunk. Returns false once none is left below the limit.
  bool Claim(size_t* index) {
    *index = next_.fetch_add(1, std::memory_order_relaxed);
    return *index < limit_.load(std::memory_order_relaxed);
  }

  // Stops handing out chunks at or after `index`. Chunks before it are still
  // processed.
  void StopAt(size_t index) {
    size_t limit = limit_.load(std::memory_order_relaxed);
    while (index < limit && !limit_.compare_exchange_weak(
                                limit, index, std::memory_order_relaxed)) {
    }
  }

 private:
  std::atomic<size_t> next_{0};
  std::atomic<size_t> limit_;
};

// Returns how many workers to run for `num_chunks` chunks.
size_t NumWorkers(int num_threads, size_t num_chunks) {
  return std::clamp<size_t>(num_chunks, 1, static_cast<size_t>(num_threads));
}

// Runs `fn(worker, chunk_index, queue)` over all chunks on `num_workers`
// threads, the calling thread included. Rethrows the first exception a worker
// raised, after every thread has stopped.
template <typename Fn>
void RunChunks(size_t num_chunks, size_t num_workers, Fn fn) {
  ChunkQueue queue(num_chunks);
  std::mutex mutex;  // NOLINT(build/c++11)
  std::exception_ptr error;
  auto worker = [&](size_t index) {
    try {
      size_t chunk;
      while (queue.Claim(&chunk)) fn(index, chunk, queue);
    } catch (...) {
      queue.StopAt(0);
      // NOLINTNEXTLINE(build/c++11)
      const std::lock_guard<std::mutex> lock(mutex);
      if (!error) error = std::current_exception();
    }
  };
  {
    // Joins on every path out of this scope, including a failed thread
    // creation, so no thread outlives the locals it references.
    struct Joiner {
      ~Joiner() {
        queue.StopAt(0);
        for (std::thread& thread : threads) thread.join();
      }
      ChunkQueue& queue;
      std::vector<std::thread> threads;
    } joiner{queue, {}};
    joiner.threads.reserve(num_workers - 1);
    for (size_t index = 1; index < num_workers; ++index) {
      joiner.threads.emplace_back(worker, index);
    }
    // Returns only when no chunk is left to claim, so stopping the queue in
    // ~Joiner does not drop work.
    worker(0);
  }
  if (error) std::rethrow_exception(error);
}

std::vector<int64_t> FullestBlockChanges(const std::vector<BitsArray>& old_bits,
                                         const std::vector<BitsArray>& new_bits,
                                         const std::vector<int64_t>& block_rows,
                                         int num_threads, bool force_scalar) {
  const std::vector<Tensor> tensors =
      MakeTensors(old_bits, new_bits, block_rows, num_threads);
  const std::vector<Chunk> chunks = MakeChunks(tensors);
  const size_t num_workers = NumWorkers(num_threads, chunks.size());
  const CountBlockFn count_block = SelectCountBlock(force_scalar);
  // fullest[worker][tensor].
  std::vector<std::vector<int64_t>> fullest(
      num_workers, std::vector<int64_t>(tensors.size()));
  {
    nb::gil_scoped_release release;
    RunChunks(chunks.size(), num_workers,
              [&](size_t worker, size_t index, ChunkQueue&) {
                const Chunk& chunk = chunks[index];
                const Tensor& t = tensors[chunk.tensor];
                int64_t most = fullest[worker][chunk.tensor];
                for (int64_t b = chunk.block_begin; b < chunk.block_end; ++b) {
                  most = std::max(
                      most, count_block(t.old_bits + b * t.BlockElements(),
                                        t.new_bits + b * t.BlockElements(),
                                        t.cols, t.block_rows));
                }
                fullest[worker][chunk.tensor] = most;
              });
  }
  std::vector<int64_t> result(tensors.size());
  for (const std::vector<int64_t>& per_worker : fullest) {
    for (size_t i = 0; i < result.size(); ++i) {
      result[i] = std::max(result[i], per_worker[i]);
    }
  }
  return result;
}

// What one worker thread owns while encoding.
struct EncodeState {
  std::vector<int32_t> scratch_offsets;
  std::vector<int32_t> scratch_values;
  int64_t changed_words = 0;
  // The block that did not fit, if any. A worker stops at its first one.
  bool overflowed = false;
  size_t overflow_chunk = 0;
  int64_t overflow_block = 0;
  int64_t overflow_count = 0;
};

int64_t EncodeBlockPatches(const std::vector<BitsArray>& old_bits,
                           const std::vector<BitsArray>& new_bits,
                           const std::vector<int64_t>& block_rows,
                           const std::vector<PatchArray>& offsets,
                           const std::vector<PatchArray>& values,
                           int num_threads, bool force_scalar) {
  std::vector<Tensor> tensors =
      MakeTensors(old_bits, new_bits, block_rows, num_threads);
  AttachPatches(old_bits, new_bits, offsets, values, tensors);
  const std::vector<Chunk> chunks = MakeChunks(tensors);
  const size_t num_workers = NumWorkers(num_threads, chunks.size());
  const EncodeBlockFn encode_block = SelectEncodeBlock(force_scalar);

  int64_t scratch_words = 0;
  for (const Tensor& t : tensors) {
    scratch_words = std::max(scratch_words, t.block_rows / 2 * t.cols);
  }
  scratch_words += kScratchSlack;
  // Allocated before any thread starts, so the workers cannot run out of
  // memory.
  std::vector<EncodeState> states(num_workers);
  for (EncodeState& state : states) {
    state.scratch_offsets.resize(scratch_words);
    state.scratch_values.resize(scratch_words);
  }

  {
    nb::gil_scoped_release release;
    RunChunks(chunks.size(), num_workers,
              [&](size_t worker, size_t index, ChunkQueue& queue) {
                EncodeState& state = states[worker];
                const Chunk& chunk = chunks[index];
                const Tensor& t = tensors[chunk.tensor];
                for (int64_t b = chunk.block_begin; b < chunk.block_end; ++b) {
                  const int64_t count =
                      encode_block(t.old_bits + b * t.BlockElements(),
                                   t.new_bits + b * t.BlockElements(), t.cols,
                                   t.block_rows, state.scratch_offsets.data(),
                                   state.scratch_values.data());
                  if (count > t.capacity) {
                    // The call fails, so later chunks are not worth encoding.
                    // Earlier ones still run, to report the first block that
                    // does not fit.
                    state.overflowed = true;
                    state.overflow_chunk = index;
                    state.overflow_block = b;
                    state.overflow_count = count;
                    queue.StopAt(index);
                    return;
                  }
                  state.changed_words += count;
                  int32_t* block_offsets = t.offsets + b * t.capacity;
                  int32_t* block_values = t.values + b * t.capacity;
                  std::memcpy(block_offsets, state.scratch_offsets.data(),
                              count * sizeof(int32_t));
                  std::memcpy(block_values, state.scratch_values.data(),
                              count * sizeof(int32_t));
                  std::fill(block_offsets + count, block_offsets + t.capacity,
                            kPadOffset);
                  std::fill(block_values + count, block_values + t.capacity, 0);
                }
              });
  }

  int64_t changed_words = 0;
  const EncodeState* first_overflow = nullptr;
  for (const EncodeState& state : states) {
    changed_words += state.changed_words;
    if (state.overflowed &&
        (first_overflow == nullptr ||
         state.overflow_chunk < first_overflow->overflow_chunk)) {
      first_overflow = &state;
    }
  }
  if (first_overflow != nullptr) {
    const size_t tensor = chunks[first_overflow->overflow_chunk].tensor;
    throw std::overflow_error(StrCat(
        "tensor ", tensor, ": block ", first_overflow->overflow_block, " has ",
        first_overflow->overflow_count,
        " changed words, more than capacity=", tensors[tensor].capacity));
  }
  return changed_words;
}

std::string SimdKernel() { return CpuHasAvx2() ? "avx2" : "scalar"; }

}  // namespace

NB_MODULE(_delta_encoder_ext, m) {
  m.doc() = "Multi-threaded host-side encoder for delta weight sync.";
  m.attr("PAD_OFFSET") = kPadOffset;
  m.attr("CAPACITY_ALIGN") = kCapacityAlign;
  m.def("simd_kernel", &SimdKernel,
        "Returns the kernel this CPU uses by default: 'avx2' or 'scalar'.");
  m.def("fullest_block_changes", &FullestBlockChanges,
        nb::arg("old_bits").noconvert(), nb::arg("new_bits").noconvert(),
        nb::kw_only(), nb::arg("block_rows"), nb::arg("num_threads"),
        nb::arg("force_scalar") = false,
        R"(Returns the changed-word count of each tensor's fullest block.

Round a count up to a positive multiple of CAPACITY_ALIGN to get a capacity
that fits.

Args:
  old_bits: per tensor, a C-contiguous uint16 [rows, cols] array holding the
    bf16 bit patterns of the previously synced weights (`.view(np.uint16)` of
    a bf16 array). cols must be a power of two >= 128.
  new_bits: the new weights, with the same shapes.
  block_rows: per tensor, the rows per block: a multiple of 16 that divides
    rows.
  num_threads: worker threads to scan with, the calling thread included. The
    scan is bound by memory bandwidth.
  force_scalar: use the portable kernel even when the CPU supports AVX2. For
    tests and benchmarks.

Raises:
  TypeError: if an array has the wrong dtype or rank or is not C-contiguous.
  ValueError: if a layout is not supported.
)");
  m.def("encode_block_patches", &EncodeBlockPatches,
        nb::arg("old_bits").noconvert(), nb::arg("new_bits").noconvert(),
        nb::kw_only(), nb::arg("block_rows"), nb::arg("offsets").noconvert(),
        nb::arg("values").noconvert(), nb::arg("num_threads"),
        nb::arg("force_scalar") = false,
        R"(Writes the patches that turn old_bits into new_bits.

Args:
  old_bits: per tensor, a C-contiguous uint16 [rows, cols] array holding the
    bf16 bit patterns of the previously synced weights (`.view(np.uint16)` of
    a bf16 array). cols must be a power of two >= 128.
  new_bits: the new weights, with the same shapes.
  block_rows: per tensor, the rows per block: a multiple of 16 that divides
    rows.
  offsets: per tensor, a writeable C-contiguous int32
    [rows // block_rows, capacity] array to fill. capacity must be a positive
    multiple of CAPACITY_ALIGN.
  values: per tensor, an array like `offsets` to fill. The output arrays must
    not share memory with each other or with the weights.
  num_threads: worker threads to encode with, the calling thread included.
  force_scalar: use the portable kernel even when the CPU supports AVX2. For
    tests and benchmarks.

Returns:
  The number of changed words across all tensors.

Raises:
  TypeError: if an array has the wrong dtype or rank, is not C-contiguous, or
    an output array is read-only.
  ValueError: if a layout or array shape is not supported.
  OverflowError: if a block has more changed words than its capacity; the
    message names the first such block. The output arrays then hold
    unspecified data and must not be applied.
)");
}

}  // namespace tunix::delta_sync
