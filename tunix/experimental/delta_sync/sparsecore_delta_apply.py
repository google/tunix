# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Applies sparse bf16 weight deltas in place with a TPU SparseCore kernel.

This is the rollout half of delta weight sync: after an RL step only ~1-2% of
the bf16 weights change, so the trainer ships just the changed words and the
rollout engine patches its own copy of the weights.

Why stream instead of scatter: writing the changed words one at a time is bound
by the number of separate HBM writes (~0.38G words/s on a v6e chip), far below
HBM bandwidth. At this change rate every 4 KiB tile holds dozens of changes, so
it is cheaper to read each block into SparseCore VMEM, patch it there and write
it back, one sequential pass over the tensor at ~1.1 TB/s. Roughly, streaming
wins once more than ~0.13% of the elements change. On one v6e chip, patching 80
routed-expert tensors (16.1 GB, 138M changed words) takes 30.7 ms, against
1.70 s for an XLA scatter and ~0.37 s for a SparseCore kernel that writes each
changed word with an indirect DMA.

Payload layout. With TensorCore tiling, TPU stores a 2D bf16 array in (16, 128)
tiles in which one 32-bit word holds rows 2p and 2p+1 of the same column.
Bitcasting a VMEM block to int32 therefore gives a [block_rows // 2, cols]
array of these row-pair words, and every patch is a plain 32-bit store. Per
block b, slot i:

  offsets[b, i] = (local_row // 2) * cols + col, or PAD_OFFSET for padding
  values[b, i]  = new_bits[2p + 1, col] << 16 | new_bits[2p, col]

A word is sent when either of its rows changed and always carries both new
halves, so a patch only sets words and applying it twice is a no-op. Every
block's list is padded to the same `capacity` so it moves in one fixed-size
DMA.

Limits (validated on v6e): `cols` is a power of two, at least 128, and small
enough for two blocks to fit in a subcore's VMEM (at most 2048 for 16-row
blocks on v6e, which has 256 KiB per subcore); `block_rows` is a multiple of
16; and `rows // block_rows` is a multiple of twice the SparseCore subcore
count (64 on v6e). The kernel patches one device's array; it is not SPMD
partitioned.
"""

from collections.abc import Callable
import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc
import jax.numpy as jnp
import numpy as np

# Marks an unused slot in `offsets`; the kernel masks those lanes off.
PAD_OFFSET = -1
# `capacity` must be a multiple of this, so per-block DMAs stay granule
# aligned and the vector loop needs no tail.
CAPACITY_ALIGN = 128

# Rows in one bf16 tile; a block must cover whole tiles.
_TILE_ROWS = 16
# Columns in one tile.
_TILE_COLS = 128


def _check_layout(rows: int, cols: int, block_rows: int) -> None:
  """Raises ValueError unless [rows, cols] splits into whole-tile blocks."""
  if block_rows <= 0 or block_rows % _TILE_ROWS:
    raise ValueError(
        f"block_rows={block_rows} must be a positive multiple of {_TILE_ROWS}"
    )
  if rows <= 0 or rows % block_rows:
    raise ValueError(
        f"rows={rows} must be a positive multiple of block_rows={block_rows}"
    )
  # A power of two lets the kernel split offsets with a shift and a mask.
  if cols < _TILE_COLS or cols & (cols - 1):
    raise ValueError(f"cols={cols} must be a power of two >= {_TILE_COLS}")


def _check_capacity(capacity: int) -> None:
  if capacity <= 0 or capacity % CAPACITY_ALIGN:
    raise ValueError(
        f"capacity={capacity} must be a positive multiple of {CAPACITY_ALIGN}"
    )


def encode_block_patches(
    old_bits: np.ndarray,
    new_bits: np.ndarray,
    *,
    block_rows: int,
    capacity: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Builds the per-block patches that turn `old_bits` into `new_bits`.

  Host-side reference for the payload `apply_block_patches` consumes.

  Args:
    old_bits: uint16 [rows, cols], bf16 bit patterns of the current weights.
    new_bits: uint16 [rows, cols], bf16 bit patterns of the target weights.
    block_rows: rows per block, a multiple of 16; the kernel streams one block
      at a time. The kernel also needs `rows // block_rows` to be a multiple of
      twice the SparseCore subcore count.
    capacity: slots per block, a positive multiple of 128. Keep it fixed per
      tensor shape so the apply does not recompile from step to step.

  Returns:
    `(offsets, values)`, both int32 [rows // block_rows, capacity], laid out as
    described in the module docstring. Within a block the words are sorted by
    offset and unique.

  Raises:
    ValueError: if the inputs are not uint16 matrices of the same shape, the
      shape does not split into whole-tile blocks, `capacity` is not a positive
      multiple of 128, or a block has more than `capacity` changed words.
  """
  if old_bits.dtype != np.uint16 or new_bits.dtype != np.uint16:
    raise ValueError(
        f"expected uint16 bit patterns, got {old_bits.dtype} and"
        f" {new_bits.dtype}"
    )
  if old_bits.ndim != 2 or old_bits.shape != new_bits.shape:
    raise ValueError(
        "expected two matrices of the same shape, got"
        f" {old_bits.shape} and {new_bits.shape}"
    )
  rows, cols = old_bits.shape
  _check_layout(rows, cols, block_rows)
  _check_capacity(capacity)
  num_blocks = rows // block_rows
  pairs_per_block = block_rows // 2

  changed = old_bits != new_bits
  # Row-major order, so the words of each block come out contiguous.
  pair_rows, pair_cols = np.nonzero(changed[0::2] | changed[1::2])
  blocks = pair_rows // pairs_per_block
  counts = np.bincount(blocks, minlength=num_blocks)
  fullest = int(counts.max())
  if fullest > capacity:
    raise ValueError(
        f"block {int(counts.argmax())} has {fullest} changed words, more than"
        f" capacity={capacity}"
    )
  slots = np.arange(blocks.size) - (np.cumsum(counts) - counts)[blocks]
  even_rows = new_bits[2 * pair_rows, pair_cols].astype(np.uint32)
  odd_rows = new_bits[2 * pair_rows + 1, pair_cols].astype(np.uint32)

  offsets = np.full((num_blocks, capacity), PAD_OFFSET, np.int32)
  values = np.zeros((num_blocks, capacity), np.int32)
  offsets[blocks, slots] = (pair_rows % pairs_per_block) * cols + pair_cols
  values[blocks, slots] = ((odd_rows << 16) | even_rows).view(np.int32)
  return offsets, values


def _build_patch_kernel(
    rows: int, cols: int, block_rows: int, capacity: int
) -> Callable[[jax.Array, jax.Array, jax.Ref], tuple[()]]:
  """Returns the SparseCore kernel that patches a [rows, cols] bf16 ref."""
  info = plsc.get_sparse_core_info()
  num_workers = info.num_cores * info.num_subcores
  num_blocks = rows // block_rows
  # Each worker owns an even run of blocks: the loop below handles two blocks
  # per step so that both buffer choices are static.
  if num_blocks % (2 * num_workers):
    raise ValueError(
        f"{num_blocks} blocks do not split into pairs over {num_workers}"
        " SparseCore subcores; rows // block_rows must be a multiple of"
        f" {2 * num_workers}"
    )
  vmem_bytes = 2 * (block_rows * cols * 2 + 2 * capacity * 4)
  if vmem_bytes > info.vmem_capacity_bytes:
    raise ValueError(
        f"double-buffered block_rows={block_rows}, cols={cols},"
        f" capacity={capacity} needs {vmem_bytes} bytes of VMEM per subcore,"
        f" more than {info.vmem_capacity_bytes}"
    )
  blocks_per_worker = num_blocks // num_workers
  lanes = info.num_lanes
  col_bits = cols.bit_length() - 1
  mesh = plsc.VectorSubcoreMesh(
      core_axis_name="core",
      subcore_axis_name="subcore",
      num_cores=info.num_cores,
  )

  @pl.kernel(
      mesh=mesh,
      # The row-pair int32 view below relies on TensorCore tiling of the bf16
      # blocks (the SparseCore default; kept explicit).
      compiler_params=pltpu.CompilerParams(use_tc_tiling_on_sc=True),
      scratch_types=dict(
          weights0=pltpu.VMEM((block_rows, cols), jnp.bfloat16),
          weights1=pltpu.VMEM((block_rows, cols), jnp.bfloat16),
          offsets0=pltpu.VMEM((capacity,), jnp.int32),
          offsets1=pltpu.VMEM((capacity,), jnp.int32),
          values0=pltpu.VMEM((capacity,), jnp.int32),
          values1=pltpu.VMEM((capacity,), jnp.int32),
          sems=pltpu.SemaphoreType.DMA((3, 2)),
      ),
  )
  def kernel(
      offsets_hbm,
      values_hbm,
      weights_hbm,
      *,
      weights0,
      weights1,
      offsets0,
      offsets1,
      values0,
      values1,
      sems,
  ):
    worker = lax.axis_index("core") * info.num_subcores + lax.axis_index(
        "subcore"
    )
    first_block = worker * blocks_per_worker
    buffers = ((weights0, offsets0, values0), (weights1, offsets1, values1))

    def rows_of(i):
      return pl.ds((first_block + i) * block_rows, block_rows)

    def fetch(i, slot):
      """The three reads that bring block `i` and its patch into `slot`."""
      weights, offsets, values = buffers[slot]
      return (
          pltpu.make_async_copy(
              weights_hbm.at[rows_of(i), :], weights, sems.at[0, slot]
          ),
          pltpu.make_async_copy(
              offsets_hbm.at[first_block + i], offsets, sems.at[1, slot]
          ),
          pltpu.make_async_copy(
              values_hbm.at[first_block + i], values, sems.at[2, slot]
          ),
      )

    for copy in fetch(0, 0):
      copy.start()

    @pl.loop(0, blocks_per_worker // 2)
    def _(step):
      for slot in (0, 1):
        i = 2 * step + slot
        for copy in fetch(i, slot):
          copy.wait()

        # Prefetch the next block into the other slot; its write-back
        # completed synchronously at the end of the previous block.
        @pl.when(i + 1 < blocks_per_worker)
        def _():
          for copy in fetch(i + 1, 1 - slot):
            copy.start()

        weights, offsets, values = buffers[slot]
        pair_words = weights.bitcast(jnp.int32)  # [block_rows // 2, cols]

        @plsc.parallel_loop(0, capacity, step=lanes)
        def _(k):
          offset = offsets[pl.ds(k, lanes)]
          valid = offset >= 0
          offset = jnp.where(valid, offset, 0)
          plsc.store_scatter(
              pair_words,
              [offset >> col_bits, offset & (cols - 1)],
              values[pl.ds(k, lanes)],
              mask=valid,
          )

        pltpu.sync_copy(weights, weights_hbm.at[rows_of(i), :])

  return kernel


@functools.partial(jax.jit, donate_argnums=0)
def apply_block_patches(
    weights: jax.Array, /, *, offsets: jax.Array, values: jax.Array
) -> jax.Array:
  """Writes per-block row-pair patches into `weights` in place.

  `weights` is donated and updated in place; use the returned array. It must
  hold the same bits the payload was encoded against (`old_bits`).

  The kernel patches one device's array and is not SPMD partitioned. For
  sharded weights, call it once per addressable shard with a payload encoded
  for that shard, and reassemble the results with
  `jax.make_array_from_single_device_arrays`.

  Args:
    weights: bf16 [rows, cols] on one TPU chip with SparseCores. `cols` must be
      a power of two >= 128, and two blocks must fit in a subcore's VMEM.
    offsets: int32 [num_blocks, capacity] from `encode_block_patches`; fixes
      block_rows = rows // num_blocks, a multiple of 16. num_blocks must be a
      multiple of twice the chip's SparseCore subcore count (64 on v6e), and
      capacity a positive multiple of 128. The kernel does not check the
      entries: each block's used slots must hold unique offsets in [0,
        block_rows // 2 * cols). An out-of-range offset writes outside the block
        and can corrupt other rows, so validate payloads that arrive over the
        network.
    values: int32 [num_blocks, capacity] from `encode_block_patches`.

  Returns:
    The patched weights, aliasing the donated input buffer.

  Raises:
    ValueError: if the shapes or dtypes do not fit the layout above, or the
      double-buffered block does not fit in SparseCore VMEM.
  """
  if weights.dtype != jnp.bfloat16 or weights.ndim != 2:
    raise ValueError(
        f"expected a bf16 matrix, got {weights.dtype}{list(weights.shape)}"
    )
  if offsets.dtype != jnp.int32 or values.dtype != jnp.int32:
    raise ValueError(
        f"expected int32 offsets and values, got {offsets.dtype} and"
        f" {values.dtype}"
    )
  if offsets.ndim != 2 or offsets.shape != values.shape:
    raise ValueError(
        "expected offsets and values of the same [num_blocks, capacity]"
        f" shape, got {offsets.shape} and {values.shape}"
    )
  rows, cols = weights.shape
  num_blocks, capacity = offsets.shape
  if num_blocks <= 0 or rows % num_blocks:
    raise ValueError(f"{num_blocks} blocks do not evenly split {rows} rows")
  block_rows = rows // num_blocks
  _check_layout(rows, cols, block_rows)
  _check_capacity(capacity)

  kernel = _build_patch_kernel(rows, cols, block_rows, capacity)
  # A mutable ref over the donated buffer lets the kernel write in place.
  weights_ref = jax.new_ref(weights)
  kernel(offsets, values, weights_ref)
  return jax.freeze(weights_ref)
