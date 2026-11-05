# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fixed-tile, batch-invariant Pallas TPU matmul.

Why this kernel exists: XLA's `jnp.dot` picks a different tiling/reduction
strategy per operand shape, so the value of one output row depends on `M` (the
number of rows in the batch). Decode (`M=1..8`) and prefill/scoring (`M=bucket`)
therefore disagree by a few ULPs. This kernel fixes the per-element contraction
order: every output element is `sum_q x[i, q*BK:(q+1)*BK] @
y[q*BK:(q+1)*BK, j]` accumulated left-to-right in `f32` over `q` and rounded to
`bf16` exactly once, independently of `M`, of the row position, and of the
contents of any other row. `block_m` / `block_n` only choose which output tile a
program writes, so they are bitwise neutral; `block_k` is the only tile that
enters the numerics and is never changed by `tile_policy`.
"""

from __future__ import annotations

import os

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

MIN_BLOCK_M = 8
BM = 128
BN = 256
BK = 256
CONTRACT_TILES = frozenset({(BM, BN, BK), (128, 128, 128)})
CONTRACT_NK_TILES = frozenset({(BN, BK), (128, 128)})
BLOCK_N_CANDIDATES = (1024, 1280, 640, 512)
INPUT_FUSION_ENV = "CANON_PALLAS_INPUT_FUSION"


def adaptive_row_bucket(m: int, default_bm: int = BM) -> tuple[int, int]:
  """Returns `(m_padded, block_m)` using power-of-2 row buckets for small `m`.

  For `m < default_bm`, selects the smallest power-of-2 bucket in
  `[MIN_BLOCK_M, default_bm)` that is `>= m` (aligned to TPU VREG sublanes
  `MIN_BLOCK_M = 8`), avoiding padding small decode batches all the way to `128`
  or `256`. Because rows are independent in the matmul contraction, `block_m`
  is bitwise neutral. For `m >= default_bm` (or when the next power of 2 reaches
  `default_bm`), rounds `m` up to a multiple of `default_bm`.

  Args:
    m: Semantic row count (`> 0`).
    default_bm: Target row block/chunk size for full batches (`> 0`).

  Returns:
    `(m_padded, block_m)`.
  """
  if m <= 0 or default_bm <= 0:
    raise ValueError(
        "adaptive_row_bucket requires positive (m, default_bm), got "
        f"{(m, default_bm)}"
    )
  if default_bm >= MIN_BLOCK_M and m < default_bm:
    bucket = max(MIN_BLOCK_M, 1 << (m - 1).bit_length())
    if bucket < default_bm:
      return bucket, bucket
  mp = ((m + default_bm - 1) // default_bm) * default_bm
  return mp, default_bm


def tile_policy(
    m: int, k: int, n: int, tiles: tuple[int, int, int] = (BM, BN, BK)
) -> tuple[int, int, int]:
  """Returns `(block_m, block_n, block_k)` for an `[m, k] @ [k, n]` bf16 matmul.

  `block_m` is promoted from `BM` (`128`) to `256` when `m % 256 == 0` (smaller
  adaptive decode `block_m` values are preserved); `block_n` is promoted to the
  first candidate in `BLOCK_N_CANDIDATES` dividing `n`; `block_k` is kept
  unchanged so the per-element accumulation order never changes.

  Args:
    m: Rows of the left operand.
    k: Contracted extent (unused; kept for signature parity).
    n: Columns of the right operand.
    tiles: The caller's `(block_m, block_n, block_k)`.

  Returns:
    The `(block_m, block_n, block_k)` the kernel runs with.
  """
  del k
  block_m, block_n, block_k = tiles
  if block_m == BM and m % 256 == 0:
    block_m = 256
  for candidate in BLOCK_N_CANDIDATES:
    if n % candidate == 0:
      block_n = candidate
      break
  return block_m, block_n, block_k


def input_fusion_enabled() -> bool:
  """Whether Pallas producer (input) fusion is requested (compile-only knob)."""
  return os.environ.get(INPUT_FUSION_ENV, "") == "1"


def matmul(
    x,
    y,
    *,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
    block_m: int = BM,
    block_n: int = BN,
    block_k: int = BK,
):
  """Computes `bf16 [M, K] @ [K, N]` with fixed `BM/BN/BK` and `f32` accumulation."""
  if x.ndim != 2 or y.ndim != 2:
    raise ValueError(
        f"pallas_matmul expects rank-2 inputs, got {x.shape}, {y.shape} "
    )
  m, k = map(int, x.shape)
  ky, n = map(int, y.shape)
  if k != ky:
    raise ValueError(f"pallas_matmul contracted dimensions differ: {k} vs {ky}")
  if x.dtype != jnp.bfloat16 or y.dtype != jnp.bfloat16:
    raise ValueError(
        f"pallas_matmul requires bf16 inputs, got {x.dtype}, {y.dtype} "
    )
  is_adaptive_small_m = (
      MIN_BLOCK_M <= block_m < BM
      and block_m % MIN_BLOCK_M == 0
      and block_m == m
      and (block_n, block_k) in CONTRACT_NK_TILES
  )
  if (block_m, block_n, block_k) in CONTRACT_TILES or is_adaptive_small_m:
    block_m, block_n, block_k = tile_policy(
        m, k, n, (block_m, block_n, block_k)
    )
  if min(block_m, block_n, block_k) <= 0:
    raise ValueError("pallas_matmul block sizes must be positive")
  if m % block_m or n % block_n or k % block_k:
    raise ValueError(
        "pallas_matmul shape must divide BM/BN/BK="
        f"{block_m}/{block_n}/{block_k}, got {(m, k, n)}"
    )

  def _kernel(x_ref, y_ref, out_ref, acc_ref):
    @pl.when(pl.program_id(2) == 0)
    def _init():
      acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] = acc_ref[...] + jnp.dot(
        x_ref[...],
        y_ref[...],
        precision=jax.lax.Precision.DEFAULT,
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
    def _finalize():
      out_ref[...] = acc_ref[...].astype(out_ref.dtype)

  return pl.pallas_call(
      _kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((block_m, block_k), lambda i, _j, q: (i, q)),
              pl.BlockSpec((block_k, block_n), lambda _i, j, q: (q, j)),
          ],
          out_specs=pl.BlockSpec((block_m, block_n), lambda i, j, _q: (i, j)),
          grid=(m // block_m, n // block_n, k // block_k),
          scratch_shapes=[pltpu.VMEM((block_m, block_n), jnp.float32)],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "arbitrary"),
          allow_input_fusion=(
              (True, True) if input_fusion_enabled() else (False, False)
          ),
          shape_invariant_numerics=shape_invariant_numerics,
      ),
      interpret=interpret,
      name=f"canon_matmul_bm{block_m}_bn{block_n}_bk{block_k}",
  )(x, y)
