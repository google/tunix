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

BM = 128
BN = 256
BK = 256
CONTRACT_TILES = frozenset({(BM, BN, BK), (128, 128, 128)})
BLOCK_N_CANDIDATES = (1024, 1280, 640, 512)
INPUT_FUSION_ENV = "CANON_PALLAS_INPUT_FUSION"


def tile_policy(
    m: int, k: int, n: int, tiles: tuple[int, int, int] = (BM, BN, BK)
) -> tuple[int, int, int]:
  """Returns `(block_m, block_n, block_k)` for an `[m, k] @ [k, n]` bf16 matmul.

  `block_m` is promoted to `256` when `m % 256 == 0`; `block_n` is promoted to
  the first candidate in `BLOCK_N_CANDIDATES` dividing `n`; `block_k` is kept
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
  if m % 256 == 0:
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
  if (block_m, block_n, block_k) in CONTRACT_TILES:
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
