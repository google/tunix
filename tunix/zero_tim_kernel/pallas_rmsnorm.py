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
"""Fixed-feature-order `bf16` RMSNorm Pallas kernel.

Each row's sum of squares is accumulated in `f32` over static `BF=128` feature
blocks strictly left-to-right, so the reduction order depends only on `F`, never
on the number of rows `M`, the row tile `BM`, or the backend's reduction
strategy. `canonical_rmsnorm` is the pure-JAX statement of this arithmetic; it
is invoked directly inside both `pallas_rmsnorm` and `pallas_norm_matmul` as a
single source of truth and differentiated by `canonical_vjp`.
"""

from __future__ import annotations

import math

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tunix.zero_tim_kernel import model_contracts
from tunix.zero_tim_kernel import pallas_matmul

BM = 8
BF = 128
ROW_TILE_WIDE = 256
# Cap wide 256-row tiles at `F <= 4096` so `(block_m, F)` `f32` intermediates
# stay comfortably within TPU scoped VMEM.
ROW_TILE_MAX_FEATURES = 4096


def validate_shape(shape_x, shape_weight) -> tuple[int, int]:
  """Validates rank-2 `x` and rank-1 `weight` shapes against the `BF` contract."""
  if len(shape_x) != 2:
    raise ValueError(f"pallas_rmsnorm kernel expects rank-2 x, got {shape_x}")
  if len(shape_weight) != 1:
    raise ValueError(
        f"pallas_rmsnorm kernel expects rank-1 weight, got {shape_weight} "
    )
  m, f = map(int, shape_x)
  fw = int(shape_weight[0])
  if f != fw:
    raise ValueError(f"pallas_rmsnorm feature/weight mismatch: {f} vs {fw}")
  if m <= 0 or f <= 0 or f % BF != 0:
    raise ValueError(
        f"pallas_rmsnorm requires M > 0 and F % BF={BF} == 0, got {(m, f)}"
    )
  return m, f


def row_tile(m: int, f: int) -> int:
  """Returns the largest power-of-2 `block_m` in `[BM, 256]` dividing `m` (`f <= 4096`)."""
  if m <= 0 or m % BM != 0 or f > ROW_TILE_MAX_FEATURES:
    return BM
  return max(BM, math.gcd(m, ROW_TILE_WIDE))


def canonical_rmsnorm(
    x: jax.Array,
    weight: jax.Array,
    *,
    epsilon: float,
    out_dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> jax.Array:
  """Declared `pallas_rmsnorm` semantic: fixed `BF`-block `f32` accumulation."""
  m, f = validate_shape(x.shape, weight.shape)
  if x.dtype != jnp.bfloat16 or weight.dtype != jnp.bfloat16:
    raise ValueError(
        f"pallas_rmsnorm requires bf16 inputs, got {x.dtype}, {weight.dtype}"
    )
  if not float(epsilon) > 0.0:
    raise ValueError(f"pallas_rmsnorm epsilon must be positive, got {epsilon}")
  xf = x.astype(jnp.float32).reshape(m, f // BF, BF)
  wf = weight.astype(jnp.float32)

  # Keep the `BF`-block reduction as a Python-unrolled loop (not `lax.fori_loop`
  # or `lax.scan`): `canonical_rmsnorm` executes directly inside the Pallas TPU
  # kernels (`pallas_rmsnorm` and `pallas_norm_matmul`), where Mosaic TPU cannot
  # lower the `dynamic_slice` emitted by a dynamic loop over `q`.
  sumsq = jnp.zeros((m,), dtype=jnp.float32)
  for q in range(f // BF):
    block = xf[:, q, :]
    sumsq = sumsq + jnp.sum(block * block, axis=-1, dtype=jnp.float32)
  inv = jax.lax.rsqrt(sumsq / jnp.float32(f) + jnp.float32(epsilon))
  return (x.astype(jnp.float32) * inv[:, None] * wf[None, :]).astype(out_dtype)


def rmsnorm(
    x: jax.Array,
    weight: jax.Array,
    *,
    epsilon: float,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
) -> jax.Array:
  """Applies `canonical_rmsnorm` inside a Pallas kernel (auto-padding `M` to `BM`)."""
  m, f = validate_shape(x.shape, weight.shape)
  if x.dtype != jnp.bfloat16 or weight.dtype != jnp.bfloat16:
    raise ValueError(
        f"pallas_rmsnorm requires bf16 inputs, got {x.dtype}, {weight.dtype}"
    )
  if not float(epsilon) > 0.0:
    raise ValueError(f"pallas_rmsnorm epsilon must be positive, got {epsilon}")

  mp = model_contracts.round_up(m, BM)
  x_in = jnp.pad(x, ((0, mp - m), (0, 0)), constant_values=0) if mp != m else x
  block_m = row_tile(mp, f)

  def _kernel(x_ref, weight_ref, out_ref):
    out_ref[...] = canonical_rmsnorm(
        x_ref[...],
        weight_ref[...],
        epsilon=epsilon,
        out_dtype=out_ref.dtype,
    )

  out = pl.pallas_call(
      _kernel,
      out_shape=jax.ShapeDtypeStruct((mp, f), jnp.bfloat16),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((block_m, f), lambda i: (i, 0)),
              pl.BlockSpec((f,), lambda _i: (0,)),
          ],
          out_specs=pl.BlockSpec((block_m, f), lambda i: (i, 0)),
          grid=(mp // block_m,),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          allow_input_fusion=(
              (True, True)
              if pallas_matmul.input_fusion_enabled()
              else (False, False)
          ),
          shape_invariant_numerics=shape_invariant_numerics,
      ),
      interpret=interpret,
      name=f"canon_rmsnorm_bm{block_m}_bf{BF}_f{f}",
  )(x_in, weight)
  return out[:m] if mp != m else out
