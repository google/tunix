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
"""Fixed-feature-order bf16 RMSNorm Pallas kernel (Zero-TIM "P22.XH").

The row's sum of squares is accumulated in f32 over static BF=128 feature
blocks strictly left to right, so the reduction order depends only on F, never
on the number of rows, the row tile or the backend's reduction strategy.
``canonical_rmsnorm`` is the pure-JAX statement of the same semantic; it is
bit-exact with the Pallas kernel and is what the custom VJPs differentiate.
"""

from __future__ import annotations

from absl import logging
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tunix.experimental.zero_tim_kernel import pallas_matmul

# Contract tiles (p22xh_contract.py).  BF is the numerics; BM only decides how
# many independent rows share a grid step.
BM = 8
BF = 128

# The contract row block BM=8 ran the per-head norms
# (rows = tokens x heads, e.g. 8192 x 128 in the 512-token prefill program) as
# 1024 grid steps of fixed overhead, ~30% of that program's device time.  The
# feature-block reduction (BF, left-to-right f32) is the numerics; the row block
# only decides how many independent rows share a grid step, so wider row blocks
# are bitwise identical (bm 8/64/256/512 all == BM8 on 12 shapes;
# [8192,128] 242 -> 58 us, [8192,4096] 272 -> 84 us).  bm=512 exhausts scoped
# VMEM at F=4096, so the policy stops at 256 and at the probed feature widths.
ROW_TILE_POLICY = "v2"
ROW_TILE_WIDE = 256
ROW_TILE_MAX_FEATURES = 4096
_ROW_RECEIPTS: set[tuple[int, int]] = set()


def validate_shape(shape_x, shape_weight) -> tuple[int, int]:
  """Validates rank-2 x / rank-1 weight shapes against the BM/BF contract."""
  if len(shape_x) != 2:
    raise ValueError(f"P22.XH kernel expects rank-2 x, got {shape_x}")
  if len(shape_weight) != 1:
    raise ValueError(f"P22.XH kernel expects rank-1 weight, got {shape_weight}")
  m, f = map(int, shape_x)
  fw = int(shape_weight[0])
  if f != fw:
    raise ValueError(f"P22.XH feature/weight mismatch: {f} vs {fw}")
  if m % BM or f % BF:
    raise ValueError(f"P22.XH shape must divide BM/BF={BM}/{BF}, got {(m, f)}")
  return m, f


def row_tile(m: int, f: int) -> int:
  """Returns block_m for an [m, f] rmsnorm.

  256 when m divides by 256 and f is within the probed width, else the
  contract BM.

  Args:
    m: Number of rows.
    f: Feature width.
  """
  if m % ROW_TILE_WIDE == 0 and f <= ROW_TILE_MAX_FEATURES:
    return ROW_TILE_WIDE
  return BM


def _row_receipt(m: int, f: int, block_m: int) -> None:
  key = (m, f)
  if key in _ROW_RECEIPTS:
    return
  _ROW_RECEIPTS.add(key)
  logging.info(
      "[PATHTRACE] rmsnorm_row_tile=%s rows=%d F=%d bm=%d bf=%d",
      ROW_TILE_POLICY,
      m,
      f,
      block_m,
      BF,
  )


def canonical_rmsnorm(x, weight, *, epsilon: float):
  """Declared P22.XH semantic: fixed BF-block f32 accumulation, bf16 output."""
  m, f = validate_shape(x.shape, weight.shape)
  if x.dtype != jnp.bfloat16 or weight.dtype != jnp.bfloat16:
    raise ValueError(
        f"P22.XH requires bf16 inputs, got {x.dtype}, {weight.dtype}"
    )
  if not float(epsilon) > 0.0:
    raise ValueError(f"P22.XH epsilon must be positive, got {epsilon}")
  xf = x.astype(jnp.float32).reshape(m, f // BF, BF)
  wf = weight.astype(jnp.float32)

  # Python-static expansion preserves the registered left-to-right BF-block
  # order and avoids a dynamic_slice primitive, which Mosaic TPU does not
  # lower inside a Pallas kernel in this exact build.
  sumsq = jnp.zeros((m,), jnp.float32)
  for q in range(f // BF):
    block = xf[:, q, :]
    sumsq = sumsq + jnp.sum(block * block, axis=-1, dtype=jnp.float32)
  inv = jax.lax.rsqrt(sumsq / jnp.float32(f) + jnp.float32(epsilon))
  return (x.astype(jnp.float32) * inv[:, None] * wf[None, :]).astype(
      jnp.bfloat16
  )


def rmsnorm(
    x,
    weight,
    *,
    epsilon: float,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
):
  """Apply the declared P22.XH semantic to TP-local rank-2 bf16 arrays."""
  m, f = validate_shape(x.shape, weight.shape)
  if x.dtype != jnp.bfloat16 or weight.dtype != jnp.bfloat16:
    raise ValueError(
        f"P22.XH requires bf16 inputs, got {x.dtype}, {weight.dtype}"
    )
  if not float(epsilon) > 0.0:
    raise ValueError(f"P22.XH epsilon must be positive, got {epsilon}")
  x, weight = pallas_matmul.p66_vma_align_operands(x, weight)
  block_m = row_tile(m, f)
  _row_receipt(m, f, block_m)

  def _kernel(x_ref, weight_ref, out_ref):
    xb = x_ref[...].astype(jnp.float32).reshape(block_m, f // BF, BF)
    wb = weight_ref[...].astype(jnp.float32)

    sumsq = jnp.zeros((block_m,), dtype=jnp.float32)
    for q in range(f // BF):
      block = xb[:, q, :]
      sumsq = sumsq + jnp.sum(block * block, axis=-1, dtype=jnp.float32)
    inv = jax.lax.rsqrt(sumsq / jnp.float32(f) + jnp.float32(epsilon))
    out_ref[...] = (
        x_ref[...].astype(jnp.float32) * inv[:, None] * wb[None, :]
    ).astype(out_ref.dtype)

  return pl.pallas_call(
      _kernel,
      out_shape=jax.ShapeDtypeStruct(
          (m, f),
          jnp.bfloat16,
          manual_axis_type=pallas_matmul.p66_vma_output_manual_axis_type(
              x, weight
          ),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((block_m, f), lambda i: (i, 0)),
              pl.BlockSpec((f,), lambda _i: (0,)),
          ],
          out_specs=pl.BlockSpec((block_m, f), lambda i: (i, 0)),
          grid=(m // block_m,),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel",),
          # P56.4.7: producer fusion changes materialization, not
          # values (elementwise-exact producers; layout is not value).
          allow_input_fusion=(
              (True, True)
              if pallas_matmul.input_fusion_enabled()
              else (False, False)
          ),
          shape_invariant_numerics=shape_invariant_numerics,
      ),
      interpret=interpret,
      name=f"canon_rmsnorm_bm{block_m}_bf{BF}_f{f}",
  )(x, weight)
