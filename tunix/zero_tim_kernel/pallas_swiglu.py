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
"""Fixed-tile, batch-invariant Pallas TPU SwiGLU (`silu(gate) * up`).

Why this kernel exists: an elementwise XLA `silu(gate) * up` can be fused into
surrounding consumers in a shape-dependent way (so decode `M=1..8` and prefill
`M=bucket` execute different instruction sequences on the same row). A Pallas
kernel is a custom-call boundary with `allow_input_fusion = (False, False)`,
giving every `(row, feature)` tile the identical `f32` `silu(gate) * up` body
and single `bf16` store regardless of `M`.

Non-multiple `(M, F)` extents are automatically zero-padded to adaptive row
buckets (`{8, 16, 32, 64}` for `M < 128`, or multiples of `BM = 128` otherwise)
and `BF = 256` column multiples before the Pallas launch and sliced back to
`[:M, :F]`.
"""

from __future__ import annotations

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tunix.zero_tim_kernel import model_contracts
from tunix.zero_tim_kernel import pallas_matmul

BM = 128
BF = 256


def canonical_swiglu(
    gate: jax.Array,
    up: jax.Array,
    *,
    out_dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> jax.Array:
  """Pure-JAX replica of the `pallas_swiglu` arithmetic (`f32` `silu(gate)*up`)."""
  if gate.ndim != 2 or up.ndim != 2:
    raise ValueError(
        f"canonical_swiglu expects rank-2 inputs, got {gate.shape}, {up.shape} "
    )
  if gate.shape != up.shape:
    raise ValueError(
        f"canonical_swiglu shape mismatch: {gate.shape} vs {up.shape}"
    )
  if gate.dtype != jnp.bfloat16 or up.dtype != jnp.bfloat16:
    raise ValueError(
        f"canonical_swiglu requires bf16 inputs, got {gate.dtype}, {up.dtype}"
    )
  m, f = map(int, gate.shape)
  if m <= 0 or f <= 0:
    raise ValueError(f"canonical_swiglu requires positive (M, F), got {(m, f)}")
  g = gate.astype(jnp.float32)
  u = up.astype(jnp.float32)
  return (jax.nn.silu(g) * u).astype(out_dtype)


def swiglu(
    gate: jax.Array,
    up: jax.Array,
    *,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
) -> jax.Array:
  """Computes `bf16 silu(gate) * up` in `f32` on fixed `(BM=128, BF=256)` tiles."""
  if gate.ndim != 2 or up.ndim != 2:
    raise ValueError(
        f"pallas_swiglu expects rank-2 inputs, got {gate.shape}, {up.shape} "
    )
  if gate.shape != up.shape:
    raise ValueError(
        f"pallas_swiglu shape mismatch: {gate.shape} vs {up.shape}"
    )
  m, f = map(int, gate.shape)
  if m <= 0 or f <= 0:
    raise ValueError(f"pallas_swiglu requires positive (M, F), got {(m, f)}")
  if gate.dtype != jnp.bfloat16 or up.dtype != jnp.bfloat16:
    raise ValueError(
        f"pallas_swiglu requires bf16 inputs, got {gate.dtype}, {up.dtype} "
    )

  mp, bm = pallas_matmul.adaptive_row_bucket(m, BM)
  fp = model_contracts.round_up(f, BF)
  if mp != m or fp != f:
    pad = ((0, mp - m), (0, fp - f))
    gate_in = jnp.pad(gate, pad, constant_values=0)
    up_in = jnp.pad(up, pad, constant_values=0)
  else:
    gate_in = gate
    up_in = up

  def _kernel(gate_ref, up_ref, out_ref):
    out_ref[...] = canonical_swiglu(
        gate_ref[...], up_ref[...], out_dtype=out_ref.dtype
    )

  out = pl.pallas_call(
      _kernel,
      out_shape=jax.ShapeDtypeStruct((mp, fp), jnp.bfloat16),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((bm, BF), lambda i, j: (i, j)),
              pl.BlockSpec((bm, BF), lambda i, j: (i, j)),
          ],
          out_specs=pl.BlockSpec((bm, BF), lambda i, j: (i, j)),
          grid=(mp // bm, fp // BF),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel"),
          allow_input_fusion=(
              (True, True)
              if pallas_matmul.input_fusion_enabled()
              else (False, False)
          ),
          shape_invariant_numerics=shape_invariant_numerics,
      ),
      interpret=interpret,
      name=f"canon_swiglu_bm{bm}_bf{BF}",
  )(gate_in, up_in)
  return out[:m, :f] if (mp != m or fp != f) else out
