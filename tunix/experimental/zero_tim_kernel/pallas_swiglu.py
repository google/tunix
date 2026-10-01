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
"""Fixed-tile bf16 SwiGLU Pallas kernel (Zero-TIM "P22.XG").

SwiGLU is elementwise, so its value never depends on M.  It is still a custom
call because XLA otherwise fuses ``silu(gate) * up`` differently into the
neighbouring matmuls in the decode, prefill and training programs (different
fusion => different intermediate rounding).  The Pallas call pins one
program: bf16 in, ``silu`` and the product in the kernel, one bf16 store.
"""

from __future__ import annotations

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tunix.experimental.zero_tim_kernel import pallas_matmul

# Contract tiles (p22xg_contract.py).
BM = 128
BF = 256


def validate_shape(shape_gate, shape_up) -> tuple[int, int]:
  """Validates equal rank-2 gate/up shapes against the BM/BF contract."""
  if tuple(shape_gate) != tuple(shape_up):
    raise ValueError(
        f"P22.XG gate/up shapes differ: {shape_gate} vs {shape_up}"
    )
  if len(shape_gate) != 2:
    raise ValueError(f"P22.XG expects rank-2 local arrays, got {shape_gate}")
  m, f = map(int, shape_gate)
  if m % BM or f % BF:
    raise ValueError(f"P22.XG shape must divide BM/BF={BM}/{BF}, got {(m, f)}")
  return m, f


def swiglu(
    gate, up, *, interpret: bool = False, shape_invariant_numerics: bool = True
):
  """Return `silu(gate) * up` for TP-local bf16 rank-2 arrays."""
  m, f = validate_shape(gate.shape, up.shape)
  if gate.dtype != jnp.bfloat16 or up.dtype != jnp.bfloat16:
    raise ValueError(
        f"P22.XG requires bf16 inputs, got {gate.dtype}, {up.dtype}"
    )
  gate, up = pallas_matmul.p66_vma_align_operands(gate, up)

  def _kernel(g_ref, u_ref, out_ref):
    g = g_ref[...]
    u = u_ref[...]
    out_ref[...] = (jax.nn.silu(g) * u).astype(out_ref.dtype)

  return pl.pallas_call(
      _kernel,
      out_shape=jax.ShapeDtypeStruct(
          (m, f),
          jnp.bfloat16,
          manual_axis_type=pallas_matmul.p66_vma_output_manual_axis_type(
              gate, up
          ),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((BM, BF), lambda i, j: (i, j)),
              pl.BlockSpec((BM, BF), lambda i, j: (i, j)),
          ],
          out_specs=pl.BlockSpec((BM, BF), lambda i, j: (i, j)),
          grid=(m // BM, f // BF),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel"),
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
      name="canon_swiglu_bm128_bf256",
  )(gate, up)
