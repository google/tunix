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
"""Fused `pallas_rmsnorm` prologue + `pallas_matmul` in one custom call.

Normalizes each `block_m` row block once into VMEM scratch via
`pallas_rmsnorm.canonical_rmsnorm` (Single Source of Truth) and then runs the
verbatim `pallas_matmul` tile loop (`bf16` tiles, `f32` accumulator, same
`block_k` order) against that scratch. Every per-element chain matches the
two-kernel chain bit for bit while keeping the normalized rows in VMEM.
"""

from __future__ import annotations

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tunix.zero_tim_kernel import pallas_matmul
from tunix.zero_tim_kernel import pallas_rmsnorm

BF = pallas_rmsnorm.BF


def norm_matmul(
    x,
    gamma,
    y,
    *,
    epsilon: float,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
    block_m: int = pallas_matmul.BM,
    block_n: int = pallas_matmul.BN,
    block_k: int = pallas_matmul.BK,
):
  """Computes `pallas_matmul.matmul(pallas_rmsnorm.rmsnorm(x, gamma), y)` as one custom call."""
  if x.ndim != 2 or y.ndim != 2 or gamma.ndim != 1:
    raise ValueError(
        "pallas_norm_matmul expects x[M,K] gamma[K] y[K,N], got "
        f"{x.shape}, {gamma.shape}, {y.shape}"
    )
  m, k = map(int, x.shape)
  ky, n = map(int, y.shape)
  if k != ky or int(gamma.shape[0]) != k:
    raise ValueError(
        f"pallas_norm_matmul contracted dims differ: x{x.shape} "
        f"gamma{gamma.shape} "
        f"y{y.shape}"
    )
  if (
      x.dtype != jnp.bfloat16
      or y.dtype != jnp.bfloat16
      or gamma.dtype != jnp.bfloat16
  ):
    raise ValueError(
        "pallas_norm_matmul requires bf16 inputs, got "
        f"{x.dtype}/{gamma.dtype}/{y.dtype}"
    )
  if not float(epsilon) > 0.0:
    raise ValueError(
        f"pallas_norm_matmul epsilon must be positive, got {epsilon} "
    )
  if m % block_m or n % block_n or k % block_k or k % BF:
    raise ValueError(
        "pallas_norm_matmul shape must divide BM/BN/BK/BF="
        f"{block_m}/{block_n}/{block_k}/{BF}, got {(m, k, n)}"
    )

  def _kernel(x_ref, gamma_ref, y_ref, out_ref, acc_ref, normed_ref):
    q = pl.program_id(1)

    @pl.when(q == 0)
    def _norm():
      normed_ref[...] = pallas_rmsnorm.canonical_rmsnorm(
          x_ref[...],
          gamma_ref[...],
          epsilon=epsilon,
          out_dtype=jnp.bfloat16,
      )

    @pl.when(q == 0)
    def _init():
      acc_ref[...] = jnp.zeros_like(acc_ref)

    xk = normed_ref[:, pl.ds(q * block_k, block_k)]
    for j in range(n // block_n):
      lo = j * block_n
      acc_ref[:, lo : lo + block_n] = acc_ref[:, lo : lo + block_n] + jnp.dot(
          xk,
          y_ref[:, lo : lo + block_n],
          precision=jax.lax.Precision.DEFAULT,
          preferred_element_type=jnp.float32,
      )

    @pl.when(q == pl.num_programs(1) - 1)
    def _store():
      out_ref[...] = acc_ref[...].astype(out_ref.dtype)

  return pl.pallas_call(
      _kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), jnp.bfloat16),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          in_specs=[
              pl.BlockSpec((block_m, k), lambda i, _q: (i, 0)),
              pl.BlockSpec((k,), lambda _i, _q: (0,)),
              pl.BlockSpec((block_k, n), lambda _i, q: (q, 0)),
          ],
          out_specs=pl.BlockSpec((block_m, n), lambda i, _q: (i, 0)),
          grid=(m // block_m, k // block_k),
          scratch_shapes=[
              pltpu.VMEM((block_m, n), jnp.float32),
              pltpu.VMEM((block_m, k), jnp.bfloat16),
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          # `q` (`k // block_k`) must be `"arbitrary"` (sequential): `q == 0`
          # populates `normed_ref` and zeros `acc_ref` read by all `q > 0`.
          dimension_semantics=("parallel", "arbitrary"),
          allow_input_fusion=(False, False, False),
          shape_invariant_numerics=shape_invariant_numerics,
      ),
      interpret=interpret,
      name=f"canon_norm_matmul_bm{block_m}_bn{block_n}_bk{block_k}_f{k}",
  )(x, gamma, y)
