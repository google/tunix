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
"""Canonical custom-VJPs whose primals remain the promoted Pallas forwards.

Pallas TPU kernels have no JVP rule, and the training forward (C) must be the
very same program as the engine's scoring forward (B) for their log-probs to
be bitwise equal. Every coat below therefore uses the Pallas kernel verbatim
as the primal and differentiates a pure-JAX replica in the backward:

* `canonical_matmul`: a fixed-K-block (contract BK) replica of the forward's
  accumulation order; optionally replaced by one plain dot's transpose
  (`CANON_MATMUL_VJP_PLAIN`, default on) because the backward only has to be
  a correct gradient, not a bitwise one;
* `canonical_rmsnorm`: the declared bit-exact semantic of the Pallas norm;
* `canonical_swiglu`: `silu(gate) * up` in bf16;
* `fused_matmul` / `fused_norm_matmul`: horizontally fused projection coats
  that keep the un-concatenated parameter views `(w_parts)` as the VJP residual
  and re-materialize the concatenation inside the backward behind an
  `optimization_barrier` to avoid keeping a duplicate copy of the layer weights
  live across the forward-to-backward span.
"""

from __future__ import annotations

from collections.abc import Callable
import os

from absl import logging
import jax
from jax import lax
import jax.numpy as jnp
from tunix.zero_tim_kernel import model_contracts
from tunix.zero_tim_kernel import padded_matmul
from tunix.zero_tim_kernel import pallas_matmul
from tunix.zero_tim_kernel import pallas_norm_matmul
from tunix.zero_tim_kernel import pallas_rmsnorm
from tunix.zero_tim_kernel import pallas_swiglu


def canonical_matmul(x, y, *, contract: model_contracts.ModelContract):
  """Pure-JAX replica of the model-pinned padded-BK accumulation order."""
  if x.ndim != 2 or y.ndim != 2 or int(x.shape[1]) != int(y.shape[0]):
    raise ValueError(
        f"canonical_vjp matmul shapes invalid: {x.shape}, {y.shape} "
    )
  if x.dtype != jnp.bfloat16 or y.dtype != jnp.bfloat16:
    raise ValueError(
        f"canonical_vjp matmul requires bf16, got {x.dtype}, {y.dtype} "
    )
  bk, bn = contract.block_k, contract.block_n
  m, k, n = int(x.shape[0]), int(x.shape[1]), int(y.shape[1])
  kp, _ = padded_matmul.padded_matmul_extents(
      k, n, contract=contract, block_k=bk, block_n=bn
  )
  if kp != k:
    x = jnp.pad(x, ((0, 0), (0, kp - k)), constant_values=0)
    y = jnp.pad(y, ((0, kp - k), (0, 0)), constant_values=0)
  acc = jnp.zeros((m, n), jnp.float32)
  for q in range(kp // bk):
    lo, hi = q * bk, (q + 1) * bk
    acc = acc + jnp.dot(
        x[:, lo:hi], y[lo:hi, :], preferred_element_type=jnp.float32
    )
  return acc.astype(jnp.bfloat16)


canonical_swiglu = pallas_swiglu.canonical_swiglu


PLAIN_VJP_ENV = "CANON_MATMUL_VJP_PLAIN"
_PLAIN_RECEIPT = set()


def plain_matmul_vjp_enabled():
  """Whether projection pullbacks are the transposes of one plain bf16 dot.

  Default on. `CANON_MATMUL_VJP_PLAIN=0` keeps the K-block replica pullback as
  an explicit A/B path.

  Returns:
    True unless `CANON_MATMUL_VJP_PLAIN=0` is set.
  """
  value = os.environ.get(PLAIN_VJP_ENV, "1")
  if value not in ("0", "1"):
    raise ValueError(f"{PLAIN_VJP_ENV} must be unset, 0 or 1, got {value!r}")
  if value not in _PLAIN_RECEIPT:
    _PLAIN_RECEIPT.add(value)
    logging.info(
        "[PATHTRACE] %s=%s projection VJPs use %s",
        PLAIN_VJP_ENV,
        value,
        "plain bf16 dots (dX = cot.W^T, dW = X^T.cot; default)"
        if value == "1"
        else "the K-block replica pullback (explicit A/B path)",
    )
  return value == "1"


def plain_matmul(x, y):
  """One plain bf16 dot (f32 MXU accumulation, rounded once to bf16).

  Only its transpose is ever executed: the pullback is JAX's own transpose of
  this dot, two plain bf16 dots.

  Args:
    x: Left operand `[M, K]` (bf16).
    y: Right operand `[K, N]` (bf16).

  Returns:
    `x @ y` as bf16.
  """
  return jnp.dot(x, y, preferred_element_type=jnp.bfloat16)


def plain_matmul_pullback(a, b, cotangent):
  """dX, dW of X @ W through JAX's transpose of one plain dot.

  Going through `jax.vjp` instead of writing the two dots by hand keeps the
  manual-axis typing of the canonical path: under `check_vma=True` the
  replicated projection input is implicitly cast to varying over TP inside
  the dot, and JAX transposes that cast into the TP `psum`, so `dX` comes back
  invariant over model like the primal input.

  Args:
    a: The primal left operand.
    b: The primal right operand.
    cotangent: The output cotangent.

  Returns:
    The `(da, db)` cotangents.
  """
  _, pullback = jax.vjp(plain_matmul, a, b)
  return pullback(cotangent)


def _require_contract(contract, what: str) -> model_contracts.ModelContract:
  if contract is None:
    raise ValueError(
        f"{what}: {PLAIN_VJP_ENV}=0 selects the K-block replica pullback, "
        "which needs the model contract (pass contract=...)"
    )
  return contract


def matmul(
    x,
    y,
    *,
    forward,
    contract: model_contracts.ModelContract | None = None,
):
  """Use `forward` verbatim for primal and `canonical_matmul` only for VJP."""

  @jax.custom_vjp
  def op(a, b):
    return forward(a, b)

  def fwd(a, b):
    return forward(a, b), (a, b)

  def bwd(residual, cotangent):
    a, b = residual
    if plain_matmul_vjp_enabled():
      return plain_matmul_pullback(a, b, cotangent)
    replica_contract = _require_contract(contract, "canonical_vjp matmul")
    _, pullback = jax.vjp(
        lambda a_, b_: canonical_matmul(a_, b_, contract=replica_contract),
        a,
        b,
    )
    return pullback(cotangent)

  op.defvjp(fwd, bwd)
  return op(x, y)


def swiglu(
    gate,
    up,
    *,
    forward: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
):
  """Use Pallas SwiGLU for primal and `canonical_swiglu` for VJP."""
  if forward is None:
    forward = pallas_swiglu.swiglu

  @jax.custom_vjp
  def op(g, u):
    return forward(g, u)

  def fwd(g, u):
    return forward(g, u), (g, u)

  def bwd(residual, cotangent):
    g, u = residual
    _, pullback = jax.vjp(canonical_swiglu, g, u)
    return pullback(cotangent)

  op.defvjp(fwd, bwd)
  return op(gate, up)


def norm_matmul(
    x,
    gamma,
    y,
    *,
    epsilon: float,
    forward,
    contract: model_contracts.ModelContract | None = None,
):
  """`pallas_norm_matmul` coat: fused Pallas primal, composed replica VJP."""

  def oracle(a, g, b):
    return canonical_matmul(
        pallas_rmsnorm.canonical_rmsnorm(a, g, epsilon=epsilon),
        b,
        contract=_require_contract(contract, "pallas_norm_matmul"),
    )

  @jax.custom_vjp
  def op(a, g, b):
    return forward(a, g, b)

  def fwd(a, g, b):
    return forward(a, g, b), (a, g, b)

  def bwd(residual, cotangent):
    a, g, b = residual
    if plain_matmul_vjp_enabled():
      h, norm_pullback = jax.vjp(
          lambda a_, g_: pallas_rmsnorm.canonical_rmsnorm(
              a_, g_, epsilon=epsilon
          ),
          a,
          g,
      )
      dh, db = plain_matmul_pullback(h, b, cotangent)
      da, dg = norm_pullback(dh)
      return da, dg, db
    _, pullback = jax.vjp(oracle, a, g, b)
    return pullback(cotangent)

  op.defvjp(fwd, bwd)
  return op(x, gamma, y)


def rmsnorm(x, weight, *, epsilon: float, forward):
  """Use promoted Pallas RMSNorm for primal and fixed-BF replica for VJP."""

  def oracle(a, w):
    return pallas_rmsnorm.canonical_rmsnorm(a, w, epsilon=epsilon)

  @jax.custom_vjp
  def op(a, w):
    return forward(a, w)

  def fwd(a, w):
    return forward(a, w), (a, w)

  def bwd(residual, cotangent):
    a, w = residual
    _, pullback = jax.vjp(oracle, a, w)
    return pullback(cotangent)

  op.defvjp(fwd, bwd)
  return op(x, weight)


def _validate_fused_operands(
    x_2d: jax.Array,
    w_parts: tuple[jax.Array, ...],
    *,
    what: str,
) -> tuple[int, int, list[int], int]:
  """Validates `x_2d` `[M, K]` and `w_parts` `([K, N_i], ...)` up front."""
  if x_2d.ndim != 2:
    raise ValueError(f"{what} expects rank-2 input x_2d, got {x_2d.shape}")
  if not w_parts:
    raise ValueError(f"{what} requires at least one weight part in w_parts")
  m, k = map(int, x_2d.shape)
  if m <= 0 or k <= 0:
    raise ValueError(f"{what} requires positive (M, K), got {(m, k)}")
  split_points: list[int] = []
  offset = 0
  for idx, w in enumerate(w_parts):
    if w.ndim != 2 or int(w.shape[0]) != k or int(w.shape[1]) <= 0:
      raise ValueError(
          f"{what} w_parts[{idx}] must have shape [K={k}, N_i>0], got {w.shape}"
      )
    offset += int(w.shape[1])
    if idx < len(w_parts) - 1:
      split_points.append(offset)
  return m, k, split_points, offset


def fused_matmul(
    x_2d: jax.Array,
    w_parts: tuple[jax.Array, ...],
    *,
    contract: model_contracts.ModelContract,
    interpret: bool = False,
    out_dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> jax.Array:
  """Horizontally fused `x @ concat(w_parts, axis=1)` with a copy-free residual.

  Keeps the original parameter views `w_parts` as the `custom_vjp` residual and
  re-materializes the concatenation only inside the backward behind an
  `optimization_barrier` tied to the cotangent so XLA CSE cannot keep the
  concatenated weight buffer live across the forward-to-backward span.

  Args:
    x_2d: `[M, K]` input rows.
    w_parts: Weights `[K, N_i]` sharing the contracting dim, concatenated along
      the output axis.
    contract: Model contract (tile policy) of the Pallas matmul.
    interpret: Pallas interpret mode.
    out_dtype: Output dtype.

  Returns:
    `[M, sum(N_i)]` matmul output.
  """
  _, _, split_points, _ = _validate_fused_operands(
      x_2d, w_parts, what="fused_matmul"
  )
  x_bf16 = x_2d.astype(jnp.bfloat16)
  parts = tuple(w.astype(jnp.bfloat16) for w in w_parts)

  def forward(a: jax.Array, b: jax.Array) -> jax.Array:
    return padded_matmul.matmul(
        a,
        b,
        contract=contract,
        interpret=interpret,
        shape_invariant_numerics=True,
        block_m=contract.block_m,
        block_n=contract.block_n,
        block_k=contract.block_k,
    )

  @jax.custom_vjp
  def op(a: jax.Array, ws: tuple[jax.Array, ...]) -> jax.Array:
    return forward(a, jnp.concatenate(ws, axis=1))

  def op_fwd(a: jax.Array, ws: tuple[jax.Array, ...]):
    return forward(a, jnp.concatenate(ws, axis=1)), (a, ws)

  def op_bwd(residual, cotangent):
    a, ws = residual
    ws_late, cotangent = lax.optimization_barrier((ws, cotangent))
    w_fused = jnp.concatenate(ws_late, axis=1)
    if plain_matmul_vjp_enabled():
      da, dw = plain_matmul_pullback(a, w_fused, cotangent)
    else:
      _, pullback = jax.vjp(
          lambda a_, b_: canonical_matmul(a_, b_, contract=contract),
          a,
          w_fused,
      )
      da, dw = pullback(cotangent)
    dws = tuple(jnp.split(dw, split_points, axis=1))
    return da, dws

  op.defvjp(op_fwd, op_bwd)
  return op(x_bf16, parts).astype(out_dtype)


def norm_matmul_admits(
    d_in: int, contract: model_contracts.ModelContract
) -> bool:
  """Whether `pallas_norm_matmul` can take `[*, d_in]` rows unpadded."""
  return d_in % contract.block_k == 0 and d_in % pallas_rmsnorm.BF == 0


def fused_norm_matmul(
    x_2d: jax.Array,
    gamma: jax.Array,
    w_parts: tuple[jax.Array, ...],
    *,
    norm_eps: float,
    contract: model_contracts.ModelContract,
    interpret: bool = False,
    out_dtype: jax.typing.DTypeLike = jnp.bfloat16,
) -> jax.Array:
  """Fused `rmsnorm(x, gamma) @ concat(w_parts)` with a copy-free residual.

  Stores only `(x, gamma, w_parts)` as the `custom_vjp` residual (neither the
  normalized activations nor the concatenated weight copy is kept alive across
  the forward-to-backward span) and re-derives `h = canonical_rmsnorm(x, gamma)`
  and `w_fused` (behind an `optimization_barrier`) inside the backward.

  Args:
    x_2d: `[M, K]` un-normalized rows (residual stream).
    gamma: `[K]` RMSNorm weight.
    w_parts: Weights `[K, N_i]` concatenated along the output axis.
    norm_eps: RMSNorm epsilon.
    contract: Model contract (tile policy) of the Pallas matmul.
    interpret: Pallas interpret mode.
    out_dtype: Output dtype.

  Returns:
    `[M, sum(N_i)]` projection of the normalized rows.
  """
  m, k, split_points, n_total = _validate_fused_operands(
      x_2d, w_parts, what="fused_norm_matmul"
  )
  if gamma.ndim != 1 or int(gamma.shape[0]) != k:
    raise ValueError(
        f"fused_norm_matmul expects gamma of shape [{k}], got {gamma.shape}"
    )
  if not norm_matmul_admits(k, contract):
    raise ValueError(
        f"fused norm_matmul needs K={k} to divide block_k={contract.block_k}"
        f" and BF={pallas_rmsnorm.BF}"
    )
  x_bf16 = x_2d.astype(jnp.bfloat16)
  gamma_bf16 = gamma.astype(jnp.bfloat16)
  parts = tuple(w.astype(jnp.bfloat16) for w in w_parts)

  # Keep `block_m = contract.block_m` (128) so `pallas_norm_matmul`'s full-width
  # VMEM scratch `[block_m, n_padded]` stays within TPU VMEM, while promoting
  # `(n_padded, block_n)` via `wide_n_tiles` / `tile_policy` so the inner MXU
  # column loop uses wide `block_n` tiles (matching `padded_matmul.matmul`).
  block_m = contract.block_m
  block_n = contract.block_n
  block_k = contract.block_k
  m_padded = model_contracts.round_up(m, block_m)
  n_padded = model_contracts.round_up(n_total, block_n)
  wide = padded_matmul.wide_n_tiles(
      m_padded, k, n_total, (block_m, block_n, block_k)
  )
  if wide is not None:
    n_padded, block_n = wide
  elif (block_m, block_n, block_k) in pallas_matmul.CONTRACT_TILES:
    _, block_n, _ = pallas_matmul.tile_policy(
        m_padded, k, n_padded, (block_m, block_n, block_k)
    )

  def forward(a: jax.Array, g: jax.Array, b: jax.Array) -> jax.Array:
    if m_padded != m:
      a = jnp.pad(a, ((0, m_padded - m), (0, 0)), constant_values=0)
    if n_padded != n_total:
      b = jnp.pad(b, ((0, 0), (0, n_padded - n_total)), constant_values=0)
    out = pallas_norm_matmul.norm_matmul(
        a,
        g,
        b,
        epsilon=norm_eps,
        interpret=interpret,
        shape_invariant_numerics=True,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )
    return out[:m, :n_total]

  @jax.custom_vjp
  def op(a: jax.Array, g: jax.Array, ws: tuple[jax.Array, ...]) -> jax.Array:
    return forward(a, g, jnp.concatenate(ws, axis=1))

  def op_fwd(a: jax.Array, g: jax.Array, ws: tuple[jax.Array, ...]):
    return forward(a, g, jnp.concatenate(ws, axis=1)), (a, g, ws)

  def op_bwd(residual, cotangent):
    a, g, ws = residual
    ws_late, cotangent = lax.optimization_barrier((ws, cotangent))
    w_fused = jnp.concatenate(ws_late, axis=1)
    h, norm_pullback = jax.vjp(
        lambda a_, g_: pallas_rmsnorm.canonical_rmsnorm(
            a_, g_, epsilon=norm_eps
        ),
        a,
        g,
    )
    if plain_matmul_vjp_enabled():
      dh, dw = plain_matmul_pullback(h, w_fused, cotangent)
    else:
      _, pullback = jax.vjp(
          lambda h_, b_: canonical_matmul(h_, b_, contract=contract),
          h,
          w_fused,
      )
      dh, dw = pullback(cotangent)
    da, dg = norm_pullback(dh)
    dws = tuple(jnp.split(dw, split_points, axis=1))
    return da, dg, dws

  op.defvjp(op_fwd, op_bwd)
  return op(x_bf16, gamma_bf16, parts).astype(out_dtype)
