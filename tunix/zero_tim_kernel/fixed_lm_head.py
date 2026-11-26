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
"""Fixed-contraction Pallas output head (`[M, K] @ [K, V_local]`).

The output head `[M, K] @ [K, V]` is evaluated at very different row counts `M`
by decode steps (`M <= 256`), prefill/scoring (`M == 256`), and the learner
(`M > 256`). `fixed_m_lm_head_local` routes every outer row count through the
same fixed-`BK = 256` Pallas contraction order:

* `M < 256` is zero-padded only to the smallest power-of-2 row bucket in
  `{8, 16, 32, 64, 128, 256}` (rows are independent and `block_m` is bitwise
  neutral, avoiding up to 32x decode MXU and HBM writeback waste) and sliced
  back;
* `M == 256` runs directly;
* `M > 256` is zero-padded to a multiple of 256 rows, reshaped into `c` chunks
  of 256 rows, and mapped with `lax.map` over the 256-row program. Its custom
  VJP accumulates the weight cotangent chunk by chunk with a loop-carried
  `lax.scan` in ascending chunk order.
"""

from __future__ import annotations

from collections.abc import Callable

from absl import logging
import jax
from jax import lax
import jax.numpy as jnp
from tunix.zero_tim_kernel import canonical_vjp
from tunix.zero_tim_kernel import model_contracts
from tunix.zero_tim_kernel import padded_matmul
from tunix.zero_tim_kernel import pallas_matmul

FIXED_M = 256
BM = 128
BN = model_contracts.LM_HEAD_TILE
BK = model_contracts.LM_HEAD_TILE


def canonical_local_matmul(
    contract: model_contracts.ModelContract | None = None,
    *,
    interpret: bool = False,
) -> Callable[..., jax.Array]:
  """Builds the canonical `local_matmul` (`canonical_vjp` over `padded_matmul`).

  Args:
    contract: Optional model contract providing default `(block_k, block_n)`.
    interpret: Run the Pallas kernel in interpret mode (CPU tests).

  Returns:
    `local_matmul(x, y, *, shape_invariant_numerics=True, **tiles)`.
  """

  def local_matmul(x, y, *, shape_invariant_numerics: bool = True, **tiles):
    def forward(a, b):
      return padded_matmul.matmul(
          a,
          b,
          contract=contract,
          interpret=interpret,
          shape_invariant_numerics=shape_invariant_numerics,
          **tiles,
      )

    return canonical_vjp.matmul(x, y, forward=forward, contract=contract)

  return local_matmul


def fixed_m_lm_head_local(
    x_bf16: jax.Array,
    w_bf16: jax.Array,
    *,
    local_matmul: Callable[..., jax.Array] | None = None,
    fixed_m: int = FIXED_M,
    block_m: int = BM,
    block_n: int = BN,
    block_k: int = BK,
    use_fast_plain_vjp: bool | None = None,
) -> jax.Array:
  """Runs the TP-local `fixed_m`-chunked or row-padded LM head.

  For `m < fixed_m`, rows are zero-padded to the smallest power-of-2 row bucket
  in `{8, 16, ..., fixed_m}` and sliced back after one `local_matmul` call at
  `block_m = min(block_m, bucket)` (preserving the exact `block_k` contraction
  order). For `m == fixed_m`, `local_matmul` runs directly. For `m > fixed_m`,
  rows are padded to a multiple of `fixed_m`, mapped over `fixed_m`-row chunks
  with `lax.map` in the forward pass, and differentiated with a custom VJP that
  accumulates `dW` across chunks in ascending order via `lax.scan`.

  Args:
    x_bf16: `[m, hidden]` bf16 local activations.
    w_bf16: `[hidden, vocab]` bf16 local LM-head weight shard.
    local_matmul: Callable executing a single `[fixed_m, hidden] @ [hidden,
      vocab]` chunk. Defaults to `canonical_local_matmul()`.
    fixed_m: Row chunk size (default `FIXED_M = 256`).
    block_m: Matmul M tile size.
    block_n: Matmul N tile size.
    block_k: Matmul K tile size.
    use_fast_plain_vjp: When True and `canonical_vjp.plain_matmul_vjp_enabled()`
      is active, calls `canonical_vjp.plain_matmul_pullback` directly in the
      chunked backward to avoid re-running the forward Pallas kernel inside
      `jax.vjp(run_fixed_chunk, ...)` on every chunk. Defaults to `True` when
      `local_matmul` is `None` (the standard `canonical_local_matmul`), and
      `False` when a custom `local_matmul` is supplied so custom pullbacks are
      never silently bypassed.

  Returns:
    `[m, vocab]` bf16 local logits.
  """
  if (
      x_bf16.ndim != 2
      or w_bf16.ndim != 2
      or int(x_bf16.shape[1]) != int(w_bf16.shape[0])
  ):
    raise ValueError(
        "fixed_m_lm_head_local requires 2D [M, K] @ [K, V], got "
        f"{x_bf16.shape} @ {w_bf16.shape}"
    )
  if x_bf16.dtype != jnp.bfloat16 or w_bf16.dtype != jnp.bfloat16:
    raise ValueError(
        "fixed_m_lm_head_local requires bf16 input/weight, got "
        f"{x_bf16.dtype}/{w_bf16.dtype}"
    )
  m, hidden = map(int, x_bf16.shape)
  _, vocab = map(int, w_bf16.shape)
  if m <= 0 or hidden <= 0 or vocab <= 0 or fixed_m <= 0:
    raise ValueError(
        "fixed_m_lm_head_local requires positive (M, K, V, fixed_m), got "
        f"{(m, hidden, vocab, fixed_m)}"
    )
  fast_plain_vjp = (
      local_matmul is None
      if use_fast_plain_vjp is None
      else bool(use_fast_plain_vjp)
  )
  if local_matmul is None:
    local_matmul = canonical_local_matmul()

  def run_chunk(
      a_chunk: jax.Array, weight: jax.Array, *, chunk_bm: int = block_m
  ) -> jax.Array:
    return local_matmul(
        a_chunk,
        weight,
        block_m=chunk_bm,
        block_n=block_n,
        block_k=block_k,
        shape_invariant_numerics=True,
    )

  if m < fixed_m:
    mp, bucket_bm = pallas_matmul.adaptive_row_bucket(m, fixed_m)
    a_in = (
        jnp.pad(x_bf16, ((0, mp - m), (0, 0)), constant_values=0)
        if mp != m
        else x_bf16
    )
    out = run_chunk(a_in, w_bf16, chunk_bm=min(block_m, bucket_bm))
    return out[:m, :] if mp != m else out
  if m == fixed_m:
    return run_chunk(x_bf16, w_bf16)

  def run_fixed_chunk(a_chunk: jax.Array, weight: jax.Array) -> jax.Array:
    return run_chunk(a_chunk, weight)

  mp = model_contracts.round_up(m, fixed_m)
  a_all = (
      jnp.pad(x_bf16, ((0, mp - m), (0, 0)), constant_values=0)
      if mp != m
      else x_bf16
  )

  def chunked_forward(a_learner: jax.Array, weight: jax.Array) -> jax.Array:
    a_chunks = a_learner.reshape((-1, fixed_m, hidden))
    return lax.map(
        lambda chunk: run_fixed_chunk(chunk, weight), a_chunks
    ).reshape((mp, vocab))

  @jax.custom_vjp
  def chunked_vjp(a_learner: jax.Array, weight: jax.Array) -> jax.Array:
    return chunked_forward(a_learner, weight)

  def chunked_fwd(a_learner: jax.Array, weight: jax.Array):
    return chunked_forward(a_learner, weight), (a_learner, weight)

  def chunked_bwd(residual, cotangent):
    a_learner, weight = residual
    a_chunks = a_learner.reshape((-1, fixed_m, hidden))
    cot_chunks = cotangent.reshape((-1, fixed_m, vocab))
    logging.info(
        "[PATHTRACE] CANON_FIXED_LM_HEAD_VJP=1 semantic_M=%d fixed_M=%d "
        "chunks=%d K=%d local_N=%d accumulation=lax.scan order=ascending",
        m,
        fixed_m,
        a_chunks.shape[0],
        hidden,
        vocab,
    )

    def accumulate(w_cot, values):
      a_chunk, out_cot = values
      # `jax.vjp(run_fixed_chunk, ...)` runs `run_fixed_chunk`'s forward pass
      # again just to capture `(a_chunk, weight)` as the `custom_vjp` residual
      # of `canonical_vjp.matmul`, which is a dead Pallas kernel launch per
      # chunk in the backward. When `use_fast_plain_vjp=True` and the plain
      # VJP is active (`CANON_MATMUL_VJP_PLAIN=1`, the default), call its
      # pullback directly on the chunks we already hold.
      if fast_plain_vjp and canonical_vjp.plain_matmul_vjp_enabled():
        a_cot, chunk_w_cot = canonical_vjp.plain_matmul_pullback(
            a_chunk, weight, out_cot
        )
      else:
        _, pullback = jax.vjp(run_fixed_chunk, a_chunk, weight)
        a_cot, chunk_w_cot = pullback(out_cot)
      # The loop-carried dependency is the backward contract: chunk q is
      # added only after chunks [0, q) have completed.
      return w_cot + chunk_w_cot, a_cot

    w_cot, a_cot_chunks = lax.scan(
        accumulate,
        jnp.zeros_like(weight),
        (a_chunks, cot_chunks),
    )
    return a_cot_chunks.reshape(a_learner.shape), w_cot

  chunked_vjp.defvjp(chunked_fwd, chunked_bwd)
  return chunked_vjp(a_all, w_bf16)[:m, :]
