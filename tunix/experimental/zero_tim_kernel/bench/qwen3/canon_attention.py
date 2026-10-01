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

"""Zero-TIM canonical attention core: Pallas forward kernel, flash backward.

The Zero-TIM attention contract is that one query row attending to one set of
keys produces the same bits in every program that evaluates it: KV-cached
decode (`T=1`), cacheless prefill (`T=L`), the learner forward of the same
rows, any batch size. Two forwards exist, selected by `ATTENTION_FWD`:

* `PALLAS_FWD` (A4, `flash_fwd_pallas`, the default and the contract): one TPU
  kernel that processes every query row as part of a fixed
  `(FWD_QUERY_BLOCK=128 rows) x (FWD_KV_SUBBLOCK=128 keys)` tile program --
  the same MXU matmuls, the same lane reductions and the same elementwise
  exp2 online-softmax update, in the same order, whatever `T`, `S` or `B` are
  (`T=1` is padded to one 128-row block; keys are visited in 128-key
  sub-blocks of the fixed `kv_block_size=512` DMA block, in key order).
  Fully masked sub-blocks (the causal upper triangle, the key padding and the
  inactive cache tail) are skipped through a scalar-prefetched block map, and
  a row without a valid key in a sub-block carries its `(m, l, acc)` state
  through unchanged, so the result does not depend on the cache length or on
  how many blocks follow the active ones either.
* `XLA_FWD` (`online_softmax_scan`): the fixed KV-block (`bkv_csz=512`)
  exp2-domain online-softmax `lax.scan`, the contract before A4 and the
  primal of `AUTODIFF`. It cannot hold the contract across query lengths: XLA
  compiles a different program for every padded query length `Tp` (the `qk`
  / `pv` dot tilings, the `l = sum(p)` reduction and the final divide are all
  free to associate differently), and on TPU (4x v5p, TP=4) the `T=1` decode
  program (`Tp=64`) and the `T=256` prefill program disagree at the f32 ULP
  level in about 1% of the rows, which the bf16 output cast turns into 1-ULP
  logit differences that persist through the KV cache; the two agree only
  when the prefill length happens to pad to the decode pad (`T <= 64`).

The two forwards are not bitwise interchangeable (different association
orders); switching is a change of contract, not an optimization A/B.

A third `ATTENTION_FWD` value, `RPA_FWD`, is a measurement setting rather
than a forward of this module: the model glue (`qwen3_model_canon` ->
`rpa_backend`) then runs the bench programs on the engine's ragged paged
attention kernel under the canonical call-site controls, so that the
cross-engine question can be measured on one host. `attention_core` rejects
it.

Backward (B4, `attention_core`): a `jax.custom_vjp` around the forward whose
gradient is the flash-attention backward instead of JAX's transpose of the
checkpointed scan. That transpose differentiates through the running max
(`jnp.max` -> a `[B, KH, G, Tp, bkv]` select against the arg-max), the
`alpha` rescaling chain and the `(m, l, acc)` carry, so it materialises
several `[B, KH, G, Tp, bkv]` f32 temporaries per layer and recomputes the
forward block arithmetic once more on top. The flash backward treats the
forward's row statistics `lse2 = m + log2(l)` as constants and needs five MXU
matmuls per `(key block, query block)` pair:

    p  = exp2(c.q.k^T - lse2)          c = scale * log2(e)
    dv = p^T . do                      dp = do . v^T
    ds = p * (dp - rowsum(do * o))
    dq = scale . ds . k                dk = scale . ds^T . q

Two implementations share this math and are selected by `ATTENTION_BWD`:

* `XLA_BWD`: dense `jnp` (CPU / reference implementation).
* `PALLAS_BWD`: one TPU kernel in the transposed `(keys, queries)` orientation
  (row statistics are `(1, bq)` lane vectors broadcast along sublanes, the
  layout `jax.experimental.pallas.ops.tpu.splash_attention` uses) that keeps
  `p / dp / ds` in VMEM and skips fully masked 128-key sub-blocks (the causal
  upper triangle and the `sp=512` key padding) through a scalar-prefetched
  block map. `dk / dv` accumulate in f32 VMEM scratch across the query blocks
  of a key block; `dq` is written per key block and reduced outside (one key
  block for `S <= 512`, so no extra HBM in the bench).
* `AUTODIFF`: no `custom_vjp` at all -- the Step 7 behaviour, kept for A/B.
  It differentiates the XLA scan (a Pallas kernel has no JVP), so it requires
  `fwd=XLA_FWD`.

Gradients carry no bitwise contract; the gates are
`qwen3_model_canon_test.test_param_grads_match_stock_tp4` (whole model, TP=4)
and `canon_attention_test` (kernel vs dense reference vs autodiff).
"""

from __future__ import annotations

import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LOG2_E = 1.4426950408889634

# Fixed KV-block granularity (`bkv_csz`) of `canonical_online_attention`.
#
# Zero-TIM only requires the value to be FIXED: decode, prefill and the learner
# forward all chunk the key axis at the same multiples of `bkv` and inactive
# blocks are strict no-ops, so the result is independent of the cache length
# and of how many blocks follow the active ones. The value itself is therefore
# purely a latency / HBM knob and mirrors the pinned RPA v3 bundle
# `rpa_canonical.CANONICAL_BLOCK_SIZES = (128, 512, 128, 512)` (`bkv_csz=512`)
# rather than the historical un-pinned vLLM default (`16` / `32`).
#
# Latency & HBM Optimization: with `bkv=16` an `S=256` prefill ran 16
# sequential `lax.scan` steps per layer and every decode step over a 128-slot
# cache ran 8, each paying a fixed per-step launch overhead (dominant in
# launch-bound `B=4` decode) and, in Fwd+Bwd, saving its fp32 `(m, l, acc)`
# carry (`[B, KH, G, Tp, D]`) to HBM for the backward pass. `bkv=512` turns
# every `S <= 512` attention into a single scan step while keeping the exact
# same per-block arithmetic (`exp2` online softmax, f32 accumulation).
DEFAULT_ATTENTION_KV_BLOCK_SIZE = 512

# Query-length padding of the XLA scan (see `online_softmax_scan`).
FORWARD_QUERY_PAD = 64

# Forward implementations (see module docstring).
XLA_FWD = 'xla'
PALLAS_FWD = 'pallas'
ATTENTION_FWD_MODES = (XLA_FWD, PALLAS_FWD)

# The engine's ragged-paged-attention kernel (tpu_inference RPA v3) under the
# canonical call-site controls, dispatched by the model glue
# (`qwen3_model_canon.CanonAttention` -> `rpa_backend`) instead of
# `attention_core`, which has no RPA forward. A measurement setting for the
# cross-engine question (bench programs on the engine kernel), not a contract
# candidate of this module; `attention_core` rejects it.
RPA_FWD = 'rpa'
ATTENTION_FWD_CHOICES = ATTENTION_FWD_MODES + (RPA_FWD,)

# Candidate A4: the Pallas forward kernel is the Zero-TIM attention contract.
ATTENTION_FWD = PALLAS_FWD

# Forward kernel geometry. FIXED on purpose: every query row of every program
# is processed as part of a `FWD_QUERY_BLOCK`-row block (`T=1` decode pads to
# one block) against `FWD_KV_SUBBLOCK`-key sub-blocks of the `kv_block_size`
# key block, so the per-row arithmetic is the same whatever `T`, `S` and `B`
# are. Row statistics are kept as `(bq, NUM_LANES)` lane-replicated tiles.
FWD_QUERY_BLOCK = 128
FWD_KV_SUBBLOCK = 128
NUM_LANES = 128

# Backward implementations (see module docstring).
AUTODIFF = 'autodiff'
XLA_BWD = 'xla'
PALLAS_BWD = 'pallas'
ATTENTION_BWD_MODES = (AUTODIFF, XLA_BWD, PALLAS_BWD)

# Candidate B4: flash-attention backward for `canonical_online_attention`.
ATTENTION_BWD = PALLAS_BWD

# Backward kernel geometry. Queries are padded to a multiple of
# `BWD_QUERY_PAD` so the `(1, bq)` row-statistics blocks are lane-aligned
# (`bq in {128, 256}`); keys keep the forward's `kv_block_size` block and are
# visited in `BWD_KV_SUBBLOCK`-key sub-blocks that can be skipped when fully
# masked. (The backward may pick its query block per shape: gradients carry
# no bitwise contract.)
BWD_QUERY_PAD = 128
BWD_KV_SUBBLOCK = 128
NUM_SUBLANES = 8

_NT_DIMS = (((1,), (1,)), ((), ()))


def round_up(value: int, multiple: int) -> int:
  return -(-value // multiple) * multiple


def _pad_rows(x: jax.Array, axis: int, size: int) -> jax.Array:
  """Zero-pads `x` along `axis` up to `size` rows (no-op if already there)."""
  cur = x.shape[axis]
  if cur == size:
    return x
  widths = [(0, 0)] * x.ndim
  widths[axis] = (0, size - cur)
  return jnp.pad(x, widths, constant_values=0)


# bf16 has an 8-bit exponent and 7 explicit mantissa bits.
_BF16_EXPONENT_BITS = 8
_BF16_MANTISSA_BITS = 7


def hard_round(x: jax.Array, dtype: jax.typing.DTypeLike) -> jax.Array:
  """`x.astype(dtype)` through a rounding point XLA cannot simplify away.

  An `astype(bf16)` lowers to an XLA `convert`. Under the default
  `--xla_allow_excess_precision=true` a `convert` to bf16 whose consumer
  upcasts to f32 again may be simplified away (and bf16 intermediates may be
  kept in f32 inside a fusion), so where the rounding happens depends on the
  fusion decisions of each program. `lax.reduce_precision(x, 8, 7)` is the
  same round-to-nearest-even to bf16 precision expressed as an arithmetic op
  in its own right (the algebraic simplifier only removes it when it is a
  no-op for the operand type), so the rounding point stays where it is
  written under either flag setting. Dtypes other than bf16 are a plain
  `astype`.

  Args:
    x: A floating-point array.
    dtype: The target dtype.

  Returns:
    `x` rounded to `dtype`; for bf16 the bits of `x.astype(bf16)` under
    `--xla_allow_excess_precision=false`.
  """
  if jnp.dtype(dtype) != jnp.bfloat16:
    return x.astype(dtype)
  rounded = lax.reduce_precision(
      x.astype(jnp.float32),
      exponent_bits=_BF16_EXPONENT_BITS,
      mantissa_bits=_BF16_MANTISSA_BITS,
  )
  return rounded.astype(jnp.bfloat16)


def hard_upcast(x: jax.Array) -> jax.Array:
  """`x.astype(f32)` that re-asserts the bf16 rounding of a bf16 `x`.

  The consumer-side counterpart of `hard_round`: if the producer's `convert`
  to bf16 was simplified away, the f32 value that reaches this upcast is
  unrounded, and the `reduce_precision` rounds it to the bits the bf16 buffer
  would have held. For a materialized bf16 `x` it is a no-op; non-bf16 inputs
  are a plain upcast.

  Args:
    x: An array.

  Returns:
    `x` as f32.
  """
  x_f32 = x.astype(jnp.float32)
  if x.dtype != jnp.bfloat16:
    return x_f32
  return lax.reduce_precision(
      x_f32,
      exponent_bits=_BF16_EXPONENT_BITS,
      mantissa_bits=_BF16_MANTISSA_BITS,
  )


def _to_f32(x: jax.Array, hard: bool) -> jax.Array:
  return hard_upcast(x) if hard else x.astype(jnp.float32)


def online_softmax_scan(
    q_grouped: jax.Array,  # [B, KH, G, T, D]
    k_grouped: jax.Array,  # [B, KH, S, D]
    v_grouped: jax.Array,  # [B, KH, S, D]
    mask_bts: jax.Array,  # [B, T, S] bool
    *,
    scale: float,
    kv_block_size: int = DEFAULT_ATTENTION_KV_BLOCK_SIZE,
    hard_rounding: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Fixed KV-block online softmax attention: the XLA forward (`XLA_FWD`).

  The Zero-TIM forward before A4, kept verbatim as the `AUTODIFF` primal and
  for A/B against `flash_fwd_pallas` (see the module docstring for why it
  cannot be the contract across query lengths).

  Args:
    q_grouped: Queries `[B, KH, G, T, D]` (`G` query heads per KV head).
    k_grouped: Keys `[B, KH, S, D]`.
    v_grouped: Values `[B, KH, S, D]`.
    mask_bts: Boolean attention mask `[B, T, S]`.
    scale: Softmax temperature (`1 / sqrt(D)`).
    kv_block_size: Fixed key block (`bkv_csz`) that pins the reduction order.
    hard_rounding: Route the bf16 boundaries of the scan (the `q` / `k` / `v`
      upcasts and the output cast) through `hard_upcast` / `hard_round` instead
      of `astype` (see `qwen3_model_canon.HARD_ROUNDING`). Same bits under
      `--xla_allow_excess_precision=false`.

  Returns:
    `(out[B, KH, G, T, D] in q dtype, m_final[B, KH, G, Tp], l_final[B, KH, G,
    Tp])` where `Tp = round_up(T, FORWARD_QUERY_PAD)`; the exp2-domain running
    max / sum are the forward's own carry (free to expose) and are consumed by
    the flash backward only.
  """
  b, kh, g, t, d = q_grouped.shape
  s = k_grouped.shape[2]
  out_dtype = q_grouped.dtype

  # Pad query length `t` to a fixed multiple of 64 (analogous to RPA v3's
  # `bq_csz=128` / `MIN_TOKEN_BUCKET`) so that `G * tp >= 128` in both
  # Decode (`t=1`) and Prefill (`t=64`), preventing XLA TPU from transposing
  # the LHS/RHS operands of the MXU `dot_general` (`qk` and `pv`) when `t=1`.
  tp = round_up(t, FORWARD_QUERY_PAD)
  if tp != t:
    pad_t = tp - t
    q_grouped = jnp.pad(
        q_grouped,
        ((0, 0), (0, 0), (0, 0), (0, pad_t), (0, 0)),
        constant_values=0,
    )
    mask_bts = jnp.pad(
        mask_bts, ((0, 0), (0, pad_t), (0, 0)), constant_values=False
    )

  bkv = int(kv_block_size)
  sp = round_up(s, bkv)
  num_kv_blocks = sp // bkv

  if sp != s:
    pad_s = sp - s
    k_grouped = jnp.pad(
        k_grouped, ((0, 0), (0, 0), (0, pad_s), (0, 0)), constant_values=0
    )
    v_grouped = jnp.pad(
        v_grouped, ((0, 0), (0, 0), (0, pad_s), (0, 0)), constant_values=0
    )
    mask_bts = jnp.pad(
        mask_bts, ((0, 0), (0, 0), (0, pad_s)), constant_values=False
    )

  k_blocks = k_grouped.reshape(b, kh, num_kv_blocks, bkv, d).transpose(
      2, 0, 1, 3, 4
  )
  v_blocks = v_grouped.reshape(b, kh, num_kv_blocks, bkv, d).transpose(
      2, 0, 1, 3, 4
  )
  mask_blocks = mask_bts.reshape(b, tp, num_kv_blocks, bkv).transpose(
      2, 0, 1, 3
  )[
      :, :, None, None, :, :
  ]  # [num_kv_blocks, B, 1, 1, tp, bkv]

  q_f32 = _to_f32(q_grouped, hard_rounding) * jnp.float32(scale * LOG2_E)
  init_m = jnp.full((b, kh, g, tp), -jnp.inf, dtype=jnp.float32)
  init_l = jnp.zeros((b, kh, g, tp), dtype=jnp.float32)
  init_acc = jnp.zeros((b, kh, g, tp, d), dtype=jnp.float32)

  # HBM Optimization: checkpointing each KV-block scan step prevents lax.scan
  # from saving per-block float32 `qk` and `p` intermediates across all S/bkv
  # blocks in HBM during Fwd+Bwd, while keeping the forward primal 100%
  # untouched (`|Fwd - FwdBwd| == 0.0`). Only the fp32 `(m, l, acc)` carry is
  # saved per step, which is why `DEFAULT_ATTENTION_KV_BLOCK_SIZE=512` (one
  # step for `S <= 512`) also minimises the Fwd+Bwd residual footprint.
  @jax.checkpoint
  def scan_kv_block(carry, kv_inputs):
    m_prev, l_prev, acc_prev = carry
    k_blk, v_blk, mask_blk = kv_inputs
    qk = jnp.einsum(
        'bhgtd,bhsd->bhgts',
        q_f32,
        _to_f32(k_blk, hard_rounding),
        preferred_element_type=jnp.float32,
    )
    qk_masked = jnp.where(mask_blk, qk, -jnp.inf)
    blk_valid_any = jnp.any(mask_blk, axis=-1)  # [B, 1, 1, tp]

    m_curr = jnp.max(qk_masked, axis=-1)  # [B, KH, G, tp]
    m_cand = jnp.maximum(m_prev, m_curr)
    m_safe = jnp.where(jnp.isneginf(m_cand), jnp.float32(0.0), m_cand)

    alpha = jnp.where(
        jnp.isneginf(m_prev),
        jnp.float32(0.0),
        jnp.exp2(m_prev - m_safe),
    )
    p = jnp.where(
        mask_blk,
        jnp.exp2(qk_masked - m_safe[..., None]),
        jnp.float32(0.0),
    )
    l_blk = jnp.sum(p, axis=-1, dtype=jnp.float32)
    l_cand = l_prev * alpha + l_blk

    pv = jnp.einsum(
        'bhgts,bhsd->bhgtd',
        p,
        _to_f32(v_blk, hard_rounding),
        preferred_element_type=jnp.float32,
    )
    acc_cand = acc_prev * alpha[..., None] + pv

    m_next = jnp.where(blk_valid_any, m_cand, m_prev)
    l_next = jnp.where(blk_valid_any, l_cand, l_prev)
    acc_next = jnp.where(blk_valid_any[..., None], acc_cand, acc_prev)
    return (m_next, l_next, acc_next), None

  (m_final, l_final, acc_final), _ = lax.scan(
      scan_kv_block,
      (init_m, init_l, init_acc),
      (k_blocks, v_blocks, mask_blocks),
  )

  denom = jnp.where(l_final == 0.0, jnp.float32(1.0), l_final)[..., None]
  out_f32 = acc_final / denom
  if hard_rounding:
    out = hard_round(out_f32, out_dtype)
  else:
    out = out_f32.astype(out_dtype)
  return out[:, :, :, :t, :], m_final, l_final


def row_lse2(m_final: jax.Array, l_final: jax.Array) -> jax.Array:
  """exp2-domain log-sum-exp of the forward; `0` for rows with no valid key."""
  return jnp.where(l_final > 0.0, m_final + jnp.log2(l_final), jnp.float32(0.0))


def _lane_tile(x: jax.Array, width: int) -> jax.Array:
  """`(rows, NUM_LANES)` lane-replicated row statistics as `(rows, width)`."""
  if width == NUM_LANES:
    return x
  tiled = jnp.tile(x, (1, -(-width // NUM_LANES)))
  return tiled if tiled.shape[1] == width else tiled[:, :width]


def flash_fwd_pallas(
    q: jax.Array,  # [B, KH, G, T, D]
    k: jax.Array,  # [B, KH, S, D]
    v: jax.Array,  # [B, KH, S, D]
    mask: jax.Array,  # [B, T, S] bool
    *,
    scale: float,
    kv_block_size: int = DEFAULT_ATTENTION_KV_BLOCK_SIZE,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Pallas TPU online-softmax attention forward (`PALLAS_FWD`, the contract).

  Grid `(B, KH, query blocks, key blocks)` with the key blocks innermost; a
  query block is `bq = FWD_QUERY_BLOCK` rows of ONE query head of the group
  (`G * Tp / bq` blocks per `(b, kh)`), a key block is the `kv_block_size`
  DMA block, visited in `FWD_KV_SUBBLOCK`-key sub-blocks in key order. Per
  sub-block the kernel runs the exp2 online-softmax update of
  `online_softmax_scan` on a `(bq, sub)` tile:

      qk  = (q * c).bf16 . k^T            m_curr = rowmax(qk | mask)
      m   = max(m_prev, m_curr)           alpha  = exp2(m_prev - m)
      p   = exp2(qk - m) | mask           l      = l_prev * alpha + rowsum(p)
      acc = acc_prev * alpha + p.bf16 . v

  with the `-inf` guards of the scan (`m_safe`, `alpha = 0` from an empty
  state) and the per-row pass-through: a row without a valid key in the
  sub-block keeps `(m, l, acc)` unchanged, exactly as if the sub-block had been
  skipped. Sub-blocks whose `(bq, sub)` mask tile is all-False ARE skipped
  (scalar-prefetched `block_map`); since every row of such a tile would pass
  through, skipping is not a numerical choice. The MXU operands are bf16 with
  f32 accumulation (what the TPU default precision gives the scan's f32
  einsums), the row statistics live in `(bq, NUM_LANES)` lane-replicated f32
  VMEM scratch, and the output is `acc / l` cast once in-kernel.

  Why this is the contract: every row of every program goes through the same
  tile program in the same order, so the forward is invariant to the query
  length (`T=1` is padded to one block), the batch size and the key length /
  cache size (blocks beyond the last valid key are skipped or pass through) by
  construction, which the XLA scan is not (module docstring).

  Args:
    q: Queries `[B, KH, G, T, D]` (`G` query heads per KV head).
    k: Keys `[B, KH, S, D]`.
    v: Values `[B, KH, S, D]`.
    mask: Boolean attention mask `[B, T, S]`.
    scale: Softmax temperature (`1 / sqrt(D)`).
    kv_block_size: Fixed key block (`bkv_csz`), a multiple of `FWD_KV_SUBBLOCK`.
    interpret: Run the kernel in Pallas interpret mode (CPU tests).

  Returns:
    `(out[B, KH, G, T, D] in q dtype, m_final[B, KH, G, Tp], l_final[B, KH, G,
    Tp])` where `Tp = round_up(T, FWD_QUERY_BLOCK)`; `m / l` are the exp2-domain
    running max / sum consumed by the flash backward (`row_lse2`).

  Raises:
    ValueError: If `kv_block_size` is not a multiple of `FWD_KV_SUBBLOCK` or
      `q`, `k`, `v` do not share one dtype.
  """
  b, kh, g, t, d = q.shape
  s = k.shape[2]
  bkv = int(kv_block_size)
  sub = FWD_KV_SUBBLOCK
  bq = FWD_QUERY_BLOCK
  if bkv % sub:
    raise ValueError(f'kv_block_size={bkv} must be a multiple of {sub}')
  if not (q.dtype == k.dtype == v.dtype):
    raise ValueError(
        'flash_fwd_pallas needs one dtype for q, k and v, got'
        f' {q.dtype}, {k.dtype}, {v.dtype}'
    )
  tp = round_up(t, bq)
  sp = round_up(s, bkv)
  nqt = tp // bq  # query blocks per head
  nq = g * nqt  # query blocks per (b, kh)
  nkv = sp // bkv
  nsub = bkv // sub

  q_p = _pad_rows(q, 3, tp)
  k_p = _pad_rows(k, 2, sp)
  v_p = _pad_rows(v, 2, sp)
  mask_p = jnp.pad(
      mask.astype(jnp.bool_),
      ((0, 0), (0, tp - t), (0, sp - s)),
      constant_values=False,
  )
  block_map = jnp.any(
      mask_p.reshape(b, nqt, bq, nkv * nsub, sub), axis=(2, 4)
  ).astype(
      jnp.int32
  )  # [B, nqt, nkv * nsub]

  # Python float (not a `jnp` scalar): a Pallas kernel may not close over JAX
  # array constants; weak-typed, rounds to f32 at the use site.
  c_f32 = float(scale * LOG2_E)
  neg_inf = float('-inf')

  def kernel(
      block_map_ref,
      q_ref,
      k_ref,
      v_ref,
      mask_ref,
      o_ref,
      m_ref,
      l_ref,
      m_acc,
      l_acc,
      acc,
  ):
    bi = pl.program_id(0)
    i = pl.program_id(2)
    j = pl.program_id(3)
    t_blk = lax.rem(i, nqt)

    @pl.when(j == 0)
    def _init():
      m_acc[...] = jnp.full_like(m_acc, neg_inf)
      l_acc[...] = jnp.zeros_like(l_acc)
      acc[...] = jnp.zeros_like(acc)

    q_blk = q_ref[...]  # (bq, D)
    q_scaled = (q_blk.astype(jnp.float32) * c_f32).astype(q_blk.dtype)

    for jj in range(nsub):

      @pl.when(block_map_ref[bi, t_blk, j * nsub + jj] != 0)
      def _sub_block(jj=jj):  # `pl.when` runs the body right away.
        keys = pl.ds(jj * sub, sub)
        k_sub = k_ref[keys, :]  # (sub, D)
        v_sub = v_ref[keys, :]
        m_sub = mask_ref[:, keys]  # (bq, sub) bool
        qk = lax.dot_general(
            q_scaled, k_sub, _NT_DIMS, preferred_element_type=jnp.float32
        )  # (bq, sub)
        qk_masked = jnp.where(m_sub, qk, neg_inf)

        m_prev = m_acc[...]  # (bq, NUM_LANES)
        l_prev = l_acc[...]
        acc_prev = acc[...]  # (bq, D)
        m_curr = lax.broadcast_in_dim(
            jnp.max(qk_masked, axis=-1), m_prev.shape, (0,)
        )
        m_cand = jnp.maximum(m_prev, m_curr)
        m_safe = jnp.where(m_cand == neg_inf, 0.0, m_cand)
        alpha = jnp.where(m_prev == neg_inf, 0.0, jnp.exp2(m_prev - m_safe))
        p = jnp.where(m_sub, jnp.exp2(qk_masked - _lane_tile(m_safe, sub)), 0.0)
        l_blk = lax.broadcast_in_dim(jnp.sum(p, axis=-1), l_prev.shape, (0,))
        l_cand = l_prev * alpha + l_blk
        pv = jnp.dot(
            p.astype(v_sub.dtype), v_sub, preferred_element_type=jnp.float32
        )  # (bq, D)
        acc_cand = acc_prev * _lane_tile(alpha, d) + pv

        # A row with a valid key in this sub-block has a finite `m_curr`
        # (`qk` is finite); the others carry their state through unchanged.
        m_acc[...] = jnp.where(m_curr > neg_inf, m_cand, m_prev)
        l_acc[...] = jnp.where(m_curr > neg_inf, l_cand, l_prev)
        acc[...] = jnp.where(
            _lane_tile(m_curr, d) > neg_inf, acc_cand, acc_prev
        )

    @pl.when(j == nkv - 1)
    def _finalize():
      l_fin = l_acc[...]
      denom = jnp.where(l_fin == 0.0, 1.0, l_fin)
      o_ref[...] = (acc[...] / _lane_tile(denom, d)).astype(o_ref.dtype)
      m_ref[...] = m_acc[...]
      l_ref[...] = l_fin

  def q_index(bi, h, i, j, _):
    del j
    return (bi, h, i // nqt, lax.rem(i, nqt), 0)

  def kv_index(bi, h, i, j, _):
    del i
    return (bi, h, j, 0)

  def mask_index(bi, h, i, j, _):
    del h
    return (bi, lax.rem(i, nqt), j)

  out_p, m_out, l_out = pl.pallas_call(
      kernel,
      out_shape=(
          jax.ShapeDtypeStruct((b, kh, g, tp, d), q.dtype),
          jax.ShapeDtypeStruct((b, kh, g, tp, NUM_LANES), jnp.float32),
          jax.ShapeDtypeStruct((b, kh, g, tp, NUM_LANES), jnp.float32),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[
              pl.BlockSpec((None, None, None, bq, d), q_index),
              pl.BlockSpec((None, None, bkv, d), kv_index),
              pl.BlockSpec((None, None, bkv, d), kv_index),
              pl.BlockSpec((None, bq, bkv), mask_index),
          ],
          out_specs=[
              pl.BlockSpec((None, None, None, bq, d), q_index),
              pl.BlockSpec((None, None, None, bq, NUM_LANES), q_index),
              pl.BlockSpec((None, None, None, bq, NUM_LANES), q_index),
          ],
          scratch_shapes=[
              pltpu.VMEM((bq, NUM_LANES), jnp.float32),
              pltpu.VMEM((bq, NUM_LANES), jnp.float32),
              pltpu.VMEM((bq, d), jnp.float32),
          ],
          grid=(b, kh, nq, nkv),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=('parallel', 'parallel', 'parallel', 'arbitrary'),
      ),
      interpret=interpret,
      name=f'canon_attention_flash_fwd_b{b}_h{kh}_g{g}_t{tp}_s{sp}',
  )(block_map, q_p, k_p, v_p, mask_p)
  return out_p[:, :, :, :t, :], m_out[..., 0], l_out[..., 0]


def _scaled_query_operand(q: jax.Array, scale: float) -> jax.Array:
  """`(q * scale * log2 e)` rounded to `q.dtype`: the forward's MXU operand.

  The forward feeds `q.astype(f32) * c` to an f32 `einsum`; at the default TPU
  matmul precision the MXU consumes it rounded to bf16, so rounding the scaled
  query (rather than scaling a bf16 `q.k` product) keeps the recomputed
  `exp2(q.k - lse2)` consistent with the forward's statistics.

  Args:
    q: Queries in the MXU operand dtype (bf16 on TPU).
    scale: Softmax temperature (`1 / sqrt(D)`).

  Returns:
    The scaled queries, rounded back to `q.dtype`.
  """
  return (q.astype(jnp.float32) * jnp.float32(scale * LOG2_E)).astype(q.dtype)


def flash_bwd_xla(
    q: jax.Array,  # [B, KH, G, T, D]
    k: jax.Array,  # [B, KH, S, D]
    v: jax.Array,  # [B, KH, S, D]
    o: jax.Array,  # [B, KH, G, T, D]
    lse2: jax.Array,  # [B, KH, G, T] f32
    mask: jax.Array,  # [B, T, S] bool
    do: jax.Array,  # [B, KH, G, T, D]
    *,
    scale: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Dense flash-attention backward (reference / CPU implementation)."""
  do = do.astype(q.dtype)
  q_scaled = _scaled_query_operand(q, scale)
  qk = jnp.einsum(
      'bhgtd,bhsd->bhgts', q_scaled, k, preferred_element_type=jnp.float32
  )
  mask5 = mask[:, None, None, :, :]
  p = jnp.where(mask5, jnp.exp2(qk - lse2[..., None]), jnp.float32(0.0))
  dv = jnp.einsum(
      'bhgts,bhgtd->bhsd',
      p.astype(do.dtype),
      do,
      preferred_element_type=jnp.float32,
  )
  dp = jnp.einsum(
      'bhgtd,bhsd->bhgts', do, v, preferred_element_type=jnp.float32
  )
  di = jnp.sum(do.astype(jnp.float32) * o.astype(jnp.float32), axis=-1)
  ds = p * (dp - di[..., None])
  ds_mxu = ds.astype(q.dtype)
  dq = jnp.einsum(
      'bhgts,bhsd->bhgtd', ds_mxu, k, preferred_element_type=jnp.float32
  )
  dk = jnp.einsum(
      'bhgts,bhgtd->bhsd', ds_mxu, q, preferred_element_type=jnp.float32
  )
  scale_f32 = jnp.float32(scale)
  return (
      (dq * scale_f32).astype(q.dtype),
      (dk * scale_f32).astype(k.dtype),
      dv.astype(v.dtype),
  )


def _bwd_query_block(tp: int) -> int:
  return 256 if tp % 256 == 0 else BWD_QUERY_PAD


def flash_bwd_pallas(
    q: jax.Array,  # [B, KH, G, T, D]
    k: jax.Array,  # [B, KH, S, D]
    v: jax.Array,  # [B, KH, S, D]
    o: jax.Array,  # [B, KH, G, T, D]
    lse2: jax.Array,  # [B, KH, G, T] f32
    mask: jax.Array,  # [B, T, S] bool
    do: jax.Array,  # [B, KH, G, T, D]
    *,
    scale: float,
    kv_block_size: int = DEFAULT_ATTENTION_KV_BLOCK_SIZE,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Pallas TPU flash-attention backward (block-sparse over the mask).

  Grid `(B, KH, key blocks, query blocks)` with the query blocks innermost;
  a query block is `bq` rows of ONE query head of the group (`G * tp / bq`
  blocks per key block), so every operand block is a plain 2-D `(rows, D)` /
  `(8, bq)` / `(bkv, bq)` tile and no in-kernel relayout is needed. Per step:
  `dk / dv` of the key block accumulate in f32 scratch (written at the last
  query block), `dq` of the query block accumulates over the key block's
  128-key sub-blocks in f32 scratch and is written per key block
  (`[nkv, ...]`, summed outside). Sub-blocks whose `(keys, queries)` mask
  tile is all-False are skipped via the scalar-prefetched `block_map`.

  Args:
    q: Queries `[B, KH, G, T, D]`.
    k: Keys `[B, KH, S, D]`.
    v: Values `[B, KH, S, D]`.
    o: Forward output `[B, KH, G, T, D]`.
    lse2: exp2-domain log-sum-exp of the forward `[B, KH, G, T]` (f32).
    mask: Boolean attention mask `[B, T, S]`.
    do: Output cotangent `[B, KH, G, T, D]`.
    scale: Softmax temperature (`1 / sqrt(D)`).
    kv_block_size: Key block of the forward (`bkv_csz`).
    interpret: Run the Pallas kernel in interpret mode (CPU tests).

  Returns:
    `(dq, dk, dv)` in the dtypes of `q`, `k` and `v`.
  """
  b, kh, g, t, d = q.shape
  s = k.shape[2]
  bkv = int(kv_block_size)
  sub = BWD_KV_SUBBLOCK
  if bkv % sub:
    raise ValueError(f'kv_block_size={bkv} must be a multiple of {sub}')
  if not (q.dtype == k.dtype == v.dtype):
    raise ValueError(
        'flash_bwd_pallas needs one dtype for q, k and v, got'
        f' {q.dtype}, {k.dtype}, {v.dtype}'
    )
  tp = round_up(t, BWD_QUERY_PAD)
  sp = round_up(s, bkv)
  bq = _bwd_query_block(tp)
  nqt = tp // bq  # query blocks per head
  nq = g * nqt  # query blocks per (b, kh)
  nkv = sp // bkv
  nsub = bkv // sub
  do = do.astype(q.dtype)

  q_p = _pad_rows(q, 3, tp)
  do_p = _pad_rows(do, 3, tp)
  k_p = _pad_rows(k, 2, sp)
  v_p = _pad_rows(v, 2, sp)

  # Row statistics as `(8, tp)` sublane-broadcast tiles (`[:1, :]` in-kernel).
  di = jnp.sum(do.astype(jnp.float32) * o.astype(jnp.float32), axis=-1)
  stats_shape = (b, kh, g, NUM_SUBLANES, tp)
  lse_b = jnp.broadcast_to(
      _pad_rows(lse2, 3, tp)[:, :, :, None, :], stats_shape
  )
  di_b = jnp.broadcast_to(_pad_rows(di, 3, tp)[:, :, :, None, :], stats_shape)

  # Mask in the kernel's `(keys, queries)` orientation plus its block map.
  mask_p = jnp.pad(
      mask.astype(jnp.bool_),
      ((0, 0), (0, tp - t), (0, sp - s)),
      constant_values=False,
  )
  mask_t = jnp.transpose(mask_p, (0, 2, 1))  # [B, sp, tp]
  block_map = jnp.any(
      mask_t.reshape(b, nkv * nsub, sub, nqt, bq), axis=(2, 4)
  ).astype(
      jnp.int32
  )  # [B, nkv * nsub, nqt]

  # Python floats (not `jnp` scalars): a Pallas kernel may not close over JAX
  # array constants. They are weak-typed and round to f32 at the use site,
  # exactly like the forward's `jnp.float32(scale * LOG2_E)`.
  c_f32 = float(scale * LOG2_E)
  scale_f32 = float(scale)

  def kernel(
      block_map_ref,
      q_ref,
      k_ref,
      v_ref,
      do_ref,
      lse_ref,
      di_ref,
      mask_ref,
      dq_ref,
      dk_ref,
      dv_ref,
      dq_acc,
      dk_acc,
      dv_acc,
  ):
    bi = pl.program_id(0)
    j = pl.program_id(2)
    i = pl.program_id(3)
    t_blk = lax.rem(i, nqt)

    @pl.when(i == 0)
    def _init_kv():
      dk_acc[...] = jnp.zeros_like(dk_acc)
      dv_acc[...] = jnp.zeros_like(dv_acc)

    dq_acc[...] = jnp.zeros_like(dq_acc)

    q_blk = q_ref[...]  # (bq, D)
    do_blk = do_ref[...]  # (bq, D)
    q_scaled = (q_blk.astype(jnp.float32) * c_f32).astype(q_blk.dtype)
    lse = lse_ref[:1, :]  # (1, bq)
    di_row = di_ref[:1, :]  # (1, bq)

    for jj in range(nsub):

      @pl.when(block_map_ref[bi, j * nsub + jj, t_blk] != 0)
      def _sub_block(jj=jj):  # `pl.when` runs the body right away.
        rows = pl.ds(jj * sub, sub)
        k_sub = k_ref[rows, :]  # (sub, D)
        v_sub = v_ref[rows, :]
        m_sub = mask_ref[rows, :]  # (sub, bq) bool
        qk = lax.dot_general(
            k_sub, q_scaled, _NT_DIMS, preferred_element_type=jnp.float32
        )  # (sub, bq)
        p = jnp.where(m_sub, jnp.exp2(qk - lse), jnp.float32(0.0))
        dv_acc[rows, :] += jnp.dot(
            p.astype(do_blk.dtype), do_blk, preferred_element_type=jnp.float32
        )
        dp = lax.dot_general(
            v_sub, do_blk, _NT_DIMS, preferred_element_type=jnp.float32
        )  # (sub, bq)
        ds = p * (dp - di_row)
        dk_acc[rows, :] += jnp.dot(
            ds.astype(q_blk.dtype), q_blk, preferred_element_type=jnp.float32
        )
        # Transpose in f32 (32-bit XLU transpose), then round for the MXU.
        dq_acc[...] += jnp.dot(
            ds.T.astype(k_sub.dtype), k_sub, preferred_element_type=jnp.float32
        )

    dq_ref[...] = (dq_acc[...] * scale_f32).astype(dq_ref.dtype)

    @pl.when(i == nq - 1)
    def _write_kv():
      dk_ref[...] = (dk_acc[...] * scale_f32).astype(dk_ref.dtype)
      dv_ref[...] = dv_acc[...].astype(dv_ref.dtype)

  def q_index(bi, h, j, i, _):
    del j
    return (bi, h, i // nqt, lax.rem(i, nqt), 0)

  def kv_index(bi, h, j, i, _):
    del i
    return (bi, h, j, 0)

  def stats_index(bi, h, j, i, _):
    del j
    return (bi, h, i // nqt, 0, lax.rem(i, nqt))

  def mask_index(bi, h, j, i, _):
    del h
    return (bi, j, lax.rem(i, nqt))

  def dq_index(bi, h, j, i, _):
    return (j, bi, h, i // nqt, lax.rem(i, nqt), 0)

  dq_unreduced, dk_p, dv_p = pl.pallas_call(
      kernel,
      out_shape=(
          jax.ShapeDtypeStruct((nkv, b, kh, g, tp, d), q.dtype),
          jax.ShapeDtypeStruct((b, kh, sp, d), k.dtype),
          jax.ShapeDtypeStruct((b, kh, sp, d), v.dtype),
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[
              pl.BlockSpec((None, None, None, bq, d), q_index),
              pl.BlockSpec((None, None, bkv, d), kv_index),
              pl.BlockSpec((None, None, bkv, d), kv_index),
              pl.BlockSpec((None, None, None, bq, d), q_index),
              pl.BlockSpec((None, None, None, NUM_SUBLANES, bq), stats_index),
              pl.BlockSpec((None, None, None, NUM_SUBLANES, bq), stats_index),
              pl.BlockSpec((None, bkv, bq), mask_index),
          ],
          out_specs=[
              pl.BlockSpec((None, None, None, None, bq, d), dq_index),
              pl.BlockSpec((None, None, bkv, d), kv_index),
              pl.BlockSpec((None, None, bkv, d), kv_index),
          ],
          scratch_shapes=[
              pltpu.VMEM((bq, d), jnp.float32),
              pltpu.VMEM((bkv, d), jnp.float32),
              pltpu.VMEM((bkv, d), jnp.float32),
          ],
          grid=(b, kh, nkv, nq),
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=('parallel', 'parallel', 'parallel', 'arbitrary'),
      ),
      interpret=interpret,
      name=f'canon_attention_flash_bwd_b{b}_h{kh}_g{g}_t{tp}_s{sp}',
  )(block_map, q_p, k_p, v_p, do_p, lse_b, di_b, mask_t)

  if nkv == 1:
    dq_p = dq_unreduced[0]
  else:
    dq_p = jnp.sum(dq_unreduced.astype(jnp.float32), axis=0).astype(q.dtype)
  return dq_p[:, :, :, :t, :], dk_p[:, :, :s, :], dv_p[:, :, :s, :]


def attention_core(
    q_grouped: jax.Array,  # [B, KH, G, T, D]
    k_grouped: jax.Array,  # [B, KH, S, D]
    v_grouped: jax.Array,  # [B, KH, S, D]
    mask_bts: jax.Array,  # [B, T, S] bool
    *,
    scale: float,
    kv_block_size: int = DEFAULT_ATTENTION_KV_BLOCK_SIZE,
    interpret: bool = False,
    bwd: str | None = None,
    fwd: str | None = None,
    hard_rounding: bool = False,
) -> jax.Array:
  """The attention forward selected by `fwd` with the backward selected by `bwd`.

  Args:
    q_grouped: Queries `[B, KH, G, T, D]` (`G` query heads per KV head).
    k_grouped: Keys `[B, KH, S, D]`.
    v_grouped: Values `[B, KH, S, D]`.
    mask_bts: Boolean attention mask `[B, T, S]` (causal, padding and segment
      masks already merged).
    scale: Softmax temperature (`1 / sqrt(D)`).
    kv_block_size: Fixed key block (`bkv_csz`).
    interpret: Run the Pallas kernels in interpret mode (CPU tests).
    bwd: `AUTODIFF`, `XLA_BWD` or `PALLAS_BWD`; `None` reads `ATTENTION_BWD`.
      `AUTODIFF` requires the XLA forward.
    fwd: `PALLAS_FWD` or `XLA_FWD`; `None` reads `ATTENTION_FWD`.
    hard_rounding: See `online_softmax_scan`; XLA forward only (the Pallas
      forward materializes its operands and output, so its rounding points
      cannot move; the backward has no bitwise contract).

  Returns:
    `out[B, KH, G, T, D]` in the query dtype; for a given `fwd` bitwise the
    same for every `bwd` (the primal is always that forward).

  Raises:
    ValueError: For an unknown `fwd` / `bwd`, or `AUTODIFF` with the Pallas
      forward.
  """
  fwd_mode = ATTENTION_FWD if fwd is None else fwd
  if fwd_mode == RPA_FWD:
    raise ValueError(
        f'{RPA_FWD!r} is the engine kernel dispatched by the model glue'
        ' (qwen3_model_canon.CanonAttention -> rpa_backend); attention_core'
        f' implements {ATTENTION_FWD_MODES}'
    )
  if fwd_mode not in ATTENTION_FWD_MODES:
    raise ValueError(f'unknown attention forward {fwd_mode!r}')
  mode = ATTENTION_BWD if bwd is None else bwd
  if mode not in ATTENTION_BWD_MODES:
    raise ValueError(f'unknown attention backward {mode!r}')
  if mode == AUTODIFF and fwd_mode != XLA_FWD:
    raise ValueError(
        f'{AUTODIFF!r} differentiates the XLA scan; the {fwd_mode!r} forward'
        f' has no JVP (use fwd={XLA_FWD!r})'
    )

  def forward(q, k, v, m):
    if fwd_mode == PALLAS_FWD:
      return flash_fwd_pallas(
          q,
          k,
          v,
          m,
          scale=scale,
          kv_block_size=kv_block_size,
          interpret=interpret,
      )
    return online_softmax_scan(
        q,
        k,
        v,
        m,
        scale=scale,
        kv_block_size=kv_block_size,
        hard_rounding=hard_rounding,
    )

  def primal(q, k, v, m):
    out, _, _ = forward(q, k, v, m)
    return out

  if mode == AUTODIFF:
    return primal(q_grouped, k_grouped, v_grouped, mask_bts)

  t = int(q_grouped.shape[3])

  @jax.custom_vjp
  def op(q, k, v, m):
    return primal(q, k, v, m)

  def op_fwd(q, k, v, m):
    out, m_final, l_final = forward(q, k, v, m)
    lse2 = row_lse2(m_final, l_final)[:, :, :, :t]
    return out, (q, k, v, out, lse2, m)

  def op_bwd(residual, d_out):
    q, k, v, out, lse2, m = residual
    if mode == PALLAS_BWD:
      dq, dk, dv = flash_bwd_pallas(
          q,
          k,
          v,
          out,
          lse2,
          m,
          d_out,
          scale=scale,
          kv_block_size=kv_block_size,
          interpret=interpret,
      )
    else:
      dq, dk, dv = flash_bwd_xla(q, k, v, out, lse2, m, d_out, scale=scale)
    return dq, dk, dv, None

  op.defvjp(op_fwd, op_bwd)
  return op(q_grouped, k_grouped, v_grouped, mask_bts)
