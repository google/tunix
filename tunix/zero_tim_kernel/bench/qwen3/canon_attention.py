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
rows, any batch size.

* `PALLAS_FWD` (`flash_fwd_pallas`, the contract): one TPU kernel that
  processes every query row as part of a fixed
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
* `RPA_FWD`: a measurement setting rather than a forward of this module: the
  model glue (`qwen3_model_canon` -> `rpa_backend`) runs the bench programs on
  the engine's ragged paged attention kernel under the canonical call-site
  controls, so that the cross-engine question can be measured on one host.
  `attention_core` rejects it.

Backward (`PALLAS_BWD`, `flash_bwd_pallas`): a `jax.custom_vjp` around the
forward whose gradient is the block-sparse Pallas flash-attention backward in
the transposed `(keys, queries)` orientation (row statistics are `(1, bq)` lane
vectors broadcast along sublanes) that keeps `p / dp / ds` in VMEM and skips
fully masked 128-key sub-blocks through a scalar-prefetched block map.
"""

from __future__ import annotations

import jax
from jax import lax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LOG2_E = 1.4426950408889634

# Fixed KV-block granularity (`bkv_csz`) of `canonical_online_attention`,
# mirroring `rpa_canonical.CANONICAL_BLOCK_SIZES = (128, 512, 128, 512)`.
DEFAULT_ATTENTION_KV_BLOCK_SIZE = 512

# Forward implementations (see module docstring).
PALLAS_FWD = 'pallas'
ATTENTION_FWD_MODES = (PALLAS_FWD,)

# The engine's ragged-paged-attention kernel (tpu_inference RPA v3) under the
# canonical call-site controls, dispatched by the model glue
# (`qwen3_model_canon.CanonAttention` -> `rpa_backend`) instead of
# `attention_core`.
RPA_FWD = 'rpa'
ATTENTION_FWD_CHOICES = ATTENTION_FWD_MODES + (RPA_FWD,)

ATTENTION_FWD = PALLAS_FWD

# Forward kernel geometry. FIXED on purpose: every query row of every program
# is processed as part of a `FWD_QUERY_BLOCK`-row block (`T=1` decode pads to
# one block) against `FWD_KV_SUBBLOCK`-key sub-blocks of the `kv_block_size`
# key block, so the per-row arithmetic is the same whatever `T`, `S` and `B`
# are. Row statistics are kept as `(bq, NUM_LANES)` lane-replicated tiles.
FWD_QUERY_BLOCK = 128
FWD_KV_SUBBLOCK = 128
NUM_LANES = 128

# Backward implementation.
PALLAS_BWD = 'pallas'
ATTENTION_BWD_MODES = (PALLAS_BWD,)
ATTENTION_BWD = PALLAS_BWD

# Backward kernel geometry. Queries are padded to a multiple of
# `BWD_QUERY_PAD` so the `(1, bq)` row-statistics blocks are lane-aligned
# (`bq in {128, 256}`); keys keep the forward's `kv_block_size` block and are
# visited in `BWD_KV_SUBBLOCK`-key sub-blocks that can be skipped when fully
# masked.
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
  sub-block the kernel runs the exp2 online-softmax update on a `(bq, sub)`
  tile:

      qk  = (q * c).bf16 . k^T            m_curr = rowmax(qk | mask)
      m   = max(m_prev, m_curr)           alpha  = exp2(m_prev - m)
      p   = exp2(qk - m) | mask           l      = l_prev * alpha + rowsum(p)
      acc = acc_prev * alpha + p.bf16 . v

  with `-inf` guards (`m_safe`, `alpha = 0` from an empty state) and per-row
  pass-through: a row without a valid key in the sub-block keeps `(m, l, acc)`
  unchanged, exactly as if the sub-block had been skipped. Sub-blocks whose
  `(bq, sub)` mask tile is all-False ARE skipped (scalar-prefetched
  `block_map`). The MXU operands are bf16 with f32 accumulation, the row
  statistics live in `(bq, NUM_LANES)` lane-replicated f32 VMEM scratch, and
  the output is `acc / l` cast once in-kernel.

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
            q_scaled,
            k_sub,
            _NT_DIMS,
            precision=lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
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
            p.astype(v_sub.dtype),
            v_sub,
            precision=lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )  # (bq, D)
        acc_cand = acc_prev * _lane_tile(alpha, d) + pv

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
            k_sub,
            q_scaled,
            _NT_DIMS,
            precision=lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )  # (sub, bq)
        p = jnp.where(m_sub, jnp.exp2(qk - lse), jnp.float32(0.0))
        dv_acc[rows, :] += jnp.dot(
            p.astype(do_blk.dtype),
            do_blk,
            precision=lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )
        dp = lax.dot_general(
            v_sub,
            do_blk,
            _NT_DIMS,
            precision=lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )  # (sub, bq)
        ds = p * (dp - di_row)
        dk_acc[rows, :] += jnp.dot(
            ds.astype(q_blk.dtype),
            q_blk,
            precision=lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )
        # Transpose in f32 (32-bit XLU transpose), then round for the MXU.
        dq_acc[...] += jnp.dot(
            ds.T.astype(k_sub.dtype),
            k_sub,
            precision=lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
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
) -> jax.Array:
  """The Pallas attention forward (`flash_fwd_pallas`) with flash backward.

  Args:
    q_grouped: Queries `[B, KH, G, T, D]` (`G` query heads per KV head).
    k_grouped: Keys `[B, KH, S, D]`.
    v_grouped: Values `[B, KH, S, D]`.
    mask_bts: Boolean attention mask `[B, T, S]` (causal, padding and segment
      masks already merged).
    scale: Softmax temperature (`1 / sqrt(D)`).
    kv_block_size: Fixed key block (`bkv_csz`).
    interpret: Run the Pallas kernels in interpret mode (CPU tests).
    bwd: `PALLAS_BWD` (or `None` to read `ATTENTION_BWD`).
    fwd: `PALLAS_FWD` (or `None` to read `ATTENTION_FWD`).

  Returns:
    `out[B, KH, G, T, D]` in the query dtype.

  Raises:
    ValueError: For an unknown `fwd` / `bwd`, or `RPA_FWD`.
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

  t = int(q_grouped.shape[3])

  @jax.custom_vjp
  def op(q, k, v, m):
    out, _, _ = flash_fwd_pallas(
        q,
        k,
        v,
        m,
        scale=scale,
        kv_block_size=kv_block_size,
        interpret=interpret,
    )
    return out

  def op_fwd(q, k, v, m):
    out, m_final, l_final = flash_fwd_pallas(
        q,
        k,
        v,
        m,
        scale=scale,
        kv_block_size=kv_block_size,
        interpret=interpret,
    )
    lse2 = row_lse2(m_final, l_final)[:, :, :, :t]
    return out, (q, k, v, out, lse2, m)

  def op_bwd(residual, d_out):
    q, k, v, out, lse2, m = residual
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
    return dq, dk, dv, None

  op.defvjp(op_fwd, op_bwd)
  return op(q_grouped, k_grouped, v_grouped, mask_bts)
