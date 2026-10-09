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
"""Chunked-cache differentiable RPA wrapper -- the F_canonical building block.

Extends rpa_diff's design to the cache-reading case (kv_len > q_len),
single-sequence mode:
  forward  = kernel_fn verbatim (real RPA v3 contract: attend + write the
             chunk's kv into the cache)
  backward = extended replica VJP (G2x2-proven math) + g[1] cache-cotangent
             routing (G2x3-proven): old cache slots pass through, this
             chunk's slots fold into dk/dv, context slots receive the
             attention cotangents dck/dcv.
All shapes static; traced kv_len/q_len handled via masks (rpa_diff's
discipline).

Cache layout: ``kv_cache[num_pages, page_size, num_kv_heads, 2, head_dim]``,
i.e. the RPA v3 combined-KV layout at bf16 packing 2 (index 0 of the packing
axis is K, index 1 is V).
"""

import os

import jax
import jax.numpy as jnp

# Number of sequences the ragged adapter's backward unrolls (see _bwd below).
MAX_SEQS_ENV = "CANON_VJP2_MAX_SEQS"


def make_chunked_replica(
    *, sm_scale, page_size, out_dtype=None, compute_dtype=jnp.float32
):
  """Returns the pure-JAX extended replica the chunked VJPs differentiate.

  Args:
    sm_scale: Softmax scale applied after the QK matmul.
    page_size: Tokens per KV-cache page.
    out_dtype: Output dtype of the replica; defaults to ``q.dtype``.
    compute_dtype: Dtype the replica computes in.

  Returns:
    ``replica(q, k, v, kv_cache, kv_len, q_len, page_indices) -> output``.
  """

  def _replica(q, k, v, kv_cache, kv_len, q_len, page_indices):
    """Extended replica: q attends [cache context ; current k/v], causal.

    q [T, nq, hd]; k/v [T, nkv, hd]; kv_cache [np, PAGE, nkv, 2, hd].  Causal
    masking uses absolute positions.  Heads are derived PER CALL from operand
    shapes (shard_map passes LOCAL shards, so the factory's global head counts
    must not be baked in).

    Args:
      q: Queries `[T, nq, hd]` of the current chunk.
      k: Keys `[T, nkv, hd]` of the current chunk.
      v: Values `[T, nkv, hd]` of the current chunk.
      kv_cache: Paged cache `[np, PAGE, nkv, 2, hd]` holding the context.
      kv_len: Total KV length (context + chunk); may be traced.
      q_len: Number of valid query rows in the chunk; may be traced.
      page_indices: This sequence's page table `int32[n_tbl]`.

    Returns:
      Attention output `[T, nq, hd]` in `out_dtype` (defaults to `q.dtype`).
    """
    T = q.shape[0]  # pylint: disable=invalid-name
    hd = q.shape[-1]
    nkv_l = kv_cache.shape[2]
    grp_l = q.shape[-2] // nkv_l
    n_tbl = page_indices.shape[0]
    S = n_tbl * page_size  # static max cache span  # pylint: disable=invalid-name
    ck = kv_cache[page_indices, :, :, 0].reshape(S, nkv_l, hd)
    cv = kv_cache[page_indices, :, :, 1].reshape(S, nkv_l, hd)
    ctx = kv_len - q_len
    qc = q.astype(compute_dtype)
    kall = jnp.concatenate([ck, k]).astype(compute_dtype)  # [S+T, nkv, hd]
    vall = jnp.concatenate([cv, v]).astype(compute_dtype)
    kg = jnp.repeat(kall, grp_l, axis=1)
    vg = jnp.repeat(vall, grp_l, axis=1)
    s = (
        jnp.einsum("qhd,shd->hqs", qc, kg, preferred_element_type=compute_dtype)
        * sm_scale
    )
    col_pos = jnp.concatenate([jnp.arange(S), ctx + jnp.arange(T)])
    col_ok = jnp.concatenate([jnp.arange(S) < ctx, jnp.arange(T) < q_len])
    row_pos = ctx + jnp.arange(T)
    row_ok = jnp.arange(T) < q_len
    keep = (
        (row_pos[:, None] >= col_pos[None, :])
        & col_ok[None, :]
        & row_ok[:, None]
    )
    neg = jnp.finfo(compute_dtype).min
    s = jnp.where(keep[None], s, neg)
    p = jax.nn.softmax(s, axis=-1)
    p = jnp.where(row_ok[None, :, None], p, 0.0)
    o = jnp.einsum("hqs,shd->qhd", p, vg, preferred_element_type=compute_dtype)
    o = jnp.where(row_ok[:, None, None], o, 0.0)
    return o.astype(out_dtype if out_dtype is not None else q.dtype)

  return _replica


def make_diff_rpa_chunked(
    kernel_fn,
    *,
    sm_scale,
    page_size,
    num_q_heads,
    num_kv_heads,
    out_dtype=None,
    compute_dtype=jnp.float32,
):
  """Wraps a single-sequence chunked RPA kernel with a replica-based VJP.

  Args:
    kernel_fn: ``kernel_fn(q, k, v, kv_cache, kv_len, q_len, page_indices)``
      returning ``(output, new_kv_cache)``; used verbatim as the forward.
    sm_scale: Softmax scale applied after the QK matmul.
    page_size: Tokens per KV-cache page.
    num_q_heads: Global number of query heads (documentation only; the replica
      derives local head counts from operand shapes).
    num_kv_heads: Global number of KV heads (see ``num_q_heads``).
    out_dtype: Output dtype of the replica; defaults to ``q.dtype``.
    compute_dtype: Dtype the replica computes in.

  Returns:
    A ``jax.custom_vjp`` function with the kernel's signature.  Its backward
    differentiates ``make_chunked_replica(...)``.
  """
  del num_q_heads, num_kv_heads  # Local head counts come from operand shapes.
  replica = make_chunked_replica(
      sm_scale=sm_scale,
      page_size=page_size,
      out_dtype=out_dtype,
      compute_dtype=compute_dtype,
  )

  @jax.custom_vjp
  def diff_rpa(q, k, v, kv_cache, kv_len, q_len, page_indices):
    return kernel_fn(q, k, v, kv_cache, kv_len, q_len, page_indices)

  def _fwd(q, k, v, kv_cache, kv_len, q_len, page_indices):
    out = diff_rpa(q, k, v, kv_cache, kv_len, q_len, page_indices)
    return out, (q, k, v, kv_cache, kv_len, q_len, page_indices)

  def _bwd(res, g):
    q, k, v, kv_cache, kv_len, q_len, page_indices = res
    g_out, g_cache = g
    T = q.shape[0]  # pylint: disable=invalid-name
    ctx = kv_len - q_len

    def f(q_, k_, v_, cache_):
      return replica(q_, k_, v_, cache_, kv_len, q_len, page_indices)

    _, vjp = jax.vjp(f, q, k, v, kv_cache)
    dq, dk, dv, dcache_attn = vjp(g_out.astype(q.dtype))
    # dcache_attn already lands on context slots in PAGED layout (replica
    # gathers via jnp indexing => JAX derives the scatter transpose) -- G2x2
    # form A ≡ form B proven.

    # g[1] routing (G2x3): this chunk's written slots fold into dk/dv; the rest
    # passes through to the incoming cache; written slots zeroed in the
    # pass-through.
    pos = ctx + jnp.arange(T)  # logical pos of chunk rows
    pg = page_indices[pos // page_size]
    off = pos % page_size
    row_ok = jnp.arange(T) < q_len
    gk = g_cache[pg, off, :, 0]
    gv = g_cache[pg, off, :, 1]
    zero = jnp.zeros_like(gk)
    dk = dk + jnp.where(row_ok[:, None, None], gk, zero)
    dv = dv + jnp.where(row_ok[:, None, None], gv, zero)
    dcache = g_cache
    dcache = dcache.at[pg, off, :, 0].set(
        jnp.where(row_ok[:, None, None], zero, gk)
    )
    dcache = dcache.at[pg, off, :, 1].set(
        jnp.where(row_ok[:, None, None], zero, gv)
    )
    dcache = dcache + dcache_attn
    return dq, dk, dv, dcache, None, None, None

  diff_rpa.defvjp(_fwd, _bwd)
  return diff_rpa


def make_diff_rpa_chunked_ragged(
    kernel_fn,
    *,
    sm_scale,
    page_size,
    num_q_heads,
    num_kv_heads,
    out_dtype=None,
    compute_dtype=jnp.float32,
    max_q_len: int | None = None,
):
  """Thin ragged adapter: full RPA v3 signature, single-sequence semantics.

  Forward calls kernel_fn VERBATIM with the original args (numerics cannot
  move).  Backward extracts per-sequence scalars and reuses the gate-verified
  chunked replica + g[1] routing for the first ``CANON_VJP2_MAX_SEQS`` (default
  1) sequences.  Metadata cotangents are None; kv_cache receives dcache.

  Args:
    kernel_fn: The RPA v3 kernel, ``kernel_fn(q, k, v, kv_cache, kv_lens,
      page_indices, cu_q_lens, distribution) -> (output, new_kv_cache)``.
    sm_scale: Softmax scale applied after the QK matmul.
    page_size: Tokens per KV-cache page.
    num_q_heads: Global number of query heads (documentation only).
    num_kv_heads: Global number of KV heads (documentation only).
    out_dtype: Output dtype of the replica; defaults to ``q.dtype``.
    compute_dtype: Dtype the replica computes in.
    max_q_len: Static upper bound of one sequence's query rows (``cu_q_lens[i +
      1] - cu_q_lens[i]``).  Each unrolled sequence is differentiated on a
      window of this many rows starting at its first row, so the replica's score
      matrix is ``[heads, max_q_len, cache + max_q_len]`` per sequence instead
      of ``[heads, T, cache + T]`` for the whole flattened buffer. ``None`` uses
      the full buffer width ``T`` (always sufficient).  Rows a sequence has
      beyond the bound would silently get no gradient, so pass the exact
      per-sequence length when it is static (the bench does).

  Returns:
    A ``jax.custom_vjp`` function with the RPA v3 signature.
  """
  del num_q_heads, num_kv_heads  # Local head counts come from operand shapes.
  if max_q_len is not None and max_q_len < 1:
    raise ValueError(f"max_q_len must be >= 1, got {max_q_len}")
  replica = make_chunked_replica(
      sm_scale=sm_scale,
      page_size=page_size,
      out_dtype=out_dtype,
      compute_dtype=compute_dtype,
  )

  @jax.custom_vjp
  def diff_rpa(
      q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution
  ):
    return kernel_fn(
        q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution
    )

  def _fwd(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution):
    out = diff_rpa(
        q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution
    )
    return out, (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens)

  def _bwd(res, g):
    q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens = res
    g_out, g_cache = g[0], g[1]
    n_seqs_max = kv_lens.shape[0]
    bpr = page_indices.shape[0] // n_seqs_max
    T = q.shape[0]  # pylint: disable=invalid-name
    # Static window of one sequence's rows (F1b: differentiate each sequence
    # on its own rows, not on the whole flattened buffer).
    L = T if max_q_len is None else min(int(max_q_len), T)  # pylint: disable=invalid-name
    # Event 21 (root cause of the compile stall): defaulting to n_seqs_max=64
    # expands backward to 64 sequences x 64 layers = 4096 replica VJPs. Trace
    # time grew from 0.014s to 151s and compilation did not converge. Default
    # to one sequence; callers must explicitly set CANON_VJP2_MAX_SEQS for
    # multi-sequence workloads.
    n_active = int(os.environ.get(MAX_SEQS_ENV, "1"))
    n_active = min(n_active, n_seqs_max)

    dq = jnp.zeros_like(q)
    dk = jnp.zeros_like(k)
    dv = jnp.zeros_like(v)
    dcache_attn = jnp.zeros_like(kv_cache)
    # Cache slots the unrolled sequences overwrote with their own k/v: their
    # cotangent routes into dk/dv and must not pass through to the input cache.
    # The mask is built per slot of each sequence's page table and merged with
    # a scatter-max, which is order independent, so neither a padded (clamped)
    # row nor a duplicated page-table entry can undo the zeroing (F1a).
    written = jnp.zeros(kv_cache.shape[:2], jnp.bool_)
    slot_pos = jnp.arange(bpr * page_size, dtype=jnp.int32).reshape(
        bpr, page_size
    )
    rows = jnp.arange(L)
    local_table = jnp.arange(bpr, dtype=jnp.int32)

    # Each sequence has its own kv_len, query interval, and page table.
    # Statically unroll n_active iterations; shapes remain static and traced
    # scalars participate only in value computations.
    for i in range(n_active):
      kv_len_i = kv_lens[i]
      q0_i = cu_q_lens[i]
      q1_i = cu_q_lens[i + 1]
      q_len_i = q1_i - q0_i
      ctx_i = kv_len_i - q_len_i
      tbl_i = jax.lax.dynamic_slice(page_indices, (i * bpr,), (bpr,))
      # Window rows [q0_i, q0_i + L) gathered to [0, L); rows past q_len_i
      # are masked (and gather row 0, in bounds by construction).
      src_ok = rows < q_len_i
      gidx = jnp.where(src_ok, jnp.clip(q0_i + rows, 0, T - 1), 0)
      qi = jnp.where(src_ok[:, None, None], q[gidx], 0)
      ki = jnp.where(src_ok[:, None, None], k[gidx], 0)
      vi = jnp.where(src_ok[:, None, None], v[gidx], 0)
      goi = jnp.where(src_ok[:, None, None], g_out[gidx], 0)
      # Only this sequence's pages take part in its attention: slice the cache
      # to its page table and let the replica index the slice.
      cache_i = kv_cache[tbl_i]

      def f_i(q_, k_, v_, cache_, kv_len_i=kv_len_i, q_len_i=q_len_i):
        return replica(q_, k_, v_, cache_, kv_len_i, q_len_i, local_table)

      _, vjp_i = jax.vjp(f_i, qi, ki, vi, cache_i)
      dqi, dki, dvi, dca_i = vjp_i(goi.astype(q.dtype))
      # Route g[1]: the slots this sequence wrote hold its k/v, so their cache
      # cotangent is a k/v cotangent.  Valid rows index inside the sequence's
      # table (kv_len_i <= bpr * page_size); masked rows are clamped and
      # discarded.
      pos_i = ctx_i + rows
      pg_i = tbl_i[jnp.minimum(pos_i // page_size, bpr - 1)]
      off_i = pos_i % page_size
      gk_i = g_cache[pg_i, off_i, :, 0]
      gv_i = g_cache[pg_i, off_i, :, 1]
      # Move cotangents back to their original rows (masked rows add zeros).
      put = src_ok[:, None, None]
      dq = dq.at[gidx].add(jnp.where(put, dqi, 0))
      dk = dk.at[gidx].add(jnp.where(put, dki + gk_i, 0))
      dv = dv.at[gidx].add(jnp.where(put, dvi + gv_i, 0))
      dcache_attn = dcache_attn.at[tbl_i].add(dca_i)
      written_i = (slot_pos >= ctx_i) & (slot_pos < kv_len_i)
      written = written.at[tbl_i].max(written_i)
    dcache_pass = jnp.where(written[:, :, None, None, None], 0, g_cache)
    dcache = dcache_pass + dcache_attn
    return dq, dk, dv, dcache, None, None, None, None

  diff_rpa.defvjp(_fwd, _bwd)
  return diff_rpa
