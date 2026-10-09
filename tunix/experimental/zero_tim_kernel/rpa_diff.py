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
"""Differentiable ragged-paged attention for the SHARED forward (P18.3a).

Why this exists
---------------
`ragged_paged_attention` (RPA v3) is inference-only: `jax.grad` through it
dies in `pallas_call.py:_pallas_call_jvp_rule -> NotImplementedError`
(measured, P14.8.0).  But P18.1b / P18.2 established that the engine's
`run_model` is the forward we must share bitwise between `old_logp` and
`new_logp`.  So the trainer needs *that* forward to admit a gradient.

Design
------
Split the requirement in two:
  * the FORWARD must be bitwise -- so the forward IS the kernel, unchanged.
    Nothing is transcribed, so nothing can drift.
  * the BACKWARD only has to be a *correct* gradient -- SGD does not care
    about 1 ULP, and FlashAttention's own backward likewise recomputes with
    its own numerics.  So the backward is plain autodiff of a pure-JAX
    attention that computes the same mathematical function.
This is the same shape as `splash_attention_kernel`'s `custom_vjp/defvjp`.

Deliberate simplification vs P14.8
----------------------------------
P14.8's replica reproduced the kernel's blocked online-softmax *bitwise*
(max|Δ| = 1 bf16 ULP), because at that time the replica was also the forward.
Here the forward is the kernel, so the replica only needs to be the same
*function*.  A single full-softmax pass with a segmented causal mask is
therefore preferred: same mathematics, far fewer transcription traps (the
recorded trap list is long -- scale after the matmul, `mask_value=finfo.min`
not -inf, cast-before-running-max, `p` not cast to v.dtype, bf16 scratch
accumulators...).  Simplicity is worth more than fidelity on a path where
fidelity buys nothing.

Scope: PREFILL ONLY (asserted, not assumed)
-------------------------------------------
Gradients are only ever needed for the scoring/training forward, which is a
full prefill; decode is never differentiated.  For a full prefill
`kv_len == q_len`, so every key/value comes from the `k`/`v` arguments and the
paged cache is never read.  `replica` asserts this and refuses otherwise,
rather than silently computing something wrong for a decode batch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _segment_ids_and_positions(cu_q_lens, n_tokens, num_seqs):
  """Maps each padded token row to (sequence index, position in sequence).

  `cu_q_lens` is the cumulative query length, e.g. [0, 160, 160, 160, ...] for
  one 160-token request padded out to max_num_reqs+1 entries.
  `searchsorted(..., 'right') - 1` turns that into a segment id; rows past the
  end of the last real sequence land on `num_seqs` and are reported invalid.

  Args:
    cu_q_lens: Cumulative query lengths `int32[max_num_reqs + 1]`.
    n_tokens: Static number of (padded) token rows.
    num_seqs: Number of active sequences; may be a traced scalar.

  Returns:
    `(seg, pos, valid)`: per-row segment id, position within the segment and
    validity mask, each of shape `[n_tokens]`.
  """
  idx = jnp.arange(n_tokens, dtype=jnp.int32)
  seg = jnp.searchsorted(cu_q_lens, idx, side="right").astype(jnp.int32) - 1
  # `num_seqs` may be a TRACED scalar (it is `distribution[2]` at the engine's
  # call site), so every use of it here is a value operation -- never a static
  # slice bound or a shape.
  total = jnp.take(cu_q_lens, jnp.clip(num_seqs, 0, cu_q_lens.shape[0] - 1))
  valid = (idx < total) & (seg >= 0) & (seg < num_seqs)
  seg = jnp.where(
      valid, seg, cu_q_lens.shape[0]
  )  # park invalid rows on an unused segment id
  pos = idx - jnp.take(cu_q_lens, jnp.clip(seg, 0, cu_q_lens.shape[0] - 1))
  return seg, pos, valid


def replica(
    q,
    k,
    v,
    kv_cache,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    *,
    sm_scale,
    use_causal_mask=True,
    out_dtype=None,
    num_seqs=None,
    compute_dtype=jnp.float32,
):
  """Pure-JAX ragged causal attention over the *local* (per-shard) view.

  Shapes follow the kernel: q [T, n_q_heads, head_dim], k/v [T, n_kv_heads,
  head_dim].  Only the arguments that carry values are used; `kv_cache` /
  `page_indices` are accepted so the signature matches the kernel's, and are
  unused because this is the prefill path (see module docstring).  `num_seqs`
  is `distribution[2]`; it may be a traced scalar -- every use of it is a value
  operation, never a static slice bound or a shape.

  Args:
    q: Queries `[T, n_q_heads, head_dim]`.
    k: Keys `[T, n_kv_heads, head_dim]`.
    v: Values `[T, n_kv_heads, head_dim]`.
    kv_cache: Unused (kernel-signature compatibility; prefill never reads it).
    kv_lens: Per-sequence KV lengths `int32[max_num_seqs]`.
    page_indices: Unused (kernel-signature compatibility).
    cu_q_lens: Cumulative query lengths `int32[max_num_seqs + 1]`.
    distribution: Unused; pass `distribution[2]` as `num_seqs` instead.
    sm_scale: Softmax scale applied to the scores after the matmul.
    use_causal_mask: Whether to apply the (segmented) causal mask.
    out_dtype: Output dtype; defaults to `q.dtype`.
    num_seqs: Number of active sequences (`distribution[2]`); may be traced.
    compute_dtype: dtype of the score / softmax computation.

  Returns:
    Attention output `[T, n_q_heads, head_dim]` in `out_dtype`; NaN if the
    prefill-only precondition (`kv_len == q_len` for every active sequence)
    is violated.
  """
  del kv_cache, page_indices, distribution  # Prefill path: never read.
  n_tokens, nq, _ = q.shape
  _, nkv, _ = k.shape
  assert (
      nq % nkv == 0
  ), f"GQA requires n_q_heads % n_kv_heads == 0, got {nq}/{nkv}"
  group = nq // nkv
  od = out_dtype if out_dtype is not None else q.dtype
  if num_seqs is None:
    raise ValueError(
        "num_seqs is required; pass distribution[2] (traced is fine)"
    )

  seg, pos, valid = _segment_ids_and_positions(cu_q_lens, n_tokens, num_seqs)

  # Prefill-only precondition, checked on the traced values rather than
  # assumed.  For a full prefill every sequence's kv_len equals its q_len, so
  # nothing is read from the paged cache.  Computed over ALL padded request
  # slots (static shapes) and masked down to the active ones, so that a traced
  # `num_seqs` never has to bound a slice.
  n_slots = min(kv_lens.shape[0], cu_q_lens.shape[0] - 1)
  q_lens_all = cu_q_lens[1 : n_slots + 1] - cu_q_lens[:n_slots]
  active = jnp.arange(n_slots, dtype=jnp.int32) < num_seqs
  kv_left = jnp.maximum(kv_lens[:n_slots] - q_lens_all, 0)
  prefill_only = jnp.all(jnp.where(active, kv_left, 0) == 0)

  qc = q.astype(compute_dtype)
  kc = k.astype(compute_dtype)
  vc = v.astype(compute_dtype)

  # Segmented causal mask: same sequence, and the query position at or after
  # the key position.
  same_seq = seg[:, None] == seg[None, :]
  causal = (
      pos[:, None] >= pos[None, :]
      if use_causal_mask
      else jnp.ones((n_tokens, n_tokens), bool)
  )
  keep = same_seq & causal & valid[:, None] & valid[None, :]

  # One [T, T] score matrix per query head.  T is the token bucket (256 in the
  # P18 workload), so this is small; for long contexts a blocked form would be
  # needed (noted in phase18 as a scale-up item, since memory here is O(T^2)
  # rather than the kernel's O(T * block)).
  kg = jnp.repeat(
      kc, group, axis=1
  )  # [T, nq, hd] -- broadcast kv heads over groups
  vg = jnp.repeat(vc, group, axis=1)
  s = (
      jnp.einsum("qhd,khd->hqk", qc, kg, preferred_element_type=compute_dtype)
      * sm_scale
  )
  neg = jnp.finfo(compute_dtype).min
  s = jnp.where(keep[None, :, :], s, neg)
  p = jax.nn.softmax(s, axis=-1)
  # Rows that are entirely masked (padding rows) produce a uniform-ish softmax
  # over -inf; zero them explicitly so padding contributes nothing to the
  # output or to the gradient.
  p = jnp.where(valid[None, :, None], p, 0.0)
  o = jnp.einsum("hqk,khd->qhd", p, vg, preferred_element_type=compute_dtype)
  o = jnp.where(valid[:, None, None], o, 0.0)
  # Fold the precondition into the value so it cannot be optimised away: NaN
  # out the result if a decode/partial batch ever reaches here, which will
  # surface loudly instead of silently.
  o = jnp.where(prefill_only, o, jnp.nan)
  return o.astype(od)


def _dtype_packing(dtype) -> int:
  """Returns the number of elements packed per 32-bit word (RPA v3 contract)."""
  itemsize = jnp.dtype(dtype).itemsize
  if itemsize <= 0 or 4 % itemsize != 0:
    raise ValueError(f"Unsupported KV dtype for 32-bit packing: {dtype}")
  return 4 // itemsize


def scratch_kv_cache_shape(
    k,
    *,
    page_size: int = 16,
    num_pages: int | None = None,
) -> tuple[int, int, int, int, int]:
  """Returns the minimal valid 5D RPA v3 scratch `kv_cache` shape for prefill.

  Matches `v3/kernel.py:get_kv_cache_shape`:
    `(total_num_pages, page_size, cdiv(actual_num_kv_heads * 2, kv_packing),
      kv_packing, align_to(actual_head_dim, 128))`

  Without modifying `v3/kernel.py` (`update_kv_cache=True`), a full-prefill
  Learner call (`kv_len == q_len <= max_num_tokens`) reads 0 tokens from
  `kv_cache` (`kv_left_frm_cache == 0`) and only writes the current layer's K/V
  blocks into `kv_cache` via DMA before `updated_kv_cache` is discarded. Sizing
  `total_num_pages` to `max(2, cdiv(max_num_tokens, page_size))` ensures every
  `_update_kv_cache` DMA page and `wait_update_kv_cache`'s
  `cache_hbm_ref.at[pl.ds(0, update_sz)]` slice stay strictly in bounds while
  reusing a single tiny buffer across all sequences and all Transformer layers.

  Args:
    k: Key tensor `[max_num_tokens, num_kv_heads, head_dim]`; only its shape
      and dtype are used.
    page_size: Tokens per KV-cache page (must be divisible by `kv_packing`).
    num_pages: Optional physical page count; defaults to
      `max(2,cdiv(max_num_tokens,page_size))`.
  """
  if len(k.shape) != 3:
    raise ValueError(
        f"Expected 3D k [max_num_tokens, num_kv_heads, head_dim], got {k.shape}"
    )
  max_num_tokens, actual_num_kv_heads, actual_head_dim = map(int, k.shape)
  if actual_num_kv_heads <= 0 or actual_head_dim <= 0:
    raise ValueError(f"Invalid k shape: {k.shape}")
  kv_packing = _dtype_packing(k.dtype)
  if page_size <= 0 or page_size % kv_packing != 0:
    raise ValueError(
        f"page_size={page_size} must be positive and divisible by"
        f" kv_packing={kv_packing}"
    )
  default_pages = max(2, (max_num_tokens + page_size - 1) // page_size)
  total_num_pages = default_pages if num_pages is None else int(num_pages)
  if total_num_pages <= 0:
    raise ValueError(f"num_pages must be positive, got {total_num_pages}")
  num_kv_heads_x2_per_packing = (
      actual_num_kv_heads * 2 + kv_packing - 1
  ) // kv_packing
  head_dim_aligned = ((actual_head_dim + 127) // 128) * 128
  return (
      total_num_pages,
      page_size,
      num_kv_heads_x2_per_packing,
      kv_packing,
      head_dim_aligned,
  )


def allocate_scratch_kv_cache(
    k: jax.Array,
    kv_lens: jax.Array,
    *,
    page_size: int = 16,
    num_pages: int | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Allocates a minimal layer-shared `(scratch_kv_cache, scratch_page_indices)`.

  Args:
    k: Key tensor `[max_num_tokens, num_kv_heads, head_dim]`.
    kv_lens: Sequence KV lengths `int32[max_num_seqs]`.
    page_size: Tokens per KV-cache page (must be divisible by `kv_packing`).
    num_pages: Optional physical page count; defaults to
      `max(2,cdiv(max_num_tokens,page_size))`.

  Returns:
    Tuple `(scratch_kv_cache, scratch_page_indices)` satisfying all static and
    dynamic validation checks of `v3/kernel.py:ragged_paged_attention`.
  """
  if len(kv_lens.shape) != 1 or kv_lens.dtype != jnp.int32:
    raise ValueError(
        "Expected 1D int32 kv_lens, got"
        f" shape={kv_lens.shape}, dtype={kv_lens.dtype}"
    )
  cache_shape = scratch_kv_cache_shape(
      k, page_size=page_size, num_pages=num_pages
  )
  total_num_pages = cache_shape[0]
  max_num_tokens = int(k.shape[0])
  max_num_seqs = int(kv_lens.shape[0])
  pages_per_seq = max(
      2, (max_num_tokens + page_size - 1) // page_size, total_num_pages
  )
  seq_pages = jnp.arange(pages_per_seq, dtype=jnp.int32) % total_num_pages
  scratch_page_indices = jnp.tile(seq_pages, max_num_seqs)
  scratch_kv_cache = jnp.zeros(cache_shape, dtype=k.dtype)
  return scratch_kv_cache, scratch_page_indices


def bind_scratch_kv_cache(
    kernel_fn,
    *,
    page_size: int = 16,
    num_pages: int | None = None,
    scratch_kv_cache: jax.Array | None = None,
    scratch_page_indices: jax.Array | None = None,
):
  """Wraps `kernel_fn` to bind a minimal layer-shared scratch KV cache.

  Supports two calling conventions without modifying `v3/kernel.py`:
    1. 8-arg RPA v3 signature:
       `bound(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution)
       -> (output, new_kv_cache)`
       Whenever `kv_cache is None` (or `page_indices is None`), automatically
       substitutes the minimal scratch buffer.
    2. 6-arg cacheless Learner prefill signature:
       `bound(q, k, v, kv_lens, cu_q_lens, distribution) -> output`
       Automatically binds the minimal scratch `kv_cache` and `page_indices`
       and discards `new_kv_cache` so the scratch buffer's lifetime ends
       immediately after each layer.

  Args:
    kernel_fn: The 8-arg RPA v3 kernel entry point.
    page_size: Tokens per KV-cache page of the auto-allocated scratch.
    num_pages: Optional physical page count of the auto-allocated scratch.
    scratch_kv_cache: Optional pre-allocated scratch cache to bind instead of
      allocating one.
    scratch_page_indices: Optional pre-allocated scratch page indices.

  Returns:
    `bound_rpa`, accepting either calling convention described above.
  """

  def _resolve_scratch(k, kv_lens, kv_cache, page_indices):
    if kv_cache is not None and page_indices is not None:
      return kv_cache, page_indices
    base_cache = scratch_kv_cache
    base_pages = scratch_page_indices
    if base_cache is None or base_pages is None:
      alloc_cache, alloc_pages = allocate_scratch_kv_cache(
          k, kv_lens, page_size=page_size, num_pages=num_pages
      )
      if base_cache is None:
        base_cache = alloc_cache
      if base_pages is None:
        base_pages = alloc_pages
    return (
        base_cache if kv_cache is None else kv_cache,
        base_pages if page_indices is None else page_indices,
    )

  def bound_rpa(*args):
    if len(args) == 6:
      q, k, v, kv_lens, cu_q_lens, distribution = args
      cache, pages = _resolve_scratch(k, kv_lens, None, None)
      out, _ = kernel_fn(
          q, k, v, cache, kv_lens, pages, cu_q_lens, distribution
      )
      return out
    if len(args) == 8:
      q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution = args
      cache, pages = _resolve_scratch(k, kv_lens, kv_cache, page_indices)
      return kernel_fn(q, k, v, cache, kv_lens, pages, cu_q_lens, distribution)
    raise TypeError(
        "bound_rpa expects either 6 args (q, k, v, kv_lens, cu_q_lens,"
        " distribution) or 8 args (q, k, v, kv_cache, kv_lens, page_indices,"
        f" cu_q_lens, distribution); got {len(args)} args"
    )

  return bound_rpa


def make_diff_rpa(
    kernel_fn,
    *,
    sm_scale,
    num_seqs=None,
    use_causal_mask=True,
    out_dtype=None,
    page_size: int = 16,
    num_scratch_pages: int | None = None,
    scratch_kv_cache: jax.Array | None = None,
    scratch_page_indices: jax.Array | None = None,
):
  """Wraps `kernel_fn`: forward is the kernel, backward is autodiff of `replica`.

  `kernel_fn(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution)`
  must return `(output, new_kv_cache)` -- the RPA v3 contract.  Only q/k/v
  receive gradients; the integer metadata and the cache are
  non-differentiable.

  For Learner prefill (`q_len == kv_len`), callers may either:
    * pass `kv_cache=None` and `page_indices=None` in the 8-arg signature, or
    * call the returned function with the 6-arg cacheless signature
      `diff_rpa(q, k, v, kv_lens, cu_q_lens, distribution) -> output`,
  which automatically binds a minimal layer-shared scratch `kv_cache` in the
  forward pass without modifying `v3/kernel.py` and without saving any cache in
  autodiff residuals.

  Args:
    kernel_fn: The 8-arg RPA v3 kernel entry point.
    sm_scale: Softmax scale used by the replica backward.
    num_seqs: Static number of active sequences; defaults to
      `distribution[2]` (traced) at call time.
    use_causal_mask: Whether the replica applies the segmented causal mask.
    out_dtype: Output dtype of the replica; defaults to `q.dtype`.
    page_size: Tokens per KV-cache page of the auto-allocated scratch.
    num_scratch_pages: Optional physical page count of the scratch cache.
    scratch_kv_cache: Optional pre-allocated scratch cache.
    scratch_page_indices: Optional pre-allocated scratch page indices.

  Returns:
    `diff_rpa`, a differentiable wrapper accepting either calling convention.
  """
  bound_kernel = bind_scratch_kv_cache(
      kernel_fn,
      page_size=page_size,
      num_pages=num_scratch_pages,
      scratch_kv_cache=scratch_kv_cache,
      scratch_page_indices=scratch_page_indices,
  )

  @jax.custom_vjp
  def diff_rpa_8(
      q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution
  ):
    return bound_kernel(
        q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution
    )

  def _fwd(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution):
    out = diff_rpa_8(
        q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution
    )
    # Residuals: q/k/v are needed to linearise the replica.  Note the kernel is
    # jitted with donate_argnames=("queries","keys","values","kv_cache"), so
    # this wrapper MUST be used inside jax.jit -- in eager mode the donated
    # buffers would be invalidated while still being held as residuals.
    return out, (q, k, v, kv_lens, cu_q_lens, distribution)

  def _bwd(res, g):
    q, k, v, kv_lens, cu_q_lens, distribution = res
    n_seqs = num_seqs if num_seqs is not None else distribution[2]
    g_out = g[0]  # cotangent of `output`; g[1] is the cache

    def f(qq, kk, vv):
      return replica(
          qq,
          kk,
          vv,
          None,
          kv_lens,
          None,
          cu_q_lens,
          None,
          sm_scale=sm_scale,
          use_causal_mask=use_causal_mask,
          out_dtype=out_dtype,
          num_seqs=n_seqs,
      )

    _, vjp = jax.vjp(f, q, k, v)
    dq, dk, dv = vjp(g_out.astype(q.dtype))
    return (dq, dk, dv, None, None, None, None, None)

  diff_rpa_8.defvjp(_fwd, _bwd)

  def diff_rpa(*args):
    if len(args) == 6:
      q, k, v, kv_lens, cu_q_lens, distribution = args
      out, _ = diff_rpa_8(q, k, v, None, kv_lens, None, cu_q_lens, distribution)
      return out
    if len(args) == 8:
      return diff_rpa_8(*args)
    raise TypeError(
        "diff_rpa expects either 6 args (q, k, v, kv_lens, cu_q_lens,"
        " distribution) or 8 args (q, k, v, kv_cache, kv_lens, page_indices,"
        f" cu_q_lens, distribution); got {len(args)} args"
    )

  return diff_rpa
