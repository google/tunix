"""Differentiable ragged-paged attention for the SHARED forward (P18.3a).

Why this exists
---------------
`ragged_paged_attention` (RPA v3) is inference-only: `jax.grad` through it dies in
`pallas_call.py:_pallas_call_jvp_rule -> NotImplementedError` (measured, P14.8.0).  But P18.1b /
P18.2 established that the engine's `run_model` is the forward we must share bitwise between
`old_logp` and `new_logp`.  So the trainer needs *that* forward to admit a gradient.

Design (carried over from P14.8, whose code was lost with the /tmp wipe; the reasoning is recorded
in phase14_shared_kernel.md "设计要点")
-----------------------------------------------------------------------------------------------
Split the requirement in two:
  * the FORWARD must be bitwise -- so the forward IS the kernel, unchanged.  Nothing is
    transcribed, so nothing can drift.
  * the BACKWARD only has to be a *correct* gradient -- SGD does not care about 1 ULP, and
    FlashAttention's own backward likewise recomputes with its own numerics.  So the backward is
    plain autodiff of a pure-JAX attention that computes the same mathematical function.
This is the same shape as `splash_attention_kernel`'s `custom_vjp/defvjp`.

Deliberate simplification vs P14.8
----------------------------------
P14.8's replica reproduced the kernel's blocked online-softmax *bitwise* (max|Δ| = 1 bf16 ULP),
because at that time the replica was also the forward.  Here the forward is the kernel, so the
replica only needs to be the same *function*.  A single full-softmax pass with a segmented causal
mask is therefore preferred: same mathematics, far fewer transcription traps (the recorded trap
list is long -- scale after the matmul, `mask_value=finfo.min` not -inf, cast-before-running-max,
`p` not cast to v.dtype, bf16 scratch accumulators...).  Simplicity is worth more than fidelity on
a path where fidelity buys nothing.

Scope: PREFILL ONLY (asserted, not assumed)
-------------------------------------------
Gradients are only ever needed for the scoring/training forward, which is a full prefill; decode
is never differentiated.  For a full prefill `kv_len == q_len`, so every key/value comes from the
`k`/`v` arguments and the paged cache is never read.  `replica` asserts this and refuses otherwise,
rather than silently computing something wrong for a decode batch.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _segment_ids_and_positions(cu_q_lens, n_tokens, num_seqs):
    """Map each padded token row to (sequence index, position within that sequence).

    `cu_q_lens` is the cumulative query length, e.g. [0, 160, 160, 160, ...] for one 160-token
    request padded out to max_num_reqs+1 entries.  `searchsorted(..., 'right') - 1` turns that
    into a segment id; rows past the end of the last real sequence land on `num_seqs` and are
    reported invalid.
    """
    idx = jnp.arange(n_tokens, dtype=jnp.int32)
    seg = jnp.searchsorted(cu_q_lens, idx, side="right").astype(jnp.int32) - 1
    # `num_seqs` may be a TRACED scalar (it is `distribution[2]` at the engine's call site), so
    # every use of it here is a value operation -- never a static slice bound or a shape.
    total = jnp.take(cu_q_lens, jnp.clip(num_seqs, 0, cu_q_lens.shape[0] - 1))
    valid = (idx < total) & (seg >= 0) & (seg < num_seqs)
    seg = jnp.where(valid, seg, cu_q_lens.shape[0])   # park invalid rows on an unused segment id
    pos = idx - jnp.take(cu_q_lens, jnp.clip(seg, 0, cu_q_lens.shape[0] - 1))
    return seg, pos, valid


def replica(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution,
            *, sm_scale, use_causal_mask=True, out_dtype=None, num_seqs=None,
            compute_dtype=jnp.float32):
    """Pure-JAX ragged causal attention over the *local* (per-shard) view.

    Shapes follow the kernel: q [T, n_q_heads, head_dim], k/v [T, n_kv_heads, head_dim].
    Only the arguments that carry values are used; `kv_cache` / `page_indices` are accepted so the
    signature matches the kernel's, and are unused because this is the prefill path (see module
    docstring).  `num_seqs` is `distribution[2]`; it may be a traced scalar -- every use of it is a
    value operation, never a static slice bound or a shape.
    """
    T, nq, hd = q.shape
    _, nkv, _ = k.shape
    assert nq % nkv == 0, f"GQA requires n_q_heads % n_kv_heads == 0, got {nq}/{nkv}"
    group = nq // nkv
    od = out_dtype if out_dtype is not None else q.dtype
    if num_seqs is None:
        raise ValueError("num_seqs is required; pass distribution[2] (traced is fine)")

    seg, pos, valid = _segment_ids_and_positions(cu_q_lens, T, num_seqs)

    # Prefill-only precondition, checked on the traced values rather than assumed.  For a full
    # prefill every sequence's kv_len equals its q_len, so nothing is read from the paged cache.
    # Computed over ALL padded request slots (static shapes) and masked down to the active ones, so
    # that a traced `num_seqs` never has to bound a slice.
    n_slots = min(kv_lens.shape[0], cu_q_lens.shape[0] - 1)
    q_lens_all = cu_q_lens[1:n_slots + 1] - cu_q_lens[:n_slots]
    active = jnp.arange(n_slots, dtype=jnp.int32) < num_seqs
    kv_left = jnp.maximum(kv_lens[:n_slots] - q_lens_all, 0)
    prefill_only = jnp.all(jnp.where(active, kv_left, 0) == 0)

    qc = q.astype(compute_dtype)
    kc = k.astype(compute_dtype)
    vc = v.astype(compute_dtype)

    # Segmented causal mask: same sequence, and the query position at or after the key position.
    same_seq = seg[:, None] == seg[None, :]
    causal = pos[:, None] >= pos[None, :] if use_causal_mask else jnp.ones((T, T), bool)
    keep = same_seq & causal & valid[:, None] & valid[None, :]

    # One [T, T] score matrix per query head.  T is the token bucket (256 in the P18 workload), so
    # this is small; for long contexts a blocked form would be needed (noted in phase18 as a
    # scale-up item, since memory here is O(T^2) rather than the kernel's O(T * block)).
    kg = jnp.repeat(kc, group, axis=1)             # [T, nq, hd] -- broadcast kv heads over groups
    vg = jnp.repeat(vc, group, axis=1)
    s = jnp.einsum("qhd,khd->hqk", qc, kg, preferred_element_type=compute_dtype) * sm_scale
    neg = jnp.finfo(compute_dtype).min
    s = jnp.where(keep[None, :, :], s, neg)
    p = jax.nn.softmax(s, axis=-1)
    # Rows that are entirely masked (padding rows) produce a uniform-ish softmax over -inf; zero
    # them explicitly so padding contributes nothing to the output or to the gradient.
    p = jnp.where(valid[None, :, None], p, 0.0)
    o = jnp.einsum("hqk,khd->qhd", p, vg, preferred_element_type=compute_dtype)
    o = jnp.where(valid[:, None, None], o, 0.0)
    # Fold the precondition into the value so it cannot be optimised away: NaN out the result if a
    # decode/partial batch ever reaches here, which will surface loudly instead of silently.
    o = jnp.where(prefill_only, o, jnp.nan)
    return o.astype(od)


def make_diff_rpa(kernel_fn, *, sm_scale, num_seqs=None, use_causal_mask=True, out_dtype=None):
    """Wrap `kernel_fn` so the forward is the kernel and the backward is autodiff of `replica`.

    `kernel_fn(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution)` must return
    `(output, new_kv_cache)` -- the RPA v3 contract.  Only q/k/v receive gradients; the integer
    metadata and the cache are non-differentiable.
    """

    @jax.custom_vjp
    def diff_rpa(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution):
        return kernel_fn(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution)

    def _fwd(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution):
        out = diff_rpa(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution)
        # Residuals: q/k/v are needed to linearise the replica.  Note the kernel is jitted with
        # donate_argnames=("queries","keys","values","kv_cache") (kernel.py:1584), so this wrapper
        # MUST be used inside jax.jit -- in eager mode the donated buffers would be invalidated
        # while still being held as residuals (recorded in phase14 as a P14.8 leftover).
        return out, (q, k, v, kv_lens, cu_q_lens, distribution)

    def _bwd(res, g):
        q, k, v, kv_lens, cu_q_lens, distribution = res
        n_seqs = num_seqs if num_seqs is not None else distribution[2]
        g_out = g[0]                                # cotangent of `output`; g[1] is the cache
        def f(qq, kk, vv):
            return replica(qq, kk, vv, None, kv_lens, None, cu_q_lens, None,
                           sm_scale=sm_scale, use_causal_mask=use_causal_mask,
                           out_dtype=out_dtype, num_seqs=n_seqs)
        _, vjp = jax.vjp(f, q, k, v)
        dq, dk, dv = vjp(g_out.astype(q.dtype))
        return (dq, dk, dv, None, None, None, None, None)

    diff_rpa.defvjp(_fwd, _bwd)
    return diff_rpa
