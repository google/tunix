"""Chunked-cache differentiable RPA wrapper -- the F_canonical building block.

Extends rpa_diff's design to the cache-reading case (kv_len > q_len), single-sequence mode:
  forward  = kernel_fn verbatim (real RPA v3 contract: attend + write chunk kv into cache)
  backward = extended replica VJP (G2x2-proven math) + g[1] cache-cotangent routing
             (G2x3-proven): old cache slots pass through, this chunk's slots fold into dk/dv,
             context slots receive the attention cotangents dck/dcv.
All shapes static; traced kv_len/q_len handled via masks (rpa_diff's discipline).
Self-test: `python3 rpa_diff_chunked.py` runs fp64 chain-vs-oracle gates on a pure-JAX
reference kernel implementing the same contract.
"""
import os

import jax
import jax.numpy as jnp


def make_diff_rpa_chunked(kernel_fn, *, sm_scale, page_size, num_q_heads, num_kv_heads,
                          out_dtype=None, compute_dtype=jnp.float32):
    group = num_q_heads // num_kv_heads

    def _positions(pages_w, kv_len, page_indices):
        # logical position of cache slot (p, o): base[p] + o, base via inverse of page_indices
        npages = pages_w
        base = jnp.full((npages,), jnp.iinfo(jnp.int32).max // 2, jnp.int32)
        idx = jnp.arange(page_indices.shape[0], dtype=jnp.int32)
        base = base.at[page_indices].set(idx * page_size)
        return base                                            # [npages]

    def _replica(q, k, v, kv_cache, kv_len, q_len, page_indices):
        """Extended replica: q attends [cache context ; current k/v], absolute-pos causal.
        q [T, nq, hd]; k/v [T, nkv, hd]; kv_cache [np, PAGE, 2, nkv, hd].
        Heads derived PER CALL from operand shapes (shard_map passes LOCAL shards, so the
        factory's global head counts must not be baked in)."""
        T = q.shape[0]
        hd = q.shape[-1]
        nkv_l = kv_cache.shape[2]
        grp_l = q.shape[-2] // nkv_l
        n_tbl = page_indices.shape[0]
        S = n_tbl * page_size                                  # static max cache span
        ck = kv_cache[page_indices, :, :, 0].reshape(S, nkv_l, hd)
        cv = kv_cache[page_indices, :, :, 1].reshape(S, nkv_l, hd)
        ctx = kv_len - q_len
        qc = q.astype(compute_dtype)
        kall = jnp.concatenate([ck, k]).astype(compute_dtype)  # [S+T, nkv, hd]
        vall = jnp.concatenate([cv, v]).astype(compute_dtype)
        kg = jnp.repeat(kall, grp_l, axis=1)
        vg = jnp.repeat(vall, grp_l, axis=1)
        s = jnp.einsum("qhd,shd->hqs", qc, kg,
                       preferred_element_type=compute_dtype) * sm_scale
        col_pos = jnp.concatenate([jnp.arange(S), ctx + jnp.arange(T)])
        col_ok = jnp.concatenate([jnp.arange(S) < ctx, jnp.arange(T) < q_len])
        row_pos = ctx + jnp.arange(T)
        row_ok = jnp.arange(T) < q_len
        keep = (row_pos[:, None] >= col_pos[None, :]) & col_ok[None, :] & row_ok[:, None]
        neg = jnp.finfo(compute_dtype).min
        s = jnp.where(keep[None], s, neg)
        p = jax.nn.softmax(s, axis=-1)
        p = jnp.where(row_ok[None, :, None], p, 0.0)
        o = jnp.einsum("hqs,shd->qhd", p, vg, preferred_element_type=compute_dtype)
        o = jnp.where(row_ok[:, None, None], o, 0.0)
        return o.astype(out_dtype if out_dtype is not None else q.dtype)

    @jax.custom_vjp
    def diff_rpa(q, k, v, kv_cache, kv_len, q_len, page_indices):
        return kernel_fn(q, k, v, kv_cache, kv_len, q_len, page_indices)

    def _fwd(q, k, v, kv_cache, kv_len, q_len, page_indices):
        out = diff_rpa(q, k, v, kv_cache, kv_len, q_len, page_indices)
        return out, (q, k, v, kv_cache, kv_len, q_len, page_indices)

    def _bwd(res, g):
        q, k, v, kv_cache, kv_len, q_len, page_indices = res
        g_out, g_cache = g
        T = q.shape[0]
        hd = q.shape[-1]
        ctx = kv_len - q_len

        def f(q_, k_, v_, cache_):
            return _replica(q_, k_, v_, cache_, kv_len, q_len, page_indices)

        _, vjp = jax.vjp(f, q, k, v, kv_cache)
        dq, dk, dv, dcache_attn = vjp(g_out.astype(q.dtype))
        # dcache_attn already lands on context slots in PAGED layout (replica gathers via
        # jnp indexing => JAX derives the scatter transpose) -- G2x2 form A ≡ form B proven.

        # g[1] routing (G2x3): this chunk's written slots fold into dk/dv; the rest passes
        # through to the incoming cache; written slots zeroed in the pass-through.
        pos = ctx + jnp.arange(T)                                # logical pos of chunk rows
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
            jnp.where(row_ok[:, None, None], zero, gk))
        dcache = dcache.at[pg, off, :, 1].set(
            jnp.where(row_ok[:, None, None], zero, gv))
        dcache = dcache + dcache_attn
        return dq, dk, dv, dcache, None, None, None

    diff_rpa.defvjp(_fwd, _bwd)
    diff_rpa._replica = _replica                                # exposed for tests
    return diff_rpa


if __name__ == "__main__":
    import numpy as np
    jax.config.update("jax_enable_x64", True)
    NQ, NKV, HD, PAGE, NP = 8, 2, 32, 16, 12
    T, CHUNK, D = 96, 32, 64
    SM = 1.0 / np.sqrt(HD)
    rng = np.random.default_rng(5)
    mk = lambda *s: jnp.asarray(rng.normal(size=s) * 0.07)
    Wq, Wk, Wv = mk(D, NQ * HD), mk(D, NKV * HD), mk(D, NKV * HD)
    X = mk(T, D)
    tbl = jnp.asarray(rng.permutation(NP)[: T // PAGE].astype(np.int32))

    def ref_kernel(q, k, v, cache, kv_len, q_len, page_indices):
        """Pure-JAX reference implementing the RPA v3 contract on this layout:
        write chunk kv at positions [ctx, kv_len), then attend causally over all kv."""
        ctx = kv_len - q_len
        pos = ctx + jnp.arange(q.shape[0])
        pg = page_indices[pos // PAGE]
        off = pos % PAGE
        ok = jnp.arange(q.shape[0]) < q_len
        newc = cache.at[pg, off, :, 0].set(
            jnp.where(ok[:, None, None], k, cache[pg, off, :, 0]))
        newc = newc.at[pg, off, :, 1].set(
            jnp.where(ok[:, None, None], v, newc[pg, off, :, 1]))
        # attend using the module's own replica math over the UPDATED cache with q_len=0 cur:
        d = make_diff_rpa_chunked(lambda *a: None, sm_scale=SM, page_size=PAGE,
                                  num_q_heads=NQ, num_kv_heads=NKV,
                                  out_dtype=jnp.float64, compute_dtype=jnp.float64)
        out = d._replica(q, k, v, cache, kv_len, q_len, page_indices)
        return out, newc

    dop = make_diff_rpa_chunked(ref_kernel, sm_scale=SM, page_size=PAGE,
                                num_q_heads=NQ, num_kv_heads=NKV,
                                out_dtype=jnp.float64, compute_dtype=jnp.float64)

    def fwd_chain(x):
        cache = jnp.zeros((NP, PAGE, NKV, 2, HD))
        outs = []
        for c0 in range(0, T, CHUNK):
            xc = x[c0:c0 + CHUNK]
            q = (xc @ Wq).reshape(-1, NQ, HD)
            k = (xc @ Wk).reshape(-1, NKV, HD)
            v = (xc @ Wv).reshape(-1, NKV, HD)
            o, cache = dop(q, k, v, cache, jnp.int32(c0 + CHUNK), jnp.int32(CHUNK), tbl)
            outs.append(o.reshape(-1, NQ * HD))
        return jnp.concatenate(outs)

    def fwd_oracle(x):
        q = (x @ Wq).reshape(-1, NQ, HD)
        kg = jnp.repeat((x @ Wk).reshape(-1, NKV, HD), NQ // NKV, axis=1)
        vg = jnp.repeat((x @ Wv).reshape(-1, NKV, HD), NQ // NKV, axis=1)
        s = jnp.einsum("qhd,shd->hqs", q, kg) * SM
        m = jnp.arange(T)[:, None] >= jnp.arange(T)[None, :]
        s = jnp.where(m[None], s, -1e30)
        return jnp.einsum("hqs,shd->qhd", jax.nn.softmax(s, -1), vg).reshape(-1, NQ * HD)

    tv = mk(T, NQ * HD)
    lC = lambda x: jnp.sum(fwd_chain(x) * tv)
    lO = lambda x: jnp.sum(fwd_oracle(x) * tv)
    vC, vO = float(lC(X)), float(lO(X))
    gC = np.asarray(jax.grad(lC)(X))
    gO = np.asarray(jax.grad(lO)(X))
    r = float(np.linalg.norm(gC - gO) / np.linalg.norm(gO))
    i, j = 40, 17
    best = 1e9
    for eps in (1e-5, 1e-6, 1e-7):
        xp = np.asarray(X).copy(); xp[i, j] += eps
        xm = np.asarray(X).copy(); xm[i, j] -= eps
        fd = (float(lC(jnp.asarray(xp))) - float(lC(jnp.asarray(xm)))) / (2 * eps)
        best = min(best, abs(fd - gC[i, j]) / (abs(gC[i, j]) + 1e-300))
    print(f"[selftest] value |chain-oracle| = {abs(vC - vO):.3e}")
    print(f"[selftest] grad rel = {r:.3e}")
    print(f"[selftest] FD best rel = {best:.3e}")
    ok = abs(vC - vO) < 1e-12 and r < 1e-12 and best < 1e-6
    print(f"[selftest] VERDICT: {'PASS' if ok else 'FAIL'}")


def make_diff_rpa_chunked_ragged(kernel_fn, *, sm_scale, page_size, num_q_heads,
                                 num_kv_heads, out_dtype=None,
                                 compute_dtype=jnp.float32):
    """Thin ragged adapter: full RPA v3 signature, single-sequence semantics (seq 0).
    Forward calls kernel_fn VERBATIM with the original args (numerics cannot move).
    Backward extracts seq-0 scalars and reuses the gate-verified chunked replica + g[1]
    routing.  Metadata cotangents are None; kv_cache receives dcache."""
    core = make_diff_rpa_chunked(lambda *a: None, sm_scale=sm_scale, page_size=page_size,
                                 num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
                                 out_dtype=out_dtype, compute_dtype=compute_dtype)

    @jax.custom_vjp
    def diff_rpa(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution):
        return kernel_fn(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution)

    def _fwd(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution):
        out = diff_rpa(q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution)
        return out, (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens)

    def _bwd(res, g):
        q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens = res
        g_out, g_cache = g[0], g[1]
        n_seqs_max = kv_lens.shape[0]
        bpr = page_indices.shape[0] // n_seqs_max
        T = q.shape[0]
        # Event 21 (root cause of the compile stall): defaulting to n_seqs_max=64 expands
        # backward to 64 sequences x 64 layers = 4096 replica VJPs. Trace time grew from
        # 0.014s to 151s and compilation did not converge. Default to one sequence; callers
        # must explicitly set CANON_VJP2_MAX_SEQS for multi-sequence workloads.
        n_active = int(os.environ.get("CANON_VJP2_MAX_SEQS", "1"))
        n_active = min(n_active, n_seqs_max)

        dq = jnp.zeros_like(q)
        dk = jnp.zeros_like(k)
        dv = jnp.zeros_like(v)
        dcache_attn = jnp.zeros_like(kv_cache)
        dcache_pass = g_cache

        # Each sequence has its own kv_len, query interval, and page table. Statically unroll
        # n_active iterations and mask rows from other sequences. Shapes remain static and
        # traced scalars participate only in value computations.
        for i in range(n_active):
            kv_len_i = kv_lens[i]
            q0_i = cu_q_lens[i]
            q1_i = cu_q_lens[i + 1]
            q_len_i = q1_i - q0_i
            tbl_i = jax.lax.dynamic_slice(page_indices, (i * bpr,), (bpr,))
            rows = jnp.arange(T)
            sel = (rows >= q0_i) & (rows < q1_i)  # Query rows owned by this sequence.
            # Move this sequence to [0, q_len_i) to reuse the single-sequence replica.
            src = q0_i + rows
            src_ok = rows < q_len_i
            gidx = jnp.where(src_ok, jnp.clip(src, 0, T - 1), 0)
            qi = jnp.where(src_ok[:, None, None], q[gidx], 0)
            ki = jnp.where(src_ok[:, None, None], k[gidx], 0)
            vi = jnp.where(src_ok[:, None, None], v[gidx], 0)
            goi = jnp.where(src_ok[:, None, None], g_out[gidx], 0)

            def f_i(q_, k_, v_, cache_):
                return core._replica(q_, k_, v_, cache_, kv_len_i, q_len_i, tbl_i)

            _, vjp_i = jax.vjp(f_i, qi, ki, vi, kv_cache)
            dqi, dki, dvi, dca_i = vjp_i(goi.astype(q.dtype))
            # Move cotangents back to their original rows.
            back = jnp.clip(rows - q0_i, 0, T - 1)
            put_ok = sel
            dq = dq + jnp.where(put_ok[:, None, None], dqi[back], 0)
            dk = dk + jnp.where(put_ok[:, None, None], dki[back], 0)
            dv = dv + jnp.where(put_ok[:, None, None], dvi[back], 0)
            dcache_attn = dcache_attn + dca_i
            # Route g[1]: fold slots written by this sequence into dk/dv; pass others through.
            ctx_i = kv_len_i - q_len_i
            pos_i = ctx_i + jnp.clip(rows - q0_i, 0, T - 1)
            pg_i = tbl_i[pos_i // page_size]
            off_i = pos_i % page_size
            gk_i = g_cache[pg_i, off_i, :, 0]
            gv_i = g_cache[pg_i, off_i, :, 1]
            zero_i = jnp.zeros_like(gk_i)
            dk = dk + jnp.where(put_ok[:, None, None], gk_i, zero_i)
            dv = dv + jnp.where(put_ok[:, None, None], gv_i, zero_i)
            dcache_pass = dcache_pass.at[pg_i, off_i, :, 0].set(
                jnp.where(put_ok[:, None, None], zero_i, dcache_pass[pg_i, off_i, :, 0]))
            dcache_pass = dcache_pass.at[pg_i, off_i, :, 1].set(
                jnp.where(put_ok[:, None, None], zero_i, dcache_pass[pg_i, off_i, :, 1]))
        dcache = dcache_pass + dcache_attn
        return dq, dk, dv, dcache, None, None, None, None

    diff_rpa.defvjp(_fwd, _bwd)
    return diff_rpa
