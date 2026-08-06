"""Multi-sequence gate for the ragged VJP2 adapter (fp64).
3 active sequences with different kv_len / q_len / page tables, packed in one buffer.
Reference = pure-JAX per-sequence attention (autodiff).  Gate: value bitwise, grads to fp64."""
import os
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
os.environ.setdefault("CANON_VJP2_MAX_SEQS", "4")
import rpa_diff_chunked as R

NQ, NKV, HD, PAGE, NP = 8, 2, 32, 16, 24
SM = 1.0/np.sqrt(HD)
MAXSEQ, BPR, T = 4, 4, 96                     # bpr*PAGE = 64 kv slots per seq
rng = np.random.default_rng(4)
mk = lambda *s: jnp.asarray(rng.normal(size=s)*0.07)

# 3 active seqs: q_lens 24/40/32 (sum 96 = T), kv_lens 40/56/32 (ctx = kv-q)
q_lens = [24, 40, 32]; kv_lens_l = [40, 56, 32]
cu = np.zeros(MAXSEQ+1, np.int32); cu[1:len(q_lens)+1] = np.cumsum(q_lens); cu[len(q_lens)+1:] = cu[len(q_lens)]
kv_lens = jnp.asarray(np.array(kv_lens_l + [0]*(MAXSEQ-3), np.int32))
cu_q_lens = jnp.asarray(cu)
tbl = np.asarray(rng.permutation(NP)[:MAXSEQ*BPR].astype(np.int32))
page_indices = jnp.asarray(tbl)
distribution = jnp.asarray(np.array([0,0,3], np.int32))

def attn_seq(q, kall, vall, qpos):
    qh = q.reshape(-1, NQ, HD)
    kg = jnp.repeat(kall, NQ//NKV, axis=1); vg = jnp.repeat(vall, NQ//NKV, axis=1)
    s = jnp.einsum("qhd,shd->hqs", qh, kg)*SM
    mask = qpos[:,None] >= jnp.arange(kall.shape[0])[None,:]
    s = jnp.where(mask[None], s, -1e30)
    return jnp.einsum("hqs,shd->qhd", jax.nn.softmax(s,-1), vg)

def ref_kernel(q, k, v, cache, kv_lens_, page_indices_, cu_q_lens_, distribution_):
    """v3-contract reference over ragged multi-seq packing: write then attend, per sequence."""
    newc = cache
    outs = jnp.zeros_like(q)
    for i in range(3):
        q0, q1 = int(cu[i]), int(cu[i+1]); n = q1-q0
        ctx = kv_lens_l[i]-n
        tb = page_indices_[i*BPR:(i+1)*BPR]
        pos = ctx+jnp.arange(n)
        pg = tb[pos//PAGE]; off = pos % PAGE
        newc = newc.at[pg, off, :, 0].set(k[q0:q1]).at[pg, off, :, 1].set(v[q0:q1])
        S = BPR*PAGE
        ck = newc[tb, :, :, 0].reshape(S, NKV, HD); cv = newc[tb, :, :, 1].reshape(S, NKV, HD)
        keep = jnp.arange(S) < kv_lens_l[i]
        ck = jnp.where(keep[:,None,None], ck, 0.0); cv = jnp.where(keep[:,None,None], cv, 0.0)
        colpos = jnp.where(keep, jnp.arange(S), 10**9)
        qh = q[q0:q1].reshape(-1, NQ, HD)
        kg = jnp.repeat(ck, NQ//NKV, axis=1); vg = jnp.repeat(cv, NQ//NKV, axis=1)
        s = jnp.einsum("qhd,shd->hqs", qh, kg)*SM
        m = pos[:,None] >= colpos[None,:]
        s = jnp.where(m[None], s, -1e30)
        o = jnp.einsum("hqs,shd->qhd", jax.nn.softmax(s,-1), vg)
        outs = outs.at[q0:q1].set(o)
    return outs, newc

dop = R.make_diff_rpa_chunked_ragged(ref_kernel, sm_scale=SM, page_size=PAGE,
                                     num_q_heads=NQ, num_kv_heads=NKV,
                                     out_dtype=jnp.float64, compute_dtype=jnp.float64)
q = mk(T, NQ, HD); k = mk(T, NKV, HD); v = mk(T, NKV, HD)
cache0 = jnp.zeros((NP, PAGE, NKV, 2, HD))
tv = mk(T, NQ, HD)
def L_custom(q_, k_, v_):
    o, _ = dop(q_, k_, v_, cache0, kv_lens, page_indices, cu_q_lens, distribution)
    return jnp.sum(o*tv)
def L_ref(q_, k_, v_):
    o, _ = ref_kernel(q_, k_, v_, cache0, kv_lens, page_indices, cu_q_lens, distribution)
    return jnp.sum(o*tv)
vc, vr = float(L_custom(q,k,v)), float(L_ref(q,k,v))
gc = jax.grad(L_custom, argnums=(0,1,2))(q,k,v)
gr = jax.grad(L_ref, argnums=(0,1,2))(q,k,v)
rel = lambda a,b: float(np.linalg.norm(np.asarray(a)-np.asarray(b))/(np.linalg.norm(np.asarray(a))+1e-300))
r_q, r_k, r_v = rel(gr[0],gc[0]), rel(gr[1],gc[1]), rel(gr[2],gc[2])
print(f"[msq] 3 seqs q_lens={q_lens} kv_lens={kv_lens_l}")
print(f"[msq] value: custom={vc:.15f} ref={vr:.15f} |Δ|={abs(vc-vr):.3e}")
print(f"[msq] grad rel: dq={r_q:.3e} dk={r_k:.3e} dv={r_v:.3e}")
ok = abs(vc-vr) < 1e-12 and max(r_q,r_k,r_v) < 1e-10
print(f"[msq] VERDICT: {'PASS -- ragged multi-sequence VJP2 matches per-seq autodiff' if ok else '<<<<< FAIL'}")
