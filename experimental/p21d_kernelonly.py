"""Measure the splash KERNEL, not the Attention module around it.

Earlier arms timed model_lib.Attention, which also runs q/k/v/o projections:
at 2048 tokens those are ~0.275 TFLOP against ~0.07 TFLOP for causal splash,
so the mask's effect was diluted roughly 4:1. This calls the kernel directly.

Also sweeps sequence length: attention is quadratic and the projections are
linear, so whatever the ratio is at 2048 it should improve with length.

Every arm is interleaved across rounds with a duplicate baseline for the noise
floor; kernels and jits are built once, outside the loop.
"""
import statistics, sys, time
import jax, numpy as np
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash, splash_attention_mask as mask_lib,
    splash_attention_mask_info as mi)
from tunix.rl import splash_mask

BLK, ROWS, HEADS, HDIM, ROUNDS = 256, 4, 16, 128, 5

def ids(l): return np.concatenate([np.full(n,k,np.int32) for k,n in enumerate(l)])
def once(fn, it=15):
    s=[]
    for _ in range(it):
        t0=time.perf_counter(); o=fn(); jax.block_until_ready(o)
        s.append((time.perf_counter()-t0)*1e3)
    return statistics.median(s)
def kern(m):
    bs=splash.BlockSizes(block_q=BLK,block_kv=BLK,block_q_dkv=BLK,block_kv_dkv=BLK,
                         block_kv_dkv_compute=BLK,block_q_dq=BLK,block_kv_dq=BLK)
    return splash.make_splash_mha(mask_lib.MultiHeadMask([m]*HEADS),block_sizes=bs,
                                  head_shards=1,q_seq_shards=1)
def shp(m):
    fi=mi.process_mask(mask_lib.MultiHeadMask([m]),(BLK,BLK))[0]
    bm=np.asarray(fi.block_mask)[0]
    return int(bm.shape[-1]),int((bm==2).sum()),int((bm==1).sum())

for SEQ in (2048, 4096, 8192):
    nseg = max(2, SEQ // 512)
    lay  = [SEQ // nseg] * nseg                      # 段长 = SEQ/nseg,恰好对齐 256
    sid  = np.stack([ids(lay)] * ROWS)
    SS   = splash_mask.seg_start_union
    ARMS = [("CausalMask", mask_lib.CausalMask((SEQ,SEQ))),
            ("CausalMask DUP", mask_lib.CausalMask((SEQ,SEQ))),
            (f"computed, {nseg} segs", splash_mask.SegmentCausalMask(SEQ, SS(sid)))]
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:ROWS]).reshape(ROWS,1),("fsdp","tp"))
    with mesh:
        k_ = jax.random.PRNGKey(0)
        q = jax.random.normal(k_, (ROWS,HEADS,SEQ,HDIM), jnp.bfloat16)
        kk= jax.random.normal(k_, (ROWS,HEADS,SEQ,HDIM), jnp.bfloat16)
        v = jax.random.normal(k_, (ROWS,HEADS,SEQ,HDIM), jnp.bfloat16)
        sg= jnp.asarray(sid)
        def call(kernel, q, kk, v, sg):
            return jax.vmap(lambda a,b,c,s: kernel(a,b,c,
                segment_ids=splash.SegmentIds(q=s, kv=s)))(q,kk,v,sg)
        prep=[(n, shp(m), jax.jit(lambda q,kk,v,sg,K=kern(m): call(K,q,kk,v,sg))) for n,m in ARMS]
        for _,_,f in prep:
            for _ in range(3): jax.block_until_ready(f(q,kk,v,sg))
        smp={n:[] for n,_,_ in prep}
        for _ in range(ROUNDS):
            for n,_,f in prep: smp[n].append(once(lambda f=f: f(q,kk,v,sg)))
        base=statistics.median(smp[prep[0][0]])
        print(f"\n=== SEQ={SEQ}  段长={lay[0]}x{nseg}  (kernel only, no projections) ===")
        print(f"  {'arm':<24}{'gw':>4}{'full':>6}{'part':>6}{'median':>9}{'vs base':>9}")
        for n,(gw,fu,pa),_ in prep:
            m_=statistics.median(smp[n])
            print(f"  {n:<24}{gw:>4}{fu:>6}{pa:>6}{m_:>9.3f}{m_/base:>9.3f}")
