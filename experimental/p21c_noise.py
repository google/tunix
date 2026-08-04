"""Establish the noise floor before believing any 5% difference.

Two arms measured back to back gave 0.995x and 0.928x for the same
configuration in consecutive runs. That spread is as large as the effect being
chased, so every arm here is measured in INTERLEAVED rounds and reported with
its spread, not a single median. An identical duplicate of the baseline is
included as a control: whatever ratio IT shows is the noise floor, and no arm
closer to 1.0 than that can be called a win.
"""
import statistics, sys, time
import jax, numpy as np
from flax import nnx
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash, splash_attention_mask as mask_lib,
    splash_attention_mask_info as mi)
from tunix.models.qwen3 import model as model_lib
from tunix.rl import splash_mask

SEQ, BLK, ROWS, ROUNDS = 2048, 256, 4, 7
def ids(l): return np.concatenate([np.full(n,k,np.int32) for k,n in enumerate(l)])
def once(fn, it=12):
    s=[]
    for _ in range(it):
        t0=time.perf_counter(); o=fn(); jax.block_until_ready(o)
        s.append((time.perf_counter()-t0)*1e3)
    return statistics.median(s)
def kern(m,qh):
    bs=splash.BlockSizes(block_q=BLK,block_kv=BLK,block_q_dkv=BLK,block_kv_dkv=BLK,
                         block_kv_dkv_compute=BLK,block_q_dq=BLK,block_kv_dq=BLK)
    return splash.make_splash_mha(mask_lib.MultiHeadMask([m]*qh),block_sizes=bs,
                                  head_shards=1,q_seq_shards=1)
def shp(m):
    fi=mi.process_mask(mask_lib.MultiHeadMask([m]),(BLK,BLK))[0]
    bm=np.asarray(fi.block_mask)[0]
    return int(bm.shape[-1]),int((bm==2).sum()),int((bm==1).sum())

UNALIGNED=[900,700,300,148]; ALIGNED=[768,512,512,256]
SS=splash_mask.seg_start_union
cfg=model_lib.ModelConfig.qwen3_1p7b(); cfg.num_layers=1
cfg.use_flash_attention=True; cfg.flash_attention_block_size=BLK; cfg.dtype=jnp.bfloat16
qh=cfg.num_heads
mesh=jax.sharding.Mesh(np.array(jax.devices()[:ROWS]).reshape(ROWS,1),("fsdp","tp"))
sid_un=np.stack([ids(UNALIGNED)]*ROWS); sid_al=np.stack([ids(ALIGNED)]*ROWS)

ARMS=[("CausalMask (base)",        mask_lib.CausalMask((SEQ,SEQ)),                  sid_un),
      ("CausalMask (DUPLICATE)",   mask_lib.CausalMask((SEQ,SEQ)),                  sid_un),
      ("computed, unaligned",      splash_mask.SegmentCausalMask(SEQ,SS(sid_un)),   sid_un),
      ("computed, ALIGNED",        splash_mask.SegmentCausalMask(SEQ,SS(sid_al)),   sid_al)]

with mesh:
    attn=model_lib.Attention(config=cfg,rngs=nnx.Rngs(params=0))
    x=jax.random.normal(jax.random.PRNGKey(0),(ROWS,SEQ,cfg.embed_dim),jnp.bfloat16)
    pos=jnp.tile(jnp.arange(SEQ)[None],(ROWS,1))
    f=jax.jit(lambda v,k,s: attn(v,pos,None,None,s,splash_kernel=k)[1])
    prep=[(n,shp(m),kern(m,qh),jnp.asarray(s)) for n,m,s in ARMS]
    for _,_,k,s in prep:                       # warm every arm before timing any
        for _ in range(3): jax.block_until_ready(f(x,k,s))
    samples={n:[] for n,_,_,_ in prep}
    for r in range(ROUNDS):                    # interleaved: one pass per round
        for n,_,k,s in prep:
            samples[n].append(once(lambda k=k,s=s: f(x,k,s)))
    base=statistics.median(samples[prep[0][0]])
    print(f"{'arm':<28}{'gw':>4}{'full':>6}{'part':>6}"
          f"{'median':>9}{'min':>8}{'max':>8}{'vs base':>9}")
    for n,(gw,fu,pa),_,_ in prep:
        v=samples[n]; m=statistics.median(v)
        print(f"{n:<28}{gw:>4}{fu:>6}{pa:>6}{m:>9.3f}{min(v):>8.3f}{max(v):>8.3f}"
              f"{m/base:>9.3f}")
    dup=statistics.median(samples[prep[1][0]])/base
    print(f"\n  噪声底(同一配置的重复臂) = {dup:.3f}x"
          f"  ⇒ 任何比 |1-{abs(1-dup):.3f}| 更小的差异都不可信")
