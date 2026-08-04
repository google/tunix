"""Why does F disagree with D, and why is attention SLOWER?

Two hypotheses, one control each:

H1  the empty layout row.  The packer emitted ((...), (...), (...), ()) -- the
    4th row holds only padding, whose segment_id is 0 everywhere.  splash's
    segment comparison is a bare `q_ids == kv_ids` with no special case for 0,
    so PAD ATTENDS PAD across that row's whole causal triangle.  The doc mask
    is a union over the layout rows, and an empty row contributes nothing, so
    for that row the mask is NOT a superset.  Control: a mask whose empty rows
    are filled in with the full causal triangle.

H2  the mask representation, not the mask content.  CausalMask is a type splash
    KNOWS: it emits q_sequence and rebuilds the triangle in-register.  NumpyMask
    is opaque, so every partial block is a 256x256 tile fetched from memory.
    Control: a NumpyMask holding EXACTLY the causal mask -- same content as
    today, different representation.  If that alone is slower, the cost is the
    representation and has nothing to do with the document mask.
"""
import statistics, time
import jax, numpy as np
from flax import nnx
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash, splash_attention_mask as mask_lib)
from tunix.models.qwen3 import model as model_lib
from tunix.rl import common, splash_mask
from tunix.rl import utils as rl_utils

BUDGET, BLOCK, PACK = 2048, 256, 4
LENGTHS = [700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150]

def make_batch(lengths):
    n=len(lengths); pf=0.3
    pmax=max(int(l*pf) for l in lengths); cmax=max(l-int(l*pf) for l in lengths)
    p=np.zeros((n,pmax),np.int32); pm=np.zeros((n,pmax),np.int32)
    c=np.zeros((n,cmax),np.int32); cm=np.zeros((n,cmax),np.int32)
    r=np.random.default_rng(0)
    for i,t in enumerate(lengths):
        pl=int(t*pf); cl=t-pl
        p[i,:pl]=r.integers(1,1000,pl); pm[i,:pl]=1
        c[i,:cl]=r.integers(1,1000,cl); cm[i,:cl]=1
    return common.TrainExample(prompt_ids=jnp.asarray(p),prompt_mask=jnp.asarray(pm),
        completion_ids=jnp.asarray(c),completion_mask=jnp.asarray(cm),
        advantages=jnp.asarray(r.normal(size=(n,)),jnp.float32),
        ref_per_token_logps=None,old_per_token_logps=None)

def timed(fn,it=8,wu=3):
    for _ in range(wu): jax.block_until_ready(fn())
    s=[]
    for _ in range(it):
        t0=time.perf_counter(); o=fn(); jax.block_until_ready(o)
        s.append((time.perf_counter()-t0)*1e3)
    return statistics.median(s)

def kern(dense, qh):
    bs=splash.BlockSizes(block_q=BLOCK,block_kv=BLOCK,block_q_dkv=BLOCK,block_kv_dkv=BLOCK,
                         block_kv_dkv_compute=BLOCK,block_q_dq=BLOCK,block_kv_dq=BLOCK)
    m = dense if isinstance(dense, mask_lib.Mask) else mask_lib.NumpyMask(dense)
    return splash.make_splash_mha(mask_lib.MultiHeadMask([m]*qh), block_sizes=bs,
                                  head_shards=1, q_seq_shards=1)

gen = rl_utils.pack_sequences(iter([[make_batch(LENGTHS)]]), max_token_budget=BUDGET,
      sequences_per_update=None, pack_size=PACK, max_segments_per_packed_row=8)
ex = next(gen)[0]
layout = ex.segment_layout
seg = np.asarray(ex.segment_ids)
print(f"layout      = {layout}")
for i in range(PACK):
    u,cts = np.unique(seg[i], return_counts=True)
    print(f"  row{i} segment_ids: " + ", ".join(f"id{int(a)}x{int(b)}" for a,b in zip(u,cts)))

cfg = model_lib.ModelConfig.qwen3_1p7b(); cfg.num_layers=1
cfg.use_flash_attention=True; cfg.flash_attention_block_size=BLOCK; cfg.dtype=jnp.bfloat16
qh = cfg.num_heads
mesh = jax.sharding.Mesh(np.array(jax.devices()[:PACK]).reshape(PACK,1), ("fsdp","tp"))

p = np.arange(BUDGET); CAUSAL = p[None,:] <= p[:,None]
DOC   = splash_mask.docmask(BUDGET, layout, BLOCK)
# H1 control: empty layout rows get the full causal triangle
lay_fix = tuple(row if row else (BUDGET,) for row in layout)
DOCFIX = splash_mask.docmask(BUDGET, lay_fix, BLOCK)
print(f"\nH1  DOC 是 causal&same-seg 的超集? ", end="")
# segment_ids 允许的真实集合(含 pad-attends-pad)
allow = np.zeros((PACK,BUDGET,BUDGET), bool)
for i in range(PACK): allow[i] = CAUSAL & (seg[i][None,:]==seg[i][:,None])
bad = [i for i in range(PACK) if (allow[i] & ~DOC).sum()]
print(f"{'是' if not bad else f'否 —— 第 {bad} 行漏了 ' + str(int(sum((allow[i]&~DOC).sum() for i in bad))) + ' 个对'}")
print(f"    补上空行后是超集? {'是' if not any((allow[i]&~DOCFIX).sum() for i in range(PACK)) else '否'}")

with mesh:
    attn = model_lib.Attention(config=cfg, rngs=nnx.Rngs(params=0))
    x = jax.random.normal(jax.random.PRNGKey(0),(PACK,BUDGET,cfg.embed_dim),jnp.bfloat16)
    pos = jnp.asarray(ex.segment_positions); sg = jnp.asarray(ex.segment_ids)
    base = jax.jit(lambda v: attn(v,pos,None,None,sg)[1])
    withk = jax.jit(lambda v,k: attn(v,pos,None,None,sg,splash_kernel=k)[1])
    ref = np.asarray(jax.device_get(base(x)))
    t_base = timed(lambda: base(x))
    print(f"\n{'arm':<34}{'ms':>9}  {'vs base':>8}   bitwise")
    print(f"  {'CausalMask (今天,base)':<32}{t_base:9.3f}  {1.0:8.3f}   —")
    for name, dense in (("NumpyMask(causal) —— H2 对照", CAUSAL),
                        ("NumpyMask(doc, 原样)", DOC),
                        ("NumpyMask(doc, 补上空行)", DOCFIX)):
        k = kern(dense, qh)
        gw = int(np.asarray(k.fwd_mask_info.data_next).shape[-1])
        pb = int(np.asarray(k.fwd_mask_info.partial_mask_blocks).shape[0])
        o = np.asarray(jax.device_get(withk(x,k)))
        t = timed(lambda: withk(x,k))
        same = np.array_equal(ref,o)
        nd = "" if same else f"  差 {int((ref!=o).sum())}/{ref.size}, 行 {sorted(set(np.argwhere(ref!=o)[:,0].tolist()))}"
        print(f"  {name:<32}{t:9.3f}  {t/t_base:8.3f}   "
              f"{'IDENTICAL' if same else 'DIFFERENT'}  (gw={gw},pb={pb}){nd}")
