"""P21.2 + P21.3 -- the computable segment-causal mask, measured on TPU.

Arms, all against the SAME q/k/v/segment_ids and the same production baseline
(mask_lib.CausalMask, which is what model.py builds today):

  base      CausalMask                              production
  uniform   every row has the same layout           the achievable win, no union
  union     the real 4-row union                    what a mixed chunk gets
  nopad     union over the non-padding rows only    P21.5 candidate B

TIMING DISCIPLINE: kernels and jits are built ONCE, outside the loop. Building
either one inside it measured the compiler and reported 19x and 0.000x in
phase 20.
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

SEQ, BLK, ROWS = 2048, 256, 4
LAYOUTS = [[900, 700, 300, 148], [550, 500, 450, 400, 148],
           [798, 350, 300, 250, 200, 150], [2048]]
UNIFORM = [[900, 700, 300, 148]] * ROWS

def ids(l): return np.concatenate([np.full(n, k, np.int32) for k, n in enumerate(l)])
def timed(fn, it=10, wu=3):
    for _ in range(wu): jax.block_until_ready(fn())
    s = []
    for _ in range(it):
        t0 = time.perf_counter(); o = fn(); jax.block_until_ready(o)
        s.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(s)

def kern(m, qh):
    bs = splash.BlockSizes(block_q=BLK, block_kv=BLK, block_q_dkv=BLK,
                           block_kv_dkv=BLK, block_kv_dkv_compute=BLK,
                           block_q_dq=BLK, block_kv_dq=BLK)
    return splash.make_splash_mha(mask_lib.MultiHeadMask([m] * qh),
                                  block_sizes=bs, head_shards=1, q_seq_shards=1)

def shape_of(m):
    fi = mi.process_mask(mask_lib.MultiHeadMask([m]), (BLK, BLK))[0]
    bm = np.asarray(fi.block_mask)[0]
    t = 0 if fi.partial_mask_blocks is None else int(np.asarray(fi.partial_mask_blocks).shape[0])
    return int(bm.shape[-1]), int((bm == 2).sum()), int((bm == 1).sum()), t

cfg = model_lib.ModelConfig.qwen3_1p7b(); cfg.num_layers = 1
cfg.use_flash_attention = True; cfg.flash_attention_block_size = BLK
cfg.dtype = jnp.bfloat16
qh = cfg.num_heads
mesh = jax.sharding.Mesh(np.array(jax.devices()[:ROWS]).reshape(ROWS, 1), ("fsdp", "tp"))
fails = []

sid_mixed = np.stack([ids(l) for l in LAYOUTS])
sid_unif = np.stack([ids(l) for l in UNIFORM])
SS = splash_mask.seg_start_union

ARMS = [
    ("CausalMask (production)",     mask_lib.CausalMask((SEQ, SEQ)),                     sid_mixed, True),
    ("computed, uniform rows",      splash_mask.SegmentCausalMask(SEQ, SS(sid_unif)),    sid_unif,  True),
    ("computed, real 4-row union",  splash_mask.SegmentCausalMask(SEQ, SS(sid_mixed)),   sid_mixed, True),
    ("computed, union w/o pad row", splash_mask.SegmentCausalMask(SEQ, SS(sid_mixed[:3])), sid_mixed, False),
]

with mesh:
    attn = model_lib.Attention(config=cfg, rngs=nnx.Rngs(params=0))
    x = jax.random.normal(jax.random.PRNGKey(0), (ROWS, SEQ, cfg.embed_dim), jnp.bfloat16)
    pos = jnp.tile(jnp.arange(SEQ)[None], (ROWS, 1))
    f = jax.jit(lambda v, k, s: attn(v, pos, None, None, s, splash_kernel=k)[1])
    prepared = [(n, shape_of(m), kern(m, qh), jnp.asarray(s), strict)
                for n, m, s, strict in ARMS]
    ref = {}
    base_t = None
    print(f"{'arm':<32}{'gw':>4}{'full':>6}{'part':>6}{'tile':>6}"
          f"{'ms':>9}{'vs base':>9}   bitwise")
    for name, (gw, fu, pa, ti), k, s, strict in prepared:
        o = np.asarray(jax.device_get(f(x, k, s)))
        key = s.tobytes()
        if key not in ref: ref[key] = o
        same = np.array_equal(ref[key], o)
        t = timed(lambda k=k, s=s: f(x, k, s))
        if base_t is None: base_t = t
        note = "IDENTICAL" if same else "DIFFERENT"
        if not same and not strict:
            # candidate B is only legitimate if every REAL token is untouched
            real = np.asarray(sid_mixed) != 0
            bad = int((ref[key][real] != o[real]).sum())
            note = (f"real-token DIFF={bad}" if bad
                    else "real tokens IDENTICAL (pad differs, by design)")
            if bad: fails.append(f"{name}: {bad} real-token elements changed")
        elif not same:
            fails.append(f"{name}: output differs but must not")
        if ti: fails.append(f"{name}: fetches {ti} tile(s) -- must be zero")
        print(f"{name:<32}{gw:>4}{fu:>6}{pa:>6}{ti:>6}{t:>9.3f}{t/base_t:>9.3f}   {note}")

print("\nVERDICT:", "PASS" if not fails else "FAIL")
for x_ in fails: print("  -", x_)
sys.exit(0 if not fails else 1)
