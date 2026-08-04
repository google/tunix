"""P21.7 + P21.8 -- the wired route, real packer, REAL budget (8192).

Everything upstream is production code:

  gsm8k-shaped TrainExamples -> rl_utils.pack_sequences(budget) -> list chunk
  -> splash_mask.attach -> common.compute_per_token_logps -> Qwen3

Gates:
  G1  the attached kernel is on the COMPUTABLE path: q_sequence present and
      partial_mask_blocks None.  If a tile ever appears, routing has fallen
      back to the falsified NumpyMask design.
  G2  logps bitwise identical to the kernel=None arm
  G3  splash kernel timing, causal vs the real union mask, with a noise floor
  G4  how often the packer emits a wholly-padded row -- the direct cause of the
      union degrading to plain causal
"""
import argparse, statistics, sys, time
import jax, numpy as np
from flax import nnx
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash, splash_attention_mask as mask_lib,
    splash_attention_mask_info as mi)
from tunix.models.qwen3 import model as model_lib
from tunix.rl import common, splash_mask
from tunix.rl import utils as rl_utils

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=int, default=8192)
ap.add_argument("--pack_size", type=int, default=4)
ap.add_argument("--nseq", type=int, default=240)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--rounds", type=int, default=5)
A = ap.parse_args()
BLK = 256

def make_batch(lengths, seed=0):
  """gsm8k-shaped: short prompt,長completion, both padded to the batch max."""
  r = np.random.default_rng(seed)
  n = len(lengths)
  pl = [max(16, int(t * 0.18)) for t in lengths]
  cl = [t - p for t, p in zip(lengths, pl)]
  pmax, cmax = max(pl), max(cl)
  p_ids = np.zeros((n, pmax), np.int32); p_m = np.zeros((n, pmax), np.int32)
  c_ids = np.zeros((n, cmax), np.int32); c_m = np.zeros((n, cmax), np.int32)
  for i in range(n):
    p_ids[i, :pl[i]] = r.integers(1, 1000, pl[i]); p_m[i, :pl[i]] = 1
    c_ids[i, :cl[i]] = r.integers(1, 1000, cl[i]); c_m[i, :cl[i]] = 1
  return common.TrainExample(
      prompt_ids=jnp.asarray(p_ids), prompt_mask=jnp.asarray(p_m),
      completion_ids=jnp.asarray(c_ids), completion_mask=jnp.asarray(c_m),
      advantages=jnp.asarray(r.normal(size=(n,)), jnp.float32),
      ref_per_token_logps=None, old_per_token_logps=None)

# gsm8k-ish: most answers short, a tail near max_response_length
rng = np.random.default_rng(7)
LENGTHS = np.clip(rng.lognormal(6.1, 0.55, A.nseq), 120, 1400).astype(int).tolist()
print(f"输入 {A.nseq} 条,长度 min/中位/max = "
      f"{min(LENGTHS)}/{int(np.median(LENGTHS))}/{max(LENGTHS)}  总 {sum(LENGTHS)} tokens")
print(f"budget={A.budget}  pack_size={A.pack_size}  "
      f"chunk 容量={A.budget*A.pack_size}")

gen = rl_utils.pack_sequences(iter([[make_batch(LENGTHS)]]),
    max_token_budget=A.budget, sequences_per_update=None,
    pack_size=A.pack_size, max_segments_per_packed_row=64)
chunks = list(gen)
print(f"\n=== G4: packer 产出 {len(chunks)} 个 chunk ===")
allpad_rows = tot_rows = 0
for ci, ch in enumerate(chunks):
  ex = ch[0]
  lay = getattr(ex, "segment_layout", None) or ()
  sid = np.asarray(ex.segment_ids)
  per_row = []
  for ri in range(sid.shape[0]):
    tot_rows += 1
    nz = int((sid[ri] != 0).sum())
    if nz == 0: allpad_rows += 1
    per_row.append((len(lay[ri]) if ri < len(lay) else 0, nz))
  print(f"  chunk{ci}: " + "  ".join(
      f"row{i}[段{a} 实token{b}]" for i, (a, b) in enumerate(per_row)))
print(f"  ⇒ 整行 padding 的行: {allpad_rows}/{tot_rows} "
      f"({allpad_rows/tot_rows*100:.0f}%)")

cfg = model_lib.ModelConfig.qwen3_1p7b(); cfg.num_layers = A.layers
cfg.use_flash_attention = True; cfg.flash_attention_block_size = BLK
cfg.dtype = jnp.bfloat16
QH = cfg.num_heads
mesh = jax.sharding.Mesh(
    np.array(jax.devices()[:A.pack_size]).reshape(A.pack_size, 1), ("fsdp", "tp"))

fails = []
ch = chunks[0]
with mesh:
  att = splash_mask.attach(ch, seq_len=A.budget, block=BLK, num_heads=QH)
  k = getattr(att[0], "splash_kernel", None)
  print(f"\n=== G1: 走的是哪条路 ===")
  if k is None:
    fails.append("attach 没挂上 kernel"); print("  ❌ kernel is None")
  else:
    fi = k.fwd_mask_info
    bm = np.asarray(fi.block_mask)[0]
    gw, full, part = int(bm.shape[-1]), int((bm==2).sum()), int((bm==1).sum())
    pb = fi.partial_mask_blocks
    print(f"  q_sequence 存在        : {fi.q_sequence is not None}")
    print(f"  partial_mask_blocks    : {pb if pb is None else np.asarray(pb).shape}"
          f"   (必须是 None,否则退回了被证伪的 NumpyMask)")
    print(f"  gw={gw} 满={full} 半={part}   (causal 基线 gw={A.budget//BLK})")
    if fi.q_sequence is None: fails.append("不是可计算路径")
    if pb is not None: fails.append("出现 tile —— 退回 NumpyMask 了")

  # ---- G2 bitwise ----
  model = model_lib.Qwen3(cfg, rngs=nnx.Rngs(params=0))
  gd, st = nnx.split(model)
  ex = att[0]
  def run(s, kk):
    return common.compute_per_token_logps(gd, s,
        prompt_tokens=ex.prompt_ids, completion_tokens=ex.completion_ids,
        pad_id=0, eos_id=1, stop_gradient=True,
        segment_ids=getattr(ex, "segment_ids", None),
        segment_positions=getattr(ex, "segment_positions", None),
        splash_kernel=kk)
  f_off = jax.jit(lambda s: run(s, None)); f_on = jax.jit(lambda s, kk: run(s, kk))
  o1 = np.asarray(jax.device_get(f_off(st))); o2 = np.asarray(jax.device_get(f_on(st, k)))
  same = np.array_equal(o1, o2)
  print(f"\n=== G2: logps bitwise {'IDENTICAL' if same else 'DIFFERENT'} ===")
  if not same:
    d = np.abs(o1 - o2); print(f"   max|diff|={d.max():.3e} ndiff={(d>0).sum()}/{d.size}")
    fails.append("logps 不同")

  # ---- G3 kernel-only timing ----
  def kern(m):
    bs = splash.BlockSizes(block_q=BLK, block_kv=BLK, block_q_dkv=BLK,
        block_kv_dkv=BLK, block_kv_dkv_compute=BLK, block_q_dq=BLK, block_kv_dq=BLK)
    return splash.make_splash_mha(mask_lib.MultiHeadMask([m]*QH),
        block_sizes=bs, head_shards=1, q_seq_shards=1)
  def once(fn, it=10):
    s=[]
    for _ in range(it):
      t0=time.perf_counter(); o=fn(); jax.block_until_ready(o)
      s.append((time.perf_counter()-t0)*1e3)
    return statistics.median(s)
  ARMS = [("CausalMask", mask_lib.CausalMask((A.budget, A.budget))),
          ("CausalMask DUP", mask_lib.CausalMask((A.budget, A.budget)))]
  for ci, c in enumerate(chunks):
    sid_c = np.asarray(c[0].segment_ids)
    npad = int(sum(1 for r in sid_c if (r != 0).sum() == 0))
    ARMS.append((f"chunk{ci} (pad行={npad})",
                 splash_mask.SegmentCausalMask(A.budget,
                     splash_mask.seg_start_union(sid_c))))
  sid = np.asarray(ex.segment_ids)
  key = jax.random.PRNGKey(0)
  q = jax.random.normal(key, (A.pack_size, QH, A.budget, cfg.head_dim), jnp.bfloat16)
  kk_ = jax.random.normal(key, (A.pack_size, QH, A.budget, cfg.head_dim), jnp.bfloat16)
  v = jax.random.normal(key, (A.pack_size, QH, A.budget, cfg.head_dim), jnp.bfloat16)
  sg = jnp.asarray(sid)   # 计时用同一份 segment_ids,只变 mask,控制变量
  prep=[]
  for n, m in ARMS:
    fi2 = mi.process_mask(mask_lib.MultiHeadMask([m]), (BLK, BLK))[0]
    b2 = np.asarray(fi2.block_mask)[0]
    K = kern(m)
    prep.append((n, (int(b2.shape[-1]), int((b2==2).sum()), int((b2==1).sum())),
        jax.jit(lambda q,kk,v,sg,K=K: jax.vmap(
            lambda a,b,c,s: K(a,b,c,segment_ids=splash.SegmentIds(q=s,kv=s)))(q,kk,v,sg))))
  for _,_,f in prep:
    for _ in range(3): jax.block_until_ready(f(q,kk_,v,sg))
  smp={n:[] for n,_,_ in prep}
  for _ in range(A.rounds):
    for n,_,f in prep: smp[n].append(once(lambda f=f: f(q,kk_,v,sg)))
  base = statistics.median(smp[prep[0][0]])
  print(f"\n=== G3: splash kernel only, budget={A.budget}, 真实并集 ===")
  print(f"  {'arm':<24}{'gw':>4}{'full':>7}{'part':>7}{'median':>10}{'vs base':>9}")
  for n,(gw,fu,pa),_ in prep:
    m_=statistics.median(smp[n])
    print(f"  {n:<24}{gw:>4}{fu:>7}{pa:>7}{m_:>10.3f}{m_/base:>9.3f}")

# G5: 退化的 chunk 必须被护栏拦下(attach 返回原对象)
print("\n=== G5: 护栏 —— 退化 chunk 必须回退,不挂 kernel ===")
for ci, c in enumerate(chunks):
  sid_c = np.asarray(c[0].segment_ids)
  npad = int(sum(1 for r in sid_c if (r != 0).sum() == 0))
  a = splash_mask.attach(c, seq_len=A.budget, block=BLK, num_heads=QH)
  got = getattr(a[0], "splash_kernel", None) is not None
  gwc = (int(np.asarray(a[0].splash_kernel.fwd_mask_info.data_next).shape[-1])
         if got else None)
  print(f"  chunk{ci} (pad行={npad}): 挂 kernel={got}"
        + (f", gw={gwc}" if got else "  <- 回退 CausalMask"))
  if npad and got:
    fails.append(f"chunk{ci} 有整行 padding 却仍挂了 kernel")
  if not npad and not got:
    fails.append(f"chunk{ci} 装满却没挂 kernel")

print("\nVERDICT:", "PASS" if not fails else "FAIL")
for x in fails: print("  -", x)
sys.exit(0 if not fails else 1)
