"""P21.1 gate -- SegmentCausalMask is EXACT, computable, and covers padding.

Four checks, in the order of what they would catch:

  1  elementwise EQUAL to `causal AND same-segment` -- not a superset, equal
  2  process_mask emits q_sequence and NO partial_mask_blocks (nothing to fetch)
  3  negative control: a flat seg_start must degrade to plain causal, proving
     check 1 can tell the two apart
  4  a wholly-padded row -- the case that broke phase 20 -- is handled, and the
     union over rows is a superset of every individual row
"""
import sys
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
    splash_attention_mask_info as mi)
from tunix.rl import splash_mask

SEQ, BLK = 2048, 256
# the last one is the packer's wholly-padded row; the middle one has a long
# padding head, which is what the real chunk in P20.6 actually looked like
ROWS = [[900, 700, 300, 148],
        [550, 500, 450, 400, 148],
        [798, 350, 300, 250, 200, 150],
        [2048]]
COSTS = dict(full=0.069, part_computed=0.102, part_tile=0.165)  # v4-8 fit

def ids(lengths):
  return np.concatenate([np.full(l, k, np.int32) for k, l in enumerate(lengths)])

def truth(sid):
  p = np.arange(SEQ)
  return (p[None, :] <= p[:, None]) & (sid[None, :] == sid[:, None])

def info(m):
  fi = mi.process_mask(mask_lib.MultiHeadMask([m]), (BLK, BLK))[0]
  bm = np.asarray(fi.block_mask)[0]
  tiles = 0 if fi.partial_mask_blocks is None else int(
      np.asarray(fi.partial_mask_blocks).shape[0])
  full, part = int((bm == 2).sum()), int((bm == 1).sum())
  cost = full * COSTS["full"] + part * COSTS[
      "part_tile" if tiles else "part_computed"]
  return dict(gw=int(bm.shape[-1]), full=full, part=part, tiles=tiles,
              qseq=fi.q_sequence is not None, cost=cost)

fails = []

print("1) 逐行:mask 与 causal & same-segment 逐元素相等?")
for r in ROWS:
  sid = ids(r)
  m = splash_mask.SegmentCausalMask(SEQ, splash_mask.seg_start_union(sid))
  eq = np.array_equal(np.asarray(m[0:SEQ, 0:SEQ]), truth(sid))
  d = info(m)
  print(f"   {str(r):<42}{'EQUAL' if eq else 'DIFFERENT'}   "
        f"gw={d['gw']} 满={d['full']:<3} 半={d['part']:<3} tiles={d['tiles']} "
        f"预测={d['cost']:.3f}ms")
  if not eq:
    fails.append(f"row {r}: mask != causal&same-segment")
  if d["tiles"]:
    fails.append(f"row {r}: {d['tiles']} tiles -- the whole point is zero")
  if not d["qseq"]:
    fails.append(f"row {r}: no q_sequence -- splash will not compute it")

base = info(mask_lib.CausalMask((SEQ, SEQ)))
print(f"\n2) 基线 CausalMask: gw={base['gw']} 满={base['full']} 半={base['part']} "
      f"tiles={base['tiles']} 预测={base['cost']:.3f}ms")
if base["tiles"]:
  fails.append("CausalMask unexpectedly needs tiles; the cost model moved")

print("\n3) 负控:seg_start 全 0 必须退化成 causal")
flat = splash_mask.SegmentCausalMask(SEQ, np.zeros(SEQ, np.int32))
p = np.arange(SEQ)
degraded = np.array_equal(np.asarray(flat[0:SEQ, 0:SEQ]), p[None, :] <= p[:, None])
print(f"   退化成 causal: {degraded}   (若为 False,说明检查 1 没有分辨率)")
if not degraded:
  fails.append("negative control did not degrade to causal")

print("\n4) 4 行并集")
sid2d = np.stack([ids(r) for r in ROWS])
u = splash_mask.SegmentCausalMask(SEQ, splash_mask.seg_start_union(sid2d))
du = np.asarray(u[0:SEQ, 0:SEQ])
sup = all(not (truth(ids(r)) & ~du).any() for r in ROWS)
d = info(u)
print(f"   是每一行的超集: {sup}")
print(f"   gw={d['gw']} 满={d['full']} 半={d['part']} tiles={d['tiles']} "
      f"预测={d['cost']:.3f}ms  ({d['cost']/base['cost']:.3f}x vs causal)")
if not sup:
  fails.append("union is not a superset of every row -- phase20's bug is back")

print("\n5) 若排除整行 padding 的那一行(P21.5 候选 B 的上界)")
sid_nopad = np.stack([ids(r) for r in ROWS[:-1]])
u2 = splash_mask.SegmentCausalMask(SEQ, splash_mask.seg_start_union(sid_nopad))
d2 = info(u2)
print(f"   gw={d2['gw']} 满={d2['full']} 半={d2['part']} "
      f"预测={d2['cost']:.3f}ms  ({d2['cost']/base['cost']:.3f}x vs causal)")

print("\nVERDICT:", "PASS" if not fails else "FAIL")
for f in fails:
  print("  -", f)
sys.exit(0 if not fails else 1)
