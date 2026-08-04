"""P19.3'a -- the full accounting for template-quantized static block-diagonal masks.

The revived candidate B: quantize every segment length UP to a multiple of the
block size, so the set of possible row layouts ("templates") is finite and each
template's block-diagonal mask can be a STATIC splash mask -- whose grid the
static path already shrinks (grid_width = max non-empty kv blocks per q row).
Exact masking stays with runtime segment_ids (the template mask is a superset),
so the numerics are untouched; only the schedule changes.

This prices the three things that decide whether it flies:

  1. how many distinct templates actually occur (upper bound: partitions of
     budget/block = p(8) = 22 at budget 2048)
  2. what the 256-quantization costs in token slots (must be << the 1.667x
     that killed equal-length bucketing)
  3. what mixing templates inside one micro-batch costs, under the two
     strategies:
       (a) group rows by template  -> each row pays its OWN template's schedule
       (b) one UNION mask per batch -> every row pays the union's schedule,
           extra blocks are zeroed by segment_ids (numerics unchanged)

Attention cost is exact block counting weighted by the P19.0 coefficients
(ratios only -- P19.1 showed absolute multi-row predictions are off by ~21%).

CPU ONLY.
"""

import sys
from collections import Counter

import jax
import numpy as np

from p18_0_blockcount import BLOCK
from p18_1_benefit import L_MAX, distributions
from p19_2_bucketcost import A_GRID, B_WORK, ffd_rows

BUDGET = 2048
NB = BUDGET // BLOCK   # 8 block slots per row


def quantize(ell):
  return min(-(-int(ell) // BLOCK) * BLOCK, L_MAX)


def template_of(row):
  """Sorted tuple of quantized segment lengths -- the row's compile key."""
  return tuple(sorted((quantize(x) for x in row), reverse=True))


def tri(n):
  return n * (n + 1) // 2


def template_schedule(tmpl):
  """(grid, work) of one row under its own static block-diagonal mask.

  grid_width shrinks to the largest segment's block count (verified on the
  real process_mask on 2026-08-03: 8x1024 -> width 4; mixed 1024+512x2+256x2
  -> width 4).  work = sum of per-segment causal triangles.
  """
  widths = [q // BLOCK for q in tmpl]
  grid_width = max(widths) if widths else 0
  q_blocks = NB  # the row is still BUDGET long; pad-tail q rows are empty
  work = sum(tri(w) for w in widths)
  return q_blocks * grid_width, work


def union_schedule(templates):
  """(grid, work) every row pays when one union mask covers the whole batch.

  The union of block-diagonal masks laid out from position 0: per q block row,
  the non-empty kv span is the max over templates.  Compute it exactly by
  materializing block-level masks.
  """
  union = np.zeros((NB, NB), dtype=bool)
  for tmpl in templates:
    pos = 0
    for q in tmpl:
      w = q // BLOCK
      first = pos // BLOCK
      for i in range(w):
        union[first + i, first : first + i + 1] = True
      pos += q
  grid_width = int(max((union[r].sum() for r in range(NB)), default=0))
  return NB * grid_width, int(union.sum())


def main():
  print(f"jax {jax.__version__}")
  print(f"budget {BUDGET}, block {BLOCK}; template upper bound = "
        f"partitions of {NB} = 22\n")

  def attn(g, w):
    return A_GRID * g + B_WORK * w

  results = []
  all_templates = Counter()
  for name, lengths in distributions():
    lengths = np.asarray(lengths)

    # baseline: FFD on raw lengths, static causal (the current default)
    base_rows = ffd_rows(lengths, BUDGET)
    g0 = len(base_rows) * NB * NB
    w0 = len(base_rows) * tri(NB)
    slots0 = len(base_rows) * BUDGET

    # template path: FFD on QUANTIZED lengths
    qlens = [quantize(x) for x in lengths]
    rows = ffd_rows(qlens, BUDGET)
    tmpls = [template_of(r) for r in rows]
    all_templates.update(tmpls)
    slots = len(rows) * BUDGET

    # strategy (a): each row pays its own template
    ga = wa = 0
    for t in tmpls:
      g, w = template_schedule(t)
      ga += g
      wa += w
    # strategy (b): every row pays the union
    gu, wu = union_schedule(set(tmpls))
    gb, wb = len(rows) * gu, len(rows) * wu

    a0, aa, ab = attn(g0, w0), attn(ga, wa), attn(gb, wb)
    results.append((name, len(set(tmpls)), slots / slots0,
                    len(base_rows), len(rows), aa / a0, ab / a0))
    print(f"--- {name}")
    print(f"    rows {len(base_rows)} -> {len(rows)}   distinct templates "
          f"{len(set(tmpls))}: {sorted(set(tmpls))}")
    print(f"    token slots {slots0} -> {slots}  ({slots/slots0:.3f}x)")
    print(f"    attention: baseline {a0:.2f} | (a) grouped {aa:.2f} "
          f"({aa/a0:.3f}x) | (b) union {ab:.2f} ({ab/a0:.3f}x)")

  n_dist = len(list(distributions()))
  if len(results) != n_dist:
    print(f"\nINCONCLUSIVE: {len(results)}/{n_dist} distributions")
    return 2

  print("\n" + "=" * 78)
  print("SUMMARY")
  print("=" * 78)
  print(f"{'distribution':<40}{'tmpls':>6}{'slots':>8}{'(a)':>8}{'(b)':>8}")
  worst_slots = 0.0
  for name, ntmpl, sl, r0, r1, ra, rb in results:
    worst_slots = max(worst_slots, sl)
    print(f"{name[:38]:<40}{ntmpl:>6}{sl:>7.3f}x{ra:>7.3f}x{rb:>7.3f}x")
  print(f"\n  distinct templates across ALL distributions: "
        f"{len(all_templates)} (bound 22): {sorted(all_templates)}")
  print(f"  worst quantization slots ratio: {worst_slots:.3f}x "
        f"(equal-length bucketing was 1.667x)")

  gate = worst_slots <= 1.15 and len(all_templates) <= 22
  print(f"\nGATE: slots <= 1.15x on every distribution AND templates <= 22: "
        f"{'PASS' if gate else 'FAIL'}")
  print("\n  (a) vs (b): (a) needs the packer to group rows by template; "
        "(b) needs nothing\n  but pays the union schedule.  The numbers above "
        "say how much (b)'s simplicity costs.")
  return 0 if gate else 1


if __name__ == "__main__":
  sys.exit(main())
