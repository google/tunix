"""P19.2a -- what does the equal-length constraint cost on ragged data?

P19.1 showed the reshape design is 0.49x the current default -- but only because
every segment in the row was exactly 1024 tokens.  Real lengths are ragged, and
forcing rows to hold equal-length segments pads them back up, which costs on the
token-LINEAR side (projections, MLP, lm_head).  This prices that trade.

Three designs, same sequences:
  A0  current: FFD pack at budget 2048, static causal mask + segment_ids
  A1  candidate C: same packing, dynamic mask (skips blocks, same grid)
  B   equal-length buckets: round each length up to a multiple of the block
      size, group by bucket, pack each group so a row holds only same-length
      segments -> the reshape applies and attention costs the ideal
      sum-of-triangles

Attention is counted EXACTLY (grid steps and work blocks, same instrument P18.0
validated against three known answers).  The token-linear side is `rows * budget`.
The two are combined with alpha = attention's share of step time, reported as a
CURVE rather than a single number, because alpha is model- and config-dependent
and we have not measured it end to end.

Absolute times from the P19.0 model are NOT used: P19.1 showed it over-predicts
multi-row calls by ~21%.  Only ratios are used, which P19.1 showed are preserved.

CPU ONLY.
"""

import sys

import jax
import numpy as np

from p18_0_blockcount import BLOCK
from p18_1_benefit import L_MAX, NUM_SEQS, distributions

# P19.0 coefficients.  Used ONLY to weight grid against work inside the
# attention term; every number derived from them is a ratio.
A_GRID = 10.370e-3   # ms per grid step (per head)
B_WORK = 16.400e-3   # ms per work block (per head)

MIN_NET = 0.10       # pre-registered decision threshold (phase19.md P19.2)
BUDGET = 2048        # the production default


def tri(n):
  """Causal blocks in an n-block square."""
  return n * (n + 1) // 2


def ffd_rows(lengths, budget):
  """First-fit-decreasing row assignment, mirroring the production packer."""
  rows = []
  for ell in sorted((int(x) for x in lengths), reverse=True):
    for r in rows:
      if sum(r) + ell <= budget:
        r.append(ell)
        break
    else:
      rows.append([ell])
  return rows


def attention_cost_packed(rows, budget, dynamic):
  """(grid, work) for a packed layout under the static or dynamic mask.

  Static: the grid is the full square and every causal block is computed.
  Dynamic: same grid (shrink_grid is ignored for dynamic masks), but only the
  per-segment triangles are computed.
  """
  nb = budget // BLOCK
  grid = len(rows) * nb * nb
  if not dynamic:
    return grid, len(rows) * tri(nb)
  work = 0
  for r in rows:
    pos = 0
    for ell in r:
      # a segment occupies the blocks it spans, including partial ones
      first, last = pos // BLOCK, (pos + ell - 1) // BLOCK
      work += tri(last - first + 1)
      pos += ell
    tail = budget - sum(r)
    if tail > 0:  # padding is segment 0 and attends itself (kernel.py:679)
      first, last = sum(r) // BLOCK, (budget - 1) // BLOCK
      work += tri(last - first + 1)
  return grid, work


def bucket_layout(lengths, budget):
  """Group into equal-length rows; returns (rows_total, segments, seg_len)."""
  buckets = {}
  for ell in lengths:
    b = min(-(-int(ell) // BLOCK) * BLOCK, L_MAX)
    buckets.setdefault(b, 0)
    buckets[b] += 1
  out = []
  for seg_len, n in sorted(buckets.items()):
    per_row = max(1, budget // seg_len)
    nrows = -(-n // per_row)
    out.append((nrows, per_row, seg_len))
  return out


def attention_cost_bucketed(layout):
  """(grid, work) when each segment is its own batch entry (the reshape)."""
  grid = work = 0
  for nrows, per_row, seg_len in layout:
    nb = seg_len // BLOCK
    segs = nrows * per_row
    grid += segs * nb * nb
    work += segs * tri(nb)
  return grid, work


def main():
  print(f"jax {jax.__version__} devices={jax.devices()}")
  if any(d.platform != "cpu" for d in jax.devices()):
    print("REFUSING: CPU-only")
    return 2
  print(f"budget {BUDGET}, block {BLOCK}, {NUM_SEQS} seqs/chip, "
        f"L_max {L_MAX}")
  print(f"pre-registered: net benefit must reach {MIN_NET:.0%} to justify "
        f"P19.3\n")

  def attn(g, w):
    return A_GRID * g + B_WORK * w

  results = []
  for name, lengths in distributions():
    lengths = np.asarray(lengths)
    rows = ffd_rows(lengths, BUDGET)
    g0, w0 = attention_cost_packed(rows, BUDGET, dynamic=False)
    g1, w1 = attention_cost_packed(rows, BUDGET, dynamic=True)
    layout = bucket_layout(lengths, BUDGET)
    gb, wb = attention_cost_bucketed(layout)

    slots_a = len(rows) * BUDGET
    slots_b = sum(nrows * BUDGET for nrows, _, _ in layout)

    a0, a1, ab = attn(g0, w0), attn(g1, w1), attn(gb, wb)
    results.append(dict(
        name=name, mean=float(lengths.mean()),
        rows_a=len(rows), rows_b=sum(n for n, _, _ in layout),
        slots_a=slots_a, slots_b=slots_b,
        a0=a0, a1=a1, ab=ab,
        attn_ratio_c=a1 / a0, attn_ratio_b=ab / a0,
        token_ratio_b=slots_b / slots_a,
        buckets=len(layout),
    ))

    print(f"--- {name}")
    print(f"    lengths mean {lengths.mean():.0f}  "
          f"rows: FFD {len(rows)} -> bucketed {sum(n for n,_,_ in layout)}"
          f"  ({len(layout)} buckets)")
    print(f"    token slots : {slots_a} -> {slots_b}"
          f"  ({slots_b/slots_a:.3f}x)")
    print(f"    attention   : static {a0:.2f} | C {a1:.2f} ({a1/a0:.3f}x)"
          f" | bucketed {ab:.2f} ({ab/a0:.3f}x)")

  if len(results) != len(list(distributions())):
    print("INCONCLUSIVE: not every distribution produced a row")
    return 2

  # --- combine: speedup as a function of attention's share alpha ------------
  print("\n" + "=" * 78)
  print("NET EFFECT vs the current default, as a function of alpha "
        "(= attention's share of step time)")
  print("=" * 78)
  alphas = (0.20, 0.30, 0.50)
  print(f"{'distribution':<40}" + "".join(f"{'a=%.2f' % a:>11}" for a in alphas)
        + f"{'break-even':>12}")

  def net(r, alpha, which):
    ar = r["attn_ratio_b"] if which == "b" else r["attn_ratio_c"]
    tr = r["token_ratio_b"] if which == "b" else 1.0
    return alpha * ar + (1 - alpha) * tr   # relative cost; <1 is a win

  print("\n  [candidate C -- no padding change, so alpha-independent]")
  for r in results:
    print(f"    {r['name'][:38]:<38} {r['attn_ratio_c']:.3f}x attention -> "
          f"{1 - (1 - r['attn_ratio_c']) * 0.30:.3f}x step at alpha=0.30")

  print("\n  [bucketed reshape -- pays tokens, wins attention]")
  any_win = False
  for r in results:
    cells = []
    for alpha in alphas:
      c = net(r, alpha, "b")
      cells.append(f"{c:>10.3f}x")
      if 1 - c >= MIN_NET:
        any_win = True
    # break-even alpha: alpha*ar + (1-alpha)*tr = 1
    ar, tr = r["attn_ratio_b"], r["token_ratio_b"]
    be = (1 - tr) / (ar - tr) if abs(ar - tr) > 1e-9 else float("nan")
    be_s = f"{be:>11.2f}" if 0 <= be <= 1 else ("   always" if tr < 1 and ar < 1
                                                else "     never")
    print(f"    {r['name'][:38]:<38}" + "".join(cells) + be_s)

  print("\n  cost < 1.000x is a win.  break-even alpha = the attention share at "
        "which the\n  bucketed design stops losing on tokens and starts "
        "winning overall.")

  # --- verdict --------------------------------------------------------------
  print("\n" + "=" * 78)
  print("VERDICT")
  wins_30 = [r for r in results if 1 - net(r, 0.30, "b") >= MIN_NET]
  print(f"  at alpha=0.30, bucketed reshape clears the {MIN_NET:.0%} bar for "
        f"{len(wins_30)}/{len(results)} distributions")
  for r in results:
    print(f"    {1 - net(r, 0.30, 'b'):>7.1%}  {r['name']}")
  c_gain = [1 - (1 - r["attn_ratio_c"]) * 0.30 for r in results]
  print(f"\n  for reference, candidate C at alpha=0.30 gives "
        f"{1 - max(c_gain):.1%}..{1 - min(c_gain):.1%} and costs no tokens")
  if not any_win:
    print("\n  => equal-length bucketing does NOT clear the bar on any "
          "distribution.\n     Per phase19.md P19.2 this ends the line; "
          "P19.3 is not justified.")
  else:
    print(f"\n  => bucketing clears the bar on {len(wins_30)} distribution(s) "
          "at alpha=0.30.\n     Whether that is enough is a judgement call for "
          "the author; alpha itself\n     has NOT been measured end to end "
          "(open item).")
  return 0


if __name__ == "__main__":
  sys.exit(main())
