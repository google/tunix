"""Paired packed-vs-unpacked comparison of two trajectory CSVs.

Usage: python3 paired_ab.py <packed.csv> <unpacked.csv> [n_steps]

Pairs on (global_step, prompt_id), which requires both runs to use the same
--seed and --shuffle. Reports the paired difference, whether it grows with step
index, and the step-0 sampling-noise floor (at step 0 both arms hold the same
policy, because generation precedes the first gradient update).
"""

import sys

import pandas as pd
from scipy import stats


def load(path, n_steps):
  d = pd.read_csv(path, engine="python")
  d = d[d.global_step < n_steps].copy()
  d["length"] = d.completion.astype(str).str.len()
  d["trunc"] = d.status.astype(str).str.contains("MAX_CONTEXT").astype(float)
  return d


def main():
  packed, unpacked = sys.argv[1], sys.argv[2]
  n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
  a, b = load(packed, n_steps), load(unpacked, n_steps)

  keys = ["global_step", "prompt_id"]
  cols = {"length": "mean", "trunc": "mean", "reward": "mean"}
  m = a.groupby(keys).agg(cols).join(
      b.groupby(keys).agg(cols), lsuffix="_p", rsuffix="_c", how="inner"
  ).reset_index()
  print(f"prompts paired: {len(m)} (packed {a.groupby(keys).ngroups}, "
        f"unpacked {b.groupby(keys).ngroups})")

  print("\npaired per-prompt difference, unpacked - packed")
  for k in ("length", "reward", "trunc"):
    d = (m[f"{k}_c"] - m[f"{k}_p"]).dropna()
    t = stats.ttest_1samp(d, 0)
    lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
    print(f"  {k:<7} {d.mean():+9.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={t.pvalue:.3f}")

  print("\ndoes the gap grow with step index? (a packing defect must)")
  for k in ("length", "reward", "trunc"):
    s = m.groupby("global_step").apply(
        lambda x, k=k: (x[f"{k}_c"] - x[f"{k}_p"]).mean(), include_groups=False
    )
    r = stats.linregress(s.index.values, s.values)
    print(f"  {k:<7} slope={r.slope:+9.5f}/step  p={r.pvalue:.3f}  "
          f"step0={s.iloc[0]:+.4f}")

  print("\nper-step KS on completion length (step 0 = sampling-noise floor)")
  ks = {s: stats.ks_2samp(a[a.global_step == s].length, b[b.global_step == s].length)
        for s in range(n_steps)}
  print(f"  step 0      D={ks[0].statistic:.4f} p={ks[0].pvalue:.3f}")
  rest = [v.statistic for s, v in ks.items() if s > 0]
  print(f"  steps 1-{n_steps-1}  D mean={sum(rest)/len(rest):.4f} "
        f"min={min(rest):.4f} max={max(rest):.4f}")
  print(f"  steps with p<0.05: {sum(v.pvalue < 0.05 for v in ks.values())}/{n_steps}")

  print("\npower check: how far does training itself move length over these steps?")
  lo_hi = lambda d: (d[d.global_step < 5].length.mean(),
                     d[d.global_step >= n_steps - 5].length.mean())
  for name, d in (("packed", a), ("unpacked", b)):
    e, l = lo_hi(d)
    k = stats.ks_2samp(d[d.global_step < 5].length,
                       d[d.global_step >= n_steps - 5].length)
    print(f"  {name:<8} {e:.0f} -> {l:.0f} chars ({l-e:+.0f})  "
          f"KS D={k.statistic:.4f} p={k.pvalue:.2e}")


if __name__ == "__main__":
  main()
