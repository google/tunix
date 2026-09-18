"""Paired packed-vs-unpacked comparison of two trajectory CSVs.

Usage: python3 qwen35_paired_ab.py <packed.csv> <unpacked.csv> [n_steps]

Pairs on (global_step, prompt_id), which requires both runs to use the same
--seed and --shuffle. Reports the paired difference, whether it grows with step
index, the step-0 sampling-noise floor (at step 0 both arms hold the same policy,
because generation precedes the first gradient update), and a direct text
comparison of the generated completions.

Needs pandas and scipy.
"""

import ast
import random
import sys

import pandas as pd
from scipy import stats

random.seed(0)

# Pairs compared per (step, prompt) group per class. The cross class has
# n_generations^2 candidates and the within classes have n*(n-1)/2, so the
# classes are subsampled to the same count to keep them comparable.
MAX_PAIRS = 40


def assistant_text(s):
  """Extracts the assistant turn from the serialized conversation.

  The `completion` column holds the whole conversation as a Python repr of a
  list of chat messages, not the generated text. Measuring it directly counts
  the prompt, which is roughly 650 characters here.
  """
  try:
    msgs = ast.literal_eval(s)
  except (ValueError, SyntaxError):
    return None
  if not isinstance(msgs, list):
    return None
  turns = [m.get("content", "") for m in msgs
           if isinstance(m, dict) and m.get("role") == "assistant"]
  return "\n".join(turns) if turns else None


def load(path, n_steps):
  d = pd.read_csv(path, engine="python")
  d = d[d.global_step < n_steps].copy()
  d["gen"] = d.completion.astype(str).map(assistant_text)
  if d.gen.isna().any():
    raise ValueError(f"{path}: {d.gen.isna().sum()} rows have no assistant turn")
  d["length"] = d.gen.str.len()
  d["trunc"] = d.status.astype(str).str.contains("MAX_CONTEXT").astype(float)
  return d


def shingles(s, k=5):
  w = s.split()
  return set(zip(*[w[i:] for i in range(k)])) if len(w) >= k else {tuple(w)}


def common_prefix(x, y):
  n = min(len(x), len(y))
  i = 0
  while i < n and x[i] == y[i]:
    i += 1
  return i


def jaccard(x, y):
  u = len(x | y)
  return len(x & y) / u if u else 1.0


def pair_stats(pairs):
  return (sum(p[0] == p[1] for p in pairs) / len(pairs),
          sum(common_prefix(p[0], p[1]) for p in pairs) / len(pairs),
          sum(jaccard(p[2], p[3]) for p in pairs) / len(pairs))


def paired(d, label):
  t = stats.ttest_1samp(d, 0)
  lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), stats.sem(d))
  print(f"  {label:<13} {d.mean():+9.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={t.pvalue:.3f}")


def main():
  packed, unpacked = sys.argv[1], sys.argv[2]
  n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
  a, b = load(packed, n_steps), load(unpacked, n_steps)
  keys = ["global_step", "prompt_id"]

  print("generated text, assistant turn only (chars)")
  for name, d in (("packed", a), ("unpacked", b)):
    print(f"  {name:<9} mean {d.length.mean():.1f}  p50 {d.length.quantile(.5):.0f}"
          f"  p90 {d.length.quantile(.9):.0f}  p99 {d.length.quantile(.99):.0f}")

  agg = {"length": "mean", "reward": "mean", "trunc": "mean"}
  m = a.groupby(keys).agg(agg).join(
      b.groupby(keys).agg(agg), lsuffix="_p", rsuffix="_c", how="inner"
  ).reset_index()
  print(f"\npaired per-prompt difference, unpacked - packed, n={len(m)}")
  print(f"  (expected {n_steps} steps x BATCH_SIZE prompts; fewer means the runs "
        "saw different data)")
  for k, lab in (("length", "length chars"), ("reward", "reward"),
                 ("trunc", "truncation")):
    paired((m[f"{k}_c"] - m[f"{k}_p"]).dropna(), lab)

  print("\ndoes the gap grow with step index? (a packing defect must)")
  for k, lab in (("length", "length"), ("reward", "reward"), ("trunc", "truncation")):
    s = m.groupby("global_step").apply(
        lambda x, k=k: (x[f"{k}_c"] - x[f"{k}_p"]).mean(), include_groups=False)
    r = stats.linregress(s.index.values, s.values)
    print(f"  {lab:<11} slope={r.slope:+9.5f}/step  p={r.pvalue:.3f}  step0={s.iloc[0]:+.4f}")

  print("\nper-step KS on generated length (step 0 = sampling-noise floor)")
  ks = {s: stats.ks_2samp(a[a.global_step == s].length, b[b.global_step == s].length)
        for s in range(n_steps)}
  rest = [v.statistic for s, v in ks.items() if s]
  print(f"  step 0      D={ks[0].statistic:.4f} p={ks[0].pvalue:.3f}")
  print(f"  steps 1-{n_steps-1}  D mean={sum(rest)/len(rest):.4f} "
        f"min={min(rest):.4f} max={max(rest):.4f}")
  print(f"  steps with p<0.05: {sum(v.pvalue < 0.05 for v in ks.values())}/{n_steps}")

  print("\npower check: how far does training itself move length over these steps?")
  for name, d in (("packed", a), ("unpacked", b)):
    e = d[d.global_step < 5]
    l = d[d.global_step >= n_steps - 5]
    k = stats.ks_2samp(e.length, l.length)
    print(f"  {name:<9} {e.length.mean():.0f} -> {l.length.mean():.0f} chars "
          f"({l.length.mean()-e.length.mean():+.0f})  KS D={k.statistic:.4f} p={k.pvalue:.2e}")

  # Direct text comparison. The within-arm classes are the same-policy baseline:
  # two draws from one arm for one prompt. If cross-arm similarity matches them,
  # the arms are sampling from indistinguishable distributions.
  for d in (a, b):
    d["sh"] = d.gen.map(shingles)
  ga, gb = dict(tuple(a.groupby(keys))), dict(tuple(b.groupby(keys)))
  rows = []
  for key in sorted(set(ga) & set(gb)):
    ca = list(zip(ga[key].gen, ga[key].sh))
    cb = list(zip(gb[key].gen, gb[key].sh))
    cross = [(x[0], y[0], x[1], y[1]) for x in ca for y in cb]
    within = [[(c[i][0], c[j][0], c[i][1], c[j][1])
               for i in range(len(c)) for j in range(i + 1, len(c))] for c in (ca, cb)]
    take = lambda l: l if len(l) <= MAX_PAIRS else random.sample(l, MAX_PAIRS)
    rows.append((key[0], *pair_stats(take(cross)),
                 *pair_stats(take(within[0])), *pair_stats(take(within[1]))))
  t = pd.DataFrame(rows, columns=["step", "x_ex", "x_lcp", "x_jac", "p_ex", "p_lcp",
                                  "p_jac", "c_ex", "c_lcp", "c_jac"])

  print("\npairwise similarity of generated text, same prompt and step")
  print(f"  {'':<18}{'exact':>9}{'prefix chars':>15}{'5-gram Jaccard':>17}")
  for lab, k in (("cross-arm", "x"), ("within-packed", "p"), ("within-unpacked", "c")):
    print(f"  {lab:<18}{t[k+'_ex'].mean():>9.4f}{t[k+'_lcp'].mean():>15.1f}"
          f"{t[k+'_jac'].mean():>17.4f}")
  print(f"\n  paired across {len(t)} groups, cross-arm minus within-arm:")
  for lab, k in (("Jaccard", "jac"), ("prefix chars", "lcp"), ("exact", "ex")):
    paired((t[f"x_{k}"] - (t[f"p_{k}"] + t[f"c_{k}"]) / 2).dropna(), lab)
  g = t.groupby("step").apply(
      lambda r: r.x_jac.mean() - (r.p_jac.mean() + r.c_jac.mean()) / 2,
      include_groups=False)
  r = stats.linregress(g.index.values, g.values)
  print(f"  Jaccard gap vs step: slope={r.slope:+.6f}  p={r.pvalue:.3f}  "
        f"step0={g.iloc[0]:+.4f}")


if __name__ == "__main__":
  main()
