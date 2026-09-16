#!/usr/bin/env python3
"""Band statistics for the NeMo-RL reference RCP trace, for calibration.

`is_oob_ratio` is not in the published RCP logs (this is the subject of
mlcommons/training#905), but the per-step training dumps carry both sides of
the ratio, so it can be recomputed:

    generation_logprobs  the inference engine's log-probs -- the sampler
    prev_logprobs        the trainer's recompute before the update
    token_loss_mask      which tokens are scored

    seq_geomean = exp(mean over scored tokens of (prev - generation))

Point it at a directory of `train_data_step*.jsonl` dumps:

    python3 reference_baseline.py --dir /path/to/nemo_logs_*/exp_001

The trace is large (GBs per step) and is NOT committed here; only this script is.

This is a tool, NOT the source of the reference acceptance figures in
README.md. Those come from NVIDIA's published curves in
mlcommons/training#905. A local recompute can disagree with them: the published
figures are "accepted among global_valid_seqs", and which sequences enter that
denominator is not reconstructable from the dumps alone. Treat any number this
prints as a property of the trace you pointed it at, not as the reference.
"""

import argparse
import glob
import json
import math
import os
import statistics

LO, HI = 0.999, 1.002


def sequences(path, limit):
  """Yields (seq_geomean, signed per-token mean, abs per-token mean, n_tok)."""
  out = []
  with open(path) as f:
    for line in f:
      if len(out) >= limit:
        break
      try:
        d = json.loads(line)
      except Exception:  # pylint: disable=broad-except
        continue  # dumps are written concurrently; tail records can be partial
      for gen, prev, mask in zip(
          d["generation_logprobs"], d["prev_logprobs"], d["token_loss_mask"]
      ):
        if not gen or not prev:
          continue
        n = min(len(gen), len(prev), len(mask))
        diffs = [prev[j] - gen[j] for j in range(n) if mask[j]]
        if len(diffs) < 2:
          continue
        mu = sum(diffs) / len(diffs)
        out.append((
            math.exp(mu),
            mu,
            sum(abs(x) for x in diffs) / len(diffs),
            len(diffs),
        ))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dir", required=True)
  ap.add_argument("--limit", type=int, default=600, help="sequences per step")
  args = ap.parse_args()

  paths = sorted(glob.glob(os.path.join(args.dir, "train_data_step*.jsonl*")))
  pooled = []
  for p in paths:
    rows = sequences(p, args.limit)
    if not rows:
      continue
    g = [r[0] for r in rows]
    below = sum(1 for x in g if x < LO)
    above = sum(1 for x in g if x > HI)
    print(
        f"{os.path.basename(p)[:24]:26s} n={len(g):4d}  "
        f"is_oob={(above + below) / len(g):.4f}  "
        f"below={below / len(g):.4f} above={above / len(g):.4f}  "
        f"absmean={statistics.mean([r[2] for r in rows]):.6f}  "
        f"tok/seq={statistics.mean([r[3] for r in rows]):.0f}"
    )
    pooled += rows

  if not pooled:
    print("no sequences found")
    return

  g = [r[0] for r in pooled]
  tm = [r[1] for r in pooled]
  ta = [r[2] for r in pooled]
  nt = [r[3] for r in pooled]
  below = sum(1 for x in g if x < LO)
  above = sum(1 for x in g if x > HI)
  obs = statistics.stdev(g)
  sigma_tok = statistics.mean(ta) / 0.7979
  pred = statistics.mean([sigma_tok / math.sqrt(t) for t in nt])

  print()
  print(f"POOLED  n={len(g)}")
  print(f"  is_oob_ratio          = {(above + below) / len(g):.4f}"
        f"   (below {below / len(g):.4f}, above {above / len(g):.4f})")
  print(f"  seq_geomean mean      = {statistics.mean(g):.6f}  stdev = {obs:.6f}")
  print(f"  token_logdiff_mean    = {statistics.mean(tm):+.6f}")
  print(f"  token_logdiff_absmean = {statistics.mean(ta):.6f}")
  print(f"  scored tokens/seq     = {statistics.mean(nt):.0f}")
  print(f"  correlation ratio     = {obs / pred:.2f}x"
        f"  -> effective independent tokens ~ {statistics.mean(nt) / (obs / pred) ** 2:.0f}")


if __name__ == "__main__":
  main()
