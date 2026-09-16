#!/usr/bin/env python3
"""Recompute every number in README.md from the logs committed next to it.

Nothing here reads the network or a TPU -- it parses the `[sampler-is]` lines
the trainer emitted and the `Train step` lines the orchestrator emitted. Run it
after changing anything in the report and diff the output against the tables.

    python3 docs/bringup/2026-09-15-dense-grpo/verify_report_numbers.py

Only stdlib, so it runs in any interpreter.
"""

import argparse
import math
import os
import re
import statistics

RUNS = ["tis4", "mb1", "mb2", "mb4", "tok1", "tok4"]


def sampler_is(path, key):
  """Every value of `key` across the trainer's [sampler-is] lines, in order."""
  out = []
  with open(path) as f:
    for line in f:
      if "[sampler-is]" not in line:
        continue
      m = re.search(re.escape(key) + r"=([-0-9.e+]+)", line)
      if m:
        out.append(float(m.group(1)))
  return out


def orchestrator(path, pattern):
  out = []
  with open(path) as f:
    for line in f:
      m = re.search(pattern, line)
      if m:
        out.append(m.group(1))
  return out


def normal_cdf(z):
  return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument(
      "--logs",
      default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"),
  )
  args = ap.parse_args()

  summary = {}

  print("=" * 68)
  print("SETUP (from the rollout node's own argument dump)")
  print("=" * 68)
  roll = os.path.join(args.logs, "tok4", "rollout.log")
  if os.path.exists(roll):
    with open(roll) as f:
      for line in f:
        if "Parsed args" in line:
          for k in (
              "model_id=",
              "sampler=",
              "mesh_fsdp=",
              "mesh_tp=",
              "tensor_parallel_size=",
              "enable_prefix_caching=",
              "max_prompt_length=",
              "max_response_length=",
          ):
            m = re.search(re.escape(k) + r"'?([^,')]*)", line)
            if m:
              print(f"  {k}{m.group(1)}")
          break
    with open(roll) as f:
      for line in f:
        if "sampler on the rollout mesh" in line:
          print(f"  log line: {line.split(' - ', 1)[-1].strip()}")

  print()
  print("=" * 68)
  print("PER-RUN METRICS")
  print("=" * 68)
  for run in RUNS:
    p = os.path.join(args.logs, run, "trainer.log")
    if not os.path.exists(p):
      continue
    oob = sampler_is(p, "tis/is_oob_ratio")
    if not oob:
      continue
    gmin = sampler_is(p, "sampler_is/seq_geomean_min")
    gmax = sampler_is(p, "sampler_is/seq_geomean_max")
    kept = sampler_is(p, "sample_mask/kept_frac")
    d = {
        "calls": len(oob),
        "oob_mean": statistics.mean(oob),
        "oob_vals": sorted(set(oob)),
        "absmean": statistics.mean(
            sampler_is(p, "sampler_is/token_logdiff_absmean")
        ),
        "signed": statistics.mean(sampler_is(p, "sampler_is/token_logdiff_mean")),
        "kept_all_one": all(k == 1.0 for k in kept),
        "degenerate": all(a == b for a, b in zip(gmin, gmax)),
        "above": sum(1 for g in gmax if g > 1.002),
        "below": sum(1 for g in gmin if g < 0.999),
        "n": len(gmax),
        "tokens": sampler_is(p, "sampler_is/scored_tokens_per_seq"),
        "gmin": gmin,
        "gmax": gmax,
    }
    summary[run] = d
    print(f"\n--- {run}: {d['calls']} loss calls ---")
    print(f"  is_oob_ratio  mean={d['oob_mean']:.4f}  distinct={d['oob_vals']}")
    print(f"  absmean       {d['absmean']:.6f}")
    print(f"  signed mean   {d['signed']:+.6f}")
    print(f"  kept_frac == 1 on every call: {d['kept_all_one']}")
    print(f"  seq_geomean min == max on every call: {d['degenerate']}")
    print(f"  above 1.002: {d['above']}/{d['n']}   below 0.999: {d['below']}/{d['n']}")
    if d["tokens"]:
      print(f"  scored_tokens_per_seq: {d['tokens']}")

  print()
  print("=" * 68)
  print("CLAIM 1  the mb1 and mb4 runs saw identical rollouts")
  print("=" * 68)
  if "tok1" in summary and "tok4" in summary:
    g = summary["tok1"]["tokens"]
    h = summary["tok4"]["tokens"]
    grouped = [sum(g[i : i + 4]) / 4 for i in range(0, len(g), 4)]
    print(f"  tok1 per-sequence     : {[int(x) for x in g]}")
    print(f"  tok1 grouped in 4s    : {grouped}")
    print(f"  tok4 per-call means   : {h}")
    print(f"  IDENTICAL             : {grouped == h}")
    print(f"  scored token range {int(min(g))}-{int(max(g))} against a 512+768=1280 buffer")

  print()
  print("=" * 68)
  print("CLAIM 2  determinism: independent reruns of the same config")
  print("=" * 68)
  for a, b in (("mb1", "tok1"), ("mb4", "tok4")):
    if a in summary and b in summary:
      same = (
          summary[a]["absmean"] == summary[b]["absmean"]
          and summary[a]["signed"] == summary[b]["signed"]
      )
      print(
          f"  {a} vs {b}: absmean {summary[a]['absmean']:.6f} / "
          f"{summary[b]['absmean']:.6f}  identical={same}"
      )

  print()
  print("=" * 68)
  print("CLAIM 3  trainer log-probs depend on forward-pass batch shape")
  print("=" * 68)
  for n, r in ((1, "mb1"), (2, "mb2"), (4, "mb4")):
    if r in summary:
      print(
          f"  train_micro_batch_size={n}: absmean={summary[r]['absmean']:.6f}"
          f"  signed={summary[r]['signed']:+.6f}"
      )
  if "mb1" in summary and "mb4" in summary:
    a1, a4 = summary["mb1"]["absmean"], summary["mb4"]["absmean"]
    # For a zero-mean Gaussian, E|x| = sigma*sqrt(2/pi) = 0.7979*sigma.
    s1, s4 = a1 / 0.7979, a4 / 0.7979
    print(f"  ratio mb4/mb1 = {a4 / a1:.2f}x")
    print(
        f"  sigma {s1:.5f} -> {s4:.5f}; in quadrature "
        f"{(s4 ** 2 - s1 ** 2) ** 0.5:.5f} nats/token of induced noise"
    )

  print()
  print("=" * 68)
  print("CLAIM 4  the OOB rate is mostly explained by noise (tis4)")
  print("=" * 68)
  if "tis4" in summary:
    d = summary["tis4"]
    rng = [b - a for a, b in zip(d["gmin"], d["gmax"])]
    mr = statistics.mean(rng)
    sigma = mr / 2.059  # E[range of n=4 normals] = 2.059 sigma
    p_in = normal_cdf(0.002 / sigma) - normal_cdf(-0.001 / sigma)
    print(f"  mean range of 4 = {mr:.5f}  ->  sigma = {sigma:.5f}")
    print(f"  predicted OOB = {1 - p_in:.4f}     measured OOB = {d['oob_mean']:.4f}")

  print()
  print("=" * 68)
  print("CLAIM 5  per-token errors are correlated within a sequence")
  print("=" * 68)
  # At micro-batch 1 there is one sequence per loss call, so seq_geomean is
  # observed directly instead of estimated from a range of 4.
  p = os.path.join(args.logs, "tok1", "trainer.log")
  if os.path.exists(p):
    g = sampler_is(p, "sampler_is/seq_geomean_mean")
    tok = sampler_is(p, "sampler_is/scored_tokens_per_seq")
    am = sampler_is(p, "sampler_is/token_logdiff_absmean")
    oob = sampler_is(p, "tis/is_oob_ratio")
    sigma_tok = [a / 0.7979 for a in am]
    predicted = statistics.mean(
        [s / math.sqrt(t) for s, t in zip(sigma_tok, tok)]
    )
    observed = statistics.stdev(g)
    ratio = observed / predicted
    print(f"  sequences                         : {len(g)}")
    print(f"  mean seq_geomean                  : {statistics.mean(g):.6f}")
    print(f"  mean sigma_token                  : {statistics.mean(sigma_tok):.5f}")
    print(f"  mean scored tokens per sequence   : {statistics.mean(tok):.1f}")
    print(f"  sigma_seq if tokens independent   : {predicted:.6f}")
    print(f"  sigma_seq OBSERVED                : {observed:.6f}")
    print(f"  ratio                             : {ratio:.2f}x")
    print(f"  effective independent tokens      : {statistics.mean(tok) / ratio ** 2:.1f}")
    p_in = normal_cdf(0.002 / observed) - normal_cdf(-0.001 / observed)
    print(f"  predicted OOB at observed sigma   : {1 - p_in:.4f}")
    print(f"  measured OOB                      : {statistics.mean(oob):.4f}")

  print()
  print("=" * 68)
  print("CLAIM 6  grad_norm tracks advantage_mean, offset one step (tis4)")
  print("=" * 68)
  p = os.path.join(args.logs, "tis4", "orchestrator.log")
  t = os.path.join(args.logs, "tis4", "trainer.log")
  if os.path.exists(p) and os.path.exists(t):
    gn = orchestrator(p, r"grad_norm: ([^ ]+)")
    adv = [float(x) for x in orchestrator(p, r"advantage_mean: ([0-9.-]+)")]
    rew = [float(x) for x in orchestrator(p, r"reward_mean: ([0-9.]+)")]
    oob = sampler_is(t, "tis/is_oob_ratio")
    per_step = [oob[i : i + 4] for i in range(0, len(oob), 4)]
    agree = 0
    print("  step  reward   advantage  is_oob   grad_norm@step+1  consistent")
    for n in range(len(gn) - 1):
      nz_g = gn[n + 1] not in ("0", "N/A")
      nz_a = adv[n] != 0.0
      agree += nz_g == nz_a
      ob = statistics.mean(per_step[n]) if n < len(per_step) else float("nan")
      print(
          f"   {n}    {rew[n]:.4f}   {adv[n]:.4f}     {ob:.4f}   "
          f"{gn[n + 1]:<16} {nz_g == nz_a}"
      )
    print(f"  consistent on {agree}/{len(gn) - 1} pairs")

  print()
  print("=" * 68)
  print("CLAIM 7  there is a live training signal (tis4)")
  print("=" * 68)
  p = os.path.join(args.logs, "tis4", "orchestrator.log")
  if os.path.exists(p):
    for pattern, name in (
        (r"grad_norm: ([^ ]+)", "grad_norm"),
        (r"loss: ([^ ]+)", "loss"),
        (r"reward_mean: ([0-9.]+)", "reward_mean"),
        (r"advantage_mean: ([0-9.-]+)", "advantage_mean"),
    ):
      print(f"  {name:15s}: {orchestrator(p, pattern)}")


if __name__ == "__main__":
  main()
