"""Distributions and closed-form metrics for a set of preemption lambdas
under the step-clock Poisson model (Section 2, 'Direct: Progress Loss').

For each lambda (hazard per MICRO-STEP) over n = k x num_steps micro-steps:

Metrics (printed as a table; step units, seconds via --seconds_per_micro_step):
  mean kill gap        1/lam
  P(kill in a step)    1 - e^{-lam}
  P(zero kills)        e^{-lam n}
  E[N] (m)             (n/m)(e^{lam m} - 1)   -- retries per checkpoint
                       interval of m steps; m = 1 (sub-batch) and m = k
                       (stock, recovery only at applies)
  E[T_discarded] (m)   (n/(m lam))(e^{lam m} - 1 - lam m) micro-steps

Figure (--out): A. one seeded schedule per lambda (what a --preempt_at
schedule from poisson_disruption_script.py looks like); B. kills-per-run
across seeds vs the Poisson(lam n) pmf.

Whether the savings pay for the snapshots is sub_batch_break_even.py.

Usage:
  python poisson_distribution.py --micro_steps_per_step 64 --num_steps 50 \
      --lams 0.00390625,0.0078125,0.0125,0.03125 \
      --seconds_per_micro_step 37.7 --out distribution.png
"""

import argparse
import math

import numpy as np


def draw_points(lam: float, n_total: int, rng: np.random.Generator):
  """Micro-steps (1-based) receiving >= 1 Poisson(lam) arrival -- identical
  construction to poisson_disruption_script.py."""
  return np.flatnonzero(rng.poisson(lam, size=n_total) > 0) + 1


def expected_interrupts(lam: float, n: int, m: int) -> float:
  """E[N] = (n/m)(e^{lam m} - 1), step-clock (t = 1 micro-step)."""
  return (n / m) * (math.exp(lam * m) - 1.0)


def expected_discarded_steps(lam: float, n: int, m: int) -> float:
  """E[T_discarded] = (n/(m lam))(e^{lam m} - 1 - lam m) micro-steps."""
  x = lam * m
  try:
    return (n / (m * lam)) * (math.exp(x) - 1.0 - x)
  except OverflowError:
    return float("inf")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--micro_steps_per_step", type=int, required=True)
  ap.add_argument("--num_steps", type=int, required=True)
  ap.add_argument("--lams", required=True,
                  help="Comma-separated hazards per micro-step.")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--draws", type=int, default=4000,
                  help="Seeds per lambda for the empirical panels.")
  ap.add_argument("--seconds_per_micro_step", type=float, default=None,
                  help="t-hat, to also report seconds (e.g. 37.7).")
  ap.add_argument("--out", default=None,
                  help="Write the 2-panel distribution figure here (.png).")
  args = ap.parse_args()

  k = args.micro_steps_per_step
  n = k * args.num_steps
  lams = [float(v) for v in args.lams.split(",") if v.strip()]
  t_hat = args.seconds_per_micro_step

  def sec(steps):
    return f" ({steps * t_hat:,.0f}s)" if t_hat else ""

  print(f"n = {n} micro-steps ({args.num_steps} steps x k={k})"
        + (f", t-hat = {t_hat}s/micro-step" if t_hat else ""))
  print()
  hdr = ("lambda/step | gap | P(step) | P(0 kills) | E[N] m=1 | E[N] m=k |"
         " E[T_disc] m=1 | E[T_disc] m=k")
  print(hdr)
  print("-" * len(hdr))
  for lam in lams:
    d1 = expected_discarded_steps(lam, n, 1)
    dk = expected_discarded_steps(lam, n, k)
    print(f"{lam:<11g} | {1 / lam:.3g}{sec(1 / lam)} |"
          f" {1 - math.exp(-lam):.4f} | {math.exp(-lam * n):.3g} |"
          f" {expected_interrupts(lam, n, 1):.2f} |"
          f" {expected_interrupts(lam, n, k):.2f} |"
          f" {d1:.3g}{sec(d1)} | {dk:.3g}{sec(dk)}")
  if not args.out:
    return

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lams)))
  fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
  fig.suptitle(
      f"Step-clock Poisson preemption -- n={n} micro-steps"
      f" ({args.num_steps} windows of k={k})", fontsize=12)

  ax = axes[0]
  for row, (lam, c) in enumerate(zip(lams, colors)):
    pts = draw_points(lam, n, np.random.default_rng([args.seed, row]))
    ax.eventplot(pts, lineoffsets=row, linelengths=0.7, colors=[c])
    ax.text(n * 1.02, row, f"lam={lam:g}: {len(pts)} kills (E={lam * n:.0f})",
            va="center", fontsize=8, color=c)
  ax.set_yticks(range(len(lams)))
  ax.set_yticklabels([f"{l:g}" for l in lams])
  ax.set_xlim(0, n * 1.35)
  ax.set_xlabel("micro-step")
  ax.set_ylabel("lambda per micro-step")
  ax.set_title(f"A. One seeded schedule per lambda (seed={args.seed})",
               fontsize=10)

  ax = axes[1]
  for lam, c in zip(lams, colors):
    rng = np.random.default_rng([args.seed, 200, round(1e9 * lam)])
    counts = [len(draw_points(lam, n, rng)) for _ in range(args.draws)]
    lo, hi = min(counts), max(counts)
    ax.hist(counts, bins=np.arange(lo - 0.5, hi + 1.5), density=True,
            histtype="step", color=c)
    mu = lam * n
    ks = np.arange(lo, hi + 1)
    ax.plot(ks, [math.exp(-mu + kk * math.log(mu) - math.lgamma(kk + 1))
                 for kk in ks], ".", color=c, ms=3,
            label=f"lam={lam:g}: Poisson({mu:.1f})")
  ax.set_xlabel(f"kills per run ({args.draws} seeds)")
  ax.set_ylabel("probability")
  ax.set_title("B. Kills per run: Poisson(lam n), no guaranteed count",
               fontsize=10)
  ax.legend(fontsize=7)

  fig.tight_layout(rect=(0, 0, 1, 0.94))
  fig.savefig(args.out, dpi=150)
  print(f"\nwrote {args.out}")


if __name__ == "__main__":
  main()

