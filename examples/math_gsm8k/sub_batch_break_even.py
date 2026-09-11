"""Break-even analysis for sub-batch checkpointing under the step-clock
Poisson preemption model: does the redo work it saves pay for its snapshots?

Closed form, no random draws. Distributions and schedules are in
poisson_distribution.py; the E[N] / E[T_discarded] formulas are imported
from there.

For n = k x num_steps micro-steps at hazard lam per micro-step:
  recovery savings     (E[T_disc](k) - E[T_disc](1)) t-hat
                       Closed-form expectation of discarded micro-steps under
                       Poisson preemption across windows of size k vs 1.
  overhead             n delta_mid + (n/k) delta_bnd
  net benefit B(lam)   savings - overhead. lam* is the break-even hazard,
                       also given as a mean kill gap G* = t-hat / lam*.
  Young tau*           sqrt(2 delta / lam) micro-steps, delta = delta_mid /
                       t-hat (the marginal cost of one more snapshot)

Overhead inputs: --delta_mid_s (seconds one mid-window snapshot adds on the
critical path) and --delta_bnd_s (seconds added per window boundary), or a
JSON file via --overhead_stats with keys t_hat_s, delta_mid_s, delta_bnd_s
(flags override the file). Measured at k=16, 8096 tokens: delta_mid ~= 0.071,
delta_bnd ~= 12.36 (median-diffs, enabled minus disabled); at k=64 the
boundary is ~110 (synchronous deletion of the previous window's snapshots).

Figure (--out): net time saved per run vs lambda -- both savings semantics,
the overhead line, the requested lambdas as dots, lam* marked.

Usage:
  python sub_batch_break_even.py --micro_steps_per_step 64 --num_steps 50 \
      --lams 0.00390625,0.0078125,0.0125,0.03125 \
      --seconds_per_micro_step 37.7 --delta_mid_s 0.07 --delta_bnd_s 110 \
      --out break_even.png
"""

import argparse
import json
import math
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
  sys.path.insert(0, _THIS_DIR)

try:
  from .poisson_distribution import expected_discarded_steps  # pytype: disable=import-error
except (ImportError, ValueError):
  from poisson_distribution import expected_discarded_steps  # pytype: disable=import-error


def young_tau_steps(delta_steps: float, lam: float) -> float:
  """Young 1974: tau* = sqrt(2 delta / lam), all in micro-steps."""
  return math.sqrt(2.0 * delta_steps / lam)


def recovery_savings(lam: float, n: int, k: int, t_hat: float) -> float:
  """Seconds of discarded work stock (m=k) loses beyond sub-batch (m=1)."""
  return (expected_discarded_steps(lam, n, k)
          - expected_discarded_steps(lam, n, 1)) * t_hat


def overhead_per_run(n: int, k: int, delta_mid: float,
                     delta_bnd: float) -> float:
  """Seconds the snapshots add to a whole run, kills or not."""
  return n * delta_mid + (n / k) * delta_bnd


def break_even_lambda(savings, overhead: float, lo: float = 1e-9,
                      hi: float = 1.0):
  """Smallest lam with savings(lam) >= overhead (savings grow with lam);
  None if even lam=hi does not pay for the overhead."""
  if savings(hi) < overhead:
    return None
  for _ in range(200):
    mid = math.sqrt(lo * hi)  # geometric bisection: lam spans decades
    if savings(mid) < overhead:
      lo = mid
    else:
      hi = mid
  return hi


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--micro_steps_per_step", type=int, required=True)
  ap.add_argument("--num_steps", type=int, required=True)
  ap.add_argument("--lams", required=True,
                  help="Comma-separated hazards per micro-step.")
  ap.add_argument("--seconds_per_micro_step", type=float, default=None,
                  help="t-hat (e.g. 37.7); overrides the JSON.")
  ap.add_argument("--delta_mid_s", type=float, default=None,
                  help="Seconds one mid-window snapshot adds (e.g. 0.07).")
  ap.add_argument("--delta_bnd_s", type=float, default=None,
                  help="Seconds added per window boundary (e.g. 110).")
  ap.add_argument("--overhead_stats", default=None,
                  help="JSON with t_hat_s, delta_mid_s, delta_bnd_s;"
                  " the flags above override it.")
  ap.add_argument("--out", default=None,
                  help="Write the net-benefit figure here (.png).")
  args = ap.parse_args()

  k = args.micro_steps_per_step
  n = k * args.num_steps
  lams = [float(v) for v in args.lams.split(",") if v.strip()]
  stats = {}
  if args.overhead_stats:
    with open(args.overhead_stats) as f:
      stats = json.load(f)
  t_hat = args.seconds_per_micro_step or stats.get("t_hat_s")
  delta_mid = (args.delta_mid_s if args.delta_mid_s is not None
               else stats.get("delta_mid_s"))
  delta_bnd = (args.delta_bnd_s if args.delta_bnd_s is not None
               else stats.get("delta_bnd_s"))
  if not t_hat or delta_mid is None or delta_bnd is None:
    ap.error("need t-hat, delta_mid and delta_bnd (flags or --overhead_stats)")

  def hrs(seconds):
    return f"{seconds / 3600:,.2f}h"

  overhead = overhead_per_run(n, k, delta_mid, delta_bnd)
  savings_fn = lambda lam: recovery_savings(lam, n, k, t_hat)

  print(f"n = {n} micro-steps ({args.num_steps} steps x k={k}),"
        f" t-hat = {t_hat}s/micro-step")
  print(f"overhead per run = n*delta_mid + (n/k)*delta_bnd ="
        f" {n}*{delta_mid:g} + {n // k}*{delta_bnd:g} = {overhead:,.0f}s"
        f" ({hrs(overhead)}, {overhead / (n * t_hat):.2%} of {hrs(n * t_hat)}"
        " clean)")
  print()
  hdr = "lambda/step | E[kills] | dT saved | Net Benefit B"
  print(hdr)
  print("-" * len(hdr))
  for lam in lams:
    s = savings_fn(lam)
    print(f"{lam:<11g} | {n * (1 - math.exp(-lam)):.1f} |"
          f" {hrs(s)} | {hrs(s - overhead)}")
  lam_star = break_even_lambda(savings_fn, overhead)
  if lam_star is None:
    print("break-even: none below lam=1")
  else:
    print(f"break-even: lam* = {lam_star:.3g}/micro-step, i.e."
          f" one kill per {t_hat / lam_star:,.0f}s"
          f" ({hrs(t_hat / lam_star)}); enable when kills come sooner.")
  print()
  for lam in lams:
    tau = young_tau_steps(delta_mid / t_hat, lam)
    print(f"lambda={lam:g}: Young tau* ~= {tau:.1f} micro-steps"
          f" ({tau * t_hat:,.0f}s) between checkpoints"
          f" (sub-batch: 1; stock: k={k})")

  if not args.out:
    return

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lams)))
  fig, ax = plt.subplots(figsize=(8, 5.5))
  lam_star = break_even_lambda(savings_fn, overhead)
  lo = lam_star / 4 if lam_star is not None else min(lams) / 10
  grid = np.geomspace(lo, max(lams) * 3, 400)
  h_savings = np.array([savings_fn(l) for l in grid]) / 3600
  h_over = overhead / 3600
  ax.plot(grid, h_savings, color="tab:blue", lw=1.5,
          label="recovery savings (Poisson preemption)")
  ax.axhline(h_over, color="black", ls=":", lw=1,
             label=f"snapshot overhead ({overhead:,.0f}s / {hrs(overhead)})")
  ax.axhline(0, color="gray", lw=0.5)
  ax.plot(grid, h_savings - h_over, color="tab:green", lw=2,
          label="net benefit B (savings - overhead)")
  for lam, c in zip(lams, colors):
    ax.plot([lam], [savings_fn(lam) / 3600], "o", color=c, ms=6, zorder=5,
            label=f"lam={lam:g}")
  if lam_star is not None:
    ax.axvline(lam_star, color="tab:green", lw=1, alpha=0.7)
    ax.annotate(
        f"lam*={lam_star:.2g}: one kill per {hrs(t_hat / lam_star)}",
        xy=(lam_star, h_over), xytext=(12, -36), textcoords="offset points",
        fontsize=8, color="tab:green",
        arrowprops=dict(arrowstyle="-", color="tab:green", lw=0.8))
  ax.set_xscale("log")
  ax.set_yscale("symlog", linthresh=max(h_over, 1e-3))
  ax.set_ylim(-1.5 * h_over, max(savings_fn(l) for l in lams) / 3600 * 1.5)
  ax.set_xlabel("lambda per micro-step (log)")
  ax.set_ylabel("hours per run (symlog)")
  ax.set_title(
      f"Net time saved per run vs lambda -- n={n} ({args.num_steps} windows"
      f" of k={k}), t-hat={t_hat}s", fontsize=10)
  ax.legend(fontsize=7, loc="upper left")
  fig.tight_layout()
  fig.savefig(args.out, dpi=150)
  print(f"\nwrote {args.out}")


if __name__ == "__main__":
  main()