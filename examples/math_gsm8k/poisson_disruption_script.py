"""Draws exact preemption points from a Poisson process on the micro-step
axis, ready to paste into the benchmark harness's --preempt_at.

The hazard is lambda per MICRO-STEP: numpy draws an independent
Poisson(lam) arrival count for every micro-step 1..n (n = micro-steps per
global step x number of steps), and every step with >= 1 arrival becomes
one preemption point (arrivals collapse within a step -- only one kill can
hit a live process; equivalently, per-step kill probability 1 - e^{-lam}).

By default, all drawn points are retained (including apply boundaries,
i.e. multiples of k) to faithfully represent the true, unconditioned
Poisson distribution. Pass --drop_boundaries if testing purely mid-window
preemption.

Usage:
 python poisson_preempt_points.py \
     --micro_steps_per_step 64 --num_steps 50 --lam 0.004 --seed 0
"""

import argparse
import numpy as np


def draw_points(lam: float, n_total: int, rng: np.random.Generator):
 """Micro-steps (1-based) receiving >= 1 Poisson(lam) arrival."""
 hits = rng.poisson(lam, size=n_total)
 return [int(i) + 1 for i in np.flatnonzero(hits > 0)]


def main() -> None:
 ap = argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--micro_steps_per_step", type=int, required=True,
                 help="k: micro-steps per mini/global step (window size).")
 ap.add_argument("--num_steps", type=int, required=True,
                 help="Total number of global steps in the run.")
 ap.add_argument("--lam", type=float, required=True,
                 help="Preemption hazard per micro-step (e.g. 0.0125 ~="
                 " one kill per 80 micro-steps on average).")
 ap.add_argument("--seed", type=int, default=0)
 ap.add_argument("--drop_boundaries", action="store_true",
                 help="Drop points landing exactly on apply boundaries"
                 " (multiples of k). Default: False (keep all points).")
 args = ap.parse_args()

 k, n = args.micro_steps_per_step, args.micro_steps_per_step * args.num_steps
 rng = np.random.default_rng(args.seed)
 points = draw_points(args.lam, n, rng)

 boundary = [p for p in points if p % k == 0]
 if boundary:
   if args.drop_boundaries:
     points = [p for p in points if p % k != 0]
     print(f"NOTE: dropped {len(boundary)} boundary point(s) {boundary}"
           " (multiples of k) per --drop_boundaries.")
   else:
     print(f"NOTE: retained {len(boundary)} boundary point(s) {boundary}"
           " (multiples of k) for true unconditioned Poisson distribution.")

 print(f"n = {n} micro-steps ({args.num_steps} steps x k={k}),"
       f" lam = {args.lam:g}/micro-step, seed = {args.seed}")
 print(f"drew {len(points)} preemption point(s)"
       f" (expected ~{args.lam * n:.1f}, count ~ Poisson({args.lam * n:.1f}))")
 print(f"points: {points}")
 print()
 print(f"--preempt_at={','.join(str(p) for p in points)}")
 print(f"--max_restarts={len(points) + 2}  # one restart per point + slack")

if __name__ == "__main__":
 main()