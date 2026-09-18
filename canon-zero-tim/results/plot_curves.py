#!/usr/bin/env python3
"""Draw one workload's arms straight from the raw exports under results/.

    python3 canon-zero-tim/results/plot_curves.py --workload frozenlake-short-horizon

Reads every results/<workload>/<arm>/<run-id>/history.csv by ascending _step and
writes <workload>/curves.svg -- left panel the mean sampler-trainer logprob gap,
right panel the raw solve rate faint under its bold ten-update trailing mean --
plus <workload>/summary.csv, one row per run.  Arms are drawn standard, tis,
zero_tim, then the rest in name order; an arm that never logged a metric is
simply absent from that panel.  The trailing mean is moving_average() imported
from the Figure 4 builder, so a curve here smooths exactly the way Figure 4
does; everything else is the standard library.
"""
import argparse
import csv
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "blog_reprod"))
from build_end_to_end_results_provisional import Point, moving_average  # noqa: E402

LOGP = "sampler_trainer/train/logp_diff_mean"
SOLVE = "rewards/train/solve_ratio"
ARM_ORDER = ("standard", "tis", "zero_tim")
COLORS = {"standard": "#5f6368", "tis": "#e37400", "zero_tim": "#0b57d0"}
SPARE = ("#188038", "#a142f4", "#c5221f", "#12b5cb", "#9aa0a6")
WIDTH, HEIGHT = 1280, 430
BOXES = ((96.0, 138.0, 596.0, 342.0), (700.0, 138.0, 1200.0, 342.0))
SOLVE_LABELS = ((0.0, "0%"), (0.25, "25%"), (0.5, "50%"), (0.75, "75%"), (1.0, "100%"))


class Run:
  """One exported run: its two plotted series and the trailing mean of one."""

  def __init__(self, arm, run_id, steps, logp, solve):
    self.arm, self.run_id, self.steps = arm, run_id, steps
    self.logp, self.solve = logp, solve
    self.trail = moving_average(solve) if solve else ()
    self.label, self.color = arm, COLORS.get(arm, SPARE[-1])


def read_run(directory):
  """Returns (row count, {metric: points}) for one run, ascending _step."""
  with (directory / "history.csv").open(newline="", encoding="utf-8") as handle:
    rows = [row for row in csv.DictReader(handle) if (row.get("_step") or "") != ""]
  rows.sort(key=lambda row: float(row["_step"]))
  series = {key: tuple(Point(int(float(row["_step"])) + 1, float(row[key]))
                       for row in rows if (row.get(key) or "") != "")
            for key in (LOGP, SOLVE)}
  return len(rows), series


def load_workload(directory):
  """Every run under a workload directory, in drawing order."""
  runs = []
  for arm in sorted(p for p in directory.iterdir() if p.is_dir() and p.name[0] not in "._"):
    for run in sorted(p for p in arm.iterdir() if p.is_dir() and p.name[0] not in "._"):
      steps, series = read_run(run)
      runs.append(Run(arm.name, run.name, steps, series[LOGP], series[SOLVE]))
  runs.sort(key=lambda run: (ARM_ORDER.index(run.arm) if run.arm in ARM_ORDER
                             else len(ARM_ORDER), run.arm, run.run_id))
  spare = iter(SPARE)
  for index, run in enumerate(runs):
    if run.arm not in COLORS:
      run.color = next(spare, SPARE[-1])
    if sum(1 for other in runs if other.arm == run.arm) > 1:
      run.label = f"{run.arm} {run.run_id.split('_')[0]}"
      run.color = SPARE[index % len(SPARE)]
  return runs


def text(x, y, value, css, anchor="start"):
  return (f'<text class="{css}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">'
          f'{html.escape(value, quote=True)}</text>')


def place(point, box, axis_max, low, high):
  x0, y0, x1, y1 = box
  return (x0 + point.update * (x1 - x0) / axis_max,
          y1 - (point.value - low) * (y1 - y0) / (high - low))


def curve(points, box, axis_max, low, high, color, width, opacity):
  path = " ".join(f"{'M' if index == 0 else 'L'} {x:.1f} {y:.1f}" for index, (x, y)
                  in enumerate(place(p, box, axis_max, low, high) for p in points))
  return (f'<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}" '
          f'stroke-opacity="{opacity}" stroke-linecap="round" stroke-linejoin="round"/>')


def axes(box, axis_max, low, high, labels):
  x0, y0, x1, y1 = box
  parts = []
  for value, caption in labels:
    y = y1 - (value - low) * (y1 - y0) / (high - low)
    parts.append(f'<line class="grid" x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}"/>')
    parts.append(text(x0 - 12, y + 4, caption, "axis-label", "end"))
  for tick in sorted({0, axis_max // 4, axis_max // 2, 3 * axis_max // 4, axis_max}):
    x = x0 + tick * (x1 - x0) / axis_max
    parts.append(f'<line class="tick" x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y1 + 5:.1f}"/>')
    parts.append(text(x, y1 + 22, str(tick), "axis-label", "middle"))
  return "\n  ".join(parts)


def render_svg(workload, runs):
  """The two-panel figure, as SVG text."""
  axis_max = max(run.steps for run in runs) or 1
  top = max([point.value for run in runs for point in run.logp], default=0.0) * 1.15 or 0.001
  left = axes(BOXES[0], axis_max, 0.0, top, tuple((top * i / 4, "%.4g" % (top * i / 4))
                                                 for i in range(5)))
  right = axes(BOXES[1], axis_max, 0.0, 1.0, SOLVE_LABELS)
  legend, x = [], 96.0
  for run in runs:
    legend.append(f'<line x1="{x:.1f}" y1="27" x2="{x + 26:.1f}" y2="27" '
                  f'stroke="{run.color}" stroke-width="3" stroke-linecap="round"/>')
    legend.append(text(x + 35, 32, run.label, "legend"))
    x += 55 + 7.6 * len(run.label)
  for run in runs:
    if run.logp:
      left += "\n  " + curve(run.logp, BOXES[0], axis_max, 0.0, top, run.color, 2.5, 1)
    if run.solve:
      right += "\n  " + curve(run.solve, BOXES[1], axis_max, 0.0, 1.0, run.color, 1.2, 0.18)
      right += "\n  " + curve(run.trail, BOXES[1], axis_max, 0.0, 1.0, run.color, 2.5, 1)
  return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" role="img" aria-labelledby="title desc">
  <title id="title">{html.escape(workload)}: logprob gap and solve rate per policy update</title>
  <desc id="desc">Two panels over the same x axis of policy updates. Left: the mean absolute sampler-trainer logprob difference per arm. Right: the raw training solve rate, faint, under its ten-update trailing mean, bold.</desc>
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; }}
    .figure-title {{ font-size: 20px; font-weight: 600; fill: #202124; }}
    .panel-title {{ font-size: 15px; font-weight: 600; fill: #3c4043; }}
    .axis-label {{ font-size: 12px; fill: #5f6368; }}
    .legend {{ font-size: 13px; fill: #3c4043; }}
    .grid {{ stroke: #e8eaed; stroke-width: 1; }}
    .tick {{ stroke: #9aa0a6; stroke-width: 1; }}
  </style>
  <rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>
  {''.join(legend)}
  {text(WIDTH / 2, 72, workload, "figure-title", "middle")}
  {text(BOXES[0][0], 116, "Mean |Δ logprob|", "panel-title")}
  {text(BOXES[1][0], 116, "Solve rate (faint: raw, bold: trailing 10)", "panel-title")}
  {left}
  {right}
  {text(WIDTH / 2, 411, "policy update", "axis-label", "middle")}
</svg>
'''


def write_summary(runs, path):
  """One row per run: the endpoints a reader would otherwise read off the curves."""
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(["arm", "run_id", "steps", "last_solve_ratio", "trailing10_last",
                     "logp_diff_mean_max"])
    for run in runs:
      writer.writerow([run.arm, run.run_id, run.steps,
                       "%.16g" % run.solve[-1].value if run.solve else "",
                       "%.16g" % run.trail[-1].value if run.trail else "",
                       "%.16g" % max(p.value for p in run.logp) if run.logp else ""])


def main(argv=None):
  parser = argparse.ArgumentParser(description="Plot one workload's arms from results/.")
  parser.add_argument("--workload", required=True, help="a directory name under results/")
  parser.add_argument("--out", type=Path, default=None,
                      help="where to write curves.svg and summary.csv (default: the workload)")
  args = parser.parse_args(argv)
  directory = ROOT / args.workload
  if not directory.is_dir():
    raise SystemExit(f"FAIL: no workload directory {directory}")
  runs = load_workload(directory)
  if not runs:
    raise SystemExit(f"FAIL: {args.workload} holds no run directories")
  out = args.out or directory
  out.mkdir(parents=True, exist_ok=True)
  (out / "curves.svg").write_text(render_svg(args.workload, runs), encoding="utf-8")
  write_summary(runs, out / "summary.csv")
  print(f"PLOT_CURVES PASS workload={args.workload} runs={len(runs)} "
        f"updates={max(run.steps for run in runs)} out={out}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
