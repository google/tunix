"""Render a compact mean-duration overlap diagram for the matched run."""

from html import escape
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent
MODES = ("agentic", "dist")
COLORS = {
    "rollout": "#3977B9",
    "weight_sync": "#259475",
    "training_during": "#CE7625",
    "training_after": "#F1B95C",
    "logprob_during": "#8D59A7",
    "logprob_after": "#C5A3D5",
}


def mean(values):
  return statistics.mean(values)


def interval_overlap(intervals, start, end):
  return sum(max(0.0, min(right, end) - max(left, start))
             for left, right in intervals)


def build_summary():
  timeline = json.loads((ROOT / "training_timeline_summary.json").read_text())
  result = {}
  for mode in MODES:
    steps = timeline[mode]["steps"]
    avg = timeline[mode]["averages"]
    logprob_during = mean([
        interval_overlap(step["intervals_from_step_start_seconds"]["actor_log_probs"],
                         *step["intervals_from_step_start_seconds"]["rollout"][0])
        for step in steps
    ])
    row = {
        "measured_steps": len(steps),
        "full_step_seconds": avg["full_step_seconds"],
        "through_last_trajectory_seconds": avg["last_rollout_from_step_start_seconds"],
        "rollout_collection_seconds": avg["rollout_seconds"],
        "before_sync_seconds": (avg["full_step_seconds"]
                                - avg["last_rollout_from_step_start_seconds"]
                                - avg["weight_sync_seconds"]),
        "weight_sync_seconds": avg["weight_sync_seconds"],
        "training_during_rollout_seconds": avg["training_during_rollout_seconds"],
        "training_after_rollout_seconds": avg["training_after_rollout_seconds"],
        "training_total_seconds": avg["training_seconds"],
        "logprob_during_rollout_seconds": logprob_during,
        "logprob_after_rollout_seconds": avg["actor_log_probs_seconds"] - logprob_during,
        "logprob_total_seconds": avg["actor_log_probs_seconds"],
    }
    assert row["measured_steps"] == 20
    assert abs(sum(row[key] for key in (
        "through_last_trajectory_seconds", "before_sync_seconds",
        "weight_sync_seconds")) - row["full_step_seconds"]) < 1e-6
    assert abs(row["training_during_rollout_seconds"]
               + row["training_after_rollout_seconds"]
               - row["training_total_seconds"]) < 1e-6
    assert abs(row["logprob_during_rollout_seconds"]
               + row["logprob_after_rollout_seconds"]
               - row["logprob_total_seconds"]) < 1e-6
    assert row["before_sync_seconds"] >= 0
    assert row["logprob_after_rollout_seconds"] >= 0
    result[mode] = row
  return result


def draw_chart(data, locale):
  """Compact mean timeline with a subtle E2E endpoint annotation."""
  assert locale in ("en", "zh")
  words = {
      "en": {
          "title": "Training pipeline · 20-step mean",
          "subtitle": "Matched 2+2 TPU · 8 trajectories per microbatch · warmup excluded",
          "rollout": "Rollout collection",
          "training": "Training calls",
          "logprob": "Actor log-prob",
          "sync": "Weight sync",
          "last": "Last trajectory",
          "e2e": "E2E",
          "axis": "Seconds from step start",
          "note": "Darker call bars run during rollout; lighter bars follow it. Their placement is schematic; see the detailed timeline.",
          "comparison": "Distributed vs Agentic",
          "less_time": "less E2E time",
          "throughput": "higher step throughput",
          "names": ("Agentic", "Distributed"),
      },
      "zh": {
          "title": "训练流程 · 20-step 平均耗时",
          "subtitle": "对齐的 2+2 TPU · 每个 microbatch 8 条轨迹 · 去除预热",
          "rollout": "Rollout 收集",
          "training": "训练调用",
          "logprob": "Actor log-prob",
          "sync": "权重同步",
          "last": "末条轨迹",
          "e2e": "E2E",
          "axis": "距 step 开始的秒数",
          "note": "训练与 log-prob 深色条表示 rollout 期间，浅色条表示之后；位置为示意，详图有真实时间戳。",
          "comparison": "Distributed 对比 Agentic",
          "less_time": "E2E 耗时降低",
          "throughput": "step 吞吐量提高",
          "names": ("Agentic", "Distributed"),
      },
  }[locale]
  width, height = 1460, 740
  plot_x, plot_w, max_seconds = 250, 880, 350
  right_x = 1200
  out = [
      f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
      f'viewBox="0 0 {width} {height}" role="img" '
      f'aria-label="{escape(words["title"])}">',
      '<rect width="100%" height="100%" fill="#FFFFFF"/>',
      '<style>text{font-family:Arial,Helvetica,sans-serif}</style>',
  ]

  def label(px, py, value, size=15, color="#263647", weight="normal", anchor=None):
    contents = escape(str(value))
    if locale == "zh":
      runs = []
      for char in str(value):
        cjk = 0x3000 <= ord(char) <= 0x9FFF or 0xFF00 <= ord(char) <= 0xFFEF
        if runs and runs[-1][0] == cjk:
          runs[-1] = (cjk, runs[-1][1] + char)
        else:
          runs.append((cjk, char))
      contents = "".join(
          f'<tspan font-family="{("Droid Sans Fallback" if cjk else "Arial")}">'
          f'{escape(chunk)}</tspan>' for cjk, chunk in runs)
    anchor_attr = f' text-anchor="{anchor}"' if anchor else ""
    out.append(f'<text x="{px}" y="{py}" font-size="{size}" fill="{color}" '
               f'font-weight="{weight}"{anchor_attr}>{contents}</text>')

  def x(seconds):
    return plot_x + seconds * plot_w / max_seconds

  def bar(start, end, center, color):
    assert 0 <= start <= end <= max_seconds
    out.append(f'<rect x="{x(start):.2f}" y="{center - 9}" '
               f'width="{max(.5, x(end) - x(start)):.2f}" height="18" '
               f'rx="3" fill="{color}"/>')

  label(42, 41, words["title"], 26, weight="bold")
  label(42, 68, words["subtitle"], 15, "#657487")
  agentic_time = data["agentic"]["full_step_seconds"]
  dist_time = data["dist"]["full_step_seconds"]
  time_reduction = 100 * (1 - dist_time / agentic_time)
  throughput_gain = 100 * (agentic_time / dist_time - 1)
  out.append('<rect x="1080" y="10" width="338" height="82" rx="12" '
             'fill="#EAF6F1" stroke="#B9DDCE"/>')
  label(1097, 33, words["comparison"], 13, "#476858", "bold")
  label(1097, 58, f"{time_reduction:.2f}% {words['less_time']}",
        18, "#176B50", "bold")
  label(1097, 80, f"+{throughput_gain:.2f}% {words['throughput']}",
        14, "#176B50")
  legend = (
      (42, COLORS["rollout"], words["rollout"]),
      (307, COLORS["training_during"], words["training"]),
      (565, COLORS["logprob_during"], words["logprob"]),
      (830, COLORS["weight_sync"], words["sync"]),
  )
  for left, color, name in legend:
    out.append(f'<circle cx="{left + 7}" cy="102" r="7" fill="{color}"/>')
    label(left + 22, 107, name, 14)
  out.append('<line x1="42" y1="376" x2="1408" y2="376" stroke="#E2E8EF"/>')

  for mode, header_y, centers in (
      ("agentic", 165, (192, 240, 288, 336)),
      ("dist", 415, (442, 490, 538, 586)),
  ):
    v = data[mode]
    last = v["through_last_trajectory_seconds"]
    full = v["full_step_seconds"]
    label(42, header_y, words["names"][0 if mode == "agentic" else 1],
          20, weight="bold")
    label(x(last) - 8, header_y,
          f"{words['last']} {last:.2f} s", 14, "#61738A", anchor="end")
    for key, cy in zip(("rollout", "training", "logprob", "sync"), centers):
      label(42, cy + 5, words[key], 15)
      out.append(f'<line x1="{plot_x}" y1="{cy}" '
                 f'x2="{plot_x + plot_w}" y2="{cy}" '
                 f'stroke="#E8EEF4" stroke-width="18"/>')
    bar(0, last, centers[0], COLORS["rollout"])
    bar(last - v["training_during_rollout_seconds"], last,
        centers[1], COLORS["training_during"])
    bar(last, last + v["training_after_rollout_seconds"],
        centers[1], COLORS["training_after"])
    bar(last - v["logprob_during_rollout_seconds"], last,
        centers[2], COLORS["logprob_during"])
    bar(last, last + v["logprob_after_rollout_seconds"],
        centers[2], COLORS["logprob_after"])
    bar(last + v["before_sync_seconds"], full,
        centers[3], COLORS["weight_sync"])
    for cy, total, detail in (
        (centers[0], v["rollout_collection_seconds"], None),
        (centers[1], v["training_total_seconds"],
         (v["training_during_rollout_seconds"], v["training_after_rollout_seconds"])),
        (centers[2], v["logprob_total_seconds"],
         (v["logprob_during_rollout_seconds"], v["logprob_after_rollout_seconds"])),
        (centers[3], v["weight_sync_seconds"], None),
    ):
      label(right_x, cy + (2 if detail else 5), f"{total:.2f} s", 16,
            weight="bold")
      if detail:
        label(right_x, cy + 17, f"{detail[0]:.2f} + {detail[1]:.2f}",
              11, "#748397")
    marker = x(last)
    out.append(f'<line x1="{marker:.2f}" y1="{centers[0] - 16}" '
               f'x2="{marker:.2f}" y2="{centers[3] + 16}" '
               f'stroke="#54718E" stroke-width="1.5" '
               f'stroke-dasharray="4,5"/>')
    end_x = x(full)
    out.append(f'<circle cx="{end_x:.2f}" cy="{centers[3]}" '
               f'r="4.5" fill="#1B765F" stroke="#FFFFFF" stroke-width="1.5"/>')
    label(end_x + 10, centers[3] + 5,
          f"{words['e2e']} {full:.2f} s", 14, "#1B765F", "bold")

  for tick in range(0, max_seconds + 1, 50):
    px = x(tick)
    out.append(f'<line x1="{px:.2f}" y1="637" x2="{px:.2f}" '
               f'y2="643" stroke="#AFC0D0"/>')
    label(px, 664, tick, 13, "#6E7F91", anchor="middle")
  out.append(f'<line x1="{plot_x}" y1="637" '
             f'x2="{plot_x + plot_w}" y2="637" stroke="#B8C7D5"/>')
  label(plot_x, 696, words["axis"], 15, "#51647A", "bold")
  label(plot_x, 723, words["note"], 13, "#6C7E90")
  out.append('</svg>')
  return "\n".join(out) + "\n"


if __name__ == "__main__":
  data = build_summary()
  (ROOT / "training_average_summary.json").write_text(json.dumps(data, indent=2) + "\n")
  import cairosvg  # Optional dependency for PNG export.
  for locale, suffix in (("en", ""),):
    svg = ROOT / f"training_average{suffix}.svg"
    png = ROOT / f"training_average{suffix}.png"
    svg.write_text(draw_chart(data, locale))
    cairosvg.svg2png(url=str(svg), write_to=str(png),
                     output_width=2920, output_height=1480)
    print(svg, png)
