#!/usr/bin/env python3
"""Render the publication-layout Figure 4 mockup from three local histories.

The renderer reads the observed mean absolute sampler–trainer logprob gap and
solve rate for all recipes.  ``--assume-certified-zero`` replaces only the
aligned-run gap series with an explicit zero-valued layout placeholder.  That
mode is for editorial preview and must not be published until a strict
zero-byte run supplies the same series.

Inside blog_reprod/ this module is imported only as the drawing library used by
build_end_to_end_results_measured.py (Point, Workload, COLORS, moving_average,
_panel, _text). It is not run directly to rebuild Figure 4.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REWARD_KEY = "rewards/train/solve_ratio"
LOGP_DIFF_KEY = "sampler_trainer/train/logp_diff_mean"
RECIPES = ("native", "importance_sampling", "zero_tim")
COLORS = {
    "native": "#5f6368",
    "importance_sampling": "#e37400",
    "zero_tim": "#0b57d0",
}
LABELS = {
    "native": "Standard IS",
    "importance_sampling": "Token-level TIS",
    "zero_tim": "Zero-TIM",
}


@dataclass(frozen=True)
class Point:
    update: int
    value: float


@dataclass(frozen=True)
class Workload:
    title: str
    subtitle: str
    axis_max: int
    observed_updates: int
    logp_diff: dict[str, tuple[Point, ...]]
    solve_rate: dict[str, tuple[Point, ...]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(raw: str, *, path: Path, row: int, key: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{path}:{row}: {key} is not numeric") from exc
    _require(math.isfinite(value), f"{path}:{row}: {key} is not finite")
    return value


def read_series(path: Path, key: str, limit: int) -> tuple[Point, ...]:
    """Read W&B rows as one-based policy-update counts up to ``limit``."""
    _require(limit > 0, "update limit must be positive")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None, f"{path}: missing CSV header")
        _require("_step" in reader.fieldnames, f"{path}: missing _step")
        _require(key in reader.fieldnames, f"{path}: missing {key}")
        points: list[Point] = []
        for row_number, row in enumerate(reader, 2):
            raw_step = row.get("_step", "")
            raw_value = row.get(key, "")
            if raw_step == "" or raw_value == "":
                continue
            step_value = _number(raw_step, path=path, row=row_number, key="_step")
            _require(step_value.is_integer(), f"{path}:{row_number}: _step is not integral")
            update = int(step_value) + 1
            if update > limit:
                continue
            value = _number(raw_value, path=path, row=row_number, key=key)
            points.append(Point(update, value))

    _require(points, f"{path}: no values for {key} in the first {limit} updates")
    updates = [point.update for point in points]
    _require(updates == sorted(set(updates)), f"{path}: {key} updates are not unique and ordered")
    return tuple(points)


def _clip(points: tuple[Point, ...], limit: int) -> tuple[Point, ...]:
    clipped = tuple(point for point in points if point.update <= limit)
    _require(clipped, "display window removed every point")
    return clipped


def load_workload(
    *,
    title: str,
    subtitle: str,
    native: Path,
    importance_sampling: Path,
    zero_tim: Path,
    axis_max: int,
    assume_certified_zero: bool,
) -> Workload:
    sources = {
        "native": native,
        "importance_sampling": importance_sampling,
        "zero_tim": zero_tim,
    }
    raw_logp = {
        recipe: read_series(path, LOGP_DIFF_KEY, axis_max)
        for recipe, path in sources.items()
    }
    raw_solve = {
        recipe: read_series(path, REWARD_KEY, axis_max)
        for recipe, path in sources.items()
    }

    observed_updates = min(
        raw_logp["zero_tim"][-1].update,
        raw_solve["zero_tim"][-1].update,
        axis_max,
    )
    logp_diff = {
        recipe: _clip(points, observed_updates)
        for recipe, points in raw_logp.items()
    }
    solve_rate = {
        recipe: _clip(points, observed_updates)
        for recipe, points in raw_solve.items()
    }

    for recipe, points in logp_diff.items():
        _require(
            all(point.value >= 0 for point in points),
            f"{title}: {recipe} mean absolute logprob difference is negative",
        )
    for recipe, points in solve_rate.items():
        _require(
            all(0 <= point.value <= 1 for point in points),
            f"{title}: {recipe} solve rate is outside [0, 1]",
        )

    if assume_certified_zero:
        logp_diff["zero_tim"] = tuple(
            replace(point, value=0.0) for point in logp_diff["zero_tim"]
        )

    return Workload(
        title=title,
        subtitle=subtitle,
        axis_max=axis_max,
        observed_updates=observed_updates,
        logp_diff=logp_diff,
        solve_rate=solve_rate,
    )


def moving_average(points: tuple[Point, ...], window: int = 10) -> tuple[Point, ...]:
    _require(window > 0, "moving-average window must be positive")
    result: list[Point] = []
    for index, point in enumerate(points):
        left = max(0, index - window + 1)
        values = [item.value for item in points[left : index + 1]]
        result.append(Point(point.update, sum(values) / len(values)))
    return tuple(result)


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _text(x: float, y: float, value: str, css: str, anchor: str = "start") -> str:
    return (
        f'<text class="{css}" x="{x:.1f}" y="{y:.1f}" '
        f'text-anchor="{anchor}">{_escape(value)}</text>'
    )


def _x(update: int, axis_max: int, x0: float, x1: float) -> float:
    return x0 + update * (x1 - x0) / axis_max


def _y(value: float, minimum: float, maximum: float, y0: float, y1: float) -> float:
    return y1 - (value - minimum) * (y1 - y0) / (maximum - minimum)


def _path(
    points: tuple[Point, ...],
    *,
    axis_max: int,
    y_min: float,
    y_max: float,
    box: tuple[float, float, float, float],
) -> str:
    x0, y0, x1, y1 = box
    return " ".join(
        f"{'M' if index == 0 else 'L'} {_x(point.update, axis_max, x0, x1):.1f} "
        f"{_y(point.value, y_min, y_max, y0, y1):.1f}"
        for index, point in enumerate(points)
    )


def _axes(
    *,
    box: tuple[float, float, float, float],
    axis_max: int,
    y_min: float,
    y_max: float,
    y_labels: tuple[tuple[float, str], ...],
    show_x_labels: bool,
) -> str:
    x0, y0, x1, y1 = box
    parts: list[str] = []
    for value, label in y_labels:
        y = _y(value, y_min, y_max, y0, y1)
        parts.append(f'<line class="grid" x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}"/>')
        parts.append(_text(x0 - 12, y + 4, label, "axis-label", "end"))
    for tick in (0, axis_max // 4, axis_max // 2, 3 * axis_max // 4, axis_max):
        x = _x(tick, axis_max, x0, x1)
        parts.append(f'<line class="tick" x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y1 + 5:.1f}"/>')
        if show_x_labels:
            parts.append(_text(x, y1 + 22, str(tick), "axis-label", "middle"))
    return "\n".join(parts)


def _panel(
    *,
    workload: Workload,
    series: dict[str, tuple[Point, ...]],
    y_min: float,
    y_max: float,
    y_labels: tuple[tuple[float, str], ...],
    box: tuple[float, float, float, float],
    show_x_labels: bool,
    show_raw: bool,
    smooth: bool,
) -> str:
    x0, y0, x1, y1 = box
    pieces = [
        _axes(
            box=box,
            axis_max=workload.axis_max,
            y_min=y_min,
            y_max=y_max,
            y_labels=y_labels,
            show_x_labels=show_x_labels,
        ),
    ]
    if show_raw:
        for recipe in RECIPES:
            raw_path = _path(
                series[recipe],
                axis_max=workload.axis_max,
                y_min=y_min,
                y_max=y_max,
                box=box,
            )
            pieces.append(
                f'<path d="{raw_path}" fill="none" stroke="{COLORS[recipe]}" '
                'stroke-opacity="0.18" stroke-width="1.2" stroke-linecap="round" '
                'stroke-linejoin="round"/>'
            )
    for recipe in RECIPES:
        foreground = moving_average(series[recipe]) if smooth else series[recipe]
        path = _path(
            foreground,
            axis_max=workload.axis_max,
            y_min=y_min,
            y_max=y_max,
            box=box,
        )
        width = 3.2 if recipe == "zero_tim" else 2.5
        pieces.append(
            f'<path d="{path}" fill="none" stroke="{COLORS[recipe]}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    return "\n".join(pieces)


def render_svg(workload: Workload, *, assumed_zero: bool) -> str:
    width, height = 1280, 430
    left_x, right_x = 96.0, 700.0
    panel_width = 500.0
    panel_y0, panel_y1 = 138.0, 342.0

    legend: list[str] = []
    legend_x = 365.0
    for recipe in RECIPES:
        legend.append(
            f'<line x1="{legend_x:.1f}" y1="27" x2="{legend_x + 26:.1f}" y2="27" '
            f'stroke="{COLORS[recipe]}" stroke-width="3" stroke-linecap="round"/>'
        )
        legend.append(_text(legend_x + 35, 32, LABELS[recipe], "legend"))
        legend_x += {"native": 148, "importance_sampling": 182, "zero_tim": 0}[recipe]

    metadata = (
        "Editorial layout assumes a certified zero-valued Zero-TIM series; "
        "do not publish before strict receipt replacement."
        if assumed_zero
        else "Observed local history values."
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-labelledby="title desc">
  <title id="title">Mean logprob difference and solve rate during FrozenLake training</title>
  <desc id="desc">Two side-by-side panels compare Standard IS, token-level truncated importance sampling, and Zero-TIM on one FrozenLake workload. The left panel shows the unsmoothed per-update mean absolute sampler–trainer logprob difference. The right panel shows raw solve-rate observations and a ten-update moving average. Both panels cover the first 120 policy updates.</desc>
  <metadata>{_escape(metadata)}</metadata>
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; }}
    .figure-title {{ font-size: 20px; font-weight: 600; fill: #202124; }}
    .configuration {{ font-size: 12px; fill: #5f6368; }}
    .panel-title {{ font-size: 15px; font-weight: 600; fill: #3c4043; }}
    .axis-label {{ font-size: 12px; fill: #5f6368; }}
    .axis-title {{ font-size: 13px; fill: #5f6368; }}
    .legend {{ font-size: 13px; fill: #3c4043; }}
    .grid {{ stroke: #e8eaed; stroke-width: 1; }}
    .tick {{ stroke: #9aa0a6; stroke-width: 1; }}
  </style>
  <rect width="{width}" height="{height}" fill="#ffffff"/>
  {''.join(legend)}
  {_text(width / 2, 72, workload.title, "figure-title", "middle")}
  {_text(left_x, 116, "Mean |Δ logprob|", "panel-title")}
  {_text(right_x, 116, "Solve rate", "panel-title")}
  {_panel(workload=workload, series=workload.logp_diff, y_min=-0.002, y_max=0.03, y_labels=((0.0, "0.000"), (0.005, "0.005"), (0.01, "0.010"), (0.015, "0.015"), (0.02, "0.020"), (0.025, "0.025"), (0.03, "0.030")), box=(left_x, panel_y0, left_x + panel_width, panel_y1), show_x_labels=True, show_raw=False, smooth=False)}
  {_panel(workload=workload, series=workload.solve_rate, y_min=0.3, y_max=0.9, y_labels=((0.3, "30%"), (0.4, "40%"), (0.5, "50%"), (0.6, "60%"), (0.7, "70%"), (0.8, "80%"), (0.9, "90%")), box=(right_x, panel_y0, right_x + panel_width, panel_y1), show_x_labels=True, show_raw=True, smooth=True)}
  {_text(width / 2, 411, "Qwen3-8B · up to 5 interaction turns · max context 6,144 tokens · max prompt 4,096 · max generation 2,048", "configuration", "middle")}
</svg>
'''


def render_html(svg_name: str, *, assumed_zero: bool) -> str:
    editorial_note = (
        '<aside><strong>Editorial mockup.</strong> The Zero-TIM mismatch line is a '
        'zero-valued layout assumption. Replace it with a strict certified run before publication.'
        '</aside>'
        if assumed_zero
        else ""
    )
    caption = (
        "Mean absolute sampler–trainer logprob difference and solve rate over the first 120 "
        "policy updates. The left panel plots every per-update logprob-difference measurement directly. "
        "In the right panel, faint lines show per-update solve rates and foreground lines "
        "show a ten-update moving average."
    )
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Figure 4 editorial preview</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: #202124; font-family: Arial, Helvetica, sans-serif; }}
    main {{ max-width: 1280px; margin: 48px auto; }}
    aside {{ margin: 0 42px 22px; padding: 12px 16px; border-left: 3px solid #f9ab00; background: #fef7e0; font-size: 14px; }}
    figure {{ margin: 0; }}
    img {{ display: block; width: 100%; height: auto; }}
    figcaption {{ margin: 14px 42px 0; color: #3c4043; font-size: 16px; line-height: 1.5; }}
  </style>
</head>
<body>
  <main>
    {editorial_note}
    <figure>
      <img src="{_escape(svg_name)}" alt="Two panels compare mean logprob difference and solve rate for one FrozenLake workload." />
      <figcaption><strong>Figure 4.</strong> {_escape(caption)}</figcaption>
    </figure>
  </main>
</body>
</html>
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    for recipe in ("native", "importance-sampling", "zero-tim"):
        parser.add_argument(f"--five-{recipe}", type=Path, required=True)
    parser.add_argument("--five-max-updates", type=int, default=120)
    parser.add_argument("--assume-certified-zero", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()

    five = load_workload(
        title="FrozenLake · 2×2 to 9×9 grids",
        subtitle="",
        native=args.five_native,
        importance_sampling=args.five_importance_sampling,
        zero_tim=args.five_zero_tim,
        axis_max=args.five_max_updates,
        assume_certified_zero=args.assume_certified_zero,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = args.output_dir / "end-to-end-results-provisional.svg"
    png_path = args.output_dir / "end-to-end-results-provisional.png"
    html_path = args.output_dir / "end-to-end-results-provisional.html"
    svg_path.write_text(
        render_svg(five, assumed_zero=args.assume_certified_zero),
        encoding="utf-8",
    )
    html_path.write_text(
        render_html(svg_path.name, assumed_zero=args.assume_certified_zero),
        encoding="utf-8",
    )
    subprocess.run(
        ["gdk-pixbuf-thumbnailer", "-s", "2560", str(svg_path), str(png_path)],
        check=True,
    )
    print(f"wrote: {svg_path}")
    print(f"wrote: {png_path}")
    print(f"wrote: {html_path}")


if __name__ == "__main__":
    main()
