#!/usr/bin/env python3
"""Build Figure 4 from pinned, measured histories; never substitute zero values.

Only drawing primitives are reused from the earlier editorial renderer. Its
placeholder loader and assume-zero option are deliberately not used here.

Run from this directory:
    python build_end_to_end_results_measured.py --data-dir data --output-dir figures
which writes figures/figure4.svg and figures/figure4.png from data/ only.
(--repo <clone> instead re-exports data/ from wandb_exports/ at the archive
commit; it needs a full clone and PyYAML.)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import subprocess
from pathlib import Path

import yaml

from build_end_to_end_results_provisional import (
    COLORS, Point, Workload, _panel, _text, moving_average,
)

ROOT = Path(__file__).resolve().parent
REVISION = "a7255cfc4b15a29ed9cabcd67a54cac416cd8b74"
STEPS = 200
REWARD = "rewards/train/solve_ratio"
MEAN = "sampler_trainer/train/logp_diff_mean"
MAXIMUM = "sampler_trainer/train/logp_diff_max"
BYTES = "canonical/train/alignment_max_differing_bytes"
FIELDS = ("_step", REWARD, MEAN, MAXIMUM, BYTES)
RUNS = {
    "standard": "jff877lt_canon-p57-fl-stan-r01-567c96d5",
    "importance_sampling": "8zjz4li7_canon-p57-fl-is-i45g-ccbcf572",
    "zero_tim": "tybj4xr0_canon-p57-fl-zero-r10a-06a0fdb9",
}
COMMON = {
    "model_id": "Qwen/Qwen3-8B", "loss_algo": "gspo-token",
    "advantage_estimator": "rloo", "batch_size": 32, "mini_batch_size": 32,
    "num_generations": 8, "seed": 42, "mesh_dp": 8, "mesh_tp": 8,
    "env_max_steps": 5, "max_prompt_length": 4096, "max_response_length": 2048,
    "num_steps": 300,
}
CONFIG_KEYS = tuple(COMMON) + (
    "old_logps_source", "sampler_is", "eval_every_n_steps", "learning_rate",
    "epsilon", "epsilon_high", "loss_agg_mode", "temperature", "top_p", "top_k",
    "train_trajectory_micro_batch_size",
)
# The legacy drawing library calls its gray slot 'native'. This mapping is
# visual only: the new Standard run is NOT the historical rollout-denominator arm.
DRAWING_KEYS = {"standard": "native", "importance_sampling": "importance_sampling", "zero_tim": "zero_tim"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def git_blob(repo: Path, path: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), "show", f"{REVISION}:{path}"],
        capture_output=True, check=False,
    )
    require(result.returncode == 0, f"Cannot read pinned source blob: {path}")
    return result.stdout


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_config(config: dict, arm: str) -> dict:
    for key, expected in COMMON.items():
        require(config.get(key) == expected, f"{arm}: unexpected {key}")
    expected_is = "token" if arm == "importance_sampling" else "none"
    require(config.get("sampler_is") == expected_is, f"{arm}: wrong sampler correction")
    if arm == "standard":
        require(config.get("old_logps_source") == "trainer", "Standard must use trainer-old")
    expected_eval = 0 if arm == "zero_tim" else 50
    require(config.get("eval_every_n_steps") == expected_eval, f"{arm}: unexpected evaluation schedule")
    return {key: config.get(key) for key in CONFIG_KEYS}


def select_rows(blob: bytes, arm: str) -> tuple[list[dict], dict]:
    rows = list(csv.DictReader(io.StringIO(blob.decode("utf-8"))))
    selected = []
    source_lines = []
    for line, row in enumerate(rows, 2):
        if not row.get("_step"):
            continue
        step = float(row["_step"])
        require(math.isfinite(step) and step.is_integer(), f"{arm}: invalid step")
        if not 0 <= step < STEPS:
            continue
        for key in (REWARD, MEAN, MAXIMUM) + ((BYTES,) if arm == "zero_tim" else ()):
            require(row.get(key) not in (None, ""), f"{arm}: missing {key} at {step:g}")
            value = float(row[key])
            require(math.isfinite(value), f"{arm}: nonfinite {key}")
            require(value >= 0, f"{arm}: negative {key}")
            if arm == "zero_tim" and key != REWARD:
                require(value == 0, f"Zero-TIM: measured nonzero {key} at {step:g}")
        require(0 <= float(row[REWARD]) <= 1, f"{arm}: solve rate outside [0, 1]")
        selected.append({key: row.get(key, "") for key in FIELDS})
        source_lines.append(line)
    require([int(float(row["_step"])) for row in selected] == list(range(STEPS)),
            f"{arm}: expected unique, ordered, complete steps 0..{STEPS - 1}")
    # Preserve CSV numeric strings exactly; no rounding or value substitution.
    summary = {}
    for key in FIELDS[1:]:
        values = [float(row[key]) for row in selected if row[key] != ""]
        if values:
            summary[key] = {"count": len(values), "minimum": min(values),
                            "maximum": max(values), "last_ten_mean": sum(values[-10:]) / 10}
    return selected, {"source_csv_lines": source_lines, "metrics": summary}


def make_workload(selected: dict[str, list[dict]]) -> Workload:
    def series(key):
        return {DRAWING_KEYS[arm]: tuple(
            Point(int(float(row["_step"])) + 1, float(row[key])) for row in rows
        ) for arm, rows in selected.items()}
    run = Workload("FrozenLake · 2×2 to 9×9 grids", "", STEPS, STEPS,
                   series(MEAN), series(REWARD))
    require(all(0 <= p.value <= .03 for points in run.logp_diff.values() for p in points),
            "Logprob axis would omit measured points")
    require(all(.3 <= p.value <= 1 for points in run.solve_rate.values() for p in points),
            "Solve-rate axis would omit measured points")
    return run


def load_exported_data(data: Path) -> tuple[dict, dict]:
    """Redraw verified local exports without touching Git or re-exporting data."""
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("source_commit") == REVISION, "Unexpected archive revision")
    require(manifest.get("window") == {
        "source_step_first": 0, "source_step_last": 199,
        "display_step_first": 1, "display_step_last": 200, "count_per_arm": STEPS,
    }, "Unexpected plotted window")
    require(set(manifest["runs"]) == set(RUNS), "Unexpected treatment inventory")
    selected = {}
    for arm, run_id in RUNS.items():
        entry = manifest["runs"][arm]
        require(entry["run_id"] == run_id, f"{arm}: unexpected archived run")
        require(entry["plotted_csv"] == arm + ".csv", f"{arm}: unexpected CSV path")
        validate_config(entry["config"], arm)
        blob = (data / entry["plotted_csv"]).read_bytes()
        require(digest(blob) == entry["plotted_csv_sha256"], f"{arm}: CSV hash mismatch")
        selected[arm], _ = select_rows(blob, arm)
    return selected, manifest


def smoothed_endpoints(run: Workload) -> list[dict]:
    """Use exactly the same trailing average as the foreground curves."""
    endpoints = []
    for arm, drawing_key in DRAWING_KEYS.items():
        point = moving_average(run.solve_rate[drawing_key])[-1]
        require(point.update == STEPS, f"{arm}: endpoint must be step {STEPS}")
        y = 342 - (point.value - .3) * (342 - 138) / (1 - .3)
        endpoints.append({"arm": arm, "step": point.update, "value": point.value,
                          "text": f"{point.value * 100:.1f}%",
                          "color": COLORS[drawing_key], "y": y})
    # Keep labels readable when the gray and orange endpoints are close.
    # Only labels move when necessary; curve coordinates remain unchanged.
    endpoints.sort(key=lambda entry: entry["y"])
    previous = 138 - 12
    for entry in endpoints:
        entry["label_y"] = max(138, entry["y"], previous + 12)
        previous = entry["label_y"]
    overflow = max(0, endpoints[-1]["label_y"] - 342)
    for entry in endpoints:
        entry["label_y"] -= overflow
    return endpoints


def render_svg(run: Workload, manifest: dict) -> str:
    legend = []
    for x, label, color in ((365, "Standard", "#5f6368"),
                            (513, "Token-level TIS", "#e37400"),
                            (695, "Zero-TIM", "#0b57d0")):
        legend.append(f'<line x1="{x}" y1="27" x2="{x+26}" y2="27" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        legend.append(_text(x+35, 32, label, "legend"))
    left = _panel(workload=run, series=run.logp_diff, y_min=-.002, y_max=.03,
                  y_labels=tuple((i / 1000, f"{i / 1000:.3f}") for i in range(0, 31, 5)),
                  box=(96, 138, 596, 342), show_x_labels=True, show_raw=False, smooth=False)
    right = _panel(workload=run, series=run.solve_rate, y_min=.3, y_max=1,
                   y_labels=tuple((i / 10, f"{i * 10}%") for i in range(3, 11)),
                   box=(700, 138, 1200, 342), show_x_labels=True, show_raw=True, smooth=True)
    endpoints = smoothed_endpoints(run)
    labels = []
    for entry in endpoints:
        label_y, color = entry["label_y"], entry["color"]
        labels.append(
            f'<g class="endpoint" data-arm="{entry["arm"]}" '
            f'data-step="{entry["step"]}" data-value="{entry["value"]}">'
            f'<text x="1212" y="{label_y + 4:.1f}" fill="{color}">'
            f'{entry["text"]}</text></g>'
        )
    metadata = html.escape(json.dumps({"evidence": "measured telemetry; no substituted values",
        "source_commit": REVISION, "window": manifest["window"],
        "inputs": {arm: entry["plotted_csv_sha256"] for arm, entry in manifest["runs"].items()},
        "signed_full_run_certification": False,
        "endpoint_labels": {"metric": "training solve rate", "window": "steps 191–200",
                            "format": "percent; one decimal place",
                            "values": {entry["arm"]: entry["value"] for entry in endpoints}}}))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1320 430" width="1320" height="430" role="img" aria-labelledby="title desc">
<title id="title">FrozenLake training: mean logprob difference and training solve rate</title>
<desc id="desc">Standard, token-level TIS, and Zero-TIM over the first 200 training observations, displayed as steps 1–200. Left: unsmoothed mean absolute sampler–trainer logprob difference over sampled action tokens. Right: raw training solve rates and a ten-step trailing average. Endpoint labels show the final ten-step averages as percentages. All values are measured; the Zero-TIM mean difference is zero at every plotted step.</desc>
<metadata>{metadata}</metadata>
<style>
text {{ font-family: Arial, Helvetica, sans-serif; }}
.figure-title {{ font-size: 20px; font-weight: 600; fill: #202124; }}
.configuration {{ font-size: 12px; fill: #5f6368; }}
.panel-title {{ font-size: 15px; font-weight: 600; fill: #3c4043; }}
.axis-label {{ font-size: 12px; fill: #5f6368; }}
.legend {{ font-size: 13px; fill: #3c4043; }}
.endpoint text {{ font-size: 12px; font-weight: 400; }}
.grid {{ stroke: #e8eaed; stroke-width: 1; }}
.tick {{ stroke: #9aa0a6; stroke-width: 1; }}
</style>
<rect width="1320" height="430" fill="#ffffff"/>
{''.join(legend)}
{_text(640, 72, run.title, 'figure-title', 'middle')}
{_text(96, 116, 'Mean |Δ logprob|', 'panel-title')}
{_text(700, 116, 'Training solve rate', 'panel-title')}
{left}
{right}
{''.join(labels)}
{_text(640, 411, 'Qwen3-8B · up to 5 interaction turns · max context 6,144 tokens · max prompt 4,096 · max generation 2,048', 'configuration', 'middle')}
</svg>
'''


def write_figure(run: Workload, manifest: dict, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    svg = output / "figure4.svg"
    svg.write_text(render_svg(run, manifest), encoding="utf-8")
    subprocess.run(["gdk-pixbuf-thumbnailer", "-s", "2640", str(svg), str(svg.with_suffix(".png"))], check=True)
    return svg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo", type=Path)
    source.add_argument("--data-dir", type=Path,
                        help="Redraw validated local CSVs; do not re-export or access Git")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    output = args.output_dir
    if args.data_dir:
        selected, manifest = load_exported_data(args.data_dir)
        svg = write_figure(make_workload(selected), manifest, output)
        print(f"PASS: {STEPS} verified local observations per arm; source data unchanged; {svg}")
        return
    data = ROOT / "data"
    selected, entries, encoded = {}, {}, {}
    # Validate every arm before writing any deliverable.
    for arm, run_id in RUNS.items():
        prefix = f"wandb_exports/{run_id}"
        config_blob = git_blob(args.repo, prefix + "/config.yaml")
        config = validate_config(yaml.safe_load(config_blob), arm)
        history_blob = git_blob(args.repo, prefix + "/history.csv")
        selected[arm], audit = select_rows(history_blob, arm)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected[arm])
        encoded[arm] = buffer.getvalue().encode("utf-8")
        entries[arm] = {"run_id": run_id, "source_history": prefix + "/history.csv",
            "source_history_sha256": digest(history_blob),
            "source_config": prefix + "/config.yaml", "source_config_sha256": digest(config_blob),
            "plotted_csv": arm + ".csv", "plotted_csv_sha256": digest(encoded[arm]),
            "config": config, **audit}
    manifest = {"schema_version": 1, "source_commit": REVISION,
        "evidence_grade": "analysis-grade full-resolution training telemetry",
        "signed_full_run_certification": False,
        "window": {"source_step_first": 0, "source_step_last": 199,
                   "display_step_first": 1, "display_step_last": 200, "count_per_arm": STEPS},
        "transformations": {"logprob_mean": "none", "solve_rate": "raw plus trailing mean",
                            "trailing_window": 10, "initial_window": "available points only"},
        "runs": entries}
    run = make_workload(selected)
    data.mkdir(parents=True, exist_ok=True)
    for arm, blob in encoded.items():
        (data / (arm + ".csv")).write_bytes(blob)
    (data / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    svg = write_figure(run, manifest, output)
    print(f"PASS: {STEPS} measured observations per arm; no substituted values; {svg}")


if __name__ == "__main__":
    main()
