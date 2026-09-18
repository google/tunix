#!/usr/bin/env python3
"""P48 Phase 0: extract per-stage timings from a one-host arm raw.log.

Reads the log binary-safely (progress bars make these logs binary to plain
grep), emits one JSON object to stdout.  Numbers only; every field carries
the marker it came from.  Usage:  extract_perf.py <raw.log path>
"""
import json
import re
import sys
from datetime import datetime

PERF_RE = re.compile(
    r"\[PERF\](?: step=(?P<step>\d+))? stage=(?P<stage>[a-z_]+) "
    r"seconds=(?P<seconds>[0-9.]+)(?P<rest>.*)"
)
KV_RE = re.compile(r"([a-z0-9_]+)=([0-9.]+)")
ALIGN_RE = re.compile(
    r"\[(?P<marker>CANON_ALIGN(?:_PRE)?)\] step=(?P<step>\d+) "
    r"verdict=(?P<verdict>[A-Z_]+) N_action=(?P<n>\d+) "
    r"bounds=(?P<bounds>\[[^\]]*\])"
)
ARM_END_RE = re.compile(
    r"ARM_END arm=(?P<arm>\w+) docker_exit=(?P<exit>-?\d+) "
    r"elapsed_seconds=(?P<elapsed>\d+)"
)
GLOBAL_STEP_RE = re.compile(
    r"Global step (?P<step>\d+) completed in (?P<seconds>[0-9.]+) seconds"
)
# absl format: "2026-08-12 23:37:00 - INFO -" ; vllm format: "INFO 08-12 23:37:00"
TS_ABSL_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
TS_VLLM_RE = re.compile(r"^(?:INFO|WARNING|ERROR) (\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
DUMMY_LOAD_RE = re.compile(r"Loading dummy weights took (?P<s>[0-9.]+) seconds")
COMPILE_ITEM_RE = re.compile(r"finished in (?P<s>[0-9.]+) \[secs?\]")
WARMUP_RE = re.compile(r"Warm-up call pass finished in (?P<s>[0-9.]+) \[secs?\]")
DIAG_RE = re.compile(r"completion_len\S*[=:]\s*(?P<len>[0-9.]+)")


def parse_ts(line: str, year_hint: int) -> float | None:
    m = TS_ABSL_RE.match(line)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
    m = TS_VLLM_RE.match(line)
    if m:
        return datetime.strptime(
            f"{year_hint}-{m.group(1)}", "%Y-%m-%d %H:%M:%S"
        ).timestamp()
    return None


def main(path: str) -> None:
    perf: list[dict] = []
    aligns: list[dict] = []
    arm_end = None
    global_steps: list[dict] = []
    dummy_load = None
    compile_item_total = 0.0
    warmup_total = 0.0
    first_ts = last_ts = None
    first_compile_ts = last_compile_ts = None
    engine_init_ts = None
    diag_lines: list[str] = []
    eval_markers: list[dict] = []
    init_marks: dict = {}
    year = 2026

    with open(path, "rb") as fh:
        for lineno, blob in enumerate(fh, 1):
            line = blob.decode("utf-8", errors="replace").rstrip("\n")
            ts = parse_ts(line, year)
            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

            m = PERF_RE.search(line)
            if m:
                extra = dict(KV_RE.findall(m.group("rest")))
                perf.append(
                    {
                        "line": lineno,
                        "step": None if m.group("step") is None else int(m.group("step")),
                        "stage": m.group("stage"),
                        "seconds": float(m.group("seconds")),
                        **{k: float(v) for k, v in extra.items()},
                    }
                )
                continue
            m = ALIGN_RE.search(line)
            if m:
                aligns.append(
                    {
                        "line": lineno,
                        "marker": m.group("marker"),
                        "step": int(m.group("step")),
                        "verdict": m.group("verdict"),
                        "N_action": int(m.group("n")),
                        "bounds": m.group("bounds"),
                        "bounds_all_zero": bool(
                            re.fullmatch(
                                r"\[(?:\('\w+', 0\)(?:, )?)+\]", m.group("bounds")
                            )
                        ),
                    }
                )
                continue
            m = ARM_END_RE.search(line)
            if m:
                arm_end = {
                    "line": lineno,
                    "arm": m.group("arm"),
                    "docker_exit": int(m.group("exit")),
                    "elapsed_seconds": int(m.group("elapsed")),
                }
                continue
            m = GLOBAL_STEP_RE.search(line)
            if m:
                global_steps.append(
                    {
                        "line": lineno,
                        "step": int(m.group("step")),
                        "seconds": float(m.group("seconds")),
                        "ts": ts,
                    }
                )
                continue
            m = DUMMY_LOAD_RE.search(line)
            if m:
                dummy_load = {"line": lineno, "seconds": float(m.group("s"))}
            m = COMPILE_ITEM_RE.search(line)
            if m and "Compilation of" in line:
                compile_item_total += float(m.group("s"))
                if ts is not None:
                    if first_compile_ts is None:
                        first_compile_ts = ts
                    last_compile_ts = ts
            m = WARMUP_RE.search(line)
            if m:
                warmup_total += float(m.group("s"))
                if ts is not None:
                    if first_compile_ts is None:
                        first_compile_ts = ts
                    last_compile_ts = ts
            if "Initializing a V1 LLM engine" in line and ts is not None:
                engine_init_ts = ts
            if "Done with loading datasets" in line and ts is not None:
                init_marks["datasets_done_ts"] = ts
            if "after loading qwen_ref / qwen_actor" in line and ts is not None:
                init_marks["models_loaded_ts"] = ts
            if "Compiling" in line and ts is not None:
                if first_compile_ts is None:
                    first_compile_ts = ts
                last_compile_ts = ts
            if "eval" in line.lower() and ("accuracy" in line.lower() or "Eval at" in line):
                eval_markers.append({"line": lineno, "text": line[:200]})
            m = DIAG_RE.search(line)
            if m:
                diag_lines.append(line[:300])

    stage_totals: dict[str, dict] = {}
    for p in perf:
        s = stage_totals.setdefault(
            p["stage"], {"count": 0, "sum_seconds": 0.0, "per_line": []}
        )
        s["count"] += 1
        s["sum_seconds"] = round(s["sum_seconds"] + p["seconds"], 3)
        s["per_line"].append(p)

    out = {
        "log_path": path,
        "arm_end": arm_end,
        "perf_stage_totals": stage_totals,
        "global_steps": global_steps,
        "alignment": aligns,
        "engine": {
            "dummy_weight_load": dummy_load,
            "compile_item_seconds_sum": round(compile_item_total, 2),
            "warmup_pass_seconds_sum": round(warmup_total, 2),
            "engine_init_to_last_compile_seconds": (
                round(last_compile_ts - engine_init_ts, 1)
                if engine_init_ts and last_compile_ts
                else None
            ),
        },
        "log_span_seconds": (
            round(last_ts - first_ts, 1) if first_ts and last_ts else None
        ),
        "init_windows": {
            "trainer_model_load_seconds": (
                round(
                    init_marks["models_loaded_ts"]
                    - init_marks["datasets_done_ts"],
                    1,
                )
                if {"models_loaded_ts", "datasets_done_ts"} <= init_marks.keys()
                else None
            ),
            "models_loaded_to_first_step_start_seconds": (
                round(
                    (global_steps[0]["ts"] - global_steps[0]["seconds"])
                    - init_marks["models_loaded_ts"],
                    1,
                )
                if global_steps
                and global_steps[0].get("ts")
                and "models_loaded_ts" in init_marks
                else None
            ),
            "log_start_to_models_loaded_seconds": (
                round(init_marks["models_loaded_ts"] - first_ts, 1)
                if first_ts and "models_loaded_ts" in init_marks
                else None
            ),
        },
        "completion_len_diagnostics": diag_lines[:10],
        "eval_markers_first10": eval_markers[:10],
    }
    json.dump(out, sys.stdout, indent=1, default=str)
    print()


if __name__ == "__main__":
    main(sys.argv[1])
