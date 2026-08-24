#!/usr/bin/env python3
"""Compares XProf/Perfetto trace events between control and treatment runs."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


def load_trace_events(path: Path) -> list[dict[str, Any]]:
  opener = gzip.open if str(path).endswith(".gz") else open
  with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
    data = json.load(f)
  if isinstance(data, dict):
    return data.get("traceEvents", [])
  if isinstance(data, list):
    return data
  return []


def summarize_events(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
  summary: dict[str, dict[str, Any]] = {}
  for ev in events:
    if not isinstance(ev, dict):
      continue
    name = ev.get("name")
    dur = ev.get("dur")
    if not name or dur is None:
      continue
    try:
      dur_val = float(dur)
    except (ValueError, TypeError):
      continue
    if name not in summary:
      summary[name] = {"count": 0, "total_dur_us": 0.0, "max_dur_us": 0.0}
    summary[name]["count"] += 1
    summary[name]["total_dur_us"] += dur_val
    if dur_val > summary[name]["max_dur_us"]:
      summary[name]["max_dur_us"] = dur_val
  for s in summary.values():
    s["avg_dur_us"] = s["total_dur_us"] / s["count"] if s["count"] > 0 else 0.0
  return summary


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--control", type=Path, required=True)
  parser.add_argument("--treatment", type=Path, required=True)
  parser.add_argument("--top", type=int, default=20)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  control_events = load_trace_events(args.control)
  treatment_events = load_trace_events(args.treatment)

  ctrl_summary = summarize_events(control_events)
  treat_summary = summarize_events(treatment_events)

  all_names = sorted(
      set(ctrl_summary.keys()) | set(treat_summary.keys()),
      key=lambda n: max(
          ctrl_summary.get(n, {}).get("total_dur_us", 0.0),
          treat_summary.get(n, {}).get("total_dur_us", 0.0),
      ),
      reverse=True,
  )

  top_names = all_names[: args.top]
  comparisons = []
  for name in top_names:
    c = ctrl_summary.get(name, {"count": 0, "total_dur_us": 0.0, "avg_dur_us": 0.0})
    t = treat_summary.get(name, {"count": 0, "total_dur_us": 0.0, "avg_dur_us": 0.0})
    dur_diff = t["total_dur_us"] - c["total_dur_us"]
    pct_change = (
        (dur_diff / c["total_dur_us"] * 100.0) if c["total_dur_us"] > 0 else None
    )
    comparisons.append({
        "name": name,
        "control_total_us": c["total_dur_us"],
        "control_count": c["count"],
        "treatment_total_us": t["total_dur_us"],
        "treatment_count": t["count"],
        "diff_total_us": dur_diff,
        "pct_change": pct_change,
    })

  total_ctrl_dur = sum(s["total_dur_us"] for s in ctrl_summary.values())
  total_treat_dur = sum(s["total_dur_us"] for s in treat_summary.values())

  out_record = {
      "schema": "xprof.trace.summary.v1",
      "control_file": str(args.control),
      "treatment_file": str(args.treatment),
      "control_total_dur_us": total_ctrl_dur,
      "treatment_total_dur_us": total_treat_dur,
      "total_diff_us": total_treat_dur - total_ctrl_dur,
      "top_events": comparisons,
  }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(out_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )

  print("=" * 80)
  print(f"XPROF TRACE SUMMARY (Top {args.top} ops by total duration)")
  print(f"Control:   {args.control} (total {total_ctrl_dur / 1e3:.2f} ms)")
  print(f"Treatment: {args.treatment} (total {total_treat_dur / 1e3:.2f} ms)")
  print("=" * 80)
  print(f"{'Operation Name':<45} | {'Control (ms)':<12} | {'Treat (ms)':<12} | {'Diff (%)':<10}")
  print("-" * 80)
  for c in comparisons:
    c_ms = c["control_total_us"] / 1000.0
    t_ms = c["treatment_total_us"] / 1000.0
    pct = f"{c['pct_change']:+.1f}%" if c["pct_change"] is not None else "N/A"
    name_display = c["name"][:43] if len(c["name"]) > 43 else c["name"]
    print(f"{name_display:<45} | {c_ms:>12.2f} | {t_ms:>12.2f} | {pct:>10}")
  print("=" * 80)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
