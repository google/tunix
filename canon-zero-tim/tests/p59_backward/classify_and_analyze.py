#!/usr/bin/env python3
"""Fail-closed P59 classifier plus training/system wall-time decomposition."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
P33_PATH = ROOT / "canon-zero-tim" / "tests" / "p33_workloads" / "classify_run.py"
P33_SPEC = importlib.util.spec_from_file_location("p59_p33_classifier", P33_PATH)
assert P33_SPEC is not None and P33_SPEC.loader is not None
P33 = importlib.util.module_from_spec(P33_SPEC)
sys.modules[P33_SPEC.name] = P33
P33_SPEC.loader.exec_module(P33)

_FIELD_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^ ]+)")
_STEP_RE = re.compile(r"Global step (\d+) completed in ([0-9.]+) seconds\.")
_ALIGN_RE = re.compile(
    r"^\[CANON_ALIGN(?:_PRE)?\].*\bverdict=(PASS|FAIL)\b"
)
_EXPECTED_STEPS = {
    "control": 3,
    "candidate": 3,
    "profile": 3,
    "tail": 8,
    "numerical-control": 1,
    "numerical-candidate": 1,
    "v1": 3,
}


def _json_lines(path: Path) -> list[dict[str, Any]]:
  rows = []
  for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
      continue
    try:
      row = json.loads(line)
    except json.JSONDecodeError as exc:
      raise ValueError(f"invalid JSONL at {path}:{number}: {exc}") from exc
    if not isinstance(row, dict):
      raise ValueError(f"expected JSON object at {path}:{number}")
    rows.append(row)
  return rows


def _perf_rows(text: str) -> dict[str, list[dict[str, str]]]:
  result: dict[str, list[dict[str, str]]] = {}
  for line in text.splitlines():
    stripped = line.strip()
    if not stripped.startswith("[PERF] "):
      continue
    fields = dict(_FIELD_RE.findall(stripped.removeprefix("[PERF] ")))
    stage = fields.pop("stage", None)
    if stage is not None:
      result.setdefault(stage, []).append(fields)
  return result


def _seconds(row: dict[str, str], label: str, reasons: list[str]) -> float:
  try:
    value = float(row["seconds"])
  except (KeyError, TypeError, ValueError):
    reasons.append(f"{label}.seconds")
    return float("nan")
  if not math.isfinite(value) or value < 0.0:
    reasons.append(f"{label}.seconds")
  return value


def _means(rows: list[dict[str, float]]) -> dict[str, float]:
  keys = (
      "wall_seconds",
      "weight_sync_seconds",
      "cycle_seconds",
      "training_seconds",
      "system_seconds",
      "system_including_sync_seconds",
      "segmented_value_and_grad_seconds",
      "p32_forward_seconds",
      "p32_reverse_seconds",
      "optimizer_seconds",
      "training_other_seconds",
  )
  return {
      key: sum(row[key] for row in rows) / len(rows)
      for key in keys
  }


def classify(
    *,
    kind: str,
    run_log: Path,
    pre_alignment_report: Path,
    update_report: Path,
    alignment_report: Path,
    workload: str = "frozenlake",
    dp_size: int = 16,
    tp_size: int = 4,
) -> dict[str, Any]:
  if kind not in _EXPECTED_STEPS:
    raise ValueError(f"unsupported P59 kind: {kind!r}")
  expected_steps = _EXPECTED_STEPS[kind]
  stage = (
      "p59-eight-update"
      if kind == "tail"
      else "one-update"
      if kind.startswith("numerical-")
      else "three-update"
  )
  base = P33.classify(
      workload=workload,
      stage=stage,
      run_log=run_log,
      pre_alignment_report=pre_alignment_report,
      update_report=update_report,
      alignment_report=alignment_report,
      dp_size=dp_size,
      tp_size=tp_size,
  )
  reasons = [] if base.get("verdict") == "PASS" else [
      f"p33:{reason}" for reason in base.get("reasons", [base.get("verdict")])
  ]
  text = run_log.read_text(encoding="utf-8", errors="replace")
  align_verdicts = [
      match.group(1)
      for line in text.splitlines()
      if (match := _ALIGN_RE.match(line.strip()))
  ]
  pass_count = align_verdicts.count("PASS")
  fail_count = align_verdicts.count("FAIL")
  local_gradient_groups = int(base.get("local_gradient_groups", 0))
  expected_alignments = expected_steps * (1 + local_gradient_groups)
  expected_hard_gate = (
      136 if kind == "tail" else 17 if kind.startswith("numerical-") else 51
  )
  if expected_alignments != expected_hard_gate:
    reasons.append(
        f"alignment_contract={expected_alignments} "
        f"expected_hard_gate={expected_hard_gate}"
    )
  if pass_count != expected_alignments:
    reasons.append(
        f"canon_align_pass={pass_count} expected={expected_alignments}"
    )
  if fail_count != 0:
    reasons.append(f"canon_align_fail={fail_count} expected=0")

  updates = _json_lines(update_report)
  expected_invocations = (
      dp_size if kind in ("control", "numerical-control") else 1
  )
  if dp_size <= 0 or dp_size & (dp_size - 1):
    reasons.append(f"dp_size={dp_size} is not a positive power of two")
  expected_rounds = 2 * int(math.log2(dp_size)) if dp_size > 0 else 0
  for index, update in enumerate(updates):
    checks = {
        "contract_name": workload,
        "dp_reduction_transactions": local_gradient_groups,
        "dp_reduction_rounds_per_transaction": expected_rounds,
        "dp_rank_pullbacks_per_transaction": dp_size,
        "dp_pullback_invocations_per_transaction": expected_invocations,
    }
    for field, expected in checks.items():
      if update.get(field) != expected:
        reasons.append(
            f"update[{index}].{field}={update.get(field)!r} expected={expected}"
        )
    if update.get("dp_replicas_exact") is not True:
      reasons.append(f"update[{index}].dp_replicas_exact")
    if (
        workload == "gsm8k-p59-dp4-tp1"
        and update.get("optimizer_placement") != "device-resident"
    ):
      reasons.append(
          f"update[{index}].optimizer_placement="
          f"{update.get('optimizer_placement')!r} expected='device-resident'"
      )
    if kind.startswith("numerical-"):
      evidence = update.get("commit_evidence")
      if not isinstance(evidence, dict):
        reasons.append(f"update[{index}].commit_evidence")
      else:
        learning_rate = evidence.get("effective_learning_rate")
        changed_elements = evidence.get("parameter_changed_elements")
        if (
            not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate)
            or learning_rate <= 0.0
        ):
          reasons.append(
              f"update[{index}].effective_learning_rate={learning_rate!r}"
          )
        if not isinstance(changed_elements, int) or changed_elements <= 0:
          reasons.append(
              f"update[{index}].parameter_changed_elements="
              f"{changed_elements!r}"
          )

  perf = _perf_rows(text)
  required_stages = (
      "p32_vag_forward",
      "p32_vag_reverse",
      "segmented_value_and_grad",
      "optimizer_transaction",
      "weight_sync",
  )
  for stage in required_stages:
    if len(perf.get(stage, [])) != expected_steps:
      reasons.append(
          f"perf.{stage}={len(perf.get(stage, []))} expected={expected_steps}"
      )
  wall_rows = [
      (int(step), float(seconds))
      for step, seconds in _STEP_RE.findall(text)
  ]
  if len(wall_rows) != expected_steps:
    reasons.append(f"global_steps={len(wall_rows)} expected={expected_steps}")
  if len(updates) != expected_steps:
    reasons.append(f"updates={len(updates)} expected={expected_steps}")

  profile_started = text.count(
      "[P59.XPROF] phase=backward_group started update=1 groups=1"
  )
  profile_stopped = text.count(
      "[P59.XPROF] phase=backward_group stopped update=1 groups=1 "
      "anchor=gradient_ready"
  )
  if kind == "profile":
    if (profile_started, profile_stopped) != (1, 1):
      reasons.append(
          f"xprof_markers={profile_started}/{profile_stopped} expected=1/1"
      )
  elif profile_started or profile_stopped:
    reasons.append(f"unexpected_xprof_markers={profile_started}/{profile_stopped}")
  if kind.startswith("numerical-"):
    for capture_name in ("model_before", "gradient", "model_after"):
      marker = (
          "[P61.NUMERICAL] capture_complete "
          f"name={capture_name} "
      )
      count = text.count(marker)
      if count != 1:
        reasons.append(
            f"p61_capture.{capture_name}={count} expected=1"
        )

  timing_rows: list[dict[str, float]] = []
  complete_timing = (
      len(wall_rows) == expected_steps
      and len(updates) == expected_steps
      and all(len(perf.get(stage, [])) == expected_steps for stage in required_stages)
  )
  if complete_timing:
    for index in range(expected_steps):
      elapsed = updates[index].get("elapsed_seconds")
      if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed):
        reasons.append(f"update[{index}].elapsed_seconds")
        elapsed = float("nan")
      wall = wall_rows[index][1]
      segmented = _seconds(
          perf["segmented_value_and_grad"][index],
          f"perf.segmented[{index}]",
          reasons,
      )
      forward = _seconds(
          perf["p32_vag_forward"][index], f"perf.forward[{index}]", reasons
      )
      reverse = _seconds(
          perf["p32_vag_reverse"][index], f"perf.reverse[{index}]", reasons
      )
      optimizer = _seconds(
          perf["optimizer_transaction"][index],
          f"perf.optimizer[{index}]",
          reasons,
      )
      weight_sync = _seconds(
          perf["weight_sync"][index],
          f"perf.weight_sync[{index}]",
          reasons,
      )
      system = wall - float(elapsed)
      training_other = float(elapsed) - segmented - optimizer
      if system < -0.05:
        reasons.append(f"timing[{index}].negative_system={system:.6f}")
      if training_other < -0.05:
        reasons.append(
            f"timing[{index}].negative_training_other={training_other:.6f}"
        )
      timing_rows.append({
          "global_step": float(wall_rows[index][0]),
          "wall_seconds": wall,
          "weight_sync_seconds": weight_sync,
          "cycle_seconds": wall + weight_sync,
          "training_seconds": float(elapsed),
          "system_seconds": max(system, 0.0),
          "system_including_sync_seconds": max(system, 0.0) + weight_sync,
          "segmented_value_and_grad_seconds": segmented,
          "p32_forward_seconds": forward,
          "p32_reverse_seconds": reverse,
          "optimizer_seconds": optimizer,
          "training_other_seconds": max(training_other, 0.0),
      })

  timing = {"steps": timing_rows}
  if timing_rows:
    timing["all_mean"] = _means(timing_rows)
    stable = timing_rows[2:]
    timing["stable_sample_count"] = len(stable)
    if stable:
      timing["stable_steps2_plus_mean"] = _means(stable)
  return {
      "schema": "canon-p59-backward-classification-v1",
      "verdict": "PASS" if not reasons else "FAIL",
      "kind": kind,
      "zero_tim": {
          "expected_pass": expected_alignments,
          "observed_pass": pass_count,
          "observed_fail": fail_count,
      },
      "pullback_invocations_per_transaction": expected_invocations,
      "topology": {"dp": dp_size, "tp": tp_size},
      "workload": workload,
      "p33": base,
      "timing": timing,
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--kind", required=True, choices=(
          "control",
          "candidate",
          "profile",
          "tail",
          "numerical-control",
          "numerical-candidate",
          "v1",
      )
  )
  parser.add_argument(
      "--workload",
      default="frozenlake",
      choices=("frozenlake", "gsm8k-p59-dp4-tp1"),
  )
  parser.add_argument("--dp-size", type=int, default=16)
  parser.add_argument("--tp-size", type=int, default=4)
  parser.add_argument("--run-log", required=True, type=Path)
  parser.add_argument("--pre-alignment-report", required=True, type=Path)
  parser.add_argument("--update-report", required=True, type=Path)
  parser.add_argument("--alignment-report", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  record = classify(
      kind=args.kind,
      run_log=args.run_log,
      pre_alignment_report=args.pre_alignment_report,
      update_report=args.update_report,
      alignment_report=args.alignment_report,
      workload=args.workload,
      dp_size=args.dp_size,
      tp_size=args.tp_size,
  )
  args.output.write_text(
      json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      "P59_BACKWARD_CLASSIFICATION "
      f"verdict={record['verdict']} kind={args.kind} "
      f"pass={record['zero_tim']['observed_pass']}/"
      f"{record['zero_tim']['expected_pass']} "
      f"fail={record['zero_tim']['observed_fail']}"
  )
  return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
