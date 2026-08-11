#!/usr/bin/env python3
"""Classify one completed P33 run from immutable local evidence files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


_FULL_STEPS = {"gsm8k": 200, "frozenlake": 450}
_BOUNDARIES = {
    "S_decode_vs_S_prefill",
    "S_prefill_vs_T_old",
    "T_old_vs_T_current",
}
_PRE_BOUNDARIES = {
    "S_decode_vs_S_prefill",
    "S_prefill_vs_T_old",
}
_EXACT_KEYS = {"w_all_exactly_1", "r_all_exactly_1", "wr_all_exactly_1"}
_AB_POLICY_ID = "gsm8k-full-ab-report-v1"
_AB_MAX_ABS = 1.0e-4
_AB_MAX_BYTE_FRACTION = 4.0e-3


def _json_lines(path: Path) -> list[dict[str, Any]]:
  records = []
  for line_number, line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1
  ):
    if not line.strip():
      continue
    try:
      record = json.loads(line)
    except json.JSONDecodeError as exc:
      raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    if not isinstance(record, dict):
      raise ValueError(f"expected JSON object at {path}:{line_number}")
    records.append(record)
  return records


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _expected_updates(workload: str, stage: str) -> int:
  if stage in ("alignment-short", "backward-no-commit"):
    return 1
  if stage == "one-update":
    return 1
  if stage == "three-update":
    return 3
  if stage == "full":
    return _FULL_STEPS[workload]
  raise ValueError(f"unsupported P33 queue stage: {stage!r}")


def _require(condition: bool, reason: str, reasons: list[str]) -> None:
  if not condition:
    reasons.append(reason)


def _ab_policy_expected(workload: str, stage: str) -> bool:
  return workload == "gsm8k" and stage == "full"


def _validate_ab_policy(
    record: dict[str, Any],
    *,
    expected: bool,
    prefix: str,
    reasons: list[str],
) -> None:
  policy = record.get("admission_policy", {})
  if not expected:
    _require(policy.get("enabled") in (None, False), f"{prefix}.policy", reasons)
    return
  _require(policy.get("id") == _AB_POLICY_ID, f"{prefix}.policy_id", reasons)
  _require(policy.get("enabled") is True, f"{prefix}.policy_enabled", reasons)
  _require(policy.get("workload") == "gsm8k", f"{prefix}.policy_workload", reasons)
  _require(policy.get("stage") == "full", f"{prefix}.policy_stage", reasons)
  _require(
      policy.get("max_abs_limit") == _AB_MAX_ABS,
      f"{prefix}.policy_max_abs",
      reasons,
  )
  _require(
      policy.get("byte_fraction_limit") == _AB_MAX_BYTE_FRACTION,
      f"{prefix}.policy_byte_fraction",
      reasons,
  )
  _require(
      policy.get("claim_level") == "alignment-degraded",
      f"{prefix}.policy_claim",
      reasons,
  )


def _validate_reported_ab(
    boundary: dict[str, Any], *, prefix: str, reasons: list[str]
) -> None:
  _require(boundary.get("valid") is True, f"{prefix}.valid", reasons)
  _require(boundary.get("finite") is True, f"{prefix}.finite", reasons)
  _require(
      isinstance(boundary.get("differing_bytes"), int)
      and boundary["differing_bytes"] > 0,
      f"{prefix}.differing_bytes",
      reasons,
  )
  _require(
      isinstance(boundary.get("max_abs"), (int, float))
      and math.isfinite(boundary["max_abs"])
      and boundary["max_abs"] <= _AB_MAX_ABS,
      f"{prefix}.max_abs",
      reasons,
  )
  _require(
      isinstance(boundary.get("byte_fraction"), (int, float))
      and boundary["byte_fraction"] <= _AB_MAX_BYTE_FRACTION,
      f"{prefix}.byte_fraction",
      reasons,
  )


def _validate_alignment_records(
    records: Iterable[dict[str, Any]],
    *,
    expected_count: int,
    optimizer_skipped: bool,
    workload: str,
    stage: str,
    reasons: list[str],
) -> tuple[int, int]:
  rows = list(records)
  reported_count = 0
  policy_expected = _ab_policy_expected(workload, stage)
  _require(
      len(rows) == expected_count,
      f"alignment_count={len(rows)} expected={expected_count}",
      reasons,
  )
  for index, record in enumerate(rows):
    prefix = f"alignment[{index}]"
    _validate_ab_policy(
        record, expected=policy_expected, prefix=prefix, reasons=reasons
    )
    degraded = record.get("verdict") == "PASS_WITH_REPORTED_DRIFT"
    if degraded:
      reported_count += 1
      _require(policy_expected, f"{prefix}.unexpected_degraded", reasons)
      _require(record.get("blocking_reds") == [], f"{prefix}.blocking_reds", reasons)
      reported = record.get("reported_reds", [])
      _require(
          isinstance(reported, list)
          and set(reported) == {
              "S_decode_vs_S_prefill",
              "w_all_exactly_1",
              "wr_all_exactly_1",
          },
          f"{prefix}.reported_reds",
          reasons,
      )
      _require(record.get("reds") == reported, f"{prefix}.reds", reasons)
    else:
      _require(record.get("verdict") == "PASS", f"{prefix}.verdict", reasons)
      _require(record.get("reds") == [], f"{prefix}.reds", reasons)
      _require(record.get("blocking_reds", []) == [], f"{prefix}.blocking_reds", reasons)
      _require(record.get("reported_reds", []) == [], f"{prefix}.reported_reds", reasons)
    _require(record.get("execution_mode") == "train", f"{prefix}.mode", reasons)
    _require(record.get("step") == index, f"{prefix}.step", reasons)
    boundaries = record.get("boundaries", {})
    _require(set(boundaries) == _BOUNDARIES, f"{prefix}.boundaries", reasons)
    for name in _BOUNDARIES:
      boundary = boundaries.get(name, {})
      if degraded and name == "S_decode_vs_S_prefill":
        _validate_reported_ab(
            boundary, prefix=f"{prefix}.{name}", reasons=reasons
        )
      else:
        _require(
            boundary.get("differing_bytes") == 0,
            f"{prefix}.{name}.differing_bytes",
            reasons,
        )
    exact = record.get("exact", {})
    _require(set(exact) == _EXACT_KEYS, f"{prefix}.exact_keys", reasons)
    _require(exact.get("r_all_exactly_1") is True, f"{prefix}.r_exact", reasons)
    if degraded:
      _require(exact.get("w_all_exactly_1") is False, f"{prefix}.w_reported", reasons)
      _require(exact.get("wr_all_exactly_1") is False, f"{prefix}.wr_reported", reasons)
    else:
      _require(all(exact.get(key) is True for key in _EXACT_KEYS), f"{prefix}.exact", reasons)
    _require(record.get("clip_hits") == 0, f"{prefix}.clip_hits", reasons)
    _require(record.get("tis_hits") == 0, f"{prefix}.tis_hits", reasons)
    _require(
        record.get("optimizer_skipped") == int(optimizer_skipped),
        f"{prefix}.optimizer_skipped",
        reasons,
    )
    gradient = record.get("gradient", {})
    _require(gradient.get("finite") is True, f"{prefix}.gradient_finite", reasons)
    _require(
        isinstance(record.get("N_action"), int) and record["N_action"] > 0,
        f"{prefix}.N_action",
        reasons,
    )
  return len(rows), reported_count


def _validate_pre_alignment_records(
    records: Iterable[dict[str, Any]],
    *,
    expected_count: int,
    workload: str,
    stage: str,
    reasons: list[str],
) -> tuple[int, int]:
  rows = list(records)
  reported_count = 0
  policy_expected = _ab_policy_expected(workload, stage)
  _require(
      len(rows) == expected_count,
      f"pre_alignment_count={len(rows)} expected={expected_count}",
      reasons,
  )
  for index, record in enumerate(rows):
    prefix = f"pre_alignment[{index}]"
    _validate_ab_policy(
        record, expected=policy_expected, prefix=prefix, reasons=reasons
    )
    degraded = record.get("verdict") == "PASS_WITH_REPORTED_DRIFT"
    if degraded:
      reported_count += 1
      _require(policy_expected, f"{prefix}.unexpected_degraded", reasons)
      _require(record.get("blocking_reds") == [], f"{prefix}.blocking_reds", reasons)
      _require(
          record.get("reported_reds") == ["S_decode_vs_S_prefill"],
          f"{prefix}.reported_reds",
          reasons,
      )
      _require(
          record.get("reds") == ["S_decode_vs_S_prefill"],
          f"{prefix}.reds",
          reasons,
      )
    else:
      _require(record.get("verdict") == "PASS", f"{prefix}.verdict", reasons)
      _require(record.get("reds") == [], f"{prefix}.reds", reasons)
      _require(record.get("blocking_reds", []) == [], f"{prefix}.blocking_reds", reasons)
      _require(record.get("reported_reds", []) == [], f"{prefix}.reported_reds", reasons)
    _require(record.get("step") == index, f"{prefix}.step", reasons)
    boundaries = record.get("boundaries", {})
    _require(set(boundaries) == _PRE_BOUNDARIES, f"{prefix}.boundaries", reasons)
    for name in _PRE_BOUNDARIES:
      boundary = boundaries.get(name, {})
      if degraded and name == "S_decode_vs_S_prefill":
        _validate_reported_ab(
            boundary, prefix=f"{prefix}.{name}", reasons=reasons
        )
      else:
        _require(
            boundary.get("differing_bytes") == 0,
            f"{prefix}.{name}.differing_bytes",
            reasons,
        )
    _require(
        isinstance(record.get("N_action"), int) and record["N_action"] > 0,
        f"{prefix}.N_action",
        reasons,
    )
  return len(rows), reported_count


def classify(
    *,
    workload: str,
    stage: str,
    run_log: Path,
    pre_alignment_report: Path,
    update_report: Path,
    alignment_report: Path,
) -> dict[str, Any]:
  if workload not in _FULL_STEPS:
    raise ValueError(f"unknown P33 workload: {workload!r}")
  expected_updates = _expected_updates(workload, stage)
  expected_alignments = expected_updates * 16
  reasons: list[str] = []

  for path, label in (
      (run_log, "run_log"),
      (pre_alignment_report, "pre_alignment_report"),
      (update_report, "update_report"),
      (alignment_report, "alignment_report"),
  ):
    _require(path.is_file() and path.stat().st_size > 0, f"missing_{label}", reasons)
  if reasons:
    return {
        "verdict": "FAIL",
        "workload": workload,
        "stage": stage,
        "reasons": reasons,
    }

  log_text = run_log.read_text(encoding="utf-8", errors="replace")
  _require(
      log_text.count("[CANON_P33_WANDB] ONLINE_RUN_PASS") == 1,
      "wandb_online_attestation_count",
      reasons,
  )
  eval_count = log_text.count("[CANON_P33_EVAL] DISABLED workload=frozenlake")
  _require(
      eval_count == (1 if workload == "frozenlake" else 0),
      f"eval_disabled_count={eval_count}",
      reasons,
  )
  metric_markers = [
      (int(last_step), int(events), int(regressions))
      for last_step, events, regressions in re.findall(
          r"\[CANON_P31_METRICS\] monotonic_direct "
          r"last_step=(\d+) events=(\d+) regressions=(\d+)",
          log_text,
      )
  ]
  _require(bool(metric_markers), "missing_monotonic_metrics_marker", reasons)
  _require(
      all(events > 0 and regressions == 0 for _, events, regressions in metric_markers),
      f"monotonic_metrics={metric_markers}",
      reasons,
  )
  _require(
      any(last_step == expected_updates - 1 for last_step, _, _ in metric_markers),
      f"monotonic_last_step={metric_markers} expected={expected_updates - 1}",
      reasons,
  )

  alignments = _json_lines(alignment_report)
  pre_alignments = _json_lines(pre_alignment_report)
  pre_alignment_count, pre_reported_count = _validate_pre_alignment_records(
      pre_alignments,
      expected_count=expected_updates,
      workload=workload,
      stage=stage,
      reasons=reasons,
  )
  alignment_count, alignment_reported_count = _validate_alignment_records(
      alignments,
      expected_count=expected_alignments,
      optimizer_skipped=stage in ("alignment-short", "backward-no-commit"),
      workload=workload,
      stage=stage,
      reasons=reasons,
  )

  if stage in ("alignment-short", "backward-no-commit"):
    try:
      update_records = [json.loads(update_report.read_text(encoding="utf-8"))]
    except json.JSONDecodeError as exc:
      raise ValueError(f"invalid no-commit report: {exc}") from exc
  else:
    update_records = _json_lines(update_report)
  _require(
      len(update_records) == expected_updates,
      f"update_count={len(update_records)} expected={expected_updates}",
      reasons,
  )

  for index, record in enumerate(update_records):
    prefix = f"update[{index}]"
    _require(record.get("verdict") == "PASS", f"{prefix}.verdict", reasons)
    _require(record.get("dp_axis") == "data", f"{prefix}.dp_axis", reasons)
    _require(record.get("microsteps") == 16, f"{prefix}.microsteps", reasons)
    activity = record.get("gradient_activity")
    _require(
        isinstance(activity, list) and len(activity) == 16,
        f"{prefix}.gradient_activity",
        reasons,
    )
    _require(
        len(record.get("alignment_hashes", [])) == 16,
        f"{prefix}.alignment_hashes",
        reasons,
    )
    norms = record.get("micro_gradient_norms", [])
    _require(
        len(norms) == 16
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in norms),
        f"{prefix}.micro_gradient_norms",
        reasons,
    )
    _require(
        record.get("optimizer_memory_kinds_before") == ["pinned_host"],
        f"{prefix}.optimizer_before",
        reasons,
    )
    if stage in ("alignment-short", "backward-no-commit"):
      _require(record.get("mode") == stage, f"{prefix}.mode", reasons)
      _require(record.get("commits") == 0, f"{prefix}.commits", reasons)
      _require(
          record.get("train_steps_after") == record.get("train_steps_before"),
          f"{prefix}.train_steps",
          reasons,
      )
      for key in (
          "model_changed_paths",
          "optimizer_changed_paths",
          "accumulator_changed_paths",
          "reference_changed_paths",
      ):
        _require(record.get(key) == [], f"{prefix}.{key}", reasons)
      _require(any(activity or ()), f"{prefix}.learning_signal", reasons)
    else:
      _require(record.get("commits") == 1, f"{prefix}.commits", reasons)
      _require(record.get("train_steps_before") == index, f"{prefix}.step_before", reasons)
      _require(record.get("train_steps_after") == index + 1, f"{prefix}.step_after", reasons)
      _require(
          record.get("optimizer_memory_kinds_after") == ["pinned_host"],
          f"{prefix}.optimizer_after",
          reasons,
      )
      _require(
          record.get("accumulator_changed_paths") == [],
          f"{prefix}.accumulator_reset",
          reasons,
      )
      _require(
          record.get("reference_changed_paths") == [],
          f"{prefix}.reference_unchanged",
          reasons,
      )
      commit_norm = record.get("commit_gradient_norm")
      _require(
          isinstance(commit_norm, (int, float)) and math.isfinite(commit_norm),
          f"{prefix}.commit_gradient_norm",
          reasons,
      )
      _require(
          record.get("optimizer_transaction_valid") is True,
          f"{prefix}.optimizer_transaction_valid",
          reasons,
      )
      evidence = record.get("commit_evidence")
      _require(isinstance(evidence, dict), f"{prefix}.commit_evidence", reasons)
      if isinstance(evidence, dict):
        effective_lr = evidence.get("effective_learning_rate")
        _require(
            effective_lr is None
            or (
                isinstance(effective_lr, (int, float))
                and math.isfinite(effective_lr)
                and effective_lr >= 0.0
            ),
            f"{prefix}.effective_learning_rate",
            reasons,
        )
        changed_elements = evidence.get("parameter_changed_elements")
        _require(
            isinstance(changed_elements, int) and changed_elements >= 0,
            f"{prefix}.parameter_changed_elements",
            reasons,
        )
        total_elements = evidence.get("parameter_total_elements")
        _require(
            isinstance(total_elements, int)
            and total_elements >= 0
            and isinstance(changed_elements, int)
            and changed_elements <= total_elements,
            f"{prefix}.parameter_total_elements",
            reasons,
        )
        gradient_nonzero = evidence.get("gradient_nonzero_elements")
        _require(
            isinstance(gradient_nonzero, int) and gradient_nonzero >= 0,
            f"{prefix}.gradient_nonzero_elements",
            reasons,
        )
        for field in ("gradient_max_abs", "parameter_delta_max_abs"):
          value = evidence.get(field)
          _require(
              isinstance(value, (int, float))
              and math.isfinite(value)
              and value >= 0.0,
              f"{prefix}.{field}",
              reasons,
          )
        _require(
            evidence.get("gradient_finite") is True,
            f"{prefix}.commit_gradient_finite",
            reasons,
        )
        _require(
            evidence.get("parameter_delta_finite") is True,
            f"{prefix}.parameter_delta_finite",
            reasons,
        )
        if workload == "gsm8k":
          _require(
              effective_lr is not None,
              f"{prefix}.gsm8k_schedule_registered",
              reasons,
          )
        if effective_lr == 0.0:
          _require(
              changed_elements == 0,
              f"{prefix}.zero_lr_model_unchanged",
              reasons,
          )
          _require(
              record.get("parameter_mutation") == "zero_lr_unchanged",
              f"{prefix}.zero_lr_mutation_class",
              reasons,
          )

  if stage in ("alignment-short", "backward-no-commit"):
    marker_count = log_text.count("[CANON_P33_DP16] backward_no_commit verdict=PASS")
  else:
    marker_count = log_text.count("[CANON_P33_DP16] update_step_committed")
  _require(
      marker_count == expected_updates,
      f"terminal_marker_count={marker_count} expected={expected_updates}",
      reasons,
  )

  policy_expected = _ab_policy_expected(workload, stage)
  verdict = (
      "FAIL"
      if reasons
      else "PASS_WITH_AB_REPORT_POLICY"
      if policy_expected
      else "PASS"
  )
  return {
      "verdict": verdict,
      "workload": workload,
      "stage": stage,
      "expected_updates": expected_updates,
      "observed_updates": len(update_records),
      "expected_alignments": expected_alignments,
      "observed_alignments": alignment_count,
      "expected_pre_alignments": expected_updates,
      "observed_pre_alignments": pre_alignment_count,
      "ab_report_policy_enabled": policy_expected,
      "pre_alignment_reported_drift_records": pre_reported_count,
      "alignment_reported_drift_records": alignment_reported_count,
      "claim_level": (
          "alignment-degraded" if policy_expected else "strict-zero-tim"
      ),
      "diagnostic_only": stage == "alignment-short",
      "evidence_sha256": {
          "run_log": _sha256(run_log),
          "pre_alignment_report": _sha256(pre_alignment_report),
          "update_report": _sha256(update_report),
          "alignment_report": _sha256(alignment_report),
      },
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--workload", required=True, choices=tuple(_FULL_STEPS))
  parser.add_argument(
      "--stage",
      required=True,
      choices=(
          "alignment-short",
          "backward-no-commit",
          "one-update",
          "three-update",
          "full",
      ),
  )
  parser.add_argument("--run-log", required=True, type=Path)
  parser.add_argument("--pre-alignment-report", required=True, type=Path)
  parser.add_argument("--update-report", required=True, type=Path)
  parser.add_argument("--alignment-report", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()

  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite P33 classification: {args.output}")
  record = classify(
      workload=args.workload,
      stage=args.stage,
      run_log=args.run_log,
      pre_alignment_report=args.pre_alignment_report,
      update_report=args.update_report,
      alignment_report=args.alignment_report,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      "[P33.RUN] VERDICT "
      f"{record['verdict']} workload={args.workload} stage={args.stage} "
      f"updates={record.get('observed_updates', 0)}/"
      f"{record.get('expected_updates', 0)} alignments="
      f"{record.get('observed_alignments', 0)}/"
      f"{record.get('expected_alignments', 0)} reasons={record['reasons']}",
      flush=True,
  )
  print(f"[P33.RUN] classification={args.output}", flush=True)
  print(f"[P33.RUN] JSON {json.dumps(record, sort_keys=True)}", flush=True)
  return 0 if record["verdict"] in (
      "PASS", "PASS_WITH_AB_REPORT_POLICY"
  ) else 1


if __name__ == "__main__":
  raise SystemExit(main())
