#!/usr/bin/env python3
"""Fail-closed accounting for one P58 64split three-update pilot.

This tool does not classify numerical correctness itself.  It requires the
ordinary P58 classifier to have passed, then extracts DeepSWE-specific rollout,
trainer, synchronization, lifecycle, solve-signal, and HBM receipts.  It is
kept in the published package so a remote evidence agent needs no outer
workspace scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping


_PERF_RE = re.compile(r"\[PERF\]\s+(.*)$")
_ALIGN_FAIL_RE = re.compile(r"\[CANON_ALIGN(?:_PRE)?\].*\bverdict=FAIL\b")
_FULL_CONSUMER = "[P58.36.BATCH] FULL_CONSUMER_PASS"
_DEADLINE_START = "[P58.36.BATCH] DEADLINE_START"
_GROUP_COMPLETE = "[P58.36.GROUP] COMPLETE"
_ENGINE_ANNOUNCE = "[ENGINE_STEP_LOG] on"
_ENGINE_STEP = "[ENGINE_STEP] n="


def _read_json(path: Path) -> dict[str, Any]:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as exc:
    raise ValueError(f"invalid JSON evidence: {path}") from exc
  if not isinstance(value, dict):
    raise ValueError(f"JSON evidence must be an object: {path}")
  return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
  try:
    lines = path.read_text(encoding="utf-8").splitlines()
  except OSError as exc:
    raise ValueError(f"cannot read JSONL evidence: {path}") from exc
  records = []
  for number, line in enumerate(lines, start=1):
    if not line.strip():
      continue
    try:
      value = json.loads(line)
    except json.JSONDecodeError as exc:
      raise ValueError(f"invalid JSONL evidence at {path}:{number}") from exc
    if not isinstance(value, dict):
      raise ValueError(f"non-object JSONL evidence at {path}:{number}")
    records.append(value)
  if not records:
    raise ValueError(f"empty JSONL evidence: {path}")
  return records


def _finite(value: Any, *, label: str) -> float:
  try:
    result = float(value)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{label} must be numeric: {value!r}") from exc
  if not math.isfinite(result) or result < 0.0:
    raise ValueError(f"{label} must be finite and nonnegative: {result}")
  return result


def _summary(values: Iterable[float]) -> dict[str, float | int]:
  ordered = sorted(float(value) for value in values)
  if not ordered:
    return {"count": 0}

  def percentile(fraction: float) -> float:
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]

  return {
      "count": len(ordered),
      "p50": float(statistics.median(ordered)),
      "p90": percentile(0.90),
      "p99": percentile(0.99),
      "max": ordered[-1],
      "sum": float(sum(ordered)),
  }


def _tokens(payload: str) -> dict[str, str]:
  result = {}
  for token in payload.split():
    if "=" in token:
      key, value = token.split("=", 1)
      result[key] = value
  return result


def _perf_stages(log_text: str) -> dict[str, list[float]]:
  stages: dict[str, list[float]] = {}
  for line in log_text.splitlines():
    match = _PERF_RE.search(line)
    if not match:
      continue
    fields = _tokens(match.group(1))
    stage = fields.get("stage")
    if not stage or "seconds" not in fields:
      continue
    stages.setdefault(stage, []).append(
        _finite(fields["seconds"], label=f"PERF {stage} seconds")
    )
  return stages


def _hbm(updates: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
  snapshots = []
  for update in updates:
    for key in ("hbm_before", "hbm_after_accumulation", "hbm_after_commit"):
      value = update.get(key)
      if isinstance(value, list):
        snapshots.extend(item for item in value if isinstance(item, Mapping))
  peaks = [
      int(item["peak_bytes_in_use"])
      for item in snapshots
      if isinstance(item.get("peak_bytes_in_use"), (int, float))
  ]
  limits = [
      int(item["bytes_limit"])
      for item in snapshots
      if isinstance(item.get("bytes_limit"), (int, float))
  ]
  if not peaks or not limits or min(limits) <= 0:
    return {"available": False}
  peak = max(peaks)
  limit = min(limits)
  return {
      "available": True,
      "peak_bytes_in_use": peak,
      "minimum_device_limit_bytes": limit,
      "peak_fraction_of_minimum_limit": peak / limit,
  }


def account(
    *,
    classification_path: Path,
    run_log_path: Path,
    debug_dir: Path,
    update_report_path: Path,
    reference_update_report_path: Path | None = None,
) -> dict[str, Any]:
  classification = _read_json(classification_path)
  metrics = _read_jsonl(debug_dir / "batch_metrics.jsonl")
  updates = _read_jsonl(update_report_path)
  try:
    log_text = run_log_path.read_text(errors="replace")
  except OSError as exc:
    raise ValueError(f"cannot read run log: {run_log_path}") from exc

  committed = [record for record in updates if record.get("commits") == 1]
  skipped = [record for record in updates if record.get("commits") == 0]
  perf = _perf_stages(log_text)
  hbm = _hbm(committed)
  deadline_receipts = log_text.count(_DEADLINE_START)
  group_receipts = log_text.count(_GROUP_COMPLETE)
  unconsumed_prefetch_batches = deadline_receipts - len(metrics)
  timing_complete = all(
      isinstance(record.get("timing"), Mapping)
      and isinstance(record["timing"].get("stage_seconds"), Mapping)
      and len(record["timing"].get("group_completion_seconds", ())) == 8
      for record in metrics
  )
  rollout_seconds = [
      _finite(record["timing"]["batch_elapsed_seconds"], label="rollout batch")
      for record in metrics
      if isinstance(record.get("timing"), Mapping)
  ]
  trainer_seconds = [
      _finite(record.get("elapsed_seconds"), label="trainer update")
      for record in committed
  ]
  sync_seconds = perf.get("weight_sync", [])
  checks = {
      "classifier_pass": classification.get("verdict") == "PASS",
      "classifier_identity": (
          classification.get("topology") == "64split"
          and classification.get("stage") == "three-update"
          and classification.get("arm") == "zero"
      ),
      "three_optimizer_commits": (
          len(committed) == 3
          and [record.get("train_steps_after") for record in committed]
          == [1, 2, 3]
      ),
      "updates_numerically_healthy": all(
          record.get("verdict") == "PASS"
          and record.get("gradient_finite") is True
          and record.get("optimizer_placement") == "device-resident"
          for record in committed
      ),
      "complete_batch_metrics": (
          len(metrics) >= 3
          and all(
              record.get("trajectories") == 128
              and record.get("prompt_groups") == 8
              for record in metrics
          )
      ),
      "complete_lifecycle_timing": timing_complete,
      "shared_deadline_receipts": (
          log_text.count(_FULL_CONSUMER) == 1
          and unconsumed_prefetch_batches in (0, 1)
          and group_receipts == 8 * deadline_receipts
      ),
      "alignment_fail_count_zero": not _ALIGN_FAIL_RE.search(log_text),
      "weight_sync_count_matches_commits": len(sync_seconds) == 3,
      "trainer_timing_count_matches_commits": len(trainer_seconds) == 3,
      "engine_step_receipts_present": (
          log_text.count(_ENGINE_ANNOUNCE) >= 1
          and log_text.count(_ENGINE_STEP) >= 1
      ),
      "trainer_hbm_receipt_present": hbm.get("available") is True,
      "trainer_hbm_peak_headroom_at_least_8gib": (
          hbm.get("available") is True
          and int(hbm["minimum_device_limit_bytes"])
          - int(hbm["peak_bytes_in_use"])
          >= 8 * 1024**3
      ),
  }

  reference = None
  if reference_update_report_path is not None:
    current_bytes = update_report_path.read_bytes()
    reference_bytes = reference_update_report_path.read_bytes()
    equal = current_bytes == reference_bytes
    checks["repeat_updates_bytewise_equal"] = equal
    reference = {
        "path": str(reference_update_report_path),
        "bytewise_equal": equal,
        "sha256": hashlib.sha256(reference_bytes).hexdigest(),
    }

  per_batch = []
  for record in metrics:
    timing = record.get("timing", {})
    per_batch.append({
        "batch_index": record.get("step"),
        "optimizer_step_before": record.get("optimizer_step"),
        "rollout_batch_seconds": timing.get("batch_elapsed_seconds"),
        "lifecycle_stage_seconds": timing.get("stage_seconds"),
        "group_completion_seconds": timing.get("group_completion_seconds"),
        "trajectory_solve_ratio": record.get("trajectory_solve_ratio"),
        "all_solved_prompt_groups": record.get("all_solved_prompt_groups"),
        "all_failed_prompt_groups": record.get("all_failed_prompt_groups"),
        "mixed_prompt_groups": record.get("mixed_prompt_groups"),
        "incomplete_prompt_groups": record.get("incomplete_prompt_groups"),
        "compact_filtered_trajectories": record.get(
            "compact_filtered_trajectories"
        ),
    })

  failed = sorted(name for name, passed in checks.items() if not passed)
  return {
      "schema": "canon.p58.deepswe-64split-account.v1",
      "verdict": "PASS" if not failed else "INCONCLUSIVE",
      "claim_level": "64split-three-update-pilot",
      "checks": checks,
      "failed": failed,
      "classification_path": str(classification_path),
      "run_log_path": str(run_log_path),
      "debug_dir": str(debug_dir),
      "update_report_path": str(update_report_path),
      "update_report_sha256": hashlib.sha256(
          update_report_path.read_bytes()
      ).hexdigest(),
      "reference_update_report": reference,
      "observed_batches": len(metrics),
      "observed_commits": len(committed),
      "observed_skipped_batches": len(skipped),
      "unconsumed_prefetch_batches": unconsumed_prefetch_batches,
      "timing": {
          "rollout_R32_seconds": _summary(rollout_seconds),
          "trainer_U32_seconds": _summary(trainer_seconds),
          "weight_sync_S_seconds": _summary(sync_seconds),
          "segmented_value_and_grad_seconds": _summary(
              perf.get("segmented_value_and_grad", [])
          ),
          "optimizer_transaction_seconds": _summary(
              perf.get("optimizer_transaction", [])
          ),
      },
      "trainer_hbm": hbm,
      "engine_receipts": {
          "announce_count": log_text.count(_ENGINE_ANNOUNCE),
          "step_count": log_text.count(_ENGINE_STEP),
          "summary_count": log_text.count("[ENGINE_STEP_SUMMARY]"),
      },
      "batches": per_batch,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--classification", type=Path, required=True)
  parser.add_argument("--run-log", type=Path, required=True)
  parser.add_argument("--debug-dir", type=Path, required=True)
  parser.add_argument("--update-report", type=Path, required=True)
  parser.add_argument("--reference-update-report", type=Path)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  report = account(
      classification_path=args.classification,
      run_log_path=args.run_log,
      debug_dir=args.debug_dir,
      update_report_path=args.update_report,
      reference_update_report_path=args.reference_update_report,
  )
  payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
  if args.output is not None:
    if args.output.exists():
      raise FileExistsError(f"refusing to overwrite P58 accounting: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
  print(payload, end="")
  if report["verdict"] != "PASS":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
