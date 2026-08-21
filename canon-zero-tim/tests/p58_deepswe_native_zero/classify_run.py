#!/usr/bin/env python3
"""Fail-closed classifier for one P58 native/zero DeepSWE arm."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any


_STAGE_UPDATES = {"three-update": 3, "full": 1000}
_TRAJECTORY_SCHEMA = "canon.p58.deepswe.trajectory.v1"
_METRICS_SCHEMA = "canon.p58.deepswe.batch-metrics.v1"
_MANIFEST_SCHEMA = "canon.p58.deepswe.run-manifest.v1"
_SOLVE_DEFINITION = "r2egym_final_reward_eq_1"
_WHITELIST_SHA256 = (
    "ec297c9cbc39cd67db15b0b9db6a229b15671b848df5ec3101de9ef8df7c9973"
)
_COMPACT = {
    "MAX_STEPS_REACHED",
    "MAX_CONTEXT_LIMIT_REACHED",
    "TIMEOUT",
    "ENV_TIMEOUT",
    "MODEL_TIMEOUT",
    "REWARD_TIMEOUT",
}
_SHA = re.compile(r"[0-9a-f]{40}")


def _records(path: Path) -> list[dict[str, Any]]:
  if not path.is_file():
    raise FileNotFoundError(f"missing P58 evidence file: {path}")
  records = []
  for number, line in enumerate(
      path.read_text(errors="replace").splitlines(), start=1
  ):
    if not line.strip():
      continue
    try:
      record = json.loads(line)
    except json.JSONDecodeError as exc:
      raise ValueError(f"invalid JSON at {path}:{number}") from exc
    if not isinstance(record, dict):
      raise ValueError(f"non-object JSON at {path}:{number}")
    records.append(record)
  if not records:
    raise ValueError(f"empty P58 evidence file: {path}")
  return records


def _artifact_checks(
    debug_dir: Path, *, arm: str, stage: str
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
  manifest_path = debug_dir / "run_manifest.json"
  metrics_path = debug_dir / "batch_metrics.jsonl"
  try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    manifest = {}
  try:
    metrics = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
  except (OSError, json.JSONDecodeError):
    metrics = []
  paths = sorted(debug_dir.glob("batch-*.trajectories.jsonl.gz"))
  records_valid = True
  identities_valid = True
  compact_valid = True
  for batch_index, path in enumerate(paths):
    try:
      with gzip.open(path, "rt", encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError):
      records_valid = False
      continue
    groups: dict[str, set[int]] = {}
    for record in records:
      groups.setdefault(str(record.get("group_id")), set()).add(
          record.get("pair_index")
      )
      identity = record.get("task_identity")
      identities_valid &= (
          isinstance(identity, dict)
          and isinstance(identity.get("docker_image"), str)
          and bool(identity["docker_image"])
      )
      status = record.get("status")
      compact_valid &= record.get("compact_filtered") is (status in _COMPACT)
      expected_solved = (
          status == "SUCCEEDED" and record.get("raw_final_reward") == 1.0
      )
      records_valid &= (
          record.get("schema") == _TRAJECTORY_SCHEMA
          and record.get("step") == batch_index
          and isinstance(record.get("optimizer_step"), int)
          and record.get("solve_definition") == _SOLVE_DEFINITION
          and record.get("solved") is expected_solved
          and isinstance(record.get("trajectory"), dict)
      )
    records_valid &= (
        len(records) == 128
        and len(groups) == 8
        and all(indices == set(range(16)) for indices in groups.values())
    )

  metrics_valid = len(metrics) == len(paths) and bool(metrics)
  optimizer_steps = []
  for batch_index, (metric, path) in enumerate(zip(metrics, paths)):
    category_total = sum(
        int(metric.get(name, -1000))
        for name in (
            "all_solved_prompt_groups",
            "all_failed_prompt_groups",
            "mixed_prompt_groups",
            "incomplete_prompt_groups",
        )
    )
    metrics_valid &= (
        metric.get("schema") == _METRICS_SCHEMA
        and metric.get("step") == batch_index
        and metric.get("trajectories") == 128
        and metric.get("prompt_groups") == 8
        and category_total == 8
        and isinstance(metric.get("optimizer_step"), int)
        and 0.0 <= float(metric.get("trajectory_solve_ratio", -1.0)) <= 1.0
        and Path(str(metric.get("trajectory_path", ""))).name == path.name
        and metric.get("trajectory_sha256")
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    optimizer_steps.append(metric.get("optimizer_step"))
  metrics_valid &= (
      optimizer_steps == sorted(optimizer_steps)
      and bool(optimizer_steps)
      and optimizer_steps[0] == 0
  )
  manifest_valid = (
      manifest.get("schema") == _MANIFEST_SCHEMA
      and manifest.get("trajectory_schema") == _TRAJECTORY_SCHEMA
      and manifest.get("metrics_schema") == _METRICS_SCHEMA
      and manifest.get("stage") == stage
      and manifest.get("tim_arm") == arm
      and manifest.get("model_id") == "Qwen/Qwen3-4B-Instruct-2507"
      and manifest.get("contract_name") == "p58-qwen4b-tim-128"
      and manifest.get("slice_topology") == "4x4x8"
      and manifest.get("role_topology") == {"dp": 8, "tp": 8, "devices": 64}
      and manifest.get("global_prompts") == 8
      and manifest.get("generations") == 16
      and manifest.get("global_trajectories") == 128
      and manifest.get("max_response_length") == 16384
      and manifest.get("max_turns") == 50
      and str(manifest.get("clean_rows")) == "1012"
      and manifest.get("whitelist_sha256") == _WHITELIST_SHA256
      and bool(_SHA.fullmatch(str(manifest.get("source_commit", ""))))
  )
  return {
      "manifest_exact": manifest_valid,
      "journal_nonempty_and_counted": len(paths) == len(metrics) and bool(paths),
      "trajectory_records_exact": records_valid,
      "trajectory_task_identity_present": identities_valid,
      "compact_status_exact": compact_valid,
      "batch_metrics_exact": metrics_valid,
  }, metrics


def _boundary_valid(record: dict[str, Any], name: str) -> bool:
  boundary = record.get("boundaries", {}).get(name, {})
  return (
      boundary.get("valid") is not False
      and boundary.get("finite") is not False
      and isinstance(boundary.get("differing_bytes"), int)
      and boundary["differing_bytes"] >= 0
  )


def classify(
    *,
    arm: str,
    stage: str,
    log_text: str,
    debug_dir: Path,
    weights: list[dict[str, Any]],
    pre_alignment: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
    updates: list[dict[str, Any]],
) -> dict[str, Any]:
  if arm not in ("native", "zero") or stage not in _STAGE_UPDATES:
    raise ValueError("P58 classifier requires a signed arm and stage")
  expected_commits = _STAGE_UPDATES[stage]
  artifact_checks, metrics = _artifact_checks(
      debug_dir, arm=arm, stage=stage
  )
  committed = [record for record in updates if record.get("commits") == 1]
  skipped = [record for record in updates if record.get("commits") == 0]
  all_alignment = pre_alignment + alignment
  exact_boundaries = all(
      all(
          _boundary_valid(record, name)
          and record["boundaries"][name]["differing_bytes"] == 0
          for name in record.get("boundaries", {})
      )
      for record in all_alignment
  )
  native_dose = any(
      _boundary_valid(record, "S_decode_vs_S_prefill")
      and record["boundaries"]["S_decode_vs_S_prefill"]["differing_bytes"] > 0
      for record in pre_alignment
  )
  native_bc_exact = all(
      _boundary_valid(record, "S_prefill_vs_T_old")
      and record["boundaries"]["S_prefill_vs_T_old"]["differing_bytes"] == 0
      for record in pre_alignment
  ) and all(
      (
          "T_old_vs_T_current" not in record.get("boundaries", {})
          or (
              _boundary_valid(record, "T_old_vs_T_current")
              and record["boundaries"]["T_old_vs_T_current"]["differing_bytes"]
              == 0
          )
      )
      for record in alignment
  )
  common_update_geometry = all(
      record.get("contract_name") == "p58-qwen4b-tim-128"
      and record.get("dp_size") == 8
      and record.get("tp_size") == 8
      and record.get("global_m") == 2048
      and record.get("verdict") == "PASS"
      and record.get("gradient_finite") is True
      and record.get("optimizer_placement") == "device-resident"
      for record in updates
  )
  if arm == "native":
    update_geometry = common_update_geometry and all(
        record.get("dp_reduction_mode") == "stock-jax-sharded-trainer"
        for record in updates
    )
  else:
    update_geometry = common_update_geometry and all(
        record.get("dp_replicas_exact") is True
        and record.get("dp_reduction_transactions") == 16
        and record.get("dp_reduction_rounds_per_transaction") == 6
        and record.get("dp_rank_pullbacks_per_transaction") == 8
        for record in updates
    )
  committed_steps = [record.get("train_steps_after") for record in committed]
  skipped_valid = all(
      record.get("mode") == "compact-filtered-no-commit"
      and record.get("loss_denominator") == 0.0
      and record.get("train_steps_before") == record.get("train_steps_after")
      and not any(record.get("changed_paths", {}).values())
      for record in skipped
  )
  checks = {
      "wandb_online": log_text.count("[CANON_P34_WANDB] ONLINE_RUN_PASS") == 1,
      "weight_attestations_present": bool(weights) and all(
          record.get("verdict") == "PASS" and record.get("equal") is True
          for record in weights
      ),
      "alignment_present": bool(pre_alignment) and bool(alignment),
      "alignment_nonblocking_finite": all(
          record.get("blocking_reds") == []
          and record.get("verdict") in ("PASS", "PASS_WITH_ALIGNMENT_WARNINGS")
          and all(_boundary_valid(record, name) for name in record.get("boundaries", {}))
          for record in all_alignment
      ),
      "registered_treatment_observed": native_dose if arm == "native" else exact_boundaries,
      "native_b_c_exact": native_bc_exact if arm == "native" else True,
      "zero_all_boundaries_exact": exact_boundaries if arm == "zero" else True,
      "update_records_nonempty": bool(updates),
      "update_geometry": update_geometry,
      "optimizer_commit_count": len(committed) == expected_commits,
      "optimizer_steps_monotonic": committed_steps == list(range(1, expected_commits + 1)),
      "compact_filtered_skips_valid": skipped_valid,
      "artifact_optimizer_steps_cover_horizon": (
          bool(metrics)
          and max(record["optimizer_step"] for record in metrics)
          == expected_commits - 1
      ),
      **artifact_checks,
  }
  failed = sorted(name for name, passed in checks.items() if not passed)
  return {
      "schema": "canon.p58.deepswe-tim.run.v1",
      "arm": arm,
      "stage": stage,
      "verdict": "PASS" if not failed else "FAIL",
      "claim_level": "paired-systems-canary" if stage == "three-update" else "paired-training-campaign",
      "expected_commits": expected_commits,
      "observed_commits": len(committed),
      "observed_skipped_batches": len(skipped),
      "observed_trajectory_batches": len(metrics),
      "native_mismatch_dose_observed": native_dose if arm == "native" else None,
      "checks": checks,
      "failed": failed,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--arm", choices=("native", "zero"), required=True)
  parser.add_argument("--stage", choices=tuple(_STAGE_UPDATES), required=True)
  parser.add_argument("--run-log", type=Path, required=True)
  parser.add_argument("--debug-dir", type=Path, required=True)
  parser.add_argument("--weight-report", type=Path, required=True)
  parser.add_argument("--pre-alignment-report", type=Path, required=True)
  parser.add_argument("--alignment-report", type=Path, required=True)
  parser.add_argument("--update-report", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  report = classify(
      arm=args.arm,
      stage=args.stage,
      log_text=args.run_log.read_text(errors="replace"),
      debug_dir=args.debug_dir,
      weights=_records(args.weight_report),
      pre_alignment=_records(args.pre_alignment_report),
      alignment=_records(args.alignment_report),
      updates=_records(args.update_report),
  )
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite P58 evidence: {args.output}")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, sort_keys=True), flush=True)
  if report["verdict"] != "PASS":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
