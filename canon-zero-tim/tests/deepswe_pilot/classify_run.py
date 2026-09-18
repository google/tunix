#!/usr/bin/env python3
"""Classifies the bounded P39 64-chip DeepSWE resident-state pilot."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


_STAGE_UPDATES = {"one-update": 1, "three-update": 3}
_MIN_HBM_FREE = 8 * 1024**3
_WARNING_POLICY = "deepswe-pilot-alignment-warning-v1"


def _records(path: Path) -> list[dict[str, Any]]:
  if not path.is_file():
    raise FileNotFoundError(f"missing evidence file: {path}")
  records = []
  for line in path.read_text(errors="replace").splitlines():
    if line.strip():
      records.append(json.loads(line))
  return records


def _scheduler(log_text: str) -> tuple[list[list[int]], list[dict[str, int]]]:
  buckets = []
  precompiles = []
  for line in log_text.splitlines():
    if "Prepared token paddings:" in line:
      try:
        value = ast.literal_eval(
            line.split("Prepared token paddings:", 1)[1].strip()
        )
      except (SyntaxError, ValueError):
        continue
      if isinstance(value, list) and all(isinstance(item, int) for item in value):
        buckets.append(value)
    if "Precompile worker0 backbone -->" in line:
      try:
        value = ast.literal_eval(
            line.split("Precompile worker0 backbone -->", 1)[1].strip()
        )
      except (SyntaxError, ValueError):
        continue
      if isinstance(value, dict):
        geometry = {key: value.get(key) for key in ("num_tokens", "num_reqs")}
        if all(isinstance(item, int) for item in geometry.values()):
          precompiles.append(geometry)
  return buckets, precompiles


def _hbm_free_bytes(record: dict[str, Any]) -> list[int]:
  snapshots = []
  for boundary in ("hbm_before", "hbm_after_accumulation", "hbm_after_commit"):
    values = record.get(boundary)
    if not isinstance(values, list):
      return []
    for value in values:
      limit = value.get("bytes_limit")
      peak = value.get("peak_bytes_in_use")
      if not isinstance(limit, int) or not isinstance(peak, int):
        return []
      snapshots.append(limit - peak)
  return snapshots


def classify(
    *,
    log_text: str,
    weight_attestations: list[dict[str, Any]],
    pre_alignment: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    stage: str,
) -> dict[str, Any]:
  """Returns a fail-closed verdict without upgrading alignment claims."""
  if stage not in _STAGE_UPDATES:
    raise ValueError(f"unknown P39 pilot stage: {stage!r}")
  expected_updates = _STAGE_UPDATES[stage]
  expected_alignment = expected_updates * 16
  buckets, precompiles = _scheduler(log_text)
  hbm_by_update = [_hbm_free_bytes(record) for record in updates]
  free_hbm = [value for snapshot in hbm_by_update for value in snapshot]
  checks = {
      "attempt_zero": log_text.count(
          "[entrypoint] JOBSET_ATTEMPT 0 (first attempt)"
      ) == 1,
      "pathways_once": log_text.count(
          "[P34.PATHWAYS] initialized_once=1 before_jax=1"
      ) == 1,
      "cli_exact": log_text.count("[P34.CLI] PASS") == 1,
      "source_exact": log_text.count("[sync] provenance ok") == 1,
      "topology_exact": log_text.count("[P34.TOPOLOGY] PASS") == 1,
      "wandb_online": log_text.count(
          "[CANON_P34_WANDB] ONLINE_RUN_PASS"
      ) == 1,
      "scheduler_bucket_exact": buckets == [[1024]],
      "scheduler_precompile_exact": precompiles
      == [{"num_tokens": 1024, "num_reqs": 64}],
      "weight_count": len(weight_attestations) == expected_updates,
      "weight_exact": all(
          record.get("verdict") == "PASS"
          and record.get("equal") is True
          and record.get("mesh_shape") == {"dp": 4, "tp": 8}
          and len(record.get("mesh_device_ids", [])) == 32
          and len(set(record.get("mesh_device_ids", []))) == 32
          for record in weight_attestations
      ),
      "pre_alignment_count": len(pre_alignment) == expected_updates,
      "pre_alignment_nonblocking": all(
          record.get("verdict") in ("PASS", "PASS_WITH_ALIGNMENT_WARNINGS")
          and record.get("blocking_reds") == []
          and record.get("N_action", 0) > 0
          and record.get("admission_policy", {}).get("id") == _WARNING_POLICY
          and record.get("admission_policy", {}).get("claim_level")
          == "convergence-only"
          for record in pre_alignment
      ),
      "alignment_count": len(alignment) == expected_alignment,
      "alignment_nonblocking": all(
          record.get("verdict") in ("PASS", "PASS_WITH_ALIGNMENT_WARNINGS")
          and record.get("blocking_reds") == []
          and record.get("ratio_finite") is True
          and record.get("gradient", {}).get("finite") is True
          and record.get("admission_policy", {}).get("id") == _WARNING_POLICY
          for record in alignment
      ),
      "update_count": len(updates) == expected_updates,
      "update_contract": all(
          record.get("contract_name") == "p39-64chip-pilot"
          and record.get("dp_size") == 4
          and record.get("tp_size") == 8
          and record.get("global_m") == 1024
          for record in updates
      ),
      "update_pass": all(record.get("verdict") == "PASS" for record in updates),
      "commit_count": sum(int(record.get("commits", -1)) for record in updates)
      == expected_updates,
      "gradient_health": all(
          record.get("gradient_finite") is True
          and any(bool(value) for value in record.get("gradient_activity", []))
          for record in updates
      ),
      "fixed_dp_transaction": all(
          record.get("dp_replicas_exact") is True
          and record.get("dp_reduction_transactions") == 16
          and record.get("dp_reduction_rounds_per_transaction") == 4
          and record.get("dp_rank_pullbacks_per_transaction") == 4
          for record in updates
      ),
      "optimizer_device_resident": all(
          record.get("optimizer_placement") == "device-resident"
          and record.get("optimizer_memory_kinds_before") == ["device"]
          and record.get("optimizer_memory_kinds_after") == ["device"]
          and record.get("optimizer_transaction_valid") is True
          for record in updates
      ),
      "optimizer_no_roundtrip": (
          "[P30.G1] OPT_STATE before_commit" not in log_text
          and "[P30.G1] OPT_STATE after_commit" not in log_text
      ),
      "hbm_telemetry_complete": (
          len(hbm_by_update) == expected_updates
          and all(
              len(snapshot) >= 3 * 32 and len(snapshot) % 3 == 0
              for snapshot in hbm_by_update
          )
      ),
      "hbm_margin": bool(free_hbm) and min(free_hbm) >= _MIN_HBM_FREE,
  }
  failed = sorted(name for name, passed in checks.items() if not passed)
  return {
      "schema": "canon.p39.deepswe-64-pilot.run.v1",
      "stage": stage,
      "verdict": "PASS" if not failed else "FAIL",
      "claim_level": "systems-pilot-alignment-degraded",
      "expected_updates": expected_updates,
      "minimum_hbm_free_bytes": min(free_hbm) if free_hbm else None,
      "required_hbm_free_bytes": _MIN_HBM_FREE,
      "scheduler_buckets": buckets,
      "scheduler_precompiles": precompiles,
      "checks": checks,
      "failed": failed,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--stage", choices=tuple(_STAGE_UPDATES), required=True)
  parser.add_argument("--run-log", type=Path, required=True)
  parser.add_argument("--weight-report", type=Path, required=True)
  parser.add_argument("--pre-alignment-report", type=Path, required=True)
  parser.add_argument("--alignment-report", type=Path, required=True)
  parser.add_argument("--update-report", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  report = classify(
      log_text=args.run_log.read_text(errors="replace"),
      weight_attestations=_records(args.weight_report),
      pre_alignment=_records(args.pre_alignment_report),
      alignment=_records(args.alignment_report),
      updates=_records(args.update_report),
      stage=args.stage,
  )
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite evidence: {args.output}")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps(report, sort_keys=True), flush=True)
  if report["verdict"] != "PASS":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
