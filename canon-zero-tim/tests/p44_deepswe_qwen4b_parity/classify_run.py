#!/usr/bin/env python3
"""Fail-closed classifier for both P44 Qwen3-4B parity topologies."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any


_STAGE_UPDATES = {"rollout-only": 0, "one-update": 1, "three-update": 3}
_MIN_HBM_FREE = 8 * 1024**3
_WARNING_POLICY = "deepswe-pilot-alignment-warning-v1"
_TRAJECTORY_SCHEMA = "canon.p44.deepswe.trajectory.v1"
_METRICS_SCHEMA = "canon.p44.deepswe.batch-metrics.v1"
_MANIFEST_SCHEMA = "canon.p44.deepswe.run-manifest.v1"
_SOLVE_DEFINITION = "r2egym_final_reward_eq_1"
_SHA = re.compile(r"[0-9a-f]{40}")
_TOPOLOGY = {
    "64": {
        "contract": "p44-qwen4b-parity-64",
        "dp": 4,
        "devices": 32,
        "total_devices": 64,
        "hosts": 16,
        "role_hosts": 8,
        "global_m": 1024,
        "local_trajectories": 4,
        "reduction_rounds": 4,
    },
    "256": {
        "contract": "p44-qwen4b-parity-256",
        "dp": 16,
        "devices": 128,
        "total_devices": 256,
        "hosts": 64,
        "role_hosts": 32,
        "global_m": 4096,
        "local_trajectories": 1,
        "reduction_rounds": 8,
    },
}


def _records(path: Path, *, required: bool) -> list[dict[str, Any]]:
  if not path.is_file():
    if required:
      raise FileNotFoundError(f"missing evidence file: {path}")
    return []
  return [
      json.loads(line)
      for line in path.read_text(errors="replace").splitlines()
      if line.strip()
  ]


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
      if isinstance(value, list) and all(
          isinstance(item, int) for item in value
      ):
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


def _artifact_checks(
    debug_dir: Path,
    *,
    stage: str,
    topology: str,
    expected_batches: int,
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
  spec = _TOPOLOGY[topology]
  manifest_path = debug_dir / "run_manifest.json"
  metrics_path = debug_dir / "batch_metrics.jsonl"
  manifest = (
      json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
  )
  metrics = _records(metrics_path, required=False)
  trajectory_paths = (
      sorted(debug_dir.glob("batch-*.trajectories.jsonl.gz"))
      if debug_dir.is_dir()
      else []
  )

  readable_batches = True
  solve_labels_exact = True
  trajectory_identity_exact = True
  for expected_step, path in enumerate(trajectory_paths):
    try:
      with gzip.open(path, "rt", encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError):
      readable_batches = False
      continue
    if len(records) != 16:
      readable_batches = False
      continue
    identities = {
        (record.get("group_id"), record.get("pair_index"))
        for record in records
    }
    trajectory_identity_exact &= (
        len(identities) == 16
        and {pair for _, pair in identities} == {0, 1, 2, 3}
    )
    for record in records:
      reward = record.get("raw_final_reward")
      expected_solved = (
          record.get("status") == "SUCCEEDED" and reward == 1.0
      )
      solve_labels_exact &= (
          record.get("schema") == _TRAJECTORY_SCHEMA
          and record.get("step") == expected_step
          and record.get("solve_definition") == _SOLVE_DEFINITION
          and record.get("solved") is expected_solved
          and isinstance(record.get("trajectory"), dict)
          and isinstance(
              record.get("trajectory", {}).get("conversation_text"), list
          )
      )

  metrics_shape = all(
      row.get("schema") == _METRICS_SCHEMA
      and row.get("step") == step
      and row.get("solve_definition") == _SOLVE_DEFINITION
      and row.get("trajectories") == 16
      and row.get("prompt_groups") == 4
      and sum(
          int(row.get(key, -100))
          for key in (
              "all_solved_prompt_groups",
              "all_failed_prompt_groups",
              "mixed_prompt_groups",
              "incomplete_prompt_groups",
          )
      ) == 4
      and 0.0 <= float(row.get("trajectory_solve_ratio", -1.0)) <= 1.0
      and 0 <= int(row.get("effective_prompt_groups", -1)) <= 4
      for step, row in enumerate(metrics)
  )
  digests_exact = len(metrics) == len(trajectory_paths) and all(
      Path(row.get("trajectory_path", "")).name == path.name
      and row.get("trajectory_sha256")
      == hashlib.sha256(path.read_bytes()).hexdigest()
      for row, path in zip(metrics, trajectory_paths)
  )
  checks = {
      "manifest_exact": (
          manifest.get("schema") == _MANIFEST_SCHEMA
          and manifest.get("stage") == stage
          and manifest.get("model_id") == "Qwen/Qwen3-4B-Instruct-2507"
          and manifest.get("contract_name") == spec["contract"]
          and manifest.get("role_topology")
          == {"dp": spec["dp"], "tp": 8, "devices": spec["devices"]}
          and manifest.get("global_prompts") == 4
          and manifest.get("generations") == 4
          and manifest.get("global_trajectories") == 16
          and manifest.get("solve_definition") == _SOLVE_DEFINITION
          and bool(_SHA.fullmatch(manifest.get("source_commit", "")))
      ),
      "batch_metric_count": len(metrics) == expected_batches,
      "trajectory_batch_count": len(trajectory_paths) == expected_batches,
      "trajectory_batches_readable": readable_batches,
      "trajectory_identity_exact": trajectory_identity_exact,
      "solve_labels_exact": solve_labels_exact,
      "batch_metrics_exact": metrics_shape,
      "trajectory_digests_exact": digests_exact,
  }
  return checks, metrics


def classify(
    *,
    log_text: str,
    debug_dir: Path,
    weight_attestations: list[dict[str, Any]],
    pre_alignment: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    stage: str,
    topology: str,
) -> dict[str, Any]:
  """Returns a systems-parity verdict without a quality or zero-TIM claim."""
  if stage not in _STAGE_UPDATES:
    raise ValueError(f"unknown P44 parity stage: {stage!r}")
  try:
    spec = _TOPOLOGY[topology]
  except KeyError as exc:
    raise ValueError("P44 topology must be exactly 64 or 256") from exc
  expected_updates = _STAGE_UPDATES[stage]
  expected_batches = max(1, expected_updates)
  expected_alignment = expected_updates * spec["local_trajectories"]
  buckets, precompiles = _scheduler(log_text)
  artifact_checks, metrics = _artifact_checks(
      debug_dir,
      stage=stage,
      topology=topology,
      expected_batches=expected_batches,
  )
  hbm_by_update = [_hbm_free_bytes(record) for record in updates]
  free_hbm = [value for snapshot in hbm_by_update for value in snapshot]
  train_steps_before = [record.get("train_steps_before") for record in updates]
  train_steps_after = [record.get("train_steps_after") for record in updates]
  expected_steps_before = list(range(expected_updates))
  expected_steps_after = list(range(1, expected_updates + 1))

  checks = {
      "attempt_zero": log_text.count(
          "[entrypoint] JOBSET_ATTEMPT 0 (first attempt)"
      ) == 1,
      "pathways_once": log_text.count(
          "[P34.PATHWAYS] initialized_once=1 before_jax=1"
      ) == 1,
      "cli_exact": log_text.count(
          "[P34.CLI] PASS model=Qwen3-4B prompts=4 generations=4"
      ) == 1,
      "source_exact": log_text.count("[sync] provenance ok") == 1,
      "device_inventory_exact": log_text.count(
          "[P34.DEVICE_INVENTORY] PASS "
          f"devices={spec['total_devices']} host_source=logical_task "
          f"hosts={spec['hosts']} devices_per_host=4 "
          f"rollout_hosts={spec['role_hosts']} "
          f"trainer_hosts={spec['role_hosts']}"
      ) == 1,
      "topology_exact": log_text.count("[P34.TOPOLOGY] PASS") == 1,
      "swiglu_feature_padding_active": bool(re.search(
          r"\[PATHTRACE\] CANON_PALLAS_SWIGLU_MPAD=1 "
          r"M=\d+ Mp=\d+ F=1216 Fp=1280 "
          r"row_padded=[01] feature_padded=1",
          log_text,
      )),
      "matmul_contract_padding_active": bool(re.search(
          r"\[PATHTRACE\] CANON_PALLAS_MPAD=1 "
          r"M=\d+ Mp=\d+ padded=[01] "
          r"K=1216 Kp=1280 N=2560 Np=2560 "
          r"contract_padded=1 output_padded=0",
          log_text,
      )),
      "matmul_output_padding_active": bool(re.search(
          r"\[PATHTRACE\] CANON_PALLAS_MPAD=1 "
          r"M=\d+ Mp=\d+ padded=[01] "
          r"K=2560 Kp=2560 N=1216 Np=1280 "
          r"contract_padded=0 output_padded=1",
          log_text,
      )),
      "logps_batch_exact": log_text.count(
          "[P44.LOGPS_BATCH] configured_prompts=4 generations=4 "
          "execution_trajectories=16 observed_trajectories=16"
      ) == expected_batches,
      "wandb_online": log_text.count(
          "[CANON_P34_WANDB] ONLINE_RUN_PASS"
      ) == 1,
      "scheduler_bucket_exact": buckets == [[spec["global_m"]]],
      "scheduler_precompile_exact": precompiles
      == [{"num_tokens": spec["global_m"], "num_reqs": 16}],
      "artifact_log_markers": log_text.count(
          "[P44.TRAJECTORY_BATCH]"
      ) == expected_batches
      and log_text.count("[P44.BATCH_METRICS_JSON]") == expected_batches,
      "rollout_only_boundary": (
          log_text.count("[P44.ROLLOUT_ONLY] PASS") == 1
          and not updates
          and "update_step_committed" not in log_text
          if stage == "rollout-only"
          else "[P44.ROLLOUT_ONLY] PASS" not in log_text
      ),
      "weight_count": len(weight_attestations) == expected_updates,
      "weight_exact": all(
          record.get("verdict") == "PASS"
          and record.get("equal") is True
          and record.get("mesh_shape") == {"dp": spec["dp"], "tp": 8}
          and len(record.get("mesh_device_ids", [])) == spec["devices"]
          and len(set(record.get("mesh_device_ids", []))) == spec["devices"]
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
          record.get("contract_name") == spec["contract"]
          and record.get("dp_size") == spec["dp"]
          and record.get("tp_size") == 8
          and record.get("global_m") == spec["global_m"]
          for record in updates
      ),
      "update_pass": all(record.get("verdict") == "PASS" for record in updates),
      "commit_count": sum(int(record.get("commits", -1)) for record in updates)
      == expected_updates,
      "monotonic_train_steps": (
          train_steps_before == expected_steps_before
          and train_steps_after == expected_steps_after
      ),
      "gradient_health": all(
          record.get("gradient_finite") is True
          and any(bool(value) for value in record.get("gradient_activity", []))
          for record in updates
      ),
      "fixed_dp_transaction": all(
          record.get("dp_replicas_exact") is True
          and record.get("dp_reduction_transactions")
          == spec["local_trajectories"]
          and record.get("dp_reduction_rounds_per_transaction")
          == spec["reduction_rounds"]
          and record.get("dp_rank_pullbacks_per_transaction") == spec["dp"]
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
          not updates
          or (
              len(hbm_by_update) == expected_updates
              and all(
                  len(snapshot) >= 3 * spec["devices"]
                  and len(snapshot) % 3 == 0
                  for snapshot in hbm_by_update
              )
          )
      ),
      "hbm_margin": not updates or (
          bool(free_hbm) and min(free_hbm) >= _MIN_HBM_FREE
      ),
      **artifact_checks,
  }
  failed = sorted(name for name, passed in checks.items() if not passed)
  return {
      "schema": "canon.p44.deepswe-parity.run.v1",
      "stage": stage,
      "topology": topology,
      "verdict": "PASS" if not failed else "FAIL",
      "claim_level": "systems-debug-functional-parity-only",
      "expected_updates": expected_updates,
      "observed_batches": len(metrics),
      "minimum_hbm_free_bytes": min(free_hbm) if free_hbm else None,
      "required_hbm_free_bytes": _MIN_HBM_FREE if updates else None,
      "scheduler_buckets": buckets,
      "scheduler_precompiles": precompiles,
      "checks": checks,
      "failed": failed,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--stage", choices=tuple(_STAGE_UPDATES), required=True)
  parser.add_argument("--topology", choices=tuple(_TOPOLOGY), required=True)
  parser.add_argument("--run-log", type=Path, required=True)
  parser.add_argument("--debug-dir", type=Path, required=True)
  parser.add_argument("--weight-report", type=Path, required=True)
  parser.add_argument("--pre-alignment-report", type=Path, required=True)
  parser.add_argument("--alignment-report", type=Path, required=True)
  parser.add_argument("--update-report", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  need_updates = _STAGE_UPDATES[args.stage] > 0
  report = classify(
      log_text=args.run_log.read_text(errors="replace"),
      debug_dir=args.debug_dir,
      weight_attestations=_records(args.weight_report, required=need_updates),
      pre_alignment=_records(
          args.pre_alignment_report, required=need_updates
      ),
      alignment=_records(args.alignment_report, required=need_updates),
      updates=_records(args.update_report, required=need_updates),
      stage=args.stage,
      topology=args.topology,
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
