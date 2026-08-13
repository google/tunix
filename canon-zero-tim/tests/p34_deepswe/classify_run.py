#!/usr/bin/env python3
"""Fail-closed classifier for one P34 DeepSWE promotion stage."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any


_STAGE_UPDATES = {
    "backward-no-commit": 1,
    "one-update": 1,
    "three-update": 3,
    "full": 1000,
}
_WARNING_POLICY = "deepswe-pilot-alignment-warning-v1"
_TRAJECTORY_SCHEMA = "canon.p34.deepswe.trajectory.v1"
_METRICS_SCHEMA = "canon.p34.deepswe.batch-metrics.v1"
_MANIFEST_SCHEMA = "canon.p34.deepswe.run-manifest.v1"
_SOLVE_DEFINITION = "r2egym_final_reward_eq_1"
_DATASET_REVISION = "2e8108ff942f24fcb5686badfaf7f9a8808566d5"
_WHITELIST_SHA256 = (
    "2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7"
)
_SHA = re.compile(r"[0-9a-f]{40}")
_TOPOLOGY = {
    "64": {
        "contract": "p46-qwen32b-train-64",
        "slice": "4x4x4",
        "dp": 4,
        "devices": 32,
        "global_m": 1024,
        "local_trajectories": 16,
        "reduction_rounds": 4,
    },
    "256": {
        "contract": "p46-qwen32b-train-256",
        "slice": "4x8x8",
        "dp": 16,
        "devices": 128,
        "global_m": 4096,
        "local_trajectories": 4,
        "reduction_rounds": 8,
    },
}


def _profile_spec(*, topology: str, p46_profile: bool) -> dict[str, Any]:
  if topology not in _TOPOLOGY:
    raise ValueError("Qwen3-32B topology must be exactly 64 or 256")
  if not p46_profile and topology != "256":
    raise ValueError("legacy P34 production is fixed to topology 256")
  if p46_profile:
    return _TOPOLOGY[topology]
  return {**_TOPOLOGY["256"], "contract": "p34-production"}


def _json_records(path: Path) -> list[dict[str, Any]]:
  if not path.is_file():
    raise ValueError(f"missing evidence file: {path}")
  records = []
  for line_number, line in enumerate(path.read_text().splitlines(), start=1):
    if not line.strip():
      continue
    try:
      value = json.loads(line)
    except json.JSONDecodeError as exc:
      raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    if not isinstance(value, dict):
      raise ValueError(f"non-object evidence at {path}:{line_number}")
    records.append(value)
  if not records:
    raise ValueError(f"empty evidence file: {path}")
  return records


def _scheduler_measurements(
    log_text: str,
) -> tuple[list[list[int]], list[dict[str, int]]]:
  buckets: list[list[int]] = []
  precompiles: list[dict[str, int]] = []
  for line in log_text.splitlines():
    if "Prepared token paddings:" in line:
      payload = line.split("Prepared token paddings:", 1)[1].strip()
      try:
        value = ast.literal_eval(payload)
      except (SyntaxError, ValueError):
        continue
      if isinstance(value, list) and all(isinstance(item, int) for item in value):
        buckets.append(value)
    if "Precompile worker0 backbone -->" in line:
      payload = line.split("Precompile worker0 backbone -->", 1)[1].strip()
      try:
        value = ast.literal_eval(payload)
      except (SyntaxError, ValueError):
        continue
      if isinstance(value, dict):
        geometry = {
            key: value.get(key) for key in ("num_tokens", "num_reqs")
        }
        if all(isinstance(item, int) for item in geometry.values()):
          precompiles.append(geometry)
  return buckets, precompiles


def _artifact_checks(
    debug_dir: Path,
    *,
    expected_batches: int,
    spec: dict[str, Any] | None = None,
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
  """Validates durable full-training trajectories without judging quality."""
  spec = _profile_spec(topology="256", p46_profile=False) if spec is None else spec
  manifest_path = debug_dir / "run_manifest.json"
  metrics_path = debug_dir / "batch_metrics.jsonl"
  manifest = (
      json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
  )
  metrics = []
  if metrics_path.is_file():
    metrics = [
        json.loads(line)
        for line in metrics_path.read_text(errors="replace").splitlines()
        if line.strip()
    ]
  paths = (
      sorted(debug_dir.glob("batch-*.trajectories.jsonl.gz"))
      if debug_dir.is_dir()
      else []
  )
  readable = True
  identities_exact = True
  labels_exact = True
  task_identity_present = True
  for expected_step, path in enumerate(paths):
    try:
      with gzip.open(path, "rt", encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError):
      readable = False
      continue
    if len(records) != 64:
      readable = False
      continue
    group_ids = {record.get("group_id") for record in records}
    pairs_by_group = {
        group_id: {
            record.get("pair_index")
            for record in records
            if record.get("group_id") == group_id
        }
        for group_id in group_ids
    }
    identities_exact &= (
        len(group_ids) == 8
        and all(pairs == set(range(8)) for pairs in pairs_by_group.values())
    )
    for record in records:
      reward = record.get("raw_final_reward")
      expected_solved = (
          record.get("status") == "SUCCEEDED" and reward == 1.0
      )
      labels_exact &= (
          record.get("schema") == _TRAJECTORY_SCHEMA
          and record.get("step") == expected_step
          and record.get("solve_definition") == _SOLVE_DEFINITION
          and record.get("solved") is expected_solved
          and isinstance(record.get("trajectory"), dict)
          and isinstance(
              record.get("trajectory", {}).get("conversation_text"), list
          )
      )
      identity = record.get("task_identity")
      task_identity_present &= (
          isinstance(identity, dict)
          and isinstance(identity.get("docker_image"), str)
          and bool(identity["docker_image"])
      )
  metrics_exact = all(
      row.get("schema") == _METRICS_SCHEMA
      and row.get("step") == step
      and row.get("solve_definition") == _SOLVE_DEFINITION
      and row.get("trajectories") == 64
      and row.get("prompt_groups") == 8
      and sum(
          int(row.get(key, -100))
          for key in (
              "all_solved_prompt_groups",
              "all_failed_prompt_groups",
              "mixed_prompt_groups",
              "incomplete_prompt_groups",
          )
      ) == 8
      and 0.0 <= float(row.get("trajectory_solve_ratio", -1.0)) <= 1.0
      and 0 <= int(row.get("effective_prompt_groups", -1)) <= 8
      for step, row in enumerate(metrics)
  )
  digests_exact = len(metrics) == len(paths) and all(
      Path(row.get("trajectory_path", "")).name == path.name
      and row.get("trajectory_sha256")
      == hashlib.sha256(path.read_bytes()).hexdigest()
      for row, path in zip(metrics, paths)
  )
  checks = {
      "trajectory_manifest_exact": (
          manifest.get("schema") == _MANIFEST_SCHEMA
          and manifest.get("stage") == "full"
          and manifest.get("model_id") == "Qwen/Qwen3-32B"
          and manifest.get("contract_name") == spec["contract"]
          and manifest.get("slice_topology") == spec["slice"]
          and manifest.get("role_topology")
          == {"dp": spec["dp"], "tp": 8, "devices": spec["devices"]}
          and manifest.get("global_prompts") == 8
          and manifest.get("generations") == 8
          and manifest.get("global_trajectories") == 64
          and manifest.get("dataset_name") == "R2E-Gym/R2E-Gym-Subset"
          and manifest.get("dataset_revision") == _DATASET_REVISION
          and manifest.get("dataset_split") == "train"
          and manifest.get("dataset_rows") == "4578"
          and manifest.get("clean_rows") == "1851"
          and manifest.get("whitelist_sha256") == _WHITELIST_SHA256
          and bool(_SHA.fullmatch(manifest.get("source_commit", "")))
      ),
      "batch_metric_count": len(metrics) == expected_batches,
      "trajectory_batch_count": len(paths) == expected_batches,
      "trajectory_batches_readable": readable,
      "trajectory_identity_exact": identities_exact,
      "trajectory_task_identity_present": task_identity_present,
      "trajectory_solve_labels_exact": labels_exact,
      "batch_metrics_exact": metrics_exact,
      "trajectory_digests_exact": digests_exact,
  }
  return checks, metrics


def classify(
    *,
    log_text: str,
    weight_attestations: list[dict[str, Any]],
    pre_alignment: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    stage: str,
    debug_dir: Path | None = None,
    topology: str = "256",
    p46_profile: bool = False,
) -> dict[str, Any]:
  """Returns the complete verdict without synthesizing missing evidence."""
  if stage not in _STAGE_UPDATES:
    raise ValueError(f"unknown P34 stage: {stage!r}")
  expected_updates = _STAGE_UPDATES[stage]
  spec = _profile_spec(topology=topology, p46_profile=p46_profile)
  if p46_profile and stage != "full":
    raise ValueError("P46 Qwen3-32B training admits only the full stage")
  expected_alignment = expected_updates * spec["local_trajectories"]
  expected_commits = 0 if stage == "backward-no-commit" else expected_updates
  warning_only = stage == "full"
  scheduler_buckets, scheduler_precompiles = _scheduler_measurements(log_text)
  checks = {
      "attempt_zero": log_text.count(
          "[entrypoint] JOBSET_ATTEMPT 0 (first attempt)"
      ) == 1,
      "pathways_once": log_text.count(
          "[P34.PATHWAYS] initialized_once=1 before_jax=1"
      ) == 1,
      "cli_exact": log_text.count("[P34.CLI] PASS") == 1,
      "source_exact": log_text.count("[sync] provenance ok") == 1,
      "whitelist_exact": log_text.count("[env] P34 whitelist SHA256 OK:") == 1,
      "topology_exact": log_text.count("[P34.TOPOLOGY] PASS") == 1,
      "dataset_filtered": log_text.count("[P34.DATASET] GOLD_FILTER_PASS") == 1,
      "clean_dataset_exact": (
          log_text.count("[P34.DATASET] CLEAN_DATA_PASS") == 1
          if warning_only
          else True
      ),
      "r2e_bounded": log_text.count(
          "[P34.R2E] BOUNDED_KUBERNETES_PATCH_PASS"
      ) == 1,
      "wandb_online": log_text.count(
          "[CANON_P34_WANDB] ONLINE_RUN_PASS"
      ) == 1,
      "fixed_ar_executed": "CANON_FIXED_AR=1 fixed-order tree" in log_text,
      "fixed_embed_executed": (
          "CANON_FIXED_AR_EMBED=1 fixed-order embed gather" in log_text
      ),
      "logprob_m_executed": "CANON_LOGPROB_M on" in log_text,
      "scheduler_bucket_exact": scheduler_buckets == [[spec["global_m"]]],
      "scheduler_precompile_exact": scheduler_precompiles
      == [{"num_tokens": spec["global_m"], "num_reqs": 64}],
      "weight_attestation_marker_count": log_text.count(
          "[P34.WEIGHTS] EXACT"
      )
      == expected_updates,
      "weight_attestation_count": len(weight_attestations)
      == expected_updates,
      "weight_attestation_exact": all(
          record.get("schema")
          == "canon.p34.deepswe.weight-attestation.v1"
          and record.get("step") == index
          and record.get("verdict") == "PASS"
          and record.get("equal") is True
          and isinstance(record.get("mapped_leaves"), int)
          and record["mapped_leaves"] > 0
          and record.get("live_leaves") == record["mapped_leaves"]
          and isinstance(record.get("total_elements"), int)
          and record["total_elements"] > 0
          and record.get("mismatch_indices") == []
          and record.get("mesh_shape") == {"dp": spec["dp"], "tp": 8}
          and len(record.get("mesh_device_ids", [])) == spec["devices"]
          and len(set(record.get("mesh_device_ids", []))) == spec["devices"]
          for index, record in enumerate(weight_attestations)
      ),
      "pre_alignment_count": len(pre_alignment) == expected_updates,
      "pre_alignment_nonblocking": all(
          record.get("verdict")
          in (
              "PASS",
              "PASS_WITH_ALIGNMENT_WARNINGS" if warning_only else "PASS",
          )
          and (not warning_only or record.get("blocking_reds") == [])
          and record.get("N_action", 0) > 0
          and (
              not warning_only
              or record.get("admission_policy", {}).get("id")
              == _WARNING_POLICY
          )
          and all(
              boundary.get("valid") is not False
              and boundary.get("finite") is not False
              for boundary in record.get("boundaries", {}).values()
          )
          for record in pre_alignment
      ),
      "pre_backward_boundaries_exact": all(
          warning_only
          or all(
              record.get("boundaries", {})
              .get(name, {})
              .get("differing_bytes")
              == 0
              for name in (
                  "S_decode_vs_S_prefill",
                  "S_prefill_vs_T_old",
              )
          )
          for record in pre_alignment
      ),
      "alignment_count": len(alignment) == expected_alignment,
      "alignment_nonblocking": all(
          record.get("verdict")
          in (
              "PASS",
              "PASS_WITH_ALIGNMENT_WARNINGS" if warning_only else "PASS",
          )
          and (not warning_only or record.get("blocking_reds") == [])
          and (
              not warning_only or record.get("ratio_finite") is True
          )
          and (
              not warning_only
              or record.get("gradient", {}).get("finite") is True
          )
          and (
              not warning_only
              or record.get("admission_policy", {}).get("id")
              == _WARNING_POLICY
          )
          for record in alignment
      ),
      "four_boundaries_exact": all(
          warning_only
          or (
            all(
              boundary.get("differing_bytes") == 0
              for boundary in record.get("boundaries", {}).values()
            )
            and record.get("exact", {}).get("w_all_exactly_1") is True
            and record.get("exact", {}).get("r_all_exactly_1") is True
            and record.get("exact", {}).get("wr_all_exactly_1") is True
            and record.get("clip_hits") == 0
            and record.get("tis_hits") == 0
          )
          for record in alignment
      ),
      "update_count": len(updates) == expected_updates,
      "update_pass": all(record.get("verdict") == "PASS" for record in updates),
      "commit_count": sum(int(record.get("commits", -1)) for record in updates)
      == expected_commits,
      "gradient_health": all(
          record.get("gradient_finite") is True
          and (
              warning_only
              or (
                  bool(record.get("gradient_activity"))
                  and any(
                      bool(value) for value in record["gradient_activity"]
                  )
              )
          )
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
          and (
              stage == "backward-no-commit"
              or record.get("optimizer_memory_kinds_after") == ["device"]
          )
          for record in updates
      ),
      "optimizer_no_host_roundtrip": (
          "[P30.G1] OPT_STATE before_commit" not in log_text
          and "[P30.G1] OPT_STATE after_commit" not in log_text
      ),
  }
  artifact_metrics: list[dict[str, Any]] = []
  if warning_only:
    if debug_dir is None:
      checks["trajectory_debug_dir_present"] = False
    else:
      artifact_checks, artifact_metrics = _artifact_checks(
          debug_dir, expected_batches=expected_updates, spec=spec
      )
      checks.update(artifact_checks)
  if stage != "backward-no-commit":
    checks["weight_sync_count"] = log_text.count(
        "[P28.G6] weight_sync_committed count=1"
    ) == expected_updates
    steps = [int(record.get("train_steps_after", -1)) for record in updates]
    checks["monotonic_update_steps"] = steps == sorted(set(steps))
  else:
    checks["gradient_deterministic_repeat"] = all(
        record.get("gradient_deterministic") is True for record in updates
    )
  failed = sorted(name for name, passed in checks.items() if not passed)
  quality_warnings = {
      "zero_signal_update_steps": [
          index
          for index, record in enumerate(updates)
          if not any(bool(value) for value in record.get("gradient_activity", []))
      ],
      "zero_effective_prompt_steps": [
          int(record.get("step", -1))
          for record in artifact_metrics
          if record.get("effective_prompt_groups") == 0
      ],
      "pre_alignment_warning_records": sum(
          record.get("verdict") == "PASS_WITH_ALIGNMENT_WARNINGS"
          for record in pre_alignment
      ),
      "post_alignment_warning_records": sum(
          record.get("verdict") == "PASS_WITH_ALIGNMENT_WARNINGS"
          for record in alignment
      ),
  }
  return {
      "schema": "canon.p34.deepswe.run.v1",
      "stage": stage,
      "profile": "p46-qwen32b" if p46_profile else "p34-production",
      "topology": topology,
      "verdict": "PASS" if not failed else "FAIL",
      "claim_level": "convergence-only" if warning_only else "strict-diagnostic",
      "expected_updates": expected_updates,
      "expected_weight_attestation_records": expected_updates,
      "expected_pre_alignment_records": expected_updates,
      "expected_alignment_records": expected_alignment,
      "scheduler_buckets": scheduler_buckets,
      "scheduler_precompiles": scheduler_precompiles,
      "checks": checks,
      "quality_warnings": quality_warnings,
      "failed": failed,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--stage", required=True, choices=tuple(_STAGE_UPDATES))
  parser.add_argument("--run-log", type=Path, required=True)
  parser.add_argument("--debug-dir", type=Path)
  parser.add_argument("--topology", choices=("64", "256"), default="256")
  parser.add_argument("--p46-profile", action="store_true")
  parser.add_argument("--weight-report", type=Path, required=True)
  parser.add_argument("--pre-alignment-report", type=Path, required=True)
  parser.add_argument("--alignment-report", type=Path, required=True)
  parser.add_argument("--update-report", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  report = classify(
      log_text=args.run_log.read_text(errors="replace"),
      weight_attestations=_json_records(args.weight_report),
      pre_alignment=_json_records(args.pre_alignment_report),
      alignment=_json_records(args.alignment_report),
      updates=_json_records(args.update_report),
      stage=args.stage,
      debug_dir=args.debug_dir,
      topology=args.topology,
      p46_profile=args.p46_profile,
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
