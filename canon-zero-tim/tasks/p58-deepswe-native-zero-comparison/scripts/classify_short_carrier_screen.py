#!/usr/bin/env python3
"""Classifies the rollout-only screen for a short P58.22 carrier."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


EXPECTED_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
EXPECTED_TASK_IMAGE = (
    "namanjain12/scrapy_final:"
    "439a3e59b8e858441f8d97dbc32f398db392330d"
)
EXPECTED_WHITELIST_SHA256 = (
    "26e06ab7469987b4bc0c66d683e8468c"
    "2f10ae7d6842b0e138e563adcf87e257"
)
ROLLOUT_PASS_MARKER = (
    "[DEEPSWE.ONEHOST.ROLLOUT_ONLY] PASS trajectories=16 "
    "backward=0 optimizer_commits=0"
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text())
  if not isinstance(value, dict):
    raise ValueError(f"expected JSON object: {path}")
  return value


def _load_last_jsonl(path: Path) -> dict[str, Any]:
  rows = [json.loads(line) for line in path.read_text().splitlines() if line]
  if not rows or not isinstance(rows[-1], dict):
    raise ValueError(f"missing JSONL record: {path}")
  return rows[-1]


def _load_trajectories(path: Path) -> list[dict[str, Any]]:
  with gzip.open(path, "rt", encoding="utf-8") as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
  if not all(isinstance(row, dict) for row in rows):
    raise ValueError("trajectory journal contains a non-object row")
  return rows


def classify(
    root: Path, *, source_sha: str, expected_hostname: str
) -> tuple[dict[str, Any], int]:
  manifest_path = root / "run_manifest.json"
  metrics_path = root / "batch_metrics.jsonl"
  process_path = root / "probe_process_status.json"
  raw_path = root / "raw.log"
  for path in (manifest_path, metrics_path, process_path, raw_path):
    if not path.is_file() or path.stat().st_size == 0:
      raise ValueError(f"missing carrier-screen artifact: {path}")
  trajectory_paths = sorted(root.glob("batch-*.trajectories.jsonl.gz"))
  if len(trajectory_paths) != 1:
    raise ValueError(
        f"expected one trajectory journal, found {len(trajectory_paths)}"
    )

  manifest = _load_json(manifest_path)
  process = _load_json(process_path)
  metrics = _load_last_jsonl(metrics_path)
  records = _load_trajectories(trajectory_paths[0])
  sampling = manifest.get("sampling_contract")
  expected_manifest = {
      "source_commit": source_sha,
      "expected_hostname": expected_hostname,
      "model_id": EXPECTED_MODEL,
      "contract_name": "local-qwen4b-dp1-tp4-zero-admission",
      "stage": "rollout-only",
      "onehost_xprof_arm": "zero-hp",
      "onehost_seam_probe": True,
      "q4_tp4_zero_admission": True,
      "q4_tp4_seam_diagnostic": "",
      "q4_tp4_continue_kv_diagnostic": False,
      "q4_tp4_short_backward": True,
      "q4_tp4_carrier_screen": True,
      "compilation_cache_dir": "",
      "max_prompt_length": 1792,
      "max_response_length": 8192,
      "max_turns": 16,
      "generations": 16,
      "global_trajectories": 16,
      "task_image": EXPECTED_TASK_IMAGE,
      "whitelist_sha256": EXPECTED_WHITELIST_SHA256,
  }
  drift = {
      key: {"expected": expected, "actual": manifest.get(key)}
      for key, expected in expected_manifest.items()
      if manifest.get(key) != expected
  }
  if manifest.get("role_topology") != {"dp": 1, "tp": 4, "devices": 4}:
    drift["role_topology"] = manifest.get("role_topology")
  if sampling != {
      "temperature": 1.0,
      "top_k": 0,
      "top_p": 1.0,
      "source": "explicit-cli",
  }:
    drift["sampling_contract"] = sampling
  if drift:
    raise ValueError(f"carrier-screen manifest drifted: {drift}")
  if process != {"profile": "seam", "training_process_status": 0}:
    raise ValueError(f"carrier-screen process status changed: {process}")
  if ROLLOUT_PASS_MARKER not in raw_path.read_text(errors="replace"):
    raise ValueError("rollout-only zero-backward terminal marker is absent")
  if any(
      path.exists()
      for path in (
          root / "backward_no_commit.json",
          root / "pre_alignment.jsonl",
          root / "alignment.jsonl",
          root / "updates.jsonl",
      )
  ):
    raise ValueError("carrier screen unexpectedly produced training evidence")
  if len(records) != 16:
    raise ValueError(
        f"carrier screen expected sixteen trajectories, got {len(records)}"
    )

  trajectory_sha256 = _sha256(trajectory_paths[0])
  if metrics.get("trajectory_sha256") != trajectory_sha256:
    raise ValueError("batch metrics trajectory hash differs from the journal")
  if metrics.get("trajectories") != 16 or metrics.get("prompt_groups") != 1:
    raise ValueError("carrier-screen batch geometry changed")

  statuses: collections.Counter[str] = collections.Counter()
  rewards: list[float] = []
  action_tokens: list[int] = []
  compact_rows = 0
  eligible_solved: list[int] = []
  eligible_unsolved: list[int] = []
  for index, record in enumerate(records):
    statuses[str(record.get("status"))] += 1
    compact_rows += int(record.get("compact_filtered") is True)
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, dict):
      raise ValueError(f"trajectory row {index} has no payload")
    tokens = trajectory.get("conversation_tokens")
    masks = trajectory.get("conversation_masks")
    logps = trajectory.get("old_logprobs")
    if not isinstance(tokens, list) or not isinstance(masks, list):
      raise ValueError(f"trajectory row {index} lacks token/mask arrays")
    if not isinstance(logps, list) or len(tokens) != len(masks) or len(tokens) != len(logps):
      raise ValueError(f"trajectory row {index} token/mask/logprob lengths differ")
    action_tokens.append(sum(int(value) for value in masks))
    reward = record.get("raw_final_reward")
    if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
      raise ValueError(f"trajectory row {index} reward is not finite")
    rewards.append(float(reward))
    if (
        record.get("status") == "SUCCEEDED"
        and record.get("complete") is True
        and record.get("compact_filtered") is False
        and action_tokens[-1] > 0
    ):
      if float(reward) == 1.0:
        eligible_solved.append(index)
      elif float(reward) == 0.0:
        eligible_unsolved.append(index)

  reasons: list[str] = []
  if not eligible_solved:
    reasons.append("eligible_solved_rows=0")
  if not eligible_unsolved:
    reasons.append("eligible_unsolved_rows=0")
  if any(count <= 0 or count > 8192 for count in action_tokens):
    reasons.append(f"action_tokens={action_tokens}")
  if metrics.get("mixed_prompt_groups") != 1:
    reasons.append(f"mixed_prompt_groups={metrics.get('mixed_prompt_groups')!r}")
  if not isinstance(metrics.get("nonzero_advantages"), int) or metrics.get(
      "nonzero_advantages"
  ) <= 0:
    reasons.append(f"nonzero_advantages={metrics.get('nonzero_advantages')!r}")

  passed = not reasons
  result = {
      "schema": "canon.p58.short-carrier-screen.classification.v1",
      "verdict": "PASS" if passed else "INCONCLUSIVE",
      "outcome": (
          "CARRIER_SCREEN_PASS" if passed else "CARRIER_SCREEN_INCONCLUSIVE"
      ),
      "source_commit": source_sha,
      "model_id": EXPECTED_MODEL,
      "role_topology": {"dp": 1, "tp": 4, "devices": 4},
      "task_image": EXPECTED_TASK_IMAGE,
      "trajectory_rows": len(records),
      "status_histogram": dict(statuses),
      "raw_rewards": rewards,
      "action_tokens": action_tokens,
      "compact_filtered_rows": compact_rows,
      "eligible_solved_rows": eligible_solved,
      "eligible_unsolved_rows": eligible_unsolved,
      "mixed_prompt_groups": metrics.get("mixed_prompt_groups"),
      "nonzero_advantages": metrics.get("nonzero_advantages"),
      "backward": 0,
      "optimizer_commits": 0,
      "reasons": reasons,
      "artifacts": {
          "trajectory": trajectory_paths[0].name,
          "trajectory_sha256": trajectory_sha256,
          "batch_metrics": metrics_path.name,
          "batch_metrics_sha256": _sha256(metrics_path),
          "manifest": manifest_path.name,
          "manifest_sha256": _sha256(manifest_path),
      },
      "claim": (
          "A PASS selects only a real, non-clipped, mixed-reward Qwen3-4B "
          "DP1xTP4 G16 trajectory journal at the P46 sampling recipe. It "
          "proves no alignment or backward."
      ),
  }
  return result, 0 if passed else 3


def _package(root: Path, output: Path) -> None:
  names = [
      "raw.log",
      "run_manifest.json",
      "probe_process_status.json",
      "batch_metrics.jsonl",
      next(root.glob("batch-*.trajectories.jsonl.gz")).name,
      output.name,
  ]
  (root / "RETURN_FILES").write_text("".join(f"{name}\n" for name in names))
  (root / "SHA256SUMS").write_text(
      "".join(f"{_sha256(root / name)}  {name}\n" for name in names)
  )


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--artifact-dir", type=Path, required=True)
  parser.add_argument("--source-sha", required=True)
  parser.add_argument("--expected-hostname", required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--package", action="store_true")
  args = parser.parse_args()
  try:
    result, status = classify(
        args.artifact_dir,
        source_sha=args.source_sha,
        expected_hostname=args.expected_hostname,
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"carrier-screen classifier error: {exc}", file=sys.stderr)
    return 1
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  if args.package:
    _package(args.artifact_dir, args.output)
  print(json.dumps(result, sort_keys=True))
  return status


if __name__ == "__main__":
  raise SystemExit(main())
