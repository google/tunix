#!/usr/bin/env python3
"""Fail-closed classifier for an isolated P57 FrozenLake checkpoint eval."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


_SCHEMA = "p57-frozenlake-isolated-evaluation-v1"
_SHA_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_JSON_MARKER = "[CANON_P57_EVAL_JSON] "
_COMPLETE_RE = re.compile(
    r"^\[CANON_P57_EVAL\] COMPLETE .* backward=0 "
    r"optimizer_commits=0 checkpoint_writes=0$"
)
_TRAIN_STEP_RE = re.compile(r"Global step \d+ completed in")


def _finite_number(value: Any) -> bool:
  return isinstance(value, (int, float)) and math.isfinite(float(value))


def classify(
    *,
    evaluation_path: Path,
    run_log_path: Path,
    arm: str,
    source_commit: str,
    checkpoint_tag: str,
    checkpoint_step: int,
    expected_updates: int,
    workload_candidate: str = "",
    data_split: str = "",
) -> dict[str, Any]:
  reasons: list[str] = []
  if arm not in ("zero", "mismatch", "is"):
    reasons.append(f"invalid arm: {arm!r}")
  expected_fixed = "1" if arm == "zero" else "0"
  if not _SHA_RE.fullmatch(source_commit):
    reasons.append("source commit is not a full lowercase SHA")
  if checkpoint_step < 0 or checkpoint_step % 10:
    reasons.append("checkpoint step is not zero or a 10-step boundary")
  if expected_updates <= 0 or checkpoint_step > expected_updates:
    reasons.append("checkpoint step lies outside the registered horizon")
  if bool(workload_candidate) != bool(data_split):
    reasons.append("workload candidate and data split must be set together")
  if workload_candidate not in ("", "l0", "m10", "m15", "m20"):
    reasons.append(f"invalid workload candidate: {workload_candidate!r}")
  if data_split not in ("", "calibration", "selection", "main"):
    reasons.append(f"invalid data split: {data_split!r}")

  try:
    record = json.loads(evaluation_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as exc:
    record = {}
    reasons.append(f"evaluation artifact unreadable: {exc}")
  try:
    log_text = run_log_path.read_text(encoding="utf-8", errors="replace")
  except OSError as exc:
    log_text = ""
    reasons.append(f"run log unreadable: {exc}")

  expected_fields = {
      "schema": _SCHEMA,
      "arm": arm,
      "fixed_lm_head": expected_fixed,
      "source_commit": source_commit,
      "checkpoint_tag": checkpoint_tag,
      "checkpoint_step": checkpoint_step,
      "policy_step": checkpoint_step,
      "expected_updates": expected_updates,
      "temperature": 0.0,
      "seed": 42,
      "held_out_rows": 100,
      "prompts": 100,
      "generations": 8,
      "batches": 100,
      "n": 800,
      "workload_candidate": workload_candidate,
      "data_split": data_split,
  }
  wrong = {
      key: record.get(key)
      for key, expected in expected_fields.items()
      if record.get(key) != expected
  }
  if wrong:
    reasons.append(f"evaluation contract fields drifted: {wrong}")

  rewards = record.get("rewards")
  if not isinstance(rewards, list) or len(rewards) != 800:
    reasons.append("evaluation rewards must contain exactly 800 values")
    reward_values = np.asarray([], dtype=np.float32)
  else:
    try:
      reward_values = np.asarray(rewards, dtype=np.float32)
    except (TypeError, ValueError) as exc:
      reward_values = np.asarray([], dtype=np.float32)
      reasons.append(f"evaluation rewards are not numeric: {exc}")
  if reward_values.size and not np.isfinite(reward_values).all():
    reasons.append("evaluation rewards contain nonfinite values")
  if reward_values.size == 800 and np.isfinite(reward_values).all():
    reward_groups = reward_values.reshape(100, 8)
    if not np.array_equal(
        reward_groups, np.repeat(reward_groups[:, :1], 8, axis=1)
    ):
      reasons.append("deterministic evaluation replicas diverged within a map")
    map_rewards = reward_groups[:, 0]
    expected_reward = float(map_rewards.mean())
    expected_solve = float((map_rewards > 0.1).mean())
    if not _finite_number(record.get("reward")) or not math.isclose(
        float(record["reward"]), expected_reward, rel_tol=0.0, abs_tol=1e-7
    ):
      reasons.append("reported reward does not match the reward vector")
    if not _finite_number(record.get("solve")) or not math.isclose(
        float(record["solve"]), expected_solve, rel_tol=0.0, abs_tol=1e-7
    ):
      reasons.append("reported solve rate does not match the reward vector")
  if not _finite_number(record.get("wall_seconds")) or float(
      record.get("wall_seconds", -1.0)
  ) < 0.0:
    reasons.append("evaluation wall time is not finite and nonnegative")
  dataset_sha = record.get("dataset_eval_sha256")
  if workload_candidate:
    if not isinstance(dataset_sha, str) or not _SHA256_RE.fullmatch(dataset_sha):
      reasons.append("materialized evaluation dataset SHA is absent or malformed")
  elif dataset_sha not in (None, ""):
    reasons.append("readiness evaluation unexpectedly names a dataset SHA")

  json_records = []
  complete_lines = []
  for line in log_text.splitlines():
    if line.startswith(_JSON_MARKER):
      try:
        json_records.append(json.loads(line[len(_JSON_MARKER):]))
      except json.JSONDecodeError as exc:
        reasons.append(f"malformed evaluation JSON marker: {exc}")
    if _COMPLETE_RE.fullmatch(line):
      complete_lines.append(line)
  if len(json_records) != 1 or (json_records and json_records[0] != record):
    reasons.append(
        "run log must contain exactly one JSON marker identical to the artifact"
    )
  if len(complete_lines) != 1:
    reasons.append("run log must contain exactly one no-update COMPLETE marker")
  if _TRAIN_STEP_RE.search(log_text):
    reasons.append("isolated evaluation unexpectedly entered the train loop")

  return {
      "schema": "p57-frozenlake-evaluation-classification-v1",
      "verdict": "PASS" if not reasons else "FAIL",
      "reasons": reasons,
      "arm": arm,
      "checkpoint_step": checkpoint_step,
      "expected_updates": expected_updates,
      "evaluation": record,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--evaluation", type=Path, required=True)
  parser.add_argument("--run-log", type=Path, required=True)
  parser.add_argument("--arm", choices=("zero", "mismatch", "is"), required=True)
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--checkpoint-tag", required=True)
  parser.add_argument("--checkpoint-step", type=int, required=True)
  parser.add_argument("--expected-updates", type=int, required=True)
  parser.add_argument("--workload-candidate", default="")
  parser.add_argument("--data-split", default="")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = classify(
      evaluation_path=args.evaluation,
      run_log_path=args.run_log,
      arm=args.arm,
      source_commit=args.source_commit,
      checkpoint_tag=args.checkpoint_tag,
      checkpoint_step=args.checkpoint_step,
      expected_updates=args.expected_updates,
      workload_candidate=args.workload_candidate,
      data_split=args.data_split,
  )
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite classification: {args.output}")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      "[P57.EVAL.CLASSIFIER] "
      f"verdict={result['verdict']} reasons={len(result['reasons'])} "
      f"output={args.output}",
      flush=True,
  )
  return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
