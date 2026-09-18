#!/usr/bin/env python3
"""Fail-closed classifier for the P57 in-process held-out curve."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

from examples.frozenlake import p57_workloads


_JSON_MARKER = "[CANON_" "FROZENLAKE_P42_JSON] "
_ENABLED_RE = re.compile(
    r"\[CANON_" r"P33_EVAL\] ENABLED workload=frozenlake "
    r"cadence=(\d+) held_out_rows=(\d+) generations=(\d+)"
)
_FINAL_RE = re.compile(
    r"\[P57\.EVAL\] FINAL policy_step=(\d+) prompts=(\d+) "
    r"generations=(\d+) n=(\d+) reward=([-+0-9.eE]+) "
    r"solve=([-+0-9.eE]+) backward=0 optimizer_commits=0 "
    r"evaluation_checkpoint_writes=0"
)
_CYCLE_RE = re.compile(
    r"\[P57\.EVAL\.CYCLE\] policy_step=(\d+) "
    r"enclosing_global_step=(\d+|none)"
)
_DATASET_RE = re.compile(
    r"\[P57\.DATASET\] MATERIALIZED_PASS candidate=(\S+) split=(\S+) "
    r"train_rows=(\d+) eval_rows=(\d+) "
    r"train_sha256=([0-9a-f]{64}) eval_sha256=([0-9a-f]{64})"
)
_SEED_RE = re.compile(
    r"\[P57\.SEED\] CONTRACT_PASS data_shuffle_seed=(\d+) "
    r"vllm_global_seed=(\d+) per_request_seed=unsupported"
)


def classify(
    run_log: Path,
    *,
    expected_updates: int,
    interval: int,
    held_out_rows: int,
    generations: int,
    workload_candidate: str,
    data_split: str,
) -> dict:
  if expected_updates <= 0 or interval <= 0:
    raise ValueError("expected updates and interval must be positive")
  if expected_updates % interval:
    raise ValueError("expected updates must be divisible by the eval interval")
  text = run_log.read_text(encoding="utf-8", errors="replace")
  enabled = _ENABLED_RE.findall(text)
  expected_enabled = [(str(interval), str(held_out_rows), str(generations))]
  if enabled != expected_enabled:
    raise ValueError(
        f"P57 evaluation enable receipt drifted: {enabled!r}"
    )

  expected_candidate = workload_candidate or "p45"
  expected_split = data_split or "legacy"
  datasets = _DATASET_RE.findall(text)
  if len(datasets) != 1:
    raise ValueError(f"P57 dataset receipt count drifted: {datasets!r}")
  candidate, split, train_rows, eval_rows, train_sha, eval_sha = datasets[0]
  expected_hashes = (
      p57_workloads.PRIMARY_DATASET_SHA256[
          (expected_candidate, expected_split, "train", 10_000)
      ],
      p57_workloads.PRIMARY_DATASET_SHA256[
          (expected_candidate, expected_split, "eval", held_out_rows)
      ],
  )
  if (
      candidate != expected_candidate
      or split != expected_split
      or train_rows != "10000"
      or eval_rows != str(held_out_rows)
      or (train_sha, eval_sha) != expected_hashes
  ):
    raise ValueError(
        "P57 dataset identity drifted: "
        f"actual={datasets[0]!r} expected_candidate={expected_candidate} "
        f"expected_split={expected_split} expected_hashes={expected_hashes}"
    )
  seeds = _SEED_RE.findall(text)
  if seeds != [("42", "0")]:
    raise ValueError(f"P57 seed receipt drifted: {seeds!r}")

  records = []
  for line in text.splitlines():
    if _JSON_MARKER not in line:
      continue
    payload = line.split(_JSON_MARKER, 1)[1]
    records.append(json.loads(payload))
  expected_steps = list(range(0, expected_updates + 1, interval))
  actual_steps = [record.get("policy_step") for record in records]
  if actual_steps != expected_steps:
    raise ValueError(
        "P57 evaluation schedule incomplete or duplicated: "
        f"actual={actual_steps} expected={expected_steps}"
    )
  raw_cycle_receipts = _CYCLE_RE.findall(text)
  cycle_receipts = [
      {
          "policy_step": int(policy_step),
          "enclosing_global_step": (
              None if enclosing_global_step == "none"
              else int(enclosing_global_step)
          ),
      }
      for policy_step, enclosing_global_step in raw_cycle_receipts
  ]
  expected_cycle_receipts = [
      {
          "policy_step": step,
          "enclosing_global_step": (
              None if step == expected_updates else step + 1
          ),
      }
      for step in expected_steps
  ]
  if cycle_receipts != expected_cycle_receipts:
    raise ValueError(
        "P57 evaluation cycle mapping incomplete or drifted: "
        f"actual={cycle_receipts!r} expected={expected_cycle_receipts!r}"
    )
  expected_rewards = held_out_rows * generations
  for record in records:
    if set(record) != {"n", "policy_step", "reward", "solve", "wall_seconds"}:
      raise ValueError(f"P57 evaluation JSON schema drifted: {sorted(record)}")
    if record["n"] != expected_rewards:
      raise ValueError(
          f"P57 evaluation coverage drifted at step {record['policy_step']}: "
          f"n={record['n']} expected={expected_rewards}"
      )
    for name in ("reward", "solve", "wall_seconds"):
      if not math.isfinite(float(record[name])):
        raise ValueError(
            f"P57 evaluation {name} is nonfinite at step {record['policy_step']}"
        )
    if not 0.0 <= float(record["solve"]) <= 1.0:
      raise ValueError(f"P57 evaluation solve fraction is outside [0,1]: {record}")
    if float(record["wall_seconds"]) < 0.0:
      raise ValueError(f"P57 evaluation wall time is negative: {record}")

  final = _FINAL_RE.findall(text)
  expected_final = [(
      str(expected_updates),
      str(held_out_rows),
      str(generations),
      str(expected_rewards),
      f"{float(records[-1]['reward']):.6f}",
      f"{float(records[-1]['solve']):.6f}",
  )]
  if final != expected_final:
    raise ValueError(f"P57 final evaluation receipt drifted: {final!r}")
  return {
      "schema": "p57-inprocess-evaluation-classification-v2",
      "verdict": "PASS",
      "expected_updates": expected_updates,
      "interval": interval,
      "steps": expected_steps,
      "held_out_rows": held_out_rows,
      "generations": generations,
      "rewards_per_step": expected_rewards,
      "experiment_seed": 42,
      "vllm_global_seed": 0,
      "dataset": {
          "candidate": candidate,
          "split": split,
          "train_sha256": train_sha,
          "eval_sha256": eval_sha,
      },
      "cycle_receipts": cycle_receipts,
      "records": records,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-log", type=Path, required=True)
  parser.add_argument("--expected-updates", type=int, required=True)
  parser.add_argument("--interval", type=int, default=50)
  parser.add_argument("--held-out-rows", type=int, default=100)
  parser.add_argument("--generations", type=int, default=8)
  parser.add_argument("--workload-candidate", default="")
  parser.add_argument("--data-split", default="")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = classify(
      args.run_log,
      expected_updates=args.expected_updates,
      interval=args.interval,
      held_out_rows=args.held_out_rows,
      generations=args.generations,
      workload_candidate=args.workload_candidate,
      data_split=args.data_split,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("x", encoding="utf-8") as output:
    json.dump(result, output, indent=2, sort_keys=True)
    output.write("\n")
  print(
      "P57_INPROCESS_EVAL_PASS "
      f"steps={','.join(map(str, result['steps']))} "
      f"rewards_per_step={result['rewards_per_step']} output={args.output}",
      flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
