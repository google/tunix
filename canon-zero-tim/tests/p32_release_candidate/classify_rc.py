#!/usr/bin/env python3
"""Fail-closed classifier for one DP16xTP4 release-candidate stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


_JSON_RE = re.compile(r"^\[P32\.RC\] JSON (\{.*\})$", re.MULTILINE)
_STAGE_COUNTS = {
    "checkpoint-forward": {
        "forward": 2,
        "backward": 0,
        "optimizer_updates": 0,
        "training_steps": 0,
    },
    "backward": {
        "forward": 36,
        "backward": 32,
        "optimizer_updates": 0,
        "training_steps": 0,
    },
    "one-update": {
        "forward": 19,
        "backward": 16,
        "optimizer_updates": 1,
        "training_steps": 1,
    },
    "three-update": {
        "forward": 53,
        "backward": 48,
        "optimizer_updates": 3,
        "training_steps": 3,
    },
}


def _exactly_one(text: str, pattern: str, label: str, reasons: list[str]):
  count = len(re.findall(pattern, text, re.MULTILINE))
  if count != 1:
    reasons.append(f"{label}: expected exactly one marker, found {count}")


def classify_text(text: str, expected_stage: str | None = None) -> dict[str, Any]:
  reasons: list[str] = []
  for marker in (
      "Traceback (most recent call last):",
      "[entrypoint] FATAL:",
      "Check failed:",
      "pthread_create() failed",
      "RESOURCE_EXHAUSTED",
      "SKIP_TAINTED",
      "TARGET NOT RUN",
      "INCONCLUSIVE",
  ):
    if marker in text:
      reasons.append(f"forbidden marker present: {marker}")
  _exactly_one(
      text,
      r"^\[T1\.PATHWAYS\] required=1 initialized=1 status=ok$",
      "pathways",
      reasons,
  )
  matches = _JSON_RE.findall(text)
  if len(matches) != 1:
    reasons.append(f"json: expected exactly one record, found {len(matches)}")
    record: dict[str, Any] = {}
  else:
    try:
      record = json.loads(matches[0])
    except json.JSONDecodeError as exc:
      reasons.append(f"json: invalid record: {exc}")
      record = {}
  stage = record.get("stage") if record else expected_stage
  if expected_stage is not None and stage != expected_stage:
    reasons.append(f"stage: expected {expected_stage}, got {stage}")
  if stage not in _STAGE_COUNTS:
    reasons.append(f"stage: unsupported value {stage!r}")
  if stage:
    _exactly_one(
        text,
        rf"^\[P32\.RC\] START stage={re.escape(stage)} .*",
        "start",
        reasons,
    )
    _exactly_one(
        text,
        rf"^\[P32\.RC\] VERDICT PASS stage={re.escape(stage)}$",
        "verdict",
        reasons,
    )
  if record:
    if record.get("attempt") != 0:
      reasons.append("attempt: release evidence must be attempt 0")
    if record.get("topology") != {
        "devices": 64,
        "dp": 16,
        "mesh_shape": [16, 4],
        "tp": 4,
        "unique_devices": 64,
    }:
      reasons.append("topology: exact DP16xTP4 full-slice contract failed")
    if record.get("batch") != {
        "global_trajectories": 256,
        "local_trajectories": 16,
        "sample_to_rank_mapping": "frozen-contiguous-16",
        "sequence_length": 16,
    }:
      reasons.append("batch: frozen 256/16 placement contract changed")
    if record.get("scope") != {
        "production_training_admitted": False,
        "rollout_engine_initialized": False,
        "zero_tim_alignment": "NOT_MEASURED",
    }:
      reasons.append("scope: systems RC must not claim zero-TIM or production admission")
    model = record.get("model", {})
    if model.get("name") != "qwen3-8b" or model.get("checkpoint_loaded") is not True:
      reasons.append("model: real Qwen3-8B checkpoint was not loaded")
    if model.get("attention_backend") != "dense-reference":
      reasons.append(
          "model: bounded RC must declare the dense-reference attention backend"
      )
    checkpoint = model.get("checkpoint", {})
    if (
        checkpoint.get("files", 0) <= 0
        or checkpoint.get("bytes", 0) <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", checkpoint.get("manifest_sha256", ""))
    ):
      reasons.append("checkpoint: immutable identity is incomplete")
    inventory = model.get("inventory", {})
    if (
        inventory.get("leaves") != 399
        or inventory.get("arrays") != 399
        or inventory.get("dp_partitioned_leaves") != 0
        or inventory.get("tp_partitioned_leaves", 0) <= 0
        or inventory.get("memory_kinds") != ["device"]
    ):
      reasons.append("model: replicated-DP/TP-sharded inventory changed")
    if record.get("forward_repeat_exact") is not True:
      reasons.append("forward: repeat was not bitwise exact")
    if record.get("forward_shape") != [256, 151936]:
      reasons.append("forward: processed row shape changed")
    if record.get("execution") != _STAGE_COUNTS.get(stage):
      reasons.append("execution: exact stage counters changed")
    before = record.get("parameter_sample_sha256_before", "")
    after = record.get("parameter_sample_sha256_after", "")
    if not re.fullmatch(r"[0-9a-f]{64}", before):
      reasons.append("parameters: before fingerprint is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", after):
      reasons.append("parameters: after fingerprint is invalid")
    if stage in ("checkpoint-forward", "backward") and before != after:
      reasons.append("parameters: no-commit stage mutated the model")
    if stage in ("one-update", "three-update") and before == after:
      reasons.append("parameters: update stage did not mutate the model")
    if stage == "checkpoint-forward":
      forbidden_work = (
          record.get("third_program_exact"),
          record.get("gradient_health"),
          record.get("post_reduction_replicas_exact"),
      )
      if forbidden_work != (None, None, None):
        reasons.append("forward: backward-only fields were populated")
    else:
      if not isinstance(record.get("third_program_exact"), bool):
        reasons.append("THIRDPROG: observation is missing")
      health = record.get("gradient_health") or {}
      if (
          health.get("finite") is not True
          or health.get("nonzero", 0) <= 0
          or not isinstance(health.get("norm"), (int, float))
          or health.get("norm", 0.0) <= 0.0
      ):
        reasons.append("gradient: health contract failed")
      if record.get("rank_local_stats_distinct") is not True:
        reasons.append("gradient: rank-local contribution was not observed")
      if record.get("post_reduction_replicas_exact") is not True:
        reasons.append("gradient: reduced replicas were not exact")
      expected_transactions = {
          "backward": 2,
          "one-update": 1,
          "three-update": 3,
      }[stage]
      if record.get("dp_reduction_transactions") != expected_transactions:
        reasons.append("reducer: transaction count changed")
      if record.get("dp_rank_pullbacks_per_transaction") != 16:
        reasons.append("reducer: every DP rank must contribute exactly once")
      if record.get("dp_rank_ordered_additions_per_transaction") != 15:
        reasons.append("reducer: DP16 serial reference must perform 15 additions")
      if record.get("dp_reduction_rounds_per_transaction") != 15:
        reasons.append("reducer: registered rank-order depth changed")
      if stage == "backward" and record.get("gradient_repeat_exact") is not True:
        reasons.append("gradient: repeated backward changed bits")
    if stage != "checkpoint-forward":
      steps = record.get("step_records", [])
      expected_steps = (
          2 if stage == "backward" else _STAGE_COUNTS[stage]["training_steps"]
      )
      if len(steps) != expected_steps:
        reasons.append("gradient: step record count changed")
      third_program_observations = [
          entry.get("third_program_exact") for entry in steps
      ]
      if any(not isinstance(value, bool) for value in third_program_observations):
        reasons.append("THIRDPROG: per-transaction observation is missing")
      elif record.get("third_program_exact") != all(
          third_program_observations
      ):
        reasons.append("THIRDPROG: aggregate does not match transaction records")
      for index, entry in enumerate(steps):
        contributions = entry.get("rank_contribution_signature_sha256", [])
        if (
            len(contributions) != 16
            or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in contributions)
            or len(set(contributions)) != 16
        ):
          reasons.append(
              f"gradient: step {index} rank contribution signatures invalid"
          )
    if stage in ("one-update", "three-update"):
      if record.get("optimizer_state_memory_between_commits") != ["pinned_host"]:
        reasons.append("optimizer: state is not pinned-host between commits")
      if record.get("optimizer_state_memory_during_commit") != ["device"]:
        reasons.append("optimizer: state was not on device during commit")
      steps = record.get("step_records", [])
      if len(steps) != _STAGE_COUNTS[stage]["training_steps"]:
        reasons.append("updates: step record count changed")
      elif [entry.get("step") for entry in steps] != list(range(len(steps))):
        reasons.append("updates: step ids are not monotonic and contiguous")
      for index, entry in enumerate(steps):
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("parameter_sample_sha256", "")):
          reasons.append(f"updates: step {index} parameter fingerprint invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("optimizer_sample_sha256", "")):
          reasons.append(f"updates: step {index} optimizer fingerprint invalid")
  return {
      "status": "PASS" if not reasons else "INCONCLUSIVE",
      "reasons": reasons,
      "record": record,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("log", type=Path)
  parser.add_argument("--stage", choices=tuple(_STAGE_COUNTS))
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  result = classify_text(
      args.log.read_text(encoding="utf-8", errors="replace"), args.stage
  )
  rendered = json.dumps(result, indent=2, sort_keys=True)
  print(rendered)
  if args.output is not None:
    args.output.write_text(rendered + "\n", encoding="utf-8")
  return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
