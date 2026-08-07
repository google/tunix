#!/usr/bin/env python3
"""Fail-closed classifier for the DP16 checkpoint-forward stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


_JSON_RE = re.compile(r"^\[P32\.RC\] JSON (\{.*\})$", re.MULTILINE)
_COUNTS = {
    "forward": 2,
    "backward": 0,
    "optimizer_updates": 0,
    "training_steps": 0,
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
  if stage != "checkpoint-forward":
    reasons.append(f"stage: expected checkpoint-forward, got {stage!r}")
  if expected_stage is not None and stage != expected_stage:
    reasons.append(f"stage: expected {expected_stage}, got {stage}")
  _exactly_one(
      text,
      r"^\[P32\.RC\] START stage=checkpoint\-forward .*",
      "start",
      reasons,
  )
  _exactly_one(
      text,
      r"^\[P32\.RC\] VERDICT PASS stage=checkpoint\-forward$",
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
    checkpoint = model.get("checkpoint", {})
    if (
        checkpoint.get("files", 0) <= 0
        or checkpoint.get("bytes", 0) <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", checkpoint.get("manifest_sha256", ""))
    ):
      reasons.append("checkpoint: manifest identity is incomplete")
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
    if record.get("execution") != _COUNTS:
      reasons.append("execution: exact checkpoint-forward counters changed")
    before = record.get("parameter_sample_sha256_before", "")
    after = record.get("parameter_sample_sha256_after", "")
    if not re.fullmatch(r"[0-9a-f]{64}", before):
      reasons.append("parameters: before fingerprint is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", after):
      reasons.append("parameters: after fingerprint is invalid")
    if before != after:
      reasons.append("parameters: checkpoint-forward mutated the model")
  return {
      "status": "PASS" if not reasons else "INCONCLUSIVE",
      "reasons": reasons,
      "record": record,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("log", type=Path)
  parser.add_argument("--stage", choices=("checkpoint-forward",))
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
