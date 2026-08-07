#!/usr/bin/env python3
"""Fail-closed classifier for the P32 model-init-only artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


JSON_RE = re.compile(r"^\[P32\.INIT\] JSON (\{.*\})$", re.MULTILINE)


def _one(text: str, pattern: str, label: str, reasons: list[str]) -> None:
  count = len(re.findall(pattern, text, re.MULTILINE))
  if count != 1:
    reasons.append(f"{label}: expected exactly one marker, found {count}")


def classify_text(text: str) -> dict[str, Any]:
  reasons: list[str] = []
  for pattern, label in (
      (r"^\[T1\.PATHWAYS\] required=1 initialized=1 status=ok$", "pathways"),
      (r"^\[P32\.INIT\] START .*checkpoint_loaded=0 forward=0 backward=0 update=0$", "start"),
      (r"^\[P32\.INIT\] MESH shape=\(16, 4\) unique=64 full_slice=1$", "mesh"),
      (r"^\[P32\.INIT\] VERDICT PASS$", "verdict"),
  ):
    _one(text, pattern, label, reasons)
  matches = JSON_RE.findall(text)
  if len(matches) != 1:
    reasons.append(f"json: expected exactly one record, found {len(matches)}")
    record: dict[str, Any] = {}
  else:
    try:
      record = json.loads(matches[0])
    except json.JSONDecodeError as exc:
      reasons.append(f"json: invalid record: {exc}")
      record = {}

  if record:
    topology = record.get("topology", {})
    if topology != {
        "devices": 64,
        "dp": 16,
        "full_slice": True,
        "mesh_shape": [16, 4],
        "tp": 4,
        "unique_devices": 64,
    }:
      reasons.append("topology: exact DP16xTP4 full-slice contract failed")
    if record.get("attempt") != 0:
      reasons.append("attempt: model-init evidence must come from attempt 0")
    model = record.get("model", {})
    expected_model = {
        "name": "qwen3-8b",
        "layers": 36,
        "vocab": 151936,
        "embed": 4096,
        "hidden": 12288,
        "heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "compute_dtype": "<class 'jax.numpy.bfloat16'>",
        "param_dtype": "<class 'jax.numpy.float32'>",
        "checkpoint_loaded": False,
        "state_kind": "zero-structural",
    }
    if model != expected_model:
      reasons.append("model: Qwen3-8B structural contract changed")
    inventory = record.get("inventory", {})
    expected_leaves = {"model": 399, "optimizer": 799, "accumulator": 399}
    expected_bytes = {
        "model": 32_762_941_440,
        "optimizer": 65_525_882_884,
        "accumulator": 32_762_941_440,
    }
    expected_memory = {
        "model": ["device"],
        "optimizer": ["pinned_host"],
        "accumulator": ["device"],
    }
    if set(inventory) != set(expected_leaves):
      reasons.append("inventory: state classes changed")
    else:
      for label in expected_leaves:
        summary = inventory[label]
        if summary.get("leaves") != expected_leaves[label]:
          reasons.append(f"inventory: {label} leaf count changed")
        if summary.get("arrays") != expected_leaves[label]:
          reasons.append(f"inventory: {label} materialized array count changed")
        if summary.get("logical_bytes") != expected_bytes[label]:
          reasons.append(f"inventory: {label} logical bytes changed")
        if summary.get("dp_partitioned_leaves") != 0:
          reasons.append(f"inventory: {label} is DP-sharded")
        if summary.get("tp_partitioned_leaves", 0) <= 0:
          reasons.append(f"inventory: {label} is not TP-sharded")
        if summary.get("memory_kinds") != expected_memory[label]:
          reasons.append(f"inventory: {label} memory kind changed")
    optimizer = record.get("optimizer", {})
    if optimizer != {
        "name": "adamw",
        "learning_rate": 1.0e-6,
        "b1": 0.9,
        "b2": 0.95,
        "weight_decay": 0.0,
        "memory_kind": "pinned_host",
        "commits": 0,
    }:
      reasons.append("optimizer: exact zero-commit AdamW contract changed")
    if record.get("physical_bytes_per_device") != {
        "model": 8_190_735_360,
        "optimizer": 16_381_470_724,
        "accumulator": 8_190_735_360,
    }:
      reasons.append("physical bytes: per-device state allocation changed")
    execution = record.get("execution", {})
    if execution != {
        "backward": 0,
        "forward": 0,
        "optimizer_updates": 0,
        "training_steps": 0,
    }:
      reasons.append("execution: model-init-only performed forbidden work")
    wandb = record.get("wandb", {})
    if (
        not wandb.get("project")
        or not wandb.get("group")
        or not wandb.get("run_name")
        or wandb.get("network_initialized") is not False
    ):
      reasons.append("wandb: non-secret identity contract failed")
  forbidden = (
      "Traceback (most recent call last):",
      "[entrypoint] FATAL:",
      "SKIP_TAINTED",
      "TARGET NOT RUN",
  )
  for marker in forbidden:
    if marker in text:
      reasons.append(f"forbidden marker present: {marker}")
  return {
      "status": "PASS" if not reasons else "INCONCLUSIVE",
      "reasons": reasons,
      "record": record,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("log", type=Path)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  result = classify_text(args.log.read_text(encoding="utf-8", errors="replace"))
  rendered = json.dumps(result, indent=2, sort_keys=True)
  print(rendered)
  if args.output is not None:
    args.output.write_text(rendered + "\n", encoding="utf-8")
  return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
