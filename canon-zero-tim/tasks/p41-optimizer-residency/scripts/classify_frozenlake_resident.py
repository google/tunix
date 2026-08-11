#!/usr/bin/env python3
"""Classify one strict Qwen3-8B device-resident FrozenLake update."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected one JSON object in {path}")
  return value


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _peak_hbm(record: dict[str, Any]) -> int | None:
  values = []
  for key in ("hbm_before", "hbm_after_accumulation", "hbm_after_commit"):
    snapshots = record.get(key, [])
    if not isinstance(snapshots, list):
      continue
    for snapshot in snapshots:
      if not isinstance(snapshot, dict):
        continue
      value = snapshot.get("peak_bytes_in_use")
      if isinstance(value, int):
        values.append(value)
  return max(values) if values else None


def classify(update_path: Path) -> dict[str, Any]:
  record = _load(update_path)
  reasons: list[str] = []

  def require(condition: bool, reason: str) -> None:
    if not condition:
      reasons.append(reason)

  require(record.get("verdict") == "PASS", "verdict")
  require(record.get("microsteps") == 4, "microsteps")
  require(record.get("commits") == 1, "commits")
  require(record.get("train_steps_before") == 0, "train_steps_before")
  require(record.get("train_steps_after") == 1, "train_steps_after")
  require(record.get("optimizer_placement") == "device-resident", "placement")
  require(record.get("optimizer_memory_kinds_before") == ["device"], "memory_before")
  require(record.get("optimizer_memory_kinds_after") == ["device"], "memory_after")
  require(record.get("optimizer_transaction_valid") is True, "transaction")
  require(record.get("reference_changed_paths") == [], "reference_changed")
  require(record.get("accumulator_changed_paths") == [], "accumulator_reset")
  require(record.get("dp_replicas_exact") is True, "replicas")
  require(record.get("gradient_finite") is True, "gradient_finite")

  activity = record.get("gradient_activity")
  require(activity == [True, True, True, True], "gradient_activity")
  micro_norms = record.get("micro_gradient_norms")
  require(
      isinstance(micro_norms, list)
      and len(micro_norms) == 4
      and all(
          isinstance(value, (int, float))
          and math.isfinite(value)
          and value > 0
          for value in micro_norms
      ),
      "micro_gradient_norms",
  )

  evidence = record.get("commit_evidence", {})
  require(isinstance(evidence, dict), "commit_evidence")
  if isinstance(evidence, dict):
    require(evidence.get("gradient_finite") is True, "commit_gradient_finite")
    require(
        isinstance(evidence.get("gradient_nonzero_elements"), int)
        and evidence["gradient_nonzero_elements"] > 0,
        "gradient_nonzero",
    )
    require(
        isinstance(evidence.get("parameter_changed_elements"), int)
        and evidence["parameter_changed_elements"] > 0,
        "parameter_changed",
    )
    timing = evidence.get("optimizer_timing", {})
    require(isinstance(timing, dict), "optimizer_timing")
    if isinstance(timing, dict):
      for key in (
          "optimizer_logical_bytes",
          "optimizer_h2d_seconds",
          "adam_commit_seconds",
          "optimizer_d2h_seconds",
          "optimizer_transaction_seconds",
      ):
        value = timing.get(key)
        require(
            isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= 0,
            f"timing.{key}",
        )
      require(timing.get("optimizer_h2d_seconds") == 0, "resident_h2d")
      require(timing.get("optimizer_d2h_seconds") == 0, "resident_d2h")

  peak_hbm = _peak_hbm(record)
  require(peak_hbm is not None, "peak_hbm")
  return {
      "verdict": "PASS" if not reasons else "FAIL",
      "scope": "DP1xTP4 Qwen3-8B FrozenLake resident one-update capacity canary",
      "resident": {
          "elapsed_seconds": record.get("elapsed_seconds"),
          "peak_hbm_bytes": peak_hbm,
          "optimizer_timing": evidence.get("optimizer_timing", {})
          if isinstance(evidence, dict)
          else {},
      },
      "evidence_sha256": _sha256(update_path),
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--update", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  record = classify(args.update)
  args.output.write_text(
      json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print("[P41.FROZENLAKE] JSON " + json.dumps(record, sort_keys=True))
  return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
