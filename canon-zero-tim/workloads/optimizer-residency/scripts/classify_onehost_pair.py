#!/usr/bin/env python3
"""Classify one bounded offload-versus-resident GSM8K update pair."""

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
    for snapshot in record.get(key, []):
      value = snapshot.get("peak_bytes_in_use")
      if isinstance(value, int):
        values.append(value)
  return max(values) if values else None


def _validate_arm(
    record: dict[str, Any], *, placement: str, memory_kind: str
) -> list[str]:
  reasons = []

  def require(condition: bool, reason: str) -> None:
    if not condition:
      reasons.append(reason)

  require(record.get("verdict") == "PASS", "verdict")
  require(record.get("commits") == 1, "commits")
  require(record.get("train_steps_before") == 0, "train_steps_before")
  require(record.get("train_steps_after") == 1, "train_steps_after")
  require(record.get("optimizer_placement") == placement, "placement")
  require(
      record.get("optimizer_memory_kinds_before") == [memory_kind],
      "memory_before",
  )
  require(
      record.get("optimizer_memory_kinds_after") == [memory_kind],
      "memory_after",
  )
  require(record.get("optimizer_transaction_valid") is True, "transaction")
  require(record.get("reference_changed_paths") == [], "reference_changed")
  require(record.get("accumulator_changed_paths") == [], "accumulator_reset")
  require(record.get("dp_replicas_exact") is True, "replicas")
  require(record.get("gradient_finite") is True, "gradient_finite")
  evidence = record.get("commit_evidence", {})
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
  for key in (
      "optimizer_logical_bytes",
      "optimizer_h2d_seconds",
      "adam_commit_seconds",
      "optimizer_d2h_seconds",
      "optimizer_transaction_seconds",
  ):
    value = timing.get(key)
    require(
        isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
        f"timing.{key}",
    )
  require(_peak_hbm(record) is not None, "peak_hbm")
  return reasons


def classify(
    offload_path: Path, resident_path: Path
) -> dict[str, Any]:
  offload = _load(offload_path)
  resident = _load(resident_path)
  reasons = [
      f"offload.{reason}"
      for reason in _validate_arm(
          offload,
          placement="pinned-host-offload",
          memory_kind="pinned_host",
      )
  ]
  reasons.extend(
      f"resident.{reason}"
      for reason in _validate_arm(
          resident,
          placement="device-resident",
          memory_kind="device",
      )
  )

  for key in ("state_fingerprints_before", "state_fingerprints_after"):
    if offload.get(key) != resident.get(key):
      reasons.append(f"pair.{key}")
  for key in ("micro_gradient_norms", "alignment_hashes"):
    if offload.get(key) != resident.get(key):
      reasons.append(f"pair.{key}")

  offload_evidence = dict(offload.get("commit_evidence", {}))
  resident_evidence = dict(resident.get("commit_evidence", {}))
  offload_timing = offload_evidence.pop("optimizer_timing", {})
  resident_timing = resident_evidence.pop("optimizer_timing", {})
  if offload_evidence != resident_evidence:
    reasons.append("pair.commit_evidence")

  offload_seconds = offload_timing.get("optimizer_transaction_seconds")
  resident_seconds = resident_timing.get("optimizer_transaction_seconds")
  speedup = None
  if (
      isinstance(offload_seconds, (int, float))
      and isinstance(resident_seconds, (int, float))
      and resident_seconds > 0
  ):
    speedup = offload_seconds / resident_seconds

  return {
      "verdict": "PASS" if not reasons else "FAIL",
      "scope": "DP1xTP4 Qwen3-1.7B one-update canary",
      "bitwise_equal": not any(reason.startswith("pair.") for reason in reasons),
      "offload": {
          "optimizer_timing": offload_timing,
          "elapsed_seconds": offload.get("elapsed_seconds"),
          "peak_hbm_bytes": _peak_hbm(offload),
      },
      "resident": {
          "optimizer_timing": resident_timing,
          "elapsed_seconds": resident.get("elapsed_seconds"),
          "peak_hbm_bytes": _peak_hbm(resident),
      },
      "optimizer_transaction_speedup": speedup,
      "evidence_sha256": {
          "offload": _sha256(offload_path),
          "resident": _sha256(resident_path),
      },
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--offload", required=True, type=Path)
  parser.add_argument("--resident", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  record = classify(args.offload, args.resident)
  args.output.write_text(
      json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print("[P41.OPTIMIZER] JSON " + json.dumps(record, sort_keys=True))
  return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
