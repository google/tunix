#!/usr/bin/env python3
"""Classify three returned P58 checked-VMA on/off/on diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_EXPECTED = {
    "on-a": "on",
    "off": "off",
    "on-b": "on",
}


def _load(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"classification is not an object: {path}")
  return value


def classify(
    *, wave_verify: Path, on_a: Path, off: Path, on_b: Path
) -> dict[str, Any]:
  wave = _load(wave_verify)
  if wave.get("schema") != "canon.p58.checked-vma-aba-verify.v1" or wave.get(
      "verdict"
  ) != "PASS":
    raise ValueError("wave verification is not PASS")
  records = {
      "on-a": _load(on_a),
      "off": _load(off),
      "on-b": _load(on_b),
  }
  reasons = []
  outcomes = {}
  for arm, selector in _EXPECTED.items():
    record = records[arm]
    if record.get("schema") != "canon.p58.checked-vma-diagnostic.v2":
      reasons.append(f"{arm}:schema")
    if record.get("verdict") != "PASS":
      reasons.append(f"{arm}:verdict={record.get('verdict')}")
    if record.get("selector") != selector:
      reasons.append(f"{arm}:selector={record.get('selector')}")
    if record.get("source_commit") != wave.get("source_commit"):
      reasons.append(f"{arm}:source_commit")
    if record.get("B_C_differing_bytes") != 0:
      reasons.append(f"{arm}:B-C={record.get('B_C_differing_bytes')}")
    if record.get("backward") != 0 or record.get("optimizer_commits") != 0:
      reasons.append(f"{arm}:training_activity")
    outcomes[arm] = record.get("outcome")

  on_red = "A_B_RED_WITH_CHECKED_VMA_ON"
  on_exact = "A_B_EXACT_WITH_CHECKED_VMA_ON"
  off_red = "A_B_RED_WITH_CHECKED_VMA_OFF"
  off_exact = "A_B_EXACT_WITH_CHECKED_VMA_OFF"
  if reasons:
    decision = "INVALID_EVIDENCE"
  elif outcomes["on-a"] != outcomes["on-b"]:
    decision = "INCONCLUSIVE_ON_REPLICATION"
  elif outcomes["on-a"] == on_red and outcomes["off"] == off_exact:
    decision = "CHECKED_VMA_CAUSAL_REPRODUCED"
  elif outcomes["on-a"] == on_red and outcomes["off"] == off_red:
    decision = "CHECKED_VMA_NOT_SUFFICIENT"
  elif outcomes["on-a"] == on_exact and outcomes["off"] == off_exact:
    decision = "BASELINE_RED_NOT_REPRODUCED"
  else:
    decision = "INCONCLUSIVE_PATTERN"
  verdict = (
      "PASS"
      if decision in (
          "CHECKED_VMA_CAUSAL_REPRODUCED",
          "CHECKED_VMA_NOT_SUFFICIENT",
          "BASELINE_RED_NOT_REPRODUCED",
      )
      else "INCONCLUSIVE" if not reasons else "FAIL"
  )
  return {
      "schema": "canon.p58.checked-vma-aba-classification.v1",
      "verdict": verdict,
      "decision": decision,
      "source_commit": wave.get("source_commit"),
      "wave_id": wave.get("wave_id"),
      "outcomes": outcomes,
      "reasons": reasons,
      "backward": 0,
      "optimizer_commits": 0,
      "claim": (
          "The result compares three independent exact-geometry Step-0 "
          "prechecks: one OFF control and two ON replicates. A concurrent "
          "launch is not a temporal ABA sandwich. It does not require "
          "cross-run token identity and does not certify backward, optimizer, "
          "or full training."
      ),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--wave-verify", type=Path, required=True)
  parser.add_argument("--on-a", type=Path, required=True)
  parser.add_argument("--off", type=Path, required=True)
  parser.add_argument("--on-b", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite output: {args.output}")
  result = classify(
      wave_verify=args.wave_verify,
      on_a=args.on_a,
      off=args.off,
      on_b=args.on_b,
  )
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(
      "P58_CHECKED_VMA_ABA_CLASSIFICATION "
      f"verdict={result['verdict']} decision={result['decision']} "
      "backward=0 optimizer_commits=0",
      flush=True,
  )
  return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
