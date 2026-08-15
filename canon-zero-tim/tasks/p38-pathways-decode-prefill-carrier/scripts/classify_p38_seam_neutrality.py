#!/usr/bin/env python3
"""Fail closed unless a seam-observed run preserves all alignment endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _read_report(path: Path) -> list[dict]:
  rows = [json.loads(line) for line in path.read_text().splitlines() if line]
  if len(rows) != 3:
    raise ValueError(f"expected exactly three alignment rounds in {path}")
  return rows


def classify(off_path: Path, observed_path: Path) -> dict:
  off_rows = _read_report(off_path)
  observed_rows = _read_report(observed_path)
  rounds = []
  for index, (off, observed) in enumerate(zip(off_rows, observed_rows)):
    if off.get("step") != observed.get("step"):
      raise ValueError(f"round {index}: step drift")
    for key in ("tokens", "action_mask"):
      if off["hashes"][key] != observed["hashes"][key]:
        raise ValueError(f"round {index}: {key} drift")
    for key in ("S_decode", "S_prefill", "T_old"):
      if off["masked_hashes"][key] != observed["masked_hashes"][key]:
        raise ValueError(f"round {index}: {key} endpoint drift")
    for boundary in ("S_decode_vs_S_prefill", "S_prefill_vs_T_old"):
      if off["boundaries"][boundary] != observed["boundaries"][boundary]:
        raise ValueError(f"round {index}: {boundary} metric drift")
    rounds.append({
        "round": index,
        "step": off["step"],
        "tokens_sha256": off["hashes"]["tokens"],
        "s_decode_sha256": off["masked_hashes"]["S_decode"],
        "s_prefill_sha256": off["masked_hashes"]["S_prefill"],
        "t_old_sha256": off["masked_hashes"]["T_old"],
    })
  return {
      "schema": "p38-seam-neutrality-v1",
      "status": "PASS",
      "classification": "observer_endpoint_bitwise_neutral",
      "off_report": str(off_path),
      "off_report_sha256": hashlib.sha256(off_path.read_bytes()).hexdigest(),
      "observed_report": str(observed_path),
      "observed_report_sha256": hashlib.sha256(
          observed_path.read_bytes()).hexdigest(),
      "rounds": rounds,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--off", type=Path, required=True)
  parser.add_argument("--observed", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  result = classify(args.off, args.observed)
  payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
  if args.output:
    args.output.write_text(payload)
  print(payload, end="")


if __name__ == "__main__":
  main()
