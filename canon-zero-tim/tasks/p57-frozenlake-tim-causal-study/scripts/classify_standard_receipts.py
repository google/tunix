#!/usr/bin/env python3
"""Require complete Native64 trainer-old/no-TIS batch-source receipts.

This gate attests treatment coverage only. Existing full-training, gradient,
checkpoint and evaluation classifiers remain mandatory.
"""

import argparse
import json
from pathlib import Path
import re

PREFIX = "[P57.TIM_STANDARD]"
RECEIPT = re.compile(
    r"\[P57\.TIM_STANDARD\] PASS step=(\d+) rows=(\d+) groups=([0-9,]+) "
    r"old_logps=trainer tis_weights=absent rollout_logps=present "
    r"trainer_rescore=training-input policy_version=matched"
)


def classify(text: str, expected_updates: int) -> dict:
  if expected_updates <= 0:
    raise ValueError("expected_updates must be positive")
  seen = set()
  receipts = 0
  for line in text.splitlines():
    if "[P57.TIM_PURITY]" in line:
      raise ValueError("legacy Bypass/TIS receipt in Standard run")
    if PREFIX not in line:
      continue
    match = RECEIPT.fullmatch(line[line.index(PREFIX):].strip())
    if match is None:
      raise ValueError("malformed or non-PASS Standard receipt")
    step, rows = int(match[1]), int(match[2])
    groups = [int(value) for value in match[3].split(",")]
    if step >= expected_updates or rows != 8 * len(groups) or len(set(groups)) != len(groups):
      raise ValueError("Standard receipt row/group geometry drifted")
    for group in groups:
      if group // 32 != step or group in seen:
        raise ValueError("Standard group is duplicated or belongs to another step")
      seen.add(group)
    receipts += 1
  if seen != set(range(expected_updates * 32)):
    raise ValueError("Standard full-horizon group coverage incomplete")
  return {
      "verdict": "PASS", "arm": "standard", "updates": expected_updates,
      "groups": len(seen), "trajectories": len(seen) * 8, "receipts": receipts,
      "claim": "batch-source coverage only; full-training postflight also required",
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-log", type=Path, required=True)
  parser.add_argument("--expected-updates", type=int, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  try:
    result = classify(args.run_log.read_text(encoding="utf-8"), args.expected_updates)
  except (ValueError, OSError) as exc:
    result = {"verdict": "FAIL", "reason": str(exc)}
  with args.output.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write("\n")
  print(f"P57_STANDARD_RECEIPTS_{result['verdict']}")
  raise SystemExit(0 if result["verdict"] == "PASS" else 1)


if __name__ == "__main__":
  main()
