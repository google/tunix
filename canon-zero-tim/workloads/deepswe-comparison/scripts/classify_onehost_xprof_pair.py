#!/usr/bin/env python3
"""Classify matched work across P58 Native and Zero one-host profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--native", type=Path, required=True)
  parser.add_argument("--zero-hp", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  native = json.loads(args.native.read_text(encoding="utf-8"))
  zero = json.loads(args.zero_hp.read_text(encoding="utf-8"))
  if args.output.exists():
    raise FileExistsError(args.output)
  reasons = []
  if native.get("arm") != "native" or zero.get("arm") != "zero-hp":
    reasons.append("arm_identity")
  if native.get("source_sha") != zero.get("source_sha"):
    reasons.append("source_sha")
  if native.get("source_diff_sha256") != zero.get("source_diff_sha256"):
    reasons.append("source_diff_sha256")
  if native.get("expected_hostname") != zero.get("expected_hostname"):
    reasons.append("hostname")
  arm_verdicts = (native.get("verdict"), zero.get("verdict"))
  work_match = native.get("work_hashes") == zero.get("work_hashes")
  if "FAIL" in arm_verdicts:
    verdict = "FAIL"
  elif any(value != "PASS" for value in arm_verdicts):
    verdict = "INCONCLUSIVE_CAPTURE"
  elif reasons:
    verdict = "INCONCLUSIVE_INPUT_MISMATCH"
  elif not work_match:
    reasons.append("work_hashes")
    verdict = "INCONCLUSIVE_INPUT_MISMATCH"
  else:
    verdict = "PASS"
  result = {
      "schema": "canon.p58.onehost-xprof.pair.v1",
      "verdict": verdict,
      "arm_verdicts": {"native": arm_verdicts[0], "zero-hp": arm_verdicts[1]},
      "source_sha": native.get("source_sha"),
      "source_diff_sha256": native.get("source_diff_sha256"),
      "expected_hostname": native.get("expected_hostname"),
      "work_hashes_match": work_match,
      "reasons": reasons,
      "claim": (
          "matched DP1xTP4 operation-attribution pair"
          if verdict == "PASS"
          else "no causal performance delta"
      ),
  }
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(json.dumps(result, sort_keys=True))
  return 0 if verdict == "PASS" else 3 if verdict.startswith("INCONCLUSIVE") else 1


if __name__ == "__main__":
  sys.exit(main())
