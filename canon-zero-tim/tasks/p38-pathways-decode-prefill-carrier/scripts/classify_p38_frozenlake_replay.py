#!/usr/bin/env python3
"""Classify measurement integrity separately from the P38 carrier result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def classify(report: dict) -> dict:
  reasons = []
  if report.get("schema") != "p38-frozenlake-causal-replay-v1":
    reasons.append("unexpected replay schema")
  if report.get("measurement_status") != "COMPLETE":
    reasons.append("measurement did not complete")
  if report.get("no_backward") is not True:
    reasons.append("run did not attest no-backward execution")
  if report.get("no_optimizer") is not True:
    reasons.append("run did not attest zero optimizer commits")
  if report.get("weight_attestation", {}).get("equal") is not True:
    reasons.append("actor and live engine weights were not bitwise equal")
  if report.get("geometry", {}).get("prefix_cache") is not False:
    reasons.append("prefix cache was not disabled")
  schedules = report.get("schedules", [])
  by_arm = {item.get("arm"): item for item in schedules}
  if set(by_arm) != {"R0", "R1", "REF"}:
    reasons.append("R0/R1/reference schedules are incomplete")
  elif (
      by_arm["R0"].get("provenance") != "mask-derived-v1"
      or by_arm["R1"].get("provenance") != "mask-derived-v1"
      or by_arm["REF"].get("provenance") != "canonical-fixed-chunk-v1"
  ):
    reasons.append("schedule provenance changed")
  repeats = report.get("repeat_comparisons", {})
  if set(repeats) != {"R0", "R1", "REF"}:
    reasons.append("repeat measurements are incomplete")
  else:
    for arm, stages in repeats.items():
      if not stages or any(stage.get("exact") is not True for stage in stages.values()):
        reasons.append(f"{arm} repeat is not bitwise exact")
  negative = report.get("negative_control", {})
  if negative.get("exact") is not False or negative.get("differing_elements") != 1:
    reasons.append("one-bit negative control did not produce one red element")
  classification = report.get("classification", "")
  admitted = {
      "MULTITURN_SCHEDULE_CARRIER_CANDIDATE",
      "LOCAL_CARRIER_NOT_REPRODUCED",
      "LOCAL_CARRIER_NOT_ISOLATED",
  }
  if classification not in admitted:
    reasons.append(f"unexpected carrier classification: {classification!r}")
  e0_lite = report.get("e0_lite_classification", "")
  e0_lite_admitted = {
      "E0_LITE_REPRODUCED",
      "E0_LITE_ENVELOPE_NOT_REPRODUCED",
      "E0_LITE_PREREQUISITE_FAILED",
  }
  if e0_lite not in e0_lite_admitted:
    reasons.append(f"unexpected E0-lite classification: {e0_lite!r}")
  comparisons = report.get("replay_vs_captured", {})
  if set(comparisons) != {"R0", "R1", "REF"}:
    reasons.append("replay-to-production comparisons are incomplete")
  elif any(
      set(values) != {"S_decode", "S_prefill", "T_old"}
      for values in comparisons.values()
  ):
    reasons.append("replay-to-production boundary set is incomplete")
  return {
      "verdict": "PASS" if not reasons else "FAIL",
      "scope": "measurement-integrity-only",
      "carrier_classification": classification,
      "e0_lite_classification": e0_lite,
      "production_repair_admitted": False,
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--report", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  report = json.loads(args.report.read_text(encoding="utf-8"))
  result = classify(report)
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  args.output.write_text(
      json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps(result, sort_keys=True))
  return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
