#!/usr/bin/env python3
"""Fail-closed classifier for the P32.D0 CPU admission gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


PREFIX = "P32.D0.CONTRACT "


def classify(text):
  reasons = []
  records = [
      json.loads(line[len(PREFIX):])
      for line in text.splitlines()
      if line.startswith(PREFIX)
  ]
  if len(records) != 1:
    reasons.append(f"contract_record_count={len(records)} expected=1")
  record = records[0] if len(records) == 1 else {}
  if record.get("status") != "pass":
    reasons.append("contract_status_not_pass")
  if record.get("topology") != {"dp": 2, "tp": 2, "devices": 4}:
    reasons.append("topology_mismatch")
  batch = record.get("batch", {})
  expected_batch = {
      "global_prompts": 4,
      "num_generations": 8,
      "global_trajectories": 32,
      "local_prompts": 2,
      "local_trajectories": 16,
      "rank_counts": [16, 16],
  }
  if batch != expected_batch:
    reasons.append("batch_contract_mismatch")
  if record.get("negative_count") != 3:
    reasons.append("negative_count_mismatch")
  negatives = record.get("negatives", [])
  if len(negatives) != 3 or not all(
      item.get("rejected") is True for item in negatives
  ):
    reasons.append("negative_control_not_rejected")
  if record.get("fixed_sum") != [0.0, 3.0]:
    reasons.append("fixed_sum_mismatch")
  passed = re.findall(r"(?m)^13 passed, .* in [0-9.]+s$", text)
  if len(passed) != 1:
    reasons.append(f"pytest_summary_count={len(passed)} expected=1")
  if "FAILED" in text or "ERROR" in text:
    reasons.append("test_failure_marker_present")
  return {
      "status": "pass" if not reasons else "fail",
      "verdict": "P32_DP2TP2_D0_PASS" if not reasons else "P32_DP2TP2_D0_FAIL",
      "reasons": reasons,
      "contract_records": len(records),
      "pytest_summaries": len(passed),
  }


def self_test():
  record = {
      "status": "pass",
      "topology": {"dp": 2, "tp": 2, "devices": 4},
      "batch": {
          "global_prompts": 4,
          "num_generations": 8,
          "global_trajectories": 32,
          "local_prompts": 2,
          "local_trajectories": 16,
          "rank_counts": [16, 16],
      },
      "fixed_sum": [0.0, 3.0],
      "negative_count": 3,
      "negatives": [{"rejected": True}] * 3,
  }
  positive = PREFIX + json.dumps(record) + "\n13 passed, 2 warnings in 1.00s\n"
  assert classify(positive)["status"] == "pass"
  negatives = [
      positive.replace("13 passed", "12 passed"),
      positive.replace('"negative_count": 3', '"negative_count": 2'),
      positive + "FAILED test\n",
  ]
  assert all(classify(value)["status"] == "fail" for value in negatives)
  print("P32.D0.CLASSIFIER_SELFTEST positives=1 negatives=3", flush=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("raw_log", type=Path)
  parser.add_argument("output_json", type=Path)
  args = parser.parse_args()
  self_test()
  result = classify(args.raw_log.read_text(encoding="utf-8", errors="replace"))
  args.output_json.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(result["verdict"], flush=True)
  if result["status"] != "pass":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
