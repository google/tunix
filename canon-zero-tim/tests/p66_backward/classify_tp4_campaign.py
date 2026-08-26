#!/usr/bin/env python3
"""Classify the four-arm P66 full-depth TP4 causal campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ARMS = ("tp4-serial", "tp4-p59-old", "tp4-p59", "tp4-gather-off")
SUCCESS_ARMS = ("tp4-serial", "tp4-p59", "tp4-gather-off")


def _json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
  return [
      json.loads(line)
      for line in path.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]


def _ratio(candidate: float, control: float) -> float:
  if control == 0.0:
    return 1.0 if candidate == 0.0 else math.inf
  return candidate / control


def classify(
    *,
    classifications: dict[str, Path],
    updates: dict[str, Path],
    pre_alignments: dict[str, Path],
) -> dict[str, Any]:
  classified = {arm: _json(path) for arm, path in classifications.items()}
  update = {arm: _json(path) for arm, path in updates.items()}
  pre = {arm: _jsonl(path) for arm, path in pre_alignments.items()}
  contract_reasons = []
  if classified["tp4-p59-old"].get("verdict") != "EXPECTED_RED":
    contract_reasons.append("unsafe_arm_not_red")
  for arm in SUCCESS_ARMS:
    if classified[arm].get("verdict") != "PASS":
      contract_reasons.append(f"{arm}.classification")
  pre_hashes = {
      arm: rows[0].get("hashes") if len(rows) == 1 else None
      for arm, rows in pre.items()
  }
  if any(value is None for value in pre_hashes.values()) or len(
      {json.dumps(value, sort_keys=True) for value in pre_hashes.values()}
  ) != 1:
    contract_reasons.append("same_input_pre_alignment")
  alignment_hashes = {
      arm: update[arm].get("alignment_hashes") for arm in SUCCESS_ARMS
  }
  if len(
      {json.dumps(value, sort_keys=True) for value in alignment_hashes.values()}
  ) != 1:
    contract_reasons.append("same_input_group_hashes")
  model_samples = {
      arm: update[arm].get("model_before_sample") for arm in SUCCESS_ARMS
  }
  if len(
      {json.dumps(value, sort_keys=True) for value in model_samples.values()}
  ) != 1:
    contract_reasons.append("same_model_before_sample")

  serial_gradient = float(update["tp4-serial"]["gradient"]["stable_norm"])
  serial_components = update["tp4-serial"]["layerwise_profile"]["components"]
  comparisons = {}
  close = {}
  for arm in ("tp4-p59", "tp4-gather-off"):
    candidate_gradient = float(update[arm]["gradient"]["stable_norm"])
    candidate_components = update[arm]["layerwise_profile"]["components"]
    norm_ratio = _ratio(candidate_gradient, serial_gradient)
    component_ratios = {
        name: _ratio(float(candidate_components[name]), float(control))
        for name, control in serial_components.items()
    }
    comparisons[arm] = {
        "mapped_gradient_norm_ratio_to_serial": norm_ratio,
        "component_ratio_to_serial": component_ratios,
    }
    close[arm] = (
        0.25 <= norm_ratio <= 4.0
        and all(0.125 <= value <= 8.0 for value in component_ratios.values())
    )
  old_engine_norm = classified["tp4-p59-old"].get("engine_norm")
  old_red = classified["tp4-p59-old"].get("verdict") == "EXPECTED_RED"
  row_cotangent = {
      arm: classified[arm].get("row_cotangent_summary") for arm in ARMS
  }
  if contract_reasons:
    verdict = "INCONCLUSIVE_CARRIER"
  elif old_red and close["tp4-p59"] and close["tp4-gather-off"]:
    verdict = "H1_VMA_SUPPORTED"
  elif old_red and not close["tp4-p59"] and close["tp4-gather-off"]:
    verdict = "H2_FIXED_GATHER_SUPPORTED"
  else:
    verdict = "P66_TP4_REPAIR_REJECT"
  return {
      "schema": "canon-p66-tp4-causal-campaign-v1",
      "verdict": verdict,
      "scope": "one-host full-Qwen3-1.7B 28-layer DP1xTP4 group0 proxy",
      "target_certification": False,
      "optimizer_commits": 0,
      "same_input_pre_alignment": "same_input_pre_alignment" not in contract_reasons,
      "same_input_group_hashes": "same_input_group_hashes" not in contract_reasons,
      "same_model_before_sample": "same_model_before_sample" not in contract_reasons,
      "unsafe_engine_norm": old_engine_norm,
      "row_cotangent": row_cotangent,
      "comparisons": comparisons,
      "contract_reasons": contract_reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  for arm in ARMS:
    key = arm.replace("-", "_")
    parser.add_argument(f"--{arm}-classification", dest=f"{key}_classification", type=Path, required=True)
    parser.add_argument(f"--{arm}-pre", dest=f"{key}_pre", type=Path, required=True)
    if arm in SUCCESS_ARMS:
      parser.add_argument(f"--{arm}-update", dest=f"{key}_update", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  result = classify(
      classifications={
          arm: getattr(args, f"{arm.replace('-', '_')}_classification")
          for arm in ARMS
      },
      updates={
          arm: getattr(args, f"{arm.replace('-', '_')}_update")
          for arm in SUCCESS_ARMS
      },
      pre_alignments={
          arm: getattr(args, f"{arm.replace('-', '_')}_pre")
          for arm in ARMS
      },
  )
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(f"P66_TP4_CAMPAIGN verdict={result['verdict']}")
  return 0 if result["verdict"] in (
      "H1_VMA_SUPPORTED", "H2_FIXED_GATHER_SUPPORTED"
  ) else 1


if __name__ == "__main__":
  raise SystemExit(main())
