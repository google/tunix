#!/usr/bin/env python3
"""Fail-closed observer-neutrality gate for the P66 TP4 VJP oracle arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REFERENCE_ARM = "tp4-p59"
ORACLE_ARM = "tp4-vma-oracle"
EXACT_UPDATE_FIELDS = (
    "alignment_hashes",
    "alignment_verdicts",
    "model_before_sample",
    "engine_vjp",
    "gradient",
    "gradient_sample",
    "state_changed_paths",
    "train_steps_before",
    "train_steps_after",
)


def _load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  values = [
      json.loads(line)
      for line in path.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]
  if not all(isinstance(value, dict) for value in values):
    raise ValueError(f"{path}: expected JSON objects")
  return values


def _without_arm(value: Any) -> Any:
  if not isinstance(value, dict):
    return value
  return {key: item for key, item in value.items() if key != "arm"}


def classify(
    *,
    reference_classification: Path,
    reference_pre_alignment: Path,
    reference_update: Path,
    oracle_classification: Path,
    oracle_pre_alignment: Path,
    oracle_update: Path,
) -> dict[str, Any]:
  reference_class = _load_json(reference_classification)
  oracle_class = _load_json(oracle_classification)
  reference_pre = _load_jsonl(reference_pre_alignment)
  oracle_pre = _load_jsonl(oracle_pre_alignment)
  reference = _load_json(reference_update)
  oracle = _load_json(oracle_update)
  contract_reasons = []
  if (
      reference_class.get("verdict") != "PASS"
      or reference_class.get("arm") != REFERENCE_ARM
      or reference_class.get("zero_tim")
      != {"expected_pass": 17, "observed_pass": 17, "observed_fail": 0}
  ):
    contract_reasons.append("reference_classification")
  if (
      oracle_class.get("verdict") != "PASS"
      or oracle_class.get("arm") != ORACLE_ARM
      or oracle_class.get("zero_tim")
      != {"expected_pass": 17, "observed_pass": 17, "observed_fail": 0}
  ):
    contract_reasons.append("oracle_classification")
  if reference.get("arm") != REFERENCE_ARM:
    contract_reasons.append("reference_update_arm")
  if oracle.get("arm") != ORACLE_ARM:
    contract_reasons.append("oracle_update_arm")
  if not isinstance(oracle.get("vjp_oracle"), dict):
    contract_reasons.append("oracle_report")
  elif oracle["vjp_oracle"].get("verdict") != "PASS":
    contract_reasons.append("oracle_report")

  input_reasons = []
  if len(reference_pre) != 1 or len(oracle_pre) != 1:
    input_reasons.append("pre_alignment_count")
  elif (
      reference_pre[0].get("verdict") != "PASS"
      or oracle_pre[0].get("verdict") != "PASS"
      or reference_pre[0].get("hashes") != oracle_pre[0].get("hashes")
  ):
    input_reasons.append("pre_alignment_hashes")
  if reference.get("model_before_sample") != oracle.get("model_before_sample"):
    input_reasons.append("model_before_sample")

  observer_reasons = [
      field
      for field in EXACT_UPDATE_FIELDS
      if reference.get(field) != oracle.get(field)
  ]
  for field in ("layerwise_profile", "row_cotangent_summary"):
    if _without_arm(reference.get(field)) != _without_arm(oracle.get(field)):
      observer_reasons.append(field)
  if contract_reasons:
    verdict = "FAIL_CONTRACT"
  elif input_reasons:
    verdict = "INCONCLUSIVE_INPUT_MISMATCH"
  elif observer_reasons:
    verdict = "FAIL_OBSERVER_RED"
  else:
    verdict = "PASS"
  return {
      "schema": "canon-p66-tp4-oracle-neutrality-v1",
      "verdict": verdict,
      "reference_arm": REFERENCE_ARM,
      "oracle_arm": ORACLE_ARM,
      "exact_update_fields": list(EXACT_UPDATE_FIELDS),
      "contract_reasons": contract_reasons,
      "input_reasons": input_reasons,
      "observer_reasons": observer_reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--reference-classification", type=Path, required=True)
  parser.add_argument("--reference-pre-alignment", type=Path, required=True)
  parser.add_argument("--reference-update", type=Path, required=True)
  parser.add_argument("--oracle-classification", type=Path, required=True)
  parser.add_argument("--oracle-pre-alignment", type=Path, required=True)
  parser.add_argument("--oracle-update", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  result = classify(
      reference_classification=args.reference_classification,
      reference_pre_alignment=args.reference_pre_alignment,
      reference_update=args.reference_update,
      oracle_classification=args.oracle_classification,
      oracle_pre_alignment=args.oracle_pre_alignment,
      oracle_update=args.oracle_update,
  )
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      f"P66_TP4_ORACLE_NEUTRALITY verdict={result['verdict']} "
      f"input_reasons={len(result['input_reasons'])} "
      f"observer_reasons={len(result['observer_reasons'])}"
  )
  return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
