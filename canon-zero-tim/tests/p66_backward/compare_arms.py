#!/usr/bin/env python3
"""Compare P66 ordinary and segmented gradients on one signed input."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


_P61_PATH = Path(__file__).parents[1] / "p61_backward" / "compare_full_trees.py"
_SPEC = importlib.util.spec_from_file_location("p61_compare_full_trees", _P61_PATH)
if _SPEC is None or _SPEC.loader is None:
  raise ImportError(f"cannot load P61 comparator from {_P61_PATH}")
_P61 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_P61)


def _load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected a JSON object")
  return value


def compare(
    *,
    ordinary_root: Path,
    segmented_root: Path,
    ordinary_update: Path,
    segmented_update: Path,
    ordinary_classification: Path,
    segmented_classification: Path,
    tier1_baseline: Path,
) -> dict[str, Any]:
  contract_reasons = []
  zero_tim_failure = False
  classifications = {
      "ordinary": _load_json(ordinary_classification),
      "segmented": _load_json(segmented_classification),
  }
  for arm, classification in classifications.items():
    zero_tim = classification.get("zero_tim", {})
    if zero_tim.get("observed_fail", 0) != 0:
      zero_tim_failure = True
      contract_reasons.append(f"{arm}.zero_tim_fail")
    if (
        classification.get("verdict") != "PASS"
        or zero_tim.get("expected_pass") != 17
        or zero_tim.get("observed_pass") != 17
    ):
      contract_reasons.append(f"{arm}.classification")

  updates = {
      "ordinary": _load_json(ordinary_update),
      "segmented": _load_json(segmented_update),
  }
  for arm, update in updates.items():
    if (
        update.get("schema") != "canon-p66-backward-gate-v1"
        or update.get("arm") != arm
        or update.get("verdict") != "PASS"
        or update.get("commits") != 0
        or (update.get("dp_size"), update.get("tp_size")) != (4, 1)
        or update.get("global_trajectories") != 64
        or update.get("gradient_groups") != 16
    ):
      contract_reasons.append(f"{arm}.update_contract")
  ordinary_hashes = updates["ordinary"].get("alignment_hashes")
  segmented_hashes = updates["segmented"].get("alignment_hashes")
  hash_schema_ok = (
      isinstance(ordinary_hashes, list)
      and isinstance(segmented_hashes, list)
      and len(ordinary_hashes) == len(segmented_hashes) == 16
      and all(
          tuple(sorted(row)) == tuple(sorted(_P61.HASH_KEYS))
          for row in ordinary_hashes + segmented_hashes
      )
  )
  same_input = hash_schema_ok and ordinary_hashes == segmented_hashes
  if not same_input:
    contract_reasons.append("same_input_seven_hashes")

  captures = {}
  for arm, root in (
      ("ordinary", ordinary_root),
      ("segmented", segmented_root),
  ):
    captures[arm] = {
        name: _P61._load_capture(root, name)  # pylint: disable=protected-access
        for name in ("model_before", "gradient")
    }
  model_leaves = _P61._compatible(  # pylint: disable=protected-access
      (
          captures["ordinary"]["model_before"],
          captures["segmented"]["model_before"],
      ),
      label="P66 model-before",
  )
  gradient_leaves = _P61._compatible(  # pylint: disable=protected-access
      (
          captures["ordinary"]["gradient"],
          captures["segmented"]["gradient"],
      ),
      label="P66 gradient",
  )
  model_before_exact = all(
      left["data_sha256"] == right["data_sha256"]
      for left, right in zip(
          captures["ordinary"]["model_before"]["leaves"],
          captures["segmented"]["model_before"]["leaves"],
          strict=True,
      )
  )
  if not model_before_exact or not model_leaves:
    contract_reasons.append("full_model_prestate")
  gradient = _P61._tree_metrics(  # pylint: disable=protected-access
      gradient_leaves,
      lambda leaf: _P61._array(  # pylint: disable=protected-access
          ordinary_root, "gradient", leaf
      ),
      lambda leaf: _P61._array(  # pylint: disable=protected-access
          segmented_root, "gradient", leaf
      ),
  )

  baseline = _load_json(tier1_baseline)
  baseline_gradient = baseline.get("gradient")
  if (
      baseline.get("schema") != "canon-p61-tier1-baseline-v1"
      or not isinstance(baseline_gradient, dict)
  ):
    raise ValueError("invalid P61 Tier-1 gradient baseline")
  thresholds = {}
  numerical_reasons = []
  for metric_name in _P61.METRIC_KEYS:
    baseline_value = baseline_gradient.get(metric_name)
    if (
        not isinstance(baseline_value, (int, float))
        or not math.isfinite(baseline_value)
        or baseline_value < 0.0
    ):
      raise ValueError(f"invalid Tier-1 gradient metric: {metric_name}")
    threshold = min(
        2.0 * baseline_value, _P61.ABSOLUTE_CAPS[metric_name]
    )
    thresholds[metric_name] = threshold
    if (
        not math.isfinite(gradient[metric_name])
        or gradient[metric_name] > threshold
    ):
      numerical_reasons.append(
          f"gradient.{metric_name}={gradient[metric_name]:.9e}>"
          f"{threshold:.9e}"
      )
  if not gradient["finite"]:
    numerical_reasons.append("gradient.nonfinite")
  if gradient["dead_candidate_leaves"]:
    numerical_reasons.append("gradient.dead_candidate_leaves")

  if zero_tim_failure:
    verdict = "REJECT_ZERO_TIM"
  elif contract_reasons:
    verdict = "INCONCLUSIVE_CARRIER"
  elif numerical_reasons:
    verdict = "P66_GRADIENT_REJECT"
  else:
    verdict = "P66_GRADIENT_KEEP"
  return {
      "schema": "canon-p66-ordinary-segmented-gradient-ab-v1",
      "verdict": verdict,
      "scope": "Qwen3-1.7B DP4xTP1 one frozen no-commit backward",
      "model_before_array_exact": model_before_exact,
      "same_input_seven_hashes": same_input,
      "gradient": gradient,
      "thresholds": thresholds,
      "tier1_baseline_sha256": _P61._sha256(  # pylint: disable=protected-access
          tier1_baseline
      ),
      "evidence_sha256": {
          "ordinary_update": _P61._sha256(ordinary_update),  # pylint: disable=protected-access
          "segmented_update": _P61._sha256(segmented_update),  # pylint: disable=protected-access
          "ordinary_classification": _P61._sha256(  # pylint: disable=protected-access
              ordinary_classification
          ),
          "segmented_classification": _P61._sha256(  # pylint: disable=protected-access
              segmented_classification
          ),
      },
      "contract_reasons": contract_reasons,
      "numerical_reasons": numerical_reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--ordinary-root", required=True, type=Path)
  parser.add_argument("--segmented-root", required=True, type=Path)
  parser.add_argument("--ordinary-update", required=True, type=Path)
  parser.add_argument("--segmented-update", required=True, type=Path)
  parser.add_argument("--ordinary-classification", required=True, type=Path)
  parser.add_argument("--segmented-classification", required=True, type=Path)
  parser.add_argument("--tier1-baseline", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  result = compare(
      ordinary_root=args.ordinary_root,
      segmented_root=args.segmented_root,
      ordinary_update=args.ordinary_update,
      segmented_update=args.segmented_update,
      ordinary_classification=args.ordinary_classification,
      segmented_classification=args.segmented_classification,
      tier1_baseline=args.tier1_baseline,
  )
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      "P66_BACKWARD_AB "
      f"verdict={result['verdict']} "
      f"gradient_rel_l2={result['gradient']['rel_l2']:.9e}"
  )
  return 0 if result["verdict"] == "P66_GRADIENT_KEEP" else 1


if __name__ == "__main__":
  raise SystemExit(main())
