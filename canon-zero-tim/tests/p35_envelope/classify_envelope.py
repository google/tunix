#!/usr/bin/env python3
"""Classify one complete P35 three-arm envelope measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


_PAIR_AB = "A_native_vs_B_canonical_serving"
_PAIR_BC = "B_canonical_serving_vs_C_adapter"
_PAIR_AC = "A_native_vs_C_adapter"
_REQUIRED_ATTESTATIONS = {
    "weights_equal",
    "policy_version_equal",
    "selected_token_ids_equal",
    "action_masks_equal",
    "validity_masks_equal",
    "rank_strided_group",
    "native_A_observed",
    "grouped_B_observed",
    "mesh_shape_expected",
    "device_order_expected",
    "local_m256_B",
    "local_m256_C",
    "positions_equal",
    "block_tables_B_observed",
    "block_tables_C_canonical",
    "request_distribution_B_one_per_rank",
    "metadata_B_matches_C",
    "prefix_cache_reset_B",
    "cache_fresh_B",
    "cache_fresh_C",
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _require(condition: bool, reason: str, reasons: list[str]) -> None:
  if not condition:
    reasons.append(reason)


def _pair_is_exact(pair: dict[str, Any], name: str, reasons: list[str]) -> bool:
  _require(pair.get("valid") is True, f"{name}.valid", reasons)
  total_elements = pair.get("total_elements")
  total_bytes = pair.get("total_bytes")
  _require(
      isinstance(total_elements, int) and total_elements > 0,
      f"{name}.total_elements",
      reasons,
  )
  _require(
      isinstance(total_bytes, int) and total_bytes > 0,
      f"{name}.total_bytes",
      reasons,
  )
  differing_elements = pair.get("differing_elements")
  differing_bytes = pair.get("differing_bytes")
  _require(
      isinstance(differing_elements, int) and differing_elements >= 0,
      f"{name}.differing_elements",
      reasons,
  )
  _require(
      isinstance(differing_bytes, int) and differing_bytes >= 0,
      f"{name}.differing_bytes",
      reasons,
  )
  hashes_equal = pair.get("masked_hashes_equal")
  _require(isinstance(hashes_equal, bool), f"{name}.masked_hashes_equal", reasons)
  if reasons:
    return False
  exact_by_count = differing_elements == 0 and differing_bytes == 0
  _require(
      hashes_equal == exact_by_count,
      f"{name}.hash_count_consistency",
      reasons,
  )
  return exact_by_count and hashes_equal


def classify(report: dict[str, Any], *, report_path: Path | None = None) -> dict[str, Any]:
  """Returns a fail-closed mechanical classification for one report."""
  reasons: list[str] = []
  _require(report.get("schema_version") == 2, "schema_version", reasons)
  _require(report.get("measurement_rows") == 1, "measurement_rows", reasons)
  _require(report.get("arms") == ["A", "B", "C"], "arms", reasons)

  attestations = report.get("attestations", {})
  _require(
      set(attestations) == _REQUIRED_ATTESTATIONS,
      "attestation_keys",
      reasons,
  )
  for key in _REQUIRED_ATTESTATIONS:
    _require(attestations.get(key) is True, f"attestation.{key}", reasons)

  negative = report.get("negative_control", {})
  _require(negative.get("injected") is True, "negative_control.injected", reasons)
  _require(
      isinstance(negative.get("differing_elements"), int)
      and negative["differing_elements"] > 0,
      "negative_control.differing_elements",
      reasons,
  )
  _require(
      negative.get("masked_hashes_equal") is False,
      "negative_control.masked_hashes_equal",
      reasons,
  )

  pairs = report.get("pairs", {})
  _require(set(pairs) == {_PAIR_AB, _PAIR_BC, _PAIR_AC}, "pair_keys", reasons)
  if reasons:
    return {
        "measurement_verdict": "INCONCLUSIVE",
        "classification": None,
        "reasons": reasons,
    }

  ab_reasons: list[str] = []
  bc_reasons: list[str] = []
  ac_reasons: list[str] = []
  ab_exact = _pair_is_exact(pairs[_PAIR_AB], _PAIR_AB, ab_reasons)
  bc_exact = _pair_is_exact(pairs[_PAIR_BC], _PAIR_BC, bc_reasons)
  ac_exact = _pair_is_exact(pairs[_PAIR_AC], _PAIR_AC, ac_reasons)
  reasons.extend(ab_reasons)
  reasons.extend(bc_reasons)
  reasons.extend(ac_reasons)
  if reasons:
    return {
        "measurement_verdict": "INCONCLUSIVE",
        "classification": None,
        "reasons": reasons,
    }

  if ab_exact and bc_exact and not ac_exact:
    return {
        "measurement_verdict": "INCONCLUSIVE",
        "classification": None,
        "reasons": ["bitwise_transitivity"],
    }
  if ac_exact:
    return {
        "measurement_verdict": "INCONCLUSIVE",
        "classification": None,
        "reasons": ["known_A_vs_C_red_not_reproduced"],
    }

  if not ab_exact and bc_exact:
    classification = "packing_metadata_carrier"
  elif ab_exact and not bc_exact:
    classification = "adapter_envelope_carrier"
  elif not ab_exact and not bc_exact:
    classification = "mixed_envelope_carriers"
  else:
    raise AssertionError("unreachable exactness combination")

  result = {
      "measurement_verdict": "COMPLETE",
      "classification": classification,
      "reasons": [],
      "A_vs_B_exact": ab_exact,
      "B_vs_C_exact": bc_exact,
      "A_vs_C_exact": ac_exact,
      "input_report_sha256": None,
  }
  if report_path is not None:
    result["input_report_sha256"] = _sha256(report_path)
  return result


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--report", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  report = json.loads(args.report.read_text(encoding="utf-8"))
  result = classify(report, report_path=args.report)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, sort_keys=True))
  return 0 if result["measurement_verdict"] == "COMPLETE" else 2


if __name__ == "__main__":
  raise SystemExit(main())
