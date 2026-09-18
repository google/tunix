#!/usr/bin/env python3
"""Classify one complete P35.3 captured-input replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PAIR_B0 = "B_serving_vs_R0_live_replay"
PAIR_01 = "R0_live_vs_R1_mapped_replay"
PAIR_12 = "R1_captured_vs_R2_adapter_direct"
PAIR_23 = "R2_adapter_direct_vs_R3_adapter_envelope"
PAIR_3C = "R3_adapter_envelope_vs_C_original"
PAIR_BC = "B_serving_vs_C_adapter"
REQUIRED_ATTESTATIONS = {
    "weights_equal",
    "captured_B_metadata_admitted",
    "selected_token_ids_equal",
    "action_masks_equal",
    "cache_fresh_B",
    "cache_fresh_replay",
    "local_m256",
    "device_order_expected",
    "repeat_exact",
}
REQUIRED_STAGES = {
    "final_hidden",
    "raw_targets",
    "processed_targets",
    "implied_log_normalizers",
    "logps",
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


def _pair_exact(pair: dict[str, Any], name: str, reasons: list[str]) -> bool:
  _require(pair.get("valid") is True, f"{name}.valid", reasons)
  differing = pair.get("differing_elements")
  total = pair.get("total_elements")
  byte_differing = pair.get("differing_bytes")
  total_bytes = pair.get("total_bytes")
  hashes_equal = pair.get("masked_hashes_equal")
  _require(isinstance(differing, int) and differing >= 0,
           f"{name}.differing_elements", reasons)
  _require(isinstance(total, int) and total > 0,
           f"{name}.total_elements", reasons)
  _require(isinstance(byte_differing, int) and byte_differing >= 0,
           f"{name}.differing_bytes", reasons)
  _require(isinstance(total_bytes, int) and total_bytes > 0,
           f"{name}.total_bytes", reasons)
  _require(isinstance(hashes_equal, bool),
           f"{name}.masked_hashes_equal", reasons)
  if reasons:
    return False
  exact = differing == 0 and byte_differing == 0
  _require(hashes_equal == exact, f"{name}.hash_count_consistency", reasons)
  return exact and hashes_equal


def _stage_group_valid(
    group: Any,
    name: str,
    reasons: list[str],
    *,
    required_stages: set[str] = REQUIRED_STAGES,
) -> bool:
  if not isinstance(group, dict) or set(group) != required_stages:
    reasons.append(f"{name}.stages")
    return False
  valid = True
  for stage, summary in group.items():
    stage_valid = bool(
        isinstance(summary, dict)
        and summary.get("valid") is True
        and isinstance(summary.get("differing_elements"), int)
        and summary["differing_elements"] >= 0
        and isinstance(summary.get("total_elements"), int)
        and summary["total_elements"] > 0
        and summary.get("exact")
        == (summary.get("differing_elements") == 0)
    )
    _require(stage_valid, f"{name}.{stage}", reasons)
    valid &= stage_valid
  return valid


def _compact_pair(pair: dict[str, Any]) -> dict[str, Any]:
  return {
      key: pair[key]
      for key in (
          "differing_elements",
          "total_elements",
          "element_fraction",
          "differing_bytes",
          "total_bytes",
          "byte_fraction",
          "masked_hashes_equal",
      )
      if key in pair
  }


def classify(report: dict[str, Any], *, report_path: Path | None = None) -> dict[str, Any]:
  """Returns one fail-closed P35.3 classification."""
  reasons: list[str] = []
  _require(report.get("schema_version") == 1, "schema_version", reasons)
  _require(report.get("measurement_rows") == 1, "measurement_rows", reasons)
  _require(
      report.get("arms") == ["B", "R0", "R1", "R2", "R3", "C"],
      "arms",
      reasons,
  )
  attestations = report.get("attestations", {})
  _require(set(attestations) == REQUIRED_ATTESTATIONS,
           "attestation_keys", reasons)
  for key in REQUIRED_ATTESTATIONS:
    _require(attestations.get(key) is True, f"attestation.{key}", reasons)
  negative = report.get("negative_control", {})
  _require(negative.get("injected") is True,
           "negative_control.injected", reasons)
  _require(
      isinstance(negative.get("differing_elements"), int)
      and negative["differing_elements"] > 0,
      "negative_control.differing_elements",
      reasons,
  )
  _require(negative.get("masked_hashes_equal") is False,
           "negative_control.masked_hashes_equal", reasons)

  repeats = report.get("repeat_comparisons", {})
  _require(set(repeats) == {
      "R0_live_repeat", "R1_mapped_repeat", "R2_adapter_direct_repeat"
  },
           "repeat_groups", reasons)
  for name in (
      "R0_live_repeat", "R1_mapped_repeat", "R2_adapter_direct_repeat"
  ):
    if name in repeats:
      expected = {"logps"} if name == "R2_adapter_direct_repeat" else REQUIRED_STAGES
      _stage_group_valid(
          repeats[name], name, reasons, required_stages=expected
      )
      for stage, summary in repeats[name].items():
        _require(summary.get("exact") is True,
                 f"{name}.{stage}.repeat_exact", reasons)
  stages = report.get("stage_comparisons", {})
  _require(set(stages) == {"R0_live_vs_R1_mapped"},
           "stage_groups", reasons)
  for name in ("R0_live_vs_R1_mapped",):
    if name not in stages:
      continue
    _stage_group_valid(
        stages[name],
        name,
        reasons,
    )

  pairs = report.get("pairs", {})
  _require(
      set(pairs) == {
          PAIR_B0, PAIR_01, PAIR_12, PAIR_23, PAIR_3C, PAIR_BC
      },
           "pair_keys", reasons)
  if reasons:
    return {"measurement_verdict": "INCONCLUSIVE",
            "classification": None, "reasons": reasons}
  pair_reasons: list[str] = []
  b0 = _pair_exact(pairs[PAIR_B0], PAIR_B0, pair_reasons)
  p01 = _pair_exact(pairs[PAIR_01], PAIR_01, pair_reasons)
  p12 = _pair_exact(pairs[PAIR_12], PAIR_12, pair_reasons)
  p23 = _pair_exact(pairs[PAIR_23], PAIR_23, pair_reasons)
  p3c = _pair_exact(pairs[PAIR_3C], PAIR_3C, pair_reasons)
  bc = _pair_exact(pairs[PAIR_BC], PAIR_BC, pair_reasons)
  if pair_reasons:
    return {"measurement_verdict": "INCONCLUSIVE",
            "classification": None, "reasons": pair_reasons}
  if bc:
    return {
        "measurement_verdict": "INCONCLUSIVE",
        "classification": None,
        "reasons": ["known_B_vs_C_red_not_reproduced"],
    }
  if not b0:
    return {
        "measurement_verdict": "INCONCLUSIVE",
        "classification": None,
        "reasons": ["serving_replay_not_anchored"],
    }
  if not p3c:
    return {
        "measurement_verdict": "INCONCLUSIVE",
        "classification": None,
        "reasons": ["adapter_repeat_not_anchored"],
    }
  carriers = []
  if not p01:
    carriers.append("weight_memory_placement")
  if not p12:
    carriers.append("metadata_cache_construction")
  if not p23:
    carriers.append("adapter_outer_program")
  if not carriers:
    return {
        "measurement_verdict": "INCONCLUSIVE",
        "classification": None,
        "reasons": ["bitwise_transitivity"],
    }
  classification = (
      f"{carriers[0]}_carrier"
      if len(carriers) == 1
      else "mixed_exact_replay_carriers"
  )
  result = {
      "measurement_verdict": "COMPLETE",
      "classification": classification,
      "reasons": [],
      "B_vs_R0_exact": b0,
      "R0_vs_R1_exact": p01,
      "R1_vs_R2_exact": p12,
      "R2_vs_R3_exact": p23,
      "R3_vs_C_exact": p3c,
      "B_vs_C_exact": bc,
      "carrier_components": carriers,
      "pair_measurements": {
          name: _compact_pair(pair) for name, pair in pairs.items()
      },
      "stage_exactness": {
          group_name: {
              stage: summary["exact"]
              for stage, summary in group.items()
          }
          for group_name, group in stages.items()
      },
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
