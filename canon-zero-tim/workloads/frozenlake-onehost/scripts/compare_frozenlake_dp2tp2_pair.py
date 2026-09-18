#!/usr/bin/env python3
"""Fail-closed causal comparator for adjacent FrozenLake optimization arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


_ARM_BASE = {
    "report_adjoint_buckets": "0",
    "chunk_dependency_ticket": "0",
    "chunk_backpressure": "0",
}
_ARMS = {
    "r0": {
        "keep_tape": "0",
        "reduce_once": "0",
        "length_sort": "0",
        **_ARM_BASE,
    },
    "r1": {
        "keep_tape": "stream",
        "reduce_once": "0",
        "length_sort": "0",
        **_ARM_BASE,
    },
    "r2": {
        "keep_tape": "stream",
        "reduce_once": "1",
        "length_sort": "0",
        **_ARM_BASE,
    },
    "r3": {
        "keep_tape": "stream",
        "reduce_once": "1",
        "length_sort": "1",
        **_ARM_BASE,
    },
}
_ADJACENT = {("r0", "r1"), ("r1", "r2"), ("r2", "r3")}
_MANIFEST_RUN_FIELDS = {"label", "arm", "selectors"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected JSON object: {path}")
  return value


def _common_manifest_mismatches(
    left: dict[str, Any], right: dict[str, Any]
) -> list[str]:
  fields = (set(left) | set(right)) - _MANIFEST_RUN_FIELDS
  return sorted(field for field in fields if left.get(field) != right.get(field))


def _valid_input_hashes(value: Any) -> bool:
  return (
      isinstance(value, dict)
      and all(
          _SHA256_RE.fullmatch(str(value.get(key))) is not None
          for key in (
              "tokens",
              "action_mask",
              "S_decode",
              "S_prefill",
              "T_old",
          )
      )
  )


def _valid_n_real(value: Any) -> bool:
  return (
      isinstance(value, list)
      and len(value) == 16
      and all(isinstance(item, int) and item > 0 for item in value)
  )


def _valid_group_chunks(value: Any) -> bool:
  return (
      isinstance(value, list)
      and len(value) == 8
      and all(isinstance(item, int) and item > 0 for item in value)
  )


def _group_chunks(n_real: list[int], local_m: int) -> list[int]:
  return [
      (max(n_real[index:index + 2]) + local_m - 1) // local_m
      for index in range(0, len(n_real), 2)
  ]


def compare(left_root: Path, right_root: Path) -> dict[str, Any]:
  left_class = _json(left_root / "classification.json")
  right_class = _json(right_root / "classification.json")
  left_manifest = _json(left_root / "run_manifest.json")
  right_manifest = _json(right_root / "run_manifest.json")

  left_arm = left_manifest.get("arm")
  right_arm = right_manifest.get("arm")
  classification_reasons = []
  configuration_reasons = []
  input_reasons = []

  if left_class.get("verdict") != "PASS":
    classification_reasons.append(
        f"left_verdict={left_class.get('verdict')}"
    )
  if right_class.get("verdict") != "PASS":
    classification_reasons.append(
        f"right_verdict={right_class.get('verdict')}"
    )
  if (left_arm, right_arm) not in _ADJACENT:
    configuration_reasons.append(f"non_adjacent_arms={left_arm}->{right_arm}")
  for side, manifest, classification, arm in (
      ("left", left_manifest, left_class, left_arm),
      ("right", right_manifest, right_class, right_arm),
  ):
    if arm not in _ARMS or manifest.get("selectors") != _ARMS.get(arm):
      configuration_reasons.append(f"{side}_selectors")
    if classification.get("arm") != arm:
      configuration_reasons.append(f"{side}_classification_arm")
    if classification.get("workload") != manifest.get("workload"):
      configuration_reasons.append(f"{side}_classification_workload")
    if manifest.get("classification_mode") != "certify":
      configuration_reasons.append(f"{side}_manifest_not_certify")
    if classification.get("classification_mode") != "certify":
      configuration_reasons.append(f"{side}_classification_not_certify")

  manifest_mismatches = _common_manifest_mismatches(
      left_manifest, right_manifest
  )
  if manifest_mismatches:
    configuration_reasons.append("manifest_identity")

  left_zero = left_class.get("zero_tim") or {}
  right_zero = right_class.get("zero_tim") or {}
  left_shape = left_class.get("landed_shape") or {}
  right_shape = right_class.get("landed_shape") or {}
  if not _valid_input_hashes(left_zero.get("input_hashes")):
    input_reasons.append("left_input_hash_inventory")
  if not _valid_input_hashes(right_zero.get("input_hashes")):
    input_reasons.append("right_input_hash_inventory")
  if left_zero.get("input_hashes") != right_zero.get("input_hashes"):
    input_reasons.append("input_hashes")
  if left_zero.get("action_tokens") != right_zero.get("action_tokens"):
    input_reasons.append("action_tokens")
  left_n_real = left_shape.get("n_real")
  right_n_real = right_shape.get("n_real")
  left_group_chunks = left_shape.get("group_chunks")
  right_group_chunks = right_shape.get("group_chunks")
  for side, shape in (("left", left_shape), ("right", right_shape)):
    n_real = shape.get("n_real")
    group_chunks = shape.get("group_chunks")
    if not _valid_n_real(n_real):
      input_reasons.append(f"{side}_landed_n_real_inventory")
    if not _valid_group_chunks(group_chunks):
      input_reasons.append(f"{side}_group_chunk_inventory")

  local_m = left_manifest.get("local_m")
  if not isinstance(local_m, int) or local_m <= 0:
    input_reasons.append("local_m_inventory")
  elif _valid_n_real(left_n_real) and _valid_group_chunks(left_group_chunks):
    if left_group_chunks != _group_chunks(left_n_real, local_m):
      input_reasons.append("left_group_chunks_from_n_real")
  if (
      isinstance(local_m, int)
      and local_m > 0
      and _valid_n_real(right_n_real)
      and _valid_group_chunks(right_group_chunks)
      and right_group_chunks != _group_chunks(right_n_real, local_m)
  ):
    input_reasons.append("right_group_chunks_from_n_real")

  input_relation = "exact-order"
  if (left_arm, right_arm) == ("r2", "r3"):
    input_relation = "stable-length-sort"
    if _valid_n_real(left_n_real) and _valid_n_real(right_n_real):
      if sorted(left_n_real) != sorted(right_n_real):
        input_reasons.append("landed_n_real_multiset")
      elif right_n_real != sorted(left_n_real, reverse=True):
        input_reasons.append("length_sort_n_real_order")
  else:
    if left_n_real != right_n_real:
      input_reasons.append("landed_n_real")
    if left_group_chunks != right_group_chunks:
      input_reasons.append("group_chunks")

  if classification_reasons:
    verdict = "FAIL"
  elif configuration_reasons:
    verdict = "INCOMPARABLE_CONFIGURATION"
  elif input_reasons:
    verdict = "INCOMPARABLE_INPUT"
  else:
    verdict = "PASS"
  return {
      "schema": "canon.v2-frozenlake-onehost.pair.v1",
      "verdict": verdict,
      "performance_eligible": verdict == "PASS",
      "workload": left_manifest.get("workload"),
      "transition": f"{left_arm}->{right_arm}",
      "classification_reasons": classification_reasons,
      "configuration_reasons": configuration_reasons,
      "input_reasons": input_reasons,
      "manifest_mismatches": manifest_mismatches,
      "matched_input": not input_reasons,
      "input_relation": input_relation,
      "claim": (
          "causal adjacent-arm one-host input pair; timing may be reported"
          if verdict == "PASS"
          else "no causal performance comparison"
      ),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--left-root", required=True, type=Path)
  parser.add_argument("--right-root", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  result = compare(args.left_root, args.right_root)
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(result, sort_keys=True, separators=(",", ":")))
  if result["verdict"] == "PASS":
    return 0
  return 3 if result["verdict"].startswith("INCOMPARABLE") else 1


if __name__ == "__main__":
  raise SystemExit(main())
