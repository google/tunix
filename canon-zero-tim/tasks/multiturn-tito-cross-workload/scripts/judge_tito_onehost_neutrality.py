#!/usr/bin/env python3
"""Judge the matched P57 exact-TiTO record-full observer pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_R7_GRADIENT_NORM_ANCHOR = (
    6.42560338973999,
    10.10729694366455,
    7.489109516143799,
)
_R7_IMPLEMENTATION_ID = (
    "tpu_inference.runner.tpu_runner.TPUModelRunner:"
    "qwen3-canonical-dp1-tp4-m256-vjp2"
)
_ALIGNMENT_HASH_KEYS = frozenset({
    "S_decode",
    "S_prefill",
    "T_current",
    "T_old",
    "action_mask",
    "policy_version",
    "tokens",
})
_INPUT_HASH_KEYS = ("tokens", "action_mask", "policy_version")


def _json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected JSON object: {path}")
  return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
  values = []
  for line in path.read_text(encoding="utf-8").splitlines():
    if line.strip():
      value = json.loads(line)
      if not isinstance(value, dict):
        raise ValueError(f"expected JSONL objects: {path}")
      values.append(value)
  return values


def _update_numerics(record: dict[str, Any]) -> dict[str, Any]:
  evidence = record.get("commit_evidence", {})
  return {
      "alignment_hashes": record.get("alignment_hashes"),
      "commit_gradient_norm": record.get("commit_gradient_norm"),
      "micro_gradient_norms": record.get("micro_gradient_norms"),
      "gradient_activity": record.get("gradient_activity"),
      "gradient_finite": record.get("gradient_finite"),
      "optimizer_transaction_valid": record.get("optimizer_transaction_valid"),
      "commit_evidence": {
          key: evidence.get(key)
          for key in (
              "gradient_finite",
              "gradient_max_abs",
              "gradient_nonzero_elements",
              "parameter_changed_elements",
              "parameter_delta_finite",
              "parameter_delta_max_abs",
              "parameter_total_elements",
          )
      },
      "state_fingerprints_before": record.get("state_fingerprints_before"),
      "state_fingerprints_after": record.get("state_fingerprints_after"),
  }


def _alignment_numerics(record: dict[str, Any]) -> dict[str, Any]:
  context = record.get("context", {})
  canonical_c = context.get("canonical_c", {})
  return {
      "N_action": record.get("N_action"),
      "boundaries": record.get("boundaries"),
      "exact": record.get("exact"),
      "gradient": record.get("gradient"),
      "hashes": record.get("hashes"),
      "masked_hashes": record.get("masked_hashes"),
      "ratio_stats": record.get("ratio_stats"),
      "clip_hits": record.get("clip_hits"),
      "tis_hits": record.get("tis_hits"),
      "implementation_id": canonical_c.get("implementation_id"),
  }


def _input_hashes(record: dict[str, Any]) -> tuple[dict[str, Any], ...] | None:
  bundles = record.get("alignment_hashes")
  if not isinstance(bundles, list) or not bundles:
    return None
  if any(
      not isinstance(bundle, dict)
      or frozenset(bundle) != _ALIGNMENT_HASH_KEYS
      for bundle in bundles
  ):
    return None
  return tuple(
      {key: bundle[key] for key in _INPUT_HASH_KEYS} for bundle in bundles
  )


def judge(*, off_root: Path, on_root: Path) -> dict[str, Any]:
  reasons: list[str] = []
  input_mismatch_reasons: list[str] = []

  def require(condition: bool, reason: str) -> None:
    if not condition:
      reasons.append(reason)

  off_classification = _json(off_root / "classification.json")
  on_classification = _json(on_root / "classification.json")
  require(
      off_classification.get("verdict") == "PASS"
      and off_classification.get("neutrality_arm") == "tito-off",
      "off_classification",
  )
  require(
      on_classification.get("verdict") == "PASS"
      and on_classification.get("neutrality_arm") == "tito-on",
      "on_classification",
  )
  require(
      off_classification.get("semantic_event_counts")
      == on_classification.get("semantic_event_counts"),
      "semantic_event_census",
  )

  off_updates = _jsonl(off_root / "updates.jsonl")
  on_updates = _jsonl(on_root / "updates.jsonl")
  require(len(off_updates) == len(on_updates) == 3, "update_count")
  off_norms = tuple(row.get("commit_gradient_norm") for row in off_updates)
  on_norms = tuple(row.get("commit_gradient_norm") for row in on_updates)
  require(off_norms == _R7_GRADIENT_NORM_ANCHOR, "off_r7_gradient_anchor")
  require(on_norms == _R7_GRADIENT_NORM_ANCHOR, "on_r7_gradient_anchor")
  for step, (off, on) in enumerate(zip(off_updates, on_updates, strict=True)):
    off_inputs = _input_hashes(off)
    on_inputs = _input_hashes(on)
    require(off_inputs is not None, f"off_seven_hash_contract:{step}")
    require(on_inputs is not None, f"on_seven_hash_contract:{step}")
    if off_inputs is not None and on_inputs is not None and off_inputs != on_inputs:
      input_mismatch_reasons.append(f"input_hashes:{step}")
    if step == 0 and (
        off.get("state_fingerprints_before")
        != on.get("state_fingerprints_before")
    ):
      input_mismatch_reasons.append("initial_state_fingerprints")
    require(
        _update_numerics(off) == _update_numerics(on),
        f"update_numerics:{step}",
    )

  off_alignment = _jsonl(off_root / "alignment.jsonl")
  on_alignment = _jsonl(on_root / "alignment.jsonl")
  require(len(off_alignment) == len(on_alignment) == 12, "alignment_count")
  for row, (off, on) in enumerate(
      zip(off_alignment, on_alignment, strict=True)
  ):
    off_numerics = _alignment_numerics(off)
    on_numerics = _alignment_numerics(on)
    require(off_numerics == on_numerics, f"alignment_numerics:{row}")
    require(
        off_numerics["implementation_id"] == _R7_IMPLEMENTATION_ID,
        f"forward_implementation_anchor:{row}",
    )

  verdict = (
      "INCONCLUSIVE_INPUT_MISMATCH"
      if input_mismatch_reasons
      else "PASS"
      if not reasons
      else "FAIL"
  )
  return {
      "schema": "canon.p57-tito-onehost-neutrality.v1",
      "verdict": verdict,
      "input_verdict": (
          "MATCH" if not input_mismatch_reasons else "MISMATCH"
      ),
      "geometry": "DP1xTP4",
      "updates": len(off_updates),
      "strict_alignment_rows": len(off_alignment),
      "gradient_norm_anchor": list(_R7_GRADIENT_NORM_ANCHOR),
      "forward_implementation_anchor": _R7_IMPLEMENTATION_ID,
      "claims": {
          "record_full_observer_neutral": verdict == "PASS",
          "target_dp8_tp8_certified": False,
          "performance_evidence": False,
      },
      "reasons": reasons,
      "input_mismatch_reasons": input_mismatch_reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--off-root", required=True, type=Path)
  parser.add_argument("--on-root", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  result = judge(off_root=args.off_root, on_root=args.on_root)
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print("P57_TITO_ONEHOST_NEUTRALITY " + json.dumps(result, sort_keys=True))
  return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
