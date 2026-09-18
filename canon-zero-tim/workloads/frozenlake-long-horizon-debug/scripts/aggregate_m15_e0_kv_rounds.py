#!/usr/bin/env python3
"""Aggregate three independently sealed E0 KV rounds for one APC arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class E0AggregateError(RuntimeError):
  """Raised when the three-round evidence is incomplete or unstable."""


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise E0AggregateError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(
    root: Path, arm: str, rounds: int, expected_source: str
) -> dict[str, Any]:
  _require(arm in ("off", "on"), "E0 arm must be off or on")
  _require(rounds == 3, "E0 stability gate requires exactly three rounds")
  rows = []
  for round_index in range(rounds):
    directory = root / f"{round_index:06d}"
    input_path = directory / "ROUND_INPUT.json"
    classifier_path = directory / "kv-observer-classification.json"
    checkpoint_path = directory / "CLASSIFIER_INPUT_RECEIPT.json"
    completion_path = directory / "ROUND_COMPLETE.json"
    _require(directory.is_dir(), f"round {round_index} directory is absent")
    for path in (input_path, classifier_path, checkpoint_path, completion_path):
      _require(path.is_file() and path.stat().st_size > 0,
               f"round {round_index} artifact is absent: {path.name}")
    round_input = json.loads(input_path.read_text(encoding="utf-8"))
    classification = json.loads(classifier_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    _require(
        round_input.get("schema") == "m15-e0-kv-round-input-v1"
        and round_input.get("arm") == arm
        and round_input.get("diagnostic_round") == round_index
        and round_input.get("expected_source_commit") == expected_source
        and round_input.get("runtime_source_commit") == expected_source
        and round_input.get("kv_records") == 16
        and round_input.get("kv_pairs") == 8,
        f"round {round_index} input receipt drifted",
    )
    _require(
        round_input.get("b_c_differing_bytes") == 0
        and round_input.get("b_c_differing_elements") == 0,
        f"round {round_index} B-C is red",
    )
    _require(
        classification.get("schema") == "p38-live-kv-classification-v2"
        and classification.get("status") == "PASS"
        and classification.get("records") == 16
        and classification.get("pairs") == 8
        and {item.get("diagnostic_round")
             for item in classification.get("comparisons", ())}
            == {round_index},
        f"round {round_index} KV classification drifted",
    )
    _require(
        checkpoint.get("schema")
            == "m15-e0-kv-classifier-input-receipt-v1"
        and checkpoint.get("status")
            == "uploaded-readback-verified-before-classification"
        and checkpoint.get("arm") == arm
        and checkpoint.get("diagnostic_round") == round_index
        and checkpoint.get("source_commit") == expected_source
        and checkpoint.get("runtime_source_commit") == expected_source
        and checkpoint.get("kv_records") == 16
        and checkpoint.get("kv_pairs") == 8
        and checkpoint.get("a_b_differing_bytes")
            == round_input.get("a_b_differing_bytes"),
        f"round {round_index} classifier-input checkpoint drifted",
    )
    _require(
        completion.get("schema") == "m15-e0-kv-round-completion-v1"
        and completion.get("status") == "sealed-uploaded-readback-verified"
        and completion.get("diagnostic_round") == round_index
        and completion.get("arm") == arm
        and completion.get("source_commit") == expected_source
        and completion.get("runtime_source_commit") == expected_source
        and completion.get("classification_sha256") == _sha256(classifier_path)
        and completion.get("classifier_input_receipt_sha256")
            == _sha256(checkpoint_path)
        and completion.get("round_input_sha256") == _sha256(input_path),
        f"round {round_index} completion receipt drifted",
    )
    a_b_bytes = int(round_input["a_b_differing_bytes"])
    a_b_elements = int(round_input["a_b_differing_elements"])
    outcome = classification.get("classification")
    if arm == "off":
      _require(
          a_b_bytes == 0
          and a_b_elements == 0
          and outcome == "observer_pairs_valid_red_join_pending",
          f"round {round_index} APC-off control is not exact",
      )
    elif a_b_bytes == 0:
      _require(
          a_b_elements == 0
          and outcome == "observer_pairs_valid_red_join_pending",
          f"round {round_index} exact treatment classification drifted",
      )
    else:
      _require(
          a_b_elements > 0
          and outcome in (
              "live_kv_fingerprint_equal_on_red_row",
              "live_kv_fingerprint_differs_on_red_row",
          )
          and classification.get("source_request_binding", {}).get("status")
              == "UNIQUE_FUTURE_PREFIX_BINDING",
          f"round {round_index} red treatment lacks a bound mechanism verdict",
      )
    rows.append({
        "a_b_differing_bytes": a_b_bytes,
        "a_b_differing_elements": a_b_elements,
        "classification": outcome,
        "classification_sha256": _sha256(classifier_path),
        "completion_sha256": _sha256(completion_path),
        "diagnostic_round": round_index,
        "round_input_sha256": _sha256(input_path),
    })

  outcomes = {row["classification"] for row in rows}
  if arm == "off":
    status = "CONTROL_EXACT_3_OF_3"
  elif outcomes == {"observer_pairs_valid_red_join_pending"}:
    status = "TARGET_NON_REPRODUCTION_3_OF_3"
  elif outcomes == {"live_kv_fingerprint_equal_on_red_row"}:
    status = "LIVE_KV_FINGERPRINT_EQUAL_3_OF_3"
  elif outcomes == {"live_kv_fingerprint_differs_on_red_row"}:
    status = "LIVE_KV_FINGERPRINT_DIFFERS_3_OF_3"
  else:
    raise E0AggregateError(
        f"three-round E0 treatment is unstable: {sorted(outcomes)}"
    )
  return {
      "arm": arm,
      "claim_level": "bit-level-diagnostic-fingerprint-not-full-kv-bytes",
      "diagnostic_rounds": rounds,
      "numerical_repair_authorized": False,
      "rounds": rows,
      "runtime_source_commit": expected_source,
      "schema": "m15-e0-kv-three-round-arm-v1",
      "status": status,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", required=True, type=Path)
  parser.add_argument("--arm", required=True, choices=("off", "on"))
  parser.add_argument("--rounds", required=True, type=int)
  parser.add_argument("--expected-source", required=True)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  result = aggregate(args.root, args.arm, args.rounds, args.expected_source)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  print(
      "[M15.E0.KV] THREE_ROUND_CLASSIFICATION_PASS "
      f"arm={args.arm} status={result['status']} rounds={args.rounds}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
