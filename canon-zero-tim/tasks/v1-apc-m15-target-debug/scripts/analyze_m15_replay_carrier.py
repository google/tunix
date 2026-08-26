#!/usr/bin/env python3
"""Verify a frozen M15 APC carrier and emit a replay-prefix input plan.

This tool performs host-only evidence analysis.  It never initializes a model,
executes serving, or claims that the target mismatch has been reproduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import package_full_replay_carrier as carrier  # pylint: disable=wrong-import-position


class AnalysisError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise AnalysisError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
  _require(path.is_file() and path.stat().st_size > 0, f"missing {label}: {path}")
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as exc:
    raise AnalysisError(f"invalid {label}: {path}: {exc}") from exc
  _require(isinstance(value, dict), f"{label} is not a JSON object")
  return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
  with path.open("x", encoding="utf-8") as target:
    target.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    target.flush()
    os.fsync(target.fileno())


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
  with path.open("x", encoding="utf-8") as target:
    for value in values:
      target.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    target.flush()
    os.fsync(target.fileno())


def _byte_difference(
    left: np.ndarray, right: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, int, float]:
  left = np.ascontiguousarray(left)
  right = np.ascontiguousarray(right)
  valid = np.asarray(valid, dtype=np.bool_)
  _require(left.shape == right.shape == valid.shape, "comparison shapes drifted")
  _require(left.dtype == right.dtype, "comparison dtypes drifted")
  byte_left = left.view(np.uint8).reshape(left.shape + (left.dtype.itemsize,))
  byte_right = right.view(np.uint8).reshape(right.shape + (right.dtype.itemsize,))
  byte_diff = byte_left != byte_right
  element_diff = np.any(byte_diff, axis=-1) & valid
  differing_bytes = int(np.count_nonzero(byte_diff & valid[..., None]))
  max_abs = (
      float(np.max(np.abs(left[element_diff].astype(np.float64) - right[element_diff].astype(np.float64))))
      if np.any(element_diff)
      else 0.0
  )
  return element_diff, differing_bytes, max_abs


def _turn_indices(action_mask: np.ndarray) -> np.ndarray:
  action = np.asarray(action_mask, dtype=np.bool_)
  previous = np.concatenate((np.array([False]), action[:-1]))
  starts = action & ~previous
  return np.cumsum(starts, dtype=np.int64) - 1


def _request_summary(item: dict[str, Any]) -> dict[str, Any]:
  return {
      "request_id": item["request_id"],
      "first_call": int(item["first_call"]),
      "last_call": int(item["last_call"]),
      "observations": int(item["observations"]),
      "max_num_tokens": int(item["max_num_tokens"]),
      "program_paths": list(item["program_paths"]),
      "candidate_source_rows": list(item["candidate_source_rows"]),
  }


def _compact_call(
    record: dict[str, Any],
    lookup: dict[tuple[int, str], list[int]],
) -> dict[str, Any]:
  requests = []
  for request in record["requests"]:
    key = (int(request["num_tokens"]), str(request["token_history_sha256"]))
    candidates = lookup[key]
    requests.append({
        name: request[name]
        for name in (
            "request_id",
            "request_index",
            "dp_rank",
            "local_scheduler_slot",
            "scheduled_tokens",
            "num_computed_tokens",
            "num_prompt_tokens",
            "num_tokens",
            "token_history_sha256",
            "request_kind",
            "block_size",
            "logical_blocks_before",
            "logical_blocks_after",
            "physical_pages",
        )
        if name in request
    } | {
        "payload_source_row": int(candidates[0]),
        "payload_candidate_rows": [int(value) for value in candidates],
    })
  return {
      "schema": "m15-apc-replay-prefix-call-v1",
      "call_index": int(record["call_index"]),
      "arm": record["arm"],
      "serving_arm": record["serving_arm"],
      "program_path": record["program_path"],
      "request_order": list(record["request_order"]),
      "requests": requests,
  }


def analyze(
    *,
    producer_path: Path,
    envelope_path: Path,
    first_red_contract_path: Path,
    replay_contract_path: Path,
    m15_classification_path: Path,
    upstream_audit_receipt_path: Path,
    source_gcs_uri: str,
    output_dir: Path,
) -> dict[str, Any]:
  _require(not output_dir.exists(), f"refusing to overwrite output: {output_dir}")
  target = _load_json(m15_classification_path, "M15 classification")
  first_contract = _load_json(first_red_contract_path, "first-red contract")
  replay_contract = _load_json(replay_contract_path, "replay contract")
  audit_receipt = _load_json(upstream_audit_receipt_path, "upstream GCS audit receipt")
  registered_prefix = "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
  _require(
      source_gcs_uri.startswith(registered_prefix) and source_gcs_uri.endswith("/attempt-0"),
      "source GCS URI is outside the registered immutable roots",
  )
  _require(target.get("status") == "FRESH_TARGET_RED_FROZEN", "target is not a frozen red")
  _require(target.get("arm") == "on", "target is not the APC-on arm")
  _require(first_contract.get("status") == "FIRST_RED_ROW_FROZEN", "captured incident is not frozen")
  _require(replay_contract.get("status") == "FULL_REPLAY_CARRIER_FROZEN", "full replay carrier is not frozen")
  source_commit = target.get("source_commit")
  _require(
      isinstance(source_commit, str)
      and len(source_commit) == 40
      and all(char in "0123456789abcdef" for char in source_commit),
      "source commit is invalid",
  )
  _require(first_contract.get("source_commit") == source_commit, "first incident source drifted")
  _require(replay_contract.get("source_commit") == source_commit, "replay source drifted")
  _require(audit_receipt.get("status") == "FRESH_TARGET_RED_FROZEN", "upstream audit is not a frozen red")
  _require(audit_receipt.get("source_commit") == source_commit, "upstream audit source drifted")
  _require(audit_receipt.get("source_gcs_uri") == source_gcs_uri, "upstream audit URI drifted")

  producer_meta, arrays = carrier._load_producer(producer_path)  # pylint: disable=protected-access
  records = carrier._load_ledger(envelope_path, "on")  # pylint: disable=protected-access
  histories = carrier._histories(arrays)  # pylint: disable=protected-access
  request_joins, lookup = carrier._request_joins(records, histories)  # pylint: disable=protected-access
  _require(producer_meta.get("source_commit") == source_commit, "producer source drifted")
  _require(int(replay_contract.get("producer_rows", -1)) == 256, "replay producer-row count drifted")
  _require(int(replay_contract.get("serving_call_count", -1)) == len(records), "replay call count drifted")
  _require(int(replay_contract.get("request_count", -1)) == len(request_joins), "replay request count drifted")

  action = np.asarray(arrays["action_mask"], dtype=np.bool_)
  valid = action & np.asarray(arrays["completion_valid_mask"], dtype=np.bool_)
  ab_elements, ab_bytes, ab_max_abs = _byte_difference(
      arrays["s_decode"], arrays["s_prefill"], valid
  )
  bc_elements, bc_bytes, bc_max_abs = _byte_difference(
      arrays["s_prefill"], arrays["t_old"], valid
  )
  expected_ab_bytes = target.get("a_b_differing_bytes")
  expected_ab_elements = target.get("a_b_differing_elements")
  expected_bc_bytes = target.get("b_c_differing_bytes")
  _require(expected_ab_bytes == [ab_bytes], "A-B byte count disagrees with classification")
  _require(expected_ab_elements == [int(np.count_nonzero(ab_elements))], "A-B element count disagrees with classification")
  _require(expected_bc_bytes == [bc_bytes], "B-C byte count disagrees with classification")
  _require(ab_bytes > 0 and np.any(ab_elements), "APC-on carrier contains no A-B red")
  _require(bc_bytes == 0 and not np.any(bc_elements), "B-C is red; carrier is not APC-only")

  joins_by_row: dict[int, list[dict[str, Any]]] = {row: [] for row in range(256)}
  for item in request_joins:
    if item["serving_arms"] != ["A"]:
      continue
    for row in item["candidate_source_rows"]:
      joins_by_row[int(row)].append(item)
  for values in joins_by_row.values():
    values.sort(key=lambda item: (int(item["first_call"]), item["request_id"]))

  row_summaries = []
  for row in np.flatnonzero(np.any(ab_elements, axis=1)).tolist():
    positions = np.flatnonzero(ab_elements[row])
    prompt_length = int(np.count_nonzero(np.asarray(arrays["prompt_mask"])[row]))
    turns = _turn_indices(action[row])
    first_position = int(positions[0])
    first_turn = int(turns[first_position])
    requests = joins_by_row[row]
    _require(first_turn >= 0 and first_turn < len(requests), f"red row {row} has no request for turn {first_turn}")
    first_request = requests[first_turn]
    row_summaries.append({
        "source_row": int(row),
        "differing_elements": int(positions.size),
        "first_completion_position": first_position,
        "last_completion_position": int(positions[-1]),
        "first_logical_kv_prefix_length": prompt_length + first_position,
        "last_logical_kv_prefix_length": prompt_length + int(positions[-1]),
        "first_turn_index": first_turn,
        "turns_with_red": sorted({int(turns[position]) for position in positions}),
        "first_request": _request_summary(first_request),
        "request_count": len(requests),
    })
  _require(row_summaries, "no red rows were derived")
  row_summaries.sort(key=lambda item: item["source_row"])
  canonical_first = min(
      row_summaries,
      key=lambda item: (item["source_row"], item["first_completion_position"]),
  )
  earliest_request = min(
      row_summaries,
      key=lambda item: (item["first_request"]["first_call"], item["source_row"]),
  )
  for item in row_summaries:
    first_request = item["first_request"]
    _require(
        int(first_request["last_call"]) >= int(first_request["first_call"]) + 1,
        f"red row {item['source_row']} lacks the post-dispatch onset interval",
    )

  captured = first_contract.get("first_incident", {})
  captured_call = int(captured.get("call_index", -1))
  captured_row = int(captured.get("source_row", -1))
  _require(captured_call > 0 and 0 <= captured_row < 256, "captured incident coordinate is invalid")
  _require(captured_row in {item["source_row"] for item in row_summaries}, "captured incident row is not red")
  _require(any(int(record["call_index"]) == captured_call for record in records), "captured incident call is absent")

  replay_prefix_end_call = max(
      int(item["first_request"]["first_call"]) + 1 for item in row_summaries
  )
  _require(replay_prefix_end_call < captured_call, "captured incident is not downstream of the onset prefix")
  replay_prefix_records = [
      _compact_call(record, lookup)
      for record in records
      if int(record["call_index"]) <= replay_prefix_end_call
  ]
  _require(
      [record["call_index"] for record in replay_prefix_records]
      == list(range(1, replay_prefix_end_call + 1)),
      "replay prefix chronology is incomplete",
  )

  output_dir.mkdir(parents=True, exist_ok=False)
  prefix_path = output_dir / "replay-prefix-plan.jsonl"
  _write_jsonl(prefix_path, replay_prefix_records)
  audit_copy_path = output_dir / "UPSTREAM_AUDIT_RECEIPT.json"
  shutil.copyfile(upstream_audit_receipt_path, audit_copy_path)
  result = {
      "schema": "m15-apc-replay-input-analysis-v1",
      "status": "M15_REPLAY_INPUT_PLAN_READY_NOT_EXECUTED",
      "source_commit": source_commit,
      "source_gcs_uri": source_gcs_uri,
      "numerical": {
          "a_b_differing_bytes": ab_bytes,
          "a_b_differing_elements": int(np.count_nonzero(ab_elements)),
          "a_b_max_abs": ab_max_abs,
          "b_c_differing_bytes": bc_bytes,
          "b_c_differing_elements": int(np.count_nonzero(bc_elements)),
          "b_c_max_abs": bc_max_abs,
          "red_rows": row_summaries,
      },
      "coordinates": {
          "canonical_first_mismatch": canonical_first,
          "canonical_first_mismatch_request": canonical_first["first_request"],
          "earliest_red_request": earliest_request,
          "first_fully_captured_incident": {
              "source_row": captured_row,
              "completion_position": int(captured.get("completion_position", -1)),
              "call_index": captured_call,
              "request_id": captured.get("request_id"),
              "num_computed_tokens": int(captured.get("num_computed_tokens", -1)),
              "dp_rank": int(captured.get("dp_rank", -1)),
              "local_scheduler_slot": int(captured.get("local_scheduler_slot", -1)),
          },
      },
      "carrier": {
          "producer_rows": 256,
          "serving_calls": len(records),
          "requests": len(request_joins),
          "program_paths_by_arm": replay_contract.get("program_paths_by_arm"),
          "replay_prefix_start_call": 1,
          "replay_prefix_end_call": replay_prefix_end_call,
          "replay_prefix_calls": len(replay_prefix_records),
          "replay_prefix_reason": "cover every red row through first_request.first_call+1",
          "payload_resolution": "slice producer prompt+completion history by num_tokens and verify token_history_sha256",
      },
      "input_sha256": {
          "producer_unit": _sha256(producer_path),
          "serving_envelope": _sha256(envelope_path),
          "first_red_contract": _sha256(first_red_contract_path),
          "replay_contract": _sha256(replay_contract_path),
          "m15_classification": _sha256(m15_classification_path),
          "upstream_audit_receipt": _sha256(upstream_audit_receipt_path),
      },
      "derived_sha256": {"replay_prefix_plan": _sha256(prefix_path)},
      "derived_bytes": {"replay_prefix_plan": prefix_path.stat().st_size},
      "claim_ceiling": "INPUT_PLAN_ONLY_MODEL_REPLAY_NOT_RUN",
      "limitations": [
          "the_saved_scheduler_chronology_has_not_been_forced_through_vllm",
          "first_call_plus_one_is_a_replay_interval_not_an_attested_output_call",
          "physical_page_ids_may_be_replaced_by_equivalent_fresh_geometry",
          "no_tensor_boundary_or_root_cause_is_localized",
      ],
  }
  result_path = output_dir / "REPLAY_ANALYSIS.json"
  _write_json(result_path, result)
  with (output_dir / "SHA256SUMS").open("x", encoding="utf-8") as manifest:
    for path in (result_path, prefix_path, audit_copy_path):
      manifest.write(f"{_sha256(path)}  {path.name}\n")
    manifest.flush()
    os.fsync(manifest.fileno())
  return result


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--producer-unit", required=True, type=Path)
  parser.add_argument("--serving-envelope", required=True, type=Path)
  parser.add_argument("--first-red-contract", required=True, type=Path)
  parser.add_argument("--replay-contract", required=True, type=Path)
  parser.add_argument("--m15-classification", required=True, type=Path)
  parser.add_argument("--upstream-audit-receipt", required=True, type=Path)
  parser.add_argument("--source-gcs-uri", required=True)
  parser.add_argument("--output-dir", required=True, type=Path)
  args = parser.parse_args()
  try:
    result = analyze(
        producer_path=args.producer_unit,
        envelope_path=args.serving_envelope,
        first_red_contract_path=args.first_red_contract,
        replay_contract_path=args.replay_contract,
      m15_classification_path=args.m15_classification,
      upstream_audit_receipt_path=args.upstream_audit_receipt,
      source_gcs_uri=args.source_gcs_uri,
        output_dir=args.output_dir,
    )
  except (OSError, KeyError, TypeError, ValueError, AnalysisError, carrier.CarrierError) as exc:
    print(json.dumps({"status": "INCONCLUSIVE", "error": str(exc)}, sort_keys=True))
    raise SystemExit(2) from exc
  print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
  main()
