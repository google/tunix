#!/usr/bin/env python3
"""Freeze a complete M15 producer/envelope carrier for serving replay.

The large immutable inputs remain beside this derived directory in the P38
capture root.  This script validates and joins them, then emits a compact
request map, a replay contract, and a manifest whose paths are relative to the
derived directory.  It does not execute a replay or make a mechanism claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class CarrierError(RuntimeError):
  pass


REQUIRED_ARRAYS = {
    "source_rows",
    "prompt_ids",
    "prompt_mask",
    "completion_ids",
    "completion_valid_mask",
    "action_mask",
    "s_decode",
    "s_prefill",
    "t_old",
    "policy_version",
    "sampling_values",
    "metadata_json",
}
PROGRAM_PATHS = {"standard", "continue_decode"}


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise CarrierError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
  _require(path.is_file() and path.stat().st_size > 0, f"missing {label}: {path}")
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as exc:
    raise CarrierError(f"invalid {label}: {path}: {exc}") from exc
  _require(isinstance(value, dict), f"{label} is not a JSON object")
  return value


def _load_producer(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
  _require(path.is_file() and path.stat().st_size > 0, f"missing producer unit: {path}")
  try:
    with np.load(path, allow_pickle=False) as archive:
      _require(
          REQUIRED_ARRAYS.issubset(archive.files),
          f"producer unit array inventory drifted: {sorted(archive.files)}",
      )
      arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
  except (OSError, ValueError) as exc:
    raise CarrierError(f"invalid producer unit: {path}: {exc}") from exc
  try:
    metadata = json.loads(arrays["metadata_json"].tobytes().decode("utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise CarrierError("producer metadata is invalid") from exc
  _require(metadata.get("schema") == "m15-apc-producer-unit-v1", "producer schema drifted")
  _require(metadata.get("rows") == 256, "producer must contain exactly 256 rows")
  _require(metadata.get("num_generations") == 8, "producer must use 8 generations")
  _require(metadata.get("prompt_groups") == 32, "producer must contain 32 prompt groups")
  source_rows = np.asarray(arrays["source_rows"]).reshape(-1)
  _require(source_rows.tolist() == list(range(256)), "producer source rows drifted")
  for name in REQUIRED_ARRAYS - {"metadata_json", "source_rows"}:
    value = arrays[name]
    _require(value.ndim > 0 and value.shape[0] == 256, f"producer {name} is not row-aligned")
    receipt = metadata.get("arrays", {}).get(name, {})
    _require(receipt.get("shape") == list(value.shape), f"producer {name} shape receipt drifted")
    _require(receipt.get("dtype") == str(value.dtype), f"producer {name} dtype receipt drifted")
    _require(receipt.get("sha256") == _array_sha256(value), f"producer {name} SHA drifted")
  return metadata, arrays


def _load_ledger(path: Path, expected_arm: str) -> list[dict[str, Any]]:
  _require(path.is_file() and path.stat().st_size > 0, f"missing serving envelope: {path}")
  records = []
  # `_p38_request_journal` increments its process-local counter before
  # returning the call index, so a complete runner chronology starts at 1.
  previous_call = 0
  with path.open("r", encoding="utf-8") as source:
    for line_number, line in enumerate(source, 1):
      try:
        record = json.loads(line)
      except json.JSONDecodeError as exc:
        raise CarrierError(f"serving envelope line {line_number} is invalid: {exc}") from exc
      _require(record.get("schema") == "m15-apc-serving-envelope-v1", f"line {line_number} schema drifted")
      _require(record.get("arm") == expected_arm, f"line {line_number} target arm drifted")
      _require(record.get("serving_arm") in ("A", "B"), f"line {line_number} serving arm drifted")
      _require(
          record.get("program_path") in PROGRAM_PATHS,
          f"line {line_number} program path drifted",
      )
      call_index = int(record.get("call_index", -1))
      _require(call_index == previous_call + 1, f"serving call chronology is not contiguous at line {line_number}")
      previous_call = call_index
      request_order = record.get("request_order")
      requests = record.get("requests")
      _require(isinstance(request_order, list) and request_order, f"line {line_number} has no request order")
      _require(isinstance(requests, list) and requests, f"line {line_number} has no requests")
      _require(
          request_order == [item.get("request_id") for item in requests],
          f"line {line_number} request order disagrees with payload",
      )
      _require(len(set(request_order)) == len(request_order), f"line {line_number} duplicates a request")
      for request in requests:
        request_id = request.get("request_id")
        _require(isinstance(request_id, str) and request_id, f"line {line_number} has an invalid request ID")
        scheduled = int(request.get("scheduled_tokens", 0))
        computed = int(request.get("num_computed_tokens", -1))
        prompt = int(request.get("num_prompt_tokens", -1))
        num_tokens = int(request.get("num_tokens", -1))
        block_size = int(request.get("block_size", 0))
        before = int(request.get("logical_blocks_before", -1))
        after = int(request.get("logical_blocks_after", -1))
        pages = request.get("physical_pages")
        token_sha = request.get("token_history_sha256")
        _require(scheduled > 0 and computed >= 0 and prompt >= 0, f"line {line_number} request geometry is invalid")
        _require(num_tokens >= computed + scheduled, f"line {line_number} request token range is incomplete")
        _require(block_size > 0, f"line {line_number} block size is invalid")
        _require(before == (computed + block_size - 1) // block_size, f"line {line_number} before-block count drifted")
        _require(after == (computed + scheduled + block_size - 1) // block_size, f"line {line_number} after-block count drifted")
        _require(isinstance(pages, list) and len(pages) == after, f"line {line_number} page table is incomplete")
        _require(all(isinstance(page, int) and page >= 0 for page in pages), f"line {line_number} has an invalid physical page")
        _require(
            isinstance(token_sha, str)
            and len(token_sha) == 64
            and all(char in "0123456789abcdef" for char in token_sha),
            f"line {line_number} token SHA is invalid",
        )
      records.append(record)
  _require(records, "serving envelope is empty")
  _require({record["serving_arm"] for record in records} == {"A", "B"}, "serving envelope must contain both A and B calls")
  paths_by_arm = {
      arm: {
          str(record["program_path"])
          for record in records
          if record["serving_arm"] == arm
      }
      for arm in ("A", "B")
  }
  _require(
      paths_by_arm["A"] == PROGRAM_PATHS,
      "serving arm A must attest standard and continue_decode program paths",
  )
  _require(
      paths_by_arm["B"] == {"standard"},
      "serving arm B must remain on the full-reset standard program path",
  )
  return records


def _histories(arrays: dict[str, np.ndarray]) -> list[np.ndarray]:
  result = []
  for row in range(256):
    prompt = np.asarray(arrays["prompt_ids"][row])[
        np.asarray(arrays["prompt_mask"][row], dtype=np.bool_)
    ]
    completion = np.asarray(arrays["completion_ids"][row])[
        np.asarray(arrays["completion_valid_mask"][row], dtype=np.bool_)
    ]
    result.append(np.concatenate((prompt, completion)).astype("<i8", copy=False))
  return result


def _candidate_lookup(
    histories: list[np.ndarray], needed: set[tuple[int, str]]
) -> dict[tuple[int, str], list[int]]:
  matches: dict[tuple[int, str], list[int]] = {key: [] for key in needed}
  needed_lengths = {length for length, _ in needed}
  max_needed = max(needed_lengths)
  for row, history in enumerate(histories):
    digest = hashlib.sha256()
    for length, token in enumerate(history[:max_needed], 1):
      digest.update(np.asarray([token], dtype="<i8").tobytes())
      if length in needed_lengths:
        key = (length, digest.hexdigest())
        if key in matches:
          matches[key].append(row)
  return matches


def _request_joins(
    records: list[dict[str, Any]], histories: list[np.ndarray]
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], list[int]]]:
  needed = {
      (int(request["num_tokens"]), str(request["token_history_sha256"]))
      for record in records
      for request in record["requests"]
  }
  lookup = _candidate_lookup(histories, needed)
  missing = sorted(key for key, rows in lookup.items() if not rows)
  _require(not missing, f"serving token histories do not join the producer unit: {missing[:8]}")
  state: dict[str, dict[str, Any]] = {}
  for record in records:
    for request in record["requests"]:
      request_id = request["request_id"]
      key = (int(request["num_tokens"]), str(request["token_history_sha256"]))
      candidates = set(lookup[key])
      current = state.setdefault(request_id, {
          "request_id": request_id,
          "candidate_source_rows": candidates,
          "first_call": int(record["call_index"]),
          "last_call": int(record["call_index"]),
          "observations": 0,
          "serving_arms": set(),
          "program_paths": set(),
          "max_num_tokens": 0,
      })
      current["candidate_source_rows"] &= candidates
      _require(current["candidate_source_rows"], f"request {request_id} changes producer identity")
      current["last_call"] = int(record["call_index"])
      current["observations"] += 1
      current["serving_arms"].add(record["serving_arm"])
      current["program_paths"].add(record["program_path"])
      current["max_num_tokens"] = max(current["max_num_tokens"], key[0])
  joins = []
  for request_id in sorted(state):
    item = state[request_id]
    joins.append({
        **item,
        "candidate_source_rows": sorted(item["candidate_source_rows"]),
        "serving_arms": sorted(item["serving_arms"]),
        "program_paths": sorted(item["program_paths"]),
    })
  return joins, lookup


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


def package(
    *,
    producer_path: Path,
    ledger_path: Path,
    first_red_dir: Path,
    capture_classification_path: Path,
    m15_classification_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
  _require(not output_dir.exists(), f"refusing to overwrite output directory: {output_dir}")
  capture = _load_json(capture_classification_path, "capture classification")
  m15 = _load_json(m15_classification_path, "M15 classification")
  _require(capture.get("verdict") == "PASS", "capture classification is not PASS")
  _require(
      m15.get("status") == "FRESH_TARGET_RED_FROZEN" and m15.get("arm") == "on",
      "full replay packaging requires a fresh APC-on target red",
  )
  producer_meta, arrays = _load_producer(producer_path)
  _require(producer_meta.get("arm") == "on", "producer unit is not the APC-on arm")
  _require(producer_meta.get("source_commit") == m15.get("source_commit"), "producer source commit drifted")
  records = _load_ledger(ledger_path, "on")
  histories = _histories(arrays)
  request_joins, lookup = _request_joins(records, histories)

  first_contract_path = first_red_dir / "first_red_contract.json"
  first_capsule_path = first_red_dir / "first_red_capsule.npz"
  first_sums_path = first_red_dir / "SHA256SUMS"
  first = _load_json(first_contract_path, "first-red contract")
  _require(first.get("status") == "FIRST_RED_ROW_FROZEN", "first-red carrier is not frozen")
  source_row = int(first.get("source_row", -1))
  _require(0 <= source_row < 256, "first-red source row is invalid")
  incident = first.get("first_incident", {})
  call_index = int(incident.get("call_index", -1))
  request_id = incident.get("request_id")
  matching_records = [record for record in records if int(record["call_index"]) == call_index]
  _require(len(matching_records) == 1, "first-red serving call is absent or duplicated")
  record = matching_records[0]
  _require(record["serving_arm"] == "A", "first-red incident did not occur in serving arm A")
  matching_requests = [request for request in record["requests"] if request["request_id"] == request_id]
  _require(len(matching_requests) == 1, "first-red request is absent or duplicated in its serving call")
  request = matching_requests[0]
  key = (int(request["num_tokens"]), str(request["token_history_sha256"]))
  _require(source_row in lookup[key], "first-red request does not join its producer row")
  _require(int(request["num_computed_tokens"]) == int(incident["num_computed_tokens"]), "first-red prefix drifted")
  _require(request["physical_pages"] == incident["physical_pages"], "first-red physical page table drifted")

  _require(first_capsule_path.is_file() and first_sums_path.is_file(), "first-red payload is incomplete")
  with np.load(first_capsule_path, allow_pickle=False) as archive:
    first_arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
  _require(np.asarray(first_arrays["selected_rows"]).reshape(-1).tolist() == [source_row], "first-red capsule row drifted")
  for name in REQUIRED_ARRAYS - {"source_rows", "metadata_json"}:
    _require(
        np.array_equal(np.asarray(first_arrays[name])[0], np.asarray(arrays[name])[source_row]),
        f"first-red {name} differs from the full producer unit",
    )

  output_dir.mkdir(parents=True, exist_ok=False)
  joins_path = output_dir / "request_row_joins.jsonl"
  _write_jsonl(joins_path, request_joins)
  relative_sources = {
      "producer_unit": Path("../") / producer_path.name,
      "serving_envelope": Path("../") / ledger_path.name,
      "first_red_capsule": Path("../") / first_red_dir.name / first_capsule_path.name,
      "first_red_contract": Path("../") / first_red_dir.name / first_contract_path.name,
      "first_red_manifest": Path("../") / first_red_dir.name / first_sums_path.name,
  }
  contract = {
      "schema": "m15-apc-full-replay-carrier-v1",
      "status": "FULL_REPLAY_CARRIER_FROZEN",
      "source_commit": m15.get("source_commit"),
      "arm": "on",
      "producer_rows": 256,
      "request_count": len(request_joins),
      "serving_call_count": len(records),
      "serving_arms": sorted({record["serving_arm"] for record in records}),
      "program_paths": sorted({record["program_path"] for record in records}),
      "program_paths_by_arm": {
          arm: sorted({
              record["program_path"]
              for record in records
              if record["serving_arm"] == arm
          })
          for arm in ("A", "B")
      },
      "first_red": {
          "source_row": source_row,
          "request_id": request_id,
          "call_index": call_index,
          "num_computed_tokens": int(request["num_computed_tokens"]),
          "dp_rank": int(request["dp_rank"]),
          "local_scheduler_slot": int(request["local_scheduler_slot"]),
          "physical_pages": request["physical_pages"],
      },
      "input_sha256": {
          "producer_unit": _sha256(producer_path),
          "serving_envelope": _sha256(ledger_path),
          "first_red_capsule": _sha256(first_capsule_path),
          "first_red_contract": _sha256(first_contract_path),
          "first_red_manifest": _sha256(first_sums_path),
          "capture_classification": _sha256(capture_classification_path),
          "m15_classification": _sha256(m15_classification_path),
      },
      "derived_sha256": {"request_row_joins": _sha256(joins_path)},
      "preserved": [
          "all_256_final_prompt_and_completion_token_streams",
          "all_A_B_logprob_rows_and_masks",
          "every_serving_call_dispatch_order",
          "every_scheduled_request_token_history_hash_and_position",
          "DP_rank_local_slot_and_physical_page_table_per_call",
          "exact_first_red_incident_and_page_generations",
      ],
      "limitations": [
          "carrier_has_not_yet_been_executed_by_a_replay_harness",
          "scheduler_dispatch_is_recorded_but_not_yet_forced",
          "physical_page_ids_may_require_equivalent_geometry_instead_of_identical_ids",
          "this_artifact_is_not_a_root_cause_or_repair_verdict",
      ],
  }
  contract_path = output_dir / "replay_contract.json"
  _write_json(contract_path, contract)
  manifest_entries = [
      (_sha256(producer_path), relative_sources["producer_unit"]),
      (_sha256(ledger_path), relative_sources["serving_envelope"]),
      (_sha256(first_capsule_path), relative_sources["first_red_capsule"]),
      (_sha256(first_contract_path), relative_sources["first_red_contract"]),
      (_sha256(first_sums_path), relative_sources["first_red_manifest"]),
      (_sha256(joins_path), Path(joins_path.name)),
      (_sha256(contract_path), Path(contract_path.name)),
  ]
  manifest_path = output_dir / "SHA256SUMS"
  with manifest_path.open("x", encoding="utf-8") as target:
    for digest, relative in manifest_entries:
      target.write(f"{digest}  {relative.as_posix()}\n")
    target.flush()
    os.fsync(target.fileno())
  return contract


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--producer-unit", required=True, type=Path)
  parser.add_argument("--serving-envelope", required=True, type=Path)
  parser.add_argument("--first-red-dir", required=True, type=Path)
  parser.add_argument("--capture-classification", required=True, type=Path)
  parser.add_argument("--m15-classification", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  args = parser.parse_args()
  try:
    result = package(
        producer_path=args.producer_unit,
        ledger_path=args.serving_envelope,
        first_red_dir=args.first_red_dir,
        capture_classification_path=args.capture_classification,
        m15_classification_path=args.m15_classification,
        output_dir=args.output_dir,
    )
  except (OSError, KeyError, TypeError, ValueError, CarrierError) as exc:
    print(json.dumps({"status": "INCONCLUSIVE", "error": str(exc)}, sort_keys=True))
    raise SystemExit(2) from exc
  print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
  main()
