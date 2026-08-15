#!/usr/bin/env python3
"""Fail-closed classifier for paired P38 live/clean KV prefix tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ARRAY_KEYS = {
    "aggregates",
    "samples",
    "token_ids",
    "physical_pages",
    "padded_global_pages",
    "valid_tokens",
}


class ObserverError(RuntimeError):
  """Raised when observer evidence is absent or internally inconsistent."""


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ObserverError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_records(directory: Path) -> list[dict[str, Any]]:
  paths = sorted(directory.glob("p38_kv_observer_*.json"))
  _require(paths, "P38 KV observer produced no JSON records")
  records = []
  for path in paths:
    record = json.loads(path.read_text(encoding="utf-8"))
    _require(
        record.get("schema") == "p38-live-kv-prefix-table-v1",
        f"observer record has an invalid schema: {path.name}",
    )
    arm = record.get("arm")
    index = int(record.get("record_index", -1))
    _require(arm in ("A", "B") and index >= 0,
             f"observer record identity is invalid: {path.name}")
    expected_json = f"p38_kv_observer_{index:04d}_{arm.lower()}.json"
    expected_npz = f"p38_kv_observer_{index:04d}_{arm.lower()}.npz"
    _require(path.name == expected_json,
             f"observer JSON filename drifted: {path.name}")
    npz_path = directory / expected_npz
    _require(npz_path.is_file(), f"observer NPZ is absent: {expected_npz}")
    _require(record.get("npz_sha256") == _sha256(npz_path),
             f"observer NPZ SHA failed: {expected_npz}")
    with np.load(npz_path, allow_pickle=False) as archive:
      arrays = {name: np.array(archive[name], copy=True)
                for name in archive.files}
    _require(set(arrays) == ARRAY_KEYS,
             f"observer array inventory drifted: {expected_npz}")
    _require(set(record.get("array_keys", ())) == ARRAY_KEYS,
             f"observer array attestation drifted: {expected_json}")

    aggregates = arrays["aggregates"]
    samples = arrays["samples"]
    token_ids = arrays["token_ids"].reshape(-1)
    physical_pages = arrays["physical_pages"].reshape(-1)
    global_pages = arrays["padded_global_pages"].reshape(-1)
    valid_tokens = arrays["valid_tokens"].reshape(-1)
    layers = int(record.get("layer_count", 0))
    page_size = int(record.get("block_size", 0))
    logical_pages = int(record.get("logical_pages", 0))
    observer_pages = int(record.get("observer_pages", 0))
    _require(
        aggregates.shape == (layers, observer_pages, page_size, 4),
        f"observer aggregate shape drifted: {expected_npz}",
    )
    _require(
        samples.ndim == 5
        and samples.shape[:4] == (layers, observer_pages, page_size, 3),
        f"observer sample shape drifted: {expected_npz}",
    )
    _require(
        token_ids.dtype.kind in "iu"
        and token_ids.size == int(record.get("target_seq_len", -1)),
        f"observer token geometry drifted: {expected_npz}",
    )
    _require(
        physical_pages.size == valid_tokens.size == logical_pages
        and global_pages.size == observer_pages
        and logical_pages <= observer_pages,
        f"observer page geometry drifted: {expected_npz}",
    )
    _require(
        valid_tokens.dtype.kind in "iu"
        and np.all((valid_tokens >= 1) & (valid_tokens <= page_size))
        and int(valid_tokens[:-1].sum()) ==
        max(0, logical_pages - 1) * page_size
        and int(valid_tokens.sum()) == token_ids.size,
        f"observer valid-token extents drifted: {expected_npz}",
    )
    token_sha = hashlib.sha256(
        np.ascontiguousarray(token_ids, dtype="<i8").tobytes()).hexdigest()
    _require(token_sha == record.get("token_history_sha256"),
             f"observer token SHA drifted: {expected_npz}")
    records.append({
        **record,
        "arrays": arrays,
        "path": path.name,
        "json_sha256": _sha256(path),
        "npz_path": npz_path.name,
    })
  indices = [int(record["record_index"]) for record in records]
  _require(indices == list(range(len(indices))),
           "observer record indices are not contiguous")
  return records


def _pair_records(records: list[dict[str, Any]]) -> list[tuple[dict, dict]]:
  by_index = {int(record["record_index"]): record for record in records}
  a_records = [record for record in records if record["arm"] == "A"]
  b_records = [record for record in records if record["arm"] == "B"]
  _require(a_records and len(a_records) == len(b_records),
           f"observer A/B record counts differ: A={len(a_records)} B={len(b_records)}")
  pairs = []
  seen_a = set()
  for clean in b_records:
    source_index = clean.get("source_a_record_index")
    _require(isinstance(source_index, int) and source_index in by_index,
             "clean observer record has no valid source A index")
    live = by_index[source_index]
    _require(live["arm"] == "A" and source_index not in seen_a,
             "clean observer record reused or misidentified its source A")
    seen_a.add(source_index)
    _require(clean.get("source_a_request_id") == live.get("request_id"),
             "observer A/B request provenance drifted")
    for key in (
        "diagnostic_round", "target_seq_len", "token_history_sha256",
        "block_size", "logical_pages", "observer_pages", "layer_count",
        "cache_shape", "cache_dtype", "cache_sharding",
    ):
      _require(clean.get(key) == live.get(key),
               f"observer A/B contract differs at {key}")
    for key in ("token_ids", "valid_tokens"):
      _require(np.array_equal(live["arrays"][key], clean["arrays"][key]),
               f"observer A/B arrays differ at {key}")
    pairs.append((live, clean))
  _require(len(seen_a) == len(a_records), "an A observer record is unpaired")
  return sorted(pairs, key=lambda pair: int(pair[0]["record_index"]))


def _compare_pair(live: dict[str, Any], clean: dict[str, Any]) -> dict[str, Any]:
  a = live["arrays"]
  b = clean["arrays"]
  valid = np.asarray(a["valid_tokens"], dtype=np.int32)
  aggregate_cells = 0
  sample_cells = 0
  first = None
  differing_layers = set()
  differing_pages = set()
  for page, extent in enumerate(valid):
    a_aggregates = a["aggregates"][:, page, :int(extent)]
    b_aggregates = b["aggregates"][:, page, :int(extent)]
    a_samples = a["samples"][:, page, :int(extent)]
    b_samples = b["samples"][:, page, :int(extent)]
    aggregate_diff = np.any(a_aggregates != b_aggregates, axis=-1)
    sample_diff = np.any(a_samples != b_samples, axis=(-1, -2))
    aggregate_cells += int(aggregate_diff.sum())
    sample_cells += int(sample_diff.sum())
    combined = aggregate_diff | sample_diff
    for layer, prefix_index in np.argwhere(combined):
      differing_layers.add(int(layer))
      differing_pages.add(int(page))
      if first is None:
        first = {
            "layer": int(layer),
            "logical_page": int(page),
            "page_prefix_extent": int(prefix_index) + 1,
            "aggregate_diff": bool(aggregate_diff[layer, prefix_index]),
            "sample_diff": bool(sample_diff[layer, prefix_index]),
        }
  return {
      "source_a_record_index": int(live["record_index"]),
      "source_a_request_id": live["request_id"],
      "clean_b_record_index": int(clean["record_index"]),
      "clean_b_request_id": clean["request_id"],
      "diagnostic_round": int(live["diagnostic_round"]),
      "target_seq_len": int(live["target_seq_len"]),
      "valid_tokens": [
          int(value) for value in np.asarray(a["valid_tokens"]).reshape(-1)
      ],
      "aggregate_prefix_cells_differing": aggregate_cells,
      "sample_prefix_cells_differing": sample_cells,
      "differing_layers": sorted(differing_layers),
      "differing_logical_pages": sorted(differing_pages),
      "first_difference": first,
      "fingerprint_equal": first is None,
  }


def _load_capsule_histories(path: Path) -> tuple[int, list[dict[str, Any]]]:
  with np.load(path, allow_pickle=False) as archive:
    arrays = {name: np.asarray(archive[name]) for name in archive.files}
  required = {
      "metadata_json", "selected_rows", "prompt_ids", "prompt_mask",
      "completion_ids", "completion_valid_mask", "action_mask",
      "s_decode", "s_prefill",
  }
  _require(required.issubset(arrays), f"mismatch capsule is incomplete: {path}")
  metadata = json.loads(arrays["metadata_json"].tobytes().decode("utf-8"))
  diagnostic_round = int(metadata.get("diagnostic_round", -1))
  _require(0 <= diagnostic_round < 8,
           f"mismatch capsule round is invalid: {path}")
  histories = []
  for index, source_row_raw in enumerate(arrays["selected_rows"].reshape(-1)):
    prompt = arrays["prompt_ids"][index][
        np.asarray(arrays["prompt_mask"][index], dtype=np.bool_)]
    completion = arrays["completion_ids"][index][
        np.asarray(arrays["completion_valid_mask"][index], dtype=np.bool_)]
    action = np.asarray(arrays["action_mask"][index], dtype=np.bool_)
    decode = np.asarray(arrays["s_decode"][index])
    prefill = np.asarray(arrays["s_prefill"][index])
    _require(action.shape == decode.shape == prefill.shape,
             f"mismatch capsule row geometry drifted: {path}")
    byte_diff = (
        np.ascontiguousarray(decode).view(np.uint8)
        != np.ascontiguousarray(prefill).view(np.uint8)
    ).reshape(decode.size, decode.dtype.itemsize).any(axis=1).reshape(
        decode.shape)
    histories.append({
        "source_row": int(source_row_raw),
        "tokens": np.concatenate((prompt, completion)).astype(
            np.int32, copy=False),
        "mismatch_positions": [
            int(value) for value in np.flatnonzero(action & byte_diff)
        ],
        "prompt_length": int(prompt.size),
        "capsule": path.name,
    })
  return diagnostic_round, histories


def _join_red_candidates(
    pairs: list[tuple[dict, dict]], capsules: list[Path]
) -> list[dict[str, Any]]:
  histories_by_round: dict[int, list[dict[str, Any]]] = {}
  for path in capsules:
    diagnostic_round, histories = _load_capsule_histories(path)
    _require(diagnostic_round not in histories_by_round,
             f"duplicate mismatch capsule round: {diagnostic_round}")
    histories_by_round[diagnostic_round] = histories
  joins = []
  for live, _ in pairs:
    diagnostic_round = int(live["diagnostic_round"])
    target = np.asarray(live["arrays"]["token_ids"], dtype=np.int32)
    candidates = [
        history for history in histories_by_round.get(diagnostic_round, ())
        if target.size <= history["tokens"].size
        and np.array_equal(target, history["tokens"][:target.size])
    ]
    _require(len(candidates) <= 1,
             "observer token prefix ambiguously joins mismatch rows")
    if not candidates:
      continue
    match = candidates[0]
    covered_mismatches = [
        int(position) for position in match["mismatch_positions"]
        if int(match["prompt_length"]) + int(position) < int(target.size)
    ]
    if not covered_mismatches:
      continue
    joins.append({
        "source_a_record_index": int(live["record_index"]),
        "diagnostic_round": diagnostic_round,
        "source_row": int(match["source_row"]),
        "capsule": match["capsule"],
        "mismatch_positions": covered_mismatches,
        "mismatch_count": len(covered_mismatches),
        "target_seq_len": int(target.size),
    })
  return joins


def classify(
    directory: Path, capsules: list[Path], require_red_join: bool
) -> dict[str, Any]:
  records = _load_records(directory)
  pairs = _pair_records(records)
  comparisons = [_compare_pair(live, clean) for live, clean in pairs]
  red_joins = _join_red_candidates(pairs, capsules) if capsules else []
  if require_red_join:
    _require(capsules, "red-join classification requires mismatch capsules")
    _require(red_joins, "no paired observer candidate joined a red capsule row")
  joined_indices = {int(join["source_a_record_index"]) for join in red_joins}
  if require_red_join:
    expected_indices = {
        int(live["record_index"]) for live, _clean in pairs
    }
    _require(
        joined_indices == expected_indices,
        "not every observer pair joined a red capsule row: "
        f"joined={sorted(joined_indices)} expected={sorted(expected_indices)}",
    )
  joined_comparisons = [
      item for item in comparisons
      if int(item["source_a_record_index"]) in joined_indices
  ]
  if red_joins and any(not item["fingerprint_equal"]
                       for item in joined_comparisons):
    classification = "live_kv_fingerprint_differs_on_red_row"
  elif red_joins:
    classification = "live_kv_fingerprint_equal_on_red_row"
  else:
    classification = "observer_pairs_valid_red_join_pending"
  source_inputs = {
      "classifier": {
          "path": Path(__file__).name,
          "sha256": _sha256(Path(__file__)),
      },
      "observer_records": [
          {
              "arm": record["arm"],
              "record_index": int(record["record_index"]),
              "json": record["path"],
              "json_sha256": record["json_sha256"],
              "npz": record["npz_path"],
              "npz_sha256": record["npz_sha256"],
              "valid_tokens": [
                  int(value)
                  for value in np.asarray(
                      record["arrays"]["valid_tokens"]
                  ).reshape(-1)
              ],
          }
          for record in records
      ],
      "capsules": [
          {
              "path": path.name,
              "sha256": _sha256(path),
          }
          for path in capsules
      ],
  }
  return {
      "schema": "p38-live-kv-classification-v2",
      "status": "PASS",
      "classification": classification,
      "records": len(records),
      "pairs": len(pairs),
      "comparisons": comparisons,
      "red_joins": red_joins,
      "source_inputs": source_inputs,
      "claim_level": "bit-level-diagnostic-fingerprint-not-full-kv-bytes",
      "claim_ceiling": [
          "A/B token prefixes and valid extents are exact.",
          "The integer aggregates and fixed samples are diagnostic fingerprints, not cryptographic hashes.",
          "An equal fingerprint does not mathematically prove full KV byte equality.",
          "Only a candidate joined to an A/B-red capsule row can choose the mechanism branch.",
      ],
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--directory", required=True, type=Path)
  parser.add_argument("--capsule", action="append", default=[], type=Path)
  parser.add_argument("--require-red-join", action="store_true")
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  report = classify(args.directory, args.capsule, args.require_red_join)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
  print(json.dumps(report, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
