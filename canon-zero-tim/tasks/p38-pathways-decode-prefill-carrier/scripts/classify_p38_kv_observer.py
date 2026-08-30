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
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


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
    layer_indices = record.get("layer_indices", list(range(layers)))
    _require(
        isinstance(layer_indices, list)
        and len(layer_indices) == layers
        and all(isinstance(value, int) and value >= 0
                for value in layer_indices)
        and len(set(layer_indices)) == layers,
        f"observer layer index contract drifted: {expected_json}",
    )
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
        "layer_indices": layer_indices,
        "arrays": arrays,
        "path": path.name,
        "json_sha256": _sha256(path),
        "npz_path": npz_path.name,
    })
  indices = [int(record["record_index"]) for record in records]
  _require(
      indices
      and indices == list(range(indices[0], indices[0] + len(indices))),
      "observer record indices are not one contiguous window",
  )
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
        "layer_indices", "cache_shape", "cache_dtype",
    ):
      _require(clean.get(key) == live.get(key),
               f"observer A/B contract differs at {key}")
    live_effective = live.get("cache_effective_sharding")
    clean_effective = clean.get("cache_effective_sharding")
    if live_effective is None and clean_effective is None:
      # Backward compatibility for archived P38 evidence produced before the
      # effective device-to-slice contract was recorded.
      _require(clean.get("cache_sharding") == live.get("cache_sharding"),
               "observer A/B contract differs at cache_sharding")
    else:
      _require(
          isinstance(live_effective, dict)
          and live_effective.get("schema") ==
          "p38-effective-device-sharding-v1",
          "live observer effective sharding is absent or malformed",
      )
      _require(
          isinstance(clean_effective, dict)
          and clean_effective.get("schema") ==
          "p38-effective-device-sharding-v1",
          "clean observer effective sharding is absent or malformed",
      )
      _require(
          clean_effective == live_effective,
          "observer A/B contract differs at cache_effective_sharding",
      )
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
    for layer_offset, prefix_index in np.argwhere(combined):
      layer = int(live["layer_indices"][int(layer_offset)])
      differing_layers.add(layer)
      differing_pages.add(int(page))
      if first is None:
        first = {
            "layer": layer,
            "logical_page": int(page),
            "page_prefix_extent": int(prefix_index) + 1,
            "aggregate_diff": bool(
                aggregate_diff[layer_offset, prefix_index]),
            "sample_diff": bool(sample_diff[layer_offset, prefix_index]),
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
      "cache_sharding_repr_equal": (
          live.get("cache_sharding") == clean.get("cache_sharding")
      ),
      "cache_effective_sharding_equal": (
          live.get("cache_effective_sharding") ==
          clean.get("cache_effective_sharding")
      ),
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


def _token_history_sha256(tokens: np.ndarray) -> str:
  values = np.ascontiguousarray(np.asarray(tokens, dtype="<i8"))
  return hashlib.sha256(values.tobytes()).hexdigest()


def _load_replay_receipts(path: Path) -> dict[tuple[int, str], list[dict]]:
  _require(path.is_file(), f"M15 replay ledger is absent: {path}")
  receipts: dict[tuple[int, str], list[dict]] = {}
  with path.open("r", encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, start=1):
      if not line.strip():
        continue
      record = json.loads(line)
      _require(
          record.get("schema") == "m15-apc-serving-envelope-v1",
          f"M15 replay ledger schema drifted at line {line_number}",
      )
      if record.get("serving_arm") != "A":
        continue
      diagnostic_round = int(record.get("diagnostic_round", -1))
      _require(
          0 <= diagnostic_round < 8,
          f"M15 replay round is invalid at line {line_number}",
      )
      for request in record.get("requests", ()):
        request_id = str(request.get("request_id", ""))
        token_count = int(request.get("num_tokens", -1))
        digest = str(request.get("token_history_sha256", ""))
        _require(
            request_id and token_count >= 0 and len(digest) == 64,
            f"M15 replay request receipt is invalid at line {line_number}",
        )
        receipts.setdefault((diagnostic_round, request_id), []).append({
            "call_index": int(record.get("call_index", -1)),
            "token_count": token_count,
            "token_history_sha256": digest,
        })
  _require(receipts, "M15 replay ledger contains no A request receipts")
  return receipts


def _bind_source_request(
    pairs: list[tuple[dict, dict]], capsules: list[Path], replay_ledger: Path
) -> dict[str, Any]:
  histories_by_round: dict[int, list[dict[str, Any]]] = {}
  for path in capsules:
    diagnostic_round, histories = _load_capsule_histories(path)
    _require(
        diagnostic_round not in histories_by_round,
        f"duplicate mismatch capsule round: {diagnostic_round}",
    )
    histories_by_round[diagnostic_round] = histories
  receipts = _load_replay_receipts(replay_ledger)
  candidates = []
  source_identity = None
  for live, _clean in pairs:
    diagnostic_round = int(live["diagnostic_round"])
    target = np.asarray(live["arrays"]["token_ids"], dtype=np.int32)
    histories = [
        history for history in histories_by_round.get(diagnostic_round, ())
        if target.size <= history["tokens"].size
        and np.array_equal(target, history["tokens"][:target.size])
        and any(
            int(history["prompt_length"]) + int(position) < target.size
            for position in history["mismatch_positions"]
        )
    ]
    _require(
        len(histories) == 1,
        "targeted KV alias does not bind exactly one red source row",
    )
    history = histories[0]
    identity = (
        diagnostic_round,
        int(history["source_row"]),
        str(history["capsule"]),
    )
    if source_identity is None:
      source_identity = identity
    _require(
        identity == source_identity,
        "targeted KV aliases do not describe one red source row",
    )
    observations = {}
    for receipt in receipts.get((diagnostic_round, live["request_id"]), ()):
      length = int(receipt["token_count"])
      if not target.size < length <= history["tokens"].size:
        continue
      expected = _token_history_sha256(history["tokens"][:length])
      matched = expected == receipt["token_history_sha256"]
      prior = observations.get(length)
      _require(
          prior is None or prior == matched,
          "M15 replay ledger has conflicting duplicate prefix evidence",
      )
      observations[length] = matched
    matching = sorted(length for length, matched in observations.items()
                      if matched)
    conflicting = sorted(length for length, matched in observations.items()
                         if not matched)
    if conflicting:
      status = "FUTURE_PREFIX_CONFLICT"
    elif matching:
      status = "FUTURE_PREFIX_MATCH"
    else:
      status = "FUTURE_PREFIX_UNOBSERVED"
    candidates.append({
        "source_a_record_index": int(live["record_index"]),
        "request_id": str(live["request_id"]),
        "status": status,
        "matching_prefix_lengths": matching,
        "conflicting_prefix_lengths": conflicting,
    })
  matching_candidates = [
      candidate for candidate in candidates
      if candidate["status"] == "FUTURE_PREFIX_MATCH"
  ]
  _require(
      len(matching_candidates) == 1,
      "targeted KV aliases do not have one future-prefix match",
  )
  alternatives = [
      candidate for candidate in candidates
      if candidate is not matching_candidates[0]
  ]
  _require(
      alternatives
      and all(candidate["status"] == "FUTURE_PREFIX_CONFLICT"
              for candidate in alternatives),
      "targeted KV aliases lack explicit future-prefix conflicts",
  )
  required_horizon = max(
      min(candidate["conflicting_prefix_lengths"])
      for candidate in alternatives
  )
  selected = matching_candidates[0]
  selected_proof = max(selected["matching_prefix_lengths"])
  _require(
      selected_proof >= required_horizon,
      "selected targeted KV alias does not reach the elimination horizon",
  )
  assert source_identity is not None
  return {
      "schema": "m15-kv-source-request-binding-v1",
      "status": "UNIQUE_FUTURE_PREFIX_BINDING",
      "diagnostic_round": source_identity[0],
      "source_row": source_identity[1],
      "capsule": source_identity[2],
      "anchor_prefix_tokens": int(
          pairs[0][0]["target_seq_len"]),
      "selected_request_id": selected["request_id"],
      "selected_source_a_record_index": selected["source_a_record_index"],
      "required_elimination_horizon": required_horizon,
      "selected_proof_prefix_tokens": selected_proof,
      "candidates": candidates,
  }


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
    directory: Path, capsules: list[Path], require_red_join: bool,
    replay_ledger: Path | None = None,
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
  source_request_binding = None
  if replay_ledger is not None:
    _require(
        require_red_join and capsules,
        "M15 replay binding requires red-join classification",
    )
    source_request_binding = _bind_source_request(
        pairs, capsules, replay_ledger
    )
    selected_index = int(
        source_request_binding["selected_source_a_record_index"]
    )
    joined_comparisons = [
        item for item in joined_comparisons
        if int(item["source_a_record_index"]) == selected_index
    ]
    _require(
        len(joined_comparisons) == 1,
        "selected M15 source request lacks one KV comparison",
    )
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
  if replay_ledger is not None:
    source_inputs["replay_ledger"] = {
        "path": replay_ledger.name,
        "sha256": _sha256(replay_ledger),
    }
  return {
      "schema": "p38-live-kv-classification-v2",
      "status": "PASS",
      "classification": classification,
      "records": len(records),
      "pairs": len(pairs),
      "comparisons": comparisons,
      "red_joins": red_joins,
      "source_request_binding": source_request_binding,
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
  parser.add_argument("--replay-ledger", type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  report = classify(
      args.directory, args.capsule, args.require_red_join,
      args.replay_ledger,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
  print(json.dumps(report, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
