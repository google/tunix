#!/usr/bin/env python3
"""Fail-closed classifier for bounded P38 continue-decode captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PRE_ARRAYS = {
    "input_ids",
    "input_positions",
    "active_mask",
    "md_input_positions",
    "md_block_tables",
    "md_seq_lens",
    "md_query_start_loc",
    "md_request_distribution",
    "tokens_indices_selector",
    "rng",
}
POST_ARRAYS = {
    "generated_tokens",
    "final_input_positions",
    "final_seq_lens",
    "logprob_token_ids",
    "logprob_values",
    "logprob_ranks",
}


class CaptureError(RuntimeError):
  """Raised when a serving capture is incomplete or internally inconsistent."""


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise CaptureError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_stage(directory: Path, seq: int, stage: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
  base = directory / f"p38_serving_{seq:04d}_{stage}"
  json_path = Path(str(base) + ".json")
  npz_path = Path(str(base) + ".npz")
  _require(json_path.is_file(), f"missing {stage} JSON for sequence {seq}")
  _require(npz_path.is_file(), f"missing {stage} NPZ for sequence {seq}")
  record = json.loads(json_path.read_text(encoding="utf-8"))
  _require(record.get("schema_version") == 1, f"bad schema for {stage} sequence {seq}")
  _require(record.get("stage") == stage, f"bad stage for sequence {seq}: {record.get('stage')!r}")
  _require(record.get("seq") == seq, f"bad sequence number in {stage} record")
  _require(record.get("npz_sha256") == _sha256(npz_path), f"NPZ SHA mismatch for {stage} sequence {seq}")
  with np.load(npz_path, allow_pickle=False) as archive:
    arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
  _require(sorted(arrays) == sorted(record.get("arrays", [])), f"array inventory mismatch for {stage} sequence {seq}")
  return record, arrays


def classify(directory: Path, expected_records: int) -> dict[str, Any]:
  _require(expected_records > 0, "expected_records must be positive")
  _require(directory.is_dir(), f"capture directory does not exist: {directory}")
  pre_files = sorted(directory.glob("p38_serving_*_pre.json"))
  post_files = sorted(directory.glob("p38_serving_*_post.json"))
  _require(len(pre_files) == expected_records, f"expected {expected_records} pre records, found {len(pre_files)}")
  _require(len(post_files) == expected_records, f"expected {expected_records} post records, found {len(post_files)}")

  summaries = []
  for seq in range(expected_records):
    pre, pre_arrays = _load_stage(directory, seq, "pre")
    post, post_arrays = _load_stage(directory, seq, "post")
    _require(PRE_ARRAYS.issubset(pre_arrays), f"missing required pre arrays for sequence {seq}: {sorted(PRE_ARRAYS - set(pre_arrays))}")
    _require(POST_ARRAYS.issubset(post_arrays), f"missing required post arrays for sequence {seq}: {sorted(POST_ARRAYS - set(post_arrays))}")
    _require(
        any(name.startswith("sampling_leaf_") for name in pre_arrays),
        f"sequence {seq} has no sampling-metadata leaves",
    )

    meta = pre.get("meta", {})
    requests = meta.get("requests", [])
    _require(meta.get("continue_decode_enabled") is True, f"sequence {seq} did not capture continue-decode")
    _require(meta.get("caller_update_kv_cache") is True, f"sequence {seq} has an invalid caller cache-update contract")
    _require(
        meta.get("output_update_kv_cache") is (not bool(meta.get("kv_unified"))),
        f"sequence {seq} has an inconsistent output cache-update contract",
    )
    _require(meta.get("request_ids"), f"sequence {seq} has no request IDs")
    _require(meta.get("request_ids_by_dp"), f"sequence {seq} has no DP request mapping")
    _require(requests, f"sequence {seq} has no request metadata")
    _require(all(item.get("block_ids") for item in requests), f"sequence {seq} has a request without physical page IDs")
    _require(all(item.get("token_ids") for item in requests), f"sequence {seq} has a request without token history")
    _require(meta.get("kv_caches_spec"), f"sequence {seq} has no KV-cache specification")
    _require(meta.get("block_size", 0) > 0, f"sequence {seq} has an invalid page size")
    _require(meta.get("observed_max_prefix", -1) >= meta.get("capture_min_prefix", 0), f"sequence {seq} violates the prefix filter")
    _require(all(meta.get("rpa_block_tuples", {}).get(name) for name in ("CANON_RPA_D", "CANON_RPA_P", "CANON_RPA_M")), f"sequence {seq} is missing a pinned RPA block tuple")

    post_meta = post.get("meta", {})
    actual_steps = int(post_meta.get("actual_steps", 0))
    _require(actual_steps > 0, f"sequence {seq} completed zero decode steps")
    _require(post_arrays["generated_tokens"].shape[0] == actual_steps, f"sequence {seq} generated-token step count mismatch")
    _require(post_arrays["logprob_values"].shape[0] == actual_steps, f"sequence {seq} logprob step count mismatch")
    _require(int(post_meta.get("completed_records", -1)) == seq + 1, f"sequence {seq} has a bad completed-record count")
    _require(int(post_meta.get("expected_max_records", -1)) >= expected_records, f"sequence {seq} reports an undersized record budget")

    summaries.append({
        "seq": seq,
        "requests": len(requests),
        "actual_steps": actual_steps,
        "observed_max_prefix": int(meta["observed_max_prefix"]),
        "kv_unified": bool(meta.get("kv_unified")),
    })

  return {
      "schema_version": 1,
      "verdict": "PASS",
      "scope": "p38-serving-capture",
      "records": summaries,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--directory", required=True, type=Path)
  parser.add_argument("--expected-records", required=True, type=int)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  try:
    report = classify(args.directory, args.expected_records)
  except CaptureError as error:
    report = {
        "schema_version": 1,
        "verdict": "INCONCLUSIVE",
        "scope": "p38-serving-capture",
        "error": str(error),
    }
    if args.output:
      args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(2) from error
  if args.output:
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
  main()
