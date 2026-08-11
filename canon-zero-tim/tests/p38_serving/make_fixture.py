#!/usr/bin/env python3
"""Create one valid bounded serving-capture fixture for shell postflight tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _write(directory: Path, stage: str, arrays: dict, meta: dict) -> None:
  base = directory / f"p38_serving_0000_{stage}"
  npz_path = Path(str(base) + ".npz")
  with npz_path.open("xb") as stream:
    np.savez(stream, **arrays)
  record = {
      "schema_version": 2,
      "stage": stage,
      "seq": 0,
      "arrays": sorted(arrays),
      "describe": {},
      "meta": meta,
      "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
  }
  Path(str(base) + ".json").write_text(
      json.dumps(record), encoding="utf-8"
  )


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--directory", required=True, type=Path)
  parser.add_argument("--mismatch-capsule", required=True, type=Path)
  args = parser.parse_args()
  args.directory.mkdir(parents=True, exist_ok=False)
  names = {
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
  pre = {name: np.arange(4, dtype=np.int32) for name in names}
  pre["input_positions"] = np.array([2, 0, 0, 0], dtype=np.int32)
  pre["active_mask"] = np.array([True, False, False, False])
  pre["md_block_tables"] = np.array(
      [[7, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      dtype=np.int32,
  )
  pre["md_seq_lens"] = np.array([3, 0, 0, 0], dtype=np.int32)
  pre["md_query_start_loc"] = np.array([0, 1, 1, 1, 1], dtype=np.int32)
  pre["tokens_indices_selector"] = np.array([0], dtype=np.int32)
  pre["sampling_leaf_0000"] = np.arange(4, dtype=np.float32)
  _write(args.directory, "pre", pre, {
      "continue_decode_enabled": True,
      "caller_update_kv_cache": True,
      "output_update_kv_cache": True,
      "request_ids": ["request-0"],
      "request_ids_by_dp": {"0": ["request-0"]},
      "requests": [{
          "request_id": "request-0",
          "input_batch_index": 0,
          "dp_rank": 0,
          "local_scheduler_slot": 0,
          "global_row": 0,
          "attention_row": 0,
          "selector_index": 0,
          "selector_range": [0, 1],
          "scheduled_tokens": 1,
          "num_computed_tokens": 2,
          "num_prompt_tokens": 2,
          "num_tokens": 3,
          "expected_seq_len": 3,
          "query_start_range": [0, 1],
          "block_ids": [[7]],
          "metadata_block_ids": [7],
          "logical_blocks": 1,
          "token_ids": [101, 102, 103],
          "token_history_sha256": hashlib.sha256(
              np.asarray([101, 102, 103], dtype="<i8").tobytes()
          ).hexdigest(),
      }],
      "req_id_to_index": {"request-0": 0},
      "scheduled_request_count": 1,
      "padded_rows_per_dp": 4,
      "max_attention_rows_per_dp": 4,
      "kv_caches_spec": [{"shape": [16, 256, 2, 1, 128]}],
      "block_size": 256,
      "observed_max_prefix": 1791,
      "capture_min_prefix": 1788,
      "rpa_block_tuples": {
          "CANON_RPA_D": "128,512,128,512",
          "CANON_RPA_P": "128,512,128,512",
          "CANON_RPA_M": "128,512,128,512",
      },
      "kv_unified": False,
  })
  post_names = {
      "generated_tokens",
      "final_input_positions",
      "final_seq_lens",
      "logprob_token_ids",
      "logprob_values",
      "logprob_ranks",
  }
  post = {
      name: np.arange(4, dtype=np.int32).reshape(1, 4)
      for name in post_names
  }
  _write(args.directory, "post", post, {
      "actual_steps": 1,
      "completed_records": 1,
      "expected_max_records": 1,
  })
  prompt_ids = np.array([[101, 102]], dtype=np.int32)
  prompt_mask = np.array([[True, True]])
  completion_ids = np.array([[103, 104]], dtype=np.int32)
  completion_valid_mask = np.array([[True, True]])
  capsule_arrays = {
      "prompt_ids": prompt_ids,
      "prompt_mask": prompt_mask,
      "completion_ids": completion_ids,
      "completion_valid_mask": completion_valid_mask,
      "action_mask": np.array([[True, True]]),
      "s_decode": np.array([[0.0, 0.1]], dtype=np.float32),
      "s_prefill": np.array([[0.0, 0.2]], dtype=np.float32),
      "t_old": np.array([[0.0, 0.2]], dtype=np.float32),
  }
  capsule_meta = {
      "schema": "p38-frozenlake-mismatch-capsule-v1",
      "arrays": {
          name: {
              "shape": list(value.shape),
              "dtype": str(value.dtype),
              "sha256": hashlib.sha256(
                  np.ascontiguousarray(value).tobytes()
              ).hexdigest(),
          }
          for name, value in capsule_arrays.items()
      },
  }
  args.mismatch_capsule.parent.mkdir(parents=True, exist_ok=True)
  with args.mismatch_capsule.open("xb") as stream:
    np.savez_compressed(
        stream,
        selected_rows=np.array([191], dtype=np.int32),
        metadata_json=np.frombuffer(
            json.dumps(capsule_meta, sort_keys=True).encode(), dtype=np.uint8
        ),
        **capsule_arrays,
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
