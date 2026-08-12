#!/usr/bin/env python3
"""Create one valid bounded serving-capture fixture for shell postflight tests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


PREFIX_BOUNDS = (1536, 1792, 2048, 2304, 2560)


def _write(
    directory: Path, stage: str, arrays: dict, meta: dict, seq: int
) -> None:
  base = directory / f"p38_serving_{seq:04d}_{stage}"
  npz_path = Path(str(base) + ".npz")
  with npz_path.open("xb") as stream:
    np.savez(stream, **arrays)
  record = {
      "schema_version": 2,
      "stage": stage,
      "seq": seq,
      "arrays": sorted(arrays),
      "describe": {},
      "meta": meta,
      "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
      "storage_guard": {
          "payload_bytes": 1,
          "estimated_total_bytes": 4,
          "free_bytes": 100,
          "required_free_bytes": 20,
          "multiplier": 5,
      },
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
  pre["input_positions"] = np.array([0, 0, 0, 0], dtype=np.int32)
  pre["active_mask"] = np.array([True, False, False, False])
  pre["md_block_tables"] = np.zeros((4, 16), dtype=np.int32)
  pre["md_seq_lens"] = np.array([0, 0, 0, 0], dtype=np.int32)
  pre["md_query_start_loc"] = np.array([0, 1, 1, 1, 1], dtype=np.int32)
  pre["tokens_indices_selector"] = np.array([0], dtype=np.int32)
  pre["sampling_leaf_0000"] = np.arange(4, dtype=np.float32)
  pre_meta = {
      "continue_decode_enabled": False,
      "program_path": "standard",
      "caller_update_kv_cache": True,
      "output_update_kv_cache": True,
      "request_ids": ["request-0"],
      "request_ids_by_dp": {"0": ["request-0"]},
      "requests": [{
          "request_id": "request-0",
          "input_batch_index": 0,
          "dp_rank": 0,
          "local_scheduler_slot": 0,
          "packed_token_offset": 0,
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
      "observed_max_prefix": 1700,
      "observed_min_prefix": 1700,
      "capture_min_prefix": 1536,
      "capture_prefix_bounds": list(PREFIX_BOUNDS),
      "capture_stratum_index": 0,
      "capture_stratum": [1536, 1792],
      "capture_anchor_request_id": "request-0",
      "capture_anchor_prefix": 1700,
      "implementation_identity": {
          "runner_class": {"module": "runner", "qualname": "Runner"},
          **{
              name: {
                  "chain": [{
                      "type_module": "builtins",
                      "type_name": "function",
                      "module": "fixture",
                      "qualname": name,
                  }],
                  "source_file": None,
                  "source_sha256": None,
              }
              for name in (
                  "continue_decode", "execute_model", "model_fn",
                  "compute_logits_fn", "sample_fn"
              )
          },
      },
      "env": {"CANON_EXPECT_COMMIT": "1" * 40},
      "rpa_block_tuples": {
          "CANON_RPA_D": "128,512,128,512",
          "CANON_RPA_P": "128,512,128,512",
          "CANON_RPA_M": "128,512,128,512",
      },
      "kv_unified": False,
  }
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
  full_history = [101, 102, *range(103, 2662)]
  for seq, observed_prefix in enumerate((1700, 1800, 2100, 2400)):
    record_meta = copy.deepcopy(pre_meta)
    record_pre = {
        name: np.array(value, copy=True) for name, value in pre.items()
    }
    token_ids = full_history[:observed_prefix + 1]
    expected_seq_len = observed_prefix + 1
    logical_blocks = (expected_seq_len + 255) // 256
    pages = list(range(7, 7 + logical_blocks))
    record_pre["input_positions"][0] = observed_prefix
    record_pre["md_seq_lens"][0] = expected_seq_len
    record_pre["md_block_tables"][0, :logical_blocks] = pages
    record_meta["observed_max_prefix"] = observed_prefix
    record_meta["observed_min_prefix"] = observed_prefix
    record_meta["capture_anchor_prefix"] = observed_prefix
    record_meta["capture_stratum_index"] = seq
    record_meta["capture_stratum"] = [
        PREFIX_BOUNDS[seq], PREFIX_BOUNDS[seq + 1]
    ]
    request = record_meta["requests"][0]
    request["num_computed_tokens"] = observed_prefix
    request["num_tokens"] = expected_seq_len
    request["expected_seq_len"] = expected_seq_len
    request["token_ids"] = token_ids
    request["token_history_sha256"] = hashlib.sha256(
        np.asarray(token_ids, dtype="<i8").tobytes()
    ).hexdigest()
    request["block_ids"] = [pages]
    request["metadata_block_ids"] = pages
    request["logical_blocks"] = logical_blocks
    _write(args.directory, "pre", record_pre, record_meta, seq)
    _write(args.directory, "post", post, {
        "actual_steps": 1,
        "completed_records": seq + 1,
        "expected_max_records": 4,
        "program_path": "standard",
    }, seq)
  prompt_ids = np.array([[101, 102]], dtype=np.int32)
  prompt_mask = np.array([[True, True]])
  completion_ids = np.asarray(full_history[2:], dtype=np.int32)[None, :]
  completion_valid_mask = np.ones_like(completion_ids, dtype=np.bool_)
  capsule_arrays = {
      "prompt_ids": prompt_ids,
      "prompt_mask": prompt_mask,
      "completion_ids": completion_ids,
      "completion_valid_mask": completion_valid_mask,
      "action_mask": np.ones_like(completion_ids, dtype=np.bool_),
      "s_decode": np.zeros_like(completion_ids, dtype=np.float32),
      "s_prefill": np.zeros_like(completion_ids, dtype=np.float32),
      "t_old": np.zeros_like(completion_ids, dtype=np.float32),
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
