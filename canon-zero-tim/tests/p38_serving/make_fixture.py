#!/usr/bin/env python3
"""Create one valid bounded serving-capture fixture for shell postflight tests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


PREFIX_BOUNDS = (1536, 1664, 1792, 1920, 2048)


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


def _write_kv_observer_pair(
    directory: Path, pair_index: int, token_ids: np.ndarray
) -> None:
  page_size = 256
  logical_pages = (int(token_ids.size) + page_size - 1) // page_size
  observer_pages = 8
  valid_tokens = np.full(logical_pages, page_size, dtype=np.int32)
  valid_tokens[-1] = token_ids.size - (logical_pages - 1) * page_size
  arrays = {
      "aggregates": np.zeros(
          (1, observer_pages, page_size, 4), dtype=np.uint32
      ),
      "samples": np.zeros(
          (1, observer_pages, page_size, 3, 2), dtype=np.uint16
      ),
      "token_ids": np.asarray(token_ids, dtype=np.int32),
      "physical_pages": np.arange(
          7, 7 + logical_pages, dtype=np.int32
      ),
      "padded_global_pages": np.pad(
          np.arange(7, 7 + logical_pages, dtype=np.int32),
          (0, observer_pages - logical_pages),
          mode="edge",
      ),
      "valid_tokens": valid_tokens,
  }
  token_sha = hashlib.sha256(
      np.ascontiguousarray(token_ids, dtype="<i8").tobytes()
  ).hexdigest()
  a_index = pair_index * 2
  for arm, record_index in (("A", a_index), ("B", a_index + 1)):
    base = directory / (
        f"p38_kv_observer_{record_index:04d}_{arm.lower()}"
    )
    npz_path = Path(str(base) + ".npz")
    with npz_path.open("xb") as stream:
      np.savez(stream, **arrays)
    record = {
        "schema": "p38-live-kv-prefix-table-v1",
        "arm": arm,
        "record_index": record_index,
        "request_id": (
            f"decode-{pair_index}" if arm == "A" else f"clean-{pair_index}"
        ),
        "source_a_request_id": f"decode-{pair_index}",
        "source_a_record_index": None if arm == "A" else a_index,
        "diagnostic_round": 0,
        "target_seq_len": int(token_ids.size),
        "token_history_sha256": token_sha,
        "block_size": page_size,
        "logical_pages": logical_pages,
        "observer_pages": observer_pages,
        "layer_count": 1,
        "cache_shape": [32, page_size, 1, 2, 4],
        "cache_dtype": "bfloat16",
        "cache_sharding": "fixture",
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "array_keys": sorted(arrays),
    }
    Path(str(base) + ".json").write_text(
        json.dumps(record, sort_keys=True), encoding="utf-8"
    )


def _write_seam_pair(
    directory: Path, token_ids: np.ndarray, source_position: int
) -> None:
  prefix_sha = hashlib.sha256(
      np.ascontiguousarray(
          token_ids[:source_position + 1], dtype="<i8"
      ).tobytes()
  ).hexdigest().encode()
  for record_index, arm in enumerate(("A", "B")):
    layer = np.zeros((1, 2, 2, 8), dtype=np.uint32)
    if arm == "B":
      layer[0, 1, 1, 3] = 1
    arrays = {
        "row_indices": np.asarray([255], dtype=np.int32),
        "positions": np.asarray([source_position], dtype=np.int32),
        "token_ids": np.asarray([token_ids[source_position]], dtype=np.int32),
        "request_ordinals": np.asarray([0], dtype=np.int32),
        "token_prefix_sha256": np.asarray([prefix_sha], dtype="S64"),
        "layer_fingerprints": layer,
        "final_norm_fingerprints": np.zeros((1, 8), dtype=np.uint32),
    }
    npz_path = directory / f"p38_seam_{record_index:06d}.npz"
    with npz_path.open("xb") as stream:
      np.savez(stream, **arrays)
    record = {
        "schema": "p38-seam-fingerprint-v1",
        "record_index": record_index,
        "arm": arm,
        "diagnostic_round": 0,
        "observer_mode": "layer",
        "checkpoint_names": ["layer_input", "layer_output"],
        "layer_indices": [0, 1],
        "array_keys": sorted(arrays),
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
    }
    (directory / f"p38_seam_{record_index:06d}.json").write_text(
        json.dumps(record, sort_keys=True), encoding="utf-8"
    )


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--directory", required=True, type=Path)
  parser.add_argument("--mismatch-capsule", required=True, type=Path)
  parser.add_argument("--omit-request-journal", action="store_true")
  parser.add_argument("--omit-incident-ledger", action="store_true")
  parser.add_argument("--seam", action="store_true")
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
      "dp_size": 1,
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
  for seq, observed_prefix in enumerate((1600, 1700, 1850, 1980)):
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
  journal_prefix = 1600
  journal_tokens = full_history[:journal_prefix + 1]
  journal_pages = list(range(7, 14))
  journal = {
      "schema": "p38-request-journal-v1",
      "call_index": 1,
      "program_path": "standard",
      "request_id": "request-0",
      "request_index": 0,
      "dp_rank": 0,
      "local_scheduler_slot": 0,
      "num_computed_tokens": journal_prefix,
      "num_prompt_tokens": 2,
      "num_tokens": len(journal_tokens),
      "stratum_index": 0,
      "stratum": [1536, 1664],
      "block_size": 256,
      "logical_blocks": len(journal_pages),
      "physical_pages": journal_pages,
      "page_generations": [
          {
              "physical_page": physical_page,
              "logical_page": logical_page,
              "observation_generation": 0,
              "previous_observed_request_id": None,
              "previous_observed_logical_page": None,
              "previous_observed_call": None,
              "observed_owner_changed": True,
          }
          for logical_page, physical_page in enumerate(journal_pages)
      ],
      "token_ids": journal_tokens,
      "token_history_sha256": hashlib.sha256(
          np.asarray(journal_tokens, dtype="<i8").tobytes()
      ).hexdigest(),
      "scheduled_request_count": 1,
      "co_batch_request_ids": ["request-0"],
      "one_token_decode_request_count": 1,
      "one_token_decode_request_ids": ["request-0"],
  }
  if not args.omit_request_journal:
    (args.directory / "p38_request_journal.jsonl").write_text(
        json.dumps(journal, sort_keys=True) + "\n", encoding="utf-8"
    )
  incident_request = {
      "request_id": "request-0",
      "request_index": 0,
      "dp_rank": 0,
      "local_scheduler_slot": 0,
      "num_computed_tokens": journal_prefix,
      "num_tokens": len(journal_tokens),
      "token_history_sha256": journal["token_history_sha256"],
      "block_size": 256,
      "logical_blocks": len(journal_pages),
      "physical_pages": journal_pages,
      "page_generations": [
          {
              "physical_page": physical_page,
              "logical_page": logical_page,
              "observation_generation": 0,
              "observed_request_id": "request-0",
              "observed_logical_page": logical_page,
          }
          for logical_page, physical_page in enumerate(journal_pages)
      ],
  }
  incident = {
      "schema": "p38-incident-ledger-v1",
      "call_index": 1,
      "diagnostic_round": 0,
      "program_path": "standard",
      "scheduled_request_count": 1,
      "co_batch_request_ids": ["request-0"],
      "one_token_decode_request_count": 1,
      "incident_min_prefix": 1400,
      "incident_max_prefix": 3072,
      "requests": [incident_request],
  }
  if not args.omit_incident_ledger:
    (args.directory / "p38_incident_ledger.jsonl").write_text(
        json.dumps(incident, sort_keys=True) + "\n", encoding="utf-8"
    )
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
  mismatch_position = journal_prefix - int(prompt_mask.sum())
  capsule_arrays["s_prefill"][0, mismatch_position] = np.nextafter(
      np.float32(0.0), np.float32(1.0)
  )
  capsule_meta = {
      "schema": "p38-frozenlake-mismatch-capsule-v1",
      "diagnostic_round": 0,
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
  observer_tokens = np.asarray(full_history[:1601], dtype=np.int32)
  if args.seam:
    _write_seam_pair(
        args.directory, np.asarray(full_history, dtype=np.int32),
        journal_prefix - 1,
    )
  else:
    for pair_index in range(3):
      _write_kv_observer_pair(args.directory, pair_index, observer_tokens)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
