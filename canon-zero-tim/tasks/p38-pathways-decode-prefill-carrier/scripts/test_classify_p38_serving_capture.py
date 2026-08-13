#!/usr/bin/env python3
"""Negative controls for the P38 serving-capture classifier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("classify_p38_serving_capture.py")
SPEC = importlib.util.spec_from_file_location("classify_p38_serving_capture", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


PREFIX_BOUNDS = (0, 4, 8, 12, 16)


def _write_stage(
    directory: Path,
    stage: str,
    arrays: dict[str, np.ndarray],
    meta: dict,
    seq: int = 0,
) -> None:
  base = directory / f"p38_serving_{seq:04d}_{stage}"
  npz_path = Path(str(base) + ".npz")
  json_path = Path(str(base) + ".json")
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
  json_path.write_text(json.dumps(record), encoding="utf-8")


def _valid_directory() -> tempfile.TemporaryDirectory:
  holder = tempfile.TemporaryDirectory()
  directory = Path(holder.name)
  pre = {name: np.arange(4, dtype=np.int32) for name in MODULE.PRE_ARRAYS}
  pre["input_positions"] = np.array([0, 2, 0, 0], dtype=np.int32)
  pre["active_mask"] = np.array([False, True, False, False])
  pre["md_block_tables"] = np.array(
      [[0, 0, 0, 0], [7, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      dtype=np.int32,
  )
  pre["md_seq_lens"] = np.array([0, 3, 0, 0], dtype=np.int32)
  pre["md_query_start_loc"] = np.array([0, 0, 1, 1, 1], dtype=np.int32)
  pre["tokens_indices_selector"] = np.array([1], dtype=np.int32)
  pre["sampling_leaf_0000"] = np.arange(4, dtype=np.float32)
  post = {name: np.arange(4, dtype=np.int32).reshape(1, 4) for name in MODULE.POST_ARRAYS}
  _write_stage(directory, "pre", pre, {
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
          "local_scheduler_slot": 1,
          "packed_token_offset": 1,
          "global_row": 1,
          "attention_row": 1,
          "selector_index": 0,
          "selector_range": [1, 2],
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
          "token_history_sha256": MODULE._token_history_sha256(
              [101, 102, 103]
          ),
      }],
      "req_id_to_index": {"request-0": 0},
      "scheduled_request_count": 1,
      "dp_size": 1,
      "padded_rows_per_dp": 4,
      "max_attention_rows_per_dp": 4,
      "kv_caches_spec": [{"shape": [16, 256, 2, 1, 128]}],
      "block_size": 256,
      "observed_max_prefix": 2,
      "observed_min_prefix": 2,
      "capture_min_prefix": 0,
      "capture_prefix_bounds": list(PREFIX_BOUNDS),
      "capture_stratum_index": 0,
      "capture_stratum": [0, 4],
      "capture_anchor_request_id": "request-0",
      "capture_anchor_prefix": 2,
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
  })
  _write_stage(directory, "post", post, {
      "actual_steps": 1,
      "completed_records": 1,
      "expected_max_records": 4,
      "program_path": "standard",
  })
  for seq, observed_prefix in enumerate((5, 9, 13), start=1):
    pre_meta = json.loads(
        (directory / "p38_serving_0000_pre.json").read_text()
    )["meta"]
    token_ids = [101, 102, *range(103, 103 + observed_prefix - 1)]
    pre_meta["observed_max_prefix"] = observed_prefix
    pre_meta["capture_anchor_prefix"] = observed_prefix
    pre_meta["requests"][0]["num_computed_tokens"] = observed_prefix
    pre_meta["requests"][0]["expected_seq_len"] = observed_prefix + 1
    pre_meta["requests"][0]["num_tokens"] = observed_prefix + 1
    pre_meta["requests"][0]["token_ids"] = token_ids
    pre_meta["requests"][0]["token_history_sha256"] = (
        MODULE._token_history_sha256(token_ids)
    )
    pre_meta["capture_stratum_index"] = seq
    pre_meta["capture_stratum"] = [PREFIX_BOUNDS[seq], PREFIX_BOUNDS[seq + 1]]
    record_pre = {name: np.array(value, copy=True) for name, value in pre.items()}
    record_pre["input_positions"][1] = observed_prefix
    record_pre["md_seq_lens"][1] = observed_prefix + 1
    _write_stage(directory, "pre", record_pre, pre_meta, seq)
    _write_stage(directory, "post", post, {
        "actual_steps": 1,
        "completed_records": seq + 1,
        "expected_max_records": 4,
        "program_path": "standard",
    }, seq)
  capsule_arrays = {
      "prompt_ids": np.array([[101, 102]], dtype=np.int32),
      "prompt_mask": np.array([[True, True]]),
      "completion_ids": np.arange(103, 115, dtype=np.int32)[None, :],
      "completion_valid_mask": np.ones((1, 12), dtype=np.bool_),
      "action_mask": np.ones((1, 12), dtype=np.bool_),
      "s_decode": np.arange(12, dtype=np.float32)[None, :],
      "s_prefill": np.arange(12, dtype=np.float32)[None, :],
      "t_old": np.arange(12, dtype=np.float32)[None, :],
  }
  metadata = {
      "schema": "p38-frozenlake-mismatch-capsule-v1",
      "arrays": {
          name: {
              "shape": list(value.shape),
              "dtype": str(value.dtype),
              "sha256": MODULE._array_sha256(value),
          }
          for name, value in capsule_arrays.items()
      },
  }
  with (directory / "mismatch.npz").open("xb") as stream:
    np.savez_compressed(
        stream,
        selected_rows=np.array([191], dtype=np.int32),
        metadata_json=np.frombuffer(
            json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8
        ),
        **capsule_arrays,
    )
  journal = {
      "schema": "p38-request-journal-v1",
      "call_index": 1,
      "program_path": "standard",
      "request_id": "request-0",
      "request_index": 0,
      "dp_rank": 0,
      "local_scheduler_slot": 1,
      "num_computed_tokens": 2,
      "num_prompt_tokens": 2,
      "num_tokens": 3,
      "stratum_index": 0,
      "stratum": [0, 4],
      "block_size": 256,
      "logical_blocks": 1,
      "physical_pages": [7],
      "page_generations": [{
          "physical_page": 7,
          "logical_page": 0,
          "observation_generation": 0,
          "previous_observed_request_id": None,
          "previous_observed_logical_page": None,
          "previous_observed_call": None,
          "observed_owner_changed": True,
      }],
      "token_ids": [101, 102, 103],
      "token_history_sha256": MODULE._token_history_sha256(
          [101, 102, 103]
      ),
      "scheduled_request_count": 1,
      "co_batch_request_ids": ["request-0"],
      "one_token_decode_request_count": 1,
      "one_token_decode_request_ids": ["request-0"],
  }
  (directory / "p38_request_journal.jsonl").write_text(
      json.dumps(journal, sort_keys=True) + "\n"
  )
  return holder


def _classify(holder: tempfile.TemporaryDirectory, expected_records: int = 4):
  directory = Path(holder.name)
  bounds = (
      PREFIX_BOUNDS
      if expected_records == 4
      else tuple(range(expected_records + 1))
  )
  return MODULE.classify(
      directory,
      expected_records,
      directory / "mismatch.npz",
      bounds,
      "standard",
  )


class ClassifyServingCaptureTest(unittest.TestCase):

  def test_accepts_complete_standard_record(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    report = _classify(holder)
    self.assertEqual(report["verdict"], "PASS")
    self.assertEqual(
        report["records"][0]["mismatch_join"]["source_row"], 191
    )
    self.assertEqual(report["joined_source_rows"], [191])
    self.assertEqual(report["request_journal_joined_source_rows"], [191])

  def test_rejects_missing_request_journal(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    (Path(holder.name) / "p38_request_journal.jsonl").unlink()
    with self.assertRaisesRegex(MODULE.CaptureError, "journal is absent"):
      _classify(holder)

  def test_accepts_flattened_production_block_table(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    directory = Path(holder.name)
    json_path = directory / "p38_serving_0000_pre.json"
    npz_path = directory / "p38_serving_0000_pre.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
      arrays = {name: np.array(archive[name]) for name in archive.files}
    arrays["md_block_tables"] = arrays["md_block_tables"].reshape(-1)
    npz_path.unlink()
    with npz_path.open("xb") as stream:
      np.savez(stream, **arrays)
    record = json.loads(json_path.read_text())
    record["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    json_path.write_text(json.dumps(record))
    self.assertEqual(_classify(holder)["verdict"], "PASS")

  def test_accepts_one_unique_request_per_source_row(self):
    prompt_ids = np.array([[10, 11], [20, 21]], dtype=np.int32)
    completion_ids = np.array([[12, 13], [22, 23]], dtype=np.int32)
    capsule = {
        "arrays": {
            "selected_rows": np.array([199, 206], dtype=np.int32),
            "prompt_ids": prompt_ids,
            "prompt_mask": np.ones_like(prompt_ids, dtype=np.bool_),
            "completion_ids": completion_ids,
            "completion_valid_mask": np.ones_like(
                completion_ids, dtype=np.bool_
            ),
        }
    }
    requests = [
        {
            "request_id": "request-199",
            "token_ids": [10, 11, 12],
            "token_history_sha256": MODULE._token_history_sha256(
                [10, 11, 12]
            ),
        },
        {
            "request_id": "request-206",
            "token_ids": [20, 21, 22],
            "token_history_sha256": MODULE._token_history_sha256(
                [20, 21, 22]
            ),
        },
    ]
    joins = MODULE._join_mismatch_capsule(requests, capsule, 2)
    self.assertEqual(
        [(item["source_row"], item["request_id"]) for item in joins],
        [(199, "request-199"), (206, "request-206")],
    )

  def test_journal_accepts_multiple_turn_requests_for_one_source_row(self):
    capsule = {
        "arrays": {
            "selected_rows": np.array([199], dtype=np.int32),
            "prompt_ids": np.array([[10, 11]], dtype=np.int32),
            "prompt_mask": np.array([[True, True]]),
            "completion_ids": np.array([[12, 13, 14]], dtype=np.int32),
            "completion_valid_mask": np.array([[True, True, True]]),
        }
    }
    common = {
        "source_row": 199,
        "stratum_index": 0,
        "dp_rank": 0,
        "local_scheduler_slot": 0,
        "physical_pages": [7],
        "page_generations": [],
        "scheduled_request_count": 1,
    }
    records = [
        {
            **common,
            "request_id": "turn-3",
            "token_ids": [10, 11, 12],
            "num_computed_tokens": 2,
            "token_history_sha256": MODULE._token_history_sha256(
                [10, 11, 12]
            ),
        },
        {
            **common,
            "request_id": "turn-4",
            "token_ids": [10, 11, 12, 13],
            "num_computed_tokens": 3,
            "token_history_sha256": MODULE._token_history_sha256(
                [10, 11, 12, 13]
            ),
        },
    ]
    joins = MODULE._join_journal_to_capsule(records, capsule)
    self.assertEqual(
        [(item["source_row"], item["request_id"]) for item in joins],
        [(199, "turn-3"), (199, "turn-4")],
    )

  def test_rejects_missing_post_record(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    Path(holder.name, "p38_serving_0000_post.json").unlink()
    with self.assertRaisesRegex(MODULE.CaptureError, "post records"):
      _classify(holder)

  def test_rejects_corrupt_npz(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.npz")
    path.write_bytes(path.read_bytes() + b"corrupt")
    with self.assertRaisesRegex(MODULE.CaptureError, "SHA mismatch"):
      _classify(holder)

  def test_rejects_missing_five_times_storage_guard(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["storage_guard"]["multiplier"] = 4
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "five-times storage guard"):
      _classify(holder)

  def test_rejects_path_configuration_contradiction(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["continue_decode_enabled"] = True
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "contradicts its path"):
      _classify(holder)

  def test_rejects_wrong_program_path(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["program_path"] = "continue_decode"
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "wrong program path"):
      _classify(holder)

  def test_rejects_missing_implementation_identity(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"].pop("implementation_identity")
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "no implementation identity"):
      _classify(holder)

  def test_rejects_duplicate_capture_stratum(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0001_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["capture_stratum_index"] = 0
    record["meta"]["capture_stratum"] = [0, 4]
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "duplicates capture stratum"):
      _classify(holder)

  def test_rejects_prefix_outside_capture_stratum(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["capture_stratum_index"] = 3
    record["meta"]["capture_stratum"] = [12, 16]
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "outside its capture stratum"):
      _classify(holder)

  def test_rejects_prefix_bound_drift(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["capture_prefix_bounds"] = [1536, 1792, 2048]
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "prefix bounds drifted"):
      _classify(holder)

  def test_rejects_source_commit_identity_drift(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0001_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["env"]["CANON_EXPECT_COMMIT"] = "2" * 40
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "source commit identity drifted"):
      _classify(holder)

  def test_rejects_callable_identity_drift(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0001_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["implementation_identity"]["model_fn"]["chain"][0][
        "qualname"
    ] = "other_model_fn"
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "implementation identity drifted"):
      _classify(holder)

  def test_accepts_unified_output_contract(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["kv_unified"] = True
    record["meta"]["output_update_kv_cache"] = False
    path.write_text(json.dumps(record))
    self.assertEqual(_classify(holder)["verdict"], "PASS")

  def test_rejects_inconsistent_unified_output_contract(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["kv_unified"] = True
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "inconsistent output"):
      _classify(holder)

  def test_rejects_missing_physical_pages(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["requests"][0]["block_ids"] = []
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "physical page IDs"):
      _classify(holder)

  def test_rejects_missing_required_array(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    directory = Path(holder.name)
    json_path = directory / "p38_serving_0000_pre.json"
    npz_path = directory / "p38_serving_0000_pre.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
      arrays = {name: np.array(archive[name]) for name in archive.files if name != "rng"}
    npz_path.unlink()
    with npz_path.open("xb") as stream:
      np.savez(stream, **arrays)
    record = json.loads(json_path.read_text())
    record["arrays"] = sorted(arrays)
    record["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    json_path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "missing required pre arrays"):
      _classify(holder)

  def test_rejects_missing_sampling_metadata(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    directory = Path(holder.name)
    json_path = directory / "p38_serving_0000_pre.json"
    npz_path = directory / "p38_serving_0000_pre.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
      arrays = {
          name: np.array(archive[name])
          for name in archive.files
          if not name.startswith("sampling_leaf_")
      }
    npz_path.unlink()
    with npz_path.open("xb") as stream:
      np.savez(stream, **arrays)
    record = json.loads(json_path.read_text())
    record["arrays"] = sorted(arrays)
    record["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    json_path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "no sampling-metadata leaves"):
      _classify(holder)

  def test_rejects_record_count_mismatch(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    with self.assertRaisesRegex(MODULE.CaptureError, "expected 5 pre records"):
      _classify(holder, 5)

  def test_rejects_unscheduled_request(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["requests"][0]["scheduled_tokens"] = 0
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "not a one-token decode"):
      _classify(holder)

  def test_rejects_global_row_mapping_drift(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["requests"][0]["global_row"] = 0
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "global row mismatch"):
      _classify(holder)

  def test_rejects_physical_page_mapping_drift(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["requests"][0]["metadata_block_ids"] = [11]
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "metadata page mismatch"):
      _classify(holder)

  def test_rejects_missing_mismatch_join(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    for seq in range(4):
      path = Path(holder.name, f"p38_serving_{seq:04d}_pre.json")
      record = json.loads(path.read_text())
      original = record["meta"]["requests"][0]["token_ids"]
      replacement = list(range(999, 999 + len(original)))
      record["meta"]["requests"][0]["token_ids"] = replacement
      record["meta"]["requests"][0]["token_history_sha256"] = (
          MODULE._token_history_sha256(replacement)
      )
      path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "no serving record joins"):
      _classify(holder)

  def test_rejects_missing_required_mismatch_capsule(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    capsule = Path(holder.name, "mismatch.npz")
    capsule.unlink()
    with self.assertRaisesRegex(MODULE.CaptureError, "capsule is absent"):
      MODULE.classify(
          Path(holder.name), 4, capsule, PREFIX_BOUNDS, "standard"
      )

  def test_allows_missing_capsule_when_join_is_not_required(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    capsule = Path(holder.name, "mismatch.npz")
    capsule.unlink()
    report = MODULE.classify(
        Path(holder.name),
        4,
        capsule,
        PREFIX_BOUNDS,
        "standard",
        require_mismatch_join=False,
    )
    self.assertEqual(report["verdict"], "PASS")
    self.assertIsNone(report["records"][0]["mismatch_join"])

  def test_allows_nonmatching_capsule_when_join_is_not_required(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["requests"][0]["token_ids"] = [999, 998, 997]
    record["meta"]["requests"][0]["token_history_sha256"] = (
        MODULE._token_history_sha256([999, 998, 997])
    )
    path.write_text(json.dumps(record))
    report = MODULE.classify(
        Path(holder.name),
        4,
        Path(holder.name, "mismatch.npz"),
        PREFIX_BOUNDS,
        "standard",
        require_mismatch_join=False,
    )
    self.assertEqual(report["verdict"], "PASS")
    self.assertIsNone(report["records"][0]["mismatch_join"])

  def test_rejects_ambiguous_mismatch_join(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "mismatch.npz")
    with np.load(path, allow_pickle=False) as archive:
      arrays = {name: np.array(archive[name]) for name in archive.files}
    metadata = json.loads(arrays.pop("metadata_json").tobytes().decode())
    row_arrays = {
        name: np.concatenate((value, value), axis=0)
        for name, value in arrays.items()
        if name != "selected_rows"
    }
    metadata["arrays"] = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": MODULE._array_sha256(value),
        }
        for name, value in row_arrays.items()
    }
    path.unlink()
    with path.open("xb") as stream:
      np.savez_compressed(
          stream,
          selected_rows=np.array([191, 199], dtype=np.int32),
          metadata_json=np.frombuffer(
              json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8
          ),
          **row_arrays,
      )
    with self.assertRaisesRegex(
        MODULE.CaptureError, "ambiguously joins source rows"
    ):
      _classify(holder)


if __name__ == "__main__":
  unittest.main()
