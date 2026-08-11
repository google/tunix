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


def _write_stage(directory: Path, stage: str, arrays: dict[str, np.ndarray], meta: dict) -> None:
  base = directory / f"p38_serving_0000_{stage}"
  npz_path = Path(str(base) + ".npz")
  json_path = Path(str(base) + ".json")
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
      "continue_decode_enabled": True,
      "caller_update_kv_cache": True,
      "output_update_kv_cache": True,
      "request_ids": ["request-0"],
      "request_ids_by_dp": {"0": ["request-0"]},
      "requests": [{
          "request_id": "request-0",
          "input_batch_index": 0,
          "dp_rank": 0,
          "local_scheduler_slot": 1,
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
  _write_stage(directory, "post", post, {
      "actual_steps": 1,
      "completed_records": 1,
      "expected_max_records": 1,
  })
  capsule_arrays = {
      "prompt_ids": np.array([[101, 102]], dtype=np.int32),
      "prompt_mask": np.array([[True, True]]),
      "completion_ids": np.array([[103, 104]], dtype=np.int32),
      "completion_valid_mask": np.array([[True, True]]),
      "action_mask": np.array([[True, True]]),
      "s_decode": np.array([[0.0, 0.1]], dtype=np.float32),
      "s_prefill": np.array([[0.0, 0.2]], dtype=np.float32),
      "t_old": np.array([[0.0, 0.2]], dtype=np.float32),
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
  return holder


def _classify(holder: tempfile.TemporaryDirectory, expected_records: int = 1):
  directory = Path(holder.name)
  return MODULE.classify(
      directory, expected_records, directory / "mismatch.npz"
  )


class ClassifyServingCaptureTest(unittest.TestCase):

  def test_accepts_complete_continue_decode_record(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    report = _classify(holder)
    self.assertEqual(report["verdict"], "PASS")
    self.assertEqual(
        report["records"][0]["mismatch_join"]["source_row"], 191
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

  def test_rejects_non_continue_decode_record(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["continue_decode_enabled"] = False
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "did not capture continue-decode"):
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
    with self.assertRaisesRegex(MODULE.CaptureError, "expected 2 pre records"):
      _classify(holder, 2)

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
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["requests"][0]["token_ids"] = [999, 998, 997]
    record["meta"]["requests"][0]["token_history_sha256"] = (
        MODULE._token_history_sha256([999, 998, 997])
    )
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "found 0"):
      _classify(holder)

  def test_rejects_missing_required_mismatch_capsule(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    capsule = Path(holder.name, "mismatch.npz")
    capsule.unlink()
    with self.assertRaisesRegex(MODULE.CaptureError, "capsule is absent"):
      MODULE.classify(Path(holder.name), 1, capsule)

  def test_allows_missing_capsule_when_join_is_not_required(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    capsule = Path(holder.name, "mismatch.npz")
    capsule.unlink()
    report = MODULE.classify(
        Path(holder.name),
        1,
        capsule,
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
        1,
        Path(holder.name, "mismatch.npz"),
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
    with self.assertRaisesRegex(MODULE.CaptureError, "found 2"):
      _classify(holder)


if __name__ == "__main__":
  unittest.main()
