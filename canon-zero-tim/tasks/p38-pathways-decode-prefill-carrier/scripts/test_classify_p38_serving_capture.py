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
      "schema_version": 1,
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
  pre["active_mask"] = np.ones(4, dtype=np.bool_)
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
          "block_ids": [[7, 11]],
          "token_ids": [101, 102, 103],
      }],
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
  return holder


class ClassifyServingCaptureTest(unittest.TestCase):

  def test_accepts_complete_continue_decode_record(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    report = MODULE.classify(Path(holder.name), 1)
    self.assertEqual(report["verdict"], "PASS")

  def test_rejects_missing_post_record(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    Path(holder.name, "p38_serving_0000_post.json").unlink()
    with self.assertRaisesRegex(MODULE.CaptureError, "post records"):
      MODULE.classify(Path(holder.name), 1)

  def test_rejects_corrupt_npz(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.npz")
    path.write_bytes(path.read_bytes() + b"corrupt")
    with self.assertRaisesRegex(MODULE.CaptureError, "SHA mismatch"):
      MODULE.classify(Path(holder.name), 1)

  def test_rejects_non_continue_decode_record(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["continue_decode_enabled"] = False
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "did not capture continue-decode"):
      MODULE.classify(Path(holder.name), 1)

  def test_accepts_unified_output_contract(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["kv_unified"] = True
    record["meta"]["output_update_kv_cache"] = False
    path.write_text(json.dumps(record))
    self.assertEqual(MODULE.classify(Path(holder.name), 1)["verdict"], "PASS")

  def test_rejects_inconsistent_unified_output_contract(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["kv_unified"] = True
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "inconsistent output"):
      MODULE.classify(Path(holder.name), 1)

  def test_rejects_missing_physical_pages(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    path = Path(holder.name, "p38_serving_0000_pre.json")
    record = json.loads(path.read_text())
    record["meta"]["requests"][0]["block_ids"] = []
    path.write_text(json.dumps(record))
    with self.assertRaisesRegex(MODULE.CaptureError, "without physical page IDs"):
      MODULE.classify(Path(holder.name), 1)

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
      MODULE.classify(directory, 1)

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
      MODULE.classify(directory, 1)

  def test_rejects_record_count_mismatch(self):
    holder = _valid_directory()
    self.addCleanup(holder.cleanup)
    with self.assertRaisesRegex(MODULE.CaptureError, "expected 2 pre records"):
      MODULE.classify(Path(holder.name), 2)


if __name__ == "__main__":
  unittest.main()
