#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src/engine_shims/p38_seam_capture.py"
SPEC = importlib.util.spec_from_file_location("p38_seam_capture", MODULE)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


class SeamCaptureTest(unittest.TestCase):

  def test_hierarchical_checkpoint_contract(self):
    self.assertEqual(
        capture.P38_LAYER_CHECKPOINTS,
        ("layer_input", "layer_output"),
    )
    self.assertEqual(len(capture.P38_SEAM_CHECKPOINTS), 15)
    self.assertEqual(capture.P38_SEAM_CHECKPOINTS[0], "layer_input")
    self.assertEqual(capture.P38_SEAM_CHECKPOINTS[-1], "layer_output")

  def test_selects_only_real_rows_in_the_registered_band(self):
    token_ids = np.arange(2 * 32, dtype=np.int32).reshape(2, 32)
    rows, arrays, requests = capture.select_standard_seam_rows(
        req_ids_dp={0: ("a",), 1: ("b",)},
        scheduled_tokens={"a": 4, "b": 3},
        req_id_to_index={"a": 0, "b": 1},
        num_computed_tokens=np.asarray([8, 14]),
        num_tokens=np.asarray([12, 17]),
        token_ids_cpu=token_ids,
        padded_rows_per_dp=8,
        total_rows=16,
        min_position=10,
        max_position=16,
    )
    np.testing.assert_array_equal(rows, np.asarray([2, 3, 8, 9]))
    np.testing.assert_array_equal(arrays["positions"], [10, 11, 14, 15])
    np.testing.assert_array_equal(arrays["token_ids"], [10, 11, 46, 47])
    self.assertEqual(len(requests), 2)
    self.assertEqual(requests[0]["packed_row_range"], [0, 4])
    self.assertEqual(requests[1]["packed_row_range"], [8, 11])
    self.assertEqual(arrays["token_prefix_sha256"].dtype, np.dtype("S64"))

  def test_mapping_rejects_padding_or_schedule_drift(self):
    kwargs = dict(
        req_ids_dp={0: ("a",)},
        scheduled_tokens={"a": 3},
        req_id_to_index={"a": 0},
        num_computed_tokens=np.asarray([2]),
        num_tokens=np.asarray([5]),
        token_ids_cpu=np.arange(8, dtype=np.int32).reshape(1, 8),
        padded_rows_per_dp=2,
        total_rows=2,
        min_position=0,
        max_position=8,
    )
    with self.assertRaisesRegex(ValueError, "overflow"):
      capture.select_standard_seam_rows(**kwargs)
    kwargs["padded_rows_per_dp"] = 4
    kwargs["total_rows"] = 4
    kwargs["scheduled_tokens"] = {"a": 3, "ghost": 1}
    with self.assertRaisesRegex(ValueError, "incomplete"):
      capture.select_standard_seam_rows(**kwargs)

  def test_record_is_exclusive_bounded_and_self_describing(self):
    with tempfile.TemporaryDirectory() as directory:
      state = {"records": 0, "bytes": 0}
      index, digest = capture.write_seam_record(
          directory,
          state,
          {"fingerprints": np.arange(16, dtype=np.uint32)},
          {"arm": "A", "checkpoint_names": capture.P38_SEAM_CHECKPOINTS},
          1 << 20,
      )
      self.assertEqual(index, 0)
      meta = json.loads(Path(directory, "p38_seam_000000.json").read_text())
      self.assertEqual(meta["npz_sha256"], digest)
      self.assertEqual(meta["schema"], "p38-seam-fingerprint-v1")
      self.assertEqual(state["records"], 1)
      with self.assertRaisesRegex(RuntimeError, "byte bound"):
        capture.write_seam_record(
            directory,
            state,
            {"too_large": np.zeros(1 << 18, dtype=np.uint32)},
            {"arm": "B"},
            state["bytes"] + 8,
        )

  def test_power_of_two_bucket(self):
    self.assertEqual(capture.next_power_of_two(1), 1)
    self.assertEqual(capture.next_power_of_two(17), 32)
    with self.assertRaises(ValueError):
      capture.next_power_of_two(0)


if __name__ == "__main__":
  unittest.main()
