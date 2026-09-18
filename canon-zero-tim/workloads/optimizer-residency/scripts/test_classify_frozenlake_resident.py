#!/usr/bin/env python3
"""Tests for the FrozenLake resident-capacity classifier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


_PATH = Path(__file__).with_name("classify_frozenlake_resident.py")
_SPEC = importlib.util.spec_from_file_location(
    "classify_frozenlake_resident", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
classifier = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = classifier
_SPEC.loader.exec_module(classifier)


def _record() -> dict:
  return {
      "verdict": "PASS",
      "microsteps": 4,
      "commits": 1,
      "train_steps_before": 0,
      "train_steps_after": 1,
      "optimizer_placement": "device-resident",
      "optimizer_memory_kinds_before": ["device"],
      "optimizer_memory_kinds_after": ["device"],
      "optimizer_transaction_valid": True,
      "reference_changed_paths": [],
      "accumulator_changed_paths": [],
      "dp_replicas_exact": True,
      "gradient_finite": True,
      "gradient_activity": [True, True, True, True],
      "micro_gradient_norms": [1.0, 2.0, 3.0, 4.0],
      "elapsed_seconds": 12.0,
      "hbm_before": [{"peak_bytes_in_use": 10}],
      "hbm_after_accumulation": [{"peak_bytes_in_use": 20}],
      "hbm_after_commit": [{"peak_bytes_in_use": 30}],
      "commit_evidence": {
          "gradient_finite": True,
          "gradient_nonzero_elements": 2,
          "parameter_changed_elements": 1,
          "optimizer_timing": {
              "optimizer_logical_bytes": 100,
              "optimizer_h2d_seconds": 0.0,
              "adam_commit_seconds": 2.0,
              "optimizer_d2h_seconds": 0.0,
              "optimizer_transaction_seconds": 2.0,
          },
      },
  }


class ClassifyFrozenLakeResidentTest(unittest.TestCase):

  def _classify(self, record: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "update.json"
      path.write_text(json.dumps(record), encoding="utf-8")
      return classifier.classify(path)

  def test_accepts_strict_resident_update(self):
    result = self._classify(_record())
    self.assertEqual(result["verdict"], "PASS")
    self.assertEqual(result["resident"]["peak_hbm_bytes"], 30)

  def test_rejects_host_placement(self):
    record = _record()
    record["optimizer_memory_kinds_after"] = ["pinned_host"]
    result = self._classify(record)
    self.assertEqual(result["verdict"], "FAIL")
    self.assertIn("memory_after", result["reasons"])

  def test_rejects_incomplete_gradient_set(self):
    record = _record()
    record["gradient_activity"][-1] = False
    result = self._classify(record)
    self.assertEqual(result["verdict"], "FAIL")
    self.assertIn("gradient_activity", result["reasons"])


if __name__ == "__main__":
  unittest.main()
