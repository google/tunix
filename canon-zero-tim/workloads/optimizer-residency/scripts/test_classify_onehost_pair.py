#!/usr/bin/env python3
"""Tests for the bounded optimizer-placement pair classifier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


_PATH = Path(__file__).with_name("classify_onehost_pair.py")
_SPEC = importlib.util.spec_from_file_location("classify_onehost_pair", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
classifier = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = classifier
_SPEC.loader.exec_module(classifier)


def _record(placement: str) -> dict:
  kind = "device" if placement == "device-resident" else "pinned_host"
  fingerprint = {"model": {"leaves": {"x": {"sha256": "a"}}}}
  return {
      "verdict": "PASS",
      "commits": 1,
      "train_steps_before": 0,
      "train_steps_after": 1,
      "optimizer_placement": placement,
      "optimizer_memory_kinds_before": [kind],
      "optimizer_memory_kinds_after": [kind],
      "optimizer_transaction_valid": True,
      "reference_changed_paths": [],
      "accumulator_changed_paths": [],
      "dp_replicas_exact": True,
      "gradient_finite": True,
      "micro_gradient_norms": [1.0],
      "alignment_hashes": [{"T_current": "a"}],
      "state_fingerprints_before": fingerprint,
      "state_fingerprints_after": fingerprint,
      "elapsed_seconds": 10.0,
      "hbm_before": [{"peak_bytes_in_use": 10}],
      "hbm_after_accumulation": [{"peak_bytes_in_use": 20}],
      "hbm_after_commit": [{"peak_bytes_in_use": 30}],
      "commit_evidence": {
          "gradient_finite": True,
          "gradient_nonzero_elements": 2,
          "parameter_changed_elements": 1,
          "optimizer_timing": {
              "optimizer_logical_bytes": 100,
              "optimizer_h2d_seconds": 1.0 if kind == "pinned_host" else 0.0,
              "adam_commit_seconds": 2.0,
              "optimizer_d2h_seconds": 1.0 if kind == "pinned_host" else 0.0,
              "optimizer_transaction_seconds": (
                  4.0 if kind == "pinned_host" else 2.0
              ),
          },
      },
  }


class ClassifyOneHostPairTest(unittest.TestCase):

  def _write(self, path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")

  def test_accepts_equal_update_pair(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      offload = root / "offload.json"
      resident = root / "resident.json"
      self._write(offload, _record("pinned-host-offload"))
      self._write(resident, _record("device-resident"))
      result = classifier.classify(offload, resident)
      self.assertEqual(result["verdict"], "PASS")
      self.assertTrue(result["bitwise_equal"])
      self.assertEqual(result["optimizer_transaction_speedup"], 2.0)

  def test_rejects_resident_arm_on_host(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      offload = root / "offload.json"
      resident = root / "resident.json"
      bad = _record("device-resident")
      bad["optimizer_memory_kinds_after"] = ["pinned_host"]
      self._write(offload, _record("pinned-host-offload"))
      self._write(resident, bad)
      result = classifier.classify(offload, resident)
      self.assertEqual(result["verdict"], "FAIL")
      self.assertIn("resident.memory_after", result["reasons"])


if __name__ == "__main__":
  unittest.main()
