#!/usr/bin/env python3
"""Negative and positive controls for the P38 replay classifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


_SCRIPT = Path(__file__).with_name("classify_p38_frozenlake_replay.py")
_SPEC = importlib.util.spec_from_file_location("classify_p38_replay", _SCRIPT)
classifier = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = classifier
_SPEC.loader.exec_module(classifier)


def _report():
  exact = {"exact": True}
  return {
      "schema": "p38-frozenlake-causal-replay-v1",
      "measurement_status": "COMPLETE",
      "classification": "LOCAL_CARRIER_NOT_REPRODUCED",
      "no_backward": True,
      "no_optimizer": True,
      "weight_attestation": {"equal": True},
      "geometry": {"prefix_cache": False},
      "schedules": [
          {"arm": "R0", "provenance": "mask-derived-v1"},
          {"arm": "R1", "provenance": "mask-derived-v1"},
          {"arm": "REF", "provenance": "canonical-fixed-chunk-v1"},
      ],
      "repeat_comparisons": {
          arm: {"logps": dict(exact)} for arm in ("R0", "R1", "REF")
      },
      "negative_control": {"exact": False, "differing_elements": 1},
  }


class ClassifyP38ReplayTest(unittest.TestCase):

  def test_accepts_complete_measurement_without_promoting_repair(self):
    result = classifier.classify(_report())
    self.assertEqual(result["verdict"], "PASS")
    self.assertFalse(result["production_repair_admitted"])

  def test_rejects_ineffective_negative_control(self):
    report = _report()
    report["negative_control"] = {"exact": True, "differing_elements": 0}
    self.assertEqual(classifier.classify(report)["verdict"], "FAIL")

  def test_rejects_nondeterministic_repeat(self):
    report = _report()
    report["repeat_comparisons"]["R1"]["logps"]["exact"] = False
    self.assertEqual(classifier.classify(report)["verdict"], "FAIL")

  def test_rejects_missing_no_optimizer_attestation(self):
    report = _report()
    report["no_optimizer"] = False
    self.assertEqual(classifier.classify(report)["verdict"], "FAIL")

  def test_rejects_weight_mismatch(self):
    report = _report()
    report["weight_attestation"]["equal"] = False
    self.assertEqual(classifier.classify(report)["verdict"], "FAIL")


if __name__ == "__main__":
  unittest.main()
