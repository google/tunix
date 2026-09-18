"""Tests for the fail-closed P35.3 exact-replay classifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


PATH = Path(__file__).with_name("classify_exact_replay.py")
SPEC = importlib.util.spec_from_file_location("p35_exact_replay_classifier", PATH)
assert SPEC is not None and SPEC.loader is not None
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)


def _pair(exact):
  return {
      "valid": True,
      "differing_elements": 0 if exact else 1,
      "total_elements": 8,
      "differing_bytes": 0 if exact else 1,
      "total_bytes": 32,
      "masked_hashes_equal": exact,
      "exact": exact,
  }


def _stages(exact=True):
  return {
      stage: {
          "valid": True,
          "differing_elements": 0 if exact else 1,
          "total_elements": 8,
          "exact": exact,
      }
      for stage in classifier.REQUIRED_STAGES
  }


def _report(
    *, b0=True, p01=True, p12=True, p23=False, p3c=True, bc=False
):
  return {
      "schema_version": 1,
      "measurement_rows": 1,
      "arms": ["B", "R0", "R1", "R2", "R3", "C"],
      "attestations": {
          key: True for key in classifier.REQUIRED_ATTESTATIONS
      },
      "negative_control": {
          "injected": True,
          "differing_elements": 1,
          "masked_hashes_equal": False,
      },
      "repeat_comparisons": {
          "R0_live_repeat": _stages(),
          "R1_mapped_repeat": _stages(),
          "R2_adapter_direct_repeat": {"logps": _pair(True)},
      },
      "stage_comparisons": {"R0_live_vs_R1_mapped": _stages(p01)},
      "pairs": {
          classifier.PAIR_B0: _pair(b0),
          classifier.PAIR_01: _pair(p01),
          classifier.PAIR_12: _pair(p12),
          classifier.PAIR_23: _pair(p23),
          classifier.PAIR_3C: _pair(p3c),
          classifier.PAIR_BC: _pair(bc),
      },
  }


class ExactReplayClassifierTest(unittest.TestCase):

  def test_classifies_placement_carrier(self):
    result = classifier.classify(_report(p01=False, p23=True))
    self.assertEqual(result["measurement_verdict"], "COMPLETE")
    self.assertEqual(result["classification"], "weight_memory_placement_carrier")

  def test_classifies_adapter_outer_program(self):
    result = classifier.classify(_report(p23=False))
    self.assertEqual(result["classification"], "adapter_outer_program_carrier")

  def test_classifies_metadata_cache_carrier(self):
    result = classifier.classify(_report(p12=False, p23=True))
    self.assertEqual(
        result["classification"], "metadata_cache_construction_carrier"
    )

  def test_rejects_unanchored_serving_replay(self):
    result = classifier.classify(_report(b0=False))
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")
    self.assertIn("serving_replay_not_anchored", result["reasons"])

  def test_rejects_unanchored_adapter_repeat(self):
    result = classifier.classify(_report(p3c=False))
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")
    self.assertIn("adapter_repeat_not_anchored", result["reasons"])

  def test_rejects_repeat_drift(self):
    report = _report()
    report["repeat_comparisons"]["R0_live_repeat"] = _stages(False)
    result = classifier.classify(report)
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")
    self.assertTrue(any("repeat_exact" in reason for reason in result["reasons"]))

  def test_rejects_missing_stage(self):
    report = _report()
    del report["stage_comparisons"]["R0_live_vs_R1_mapped"]["final_hidden"]
    result = classifier.classify(report)
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")

  def test_rejects_missing_known_red(self):
    result = classifier.classify(_report(bc=True, p23=True))
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")
    self.assertIn("known_B_vs_C_red_not_reproduced", result["reasons"])


if __name__ == "__main__":
  unittest.main()
