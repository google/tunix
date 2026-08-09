"""Tests for the fail-closed P35 envelope classifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


_CLASSIFIER_PATH = Path(__file__).with_name("classify_envelope.py")
_MODULE_SPEC = importlib.util.spec_from_file_location(
    "classify_p35_envelope", _CLASSIFIER_PATH
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
classifier = importlib.util.module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = classifier
_MODULE_SPEC.loader.exec_module(classifier)


def _pair(exact: bool) -> dict:
  return {
      "valid": True,
      "differing_elements": 0 if exact else 2,
      "total_elements": 8,
      "differing_bytes": 0 if exact else 3,
      "total_bytes": 32,
      "masked_hashes_equal": exact,
  }


def _report(ab_exact: bool, bc_exact: bool) -> dict:
  return {
      "schema_version": 1,
      "measurement_rows": 1,
      "arms": ["A", "B", "C"],
      "attestations": {
          key: True for key in classifier._REQUIRED_ATTESTATIONS
      },
      "negative_control": {
          "injected": True,
          "differing_elements": 1,
          "masked_hashes_equal": False,
      },
      "pairs": {
          classifier._PAIR_AB: _pair(ab_exact),
          classifier._PAIR_BC: _pair(bc_exact),
      },
  }


class ClassifyEnvelopeTest(unittest.TestCase):

  def test_classifies_all_four_outcomes(self):
    expected = {
        (False, True): "packing_metadata_carrier",
        (True, False): "wrapper_program_context_carrier",
        (False, False): "both_carriers",
        (True, True): "pre_backward_envelope_exact",
    }
    for exactness, classification in expected.items():
      with self.subTest(exactness=exactness):
        result = classifier.classify(_report(*exactness))
        self.assertEqual(result["measurement_verdict"], "COMPLETE")
        self.assertEqual(result["classification"], classification)

  def test_missing_arm_is_inconclusive(self):
    report = _report(True, True)
    report["arms"] = ["A", "B"]
    result = classifier.classify(report)
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")
    self.assertIn("arms", result["reasons"])

  def test_red_contract_is_inconclusive(self):
    report = _report(True, True)
    report["attestations"]["weights_equal"] = False
    result = classifier.classify(report)
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")
    self.assertIn("attestation.weights_equal", result["reasons"])

  def test_negative_control_must_be_observed(self):
    report = _report(True, True)
    report["negative_control"]["differing_elements"] = 0
    report["negative_control"]["masked_hashes_equal"] = True
    result = classifier.classify(report)
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")
    self.assertIn("negative_control.differing_elements", result["reasons"])

  def test_hash_and_counts_must_agree(self):
    report = _report(True, True)
    report["pairs"][classifier._PAIR_BC]["masked_hashes_equal"] = False
    result = classifier.classify(report)
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")
    self.assertIn(
        f"{classifier._PAIR_BC}.hash_count_consistency", result["reasons"]
    )


if __name__ == "__main__":
  unittest.main()
