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


def _report(ab_exact: bool, bc_exact: bool, ac_exact: bool = False) -> dict:
  return {
      "schema_version": 2,
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
          classifier._PAIR_AC: _pair(ac_exact),
      },
  }


class ClassifyEnvelopeTest(unittest.TestCase):

  def test_classifies_three_reproduced_red_outcomes(self):
    expected = {
        (False, True): "packing_metadata_carrier",
        (True, False): "adapter_envelope_carrier",
        (False, False): "mixed_envelope_carriers",
    }
    for exactness, classification in expected.items():
      with self.subTest(exactness=exactness):
        result = classifier.classify(_report(*exactness))
        self.assertEqual(result["measurement_verdict"], "COMPLETE")
        self.assertEqual(result["classification"], classification)

  def test_exact_exact_is_inconclusive_when_known_red_is_not_reproduced(self):
    result = classifier.classify(_report(True, True, True))
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")
    self.assertIn("known_A_vs_C_red_not_reproduced", result["reasons"])

  def test_transitivity_conflict_is_inconclusive(self):
    result = classifier.classify(_report(True, True, False))
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")
    self.assertIn("bitwise_transitivity", result["reasons"])

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

  def test_missing_direct_reproduction_pair_is_inconclusive(self):
    report = _report(False, True)
    del report["pairs"][classifier._PAIR_AC]
    result = classifier.classify(report)
    self.assertEqual(result["measurement_verdict"], "INCONCLUSIVE")
    self.assertIn("pair_keys", result["reasons"])


if __name__ == "__main__":
  unittest.main()
