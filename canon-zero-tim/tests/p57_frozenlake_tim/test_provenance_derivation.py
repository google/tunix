"""Regression gate for the auditable p57cal6 provenance derivation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "canon-zero-tim/tasks/p57-frozenlake-tim-causal-study"
SOURCE = TASK / "evidence/p57cal6/p57_calibration.json"


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec and spec.loader
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


deriver = _load(
    "p57_provenance_deriver", TASK / "scripts/derive_calibration_provenance.py"
)
classifier = _load(
    "p57_derived_classifier", TASK / "scripts/classify_stock_discovery.py"
)


class P57ProvenanceDerivationTest(unittest.TestCase):

  def test_pre_p78_exception_is_exact_artifact_not_a_generic_missing_receipt(self):
    with tempfile.TemporaryDirectory() as tmp:
      output = Path(tmp) / "derived.json"
      deriver.derive(SOURCE, output, Path(tmp) / "proof.json")
      value = json.loads(output.read_text())
      value["source_commit"] = "a" * 40
      output.write_text(json.dumps(value))
      self.assertEqual(classifier.classify(output)["verdict"], "FAIL")

  def test_committed_cal6_preserves_measurements_and_historical_verdict(self):
    source_before = SOURCE.read_bytes()
    historic_path = SOURCE.with_name("classification.derived.json")
    historic_before = historic_path.read_bytes()
    historic = json.loads(historic_before)
    with tempfile.TemporaryDirectory() as tmp:
      output = Path(tmp) / "derived.json"
      proof_path = Path(tmp) / "proof.json"
      proof = deriver.derive(SOURCE, output, proof_path)
      result = classifier.classify(output)
      derived = json.loads(output.read_text(encoding="utf-8"))

    self.assertEqual(SOURCE.read_bytes(), source_before)
    self.assertEqual(historic_path.read_bytes(), historic_before)
    self.assertEqual(proof["verdict"], "PASS")
    self.assertEqual(proof["records_derived"], 2400)
    self.assertFalse(proof["measured_fields_modified"])
    self.assertEqual(historic["verdict"], "PASS")
    self.assertEqual(historic["selection"], "FREEZE_M15")
    self.assertEqual(historic["selected_recipe"], "m15")
    self.assertEqual(result["summaries"], historic["summaries"])
    self.assertTrue(
        derived == json.loads(
            SOURCE.with_name("p57_calibration.derived.json").read_text()
        ),
        "the provenance derivation changed the committed derived receipt",
    )
    # Old measurements and their original PASS remain immutable. The newer
    # runtime contract cannot infer an unrecorded P78-off check from them.
    self.assertEqual(result["verdict"], "FAIL")
    self.assertEqual(result["selection"], "INVALID")
    self.assertIsNone(result["selected_recipe"])
    self.assertEqual(len(result["reasons"]), 1)
    self.assertIn("zero_tim_off_attestation", result["reasons"][0])
    self.assertNotIn(
        "CANON_P78_SEGMENTED_ACTOR_LOGPS",
        derived["zero_tim_off_attestation"]["zero_switches"],
    )
    self.assertEqual(
        derived["provenance_derivation"]["source_sha256"],
        proof["source_sha256"],
    )


if __name__ == "__main__":
  unittest.main()
