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

  def test_committed_cal6_derives_without_mutating_measurements(self):
    source_before = SOURCE.read_bytes()
    with tempfile.TemporaryDirectory() as tmp:
      output = Path(tmp) / "derived.json"
      proof_path = Path(tmp) / "proof.json"
      proof = deriver.derive(SOURCE, output, proof_path)
      result = classifier.classify(output)
      derived = json.loads(output.read_text(encoding="utf-8"))

    self.assertEqual(SOURCE.read_bytes(), source_before)
    self.assertEqual(proof["verdict"], "PASS")
    self.assertEqual(proof["records_derived"], 2400)
    self.assertFalse(proof["measured_fields_modified"])
    self.assertEqual(result["verdict"], "PASS")
    self.assertEqual(result["selection"], "FREEZE_M15")
    self.assertEqual(result["selected_recipe"], "m15")
    self.assertEqual(
        derived["provenance_derivation"]["source_sha256"],
        proof["source_sha256"],
    )


if __name__ == "__main__":
  unittest.main()
