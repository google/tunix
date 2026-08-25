#!/usr/bin/env python3
"""Regression tests for the immutable M15 Attempt-2 evidence decoder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("analyze_m15i_evidence.py")
SPEC = importlib.util.spec_from_file_location("analyze_m15i_evidence", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PACKAGE = MODULE_PATH.parents[3]
EVIDENCE = (
    PACKAGE
    / "tasks/v1-phase4-three-full-recipes/evidence"
    / "v1_hp_three_full_attempt2_20260824"
)


class AnalyzeM15iEvidenceTest(unittest.TestCase):

  def test_frozen_m15i_contract(self):
    manifest = MODULE.verify_manifest(EVIDENCE)
    self.assertTrue(manifest["valid"])

    decoded = MODULE.decode_log(EVIDENCE / "m15_m15i_error.log")
    alignment = decoded["alignment"]
    mismatch = alignment["a_minus_b"]
    self.assertEqual(alignment["n_action"], 110844)
    self.assertEqual(mismatch["reported_elements"], 760)
    self.assertEqual(mismatch["reported_bytes"], 1389)
    self.assertEqual(mismatch["reported_max_abs"], 0.998443603515625)
    self.assertEqual(alignment["b_minus_c"]["differing_bytes"], 0)
    self.assertEqual(mismatch["prompt_groups"], {"24": 760})
    self.assertEqual(
        mismatch["sequence_rows"],
        {"192": 76, "193": 46, "194": 79, "196": 134,
         "197": 195, "198": 80, "199": 150},
    )
    self.assertEqual(mismatch["exact_256_boundary_count"], 6)
    self.assertEqual(mismatch["first_mismatch"]["logical_kv_prefix_length"], 1226)
    self.assertEqual(decoded["runtime"]["source_head"]["sha"],
                     "71d889a32f4668353c758d5c00df88299e6c0d35")

  def test_manifest_corruption_is_detected(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      receipt = json.loads((EVIDENCE / "receipt.json").read_text())
      (root / "receipt.json").write_text(json.dumps(receipt) + "\n")
      (root / "SHA256SUMS").write_text(
          "0" * 64 + "  receipt.json\n", encoding="utf-8"
      )
      self.assertFalse(MODULE.verify_manifest(root)["valid"])


if __name__ == "__main__":
  unittest.main()
