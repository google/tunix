#!/usr/bin/env python3
"""Fail-closed tests for the archived 64-chip evidence classifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


PACKAGE = Path(__file__).resolve().parents[2]
CLASSIFIER_PATH = PACKAGE / "debug_logs" / "classify_64chip_admission.py"
LOG_PATH = PACKAGE / "debug_logs" / "head_jax_tpu.log"
EXPECTED_SHA256 = "da3f7ff78ef43d8a55026cd4d40224a608d4c663a5888b316b23605e27a2f333"

SPEC = importlib.util.spec_from_file_location("classify_64chip_admission", CLASSIFIER_PATH)
assert SPEC and SPEC.loader
CLASSIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLASSIFIER)


class EvidenceClassifierTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.text = LOG_PATH.read_text(encoding="utf-8")

  def classify(self, text: str, expected_sha256: str | None = None):
    return CLASSIFIER.classify_text(
        text,
        artifact_sha256=EXPECTED_SHA256,
        expected_sha256=expected_sha256,
    )

  def test_archived_log_passes_with_bounded_scope(self):
    result = self.classify(self.text, EXPECTED_SHA256)
    self.assertEqual(result["status"], "TARGET PASS")
    self.assertEqual(result["reasons"], [])
    self.assertEqual(
        result["claim_scope"]["bounded_canonical_qwen_operator"],
        "TARGET PASS",
    )
    self.assertEqual(result["claim_scope"]["training"], "TARGET NOT RUN")
    self.assertEqual(result["measurements"]["generic_waycount"]["rows"], 18)
    self.assertEqual(
        result["measurements"]["generic_waycount"]["dirty_rows"], 18
    )

  def test_wrong_attempt_is_rejected(self):
    mutated = self.text.replace("JOBSET_ATTEMPT 0", "JOBSET_ATTEMPT 1", 1)
    self.assertEqual(self.classify(mutated)["status"], "INCONCLUSIVE")

  def test_missing_canonical_depth_is_rejected(self):
    line = next(
        line
        for line in self.text.splitlines()
        if line.startswith("[canonical-op] depth= 4 ")
    )
    mutated = self.text.replace(line + "\n", "", 1)
    self.assertEqual(self.classify(mutated)["status"], "INCONCLUSIVE")

  def test_false_t2_check_is_rejected(self):
    mutated = self.text.replace(
        '"fault_rejected": true', '"fault_rejected": false'
    )
    self.assertEqual(self.classify(mutated)["status"], "INCONCLUSIVE")

  def test_tainted_session_is_rejected(self):
    mutated = self.text + "\n[t1.unified] SKIP_TAINTED after=P1b skipped=T2\n"
    self.assertEqual(self.classify(mutated)["status"], "INCONCLUSIVE")

  def test_traceback_is_rejected(self):
    mutated = self.text + "\nTraceback (most recent call last):\n"
    self.assertEqual(self.classify(mutated)["status"], "INCONCLUSIVE")

  def test_incomplete_generic_diagnostic_is_rejected(self):
    line = next(
        line
        for line in self.text.splitlines()
        if line.startswith("[waycount] width= 8 replicas= 8 depth= 15 arm=f4-fixed")
    )
    mutated = self.text.replace(line + "\n", "", 1)
    self.assertEqual(self.classify(mutated)["status"], "INCONCLUSIVE")

  def test_hash_mismatch_is_rejected(self):
    result = self.classify(self.text, "0" * 64)
    self.assertEqual(result["status"], "INCONCLUSIVE")


if __name__ == "__main__":
  unittest.main()
