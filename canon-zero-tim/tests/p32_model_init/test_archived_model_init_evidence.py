#!/usr/bin/env python3
"""Fail-closed checks for the archived P32.2c target artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest


PACKAGE = Path(__file__).resolve().parents[2]
DEBUG_LOGS = PACKAGE / "debug_logs"
PASS_LOG = DEBUG_LOGS / "p32_2c_model_init_attempt0_pass.raw.log"
PASS_REPORT = (
    DEBUG_LOGS / "p32_2c_model_init_attempt0_pass.classification.json"
)
FAIL_LOG = DEBUG_LOGS / "p32_2c_model_init_attempt0_hostbuffer_fail.raw.log"
EXPECTED_PASS_LOG_SHA256 = (
    "4a98384920de136da753114963d8edc216e0b564e535276091b4b2178d1fd140"
)
EXPECTED_PASS_REPORT_SHA256 = (
    "1097f0b67410a9eb5178121dccf0a7c9a84b0e36f5c9601a3b82330c6f84eb59"
)
EXPECTED_FAIL_LOG_SHA256 = (
    "af4e8baaa9a325fac32b8187b0fbab84cd22b005a45ec2e77507127fc6ec6c5c"
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify_model_init  # pylint: disable=g-import-not-at-top


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


class ArchivedModelInitEvidenceTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.pass_text = PASS_LOG.read_text(encoding="utf-8", errors="replace")
    cls.fail_text = FAIL_LOG.read_text(encoding="utf-8", errors="replace")

  def test_artifact_hashes_are_pinned(self):
    self.assertEqual(_sha256(PASS_LOG), EXPECTED_PASS_LOG_SHA256)
    self.assertEqual(_sha256(PASS_REPORT), EXPECTED_PASS_REPORT_SHA256)
    self.assertEqual(_sha256(FAIL_LOG), EXPECTED_FAIL_LOG_SHA256)

  def test_pass_artifact_reclassifies(self):
    result = classify_model_init.classify_text(self.pass_text)
    self.assertEqual(result["status"], "PASS")
    self.assertEqual(result["reasons"], [])

  def test_pass_artifact_has_clean_outer_lifecycle(self):
    required_once = (
        "[entrypoint] JOBSET_ATTEMPT 0 (first attempt)",
        "[sync] HEAD=ce0511ee068be38431e005faccfc149694dccb5d",
        "[sync] tracked_dirty=0",
        "[sync] package_untracked=0",
        "[entrypoint] <-- 80_model_init.sh ok",
        "[entrypoint] mode=model-init-only -- structural state materialized; "
        "no checkpoint, forward, backward, update or training was run.",
    )
    for marker in required_once:
      with self.subTest(marker=marker):
        self.assertEqual(self.pass_text.count(marker), 1)

  def test_failed_artifact_remains_a_rejected_negative(self):
    result = classify_model_init.classify_text(self.fail_text)
    self.assertEqual(result["status"], "INCONCLUSIVE")
    self.assertIn("pthread_create() failed", self.fail_text)
    self.assertNotIn("[P32.INIT] VERDICT PASS", self.fail_text)


if __name__ == "__main__":
  unittest.main()
