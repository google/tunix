#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("classify_m15_apc_debug_tito.py")
SPEC = importlib.util.spec_from_file_location("classify_m15_apc_debug_tito", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)


def _receipt(*, verdict: str = "TOKEN_STREAM_EQUAL") -> str:
  mismatch = "-1" if verdict == "TOKEN_STREAM_EQUAL" else "2"
  token = "NA" if verdict == "TOKEN_STREAM_EQUAL" else "8"
  expected = "NA" if verdict == "TOKEN_STREAM_EQUAL" else "3"
  actual_sha = "a" * 64
  expected_sha = actual_sha if verdict == "TOKEN_STREAM_EQUAL" else "b" * 64
  return (
      "[CANON_M15_TOKEN_CONTINUITY] mode=exact turn=1 "
      f"verdict={verdict} actual_tokens=10 expected_tokens=10 "
      f"actual_sha256={actual_sha} expected_sha256={expected_sha} "
      f"first_mismatch={mismatch} actual_token={token} "
      f"expected_token={expected}"
  )


def _round(index: int) -> str:
  return (
      "[CANON_P38] PRECHECK_ROUND_COMPLETE "
      f"round={index}/3 step=0 N_action=256 verdict=PASS "
      "a_b_differing_bytes=0 backward=0 optimizer_commits=0"
  )


def _log(arm: str = "on") -> list[str]:
  lines = [
      "[env] M15 APC debug exact TITO enabled mode=exact "
      f"arm={arm} observer=layer rounds=3"
  ]
  for index in range(1, 4):
    lines.extend((_receipt(), _round(index)))
  return lines


def _onehost_log(arm: str = "on") -> list[str]:
  lines = [
      "[M15.E0V.ONEHOST] exact TITO enabled mode=exact "
      f"arm={arm} topology=DP1xTP4 rounds=3"
  ]
  for index in range(1, 4):
    lines.extend((_receipt(), _round(index)))
  return lines


class TitoPostflightTest(unittest.TestCase):

  def _classify(
      self, lines: list[str], arm: str = "on", scope: str = "target"
  ):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "run.log"
      path.write_text("\n".join(lines) + "\n", encoding="utf-8")
      return classifier.classify(run_log=path, arm=arm, scope=scope)

  def test_three_round_exact_receipts_pass(self):
    report = self._classify(_log())
    self.assertEqual(report["status"], "PASS")
    self.assertEqual(report["round_receipt_counts"], [1, 1, 1])
    self.assertEqual(report["total_exact_equal_receipts"], 3)
    self.assertFalse(report["historical_1226_prefix_reused"])

  def test_different_receipt_fails_closed(self):
    lines = _log()
    lines[3] = _receipt(verdict="TOKEN_STREAM_DIFFERENT")
    with self.assertRaisesRegex(classifier.TitoAuditError, "non-exact"):
      self._classify(lines)

  def test_each_round_requires_a_receipt(self):
    lines = _log()
    del lines[3]
    with self.assertRaisesRegex(classifier.TitoAuditError, "round 1 has no"):
      self._classify(lines)

  def test_arm_specific_environment_receipt_is_required(self):
    with self.assertRaisesRegex(classifier.TitoAuditError, "environment"):
      self._classify(_log("off"), arm="on")

  def test_round_sequence_cannot_skip(self):
    lines = _log()
    lines[2] = _round(2)
    with self.assertRaisesRegex(classifier.TitoAuditError, "sequence"):
      self._classify(lines)

  def test_onehost_three_round_exact_receipts_pass(self):
    report = self._classify(_onehost_log(), scope="onehost")
    self.assertEqual(report["status"], "PASS")
    self.assertEqual(report["scope"], "onehost")
    self.assertEqual(report["topology"], "DP1xTP4")
    self.assertIsNone(report["observer"])

  def test_target_marker_cannot_satisfy_onehost_scope(self):
    with self.assertRaisesRegex(classifier.TitoAuditError, "onehost"):
      self._classify(_log(), scope="onehost")


if __name__ == "__main__":
  unittest.main()
