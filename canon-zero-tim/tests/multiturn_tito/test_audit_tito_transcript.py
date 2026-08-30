#!/usr/bin/env python3
"""Focused tests for the bounded TiTO transcript oracle."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "tasks/multiturn-tito-cross-workload/scripts/audit_tito_transcript.py"
)
SPEC = importlib.util.spec_from_file_location("audit_tito_transcript", SCRIPT)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import TiTO transcript oracle")
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class TokenComparisonTest(unittest.TestCase):

  def test_equal_receipt_is_content_free(self):
    receipt = audit.compare_tokens([1, 2, 3], np.asarray([1, 2, 3]))
    self.assertEqual(receipt["verdict"], "TOKEN_STREAM_EQUAL")
    self.assertEqual(receipt["first_mismatch"], -1)
    self.assertNotIn("tokens_value", receipt)

  def test_value_and_length_mismatches_report_first_coordinate(self):
    value = audit.compare_tokens([1, 2, 3], [1, 9, 3])
    self.assertEqual(value["first_mismatch"], 1)
    length = audit.compare_tokens([1, 2], [1, 2, 3])
    self.assertEqual(length["first_mismatch"], 2)

  def test_invalid_vectors_fail_closed(self):
    with self.assertRaisesRegex(ValueError, "one-dimensional"):
      audit.compare_tokens([[1, 2]], [1, 2])
    with self.assertRaisesRegex(ValueError, "negative"):
      audit.compare_tokens([-1], [1])


if __name__ == "__main__":
  unittest.main()
