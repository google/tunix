"""Fail-closed controls for the persisted same-session validator."""

from __future__ import annotations

import unittest

import validate_same_session as target


def _good_lines():
    return [
        "[canonical-op] VERDICT: PASS",
        "[P32.DP] CONFIG dp=16 tp=4",
        "[P32.DP] MESH ids=(0,) shape=(16, 4) full_slice=1",
        "[P32.DP] CHECKS {}",
        "[P32.DP] OBSERVATIONS {}",
        "[P32.DP] UPDATE {}",
        "[P32.DP] DECISION FIXED_TOPOLOGY_STOCK_ADMISSIBLE",
        "[P32.DP] VERDICT PASS",
    ]


class SameSessionValidatorTest(unittest.TestCase):
    def test_complete_singleton_markers_pass(self):
        passed, reasons = target.validate_lines(_good_lines())
        self.assertTrue(passed)
        self.assertEqual(reasons, ())

    def test_missing_duplicate_and_red_markers_are_rejected(self):
        cases = {
            "missing": _good_lines()[:-1],
            "duplicate": _good_lines() + ["[P32.DP] UPDATE {}"],
            "canonical_red": [
                line.replace("VERDICT: PASS", "VERDICT: FAIL")
                for line in _good_lines()
            ],
            "t2_red": [
                line.replace("VERDICT PASS", "VERDICT FAIL")
                for line in _good_lines()
            ],
        }
        for name, lines in cases.items():
            with self.subTest(name=name):
                passed, reasons = target.validate_lines(lines)
                self.assertFalse(passed)
                self.assertTrue(reasons)


if __name__ == "__main__":
    unittest.main()
