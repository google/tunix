#!/usr/bin/env python3
"""Unit tests for the P38 lm_head operator-screen verdict."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts"
    / "probe_p38_lm_head.py"
)
SPEC = importlib.util.spec_from_file_location("probe_p38_lm_head", SCRIPT)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class LmHeadProbeTest(unittest.TestCase):

  def test_verdict_table(self):
    exact = [{
        "default_differing_elements": 0,
        "algorithm_differing_elements": 0,
    }]
    eliminated = [{
        "default_differing_elements": 3,
        "algorithm_differing_elements": 0,
    }]
    insufficient = [{
        "default_differing_elements": 3,
        "algorithm_differing_elements": 1,
    }]
    self.assertEqual(
        probe.classify(exact, 1),
        "BOTH_EXACT_OPERATOR_SCREEN_INCONCLUSIVE",
    )
    self.assertEqual(
        probe.classify(eliminated, 1),
        "ALGORITHM_ELIMINATES_OPERATOR_DRIFT",
    )
    self.assertEqual(
        probe.classify(insufficient, 1),
        "ALGORITHM_NOT_SUFFICIENT",
    )
    self.assertEqual(
        probe.classify(exact, 0),
        "FAIL_NEGATIVE_CONTROL",
    )


if __name__ == "__main__":
  unittest.main()
