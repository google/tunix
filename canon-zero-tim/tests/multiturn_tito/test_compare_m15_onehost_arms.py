#!/usr/bin/env python3
"""Tests for matched M15 verify/exact one-host receipt comparison."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tasks/multiturn-tito-cross-workload/scripts/compare_m15_onehost_arms.py"
SPEC = importlib.util.spec_from_file_location("compare_m15_onehost_arms", SCRIPT)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import M15 cross-arm comparator")
comparator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparator
SPEC.loader.exec_module(comparator)


def _write_arm(root: Path, mode: str, *, prompt_sha: str = "a" * 64) -> None:
  root.mkdir()
  root.joinpath("raw.log").write_text(
      "[CANON_M15_TOKEN_CONTINUITY] "
      f"mode={mode} turn=1 verdict=TOKEN_STREAM_EQUAL "
      "actual_tokens=3 expected_tokens=3 "
      f"actual_sha256={prompt_sha} expected_sha256={prompt_sha} "
      "first_mismatch=-1 actual_token=NA expected_token=NA\n"
  )
  root.joinpath("pre_alignment.jsonl").write_text(json.dumps({
      "diagnostic_round": 0,
      "N_action": 7,
      "hashes": {"tokens": "t", "action_mask": "m"},
  }) + "\n")


class ComparatorTest(unittest.TestCase):

  def test_match_and_prompt_negative(self):
    with tempfile.TemporaryDirectory() as temp:
      base = Path(temp)
      verify = base / "verify"
      exact = base / "exact"
      _write_arm(verify, "verify")
      _write_arm(exact, "exact")
      self.assertEqual(comparator.compare(verify, exact)["status"], "MATCH")
      _write_arm(base / "different", "exact", prompt_sha="b" * 64)
      result = comparator.compare(verify, base / "different")
      self.assertEqual(result["status"], "DIFFERENT")
      self.assertEqual(result["first_prompt_difference"], 0)


if __name__ == "__main__":
  unittest.main()
