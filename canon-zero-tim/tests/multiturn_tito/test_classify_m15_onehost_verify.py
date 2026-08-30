#!/usr/bin/env python3
"""Tests for the M15 one-host TiTO observer classifier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tasks/multiturn-tito-cross-workload/scripts/classify_m15_onehost_verify.py"
SPEC = importlib.util.spec_from_file_location("classify_m15_onehost_verify", SCRIPT)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import M15 one-host classifier")
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)


def _row(index: int) -> dict:
  boundary = {"differing_bytes": 0, "finite": True}
  return {
      "diagnostic_round": index,
      "verdict": "PASS",
      "N_action": 100 + index,
      "boundaries": {
          "S_decode_vs_S_prefill": dict(boundary),
          "S_prefill_vs_T_old": dict(boundary),
      },
  }


class ClassifierTest(unittest.TestCase):

  def _classify(
      self,
      verdict: str = "TOKEN_STREAM_EQUAL",
      mutate=None,
      *,
      mode: str = "verify",
  ):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      raw = root / "raw.log"
      report = root / "pre.jsonl"
      lines = []
      for index in range(3):
        lines.extend([
            f"[CANON_M15_TOKEN_CONTINUITY] mode={mode} turn=1 verdict={verdict} actual_tokens=3 expected_tokens=3 actual_sha256=a expected_sha256=b first_mismatch={-1 if verdict.endswith('EQUAL') else 1} actual_token=NA expected_token=NA",
            "[CANON_APC_M15_B_CONTRACT] reset_prefix_cache=True all_num_cached_tokens_zero=True",
            f"[CANON_P38] PRECHECK_ROUND_COMPLETE round={index + 1}/3 step={index} N_action=100 verdict=PASS a_b_differing_bytes=0 backward=0 optimizer_commits=0",
        ])
      lines.append("[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0")
      if mutate:
        lines, rows = mutate(lines, [_row(i) for i in range(3)])
      else:
        rows = [_row(i) for i in range(3)]
      raw.write_text("\n".join(lines) + "\n")
      report.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
      return classifier.classify(raw, report, mode=mode)

  def test_equal_and_drift_are_both_scientific_results(self):
    self.assertEqual(self._classify()["status"], "LEGACY_TOKEN_EQUAL")
    drift = self._classify("TOKEN_STREAM_DIFFERENT")
    self.assertEqual(drift["status"], "LEGACY_TOKEN_DRIFT")
    self.assertEqual(drift["first_red"]["first_mismatch"], 1)

  def test_exact_requires_every_consumed_prompt_to_match(self):
    result = self._classify(mode="exact")
    self.assertEqual(
        result["status"], "EXACT_TOKEN_CONTINUITY_ALIGNMENT_PASS"
    )
    drift = self._classify("TOKEN_STREAM_DIFFERENT", mode="exact")
    self.assertEqual(drift["status"], "FAIL")
    self.assertIn("token_receipts.exact_mismatch", drift["reasons"])

  def test_b_c_red_is_fatal(self):
    def mutate(lines, rows):
      rows[1]["boundaries"]["S_prefill_vs_T_old"]["differing_bytes"] = 4
      return lines, rows
    result = self._classify(mutate=mutate)
    self.assertEqual(result["status"], "FAIL")
    self.assertIn("round.1.B-C", result["reasons"])

  def test_missing_round_or_token_receipts_is_fatal(self):
    def mutate(lines, rows):
      return [line for line in lines if "TOKEN_CONTINUITY" not in line], rows[:2]
    result = self._classify(mutate=mutate)
    self.assertEqual(result["status"], "FAIL")
    self.assertIn("token_receipts.coverage", result["reasons"])


if __name__ == "__main__":
  unittest.main()
