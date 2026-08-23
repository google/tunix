#!/usr/bin/env python3
"""Cross-arm classification tests for the P58 one-host XProf pair."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
CLASSIFIER = (
    ROOT / "canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/"
    "classify_onehost_xprof_pair.py"
)


class OnehostXprofPairTest(unittest.TestCase):

  def _run(self, native: dict, zero: dict) -> tuple[int, dict]:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      native_path = root / "native.json"
      zero_path = root / "zero.json"
      output = root / "pair.json"
      native_path.write_text(json.dumps(native))
      zero_path.write_text(json.dumps(zero))
      result = subprocess.run(
          [
              sys.executable,
              str(CLASSIFIER),
              "--native", str(native_path),
              "--zero-hp", str(zero_path),
              "--output", str(output),
          ],
          check=False,
          capture_output=True,
          text=True,
      )
      return result.returncode, json.loads(output.read_text())

  def _arm(self, arm: str) -> dict:
    return {
        "arm": arm,
        "verdict": "PASS",
        "source_sha": "1" * 40,
        "source_diff_sha256": "2" * 64,
        "expected_hostname": "host",
        "work_hashes": {"completion_ids": "a", "actor_update_calls": 2},
    }

  def test_matched_pair_passes(self):
    code, result = self._run(self._arm("native"), self._arm("zero-hp"))
    self.assertEqual(code, 0)
    self.assertEqual(result["verdict"], "PASS")

  def test_work_mismatch_is_inconclusive_not_a_speed_claim(self):
    native = self._arm("native")
    zero = self._arm("zero-hp")
    zero["work_hashes"] = {"completion_ids": "b", "actor_update_calls": 2}
    code, result = self._run(native, zero)
    self.assertEqual(code, 3)
    self.assertEqual(result["verdict"], "INCONCLUSIVE_INPUT_MISMATCH")
    self.assertEqual(result["claim"], "no causal performance delta")


if __name__ == "__main__":
  unittest.main()
