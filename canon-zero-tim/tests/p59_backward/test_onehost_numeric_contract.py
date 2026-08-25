#!/usr/bin/env python3
"""Static fail-closed contracts for the bounded P62 numeric carrier."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
RUNNER = PKG / "tests/p59_backward/run_onehost_numeric_v5p.sh"
PROBE = PKG / "tests/p59_backward/probe_onehost_numeric_v5p.py"


class OneHostNumericContractTest(unittest.TestCase):

  def test_runner_pins_lane_image_and_zero_commit_scope(self):
    text = RUNNER.read_text(encoding="utf-8")
    self.assertIn('test "$(hostname)" = t1v-n-4a77ebd0-w-0', text)
    self.assertIn(
        "sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a",
        text,
    )
    self.assertIn('if [[ -e "$root" ]]', text)
    self.assertIn("grep -E '^(p51_|p59_|p62_)'", text)
    self.assertIn("optimizer_commits=0", text)
    self.assertNotIn("P62_ONEHOST_CONSTRUCTION_ONLY", text)

  def test_probe_covers_scaling_reduction_and_fp64_oracle(self):
    text = PROBE.read_text(encoding="utf-8")
    self.assertIn('devices.reshape(_DP, _TP), ("data", "model")', text)
    self.assertIn("jax.vjp(forward", text)
    self.assertIn("FixedDPRankGradientReducer", text)
    self.assertIn("optimization_barrier", text)
    self.assertIn("_STREAMED_MULTIPLIER", text)
    self.assertIn("np.float64", text)
    self.assertIn("wrong_multiplier_rel_l2", text)
    self.assertIn("wrong_dp_rel_l2", text)
    self.assertIn("optimizer_commits=0", text)

  def test_construction_receipt_cannot_impersonate_v5p(self):
    text = PROBE.read_text(encoding="utf-8")
    self.assertIn("P62_NUMERIC_ONEHOST_CONSTRUCTION_PASS", text)
    self.assertIn("P62_NUMERIC_ONEHOST_V5P_PASS", text)
    self.assertIn('construction_value not in ("", "0", "1")', text)


if __name__ == "__main__":
  unittest.main()
