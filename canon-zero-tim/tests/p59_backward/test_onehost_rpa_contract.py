#!/usr/bin/env python3
"""Static fail-closed contracts for the bounded real-v5p RPA carrier."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
RUNNER = PKG / "tests/p59_backward/run_onehost_rpa_v5p.sh"
PROBE = PKG / "tests/p59_backward/probe_onehost_rpa_v5p.py"


class OneHostRpaContractTest(unittest.TestCase):

  def test_runner_pins_lane_image_and_first_use_evidence(self):
    text = RUNNER.read_text(encoding="utf-8")
    self.assertIn("test \"$(hostname)\" = t1v-n-4a77ebd0-w-0", text)
    self.assertIn("sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a", text)
    self.assertIn("if [[ -e \"$root\" ]]", text)
    self.assertIn("grep -E '^(p51_|p59_)'", text)
    self.assertIn("optimizer_commits=0", text)

  def test_probe_executes_both_four_chip_topologies(self):
    text = PROBE.read_text(encoding="utf-8")
    self.assertIn('devices.size != 4', text)
    self.assertIn('devices.reshape(2, 2), ("data", "model")', text)
    self.assertIn('devices.reshape(1, 1, 1, 1, 4, 1)', text)
    self.assertIn("jax.vjp(primal", text)
    self.assertIn("sharded_ragged_paged_attention", text)
    self.assertIn("_run_staged_spec_restore(outer_mesh)", text)
    self.assertIn(
        "_p59_restore_physically_equal_staged_specs", text
    )
    self.assertNotIn("fake_rpa", text)

  def test_negative_and_receipt_are_mandatory(self):
    runner = RUNNER.read_text(encoding="utf-8")
    probe = PROBE.read_text(encoding="utf-8")
    self.assertIn("P59 local attention cache shape mismatch", probe)
    self.assertIn("wrong-cache negative did not fire", probe)
    self.assertIn("ordinary_global_gqa=1", probe)
    self.assertIn("P59_STAGED_SPEC_ONEHOST_PASS", probe)
    self.assertIn("wrong-placement negative did not fire", probe)
    self.assertIn("local_marker_count", runner)
    self.assertIn("restore_marker_count", runner)
    self.assertIn("terminal_count", runner)
    self.assertIn("traceback_count", runner)


if __name__ == "__main__":
  unittest.main()
