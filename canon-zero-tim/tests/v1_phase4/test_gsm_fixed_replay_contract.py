#!/usr/bin/env python3
"""Static and pure-capsule contracts for the GSM fixed replay carrier."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / (
    "canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/"
    "probe_gsm_fixed_replay_scale.py"
)


class GsmFixedReplayContractTest(unittest.TestCase):

  def test_script_is_syntactically_valid_and_freezes_exact_topology(self):
    text = SCRIPT.read_text(encoding="utf-8")
    ast.parse(text)
    for contract in (
        "_DP = 16",
        "_TP = 4",
        "_GROUPS = 16",
        "_GLOBAL_TRAJECTORIES = 256",
        "_SEED = 42",
        '"optimizer_commits": 0',
        '"bounded_projection_topology_and_scale_only"',
    ):
      self.assertIn(contract, text)

  def test_both_arms_share_one_capsule_and_fp64_oracle(self):
    text = SCRIPT.read_text(encoding="utf-8")
    self.assertEqual(text.count("replay = _frozen_replay()"), 1)
    self.assertIn("ordinary_pullback = jax.vjp", text)
    self.assertIn("parallel_pullback = jax.jit(jax.shard_map", text)
    self.assertIn("oracle = np.einsum", text)
    self.assertIn("FixedDPRankGradientReducer", text)
    self.assertIn("wrong_denominator_rel_l2", text)
    self.assertIn("duplicate_dp_sum_rel_l2", text)

  def test_pinned_image_executes_forced_cpu_dp16_tp4(self):
    exact = (
        ROOT / "canon-zero-tim/tests/v1_phase4/run_exact_image.sh"
    ).read_text(encoding="utf-8")
    self.assertIn("probe_gsm_fixed_replay_scale.py", exact)
    self.assertIn("--xla_force_host_platform_device_count=64", exact)
    self.assertIn("gsm_scale_replay=1", exact)


if __name__ == "__main__":
  unittest.main()
