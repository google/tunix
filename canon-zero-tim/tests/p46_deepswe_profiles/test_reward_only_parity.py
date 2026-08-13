"""Layered L2/L3 policy tests for P46 reward-only evaluation."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/deepswe"))

import deepswe_reward_only_parity as parity  # pylint: disable=wrong-import-position


class RewardOnlyParityTest(unittest.TestCase):

  def test_l2_identity_and_suffix_divergence_both_pass(self):
    identical = parity.classify_l2_tokens([1, 2, 3], [1, 2, 3])
    self.assertTrue(identical["hard_gate_pass"])
    self.assertEqual(identical["classification"], "IDENTICAL_OBSERVER")
    diverged = parity.classify_l2_tokens([1, 2, 3], [1, 9, 8, 7])
    self.assertTrue(diverged["hard_gate_pass"])
    self.assertEqual(diverged["classification"], "LAW1_SUFFIX_DIVERGENCE")
    self.assertEqual(diverged["first_divergence_index"], 1)

  def test_l2_rejects_malformed_streams(self):
    for right in ([], [1, -1], [1, "bad"]):
      with self.subTest(right=right):
        with self.assertRaisesRegex(ValueError, "token stream"):
          parity.classify_l2_tokens([1, 2], right)

  def test_l3_exact_paired_binomial_gate(self):
    balanced = [
        {
            "identity": f"sample-{index}",
            "logprob_solved": index in (0, 1, 2, 3),
            "reward_only_solved": index in (0, 1, 2, 4),
        }
        for index in range(16)
    ]
    report = parity.classify_l3_paired_solve_rate(balanced)
    self.assertEqual(report["verdict"], "PASS")
    shifted = [
        {
            "identity": f"sample-{index}",
            "logprob_solved": False,
            "reward_only_solved": True,
        }
        for index in range(16)
    ]
    report = parity.classify_l3_paired_solve_rate(shifted)
    self.assertEqual(report["verdict"], "FAIL")

  def test_l3_artifact_gate_requires_exact_n16_and_mode_provenance(self):
    observer = []
    reward_only = []
    for sample_index in range(16):
      common = {
          "task_key": "task-a",
          "sample_index": sample_index,
          "sampled_by": "stock@" + "6" * 40,
          "valid": True,
          "solved": sample_index in (0, 1, 2, 3),
      }
      observer.append({
          **common,
          "trajectory_mode": "observer_with_sampled_logprobs",
          "trajectory": {"steps": [{"logprobs": [-0.2]}]},
      })
      reward_only.append({
          **common,
          "trajectory_mode": "reward_only_no_logprobs",
          "trajectory": {"steps": [{"logprobs": None}]},
      })
    report = parity.build_l3_report(
        observer,
        reward_only,
        observer_wall_secs=1800,
        reward_only_wall_secs=1200,
    )
    self.assertEqual(report["verdict"], "PASS")
    self.assertEqual(report["pairs"], 16)
    self.assertEqual(report["observer_valid_trajectories_per_hour"], 32)
    self.assertEqual(report["reward_only_valid_trajectories_per_hour"], 48)
    reward_only[0]["trajectory"] = {
        "steps": [{"logprobs": [None], "logprob_note": "absent"}]
    }
    parity.build_l3_report(
        observer,
        reward_only,
        observer_wall_secs=1800,
        reward_only_wall_secs=1200,
    )
    reward_only[0]["trajectory"] = {"steps": [{"logprobs": [0.0]}]}
    with self.assertRaisesRegex(ValueError, "contains numeric logprobs"):
      parity.build_l3_report(
          observer,
          reward_only,
          observer_wall_secs=1800,
          reward_only_wall_secs=1200,
      )
    with self.assertRaisesRegex(ValueError, "exact 16"):
      parity.build_l3_report(
          observer[:-1],
          reward_only,
          observer_wall_secs=1800,
          reward_only_wall_secs=1200,
      )


if __name__ == "__main__":
  unittest.main()
