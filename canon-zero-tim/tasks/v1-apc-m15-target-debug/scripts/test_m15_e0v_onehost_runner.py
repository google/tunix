#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
ARM = Path(__file__).with_name("run_m15_e0v_tito_onehost_arm.sh")
PAIR = Path(__file__).with_name("run_m15_e0v_tito_onehost_pair.sh")


class OnehostRunnerContractTest(unittest.TestCase):

  def test_arm_is_exact_three_round_zero_commit_dp1tp4(self):
    text = ARM.read_text(encoding="utf-8")
    for fragment in (
        "-e CANON_M15_TOKEN_CONTINUITY=exact",
        "-e CANON_P38_DIAGNOSTIC_ROUNDS=3",
        "-e CANON_P38_ONEHOST_REHEARSAL=1",
        "-e CANON_P32_WORKLOAD=",
        "-e CANON_DP_SIZE=1 -e CANON_TP_SIZE=4",
        "--mesh_dp=1 --mesh_tp=4",
        "-e CANON_ALIGNMENT_TRAIN=0",
        "classify_m15_e0v_onehost_arm.py",
        "--scope onehost",
        "docker_rc\" -ne 42",
        "test ! -e \"$state\"",
        "--p57_workload_candidate=m15 --p57_data_split=main",
        "--vllm_max_num_batched_tokens=256 --env_max_steps=15",
    ):
      self.assertIn(fragment, text)
    self.assertIn("-e CANON_APC_M15_TARGET_DEBUG=", text)
    self.assertNotIn("-e CANON_APC_M15_TARGET_DEBUG=off", text)
    self.assertNotIn("-e CANON_APC_M15_TARGET_DEBUG=on", text)
    self.assertNotIn("--observer kv", text)

  def test_pair_is_sequential_fail_closed_and_never_remote(self):
    text = PAIR.read_text(encoding="utf-8")
    self.assertLess(
        text.index('bash "$arm_runner" off'),
        text.index('bash "$arm_runner" on'),
    )
    for fragment in (
        "PAIR_STATUS.json",
        "status\": \"INCONCLUSIVE",
        "target_executed\": False",
        "numerical_repair_authorized\": False",
        "classify_m15_e0v_onehost_pair.py",
        "test ! -e \"$root\"",
    ):
      self.assertIn(fragment, text)
    for forbidden in ("kubectl", "gcloud", "gsutil", "rm -", "gs://"):
      self.assertNotIn(forbidden, text)

  def test_target_identity_still_rejects_onehost_masquerade(self):
    target_test = (
        ROOT
        / "tasks/v1-apc-m15-target-debug/scripts/test_target_carrier.py"
    ).read_text(encoding="utf-8")
    self.assertIn(
        "test_m15_target_cannot_masquerade_as_onehost_rehearsal",
        target_test,
    )
    self.assertIn("not a one-host rehearsal", target_test)
    rollout = (ROOT.parent / "tunix/rl/rollout/vllm_rollout.py").read_text(
        encoding="utf-8"
    )
    self.assertIn(
        'os.environ.get("CANON_M15_TOKEN_CONTINUITY", "")\n'
        '            in ("verify", "exact")',
        rollout,
    )
    self.assertIn('os.environ.get("CANON_P38_ONEHOST_REHEARSAL", "0") == "1"', rollout)
    self.assertIn("all_num_cached_tokens_zero=True", rollout)
    entrypoint = (
        ROOT.parent / "examples/frozenlake/train_frozenlake_qwen3.py"
    ).read_text(encoding="utf-8")
    self.assertIn("_M15_ONEHOST_TOKEN_CONTINUITY", entrypoint)
    self.assertIn("token_continuity_lib.m15_token_continuity_mode", entrypoint)


if __name__ == "__main__":
  unittest.main()
