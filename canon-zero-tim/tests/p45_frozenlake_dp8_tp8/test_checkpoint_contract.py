"""Tests for the fail-closed P45 FrozenLake checkpoint contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

_ROOT = Path(__file__).resolve().parents[3]
_RECIPE = _ROOT / "examples/frozenlake/train_frozenlake_qwen3.py"
_RL_CLUSTER = _ROOT / "tunix/rl/rl_cluster.py"
_CONTRACT_PATH = _ROOT / "tunix/rl/frozenlake_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location(
    "frozenlake_checkpoint", _CONTRACT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
frozenlake_checkpoint = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = frozenlake_checkpoint
_SPEC.loader.exec_module(frozenlake_checkpoint)


def _env(mode: str = "new") -> dict[str, str]:
  return {
      "CANON_FROZENLAKE_CKPT_MODE": mode,
      "CANON_FROZENLAKE_CKPT_ROOT": frozenlake_checkpoint.GCS_ROOT,
      "CANON_FROZENLAKE_CKPT_TAG": "fl-prod-001",
      "CANON_FROZENLAKE_CKPT_INTERVAL": "10",
      "CANON_FROZENLAKE_CKPT_MAX_TO_KEEP": "1",
      "ENABLE_PATHWAYS_PERSISTENCE": "1",
      "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
      "CANON_P33_RUN_STAGE": "full",
      "CANON_P33_NO_COMMIT": "0",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
  }


def _contract(config: frozenlake_checkpoint.Config) -> dict[str, object]:
  return frozenlake_checkpoint.build_contract(
      config, {"source_commit": "a" * 40, "mesh_dp": 8, "mesh_tp": 8}
  )


class FrozenLakeCheckpointContractTest(unittest.TestCase):

  def test_new_and_resume_resolve_to_same_campaign_directory(self):
    new = frozenlake_checkpoint.from_env(_env("new"))
    resume = frozenlake_checkpoint.from_env(_env("resume"))
    self.assertEqual(
        new.directory,
        "gs://yuxzhang-tunix-models/canon-zero-tim/checkpoints/"
        "frozenlake/fl-prod-001",
    )
    self.assertEqual(new.directory, resume.directory)
    frozenlake_checkpoint.require_p45(new, _env("new"))
    frozenlake_checkpoint.require_p45(resume, _env("resume"))
    self.assertNotIn("checkpoint_milestone_interval", _contract(new))

  def test_disabled_rejects_partial_contract(self):
    self.assertFalse(frozenlake_checkpoint.from_env({}).enabled)
    with self.assertRaisesRegex(ValueError, "explicit new/resume mode"):
      frozenlake_checkpoint.from_env(
          {"CANON_FROZENLAKE_CKPT_TAG": "fl-prod-001"}
      )

  def test_rejects_drifted_bounds_and_storage(self):
    cases = (
        ("CANON_FROZENLAKE_CKPT_ROOT", "gs://wrong", "root drifted"),
        ("CANON_FROZENLAKE_CKPT_TAG", "Bad/Tag", "tag must be"),
        ("CANON_FROZENLAKE_CKPT_INTERVAL", "11", "expected 10"),
        ("CANON_FROZENLAKE_CKPT_MAX_TO_KEEP", "2", "exactly one"),
        (
            "CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL",
            "25",
            "disabled or 50",
        ),
        ("ENABLE_PATHWAYS_PERSISTENCE", "0", "requires Pathways"),
    )
    for key, value, message in cases:
      with self.subTest(key=key):
        env = _env()
        env[key] = value
        with self.assertRaisesRegex(ValueError, message):
          frozenlake_checkpoint.from_env(env)

  def test_active_p57_primary_uses_only_the_final_checkpoint(self):
    for arm in ("mismatch", "is", "zero"):
      for candidate, split in (("", ""), ("m15", "main")):
        with self.subTest(arm=arm, candidate=candidate):
          env = _env()
          env.update({
              "CANON_P57_RUN_KIND": "train",
              "CANON_P57_TIM_ARM": arm,
              "CANON_P57_EXPECTED_UPDATES": "300",
              "CANON_P57_WORKLOAD_CANDIDATE": candidate,
              "CANON_P57_DATA_SPLIT": split,
              "CANON_FROZENLAKE_CKPT_INTERVAL": "300",
          })
          config = frozenlake_checkpoint.from_env(env)
          self.assertEqual(config.interval, 300)
          policy = frozenlake_checkpoint.build_preservation_policy(config)
          self.assertEqual(policy.n, 1)

          from orbax.checkpoint import v1 as ocp
          interval_policy = (
              ocp.training.save_decision_policies.FixedIntervalPolicy(
                  config.interval
              )
          )
          self.assertFalse(
              interval_policy.should_save(
                  SimpleNamespace(step=299), (), context=None
              )
          )
          self.assertTrue(
              interval_policy.should_save(
                  SimpleNamespace(step=300), (), context=None
              )
          )

  def test_final_only_interval_is_scoped_to_registered_p57_primary(self):
    base = {
        "CANON_P57_RUN_KIND": "train",
        "CANON_P57_TIM_ARM": "mismatch",
        "CANON_P57_EXPECTED_UPDATES": "300",
        "CANON_P57_WORKLOAD_CANDIDATE": "",
        "CANON_P57_DATA_SPLIT": "",
    }
    mutations = (
        ("CANON_P57_RUN_KIND", "calibration"),
        ("CANON_P57_TIM_ARM", "unknown"),
        ("CANON_P57_EXPECTED_UPDATES", "200"),
        ("CANON_P57_WORKLOAD_CANDIDATE", "m15"),
        ("CANON_P57_DATA_SPLIT", "selection"),
    )
    for key, value in mutations:
      with self.subTest(key=key):
        env = _env()
        env.update(base)
        env[key] = value
        env["CANON_FROZENLAKE_CKPT_INTERVAL"] = "300"
        with self.assertRaisesRegex(ValueError, "expected 10"):
          frozenlake_checkpoint.from_env(env)

  def test_active_p57_primary_rejects_legacy_ten_step_interval(self):
    env = _env()
    env.update({
        "CANON_P57_RUN_KIND": "train",
        "CANON_P57_TIM_ARM": "mismatch",
        "CANON_P57_EXPECTED_UPDATES": "300",
        "CANON_P57_WORKLOAD_CANDIDATE": "m15",
        "CANON_P57_DATA_SPLIT": "main",
    })
    with self.assertRaisesRegex(ValueError, "expected 300"):
      frozenlake_checkpoint.from_env(env)

  def test_p57_milestones_keep_latest_one_plus_every_fifty(self):
    env = _env()
    env.update({
        "CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL": "50",
        "CANON_P57_EXPECTED_UPDATES": "450",
        "CANON_P57_RUN_KIND": "train",
        "CANON_P57_TIM_ARM": "mismatch",
    })
    config = frozenlake_checkpoint.from_env(env)
    policy = frozenlake_checkpoint.build_preservation_policy(config)
    self.assertEqual(config.milestone_interval, 50)
    self.assertEqual(len(policy.policies), 2)
    self.assertEqual(policy.policies[0].n, 1)
    self.assertEqual(policy.policies[1].interval_steps, 50)
    self.assertTrue(policy.policies[1].exact_interval)
    self.assertEqual(_contract(config)["checkpoint_milestone_interval"], 50)

  def test_milestones_reject_non_p57_or_wrong_horizon(self):
    for key, value in (
        ("CANON_P57_EXPECTED_UPDATES", "200"),
        ("CANON_P57_RUN_KIND", ""),
        ("CANON_P57_TIM_ARM", "unknown"),
    ):
      with self.subTest(key=key):
        env = _env()
        env.update({
            "CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL": "50",
            "CANON_P57_EXPECTED_UPDATES": "450",
            "CANON_P57_RUN_KIND": "train",
            "CANON_P57_TIM_ARM": "mismatch",
        })
        env[key] = value
        with self.assertRaisesRegex(ValueError, "isolated to"):
          frozenlake_checkpoint.from_env(env)

  def test_requires_committed_resident_p45(self):
    for key, value in (
        ("CANON_P32_WORKLOAD", "frozenlake"),
        ("CANON_P33_RUN_STAGE", "gate"),
        ("CANON_P33_NO_COMMIT", "1"),
        ("CANON_OPT_STATE_RESIDENT", "0"),
        ("CANON_P30_OPT_STATE_OFFLOAD", "1"),
    ):
      with self.subTest(key=key):
        env = _env()
        env[key] = value
        config = frozenlake_checkpoint.from_env(env)
        with self.assertRaisesRegex(ValueError, "workload contract drifted"):
          frozenlake_checkpoint.require_p45(config, env)

  def test_latest_checkpoint_mode_is_fail_closed(self):
    new = frozenlake_checkpoint.from_env(_env("new"))
    resume = frozenlake_checkpoint.from_env(_env("resume"))
    frozenlake_checkpoint.validate_latest(new, None)
    frozenlake_checkpoint.validate_latest(resume, 10)
    with self.assertRaisesRegex(ValueError, "refuses an existing"):
      frozenlake_checkpoint.validate_latest(new, 10)
    with self.assertRaisesRegex(ValueError, "requires an existing"):
      frozenlake_checkpoint.validate_latest(resume, None)
    with self.assertRaisesRegex(ValueError, "committed boundary"):
      frozenlake_checkpoint.validate_latest(resume, 11)

  def test_restore_requires_optimizer_step_role_and_exact_contract(self):
    config = frozenlake_checkpoint.from_env(_env("resume"))
    contract = _contract(config)
    metadata = {
        "global_step": 20,
        "role": "actor",
        "canon_resume_contract": contract,
    }
    frozenlake_checkpoint.validate_restored(
        config,
        restored_step=20,
        optimizer_restored=True,
        metadata=metadata,
        expected_contract=contract,
    )
    with self.assertRaisesRegex(ValueError, "optimizer state"):
      frozenlake_checkpoint.validate_restored(
          config,
          restored_step=20,
          optimizer_restored=False,
          metadata=metadata,
          expected_contract=contract,
      )
    drifted = {**metadata, "canon_resume_contract": {**contract, "mesh_tp": 4}}
    with self.assertRaisesRegex(ValueError, "contract mismatch"):
      frozenlake_checkpoint.validate_restored(
          config,
          restored_step=20,
          optimizer_restored=True,
          metadata=drifted,
          expected_contract=contract,
      )

  def test_new_mode_rejects_any_restored_state(self):
    config = frozenlake_checkpoint.from_env(_env("new"))
    contract = _contract(config)
    frozenlake_checkpoint.validate_restored(
        config,
        restored_step=0,
        optimizer_restored=False,
        metadata={},
        expected_contract=contract,
    )
    with self.assertRaisesRegex(ValueError, "unexpectedly restored"):
      frozenlake_checkpoint.validate_restored(
          config,
          restored_step=10,
          optimizer_restored=True,
          metadata={"global_step": 10},
          expected_contract=contract,
      )

  def test_resume_sync_is_before_train_and_does_not_advance_step(self):
    recipe = _RECIPE.read_text()
    sync = recipe.index("frozenlake_checkpoint.sync_rollout_for_no_update(")
    train = recipe.index("grpo_trainer.train(")
    self.assertLess(sync, train)
    sync_call = recipe[sync:train]
    self.assertIn("sync_rollout_for_no_update(\n      grpo_trainer,", sync_call)

    cluster = _RL_CLUSTER.read_text()
    resume_start = cluster.index("def sync_weights_for_resume")
    normal_start = cluster.index("def sync_weights(self)", resume_start)
    resume_body = cluster[resume_start:normal_start]
    self.assertIn("_sync_weights_without_advancing_step()", resume_body)
    self.assertNotIn("global_steps += 1", resume_body)

  def test_no_update_sync_uses_honest_stock_and_exact_canonical_gates(self):
    class FakeCluster:
      def __init__(self, *, equal=True):
        self.equal = equal
        self.sync_calls = 0
        self.attest_calls = 0

      def sync_weights_for_resume(self):
        self.sync_calls += 1

      def attest_actor_anchor_matches_engine(self):
        self.attest_calls += 1
        return {"equal": self.equal}

    class FakeLearner:
      def __init__(self, *, should_sync_weights=True, equal=True):
        self.should_sync_weights = should_sync_weights
        self.rl_cluster = FakeCluster(equal=equal)

    stock = FakeLearner()
    stock_receipt = frozenlake_checkpoint.sync_rollout_for_no_update(
        stock, stock_fast=True
    )
    self.assertEqual(
        stock_receipt, frozenlake_checkpoint.P57_STOCK_SYNC_RECEIPT
    )
    self.assertEqual(
        (stock.rl_cluster.sync_calls, stock.rl_cluster.attest_calls), (1, 0)
    )

    canonical = FakeLearner()
    canonical_receipt = frozenlake_checkpoint.sync_rollout_for_no_update(
        canonical, stock_fast=False
    )
    self.assertEqual(canonical_receipt["exact_weight_attestation"], "pass")
    self.assertEqual(
        (canonical.rl_cluster.sync_calls, canonical.rl_cluster.attest_calls),
        (1, 1),
    )

    mismatch = FakeLearner(equal=False)
    with self.assertRaisesRegex(ValueError, "did not match"):
      frozenlake_checkpoint.sync_rollout_for_no_update(
          mismatch, stock_fast=False
      )

    disabled = FakeLearner(should_sync_weights=False)
    with self.assertRaisesRegex(ValueError, "requires an explicit"):
      frozenlake_checkpoint.sync_rollout_for_no_update(
          disabled, stock_fast=True
      )
    self.assertEqual(
        (disabled.rl_cluster.sync_calls, disabled.rl_cluster.attest_calls),
        (0, 0),
    )

  def test_recipe_wires_signed_checkpoint_contract_into_g6_trainer(self):
    recipe = _RECIPE.read_text()
    self.assertIn(
        "precomputed_gradient_checkpointing_contract=(", recipe
    )
    self.assertIn(
        "frozenlake_checkpoint.SCHEMA if P45_CHECKPOINT.enabled", recipe
    )

    trainer = (_ROOT / "tunix/sft/peft_trainer.py").read_text()
    self.assertIn("_p45_precomputed_checkpointing_admitted", trainer)
    self.assertIn("committed ", trainer)
    self.assertIn("P45 checkpoint contract is admitted", trainer)
    self.assertIn("step=self.config.checkpoint_restore_step", trainer)


if __name__ == "__main__":
  unittest.main()
