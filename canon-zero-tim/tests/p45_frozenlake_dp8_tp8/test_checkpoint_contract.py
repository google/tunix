"""Tests for the fail-closed P45 FrozenLake checkpoint contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
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
        ("CANON_FROZENLAKE_CKPT_INTERVAL", "11", "exactly 10"),
        ("CANON_FROZENLAKE_CKPT_MAX_TO_KEEP", "2", "exactly one"),
        ("ENABLE_PATHWAYS_PERSISTENCE", "0", "requires Pathways"),
    )
    for key, value, message in cases:
      with self.subTest(key=key):
        env = _env()
        env[key] = value
        with self.assertRaisesRegex(ValueError, message):
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
    sync = recipe.index("rl_cluster.sync_weights_for_resume()")
    attest = recipe.index("rl_cluster.attest_actor_anchor_matches_engine()", sync)
    train = recipe.index("grpo_trainer.train(")
    self.assertLess(sync, attest)
    self.assertLess(attest, train)

    cluster = _RL_CLUSTER.read_text()
    resume_start = cluster.index("def sync_weights_for_resume")
    normal_start = cluster.index("def sync_weights(self)", resume_start)
    resume_body = cluster[resume_start:normal_start]
    self.assertIn("_sync_weights_without_advancing_step()", resume_body)
    self.assertNotIn("global_steps += 1", resume_body)


if __name__ == "__main__":
  unittest.main()
