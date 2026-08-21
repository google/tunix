"""Checkpoint provenance gates for the isolated P57 evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tunix/rl/frozenlake_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("p57_frozenlake_checkpoint", MODULE_PATH)
assert SPEC and SPEC.loader
checkpoint = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checkpoint
SPEC.loader.exec_module(checkpoint)


def _env(*, arm="zero", step="20"):
  return {
      "CANON_EXPECT_COMMIT": "a" * 40,
      "CANON_P57_TIM_ARM": arm,
      "CANON_P38_FIXED_LM_HEAD": "1" if arm == "zero" else "0",
      "CANON_P57_EVAL_CHECKPOINT_STEP": step,
      "CANON_P57_WORKLOAD_CANDIDATE": "",
      "CANON_P57_DATA_SPLIT": "",
  }


def _config(arm="zero"):
  return checkpoint.Config(
      mode="resume",
      root=checkpoint.GCS_ROOT,
      tag=f"p57-campaign-{arm}",
      interval=10,
      max_to_keep=1,
  )


def _metadata(config, *, arm="zero", step=20):
  return {
      "global_step": step,
      "role": "actor",
      "canon_resume_contract": {
          "schema": checkpoint.SCHEMA,
          "checkpoint_root": config.root,
          "checkpoint_tag": config.tag,
          "source_commit": "a" * 40,
          "profile": "qwen3-8b-dp8-tp8-frozenlake-tim",
          "workload": "frozenlake-dp8-tp8",
          "model_version": "Qwen/Qwen3-8B",
          "model_dir_name": "qwen8b_tp8",
          "mesh_dp": 8,
          "mesh_tp": 8,
          "p57_tim_arm": arm,
          "p57_fixed_lm_head": "1" if arm == "zero" else "0",
          "p57_workload_candidate": "",
          "p57_data_split": "",
          "eval_enabled": False,
      },
  }


class P57CheckpointEvalTest(unittest.TestCase):

  def test_accepts_exact_training_provenance(self):
    for arm in ("zero", "mismatch"):
      with self.subTest(arm=arm):
        config = _config(arm)
        checkpoint.validate_p57_evaluation_restored(
            config,
            restored_step=20,
            metadata=_metadata(config, arm=arm),
            env=_env(arm=arm),
        )

  def test_rejects_wrong_step_role_or_treatment(self):
    config = _config()
    cases = []
    wrong_role = _metadata(config)
    wrong_role["role"] = "reference"
    cases.append((20, wrong_role, _env(), "actor"))
    wrong_treatment = _metadata(config)
    wrong_treatment["canon_resume_contract"]["p57_fixed_lm_head"] = "0"
    cases.append((20, wrong_treatment, _env(), "provenance"))
    cases.append((10, _metadata(config), _env(), "wrong checkpoint"))
    for restored, metadata, env, message in cases:
      with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
        checkpoint.validate_p57_evaluation_restored(
            config,
            restored_step=restored,
            metadata=metadata,
            env=env,
        )

  def test_rejects_nonboundary_or_new_mode(self):
    config = _config()
    with self.assertRaisesRegex(ValueError, "boundary"):
      checkpoint.validate_p57_evaluation_restored(
          config,
          restored_step=20,
          metadata=_metadata(config),
          env=_env(step="25"),
      )
    new_config = checkpoint.Config(
        mode="new",
        root=config.root,
        tag=config.tag,
        interval=10,
        max_to_keep=1,
    )
    with self.assertRaisesRegex(ValueError, "mode=resume"):
      checkpoint.validate_p57_evaluation_restored(
          new_config,
          restored_step=20,
          metadata=_metadata(config),
          env=_env(),
      )

  def test_step_zero_requires_clean_new_mode(self):
    config = checkpoint.Config(
        mode="new",
        root=checkpoint.GCS_ROOT,
        tag="p57-campaign-zero",
        interval=10,
        max_to_keep=1,
    )
    checkpoint.validate_p57_evaluation_restored(
        config,
        restored_step=0,
        metadata={},
        env=_env(step="0"),
    )
    with self.assertRaisesRegex(ValueError, "unexpectedly restored"):
      checkpoint.validate_p57_evaluation_restored(
          config,
          restored_step=0,
          metadata={"global_step": 0},
          env=_env(step="0"),
      )


if __name__ == "__main__":
  unittest.main()
