#!/usr/bin/env python3
"""P58 native-dose and zero-exact run-classifier controls."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest

from tunix.rl import deepswe_debug


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "p58_classifier", Path(__file__).with_name("classify_run.py")
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P58 classifier")
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)

_WANDB_PASS = "[CANON_" "P34_WANDB] ONLINE_RUN_PASS\n"


def _values(root: Path, arm: str) -> dict[str, str]:
  return {
      "CANON_P34_DEEPSWE": "1",
      "CANON_P58_DEEPSWE_TIM": "1",
      "CANON_P58_TIM_ARM": arm,
      "CANON_P58_DEBUG_DIR": str(root),
      "CANON_EXPECT_COMMIT": "1" * 40,
      "CANON_SOURCE_BRANCH": "yuxzhang/canon-zero-tim",
      "CANON_RUN_ID": "classifier-test",
      "CANON_P34_RUN_STAGE": "three-update",
      "CANON_P34_CLEAN_ROWS": "1012",
      "CANON_P34_WHITELIST_SHA256": classifier._WHITELIST_SHA256,
  }


def _batch():
  items = []
  rewards = []
  advantages = []
  for group in range(8):
    for pair in range(16):
      reward = float(pair == 0)
      items.append(types.SimpleNamespace(
          group_id=f"group-{group}",
          pair_index=pair,
          metadata={"task_identity": {"docker_image": f"task-{group}"}},
          traj={
              "status": "SUCCEEDED",
              "trajectory_reward": reward,
              "conversation_text": [],
          },
      ))
      rewards.append(reward)
      advantages.append(1.0 if pair == 0 else -1.0 / 15.0)
  return items, rewards, advantages


def _boundary(differing: int) -> dict:
  return {"valid": True, "finite": True, "differing_bytes": differing}


def _pre(arm: str) -> dict:
  return {
      "verdict": "PASS_WITH_ALIGNMENT_WARNINGS" if arm == "native" else "PASS",
      "blocking_reds": [],
      "boundaries": {
          "S_decode_vs_S_prefill": _boundary(1 if arm == "native" else 0),
          "S_prefill_vs_T_old": _boundary(1 if arm == "native" else 0),
      },
  }


def _post(arm: str) -> dict:
  return {
      "verdict": "PASS_WITH_ALIGNMENT_WARNINGS" if arm == "native" else "PASS",
      "blocking_reds": [],
      "boundaries": {
          "S_decode_vs_S_prefill": _boundary(1 if arm == "native" else 0),
          "S_prefill_vs_T_old": _boundary(1 if arm == "native" else 0),
          "T_old_vs_T_current": _boundary(0),
      },
  }


def _update(step: int, arm: str) -> dict:
  record = {
      "contract_name": "p58-qwen4b-tim-128",
      "dp_size": 8,
      "tp_size": 8,
      "global_m": 2048,
      "verdict": "PASS",
      "commits": 1,
      "train_steps_before": step,
      "train_steps_after": step + 1,
      "gradient_finite": True,
      "dp_replicas_exact": True,
      "dp_reduction_transactions": 16,
      "dp_reduction_rounds_per_transaction": 6,
      "dp_rank_pullbacks_per_transaction": 8,
      "optimizer_placement": "device-resident",
  }
  if arm == "native":
    record["dp_reduction_mode"] = "stock-jax-sharded-trainer"
  else:
    record.update({
        "dp_replicas_exact": True,
        "dp_reduction_transactions": 16,
        "dp_reduction_rounds_per_transaction": 6,
        "dp_rank_pullbacks_per_transaction": 8,
    })
  return record


class P58ClassifierTest(unittest.TestCase):

  def _classify(self, root: Path, arm: str):
    with contextlib.redirect_stdout(io.StringIO()):
      for step in range(3):
        deepswe_debug.persist_batch(
            *_batch(),
            expected_step=step,
            optimizer_step=step,
            output_dir=root,
            model_id="Qwen/Qwen3-4B-Instruct-2507",
            values=_values(root, arm),
        )
    return classifier.classify(
        arm=arm,
        stage="three-update",
        log_text=_WANDB_PASS,
        debug_dir=root,
        weights=[{"verdict": "PASS", "equal": True}],
        pre_alignment=[_pre(arm)],
        alignment=[_post(arm)],
        updates=[_update(step, arm) for step in range(3)],
    )

  def test_native_requires_a_finite_nonzero_treatment_dose(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      report = self._classify(root, "native")
      self.assertEqual(report["verdict"], "PASS")
      zero_dose = [_pre("zero")]
      failed = classifier.classify(
          arm="native",
          stage="three-update",
          log_text=_WANDB_PASS,
          debug_dir=root,
          weights=[{"verdict": "PASS", "equal": True}],
          pre_alignment=zero_dose,
          alignment=[_post("zero")],
          updates=[_update(step, "native") for step in range(3)],
      )
      self.assertIn("registered_treatment_observed", failed["failed"])

  def test_native_b_c_only_is_a_registered_treatment_dose(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self._classify(root, "native")
      b_c_only = _pre("native")
      b_c_only["boundaries"]["S_decode_vs_S_prefill"] = _boundary(0)
      report = classifier.classify(
          arm="native",
          stage="three-update",
          log_text=_WANDB_PASS,
          debug_dir=root,
          weights=[{"verdict": "PASS", "equal": True}],
          pre_alignment=[b_c_only],
          alignment=[_post("native")],
          updates=[_update(step, "native") for step in range(3)],
      )
      self.assertNotIn("registered_treatment_observed", report["failed"])

  def test_native_trainer_repeat_drift_remains_blocking(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self._classify(root, "native")
      drift = _post("native")
      drift["boundaries"]["T_old_vs_T_current"] = _boundary(1)
      failed = classifier.classify(
          arm="native",
          stage="three-update",
          log_text=_WANDB_PASS,
          debug_dir=root,
          weights=[{"verdict": "PASS", "equal": True}],
          pre_alignment=[_pre("native")],
          alignment=[drift],
          updates=[_update(step, "native") for step in range(3)],
      )
      self.assertIn("native_trainer_repeat_exact", failed["failed"])

  def test_zero_requires_all_boundaries_exact(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      report = self._classify(root, "zero")
      self.assertEqual(report["verdict"], "PASS")
      drift = _post("zero")
      drift["boundaries"]["T_old_vs_T_current"] = _boundary(1)
      failed = classifier.classify(
          arm="zero",
          stage="three-update",
          log_text=_WANDB_PASS,
          debug_dir=root,
          weights=[{"verdict": "PASS", "equal": True}],
          pre_alignment=[_pre("zero")],
          alignment=[drift],
          updates=[_update(step, "zero") for step in range(3)],
      )
      self.assertIn("zero_all_boundaries_exact", failed["failed"])


if __name__ == "__main__":
  unittest.main()
