"""Positive and negative controls for the P43 debug classifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[3]


def _load(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


classifier = _load(
    "p43_debug_classifier",
    ROOT / "canon-zero-tim/tests/p43_deepswe_debug/classify_run.py",
)
artifacts = _load("p43_debug_artifacts", ROOT / "tunix/rl/deepswe_debug.py")


def _trajectory_batch():
  trajectories = []
  rewards = []
  advantages = []
  for index in range(16):
    group, pair = divmod(index, 4)
    reward = float(pair % 2)
    trajectories.append(types.SimpleNamespace(
        group_id=group,
        pair_index=pair,
        traj={
            "status": "SUCCEEDED",
            "trajectory_reward": reward,
            "conversation_text": [
                {"role": "user", "content": f"p{group}"},
                {"role": "assistant", "content": f"a{pair}"},
            ],
        },
    ))
    rewards.append(reward)
    advantages.append(-1.0 if pair % 2 == 0 else 1.0)
  return trajectories, rewards, advantages


def _log(stage, batches):
  lines = [
      "[entrypoint] JOBSET_ATTEMPT 0 (first attempt)",
      "[P34.PATHWAYS] initialized_once=1 before_jax=1",
      "[P34.CLI] PASS model=Qwen3-8B prompts=4 generations=4",
      "[sync] provenance ok",
      "[P34.TOPOLOGY] PASS",
      "[CANON_P34_WANDB] ONLINE_RUN_PASS",
      "Prepared token paddings: [1024]",
      "Precompile worker0 backbone --> {'num_tokens': 1024, 'num_reqs': 16}",
  ]
  lines.extend("[P43.TRAJECTORY_BATCH]" for _ in range(batches))
  lines.extend("[P43.BATCH_METRICS_JSON]" for _ in range(batches))
  if stage == "rollout-only":
    lines.append("[P43.ROLLOUT_ONLY] PASS")
  return "\n".join(lines)


def _policy():
  return {
      "id": "deepswe-pilot-alignment-warning-v1",
      "claim_level": "convergence-only",
  }


def _weight():
  return {
      "verdict": "PASS",
      "equal": True,
      "mesh_shape": {"dp": 4, "tp": 8},
      "mesh_device_ids": list(range(32)),
  }


def _pre():
  return {
      "verdict": "PASS_WITH_ALIGNMENT_WARNINGS",
      "blocking_reds": [],
      "N_action": 10,
      "admission_policy": _policy(),
  }


def _alignment():
  return {
      "verdict": "PASS_WITH_ALIGNMENT_WARNINGS",
      "blocking_reds": [],
      "ratio_finite": True,
      "gradient": {"finite": True},
      "admission_policy": _policy(),
  }


def _update(step):
  limit = 100 * 1024**3
  free = 10 * 1024**3
  snapshot = [
      {
          "device": index,
          "peak_bytes_in_use": limit - free,
          "bytes_limit": limit,
      }
      for index in range(32)
  ]
  return {
      "contract_name": "p43-64chip-debug",
      "dp_size": 4,
      "tp_size": 8,
      "global_m": 1024,
      "verdict": "PASS",
      "commits": 1,
      "train_steps_before": step,
      "train_steps_after": step + 1,
      "gradient_finite": True,
      "gradient_activity": [True] * 4,
      "dp_replicas_exact": True,
      "dp_reduction_transactions": 4,
      "dp_reduction_rounds_per_transaction": 4,
      "dp_rank_pullbacks_per_transaction": 4,
      "optimizer_placement": "device-resident",
      "optimizer_memory_kinds_before": ["device"],
      "optimizer_memory_kinds_after": ["device"],
      "optimizer_transaction_valid": True,
      "hbm_before": snapshot,
      "hbm_after_accumulation": snapshot,
      "hbm_after_commit": snapshot,
  }


class P43ClassifierTest(unittest.TestCase):

  def _artifacts(self, root, *, stage, batches):
    trajectory_batch = _trajectory_batch()
    for step in range(batches):
      artifacts.persist_batch(
          *trajectory_batch,
          expected_step=step,
          output_dir=root,
          model_id="Qwen/Qwen3-8B",
          values={
              "CANON_EXPECT_COMMIT": "1" * 40,
              "CANON_SOURCE_BRANCH": "yuxzhang/canon-zero-tim",
              "CANON_RUN_ID": "classify",
              "CANON_P34_RUN_STAGE": stage,
          },
      )

  def _classify(self, root, *, stage):
    updates_count = classifier._STAGE_UPDATES[stage]
    batches = max(1, updates_count)
    return classifier.classify(
        log_text=_log(stage, batches),
        debug_dir=root,
        weight_attestations=[_weight() for _ in range(updates_count)],
        pre_alignment=[_pre() for _ in range(updates_count)],
        alignment=[_alignment() for _ in range(updates_count * 4)],
        updates=[_update(step) for step in range(updates_count)],
        stage=stage,
    )

  def test_rollout_only_passes_without_update_evidence(self):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text).resolve()
      self._artifacts(root, stage="rollout-only", batches=1)
      report = self._classify(root, stage="rollout-only")
      self.assertEqual(report["verdict"], "PASS")
      self.assertTrue(report["checks"]["rollout_only_boundary"])

  def test_three_sequential_updates_pass(self):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text).resolve()
      self._artifacts(root, stage="three-update", batches=3)
      report = self._classify(root, stage="three-update")
      self.assertEqual(report["verdict"], "PASS")
      self.assertTrue(report["checks"]["monotonic_train_steps"])

  def test_missing_trajectory_batch_is_rejected(self):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text).resolve()
      self._artifacts(root, stage="one-update", batches=1)
      (root / "batch-000000.trajectories.jsonl.gz").unlink()
      report = self._classify(root, stage="one-update")
      self.assertIn("trajectory_batch_count", report["failed"])

  def test_nonmonotonic_commits_are_rejected(self):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text).resolve()
      self._artifacts(root, stage="three-update", batches=3)
      updates = [_update(step) for step in range(3)]
      updates[2]["train_steps_before"] = 9
      report = classifier.classify(
          log_text=_log("three-update", 3),
          debug_dir=root,
          weight_attestations=[_weight() for _ in range(3)],
          pre_alignment=[_pre() for _ in range(3)],
          alignment=[_alignment() for _ in range(12)],
          updates=updates,
          stage="three-update",
      )
      self.assertIn("monotonic_train_steps", report["failed"])


if __name__ == "__main__":
  unittest.main()
