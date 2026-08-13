"""Cross-topology tests for P44 durable DeepSWE artifacts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "p44_deepswe_debug_artifacts", ROOT / "tunix/rl/deepswe_debug.py"
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import DeepSWE artifact writer")
artifacts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = artifacts
SPEC.loader.exec_module(artifacts)


def _batch():
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
                {"role": "user", "content": f"prompt-{group}"},
                {"role": "assistant", "content": f"answer-{pair}"},
            ],
        },
    ))
    rewards.append(reward)
    advantages.append(-1.0 if pair % 2 == 0 else 1.0)
  return trajectories, rewards, advantages


def _values(topology: str) -> dict[str, str]:
  return {
      "CANON_P43_DEEPSWE_DEBUG": "0",
      "CANON_P44_DEEPSWE_PARITY": "1",
      "CANON_P44_TOPOLOGY": topology,
      "CANON_EXPECT_COMMIT": "1" * 40,
      "CANON_SOURCE_BRANCH": "yuxzhang/canon-zero-tim",
      "CANON_RUN_ID": f"artifact-{topology}",
      "CANON_P34_RUN_STAGE": "rollout-only",
      "CANON_DEEPSWE_PER_TURN_TIMEOUT_SECS": "300",
      "CANON_DEEPSWE_TRAJECTORY_TIMEOUT_SECS": "3000",
      "CANON_DEEPSWE_STEP_TIMEOUT_SECS": "600",
      "CANON_DEEPSWE_REWARD_TIMEOUT_SECS": "600",
      "CANON_DEEPSWE_CLEANUP_TIMEOUT_SECS": "300",
      "CANON_DEEPSWE_ROLLOUT_BATCH_TIMEOUT_SECS": "3600",
      "R2E_ACTIVE_DEADLINE_SECONDS": "3300",
  }


def _onehost_batch():
  trajectories = []
  for pair, reward in enumerate((0.0, 1.0)):
    trajectories.append(types.SimpleNamespace(
        group_id=0,
        pair_index=pair,
        traj={
            "status": "SUCCEEDED",
            "trajectory_reward": reward,
            "conversation_text": [
                {"role": "user", "content": "one-host-prompt"},
                {"role": "assistant", "content": f"answer-{pair}"},
            ],
        },
    ))
  return trajectories, [0.0, 1.0], [-1.0, 1.0]


def _onehost_values() -> dict[str, str]:
  return {
      "CANON_P43_DEEPSWE_DEBUG": "0",
      "CANON_P44_DEEPSWE_PARITY": "0",
      "CANON_DEEPSWE_ONEHOST_SMOKE": "1",
      "CANON_DEEPSWE_ONEHOST_STAGE": "backward-no-commit",
      "CANON_DEEPSWE_ONEHOST_NO_COMMIT": "1",
      "CANON_EXPECT_COMMIT": "2" * 40,
      "CANON_SOURCE_BRANCH": "yuxzhang/canon-zero-tim",
      "CANON_RUN_ID": "onehost-artifact",
  }


class P44ArtifactTest(unittest.TestCase):

  def test_both_topologies_write_the_same_artifact_schema(self):
    manifests = []
    for topology in ("64", "128"):
      with self.subTest(topology=topology), tempfile.TemporaryDirectory() as text:
        root = Path(text).resolve()
        metrics = artifacts.persist_batch(
            *_batch(),
            expected_step=0,
            output_dir=root,
            model_id="Qwen/Qwen3-4B-Instruct-2507",
            values=_values(topology),
        )
        manifest = json.loads((root / "run_manifest.json").read_text())
        manifests.append(manifest)
        self.assertEqual(manifest["schema"], artifacts.P44_MANIFEST_SCHEMA)
        self.assertEqual(
            manifest["trajectory_schema"], artifacts.P44_TRAJECTORY_SCHEMA
        )
        self.assertEqual(manifest["metrics_schema"], artifacts.P44_METRICS_SCHEMA)
        self.assertEqual(metrics["schema"], artifacts.P44_METRICS_SCHEMA)
        self.assertEqual(metrics["trajectories"], 16)
        self.assertEqual(metrics["prompt_groups"], 4)
        self.assertEqual(manifest["timeouts_seconds"], {
            "per_turn": "300",
            "trajectory": "3000",
            "step": "600",
            "reward": "600",
            "cleanup": "300",
            "rollout_batch": "3600",
            "sandbox_active_deadline": "3300",
        })
    self.assertEqual(manifests[0]["model_id"], manifests[1]["model_id"])
    self.assertEqual(manifests[0]["global_trajectories"], 16)
    self.assertEqual(manifests[1]["global_trajectories"], 16)
    self.assertEqual(manifests[0]["role_topology"]["dp"], 4)
    self.assertEqual(manifests[1]["role_topology"]["dp"], 8)

  def test_mode_helpers_are_fail_closed(self):
    values = {
        **_values("64"),
        "CANON_P44_DEBUG_DIR": "/tmp/p44-debug",
        "CANON_P44_ROLLOUT_ONLY": "1",
    }
    self.assertTrue(artifacts.enabled(values))
    self.assertTrue(artifacts.rollout_only(values))
    self.assertEqual(artifacts.artifact_directory(values), "/tmp/p44-debug")
    self.assertEqual(artifacts.marker_prefix(values), "P44")
    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
      artifacts.enabled({
          "CANON_P43_DEEPSWE_DEBUG": "1",
          "CANON_P44_DEEPSWE_PARITY": "1",
      })

  def test_onehost_writes_local_geometry_and_group_metrics(self):
    values = {
        **_onehost_values(),
        "CANON_DEEPSWE_ONEHOST_DEBUG_DIR": "/tmp/deepswe-onehost",
        "CANON_DEEPSWE_ONEHOST_ROLLOUT_ONLY": "0",
    }
    with tempfile.TemporaryDirectory() as text:
      root = Path(text).resolve()
      metrics = artifacts.persist_batch(
          *_onehost_batch(),
          expected_step=0,
          output_dir=root,
          model_id="Qwen/Qwen3-4B-Instruct-2507",
          values=values,
      )
      manifest = json.loads((root / "run_manifest.json").read_text())
    self.assertEqual(manifest["schema"], artifacts.ONEHOST_MANIFEST_SCHEMA)
    self.assertEqual(manifest["contract_name"], "local-qwen4b-dp1-tp4")
    self.assertEqual(manifest["role_topology"], {"dp": 1, "tp": 4, "devices": 4})
    self.assertEqual(manifest["global_trajectories"], 2)
    self.assertEqual(metrics["trajectories"], 2)
    self.assertEqual(metrics["prompt_groups"], 1)
    self.assertEqual(metrics["mixed_prompt_groups"], 1)
    self.assertEqual(metrics["trajectory_solve_ratio"], 0.5)
    self.assertTrue(artifacts.onehost(values))
    self.assertTrue(artifacts.no_commit(values))
    self.assertEqual(artifacts.marker_prefix(values), "DEEPSWE.ONEHOST")

  def test_onehost_is_mutually_exclusive_and_model_pinned(self):
    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
      artifacts.enabled({
          **_onehost_values(),
          "CANON_P44_DEEPSWE_PARITY": "1",
      })
    with tempfile.TemporaryDirectory() as text:
      with self.assertRaisesRegex(ValueError, "4B-Instruct-2507"):
        artifacts.persist_batch(
            *_onehost_batch(),
            expected_step=0,
            output_dir=Path(text).resolve(),
            model_id="Qwen/Qwen3-4B",
            values=_onehost_values(),
        )

  def test_wrong_model_or_topology_is_rejected(self):
    with tempfile.TemporaryDirectory() as text:
      with self.assertRaisesRegex(ValueError, "Qwen/Qwen3-4B-Instruct-2507"):
        artifacts.persist_batch(
            *_batch(),
            expected_step=0,
            output_dir=Path(text).resolve(),
            model_id="Qwen/Qwen3-8B",
            values=_values("64"),
        )
    with tempfile.TemporaryDirectory() as text:
      with self.assertRaisesRegex(ValueError, "exactly 64 or 128"):
        artifacts.persist_batch(
            *_batch(),
            expected_step=0,
            output_dir=Path(text).resolve(),
            model_id="Qwen/Qwen3-4B-Instruct-2507",
            values=_values("256"),
        )


if __name__ == "__main__":
  unittest.main()
