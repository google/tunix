"""Round-trip tests for P43 real-trajectory debug artifacts."""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "p43_deepswe_debug", ROOT / "tunix/rl/deepswe_debug.py"
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P43 artifact writer")
artifacts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = artifacts
SPEC.loader.exec_module(artifacts)


def _batch():
  raw_rewards = [
      1.0, 1.0, 1.0, 1.0,
      0.0, 0.0, 0.5, 0.0,
      1.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
  ]
  statuses = ["SUCCEEDED"] * 15 + ["TIMEOUT"]
  advantages = [0.0] * 8 + [-1.0, 1.0, -1.0, 1.0] + [0.0] * 4
  trajectories = []
  for index, (reward, status) in enumerate(zip(raw_rewards, statuses)):
    group_id, pair_index = divmod(index, 4)
    trajectories.append(types.SimpleNamespace(
        group_id=group_id,
        pair_index=pair_index,
        traj={
            "status": status,
            "trajectory_reward": reward,
            "conversation_text": [
                {"role": "user", "content": f"prompt-{group_id}"},
                {"role": "assistant", "content": f"answer-{pair_index}"},
                {"role": "tool", "content": "real environment output"},
            ],
            "original_input": {
                "prompts": f"prompt-{group_id}",
                "api_key": "must-not-survive",
            },
            "conversation_tokens": np.asarray([1, 2, 3]),
            "policy_version": 0,
        },
    ))
  return trajectories, raw_rewards, advantages


class P43ArtifactTest(unittest.TestCase):

  def test_group_metrics_and_compressed_round_trip(self):
    trajectories, rewards, advantages = _batch()
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text).resolve()
      metrics = artifacts.persist_batch(
          trajectories,
          rewards,
          advantages,
          expected_step=0,
          output_dir=root,
          model_id="Qwen/Qwen3-8B",
          values={
              "CANON_EXPECT_COMMIT": "1" * 40,
              "CANON_SOURCE_BRANCH": "yuxzhang/canon-zero-tim",
              "CANON_RUN_ID": "unit",
              "CANON_P34_RUN_STAGE": "rollout-only",
          },
      )
      self.assertEqual(metrics["all_solved_prompt_groups"], 1)
      self.assertEqual(metrics["all_failed_prompt_groups"], 1)
      self.assertEqual(metrics["mixed_prompt_groups"], 1)
      self.assertEqual(metrics["incomplete_prompt_groups"], 1)
      self.assertEqual(metrics["solved_trajectories"], 6)
      self.assertEqual(metrics["nonbinary_final_rewards"], 1)
      self.assertEqual(metrics["effective_prompt_groups"], 1)
      self.assertEqual(metrics["nonzero_advantages"], 4)

      trajectory_path = root / "batch-000000.trajectories.jsonl.gz"
      with gzip.open(trajectory_path, "rt", encoding="utf-8") as source:
        records = [json.loads(line) for line in source]
      self.assertEqual(len(records), 16)
      self.assertEqual(
          records[0]["trajectory"]["conversation_text"][2]["content"],
          "real environment output",
      )
      self.assertEqual(
          records[0]["trajectory"]["original_input"]["api_key"],
          "<redacted>",
      )
      manifest = json.loads((root / "run_manifest.json").read_text())
      self.assertEqual(manifest["solve_definition"], artifacts.SOLVE_DEFINITION)
      metric_rows = [
          json.loads(line)
          for line in (root / "batch_metrics.jsonl").read_text().splitlines()
      ]
      self.assertEqual(metric_rows, [metrics])

  def test_existing_batch_is_never_overwritten(self):
    trajectories, rewards, advantages = _batch()
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text).resolve()
      kwargs = dict(
          expected_step=2,
          output_dir=root,
          model_id="Qwen/Qwen3-8B",
          values={"CANON_P34_RUN_STAGE": "three-update"},
      )
      artifacts.persist_batch(trajectories, rewards, advantages, **kwargs)
      original = (root / "batch-000002.trajectories.jsonl.gz").read_bytes()
      with self.assertRaises(FileExistsError):
        artifacts.persist_batch(trajectories, rewards, advantages, **kwargs)
      self.assertEqual(
          (root / "batch-000002.trajectories.jsonl.gz").read_bytes(),
          original,
      )

  def test_nonfinite_reward_is_rejected(self):
    trajectories, rewards, advantages = _batch()
    trajectories[0].traj["trajectory_reward"] = float("nan")
    with tempfile.TemporaryDirectory() as root_text:
      with self.assertRaisesRegex(ValueError, "must be finite"):
        artifacts.persist_batch(
            trajectories,
            rewards,
            advantages,
            expected_step=0,
            output_dir=Path(root_text).resolve(),
            model_id="Qwen/Qwen3-8B",
        )


if __name__ == "__main__":
  unittest.main()
