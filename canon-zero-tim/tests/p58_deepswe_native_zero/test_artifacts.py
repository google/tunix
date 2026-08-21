#!/usr/bin/env python3
"""P58 durable batch-index and optimizer-step journal contracts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import types
import unittest

from tunix.rl import deepswe_debug


def _values(root: Path) -> dict[str, str]:
  return {
      "CANON_P34_DEEPSWE": "1",
      "CANON_P58_DEEPSWE_TIM": "1",
      "CANON_P58_TIM_ARM": "native",
      "CANON_P58_DEBUG_DIR": str(root),
      "CANON_EXPECT_COMMIT": "1" * 40,
      "CANON_SOURCE_BRANCH": "yuxzhang/canon-zero-tim",
      "CANON_RUN_ID": "artifact-test",
      "CANON_P34_RUN_STAGE": "three-update",
      "CANON_P34_DATASET_NAME": "R2E-Gym/R2E-Gym-Subset",
      "CANON_P34_DATASET_REVISION": "2e8108ff942f24fcb5686badfaf7f9a8808566d5",
      "CANON_P34_DATASET_SPLIT": "train",
      "CANON_P34_DATASET_ROWS": "4578",
      "CANON_P34_CLEAN_ROWS": "1012",
      "CANON_P34_WHITELIST_SHA256": (
          "ec297c9cbc39cd67db15b0b9db6a229b15671b848df5ec3101de9ef8df7c9973"
      ),
  }


def _batch(
    status: str = "SUCCEEDED",
    *,
    timeout_stage: str = "",
    timeout_scheduler_reason: str = "",
    timeout_resource: str = "",
):
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
              "status": status,
              "timeout_stage": timeout_stage,
              "timeout_scheduler_reason": timeout_scheduler_reason,
              "timeout_resource": timeout_resource,
              "trajectory_reward": reward,
              "conversation_text": [
                  {"role": "assistant", "content": "redacted test"}
              ],
          },
      ))
      rewards.append(reward)
      advantages.append(1.0 if pair == 0 else -1.0 / 15.0)
  return items, rewards, advantages


class P58ArtifactTest(unittest.TestCase):

  def test_sandbox_start_timeouts_are_bounded_wandb_metrics(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      metrics = deepswe_debug.persist_batch(
          *_batch(
              "ENV_TIMEOUT",
              timeout_stage="sandbox_start",
              timeout_scheduler_reason="unschedulable",
              timeout_resource="cpu",
          ),
          expected_step=0,
          optimizer_step=0,
          output_dir=root,
          model_id="Qwen/Qwen3-4B-Instruct-2507",
          values=_values(root),
      )
      self.assertEqual(metrics["env_timeout_trajectories"], 128)
      self.assertEqual(metrics["sandbox_start_timeout_trajectories"], 128)
      self.assertEqual(metrics["unschedulable_trajectories"], 128)
      self.assertEqual(metrics["insufficient_cpu_trajectories"], 128)
      self.assertTrue(metrics["all_env_timeout_batch"])
      self.assertTrue(metrics["all_sandbox_start_timeout_batch"])
      wandb_metrics = deepswe_debug.timeout_wandb_metrics(metrics)
      self.assertEqual(wandb_metrics["deepswe/env_timeout_ratio"], 1.0)
      self.assertEqual(
          wandb_metrics["deepswe/sandbox_start_timeout_ratio"], 1.0
      )
      self.assertEqual(wandb_metrics["deepswe/all_env_timeout_batch"], 1.0)
      self.assertEqual(
          wandb_metrics["deepswe/all_sandbox_start_timeout_batch"], 1.0
      )

  def test_batch_index_survives_no_commit_and_resume(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      values = _values(root)
      first = deepswe_debug.persist_batch(
          *_batch("MAX_CONTEXT_LIMIT_REACHED"),
          expected_step=0,
          optimizer_step=0,
          output_dir=root,
          model_id="Qwen/Qwen3-4B-Instruct-2507",
          values=values,
      )
      self.assertEqual(first["step"], 0)
      self.assertEqual(first["optimizer_step"], 0)
      self.assertEqual(first["compact_filtered_trajectories"], 128)
      self.assertEqual(first["effective_prompt_groups"], 0)
      self.assertEqual(first["nonzero_advantages"], 0)
      self.assertEqual(first["raw_nonzero_advantages"], 128)
      self.assertEqual(deepswe_debug.next_batch_index(root), 1)

      second = deepswe_debug.persist_batch(
          *_batch(),
          expected_step=1,
          optimizer_step=0,
          output_dir=root,
          model_id="Qwen/Qwen3-4B-Instruct-2507",
          values=values,
      )
      self.assertEqual(second["step"], 1)
      self.assertEqual(second["optimizer_step"], 0)
      self.assertEqual(deepswe_debug.next_batch_index(root), 2)
      with gzip.open(
          root / "batch-000001.trajectories.jsonl.gz", "rt", encoding="utf-8"
      ) as source:
        records = [json.loads(line) for line in source if line.strip()]
      self.assertEqual(len(records), 128)
      self.assertTrue(all(record["optimizer_step"] == 0 for record in records))

  def test_partial_journal_fails_closed(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      root.joinpath("batch_metrics.jsonl").write_text(
          json.dumps({"step": 0}) + "\n", encoding="utf-8"
      )
      with self.assertRaisesRegex(ValueError, "partial"):
        deepswe_debug.next_batch_index(root)


if __name__ == "__main__":
  unittest.main()
