# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import os
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
import numpy as np
from examples.math_gsm8k import sub_batch_bench_hooks
from tunix.rl import sub_batch_checkpoint
from tunix.rl.agentic.agents import agent_types

_GEOMETRY = sub_batch_checkpoint.BatchGeometry(
    full_batch_size=2,
    mini_batch_size=2,
    train_micro_batch_size=1,
    gradient_accumulation_steps=2,
    num_generations=1,
    num_iterations=1,
)


def _group(prompt_id):
  return [
      agent_types.TrajectoryItem(
          prompt_id=prompt_id,
          group_index=0,
          traj={"conversation_tokens": np.array([1])},
          metadata={},
      )
  ]


class _Trainer(SimpleNamespace):
  """Applies an update every 2 micro-steps."""

  def train(self, chunks):
    for _ in chunks:
      self.iter_steps += 1
      if self.iter_steps % 2 == 0:
        self.train_steps += 1


class SubBatchBenchHooksTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.log = self.create_tempfile().full_path
    self.enter_context(
        mock.patch.dict(
            os.environ,
            {"TUNIX_SB_EVENT_LOG": self.log, "TUNIX_SB_PREEMPT_AT": "3"},
        )
    )
    # `install` patches the manager class; restore it after each test.
    manager_cls = sub_batch_checkpoint.SubBatchCheckpointManager
    for name in ("record_rollout_group", "_save_records", "restore"):
      self.enter_context(
          mock.patch.object(manager_cls, name, getattr(manager_cls, name))
      )
    self.trainer = _Trainer(
        iter_steps=0,
        train_steps=0,
        checkpoint_manager=mock.Mock(),
        _buffered_train_metrics=None,
        _prev_buffered_train_metrics=None,
    )
    self.learner = SimpleNamespace(
        rl_engine=SimpleNamespace(
            actor_trainer=self.trainer,
            global_steps=0,
            update_actor=lambda chunks, *_: self.trainer.train(chunks),
        ),
        _orchestrator_producer=self._produce,
    )

  async def _produce(self, orchestrator, prompt_iterator, **unused_kwargs):
    del orchestrator
    for prompt_id in prompt_iterator:
      yield _group(prompt_id)

  def _events(self, name):
    with open(self.log) as f:
      events = [json.loads(line) for line in f]
    return [e for e in events if e["event"] == name]

  def test_kills_once_training_passes_a_preemption_point(self):
    sub_batch_bench_hooks.install(self.learner)
    update_actor = self.learner.rl_engine.update_actor
    update_actor([1, 2])
    with mock.patch.object(os, "_exit", side_effect=SystemExit) as exit_:
      with self.assertRaises(SystemExit):
        update_actor([3, 4])
    exit_.assert_called_once_with(sub_batch_bench_hooks.KILL_EXIT_CODE)
    # The kill waits for the update's checkpoint first.
    self.trainer.checkpoint_manager.wait.assert_called_once()
    self.assertEqual(
        [(e["iter_before"], e["iter_steps"]) for e in self._events("train")],
        [(0, 2), (2, 4)],
    )
    self.assertEqual(
        [(e["train_before"], e["train_steps"]) for e in self._events("train")],
        [(0, 1), (1, 2)],
    )
    [preempt] = self._events("preempt")
    self.assertEqual(preempt["iter_steps"], 3)
    self.assertEqual(preempt["killed_after_iter"], 4)
    [start] = self._events("start")
    self.assertEqual(start["iter_steps"], 0)

  def test_records_snapshots_restores_and_training_rollouts(self):
    sub_batch_bench_hooks.install(self.learner)
    root = self.create_tempdir().full_path
    manager = sub_batch_checkpoint.SubBatchCheckpointManager(
        root, enable_async=False
    )
    manager.set_geometry(_GEOMETRY)
    manager.record_rollout_group(0, _group(7))
    # Closing saves the recorded group.
    manager.close()
    restarted = sub_batch_checkpoint.SubBatchCheckpointManager(
        root, enable_async=False
    )
    self.addCleanup(restarted.close)
    restarted.restore(global_step=0)

    async def consume(prompts):
      producer = self.learner._orchestrator_producer
      return [group async for group in producer(None, prompts)]

    asyncio.run(consume(iter([1, 2])))
    # Evaluation passes a list and is not recorded.
    asyncio.run(consume([3]))
    [snapshot] = self._events("snapshot")
    self.assertFalse(snapshot["wait"])
    [resume] = self._events("resume")
    self.assertEqual(resume["groups"], [7])
    self.assertEqual(resume["global_step"], 0)
    self.assertEqual([e["group_id"] for e in self._events("rollout")], [1, 2])

  def test_preemption_waits_for_recorded_groups(self):
    sub_batch_bench_hooks.install(self.learner)
    manager = sub_batch_checkpoint.SubBatchCheckpointManager(
        self.create_tempdir().full_path, enable_async=False
    )
    self.addCleanup(manager.close)
    manager.set_geometry(_GEOMETRY)
    manager.record_rollout_group(0, _group(1))
    update_actor = self.learner.rl_engine.update_actor
    update_actor([1, 2])
    with mock.patch.object(manager, "wait", wraps=manager.wait) as wait:
      with mock.patch.object(os, "_exit", side_effect=SystemExit):
        with self.assertRaises(SystemExit):
          update_actor([3, 4])
    wait.assert_called_once()


if __name__ == "__main__":
  absltest.main()
