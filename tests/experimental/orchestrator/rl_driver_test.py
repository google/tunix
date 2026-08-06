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

"""Tests for RLDriver."""

import asyncio
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import rl_driver


def _create_item(
    group_id: str,
    pair_index: int = 0,
    policy_version: int = 0,
    reward: float = 1.0,
) -> datatypes.TrajectoryItem:
  traj = datatypes.Trajectory(
      reward=reward,
  )
  item = datatypes.TrajectoryItem(
      pair_index=pair_index,
      group_id=group_id,
      start_step=0,
      traj=traj,
  )
  item.policy_version = policy_version
  return item


class RLDriverTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_engine = mock.MagicMock()
    self.mock_engine.global_steps = 42

    self.mock_config = mock.MagicMock()
    self.mock_config.reward_manager = "agentic-sequence-level"
    self.mock_config.advantage_estimator = "grpo"
    self.mock_config.num_generations = 2

  def test_initialization_and_policy_version(self):
    driver = rl_driver.RLDriver(
        rl_engine=self.mock_engine,
        algo_config=self.mock_config,
    )
    self.assertEqual(driver.policy_version, 0)
    driver.policy_version = 100
    self.assertEqual(driver.policy_version, 100)

  def test_sync_weights_delegates_to_engine(self):
    driver = rl_driver.RLDriver(
        rl_engine=self.mock_engine,
        algo_config=self.mock_config,
    )
    driver.sync_weights()
    self.mock_engine.sync_weights.assert_called_once()

  def test_train_delegates_to_engine(self):
    driver = rl_driver.RLDriver(
        rl_engine=self.mock_engine,
        algo_config=self.mock_config,
    )
    driver.train(
        role=datatypes.Role.ACTOR, train_ds="train_ds", eval_ds="eval_ds"
    )
    self.mock_engine.train.assert_called_once_with(
        role=datatypes.Role.ACTOR,
        train_ds="train_ds",
        eval_ds="eval_ds",
        skip_jit=False,
    )

  def test_queue_manager_out_of_order_grouping(self):
    async def _run():
      driver = rl_driver.RLDriver(
          rl_engine=self.mock_engine,
          algo_config=self.mock_config,
      )
      queue = driver.create_queue_manager(group_size=2)

      item_g1_0 = _create_item("prompt_g1", pair_index=0)
      item_g2_0 = _create_item("prompt_g2", pair_index=0)
      item_g1_1 = _create_item("prompt_g1", pair_index=1)

      await queue.put(item_g1_0)
      await queue.put(item_g2_0)
      self.assertEmpty(queue._ready_groups)

      await queue.put(item_g1_1)
      self.assertLen(queue._ready_groups, 1)

      batch = await queue.get_batch(2)
      self.assertEqual(batch, [item_g1_0, item_g1_1])

    asyncio.run(_run())

  def test_queue_manager_staleness_filter(self):
    async def _run():
      driver = rl_driver.RLDriver(
          rl_engine=self.mock_engine,
          algo_config=self.mock_config,
      )
      driver.expected_policy_version = 10
      queue = driver.create_queue_manager(group_size=2, max_staleness=1)

      fresh_0 = _create_item("p1", pair_index=0, policy_version=10)
      stale_1 = _create_item("p1", pair_index=1, policy_version=8)

      await queue.put(fresh_0)
      await queue.put(stale_1)
      filtered = await queue.get_filtered_groups()
      self.assertLen(filtered, 1)
      self.assertEqual(filtered[0], [stale_1])

    asyncio.run(_run())

  def test_generate_delegates_to_engine(self):
    async def _run():
      self.mock_engine.generate = mock.AsyncMock(return_value="async_rollout")
      driver = rl_driver.RLDriver(
          rl_engine=self.mock_engine,
          algo_config=self.mock_config,
      )
      res = await driver.generate(prompts=["hello"])
      self.assertEqual(res, "async_rollout")
      self.mock_engine.generate.assert_called_once()

    asyncio.run(_run())

  def test_dispatch_rollouts_and_poll_rollouts(self):
    async def _run():
      self.mock_engine.dispatch_generate = mock.AsyncMock()
      self.mock_engine.poll_rollouts = mock.AsyncMock(return_value=["poll_1"])
      driver = rl_driver.RLDriver(
          rl_engine=self.mock_engine,
          algo_config=self.mock_config,
      )
      await driver.dispatch_rollouts(["req1"])
      self.mock_engine.dispatch_generate.assert_called_once_with(["req1"])

      polled = await driver.poll_rollouts(timeout_s=0.2)
      self.assertEqual(polled, ["poll_1"])

    asyncio.run(_run())

  def test_score_async(self):
    async def _run():
      driver = rl_driver.RLDriver(
          rl_engine=self.mock_engine,
          algo_config=self.mock_config,
      )
      driver.process_results = mock.MagicMock(return_value=["scored_example"])
      res = await driver.score_async(["traj_1"])
      self.assertEqual(res, ["scored_example"])
      driver.process_results.assert_called_once()

    asyncio.run(_run())

  def test_train_step_and_sync_weights(self):
    async def _run():
      self.mock_engine.train_step = mock.AsyncMock(return_value="step_done")
      self.mock_engine.sync_weights = mock.AsyncMock()
      driver = rl_driver.RLDriver(
          rl_engine=self.mock_engine,
          algo_config=self.mock_config,
      )
      res = await driver.train_step("batch_1")
      self.assertEqual(res, "step_done")
      self.mock_engine.train_step.assert_called_once_with(
          batch="batch_1", role=datatypes.Role.ACTOR, skip_jit=False
      )

      await driver.sync_weights()
      self.mock_engine.sync_weights.assert_called_once()

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()

