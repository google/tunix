# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for AsyncRLProgram."""

import asyncio
from unittest import mock

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import async_rl_program
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.orchestrator import rl_driver
from tunix.rl import rl_cluster as rl_engine_lib


def _create_rollout_response(
    request_id: str,
    prompt_id: str,
    group_id: str,
    pair_index: int = 0,
    policy_version: int = 0,
    reward: float = 1.0,
) -> datatypes.RolloutResponse:
  return datatypes.RolloutResponse(
      request_id=request_id,
      prompt_id=prompt_id,
      status="COMPLETED",
      env_reward=reward,
      policy_version=policy_version,
      metadata={
          "group_id": group_id,
          "pair_index": pair_index,
      },
  )


class AsyncRLProgramTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_engine = mock.MagicMock(spec=distributed_rl_engine.DistributedRLEngine)
    self.mock_config = mock.MagicMock()
    self.mock_config.reward_manager = "agentic-sequence-level"
    self.mock_config.advantage_estimator = "grpo"
    self.mock_config.num_generations = 2

    self.driver = rl_driver.RLDriver(
        rl_engine=self.mock_engine,
        algo_config=self.mock_config,
    )

  def test_initialization(self):
    program = async_rl_program.AsyncRLProgram(
        driver=self.driver,
        group_size=2,
        batch_size_groups=1,
    )
    self.assertEqual(program.step, 0)
    self.assertEqual(program.group_size, 2)
    self.assertEqual(program.batch_size_groups, 1)
    self.assertIsNotNone(program.queue_manager)

  def test_run_async_three_stages_with_long_polling(self):
    async def _run():
      poll_results = [
          [
              _create_rollout_response(
                  "req_0_0", "prompt_0", "group_0", pair_index=0
              ),
              _create_rollout_response(
                  "req_0_1", "prompt_0", "group_0", pair_index=1
              ),
          ],
          [],
      ]
      call_idx = 0

      async def mock_poll(timeout_s=0.1):
        nonlocal call_idx
        if call_idx < len(poll_results):
          res = poll_results[call_idx]
          call_idx += 1
          return res
        await asyncio.sleep(0.01)
        return []

      self.mock_engine.poll_rollouts.side_effect = mock_poll
      self.mock_engine.dispatch_generate = mock.AsyncMock()

      begin_steps = []
      end_steps = []

      def on_begin(step):
        begin_steps.append(step)

      def on_end(step, result):
        end_steps.append((step, result))

      self.driver.process_results = mock.MagicMock(return_value=["mock_batch"])
      self.driver.train_step = mock.AsyncMock(return_value="step_result_1")
      self.driver.sync_weights = mock.AsyncMock()

      program = async_rl_program.AsyncRLProgram(
          driver=self.driver,
          group_size=2,
          batch_size_groups=1,
          on_step_begin=on_begin,
          on_step_end=on_end,
      )

      await program.run_async(train_dataset=["prompt_data_0"], num_steps=1)

      self.assertEqual(program.step, 1)
      self.assertEqual(begin_steps, [0])
      self.assertEqual(end_steps, [(1, "step_result_1")])
      self.mock_engine.dispatch_generate.assert_called_once()
      self.driver.train_step.assert_called_once_with("mock_batch")
      self.driver.sync_weights.assert_called_once()

    asyncio.run(_run())

  def test_out_of_order_prompt_grouping(self):
    async def _run():
      # Completions for prompt_0 and prompt_1 arrive interleaved:
      # prompt_0 #0, prompt_1 #0, prompt_0 #1, prompt_1 #1
      poll_results = [
          [
              _create_rollout_response(
                  "req_0_0", "prompt_0", "group_0", pair_index=0
              ),
              _create_rollout_response(
                  "req_1_0", "prompt_1", "group_1", pair_index=0
              ),
          ],
          [
              _create_rollout_response(
                  "req_0_1", "prompt_0", "group_0", pair_index=1
              ),
              _create_rollout_response(
                  "req_1_1", "prompt_1", "group_1", pair_index=1
              ),
          ],
          [],
      ]
      call_idx = 0

      async def mock_poll(timeout_s=0.1):
        nonlocal call_idx
        if call_idx < len(poll_results):
          res = poll_results[call_idx]
          call_idx += 1
          return res
        await asyncio.sleep(0.01)
        return []

      self.mock_engine.poll_rollouts.side_effect = mock_poll
      self.mock_engine.dispatch_generate = mock.AsyncMock()

      processed_groups = []

      def mock_process_results(trajectories, mode=None, expected_step=None):
        del mode, expected_step
        group_ids = [getattr(t, "group_id", None) for t in trajectories]
        processed_groups.append(group_ids)
        return [f"batch_{len(processed_groups)}"]

      self.driver.process_results = mock_process_results
      self.driver.train_step = mock.AsyncMock(return_value="ok")
      self.driver.sync_weights = mock.AsyncMock()

      program = async_rl_program.AsyncRLProgram(
          driver=self.driver,
          group_size=2,
          batch_size_groups=1,
      )

      await program.run_async(
          train_dataset=["prompt_0", "prompt_1"], num_steps=2
      )

      self.assertEqual(program.step, 2)
      # Both groups are processed with full matching pairs
      self.assertEqual(processed_groups, [["group_0", "group_0"], ["group_1", "group_1"]])

    asyncio.run(_run())

  def test_staleness_filtering(self):
    async def _run():
      self.driver.expected_policy_version = 10
      # Create poll responses: one fresh (v10) and one stale (v8)
      poll_results = [
          [
              _create_rollout_response(
                  "req_0_0", "prompt_0", "group_0", pair_index=0, policy_version=10
              ),
              _create_rollout_response(
                  "req_0_1", "prompt_0", "group_0", pair_index=1, policy_version=8
              ),
          ],
          [],
      ]
      call_idx = 0

      async def mock_poll(timeout_s=0.1):
        nonlocal call_idx
        if call_idx < len(poll_results):
          res = poll_results[call_idx]
          call_idx += 1
          return res
        await asyncio.sleep(0.01)
        return []

      self.mock_engine.poll_rollouts.side_effect = mock_poll
      self.mock_engine.dispatch_generate = mock.AsyncMock()

      program = async_rl_program.AsyncRLProgram(
          driver=self.driver,
          group_size=2,
          max_staleness=1,
      )

      # Start polling stage task briefly
      task = asyncio.create_task(program.polling_stage())
      program._is_running = True
      await asyncio.sleep(0.05)
      program._is_running = False
      await task

      filtered = await program.queue_manager.get_filtered_groups()
      self.assertLen(filtered, 1)
      self.assertEqual(filtered[0][0].policy_version, 8)

    asyncio.run(_run())

  def test_custom_multi_stage_subclass(self):
    class MultiStageAgenticProgram(async_rl_program.AsyncRLProgram):

      def __init__(self, driver):
        super().__init__(driver=driver, group_size=2, batch_size_groups=1)
        self.raw_rollouts_q = driver.create_queue_manager(group_size=2)
        self.scored_rollouts_q = driver.create_queue_manager(group_size=2)
        self.rollouts_dispatched = 0
        self.critiques_scored = 0
        self.train_steps_completed = 0

      async def rollout_stage(self, train_dataset):
        for prompt in train_dataset:
          rollout = await self.driver.generate([prompt] * self.group_size)
          for g_idx, r in enumerate(rollout):
            item = self._response_to_trajectory_item(r)
            item.group_id = prompt
            item.pair_index = g_idx
            await self.raw_rollouts_q.put(item)
          self.rollouts_dispatched += 1

      async def critique_stage(self):
        async for group in self.raw_rollouts_q:
          scored = await self.driver.score_async(group)
          for item in scored:
            await self.scored_rollouts_q.put(
                datatypes.TrajectoryItem(
                    pair_index=0, group_id="scored", start_step=0, traj=item
                )
            )
          self.critiques_scored += 1
          if self.critiques_scored >= 2:
            break

      async def train_stage(self, num_steps):
        for _ in range(num_steps):
          scored_items = await self.scored_rollouts_q.get_batch(num_groups=1)
          if not scored_items:
            break
          await self.driver.train_step(scored_items)
          await self.driver.sync_weights()
          self.driver.policy_version += 1
          self.train_steps_completed += 1

    async def _run():
      self.driver.generate = mock.AsyncMock(
          return_value=[
              datatypes.Trajectory(reward=1.0),
              datatypes.Trajectory(reward=1.0),
          ]
      )
      self.driver.score_async = mock.AsyncMock(
          return_value=[{"train_batch": 1}, {"train_batch": 2}]
      )
      self.driver.train_step = mock.AsyncMock(return_value="ok")
      self.driver.sync_weights = mock.AsyncMock()

      prog = MultiStageAgenticProgram(driver=self.driver)
      await prog.run_async(train_dataset=["prompt_a", "prompt_b"], num_steps=2)

      self.assertEqual(prog.rollouts_dispatched, 2)
      self.assertEqual(prog.critiques_scored, 2)
      self.assertEqual(prog.train_steps_completed, 2)
      self.assertEqual(self.driver.policy_version, 2)

    asyncio.run(_run())

  def test_stage_exception_aborts_queue_and_propagates(self):
    class FailingProgram(async_rl_program.AsyncRLProgram):

      async def rollout_stage(self, train_dataset):
        del train_dataset
        raise RuntimeError("Rollout worker cluster down!")

    async def _run():
      prog = FailingProgram(driver=self.driver, group_size=2)
      with self.assertRaises(RuntimeError) as cm:
        await prog.run_async(train_dataset=["prompt"], num_steps=1)
      self.assertIn("Rollout worker cluster down!", str(cm.exception))

    asyncio.run(_run())

  def test_multi_stage_agentic_program_class(self):
    async def _run():
      poll_results = [
          [
              _create_rollout_response(
                  "req_0_0", "prompt_1", "group_0", pair_index=0
              ),
              _create_rollout_response(
                  "req_0_1", "prompt_1", "group_0", pair_index=1
              ),
          ],
          [],
      ]
      call_idx = 0

      async def mock_poll(timeout_s=0.1):
        nonlocal call_idx
        if call_idx < len(poll_results):
          res = poll_results[call_idx]
          call_idx += 1
          return res
        await asyncio.sleep(0.01)
        return []

      self.mock_engine.poll_rollouts.side_effect = mock_poll
      self.mock_engine.dispatch_generate = mock.AsyncMock()

      self.driver.score_async = mock.AsyncMock(
          return_value=[
              datatypes.Trajectory(reward=1.0),
              datatypes.Trajectory(reward=1.0),
          ]
      )
      self.driver.train_step = mock.AsyncMock(return_value="step_ok")
      self.driver.sync_weights = mock.AsyncMock()

      program = async_rl_program.MultiStageAgenticProgram(
          driver=self.driver,
          group_size=2,
          batch_size_groups=1,
      )
      await program.run_async(train_dataset=["prompt_1"], num_steps=1)

      self.assertEqual(program.step, 1)
      self.mock_engine.dispatch_generate.assert_called_once()
      self.driver.score_async.assert_called_once()
      self.driver.train_step.assert_called_once()
      self.driver.sync_weights.assert_called_once()

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
