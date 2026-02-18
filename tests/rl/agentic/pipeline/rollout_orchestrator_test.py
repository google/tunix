"""Tests for rollout_orchestrator."""

import asyncio
from collections.abc import Mapping
import math
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.rl.agentic import utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.pipeline import rollout_orchestrator


# Mock classes for dependencies
class MockAgent(base_agent.ConversationAgentBase):
  """A mock agent."""

  def __init__(self):
    super().__init__('')

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    return agent_types.Action()


class MockEnv(base_environment.BaseTaskEnv):
  """A mock environment."""

  def __init__(self, task: Mapping[str, Any] | None = None, env_id: int = 0):
    super().__init__(task=task)
    self.env_id = env_id

  def _initial_observation(self):
    return {'obs': f'initial_obs_{self.env_id}'}

  def _step_impl(self, action):
    return base_environment.EnvStepResult(
        observation={'obs': f'next_obs_{self.env_id}'},
        reward=1.0,
        done=False,
        info={},
    )


class RolloutOrchestratorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.collect_patcher = mock.patch.object(
        rollout_orchestrator.RolloutOrchestrator,
        '_collect_trajectory',
        new_callable=mock.AsyncMock,
    )
    self.mock_collect = self.collect_patcher.start()
    self.addCleanup(self.collect_patcher.stop)

  @parameterized.named_parameters(
      dict(
          testcase_name='group_size_1_batch_size_2',
          num_pairs=3,
          group_size=1,
          batch_size=2,
      ),
      dict(
          testcase_name='group_size_2_batch_size_2',
          num_pairs=4,
          group_size=2,
          batch_size=2,
      ),
      dict(
          testcase_name='group_size_1_batch_size_5',
          num_pairs=5,
          group_size=1,
          batch_size=5,
      ),
      dict(
          testcase_name='group_size_3_batch_size_3',
          num_pairs=6,
          group_size=3,
          batch_size=3,
      ),
      dict(
          testcase_name='group_size_2_batch_size_4',
          num_pairs=6,
          group_size=2,
          batch_size=4,
      ),
  )
  def test_streaming_successful_run(self, num_pairs, group_size, batch_size):
    asyncio.run(
        self._test_streaming_successful_run(num_pairs, group_size, batch_size)
    )

  async def _test_streaming_successful_run(
      self, num_pairs, group_size, batch_size
  ):
    orchestrator = rollout_orchestrator.RolloutOrchestrator(
        max_concurrency=2,
        rollout_sync_lock=utils.RolloutSyncLock(),
    )

    def pair_generator():
      for i in range(num_pairs):
        yield MockAgent(), MockEnv(env_id=i)

    async def side_effect_fn(*args, **kwargs):
      env = args[1]
      return {'trajectory': [f'traj_for_env_{env.env_id}']}

    self.mock_collect.side_effect = side_effect_fn

    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pair_generator(),
            group_size=group_size,
            group_key_fn=lambda i, env, traj: i // group_size,
        )
    )
    await asyncio.sleep(0)

    batches = []
    async for batch in orchestrator.yield_batches(batch_size=batch_size):
      batches.append(batch)
    await producer_task

    # Check if the orchestrator yields the correct number of batches.
    self.assertLen(batches, math.ceil(num_pairs / batch_size))
    for batch in batches:
      self.assertLessEqual(len(batch), batch_size)
      if group_size > 1 and batch_size <= group_size:
        # If group_size > 1 and batch_size <= group_size, items in a batch
        # are expected to come from the same group.
        group_ids = set(item.group_id for item in batch)
        self.assertLen(group_ids, 1)

    all_items = []
    for batch in batches:
      all_items.extend(batch)

    self.assertLen(all_items, num_pairs)

    pair_indices = sorted([item.pair_index for item in all_items])
    self.assertEqual(pair_indices, list(range(num_pairs)))

    items_by_group = {}
    for item in all_items:
      self.assertEqual(
          item.traj, {'trajectory': [f'traj_for_env_{item.pair_index}']}
      )
      self.assertEqual(item.group_id, item.pair_index // group_size)
      if item.group_id not in items_by_group:
        items_by_group[item.group_id] = []
      items_by_group[item.group_id].append(item)

    self.assertLen(items_by_group, num_pairs // group_size)
    for group_id in items_by_group:
      self.assertLen(items_by_group[group_id], group_size)
      pair_indices_in_group = sorted(
          [item.pair_index for item in items_by_group[group_id]]
      )
      expected_pair_indices = list(
          range(
              group_id * group_size,
              group_id * group_size + group_size,
          )
      )
      self.assertEqual(pair_indices_in_group, expected_pair_indices)

  def test_streaming_producer_runner_exception(self):
    asyncio.run(self._test_streaming_producer_runner_exception())

  async def _test_streaming_producer_runner_exception(self):
    orchestrator = rollout_orchestrator.RolloutOrchestrator(
        max_concurrency=2,
        rollout_sync_lock=utils.RolloutSyncLock(),
    )
    num_pairs = 5
    failing_pair_index = 2

    def pair_generator():
      for i in range(num_pairs):
        yield MockAgent(), MockEnv(env_id=i)

    async def failing_side_effect(*args, **kwargs):
      env = args[1]
      if env.env_id == failing_pair_index:
        raise ValueError('Collection failed!')
      return {'trajectory': [f'traj_for_env_{env.env_id}']}

    self.mock_collect.side_effect = failing_side_effect

    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pair_generator(),
            group_size=1,
            group_key_fn=lambda i, *_: i,
        )
    )
    await asyncio.sleep(0)
    with self.assertRaisesRegex(ValueError, 'Collection failed!'):
      # Consumer loop.
      async for _ in orchestrator.yield_batches(batch_size=1):
        pass
      # Await producer to get the exception if not raised during consumption.
      await producer_task

  def test_streaming_generator_exception(self):
    asyncio.run(self._test_streaming_generator_exception())

  async def _test_streaming_generator_exception(self):
    orchestrator = rollout_orchestrator.RolloutOrchestrator(
        max_concurrency=2,
        rollout_sync_lock=utils.RolloutSyncLock(),
    )
    failing_pair_index = 2

    def faulty_generator():
      for i in range(5):
        if i == failing_pair_index:
          raise ValueError('Generator failed!')
        yield MockAgent(), MockEnv(env_id=i)

    self.mock_collect.side_effect = None
    self.mock_collect.return_value = {'trajectory': ['mock_traj']}

    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=faulty_generator(),
            group_size=1,
            group_key_fn=lambda i, *_: i,
        )
    )
    await asyncio.sleep(0)
    with self.assertRaisesRegex(ValueError, 'Generator failed!'):
      async for _ in orchestrator.yield_batches(batch_size=1):
        pass
      await producer_task


if __name__ == '__main__':
  absltest.main()
