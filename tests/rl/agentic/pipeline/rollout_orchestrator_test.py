"""Tests for rollout_orchestrator."""

import asyncio
from collections.abc import Mapping
import math
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.rl.agentic import utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.pipeline import rollout_orchestrator
from tunix.rl.agentic.trajectory import trajectory_collect_engine


# Mock classes for dependencies
class MockAgent(base_agent.ConversationAgentBase):
  """A mock agent."""

  def __init__(self):
    super().__init__('')

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    return agent_types.Action()


class MockEnv(base_environment.BaseTaskEnv):
  """A mock environment."""

  def __init__(
      self, task: Mapping[str, Any] | None = None, env_id: int = 0, **kwargs
  ):
    super().__init__(task=task, **kwargs)
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


class PoolNotFoundError(Exception):
  """Stands in for agent_sandbox_rl's exception of the same name.

  The orchestrator matches provisioning failures by class name rather than by
  isinstance, precisely so that tunix does not need a dependency on the fleet
  client, so a local class with the same name is a faithful test double.
  """


def _token_trajectory(env_id: int) -> dict[str, Any]:
  """A minimal payload shaped like TrajectoryCollectEngine(mode='Token')."""
  return {
      'conversation_text': [],
      'prompt_tokens': np.array([1], dtype=np.int32),
      'conversation_tokens': np.array([2, 3], dtype=np.int32),
      'conversation_masks': np.array([1, 1], dtype=np.int32),
      'status': agent_types.TrajectoryStatus.SUCCEEDED.name,
      'trajectory_reward': 1.0,
      'env_time': {},
      'reward_time': {},
      'old_logprobs': None,
      'policy_version': 0,
      'original_input': {},
      'group_id': env_id,
  }


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
        yield MockAgent(), MockEnv(env_id=i, group_id=0, pair_index=i)

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
        group_ids = set(item.prompt_id for item in batch)
        self.assertLen(group_ids, 1)

    all_items = []
    for batch in batches:
      all_items.extend(batch)

    self.assertLen(all_items, num_pairs)

    pair_indices = sorted([item.group_index for item in all_items])
    self.assertEqual(pair_indices, list(range(num_pairs)))

    items_by_group = {}
    for item in all_items:
      self.assertEqual(
          item.traj, {'trajectory': [f'traj_for_env_{item.group_index}']}
      )
      self.assertEqual(item.prompt_id, item.group_index // group_size)
      if item.prompt_id not in items_by_group:
        items_by_group[item.prompt_id] = []
      items_by_group[item.prompt_id].append(item)

    self.assertLen(items_by_group, num_pairs // group_size)
    for group_id in items_by_group:
      self.assertLen(items_by_group[group_id], group_size)
      pair_indices_in_group = sorted(
          [item.group_index for item in items_by_group[group_id]]
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
        yield MockAgent(), MockEnv(env_id=i, group_id=0, pair_index=i)

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
        yield MockAgent(), MockEnv(env_id=i, group_id=0, pair_index=i)

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

  def test_env_provisioning_failure_degrades_instead_of_killing_run(self):
    asyncio.run(self._test_env_provisioning_failure_degrades())

  async def _test_env_provisioning_failure_degrades(self):
    orchestrator = rollout_orchestrator.RolloutOrchestrator(
        max_concurrency=2,
        rollout_sync_lock=utils.RolloutSyncLock(),
        degrade_on_env_failure=True,
    )
    num_pairs = 4
    group_size = 2
    failing_pair_index = 1

    def pair_generator():
      for i in range(num_pairs):
        yield MockAgent(), MockEnv(env_id=i, group_id=0, pair_index=i)

    async def failing_side_effect(*args, **kwargs):
      env = args[1]
      if env.env_id == failing_pair_index:
        raise PoolNotFoundError('warm pool missing for task image')
      return _token_trajectory(env.env_id)

    self.mock_collect.side_effect = failing_side_effect

    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pair_generator(),
            group_size=group_size,
            group_key_fn=lambda i, *_: i // group_size,
            collect_mode='Token',
        )
    )
    await asyncio.sleep(0)

    items = []
    async for batch in orchestrator.yield_batches(batch_size=group_size):
      items.extend(batch)
    await producer_task

    # Every group still closes: no trajectory is silently dropped, which would
    # stall the consumer waiting for a group that can never fill.
    self.assertLen(items, num_pairs)

    by_index = {item.group_index: item for item in items}
    failed = by_index[failing_pair_index]
    self.assertEqual(
        failed.traj['status'], agent_types.TrajectoryStatus.FAILED.name
    )
    self.assertEqual(failed.traj['trajectory_reward'], 0.0)
    # An empty mask pads to all zeros downstream, so the placeholder
    # contributes no gradient.
    self.assertEmpty(failed.traj['conversation_masks'])
    self.assertEmpty(failed.traj['conversation_tokens'])

    for index, item in by_index.items():
      if index != failing_pair_index:
        self.assertEqual(
            item.traj['status'], agent_types.TrajectoryStatus.SUCCEEDED.name
        )

  def test_env_provisioning_failure_propagates_when_degrade_disabled(self):
    asyncio.run(self._test_env_provisioning_failure_propagates())

  async def _test_env_provisioning_failure_propagates(self):
    orchestrator = rollout_orchestrator.RolloutOrchestrator(
        max_concurrency=2,
        rollout_sync_lock=utils.RolloutSyncLock(),
        degrade_on_env_failure=False,
    )

    def pair_generator():
      for i in range(3):
        yield MockAgent(), MockEnv(env_id=i, group_id=0, pair_index=i)

    async def failing_side_effect(*args, **kwargs):
      raise PoolNotFoundError('warm pool missing for task image')

    self.mock_collect.side_effect = failing_side_effect

    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pair_generator(),
            group_size=1,
            group_key_fn=lambda i, *_: i,
            collect_mode='Token',
        )
    )
    await asyncio.sleep(0)
    with self.assertRaisesRegex(PoolNotFoundError, 'warm pool missing'):
      async for _ in orchestrator.yield_batches(batch_size=1):
        pass
      await producer_task

  def test_env_provisioning_failure_propagates_for_non_token_modes(self):
    asyncio.run(self._test_non_token_mode_propagates())

  async def _test_non_token_mode_propagates(self):
    """The placeholder is a Token payload, so it cannot stand in elsewhere."""
    orchestrator = rollout_orchestrator.RolloutOrchestrator(
        max_concurrency=1,
        rollout_sync_lock=utils.RolloutSyncLock(),
        degrade_on_env_failure=True,
    )

    def pair_generator():
      yield MockAgent(), MockEnv(env_id=0, group_id=0, pair_index=0)

    async def failing_side_effect(*args, **kwargs):
      raise PoolNotFoundError('warm pool missing for task image')

    self.mock_collect.side_effect = failing_side_effect

    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pair_generator(),
            group_size=1,
            group_key_fn=lambda i, *_: i,
            collect_mode='Trajectory',
        )
    )
    await asyncio.sleep(0)
    with self.assertRaisesRegex(PoolNotFoundError, 'warm pool missing'):
      async for _ in orchestrator.yield_batches(batch_size=1):
        pass
      await producer_task


class TokenTrajectoryTest(absltest.TestCase):
  """One builder for both producers of the `mode="Token"` payload.

  `AsyncTrajectoryLogger` fixes CSV column order from the first payload it sees
  and then appends without re-checking, so a field present in one producer and
  missing from the other shifts every later column in the rows that lack it,
  and every consumer reading the payload with `.get()` sees None rather than a
  missing key. Router replay is the concrete trigger: adding `routed_experts`
  to the engine payload and not to the placeholder does exactly this.

  `trajectory_collect_engine.token_trajectory` is the single definition that
  makes the two agree. These tests pin the properties it relies on.
  """

  def test_overriding_a_field_does_not_move_it(self):
    defaults = trajectory_collect_engine.token_trajectory()
    overridden = trajectory_collect_engine.token_trajectory(
        conversation_text=['hi'], group_id=7
    )
    self.assertEqual(list(overridden), list(defaults))
    self.assertEqual(overridden['conversation_text'], ['hi'])
    self.assertEqual(overridden['group_id'], 7)

  def test_unknown_field_is_rejected(self):
    # A new field has to be added to the builder, where both producers get it,
    # rather than smuggled into one call site. (`routed_experts` was the
    # original example here; it is now a real field, added via the builder.)
    with self.assertRaises(KeyError):
      trajectory_collect_engine.token_trajectory(not_a_payload_field=None)

  def test_placeholder_has_the_whole_schema(self):
    orchestrator = rollout_orchestrator.RolloutOrchestrator(
        max_concurrency=1,
        rollout_sync_lock=utils.RolloutSyncLock(),
    )
    env = MockEnv(task={'policy_version': 4}, env_id=0, group_id=3)

    # pylint: disable-next=protected-access
    placeholder = orchestrator._make_failed_trajectory(env)

    self.assertEqual(
        list(placeholder),
        list(trajectory_collect_engine.token_trajectory()),
    )
    self.assertEqual(placeholder['group_id'], 3)
    self.assertEqual(placeholder['policy_version'], 4)
    self.assertEqual(placeholder['trajectory_reward'], 0.0)
    self.assertEqual(
        placeholder['status'], agent_types.TrajectoryStatus.FAILED.name
    )
    self.assertEqual(placeholder['conversation_masks'].size, 0)


if __name__ == '__main__':
  absltest.main()
