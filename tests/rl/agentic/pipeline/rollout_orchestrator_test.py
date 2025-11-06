"""Tests for rollout_orchestrator."""

import asyncio
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.rl.agentic.pipeline import rollout_orchestrator
from tunix.rl.agentic.trajectory import trajectory_collect_engine


# Mock classes for dependencies
class MockAgent(trajectory_collect_engine.LLMBaseAgent):
  """A mock agent."""

  async def predict(self, *args, **kwargs):
    return {'response': 'mock_response'}


class MockEnv(trajectory_collect_engine.BaseEnv):
  """A mock environment."""

  def __init__(self, task: dict | None = None, env_id: int = 0):
    self.task = task or {}
    self.env_id = env_id

  async def reset(self, *args, **kwargs):
    return {'obs': f'initial_obs_{self.env_id}'}

  async def step(self, action):
    return {'obs': f'next_obs_{self.env_id}', 'reward': 1.0, 'done': False}


def _group_by_pair_index(
    pair_index: int, env: MockEnv, traj: rollout_orchestrator.Trajectory
) -> int:
  return pair_index


class RolloutOrchestratorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Using AsyncMock for async methods
    self.collect_patcher = mock.patch.object(
        rollout_orchestrator.RolloutOrchestrator,
        '_collect_trajectory',
        new_callable=mock.AsyncMock,
    )
    self.mock_collect = self.collect_patcher.start()

    def _side_effect(*args, **kwargs):
      # args are (self, agent, env, mode=)
      env = args[2]
      return {'trajectory': [f'traj_for_env_{env.env_id}']}

    self.mock_collect.side_effect = _side_effect
    self.addCleanup(self.collect_patcher.stop)

  async def test_run_and_yield_batches_drains_queue(self):
    orchestrator = rollout_orchestrator.RolloutOrchestrator()
    num_pairs = 4
    episodes_per_pair = 3
    pairs = [(MockAgent(), MockEnv(env_id=i)) for i in range(num_pairs)]

    # Group size is larger than episodes_per_pair, so groups are formed at end.
    group_size = 5
    batch_size = 2

    results = []
    async for batch in orchestrator.run_and_yield_batches(
        pairs=pairs,
        group_size=group_size,
        batch_size=batch_size,
        group_key=_group_by_pair_index,
        episodes_per_pair=episodes_per_pair,
    ):
      results.extend(batch)

    self.assertLen(
        results,
        num_pairs * episodes_per_pair,
        'Should collect all trajectories.',
    )
    self.assertEqual(
        self.mock_collect.call_count, num_pairs * episodes_per_pair
    )

  async def test_streaming_producers_basic_functionality(self):
    orchestrator = rollout_orchestrator.RolloutOrchestrator(max_concurrency=2)
    num_pairs = 5
    episodes_per_pair = 2

    def pair_generator():
      for i in range(num_pairs):
        yield MockAgent(), MockEnv(env_id=i)

    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pair_generator(),
            group_size=episodes_per_pair,
            group_key=_group_by_pair_index,
            episodes_per_pair=episodes_per_pair,
        )
    )

    results = []
    async for batch in orchestrator.yield_batches(batch_size=3):
      results.extend(batch)

    await producer_task

    self.assertLen(
        results,
        num_pairs * episodes_per_pair,
        'Should collect all trajectories.',
    )
    self.assertEqual(
        self.mock_collect.call_count, num_pairs * episodes_per_pair
    )

    # Check if trajectories from all envs are present.
    collected_envs = set()
    for item in results:
      traj_content = item.traj['trajectory'][0]
      env_id = int(traj_content.split('_')[-1])
      collected_envs.add(env_id)
    self.assertSetEqual(collected_envs, set(range(num_pairs)))

  async def test_streaming_consumer_stops_early(self):
    orchestrator = rollout_orchestrator.RolloutOrchestrator(max_concurrency=2)
    num_pairs = 10
    episodes_per_pair = 5  # Set high to ensure it's still running

    def pair_generator():
      for i in range(num_pairs):
        yield MockAgent(), MockEnv(env_id=i)

    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pair_generator(),
            group_size=2,
            group_key=_group_by_pair_index,
            episodes_per_pair=episodes_per_pair,
        )
    )

    results = []
    batches_to_take = 2
    async for batch in orchestrator.yield_batches(batch_size=1):
      results.extend(batch)
      batches_to_take -= 1
      if batches_to_take == 0:
        break

    # Consumer loop breaks, so producer should be cancelled.
    # Give it a moment to process cancellation.
    await asyncio.sleep(0.01)

    self.assertTrue(producer_task.done())
    # Depending on timing, it might be cancelled or finished gracefully if all
    # tasks happened to finish before cancellation was processed. The key part
    # is that it doesn't hang.
    if not producer_task.cancelled():
      # if not cancelled, check if any exception
      exc = producer_task.exception()
      self.assertIsNone(exc, f'Producer task failed with {exc}')

    # Check that not all trajectories were collected
    self.assertLess(self.mock_collect.call_count, num_pairs * episodes_per_pair)
    # At least some should have been collected before stopping.
    self.assertGreater(self.mock_collect.call_count, 0)

  async def test_streaming_producer_runner_exception(self):
    orchestrator = rollout_orchestrator.RolloutOrchestrator(max_concurrency=2)
    num_pairs = 5

    failing_pair_index = 2

    # Mock collect to fail for a specific pair index
    original_side_effect = self.mock_collect.side_effect

    async def failing_side_effect(*args, **kwargs):
      env = args[2]
      if env.env_id == failing_pair_index:
        raise ValueError('Collection failed!')
      return await mock.AsyncMock(
          return_value=original_side_effect(*args, **kwargs)
      )()

    self.mock_collect.side_effect = failing_side_effect

    def pair_generator():
      for i in range(num_pairs):
        yield MockAgent(), MockEnv(env_id=i)

    producer_task = asyncio.create_task(
        orchestrator.run_producers_from_stream(
            pairs_stream=pair_generator(),
            group_size=1,
            group_key=_group_by_pair_index,
            episodes_per_pair=1,
        )
    )

    with self.assertRaisesRegex(ValueError, 'Collection failed!'):
      # Consumer loop.
      async for _ in orchestrator.yield_batches(batch_size=1):
        pass
      # Await producer to get the exception if not raised during consumption.
      await producer_task

  async def test_streaming_generator_exception(self):
    orchestrator = rollout_orchestrator.RolloutOrchestrator(max_concurrency=2)
    failing_pair_index = 2

    def faulty_generator():
      for i in range(5):
        if i == failing_pair_index:
          raise ValueError('Generator failed!')
        yield MockAgent(), MockEnv(env_id=i)

    with self.assertRaisesRegex(ValueError, 'Generator failed!'):
      producer_task = asyncio.create_task(
          orchestrator.run_producers_from_stream(
              pairs_stream=faulty_generator(),
              group_size=1,
              group_key=_group_by_pair_index,
              episodes_per_pair=1,
          )
      )
      async for _ in orchestrator.yield_batches(batch_size=1):
        pass
      await producer_task


if __name__ == '__main__':
  absltest.main()
