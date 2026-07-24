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

"""Tests for experimental trajectory_queue_manager."""

import asyncio

from absl.testing import absltest
from tunix.experimental.queue_manager import trajectory_queue_manager
from tunix.rl.agentic.agents import agent_types


def _create_item(
    group_id: str,
    pair_index: int = 0,
    task_id: str = "",
    reward: float = 1.0,
) -> agent_types.TrajectoryItem:
  """Helper to create a TrajectoryItem for testing."""
  traj = agent_types.Trajectory(reward=reward)
  return agent_types.TrajectoryItem(
      pair_index=pair_index,
      group_id=group_id,
      start_step=0,
      traj=traj,
      metadata={"task_id": task_id},
  )


class TrajectoryQueueManagerTest(absltest.TestCase):

  def test_default_grouping_same_group_different_pair(self):
    """Tests grouping by group_id and ensuring distinct pair_index."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager(group_size=2)
      item1 = _create_item("g1", pair_index=0)
      item2 = _create_item("g1", pair_index=1)

      await manager.put(item1)
      self.assertEmpty(manager._ready_groups)

      await manager.put(item2)
      self.assertLen(manager._ready_groups, 1)

      batch = await manager.get_batch(2)
      self.assertLen(batch, 2)
      self.assertCountEqual([item1, item2], batch)

    asyncio.run(_run_test())

  def test_grouping_with_duplicate_pair_index(self):
    """Tests that duplicate pair_indices for same group_id go to separate sub-buckets."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager(group_size=2)
      item_g1_p0_a = _create_item("g1", pair_index=0, task_id="a")
      item_g1_p0_b = _create_item("g1", pair_index=0, task_id="b")
      item_g1_p1_a = _create_item("g1", pair_index=1, task_id="a")

      # Put two items with same pair_index=0
      await manager.put(item_g1_p0_a)
      await manager.put(item_g1_p0_b)
      # Neither bucket is ready yet (needs group_size=2 distinct pair_indices)
      self.assertEmpty(manager._ready_groups)

      # Now add pair_index=1 to complete the first sub-bucket
      await manager.put(item_g1_p1_a)
      self.assertLen(manager._ready_groups, 1)

      batch = await manager.get_batch(2)
      self.assertCountEqual([item_g1_p0_a, item_g1_p1_a], batch)

    asyncio.run(_run_test())

  def test_pluggable_group_fn_key_extraction(self):
    """Tests custom pluggable group_fn extracting custom grouping key."""

    async def _run_test():
      # Group by task_id in metadata rather than group_id
      def custom_key_fn(item: agent_types.TrajectoryItem) -> str:
        return item.metadata.get("task_id", "")

      manager = trajectory_queue_manager.TrajectoryQueueManager(
          group_size=2, group_fn=custom_key_fn
      )

      item1 = _create_item("g1", pair_index=0, task_id="task_A")
      item2 = _create_item(
          "g2", pair_index=1, task_id="task_A"
      )  # different group_id, same task_id

      await manager.put(item1)
      self.assertEmpty(manager._ready_groups)

      await manager.put(item2)
      self.assertLen(manager._ready_groups, 1)

      batch = await manager.get_batch(2)
      self.assertLen(batch, 2)
      self.assertCountEqual([item1, item2], batch)

    asyncio.run(_run_test())

  def test_pluggable_filter_fn(self):
    """Tests filtering function filtering candidate groups and returning filtered items."""

    async def _run_test():
      # Filter out items with reward <= 0
      def positive_reward_filter_fn(
          group: list[agent_types.TrajectoryItem],
      ) -> list[agent_types.TrajectoryItem]:
        return [item for item in group if item.traj.reward > 0]

      manager = trajectory_queue_manager.TrajectoryQueueManager(
          group_size=2, filter_fn=positive_reward_filter_fn
      )

      item_good = _create_item("g1", pair_index=0, reward=1.0)
      item_bad = _create_item("g1", pair_index=1, reward=-1.0)

      await manager.put(item_good)
      await manager.put(item_bad)

      # Candidate group of 2 formed; filter_fn filtered out item_bad
      filtered_groups = manager.get_filtered_groups()
      self.assertLen(filtered_groups, 1)
      self.assertEqual(filtered_groups[0], [item_bad])

      # Ready group contains valid item_good
      batch = await manager.get_batch(1)
      self.assertEqual(batch, [item_good])

    asyncio.run(_run_test())

  def test_batching_with_leftovers(self):
    """Tests batching where a group is split across get_batch calls."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager(group_size=3)
      items = [_create_item("g1", pair_index=i) for i in range(3)]
      for item in items:
        await manager.put(item)

      self.assertEmpty(manager._batch_buf)
      batch1 = await manager.get_batch(2)
      self.assertLen(batch1, 2)
      self.assertCountEqual(items[:2], batch1)
      self.assertLen(manager._batch_buf, 1)
      self.assertEqual(manager._batch_buf[0], items[2])

      batch2 = await manager.get_batch(1)
      self.assertLen(batch2, 1)
      self.assertEqual(batch2[0], items[2])
      self.assertEmpty(manager._batch_buf)

    asyncio.run(_run_test())

  def test_put_exception(self):
    """Tests exception propagation."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager(group_size=2)
      exc = ValueError("Test Exception")
      await manager.put_exception(exc)

      with self.assertRaises(ValueError):
        await manager.put(_create_item("g1", 0))

      with self.assertRaises(ValueError):
        await manager.get_batch(1)

    asyncio.run(_run_test())


if __name__ == "__main__":
  absltest.main()
