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

"""Tests for group_queue_manager and trajectory_queue_manager."""

import asyncio

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.queue_manager import trajectory_queue_manager
from tunix.rl.agentic.queue_manager import group_queue_manager


def _create_item(
    prompt_id: str,
    group_index: int = 0,
    task_id: str = "",
    reward: float = 1.0,
) -> datatypes.TrajectoryItem:
  """Helper to create a TrajectoryItem for testing."""
  traj = datatypes.Trajectory(reward=reward)
  return datatypes.TrajectoryItem(
      group_index=group_index,
      prompt_id=prompt_id,
      start_step=0,
      traj=traj,
      metadata={"task_id": task_id},
  )


class QueueManagerTest(absltest.TestCase):

  def test_default_trajectory_grouping(self):
    """Tests default trajectory grouping by prompt_id up to group_size."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager(
          group_size=2
      )
      item1 = _create_item("g1", group_index=0)
      item2 = _create_item("g1", group_index=1)

      await manager.put(item1)
      self.assertEmpty(manager._ready_groups)

      await manager.put(item2)
      self.assertLen(manager._ready_groups, 1)

      batch = await manager.get_batch(2)
      self.assertLen(batch, 2)
      self.assertCountEqual([item1, item2], batch)

    asyncio.run(_run_test())

  def test_generic_group_queue_manager(self):
    """Tests generic GroupQueueManager with string payloads."""

    async def _run_test():

      def string_len_group_fn(buckets, item):
        key = len(item)
        bucket = buckets[key]
        bucket.append(item)
        if len(bucket) == 2:
          del buckets[key]
          return bucket
        return None

      manager = group_queue_manager.GroupQueueManager(
          group_fn=string_len_group_fn
      )

      await manager.put("cat")
      self.assertEmpty(manager._ready_groups)

      await manager.put("dog")
      self.assertLen(manager._ready_groups, 1)

      batch = await manager.get_batch(2)
      self.assertEqual(batch, ["cat", "dog"])

    asyncio.run(_run_test())

  def test_pluggable_custom_group_fn(self):
    """Tests custom group_fn providing full bucket assembly."""

    async def _run_test():

      def custom_builder(buckets, item):
        sub = buckets["all"]
        sub.append(item)
        if len(sub) == 2:
          ready = sub.copy()
          buckets["all"].clear()
          return ready
        return None

      manager = trajectory_queue_manager.TrajectoryQueueManager(
          group_fn=custom_builder
      )

      item1 = _create_item("g1", group_index=0)
      item2 = _create_item("g2", group_index=0)

      await manager.put(item1)
      self.assertEmpty(manager._ready_groups)

      await manager.put(item2)
      self.assertLen(manager._ready_groups, 1)

      batch = await manager.get_batch(2)
      self.assertCountEqual([item1, item2], batch)

    asyncio.run(_run_test())

  def test_pluggable_filter_fn(self):
    """Tests filtering function filtering candidate groups and returning filtered items."""

    async def _run_test():
      def positive_reward_filter_fn(
          group: list[datatypes.TrajectoryItem],
      ) -> list[datatypes.TrajectoryItem]:
        return [item for item in group if item.traj.reward > 0]

      manager = trajectory_queue_manager.TrajectoryQueueManager(
          group_size=2, filter_fn=positive_reward_filter_fn
      )

      item_good = _create_item("g1", group_index=0, reward=1.0)
      item_bad = _create_item("g1", group_index=1, reward=-1.0)

      await manager.put(item_good)
      await manager.put(item_bad)

      filtered_groups = await manager.get_filtered_groups()
      self.assertLen(filtered_groups, 1)
      self.assertEqual(filtered_groups[0], [item_bad])

      batch = await manager.get_batch(1)
      self.assertEqual(batch, [item_good])

    asyncio.run(_run_test())

  def test_batching_with_leftovers(self):
    """Tests batching where a group is split across get_batch calls."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager(
          group_size=3
      )
      items = [_create_item("g1", group_index=i) for i in range(3)]
      for item in items:
        await manager.put(item)

      batch1 = await manager.get_batch(2)
      self.assertLen(batch1, 2)
      self.assertCountEqual(items[:2], batch1)
      self.assertLen(manager._ready_groups, 1)

      batch2 = await manager.get_batch(1)
      self.assertLen(batch2, 1)
      self.assertEqual(batch2[0], items[2])
      self.assertEmpty(manager._ready_groups)

    asyncio.run(_run_test())

  def test_put_exception(self):
    """Tests exception propagation."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager(
          group_size=2
      )
      exc = ValueError("Test Exception")
      await manager.put_exception(exc)

      with self.assertRaises(ValueError):
        await manager.put(_create_item("g1", 0))

      with self.assertRaises(ValueError):
        await manager.get_batch(1)

    asyncio.run(_run_test())


if __name__ == "__main__":
  absltest.main()
