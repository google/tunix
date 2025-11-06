# Copyright 2025 Google LLC
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

"""Tests for group_queue_manager."""

import asyncio

from absl.testing import absltest
from tunix.rl.agentic.queue_manager import group_queue_manager


def _create_item(
    group_id: str, episode_id: int, pair_index: int = 0
) -> group_queue_manager.TrajectoryItem:
  """Helper to create a TrajectoryItem for testing."""
  return group_queue_manager.TrajectoryItem(
      pair_index=pair_index,
      group_id=group_id,
      episode_id=episode_id,
      start_step=0,
      traj=None,
  )


class GroupQueueManagerTest(absltest.TestCase):

  def test_put_and_get_simple_batch(self):
    """Tests basic put and get functionality."""
    async def _run_test():
      manager = group_queue_manager.GroupQueueManager(group_size=2)
      item1 = _create_item("g1", 1)
      item2 = _create_item("g1", 1)

      await manager.put(item1)
      self.assertEqual(manager._open_bucket_count(), 1)
      self.assertEmpty(manager._ready_groups)

      await manager.put(item2)
      self.assertEqual(manager._open_bucket_count(), 0)
      self.assertLen(manager._ready_groups, 1)

      batch = await manager.get_batch(2)
      self.assertLen(batch, 2)
      self.assertCountEqual([item1, item2], batch)
    asyncio.run(_run_test())

  def test_get_batch_waits_for_items(self):
    """Tests that get_batch waits until a group is ready."""
    async def _run_test():
      manager = group_queue_manager.GroupQueueManager(group_size=2)
      item1 = _create_item("g1", 1)
      item2 = _create_item("g1", 1)

      async def producer():
        await asyncio.sleep(0.01)
        await manager.put(item1)
        await asyncio.sleep(0.01)
        await manager.put(item2)

      producer_task = asyncio.create_task(producer())
      batch = await manager.get_batch(2)

      self.assertLen(batch, 2)
      await producer_task
    asyncio.run(_run_test())

  def test_batching_with_leftovers(self):
    """Tests batching where a group is split across two get_batch calls."""
    async def _run_test():
      manager = group_queue_manager.GroupQueueManager(group_size=3)
      items = [_create_item("g1", 1, i) for i in range(3)]
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

  def test_max_open_buckets(self):
    """Tests that put blocks when max_open_buckets is reached."""
    async def _run_test():
      manager = group_queue_manager.GroupQueueManager(
          group_size=2, max_open_buckets=1
      )
      item_g1 = _create_item("g1", 1)
      item_g2 = _create_item("g2", 1)

      await manager.put(item_g1)

      put_task = asyncio.create_task(manager.put(item_g2))

      await asyncio.sleep(0.01)
      self.assertFalse(put_task.done())

      await manager.put(_create_item("g1", 1))
      await asyncio.sleep(0.01)

      await put_task
      self.assertEqual(manager._open_bucket_count(), 1)
      self.assertIn(("g2", 1), manager._buckets)
    asyncio.run(_run_test())

  def test_put_exception(self):
    """Tests that an exception is propagated to put and get calls."""
    async def _run_test():
      manager = group_queue_manager.GroupQueueManager(group_size=2)
      exc = ValueError("Test Exception")
      await manager.put_exception(exc)

      with self.assertRaises(ValueError):
        await manager.put(_create_item("g1", 1))

      with self.assertRaises(ValueError):
        await manager.get_batch(1)
    asyncio.run(_run_test())


if __name__ == "__main__":
  absltest.main()
