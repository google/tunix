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
    policy_version: int = 0,
    batch_idx: int | None = None,
) -> datatypes.TrajectoryItem:
  """Helper to create a TrajectoryItem for testing."""
  traj = datatypes.Trajectory(reward=reward)
  metadata: dict[str, object] = {"task_id": task_id}
  if batch_idx is not None:
    metadata["batch_idx"] = batch_idx
  return datatypes.TrajectoryItem(
      group_index=group_index,
      prompt_id=prompt_id,
      start_step=0,
      traj=traj,
      metadata=metadata,
      policy_version=policy_version,
  )


class QueueManagerTest(absltest.TestCase):

  def test_default_trajectory_grouping(self):
    """Tests default trajectory grouping by prompt_id up to num_generations."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager(
          num_generations=2
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
          num_generations=2, filter_fn=positive_reward_filter_fn
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
    """Tests item-granular get_batch splitting a group across calls."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager(
          num_generations=3
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
          num_generations=2
      )
      exc = ValueError("Test Exception")
      await manager.put_exception(exc)

      with self.assertRaises(ValueError):
        await manager.put(_create_item("g1", 0))

      with self.assertRaises(ValueError):
        await manager.get_batch(1)

    asyncio.run(_run_test())

  def test_staleness_filter_drops_stale_group_and_invokes_on_group_filtered(
      self,
  ):
    """Tests create() drops stale groups and notifies on_group_filtered."""

    async def _run_test():
      current_version = 5
      dropped_groups = []
      manager = trajectory_queue_manager.TrajectoryQueueManager.create(
          num_generations=2,
          max_staleness=1,
          current_policy_version=lambda: current_version,
          on_group_filtered=dropped_groups.append,
      )

      stale0 = _create_item("g_stale", group_index=0, policy_version=3)
      stale1 = _create_item("g_stale", group_index=1, policy_version=3)
      fresh0 = _create_item("g_fresh", group_index=0, policy_version=4)
      fresh1 = _create_item("g_fresh", group_index=1, policy_version=5)

      await manager.put(stale0)
      await manager.put(stale1)
      self.assertEqual(manager.ready_groups_count, 0)
      self.assertEqual(manager.filtered_groups_count, 1)
      self.assertEqual(dropped_groups, [[stale0, stale1]])

      await manager.put(fresh0)
      await manager.put(fresh1)
      self.assertEqual(manager.ready_groups_count, 1)
      self.assertEqual(manager.filtered_groups_count, 1)
      self.assertLen(dropped_groups, 1)

      batch = await manager.get_batch(2)
      self.assertEqual(batch, [fresh0, fresh1])

    asyncio.run(_run_test())

  def test_staleness_filter_drops_partially_stale_group(self):
    """Tests a group with mixed stale and fresh items is dropped as a whole."""

    async def _run_test():
      dropped_groups = []
      manager = trajectory_queue_manager.TrajectoryQueueManager.create(
          num_generations=2,
          max_staleness=1,
          current_policy_version=lambda: 5,
          on_group_filtered=dropped_groups.append,
      )

      stale_item = _create_item("g_mixed", group_index=0, policy_version=2)
      fresh_item = _create_item("g_mixed", group_index=1, policy_version=5)

      await manager.put(stale_item)
      await manager.put(fresh_item)

      self.assertEqual(manager.ready_groups_count, 0)
      self.assertEqual(manager.filtered_groups_count, 1)
      self.assertEqual(dropped_groups, [[stale_item, fresh_item]])

    asyncio.run(_run_test())

  def test_staleness_filter_falls_back_to_metadata_when_traj_policy_version_is_none(
      self,
  ):
    """Tests staleness filter reading metadata['policy_version'] when traj['policy_version'] is None."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager.create(
          num_generations=1,
          max_staleness=1,
          current_policy_version=lambda: 2,
      )
      fresh = datatypes.TrajectoryItem(
          prompt_id="p_fresh",
          group_index=0,
          traj={"policy_version": None, "status": "SUCCEEDED"},
          metadata={"policy_version": 1},
      )
      stale = datatypes.TrajectoryItem(
          prompt_id="p_stale",
          group_index=0,
          traj={"policy_version": None, "status": "SUCCEEDED"},
          metadata={"policy_version": 0},
      )

      await manager.put(fresh)
      await manager.put(stale)

      self.assertEqual(manager.ready_groups_count, 1)
      self.assertEqual(manager.filtered_groups_count, 1)
      self.assertEqual(await manager.get_group(), [fresh])

    asyncio.run(_run_test())


def _create_batch_manager(
    full_batch_size: int = 2,
    num_generations: int = 1,
    filter_fn=None,
) -> trajectory_queue_manager.BatchOrderedQueueManager:
  """Helper to create a BatchOrderedQueueManager for testing."""
  return trajectory_queue_manager.BatchOrderedQueueManager(
      full_batch_size=full_batch_size,
      num_generations=num_generations,
      filter_fn=filter_fn,
  )


async def _put_group(
    manager: trajectory_queue_manager.BatchOrderedQueueManager,
    prompt_id: str,
    batch_idx: int,
    num_generations: int = 1,
    reward: float = 1.0,
) -> list[datatypes.TrajectoryItem]:
  """Puts one whole group of `num_generations` items, and returns it."""
  group = [
      _create_item(prompt_id, group_index=i, reward=reward, batch_idx=batch_idx)
      for i in range(num_generations)
  ]
  for item in group:
    await manager.put(item)
  return group


class BatchOrderedQueueManagerTest(absltest.TestCase):

  def test_serves_batches_in_order_regardless_of_arrival(self):
    """Tests a later batch arriving first is still served second."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=1)
      late = await _put_group(manager, "p1", batch_idx=1)
      early = await _put_group(manager, "p0", batch_idx=0)

      self.assertEqual(await manager.get_ordered_group(), (0, early))
      self.assertIsNone(await manager.get_ordered_group())
      manager.commit_batch(0)
      self.assertEqual(await manager.get_ordered_group(), (1, late))

    asyncio.run(_run_test())

  def test_withholds_later_batch_until_cursor_batch_completes(self):
    """Tests the cursor batch blocking a ready group of the next batch."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=2)
      first = await _put_group(manager, "p0", batch_idx=0)
      await _put_group(manager, "p2", batch_idx=1)

      self.assertEqual(await manager.get_ordered_group(), (0, first))

      # Batch 0 still owes a group, so batch 1 must not be served yet.
      parked = asyncio.ensure_future(manager.get_ordered_group())
      await asyncio.sleep(0)  # Let the task reach the `_have_ready` wait.
      self.assertFalse(parked.done())

      second = await _put_group(manager, "p1", batch_idx=0)
      self.assertEqual(await parked, (0, second))

    asyncio.run(_run_test())

  def test_get_ordered_group_stops_at_batch_boundary_until_commit(self):
    """Tests get_ordered_group returning None once next_batch_idx is drained."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=1)
      await _put_group(manager, "p0", batch_idx=0)
      later = await _put_group(manager, "p1", batch_idx=1)

      batch_idx, _ = await manager.get_ordered_group()
      self.assertEqual(batch_idx, 0)

      # Drained full_batch_size groups: returns None without crossing into
      # batch 1.
      self.assertIsNone(await manager.get_ordered_group())
      self.assertEqual(manager.next_batch_idx, 0)

      manager.commit_batch(0)
      self.assertEqual(manager.next_batch_idx, 1)
      self.assertEqual(await manager.get_ordered_group(), (1, later))

    asyncio.run(_run_test())

  def test_skip_seeds_the_cursor_for_resume(self):
    """Tests skip() starting the cursor mid-stream and discarding stragglers."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=1)
      manager.skip(0, 2)

      self.assertEqual(manager.next_batch_idx, 2)

      # A group of an already-passed batch is discarded, not staged.
      await _put_group(manager, "p1", batch_idx=1)
      self.assertEqual(manager.pending_groups_count, 0)

      resumed = await _put_group(manager, "p2", batch_idx=2)
      self.assertEqual(await manager.get_ordered_group(), (2, resumed))

    asyncio.run(_run_test())

  def test_commit_releases_batch_state(self):
    """Tests commit() advancing the cursor and forgetting the batch."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=2)
      await _put_group(manager, "p0", batch_idx=0)
      await _put_group(manager, "p1", batch_idx=0)

      await manager.get_ordered_group()
      await manager.get_ordered_group()
      manager.commit_batch(0)

      self.assertEqual(manager.next_batch_idx, 1)
      self.assertEqual(manager.pending_groups_count, 0)

    asyncio.run(_run_test())

  def test_close_ends_an_incomplete_batch(self):
    """Tests EOF returning None rather than waiting for a batch to fill."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=2)
      only = await _put_group(manager, "p0", batch_idx=0)

      self.assertEqual(await manager.get_ordered_group(), (0, only))
      await manager.close()
      self.assertIsNone(await manager.get_ordered_group())

    asyncio.run(_run_test())

  def test_abort_unblocks_a_waiting_consumer(self):
    """Tests abort() raising in a consumer parked on the cursor."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=1)
      waiter = asyncio.ensure_future(manager.get_ordered_group())
      await asyncio.sleep(0)

      await manager.abort(ValueError("Test Exception"))

      with self.assertRaises(ValueError):
        await waiter

    asyncio.run(_run_test())

  def test_requires_batch_idx_metadata(self):
    """Tests an untagged item failing loudly instead of being mis-routed."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=1)

      with self.assertRaises(ValueError):
        await manager.put(_create_item("p0"))

    asyncio.run(_run_test())

  def test_each_class_has_its_own_create(self):
    """Tests both factories returning a single concrete class, not a union."""
    arrival = trajectory_queue_manager.TrajectoryQueueManager.create(
        num_generations=2
    )
    self.assertIsInstance(
        arrival, trajectory_queue_manager.TrajectoryQueueManager
    )
    self.assertNotIsInstance(
        arrival, trajectory_queue_manager.BatchOrderedQueueManager
    )

    ordered = trajectory_queue_manager.BatchOrderedQueueManager.create(
        num_generations=2, full_batch_size=4
    )
    self.assertIsInstance(
        ordered, trajectory_queue_manager.BatchOrderedQueueManager
    )
    self.assertNotIsInstance(
        ordered, trajectory_queue_manager.TrajectoryQueueManager
    )
    self.assertEqual(ordered.full_batch_size, 4)

    with self.assertRaises(ValueError):
      trajectory_queue_manager.BatchOrderedQueueManager.create(
          num_generations=2, full_batch_size=0
      )

  def test_ordered_queue_get_batch_stops_at_batch_boundary_until_commit(self):
    """Tests get_batch draining the active batch without crossing into the next batch."""

    async def _run_test():
      manager = _create_batch_manager(num_generations=2, full_batch_size=1)
      b1_g0 = await _put_group(manager, "p1", batch_idx=1, num_generations=2)
      b0_g0 = await _put_group(manager, "p0", batch_idx=0, num_generations=2)

      # First get_batch(2) returns the only group of batch 0.
      self.assertEqual(await manager.get_batch(2), b0_g0)
      # Second get_batch(2) before commit() returns [] even though batch 1 is
      # ready.
      self.assertEqual(await manager.get_batch(2), [])
      self.assertEqual(manager.next_batch_idx, 0)

      # After commit(), get_batch(2) advances to batch 1.
      manager.commit(step=0)
      self.assertEqual(manager.next_batch_idx, 1)
      self.assertEqual(await manager.get_batch(2), b1_g0)

    asyncio.run(_run_test())

  def test_clear_resets_pending_and_batch_cursor(self):
    """Tests clear() emptying _pending and resetting _served_in_batch and _next_batch_idx."""

    async def _run_test():
      manager = _create_batch_manager(num_generations=2, full_batch_size=2)
      await _put_group(manager, "p0", batch_idx=0, num_generations=2)
      await _put_group(manager, "p1", batch_idx=1, num_generations=2)
      await manager.put(_create_item("incomplete", group_index=0, batch_idx=0))

      # Serve one group of batch 0 and advance cursor to batch 1.
      self.assertIsNotNone(await manager.get_ordered_group())
      manager.commit_batch(0)
      self.assertEqual(manager.next_batch_idx, 1)
      self.assertEqual(manager.pending_groups_count, 1)
      self.assertEqual(manager.incomplete_buckets_count, 1)

      await manager.clear()
      self.assertEqual(manager.next_batch_idx, 0)
      self.assertEqual(manager.pending_groups_count, 0)
      self.assertEqual(manager.incomplete_buckets_count, 0)

      fresh = await _put_group(
          manager, "p_fresh", batch_idx=0, num_generations=2
      )
      self.assertEqual(await manager.get_ordered_group(), (0, fresh))

    asyncio.run(_run_test())


if __name__ == "__main__":
  absltest.main()
