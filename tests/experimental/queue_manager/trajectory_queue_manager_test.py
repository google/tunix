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

  def test_get_group_batch_is_group_granular(self):
    """Tests get_group_batch counting groups, never splitting one."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager(
          num_generations=2
      )
      items = [
          _create_item(f"g{prompt}", group_index=i)
          for prompt in range(3)
          for i in range(2)
      ]
      for item in items:
        await manager.put(item)
      self.assertLen(manager._ready_groups, 3)

      batch1 = await manager.get_group_batch(2)
      self.assertLen(batch1, 4)
      self.assertCountEqual(items[:4], batch1)

      batch2 = await manager.get_group_batch(1)
      self.assertCountEqual(items[4:], batch2)
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

  def test_get_group_batch(self):
    """Tests retrieving multiple groups via get_group_batch."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager(
          num_generations=2
      )
      g1_items = [
          _create_item("g1", group_index=0),
          _create_item("g1", group_index=1),
      ]
      g2_items = [
          _create_item("g2", group_index=0),
          _create_item("g2", group_index=1),
      ]
      g3_items = [
          _create_item("g3", group_index=0),
          _create_item("g3", group_index=1),
      ]

      for item in g1_items + g2_items + g3_items:
        await manager.put(item)

      self.assertEqual(manager.ready_groups_count, 3)

      batch = await manager.get_group_batch(2)
      self.assertEqual(batch, g1_items + g2_items)
      self.assertEqual(manager.ready_groups_count, 1)

      remaining = await manager.get_group_batch(1)
      self.assertEqual(remaining, g3_items)
      self.assertEqual(manager.ready_groups_count, 0)

    asyncio.run(_run_test())

  def test_get_group_batch_early_exit_on_close(self):
    """Tests get_group_batch stops early when queue closes."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager(
          num_generations=2
      )
      g1_items = [
          _create_item("g1", group_index=0),
          _create_item("g1", group_index=1),
      ]
      for item in g1_items:
        await manager.put(item)
      await manager.close()

      batch = await manager.get_group_batch(3)
      self.assertEqual(batch, g1_items)
      self.assertEqual(manager.ready_groups_count, 0)

    asyncio.run(_run_test())


def _create_batch_manager(
    full_batch_size: int = 2,
    num_generations: int = 1,
    filter_fn=None,
    cursor_log: list[int] | None = None,
) -> trajectory_queue_manager.BatchOrderedQueueManager:
  """Helper to create a BatchOrderedQueueManager for testing."""
  return trajectory_queue_manager.BatchOrderedQueueManager(
      full_batch_size=full_batch_size,
      num_generations=num_generations,
      filter_fn=filter_fn,
      on_cursor_advance=None if cursor_log is None else cursor_log.append,
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
      _create_item(
          prompt_id, group_index=i, reward=reward, batch_idx=batch_idx
      )
      for i in range(num_generations)
  ]
  for item in group:
    await manager.put(item)
  return group


def _reject_negative_reward(
    group: list[datatypes.TrajectoryItem],
) -> list[datatypes.TrajectoryItem]:
  return [item for item in group if item.traj.reward > 0]


class BatchOrderedQueueManagerTest(absltest.TestCase):

  def test_serves_batches_in_order_regardless_of_arrival(self):
    """Tests a later batch arriving first is still served second."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=1)
      late = await _put_group(manager, "p1", batch_idx=1)
      early = await _put_group(manager, "p0", batch_idx=0)

      self.assertEqual(await manager.get_ordered_group(), (0, early))
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

  def test_batch_boundary_precedes_commit(self):
    """Tests the next batch being served before the previous one commits."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=1)
      await _put_group(manager, "p0", batch_idx=0)
      await _put_group(manager, "p1", batch_idx=1)

      batch_idx, _ = await manager.get_ordered_group()
      self.assertEqual(batch_idx, 0)

      # The index change is the boundary signal, so it arrives before the
      # consumer can commit the batch it just finished reading.
      next_batch_idx, _ = await manager.get_ordered_group()
      self.assertEqual(next_batch_idx, 1)
      self.assertEqual(manager.next_batch_idx, 0)

      manager.commit_batch(0)
      self.assertEqual(manager.next_batch_idx, 1)

    asyncio.run(_run_test())

  def test_short_batch_when_filter_drops_a_group(self):
    """Tests a batch serving fewer groups than expected after a drop."""

    async def _run_test():
      manager = _create_batch_manager(
          full_batch_size=2, filter_fn=_reject_negative_reward
      )
      kept = await _put_group(manager, "p0", batch_idx=0)
      await _put_group(manager, "p1", batch_idx=0, reward=-1.0)
      later = await _put_group(manager, "p2", batch_idx=1)

      self.assertEqual(await manager.get_ordered_group(), (0, kept))
      # The drop sealed batch 0 at one group, so batch 1 follows immediately.
      self.assertEqual(await manager.get_ordered_group(), (1, later))

      dropped = await manager.get_filtered_groups()
      self.assertLen(dropped, 1)

    asyncio.run(_run_test())

  def test_hole_batch_is_passed_over(self):
    """Tests a batch that loses every group not stranding the cursor."""

    async def _run_test():
      cursor_log: list[int] = []
      manager = _create_batch_manager(
          full_batch_size=1,
          filter_fn=_reject_negative_reward,
          cursor_log=cursor_log,
      )
      await _put_group(manager, "p0", batch_idx=0, reward=-1.0)
      survivor = await _put_group(manager, "p1", batch_idx=1)

      self.assertEqual(await manager.get_ordered_group(), (1, survivor))
      # Nothing to train in batch 0, so it advances without a commit.
      self.assertEqual(manager.next_batch_idx, 1)
      self.assertEqual(cursor_log, [1])

    asyncio.run(_run_test())

  def test_expect_seals_a_short_final_batch(self):
    """Tests expect() letting a batch seal below full_batch_size."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=4)
      manager.expect(0, num_groups=1)
      only = await _put_group(manager, "p0", batch_idx=0)
      later = await _put_group(manager, "p1", batch_idx=1)

      self.assertEqual(await manager.get_ordered_group(), (0, only))
      self.assertEqual(await manager.get_ordered_group(), (1, later))

    asyncio.run(_run_test())

  def test_expect_reopens_a_batch_that_sealed_early(self):
    """Tests a raised expect() un-sealing a batch still at the cursor."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=1)
      first = await _put_group(manager, "p0", batch_idx=0)
      # Sealed at the default count of 1, but not yet served, so the cursor
      # still sits on batch 0.
      self.assertEqual(manager.serving_batch_idx, 0)

      manager.expect(0, num_groups=2)
      second = await _put_group(manager, "p1", batch_idx=0)
      await _put_group(manager, "p2", batch_idx=1)

      # Both groups of batch 0 precede batch 1.
      self.assertEqual(await manager.get_ordered_group(), (0, first))
      self.assertEqual(await manager.get_ordered_group(), (0, second))
      self.assertEqual((await manager.get_ordered_group())[0], 1)

    asyncio.run(_run_test())

  def test_expect_does_not_reopen_a_drained_batch(self):
    """Tests the un-seal guard once the serving cursor has moved past."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=1)
      first = await _put_group(manager, "p0", batch_idx=0)
      later = await _put_group(manager, "p1", batch_idx=1)

      self.assertEqual(await manager.get_ordered_group(), (0, first))
      # Serving batch 1 is what tells the consumer batch 0 ended.
      self.assertEqual(await manager.get_ordered_group(), (1, later))
      self.assertEqual(manager.serving_batch_idx, 1)

      # Too late: batch 0 was already handed out in full.
      manager.expect(0, num_groups=2)
      self.assertEqual(manager.serving_batch_idx, 1)

      # A straggler for batch 0 must not stall batch 1's boundary, and is
      # released along with the rest of batch 0's state.
      await _put_group(manager, "p2", batch_idx=0)
      manager.commit_batch(0)
      self.assertEqual(manager.next_batch_idx, 1)
      self.assertEqual(manager.pending_groups_count, 0)

    asyncio.run(_run_test())

  def test_skip_seeds_the_cursor_for_resume(self):
    """Tests skip() starting the cursor mid-stream and discarding stragglers."""

    async def _run_test():
      cursor_log: list[int] = []
      manager = _create_batch_manager(full_batch_size=1, cursor_log=cursor_log)
      manager.skip(0, 2)

      self.assertEqual(manager.next_batch_idx, 2)
      self.assertEqual(cursor_log, [2])

      # A group of an already-passed batch is discarded, not staged.
      await _put_group(manager, "p1", batch_idx=1)
      self.assertEqual(manager.pending_groups_count, 0)

      resumed = await _put_group(manager, "p2", batch_idx=2)
      self.assertEqual(await manager.get_ordered_group(), (2, resumed))

    asyncio.run(_run_test())

  def test_commit_releases_batch_state(self):
    """Tests commit() advancing the cursor and forgetting the batch."""

    async def _run_test():
      cursor_log: list[int] = []
      manager = _create_batch_manager(full_batch_size=2, cursor_log=cursor_log)
      await _put_group(manager, "p0", batch_idx=0)
      await _put_group(manager, "p1", batch_idx=0)

      await manager.get_ordered_group()
      await manager.get_ordered_group()
      manager.commit_batch(0)

      self.assertEqual(manager.next_batch_idx, 1)
      self.assertEqual(cursor_log, [1])
      self.assertEqual(manager.pending_groups_count, 0)
      self.assertEmpty(manager._sealed)

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

  def test_staleness_filter_is_shared_by_both_factories(self):
    """Tests build_filter reaching both classes identically."""

    async def _run_test():
      version = 5
      for manager in (
          trajectory_queue_manager.TrajectoryQueueManager.create(
              num_generations=1, max_staleness=1, current_policy_version=lambda: 5
          ),
          trajectory_queue_manager.BatchOrderedQueueManager.create(
              num_generations=1,
              full_batch_size=1,
              max_staleness=1,
              current_policy_version=lambda: 5,
          ),
      ):
        stale = _create_item("p0", batch_idx=0)
        stale.policy_version = version - 3
        await manager.put(stale)
        self.assertLen(await manager.get_filtered_groups(), 1)

    asyncio.run(_run_test())

  def test_ordered_queue_rejects_the_inherited_get_batch(self):
    """Tests get_batch failing loudly instead of parking on an empty FIFO."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=1)
      await _put_group(manager, "p0", batch_idx=0)
      with self.assertRaises(NotImplementedError):
        await manager.get_batch(1)

    asyncio.run(_run_test())

  def test_get_ordered_group_with_batch_idx_stops_at_sealed_short_batch(self):
    """Tests get_ordered_group(batch_idx=b) returning None once b is drained."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=2)
      manager.expect(0, num_groups=1)
      only = await _put_group(manager, "p0", batch_idx=0)
      later = await _put_group(manager, "p1", batch_idx=1)

      self.assertEqual(await manager.get_ordered_group(), (0, only))
      self.assertIsNone(await manager.get_ordered_group(batch_idx=0))
      self.assertEqual(manager.serving_batch_idx, 1)
      self.assertEqual(await manager.get_ordered_group(), (1, later))

    asyncio.run(_run_test())

  def test_dropped_seals_short_batch_and_advances_over_hole(self):
    """Tests dropped() sealing a short batch and advancing over a 0-admitted hole."""

    async def _run_test():
      advanced: list[int] = []
      manager = _create_batch_manager(
          full_batch_size=2,
          cursor_log=advanced,
      )
      # Batch 0 loses all 2 groups -> hole: next_batch_idx advances to 1 immediately.
      manager.dropped(0, num_groups=2)
      self.assertEqual(manager.next_batch_idx, 1)
      self.assertEqual(manager.serving_batch_idx, 1)
      self.assertEqual(advanced, [1])

      # Batch 1 admits 1 group and drops 1 group -> seals at 1 group.
      g1 = await _put_group(manager, "p2", batch_idx=1)
      manager.dropped(1, num_groups=1)
      g2 = await _put_group(manager, "p4", batch_idx=2)

      self.assertEqual(await manager.get_ordered_group(), (1, g1))
      self.assertIsNone(await manager.get_ordered_group(batch_idx=1))
      self.assertEqual(manager.serving_batch_idx, 2)
      manager.commit_batch(1)
      self.assertEqual(manager.next_batch_idx, 2)
      self.assertEqual(await manager.get_ordered_group(), (2, g2))

      with self.assertRaises(ValueError):
        manager.dropped(2, num_groups=0)

    asyncio.run(_run_test())

  def test_raw_queue_on_drop_reports_to_ordered_queue(self):
    """Tests TrajectoryQueueManager(on_drop=ordered_q.dropped) forwarding drops."""

    async def _run_test():
      ordered = trajectory_queue_manager.BatchOrderedQueueManager.create(
          num_generations=2,
          full_batch_size=2,
      )
      raw = trajectory_queue_manager.TrajectoryQueueManager.create(
          num_generations=2,
          max_staleness=1,
          current_policy_version=lambda: 5,
          on_drop=ordered.dropped,
      )
      # Stale group from batch 0 (policy_version=1, staleness=4 > 1) is dropped by raw.
      for g_idx in range(2):
        stale = _create_item("p0", group_index=g_idx, batch_idx=0)
        stale.policy_version = 1
        await raw.put(stale)

      # Fresh group from batch 0 reaches ordered_q.
      fresh = await _put_group(ordered, "p1", batch_idx=0, num_generations=2)
      next_batch_group = await _put_group(
          ordered, "p2", batch_idx=1, num_generations=2
      )

      # Because raw reported dropped(0, 1), batch 0 is sealed after its 1 fresh group.
      self.assertEqual(await ordered.get_ordered_group(), (0, fresh))
      self.assertIsNone(await ordered.get_ordered_group(batch_idx=0))
      self.assertEqual(await ordered.get_ordered_group(), (1, next_batch_group))

    asyncio.run(_run_test())


if __name__ == "__main__":
  absltest.main()
