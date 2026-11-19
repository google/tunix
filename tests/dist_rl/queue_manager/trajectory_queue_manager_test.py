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
from absl.testing import parameterized
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.queue_manager import trajectory_queue_manager
from tunix.rl.agentic.queue_manager import group_queue_manager


def _token_traj(
    reward: float = 1.0,
    status: datatypes.TrajectoryStatus = datatypes.TrajectoryStatus.SUCCEEDED,
) -> dict[str, object]:
  """Minimal Token-mode trajectory dict, the shape `raw_q` sees in production."""
  return {"status": status, "trajectory_reward": reward}


def _create_item(
    prompt_id: str,
    group_index: int = 0,
    task_id: str = "",
    reward: float = 1.0,
    policy_version: int = 0,
    batch_idx: int | None = None,
) -> datatypes.TrajectoryItem:
  """Helper to create a TrajectoryItem for testing."""
  metadata: dict[str, object] = {
      "task_id": task_id,
  }
  if batch_idx is not None:
    metadata["batch_idx"] = batch_idx
  return datatypes.TrajectoryItem(
      group_index=group_index,
      prompt_id=prompt_id,
      start_step=0,
      traj={
          **_token_traj(reward=reward),
          "policy_versions": [policy_version],
      },
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
        return [item for item in group if item.traj["trajectory_reward"] > 0]

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

  def test_staleness_filter_uses_latest_in_flight_policy_versions(self):
    """Tests staleness filter evaluating the latest turn of `traj['policy_versions']`."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager.create(
          num_generations=1,
          max_staleness=1,
          current_policy_version=lambda: 3,
      )
      updated_inflight = datatypes.TrajectoryItem(
          prompt_id="p_inflight",
          group_index=0,
          traj={"policy_versions": [1, 2], "status": "SUCCEEDED"},
      )
      genuinely_stale = datatypes.TrajectoryItem(
          prompt_id="p_stale",
          group_index=0,
          traj={"policy_versions": [1, 1], "status": "SUCCEEDED"},
      )

      await manager.put(updated_inflight)
      await manager.put(genuinely_stale)

      self.assertEqual(manager.ready_groups_count, 1)
      self.assertEqual(manager.filtered_groups_count, 1)
      self.assertEqual(await manager.get_group(), [updated_inflight])

    asyncio.run(_run_test())

  def test_max_version_span_filters_groups_spanning_too_many_versions(self):
    """Tests max_version_span rejecting groups whose max-min policy_versions exceeds the bound."""
    with self.assertRaises(ValueError):
      trajectory_queue_manager.TrajectoryQueueManager.create(
          num_generations=1,
          max_version_span=-1,
      )
    with self.assertRaises(ValueError):
      trajectory_queue_manager.item_policy_version(
          datatypes.TrajectoryItem(
              prompt_id="p_empty_versions",
              group_index=0,
              traj={"status": "SUCCEEDED", "policy_versions": []},
          )
      )
    with self.assertRaises(ValueError):
      trajectory_queue_manager.item_policy_version(
          datatypes.TrajectoryItem(
              prompt_id="p_missing_version",
              group_index=0,
              traj={"status": "SUCCEEDED"},
          )
      )

    async def _run_test():
      # 1. Combined staleness + max_version_span filtering.
      manager = trajectory_queue_manager.TrajectoryQueueManager.create(
          num_generations=1,
          max_staleness=1,
          max_version_span=2,
          current_policy_version=lambda: 3,
      )
      bounded_span = datatypes.TrajectoryItem(
          prompt_id="p_bounded",
          group_index=0,
          traj={"policy_versions": [1, 2, 3], "status": "SUCCEEDED"},
      )
      excessive_span = datatypes.TrajectoryItem(
          prompt_id="p_excessive",
          group_index=0,
          traj={"policy_versions": [0, 1, 2, 3], "status": "SUCCEEDED"},
      )

      await manager.put(bounded_span)
      await manager.put(excessive_span)

      self.assertEqual(manager.ready_groups_count, 1)
      self.assertEqual(manager.filtered_groups_count, 1)
      self.assertEqual(await manager.get_group(), [bounded_span])

      # 2. Span-only filtering (max_staleness=0, current_policy_version=None).
      span_only = trajectory_queue_manager.TrajectoryQueueManager.create(
          num_generations=1,
          max_staleness=0,
          max_version_span=1,
      )
      await span_only.put(bounded_span)
      self.assertEqual(span_only.ready_groups_count, 0)
      self.assertEqual(span_only.filtered_groups_count, 1)

      # 3. Unbounded span when max_version_span=None (default).
      unbounded = trajectory_queue_manager.TrajectoryQueueManager.create(
          num_generations=1,
          max_staleness=1,
          max_version_span=None,
          current_policy_version=lambda: 3,
      )
      await unbounded.put(excessive_span)
      self.assertEqual(unbounded.ready_groups_count, 1)
      self.assertEqual(unbounded.filtered_groups_count, 0)

    asyncio.run(_run_test())

  def test_multi_generation_partial_rollout_group_staleness_and_invalid_items(
      self,
  ):
    """Tests G>1 group completion staleness, group-wide span, and skipping invalid items."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager.create(
          num_generations=2,
          max_staleness=1,
          max_version_span=2,
          current_policy_version=lambda: 3,
      )

      # 1. Early-finishing rollout (v=1) paired with multi-turn sibling
      # ([1, 2, 3]): group completed at v=3 (>= 2) with group span 3 - 1 = 2.
      early_sibling = datatypes.TrajectoryItem(
          prompt_id="p_mixed_valid",
          group_index=0,
          traj={"policy_versions": [1], "status": "SUCCEEDED"},
      )
      long_sibling = datatypes.TrajectoryItem(
          prompt_id="p_mixed_valid",
          group_index=1,
          traj={"policy_versions": [1, 2, 3], "status": "SUCCEEDED"},
      )
      await manager.put(early_sibling)
      self.assertEqual(manager.incomplete_buckets_count, 1)
      await manager.put(long_sibling)
      self.assertEqual(manager.incomplete_buckets_count, 0)
      self.assertEqual(manager.ready_groups_count, 1)
      self.assertEqual(
          await manager.get_group(), [early_sibling, long_sibling]
      )

      # 2. Cross-sibling span 3 - 0 = 3 > max_version_span=2 -> rejected.
      too_old_sibling = datatypes.TrajectoryItem(
          prompt_id="p_cross_span_exceeded",
          group_index=0,
          traj={"policy_versions": [0], "status": "SUCCEEDED"},
      )
      recent_sibling = datatypes.TrajectoryItem(
          prompt_id="p_cross_span_exceeded",
          group_index=1,
          traj={"policy_versions": [1, 2, 3], "status": "SUCCEEDED"},
      )
      await manager.put(too_old_sibling)
      await manager.put(recent_sibling)
      self.assertEqual(manager.ready_groups_count, 0)
      self.assertEqual(manager.filtered_groups_count, 1)

      # 3. Tombstoned/failed error item (is_valid=False, even without version
      # or with old version) does not cause valid sibling group to be dropped.
      failed_tombstone = datatypes.TrajectoryItem(
          prompt_id="p_one_failed",
          group_index=0,
          traj={"status": "FAILED"},
          metadata={},
          is_valid=False,
      )
      valid_sibling = datatypes.TrajectoryItem(
          prompt_id="p_one_failed",
          group_index=1,
          traj={"policy_versions": [2, 3], "status": "SUCCEEDED"},
      )
      await manager.put(failed_tombstone)
      await manager.put(valid_sibling)
      self.assertEqual(manager.ready_groups_count, 1)
      self.assertEqual(
          await manager.get_batch(num_groups=1),
          [failed_tombstone, valid_sibling],
      )

      # 4. Group where all valid items completed at v=1 (< 2) is dropped.
      stale_a = datatypes.TrajectoryItem(
          prompt_id="p_all_stale",
          group_index=0,
          traj={"policy_versions": [1], "status": "SUCCEEDED"},
      )
      stale_b = datatypes.TrajectoryItem(
          prompt_id="p_all_stale",
          group_index=1,
          traj={"policy_versions": [1], "status": "SUCCEEDED"},
      )
      await manager.put(stale_a)
      await manager.put(stale_b)
      self.assertEqual(manager.ready_groups_count, 0)
      self.assertEqual(manager.filtered_groups_count, 2)

      manager.commit(step=1, groups=[])
      await manager.close()
      collected = [g async for g in manager]
      self.assertEmpty(collected)

    asyncio.run(_run_test())

  def test_staleness_filter_enforced_when_max_staleness_zero(self):
    """Tests max_staleness=0 installs filter and enforces exact version."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager.create(
          num_generations=1,
          max_staleness=0,
          current_policy_version=lambda: 3,
      )
      await manager.put(_create_item("p_old", group_index=0, policy_version=2))
      await manager.put(
          _create_item("p_future", group_index=0, policy_version=4)
      )
      exact = _create_item("p_exact", group_index=0, policy_version=3)
      await manager.put(exact)

      self.assertEqual(manager.ready_groups_count, 1)
      self.assertEqual(manager.filtered_groups_count, 2)
      self.assertEqual(await manager.get_group(), [exact])

    asyncio.run(_run_test())

  def test_no_staleness_filter_without_current_policy_version(self):
    """Tests create() without current_policy_version installs no filter."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager.create(
          num_generations=2, max_staleness=0
      )
      # This is how PROMPT_ARRIVAL mode builds `raw_q`: no version callback,
      # so nothing is checked or dropped here, not even a missing version.
      unversioned = datatypes.TrajectoryItem(
          prompt_id="p_any", group_index=0, traj=_token_traj()
      )
      versioned = _create_item("p_any", group_index=1, policy_version=99)
      await manager.put(unversioned)
      await manager.put(versioned)

      self.assertEqual(manager.ready_groups_count, 1)
      self.assertEqual(manager.filtered_groups_count, 0)
      self.assertEqual(await manager.get_group(), [unversioned, versioned])

    asyncio.run(_run_test())

  def test_staleness_filter_two_sided_scalar_policy_version(self):
    """Tests scalar policy_version two-sided window and missing version."""

    async def _run_test():
      manager = trajectory_queue_manager.TrajectoryQueueManager.create(
          num_generations=2,
          max_staleness=2,
          current_policy_version=lambda: 5,
      )
      await manager.put(_create_item("p_ok", group_index=0, policy_version=3))
      await manager.put(_create_item("p_ok", group_index=1, policy_version=5))
      self.assertEqual(manager.ready_groups_count, 1)

      # Future version v > cur_v (6 > 5) is dropped.
      await manager.put(
          _create_item("p_future", group_index=0, policy_version=6)
      )
      await manager.put(
          _create_item("p_future", group_index=1, policy_version=5)
      )
      self.assertEqual(manager.ready_groups_count, 1)
      self.assertEqual(manager.filtered_groups_count, 1)

      # Non-monotonic per-turn versions (e.g. [5, 4] straddling a checkpoint
      # rollback) are dropped even though both 5 and 4 lie inside [3, 5].
      await manager.put(
          datatypes.TrajectoryItem(
              prompt_id="p_regressed",
              group_index=0,
              traj={
                  **_token_traj(),
                  "policy_versions": [5, 4],
              },
          )
      )
      await manager.put(
          _create_item("p_regressed", group_index=1, policy_version=5)
      )
      self.assertEqual(manager.ready_groups_count, 1)
      self.assertEqual(manager.filtered_groups_count, 2)

      # Missing policy_version on a valid item fails fast with ValueError.
      empty_pv_0 = datatypes.TrajectoryItem(
          prompt_id="p_empty_pv", group_index=0, traj=_token_traj()
      )
      empty_pv_1 = _create_item("p_empty_pv", group_index=1, policy_version=5)
      await manager.put(empty_pv_0)
      with self.assertRaisesRegex(ValueError, "is missing policy_version"):
        await manager.put(empty_pv_1)

      # Stale sibling arriving BEFORE a missing-policy_version sibling must also
      # fail fast regardless of arrival order.
      await manager.put(
          _create_item("p_stale_then_empty", group_index=0, policy_version=1)
      )
      with self.assertRaisesRegex(ValueError, "is missing policy_version"):
        await manager.put(
            datatypes.TrajectoryItem(
                prompt_id="p_stale_then_empty",
                group_index=1,
                traj=_token_traj(),
            )
        )

    asyncio.run(_run_test())

  def test_staleness_filter_judges_valid_items_only(self):
    """Tests invalid items neither fail fast nor count toward the window.

    A 0-turn FAILED tombstone has no sampled version, so it must not raise
    and must not drag its valid sibling out of the window. A group with no
    valid item has nothing to judge and passes through (it trains with zero
    advantage) instead of being dropped as if it were version 0.
    """

    async def _run_test():
      dropped_groups = []
      manager = trajectory_queue_manager.TrajectoryQueueManager.create(
          num_generations=2,
          max_staleness=1,
          current_policy_version=lambda: 5,
          on_group_filtered=dropped_groups.append,
      )
      failed_tombstone = datatypes.TrajectoryItem(
          prompt_id="p_partial_fail",
          group_index=0,
          traj={"status": datatypes.TrajectoryStatus.FAILED},
      )
      valid_sibling = _create_item(
          "p_partial_fail", group_index=1, policy_version=5
      )
      await manager.put(failed_tombstone)
      await manager.put(valid_sibling)
      self.assertEqual(manager.ready_groups_count, 1)
      self.assertEqual(
          await manager.get_group(), [failed_tombstone, valid_sibling]
      )

      # Every item invalid: versions 0 (would be stale) and None are ignored.
      all_failed_0 = datatypes.TrajectoryItem(
          prompt_id="p_all_fail",
          group_index=0,
          traj={"status": datatypes.TrajectoryStatus.FAILED},
          metadata={"policy_version": 0},
      )
      all_failed_1 = datatypes.TrajectoryItem(
          prompt_id="p_all_fail",
          group_index=1,
          traj={"status": datatypes.TrajectoryStatus.TIMEOUT},
      )
      await manager.put(all_failed_0)
      await manager.put(all_failed_1)
      self.assertEqual(manager.ready_groups_count, 1)
      self.assertEqual(await manager.get_group(), [all_failed_0, all_failed_1])
      self.assertEmpty(dropped_groups)

    asyncio.run(_run_test())


class FifoGroupQueueTest(absltest.TestCase):

  def test_serves_groups_in_fifo_order_and_returns_none_at_eof(self):
    async def _run_test():
      q: trajectory_queue_manager.FifoGroupQueue[str] = (
          trajectory_queue_manager.FifoGroupQueue.create()
      )
      await q.put("g0")
      await q.put("g1")
      self.assertEqual(q.ready_groups_count, 2)

      self.assertEqual(await q.get(), "g0")
      self.assertEqual(await q.get(), "g1")
      self.assertEqual(q.ready_groups_count, 0)

      await q.close()
      self.assertIsNone(await q.get())

      with self.assertRaises(RuntimeError):
        await q.put("g2")

    asyncio.run(_run_test())

  def test_priority_fn_breaks_ties_among_simultaneously_ready_groups(self):
    async def _run_test():
      calls = 0

      def _priority(g: tuple[str, int]) -> int:
        nonlocal calls
        calls += 1
        return g[1]

      q: trajectory_queue_manager.FifoGroupQueue[tuple[str, int]] = (
          trajectory_queue_manager.FifoGroupQueue.create(priority_fn=_priority)
      )
      await q.put(("newer", 2))
      await q.put(("older_a", 1))
      await q.put(("older_b", 1))
      self.assertEqual(calls, 3)

      self.assertEqual(await q.get(), ("older_a", 1))
      self.assertEqual(await q.get(), ("older_b", 1))
      self.assertEqual(await q.get(), ("newer", 2))
      self.assertEqual(calls, 3)

    asyncio.run(_run_test())

  def test_abort_unblocks_waiting_get_and_raises_on_future_ops(self):
    async def _run_test():
      q: trajectory_queue_manager.FifoGroupQueue[str] = (
          trajectory_queue_manager.FifoGroupQueue.create()
      )
      waiter = asyncio.ensure_future(q.get())
      await asyncio.sleep(0)
      self.assertFalse(waiter.done())

      await q.abort(ValueError("FIFO aborted"))
      with self.assertRaisesRegex(ValueError, "FIFO aborted"):
        await waiter

      await q.clear()
      await q.put("g_after_clear")
      self.assertEqual(await q.get(), "g_after_clear")

    asyncio.run(_run_test())


def _create_batch_manager(
    full_batch_size: int = 2,
) -> trajectory_queue_manager.BatchOrderedGroupQueue[
    list[datatypes.TrajectoryItem]
]:
  """Helper to create a BatchOrderedGroupQueue for testing."""
  return trajectory_queue_manager.BatchOrderedGroupQueue.create(
      full_batch_size=full_batch_size,
      batch_idx_fn=lambda group: trajectory_queue_manager.default_batch_idx_fn(
          group[0]
      ),
  )


async def _put_group(
    manager: trajectory_queue_manager.BatchOrderedGroupQueue[
        list[datatypes.TrajectoryItem]
    ],
    prompt_id: str,
    batch_idx: int,
    num_generations: int = 1,
    reward: float = 1.0,
) -> list[datatypes.TrajectoryItem]:
  """Puts one whole pre-formed group of `num_generations` items, and returns it."""
  group = [
      _create_item(prompt_id, group_index=i, reward=reward, batch_idx=batch_idx)
      for i in range(num_generations)
  ]
  await manager.put(group)
  return group


class BatchOrderedGroupQueueTest(absltest.TestCase):

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

      # A group of an already-passed batch is discarded and counted, not staged.
      await _put_group(manager, "p1", batch_idx=1)
      self.assertEqual(manager.pending_groups_count, 0)
      self.assertEqual(manager.dropped_late_groups_count, 1)

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

      with self.assertRaises(RuntimeError):
        await _put_group(manager, "p_after_close", batch_idx=0)

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
    """Tests an untagged group failing loudly instead of being mis-routed."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=1)

      with self.assertRaises(ValueError):
        await manager.put([_create_item("p0")])

    asyncio.run(_run_test())

  def test_each_class_has_its_own_create(self):
    """Tests factories returning concrete queue classes."""
    raw = trajectory_queue_manager.TrajectoryQueueManager.create(
        num_generations=2
    )
    self.assertIsInstance(raw, trajectory_queue_manager.TrajectoryQueueManager)

    fifo = trajectory_queue_manager.FifoGroupQueue.create()
    self.assertIsInstance(fifo, trajectory_queue_manager.FifoGroupQueue)

    ordered = trajectory_queue_manager.BatchOrderedGroupQueue.create(
        full_batch_size=4,
        batch_idx_fn=lambda x: 0,
    )
    self.assertIsInstance(
        ordered, trajectory_queue_manager.BatchOrderedGroupQueue
    )
    self.assertEqual(ordered.full_batch_size, 4)

    with self.assertRaises(ValueError):
      trajectory_queue_manager.BatchOrderedGroupQueue.create(
          full_batch_size=0,
          batch_idx_fn=lambda x: 0,
      )

  def test_ordered_queue_get_stops_at_batch_boundary_until_commit(self):
    """Tests get() draining the active batch without crossing into the next batch."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=1)
      b1_g0 = await _put_group(manager, "p1", batch_idx=1, num_generations=2)
      b0_g0 = await _put_group(manager, "p0", batch_idx=0, num_generations=2)

      # First get() returns the only group of batch 0.
      self.assertEqual(await manager.get(), b0_g0)
      # Second get() before commit() returns None even though batch 1 is ready.
      self.assertIsNone(await manager.get())
      self.assertEqual(manager.next_batch_idx, 0)

      # After commit(), get() advances to batch 1.
      manager.commit(step=0)
      self.assertEqual(manager.next_batch_idx, 1)
      self.assertEqual(await manager.get(), b1_g0)

    asyncio.run(_run_test())

  def test_clear_resets_pending_and_batch_cursor(self):
    """Tests clear() emptying _pending and resetting _served_in_batch and _next_batch_idx."""

    async def _run_test():
      manager = _create_batch_manager(full_batch_size=2)
      await _put_group(manager, "p0", batch_idx=0, num_generations=2)
      await _put_group(manager, "p1", batch_idx=1, num_generations=2)

      # Serve one group of batch 0 and advance cursor to batch 1.
      self.assertIsNotNone(await manager.get_ordered_group())
      manager.commit_batch(0)
      await _put_group(manager, "p_late", batch_idx=0, num_generations=2)
      self.assertEqual(manager.next_batch_idx, 1)
      self.assertEqual(manager.pending_groups_count, 1)
      self.assertEqual(manager.dropped_late_groups_count, 1)

      await manager.clear()
      self.assertEqual(manager.next_batch_idx, 0)
      self.assertEqual(manager.pending_groups_count, 0)
      self.assertEqual(manager.dropped_late_groups_count, 0)

      fresh = await _put_group(
          manager, "p_fresh", batch_idx=0, num_generations=2
      )
      self.assertEqual(await manager.get_ordered_group(), (0, fresh))

    asyncio.run(_run_test())


class GroupQueueDropCallbackTest(parameterized.TestCase):
  """Covers `on_group_dropped` across FIFO and batch-ordered group queues."""

  @parameterized.named_parameters(
      dict(testcase_name="fifo_clear", mode="fifo_clear"),
      dict(testcase_name="fifo_put_after_abort", mode="fifo_put_after_abort"),
      dict(
          testcase_name="fifo_put_abort_under_lock",
          mode="fifo_put_abort_under_lock",
      ),
      dict(testcase_name="ordered_clear", mode="ordered_clear"),
      dict(
          testcase_name="ordered_put_after_abort",
          mode="ordered_put_after_abort",
      ),
      dict(
          testcase_name="ordered_put_abort_under_lock",
          mode="ordered_put_abort_under_lock",
      ),
      dict(testcase_name="ordered_late_arrival", mode="ordered_late_arrival"),
      dict(testcase_name="ordered_skip_pending", mode="ordered_skip_pending"),
  )
  def test_on_group_dropped_invoked_once_per_discarded_group(self, mode: str):
    async def _run():
      dropped: list[tuple[str, int]] = []
      if mode.startswith("fifo_"):
        q = trajectory_queue_manager.FifoGroupQueue.create(
            on_group_dropped=dropped.append
        )
        kept = ("kept", 0)
        await q.put(kept)
        self.assertEqual(await q.get(), kept)
        self.assertEmpty(dropped)

        if mode == "fifo_clear":
          await q.put(("g0", 0))
          await q.put(("g1", 0))
          await q.clear()
          self.assertEqual(dropped, [("g0", 0), ("g1", 0)])
        elif mode == "fifo_put_after_abort":
          await q.abort(RuntimeError("boom"))
          await q.put(("g_abort", 0))
          self.assertEqual(dropped, [("g_abort", 0)])
        else:
          async with q._lock:
            put_task = asyncio.create_task(q.put(("g_locked_abort", 0)))
            await asyncio.sleep(0)
            q._clearing = True
          await put_task
          self.assertEqual(dropped, [("g_locked_abort", 0)])
        return

      ordered = trajectory_queue_manager.BatchOrderedGroupQueue.create(
          full_batch_size=1,
          batch_idx_fn=lambda g: g[1],
          on_group_dropped=dropped.append,
      )
      kept = ("kept", 0)
      await ordered.put(kept)
      self.assertEqual(await ordered.get(), kept)
      self.assertEmpty(dropped)

      if mode == "ordered_clear":
        ordered.commit_batch(0)
        await ordered.put(("b1", 1))
        await ordered.put(("b2", 2))
        await ordered.clear()
        self.assertEqual(dropped, [("b1", 1), ("b2", 2)])
      elif mode == "ordered_put_after_abort":
        await ordered.abort(RuntimeError("boom"))
        await ordered.put(("b_abort", 1))
        self.assertEqual(dropped, [("b_abort", 1)])
      elif mode == "ordered_put_abort_under_lock":
        async with ordered._lock:
          put_task = asyncio.create_task(ordered.put(("b_locked_abort", 1)))
          await asyncio.sleep(0)
          ordered._clearing = True
        await put_task
        self.assertEqual(dropped, [("b_locked_abort", 1)])
      elif mode == "ordered_late_arrival":
        ordered.commit_batch(0)
        await ordered.put(("b_late", 0))
        self.assertEqual(dropped, [("b_late", 0)])
      else:
        ordered.commit_batch(0)
        await ordered.put(("b1_skip", 1))
        await ordered.put(("b2_skip", 2))
        await ordered.put(("b3_keep", 3))
        ordered.skip(1, 3)
        self.assertEqual(dropped, [("b1_skip", 1), ("b2_skip", 2)])
        self.assertEqual(await ordered.get(), ("b3_keep", 3))

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
