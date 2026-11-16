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

"""Unit tests for RolloutStagingBuffer."""

from __future__ import annotations

import asyncio
import threading
import time

from absl.testing import absltest
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.worker import staging_buffer as sb_lib


class RolloutStagingBufferTest(absltest.TestCase):

  def test_commit_fetch_release_and_incarnation_isolation(self):
    buf_a = sb_lib.RolloutStagingBuffer(
        "rollout-0", "inst-a", max_bytes=1024, incarnation=7
    )
    buf_b = sb_lib.RolloutStagingBuffer(
        "rollout-0", "inst-b", max_bytes=1024, incarnation=8
    )

    async def _stage(buf: sb_lib.RolloutStagingBuffer, val: int):
      res = await buf.reserve(64)
      arr = np.full((4, 2, 2), val, dtype=np.uint8)
      return buf.commit(res, arr, traj_id=f"traj-{val}")

    ref_a = asyncio.run(_stage(buf_a, 11))
    ref_b = asyncio.run(_stage(buf_b, 22))
    self.assertEqual(ref_a.worker_id, "rollout-0")
    self.assertEqual(ref_a.instance_id, "inst-a")
    self.assertEqual(ref_b.instance_id, "inst-b")
    # Same sequence #1 on both buffers, but distinct 63-bit ref_ids.
    self.assertEqual(ref_a.ref_id & 0xFFFFFFFF, 1)
    self.assertEqual(ref_b.ref_id & 0xFFFFFFFF, 1)
    self.assertNotEqual(ref_a.ref_id, ref_b.ref_id)

    fetched = buf_a.fetch_staged([ref_a.ref_id])[ref_a.ref_id]
    self.assertFalse(fetched.flags.writeable)
    np.testing.assert_array_equal(fetched, 11)
    self.assertEqual(buf_a.used_bytes, 16)
    self.assertEqual(buf_a.reserved_bytes, 0)

    # Restart ABA check: buf_b rejects ref_a with KeyError on fetch, and ignores
    # ref_a idempotently on release without dropping its own entry #1!
    with self.assertRaisesRegex(KeyError, "Stale worker incarnation"):
      buf_b.fetch_staged([ref_a.ref_id])
    self.assertEqual(buf_b.release_staged([ref_a.ref_id]), 0)
    np.testing.assert_array_equal(
        buf_b.fetch_staged([ref_b.ref_id])[ref_b.ref_id], 22
    )

    self.assertEqual(buf_a.release_staged([ref_a.ref_id, ref_a.ref_id]), 16)
    self.assertEqual(buf_a.used_bytes, 0)
    with self.assertRaisesRegex(KeyError, "Missing or already-released"):
      buf_a.fetch_staged([ref_a.ref_id])

    with self.assertRaises(ValueError):
      sb_lib.RolloutStagingBuffer("rollout-0", "")

  def test_strict_fifo_order_and_cross_thread_release_wakeup(self):
    buf = sb_lib.RolloutStagingBuffer("rollout-0", "inst-0", max_bytes=100)
    granted_order: list[str] = []

    async def _run():
      r0 = await buf.reserve(80)
      ref0 = buf.commit(r0, np.zeros(80, dtype=np.uint8), traj_id="t0")

      async def _waiter(name: str, size: int):
        res = await buf.reserve(size)
        granted_order.append(name)
        res.cancel()

      # w1 needs 60 (blocked: 80 + 60 > 100). w2 needs 10 (would fit in 20 free
      # bytes, MUST NOT jump ahead of w1 in strict FIFO order!).
      t1 = asyncio.create_task(_waiter("w1", 60))
      await asyncio.sleep(0.01)
      t2 = asyncio.create_task(_waiter("w2", 10))
      await asyncio.sleep(0.01)
      self.assertEmpty(granted_order)

      # Release ref0 from a separate OS thread to verify cross-thread wakeup.
      release_thread = threading.Thread(
          target=buf.release_staged, args=([ref0.ref_id],)
      )
      release_thread.start()
      await asyncio.gather(t1, t2)
      release_thread.join()

    asyncio.run(_run())
    self.assertEqual(granted_order, ["w1", "w2"])
    self.assertEqual(buf.used_bytes, 0)
    self.assertEqual(buf.reserved_bytes, 0)

  def test_cancel_after_grant_refunds_and_cascades_to_next_fifo_waiter(self):
    buf = sb_lib.RolloutStagingBuffer("rollout-0", "inst-0", max_bytes=100)

    async def _run():
      r0 = await buf.reserve(80)
      ref0 = buf.commit(r0, np.zeros(80, dtype=np.uint8), traj_id="t0")

      t_cancelled = asyncio.create_task(buf.reserve(80))
      await asyncio.sleep(0.01)
      t_survivor = asyncio.create_task(buf.reserve(80))
      await asyncio.sleep(0.01)

      # Grant `t_cancelled` via release_staged (schedules call_soon_threadsafe),
      # and cancel `t_cancelled` on the event loop in the same turn BEFORE it
      # can consume the reservation.
      buf.release_staged([ref0.ref_id])
      t_cancelled.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await t_cancelled

      r_survivor = await asyncio.wait_for(t_survivor, timeout=1.0)
      self.assertEqual(buf.reserved_bytes, 80)
      r_survivor.cancel()
      self.assertEqual(buf.reserved_bytes, 0)

    asyncio.run(_run())

  def test_oversized_request_and_oversized_commit_fail_fast(self):
    buf = sb_lib.RolloutStagingBuffer("rollout-0", "inst-0", max_bytes=64)

    async def _run():
      with self.assertRaisesRegex(
          ValueError, "exceeds worker staging max_bytes"
      ):
        await buf.reserve(65)
      res = await buf.reserve(64)
      t_queued = asyncio.create_task(buf.reserve(32))
      await asyncio.sleep(0.01)
      with self.assertRaisesRegex(
          ValueError, "exceeds reserved worst_case_bytes"
      ):
        buf.commit(res, np.zeros(80, dtype=np.uint8), traj_id="over")
      r_woken = await asyncio.wait_for(t_queued, timeout=1.0)
      r_woken.cancel()
      self.assertEqual(buf.reserved_bytes, 0)
      self.assertEqual(buf.used_bytes, 0)

    asyncio.run(_run())

  def test_policy_version_type_safety_and_zombie_eviction(self):
    buf = sb_lib.RolloutStagingBuffer("rollout-0", "inst-0", max_bytes=1024)

    with self.assertRaises(TypeError):
      buf.on_post_weight_sync(True)  # pyrefly: ignore[bad-argument-type]

    async def _stage_at(version: int, name: str):
      buf.on_post_weight_sync(version)
      res = await buf.reserve(32)
      return buf.commit(res, np.ones(16, dtype=np.uint8), traj_id=name)

    ref_v0 = asyncio.run(_stage_at(0, "v0"))
    ref_v2 = asyncio.run(_stage_at(2, "v2"))
    ref_v5 = asyncio.run(_stage_at(5, "v5"))
    self.assertEqual(buf.used_bytes, 48)

    # Initial/checkpoint sync passes None -> zero evictions.
    self.assertEqual(
        buf.on_post_weight_sync(6, min_retained_policy_version=None), 0
    )
    self.assertEqual(buf.used_bytes, 48)

    # min_retained_policy_version=2 evicts v0 (1 entry) and retains v2 and v5.
    evicted = buf.on_post_weight_sync(6, min_retained_policy_version=2)
    self.assertEqual(evicted, 1)
    self.assertEqual(buf.zombie_evictions_total, 1)
    self.assertEqual(buf.used_bytes, 32)
    with self.assertRaises(KeyError):
      buf.fetch_staged([ref_v0.ref_id])
    self.assertLen(buf.fetch_staged([ref_v2.ref_id, ref_v5.ref_id]), 2)

  def test_sync_frozen_stall_watchdog_freezes_while_paused_and_fires_when_open(
      self,
  ):
    buf = sb_lib.RolloutStagingBuffer(
        "rollout-0", "inst-0", max_bytes=64, stall_timeout_s=0.08
    )
    paused = True

    async def _run():
      nonlocal paused
      r0 = await buf.reserve(64)
      _ = buf.commit(r0, np.zeros(64, dtype=np.uint8), traj_id="pin")

      waiter_task = asyncio.create_task(
          buf.reserve(16, is_paused_fn=lambda: paused)
      )
      # Stay paused for > 2x stall_timeout_s: timer MUST stay frozen!
      await asyncio.sleep(0.18)
      self.assertFalse(waiter_task.done())

      # Unpause with zero buffer progress: watchdog fires after ~0.08s!
      t_open = time.monotonic()
      paused = False
      with self.assertRaisesRegex(TimeoutError, "stalled"):
        await waiter_task
      self.assertGreaterEqual(time.monotonic() - t_open, 0.06)
      self.assertEqual(buf.reserved_bytes, 0)

    asyncio.run(_run())

  def test_begin_and_finish_clear_revokes_pre_task_grants_and_zeros_bytes(self):
    buf = sb_lib.RolloutStagingBuffer("rollout-0", "inst-0", max_bytes=256)

    async def _run():
      r_committed = await buf.reserve(64)
      _ = buf.commit(r_committed, np.zeros(32, dtype=np.uint8), traj_id="done")
      r_pre_task = await buf.reserve(64)  # Not yet attached to task!
      r_attached = await buf.reserve(64)
      self.assertTrue(r_attached.attach_to_task())
      t_queued = asyncio.create_task(buf.reserve(128))
      await asyncio.sleep(0.01)

      buf.begin_clear()
      self.assertTrue(buf.is_clearing)
      self.assertTrue(r_pre_task.is_revoked)
      with self.assertRaisesRegex(RuntimeError, "clearing"):
        await t_queued
      # Pre-task grant was refunded by begin_clear; attached task refunds on
      # task completion callback.
      self.assertEqual(buf.reserved_bytes, 64)
      r_pre_task.cancel()  # Idempotent no-op.
      r_attached.cancel()
      self.assertEqual(buf.reserved_bytes, 0)

      cleared = buf.finish_clear()
      self.assertEqual(cleared, 1)
      self.assertFalse(buf.is_clearing)
      self.assertEqual(buf.used_bytes, 0)
      self.assertEqual(buf.reserved_bytes, 0)

    asyncio.run(_run())

  def test_grant_revoked_by_begin_clear_before_delivery_wakes_with_error(self):
    buf = sb_lib.RolloutStagingBuffer("rollout-0", "inst-0", max_bytes=100)

    async def _run():
      r0 = await buf.reserve(80)
      t_waiter = asyncio.create_task(buf.reserve(80))
      await asyncio.sleep(0.01)
      # Cancel r0 grants t_waiter and schedules _deliver_grant; begin_clear()
      # revokes that undelivered grant before _deliver_grant runs.
      r0.cancel()
      buf.begin_clear()
      buf.finish_clear()
      with self.assertRaisesRegex(RuntimeError, "clearing"):
        await asyncio.wait_for(t_waiter, timeout=1.0)
      self.assertEqual(buf.reserved_bytes, 0)
      self.assertEqual(buf.used_bytes, 0)

    asyncio.run(_run())

  def test_stale_clear_epoch_rejected_on_commit_and_release_validates_upfront(
      self,
  ):
    buf = sb_lib.RolloutStagingBuffer("rollout-0", "inst-0", max_bytes=256)

    async def _run():
      r_attached = await buf.reserve(64)
      self.assertTrue(r_attached.attach_to_task())
      buf.begin_clear()
      buf.abort_clear()
      with self.assertRaisesRegex(RuntimeError, "Cannot commit reservation"):
        buf.commit(r_attached, np.zeros(16, dtype=np.uint8), traj_id="stale")
      self.assertEqual(buf.reserved_bytes, 0)

      r_valid = await buf.reserve(64)
      ref = buf.commit(r_valid, np.ones(16, dtype=np.uint8), traj_id="valid")
      with self.assertRaises(ValueError):
        buf.release_staged([ref.ref_id, -1])
      # Atomic validation: valid entry remains intact when validation fails.
      self.assertEqual(buf.used_bytes, 16)
      self.assertLen(buf.fetch_staged([ref.ref_id]), 1)
      self.assertEqual(buf.release_staged([ref.ref_id]), 16)

      r_bad_item = await buf.reserve(32)
      with self.assertRaises(TypeError):
        buf.stage_trajectory_item(r_bad_item, "not_item", traj_id="bad")  # pyrefly: ignore[bad-argument-type]
      self.assertEqual(buf.reserved_bytes, 0)

    asyncio.run(_run())

  def test_cgroup_permission_error_falls_back_cleanly(self):
    with absltest.mock.patch.object(
        sb_lib.pathlib.Path, "read_text", side_effect=PermissionError("sandbox")
    ), absltest.mock.patch.object(
        sb_lib.pathlib.Path, "is_file", return_value=True
    ):
      self.assertIsNone(sb_lib._read_cgroup_memory_limit())
      self.assertGreater(sb_lib.default_staging_max_bytes(), 0)

  def test_close_wakes_waiters_and_stage_trajectory_item_cleans_metadata(self):
    buf = sb_lib.RolloutStagingBuffer("rollout-0", "inst-0", max_bytes=128)

    async def _run():
      r0 = await buf.reserve(64)
      item = datatypes.TrajectoryItem(
          prompt_id="p0",
          group_index=0,
          traj={"routed_experts": np.ones((4, 2), dtype=np.uint8)},
          metadata={"routed_experts": np.ones((4, 2), dtype=np.uint8)},
      )
      buf.stage_trajectory_item(r0, item, traj_id="traj-0")
      self.assertIsNotNone(item.routed_experts_ref)
      self.assertNotIn("routed_experts", item.metadata)
      self.assertEqual(buf.used_bytes, 8)

      # If setting `routed_experts_ref` fails after commit, the entry is freed.
      r1 = await buf.reserve(32)
      bad_item = datatypes.TrajectoryItem(
          prompt_id="p1",
          group_index=0,
          traj={"routed_experts": np.ones((2, 2), dtype=np.uint8)},
      )
      with absltest.mock.patch.object(
          datatypes.TrajectoryItem,
          "routed_experts_ref",
          new_callable=absltest.mock.PropertyMock,
          side_effect=ValueError("bad ref"),
      ):
        with self.assertRaisesRegex(ValueError, "bad ref"):
          buf.stage_trajectory_item(r1, bad_item, traj_id="traj-1")
      self.assertEqual(buf.used_bytes, 8)
      self.assertEqual(buf.reserved_bytes, 0)

      # Waiter blocked at capacity is woken immediately on close().
      waiter = asyncio.create_task(buf.reserve(128))
      await asyncio.sleep(0.01)
      buf.close()
      with self.assertRaisesRegex(RuntimeError, "clearing"):
        await asyncio.wait_for(waiter, timeout=1.0)
      buf.abort_clear()
      self.assertTrue(buf.is_clearing)
      with self.assertRaisesRegex(RuntimeError, "closed"):
        await buf.reserve(16)

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
