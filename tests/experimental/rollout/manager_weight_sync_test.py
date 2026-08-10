# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for RolloutManager's weight-sync admission gate.

Silent failures this file locks out:
  - a generate() slipping past the admission gate during a sync drain, so an
    episode samples over a KV cache that is being freed;
  - pre_weight_sync's drain timeout cancelling live collector tasks (a
    cancelled collector never resolves its caller's future) instead of
    failing closed and leaving stragglers untouched;
  - a caller hanging forever when its episode task is cancelled before the
    task body ever runs (no CancelledError handler executes in that window);
  - a raising on_complete callback swallowing the caller's result or the
    completed-queue entry;
  - the gate reopening after a FAILED post_weight_sync, after weight_sync
    (too early), inside post/abort themselves (before the caller confirmed
    the round is serving), or after cancel_all() (stop must be permanent);
  - a duplicate traj_id silently shadowing a running collector, or leaking
    the env acquired for the rejected duplicate request;
  - a re-used traj_id being deleted from the active maps by the episode that
    just finished under that id, hiding a live collector from the drain;
  - a sampler that predates the weight-sync capability failing the Sampler
    Protocol check and no longer constructing a RolloutManager at all.
"""

import asyncio
from unittest import mock

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.rollout import manager as manager_lib
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.trajectory import trajectory as trajectory_lib


class FakeSampler:
  def initialize(self) -> None: return None
  """Implements all 16 Sampler protocol methods; logs weight-sync calls."""

  def __init__(self, log=None):
    # Shared ordered log: tests append episode events to the same list to
    # assert cross-component ordering (drain barrier vs sampler quiesce).
    self.log = log if log is not None else []
    self.fail_post = False

  async def start(self, **kwargs):
    return None

  async def stop(self, **kwargs):
    return None

  async def pause(self, **kwargs):
    return None

  async def resume(self, **kwargs):
    return None

  async def get_mesh(self, **kwargs):
    return None

  async def sample(self, sampling_requests, **kwargs):
    return []

  async def get_weight_sync_metadata(self, **kwargs):
    return [{"host": 0}]

  async def bind_weight_sync(self, **kwargs):
    return None

  async def pre_weight_sync(self, sync_request=None, **kwargs):
    self.log.append("sampler_pre")

  async def weight_sync(self, sync_request=None, **kwargs):
    self.log.append("sampler_weight")

  async def post_weight_sync(self, sync_request=None, **kwargs):
    if self.fail_post:
      raise RuntimeError("post_weight_sync failed (injected)")
    self.log.append("sampler_post")

  async def abort_weight_sync(self, sync_request=None, **kwargs):
    self.log.append("sampler_abort")

  async def get_weight_sync_status(self, **kwargs):
    return {}

  async def get_transfer_status(self, req_id, **kwargs):
    return "DONE"

  async def migrate_kv_cache(
      self, source_server_id, target_server_id, token_ids, **kwargs
  ):
    return True

  async def get_load_info(self, **kwargs):
    return None


class LegacySampler:
  def initialize(self) -> None: return None
  """A pre-weight-sync sampler: the Sampler members and nothing else.

  Stands in for the upstream adapters (VanillaSamplerAdapter,
  LegacyVllmSamplerAdapter), which conform to Sampler structurally and know
  nothing about coordinated weight sync.
  """

  async def start(self, **kwargs):
    return None

  async def stop(self, **kwargs):
    return None

  async def pause(self, **kwargs):
    return None

  async def resume(self, **kwargs):
    return None

  async def get_mesh(self, **kwargs):
    return None

  async def sample(self, sampling_requests, **kwargs):
    return []

  async def get_weight_sync_metadata(self, **kwargs):
    return [{"host": 0}]

  async def pre_weight_sync(self, sync_request=None, **kwargs):
    return None

  async def weight_sync(self, sync_request=None, **kwargs):
    return None

  async def post_weight_sync(self, sync_request=None, **kwargs):
    return None

  async def get_transfer_status(self, req_id, **kwargs):
    return "DONE"

  async def migrate_kv_cache(
      self, source_server_id, target_server_id, token_ids, **kwargs
  ):
    return True

  async def get_load_info(self, **kwargs):
    return None


class FakeCollector:
  """Episode fake driven by asyncio.Events carried in request.metadata.

  metadata["started"]: set once the episode body is running.
  metadata["gate"]: episode blocks until the test sets it.
  metadata["log"]: shared ordered log; "episode_done:<id>" appended at end.
  """

  def __init__(
      self,
      *,
      traj_id,
      request,
      sampler,
      env_client,
      agent,
      tokenizer,
      chat_parser,
  ):
    self.traj_id = traj_id
    self.request = request
    self.env = env_client
    self.is_done = False
    self.is_paused = False

  async def run_episode(self):
    started = self.request.metadata.get("started")
    if started is not None:
      started.set()
    gate = self.request.metadata.get("gate")
    if gate is not None:
      await gate.wait()
    log = self.request.metadata.get("log")
    if log is not None:
      log.append("episode_done:" + self.traj_id)
    self.is_done = True
    # Trajectory requires a nested `agent`, which the admission gate never
    # reads. model_construct skips that validation so these tests do not
    # depend on a schema they are not exercising.
    return trajectory_lib.Trajectory.model_construct(
        trajectory_id=self.traj_id
    )

  def pause(self):
    self.is_paused = True

  def resume(self):
    self.is_paused = False

  def cancel(self):
    pass


class FakeEnvPool:
  """Counts acquire/release so an env leak shows up as a counter mismatch."""

  def __init__(self):
    self.acquired = 0
    self.released = 0

  def acquire_env(self, env_config=None):
    self.acquired += 1
    return {"env_id": self.acquired}

  def release_env(self, env):
    self.released += 1


def _ws_req(uuid):
  return datatypes.WeightSyncRequest(
      policy_version=uuid, extra_config={"req_id": "r1", "uuid": uuid}
  )


def _rollout_req(prompt_id, **metadata):
  return datatypes.RolloutRequest(prompt_id=prompt_id, metadata=metadata)


def _traj(prompt_id):
  """The traj_id the request derives from a prompt_id.

  `RolloutRequest.traj_id` is a read-only property, so a test cannot pick
  one: it asserts against what the request will actually key itself by.
  """
  return datatypes.RolloutRequest(prompt_id=prompt_id).traj_id


def _make_manager(sampler, env_pool=None):
  return manager_lib.RolloutManager(
      sampler=sampler,
      env_pool=env_pool,
      tokenizer=object(),
      chat_parser=object(),
  )


def _patch_collector():
  return mock.patch.object(
      manager_lib.collector_lib, "TrajectoryCollectorEngine", FakeCollector
  )


class ManagerWeightSyncTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # A task created inside a `with` block can reach its collector
    # construction after the block exits, so the patch has to outlive any
    # single block. The real engine rejects the None sampler/env/agent these
    # tests run without.
    patcher = mock.patch.object(
        manager_lib.collector_lib, "TrajectoryCollectorEngine", FakeCollector
    )
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_weight_sync_request_alias_is_canonical_wire_type(self):
    # Coordinator, trainer and rollout RPCs must exchange one concrete class;
    # two same-shaped dataclasses are not interchangeable to pytype or every
    # serializer used by the remote actor path.
    self.assertIs(
        sampler_lib.WeightSyncRequest, datatypes.WeightSyncRequest
    )

  def test_generate_rejected_while_sync_in_progress(self):
    asyncio.run(
        asyncio.wait_for(
            self._generate_rejected_while_sync_in_progress(), timeout=30
        )
    )

  async def _generate_rejected_while_sync_in_progress(self):
    log = []
    sampler = FakeSampler(log=log)
    with _patch_collector():
      manager = _make_manager(sampler)
      gate = asyncio.Event()
      started = asyncio.Event()
      gen_task = asyncio.ensure_future(
          manager.generate(
              _rollout_req("t1", gate=gate, started=started, log=log)
          )
      )
      await asyncio.wait_for(started.wait(), 5)

      pre_task = asyncio.ensure_future(manager.pre_weight_sync(_ws_req(1)))
      # Let pre_task run up to its drain barrier (asyncio.wait on t1).
      await asyncio.sleep(0.01)
      self.assertFalse(pre_task.done())
      self.assertNotIn("sampler_pre", log)  # draining, sampler untouched

      with self.assertRaisesRegex(RuntimeError, "admission is closed"):
        await manager.generate(_rollout_req("t2"))

      gate.set()
      await asyncio.wait_for(pre_task, 5)
      result = await asyncio.wait_for(gen_task, 5)
      self.assertIsInstance(result, trajectory_lib.Trajectory)
      self.assertEqual(_traj("t1"), result.trajectory_id)
      # The drain barrier must hold: the in-flight episode finished strictly
      # before the sampler was quiesced.
      self.assertEqual([f"episode_done:{_traj('t1')}", "sampler_pre"], log)

      # pre completing does NOT reopen admission; only post does.
      with self.assertRaisesRegex(RuntimeError, "admission is closed"):
        await manager.generate(_rollout_req("t3"))

      await manager.post_weight_sync(_ws_req(1))
      # The publish alone does not admit traffic either: the caller reopens
      # once it has confirmed the round is serving.
      with self.assertRaisesRegex(RuntimeError, "admission is closed"):
        await manager.generate(_rollout_req("t4"))

      manager.reopen_admission()
      result = await asyncio.wait_for(manager.generate(_rollout_req("t5")), 5)
      self.assertIsInstance(result, trajectory_lib.Trajectory)

  def test_pre_drain_timeout_fails_closed_without_cancel(self):
    asyncio.run(
        asyncio.wait_for(
            self._pre_drain_timeout_fails_closed_without_cancel(), timeout=30
        )
    )

  async def _pre_drain_timeout_fails_closed_without_cancel(self):
    log = []
    sampler = FakeSampler(log=log)
    with _patch_collector():
      manager = _make_manager(sampler)
      gate = asyncio.Event()
      started = asyncio.Event()
      gen_task = asyncio.ensure_future(
          manager.generate(_rollout_req("t1", gate=gate, started=started))
      )
      await asyncio.wait_for(started.wait(), 5)

      with self.assertRaisesRegex(RuntimeError, "drain timeout"):
        await manager.pre_weight_sync(_ws_req(1), drain_timeout_s=0.2)

      # Fails closed WITHOUT cancelling: the straggler episode is untouched
      # and its caller's future stays unresolved.
      self.assertFalse(gen_task.done())
      episode_task = manager._active_tasks.get(_traj("t1"))
      self.assertIsNotNone(episode_task)
      self.assertFalse(episode_task.done())
      self.assertNotIn("sampler_pre", log)
      with self.assertRaisesRegex(RuntimeError, "admission is closed"):
        await manager.generate(_rollout_req("t2"))

      await manager.abort_weight_sync(_ws_req(1))
      manager.reopen_admission()
      result = await asyncio.wait_for(manager.generate(_rollout_req("t3")), 5)
      self.assertIsInstance(result, trajectory_lib.Trajectory)

      gate.set()
      result = await asyncio.wait_for(gen_task, 5)
      self.assertIsInstance(result, trajectory_lib.Trajectory)
      self.assertEqual(_traj("t1"), result.trajectory_id)

  def test_cancel_before_body_starts_still_settles_caller(self):
    asyncio.run(
        asyncio.wait_for(
            self._cancel_before_body_starts_still_settles_caller(), timeout=30
        )
    )

  async def _cancel_before_body_starts_still_settles_caller(self):
    sampler = FakeSampler()
    with _patch_collector():
      manager = _make_manager(sampler)
      gen = asyncio.ensure_future(manager.generate(_rollout_req("t1")))
      # Exactly one tick: _generate_one registers the episode task, but the
      # task body has not run when the synchronous cancel_all() lands, so
      # _run_and_enqueue's own CancelledError handler can never fire.
      await asyncio.sleep(0)
      manager.cancel_all()
      result = await asyncio.wait_for(gen, 5)
      self.assertIsInstance(result, trajectory_lib.TrajectoryError)
      self.assertEqual("CancelledError", result.error_type)
      self.assertIn(
          result.error_message,
          (
              "cancelled before the episode started",
              "cancelled (manager stopping)",
          ),
      )
      self.assertEmpty(manager._active_tasks)
      self.assertEmpty(manager._active_collectors)
      # The completion stream is a SECOND consumer, not a mirror of the
      # future: a result that never reaches the queue leaves
      # as_completed_stream() waiting forever for an episode already over.
      streamed = await asyncio.wait_for(manager.pop_next_completed(), 5)
      self.assertIsInstance(streamed, trajectory_lib.TrajectoryError)
      self.assertEqual("CancelledError", streamed.error_type)

  def test_on_complete_raising_does_not_strand_caller(self):
    asyncio.run(
        asyncio.wait_for(
            self._on_complete_raising_does_not_strand_caller(), timeout=30
        )
    )

  async def _on_complete_raising_does_not_strand_caller(self):
    sampler = FakeSampler()
    with _patch_collector():
      manager = _make_manager(sampler)

      def bad_on_complete(result):
        raise ValueError("injected on_complete failure")

      with self.assertLogs(level="ERROR"):
        result = await asyncio.wait_for(
            manager.generate(_rollout_req("t1"), on_complete=bad_on_complete),
            5,
        )
      self.assertIsInstance(result, trajectory_lib.Trajectory)
      self.assertEqual(_traj("t1"), result.trajectory_id)
      streamed = await asyncio.wait_for(manager.pop_next_completed(), 5)
      self.assertEqual(result, streamed)

  def test_post_failure_leaves_gate_closed(self):
    asyncio.run(
        asyncio.wait_for(self._post_failure_leaves_gate_closed(), timeout=30)
    )

  async def _post_failure_leaves_gate_closed(self):
    sampler = FakeSampler()
    sampler.fail_post = True
    with _patch_collector():
      manager = _make_manager(sampler)
      await manager.pre_weight_sync(_ws_req(1))
      with self.assertRaisesRegex(RuntimeError, "injected"):
        await manager.post_weight_sync(_ws_req(1))
      # A failed publish must NOT reopen admission over unknown sampler state.
      with self.assertRaisesRegex(RuntimeError, "admission is closed"):
        await manager.generate(_rollout_req("t1"))
      await manager.abort_weight_sync(_ws_req(2))
      manager.reopen_admission()
      result = await asyncio.wait_for(manager.generate(_rollout_req("t2")), 5)
      self.assertIsInstance(result, trajectory_lib.Trajectory)

  def test_duplicate_traj_id_rejected_and_env_released(self):
    asyncio.run(
        asyncio.wait_for(
            self._duplicate_traj_id_rejected_and_env_released(), timeout=30
        )
    )

  async def _duplicate_traj_id_rejected_and_env_released(self):
    sampler = FakeSampler()
    pool = FakeEnvPool()
    with _patch_collector():
      manager = _make_manager(sampler, env_pool=pool)
      gate = asyncio.Event()
      started = asyncio.Event()
      gen1 = asyncio.ensure_future(
          manager.generate(_rollout_req("t1", gate=gate, started=started))
      )
      await asyncio.wait_for(started.wait(), 5)
      self.assertEqual(1, pool.acquired)

      with self.assertRaisesRegex(ValueError, "already has an active task"):
        await manager.generate(_rollout_req("t1"))
      # The duplicate acquired its env before the atomic registration check;
      # it must give the env back, or every duplicate leaks one.
      self.assertEqual(2, pool.acquired)
      self.assertEqual(1, pool.released)

      gate.set()
      result = await asyncio.wait_for(gen1, 5)
      self.assertIsInstance(result, trajectory_lib.Trajectory)
      self.assertEqual(_traj("t1"), result.trajectory_id)
      self.assertEqual(pool.acquired, pool.released)  # net leak is zero

  def test_closed_manager_rejects_before_env_acquisition(self):
    asyncio.run(
        asyncio.wait_for(
            self._closed_manager_rejects_before_env_acquisition(), timeout=30
        )
    )

  async def _closed_manager_rejects_before_env_acquisition(self):
    sampler = FakeSampler()
    pool = FakeEnvPool()
    with _patch_collector():
      manager = _make_manager(sampler, env_pool=pool)
      manager.cancel_all()
      with self.assertRaisesRegex(RuntimeError, "admission is closed"):
        await manager.generate(_rollout_req("t1"))
      # The cheap early check fires before any env is acquired.
      self.assertEqual(0, pool.acquired)

  def test_gate_reopens_only_after_post_not_weight(self):
    asyncio.run(
        asyncio.wait_for(
            self._gate_reopens_only_after_post_not_weight(), timeout=30
        )
    )

  async def _gate_reopens_only_after_post_not_weight(self):
    sampler = FakeSampler()
    with _patch_collector():
      manager = _make_manager(sampler)
      await manager.pre_weight_sync(_ws_req(1))
      with self.assertRaisesRegex(RuntimeError, "admission is closed"):
        await manager.generate(_rollout_req("t1"))
      await manager.weight_sync(_ws_req(1))
      # weight_sync only moves bytes; it must NOT admit traffic before the
      # publish in post_weight_sync.
      with self.assertRaisesRegex(RuntimeError, "admission is closed"):
        await manager.generate(_rollout_req("t2"))
      await manager.post_weight_sync(_ws_req(1))
      manager.reopen_admission()
      result = await asyncio.wait_for(manager.generate(_rollout_req("t3")), 5)
      self.assertIsInstance(result, trajectory_lib.Trajectory)

  def test_under_lock_recheck_rejects_and_releases_env(self):
    asyncio.run(
        asyncio.wait_for(
            self._under_lock_recheck_rejects_and_releases_env(), timeout=30
        )
    )

  async def _under_lock_recheck_rejects_and_releases_env(self):
    # The cheap early check and the under-lock re-check guard the SAME
    # invariant at different points; this test pins the under-lock one,
    # which nothing else can reach (today there is no await between them,
    # but the lock exists precisely so a future await in that window --
    # e.g. async env acquisition -- cannot reintroduce the race). Recipe:
    # hold the admission lock exactly as pre_weight_sync's close-and-
    # snapshot does, park a generate() that already passed the early check
    # (and acquired its env) on the lock, close the gate, release the lock.
    sampler = FakeSampler()
    pool = FakeEnvPool()
    with _patch_collector():
      manager = _make_manager(sampler, env_pool=pool)
      await manager._admission_lock.acquire()
      gen = asyncio.ensure_future(manager.generate(_rollout_req("t1")))
      await asyncio.sleep(0)  # generate is now parked on the lock
      self.assertEqual(1, pool.acquired)
      manager._admission_open.clear()
      manager._admission_lock.release()
      with self.assertRaisesRegex(RuntimeError, "admission is closed"):
        await asyncio.wait_for(gen, 5)
      # The re-check must give back the env acquired before the lock.
      self.assertEqual(1, pool.released)
      self.assertEmpty(manager._active_tasks)
      self.assertEmpty(manager._active_collectors)

  def test_cancel_all_is_permanent(self):
    asyncio.run(
        asyncio.wait_for(self._cancel_all_is_permanent(), timeout=30)
    )

  async def _cancel_all_is_permanent(self):
    sampler = FakeSampler()
    manager = _make_manager(sampler)
    manager.cancel_all()
    await manager.post_weight_sync(_ws_req(1))
    manager.reopen_admission()
    with self.assertRaisesRegex(RuntimeError, "admission is closed"):
      await manager.generate(_rollout_req("t1"))
    await manager.abort_weight_sync(_ws_req(2))
    manager.reopen_admission()
    with self.assertRaisesRegex(RuntimeError, "admission is closed"):
      await manager.generate(_rollout_req("t2"))

  def test_legacy_sampler_without_weight_sync_capability_constructs(self):
    asyncio.run(
        asyncio.wait_for(
            self._legacy_sampler_without_weight_sync_capability_constructs(),
            timeout=30,
        )
    )

  async def _legacy_sampler_without_weight_sync_capability_constructs(self):
    # The weight-sync members live in a SEPARATE optional Protocol: a
    # runtime_checkable isinstance test only checks member presence, so
    # carrying them in Sampler would stop every sampler that predates
    # coordinated weight sync from constructing a manager at all.
    sampler = LegacySampler()
    with _patch_collector():
      manager = _make_manager(sampler)
      self.assertIs(sampler, manager.sampler)
      result = await asyncio.wait_for(manager.generate(_rollout_req("t1")), 5)
      self.assertIsInstance(result, trajectory_lib.Trajectory)
      self.assertEqual(_traj("t1"), result.trajectory_id)

  def test_weight_sync_on_incapable_sampler_raises_typeerror(self):
    asyncio.run(
        asyncio.wait_for(
            self._weight_sync_on_incapable_sampler_raises_typeerror(),
            timeout=30,
        )
    )

  async def _weight_sync_on_incapable_sampler_raises_typeerror(self):
    manager = _make_manager(LegacySampler())
    # Named as a whole capability, not as one missing attribute: a sampler
    # with bind but no abort/status cannot be driven through a round.
    with self.assertRaisesRegex(TypeError, "get_weight_sync_status"):
      await manager.bind_weight_sync()
    # Every capability entry point, not just bind. Guarding bind alone let
    # pre/weight_sync succeed (quiescing the worker and freeing its cache)
    # and only then died with AttributeError inside abort, stranding the
    # worker SYNCING with admission shut and no way to roll back.
    with self.assertRaisesRegex(TypeError, "get_weight_sync_status"):
      await manager.abort_weight_sync(_ws_req(1))
    with self.assertRaisesRegex(TypeError, "get_weight_sync_status"):
      await manager.get_weight_sync_status()

  def test_traj_id_reuse_after_completion_stays_visible_to_drain(self):
    asyncio.run(
        asyncio.wait_for(
            self._traj_id_reuse_after_completion_stays_visible_to_drain(),
            timeout=30,
        )
    )

  async def _traj_id_reuse_after_completion_stays_visible_to_drain(self):
    # ABA on the active maps: the caller is woken by future.set_result BEFORE
    # the finished task's done-callback runs, so it can legally start a new
    # episode under the same traj_id in that window. An unconditional pop in
    # the done-callback then deletes the NEW episode, and pre_weight_sync
    # quiesces (freeing the KV cache) over a collector that is still
    # sampling.
    sampler = FakeSampler()
    gate = asyncio.Event()
    started = asyncio.Event()
    reused = []
    with _patch_collector():
      manager = _make_manager(sampler)

      def start_reused(result):
        del result
        reused.append(
            asyncio.ensure_future(
                manager.generate(
                    _rollout_req("t1", gate=gate, started=started)
                )
            )
        )

      first = await asyncio.wait_for(
          manager.generate(_rollout_req("t1"), on_complete=start_reused), 5
      )
      self.assertIsInstance(first, trajectory_lib.Trajectory)
      await asyncio.wait_for(started.wait(), 5)

      with self.assertRaisesRegex(RuntimeError, "drain timeout"):
        await manager.pre_weight_sync(_ws_req(1), drain_timeout_s=0.2)
      episode_task = manager._active_tasks.get(_traj("t1"))
      self.assertIsNotNone(episode_task)
      self.assertFalse(episode_task.done())
      self.assertIn(_traj("t1"), manager._active_collectors)

      gate.set()
      result = await asyncio.wait_for(reused[0], 5)
      self.assertIsInstance(result, trajectory_lib.Trajectory)
      self.assertEqual(_traj("t1"), result.trajectory_id)
      self.assertEmpty(manager._active_tasks)
      self.assertEmpty(manager._active_collectors)


if __name__ == "__main__":
  absltest.main()
