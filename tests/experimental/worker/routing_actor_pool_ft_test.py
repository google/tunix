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

"""Tests for RoutingActorPool dynamic membership, HRW/least-loaded routing, and HealthMonitor isolation."""

import asyncio
from unittest import mock
from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import health_monitor
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.worker import mock_worker
from tunix.experimental.worker import remote_execution as remote_lib

WorkerState = datatypes.WorkerState


class _StubActor:
  """Simple stub bound to InProcessRemoteExecutionServer for pool routing tests."""

  def __init__(self, worker_id: str, tag: str = "v1"):
    self.worker_id = worker_id
    self.tag = tag

  def echo(self, payload: str = "") -> str:
    return f"{self.worker_id}:{self.tag}:{payload}"


def _make_handle(worker_id: str, tag: str = "v1") -> remote_lib.ActorHandle:
  server = remote_lib.InProcessRemoteExecutionServer(
      _StubActor(worker_id=worker_id, tag=tag)
  )
  handle = remote_lib.InProcessActorHandle(server)
  handle.worker_id = worker_id
  return handle


class RoutingActorPoolFaultToleranceTest(absltest.TestCase):

  def test_hrw_routing_pins_unaffected_keys_on_remove_and_rejoin(self):
    w0 = _make_handle("rollout_0")
    w1 = _make_handle("rollout_1")
    w2 = _make_handle("rollout_2")
    pool = remote_lib.RoutingActorPool([w0, w1, w2])

    keys = [f"traj_prompt_{i}" for i in range(100)]
    initial_mapping = {
        k: pool.select_actor(route_key=k).worker_id for k in keys
    }
    # Ensure all 3 workers received traffic initially.
    self.assertEqual(
        set(initial_mapping.values()), {"rollout_0", "rollout_1", "rollout_2"}
    )

    # Evict rollout_1 and verify keys on rollout_0 and rollout_2 never move.
    removed = pool.remove_actor("rollout_1")
    self.assertIs(removed, w1)
    self.assertLen(pool, 2)

    after_evict_mapping = {
        k: pool.select_actor(route_key=k).worker_id for k in keys
    }
    for k in keys:
      if initial_mapping[k] in ("rollout_0", "rollout_2"):
        self.assertEqual(
            after_evict_mapping[k],
            initial_mapping[k],
            f"Key {k} moved from {initial_mapping[k]} to"
            f" {after_evict_mapping[k]} when rollout_1 was evicted!",
        )
      else:
        self.assertIn(after_evict_mapping[k], ("rollout_0", "rollout_2"))

    # Rejoin rollout_1 with a new handle and verify deterministic restoration.
    w1_rejoined = _make_handle("rollout_1", tag="v2")
    pool.upsert_actor(w1_rejoined, worker_id="rollout_1")
    self.assertLen(pool, 3)

    restored_mapping = {
        k: pool.select_actor(route_key=k).worker_id for k in keys
    }
    self.assertEqual(restored_mapping, initial_mapping)

  def test_upsert_actor_replaces_existing_handle_in_place(self):
    w0_v1 = _make_handle("rollout_0", tag="v1")
    w1_v1 = _make_handle("rollout_1", tag="v1")
    pool = remote_lib.RoutingActorPool([w0_v1, w1_v1])
    self.assertLen(pool, 2)
    self.assertIn(w0_v1, pool)
    self.assertIn("rollout_0", pool)
    self.assertNotIn("rollout_99", pool)

    w0_v2 = _make_handle("rollout_0", tag="v2")
    pool.upsert_actor(w0_v2, worker_id="rollout_0")
    self.assertLen(pool, 2)
    self.assertNotIn(w0_v1, pool)
    self.assertIn(w0_v2, pool)
    self.assertIn("rollout_0", pool)
    self.assertIs(pool.get_actor("rollout_0"), w0_v2)
    self.assertEqual(
        pool.get_actor("rollout_0").submit("echo", "ping"),
        "rollout_0:v2:ping",
    )

  def test_wait_for_available_actor_blocks_when_empty_and_wakes_on_upsert(self):
    async def _run():
      w0 = _make_handle("rollout_0")
      pool = remote_lib.RoutingActorPool([w0])
      self.assertTrue(await pool.wait_for_available_actor(timeout_s=0.05))

      pool.remove_actor("rollout_0")
      self.assertFalse(bool(pool))
      self.assertFalse(await pool.wait_for_available_actor(timeout_s=0.03))

      async def _delayed_rejoin():
        await asyncio.sleep(0.04)
        pool.upsert_actor(
            _make_handle("rollout_0", tag="v2"), worker_id="rollout_0"
        )

      rejoin_task = asyncio.create_task(_delayed_rejoin())
      available = await pool.wait_for_available_actor(timeout_s=1.0)
      await rejoin_task
      self.assertTrue(available)
      self.assertLen(pool, 1)

    asyncio.run(_run())

  def test_select_actor_least_loaded_and_per_worker_capacity_gate(self):
    w0 = _make_handle("rollout_0")
    w1 = _make_handle("rollout_1")
    w2 = _make_handle("rollout_2")
    pool = remote_lib.RoutingActorPool([w0, w1, w2])

    loads = {"rollout_0": 4, "rollout_1": 1, "rollout_2": 3}
    load_fn = lambda a: loads[a.worker_id]

    # 1. Picks rollout_1 because it has the lowest in-flight count (1).
    chosen = pool.select_actor(
        route_key="traj_any", load_fn=load_fn, max_load=4
    )
    self.assertIs(chosen, w1)

    # 2. Drain 4 retries (filling 3 free slots on rollout_1, 1 on rollout_2):
    dispatched_to = []
    for i in range(4):
      target = pool.select_actor(
          route_key=f"retry_{i}", load_fn=load_fn, max_load=4
      )
      self.assertIsNotNone(target)
      loads[target.worker_id] += 1
      dispatched_to.append(target.worker_id)

    # All 3 workers should now be at max_load=4 (total capacity 12).
    self.assertEqual(loads, {"rollout_0": 4, "rollout_1": 4, "rollout_2": 4})

    # 3. With all workers at max_load=4, select_actor returns None.
    self.assertIsNone(
        pool.select_actor(route_key="overflow_req", load_fn=load_fn, max_load=4)
    )

    # 4. Once rollout_2 finishes 1 trajectory (4 -> 3), 1 slot opens on w2.
    loads["rollout_2"] = 3
    self.assertIs(
        pool.select_actor(
            route_key="overflow_req", load_fn=load_fn, max_load=4
        ),
        w2,
    )

  def test_health_monitor_isolate_errors_marks_dead_worker_error(self):
    w0 = mock_worker.MockWorker("rollout_0", roles={"rollout"})
    w1 = mock_worker.MockWorker("rollout_1", roles={"rollout"})
    w2 = mock_worker.MockWorker("rollout_2", roles={"rollout"})
    for w in (w0, w1, w2):
      w._state = WorkerState.READY

    registry = worker_registry.WorkerRegistry()
    for w in (w0, w1, w2):
      registry.register(w)

    clock_val = [10.0]
    monitor = health_monitor.HealthMonitor(
        registry,
        state_deadlines_s={WorkerState.ERROR: 30.0},
        clock=lambda: clock_val[0],
        isolate_errors=True,
    )
    with mock.patch.object(
        w1, "heartbeat", side_effect=ConnectionError("gRPC socket reset")
    ):
      reports = monitor.poll()

    self.assertEqual(reports["rollout_0"].state, WorkerState.READY)
    self.assertEqual(reports["rollout_2"].state, WorkerState.READY)
    self.assertEqual(reports["rollout_1"].state, WorkerState.ERROR)
    self.assertIn(
        "ConnectionError: gRPC socket reset", reports["rollout_1"].last_error
    )

    # Advance clock past ERROR deadline and verify overdue() flags rollout_1.
    clock_val[0] = 45.0
    overdue = monitor.overdue()
    self.assertLen(overdue, 1)
    self.assertEqual(overdue[0].worker_id, "rollout_1")
    self.assertEqual(overdue[0].state, WorkerState.ERROR)


if __name__ == "__main__":
  absltest.main()
