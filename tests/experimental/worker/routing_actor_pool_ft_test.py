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

"""Tests for RoutingActorPool dynamic membership, HRW routing, and sticky placements."""

import asyncio
from absl.testing import absltest
from tunix.experimental.worker import remote_execution


class _DummyActor(remote_execution.ActorHandle):

  def __init__(self, worker_id: str, target_address: str = ""):
    self.worker_id = worker_id
    self.target_address = target_address or f"localhost:{worker_id}"
    self.calls = []

  def submit(self, method_name: str, *args, **kwargs):
    self.calls.append((method_name, args, kwargs))
    return f"{self.worker_id}:{method_name}"

  async def asubmit(self, method_name: str, *args, **kwargs):
    self.calls.append((method_name, args, kwargs))
    return f"{self.worker_id}:{method_name}"

  async def dispatch_task(
      self,
      request_id: str | None = None,
      method_name: str | None = None,
      *args,
      **kwargs,
  ) -> str:
    self.calls.append((method_name, args, kwargs))
    return request_id or "req"

  async def poll_responses(
      self, timeout_s: float = remote_execution.LONG_POLL_TIMEOUT_S
  ):
    del timeout_s
    return None


class RoutingActorPoolFaultToleranceTest(absltest.TestCase):

  def test_hrw_routing_preserves_unaffected_keys_on_worker_removal(self):
    w0 = _DummyActor("rollout_0")
    w1 = _DummyActor("rollout_1")
    w2 = _DummyActor("rollout_2")
    pool = remote_execution.RoutingActorPool([w0, w1, w2])

    keys = [f"traj_{i}" for i in range(300)]
    before = {k: pool.select_actor(route_key=k) for k in keys}
    # Ensure all 3 workers received traffic.
    self.assertEqual({a.worker_id for a in before.values()}, {
        "rollout_0",
        "rollout_1",
        "rollout_2",
    })

    # Remove rollout_1: keys previously routed to rollout_0 or rollout_2 must
    # stay on the exact same worker.
    removed = pool.remove_actor("rollout_1")
    self.assertIs(removed, w1)
    self.assertLen(pool, 2)

    after = {k: pool.select_actor(route_key=k) for k in keys}
    for k in keys:
      if before[k].worker_id != "rollout_1":
        self.assertIs(
            after[k],
            before[k],
            f"Key {k} moved from {before[k].worker_id} to {after[k].worker_id}",
        )
      else:
        self.assertIn(after[k].worker_id, {"rollout_0", "rollout_2"})

  def test_sticky_placements_prevent_reshuffling_when_worker_rejoins(self):
    w0 = _DummyActor("rollout_0")
    w1 = _DummyActor("rollout_1")
    w2 = _DummyActor("rollout_2")
    pool = remote_execution.RoutingActorPool([w0, w1, w2])

    keys = [f"traj_{i}" for i in range(120)]
    initial = {k: pool.select_actor(route_key=k) for k in keys}
    w1_keys = [k for k, a in initial.items() if a.worker_id == "rollout_1"]
    self.assertNotEmpty(w1_keys)

    # Evict rollout_1 and re-route its keys to surviving workers (w0 / w2).
    pool.remove_actor("rollout_1")
    reassigned = {k: pool.select_actor(route_key=k) for k in w1_keys}
    for k, actor in reassigned.items():
      self.assertIn(actor.worker_id, {"rollout_0", "rollout_2"})

    # Now rollout_1 rejoins with a new handle. Previously reassigned keys MUST
    # stay pinned to their surviving workers (w0 / w2), while brand-new keys
    # can route to the rejoined rollout_1.
    w1_new = _DummyActor("rollout_1", target_address="localhost:9999")
    pool.upsert_actor(w1_new)
    for k in w1_keys:
      self.assertIs(
          pool.select_actor(route_key=k),
          reassigned[k],
          f"Reassigned key {k} reshuffled after rollout_1 rejoined!",
      )

    new_keys = [f"fresh_traj_{i}" for i in range(120)]
    new_assignments = {pool.select_actor(route_key=k).worker_id for k in new_keys}
    self.assertEqual(new_assignments, {"rollout_0", "rollout_1", "rollout_2"})

  def test_upsert_replaces_existing_worker_in_place(self):
    w0_old = _DummyActor("rollout_0", target_address="localhost:1000")
    w1 = _DummyActor("rollout_1", target_address="localhost:1001")
    pool = remote_execution.RoutingActorPool([w0_old, w1])

    w0_new = _DummyActor("rollout_0", target_address="localhost:2000")
    pool.upsert_actor(w0_new)

    self.assertLen(pool, 2)
    self.assertIs(pool.get_actor("rollout_0"), w0_new)
    self.assertEqual([a.worker_id for a in pool.actors], ["rollout_0", "rollout_1"])

  def test_select_actor_respects_load_fn_and_max_load(self):
    w0 = _DummyActor("rollout_0")
    w1 = _DummyActor("rollout_1")
    pool = remote_execution.RoutingActorPool([w0, w1])

    loads = {"rollout_0": 4, "rollout_1": 1}
    chosen = pool.select_actor(
        route_key="traj_0",
        load_fn=lambda a: loads[a.worker_id],
        max_load=4,
    )
    self.assertIs(chosen, w1)

    # When all workers are at capacity, select_actor returns None for a new key.
    loads["rollout_1"] = 4
    self.assertIsNone(
        pool.select_actor(
            route_key="traj_new",
            load_fn=lambda a: loads[a.worker_id],
            max_load=4,
        )
    )

  def test_wait_for_available_actor_blocks_until_upsert(self):
    async def _run():
      pool = remote_execution.RoutingActorPool([])
      self.assertFalse(pool)

      async def _add_later():
        await asyncio.sleep(0.02)
        pool.upsert_actor(_DummyActor("rollout_0"))

      task = asyncio.create_task(_add_later())
      ok = await pool.wait_for_available_actor(timeout_s=1.0)
      await task
      self.assertTrue(ok)
      self.assertLen(pool, 1)

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
