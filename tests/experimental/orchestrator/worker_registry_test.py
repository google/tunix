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

"""Tests for the WorkerRegistry and WorkerGroup."""

from absl.testing import absltest
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.worker import mock_worker


class WorkerRegistryTest(absltest.TestCase):

  def test_register_and_group_by_role(self):
    registry = worker_registry.WorkerRegistry()
    rollout = mock_worker.MockWorker(worker_id="r0", roles={"rollout"})
    trainer0 = mock_worker.MockWorker(worker_id="t0", roles={"trainer"})
    trainer1 = mock_worker.MockWorker(worker_id="t1", roles={"trainer"})
    registry.register(rollout)
    registry.register(trainer0)
    registry.register(trainer1)

    self.assertEqual(registry.roles(), {"rollout", "trainer"})

    rollout_group = registry.group("rollout")
    self.assertEqual(rollout_group.role, "rollout")
    self.assertLen(rollout_group, 1)
    self.assertEqual(list(rollout_group), [rollout])

    trainer_group = registry.group("trainer")
    self.assertEqual(trainer_group.role, "trainer")
    self.assertLen(trainer_group, 2)
    self.assertEqual(trainer_group.members(), [trainer0, trainer1])

    self.assertIs(registry.get("r0"), rollout)
    self.assertLen(registry, 3)
    self.assertIn("t0", registry)

  def test_worker_group_properties(self):
    registry = worker_registry.WorkerRegistry()
    registry.register(mock_worker.MockWorker("r0", {"rollout"}))
    registry.register(mock_worker.MockWorker("t0", {"trainer"}))
    registry.register(mock_worker.MockWorker("t1", {"trainer"}))

    rollout_group = registry.group("rollout")
    trainer_group = registry.group("trainer")

    self.assertEqual(rollout_group.role, "rollout")
    self.assertFalse(rollout_group.is_empty())
    self.assertLen(rollout_group, 1)
    self.assertLen(list(rollout_group), 1)

    self.assertEqual(trainer_group.role, "trainer")
    self.assertFalse(trainer_group.is_empty())
    self.assertLen(trainer_group, 2)
    self.assertLen(list(trainer_group), 2)

    empty_group = registry.group("inference")
    self.assertEqual(empty_group.role, "inference")
    self.assertTrue(empty_group.is_empty())
    self.assertEmpty(empty_group)
    self.assertEmpty(list(empty_group))

  def test_fused_worker_joins_every_role(self):
    registry = worker_registry.WorkerRegistry()
    fused = mock_worker.MockWorker("f0", {"trainer", "inference"})
    registry.register(fused)

    self.assertEqual(registry.group("trainer").members(), [fused])
    self.assertEqual(registry.group("inference").members(), [fused])

  def test_duplicate_worker_id_raises(self):
    registry = worker_registry.WorkerRegistry()
    registry.register(
        mock_worker.MockWorker(worker_id="dup", roles={"trainer"})
    )
    with self.assertRaises(ValueError):
      registry.register(
          mock_worker.MockWorker(worker_id="dup", roles={"rollout"})
      )

  def test_worker_without_roles_raises(self):
    registry = worker_registry.WorkerRegistry()
    with self.assertRaises(ValueError):
      registry.register(mock_worker.MockWorker("no-roles", set()))

  def test_unknown_role_returns_empty_group(self):
    registry = worker_registry.WorkerRegistry()
    registry.register(mock_worker.MockWorker(worker_id="t0", roles={"trainer"}))
    group = registry.group("inference")
    self.assertTrue(group.is_empty())
    self.assertEmpty(group.members())

  def test_unregister_removes_from_registry_and_groups(self):
    registry = worker_registry.WorkerRegistry()
    registry.register(mock_worker.MockWorker(worker_id="t0", roles={"trainer"}))
    registry.unregister("t0")
    self.assertNotIn("t0", registry)
    self.assertTrue(registry.group("trainer").is_empty())
    self.assertNotIn("trainer", registry.roles())
    with self.assertRaises(KeyError):
      registry.unregister("t0")
    with self.assertRaises(KeyError):
      registry.get("t0")
    with self.assertRaises(KeyError):
      registry.info("t0")

  def test_unregister_retains_role_if_members_remain(self):
    registry = worker_registry.WorkerRegistry()
    registry.register(mock_worker.MockWorker(worker_id="t0", roles={"trainer"}))
    t1 = mock_worker.MockWorker(worker_id="t1", roles={"trainer"})
    registry.register(t1)
    registry.unregister("t0")
    self.assertIn("trainer", registry.roles())
    self.assertEqual(registry.group("trainer").members(), [t1])

  def test_registry_retrieval_methods(self):
    registry = worker_registry.WorkerRegistry()
    r0 = mock_worker.MockWorker(worker_id="r0", roles={"rollout"})
    t0 = mock_worker.MockWorker(worker_id="t0", roles={"trainer"})
    registry.register(t0)
    registry.register(r0)

    self.assertEqual(registry.info("r0").worker_id, "r0")
    self.assertEqual(registry.worker_ids(), ["r0", "t0"])
    self.assertEqual(registry.workers(), [r0, t0])
    self.assertEqual(registry.infos(), [r0.info(), t0.info()])

  def test_register_override_cleans_up_empty_roles(self):
    registry = worker_registry.WorkerRegistry()
    t0 = mock_worker.MockWorker(worker_id="t0", roles={"trainer"})
    t1 = mock_worker.MockWorker(worker_id="t1", roles={"trainer"})
    registry.register(t0)
    registry.register(t1)

    # Override t0 with a new worker that no longer has the "trainer" role
    t0_new = mock_worker.MockWorker(worker_id="t0", roles={"rollout"})
    registry.register(t0_new, override=True)

    # The new t0_new should be removed from the "trainer" group
    self.assertNotIn(t0_new, registry.group("trainer").members())
    self.assertEqual(registry.group("trainer").members(), [t1])
    self.assertIn("trainer", registry.roles())

    # Then override t1 to rollout as well to empty the role
    t1_new = mock_worker.MockWorker(worker_id="t1", roles={"rollout"})
    registry.register(t1_new, override=True)
    self.assertNotIn("trainer", registry.roles())

    # The worker should just be "rollout" now
    self.assertEqual(registry.roles(), {"rollout"})
    self.assertCountEqual(registry.group("rollout").members(), [t0_new, t1_new])

  def test_membership_state_filtering_in_group_and_active_group(self):
    registry = worker_registry.WorkerRegistry()
    state_cls = worker_registry.MembershipState

    w_init = mock_worker.MockWorker(worker_id="w0", roles={"rollout"})
    w_pending = mock_worker.MockWorker(worker_id="w1", roles={"rollout"})
    w_active = mock_worker.MockWorker(worker_id="w2", roles={"rollout"})
    w_evicted = mock_worker.MockWorker(worker_id="w3", roles={"rollout"})

    registry.register(w_init, state=state_cls.INITIALIZING)
    registry.register(w_pending, state=state_cls.PENDING_WEIGHT_SYNC)
    registry.register(w_active, state=state_cls.ACTIVE)
    registry.register(w_evicted, state=state_cls.EVICTED)

    self.assertEqual(registry.state("w0"), state_cls.INITIALIZING)
    self.assertEqual(registry.state("w1"), state_cls.PENDING_WEIGHT_SYNC)
    self.assertEqual(registry.state("w2"), state_cls.ACTIVE)
    self.assertEqual(registry.state("w3"), state_cls.EVICTED)

    # Default group() includes only ACTIVE, excluding INITIALIZING,
    # PENDING_WEIGHT_SYNC, and EVICTED.
    self.assertEqual(registry.group("rollout").members(), [w_active])

    # Explicit states filter (e.g. WeightSyncCoordinator includes
    # PENDING_WEIGHT_SYNC)
    self.assertEqual(
        registry.group(
            "rollout",
            states=(state_cls.ACTIVE, state_cls.PENDING_WEIGHT_SYNC),
        ).members(),
        [w_pending, w_active],
    )
    self.assertEqual(
        registry.group("rollout", states=(state_cls.INITIALIZING,)).members(),
        [w_init],
    )
    self.assertEqual(
        registry.group(
            "rollout", states=(state_cls.EVICTED, state_cls.ACTIVE)
        ).members(),
        [w_active, w_evicted],
    )

  def test_monotonic_incarnation_across_override_and_re_register(self):
    registry = worker_registry.WorkerRegistry()
    w_v1 = mock_worker.MockWorker(worker_id="r0", roles={"rollout"})
    w_v2 = mock_worker.MockWorker(worker_id="r0", roles={"rollout"})
    w_v3 = mock_worker.MockWorker(worker_id="r0", roles={"rollout"})

    registry.register(w_v1)
    self.assertEqual(registry.incarnation("r0"), 1)

    registry.register(w_v2, override=True)
    self.assertEqual(registry.incarnation("r0"), 2)

    registry.unregister("r0")
    with self.assertRaises(KeyError):
      registry.incarnation("r0")
    with self.assertRaises(KeyError):
      registry.state("r0")

    registry.register(w_v3)
    self.assertEqual(registry.incarnation("r0"), 3)

  def test_set_state_and_evict_with_expected_incarnation(self):
    registry = worker_registry.WorkerRegistry()
    state_cls = worker_registry.MembershipState

    w0 = mock_worker.MockWorker(worker_id="r0", roles={"rollout"})
    registry.register(w0, state=state_cls.INITIALIZING)
    inc1 = registry.incarnation("r0")
    self.assertEqual(inc1, 1)

    # Stale/wrong expected_incarnation returns False and does not change state
    self.assertFalse(
        registry.set_state(
            "r0", state_cls.PENDING_WEIGHT_SYNC, expected_incarnation=99
        )
    )
    self.assertEqual(registry.state("r0"), state_cls.INITIALIZING)

    # Matching expected_incarnation succeeds
    self.assertTrue(
        registry.set_state(
            "r0", state_cls.PENDING_WEIGHT_SYNC, expected_incarnation=inc1
        )
    )
    self.assertEqual(registry.state("r0"), state_cls.PENDING_WEIGHT_SYNC)

    # Unconditional set_state succeeds
    self.assertTrue(registry.set_state("r0", state_cls.ACTIVE))
    self.assertEqual(registry.state("r0"), state_cls.ACTIVE)

    # set_state on unknown worker raises KeyError
    with self.assertRaises(KeyError):
      registry.set_state("unknown", state_cls.ACTIVE)

    # Re-register r0 with override=True (incarnation -> 2)
    w0_new = mock_worker.MockWorker(worker_id="r0", roles={"rollout"})
    registry.register(w0_new, override=True, state=state_cls.ACTIVE)
    inc2 = registry.incarnation("r0")
    self.assertEqual(inc2, 2)

    # Evict with stale incarnation (1) fails and leaves new incarnation ACTIVE
    self.assertFalse(registry.evict("r0", expected_incarnation=inc1))
    self.assertEqual(registry.state("r0"), state_cls.ACTIVE)
    self.assertEqual(registry.group("rollout").members(), [w0_new])

    # Evict with matching incarnation (2) succeeds and removes from group
    self.assertTrue(registry.evict("r0", expected_incarnation=inc2))
    self.assertEqual(registry.state("r0"), state_cls.EVICTED)
    self.assertEmpty(registry.group("rollout").members())

    # Evict on unknown worker returns False
    self.assertFalse(registry.evict("unknown"))

  def test_retrieval_methods_exclude_evicted_by_default(self):
    registry = worker_registry.WorkerRegistry()
    r0 = mock_worker.MockWorker(worker_id="r0", roles={"rollout"})
    r1 = mock_worker.MockWorker(worker_id="r1", roles={"rollout"})
    registry.register(r0)
    registry.register(r1)
    self.assertTrue(registry.evict("r0"))

    self.assertEqual(registry.worker_ids(), ["r1"])
    self.assertEqual(registry.workers(), [r1])
    self.assertEqual(registry.infos(), [r1.info()])
    self.assertEqual(registry.worker_ids(include_evicted=True), ["r0", "r1"])
    self.assertEqual(registry.workers(include_evicted=True), [r0, r1])
    self.assertEqual(
        registry.infos(include_evicted=True), [r0.info(), r1.info()]
    )
    # Identity-level membership still tracks evicted workers so incarnation
    # history and re-registration keep working.
    self.assertLen(registry, 2)
    self.assertIn("r0", registry)
    self.assertEqual(
        registry.state("r0"), worker_registry.MembershipState.EVICTED
    )

  def test_state_listeners_notified_only_on_transitions(self):
    registry = worker_registry.WorkerRegistry()
    state_cls = worker_registry.MembershipState
    events: list[tuple[str, worker_registry.MembershipState]] = []
    registry.add_state_listener(lambda wid, state: events.append((wid, state)))

    r0 = mock_worker.MockWorker(worker_id="r0", roles={"rollout"})
    registry.register(r0, state=state_cls.INITIALIZING)
    self.assertEqual(events, [("r0", state_cls.INITIALIZING)])

    # Re-applying the current state is a no-op for listeners.
    self.assertTrue(registry.set_state("r0", state_cls.INITIALIZING))
    self.assertEqual(events, [("r0", state_cls.INITIALIZING)])

    self.assertTrue(registry.set_state("r0", state_cls.ACTIVE))
    self.assertTrue(registry.set_state("r0", state_cls.ACTIVE))
    self.assertEqual(
        events, [("r0", state_cls.INITIALIZING), ("r0", state_cls.ACTIVE)]
    )

    # Repeated evictions (e.g. session callback + coordinator) notify once.
    self.assertTrue(registry.evict("r0"))
    self.assertTrue(registry.evict("r0"))
    self.assertEqual(
        events,
        [
            ("r0", state_cls.INITIALIZING),
            ("r0", state_cls.ACTIVE),
            ("r0", state_cls.EVICTED),
        ],
    )
    self.assertEqual(registry.state("r0"), state_cls.EVICTED)

    # A fresh incarnation re-arms notifications.
    r0_new = mock_worker.MockWorker(worker_id="r0", roles={"rollout"})
    registry.register(r0_new, override=True, state=state_cls.ACTIVE)
    self.assertEqual(events[-1], ("r0", state_cls.ACTIVE))
    self.assertLen(events, 4)


if __name__ == "__main__":
  absltest.main()
