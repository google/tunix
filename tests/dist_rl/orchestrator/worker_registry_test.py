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

"""Tests for the WorkerRegistry."""

import pickle
import threading
import time
from unittest import mock

from absl.testing import absltest
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.worker import mock_worker
from tunix.dist_rl.worker import remote_execution


class WorkerRegistryTest(absltest.TestCase):

  def _activate_all(self, registry: worker_registry.WorkerRegistry) -> None:
    for wid in registry.worker_ids():
      registry.set_status(wid, worker_registry.AdmissionStatus.ACTIVE)

  def test_register_and_handles_by_role(self):
    registry = worker_registry.WorkerRegistry()
    rollout = mock_worker.MockWorker(worker_id="r0", roles={"rollout"})
    trainer0 = mock_worker.MockWorker(worker_id="t0", roles={"trainer"})
    trainer1 = mock_worker.MockWorker(worker_id="t1", roles={"trainer"})
    registry.register(rollout)
    registry.register(trainer0)
    registry.register(trainer1)

    self.assertEqual(
        registry.status("r0"), worker_registry.AdmissionStatus.REGISTERED
    )
    self.assertEmpty(registry.handles("rollout"))
    self._activate_all(registry)

    self.assertEqual(registry.roles(), {"rollout", "trainer"})
    self.assertEqual(registry.handles("rollout"), [registry.handle("r0")])
    self.assertEqual(
        registry.handles("trainer"),
        [registry.handle("t0"), registry.handle("t1")],
    )

    self.assertIsInstance(
        registry.handle("r0"), remote_execution.InProcessActorHandle
    )
    self.assertIs(registry.handle("r0").server.bound_instance, rollout)
    self.assertLen(registry, 3)
    self.assertIn("t0", registry)

  def test_fused_worker_joins_every_role(self):
    registry = worker_registry.WorkerRegistry()
    fused = mock_worker.MockWorker("f0", {"trainer", "inference"})
    registry.register(fused)
    self._activate_all(registry)

    self.assertEqual(registry.handles("trainer"), [registry.handle("f0")])
    self.assertEqual(registry.handles("inference"), [registry.handle("f0")])

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

  def test_unknown_role_returns_empty_handles(self):
    registry = worker_registry.WorkerRegistry()
    registry.register(mock_worker.MockWorker(worker_id="t0", roles={"trainer"}))
    self._activate_all(registry)
    self.assertEmpty(registry.handles("inference"))

  def test_unregister_removes_from_registry_and_closes_handle(self):
    registry = worker_registry.WorkerRegistry()
    remote_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    registry.register_handle("t0", ["trainer"], remote_handle)
    registry.unregister("t0")
    remote_handle.sync_close.assert_called_once()
    self.assertNotIn("t0", registry)
    self.assertEmpty(registry.handles("trainer"))
    self.assertNotIn("trainer", registry.roles())
    with self.assertRaises(KeyError):
      registry.unregister("t0")
    with self.assertRaises(KeyError):
      registry.handle("t0")
    with self.assertRaises(KeyError):
      registry.info("t0")

  def test_unregister_retains_role_if_members_remain(self):
    registry = worker_registry.WorkerRegistry()
    registry.register(mock_worker.MockWorker(worker_id="t0", roles={"trainer"}))
    t1 = mock_worker.MockWorker(worker_id="t1", roles={"trainer"})
    registry.register(t1)
    self._activate_all(registry)
    registry.unregister("t0")
    self.assertIn("trainer", registry.roles())
    self.assertEqual(registry.handles("trainer"), [registry.handle("t1")])

  def test_registry_retrieval_methods(self):
    registry = worker_registry.WorkerRegistry()
    r0 = mock_worker.MockWorker(worker_id="r0", roles={"rollout"})
    t0 = mock_worker.MockWorker(worker_id="t0", roles={"trainer"})
    registry.register(t0)
    registry.register(r0)

    self.assertEqual(registry.info("r0").worker_id, "r0")
    self.assertEqual(registry.worker_ids(), ["r0", "t0"])
    self.assertEqual(registry.infos(), [r0.info(), t0.info()])
    self.assertEqual(registry.infos("rollout"), [r0.info()])
    self.assertEqual(registry.infos("trainer"), [t0.info()])
    self.assertEmpty(registry.infos("inference"))

  def test_register_override_cleans_up_empty_roles(self):
    registry = worker_registry.WorkerRegistry()
    t0 = mock_worker.MockWorker(worker_id="t0", roles={"trainer"})
    t1 = mock_worker.MockWorker(worker_id="t1", roles={"trainer"})
    registry.register(t0)
    registry.register(t1)

    # Override t0 with a new worker that no longer has the "trainer" role
    t0_new = mock_worker.MockWorker(worker_id="t0", roles={"rollout"})
    registry.register(t0_new, override=True)

    # The new t0_new should be removed from the "trainer" role
    self.assertEqual(registry.infos("trainer"), [t1.info()])
    self.assertIn("trainer", registry.roles())

    # Then override t1 to rollout as well to empty the role
    t1_new = mock_worker.MockWorker(worker_id="t1", roles={"rollout"})
    registry.register(t1_new, override=True)
    self.assertNotIn("trainer", registry.roles())

    # The worker should just be "rollout" now
    self.assertEqual(registry.roles(), {"rollout"})
    self.assertEqual(
        registry.infos("rollout"), [t0_new.info(), t1_new.info()]
    )

  def test_register_handle_and_handles_by_role(self):
    registry = worker_registry.WorkerRegistry()
    local_worker = mock_worker.MockWorker(worker_id="r-local", roles={"rollout"})
    remote_handle = mock.MagicMock(spec=remote_execution.ActorHandle)

    registry.register(local_worker)
    info = registry.register_handle(
        "r-remote",
        [datatypes.Role.ROLLOUT],
        remote_handle,
        resources={"tpu": 4},
    )
    self._activate_all(registry)

    self.assertEqual(info.resources, {"remote": True, "tpu": 4})
    self.assertIs(registry.handle("r-remote"), remote_handle)
    self.assertIsInstance(
        registry.handle("r-local"), remote_execution.InProcessActorHandle
    )
    self.assertEqual(
        registry.handles(datatypes.Role.ROLLOUT),
        [registry.handle("r-local"), remote_handle],
    )

  @mock.patch.object(remote_execution.ActorHandle, "from_address")
  def test_register_from_hostname(self, mock_from_address):
    registry = worker_registry.WorkerRegistry()
    for port, (service_type, role) in enumerate(
        [
            ("trainer", datatypes.Role.ACTOR),
            ("rollout", datatypes.Role.ROLLOUT),
            ("inference", datatypes.Role.REFERENCE),
        ],
        start=5000,
    ):
      handle = mock.MagicMock(spec=remote_execution.ActorHandle)
      mock_from_address.return_value = handle
      wid = f"{service_type}-0"
      meta = pickle.dumps({
          "service_type": service_type,
          "service_port": port,
          "worker_id": wid,
      })
      registry.register_from_hostname("host", 0, meta, rpc_timeout_s=120.0)
      mock_from_address.assert_called_with(
          f"grpc://host:{port}", rpc_timeout_s=120.0
      )
      self.assertEqual(
          registry.status(wid), worker_registry.AdmissionStatus.REGISTERED
      )
      self.assertEmpty(registry.handles(role))
      registry.set_status(wid, worker_registry.AdmissionStatus.ACTIVE)
      self.assertEqual(registry.handles(role), [handle])

    # Re-registering on the same address via register_from_hostname overrides by
    # default without probing, closing the displaced handle and staging the new
    # handle in REGISTERED.
    old_rollout_handle = registry.handle("rollout-0")
    new_rollout_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_from_address.return_value = new_rollout_handle
    rollout_meta = pickle.dumps({
        "service_type": "rollout",
        "service_port": 5001,
        "worker_id": "rollout-0",
    })
    registry.register_from_hostname("host", 0, rollout_meta)
    old_rollout_handle.sync_close.assert_called_once()
    self.assertIs(registry.handle("rollout-0"), new_rollout_handle)
    self.assertEqual(
        registry.status("rollout-0"), worker_registry.AdmissionStatus.REGISTERED
    )
    self.assertEmpty(registry.handles(datatypes.Role.ROLLOUT))

    info_by_id = {i.worker_id: i for i in registry.infos()}
    self.assertEqual(
        info_by_id["trainer-0"],
        datatypes.WorkerInfo(
            worker_id="trainer-0",
            roles=frozenset({"actor"}),
            resources={"remote": True, "address": "host:5000"},
        ),
    )

  @mock.patch.object(remote_execution.ActorHandle, "from_address")
  def test_register_from_hostname_collision_rejects_live_rollout_and_replaces_dead(
      self, mock_from_address
  ):
    registry = worker_registry.WorkerRegistry()
    old_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    live_probe_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    live_probe_handle.submit.return_value = datatypes.HealthReport(
        state=datatypes.WorkerState.READY
    )
    rejected_new_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    dead_probe_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    dead_probe_handle.submit.side_effect = ConnectionError("connection refused")
    accepted_new_handle = mock.MagicMock(spec=remote_execution.ActorHandle)

    meta_a = pickle.dumps({
        "service_type": "rollout",
        "service_port": 5001,
        "worker_id": "rollout-0",
    })
    meta_b = pickle.dumps({
        "service_type": "rollout",
        "service_port": 6001,
        "worker_id": "rollout-0",
    })

    mock_from_address.side_effect = [
        old_handle,
        rejected_new_handle,
        live_probe_handle,
        accepted_new_handle,
        dead_probe_handle,
    ]
    registry.register_from_hostname("hostA", 0, meta_a)
    registry.set_status("rollout-0", worker_registry.AdmissionStatus.ACTIVE)

    # 1. Existing worker at hostA:5001 is still alive -> reject hostB:6001.
    with self.assertRaisesRegex(
        ValueError,
        "existing rollout worker at hostA:5001 is still alive when hostB:6001"
        " attempted to register",
    ):
      registry.register_from_hostname("hostB", 0, meta_b)
    live_probe_handle.submit.assert_called_once_with("heartbeat")
    live_probe_handle.sync_close.assert_called_once()
    rejected_new_handle.sync_close.assert_called_once()
    old_handle.sync_close.assert_not_called()
    self.assertIs(registry.handle("rollout-0"), old_handle)
    self.assertEqual(
        registry.status("rollout-0"), worker_registry.AdmissionStatus.ACTIVE
    )

    # 2. Existing worker at hostA:5001 is dead -> replace with hostB:6001.
    registry.register_from_hostname("hostB", 0, meta_b)
    dead_probe_handle.submit.assert_called_once_with("heartbeat")
    dead_probe_handle.sync_close.assert_called_once()
    old_handle.sync_close.assert_called_once()
    accepted_new_handle.sync_close.assert_not_called()
    self.assertIs(registry.handle("rollout-0"), accepted_new_handle)
    self.assertEqual(
        registry.status("rollout-0"), worker_registry.AdmissionStatus.REGISTERED
    )
    self.assertEqual(
        registry.info("rollout-0").resources["address"], "hostB:6001"
    )

  @mock.patch.object(remote_execution.ActorHandle, "from_address")
  def test_register_from_hostname_rejects_non_rollout_duplicate_and_concurrent_eviction(
      self, mock_from_address
  ):
    registry = worker_registry.WorkerRegistry()
    trainer_h0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    trainer_h1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_from_address.side_effect = [trainer_h0, trainer_h1]
    trainer_meta = pickle.dumps({
        "service_type": "trainer",
        "service_port": 5000,
        "worker_id": "trainer-0",
    })
    registry.register_from_hostname("hostA", 0, trainer_meta)
    with self.assertRaisesRegex(
        ValueError,
        "duplicate worker_id 'trainer-0' for role 'actor'; only rollout"
        " workers support re-registration",
    ):
      registry.register_from_hostname("hostA", 0, trainer_meta)
    trainer_h1.sync_close.assert_called_once()
    trainer_h0.sync_close.assert_not_called()
    self.assertIs(registry.handle("trainer-0"), trainer_h0)

    # Concurrent eviction of old rollout handle while liveness probe is running
    # outside the lock inserts the new handle cleanly without double-closing
    # old_handle.
    old_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    new_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    probe_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)

    def _evict_during_probe(method_name):
      self.assertEqual(method_name, "heartbeat")
      self.assertEqual(registry.unregister_handle(old_rollout), "rollout-race")
      raise ConnectionError("evicted while probing")

    probe_rollout.submit.side_effect = _evict_during_probe
    mock_from_address.side_effect = [old_rollout, new_rollout, probe_rollout]
    meta_race_a = pickle.dumps({
        "service_type": "rollout",
        "service_port": 5001,
        "worker_id": "rollout-race",
    })
    meta_race_b = pickle.dumps({
        "service_type": "rollout",
        "service_port": 6001,
        "worker_id": "rollout-race",
    })
    registry.register_from_hostname("hostA", 0, meta_race_a)
    registry.register_from_hostname("hostB", 0, meta_race_b)
    old_rollout.sync_close.assert_called_once()
    probe_rollout.sync_close.assert_called_once()
    new_rollout.sync_close.assert_not_called()
    self.assertIs(registry.handle("rollout-race"), new_rollout)

  def test_register_from_hostname_unknown_service_type(self):
    registry = worker_registry.WorkerRegistry()
    meta = pickle.dumps({
        "service_type": "unknown",
        "service_port": 5000,
        "worker_id": "bad-0",
    })
    with self.assertRaisesRegex(RuntimeError, "unknown service type unknown"):
      registry.register_from_hostname("host", 0, meta)

  def test_wait_for_workers_already_available(self):
    mock_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    registry = worker_registry.WorkerRegistry()
    registry.register_handle("actor-0", [datatypes.Role.ACTOR], mock_actor)
    registry.register_handle(
        "rollout-0", [datatypes.Role.ROLLOUT], mock_rollout
    )

    registry.wait_for_workers(
        {
            datatypes.Role.ACTOR: 1,
            datatypes.Role.ROLLOUT: 1,
            datatypes.Role.REFERENCE: 0,
        },
        timeout=1.0,
    )

  def test_wait_for_workers_delayed_registration(self):
    mock_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    registry = worker_registry.WorkerRegistry()

    def register_later():
      time.sleep(0.05)
      registry.register_handle("actor-0", [datatypes.Role.ACTOR], mock_actor)

    t = threading.Thread(target=register_later)
    t.start()
    try:
      registry.wait_for_workers(
          {datatypes.Role.ACTOR: 1},
          timeout=2.0,
      )
    finally:
      t.join()

    self.assertIn("actor-0", registry)
    self.assertEqual(
        registry.status("actor-0"), worker_registry.AdmissionStatus.REGISTERED
    )

  def test_wait_for_workers_timeout(self):
    registry = worker_registry.WorkerRegistry()
    with self.assertRaises(TimeoutError):
      registry.wait_for_workers(
          {datatypes.Role.ACTOR: 1},
          timeout=0.05,
      )

  def test_close_closes_all_handles_and_clears_state(self):
    registry = worker_registry.WorkerRegistry()
    h0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    registry.register_handle("w0", [datatypes.Role.ACTOR], h0)
    registry.register_handle("w1", [datatypes.Role.ROLLOUT], h1)

    registry.close()

    h0.sync_close.assert_called_once()
    h1.sync_close.assert_called_once()
    self.assertEmpty(registry)
    self.assertEmpty(registry.roles())

  def test_unregister_handle_removes_matching_worker_and_closes_handle(self):
    registry = worker_registry.WorkerRegistry()
    h0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    registry.register_handle("w0", [datatypes.Role.ROLLOUT], h0)

    self.assertEqual(registry.unregister_handle(h0), "w0")
    self.assertNotIn("w0", registry)
    h0.sync_close.assert_called_once()

    # Unregistering an unknown or already removed handle is a safe no-op.
    self.assertIsNone(registry.unregister_handle(h0))
    self.assertIsNone(registry.unregister_handle(h1))
    h1.sync_close.assert_not_called()

  def test_register_override_closes_displaced_handle(self):
    registry = worker_registry.WorkerRegistry()
    h0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    registry.register_handle("w0", [datatypes.Role.ROLLOUT], h0)
    registry.register_handle("w0", [datatypes.Role.ROLLOUT], h1, override=True)

    h0.sync_close.assert_called_once()
    h1.sync_close.assert_not_called()
    self.assertIs(registry.handle("w0"), h1)

  def test_update_resources_and_expected_handle_guard(self):
    registry = worker_registry.WorkerRegistry()
    h0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    registry.register_handle("w0", [datatypes.Role.ROLLOUT], h0)

    self.assertTrue(
        registry.update_resources(
            "w0", {"drain_timeout_s": 300.0}, expected_handle=h0
        )
    )
    self.assertEqual(registry.info("w0").resources["drain_timeout_s"], 300.0)

    self.assertFalse(
        registry.update_resources(
            "w0", {"drain_timeout_s": 600.0}, expected_handle=h1
        )
    )
    self.assertEqual(registry.info("w0").resources["drain_timeout_s"], 300.0)
    self.assertFalse(
        registry.update_resources(
            "missing", {"drain_timeout_s": 100.0}, expected_handle=h0
        )
    )

    # Guarded update_resources after worker is unregistered returns False
    # without raising.
    registry.unregister("w0")
    self.assertFalse(
        registry.update_resources(
            "w0", {"drain_timeout_s": 900.0}, expected_handle=h0
        )
    )

  def test_unregister_expected_handle_guard(self):
    registry = worker_registry.WorkerRegistry()
    h0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    registry.register_handle("w0", [datatypes.Role.ROLLOUT], h0)

    self.assertFalse(registry.unregister("w0", expected_handle=h1))
    h0.sync_close.assert_not_called()
    self.assertIn("w0", registry)

    self.assertTrue(registry.unregister("w0", expected_handle=h0))
    h0.sync_close.assert_called_once()
    self.assertNotIn("w0", registry)

    # Guarded unregister after worker was already unregistered returns False
    # without raising or closing again.
    self.assertFalse(registry.unregister("w0", expected_handle=h0))
    h0.sync_close.assert_called_once()

  def test_reregistered_worker_stays_unroutable_until_promoted_to_active(self):
    registry = worker_registry.WorkerRegistry()
    h0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    registry.register_handle("w0", [datatypes.Role.ROLLOUT], h0)
    registry.set_status("w0", worker_registry.AdmissionStatus.ACTIVE)
    self.assertEqual(registry.handles(datatypes.Role.ROLLOUT), [h0])

    self.assertEqual(registry.unregister_handle(h0), "w0")
    self.assertEmpty(registry.handles(datatypes.Role.ROLLOUT))

    # A restarted worker re-registering under the same id enters REGISTERED and
    # is invisible to data-plane handles() until promoted to ACTIVE.
    restarted = mock.MagicMock(spec=remote_execution.ActorHandle)
    registry.register_handle("w0", [datatypes.Role.ROLLOUT], restarted)
    self.assertEqual(
        registry.status("w0"), worker_registry.AdmissionStatus.REGISTERED
    )
    self.assertEmpty(registry.handles(datatypes.Role.ROLLOUT))

    registry.set_status("w0", worker_registry.AdmissionStatus.ACTIVE)
    self.assertEqual(registry.handles(datatypes.Role.ROLLOUT), [restarted])

  def test_status_filtering_and_promotion_lifecycle(self):
    registry = worker_registry.WorkerRegistry()
    h0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    r0_info = registry.register_handle("r0", [datatypes.Role.ROLLOUT], h0)
    self.assertEqual(
        registry.status("r0"), worker_registry.AdmissionStatus.REGISTERED
    )
    registry.set_status("r0", worker_registry.AdmissionStatus.ACTIVE)

    r1_info = registry.register_handle(
        "r1",
        [datatypes.Role.ROLLOUT],
        h1,
    )
    self.assertEqual(
        registry.status("r1"), worker_registry.AdmissionStatus.REGISTERED
    )
    self.assertEqual(
        registry.targets(
            statuses=(worker_registry.AdmissionStatus.REGISTERED,)
        ),
        [(r1_info, h1)],
    )
    self.assertEqual(
        registry.targets(statuses=(worker_registry.AdmissionStatus.ACTIVE,)),
        [(r0_info, h0)],
    )
    self.assertEqual(
        registry.targets_with_status(datatypes.Role.ROLLOUT, statuses=None),
        [
            (r0_info, h0, worker_registry.AdmissionStatus.ACTIVE),
            (r1_info, h1, worker_registry.AdmissionStatus.REGISTERED),
        ],
    )
    self.assertEqual(
        registry.targets_with_status(
            datatypes.Role.ROLLOUT,
            statuses=(worker_registry.AdmissionStatus.ACTIVE,),
        ),
        [(r0_info, h0, worker_registry.AdmissionStatus.ACTIVE)],
    )

    # Default handles() only returns ACTIVE workers; infos() and worker_ids()
    # return all registered workers.
    self.assertEqual(registry.handles(datatypes.Role.ROLLOUT), [h0])
    self.assertEqual(
        [i.worker_id for i in registry.infos(datatypes.Role.ROLLOUT)],
        ["r0", "r1"],
    )
    self.assertEqual(registry.worker_ids(), ["r0", "r1"])

    # Transition REGISTERED -> PENDING_WEIGHT_SYNC.
    self.assertTrue(
        registry.set_status(
            "r1",
            worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
            expected_handle=h1,
            expected_status=worker_registry.AdmissionStatus.REGISTERED,
        )
    )
    self.assertEqual(
        registry.status("r1"),
        worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
    )
    self.assertEqual(
        registry.handles(
            datatypes.Role.ROLLOUT,
            statuses=(
                worker_registry.AdmissionStatus.ACTIVE,
                worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
            ),
        ),
        [h0, h1],
    )

    # expected_status mismatch returns False and does not mutate status.
    self.assertFalse(
        registry.set_status(
            "r1",
            worker_registry.AdmissionStatus.ACTIVE,
            expected_status=worker_registry.AdmissionStatus.REGISTERED,
        )
    )
    self.assertEqual(
        registry.status("r1"),
        worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
    )

    # Promote PENDING_WEIGHT_SYNC -> ACTIVE.
    self.assertTrue(
        registry.set_status(
            "r1",
            worker_registry.AdmissionStatus.ACTIVE,
            expected_status=worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
        )
    )
    self.assertEqual(
        registry.status("r1"), worker_registry.AdmissionStatus.ACTIVE
    )
    self.assertEqual(registry.handles(datatypes.Role.ROLLOUT), [h0, h1])

  def test_expected_handle_guards_and_override_closes_displaced_handle(self):
    registry = worker_registry.WorkerRegistry()
    h_old = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_new = mock.MagicMock(spec=remote_execution.ActorHandle)
    registry.register_handle("r0", [datatypes.Role.ROLLOUT], h_old)

    registry.register_handle(
        "r0",
        [datatypes.Role.ROLLOUT],
        h_new,
        override=True,
    )
    h_old.sync_close.assert_called_once()
    self.assertIs(registry.handle("r0"), h_new)

    # Stale expected_handle must not mutate or unregister the new worker.
    self.assertFalse(
        registry.set_status(
            "r0",
            worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
            expected_handle=h_old,
        )
    )
    self.assertFalse(
        registry.set_status(
            "r0",
            worker_registry.AdmissionStatus.ACTIVE,
            expected_handle=h_old,
        )
    )
    self.assertFalse(registry.unregister("r0", expected_handle=h_old))
    self.assertIn("r0", registry)
    h_new.sync_close.assert_not_called()

    # Matching expected_handle succeeds.
    self.assertTrue(registry.unregister("r0", expected_handle=h_new))
    self.assertNotIn("r0", registry)
    h_new.sync_close.assert_called_once()

  def test_wait_for_change_supports_reentrant_queries(self):
    registry = worker_registry.WorkerRegistry()
    h = mock.MagicMock(spec=remote_execution.ActorHandle)

    def _register_soon():
      time.sleep(0.05)
      registry.register_handle("r0", [datatypes.Role.ROLLOUT], h)

    t = threading.Thread(target=_register_soon)
    t.start()
    try:
      self.assertTrue(
          registry.wait_for_change(
              lambda: bool(registry.targets(statuses=None))
              and bool(registry.infos()),
              timeout=2.0,
          )
      )
    finally:
      t.join()

  @mock.patch.object(remote_execution.ActorHandle, "from_address")
  def test_close_registration_and_close_reject_new_registrations(
      self, mock_from_address
  ):
    remote_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_from_address.return_value = remote_handle
    registry = worker_registry.WorkerRegistry()
    registry.close_registration()

    h = mock.MagicMock(spec=remote_execution.ActorHandle)
    with self.assertRaisesRegex(RuntimeError, "registration is closed"):
      registry.register_handle("r0", [datatypes.Role.ROLLOUT], h, override=True)
    meta = pickle.dumps({
        "service_type": "rollout",
        "service_port": 5000,
        "worker_id": "r0",
    })
    with self.assertRaisesRegex(RuntimeError, "registration is closed"):
      registry.register_from_hostname("host", 0, meta)
    remote_handle.sync_close.assert_called_once()


if __name__ == "__main__":
  absltest.main()
