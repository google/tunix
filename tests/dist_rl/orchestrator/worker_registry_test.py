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

  def test_register_and_handles_by_role(self):
    registry = worker_registry.WorkerRegistry()
    rollout = mock_worker.MockWorker(worker_id="r0", roles={"rollout"})
    trainer0 = mock_worker.MockWorker(worker_id="t0", roles={"trainer"})
    trainer1 = mock_worker.MockWorker(worker_id="t1", roles={"trainer"})
    registry.register(rollout)
    registry.register(trainer0)
    registry.register(trainer1)

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
    self.assertEmpty(registry.handles("inference"))

  def test_unregister_removes_from_registry_and_closes_handle(self):
    registry = worker_registry.WorkerRegistry()
    remote_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    registry.register_handle("t0", ["trainer"], remote_handle)
    registry.unregister("t0")
    remote_handle.close.assert_awaited_once()
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
    mock_from_address.return_value = mock.MagicMock(
        spec=remote_execution.ActorHandle
    )
    registry = worker_registry.WorkerRegistry()
    for port, (service_type, role) in enumerate(
        [
            ("trainer", datatypes.Role.ACTOR),
            ("rollout", datatypes.Role.ROLLOUT),
            ("inference", datatypes.Role.REFERENCE),
        ],
        start=5000,
    ):
      meta = pickle.dumps({
          "service_type": service_type,
          "service_port": port,
          "worker_id": f"{service_type}-0",
      })
      registry.register_from_hostname("host", 0, meta, rpc_timeout_s=120.0)
      mock_from_address.assert_called_with(
          f"grpc://host:{port}", rpc_timeout_s=120.0
      )
      self.assertEqual(registry.handles(role), [mock_from_address.return_value])

    info_by_id = {i.worker_id: i for i in registry.infos()}
    self.assertEqual(
        info_by_id["trainer-0"],
        datatypes.WorkerInfo(
            worker_id="trainer-0",
            roles=frozenset({"actor"}),
            resources={"remote": True, "address": "host:5000"},
        ),
    )

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

    self.assertLen(registry.handles(datatypes.Role.ACTOR), 1)

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

    h0.close.assert_awaited_once()
    h1.close.assert_awaited_once()
    self.assertEmpty(registry)
    self.assertEmpty(registry.roles())


if __name__ == "__main__":
  absltest.main()
