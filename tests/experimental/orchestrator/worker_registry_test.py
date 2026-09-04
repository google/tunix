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

import pickle
import threading
import time
from unittest import mock

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.worker import mock_worker
from tunix.experimental.worker import remote_execution


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

    r0_handle = registry.get("r0")
    self.assertIsInstance(r0_handle, remote_execution.InProcessActorHandle)
    self.assertEqual(list(rollout_group), [r0_handle])
    self.assertEqual(rollout_group.handles(), [r0_handle])
    self.assertEqual(rollout_group.members(), [r0_handle])
    self.assertEqual(rollout_group.worker_ids(), ["r0"])
    self.assertEqual(rollout_group.infos(), [registry.info("r0")])

    trainer_group = registry.group("trainer")
    self.assertEqual(trainer_group.role, "trainer")
    self.assertLen(trainer_group, 2)
    t0_handle = registry.get("t0")
    t1_handle = registry.get("t1")
    self.assertEqual(trainer_group.members(), [t0_handle, t1_handle])
    self.assertEqual(trainer_group.handles(), [t0_handle, t1_handle])
    self.assertEqual(trainer_group.worker_ids(), ["t0", "t1"])

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
    self.assertEqual(rollout_group[0], registry.get("r0"))

    self.assertEqual(trainer_group.role, "trainer")
    self.assertFalse(trainer_group.is_empty())
    self.assertLen(trainer_group, 2)
    self.assertEqual(trainer_group[0], registry.get("t0"))
    self.assertEqual(trainer_group[1], registry.get("t1"))

    empty_group = registry.group("inference")
    self.assertEqual(empty_group.role, "inference")
    self.assertTrue(empty_group.is_empty())
    self.assertEmpty(empty_group)
    self.assertEmpty(list(empty_group))
    self.assertEmpty(empty_group.handles())
    self.assertEmpty(empty_group.members())
    self.assertEmpty(empty_group.infos())
    self.assertEmpty(empty_group.worker_ids())

  def test_fused_worker_joins_every_role(self):
    registry = worker_registry.WorkerRegistry()
    fused = mock_worker.MockWorker("f0", {"trainer", "inference"})
    registry.register(fused)

    fused_handle = registry.get("f0")
    self.assertEqual(registry.group("trainer").members(), [fused_handle])
    self.assertEqual(registry.group("inference").members(), [fused_handle])

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
    t1_handle = registry.get("t1")
    registry.unregister("t0")
    self.assertIn("trainer", registry.roles())
    self.assertEqual(registry.group("trainer").members(), [t1_handle])

  def test_registry_retrieval_methods(self):
    registry = worker_registry.WorkerRegistry()
    r0 = mock_worker.MockWorker(worker_id="r0", roles={"rollout"})
    t0 = mock_worker.MockWorker(worker_id="t0", roles={"trainer"})
    registry.register(t0)
    registry.register(r0)

    r0_handle = registry.get("r0")
    t0_handle = registry.get("t0")

    self.assertEqual(registry.info("r0").worker_id, "r0")
    self.assertEqual(registry.get_handle("r0"), r0_handle)
    self.assertEqual(registry.get_handle("t0"), t0_handle)
    with self.assertRaises(KeyError):
      registry.get_handle("non-existent")
    self.assertEqual(registry.worker_ids(), ["t0", "r0"])
    self.assertEqual(registry.handles(), [t0_handle, r0_handle])
    self.assertEqual(registry.workers(), [t0_handle, r0_handle])
    self.assertEqual(registry.infos(), [t0.info(), r0.info()])

  def test_register_override_cleans_up_empty_roles(self):
    registry = worker_registry.WorkerRegistry()
    t0 = mock_worker.MockWorker(worker_id="t0", roles={"trainer"})
    t1 = mock_worker.MockWorker(worker_id="t1", roles={"trainer"})
    registry.register(t0)
    registry.register(t1)
    t1_handle = registry.get("t1")

    # Override t0 with a new worker that no longer has the "trainer" role
    t0_new = mock_worker.MockWorker(worker_id="t0", roles={"rollout"})
    registry.register(t0_new, override=True)
    t0_new_handle = registry.get("t0")

    # The new t0_new should be removed from the "trainer" group
    self.assertNotIn(t0_new_handle, registry.group("trainer").members())
    self.assertEqual(registry.group("trainer").members(), [t1_handle])
    self.assertIn("trainer", registry.roles())

    # Then override t1 to rollout as well to empty the role
    t1_new = mock_worker.MockWorker(worker_id="t1", roles={"rollout"})
    registry.register(t1_new, override=True)
    t1_new_handle = registry.get("t1")
    self.assertNotIn("trainer", registry.roles())

    # The worker should just be "rollout" now
    self.assertEqual(registry.roles(), {"rollout"})
    self.assertCountEqual(
        registry.group("rollout").members(), [t0_new_handle, t1_new_handle]
    )

  def test_register_handle_direct(self):
    registry = worker_registry.WorkerRegistry()
    mock_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    info = registry.register_handle(
        worker_id="actor-0",
        roles=[datatypes.Role.ACTOR],
        handle=mock_handle,
        resources={"cores": 8},
    )

    self.assertEqual(info.worker_id, "actor-0")
    self.assertEqual(info.roles, frozenset({"actor"}))
    self.assertEqual(info.resources, {"remote": True, "cores": 8})
    self.assertIs(registry.get("actor-0"), mock_handle)
    self.assertEqual(registry.handles("actor"), [mock_handle])
    self.assertEqual(registry.handles(datatypes.Role.ACTOR), [mock_handle])

    # Rejects non-ActorHandle
    with self.assertRaises(TypeError):
      registry.register_handle(
          worker_id="bad",
          roles=["actor"],
          handle="not_a_handle",  # pytype: disable=wrong-arg-types
      )

    # Rejects empty roles
    with self.assertRaises(ValueError):
      registry.register_handle(
          worker_id="no-roles",
          roles=[],
          handle=mock_handle,
      )

    # Rejects duplicate worker_id unless override
    with self.assertRaises(ValueError):
      registry.register_handle(
          worker_id="actor-0",
          roles=["actor"],
          handle=mock_handle,
      )

  @mock.patch.object(remote_execution.ActorHandle, "from_address")
  def test_register_from_hostname(self, mock_from_address):
    mock_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_from_address.return_value = mock_handle

    registry = worker_registry.WorkerRegistry()
    meta = pickle.dumps({
        "service_type": "trainer",
        "service_port": 5001,
        "worker_id": "trainer-0",
    })
    info = registry.register_from_hostname(
        hostname="test-host",
        port=5000,
        metadata=meta,
        rpc_timeout_s=60.0,
    )

    mock_from_address.assert_called_once_with(
        "grpc://test-host:5001", rpc_timeout_s=60.0
    )
    self.assertEqual(info.worker_id, "trainer-0")
    self.assertEqual(info.roles, frozenset({"actor"}))
    self.assertEqual(
        info.resources, {"remote": True, "address": "test-host:5001"}
    )
    self.assertIs(registry.get("trainer-0"), mock_handle)

  def test_register_from_hostname_unknown_service_type(self):
    registry = worker_registry.WorkerRegistry()
    meta = pickle.dumps({
        "service_type": "unknown_service",
        "service_port": 5000,
        "worker_id": "bad-0",
    })
    with self.assertRaisesRegex(
        RuntimeError, "unknown service type unknown_service"
    ):
      registry.register_from_hostname("host", 0, meta)

  def test_role_normalization(self):
    registry = worker_registry.WorkerRegistry()
    handle_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    handle_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)

    registry.register_handle(
        worker_id="a0",
        roles=[datatypes.Role.ACTOR],
        handle=handle_actor,
    )
    registry.register_handle(
        worker_id="r0",
        roles=["rollout"],
        handle=handle_rollout,
    )

    # Lookup by enum Role.ACTOR or string "actor" both work
    self.assertEqual(
        registry.group(datatypes.Role.ACTOR).handles(), [handle_actor]
    )
    self.assertEqual(registry.group("actor").handles(), [handle_actor])
    self.assertEqual(registry.handles(datatypes.Role.ACTOR), [handle_actor])
    self.assertEqual(registry.handles("actor"), [handle_actor])

    # Lookup by enum Role.ROLLOUT or string "rollout" both work
    self.assertEqual(
        registry.group(datatypes.Role.ROLLOUT).handles(), [handle_rollout]
    )
    self.assertEqual(registry.group("rollout").handles(), [handle_rollout])
    self.assertEqual(registry.handles(datatypes.Role.ROLLOUT), [handle_rollout])
    self.assertEqual(registry.handles("rollout"), [handle_rollout])

  def test_wait_for_workers_already_available(self):
    registry = worker_registry.WorkerRegistry()
    registry.register_handle(
        "a0",
        [datatypes.Role.ACTOR],
        mock.MagicMock(spec=remote_execution.ActorHandle),
    )
    registry.register_handle(
        "r0",
        [datatypes.Role.ROLLOUT],
        mock.MagicMock(spec=remote_execution.ActorHandle),
    )

    # Should return immediately without timing out
    registry.wait_for_workers(
        {
            datatypes.Role.ACTOR: 1,
            datatypes.Role.ROLLOUT: 1,
            datatypes.Role.REFERENCE: 0,
        },
        timeout=1.0,
    )

  def test_wait_for_workers_delayed_registration(self):
    registry = worker_registry.WorkerRegistry()
    mock_actor = mock.MagicMock(spec=remote_execution.ActorHandle)

    def register_later():
      time.sleep(0.05)
      registry.register_handle(
          worker_id="actor-0",
          roles=[datatypes.Role.ACTOR],
          handle=mock_actor,
      )

    t = threading.Thread(target=register_later)
    t.start()
    registry.wait_for_workers(
        min_workers={datatypes.Role.ACTOR: 1},
        timeout=2.0,
        poll_interval_s=0.01,
    )
    t.join()
    self.assertLen(registry.handles(datatypes.Role.ACTOR), 1)

  def test_wait_for_workers_timeout(self):
    registry = worker_registry.WorkerRegistry()
    with self.assertRaises(TimeoutError):
      registry.wait_for_workers(
          {datatypes.Role.ACTOR: 1},
          timeout=0.05,
          poll_interval_s=0.01,
      )


if __name__ == "__main__":
  absltest.main()
