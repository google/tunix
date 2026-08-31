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


if __name__ == "__main__":
  absltest.main()
