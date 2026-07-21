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
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.testing import fake_rollout_worker
from tunix.experimental.testing import fake_trainer_worker
from tunix.experimental.worker import abstract_worker


class _FusedWorker(abstract_worker.Worker):
  """A minimal worker declaring multiple roles, for grouping tests."""

  def __init__(self, worker_id: str, roles):
    self._info = datatypes.WorkerInfo(
        worker_id=worker_id, roles=frozenset(roles)
    )

  def initialize(self) -> None:
    pass

  def compile(self, shape_config) -> None:
    del shape_config

  def start(self) -> None:
    pass

  def stop(self) -> None:
    pass

  def health(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(state="READY")

  def info(self) -> datatypes.WorkerInfo:
    return self._info


class WorkerRegistryTest(absltest.TestCase):

  def test_register_and_group_by_role(self):
    registry = worker_registry.WorkerRegistry()
    rollout = fake_rollout_worker.FakeRolloutWorker(worker_id="r0")
    trainer = fake_trainer_worker.FakeTrainerWorker(worker_id="t0")
    registry.register(rollout)
    registry.register(trainer)

    self.assertEqual(registry.roles(), {"rollout", "trainer"})
    self.assertEqual([w for w in registry.group("rollout")], [rollout])
    self.assertEqual(registry.group("trainer").members(), [trainer])
    self.assertIs(registry.get("r0"), rollout)
    self.assertLen(registry, 2)
    self.assertIn("t0", registry)

  def test_fused_worker_joins_every_role(self):
    registry = worker_registry.WorkerRegistry()
    fused = _FusedWorker("f0", {"trainer", "inference"})
    registry.register(fused)

    self.assertEqual(registry.group("trainer").members(), [fused])
    self.assertEqual(registry.group("inference").members(), [fused])

  def test_duplicate_worker_id_raises(self):
    registry = worker_registry.WorkerRegistry()
    registry.register(fake_trainer_worker.FakeTrainerWorker(worker_id="dup"))
    with self.assertRaises(ValueError):
      registry.register(fake_rollout_worker.FakeRolloutWorker(worker_id="dup"))

  def test_worker_without_roles_raises(self):
    registry = worker_registry.WorkerRegistry()
    with self.assertRaises(ValueError):
      registry.register(_FusedWorker("no-roles", set()))

  def test_unknown_role_returns_empty_group(self):
    registry = worker_registry.WorkerRegistry()
    registry.register(fake_trainer_worker.FakeTrainerWorker(worker_id="t0"))
    group = registry.group("inference")
    self.assertTrue(group.is_empty())
    self.assertEmpty(group.members())

  def test_unregister_removes_from_registry_and_groups(self):
    registry = worker_registry.WorkerRegistry()
    registry.register(fake_trainer_worker.FakeTrainerWorker(worker_id="t0"))
    registry.unregister("t0")
    self.assertNotIn("t0", registry)
    self.assertTrue(registry.group("trainer").is_empty())
    self.assertNotIn("trainer", registry.roles())
    with self.assertRaises(KeyError):
      registry.unregister("t0")


if __name__ == "__main__":
  absltest.main()
