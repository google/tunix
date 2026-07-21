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

"""Tests for the WeightSyncCoordinator."""

from absl.testing import absltest
from tunix.experimental.orchestrator import weight_sync_coordinator
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.testing import fake_rollout_worker
from tunix.experimental.testing import fake_trainer_worker


class _FailingRollout(fake_rollout_worker.FakeRolloutWorker):
  """A rollout replica whose install always fails."""

  def sync_weights(self, metadata):
    raise RuntimeError("install failed")


def _registry(*rollouts, with_trainer=True):
  registry = worker_registry.WorkerRegistry()
  if with_trainer:
    registry.register(fake_trainer_worker.FakeTrainerWorker(worker_id="t0"))
  for rollout in rollouts:
    registry.register(rollout)
  return registry


class WeightSyncCoordinatorTest(absltest.TestCase):

  def test_syncs_all_replicas_to_version(self):
    r0 = fake_rollout_worker.FakeRolloutWorker(worker_id="r0")
    r1 = fake_rollout_worker.FakeRolloutWorker(worker_id="r1")
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(_registry(r0, r1))

    outcome = coordinator.sync(5)

    self.assertEqual(sorted(outcome.synced), ["r0", "r1"])
    self.assertEmpty(outcome.quarantined)
    self.assertTrue(outcome.all_synced)
    self.assertEqual(outcome.version, 5)
    self.assertEqual(outcome.metadata.version, 5)
    self.assertEqual(r0.health().policy_version, 5)
    self.assertEqual(r1.health().policy_version, 5)

  def test_quarantines_failing_replica_and_advances_the_rest(self):
    good = fake_rollout_worker.FakeRolloutWorker(worker_id="good")
    bad = _FailingRollout(worker_id="bad")
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        _registry(good, bad), max_retries=1
    )

    outcome = coordinator.sync(2)

    self.assertEqual(outcome.synced, ["good"])
    self.assertEqual(outcome.quarantined, ["bad"])
    self.assertFalse(outcome.all_synced)
    self.assertEqual(good.health().policy_version, 2)

  def test_no_trainer_raises(self):
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        _registry(
            fake_rollout_worker.FakeRolloutWorker(worker_id="r0"),
            with_trainer=False,
        )
    )
    with self.assertRaises(ValueError):
      coordinator.sync(1)

  def test_no_replicas_stages_but_syncs_none(self):
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(_registry())
    outcome = coordinator.sync(1)
    self.assertEmpty(outcome.synced)
    self.assertEmpty(outcome.quarantined)
    self.assertEqual(outcome.metadata.version, 1)


if __name__ == "__main__":
  absltest.main()
