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

"""A sync round must never be quietly partial."""

from typing import Any, Optional

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import weight_sync_coordinator


class _Replica:
  """A rollout replica that installs versions, or refuses to."""

  def __init__(
      self,
      worker_id: str,
      *,
      fail_times: int = 0,
      raises: Optional[Exception] = None,
      acks_version: Optional[int] = None,
  ):
    self._worker_id = worker_id
    self._fail_times = fail_times
    self._raises = raises
    self._acks_version = acks_version
    self.version = 0
    self.fenced = 0
    self.install_attempts = 0

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id, roles=frozenset({"rollout"})
    )

  def prepare_weight_sync(self, metadata: Any) -> datatypes.Response:
    del metadata
    self.fenced += 1
    return datatypes.Response()

  def sync_weights(self, metadata: Any) -> int:
    self.install_attempts += 1
    if self._raises is not None:
      raise self._raises
    if self._fail_times > 0:
      self._fail_times -= 1
      raise ConnectionError(f"{self._worker_id} unreachable")
    if self._acks_version is not None:
      return self._acks_version
    self.version = metadata.policy_version
    return self.version


class _Trainer:

  def __init__(self):
    self.staged = []

  def prepare_weight_sync(self, request: Any) -> str:
    self.staged.append(request.policy_version)
    return f"weights-at-{request.policy_version}"


class WeightSyncCoordinatorTest(absltest.TestCase):

  def test_all_replicas_reach_the_version(self):
    replicas = [_Replica("r0"), _Replica("r1"), _Replica("r2")]
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        trainer=_Trainer(), replicas=replicas
    )

    outcome = coordinator.sync(version=1)

    self.assertTrue(outcome.all_synced)
    self.assertCountEqual(outcome.synced, ["r0", "r1", "r2"])
    self.assertTrue(all(r.version == 1 for r in replicas))
    # Each was fenced before installing.
    self.assertTrue(all(r.fenced == 1 for r in replicas))

  def test_the_trainer_stages_before_replicas_install(self):
    trainer = _Trainer()
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        trainer=trainer, replicas=[_Replica("r0")]
    )

    coordinator.sync(version=3)

    self.assertEqual(trainer.staged, [3])

  def test_an_unreachable_replica_is_quarantined_and_the_rest_advance(self):
    healthy, broken = _Replica("healthy"), _Replica(
        "broken", raises=ConnectionError("gone")
    )
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        trainer=_Trainer(), replicas=[healthy, broken], max_retries=1
    )

    outcome = coordinator.sync(version=2)

    self.assertFalse(outcome.all_synced)
    self.assertEqual(outcome.synced, ["healthy"])
    self.assertEqual(outcome.quarantined_ids, ["broken"])
    self.assertEqual(
        outcome.quarantined[0].reason,
        weight_sync_coordinator.QuarantineReason.UNREACHABLE,
    )
    self.assertIn("gone", outcome.quarantined[0].detail)
    self.assertEqual(healthy.version, 2)

  def test_a_replica_that_recovers_within_its_retries_is_not_quarantined(self):
    flaky = _Replica("flaky", fail_times=1)
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        trainer=_Trainer(), replicas=[flaky], max_retries=1
    )

    outcome = coordinator.sync(version=5)

    self.assertTrue(outcome.all_synced)
    self.assertEqual(flaky.install_attempts, 2)
    self.assertEqual(flaky.version, 5)

  def test_retries_are_bounded(self):
    always = _Replica("always", fail_times=99)
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        trainer=_Trainer(), replicas=[always], max_retries=2
    )

    outcome = coordinator.sync(version=1)

    self.assertEqual(always.install_attempts, 3)
    self.assertEqual(outcome.quarantined_ids, ["always"])

  def test_installing_the_wrong_version_is_reported_as_such(self):
    """Reachable but running something else is a different problem."""
    liar = _Replica("liar", acks_version=4)
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        trainer=_Trainer(), replicas=[liar], max_retries=3
    )

    outcome = coordinator.sync(version=7)

    self.assertEqual(
        outcome.quarantined[0].reason,
        weight_sync_coordinator.QuarantineReason.WRONG_VERSION,
    )
    self.assertIn("expected 7", outcome.quarantined[0].detail)
    # No point retrying a replica that installs the wrong thing.
    self.assertEqual(liar.install_attempts, 1)

  def test_a_programming_error_surfaces_instead_of_quarantining_everyone(self):
    """A broken call must not look like a fleet-wide outage."""
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        trainer=_Trainer(),
        replicas=[_Replica("r0", raises=TypeError("bad signature"))],
    )

    with self.assertRaises(TypeError):
      coordinator.sync(version=1)

  def test_runs_without_a_trainer(self):
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        replicas=[_Replica("r0")]
    )

    self.assertTrue(coordinator.sync(version=1).all_synced)

  def test_rejects_a_negative_retry_budget(self):
    with self.assertRaises(ValueError):
      weight_sync_coordinator.WeightSyncCoordinator(max_retries=-1)


if __name__ == "__main__":
  absltest.main()
