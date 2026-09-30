# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for TrainerWorker weight sync staging."""

import types

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.weight_sync import weight_sync
from tunix.experimental.worker import trainer_worker as trainer_worker_lib

WorkerState = datatypes.WorkerState


class _FakeTrainer:

  def __init__(self):
    self.calls = []

  def prepare_weight_sync(self, sync_request=None, **kwargs):
    self.calls.append("prepare")
    return [{"unit": "trainer0"}]


class _ReleasingTrainer(_FakeTrainer):

  def release_weight_sync(self, **kwargs):
    self.calls.append("release")
    self.release_kwargs = kwargs
    return "released"

class _StagingTrainer(_ReleasingTrainer):
  """MaxText-style trainer: the synchronizer lives in `_weight_sync`."""

  def __init__(self, staged_on_host):
    super().__init__()
    self._weight_sync = types.SimpleNamespace(staged_on_host=staged_on_host)


def _background_request():
  return types.SimpleNamespace(
      extra_config={weight_sync.RELEASE_SOURCE_AFTER_STAGE: True}
  )


class _FailingTrainer(_FakeTrainer):

  def prepare_weight_sync(self, sync_request=None, **kwargs):
    raise RuntimeError("boom")


class WeightSyncStagingTest(absltest.TestCase):

  def _worker(self, trainer):
    worker = trainer_worker_lib.TrainerWorker(
        trainer_factory=lambda: trainer, worker_id="t0"
    )
    worker.initialize()
    worker._state = WorkerState.READY
    return worker

  def test_prepare_stays_syncing(self):
    worker = self._worker(_FakeTrainer())
    worker.prepare_weight_sync()
    self.assertEqual(worker.state, WorkerState.SYNCING)

  def test_prepare_returns_trainer_metadata(self):
    worker = self._worker(_FakeTrainer())
    self.assertEqual(worker.prepare_weight_sync(), [{"unit": "trainer0"}])

  def test_prepare_failure_sets_error_state(self):
    worker = self._worker(_FailingTrainer())
    with self.assertRaises(RuntimeError):
      worker.prepare_weight_sync()
    self.assertEqual(worker.state, WorkerState.ERROR)

  def test_release_restores_ready(self):
    trainer = _ReleasingTrainer()
    worker = self._worker(trainer)
    worker.prepare_weight_sync()
    self.assertEqual(worker.release_weight_sync(), "released")
    self.assertEqual(worker.state, WorkerState.READY)
    self.assertEqual(trainer.calls, ["prepare", "release"])

  def test_release_without_trainer_hook(self):
    worker = self._worker(_FakeTrainer())
    worker.prepare_weight_sync()
    self.assertIsNone(worker.release_weight_sync())
    self.assertEqual(worker.state, WorkerState.READY)

  def test_background_round_host_staged_returns_to_ready(self):
    trainer = _StagingTrainer(staged_on_host=True)
    worker = self._worker(trainer)
    worker.prepare_weight_sync(_background_request())
    # The next train step may run while the transfer is in flight.
    self.assertEqual(worker.state, WorkerState.READY)
    worker.release_weight_sync()
    self.assertEqual(worker.state, WorkerState.READY)
    self.assertEqual(trainer.calls, ["prepare", "release"])

  def test_background_round_finds_peft_synchronizer(self):
    trainer = _ReleasingTrainer()
    trainer._weight_sync_worker = types.SimpleNamespace(staged_on_host=True)
    worker = self._worker(trainer)
    worker.prepare_weight_sync(_background_request())
    self.assertEqual(worker.state, WorkerState.READY)

  def test_background_round_from_device_memory_fails(self):
    worker = self._worker(_StagingTrainer(staged_on_host=False))
    with self.assertRaisesRegex(RuntimeError, "DIRECT_DEVICE_BUFFER=0"):
      worker.prepare_weight_sync(_background_request())
    self.assertEqual(worker.state, WorkerState.ERROR)

  def test_blocking_round_stays_syncing_even_if_host_staged(self):
    worker = self._worker(_StagingTrainer(staged_on_host=True))
    worker.prepare_weight_sync(types.SimpleNamespace(extra_config={}))
    self.assertEqual(worker.state, WorkerState.SYNCING)

  def test_release_forwards_sync_request(self):
    trainer = _ReleasingTrainer()
    worker = self._worker(trainer)
    request = object()
    worker.prepare_weight_sync()
    worker.release_weight_sync(request)
    self.assertIs(trainer.release_kwargs["sync_request"], request)


if __name__ == "__main__":
  absltest.main()
