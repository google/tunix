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

import asyncio
import threading

from absl.testing import absltest
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.train import abstract_trainer
from tunix.dist_rl.worker import trainer_worker as trainer_worker_lib

WorkerState = datatypes.WorkerState


class _FakeTrainer:

  def __init__(self):
    self.calls = []

  def with_fwd_only_fn(self, fn):
    del fn
    return self

  def prepare_weight_sync(self, *, sync_request=None):
    del sync_request
    self.calls.append("prepare")
    return [{"unit": "trainer0"}]

  # Not overridden: the AbstractTrainer default every trainer inherits.
  release_weight_sync = abstract_trainer.AbstractTrainer.release_weight_sync


class _ReleasingTrainer(_FakeTrainer):

  def release_weight_sync(self, *, sync_request=None):
    self.calls.append("release")
    self.release_request = sync_request
    return "released"


class _FailingTrainer(_FakeTrainer):

  def prepare_weight_sync(self, *, sync_request=None):
    raise RuntimeError("boom")


class WeightSyncStagingTest(absltest.TestCase):

  def _worker(self, trainer):
    worker = trainer_worker_lib.TrainerWorker(
        trainer_factory=lambda: trainer, worker_id="t0"
    )
    worker.initialize()
    worker._state = WorkerState.READY
    return worker

  def test_prepare_leaves_the_worker_ready(self):
    # Staging is complete when prepare returns, so the trainer may take the
    # next batch while the rest of the round runs.
    worker = self._worker(_FakeTrainer())
    worker.prepare_weight_sync()
    self.assertEqual(worker.state, WorkerState.READY)

  def test_release_does_not_wait_for_a_running_worker_call(self):
    # A release that queued behind `fwd_bwd` would hold the round open, and
    # with it the rollouts' next version.
    worker = self._worker(_ReleasingTrainer())
    worker.prepare_weight_sync()
    call_running = threading.Event()
    finish_call = threading.Event()

    def long_worker_call():
      with worker._worker_lock:
        call_running.set()
        finish_call.wait()

    caller = threading.Thread(target=long_worker_call)
    caller.start()
    self.addCleanup(caller.join)
    self.addCleanup(finish_call.set)
    call_running.wait()

    async def release():
      return await asyncio.wait_for(worker.release_weight_sync(), timeout=10)

    self.assertEqual(asyncio.run(release()), "released")

  def test_release_waits_for_a_running_prepare(self):
    release_may_run = threading.Event()
    calls = []

    class _SlowPrepareTrainer(_ReleasingTrainer):

      def prepare_weight_sync(self, *, sync_request=None):
        del sync_request
        calls.append("prepare_started")
        preparing.set()
        release_may_run.wait()
        calls.append("prepare_done")
        return [{"unit": "trainer0"}]

      def release_weight_sync(self, *, sync_request=None):
        del sync_request
        calls.append("release")
        return "released"

    preparing = threading.Event()
    worker = self._worker(_SlowPrepareTrainer())
    preparer = threading.Thread(target=worker.prepare_weight_sync)
    preparer.start()
    self.addCleanup(preparer.join)
    self.addCleanup(release_may_run.set)
    preparing.wait()

    async def release_while_preparing():
      release = asyncio.create_task(worker.release_weight_sync())
      await asyncio.sleep(0.2)
      still_waiting = not release.done()
      release_may_run.set()
      return still_waiting, await release

    still_waiting, result = asyncio.run(release_while_preparing())

    self.assertTrue(still_waiting)
    self.assertEqual(result, "released")
    self.assertEqual(calls, ["prepare_started", "prepare_done", "release"])

  def test_prepare_returns_trainer_metadata(self):
    worker = self._worker(_FakeTrainer())
    self.assertEqual(worker.prepare_weight_sync(), [{"unit": "trainer0"}])

  def test_prepare_failure_sets_error_state(self):
    worker = self._worker(_FailingTrainer())
    with self.assertRaises(RuntimeError):
      worker.prepare_weight_sync()
    self.assertEqual(worker.state, WorkerState.ERROR)

  def test_release_calls_the_trainer_and_leaves_the_worker_ready(self):
    trainer = _ReleasingTrainer()
    worker = self._worker(trainer)
    worker.prepare_weight_sync()
    self.assertEqual(asyncio.run(worker.release_weight_sync()), "released")
    self.assertEqual(worker.state, WorkerState.READY)
    self.assertEqual(trainer.calls, ["prepare", "release"])

  def test_release_with_default_trainer_hook(self):
    worker = self._worker(_FakeTrainer())
    worker.prepare_weight_sync()
    self.assertIsNone(asyncio.run(worker.release_weight_sync()))
    self.assertEqual(worker.state, WorkerState.READY)

  def test_release_forwards_sync_request(self):
    trainer = _ReleasingTrainer()
    worker = self._worker(trainer)
    request = object()
    worker.prepare_weight_sync()
    asyncio.run(worker.release_weight_sync(request))
    self.assertIs(trainer.release_request, request)


if __name__ == "__main__":
  absltest.main()
