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

"""Tests for TrainerWorker.stage_fwd_bwd / fwd_bwd_staged.

A staged micro-batch must train exactly as the same micro-batch sent with
`fwd_bwd`, staging must not touch the trainer's state, and the worker must
never train a staged micro-batch in another's place.
"""

import asyncio
from typing import Any
from unittest import mock

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import trainer_worker

WorkerState = datatypes.WorkerState


class _Trainer:
  """Records what reaches the trainer."""

  def __init__(self):
    self.calls: list[tuple[str, Any, dict[str, Any]]] = []
    self.policy_version = 2

  def fwd_bwd(self, payload, **kwargs):
    self.calls.append(("fwd_bwd", payload, kwargs))

  def update(self, **kwargs):
    self.calls.append(("update", None, kwargs))
    return 1

  def close(self):
    pass


class _PreparingTrainer(_Trainer):
  """Has `prepare_fwd_bwd_input`, e.g. to place the inputs on its devices."""

  def prepare_fwd_bwd_input(self, payload):
    self.calls.append(("prepare", payload, {}))
    return ("on_device", payload)


class _FailingPrepareTrainer(_Trainer):

  def prepare_fwd_bwd_input(self, payload):
    raise RuntimeError("device_put failed")


class _WaitingTrainer(_PreparingTrainer):
  """Also has `wait_for_device`, as MaxText's engine does."""

  def __init__(self, waited_s: float = 0.75):
    super().__init__()
    self._waited_s = waited_s

  def wait_for_device(self):
    self.calls.append(("wait_for_device", None, {}))
    return self._waited_s


class _FailingWaitTrainer(_Trainer):

  def wait_for_device(self):
    raise RuntimeError("device lost")


class _TokenTrainer(_WaitingTrainer):
  """Its update publishes an output to wait on (`last_update_token`)."""

  def __init__(self, waited_s: float = 0.75):
    super().__init__(waited_s=waited_s)
    self.last_update_token = None

  def update(self, **kwargs):
    step = super().update(**kwargs)
    self.last_update_token = jnp.array(float(step))
    return step


class _UpdateFailsAtTheDeviceTrainer(_TokenTrainer):
  """An update that fails on the device, met by the device wait first.

  MaxText's `wait_for_device` is its throttler's `block_until_ready()`, which
  also waits on the update's gradient norm (AI-Hypercomputer/maxtext#5718).
  """

  def wait_for_device(self):
    if self.last_update_token is not None:
      raise RuntimeError("XLA: update 1 failed on device")
    return super().wait_for_device()


class _RestorableTrainer(_Trainer):
  """Also takes the calls a restarted orchestrator brings a worker up with."""

  def compile(self, dummy_data):
    self.calls.append(("compile", dummy_data, {}))

  def restore_checkpoint(self, **kwargs):
    self.calls.append(("restore_checkpoint", None, kwargs))
    return {"step": 3}


class _RecordingWaits:
  """Patches `jax.block_until_ready` in the worker to log each wait."""

  def __init__(self, trainer: _Trainer):
    self._trainer = trainer
    self._real = trainer_worker.jax.block_until_ready
    self._patch = mock.patch.object(
        trainer_worker.jax, "block_until_ready", side_effect=self._wait
    )

  def _wait(self, x):
    self._trainer.calls.append(("block_until_ready", x, {}))
    return self._real(x)

  def __enter__(self):
    self._patch.start()
    return self

  def __exit__(self, *exc):
    self._patch.stop()


def _payload(value: int) -> datatypes.RLTrainerPayload:
  return datatypes.RLTrainerPayload(
      prompt_ids=np.array([[value, value + 1]], dtype=np.int32),
      prompt_mask=np.ones((1, 2), dtype=np.float32),
      completion_ids=np.array([[value + 2, value + 3]], dtype=np.int32),
      completion_mask=np.ones((1, 2), dtype=np.float32),
      advantages=np.array([1.0], dtype=np.float32),
  )


def _request(request_id: str, value: int) -> datatypes.TrainRequest:
  return datatypes.TrainRequest(
      request_id=request_id,
      payload=_payload(value),
      metadata={"batch_id": f"b{value}"},
  )


def _worker(trainer: _Trainer) -> trainer_worker.TrainerWorker:
  worker = trainer_worker.TrainerWorker(
      trainer_factory=lambda: trainer, worker_id="trainer_0"
  )
  worker.initialize()
  return worker


class TrainerWorkerStagingTest(absltest.TestCase):

  def test_staged_micro_batch_trains_exactly_like_fwd_bwd(self):
    direct, staged = _Trainer(), _Trainer()
    request = _request("req-1", 7)
    resp_direct = _worker(direct).fwd_bwd(request, skip_jit=True, foo=3)
    worker = _worker(staged)
    worker.stage_fwd_bwd(request, skip_jit=True, foo=3)
    resp_staged = worker.fwd_bwd_staged("req-1")
    self.assertEqual(len(direct.calls), len(staged.calls))
    for (kind_a, pay_a, kw_a), (kind_b, pay_b, kw_b) in zip(
        direct.calls, staged.calls
    ):
      self.assertEqual(kind_a, kind_b)
      self.assertIs(pay_a, request.payload)
      self.assertIs(pay_b, request.payload)
      self.assertEqual(kw_a, kw_b)
    self.assertEqual(resp_direct.request_id, resp_staged.request_id)
    self.assertEqual(resp_direct.metadata, resp_staged.metadata)

  def test_staging_alone_does_not_touch_the_trainer(self):
    trainer = _Trainer()
    worker = _worker(trainer)
    resp = worker.stage_fwd_bwd(_request("req-1", 1))
    self.assertTrue(resp.metadata["staged"])
    self.assertEqual(resp.request_id, "req-1")
    self.assertEmpty(trainer.calls)
    self.assertEqual(worker.state, WorkerState.READY)

  def test_a_wrong_id_is_refused_and_the_staged_batch_is_kept(self):
    trainer = _Trainer()
    worker = _worker(trainer)
    request = _request("req-1", 1)
    worker.stage_fwd_bwd(request)
    with self.assertRaisesRegex(RuntimeError, "no staged micro-batch"):
      worker.fwd_bwd_staged("req-2")
    self.assertEmpty(trainer.calls)
    worker.fwd_bwd_staged("req-1")
    self.assertIs(trainer.calls[0][1], request.payload)

  def test_a_second_stage_is_refused_rather_than_replacing_the_first(self):
    trainer = _Trainer()
    worker = _worker(trainer)
    first = _request("req-1", 1)
    worker.stage_fwd_bwd(first)
    with self.assertRaisesRegex(RuntimeError, "already holds"):
      worker.stage_fwd_bwd(_request("req-2", 2))
    worker.fwd_bwd_staged("req-1")
    self.assertIs(trainer.calls[0][1], first.payload)

  def test_running_without_staging_is_refused(self):
    worker = _worker(_Trainer())
    with self.assertRaisesRegex(RuntimeError, "no staged micro-batch"):
      worker.fwd_bwd_staged("req-1")

  def test_a_staged_batch_runs_once(self):
    worker = _worker(_Trainer())
    worker.stage_fwd_bwd(_request("req-1", 1))
    worker.fwd_bwd_staged("req-1")
    with self.assertRaisesRegex(RuntimeError, "no staged micro-batch"):
      worker.fwd_bwd_staged("req-1")

  def test_staging_is_allowed_during_a_sync_but_running_is_not(self):
    trainer = _Trainer()
    worker = _worker(trainer)
    worker.state = WorkerState.SYNCING
    worker.stage_fwd_bwd(_request("req-1", 1))
    with self.assertRaisesRegex(RuntimeError, "not ready"):
      worker.fwd_bwd_staged("req-1")
    worker.state = WorkerState.READY
    worker.fwd_bwd_staged("req-1")
    self.assertLen(trainer.calls, 1)

  def test_staging_is_refused_on_a_failed_or_stopped_worker(self):
    for state in (WorkerState.ERROR, WorkerState.STOPPED, WorkerState.DRAINING):
      worker = _worker(_Trainer())
      worker.state = state
      with self.assertRaisesRegex(RuntimeError, "cannot stage"):
        worker.stage_fwd_bwd(_request("req-1", 1))

  def test_prepare_fwd_bwd_input_is_what_trains(self):
    trainer = _PreparingTrainer()
    worker = _worker(trainer)
    request = _request("req-1", 1)
    worker.stage_fwd_bwd(request)
    self.assertEqual([c[0] for c in trainer.calls], ["prepare"])
    worker.fwd_bwd_staged("req-1")
    self.assertEqual(trainer.calls[1][1], ("on_device", request.payload))

  def test_a_failed_prepare_marks_the_worker_failed_and_stages_nothing(self):
    worker = _worker(_FailingPrepareTrainer())
    with self.assertRaisesRegex(RuntimeError, "device_put failed"):
      worker.stage_fwd_bwd(_request("req-1", 1))
    self.assertEqual(worker.state, WorkerState.ERROR)
    self.assertIsNone(worker._staged_fwd_bwd)

  def test_stage_reports_how_long_it_waited_for_the_device(self):
    trainer = _WaitingTrainer(waited_s=0.75)
    worker = _worker(trainer)
    resp = worker.stage_fwd_bwd(_request("req-1", 1))
    self.assertEqual(resp.metadata["device_wait_s"], 0.75)
    # The micro-batch is in place first, then the wait: the wait measures the
    # device work left after the transfer.
    self.assertEqual(
        [c[0] for c in trainer.calls], ["prepare", "wait_for_device"]
    )
    # Waiting changes nothing the trainer trains.
    worker.fwd_bwd_staged("req-1")
    self.assertEqual(trainer.calls[-1][0], "fwd_bwd")

  def test_stage_without_wait_for_device_reports_none(self):
    resp = _worker(_Trainer()).stage_fwd_bwd(_request("req-1", 1))
    self.assertIn("device_wait_s", resp.metadata)
    self.assertIsNone(resp.metadata["device_wait_s"])

  def test_a_failed_device_wait_marks_the_worker_failed_and_stages_nothing(
      self,
  ):
    worker = _worker(_FailingWaitTrainer())
    with self.assertRaisesRegex(RuntimeError, "device lost"):
      worker.stage_fwd_bwd(_request("req-1", 1))
    self.assertEqual(worker.state, WorkerState.ERROR)
    self.assertIsNone(worker._staged_fwd_bwd)

  def test_info_advertises_staging(self):
    with_wait = _worker(_WaitingTrainer()).info().resources
    self.assertTrue(with_wait["stage_fwd_bwd"])
    self.assertTrue(with_wait["reports_device_wait"])
    self.assertTrue(with_wait["stage_before_update"])
    without = _worker(_Trainer()).info().resources
    self.assertTrue(without["stage_fwd_bwd"])
    self.assertFalse(without["reports_device_wait"])
    self.assertTrue(without["stage_before_update"])

  def test_stop_drops_the_staged_batch(self):
    worker = _worker(_Trainer())
    worker.stage_fwd_bwd(_request("req-1", 1))
    worker.stop()
    self.assertIsNone(worker._staged_fwd_bwd)

  def test_staging_through_the_rpc_layer(self):
    trainer = _Trainer()
    handle = remote_execution.InProcessActorHandle(
        remote_execution.InProcessRemoteExecutionServer(_worker(trainer))
    )
    request = _request("req-9", 9)

    async def _run():
      staged = await handle.asubmit("stage_fwd_bwd", request=request)
      ran = await handle.asubmit(
          "fwd_bwd_staged", staged_request_id="req-9", skip_jit=False
      )
      return staged, ran

    staged, ran = asyncio.run(_run())
    self.assertTrue(staged.metadata["staged"])
    self.assertTrue(ran.metadata["queued"])
    self.assertEqual(ran.request_id, "req-9")
    np.testing.assert_array_equal(
        trainer.calls[0][1].completion_ids, request.payload.completion_ids
    )


class FusedUpdateStagingTest(absltest.TestCase):
  """Staging on top of the fused final update (fwd_bwd(apply_optimizer=True))."""

  def test_a_staged_last_micro_batch_runs_the_fused_update_like_fwd_bwd(self):
    for trainer_cls in (_Trainer, _TokenTrainer):
      for apply_optimizer in (False, True):
        direct, staged = trainer_cls(), trainer_cls()
        request = _request("req-1", 7)
        with _RecordingWaits(direct):
          resp_direct = _worker(direct).fwd_bwd(
              request, apply_optimizer=apply_optimizer, foo=3
          )
        worker = _worker(staged)
        worker.stage_fwd_bwd(request, foo=3)
        # Only what fwd_bwd_staged does is compared.
        del staged.calls[:]
        with _RecordingWaits(staged):
          resp_staged = worker.fwd_bwd_staged(
              "req-1", apply_optimizer=apply_optimizer
          )
        label = (trainer_cls.__name__, apply_optimizer)
        self.assertEqual(
            [(c[0], c[2]) for c in direct.calls],
            [(c[0], c[2]) for c in staged.calls],
            label,
        )
        self.assertEqual(resp_direct.metadata, resp_staged.metadata, label)
        self.assertEqual(resp_direct.request_id, resp_staged.request_id)
        self.assertEqual(
            bool(resp_staged.metadata.get("updated")), apply_optimizer, label
        )

  def test_wait_for_update_false_returns_at_dispatch_and_claims_nothing(self):
    trainer = _TokenTrainer()
    worker = _worker(trainer)
    with _RecordingWaits(trainer):
      resp = worker.fwd_bwd(
          _request("req-1", 1), apply_optimizer=True, wait_for_update=False
      )
    self.assertEqual([c[0] for c in trainer.calls], ["fwd_bwd", "update"])
    self.assertTrue(resp.metadata["updated"])
    self.assertEqual(resp.metadata["train_step"], 1)
    self.assertNotIn("update_ready", resp.metadata)
    # It is the worker's option, never the trainer's.
    self.assertEqual(trainer.calls[0][2], {"apply_optimizer": True})

  def test_the_default_still_waits_for_the_update(self):
    # Control for the test above: the same call without the option.
    trainer = _TokenTrainer()
    with _RecordingWaits(trainer):
      resp = _worker(trainer).fwd_bwd(
          _request("req-1", 1), apply_optimizer=True
      )
    self.assertEqual(
        [c[0] for c in trainer.calls],
        ["fwd_bwd", "update", "block_until_ready"],
    )
    self.assertTrue(resp.metadata["update_ready"])

  def test_wait_for_update_reaches_fwd_bwd_through_a_staged_call(self):
    trainer = _TokenTrainer()
    worker = _worker(trainer)
    worker.stage_fwd_bwd(_request("req-1", 1))
    del trainer.calls[:]
    with _RecordingWaits(trainer):
      resp = worker.fwd_bwd_staged(
          "req-1", apply_optimizer=True, wait_for_update=False
      )
    self.assertEqual([c[0] for c in trainer.calls], ["fwd_bwd", "update"])
    self.assertNotIn("update_ready", resp.metadata)

  def test_stage_confirms_the_update_after_its_device_wait(self):
    trainer = _TokenTrainer()
    worker = _worker(trainer)
    worker.fwd_bwd(
        _request("req-0", 0), apply_optimizer=True, wait_for_update=False
    )
    token = trainer.last_update_token
    del trainer.calls[:]
    with _RecordingWaits(trainer):
      resp = worker.stage_fwd_bwd(_request("req-1", 1))
    self.assertEqual(
        [c[0] for c in trainer.calls],
        ["prepare", "wait_for_device", "block_until_ready"],
    )
    self.assertIs(trainer.calls[2][1], token)
    self.assertTrue(resp.metadata["update_ready"])
    self.assertEqual(resp.metadata["device_wait_s"], 0.75)

  def test_stage_without_an_update_token_claims_nothing(self):
    trainer = _WaitingTrainer()
    worker = _worker(trainer)
    with _RecordingWaits(trainer):
      resp = worker.stage_fwd_bwd(_request("req-1", 1))
    self.assertNotIn("update_ready", resp.metadata)
    self.assertNotIn("block_until_ready", [c[0] for c in trainer.calls])

  def test_caller_metadata_cannot_claim_for_the_worker(self):
    request = datatypes.TrainRequest(
        request_id="req-1",
        payload=_payload(1),
        metadata={
            "batch_id": "b1",
            "update_ready": True,
            "device_wait_s": 9.0,
            "staged": "caller",
        },
    )
    resp = _worker(_Trainer()).stage_fwd_bwd(request)
    self.assertNotIn("update_ready", resp.metadata)
    self.assertIsNone(resp.metadata["device_wait_s"])
    self.assertIs(resp.metadata["staged"], True)
    self.assertEqual(resp.metadata["batch_id"], "b1")

  def test_a_failed_update_wait_marks_the_worker_failed_and_stages_nothing(
      self,
  ):
    trainer = _TokenTrainer()
    worker = _worker(trainer)
    worker.fwd_bwd(
        _request("req-0", 0), apply_optimizer=True, wait_for_update=False
    )
    with mock.patch.object(
        trainer_worker.jax,
        "block_until_ready",
        side_effect=ValueError("update failed on device"),
    ):
      # Reported as the optimizer step, as the fused call reports it, in the
      # error and in the worker's health report.
      with self.assertRaisesRegex(
          RuntimeError,
          r"^optimizer update\(\) failed: .*stage_fwd_bwd.*the update\b.*"
          r"ValueError: update failed on device",
      ) as raised:
        worker.stage_fwd_bwd(_request("req-1", 1))
    self.assertIsInstance(raised.exception.__cause__, ValueError)
    self.assertEqual(worker.state, WorkerState.ERROR)
    self.assertIn("optimizer update() failed", worker.heartbeat().last_error)
    self.assertIsNone(worker._staged_fwd_bwd)

  def test_a_failed_update_met_by_the_device_wait_is_reported_as_the_update(
      self,
  ):
    trainer = _UpdateFailsAtTheDeviceTrainer()
    worker = _worker(trainer)
    worker.fwd_bwd(
        _request("req-0", 0), apply_optimizer=True, wait_for_update=False
    )
    with self.assertRaisesRegex(
        RuntimeError,
        r"^optimizer update\(\) failed: .*stage_fwd_bwd.*the device\b.*"
        r"RuntimeError: XLA: update 1 failed on device",
    ) as raised:
      worker.stage_fwd_bwd(_request("req-1", 1))
    self.assertIsInstance(raised.exception.__cause__, RuntimeError)
    self.assertEqual(worker.state, WorkerState.ERROR)
    self.assertIn("optimizer update() failed", worker.heartbeat().last_error)
    self.assertIsNone(worker._staged_fwd_bwd)

  def test_a_failed_prepare_is_not_reported_as_the_update(self):
    # Control: only the waits for the work dispatched before the stage are
    # the update's; preparing the stage's own input is the stage's.
    worker = _worker(_FailingPrepareTrainer())
    with self.assertRaisesRegex(RuntimeError, "^device_put failed$"):
      worker.stage_fwd_bwd(_request("req-1", 1))
    self.assertEqual(worker.heartbeat().last_error, "device_put failed")


class StaleStageTest(absltest.TestCase):
  """A stage left by an orchestrator that died without stop() (OOM kill,
  preemption) is dropped when a successor brings the worker up again."""

  def _held(self) -> trainer_worker.TrainerWorker:
    worker = _worker(_RestorableTrainer())
    worker.stage_fwd_bwd(_request("dead-orchestrator", 1))
    return worker

  def test_initialize_alone_keeps_the_held_stage(self):
    # Control: a held stage is refused a successor, and initialize() (a no-op
    # on a ready worker) does not change that.
    worker = self._held()
    worker.initialize()
    with self.assertRaisesRegex(
        RuntimeError, "already holds staged micro-batch 'dead-orchestrator'"
    ):
      worker.stage_fwd_bwd(_request("req-2", 2))

  def test_compile_drops_the_held_stage(self):
    worker = self._held()
    with self.assertLogs(level="WARNING") as logs:
      worker.compile(None)
    self.assertIsNone(worker._staged_fwd_bwd)
    self.assertIn("dead-orchestrator", logs.output[0])
    self.assertTrue(
        worker.stage_fwd_bwd(_request("req-2", 2)).metadata["staged"]
    )

  def test_restore_checkpoint_drops_the_held_stage(self):
    worker = self._held()
    with self.assertLogs(level="WARNING") as logs:
      self.assertEqual(worker.restore_checkpoint(), {"step": 3})
    self.assertIsNone(worker._staged_fwd_bwd)
    self.assertIn("dead-orchestrator", logs.output[0])
    self.assertTrue(
        worker.stage_fwd_bwd(_request("req-2", 2)).metadata["staged"]
    )

  def test_a_dropped_stage_cannot_be_run(self):
    worker = self._held()
    worker.restore_checkpoint()
    with self.assertRaisesRegex(RuntimeError, "no staged micro-batch"):
      worker.fwd_bwd_staged("dead-orchestrator")


if __name__ == "__main__":
  absltest.main()
