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

"""Tests for TrainerWorker's deferred update.

`fwd_bwd(apply_optimizer=True, defer_update=True)` runs a step's last
micro-batch and leaves its update to `apply_deferred_update`, so the next
step's first micro-batch can be staged in between. The trainer must see what
a fused call shows it (the micro-batch marked last, then one update), the
worker must accept nothing that needs the step closed while the update is
deferred, a stage in between must not wait for the device, and the update
call must wait for the device before it dispatches the update.
"""

from absl.testing import absltest
from absl.testing import parameterized
import trainer_worker_staging_test as st
import types
from tunix.experimental.common import datatypes
from tunix.experimental.worker import trainer_worker

WorkerState = datatypes.WorkerState


def _kinds(trainer: st._Trainer) -> list[str]:
  return [c[0] for c in trainer.calls]


def _deferred(trainer: st._Trainer) -> trainer_worker.TrainerWorker:
  worker = st._worker(trainer)
  resp = worker.fwd_bwd(
      st._request("req-1", 1), apply_optimizer=True, defer_update=True
  )
  assert resp.metadata.get("update_deferred"), resp.metadata
  return worker


class _FailingUpdateTrainer(st._WaitingTrainer):

  def update(self, **kwargs):
    raise RuntimeError("XLA: update failed on device")


class DeferredUpdateTest(parameterized.TestCase):

  def test_defer_runs_the_last_micro_batch_and_holds_the_update(self):
    trainer = st._TokenTrainer()
    worker = st._worker(trainer)
    request = st._request("req-1", 1)
    resp = worker.fwd_bwd(request, apply_optimizer=True, defer_update=True)
    self.assertEqual(
        trainer.calls, [("fwd_bwd", request.payload, {"apply_optimizer": True})]
    )
    self.assertTrue(resp.metadata["update_deferred"])
    for key in ("updated", "train_step", "update_ready"):
      self.assertNotIn(key, resp.metadata)
    self.assertEqual(resp.request_id, "req-1")
    self.assertEqual(worker._deferred_update, "req-1")
    self.assertEqual(worker.state, WorkerState.READY)

  def test_without_defer_the_call_is_the_fused_one(self):
    trainer = st._TokenTrainer()
    resp = st._worker(trainer).fwd_bwd(
        st._request("req-1", 1), apply_optimizer=True
    )
    self.assertEqual(_kinds(trainer), ["fwd_bwd", "update"])
    self.assertTrue(resp.metadata["updated"])
    self.assertNotIn("update_deferred", resp.metadata)

  def test_a_caller_cannot_spoof_the_deferral(self):
    trainer = st._Trainer()
    request = st._request("req-1", 1)
    request.metadata["update_deferred"] = True
    resp = st._worker(trainer).fwd_bwd(request, apply_optimizer=True)
    self.assertNotIn("update_deferred", resp.metadata)
    self.assertEqual(_kinds(trainer), ["fwd_bwd", "update"])

  def test_defer_needs_apply_optimizer(self):
    trainer = st._Trainer()
    worker = st._worker(trainer)
    with self.assertRaisesRegex(ValueError, "needs apply_optimizer=True"):
      worker.fwd_bwd(st._request("req-1", 1), defer_update=True)
    self.assertEmpty(trainer.calls)
    self.assertEqual(worker.state, WorkerState.READY)
    self.assertIsNone(worker._deferred_update)

  @parameterized.named_parameters(
      ("fwd_bwd", lambda w: w.fwd_bwd(st._request("req-2", 2))),
      ("fused_fwd_bwd", lambda w: w.fwd_bwd(st._request("req-2", 2), True)),
      ("update", lambda w: w.update()),
      ("get_metrics", lambda w: w.get_metrics()),
      ("save_checkpoint", lambda w: w.save_checkpoint({"step": 1})),
      ("prepare_weight_sync", lambda w: w.prepare_weight_sync()),
      (
          "per_token_logps",
          lambda w: w.per_token_logps(
              types.SimpleNamespace(pad_id=0, eos_id=0, request_id="lp")
          ),
      ),
      ("eval_step", lambda w: w.eval_step(st._request("req-2", 2))),
      ("run_eval", lambda w: w.run_eval([st._payload(2)])),
  )
  def test_while_deferred_the_step_closing_calls_are_refused(self, call):
    trainer = st._TokenTrainer()
    worker = _deferred(trainer)
    before = list(trainer.calls)
    with self.assertRaisesRegex(RuntimeError, "is deferred"):
      call(worker)
    self.assertEqual(trainer.calls, before)
    self.assertEqual(worker.state, WorkerState.READY)
    self.assertEqual(worker._deferred_update, "req-1")

  def test_after_the_deferred_update_the_next_micro_batch_is_accepted(self):
    # Two-sided for the refusals above.
    trainer = st._TokenTrainer()
    worker = _deferred(trainer)
    worker.apply_deferred_update()
    worker.fwd_bwd(st._request("req-2", 2))
    self.assertEqual(
        _kinds(trainer), ["fwd_bwd", "wait_for_device", "update", "fwd_bwd"]
    )

  def test_a_stage_in_between_waits_for_nothing(self):
    trainer = st._TokenTrainer()
    worker = _deferred(trainer)
    with st._RecordingWaits(trainer):
      resp = worker.stage_fwd_bwd(st._request("req-2", 2))
    self.assertEqual(_kinds(trainer), ["fwd_bwd", "prepare"])
    self.assertTrue(resp.metadata["staged"])
    self.assertTrue(resp.metadata["before_update"])
    self.assertIsNone(resp.metadata["device_wait_s"])
    self.assertNotIn("update_ready", resp.metadata)
    self.assertEqual(worker._staged_fwd_bwd[0], "req-2")

  def test_a_stage_after_a_fused_call_still_waits_and_confirms(self):
    # Control: the hook after the last call is unchanged.
    trainer = st._TokenTrainer()
    worker = st._worker(trainer)
    worker.fwd_bwd(
        st._request("req-1", 1), apply_optimizer=True, wait_for_update=False
    )
    with st._RecordingWaits(trainer):
      resp = worker.stage_fwd_bwd(st._request("req-2", 2))
    self.assertEqual(
        _kinds(trainer),
        [
            "fwd_bwd",
            "update",
            "prepare",
            "wait_for_device",
            "block_until_ready",
        ],
    )
    self.assertNotIn("before_update", resp.metadata)
    self.assertEqual(resp.metadata["device_wait_s"], 0.75)
    self.assertTrue(resp.metadata["update_ready"])

  def test_a_caller_cannot_spoof_before_update(self):
    trainer = st._WaitingTrainer()
    worker = st._worker(trainer)
    request = st._request("req-2", 2)
    request.metadata["before_update"] = True
    resp = worker.stage_fwd_bwd(request)
    self.assertNotIn("before_update", resp.metadata)
    self.assertEqual(resp.metadata["device_wait_s"], 0.75)

  def test_the_update_waits_for_the_device_first_then_confirms(self):
    trainer = st._TokenTrainer(waited_s=4.9)
    worker = _deferred(trainer)
    worker.stage_fwd_bwd(st._request("req-2", 2))
    with st._RecordingWaits(trainer):
      resp = worker.apply_deferred_update()
    self.assertEqual(
        _kinds(trainer),
        [
            "fwd_bwd",
            "prepare",
            "wait_for_device",
            "update",
            "block_until_ready",
        ],
    )
    self.assertIs(trainer.calls[-1][1], trainer.last_update_token)
    self.assertTrue(resp.metadata["updated"])
    self.assertEqual(resp.metadata["train_step"], 1)
    self.assertEqual(resp.metadata["device_wait_s"], 4.9)
    self.assertTrue(resp.metadata["update_ready"])
    self.assertIsNone(worker._deferred_update)
    # The staged micro-batch is untouched and runs next.
    self.assertEqual(worker._staged_fwd_bwd[0], "req-2")
    worker.fwd_bwd_staged("req-2")
    self.assertEqual(trainer.calls[-1][0], "fwd_bwd")
    self.assertEqual(trainer.calls[-1][1][0], "on_device")

  def test_the_update_without_a_token_is_not_confirmed(self):
    trainer = st._WaitingTrainer()
    resp = _deferred(trainer).apply_deferred_update()
    self.assertEqual(_kinds(trainer), ["fwd_bwd", "wait_for_device", "update"])
    self.assertNotIn("update_ready", resp.metadata)
    self.assertEqual(resp.metadata["device_wait_s"], 0.75)

  def test_the_update_without_wait_for_device_reports_none(self):
    trainer = st._Trainer()
    resp = _deferred(trainer).apply_deferred_update()
    self.assertEqual(_kinds(trainer), ["fwd_bwd", "update"])
    self.assertIsNone(resp.metadata["device_wait_s"])

  def test_nothing_deferred_is_refused(self):
    trainer = st._WaitingTrainer()
    worker = st._worker(trainer)
    with self.assertRaisesRegex(RuntimeError, "holds no deferred update"):
      worker.apply_deferred_update()
    self.assertEmpty(trainer.calls)
    self.assertEqual(worker.state, WorkerState.READY)

  def test_a_second_call_is_refused(self):
    trainer = st._WaitingTrainer()
    worker = _deferred(trainer)
    worker.apply_deferred_update()
    with self.assertRaisesRegex(RuntimeError, "holds no deferred update"):
      worker.apply_deferred_update()
    self.assertEqual(_kinds(trainer).count("update"), 1)

  def test_a_failed_device_wait_is_the_deferred_updates_failure(self):
    trainer = st._FailingWaitTrainer()
    worker = _deferred(trainer)
    with self.assertRaisesRegex(
        RuntimeError,
        r"deferred optimizer update failed.*waiting for the device:"
        r" RuntimeError: device lost",
    ):
      worker.apply_deferred_update()
    self.assertEqual(worker.state, WorkerState.ERROR)
    self.assertIn(
        "deferred optimizer update failed", worker.heartbeat().last_error
    )
    self.assertNotIn("update", _kinds(trainer))

  def test_a_failed_update_is_the_deferred_updates_failure(self):
    trainer = _FailingUpdateTrainer()
    worker = _deferred(trainer)
    with self.assertRaisesRegex(
        RuntimeError,
        r"deferred optimizer update failed.*waiting for the update:"
        r" RuntimeError: XLA: update failed on device",
    ):
      worker.apply_deferred_update()
    self.assertEqual(worker.state, WorkerState.ERROR)

  @parameterized.named_parameters(
      ("compile", lambda w: w.compile(None)),
      ("restore_checkpoint", lambda w: w.restore_checkpoint()),
  )
  def test_a_successors_bring_up_drops_a_deferred_update(self, bring_up):
    trainer = st._RestorableTrainer()
    worker = st._worker(trainer)
    worker.fwd_bwd(
        st._request("req-1", 1), apply_optimizer=True, defer_update=True
    )
    with self.assertLogs(level="WARNING") as logs:
      bring_up(worker)
    self.assertIsNone(worker._deferred_update)
    self.assertTrue(any("never applied" in m for m in logs.output), logs.output)
    worker.fwd_bwd(st._request("req-2", 2))
    self.assertEqual(_kinds(trainer)[-1], "fwd_bwd")

  def test_initialize_keeps_a_deferred_update(self):
    # Control: only a bring-up drops it.
    worker = _deferred(st._Trainer())
    worker.initialize()
    self.assertEqual(worker._deferred_update, "req-1")

  def test_stop_drops_a_deferred_update(self):
    worker = _deferred(st._Trainer())
    worker.stop()
    self.assertIsNone(worker._deferred_update)

  def test_info_advertises_it(self):
    self.assertTrue(
        st._worker(st._Trainer()).info().resources["stage_before_update"]
    )

  @parameterized.parameters(st._PreparingTrainer, st._TokenTrainer)
  def test_staged_before_the_update_trains_like_the_fused_sequence(
      self, trainer_cls
  ):
    first, second = st._request("req-1", 1), st._request("req-2", 2)
    fused = trainer_cls()
    worker = st._worker(fused)
    worker.fwd_bwd(first, apply_optimizer=True, wait_for_update=False)
    worker.stage_fwd_bwd(second)
    worker.fwd_bwd_staged("req-2")
    deferred = trainer_cls()
    worker = st._worker(deferred)
    worker.fwd_bwd(first, apply_optimizer=True, defer_update=True)
    worker.stage_fwd_bwd(second)
    worker.apply_deferred_update()
    worker.fwd_bwd_staged("req-2")
    trained = lambda t: [c for c in t.calls if c[0] in ("fwd_bwd", "update")]
    self.assertEqual(trained(fused), trained(deferred))
    self.assertEqual(
        [c[0] for c in trained(deferred)], ["fwd_bwd", "update", "fwd_bwd"]
    )


if __name__ == "__main__":
  absltest.main()
