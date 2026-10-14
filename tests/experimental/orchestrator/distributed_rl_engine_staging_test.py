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

"""Tests for DistributedRLEngine.stage_train_step and staged train_step."""

import asyncio
import dataclasses
from typing import Any
from unittest import mock

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import trainer_worker


class _RecordingHandle(remote_execution.ActorHandle):
  """Records every RPC; answers like a TrainerWorker would.

  `fused` answers a train call with `apply_optimizer=True` as a worker that
  ran the update inside it (`updated`, `train_step` 7 and `update_ready`);
  otherwise it answers as a worker without the fused update, so the engine
  sends `update` itself.
  """

  def __init__(
      self,
      stage_metadata: dict[str, Any] | None = None,
      fused: bool = False,
  ):
    self.calls: list[tuple[str, dict[str, Any]]] = []
    self._stage_metadata = stage_metadata or {"staged": True}
    self._fused = fused

  async def asubmit(self, method_name=None, *args, **kwargs):
    del args
    self.calls.append((method_name, kwargs))
    if method_name == "update":
      return 5
    if method_name == "stage_fwd_bwd":
      return datatypes.Response(request_id="r", metadata=self._stage_metadata)
    if self._fused and kwargs.get("apply_optimizer"):
      return datatypes.Response(
          request_id="r",
          metadata={
              "queued": True,
              "updated": True,
              "train_step": 7,
              "update_ready": True,
          },
      )
    return datatypes.Response(request_id="r", metadata={"queued": True})

  def submit(self, method_name=None, *args, **kwargs):
    raise NotImplementedError

  async def dispatch_task(self, request_id=None, method_name=None, *a, **k):
    raise NotImplementedError

  async def poll_responses(self, timeout_s=0.0):
    raise NotImplementedError


def _payload() -> datatypes.RLTrainerPayload:
  return datatypes.RLTrainerPayload(
      prompt_ids=np.array([[1, 2]], dtype=np.int32),
      prompt_mask=np.ones((1, 2), dtype=np.float32),
      completion_ids=np.array([[3, 4]], dtype=np.int32),
      completion_mask=np.ones((1, 2), dtype=np.float32),
      advantages=np.array([1.0], dtype=np.float32),
      metadata={"traj": "t0"},
  )


def _engine(handle) -> distributed_rl_engine.DistributedRLEngine:
  return distributed_rl_engine.DistributedRLEngine(
      rollout_workers=[], trainer_workers={datatypes.Role.ACTOR: handle}
  )


class StagedTrainStepTest(absltest.TestCase):

  def test_stage_sends_the_payload_once_and_returns_its_id(self):
    handle = _RecordingHandle()
    payload = _payload()
    staged = asyncio.run(_engine(handle).stage_train_step(payload))
    self.assertLen(handle.calls, 1)
    method, kwargs = handle.calls[0]
    self.assertEqual(method, "stage_fwd_bwd")
    self.assertIs(kwargs["request"].payload, payload)
    self.assertEqual(kwargs["request"].request_id, staged.request_id)
    self.assertEqual(kwargs["request"].metadata, {"traj": "t0"})
    # The trainer said nothing about its device.
    self.assertIsNone(staged.device_wait_s)

  def test_stage_reports_the_trainers_device_wait(self):
    handle = _RecordingHandle({"staged": True, "device_wait_s": 1.25})
    staged = asyncio.run(_engine(handle).stage_train_step(_payload()))
    self.assertEqual(staged.device_wait_s, 1.25)
    self.assertIsInstance(staged, distributed_rl_engine.StagedTrainStep)
    self.assertFalse(staged.update_ready)

  def test_stage_reports_the_trainers_update_confirmation(self):
    handle = _RecordingHandle(
        {"staged": True, "device_wait_s": 1.25, "update_ready": True}
    )
    staged = asyncio.run(_engine(handle).stage_train_step(_payload()))
    self.assertTrue(staged.update_ready)

  def test_staged_train_step_without_a_fused_worker_sends_the_update(self):
    # A worker that did not run the update inside the call (no `updated`):
    # the engine sends `update` itself, as for an unstaged call.
    handle = _RecordingHandle()
    engine = _engine(handle)
    result = asyncio.run(
        engine.train_step(
            None,
            accumulate_gradients=True,
            apply_optimizer=True,
            staged_request_id="train_req_x",
        )
    )
    self.assertEqual(
        handle.calls,
        [
            (
                "fwd_bwd_staged",
                {
                    "staged_request_id": "train_req_x",
                    "skip_jit": False,
                    "apply_optimizer": True,
                },
            ),
            ("update", {}),
        ],
    )
    self.assertTrue(result["updated"])
    self.assertEqual(result["train_step"], 5)
    self.assertFalse(result["update_ready"])
    self.assertTrue(engine._actor_weights_dirty)

  def test_staged_train_step_takes_the_fused_update(self):
    # A worker that ran the update inside the call: no second update RPC,
    # and its train_step and update_ready are passed on, as unstaged.
    handle = _RecordingHandle(fused=True)
    result = asyncio.run(
        _engine(handle).train_step(
            None, apply_optimizer=True, staged_request_id="train_req_x"
        )
    )
    self.assertEqual([c[0] for c in handle.calls], ["fwd_bwd_staged"])
    self.assertEqual(result["train_step"], 7)
    self.assertTrue(result["update_ready"])

  def test_staged_and_unstaged_results_have_the_same_shape(self):
    for fused in (False, True):
      staged = asyncio.run(
          _engine(_RecordingHandle(fused=fused)).train_step(
              None, apply_optimizer=True, staged_request_id="train_req_x"
          )
      )
      unstaged = asyncio.run(
          _engine(_RecordingHandle(fused=fused)).train_step(
              _payload(), apply_optimizer=True
          )
      )
      self.assertEqual(sorted(staged), sorted(unstaged))
      for key in ("updated", "train_step", "accumulated", "update_ready"):
        self.assertEqual(staged[key], unstaged[key], (fused, key))

  def test_staged_train_step_without_the_update(self):
    handle = _RecordingHandle()
    asyncio.run(
        _engine(handle).train_step(
            None, apply_optimizer=False, staged_request_id="train_req_x"
        )
    )
    self.assertEqual([c[0] for c in handle.calls], ["fwd_bwd_staged"])
    # The trainer still learns it is not the step's last micro-batch.
    self.assertIs(handle.calls[0][1]["apply_optimizer"], False)

  def test_a_payload_and_a_staged_id_together_are_refused(self):
    handle = _RecordingHandle()
    with self.assertRaisesRegex(ValueError, "not both"):
      asyncio.run(_engine(handle).train_step(_payload(), staged_request_id="x"))
    self.assertEmpty(handle.calls)

  def test_unstaged_train_step_is_unchanged(self):
    handle = _RecordingHandle()
    payload = _payload()
    asyncio.run(_engine(handle).train_step(payload, apply_optimizer=True))
    self.assertEqual([c[0] for c in handle.calls], ["fwd_bwd", "update"])
    self.assertEqual(
        sorted(handle.calls[0][1]), ["apply_optimizer", "request", "skip_jit"]
    )
    self.assertIs(handle.calls[0][1]["request"].payload, payload)

  def test_wait_for_update_reaches_the_worker_only_when_given(self):
    for staged_request_id, payload in ((None, _payload()), ("x", None)):
      default = _RecordingHandle(fused=True)
      asyncio.run(
          _engine(default).train_step(
              payload,
              apply_optimizer=True,
              staged_request_id=staged_request_id,
          )
      )
      self.assertNotIn("wait_for_update", default.calls[0][1])
      at_dispatch = _RecordingHandle(fused=True)
      asyncio.run(
          _engine(at_dispatch).train_step(
              payload,
              apply_optimizer=True,
              staged_request_id=staged_request_id,
              wait_for_update=False,
          )
      )
      self.assertIs(at_dispatch.calls[0][1]["wait_for_update"], False)

  def test_stage_then_train_reaches_a_real_trainer_worker(self):

    class _Trainer:
      policy_version = 0

      def __init__(self):
        self.trained = []
        self.fwd_bwd_kwargs = []
        self.updates = 0

      def fwd_bwd(self, payload, **kwargs):
        self.trained.append(payload)
        self.fwd_bwd_kwargs.append(kwargs)

      def update(self, **kwargs):
        self.updates += 1
        return 1

      def close(self):
        pass

    trainer = _Trainer()
    worker = trainer_worker.TrainerWorker(trainer_factory=lambda: trainer)
    worker.initialize()
    handle = remote_execution.InProcessActorHandle(
        remote_execution.InProcessRemoteExecutionServer(worker)
    )
    engine = _engine(handle)
    payload = _payload()

    async def _run():
      staged = await engine.stage_train_step(payload)
      self.assertEmpty(trainer.trained)
      self.assertIsNone(staged.device_wait_s)  # no wait_for_device
      self.assertFalse(staged.update_ready)  # no last_update_token
      return await engine.train_step(
          None, apply_optimizer=True, staged_request_id=staged.request_id
      )

    result = asyncio.run(_run())
    self.assertEqual(result["train_step"], 1)
    self.assertLen(trainer.trained, 1)
    self.assertIs(trainer.trained[0], payload)
    # The worker ran the update inside the staged call, exactly once, and the
    # trainer saw the step's-last-micro-batch flag as for an unstaged call.
    self.assertEqual(trainer.updates, 1)
    self.assertEqual(trainer.fwd_bwd_kwargs, [{"apply_optimizer": True}])

  def test_fused_call_at_dispatch_then_stage_confirms_the_update(self):
    """`wait_for_update=False` moves the update wait from the call to the stage.

    The trainer publishes an update token. The step's last call, sent with
    `wait_for_update=False`, must return without waiting on it and without
    claiming `update_ready`; the stage right after must wait on it and claim
    it. Control: the same call with the default waits and claims, as before.
    """

    class _TokenTrainer:
      policy_version = 0

      def __init__(self):
        self.last_update_token = None

      def fwd_bwd(self, payload, **kwargs):
        del payload, kwargs

      def update(self, **kwargs):
        del kwargs
        self.last_update_token = jnp.array(2.0)
        return 3

      def close(self):
        pass

    trainer = _TokenTrainer()
    engine = _real_engine(trainer_worker.TrainerWorker, trainer)
    waited = []
    real_block_until_ready = trainer_worker.jax.block_until_ready

    def _recording_block_until_ready(x):
      waited.append(x)
      return real_block_until_ready(x)

    async def _run():
      at_dispatch = await engine.train_step(
          _payload(), apply_optimizer=True, wait_for_update=False
      )
      waits_after_call = len(waited)
      staged = await engine.stage_train_step(_payload())
      waits_after_stage = len(waited)
      await engine.train_step(
          None, apply_optimizer=False, staged_request_id=staged.request_id
      )
      default = await engine.train_step(_payload(), apply_optimizer=True)
      return (
          at_dispatch,
          waits_after_call,
          staged,
          waits_after_stage,
          default,
      )

    with mock.patch.object(
        trainer_worker.jax,
        "block_until_ready",
        side_effect=_recording_block_until_ready,
    ):
      at_dispatch, waits_after_call, staged, waits_after_stage, default = (
          asyncio.run(_run())
      )

    self.assertTrue(at_dispatch["updated"])
    self.assertFalse(at_dispatch["update_ready"])
    self.assertEqual(waits_after_call, 0)
    self.assertTrue(staged.update_ready)
    self.assertEqual(waits_after_stage, 1)
    # Control: the default waits for the update and claims it.
    self.assertTrue(default["update_ready"])
    self.assertLen(waited, 2)


class _MinimalTrainer:
  policy_version = 0

  def fwd_bwd(self, payload, **kwargs):
    del payload, kwargs

  def update(self, **kwargs):
    del kwargs
    return 1

  def close(self):
    pass


class _WaitingTrainer(_MinimalTrainer):

  def wait_for_device(self):
    return 0.5


class _OldTrainerWorker(trainer_worker.TrainerWorker):
  """`info()` as before staging: no capability keys."""

  def info(self):
    info = super().info()
    return dataclasses.replace(
        info,
        resources={
            k: v
            for k, v in info.resources.items()
            if k
            not in (
                "stage_fwd_bwd",
                "reports_device_wait",
                "stage_before_update",
            )
        },
    )


def _real_engine(worker_cls, trainer):
  worker = worker_cls(trainer_factory=lambda: trainer)
  worker.initialize()
  handle = remote_execution.InProcessActorHandle(
      remote_execution.InProcessRemoteExecutionServer(worker)
  )
  return _engine(handle)


class TrainerStagingSupportTest(absltest.TestCase):

  def test_a_trainer_with_wait_for_device_supports_staging(self):
    engine = _real_engine(trainer_worker.TrainerWorker, _WaitingTrainer())
    self.assertEqual(
        asyncio.run(engine.trainer_staging_support()),
        {
            "stage_fwd_bwd": True,
            "reports_device_wait": True,
            "stage_before_update": True,
        },
    )

  def test_a_trainer_without_it_cannot_report_its_device(self):
    engine = _real_engine(trainer_worker.TrainerWorker, _MinimalTrainer())
    self.assertEqual(
        asyncio.run(engine.trainer_staging_support()),
        {
            "stage_fwd_bwd": True,
            "reports_device_wait": False,
            "stage_before_update": True,
        },
    )

  def test_an_older_trainer_worker_supports_nothing(self):
    engine = _real_engine(_OldTrainerWorker, _WaitingTrainer())
    self.assertEqual(
        asyncio.run(engine.trainer_staging_support()),
        {
            "stage_fwd_bwd": False,
            "reports_device_wait": False,
            "stage_before_update": False,
        },
    )

  def test_the_device_wait_comes_back_through_a_real_worker(self):
    engine = _real_engine(trainer_worker.TrainerWorker, _WaitingTrainer())
    staged = asyncio.run(engine.stage_train_step(_payload()))
    self.assertEqual(staged.device_wait_s, 0.5)

  def test_no_trainer_for_the_role_is_an_error(self):
    engine = distributed_rl_engine.DistributedRLEngine(
        rollout_workers=[], trainer_workers={}
    )
    with self.assertRaisesRegex(ValueError, "No trainer worker"):
      asyncio.run(engine.trainer_staging_support())


if __name__ == "__main__":
  absltest.main()
