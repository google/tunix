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

"""Tests for `TrainerWorker.stage_batch`, the RPC server's executor hook and the
orchestrator's `stage_train_batch` (TUNIX_STAGE_BATCH=stage)."""

import asyncio
import contextlib
import os
import threading
from typing import Any
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.train import abstract_trainer
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import trainer_worker


class FakeTrainer(abstract_trainer.AbstractTrainer):
  """The parts of trainer_worker_test's FakeTrainer these tests touch."""

  def __init__(self):
    self.fwd_bwd_calls = []
    self.policy_version = 3
    self.step_count = 10

  @property
  def checkpoint_dir(self) -> str | None:
    return None

  def compile(self, dummy_data=None):
    pass

  def with_loss_fn(self, loss_fn, has_aux=False):
    pass

  def with_gen_model_input_fn(self, gen_model_input_fn):
    pass

  def fwd_bwd(self, payload, **kwargs):
    self.fwd_bwd_calls.append((payload, kwargs))

  def update(self, **kwargs):
    self.step_count += 1
    return self.step_count

  def eval_step(self, payload, **kwargs):
    pass

  @contextlib.contextmanager
  def model_scope(self, *args, **kwargs):
    yield None, args, kwargs

  def save_checkpoint(self, metadata, **kwargs):
    pass

  def restore_checkpoint(self, **kwargs):
    return {}

  def get_metrics(self):
    return {}

  def prepare_weight_sync(self, **kwargs):
    pass

  def close(self):
    pass


class StagingFakeTrainer(FakeTrainer):
  """FakeTrainer with the engine's `stage_batch(payload, key=...)` hook."""

  def __init__(self, staged: bool = True):
    super().__init__()
    self.stage_calls = []
    self.staged = staged

  def stage_batch(self, payload, key=None):
    self.stage_calls.append((payload, key, threading.current_thread().name))
    return self.staged


def _payload():
  return datatypes.RLTrainerPayload(
      prompt_ids=np.array([[1, 2]], dtype=np.int32),
      prompt_mask=np.ones((1, 2), dtype=np.float32),
      completion_ids=np.array([[3, 4]], dtype=np.int32),
      completion_mask=np.ones((1, 2), dtype=np.float32),
      advantages=np.array([1.0], dtype=np.float32),
  )


class TrainerWorkerStageBatchTest(absltest.TestCase):

  def _worker(self, trainer):
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: trainer, worker_id="trainer_0"
    )
    worker.initialize()
    return worker

  def test_stage_batch_forwards_payload_and_request_id_as_key(self):
    trainer = StagingFakeTrainer()
    worker = self._worker(trainer)
    payload = _payload()
    req = datatypes.TrainRequest(request_id="mb-7", payload=payload)
    resp = worker.stage_batch(request=req)
    self.assertTrue(resp.metadata["staged"])
    self.assertEqual(resp.request_id, "mb-7")
    self.assertLen(trainer.stage_calls, 1)
    self.assertIs(trainer.stage_calls[0][0], payload)
    self.assertEqual(trainer.stage_calls[0][1], "mb-7")
    # Not a step: no fwd_bwd, no state change, worker still ready.
    self.assertEmpty(trainer.fwd_bwd_calls)
    self.assertEqual(worker.state, datatypes.WorkerState.READY)

  def test_stage_batch_without_trainer_support_says_not_staged(self):
    worker = self._worker(FakeTrainer())
    resp = worker.stage_batch(
        request=datatypes.TrainRequest(request_id="x", payload=_payload())
    )
    self.assertFalse(resp.metadata["staged"])
    self.assertIn("reason", resp.metadata)

  def test_stage_batch_trainer_declines(self):
    worker = self._worker(StagingFakeTrainer(staged=False))
    resp = worker.stage_batch(
        request=datatypes.TrainRequest(request_id="x", payload=_payload())
    )
    self.assertFalse(resp.metadata["staged"])

  def test_stage_batch_materializes_compact_routed_experts(self):
    class _CompactRouted:

      def __init__(self, dense):
        self._dense = dense

      def materialize(self):
        return self._dense

    dense = np.arange(8, dtype=np.int32).reshape(1, 2, 2, 2)
    trainer = StagingFakeTrainer()
    worker = self._worker(trainer)
    payload = _payload().replace(routed_experts=_CompactRouted(dense))
    resp = worker.stage_batch(
        request=datatypes.TrainRequest(request_id="mb-compact", payload=payload)
    )
    self.assertTrue(resp.metadata["staged"])
    staged_payload = trainer.stage_calls[0][0]
    self.assertIsInstance(staged_payload.routed_experts, np.ndarray)
    np.testing.assert_array_equal(staged_payload.routed_experts, dense)

  def test_stage_batch_exception_degrades_without_error_state(self):
    class _FailingStageTrainer(StagingFakeTrainer):

      def stage_batch(self, payload, key=None):
        raise RuntimeError("device_put failed")

      def reset_staged(self):
        return 2

    worker = self._worker(_FailingStageTrainer())
    resp = worker.stage_batch(
        request=datatypes.TrainRequest(request_id="mb-err", payload=_payload())
    )
    self.assertFalse(resp.metadata["staged"])
    self.assertIn("device_put failed", resp.metadata["reason"])
    self.assertEqual(worker.state, datatypes.WorkerState.READY)
    reset_resp = worker.reset_staged()
    self.assertTrue(reset_resp.metadata["reset_staged"])
    self.assertEqual(reset_resp.metadata["dropped"], 2)

  def test_fwd_bwd_passes_stage_key_and_no_payload(self):
    trainer = StagingFakeTrainer()
    worker = self._worker(trainer)
    req = datatypes.TrainRequest(request_id="mb-1", payload=None)
    resp = worker.fwd_bwd(request=req, stage_key="mb-1")
    self.assertTrue(resp.metadata["queued"])
    self.assertLen(trainer.fwd_bwd_calls, 1)
    payload, kwargs = trainer.fwd_bwd_calls[0]
    self.assertIsNone(payload)
    self.assertEqual(kwargs["stage_key"], "mb-1")
    self.assertFalse(kwargs["apply_optimizer"])

  def test_executor_methods_default_and_with_env(self):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop(trainer_worker.STAGE_BATCH_ENV, None)
      worker = self._worker(StagingFakeTrainer())
      self.assertEqual(
          worker.remote_executor_methods, frozenset({"stage_batch"})
      )
    with mock.patch.dict(
        os.environ, {trainer_worker.STAGE_BATCH_ENV: "stage"}, clear=False
    ):
      worker = self._worker(StagingFakeTrainer())
      self.assertEqual(
          worker.remote_executor_methods, frozenset({"stage_batch", "fwd_bwd"})
      )
    with mock.patch.dict(
        os.environ, {trainer_worker.STAGE_BATCH_ENV: "serialize"}, clear=False
    ):
      self.assertFalse(trainer_worker.stage_batch_env_on())


class RemoteExecutionExecutorHookTest(absltest.TestCase):

  def test_listed_method_runs_off_the_loop_thread(self):
    trainer = StagingFakeTrainer()
    with mock.patch.dict(
        os.environ, {trainer_worker.STAGE_BATCH_ENV: "stage"}, clear=False
    ):
      worker = trainer_worker.TrainerWorker(
          trainer_factory=lambda: trainer, worker_id="trainer_0"
      )
      worker.initialize()
    server = remote_execution.InProcessRemoteExecutionServer(worker)
    loop_thread = threading.current_thread().name

    async def run():
      stage = remote_execution.ExecutionRequest(
          method_name="stage_batch",
          args=(),
          kwargs={
              "request": datatypes.TrainRequest(
                  request_id="s1", payload=_payload()
              )
          },
          request_id="r-stage",
      )
      info = remote_execution.ExecutionRequest(
          method_name="info", args=(), kwargs={}, request_id="r-info"
      )
      return await server.execute_request(stage), await server.execute_request(
          info
      )

    stage_resp, info_resp = asyncio.run(run())
    self.assertIsNone(stage_resp.error_message, stage_resp.traceback)
    self.assertTrue(stage_resp.result.metadata["staged"])
    self.assertLen(trainer.stage_calls, 1)
    # stage_batch ran on the rpc-method pool, not on the loop's thread; `info`
    # (not listed) ran inline as before.
    self.assertTrue(
        trainer.stage_calls[0][2].startswith("rpc-method"),
        trainer.stage_calls[0][2],
    )
    self.assertNotEqual(trainer.stage_calls[0][2], loop_thread)
    self.assertIsNone(info_resp.error_message)
    self.assertEqual(info_resp.result.worker_id, "trainer_0")
    server.close_executor()

  def test_stage_batch_overlaps_blocked_fwd_bwd_on_executor_pool(self):
    fwd_bwd_entered = threading.Event()
    release_fwd_bwd = threading.Event()

    class _BlockingTrainer(StagingFakeTrainer):

      def fwd_bwd(self, payload, **kwargs):
        fwd_bwd_entered.set()
        release_fwd_bwd.wait(timeout=5.0)
        super().fwd_bwd(payload, **kwargs)

    trainer = _BlockingTrainer()
    with mock.patch.dict(
        os.environ, {trainer_worker.STAGE_BATCH_ENV: "stage"}, clear=False
    ):
      worker = trainer_worker.TrainerWorker(
          trainer_factory=lambda: trainer, worker_id="trainer_0"
      )
      worker.initialize()
    server = remote_execution.InProcessRemoteExecutionServer(worker)

    async def run():
      fwd_req = remote_execution.ExecutionRequest(
          method_name="fwd_bwd",
          args=(),
          kwargs={
              "request": datatypes.TrainRequest(
                  request_id="f1", payload=_payload()
              )
          },
          request_id="r-fwd",
      )
      stage_req = remote_execution.ExecutionRequest(
          method_name="stage_batch",
          args=(),
          kwargs={
              "request": datatypes.TrainRequest(
                  request_id="s2", payload=_payload()
              )
          },
          request_id="r-stage",
      )
      fwd_task = asyncio.create_task(server.execute_request(fwd_req))
      await asyncio.to_thread(fwd_bwd_entered.wait, 5.0)
      stage_resp = await server.execute_request(stage_req)
      self.assertTrue(stage_resp.result.metadata["staged"])
      self.assertFalse(fwd_task.done())
      release_fwd_bwd.set()
      fwd_resp = await fwd_task
      self.assertTrue(fwd_resp.result.metadata["queued"])

    try:
      asyncio.run(run())
    finally:
      server.close_executor()


class _FakeHandle:
  """A trainer handle whose `asubmit` records RPCs like TrainerWorker."""

  def __init__(self, staged=True, has_stage=True, raise_runtime=False):
    self.calls = []
    self.staged = staged
    self.has_stage = has_stage
    self.raise_runtime = raise_runtime

  async def asubmit(self, method_name, **kwargs):
    self.calls.append((method_name, kwargs))
    if method_name == "stage_batch":
      if not self.has_stage:
        raise AttributeError("Method 'stage_batch' not found")
      if self.raise_runtime:
        raise RuntimeError(
            "RemoteExecutionError [AttributeError]: Method 'stage_batch' not"
            " found"
        )
      return datatypes.Response(metadata={"staged": self.staged})
    if method_name == "reset_staged":
      if self.raise_runtime:
        raise RuntimeError("RemoteExecutionError [AttributeError]")
      return datatypes.Response(metadata={"reset_staged": True, "dropped": 1})
    if method_name == "fwd_bwd":
      return datatypes.Response(metadata={"queued": True})
    raise AssertionError(method_name)


class EngineStageTrainBatchTest(absltest.TestCase):

  def _engine(self, handle):
    engine = distributed_rl_engine.DistributedRLEngine.__new__(
        distributed_rl_engine.DistributedRLEngine
    )
    engine._trainer_workers = {datatypes.Role.ACTOR: handle}
    return engine

  def test_stage_returns_key_and_train_step_ships_no_payload(self):
    handle = _FakeHandle()
    engine = self._engine(handle)
    payload = _payload()
    key = asyncio.run(engine.stage_train_batch(payload))
    self.assertIsNotNone(key)
    self.assertEqual(handle.calls[0][0], "stage_batch")
    req = handle.calls[0][1]["request"]
    self.assertEqual(req.request_id, key)
    self.assertIs(req.payload, payload)
    # The train_step for it carries the key and no payload.
    engine._actor_busy_guard = mock.MagicMock()
    engine._actor_busy_guard.return_value.__aenter__ = mock.AsyncMock()
    engine._actor_busy_guard.return_value.__aexit__ = mock.AsyncMock(
        return_value=False
    )
    engine._actor_weights_dirty = False
    asyncio.run(
        engine.train_step(
            None,
            accumulate_gradients=True,
            apply_optimizer=False,
            stage_key=key,
        )
    )
    name, kwargs = handle.calls[1]
    self.assertEqual(name, "fwd_bwd")
    self.assertIsNone(kwargs["request"].payload)
    self.assertEqual(kwargs["stage_key"], key)

  def test_not_staged_or_missing_method_returns_none(self):
    for handle in (
        _FakeHandle(staged=False),
        _FakeHandle(has_stage=False),
        _FakeHandle(raise_runtime=True),
    ):
      engine = self._engine(handle)
      self.assertIsNone(asyncio.run(engine.stage_train_batch(_payload())))
      asyncio.run(engine.reset_staged_batches())


class RLProgramMultiStageAheadTest(absltest.TestCase):
  """Tests StandardRLProgram continuous 2-deep background staging."""

  def test_stages_two_microbatches_ahead_while_train_step_in_flight(self):
    from tunix.experimental.orchestrator import batch_assembly
    from tunix.experimental.orchestrator import rl_program

    events = []
    mb0_release = asyncio.Event()
    mb2_stage_started = asyncio.Event()

    class _FakeAlgo:
      num_generations = 1
      mini_batch_size = 4
      train_micro_batch_size = 1
      max_response_length = 4
      requires_reference_kl = False

      class _Config:
        temperature = 1.0
        use_rollout_logps = False
        truncated_importance_sampling_type = None
        seq_logprob_error_threshold = None
        sampler_is_length_buckets = None
        sampler_is = None
        sampler_is_threshold = 2.0

      algo_config = _Config()

    class _FakeAssembler(batch_assembly.BatchAssembler):

      def __init__(self):
        super().__init__(num_generations=1, mini_batch_size=1)
        self._count = 0

      def feed(self, payloads):
        out = []
        for idx, p in enumerate(payloads):
          self._count += 1
          out.append(
              batch_assembly.AssembledBatch(
                  payload=p,
                  trajectory_ids=(f"t_{self._count}",),
                  is_final_batch=(self._count == 4),
              )
          )
        return out

      def flush(self):
        return []

      def reset(self):
        self._count = 0

    class _FakeEngine:

      def __init__(self):
        self.staged_keys = []
        self.trained_keys = []

      async def stage_train_batch(
          self, payload, role=datatypes.Role.ACTOR, stage_key=None
      ):
        events.append(("stage_start", stage_key))
        self.staged_keys.append(stage_key)
        if len(self.staged_keys) == 2:
          # Second staged batch (MB 2) started while MB 0 is still running!
          mb2_stage_started.set()
          # Release MB 0 while MB 2 is still mid-transfer to verify MB 1
          # dispatches immediately without waiting for MB 2's stage to finish.
          mb0_release.set()
          await asyncio.sleep(0.02)
        events.append(("stage_end", stage_key))
        return stage_key

      async def train_step(
          self,
          payload,
          role=datatypes.Role.ACTOR,
          accumulate_gradients=True,
          apply_optimizer=False,
          stage_key=None,
          **kwargs,
      ):
        events.append(
            ("train_start", stage_key, payload is None, apply_optimizer)
        )
        self.trained_keys.append((stage_key, payload is None, apply_optimizer))
        if len(self.trained_keys) == 1:
          await mb0_release.wait()
        events.append(("train_end", stage_key))
        return {
            "updated": apply_optimizer,
            "train_step": 1 if apply_optimizer else None,
        }

      async def get_metrics(self, role=datatypes.Role.ACTOR):
        return {}

      async def save_checkpoint(
          self, role=datatypes.Role.ACTOR, metadata=None, **kwargs
      ):
        return datatypes.Response(metadata={"checkpoint_path": "/tmp/ckpt"})

    program = rl_program.StandardRLProgram(
        algo=_FakeAlgo(),
        dataset=[{"prompt": f"p{i}"} for i in range(4)],
        max_steps=1,
        batch_size=4,
        sync_weights=False,
        assembler=_FakeAssembler(),
        pipeline_train_microbatches=True,
        stage_train_batches=True,
        max_staged_batches=2,
    )
    engine = _FakeEngine()
    program.engine = engine
    logged_metrics = {}
    orig_log_metric = program._log_metric

    def _capture_metric(name, val, step, prefix=None):
      key = f"{prefix}/{name}" if prefix else name
      logged_metrics[key] = val
      return orig_log_metric(name, val, step, prefix=prefix)

    program._log_metric = _capture_metric

    async def _run():
      for i in range(4):
        item = datatypes.TrajectoryItem(
            prompt_id=f"p_{i}",
            group_index=0,
            start_step=0,
            traj={
                "trajectory_reward": 1.0,
                "status": "SUCCEEDED",
                "steps": [0],
            },
            is_valid=True,
            prompt_tokens=np.array([1, 2], dtype=np.int32),
            completion_tokens=np.array([3, 4], dtype=np.int32),
            policy_version=0,
            metadata={"prompt_idx": i, "batch_idx": 0, "intra_batch_idx": i},
        )
        item.payload = _payload()
        await program.scored_q.put(item)
      await program.scored_q.close()
      await program._train_loop()

    asyncio.run(_run())
    self.assertTrue(mb2_stage_started.is_set())
    self.assertLen(engine.staged_keys, 3)
    self.assertEqual(
        engine.trained_keys,
        [
            (None, False, False),
            ("step_0_stage_1", True, False),
            ("step_0_stage_2", True, False),
            ("step_0_stage_3", True, True),
        ],
    )
    # Verify MB 1 train_start happened BEFORE MB 2 stage_end (proving MB 1
    # dispatched as soon as MB 0 finished, while MB 2 was still streaming).
    idx_mb1_train_start = events.index(
        ("train_start", "step_0_stage_1", True, False)
    )
    idx_mb2_stage_end = events.index(("stage_end", "step_0_stage_2"))
    self.assertLess(idx_mb1_train_start, idx_mb2_stage_end)

  def test_stage_ahead_does_not_block_staged_mb_on_long_tail_group(self):
    """Staged MB 1 dispatches when MB 0 finishes even if MB 2 is slow."""
    from tunix.experimental.orchestrator import batch_assembly
    from tunix.experimental.orchestrator import rl_program

    events: list[tuple] = []
    mb1_staged = asyncio.Event()
    mb1_train_started = asyncio.Event()

    class _FakeAlgo:
      num_generations = 1
      mini_batch_size = 3
      train_micro_batch_size = 1
      max_response_length = 4
      requires_reference_kl = False

      class _Config:
        temperature = 1.0
        use_rollout_logps = False
        truncated_importance_sampling_type = None
        seq_logprob_error_threshold = None
        sampler_is_length_buckets = None
        sampler_is = None
        sampler_is_threshold = 2.0

      algo_config = _Config()

    class _FakeAssembler(batch_assembly.BatchAssembler):

      def __init__(self):
        super().__init__(num_generations=1, mini_batch_size=1)
        self._count = 0

      def feed(self, payloads):
        out = []
        for p in payloads:
          self._count += 1
          out.append(
              batch_assembly.AssembledBatch(
                  payload=p,
                  trajectory_ids=(f"t_{self._count}",),
                  is_final_batch=(self._count == 3),
              )
          )
        return out

      def flush(self):
        return []

      def reset(self):
        self._count = 0

    class _FakeEngine:

      def __init__(self):
        self.staged_keys = []
        self.trained_keys = []

      async def stage_train_batch(
          self, payload, role=datatypes.Role.ACTOR, stage_key=None
      ):
        events.append(("stage", stage_key))
        self.staged_keys.append(stage_key)
        if len(self.staged_keys) == 1:
          mb1_staged.set()
        return stage_key

      async def train_step(
          self,
          payload,
          role=datatypes.Role.ACTOR,
          accumulate_gradients=True,
          apply_optimizer=False,
          stage_key=None,
          **kwargs,
      ):
        events.append(("train_start", stage_key))
        self.trained_keys.append(stage_key)
        if len(self.trained_keys) == 1:
          # MB 0 finishes shortly after MB 1 is staged, while Group 2 (for MB 2)
          # has NOT arrived yet.
          await mb1_staged.wait()
          await asyncio.sleep(0.01)
        elif len(self.trained_keys) == 2:
          # MB 1 started training without waiting for Group 2!
          mb1_train_started.set()
          await asyncio.sleep(0.02)
        events.append(("train_end", stage_key))
        return {
            "updated": apply_optimizer,
            "train_step": 1 if apply_optimizer else None,
        }

      async def get_metrics(self, role=datatypes.Role.ACTOR):
        return {}

      async def save_checkpoint(
          self, role=datatypes.Role.ACTOR, metadata=None, **kwargs
      ):
        return datatypes.Response(metadata={"checkpoint_path": "/tmp/ckpt"})

    program = rl_program.StandardRLProgram(
        algo=_FakeAlgo(),
        dataset=[{"prompt": f"p{i}"} for i in range(3)],
        max_steps=1,
        batch_size=3,
        sync_weights=False,
        assembler=_FakeAssembler(),
        pipeline_train_microbatches=True,
        stage_train_batches=True,
        max_staged_batches=2,
        group_order="prompt_batch",
    )
    engine = _FakeEngine()
    program.engine = engine

    def _make_item(i: int) -> datatypes.TrajectoryItem:
      item = datatypes.TrajectoryItem(
          prompt_id=f"p_{i}",
          group_index=0,
          start_step=0,
          traj={"trajectory_reward": 1.0, "status": "SUCCEEDED", "steps": [0]},
          is_valid=True,
          prompt_tokens=np.array([1, 2], dtype=np.int32),
          completion_tokens=np.array([3, 4], dtype=np.int32),
          policy_version=0,
          metadata={"prompt_idx": i, "batch_idx": 0, "intra_batch_idx": i},
      )
      item.payload = _payload()
      return item

    async def _feed_straggler():
      # Wait until MB 1 has actually started training before delivering Group 2.
      await asyncio.wait_for(mb1_train_started.wait(), timeout=2.0)
      await program.scored_q.put(_make_item(2))
      await program.scored_q.close()

    async def _run():
      await program.scored_q.put(_make_item(0))
      await program.scored_q.put(_make_item(1))
      straggler_task = asyncio.create_task(_feed_straggler())
      await program._train_loop()
      await straggler_task

    asyncio.run(_run())
    self.assertTrue(mb1_train_started.is_set())
    self.assertEqual(
        engine.trained_keys,
        [None, "step_0_stage_1", "step_0_stage_2"],
    )

  def test_env_default_ref_kl_sampler_agreement_and_fallback(self):
    """Covers env default, reference KL, sampler agreement, and fallback."""
    from tunix.experimental.orchestrator import batch_assembly
    from tunix.experimental.orchestrator import rl_program

    class _FakeAlgo:
      num_generations = 1
      mini_batch_size = 3
      train_micro_batch_size = 1
      max_response_length = 4
      requires_reference_kl = True

      class _Config:
        temperature = 1.0
        use_rollout_logps = False
        truncated_importance_sampling_type = None
        seq_logprob_error_threshold = None
        sampler_is_length_buckets = None
        sampler_is = None
        sampler_is_threshold = 2.0

      algo_config = _Config()

    class _FakeAssembler(batch_assembly.BatchAssembler):

      def __init__(self):
        super().__init__(num_generations=1, mini_batch_size=1)
        self._count = 0

      def feed(self, payloads):
        out = []
        for p in payloads:
          self._count += 1
          out.append(
              batch_assembly.AssembledBatch(
                  payload=p,
                  trajectory_ids=(f"t_{self._count}",),
                  is_final_batch=(self._count == 3),
              )
          )
        return out

      def flush(self):
        return []

      def reset(self):
        self._count = 0

    class _FakeEngine:

      def __init__(self):
        self.ref_calls = 0
        self.stage_calls = 0
        self.trained = []
        self.reset_called = False

      async def per_token_logps(self, role, items=None, **kwargs):
        self.ref_calls += 1
        return np.zeros_like(items.completion_ids, dtype=np.float32)

      async def stage_train_batch(
          self, payload, role=datatypes.Role.ACTOR, stage_key=None
      ):
        self.stage_calls += 1
        if self.stage_calls == 1:
          # Simulate transient RPC failure on MB 1 -> falls back inline.
          raise RuntimeError("RemoteExecutionError: transient failure")
        return stage_key

      async def train_step(
          self,
          payload,
          role=datatypes.Role.ACTOR,
          accumulate_gradients=True,
          apply_optimizer=False,
          stage_key=None,
          **kwargs,
      ):
        await asyncio.sleep(0.01)
        self.trained.append((stage_key, payload is not None, apply_optimizer))
        return {
            "updated": apply_optimizer,
            "train_step": 1 if apply_optimizer else None,
        }

      async def get_metrics(self, role=datatypes.Role.ACTOR):
        return {}

      async def save_checkpoint(
          self, role=datatypes.Role.ACTOR, metadata=None, **kwargs
      ):
        return datatypes.Response(metadata={"checkpoint_path": "/tmp/ckpt"})

      async def reset_staged_batches(self, role=datatypes.Role.ACTOR):
        self.reset_called = True

    with mock.patch.dict(
        os.environ, {trainer_worker.STAGE_BATCH_ENV: "stage"}, clear=False
    ):
      program = rl_program.StandardRLProgram(
          algo=_FakeAlgo(),
          dataset=[{"prompt": f"p{i}"} for i in range(3)],
          max_steps=1,
          batch_size=3,
          sync_weights=False,
          assembler=_FakeAssembler(),
          pipeline_train_microbatches=True,
      )
      self.assertTrue(program.stage_train_batches)

    engine = _FakeEngine()
    program.engine = engine
    logged_metrics = {}
    orig_log_metric = program._log_metric

    def _capture_metric(name, val, step, prefix=None):
      key = f"{prefix}/{name}" if prefix else name
      logged_metrics[key] = val
      return orig_log_metric(name, val, step, prefix=prefix)

    program._log_metric = _capture_metric

    async def _run():
      for i in range(3):
        item = datatypes.TrajectoryItem(
            prompt_id=f"p_{i}",
            group_index=0,
            start_step=0,
            traj={
                "trajectory_reward": 1.0,
                "status": "SUCCEEDED",
                "steps": [0],
            },
            is_valid=True,
            prompt_tokens=np.array([1, 2], dtype=np.int32),
            completion_tokens=np.array([3, 4], dtype=np.int32),
            policy_version=0,
            metadata={"prompt_idx": i, "batch_idx": 0, "intra_batch_idx": i},
        )
        item.payload = _payload()
        await program.scored_q.put(item)
      await program.scored_q.close()
      await program._train_loop()

    asyncio.run(_run())
    self.assertEqual(engine.ref_calls, 3)
    # MB 0 ran inline; MB 1 stage raised RuntimeError so fell back inline
    # (payload is not None); MB 2 staged and ran with stage_key.
    self.assertEqual(
        engine.trained,
        [
            (None, True, False),
            (None, True, False),
            ("step_0_stage_2", False, True),
        ],
    )
    self.assertEqual(logged_metrics.get("stage/fallbacks"), 1.0)

    # Also verify pipeline_train_microbatches=False logs a warning when
    # stage_train_batches=True.
    with self.assertLogs(level="WARNING") as warn_logs:
      rl_program.StandardRLProgram(
          algo=_FakeAlgo(),
          dataset=[],
          max_steps=1,
          batch_size=3,
          sync_weights=False,
          assembler=_FakeAssembler(),
          pipeline_train_microbatches=False,
          stage_train_batches=True,
      )
    self.assertTrue(
        any("pipeline_train_microbatches=False" in m for m in warn_logs.output)
    )


if __name__ == "__main__":
  absltest.main()
