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

"""Unit tests for TrainerWorker.

Verifies TrainerWorker RPC request unpacking (TrainRequest), delegation to
AbstractTrainer (fwd_bwd, eval_step, update), response metadata stamping, and
per-token log-prob scorer run through AbstractTrainer.fwd_only.
"""

import dataclasses
import functools
import importlib
import os
import sys
import threading
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
from opentelemetry.sdk import metrics as sdk_metrics
from opentelemetry.sdk.metrics import export as sdk_metrics_export
import optax
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.testing import otel_testing
from tunix.dist_rl.train import abstract_trainer
from tunix.dist_rl.train import jax_logps
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.worker import otel_metrics
from tunix.dist_rl.worker import trainer_worker
from tunix.rl import algo_core
from tunix.rl import algorithm_config
from tunix.rl import common as rl_common
from tunix.tests import test_common as tc

# CPU environment setup to simulate multi device env. Use 4 devices so pytest
# collection does not downgrade XLA_FLAGS below what sibling dist_rl tests need.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


class FakeTrainer(abstract_trainer.AbstractTrainer):

  def __init__(self):
    self.compile_calls = []
    self.fwd_bwd_calls = []
    self.eval_step_calls = []
    # Payloads passed to `fwd_only`.
    self.fwd_only_calls = []
    self._fwd_only_fn = None
    self.model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    self.policy_version = 3
    self.step_count = 10
    self.target_state = None
    self.gen_model_input_fn = None

  def compile(self, dummy_data=None):
    self.compile_calls.append(dummy_data)

  def with_loss_fn(self, loss_fn, has_aux=False):
    self.loss_fn = loss_fn
    self.has_aux = has_aux

  def with_gen_model_input_fn(self, gen_model_input_fn):
    self.gen_model_input_fn = gen_model_input_fn

  def fwd_bwd(self, payload):
    self.fwd_bwd_calls.append(payload)

  def update(self):
    self.step_count += 1
    return self.step_count

  def eval_step(self, payload):
    self.eval_step_calls.append(payload)

  def with_fwd_only_fn(self, fn):
    self._fwd_only_fn = fn
    return self

  def fwd_only(self, payload):
    if self._fwd_only_fn is None:
      raise ValueError("FakeTrainer.fwd_only needs with_fwd_only_fn() first.")
    self.fwd_only_calls.append(payload)
    return self._fwd_only_fn(self.model, payload)

  compute_per_token_logps = jax_logps.compute_trainer_per_token_logps

  def save_checkpoint(self, metadata=None, *, step=None, force=False):
    pass

  def restore_checkpoint(self, *, step=None):
    return {}

  def get_metrics(self):
    return {"loss": 0.25}

  def prepare_weight_sync(self, *, sync_request=None):
    pass

  def set_target_state(self, target_state: Any) -> None:
    self.target_state = target_state

  def close(self):
    pass


class TrainerWorkerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.fake_trainer = FakeTrainer()
    self.worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer,
        worker_id="trainer_0",
    )
    self.worker.initialize()

  def test_trainer_factory_deferred_until_initialize(self):
    calls = []

    def _factory():
      calls.append("called")
      return self.fake_trainer

    worker = trainer_worker.TrainerWorker(
        trainer_factory=_factory, worker_id="lazy_trainer"
    )
    gen_fn = lambda x: {"input": x}
    worker.with_gen_model_input_fn(gen_fn)
    self.assertEmpty(calls)
    self.assertEqual(worker.state, datatypes.WorkerState.PENDING)
    self.assertEqual(worker.heartbeat().policy_version, 0)
    self.assertEqual(worker.info().resources["trainer"], "UninitializedTrainer")

    worker.initialize()
    self.assertEqual(calls, ["called"])
    self.assertEqual(worker.state, datatypes.WorkerState.READY)
    self.assertEqual(worker.heartbeat().policy_version, 3)
    self.assertEqual(worker.info().resources["trainer"], "FakeTrainer")
    self.assertIs(self.fake_trainer.gen_model_input_fn, gen_fn)

  def test_initialize_failure_sets_error_state(self):
    def _failing_factory():
      raise RuntimeError("weight load failed")

    worker = trainer_worker.TrainerWorker(
        trainer_factory=_failing_factory, worker_id="bad_trainer"
    )
    with self.assertRaisesRegex(RuntimeError, "weight load failed"):
      worker.initialize()
    self.assertEqual(worker.state, datatypes.WorkerState.ERROR)
    self.assertEqual(worker.heartbeat().last_error, "weight load failed")

    # Verify a failure in a pending hook leaves _trainer as None so a retry
    # re-runs initialization cleanly.
    should_fail = True
    orig_with_loss_fn = self.fake_trainer.with_loss_fn

    def _flaky_with_loss_fn(loss_fn, has_aux=False):
      if should_fail:
        raise RuntimeError("loss hook failed")
      orig_with_loss_fn(loss_fn, has_aux)

    self.fake_trainer.with_loss_fn = _flaky_with_loss_fn
    retry_worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer, worker_id="retry_trainer"
    )
    retry_worker.with_loss_fn(lambda m, **kw: kw)
    with self.assertRaisesRegex(RuntimeError, "loss hook failed"):
      retry_worker.initialize()
    self.assertIsNone(retry_worker._trainer)
    self.assertEqual(retry_worker.state, datatypes.WorkerState.ERROR)

    should_fail = False
    retry_worker._state = datatypes.WorkerState.PENDING
    retry_worker.initialize()
    self.assertIs(retry_worker._trainer, self.fake_trainer)
    self.assertEqual(retry_worker.state, datatypes.WorkerState.READY)
    self.assertIsNotNone(self.fake_trainer.loss_fn)

  def test_with_loss_fn_binds_logps_chunk_size(self):
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer, logps_chunk_size=2048
    )
    worker.with_loss_fn(lambda model, **kwargs: kwargs, has_aux=True)
    worker.initialize()

    self.assertTrue(self.fake_trainer.has_aux)
    self.assertEqual(
        self.fake_trainer.loss_fn(None, pad_id=0),
        {"pad_id": 0, "compute_logps_chunk_size": 2048},
    )

  def test_with_loss_fn_per_call_chunk_size_wins(self):
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer, logps_chunk_size=2048
    )
    worker.with_loss_fn(lambda model, **kwargs: kwargs)
    worker.initialize()

    kwargs = self.fake_trainer.loss_fn(None, compute_logps_chunk_size=512)

    self.assertEqual(kwargs["compute_logps_chunk_size"], 512)

  def test_with_loss_fn_unchanged_without_chunk_size(self):
    fn = lambda model, **kwargs: kwargs
    self.worker.with_loss_fn(fn)

    self.assertIs(self.fake_trainer.loss_fn, fn)

  def test_compile_forwards_dummy_data_to_trainer(self):
    dummy = {"dummy": 1}
    resp = self.worker.compile(dummy)
    self.assertTrue(resp.metadata["compiled"])
    self.assertEqual(self.worker.state, datatypes.WorkerState.READY)
    self.assertEqual(self.fake_trainer.compile_calls, [dummy])

  def test_compile_resolves_dummy_routed_experts_for_maxtext_engine(self):
    encode_fn = functools.partial(
        trainer_worker.packing.encode_routed_experts,
        mode="auto",
        batched=True,
    )

    class _MaxTextCfg:
      num_decoder_layers = 40
      num_experts_per_tok = 8
      num_experts = 256

    class _BaseModel:
      config = _MaxTextCfg()

    class _AdapterModel:
      # Mirrors `maxtext.utils.model_creation_utils.from_pretrained(...,
      # wrap_with_tunix_adapter=True)`, which explicitly sets `model.config =
      # None` on `TunixMaxTextAdapter`.
      config = None
      base = _BaseModel()

    # 1. Public `trainer.model.base.config` (TunixMaxTextAdapter) with real
    # `packing.encode_routed_experts` -> 5D uint10 `(2, 8, 40, 2, 5)`.
    self.fake_trainer.model = _AdapterModel()
    dummy = datatypes.RLTrainerPayload(
        prompt_ids=np.zeros((2, 4), dtype=np.int32),
        prompt_mask=np.ones((2, 4), dtype=np.int32),
        completion_ids=np.zeros((2, 4), dtype=np.int32),
        completion_mask=np.ones((2, 4), dtype=np.int32),
        advantages=np.zeros((2, 8), dtype=np.float32),
        routed_experts=np.zeros((2, 8, 0, 0), dtype=np.int16),
        metadata={"routed_experts_encode_fn": encode_fn, "tag": "moe"},
    )
    resp = self.worker.compile(dummy)
    self.assertTrue(resp.metadata["compiled"])
    self.assertLen(self.fake_trainer.compile_calls, 1)
    compiled_payload = self.fake_trainer.compile_calls[0]
    self.assertEqual(compiled_payload.routed_experts.shape, (2, 8, 40, 2, 5))
    self.assertEqual(compiled_payload.routed_experts.dtype, np.uint8)
    self.assertEqual(compiled_payload.metadata, {"tag": "moe"})

    # 2. No `routed_experts_encode_fn` -> raw `(2, 8, 40, 8)` int16 with
    # metadata preserved unchanged.
    self.fake_trainer.compile_calls.clear()
    raw_dummy = dataclasses.replace(dummy, metadata={"keep": 1})
    self.worker.compile(raw_dummy)
    raw_compiled = self.fake_trainer.compile_calls[0]
    self.assertEqual(raw_compiled.routed_experts.shape, (2, 8, 40, 8))
    self.assertEqual(raw_compiled.routed_experts.dtype, np.int16)
    self.assertEqual(raw_compiled.metadata, {"keep": 1})

    # 3. Native `trainer.model.config.num_layers` path (`num_experts <= 255`)
    # -> 4D uint8 `(2, 8, 12, 4)`.
    class _NativeCfg:
      num_layers = 12
      num_experts_per_tok = 4
      num_experts = 64

    class _NativeModel:
      config = _NativeCfg()

    self.fake_trainer.model = _NativeModel()
    self.fake_trainer.compile_calls.clear()
    self.worker.compile(dummy)
    native_compiled = self.fake_trainer.compile_calls[0]
    self.assertEqual(native_compiled.routed_experts.shape, (2, 8, 12, 4))
    self.assertEqual(native_compiled.routed_experts.dtype, np.uint8)
    self.assertEqual(native_compiled.metadata, {"tag": "moe"})

    # 4. Non-empty `routed_experts` is forwarded untouched.
    self.fake_trainer.compile_calls.clear()
    non_empty = dataclasses.replace(
        dummy,
        routed_experts=np.ones((2, 8, 12, 4), dtype=np.int16),
        metadata={"keep": 2},
    )
    self.worker.compile(non_empty)
    passthrough = self.fake_trainer.compile_calls[0]
    self.assertIs(passthrough, non_empty)
    self.assertIs(
        abstract_trainer.resolve_dummy_routed_experts(
            self.fake_trainer, compiled_payload
        ),
        compiled_payload,
    )

    # 5. Malformed rank < 2 zero-sized placeholder fails loud.
    malformed = dataclasses.replace(
        dummy, routed_experts=np.zeros((0,), dtype=np.int16)
    )
    with self.assertRaisesRegex(ValueError, "rank >= 2"):
      abstract_trainer.resolve_dummy_routed_experts(
          self.fake_trainer, malformed
      )

    # 6. Model without any non-None config fails loud with a clear ValueError.
    class _ConfiglessModel:
      config = None

    self.fake_trainer.model = _ConfiglessModel()
    with self.assertRaisesRegex(ValueError, "non-None"):
      abstract_trainer.resolve_dummy_routed_experts(self.fake_trainer, dummy)

  def test_offload_markers_and_stop_serializes_with_inflight_step(self):
    attr = "_tunix_offload_to_thread"
    for offloaded_name in (
        "initialize",
        "compile",
        "start",
        "stop",
        "with_loss_fn",
        "with_gen_model_input_fn",
        "set_target_state",
        "fwd_bwd",
        "update",
        "eval_step",
        "run_eval",
        "per_token_logps",
        "save_checkpoint",
        "restore_checkpoint",
        "prepare_weight_sync",
        "release_weight_sync",
        "get_metrics",
    ):
      self.assertTrue(
          getattr(getattr(self.worker, offloaded_name), attr, False),
          msg=f"{offloaded_name} should opt in to worker-thread offloading",
      )
    for inline_name in ("heartbeat", "info"):
      self.assertFalse(
          getattr(getattr(self.worker, inline_name), attr, False),
          msg=f"{inline_name} must remain inline on the event loop",
      )

    entered_step = threading.Event()
    release_step = threading.Event()
    close_called = threading.Event()

    def _blocking_fwd_bwd(payload):
      del payload
      entered_step.set()
      if not release_step.wait(timeout=10.0):
        raise TimeoutError("release_step timed out")

    def _recording_close():
      close_called.set()

    self.fake_trainer.fwd_bwd = _blocking_fwd_bwd
    self.fake_trainer.close = _recording_close
    req = datatypes.TrainRequest(
        request_id="req-lock",
        payload=datatypes.RLTrainerPayload(
            prompt_ids=np.array([[1]], dtype=np.int32),
            prompt_mask=np.ones((1, 1), dtype=np.float32),
            completion_ids=np.array([[2]], dtype=np.int32),
            completion_mask=np.array([[1]], dtype=np.float32),
            advantages=np.array([1.0], dtype=np.float32),
        ),
    )
    step_thread = threading.Thread(
        target=lambda: self.worker.fwd_bwd(request=req)
    )
    stop_thread = threading.Thread(target=self.worker.stop)
    step_thread.start()
    self.assertTrue(entered_step.wait(timeout=5.0))
    # heartbeat() and info() must remain non-blocking while _trainer_lock is
    # held by an in-flight fwd_bwd step.
    self.assertEqual(self.worker.heartbeat().state, datatypes.WorkerState.READY)
    self.assertEqual(self.worker.info().worker_id, "trainer_0")

    stop_thread.start()
    self.assertFalse(close_called.wait(timeout=0.05))
    release_step.set()
    step_thread.join(timeout=5.0)
    stop_thread.join(timeout=5.0)
    self.assertFalse(step_thread.is_alive())
    self.assertFalse(stop_thread.is_alive())
    self.assertTrue(close_called.is_set())
    self.assertEqual(self.worker.state, datatypes.WorkerState.STOPPED)

  def test_with_loss_fn_chunked_grpo_loss_matches_unchunked(self):
    batch, prompt_len, completion_len = 2, 3, 4
    example = datatypes.RLTrainerPayload(
        prompt_ids=np.arange(1, 1 + batch * prompt_len, dtype=np.int32).reshape(
            batch, prompt_len
        ),
        prompt_mask=np.ones((batch, prompt_len), dtype=np.int32),
        completion_ids=np.arange(
            1, 1 + batch * completion_len, dtype=np.int32
        ).reshape(batch, completion_len),
        completion_mask=np.ones((batch, completion_len), dtype=np.int32),
        advantages=np.array([1.0, -1.0], dtype=np.float32),
    )
    algo_config = algorithm_config.GRPOConfig(beta=0.0)
    algo_config.temperature = 1.0

    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer, logps_chunk_size=3
    )
    worker.with_loss_fn(algo_core.grpo_loss_fn)
    worker.initialize()
    chunked_fn = self.fake_trainer.loss_fn
    self.assertEqual(chunked_fn.keywords, {"compute_logps_chunk_size": 3})

    kwargs = dict(
        train_example=example, algo_config=algo_config, pad_id=0, eos_id=0
    )
    chunked = chunked_fn(self.fake_trainer.model, **kwargs)
    unchunked = algo_core.grpo_loss_fn(self.fake_trainer.model, **kwargs)

    np.testing.assert_allclose(
        np.asarray(chunked.primary_loss.compute()),
        np.asarray(unchunked.primary_loss.compute()),
        rtol=1e-5,
        atol=1e-5,
    )

  def test_fwd_bwd_with_train_request(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[1, 2], [1, 2]], dtype=np.int32),
        prompt_mask=np.ones((2, 2), dtype=np.float32),
        completion_ids=np.array([[3, 4], [3, 4]], dtype=np.int32),
        completion_mask=np.array([[1, 1], [1, 0]], dtype=np.float32),
        advantages=np.array([1.0, 2.0], dtype=np.float32),
        metadata={"step": 1},
    )
    request = datatypes.TrainRequest(
        request_id="req-train-123",
        payload=payload,
        metadata={"batch_id": "b0"},
    )

    resp = self.worker.fwd_bwd(request=request)

    self.assertIsInstance(resp, datatypes.Response)
    self.assertEqual(resp.request_id, "req-train-123")
    self.assertEqual(resp.metadata["worker_id"], "trainer_0")
    self.assertEqual(resp.metadata["batch_id"], "b0")
    self.assertEqual(resp.metadata["policy_version"], 3)
    self.assertTrue(resp.metadata["queued"])
    self.assertLen(self.fake_trainer.fwd_bwd_calls, 1)
    self.assertIs(self.fake_trainer.fwd_bwd_calls[0], payload)

  def test_eval_step_with_train_request(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[1]], dtype=np.int32),
        prompt_mask=np.ones((1, 1), dtype=np.float32),
        completion_ids=np.array([[2]], dtype=np.int32),
        completion_mask=np.array([[1]], dtype=np.float32),
        advantages=np.array([1.0], dtype=np.float32),
    )
    request = datatypes.TrainRequest(
        request_id="req-eval-456",
        payload=payload,
        metadata={"eval_split": "val"},
    )

    resp = self.worker.eval_step(request=request)

    self.assertIsInstance(resp, datatypes.Response)
    self.assertEqual(resp.request_id, "req-eval-456")
    self.assertEqual(resp.metadata["eval_split"], "val")
    self.assertTrue(resp.metadata["evaluated"])
    self.assertLen(self.fake_trainer.eval_step_calls, 1)
    self.assertIs(self.fake_trainer.eval_step_calls[0], payload)

  def test_update_returns_step_count(self):
    step, metrics = self.worker.update()
    self.assertEqual(step, 11)
    self.assertEqual(metrics, {"loss": 0.25})

  def test_set_target_state_configures_trainer(self):
    target_state = {"params": np.zeros((4, 4))}
    resp = self.worker.set_target_state(target_state=target_state)

    self.assertIsInstance(resp, datatypes.Response)
    self.assertTrue(resp.metadata["target_state_configured"])
    self.assertEqual(self.fake_trainer.target_state, target_state)

  def test_save_checkpoint_metadata_is_optional(self):
    with mock.patch.object(
        self.fake_trainer, "save_checkpoint", autospec=True
    ) as save:
      self.worker.save_checkpoint(step=42, force=True)
    save.assert_called_once_with(None, step=42, force=True)

  def test_set_target_state_falls_back_to_abstract_default(self):
    with mock.patch.object(
        FakeTrainer,
        "set_target_state",
        abstract_trainer.AbstractTrainer.set_target_state,
    ):
      resp = self.worker.set_target_state(target_state={"params": 1})
    self.assertTrue(resp.metadata["target_state_configured"])
    self.assertIsNone(self.fake_trainer.target_state)

  def _expected_logps(self, request, chunk_size=0):
    graphdef, state = nnx.split(self.fake_trainer.model)
    return np.asarray(
        rl_common.compute_per_token_logps(
            graphdef,
            state,
            prompt_tokens=jnp.asarray(request.prompt_tokens),
            completion_tokens=jnp.asarray(request.completion_tokens),
            pad_id=request.pad_id,
            eos_id=request.eos_id,
            stop_gradient=True,
            temperature=request.temperature,
            chunk_size=chunk_size,
            segment_ids=(
                None
                if request.segment_ids is None
                else jnp.asarray(request.segment_ids)
            ),
            segment_positions=(
                None
                if request.segment_positions is None
                else jnp.asarray(request.segment_positions)
            ),
        )
    )

  def _unpacked_request(self, temperature=1.0):
    batch, prompt_len, completion_len = 3, 3, 4
    return datatypes.LogprobsRequest(
        prompt_tokens=np.arange(
            1, 1 + batch * prompt_len, dtype=np.int32
        ).reshape(batch, prompt_len),
        completion_tokens=np.arange(
            1, 1 + batch * completion_len, dtype=np.int32
        ).reshape(batch, completion_len),
        temperature=temperature,
        pad_id=0,
        eos_id=0,
    )

  def test_per_token_logps_matches_reference_scorer(self):
    request = self._unpacked_request()

    result = self.worker.per_token_logps(items=request)

    self.assertIsInstance(result, datatypes.LogprobsResponse)
    self.assertEqual(result.request_id, request.request_id)
    self.assertEqual(result.model_version, self.fake_trainer.policy_version)
    self.assertIsInstance(result.per_token_logps, np.ndarray)
    self.assertEqual(result.per_token_logps.dtype, np.float32)
    self.assertEqual(result.per_token_logps.shape, (3, 4))
    # Whole request in one forward by default.
    self.assertLen(self.fake_trainer.fwd_only_calls, 1)
    np.testing.assert_allclose(
        result.per_token_logps,
        self._expected_logps(request),
        rtol=1e-4,
        atol=1e-4,
    )

  def test_per_token_logps_micro_batches_in_order(self):
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer,
        worker_id="trainer_1",
        logps_micro_batch_size=2,
    )
    worker.initialize()
    request = self._unpacked_request()

    result = worker.per_token_logps(items=request)

    self.assertLen(self.fake_trainer.fwd_only_calls, 2)
    first_payload = self.fake_trainer.fwd_only_calls[0]
    second_payload = self.fake_trainer.fwd_only_calls[1]
    self.assertIsInstance(first_payload, datatypes.LogprobsScoringPayload)
    np.testing.assert_array_equal(
        first_payload.completion_ids, request.completion_tokens[:2]
    )
    np.testing.assert_array_equal(
        second_payload.completion_ids, request.completion_tokens[2:]
    )
    np.testing.assert_allclose(
        result.per_token_logps,
        self._expected_logps(request),
        rtol=1e-4,
        atol=1e-4,
    )

  def test_per_token_logps_temperature_and_chunked_vocab(self):
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer,
        worker_id="trainer_2",
        logps_chunk_size=3,
    )
    worker.initialize()
    request = self._unpacked_request(temperature=0.7)

    result = worker.per_token_logps(items=request)

    payload = self.fake_trainer.fwd_only_calls[0]
    self.assertEqual(payload.pad_id, 0)
    self.assertEqual(payload.eos_id, 0)
    self.assertEqual(payload.temperature, 0.7)
    np.testing.assert_allclose(
        result.per_token_logps,
        self._expected_logps(request, chunk_size=3),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        result.per_token_logps,
        self._expected_logps(request),
        rtol=1e-4,
        atol=1e-4,
    )

  def test_per_token_logps_packed_request(self):
    request = datatypes.LogprobsRequest(
        prompt_tokens=np.zeros((2, 0), dtype=np.int32),
        completion_tokens=np.array([[3, 4, 5], [3, 4, 0]], dtype=np.int32),
        temperature=1.0,
        pad_id=0,
        eos_id=0,
        segment_ids=np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int32),
        segment_positions=np.array([[0, 1, 2], [0, 1, 0]], dtype=np.int32),
    )

    result = self.worker.per_token_logps(items=request)

    self.assertEqual(result.per_token_logps.shape, (2, 3))
    np.testing.assert_allclose(
        result.per_token_logps,
        self._expected_logps(request),
        rtol=1e-4,
        atol=1e-4,
    )

  def test_per_token_logps_empty_batch_raises(self):
    request = datatypes.LogprobsRequest(
        prompt_tokens=np.zeros((0, 3), dtype=np.int32),
        completion_tokens=np.zeros((0, 4), dtype=np.int32),
        temperature=1.0,
        pad_id=0,
        eos_id=0,
    )
    with self.assertRaises(ValueError):
      self.worker.per_token_logps(items=request)
    self.assertEmpty(self.fake_trainer.fwd_only_calls)
    self.assertEqual(self.worker.state, datatypes.WorkerState.ERROR)

  def test_per_token_logps_requires_pad_and_eos(self):
    request = datatypes.LogprobsRequest(
        prompt_tokens=np.zeros((1, 0), dtype=np.int32),
        completion_tokens=np.array([[3, 4]], dtype=np.int32),
        temperature=1.0,
    )
    with self.assertRaises(ValueError):
      self.worker.per_token_logps(items=request)
    self.assertEmpty(self.fake_trainer.fwd_only_calls)

  def test_per_token_logps_without_routing_passes_none(self):
    self.worker.per_token_logps(items=self._unpacked_request())
    payload = self.fake_trainer.fwd_only_calls[0]
    self.assertIsNone(payload.routed_experts)

  def test_per_token_logps_payload_uses_scoring_payload_without_rl_masks(self):
    request = self._unpacked_request(temperature=0.7)
    self.worker.per_token_logps(items=request)
    payload = self.fake_trainer.fwd_only_calls[0]
    self.assertIsInstance(payload, datatypes.LogprobsScoringPayload)
    np.testing.assert_array_equal(payload.prompt_ids, request.prompt_tokens)
    np.testing.assert_array_equal(
        payload.completion_ids, request.completion_tokens
    )
    self.assertEqual(payload.pad_id, request.pad_id)
    self.assertEqual(payload.eos_id, request.eos_id)
    self.assertEqual(payload.temperature, request.temperature)
    self.assertFalse(hasattr(payload, "prompt_mask"))
    self.assertFalse(hasattr(payload, "completion_mask"))
    self.assertFalse(hasattr(payload, "advantages"))

  def test_per_token_logps_falls_back_to_legacy_trainer_without_with_fwd_only_fn(
      self,
  ):
    class _LegacyFakeTrainer:

      def __init__(self):
        self.model = tc.ToyTransformer(
            config=tc.ModelConfig(), rngs=nnx.Rngs(0)
        )
        self.policy_version = 3
        self.step_count = 10
        self.fwd_only_calls = []

      def fwd_only(self, fn, *args, **kwargs):
        self.fwd_only_calls.append((fn, args, kwargs))
        return fn(self.model, *args, **kwargs)

      compute_per_token_logps = jax_logps.compute_trainer_per_token_logps

    legacy_trainer = _LegacyFakeTrainer()
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: legacy_trainer,
        worker_id="legacy_trainer",
        logps_micro_batch_size=2,
        logps_chunk_size=3,
    )
    worker.initialize()
    request = self._unpacked_request(temperature=0.7)

    result = worker.per_token_logps(items=request)

    self.assertLen(legacy_trainer.fwd_only_calls, 2)
    np.testing.assert_allclose(
        result.per_token_logps,
        self._expected_logps(request),
        rtol=1e-4,
        atol=1e-4,
    )

  def test_per_token_logps_dispatches_all_micro_batches_before_host_transfer(
      self,
  ):
    events: list[str] = []

    class _DeferredDeviceArray:

      def __init__(self, arr: np.ndarray, idx: int):
        self._arr = arr
        self._idx = idx

      def __array__(self, dtype=None):
        events.append(f"materialize_{self._idx}")
        return np.asarray(self._arr, dtype=dtype)

    class _AsyncTrackingTrainer(FakeTrainer):

      def compute_per_token_logps(self, prompt_tokens, completion_tokens, **kw):
        idx = len(self.fwd_only_calls)
        events.append(f"dispatch_{idx}")
        out = super().compute_per_token_logps(
            prompt_tokens, completion_tokens, **kw
        )
        return _DeferredDeviceArray(np.asarray(out), idx)

    async_trainer = _AsyncTrackingTrainer()
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: async_trainer,
        worker_id="async_trainer",
        logps_micro_batch_size=2,
    )
    worker.initialize()
    request = self._unpacked_request(temperature=0.7)

    result = worker.per_token_logps(items=request)

    self.assertEqual(
        events,
        ["dispatch_0", "dispatch_1", "materialize_0", "materialize_1"],
    )
    np.testing.assert_allclose(
        result.per_token_logps,
        self._expected_logps(request),
        rtol=1e-4,
        atol=1e-4,
    )

  def test_per_token_logps_forwards_routed_experts_per_micro_batch(self):
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer,
        worker_id="trainer_3",
        logps_micro_batch_size=2,
    )
    worker.initialize()
    request = self._unpacked_request()
    batch, seq_len = 3, 3 + 4
    # Row i routes every token to expert i, so a mis-sliced row is visible.
    routed = np.broadcast_to(
        np.arange(batch, dtype=np.int32).reshape(batch, 1, 1, 1),
        (batch, seq_len, 2, 2),
    )
    request = dataclasses.replace(request, routed_experts=routed)

    result = worker.per_token_logps(items=request)

    self.assertLen(self.fake_trainer.fwd_only_calls, 2)
    first_payload = self.fake_trainer.fwd_only_calls[0]
    second_payload = self.fake_trainer.fwd_only_calls[1]
    self.assertEqual(first_payload.routed_experts.dtype, np.int16)
    np.testing.assert_array_equal(first_payload.routed_experts, routed[:2])
    np.testing.assert_array_equal(second_payload.routed_experts, routed[2:])
    # The fake model does not accept `forced_routed_experts`, so scoring is
    # unchanged; the routing is only forwarded.
    np.testing.assert_allclose(
        result.per_token_logps,
        self._expected_logps(request),
        rtol=1e-4,
        atol=1e-4,
    )

    # Also verify compact uint8 (4D) and uint10 (5D) routed_experts are kept
    # in compact uint8 form without widening to int16.
    for shape in ((batch, seq_len, 2, 4), (batch, seq_len, 2, 1, 5)):
      self.fake_trainer.fwd_only_calls.clear()
      compact_routed = np.ones(shape, dtype=np.uint8)
      compact_req = dataclasses.replace(request, routed_experts=compact_routed)
      worker.per_token_logps(items=compact_req)
      self.assertLen(self.fake_trainer.fwd_only_calls, 2)
      payload0 = self.fake_trainer.fwd_only_calls[0]
      payload1 = self.fake_trainer.fwd_only_calls[1]
      self.assertEqual(payload0.routed_experts.dtype, np.uint8)
      self.assertEqual(payload0.routed_experts.shape, (2,) + shape[1:])
      self.assertEqual(payload1.routed_experts.dtype, np.uint8)
      self.assertEqual(payload1.routed_experts.shape, (1,) + shape[1:])

  def test_per_token_logps_end_to_end_with_real_peft_trainer(self):
    tc.safe_set_n_cpu_devices(2)
    mesh = shd.Mesh(
        devices=np.array(jax.devices()[:2]).reshape(2, 1),
        axis_names=("fsdp", "tp"),
    )
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    config = peft_trainer_v2.TrainingConfig(
        eval_every_n_steps=100, max_steps=100
    )
    real_trainer = peft_trainer_v2.PeftTrainer(model, optax.sgd(1e-3), config)
    with self.assertRaises(ValueError):
      FakeTrainer().fwd_only(
          datatypes.LogprobsScoringPayload(
              prompt_ids=np.zeros((2, 2), dtype=np.int32),
              completion_ids=np.ones((2, 2), dtype=np.int32),
              pad_id=0,
              eos_id=1,
              temperature=1.0,
          )
      )

    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: real_trainer,
        worker_id="real_peft_trainer",
        logps_micro_batch_size=2,
        logps_chunk_size=2,
    )
    worker.initialize()
    self.assertIsNone(real_trainer._jitted_fwd_only_fn)

    unpacked_req = datatypes.LogprobsRequest(
        prompt_tokens=np.arange(1, 13, dtype=np.int32).reshape(4, 3),
        completion_tokens=np.arange(1, 17, dtype=np.int32).reshape(4, 4),
        temperature=0.8,
        pad_id=0,
        eos_id=0,
    )
    packed_req = datatypes.LogprobsRequest(
        prompt_tokens=np.zeros((4, 0), dtype=np.int32),
        completion_tokens=np.array(
            [[3, 4, 5, 0], [3, 4, 0, 0], [6, 7, 8, 9], [1, 2, 0, 0]],
            dtype=np.int32,
        ),
        temperature=0.8,
        pad_id=0,
        eos_id=0,
        segment_ids=np.array(
            [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]],
            dtype=np.int32,
        ),
        segment_positions=np.array(
            [[0, 1, 2, 0], [0, 1, 0, 0], [0, 1, 2, 3], [0, 1, 0, 0]],
            dtype=np.int32,
        ),
    )

    with mesh:
      unpacked_resp = worker.per_token_logps(items=unpacked_req)
      self.assertIsNotNone(real_trainer._jitted_fwd_only_fn)

      packed_resp = worker.per_token_logps(items=packed_req)
      self.assertIsNotNone(real_trainer._jitted_fwd_only_fn)

    graphdef, state = nnx.split(model)
    for req, resp in ((unpacked_req, unpacked_resp), (packed_req, packed_resp)):
      expected = np.asarray(
          rl_common.compute_per_token_logps(
              graphdef,
              state,
              prompt_tokens=jnp.asarray(req.prompt_tokens),
              completion_tokens=jnp.asarray(req.completion_tokens),
              pad_id=req.pad_id,
              eos_id=req.eos_id,
              stop_gradient=True,
              temperature=req.temperature,
              chunk_size=2,
              segment_ids=(
                  None
                  if req.segment_ids is None
                  else jnp.asarray(req.segment_ids)
              ),
              segment_positions=(
                  None
                  if req.segment_positions is None
                  else jnp.asarray(req.segment_positions)
              ),
          )
      )
      np.testing.assert_allclose(
          resp.per_token_logps, expected, rtol=1e-4, atol=1e-4
      )


class TrainerWorkerExecutionContextTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.events = []

    class TrackingContext:

      def __init__(self, events):
        self._events = events

      def __enter__(self):
        self._events.append("enter_ctx")
        return self

      def __exit__(self, *args):
        self._events.append("exit_ctx")

    self.ctx = TrackingContext(self.events)
    self.fake_trainer = FakeTrainer()

    def _factory():
      self.events.append("create_trainer")
      return self.fake_trainer

    self.worker = trainer_worker.TrainerWorker(
        trainer_factory=_factory,
        worker_id="trainer_ctx",
        execution_context=self.ctx,
    )

  def test_trainer_factory_runs_within_execution_context_on_initialize(self):
    self.assertEmpty(self.events)
    self.worker.initialize()
    self.assertEqual(self.events, ["enter_ctx", "create_trainer", "exit_ctx"])

  def test_all_worker_operations_run_within_execution_context(self):
    self.events.clear()
    self.worker.initialize()
    self.assertEqual(self.events, ["enter_ctx", "create_trainer", "exit_ctx"])

    # Verify per_token_logps runs inside execution_context.
    self.events.clear()
    logps_req = datatypes.LogprobsRequest(
        prompt_tokens=np.zeros((1, 0), dtype=np.int32),
        completion_tokens=np.array([[3, 4]], dtype=np.int32),
        temperature=1.0,
        pad_id=0,
        eos_id=1,
    )
    self.worker.per_token_logps(items=logps_req)
    self.assertEqual(self.events, ["enter_ctx", "exit_ctx"])

    # Verify fwd_bwd runs inside execution_context.
    self.events.clear()
    train_req = datatypes.TrainRequest(
        request_id="req-1",
        payload=datatypes.RLTrainerPayload(
            prompt_ids=np.array([[1]], dtype=np.int32),
            prompt_mask=np.ones((1, 1), dtype=np.float32),
            completion_ids=np.array([[2]], dtype=np.int32),
            completion_mask=np.array([[1]], dtype=np.float32),
            advantages=np.array([1.0], dtype=np.float32),
        ),
    )
    self.worker.fwd_bwd(request=train_req)
    self.assertEqual(self.events, ["enter_ctx", "exit_ctx"])

    # Verify update, checkpoints, weight sync, and stop operations.
    for op in [
        self.worker.update,
        lambda: self.worker.save_checkpoint(metadata={"step": 1}),
        lambda: self.worker.restore_checkpoint(step=1),
        self.worker.prepare_weight_sync,
        self.worker.release_weight_sync,
        self.worker.stop,
    ]:
      self.events.clear()
      op()
      self.assertEqual(self.events, ["enter_ctx", "exit_ctx"])

  def test_execution_context_exits_cleanly_on_exception(self):
    self.worker.initialize()
    self.events.clear()

    def _failing_save(*args, **kwargs):
      del args, kwargs
      self.events.append("save_failed")
      raise RuntimeError("Disk full")

    self.fake_trainer.save_checkpoint = _failing_save
    with self.assertRaisesRegex(RuntimeError, "Disk full"):
      self.worker.save_checkpoint(metadata={"step": 1})

    self.assertEqual(self.events, ["enter_ctx", "save_failed", "exit_ctx"])

  def test_repeated_stop_closes_trainer_once(self):
    self.worker.initialize()
    close_calls = []
    self.fake_trainer.close = lambda: close_calls.append(1)

    first = self.worker.stop()
    second = self.worker.stop()

    self.assertEqual(self.worker.state, datatypes.WorkerState.STOPPED)
    self.assertNotIn("already_stopped", first.metadata)
    self.assertTrue(second.metadata["already_stopped"])
    self.assertEqual(close_calls, [1])

  def test_stop_keeps_failure_signal_without_reclosing_if_close_raises(self):
    self.worker.initialize()
    close_calls = []

    def _failing_close():
      close_calls.append(1)
      raise RuntimeError("checkpoint flush failed")

    self.fake_trainer.close = _failing_close
    observed_aborts = []
    self.worker.set_on_stop(observed_aborts.append)

    # A clean (non-abort) stop whose close fails is escalated to an abort.
    with self.assertRaisesRegex(RuntimeError, "checkpoint flush failed"):
      self.worker.stop()

    self.assertEqual(self.worker.state, datatypes.WorkerState.STOPPED)
    self.assertEqual(close_calls, [1])
    self.assertEqual(observed_aborts, [True])
    self.assertEqual(
        self.worker.heartbeat().last_error, "checkpoint flush failed"
    )

    # Local teardown (`worker_service.stop()`) does not re-close the trainer,
    # but still surfaces the earlier failure with the abort flag latched.
    with self.assertRaisesRegex(
        RuntimeError, "failed to stop cleanly earlier: checkpoint flush failed"
    ):
      self.worker.stop()
    self.assertEqual(close_calls, [1])
    self.assertEqual(observed_aborts, [True, True])


class TrainerWorkerOTelMetricsTest(parameterized.TestCase):
  """Verifies direct OpenTelemetry SDK instruments emitted by TrainerWorker."""

  def setUp(self):
    super().setUp()
    self.reader = sdk_metrics_export.InMemoryMetricReader()
    self.meter_provider = sdk_metrics.MeterProvider(
        metric_readers=[self.reader]
    )
    self.addCleanup(self.meter_provider.shutdown)
    self.fake_trainer = FakeTrainer()
    self.worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer,
        worker_id="trainer_otel_0",
        otel_meter_provider=self.meter_provider,
    )
    self.worker.initialize()

  def test_emits_step_duration_tokens_processed_and_weight_sync_duration(self):
    # 1. Padded RLTrainerPayload with BatchAssembler-stamped token counts:
    # total=14, valid=9, padded=5.
    padded_rl_req = datatypes.TrainRequest(
        request_id="req-padded",
        payload=datatypes.RLTrainerPayload(
            prompt_ids=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
            prompt_mask=np.array(
                [[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float32
            ),
            completion_ids=np.array(
                [[10, 11, 12, 0], [20, 21, 0, 0]], dtype=np.int32
            ),
            completion_mask=np.array(
                [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=np.float32
            ),
            advantages=np.array([1.0, -1.0], dtype=np.float32),
            metadata={"valid_tokens": 9, "capacity_tokens": 14},
        ),
    )
    # 2. Sequence-packed RLTrainerPayload with BatchAssembler-stamped token
    # counts: total=6, valid=4, padded=2.
    # Also pass `mb_ready_time_s` to verify `data_fetch_delay` starvation.
    packed_rl_req = datatypes.TrainRequest(
        request_id="req-packed",
        payload=datatypes.RLTrainerPayload(
            prompt_ids=np.zeros((1, 0), dtype=np.int32),
            prompt_mask=np.zeros((1, 0), dtype=np.float32),
            completion_ids=np.array([[1, 2, 3, 4, 0, 0]], dtype=np.int32),
            completion_mask=np.array(
                [[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32
            ),
            advantages=np.zeros((1, 6), dtype=np.float32),
            segment_ids=np.array([[1, 1, 2, 2, 0, 0]], dtype=np.int32),
            segment_positions=np.array([[0, 1, 0, 1, 0, 0]], dtype=np.int32),
            metadata={"valid_tokens": 4, "capacity_tokens": 6},
        ),
        metadata={"mb_ready_time_s": 0.0},
    )
    # 3. SFTTrainerPayload without BatchAssembler token metadata: records
    # step.duration without emitting token counts.
    sft_req = datatypes.TrainRequest(
        request_id="req-sft",
        payload=datatypes.SFTTrainerPayload(
            token_ids=np.array([[1, 2, 3, 0]], dtype=np.int32),
            token_mask=np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32),
        ),
    )

    # Pin exact duration windows via `otel_metrics.perf_counter_s`:
    # - fwd_bwd #1: [1.0, 1.1] -> 0.1s
    # - fwd_bwd #2: [2.0, 2.2] -> 0.2s
    # - fwd_bwd #3: [3.0, 3.3] -> 0.3s (sum = 0.6s, count = 3)
    # - update (including get_metrics): [4.0, 4.25] -> 0.25s
    # - eval_step: [5.0, 5.15] -> 0.15s
    # - per_token_logps: [6.0, 6.4] -> 0.4s
    # - save_checkpoint: [7.0, 7.5] -> 0.5s
    # - prepare_weight_sync ("send"): [8.0, 8.35] -> 0.35s
    # - release_weight_sync ("release"): [9.0, 9.05] -> 0.05s
    orig_get_metrics = self.fake_trainer.get_metrics
    with mock.patch.object(
        otel_metrics,
        "perf_counter_s",
        side_effect=[
            1.0,
            1.1,
            2.0,
            2.2,
            3.0,
            3.3,
            4.0,
            4.25,
            5.0,
            5.15,
            6.0,
            6.4,
            7.0,
            7.5,
            8.0,
            8.35,
            9.0,
            9.05,
        ],
    ) as perf_mock:
      with mock.patch.object(
          self.fake_trainer,
          "get_metrics",
          autospec=True,
          side_effect=lambda: (
              self.assertEqual(perf_mock.call_count, 7) or orig_get_metrics()
          ),
      ):
        self.worker.fwd_bwd(request=padded_rl_req)
        self.worker.fwd_bwd(request=packed_rl_req)
        self.worker.fwd_bwd(request=sft_req)
        self.worker.update()
      self.worker.eval_step(request=sft_req)
      self.worker.per_token_logps(
          items=datatypes.LogprobsRequest(
              prompt_tokens=np.zeros((1, 0), dtype=np.int32),
              completion_tokens=np.array([[3, 4]], dtype=np.int32),
              temperature=1.0,
              pad_id=0,
              eos_id=1,
          )
      )
      self.worker.save_checkpoint(metadata={"step": 11}, step=11)
      self.worker.prepare_weight_sync()
      self.worker.release_weight_sync()

    metrics = otel_testing.collect_metrics(self.reader)

    # Verify tokens.processed: valid = 9 + 4 = 13, padded = 5 + 2 = 7
    tokens_by_type = {
        attrs["token_type"]: (val, attrs)
        for val, _, attrs in metrics["tunix.trainer.tokens.processed"]
    }
    self.assertEqual(
        tokens_by_type["valid"],
        (13, {"worker_id": "trainer_otel_0", "token_type": "valid"}),
    )
    self.assertEqual(
        tokens_by_type["padded"],
        (7, {"worker_id": "trainer_otel_0", "token_type": "padded"}),
    )

    # Verify step.duration phases and exact recorded durations
    step_by_phase = {
        attrs["phase"]: (val, cnt, attrs)
        for val, cnt, attrs in metrics["tunix.trainer.step.duration"]
    }
    self.assertCountEqual(
        step_by_phase.keys(),
        [
            "fwd_bwd",
            "optimizer_step",
            "eval_step",
            "per_token_logps",
            "checkpoint_save",
        ],
    )
    self.assertAlmostEqual(step_by_phase["fwd_bwd"][0], 0.6, places=5)
    self.assertEqual(step_by_phase["fwd_bwd"][1], 3)
    self.assertAlmostEqual(step_by_phase["optimizer_step"][0], 0.25, places=5)
    self.assertEqual(step_by_phase["optimizer_step"][1], 1)
    self.assertAlmostEqual(step_by_phase["eval_step"][0], 0.15, places=5)
    self.assertEqual(step_by_phase["eval_step"][1], 1)
    self.assertAlmostEqual(step_by_phase["per_token_logps"][0], 0.4, places=5)
    self.assertEqual(step_by_phase["per_token_logps"][1], 1)
    self.assertAlmostEqual(step_by_phase["checkpoint_save"][0], 0.5, places=5)
    self.assertEqual(step_by_phase["checkpoint_save"][1], 1)
    for _, _, phase_attrs in step_by_phase.values():
      self.assertEqual(phase_attrs["worker_id"], "trainer_otel_0")

    # Verify weight_sync.duration (both prepare "send" and "release")
    ws_by_dir = {
        attrs["direction"]: (val, cnt, attrs)
        for val, cnt, attrs in metrics["tunix.trainer.weight_sync.duration"]
    }
    self.assertCountEqual(ws_by_dir.keys(), ["send", "release"])
    self.assertAlmostEqual(ws_by_dir["send"][0], 0.35, places=5)
    self.assertEqual(ws_by_dir["send"][1], 1)
    self.assertAlmostEqual(ws_by_dir["release"][0], 0.05, places=5)
    self.assertEqual(ws_by_dir["release"][1], 1)
    for _, _, ws_attrs in ws_by_dir.values():
      self.assertEqual(ws_attrs["worker_id"], "trainer_otel_0")

    # Verify starvation.duration (`trainer_idle` on microbatches 2 & 3,
    # `data_fetch_delay` on microbatch 2 with `mb_ready_time_s`).
    starv_by_kind = {
        attrs["kind"]: (val, cnt, attrs)
        for val, cnt, attrs in metrics["tunix.trainer.starvation.duration"]
    }
    self.assertCountEqual(
        starv_by_kind.keys(), ["trainer_idle", "data_fetch_delay"]
    )
    self.assertEqual(starv_by_kind["trainer_idle"][1], 2)
    self.assertEqual(starv_by_kind["data_fetch_delay"][1], 1)
    for st_val, _, st_attrs in starv_by_kind.values():
      self.assertGreaterEqual(st_val, 0.0)
      self.assertEqual(st_attrs["worker_id"], "trainer_otel_0")

  def test_token_metadata_validation_and_failed_fwd_bwd_edge_cases(self):
    # 1. Multi-turn RLTrainerPayload with loss-masked observation tokens
    # (`completion_mask == 0` on turn 1 observation): BatchAssembler stamps
    # `valid_tokens=3, capacity_tokens=4` so observation tokens count as valid.
    first_rl_req = datatypes.TrainRequest(
        request_id="req-first-rl",
        payload=datatypes.RLTrainerPayload(
            prompt_ids=np.array([[10]], dtype=np.int32),
            prompt_mask=np.array([[1.0]], dtype=np.float32),
            completion_ids=np.array([[11, 12, 0]], dtype=np.int32),
            completion_mask=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            advantages=np.zeros((1, 3), dtype=np.float32),
            metadata={"valid_tokens": 3, "capacity_tokens": 4},
        ),
    )
    # 2. Second RLTrainerPayload: valid=4, capacity=5 (padded=1).
    second_rl_req = datatypes.TrainRequest(
        request_id="req-second-rl",
        payload=datatypes.RLTrainerPayload(
            prompt_ids=np.array([[1, 2]], dtype=np.int32),
            prompt_mask=np.array([[1.0, 1.0]], dtype=np.float32),
            completion_ids=np.array([[3, 4, 0]], dtype=np.int32),
            completion_mask=np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
            advantages=np.zeros((1, 3), dtype=np.float32),
            metadata={"valid_tokens": 4, "capacity_tokens": 5},
        ),
    )
    # Mock wall clock via `otel_metrics.now_s`:
    # - Call 1 (`first_rl_req`): `finally` arms free_time = 10.0
    # - Call 2 (failing `fwd_bwd`): `record_starvation` reads 12.0 (idle=2.0),
    #   then raises `RuntimeError`, and `finally` sets free_time = 15.0
    # - Call 3 (`second_rl_req` after resetting `_state = WorkerState.READY`
    #   since `ERROR` is terminal): `record_starvation` reads 15.5
    #   (idle = 15.5 - 15.0 = 0.5, proving the failed call's `finally` updated
    #   free_time to 15.0), and `finally` sets 16.0.
    with mock.patch.object(
        otel_metrics,
        "now_s",
        side_effect=[10.0, 12.0, 15.0, 15.5, 16.0],
    ):
      self.worker.fwd_bwd(request=first_rl_req)
      with mock.patch.object(
          self.fake_trainer,
          "fwd_bwd",
          autospec=True,
          side_effect=RuntimeError("nan loss"),
      ):
        with self.assertRaisesRegex(RuntimeError, "nan loss"):
          self.worker.fwd_bwd(request=first_rl_req)
      # `WorkerState.ERROR` has no public transition back to `INITIALIZING` or
      # `READY` (`WorkerState.can_transition_to`).
      self.worker._state = datatypes.WorkerState.READY
      self.worker.fwd_bwd(request=second_rl_req)

    # A failing `release_weight_sync` must keep `WorkerState.SYNCING` (fail
    # closed, matching HEAD) and record no `direction="release"` sample.
    self.worker.prepare_weight_sync()
    self.assertEqual(self.worker.state, datatypes.WorkerState.SYNCING)
    with mock.patch.object(
        self.fake_trainer,
        "release_weight_sync",
        autospec=True,
        side_effect=RuntimeError("release failed"),
    ):
      with self.assertRaisesRegex(RuntimeError, "release failed"):
        self.worker.release_weight_sync()
    self.assertEqual(self.worker.state, datatypes.WorkerState.SYNCING)

    metrics = otel_testing.collect_metrics(self.reader)
    tokens_by_type = {
        attrs["token_type"]: val
        for val, _, attrs in metrics["tunix.trainer.tokens.processed"]
    }
    # valid = 3 + 4 = 7; padded = 1 + 1 = 2.
    # The failed `fwd_bwd` recorded 0 tokens and 0 step durations.
    self.assertEqual(tokens_by_type, {"valid": 7, "padded": 2})
    step_by_phase = {
        attrs["phase"]: cnt
        for _, cnt, attrs in metrics["tunix.trainer.step.duration"]
    }
    self.assertEqual(step_by_phase, {"fwd_bwd": 2})
    ws_by_dir = {
        attrs["direction"]: cnt
        for _, cnt, attrs in metrics["tunix.trainer.weight_sync.duration"]
    }
    self.assertEqual(ws_by_dir, {"send": 1})
    starv_by_kind = {
        attrs["kind"]: (val, cnt)
        for val, cnt, attrs in metrics["tunix.trainer.starvation.duration"]
    }
    # Total idle = 2.0 (before failed call) + 0.5 (after failed call at 15.0)
    self.assertAlmostEqual(starv_by_kind["trainer_idle"][0], 2.5, places=5)
    self.assertEqual(starv_by_kind["trainer_idle"][1], 2)

  @parameterized.named_parameters(
      ("only_valid_tokens", {"valid_tokens": 3}, None, ValueError),
      ("only_capacity_tokens", {"capacity_tokens": 4}, None, ValueError),
      (
          "negative_valid_tokens",
          {"valid_tokens": -1, "capacity_tokens": 4},
          None,
          ValueError,
      ),
      (
          "valid_exceeds_capacity",
          {"valid_tokens": 6, "capacity_tokens": 5},
          None,
          ValueError,
      ),
      (
          "bool_valid_tokens",
          {"valid_tokens": True, "capacity_tokens": 4},
          None,
          TypeError,
      ),
      (
          "float_valid_tokens",
          {"valid_tokens": 3.5, "capacity_tokens": 4},
          None,
          TypeError,
      ),
      (
          "str_capacity_tokens",
          {"valid_tokens": 3, "capacity_tokens": "4"},
          None,
          TypeError,
      ),
      ("non_mapping_metadata", None, None, TypeError),
      ("bool_mb_ready_time_s", {}, {"mb_ready_time_s": True}, TypeError),
      (
          "str_mb_ready_time_s",
          {},
          {"mb_ready_time_s": "not-a-float"},
          TypeError,
      ),
      (
          "nan_mb_ready_time_s",
          {},
          {"mb_ready_time_s": float("nan")},
          ValueError,
      ),
      (
          "inf_mb_ready_time_s",
          {},
          {"mb_ready_time_s": float("inf")},
          ValueError,
      ),
      ("negative_mb_ready_time_s", {}, {"mb_ready_time_s": -0.5}, ValueError),
  )
  def test_extract_payload_tokens_validation_edge_cases(
      self, bad_payload_metadata, bad_req_metadata, expected_exc
  ):
    bad_req = datatypes.TrainRequest(
        request_id="req-bad",
        payload=datatypes.RLTrainerPayload(
            prompt_ids=np.array([[1]], dtype=np.int32),
            prompt_mask=np.array([[1.0]], dtype=np.float32),
            completion_ids=np.array([[2]], dtype=np.int32),
            completion_mask=np.array([[1.0]], dtype=np.float32),
            advantages=np.zeros((1, 1), dtype=np.float32),
            metadata=bad_payload_metadata,
        ),
        metadata=bad_req_metadata,
    )
    # 1. On a PENDING worker, fail-early validation (for both payload tokens
    # and `mb_ready_time_s`) runs before `_ensure_ready` so lazy `initialize()`
    # is never triggered.
    factory_calls = []
    pending_worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: (factory_calls.append(1) or self.fake_trainer),
        worker_id="pending_trainer",
        otel_meter_provider=self.meter_provider,
    )
    with self.assertRaises(expected_exc):
      pending_worker.fwd_bwd(request=bad_req)
    self.assertEmpty(factory_calls)
    self.assertEqual(pending_worker.state, datatypes.WorkerState.PENDING)

    # 2. On an already READY (and armed) worker, fail-early validation leaves
    # `state == READY`, does not invoke `trainer.fwd_bwd`, and emits no
    # partial `trainer_idle` starvation metric.
    self.worker._otel.last_free_time_s = 10.0
    with self.assertRaises(expected_exc):
      self.worker.fwd_bwd(request=bad_req)
    self.assertEmpty(self.fake_trainer.fwd_bwd_calls)
    self.assertEqual(self.worker.state, datatypes.WorkerState.READY)
    self.assertEqual(self.worker._otel.last_free_time_s, 10.0)
    self.assertEmpty(otel_testing.collect_metrics(self.reader))

  def test_token_metadata_boundary_back_compat_and_uninitialized_release(self):
    # 1. Uninitialized worker: `release_weight_sync` is a no-op and must not
    # emit a `direction="release"` sample.
    uninit_worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer,
        worker_id="trainer_otel_0",
        otel_meter_provider=self.meter_provider,
    )
    self.assertIsNone(uninit_worker.release_weight_sync())

    # 2. Boundary `valid_tokens == capacity_tokens` records `padded == 0`.
    full_req = datatypes.TrainRequest(
        request_id="req-full",
        payload=datatypes.RLTrainerPayload(
            prompt_ids=np.array([[1, 2]], dtype=np.int32),
            prompt_mask=np.array([[1.0, 1.0]], dtype=np.float32),
            completion_ids=np.array([[3, 4]], dtype=np.int32),
            completion_mask=np.array([[1.0, 1.0]], dtype=np.float32),
            advantages=np.zeros((1, 2), dtype=np.float32),
            metadata={"valid_tokens": 4, "capacity_tokens": 4},
        ),
    )
    self.worker.fwd_bwd(request=full_req)

    # 3. Boundary `valid_tokens == 0` and `np.int64` token counts:
    # records `valid += 0 + 3 = 3`, `padded += 4 + 2 = 6`.
    zero_valid_req = dataclasses.replace(
        full_req,
        request_id="req-zero-valid",
        payload=dataclasses.replace(
            full_req.payload,
            metadata={"valid_tokens": 0, "capacity_tokens": 4},
        ),
    )
    np_int_req = dataclasses.replace(
        full_req,
        request_id="req-np-int",
        payload=dataclasses.replace(
            full_req.payload,
            metadata={
                "valid_tokens": np.int64(3),
                "capacity_tokens": np.int64(5),
            },
        ),
    )
    self.worker.fwd_bwd(request=zero_valid_req)
    self.worker.fwd_bwd(request=np_int_req)

    # 4. Back-compat: `RLTrainerPayload` without token metadata keys succeeds
    # without recording extra token samples.
    legacy_rl_req = datatypes.TrainRequest(
        request_id="req-legacy-rl",
        payload=datatypes.RLTrainerPayload(
            prompt_ids=np.array([[1, 2]], dtype=np.int32),
            prompt_mask=np.array([[1.0, 1.0]], dtype=np.float32),
            completion_ids=np.array([[3, 4]], dtype=np.int32),
            completion_mask=np.array([[1.0, 1.0]], dtype=np.float32),
            advantages=np.zeros((1, 2), dtype=np.float32),
            metadata={"step": 1},
        ),
    )
    self.worker.fwd_bwd(request=legacy_rl_req)

    # 5. Negligible-overhead no-op mode when `opentelemetry` is not installed
    # (`otel_metrics.otel_api is None`): all RPCs succeed without emitting
    # metrics, while boundary validation still fails fast.
    with mock.patch.object(otel_metrics, "otel_api", None):
      disabled_worker = trainer_worker.TrainerWorker(
          trainer_factory=lambda: self.fake_trainer,
          worker_id="trainer_disabled",
          otel_meter_provider=self.meter_provider,
      )
      disabled_worker.initialize()
      self.assertFalse(disabled_worker._otel._meter.enabled)
      disabled_worker.fwd_bwd(request=full_req)
      disabled_worker.fwd_bwd(
          request=dataclasses.replace(
              full_req, metadata={"mb_ready_time_s": 1.0}
          )
      )
      disabled_worker.update()
      disabled_worker.eval_step(request=legacy_rl_req)
      disabled_worker.per_token_logps(
          items=datatypes.LogprobsRequest(
              prompt_tokens=np.zeros((1, 0), dtype=np.int32),
              completion_tokens=np.array([[3, 4]], dtype=np.int32),
              temperature=1.0,
              pad_id=0,
              eos_id=1,
          )
      )
      disabled_worker.save_checkpoint(metadata={"step": 1}, step=1)
      disabled_worker.prepare_weight_sync()
      disabled_worker.release_weight_sync()

    metrics = otel_testing.collect_metrics(self.reader)
    self.assertNotIn("tunix.trainer.weight_sync.duration", metrics)
    tokens_by_type = {
        attrs["token_type"]: val
        for val, _, attrs in metrics["tunix.trainer.tokens.processed"]
    }
    self.assertEqual(tokens_by_type, {"valid": 7, "padded": 6})

  def test_end_to_end_batch_assembler_to_trainer_worker_otel_parity(self):
    # `unbatched_1` has a leading `completion_mask == 0` (multi-turn
    # observation token at index 0), verifying that both assemblers count all 5
    # sequence tokens (`len(prompt_ids)=2 + len(completion_ids)=3`) as valid.
    unbatched_1 = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([10, 11, 12], dtype=np.int32),
        completion_mask=np.array([0.0, 1.0, 1.0], dtype=np.float32),
        advantages=np.ones(3, dtype=np.float32),
    )
    unbatched_2 = datatypes.RLTrainerPayload(
        prompt_ids=np.array([3], dtype=np.int32),
        prompt_mask=np.ones(1, dtype=np.float32),
        completion_ids=np.array([20, 21], dtype=np.int32),
        completion_mask=np.ones(2, dtype=np.float32),
        advantages=np.ones(2, dtype=np.float32),
    )
    padded_asm = batch_assembly.PaddedBatchAssembler(
        batch_size=2,
        max_prompt_length=3,
        max_response_length=4,
        pad_id=0,
        num_generations=1,
        mini_batch_size=2,
    )
    packed_asm = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        max_packed_len=12,
        pad_id=0,
        num_generations=1,
        mini_batch_size=2,
    )
    (padded_mb,) = list(padded_asm.feed([unbatched_1, unbatched_2]))
    (packed_mb,) = list(packed_asm.feed([unbatched_1, unbatched_2]))

    for idx, mb in enumerate((padded_mb, packed_mb)):
      # Mirror `DistributedRLEngine.train_step` wrapping `mb.payload` and
      # copying `mb.payload.metadata` into `TrainRequest.metadata`.
      req = datatypes.TrainRequest(
          request_id=f"req-e2e-{idx}",
          payload=mb.payload,
          metadata=dict(mb.payload.metadata),
      )
      self.worker.fwd_bwd(request=req)

    # Derive expected token counts directly from the raw fed items rather than
    # `mb.padding_stats`:
    # - Per assembler, raw valid tokens = (2 + 3) + (1 + 2) = 8 -> 16 total.
    # - Padded capacity = 2 * (3 + 4) = 14 (6 padded); packed capacity = 1 * 12
    #   = 12 (4 padded) -> 10 padded total.
    raw_valid_per_assembler = (
        unbatched_1.prompt_ids.size
        + unbatched_1.completion_ids.size
        + unbatched_2.prompt_ids.size
        + unbatched_2.completion_ids.size
    )
    expected_valid = 2 * raw_valid_per_assembler
    expected_padded = (2 * (3 + 4) - raw_valid_per_assembler) + (
        1 * 12 - raw_valid_per_assembler
    )
    self.assertEqual((expected_valid, expected_padded), (16, 10))
    metrics = otel_testing.collect_metrics(self.reader)
    tokens_by_type = {
        attrs["token_type"]: val
        for val, _, attrs in metrics["tunix.trainer.tokens.processed"]
    }
    self.assertEqual(
        tokens_by_type,
        {"valid": expected_valid, "padded": expected_padded},
    )
    starv_by_kind = {
        attrs["kind"]: cnt
        for _, cnt, attrs in metrics["tunix.trainer.starvation.duration"]
    }
    # Second microbatch records both `trainer_idle` and `data_fetch_delay`
    # using the `mb_ready_time_s` stamped by `BatchAssembler`.
    self.assertEqual(
        starv_by_kind,
        {"trainer_idle": 1, "data_fetch_delay": 1},
    )

  def test_starvation_values_and_non_train_rpc_busy_anchor(self):
    sft_req = datatypes.TrainRequest(
        request_id="req-sft",
        payload=datatypes.SFTTrainerPayload(
            token_ids=np.array([[1, 2]], dtype=np.int32),
            token_mask=np.array([[1.0, 1.0]], dtype=np.float32),
        ),
    )
    # Startup `compile` and initial weight sync before the first `fwd_bwd` must
    # NOT arm `last_free_time_s`, so MB1 emits zero cold-start starvation.
    self.worker.compile({"dummy": 1})
    self.worker.prepare_weight_sync()
    self.worker.release_weight_sync()
    self.assertIsNone(self.worker._otel.last_free_time_s)

    # Wall-clock sequence via `otel_metrics.now_s`:
    # 1. MB1 `fwd_bwd`: cold start (no starvation), `finally` arms -> 10.0
    # 2. MB2 (`mb_ready=8.0 < last_free=10.0`): start=10.5 -> idle=0.5,
    #    fetch_delay = 10.5 - max(10.0, 8.0) = 0.5; `finally` -> 11.0
    # 3. MB3 (`mb_ready=13.0 > last_free=11.0`): start=13.5 -> idle=2.5,
    #    fetch_delay = 13.5 - max(11.0, 13.0) = 0.5; `finally` -> 14.0
    # 4. MB4 (`mb_ready=20.0 > start=16.0`, future clock skew): start=16.0 ->
    #    idle=2.0, fetch_delay = max(0.0, 16.0 - 20.0) = 0.0; `finally` -> 17.0
    # 5. `per_token_logps`: `finally` updates armed free_time -> 21.0
    # 6. `compile` (post-arming): `finally` updates armed free_time -> 23.0
    # 7. `restore_checkpoint` (post-arming): `finally` updates free_time -> 25.0
    # 8. MB5 (after `per_token_logps`, `compile`, and `restore_checkpoint`, no
    #    `mb_ready_time_s`): start=25.25 -> idle = 25.25 - 25.0 = 0.25
    #    (excludes [17.0, 25.0] non-train RPC windows); `finally` -> 26.0
    # 9. `run_eval` over 2 batches: updates free_time after each batch -> 28.0,
    #    30.0
    with mock.patch.object(
        otel_metrics,
        "now_s",
        side_effect=[
            10.0,
            10.5,
            11.0,
            13.5,
            14.0,
            16.0,
            17.0,
            21.0,
            23.0,
            25.0,
            25.25,
            26.0,
            28.0,
            30.0,
        ],
    ):
      self.worker.fwd_bwd(request=sft_req)
      self.worker.fwd_bwd(
          request=dataclasses.replace(
              sft_req, metadata={"mb_ready_time_s": 8.0}
          )
      )
      self.worker.fwd_bwd(
          request=dataclasses.replace(
              sft_req, metadata={"mb_ready_time_s": 13.0}
          )
      )
      self.worker.fwd_bwd(
          request=dataclasses.replace(
              sft_req, metadata={"mb_ready_time_s": 20.0}
          )
      )
      self.worker.per_token_logps(
          items=datatypes.LogprobsRequest(
              prompt_tokens=np.zeros((1, 0), dtype=np.int32),
              completion_tokens=np.array([[3, 4]], dtype=np.int32),
              temperature=1.0,
              pad_id=0,
              eos_id=1,
          )
      )
      self.worker.compile({"dummy": 2})
      self.worker.restore_checkpoint(step=5)
      self.worker.fwd_bwd(request=sft_req)
      self.worker.run_eval(eval_ds=[sft_req.payload, sft_req.payload])
    self.assertEqual(self.worker._otel.last_free_time_s, 30.0)

    metrics = otel_testing.collect_metrics(self.reader)
    starv_by_kind = {
        attrs["kind"]: (val, cnt)
        for val, cnt, attrs in metrics["tunix.trainer.starvation.duration"]
    }
    # trainer_idle sum = 0.5 + 2.5 + 2.0 + 0.25 = 5.25 across 4 microbatches.
    self.assertAlmostEqual(starv_by_kind["trainer_idle"][0], 5.25, places=5)
    self.assertEqual(starv_by_kind["trainer_idle"][1], 4)
    # data_fetch_delay sum = 0.5 + 0.5 + 0.0 = 1.0 across 3 valid samples.
    self.assertAlmostEqual(starv_by_kind["data_fetch_delay"][0], 1.0, places=5)
    self.assertEqual(starv_by_kind["data_fetch_delay"][1], 3)

    step_by_phase = {
        attrs["phase"]: cnt
        for _, cnt, attrs in metrics["tunix.trainer.step.duration"]
    }
    self.assertEqual(step_by_phase["eval_step"], 2)

  @parameterized.named_parameters(
      ("fwd_bwd", "fwd_bwd"),
      ("update", "update"),
      ("eval_step", "eval_step"),
      ("run_eval", "run_eval"),
      ("run_eval_native", "run_eval_native"),
      ("per_token_logps", "per_token_logps"),
      ("save_checkpoint", "save_checkpoint"),
      ("restore_checkpoint", "restore_checkpoint"),
      ("compile", "compile"),
      ("prepare_and_release_weight_sync", "prepare_and_release_weight_sync"),
      ("failed_update", "failed_update"),
  )
  def test_post_arming_busy_anchor_refresh_across_all_rpcs(self, rpc_case):
    sft_req = datatypes.TrainRequest(
        request_id="req-sft",
        payload=datatypes.SFTTrainerPayload(
            token_ids=np.array([[1, 2]], dtype=np.int32),
            token_mask=np.array([[1.0, 1.0]], dtype=np.float32),
        ),
    )
    self.worker._otel.last_free_time_s = 10.0
    if rpc_case == "fwd_bwd":
      with mock.patch.object(otel_metrics, "now_s", side_effect=[11.0, 42.0]):
        self.worker.fwd_bwd(request=sft_req)
      self.assertEqual(self.worker._otel.last_free_time_s, 42.0)
      return
    if rpc_case == "prepare_and_release_weight_sync":
      with mock.patch.object(otel_metrics, "now_s", side_effect=[35.0, 42.0]):
        self.worker.prepare_weight_sync()
        self.assertEqual(self.worker._otel.last_free_time_s, 35.0)
        self.worker.release_weight_sync()
      self.assertEqual(self.worker._otel.last_free_time_s, 42.0)
      return
    if rpc_case == "failed_update":
      with (
          mock.patch.object(otel_metrics, "now_s", return_value=42.0),
          mock.patch.object(
              self.fake_trainer,
              "update",
              autospec=True,
              side_effect=RuntimeError("update failed"),
          ),
      ):
        with self.assertRaisesRegex(RuntimeError, "update failed"):
          self.worker.update()
      self.assertEqual(self.worker._otel.last_free_time_s, 42.0)
      return

    with mock.patch.object(otel_metrics, "now_s", return_value=42.0):
      if rpc_case == "update":
        self.worker.update()
      elif rpc_case == "eval_step":
        self.worker.eval_step(request=sft_req)
      elif rpc_case == "run_eval":
        self.worker.run_eval(eval_ds=[sft_req.payload])
      elif rpc_case == "run_eval_native":
        with mock.patch.object(
            self.fake_trainer, "run_eval", create=True
        ) as native_eval:
          self.worker.run_eval(eval_ds=[sft_req.payload])
          native_eval.assert_called_once_with([sft_req.payload])
      elif rpc_case == "per_token_logps":
        self.worker.per_token_logps(
            items=datatypes.LogprobsRequest(
                prompt_tokens=np.zeros((1, 0), dtype=np.int32),
                completion_tokens=np.array([[3, 4]], dtype=np.int32),
                temperature=1.0,
                pad_id=0,
                eos_id=1,
            )
        )
      elif rpc_case == "save_checkpoint":
        self.worker.save_checkpoint(metadata={"step": 1}, step=1)
      elif rpc_case == "restore_checkpoint":
        self.worker.restore_checkpoint(step=1)
      elif rpc_case == "compile":
        self.worker.compile({"dummy": 1})
    self.assertEqual(self.worker._otel.last_free_time_s, 42.0)


class NonJaxTrainerWorkerTest(absltest.TestCase):

  def test_non_jax_trainer_scores_logps_without_jax(self):
    class PureNumpyTrainer(abstract_trainer.AbstractTrainer):

      def __init__(self, config: Any = None):
        del config
        self.policy_version = 7

      def with_loss_fn(self, loss_fn, has_aux=False):
        del loss_fn, has_aux
        return self

      def with_gen_model_input_fn(self, gen_model_input_fn):
        del gen_model_input_fn
        return self

      def compile(self, dummy_data=None):
        del dummy_data

      def fwd_bwd(self, payload):
        del payload

      def update(self):
        return 1

      def eval_step(self, payload):
        del payload

      def with_fwd_only_fn(self, fn):
        del fn
        return self

      def fwd_only(self, payload):
        raise AssertionError("TrainerWorker must not call JAX fwd_only")

      def compute_per_token_logps(
          self,
          prompt_tokens,
          completion_tokens,
          *,
          pad_id,
          eos_id,
          temperature=1.0,
          chunk_size=0,
          segment_ids=None,
          segment_positions=None,
          routed_experts=None,
      ):
        del prompt_tokens, pad_id, eos_id, chunk_size
        del segment_ids, segment_positions, routed_experts
        return -float(temperature) * np.ones(
            completion_tokens.shape, dtype=np.float32
        )

      def save_checkpoint(self, metadata=None, *, step=None, force=False):
        del metadata, step, force

      def restore_checkpoint(self, *, step=None):
        del step
        return {}

      def prepare_weight_sync(self, *, sync_request=None):
        del sync_request
        return None

      def get_metrics(self):
        return {}

      def close(self):
        pass

    worker = trainer_worker.TrainerWorker(
        trainer_factory=PureNumpyTrainer,
        worker_id="torch_trainer_0",
        logps_micro_batch_size=2,
    )
    worker.initialize()
    req = datatypes.LogprobsRequest(
        request_id="req-pure-np",
        prompt_tokens=np.ones((3, 2), dtype=np.int32),
        completion_tokens=np.ones((3, 4), dtype=np.int32),
        pad_id=0,
        eos_id=2,
        temperature=0.5,
    )
    resp = worker.per_token_logps(req)
    self.assertEqual(resp.model_version, 7)
    self.assertIsInstance(resp.per_token_logps, np.ndarray)
    np.testing.assert_allclose(
        resp.per_token_logps,
        np.full((3, 4), -0.5, dtype=np.float32),
    )
    worker.stop()

  def test_importing_trainer_worker_keeps_ml_frameworks_unloaded(self):
    forbidden = (
        "jax",
        "flax",
        "optax",
        "torch",
        "tpu_raiden",
        "tpu_sync",
    )
    with mock.patch.dict(sys.modules):
      for name in list(sys.modules):
        if (
            name == "tunix"
            or name.startswith("tunix.")
            or name.endswith(".tunix")
            or ".tunix." in name
        ):
          sys.modules.pop(name, None)
        elif any(
            name == pkg or name.startswith(f"{pkg}.") or f".{pkg}" in name
            for pkg in forbidden
        ):
          sys.modules[name] = None
      for pkg in forbidden:
        sys.modules[pkg] = None
      mod = importlib.import_module("tunix.dist_rl.worker.trainer_worker")
      self.assertTrue(hasattr(mod, "TrainerWorker"))


if __name__ == "__main__":
  absltest.main()
