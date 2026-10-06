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
per-token log-prob scorer run through AbstractTrainer.model_scope.
"""

import contextlib
from typing import Any
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.train import abstract_trainer
from tunix.experimental.worker import trainer_worker
from tunix.rl import common as rl_common
from tunix.tests import test_common as tc


class FakeTrainer(abstract_trainer.AbstractTrainer):

  def __init__(self):
    self.fwd_bwd_calls = []
    self.eval_step_calls = []
    self.model_scope_calls = []
    self.model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    self.policy_version = 3
    self.step_count = 10
    self.target_state = None
    self.gen_model_input_fn = None
    self._checkpoint_dir = None

  @property
  def checkpoint_dir(self) -> str | None:
    return self._checkpoint_dir

  def compile(self, dummy_data=None):
    pass

  def with_loss_fn(self, loss_fn, has_aux=False):
    pass

  def with_gen_model_input_fn(self, gen_model_input_fn):
    self.gen_model_input_fn = gen_model_input_fn

  def fwd_bwd(self, payload, **kwargs):
    self.fwd_bwd_calls.append((payload, kwargs))

  def update(self, **kwargs):
    self.step_count += 1
    return self.step_count

  def eval_step(self, payload, **kwargs):
    self.eval_step_calls.append((payload, kwargs))

  @contextlib.contextmanager
  def model_scope(self, *args, **kwargs):
    self.model_scope_calls.append((args, kwargs))
    yield self.model, args, kwargs

  def save_checkpoint(self, metadata, **kwargs):
    pass

  def restore_checkpoint(self, **kwargs):
    return {}

  def get_metrics(self):
    return {"loss": 0.25}

  def prepare_weight_sync(self, **kwargs):
    pass

  def set_target_state(self, target_state: Any) -> None:
    self.target_state = target_state

  def close(self):
    pass


class TrainerWorkerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.fake_trainer = FakeTrainer()
    self.worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer,
        worker_id="trainer_0",
    )
    self.worker.initialize()

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
    self.assertNotIn("updated", resp.metadata)
    self.assertLen(self.fake_trainer.fwd_bwd_calls, 1)
    self.assertIs(self.fake_trainer.fwd_bwd_calls[0][0], payload)

  def test_fwd_bwd_with_apply_optimizer_runs_update_in_same_call(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[1, 2]], dtype=np.int32),
        prompt_mask=np.ones((1, 2), dtype=np.float32),
        completion_ids=np.array([[3, 4]], dtype=np.int32),
        completion_mask=np.ones((1, 2), dtype=np.float32),
        advantages=np.array([1.0], dtype=np.float32),
    )
    # First microbatch (apply_optimizer=False): strips caller-supplied
    # updated/train_step keys and does not advance step_count.
    req_0 = datatypes.TrainRequest(
        request_id="req-accum-0",
        payload=payload,
        metadata={"batch_id": "b0", "updated": True, "train_step": 999},
    )
    resp_0 = self.worker.fwd_bwd(request=req_0, apply_optimizer=False)
    self.assertEqual(resp_0.request_id, "req-accum-0")
    self.assertNotIn("updated", resp_0.metadata)
    self.assertNotIn("train_step", resp_0.metadata)
    self.assertEqual(self.fake_trainer.step_count, 10)

    # Final microbatch (apply_optimizer=True): runs fwd_bwd + update in one RPC.
    request = datatypes.TrainRequest(
        request_id="req-fused-1",
        payload=payload,
        metadata={"batch_id": "b1"},
    )
    resp = self.worker.fwd_bwd(request=request, apply_optimizer=True)

    self.assertIsInstance(resp, datatypes.Response)
    self.assertEqual(resp.request_id, "req-fused-1")
    self.assertEqual(resp.metadata["worker_id"], "trainer_0")
    self.assertEqual(resp.metadata["batch_id"], "b1")
    self.assertTrue(resp.metadata["queued"])
    self.assertTrue(resp.metadata["updated"])
    self.assertEqual(resp.metadata["train_step"], 11)
    self.assertEqual(self.fake_trainer.step_count, 11)
    self.assertLen(self.fake_trainer.fwd_bwd_calls, 2)
    self.assertIs(self.fake_trainer.fwd_bwd_calls[1][0], payload)

  def test_fwd_bwd_with_apply_optimizer_update_failure_marks_worker_error(
      self,
  ):
    """A failing fused update() is reported as such and puts the worker in ERROR."""

    def _failing_update(**kwargs):
      del kwargs
      raise ValueError("optimizer exploded")

    self.fake_trainer.update = _failing_update
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[1, 2]], dtype=np.int32),
        prompt_mask=np.ones((1, 2), dtype=np.float32),
        completion_ids=np.array([[3, 4]], dtype=np.int32),
        completion_mask=np.ones((1, 2), dtype=np.float32),
        advantages=np.array([1.0], dtype=np.float32),
    )
    request = datatypes.TrainRequest(
        request_id="req-fused-err",
        payload=payload,
        metadata={"batch_id": "b2"},
    )

    # The error comes back on a fwd_bwd call, so it must name the optimizer
    # step and keep the original exception type and message.
    with self.assertRaisesRegex(
        RuntimeError,
        r"optimizer update\(\) failed inside fwd_bwd\(apply_optimizer=True\)"
        r".*ValueError: optimizer exploded",
    ) as cm:
      self.worker.fwd_bwd(request=request, apply_optimizer=True)
    self.assertIsInstance(cm.exception.__cause__, ValueError)

    # fwd_bwd ran; update failed after it.
    self.assertLen(self.fake_trainer.fwd_bwd_calls, 1)
    self.assertEqual(self.worker.state, datatypes.WorkerState.ERROR)
    self.assertEqual(self.worker.heartbeat().last_error, str(cm.exception))
    with self.assertRaisesRegex(RuntimeError, "not ready"):
      self.worker.fwd_bwd(request=request, apply_optimizer=False)
    self.assertLen(self.fake_trainer.fwd_bwd_calls, 1)

  def test_fwd_bwd_failure_with_apply_optimizer_is_not_blamed_on_update(self):
    """A failing forward/backward is re-raised unchanged; update() never runs."""
    update_calls = []

    def _failing_fwd_bwd(payload, **kwargs):
      del payload, kwargs
      raise ValueError("bad microbatch")

    def _recording_update(**kwargs):
      update_calls.append(kwargs)
      return 0

    self.fake_trainer.fwd_bwd = _failing_fwd_bwd
    self.fake_trainer.update = _recording_update
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[1, 2]], dtype=np.int32),
        prompt_mask=np.ones((1, 2), dtype=np.float32),
        completion_ids=np.array([[3, 4]], dtype=np.int32),
        completion_mask=np.ones((1, 2), dtype=np.float32),
        advantages=np.array([1.0], dtype=np.float32),
    )
    request = datatypes.TrainRequest(
        request_id="req-fwd-err", payload=payload, metadata={}
    )

    with self.assertRaisesRegex(ValueError, "^bad microbatch$"):
      self.worker.fwd_bwd(request=request, apply_optimizer=True)

    self.assertEmpty(update_calls)
    self.assertEqual(self.worker.state, datatypes.WorkerState.ERROR)
    self.assertEqual(self.worker.heartbeat().last_error, "bad microbatch")

  def test_fwd_bwd_does_not_leak_control_kwargs_to_trainer(self):
    """`apply_optimizer`/`skip_jit` stay in the worker; update runs after fwd_bwd."""
    update_seen_fwd_bwd_calls = []
    original_update = self.fake_trainer.update

    def _recording_update(**kwargs):
      update_seen_fwd_bwd_calls.append(len(self.fake_trainer.fwd_bwd_calls))
      return original_update(**kwargs)

    self.fake_trainer.update = _recording_update
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[1, 2]], dtype=np.int32),
        prompt_mask=np.ones((1, 2), dtype=np.float32),
        completion_ids=np.array([[3, 4]], dtype=np.int32),
        completion_mask=np.ones((1, 2), dtype=np.float32),
        advantages=np.array([1.0], dtype=np.float32),
    )
    request = datatypes.TrainRequest(
        request_id="req-kw", payload=payload, metadata={}
    )

    self.worker.fwd_bwd(request=request, apply_optimizer=False, skip_jit=True)
    self.worker.fwd_bwd(
        request=request,
        apply_optimizer=True,
        skip_jit=True,
        cache_nnx_graph=False,
    )

    self.assertEqual(self.fake_trainer.fwd_bwd_calls[0][1], {})
    self.assertEqual(
        self.fake_trainer.fwd_bwd_calls[1][1], {"cache_nnx_graph": False}
    )
    # update() ran exactly once, after the second fwd_bwd.
    self.assertEqual(update_seen_fwd_bwd_calls, [2])

  def test_fwd_bwd_with_apply_optimizer_goes_through_worker_update(self):
    """The fused update is this worker's update(), not a bare trainer call.

    Anything the worker does around an update RPC (error state, step
    instrumentation) must apply to the fused optimizer step as well.
    """
    worker_update_calls = []
    original_worker_update = self.worker.update

    def _recording_worker_update(**kwargs):
      worker_update_calls.append(
          (len(self.fake_trainer.fwd_bwd_calls), dict(kwargs))
      )
      return original_worker_update(**kwargs)

    # An instance attribute shadows the method, so self.update() resolves here.
    self.worker.update = _recording_worker_update
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[1, 2]], dtype=np.int32),
        prompt_mask=np.ones((1, 2), dtype=np.float32),
        completion_ids=np.array([[3, 4]], dtype=np.int32),
        completion_mask=np.ones((1, 2), dtype=np.float32),
        advantages=np.array([1.0], dtype=np.float32),
    )
    request = datatypes.TrainRequest(
        request_id="req-via-worker", payload=payload, metadata={}
    )

    self.worker.fwd_bwd(request=request, apply_optimizer=False)
    self.assertEmpty(worker_update_calls)
    resp = self.worker.fwd_bwd(request=request, apply_optimizer=True)

    # Once, with no kwargs, after this RPC's fwd_bwd.
    self.assertEqual(worker_update_calls, [(2, {})])
    self.assertEqual(resp.metadata["train_step"], 11)
    self.assertEqual(self.fake_trainer.step_count, 11)

  def test_fused_update_waits_for_the_update_token_and_reports_update_ready(
      self,
  ):
    """With an update output to wait on, the fused call returns once it is ready."""
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[1, 2]], dtype=np.int32),
        prompt_mask=np.ones((1, 2), dtype=np.float32),
        completion_ids=np.array([[3, 4]], dtype=np.int32),
        completion_mask=np.ones((1, 2), dtype=np.float32),
        advantages=np.array([1.0], dtype=np.float32),
    )
    # A caller-supplied `update_ready` must never be echoed back as the
    # worker's own claim.
    request = datatypes.TrainRequest(
        request_id="req-token", payload=payload, metadata={"update_ready": True}
    )
    events = []
    real_block_until_ready = trainer_worker.jax.block_until_ready

    def _recording_block_until_ready(x):
      events.append(("block_until_ready", x))
      return real_block_until_ready(x)

    with mock.patch.object(
        trainer_worker.jax,
        "block_until_ready",
        side_effect=_recording_block_until_ready,
    ):
      # A trainer without a token: nothing to wait on, so no claim.
      resp = self.worker.fwd_bwd(request=request, apply_optimizer=True)
      self.assertNotIn("update_ready", resp.metadata)
      self.assertEmpty(events)

      # A trainer whose update publishes a token: waited on after the update.
      original_update = self.fake_trainer.update
      token = jnp.array(1.0)

      def _update_with_token(**kwargs):
        events.append("update")
        self.fake_trainer.last_update_token = token
        return original_update(**kwargs)

      self.fake_trainer.update = _update_with_token
      resp = self.worker.fwd_bwd(request=request, apply_optimizer=True)

    self.assertEqual(events, ["update", ("block_until_ready", token)])
    self.assertTrue(resp.metadata["update_ready"])
    self.assertTrue(resp.metadata["updated"])

  def test_fused_update_token_failure_is_reported_as_the_optimizer_step(self):
    """An update that fails on device surfaces through the token wait as the optimizer step."""
    self.fake_trainer.last_update_token = jnp.array(1.0)
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[1, 2]], dtype=np.int32),
        prompt_mask=np.ones((1, 2), dtype=np.float32),
        completion_ids=np.array([[3, 4]], dtype=np.int32),
        completion_mask=np.ones((1, 2), dtype=np.float32),
        advantages=np.array([1.0], dtype=np.float32),
    )
    request = datatypes.TrainRequest(
        request_id="req-token-err", payload=payload, metadata={}
    )

    with mock.patch.object(
        trainer_worker.jax,
        "block_until_ready",
        side_effect=ValueError("device lost"),
    ):
      with self.assertRaisesRegex(
          RuntimeError,
          r"optimizer update\(\) failed inside fwd_bwd\(apply_optimizer=True\)"
          r".*ValueError: device lost",
      ):
        self.worker.fwd_bwd(request=request, apply_optimizer=True)

    self.assertEqual(self.worker.state, datatypes.WorkerState.ERROR)

  def test_fused_update_is_called_without_the_fwd_bwd_kwargs(self):
    """Per-call kwargs are for the trainer's fwd_bwd only; the fused update() is called bare."""
    update_kwargs = []
    original_update = self.fake_trainer.update

    def _recording_update(**kwargs):
      update_kwargs.append(kwargs)
      return original_update()

    self.fake_trainer.update = _recording_update
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[1, 2]], dtype=np.int32),
        prompt_mask=np.ones((1, 2), dtype=np.float32),
        completion_ids=np.array([[3, 4]], dtype=np.int32),
        completion_mask=np.ones((1, 2), dtype=np.float32),
        advantages=np.array([1.0], dtype=np.float32),
    )
    request = datatypes.TrainRequest(
        request_id="req-kw-update", payload=payload, metadata={}
    )

    self.worker.fwd_bwd(
        request=request, apply_optimizer=True, cache_nnx_graph=False
    )

    self.assertEqual(
        self.fake_trainer.fwd_bwd_calls[-1][1], {"cache_nnx_graph": False}
    )
    self.assertEqual(update_kwargs, [{}])

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
    self.assertIs(self.fake_trainer.eval_step_calls[0][0], payload)

  def test_update_returns_step_count(self):
    step = self.worker.update()
    self.assertEqual(step, 11)

  def test_save_checkpoint_returns_checkpoint_path_from_checkpoint_dir(self):
    resp_empty = self.worker.save_checkpoint(metadata={"step": 5})
    self.assertTrue(resp_empty.metadata["checkpoint_saved"])
    self.assertEqual(resp_empty.metadata["checkpoint_path"], "")

    self.fake_trainer._checkpoint_dir = "gs://bucket/checkpoints"
    resp = self.worker.save_checkpoint(metadata={"step": 5})
    self.assertTrue(resp.metadata["checkpoint_saved"])
    self.assertEqual(
        resp.metadata["checkpoint_path"],
        "gs://bucket/checkpoints/5/model_params",
    )

  def test_restore_checkpoint_transitions_to_error_state_on_failure(self):
    def _failing_restore(**kwargs):
      del kwargs
      raise RuntimeError("CheckpointRestoreError: corrupted step 2")

    self.fake_trainer.restore_checkpoint = _failing_restore
    with self.assertRaisesRegex(RuntimeError, "corrupted step 2"):
      self.worker.restore_checkpoint(step=2)

    self.assertEqual(self.worker.state, datatypes.WorkerState.ERROR)
    self.assertIn("corrupted step 2", self.worker.heartbeat().last_error)

  def test_set_target_state_configures_trainer(self):
    target_state = {"params": np.zeros((4, 4))}
    resp = self.worker.set_target_state(target_state=target_state)

    self.assertIsInstance(resp, datatypes.Response)
    self.assertTrue(resp.metadata["target_state_configured"])
    self.assertEqual(self.fake_trainer.target_state, target_state)

  def test_set_target_state_raises_when_trainer_unsupported(self):
    self.fake_trainer.set_target_state = None
    with self.assertRaises(AttributeError):
      self.worker.set_target_state(target_state={"params": 1})

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
    self.assertLen(self.fake_trainer.model_scope_calls, 1)
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

    self.assertLen(self.fake_trainer.model_scope_calls, 2)
    first_args, _ = self.fake_trainer.model_scope_calls[0]
    second_args, _ = self.fake_trainer.model_scope_calls[1]
    np.testing.assert_array_equal(first_args[1], request.completion_tokens[:2])
    np.testing.assert_array_equal(second_args[1], request.completion_tokens[2:])
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

    _, kwargs = self.fake_trainer.model_scope_calls[0]
    self.assertEqual(kwargs["temperature"], 0.7)
    self.assertEqual(kwargs["chunk_size"], 3)
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

  def test_with_gen_model_input_fn_injects_logps_chunk_size(self):
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer,
        worker_id="trainer_chunked",
        logps_chunk_size=256,
    )
    worker.initialize()
    worker.with_gen_model_input_fn(lambda payload: {"train_example": payload})

    configured_fn = self.fake_trainer.gen_model_input_fn
    self.assertIsNotNone(configured_fn)
    mapped = configured_fn("dummy_payload")
    self.assertEqual(mapped["train_example"], "dummy_payload")
    self.assertEqual(mapped["compute_logps_chunk_size"], 256)

    # Explicit caller value should not be overwritten by setdefault.
    worker.with_gen_model_input_fn(
        lambda payload: {
            "train_example": payload,
            "compute_logps_chunk_size": 64,
        }
    )
    self.assertEqual(
        self.fake_trainer.gen_model_input_fn("p")["compute_logps_chunk_size"],
        64,
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
    self.assertEmpty(self.fake_trainer.model_scope_calls)
    self.assertEqual(self.worker.state, datatypes.WorkerState.ERROR)

  def test_per_token_logps_requires_pad_and_eos(self):
    request = datatypes.LogprobsRequest(
        prompt_tokens=np.zeros((1, 0), dtype=np.int32),
        completion_tokens=np.array([[3, 4]], dtype=np.int32),
        temperature=1.0,
    )
    with self.assertRaises(ValueError):
      self.worker.per_token_logps(items=request)
    self.assertEmpty(self.fake_trainer.model_scope_calls)


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

  def test_trainer_factory_runs_within_execution_context(self):
    self.assertEqual(self.events, [])
    self.worker.initialize()
    self.assertEqual(self.events, ["enter_ctx", "create_trainer", "exit_ctx"])

  def test_all_worker_operations_run_within_execution_context(self):
    self.worker.initialize()
    self.events.clear()

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


if __name__ == "__main__":
  absltest.main()
