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

"""Tests for the frozen-model InferenceWorker wrapper."""

import threading
from absl.testing import absltest
import cloudpickle
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import rpc_utils
from tunix.dist_rl.worker import inference_worker as inference_lib

WorkerState = datatypes.WorkerState


class _StubCore:
  """Deterministic, row-independent stand-in for the real inference core."""

  def __init__(self):
    self.calls = 0
    self.last_prompt_tokens = None
    self.last_completion_tokens = None
    self.last_segment_ids = None
    self.last_segment_positions = None
    self.chunk_shapes = []

  def get_ref_per_token_logps(
      self,
      prompt_tokens,
      completion_tokens,
      pad_id,
      eos_id,
      temperature=1.0,
      segment_ids=None,
      segment_positions=None,
  ):
    del pad_id, eos_id
    self.calls += 1
    self.last_prompt_tokens = np.asarray(prompt_tokens, dtype=np.int32)
    self.last_completion_tokens = np.asarray(completion_tokens, dtype=np.int32)
    self.last_segment_ids = (
        None if segment_ids is None else np.asarray(segment_ids, dtype=np.int32)
    )
    self.last_segment_positions = (
        None
        if segment_positions is None
        else np.asarray(segment_positions, dtype=np.int32)
    )
    self.chunk_shapes.append((
        self.last_prompt_tokens.shape,
        self.last_completion_tokens.shape,
        None if self.last_segment_ids is None else self.last_segment_ids.shape,
        None
        if self.last_segment_positions is None
        else self.last_segment_positions.shape,
    ))
    offset = (
        0.0
        if segment_positions is None
        else jnp.asarray(segment_positions, dtype=jnp.float32) * 0.1
    )
    return (
        jnp.asarray(completion_tokens, dtype=jnp.float32) * temperature + offset
    )

  def get_rewards(self, prompt_tokens, completion_tokens, pad_id, eos_id):
    del prompt_tokens, pad_id, eos_id
    self.calls += 1
    return jnp.asarray(completion_tokens, dtype=jnp.float32).sum(axis=1)


def _worker(core=None, chunk_size=None):
  return inference_lib.InferenceWorker(
      core if core is not None else _StubCore(),
      worker_id="test_worker_1",
      pad_id=0,
      eos_id=1,
      model_version=3,
      chunk_size=chunk_size,
  )


def _logprobs_request(batch=4, **overrides):
  kwargs = dict(
      request_id="r1",
      prompt_tokens=np.ones((batch, 2), dtype=np.int32),
      completion_tokens=np.arange(batch * 3, dtype=np.int32).reshape(batch, 3),
      temperature=1.0,
  )
  kwargs.update(overrides)
  return datatypes.LogprobsRequest(**kwargs)


class InferenceWorkerTest(absltest.TestCase):

  def test_compute_logps_shape_and_echo(self):
    result = _worker().compute_logps(_logprobs_request(batch=4))

    self.assertEqual(result.request_id, "r1")
    self.assertEqual(result.model_version, 3)
    self.assertEqual(result.per_token_logps.shape, (4, 3))
    self.assertEqual(result.per_token_logps.dtype, np.float32)

  def test_compute_logps_applies_temperature(self):
    req = _logprobs_request(batch=2, temperature=2.0)

    result = _worker().compute_logps(req)

    np.testing.assert_allclose(
        result.per_token_logps, req.completion_tokens.astype(np.float32) * 2.0
    )

  def test_chunking_matches_single_pass(self):
    core_single = _StubCore()
    single = _worker(core_single).compute_logps(_logprobs_request(batch=6))

    core_chunked = _StubCore()
    chunked = _worker(core_chunked, chunk_size=2).compute_logps(
        _logprobs_request(batch=6)
    )

    np.testing.assert_array_equal(
        single.per_token_logps, chunked.per_token_logps
    )
    self.assertEqual(core_single.calls, 1)  # single pass
    self.assertEqual(core_chunked.calls, 3)  # 6 rows / 2 = 3 micro-batches

  def test_per_token_logps_uses_padded_batch_without_repadding(self):
    core = _StubCore()
    batch = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[0, 0, 5, 6], [0, 7, 8, 9]], dtype=np.int32),
        prompt_mask=np.array([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=np.float32),
        completion_ids=np.array([[10, 11, 0], [12, 0, 0]], dtype=np.int32),
        completion_mask=np.array([[1, 1, 0], [1, 0, 0]], dtype=np.float32),
        advantages=np.ones((2, 3), dtype=np.float32),
        ref_per_token_logps=None,
        old_per_token_logps=None,
    )

    result = _worker(core).per_token_logps(batch, temperature=2.0)

    np.testing.assert_array_equal(core.last_prompt_tokens, batch.prompt_ids)
    np.testing.assert_array_equal(
        core.last_completion_tokens, batch.completion_ids
    )
    np.testing.assert_allclose(
        result, batch.completion_ids.astype(np.float32) * 2.0
    )
    self.assertIsNone(core.last_segment_ids)
    self.assertIsNone(core.last_segment_positions)

  def test_per_token_logps_forwards_sequence_packing_segment_metadata(self):
    core = _StubCore()
    batch = datatypes.RLTrainerPayload(
        prompt_ids=np.zeros((2, 0), dtype=np.int32),
        prompt_mask=np.zeros((2, 0), dtype=np.float32),
        completion_ids=np.array(
            [[3, 4, 5, 6, 7, 8], [9, 10, 11, 0, 0, 0]], dtype=np.int32
        ),
        completion_mask=np.array(
            [[0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 0, 0]], dtype=np.float32
        ),
        advantages=np.ones((2, 6), dtype=np.float32),
        segment_ids=np.array(
            [[1, 1, 1, 2, 2, 2], [1, 1, 1, 0, 0, 0]], dtype=np.int32
        ),
        segment_positions=np.array(
            [[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 0, 0]], dtype=np.int32
        ),
    )

    result = _worker(core).per_token_logps(batch, temperature=1.5)

    self.assertEqual(core.last_prompt_tokens.shape, (2, 0))
    np.testing.assert_array_equal(
        core.last_completion_tokens, batch.completion_ids
    )
    np.testing.assert_array_equal(core.last_segment_ids, batch.segment_ids)
    np.testing.assert_array_equal(
        core.last_segment_positions, batch.segment_positions
    )
    expected = (
        batch.completion_ids.astype(np.float32) * 1.5
        + batch.segment_positions.astype(np.float32) * 0.1
    )
    np.testing.assert_allclose(result, expected)

  def test_compute_logps_chunks_segment_metadata_along_batch_axis(self):
    segment_ids = np.tile(np.array([[1, 1, 2]], dtype=np.int32), (6, 1))
    segment_positions = np.tile(np.array([[0, 1, 0]], dtype=np.int32), (6, 1))
    req = _logprobs_request(
        batch=6,
        prompt_tokens=np.zeros((6, 0), dtype=np.int32),
        segment_ids=segment_ids,
        segment_positions=segment_positions,
    )

    core_single = _StubCore()
    single = _worker(core_single).compute_logps(req)

    core_chunked = _StubCore()
    chunked = _worker(core_chunked, chunk_size=2).compute_logps(req)

    np.testing.assert_allclose(single.per_token_logps, chunked.per_token_logps)
    self.assertEqual(core_single.calls, 1)
    self.assertEqual(core_chunked.calls, 3)
    self.assertEqual(
        core_chunked.chunk_shapes,
        [((2, 0), (2, 3), (2, 3), (2, 3))] * 3,
    )

  def test_compute_logps_rejects_half_specified_segment_metadata(self):
    req = _logprobs_request(
        batch=2,
        segment_ids=np.ones((2, 3), dtype=np.int32),
        segment_positions=None,
    )
    result = _worker().compute_logps(req)
    self.assertIsNotNone(result.error)
    self.assertEqual(result.error.error_type, "ValueError")
    self.assertIn("segment_ids and segment_positions", result.error.message)

  def test_per_token_logps_rejects_unpadded_trajectory_items(self):
    item = datatypes.TrajectoryItem(
        group_index=0,
        prompt_id="g1",
        start_step=0,
        traj=datatypes.Trajectory(
            reward=1.0,
            status=datatypes.TrajectoryStatus.SUCCEEDED,
        ),
        prompt_tokens=np.array([5, 6], dtype=np.int32),
        completion_tokens=np.array([7, 8], dtype=np.int32),
        action_mask=np.array([1, 1], dtype=np.float32),
    )
    with self.assertRaisesRegex(TypeError, "BatchAssembler"):
      _worker().per_token_logps([item])

  def test_compute_logps_rejects_non_reference_role(self):
    result = _worker().compute_logps(_logprobs_request(model_role="policy"))
    self.assertIsNotNone(result.error)
    self.assertEqual(result.error.error_type, "NotImplementedError")

  def test_compute_logps_rejects_non_positive_temperature(self):
    req = _logprobs_request(temperature=0.0)
    result = _worker().compute_logps(req)
    self.assertIsNotNone(result.error)
    self.assertEqual(result.error.error_type, "ValueError")
    self.assertIn("Temperature must be strictly positive", result.error.message)

  def test_score_shape_and_echo(self):
    req = datatypes.ScoreRequest(
        request_id="s1",
        prompt_tokens=np.ones((3, 2), dtype=np.int32),
        completion_tokens=np.arange(9, dtype=np.int32).reshape(3, 3),
    )

    result = _worker().score(req)

    self.assertEqual(result.request_id, "s1")
    self.assertEqual(result.model_version, 3)
    self.assertEqual(result.scores.shape, (3,))
    self.assertEqual(result.scores.dtype, np.float32)

  def test_score_rejects_non_reward_role(self):
    req = datatypes.ScoreRequest(
        request_id="s1",
        prompt_tokens=np.ones((3, 2), dtype=np.int32),
        completion_tokens=np.arange(9, dtype=np.int32).reshape(3, 3),
        model_role="policy",
    )
    result = _worker().score(req)
    self.assertIsNotNone(result.error)
    self.assertEqual(result.error.error_type, "NotImplementedError")

  def test_score_chunking_matches_single_pass(self):
    req = datatypes.ScoreRequest(
        request_id="s1",
        prompt_tokens=np.ones((6, 2), dtype=np.int32),
        completion_tokens=np.arange(18, dtype=np.int32).reshape(6, 3),
    )

    core_single = _StubCore()
    single = _worker(core_single).score(req)

    core_chunked = _StubCore()
    chunked = _worker(core_chunked, chunk_size=2).score(req)

    np.testing.assert_array_equal(single.scores, chunked.scores)
    self.assertEqual(core_single.calls, 1)  # single pass
    self.assertEqual(core_chunked.calls, 3)  # 6 rows / 2 = 3 micro-batches

  def test_result_dtos_are_wire_safe_and_round_trip(self):
    result = _worker().compute_logps(_logprobs_request(batch=2))

    rpc_utils.validate_wire_safe(result)  # numpy result must pass the guardd
    restored = cloudpickle.loads(cloudpickle.dumps(result))

    np.testing.assert_array_equal(
        restored.per_token_logps, result.per_token_logps
    )

  def test_heartbeat_reports_current_state(self):
    worker = _worker()
    worker.state = WorkerState.INITIALIZING
    worker.state = WorkerState.READY
    worker.state = WorkerState.COMPILING
    self.assertEqual(worker.heartbeat().state, "COMPILING")

  def test_execution_context_wraps_compute_logps_and_score(self):
    events = []

    class TrackingContext:

      def __enter__(self):
        events.append("enter_ctx")
        return self

      def __exit__(self, *args):
        events.append("exit_ctx")

    worker = inference_lib.InferenceWorker(
        _StubCore(),
        worker_id="test_worker_ctx",
        pad_id=0,
        eos_id=1,
        execution_context=TrackingContext(),
    )

    worker.compute_logps(_logprobs_request(batch=2))
    self.assertEqual(events, ["enter_ctx", "exit_ctx"])

    events.clear()
    req = datatypes.ScoreRequest(
        request_id="s1",
        prompt_tokens=np.ones((2, 2), dtype=np.int32),
        completion_tokens=np.ones((2, 3), dtype=np.int32),
    )
    worker.score(req)
    self.assertEqual(events, ["enter_ctx", "exit_ctx"])

  def test_offload_markers_and_lock_serialization(self):
    worker = _worker()
    attr = inference_lib.abstract_worker.OFFLOAD_TO_THREAD_ATTR
    for offloaded_name in (
        "initialize",
        "compile",
        "start",
        "stop",
        "compute_logps",
        "per_token_logps",
        "score",
    ):
      self.assertTrue(
          getattr(getattr(worker, offloaded_name), attr, False),
          msg=f"{offloaded_name} should opt in to worker-thread offloading",
      )
    for inline_name in ("heartbeat", "info"):
      self.assertFalse(
          getattr(getattr(worker, inline_name), attr, False),
          msg=f"{inline_name} must remain inline on the event loop",
      )

    entered_logps = threading.Event()
    release_logps = threading.Event()
    stop_finished = threading.Event()
    logps_results: list[np.ndarray] = []
    logps_errors: list[BaseException] = []

    class _BlockingCore(_StubCore):

      def get_ref_per_token_logps(
          self,
          prompt_tokens,
          completion_tokens,
          pad_id,
          eos_id,
          temperature=1.0,
      ):
        entered_logps.set()
        if not release_logps.wait(timeout=5.0):
          raise TimeoutError("release_logps timed out")
        return super().get_ref_per_token_logps(
            prompt_tokens, completion_tokens, pad_id, eos_id, temperature
        )

    blocking_worker = _worker(_BlockingCore())
    blocking_worker.initialize()
    batch = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[1, 2]], dtype=np.int32),
        prompt_mask=np.ones((1, 2), dtype=np.float32),
        completion_ids=np.array([[3, 4]], dtype=np.int32),
        completion_mask=np.ones((1, 2), dtype=np.float32),
        advantages=np.ones((1, 2), dtype=np.float32),
    )

    def _run_logps():
      try:
        logps_results.append(blocking_worker.per_token_logps(batch))
      except BaseException as exc:  # pylint: disable=broad-exception-caught
        logps_errors.append(exc)

    t_logps = threading.Thread(target=_run_logps)
    t_logps.start()
    self.assertTrue(entered_logps.wait(timeout=5.0))

    # Inline probes do not acquire `_worker_lock` and respond immediately.
    self.assertEqual(blocking_worker.heartbeat().state, WorkerState.READY)
    self.assertEqual(blocking_worker.info().worker_id, "test_worker_1")

    def _run_stop():
      blocking_worker.stop()
      stop_finished.set()

    t_stop = threading.Thread(target=_run_stop)
    t_stop.start()
    self.assertFalse(stop_finished.wait(timeout=0.05))

    release_logps.set()
    t_logps.join(timeout=5.0)
    t_stop.join(timeout=5.0)
    self.assertEmpty(logps_errors)
    self.assertLen(logps_results, 1)
    np.testing.assert_allclose(logps_results[0], batch.completion_ids)
    self.assertTrue(stop_finished.is_set())
    self.assertEqual(blocking_worker.heartbeat().state, WorkerState.STOPPED)

  def test_offload_to_thread_rejects_async_and_supports_reentrant_lock(self):
    aw = inference_lib.abstract_worker

    async def _async_fn(self):
      return self

    with self.assertRaisesRegex(TypeError, "synchronous methods"):
      aw.offload_to_thread(_async_fn)

    class _MinimalWorker(aw.Worker):

      @aw.offload_to_thread
      def initialize(self):
        return datatypes.Response()

      @aw.offload_to_thread
      def compile(self, dummy_data=None):
        del dummy_data
        return self.initialize()

      def start(self):
        return datatypes.Response()

      def stop(self):
        return datatypes.Response()

      def info(self):
        return datatypes.WorkerInfo(worker_id="m", roles=frozenset())

      def heartbeat(self):
        return datatypes.HealthReport(state=self.state)

    w = _MinimalWorker()
    self.assertTrue(getattr(w.compile, aw.OFFLOAD_TO_THREAD_ATTR, False))
    self.assertFalse(getattr(w.heartbeat, aw.OFFLOAD_TO_THREAD_ATTR, False))
    self.assertIsInstance(w.compile(), datatypes.Response)


if __name__ == "__main__":
  absltest.main()
