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

from unittest import mock

from absl.testing import absltest
import cloudpickle
from flax import nnx
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import rpc_utils
from tunix.dist_rl.worker import inference_worker as inference_lib
from tunix.rl.inference import inference_worker as inference_core

WorkerState = datatypes.WorkerState


class _StubCore:
  """Deterministic, row-independent stand-in for the real inference core."""

  def __init__(self):
    self.calls = 0
    self.last_prompt_tokens = None
    self.last_completion_tokens = None

  def get_ref_per_token_logps(
      self, prompt_tokens, completion_tokens, pad_id, eos_id, temperature=1.0
  ):
    del pad_id, eos_id
    self.calls += 1
    self.last_prompt_tokens = np.asarray(prompt_tokens, dtype=np.int32)
    self.last_completion_tokens = np.asarray(completion_tokens, dtype=np.int32)
    return jnp.asarray(completion_tokens, dtype=jnp.float32) * temperature

  def get_rewards(self, prompt_tokens, completion_tokens, pad_id, eos_id):
    del prompt_tokens, pad_id, eos_id
    self.calls += 1
    return jnp.asarray(completion_tokens, dtype=jnp.float32).sum(axis=1)


class _MaskedCore(_StubCore):

  def __init__(self):
    super().__init__()
    self.mask_chunks = []

  def get_ref_per_token_logps(
      self,
      prompt_tokens,
      completion_tokens,
      pad_id,
      eos_id,
      temperature=1.0,
      token_mask=None,
  ):
    self.mask_chunks.append(np.asarray(token_mask).copy())
    return super().get_ref_per_token_logps(
        prompt_tokens, completion_tokens, pad_id, eos_id, temperature
    )


class _SegmentContextModel(nnx.Module):
  """Scores depend on segment-confined causal context and local positions."""

  def __call__(
      self, tokens, positions, cache=None, attention_mask=None, segment_ids=None
  ):
    length = tokens.shape[1]
    causal = jnp.tril(jnp.ones((length, length), dtype=jnp.bool_))
    if segment_ids is None:
      valid = attention_mask
    else:
      valid = (
          (segment_ids[:, :, None] == segment_ids[:, None, :])
          & (segment_ids[:, :, None] > 0)
          & causal
      )
    context = jnp.einsum("bqk,bk->bq", valid.astype(jnp.float32), tokens + 1.0)
    features = context + positions.astype(jnp.float32) / 2.0
    return features[..., None] * jnp.arange(16, dtype=jnp.float32) / 32.0, cache


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

  def test_per_token_logps_uses_attention_validity_not_loss_or_pad_ids(self):
    core = _MaskedCore()
    batch = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[0, 2], [0, 3]], dtype=np.int32),
        prompt_mask=np.array([[0, 1], [1, 1]], dtype=np.bool_),
        completion_ids=np.array([[1, 0, 4, 0], [5, 0, 6, 0]], dtype=np.int32),
        # The internal tool token has no loss but remains attention context.
        completion_mask=np.array([[1, 0, 1, 0], [1, 0, 1, 0]], dtype=np.bool_),
        completion_attention_mask=np.array(
            [[1, 1, 1, 0], [1, 1, 1, 0]], dtype=np.bool_
        ),
        advantages=np.ones((2, 4), dtype=np.float32),
    )

    result = _worker(core, chunk_size=1).per_token_logps(batch)

    self.assertLen(core.mask_chunks, 2)
    np.testing.assert_array_equal(
        np.concatenate(core.mask_chunks),
        [[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]],
    )
    np.testing.assert_array_equal(
        result, batch.completion_ids.astype(np.float32)
    )

  def test_compute_logps_rejects_misaligned_attention_mask(self):
    core = _MaskedCore()
    req = _logprobs_request(token_mask=np.ones((4, 4), dtype=np.bool_))

    result = _worker(core, chunk_size=2).compute_logps(req)

    self.assertIsNotNone(result.error)
    self.assertEqual(result.error.error_type, "ValueError")
    self.assertIn("token_mask must match", result.error.message)
    self.assertEmpty(core.mask_chunks)

  def test_packed_scoring_preserves_segments_positions_and_chunk_order(self):
    core = inference_core.InferenceWorker({"reference": _SegmentContextModel()})
    worker = _worker(core, chunk_size=2)
    tokens = np.array(
        [[2, 3, 4, 7, 8, 9], [5, 6, 7, 2, 4, 5], [3, 5, 7, 6, 8, 10]],
        dtype=np.int32,
    )
    ids = np.repeat(np.arange(1, 7, dtype=np.int32).reshape(3, 2), 3, axis=1)
    positions = np.tile([0, 1, 2, 0, 1, 2], (3, 1)).astype(np.int32)
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.zeros((3, 0), dtype=np.int32),
        prompt_mask=np.zeros((3, 0), dtype=np.bool_),
        completion_ids=tokens,
        completion_mask=np.tile([0, 1, 1, 0, 1, 1], (3, 1)).astype(np.bool_),
        advantages=np.ones((3, 6), dtype=np.float32),
        segment_ids=ids,
        segment_positions=positions,
    )

    with mock.patch.object(
        core, "get_ref_per_token_logps", wraps=core.get_ref_per_token_logps
    ) as score:
      packed = worker.per_token_logps(payload)

    self.assertLen(score.call_args_list, 2)
    for call, rows in zip(score.call_args_list, (slice(0, 2), slice(2, 3))):
      np.testing.assert_array_equal(call.kwargs["segment_ids"], ids[rows])
      np.testing.assert_array_equal(
          call.kwargs["segment_positions"], positions[rows]
      )
      self.assertNotIn("token_mask", call.kwargs)

    # Independent rows provide an oracle for each packed segment's context.
    unpacked = payload.replace(
        prompt_ids=np.zeros((6, 0), dtype=np.int32),
        prompt_mask=np.zeros((6, 0), dtype=np.bool_),
        completion_ids=tokens.reshape(6, 3),
        completion_mask=np.tile([0, 1, 1], (6, 1)).astype(np.bool_),
        advantages=np.ones((6, 3), dtype=np.float32),
        segment_ids=np.ones((6, 3), dtype=np.int32),
        segment_positions=np.tile(np.arange(3, dtype=np.int32), (6, 1)),
    )
    separate = worker.per_token_logps(unpacked)
    np.testing.assert_allclose(
        packed.reshape(6, 3)[:, 1:], separate[:, 1:], rtol=1e-5, atol=1e-5
    )

    # Changing another segment cannot affect this segment's interior scores.
    changed_tokens = tokens.copy()
    changed_tokens[0, :3] = [10, 11, 12]
    changed = worker.per_token_logps(
        payload.replace(completion_ids=changed_tokens)
    )
    np.testing.assert_allclose(changed[0, 4:], packed[0, 4:], atol=1e-6)
    no_reset = worker.per_token_logps(
        payload.replace(
            segment_positions=np.tile(np.arange(6, dtype=np.int32), (3, 1))
        )
    )
    self.assertTrue(np.any(np.abs(no_reset[:, 4:] - packed[:, 4:]) > 1e-4))
    joined_segments = worker.per_token_logps(
        payload.replace(segment_ids=np.ones_like(ids))
    )
    self.assertTrue(
        np.any(np.abs(joined_segments[:, 4:] - packed[:, 4:]) > 1e-4)
    )

  def test_compute_logps_rejects_misaligned_or_incomplete_segment_metadata(
      self,
  ):
    for overrides in (
        {
            "segment_ids": np.ones((4, 4), dtype=np.int32),
            "segment_positions": np.ones((4, 5), dtype=np.int32),
        },
        {"segment_ids": np.ones((4, 5), dtype=np.int32)},
        {"segment_positions": np.ones((4, 5), dtype=np.int32)},
        {
            "segment_ids": np.ones((4, 5), dtype=np.int32),
            "segment_positions": np.ones((4, 5), dtype=np.int32),
            "token_mask": np.ones((4, 5), dtype=np.bool_),
        },
    ):
      with self.subTest(overrides=tuple(overrides)):
        core = _StubCore()
        result = _worker(core).compute_logps(_logprobs_request(**overrides))
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.error_type, "ValueError")
        self.assertEqual(core.calls, 0)

  def test_per_token_logps_rejects_completion_attention_shape_mismatch(self):
    batch = datatypes.RLTrainerPayload(
        prompt_ids=np.ones((2, 2), dtype=np.int32),
        prompt_mask=np.ones((2, 2), dtype=np.bool_),
        completion_ids=np.ones((2, 3), dtype=np.int32),
        completion_mask=np.ones((2, 3), dtype=np.float32),
        completion_attention_mask=np.ones((2, 2), dtype=np.bool_),
        advantages=np.ones((2, 3), dtype=np.float32),
    )
    with self.assertRaisesRegex(ValueError, "completion_attention_mask"):
      _worker(_MaskedCore()).per_token_logps(batch)

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


if __name__ == "__main__":
  absltest.main()
