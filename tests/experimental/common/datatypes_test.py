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

"""Serialization-discipline tests for the common wire DTOs."""

from absl.testing import absltest
import cloudpickle
import numpy as np
from tunix.experimental.common import datatypes


def _sample_result() -> datatypes.RolloutResult:
  return datatypes.RolloutResult(
      request_id="req-1",
      status="SUCCEEDED",
      prompt_tokens=np.array([10, 11, 12], dtype=np.int32),
      segments=[
          datatypes.TokenSegment(
              source="assistant",
              tokens=np.array([20, 21], dtype=np.int32),
              loss_mask=np.array([1, 1], dtype=np.int32),
              logps=np.array([-0.5, -1.5], dtype=np.float32),
          ),
          datatypes.TokenSegment(
              source="env",
              tokens=np.array([30], dtype=np.int32),
              loss_mask=np.array([0], dtype=np.int32),
          ),
      ],
      env_reward=1.25,
      policy_version=7,
  )


def _sample_dto() -> datatypes.SamplingResult:
  return datatypes.SamplingResult(
      request_id="sample-req-1",
      text="Hello from Tunix sampler!",
      token_ids=np.array([101, 102, 103], dtype=np.int32),
      logprobs=np.array([-0.1, -0.2, -0.05], dtype=np.float32),
      finish_reason="stop",
  )


def _sampling_request_dto() -> datatypes.SamplingRequest:
  return datatypes.SamplingRequest(
      prompt="Solve 2+2",
      request_id="req-sample-42",
      sampling_params=datatypes.SamplingParams(max_tokens=64, temperature=0.7),
  )


def _weight_sync_request_dto() -> datatypes.WeightSyncRequest:
  return datatypes.WeightSyncRequest(
      controller_id="raiden-ctrl-0",
      policy_version=14,
      source_metadata={"mesh": "2x4"},
      extra_config={"timeout": 30.0},
  )


class WireSerializationTest(absltest.TestCase):

  def test_weight_sync_request_round_trips_through_cloudpickle(self):
    original = _weight_sync_request_dto()

    restored = cloudpickle.loads(cloudpickle.dumps(original))

    self.assertEqual(restored.controller_id, original.controller_id)
    self.assertEqual(restored.policy_version, original.policy_version)
    self.assertEqual(restored.source_metadata, original.source_metadata)
    self.assertEqual(restored.extra_config, original.extra_config)

  def test_sampling_request_round_trips_through_cloudpickle(self):
    original = _sampling_request_dto()

    restored = cloudpickle.loads(cloudpickle.dumps(original))

    self.assertEqual(restored.request_id, original.request_id)
    self.assertEqual(restored.prompt, original.prompt)
    self.assertIsNotNone(restored.sampling_params)
    self.assertEqual(
        restored.sampling_params.max_tokens, original.sampling_params.max_tokens
    )
    self.assertEqual(
        restored.sampling_params.temperature,
        original.sampling_params.temperature,
    )

  def test_trajectory_result_round_trips_through_cloudpickle(self):
    original = _sample_result()

    restored = cloudpickle.loads(cloudpickle.dumps(original))

    self.assertEqual(restored.request_id, original.request_id)
    self.assertEqual(restored.status, original.status)
    self.assertEqual(restored.env_reward, original.env_reward)
    self.assertEqual(restored.policy_version, original.policy_version)
    self.assertIsNone(restored.error)
    np.testing.assert_array_equal(
        restored.prompt_tokens, original.prompt_tokens
    )
    self.assertLen(restored.segments, 2)
    np.testing.assert_array_equal(
        restored.segments[0].tokens, original.segments[0].tokens
    )
    np.testing.assert_array_equal(
        restored.segments[0].loss_mask, original.segments[0].loss_mask
    )
    np.testing.assert_allclose(
        restored.segments[0].logps, original.segments[0].logps
    )
    self.assertIsNone(restored.segments[1].logps)

  def test_error_result_round_trips(self):
    result = datatypes.RolloutResult(
        request_id="req-2",
        status="TIMEOUT",
        error=datatypes.ErrorInfo(
            error_type="TimeoutError",
            message="deadline exceeded",
            retryable=True,
        ),
    )

    restored = cloudpickle.loads(cloudpickle.dumps(result))

    self.assertEqual(restored.status, "TIMEOUT")
    self.assertEqual(restored.error.error_type, "TimeoutError")
    self.assertTrue(restored.error.retryable)
    self.assertEqual(restored.prompt_tokens.size, 0)
    self.assertEmpty(restored.segments)

  def test_token_segment_enforces_shapes(self):
    with self.assertRaisesRegex(
        ValueError, "loss_mask shape .* != tokens shape"
    ):
      datatypes.TokenSegment(
          source="env",
          tokens=np.array([1, 2]),
          loss_mask=np.array([1]),
      )

    with self.assertRaisesRegex(
        ValueError, "logps shape .* != tokens shape"
    ):
      datatypes.TokenSegment(
          source="assistant",
          tokens=np.array([1, 2]),
          loss_mask=np.array([1, 1]),
          logps=np.array([0.5]),
      )

  def test_from_trajectory(self):
    step1 = datatypes.Step(
        assistant_tokens=np.array([20, 21], dtype=np.int32),
        assistant_masks=np.array([1, 1], dtype=np.int32),
        logprobs=np.array([-0.5, -1.5], dtype=np.float32),
        env_tokens=np.array([30], dtype=np.int32),
        env_masks=np.array([0], dtype=np.int32),
    )
    traj = datatypes.Trajectory(
        steps=[step1],
        reward=1.25,
        status=datatypes.TrajectoryStatus.SUCCEEDED,
    )
    request = datatypes.RolloutRequest(
        request_id="req-1",
        prompt_text="hello",
        sampling_params=datatypes.SamplingParams(max_tokens=10),
    )

    result = datatypes.RolloutResult.from_trajectory(
        request_id=request.request_id,
        traj=traj,
        prompt_tokens=np.array([10, 11, 12], dtype=np.int32),
        policy_version=7,
    )

    self.assertEqual(result.request_id, "req-1")
    self.assertEqual(result.status, "SUCCEEDED")
    self.assertEqual(result.env_reward, 1.25)
    self.assertEqual(result.policy_version, 7)
    np.testing.assert_array_equal(result.prompt_tokens, [10, 11, 12])

    self.assertLen(result.segments, 2)

    # Assistant segment
    self.assertEqual(result.segments[0].source, "assistant")
    np.testing.assert_array_equal(result.segments[0].tokens, [20, 21])
    np.testing.assert_array_equal(result.segments[0].loss_mask, [1, 1])
    np.testing.assert_allclose(result.segments[0].logps, [-0.5, -1.5])

    # Env segment
    self.assertEqual(result.segments[1].source, "env")
    np.testing.assert_array_equal(result.segments[1].tokens, [30])
    np.testing.assert_array_equal(result.segments[1].loss_mask, [0])
    self.assertIsNone(result.segments[1].logps)

  def test_sampling_result_round_trips_through_cloudpickle(self):
    original = _sample_dto()

    restored = cloudpickle.loads(cloudpickle.dumps(original))

    self.assertEqual(restored.request_id, original.request_id)
    self.assertEqual(restored.text, original.text)
    self.assertEqual(restored.finish_reason, original.finish_reason)
    self.assertIsNone(restored.error)
    np.testing.assert_array_equal(restored.token_ids, original.token_ids)
    np.testing.assert_allclose(restored.logprobs, original.logprobs)

  def test_sampling_result_enforces_shapes(self):
    with self.assertRaisesRegex(
        ValueError, "logprobs shape .* != token_ids shape"
    ):
      datatypes.SamplingResult(
          token_ids=np.array([1, 2, 3]),
          logprobs=np.array([-0.1, -0.2]),
      )


if __name__ == "__main__":
  absltest.main()
