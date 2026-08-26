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

import time

from absl.testing import absltest
import cloudpickle
import numpy as np
from tunix.experimental.common import datatypes

WorkerState = datatypes.WorkerState


def _rollout_response_dto() -> datatypes.RolloutResponse:
  return datatypes.RolloutResponse(
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
      metadata={"response_time": 0.5},
  )


def _rollout_request_dto() -> datatypes.RolloutRequest:
  return datatypes.RolloutRequest(
      request_id="req-123",
      prompt="Solve 2+2",
      prompt_id="req-rollout-42",
      group_offset_id="group-1",
      generation_kwargs={"max_tokens": 128, "temperature": 0.5},
      max_turns=5,
      target_policy_version=3,
      metadata={"env": "math"},
  )


class WireSerializationTest(absltest.TestCase):

  def test_rollout_request_round_trips_through_cloudpickle(self):
    original = _rollout_request_dto()

    restored = cloudpickle.loads(cloudpickle.dumps(original))

    self.assertEqual(restored.request_id, original.request_id)
    self.assertEqual(restored.prompt, original.prompt)
    self.assertEqual(restored.prompt_id, original.prompt_id)
    self.assertEqual(restored.group_offset_id, original.group_offset_id)
    self.assertEqual(restored.generation_kwargs, original.generation_kwargs)
    self.assertEqual(restored.max_turns, original.max_turns)
    self.assertEqual(
        restored.target_policy_version, original.target_policy_version
    )
    self.assertEqual(restored.metadata, original.metadata)

  def test_train_request_round_trips_through_cloudpickle(self):
    payload = datatypes.RLTrainerPayload(
        advantages=np.array([1.0, 2.0], dtype=np.float32),
        loss_mask=np.array([[1, 1], [1, 0]], dtype=np.int32),
        metadata={"step": 42},
    )
    original = datatypes.TrainRequest(
        request_id="train-req-1",
        payload=payload,
        target_policy_version=2,
        metadata={"lineage_id": "batch_0"},
    )

    restored = cloudpickle.loads(cloudpickle.dumps(original))

    self.assertEqual(restored.request_id, "train-req-1")
    self.assertEqual(restored.target_policy_version, 2)
    self.assertEqual(restored.metadata, {"lineage_id": "batch_0"})
    np.testing.assert_allclose(restored.payload.advantages, [1.0, 2.0])
    np.testing.assert_array_equal(
        restored.payload.loss_mask, [[1, 1], [1, 0]]
    )

  def test_trajectory_response_round_trips_through_cloudpickle(self):
    original = _rollout_response_dto()

    restored = cloudpickle.loads(cloudpickle.dumps(original))

    self.assertEqual(restored.request_id, original.request_id)
    self.assertEqual(restored.status, original.status)
    self.assertEqual(restored.env_reward, original.env_reward)
    self.assertEqual(restored.policy_version, original.policy_version)
    self.assertEqual(restored.metadata, original.metadata)
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
    result = datatypes.RolloutResponse(
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
        prompt_id="prompt-1",
        prompt="hello",
        generation_kwargs={"max_tokens": 10},
    )

    result = datatypes.RolloutResponse.from_trajectory(
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

  def test_from_trajectory_preserves_metadata(self):
    traj = datatypes.Trajectory(
        steps=[],
        reward=1.0,
        status=datatypes.TrajectoryStatus.SUCCEEDED,
    )
    traj.metadata = {"traj_meta": "foo"}

    result = datatypes.RolloutResponse.from_trajectory(
        request_id="req-2",
        traj=traj,
        prompt_tokens=np.array([1, 2], dtype=np.int32),
        policy_version=1,
        metadata={"caller_meta": "bar"},
    )

    self.assertEqual(
        result.metadata, {"caller_meta": "bar", "traj_meta": "foo"}
    )

  def test_from_trajectory_metadata_edge_cases(self):
    # 1. Neither metadata nor traj.metadata provided
    traj_none = datatypes.Trajectory(
        steps=[], reward=1.0, status=datatypes.TrajectoryStatus.SUCCEEDED
    )
    res_none = datatypes.RolloutResponse.from_trajectory(
        request_id="req-1",
        traj=traj_none,
        prompt_tokens=np.array([1], dtype=np.int32),
        policy_version=1,
        metadata=None,
    )
    self.assertEqual(res_none.metadata, {})

    # 2. traj.metadata is non-dict (e.g., string or None)
    traj_non_dict = datatypes.Trajectory(
        steps=[], reward=1.0, status=datatypes.TrajectoryStatus.SUCCEEDED
    )
    traj_non_dict.metadata = "not-a-dict"  # pytype: disable=annotation-type-mismatch
    res_non_dict = datatypes.RolloutResponse.from_trajectory(
        request_id="req-2",
        traj=traj_non_dict,
        prompt_tokens=np.array([1], dtype=np.int32),
        policy_version=1,
        metadata={"key": "val"},
    )
    self.assertEqual(res_non_dict.metadata, {"key": "val"})

    # 3. Caller metadata overrides traj.metadata on collision
    traj_collision = datatypes.Trajectory(
        steps=[], reward=1.0, status=datatypes.TrajectoryStatus.SUCCEEDED
    )
    traj_collision.metadata = {"shared_key": "traj_val", "traj_only": 123}
    res_collision = datatypes.RolloutResponse.from_trajectory(
        request_id="req-3",
        traj=traj_collision,
        prompt_tokens=np.array([1], dtype=np.int32),
        policy_version=1,
        metadata={"shared_key": "caller_val", "caller_only": 456},
    )
    self.assertEqual(
        res_collision.metadata,
        {
            "shared_key": "caller_val",
            "traj_only": 123,
            "caller_only": 456,
        },
    )

  def test_health_report_defaults_heartbeat_unix_s_to_current_time(self):
    before = time.time()
    report = datatypes.HealthReport(state=WorkerState.READY)
    after = time.time()
    self.assertGreaterEqual(report.heartbeat_unix_s, before)
    self.assertLessEqual(report.heartbeat_unix_s, after)


if __name__ == "__main__":
  absltest.main()
