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
      group_index=1,
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
    self.assertEqual(restored.group_index, original.group_index)
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
        group_index=0,
        prompt="hello",
        generation_kwargs={"max_tokens": 10},
    )

    result = datatypes.RolloutResponse.from_trajectory(
        request_id=request.request_id,
        traj=traj,
        prompt_tokens=np.array([10, 11, 12], dtype=np.int32),
        policy_version=7,
        metadata={"group_index": 0},
    )

    self.assertEqual(result.request_id, "req-1")
    self.assertEqual(result.group_index, 0)
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
    traj.metadata = {"traj_meta": "foo", "group_index": 0}

    result = datatypes.RolloutResponse.from_trajectory(
        request_id="req-2",
        traj=traj,
        prompt_tokens=np.array([1, 2], dtype=np.int32),
        policy_version=1,
        metadata={"caller_meta": "bar"},
    )

    self.assertEqual(
        result.metadata,
        {"caller_meta": "bar", "traj_meta": "foo", "group_index": 0},
    )

  def test_from_trajectory_metadata_edge_cases(self):
    # 1. Neither metadata nor traj.metadata provided besides group_index
    traj_none = datatypes.Trajectory(
        steps=[], reward=1.0, status=datatypes.TrajectoryStatus.SUCCEEDED
    )
    traj_none.metadata = {"group_index": 0}
    res_none = datatypes.RolloutResponse.from_trajectory(
        request_id="req-1",
        traj=traj_none,
        prompt_tokens=np.array([1], dtype=np.int32),
        policy_version=1,
        metadata=None,
    )
    self.assertEqual(res_none.metadata, {"group_index": 0})

    # 2. traj.metadata is non-dict (e.g., string or None) with caller metadata providing group_index
    traj_non_dict = datatypes.Trajectory(
        steps=[], reward=1.0, status=datatypes.TrajectoryStatus.SUCCEEDED
    )
    traj_non_dict.metadata = "not-a-dict"  # pytype: disable=annotation-type-mismatch
    res_non_dict = datatypes.RolloutResponse.from_trajectory(
        request_id="req-2",
        traj=traj_non_dict,
        prompt_tokens=np.array([1], dtype=np.int32),
        policy_version=1,
        metadata={"key": "val", "group_index": 1},
    )
    self.assertEqual(res_non_dict.metadata, {"key": "val", "group_index": 1})
    self.assertEqual(res_non_dict.group_index, 1)

    # 3. Caller metadata overrides traj.metadata on collision
    traj_collision = datatypes.Trajectory(
        steps=[], reward=1.0, status=datatypes.TrajectoryStatus.SUCCEEDED
    )
    traj_collision.metadata = {
        "shared_key": "traj_val",
        "traj_only": 123,
        "group_index": 0,
    }
    res_collision = datatypes.RolloutResponse.from_trajectory(
        request_id="req-3",
        traj=traj_collision,
        prompt_tokens=np.array([1], dtype=np.int32),
        policy_version=1,
        metadata={
            "shared_key": "caller_val",
            "caller_only": 456,
            "group_index": 2,
        },
    )
    self.assertEqual(
        res_collision.metadata,
        {
            "shared_key": "caller_val",
            "traj_only": 123,
            "caller_only": 456,
            "group_index": 2,
        },
    )
    self.assertEqual(res_collision.group_index, 2)

  def test_from_trajectory_requires_group_index(self):
    traj = datatypes.Trajectory(
        steps=[],
        reward=1.0,
        status=datatypes.TrajectoryStatus.SUCCEEDED,
    )
    with self.assertRaisesRegex(ValueError, "lacks 'group_index'"):
      datatypes.RolloutResponse.from_trajectory(
          request_id="req-missing-group",
          traj=traj,
          prompt_tokens=np.zeros(0, dtype=np.int32),
          policy_version=1,
          metadata={"prompt_id": "p1"},
      )

    with self.assertRaisesRegex(ValueError, "lacks 'group_index'"):
      datatypes.RolloutResponse.from_trajectory(
          request_id="req-none-group",
          traj=traj,
          prompt_tokens=np.zeros(0, dtype=np.int32),
          policy_version=1,
          metadata={"prompt_id": "p1", "group_index": None},
      )

    with self.assertRaisesRegex(ValueError, "must be an integer"):
      datatypes.RolloutResponse.from_trajectory(
          request_id="req-bad-group",
          traj=traj,
          prompt_tokens=np.zeros(0, dtype=np.int32),
          policy_version=1,
          metadata={"prompt_id": "p1", "group_index": "invalid_group"},
      )

  def test_from_trajectory_handles_integer_prompt_id_zero(self):
    traj = datatypes.Trajectory(
        steps=[],
        reward=1.0,
        status=datatypes.TrajectoryStatus.SUCCEEDED,
    )
    resp = datatypes.RolloutResponse.from_trajectory(
        request_id="req-int-0",
        traj=traj,
        prompt_tokens=np.zeros(0, dtype=np.int32),
        policy_version=1,
        metadata={"prompt_id": 0, "group_index": 0},
    )
    self.assertEqual(resp.prompt_id, "0")
    self.assertEqual(resp.group_index, 0)

  def test_from_trajectory_unpacks_extra_reward(self):
    traj = datatypes.Trajectory(
        steps=[],
        status=datatypes.TrajectoryStatus.SUCCEEDED,
    )
    traj.extra = {"prompt_id": "p1", "group_index": 1, "reward": 3.5}
    resp = datatypes.RolloutResponse.from_trajectory(
        request_id="req-extra",
        traj=traj,
        prompt_tokens=np.zeros(0, dtype=np.int32),
        policy_version=1,
    )
    self.assertEqual(resp.prompt_id, "p1")
    self.assertEqual(resp.group_index, 1)
    self.assertEqual(resp.env_reward, 3.5)

  def test_from_trajectory_invalid_reward_fallback(self):
    traj = datatypes.Trajectory(
        steps=[],
        status=datatypes.TrajectoryStatus.SUCCEEDED,
    )
    # 1. Unparseable string reward in metadata falls back to 0.0
    resp_str = datatypes.RolloutResponse.from_trajectory(
        request_id="req-bad-str-reward",
        traj=traj,
        prompt_tokens=np.zeros(0, dtype=np.int32),
        policy_version=1,
        metadata={"prompt_id": "p1", "group_index": 0, "reward": "not_a_float"},
    )
    self.assertEqual(resp_str.env_reward, 0.0)

    # 2. Dictionary reward in metadata (TypeError on float cast) falls back to 0.0
    resp_dict = datatypes.RolloutResponse.from_trajectory(
        request_id="req-dict-reward",
        traj=traj,
        prompt_tokens=np.zeros(0, dtype=np.int32),
        policy_version=1,
        metadata={"prompt_id": "p1", "group_index": 0, "reward": {"bad": 123}},
    )
    self.assertEqual(resp_dict.env_reward, 0.0)

    # 3. Invalid reward attribute on traj falls back to 0.0 when metadata has no reward
    traj_bad_reward = datatypes.Trajectory(
        steps=[],
        reward="unparseable",  # pytype: disable=annotation-type-mismatch
        status=datatypes.TrajectoryStatus.SUCCEEDED,
    )
    resp_traj_reward = datatypes.RolloutResponse.from_trajectory(
        request_id="req-traj-bad-reward",
        traj=traj_bad_reward,
        prompt_tokens=np.zeros(0, dtype=np.int32),
        policy_version=1,
        metadata={"prompt_id": "p1", "group_index": 0},
    )
    self.assertEqual(resp_traj_reward.env_reward, 0.0)

    # 4. None / empty reward falls back to 0.0
    resp_none = datatypes.RolloutResponse.from_trajectory(
        request_id="req-none-reward",
        traj=traj,
        prompt_tokens=np.zeros(0, dtype=np.int32),
        policy_version=1,
        metadata={"prompt_id": "p1", "group_index": 0, "reward": None},
    )
    self.assertEqual(resp_none.env_reward, 0.0)

    # 5. Valid numerical string parses correctly to float
    resp_valid_str = datatypes.RolloutResponse.from_trajectory(
        request_id="req-valid-str-reward",
        traj=traj,
        prompt_tokens=np.zeros(0, dtype=np.int32),
        policy_version=1,
        metadata={"prompt_id": "p1", "group_index": 0, "reward": "4.25"},
    )
    self.assertEqual(resp_valid_str.env_reward, 4.25)

  def test_from_trajectory_falls_back_to_traj_task_for_prompt_id(self):
    # 1. traj.task string fallback when prompt_id is missing from metadata
    traj_with_task = datatypes.Trajectory(
        steps=[],
        reward=1.0,
        status=datatypes.TrajectoryStatus.SUCCEEDED,
    )
    traj_with_task.task = "task_prompt_99"
    resp = datatypes.RolloutResponse.from_trajectory(
        request_id="req-task-fallback",
        traj=traj_with_task,
        prompt_tokens=np.zeros(0, dtype=np.int32),
        policy_version=1,
        metadata={"group_index": 0},
    )
    self.assertEqual(resp.prompt_id, "task_prompt_99")

    # 2. traj.task non-string (int) fallback gets converted to str
    traj_int_task = datatypes.Trajectory(
        steps=[],
        reward=1.0,
        status=datatypes.TrajectoryStatus.SUCCEEDED,
    )
    traj_int_task.task = 42
    resp_int = datatypes.RolloutResponse.from_trajectory(
        request_id="req-int-task-fallback",
        traj=traj_int_task,
        prompt_tokens=np.zeros(0, dtype=np.int32),
        policy_version=1,
        metadata={"group_index": 0},
    )
    self.assertEqual(resp_int.prompt_id, "42")

    # 3. traj without task and without metadata prompt_id defaults to empty str
    traj_no_task = datatypes.Trajectory(
        steps=[],
        reward=1.0,
        status=datatypes.TrajectoryStatus.SUCCEEDED,
    )
    resp_empty = datatypes.RolloutResponse.from_trajectory(
        request_id="req-no-task",
        traj=traj_no_task,
        prompt_tokens=np.zeros(0, dtype=np.int32),
        policy_version=1,
        metadata={"group_index": 0},
    )
    self.assertEqual(resp_empty.prompt_id, "")

    # 4. Explicit metadata prompt_id takes precedence over traj.task
    resp_override = datatypes.RolloutResponse.from_trajectory(
        request_id="req-override-task",
        traj=traj_with_task,
        prompt_tokens=np.zeros(0, dtype=np.int32),
        policy_version=1,
        metadata={"prompt_id": "explicit_id", "group_index": 0},
    )
    self.assertEqual(resp_override.prompt_id, "explicit_id")

  def test_health_report_defaults_heartbeat_unix_s_to_current_time(self):
    before = time.time()
    report = datatypes.HealthReport(state=WorkerState.READY)
    after = time.time()
    self.assertGreaterEqual(report.heartbeat_unix_s, before)
    self.assertLessEqual(report.heartbeat_unix_s, after)

  def test_request_defaults_request_id_to_uuid(self):
    req1 = datatypes.Request()
    req2 = datatypes.Request()
    self.assertTrue(req1.request_id.startswith("req_"))
    self.assertTrue(req2.request_id.startswith("req_"))
    self.assertNotEqual(req1.request_id, req2.request_id)

  def test_rollout_request_traj_id_formatting(self):
    # Default group_index is 0
    r_default = datatypes.RolloutRequest(prompt_id="42")
    self.assertEqual(r_default.traj_id, "traj_42_g0")

    # String prompt ID
    r_str = datatypes.RolloutRequest(prompt_id="math_101")
    self.assertEqual(r_str.traj_id, "traj_math_101_g0")

    # Grouped at index 1
    r_grouped_1 = datatypes.RolloutRequest(prompt_id="42", group_index=1)
    self.assertEqual(r_grouped_1.traj_id, "traj_42_g1")

  def test_from_trajectory_populates_prompt_id_and_group_index(self):
    traj = datatypes.Trajectory(
        steps=[],
        reward=2.0,
        status=datatypes.TrajectoryStatus.SUCCEEDED,
    )
    resp = datatypes.RolloutResponse.from_trajectory(
        request_id="req_test",
        traj=traj,
        prompt_tokens=np.array([1, 2], dtype=np.int32),
        policy_version=2,
        metadata={"prompt_id": "math_42", "group_index": 3},
    )
    self.assertEqual(resp.prompt_id, "math_42")
    self.assertEqual(resp.group_index, 3)

  def test_trajectory_item_fields(self):
    item = datatypes.TrajectoryItem(prompt_id="prompt_88", group_index=4)
    self.assertEqual(item.prompt_id, "prompt_88")
    self.assertEqual(item.group_index, 4)

    item_default = datatypes.TrajectoryItem(prompt_id="prompt_99")
    self.assertEqual(item_default.prompt_id, "prompt_99")
    self.assertEqual(item_default.group_index, 0)


class TokenSegmentRoutingTest(absltest.TestCase):
  """`routed_experts` must line up with the tokens it describes."""

  def test_rejects_length_mismatch(self):
    """Only the per-token axis is checked; trailing axes are model-specific."""
    tokens = np.arange(4, dtype=np.int32)
    with self.assertRaisesRegex(ValueError, "routed_experts shape"):
      datatypes.TokenSegment(
          source="assistant",
          tokens=tokens,
          loss_mask=np.ones_like(tokens),
          routed_experts=np.zeros((3, 2, 2), dtype=np.int32),
      )


if __name__ == "__main__":
  absltest.main()
