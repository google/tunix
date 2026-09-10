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
import jax
import numpy as np
from tunix.experimental.common import datatypes

WorkerState = datatypes.WorkerState


def _rollout_response_dto() -> datatypes.RolloutResponse:
  traj_item = datatypes.TrajectoryItem(
      prompt_id="req-rollout-42",
      group_index=1,
      start_step=0,
      traj=datatypes.Trajectory(
          reward=1.25,
          status=datatypes.TrajectoryStatus.SUCCEEDED,
      ),
      prompt_tokens=np.array([10, 11, 12], dtype=np.int32),
      completion_tokens=np.array([20, 21], dtype=np.int32),
      action_mask=np.array([1, 1], dtype=np.float32),
      policy_version=7,
      metadata={"response_time": 0.5},
  )
  return datatypes.RolloutResponse(
      request_id="req-1",
      status="SUCCEEDED",
      payload=traj_item,
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
        prompt_ids=np.array([[1], [2]], dtype=np.int32),
        prompt_mask=np.ones((2, 1), dtype=np.float32),
        completion_ids=np.array([[3], [4]], dtype=np.int32),
        completion_mask=np.ones((2, 1), dtype=np.float32),
        advantages=np.array([1.0, 2.0], dtype=np.float32),
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
    np.testing.assert_array_equal(restored.payload.completion_mask, [[1], [1]])

  def test_trajectory_response_round_trips_through_cloudpickle(self):
    original = _rollout_response_dto()

    restored = cloudpickle.loads(cloudpickle.dumps(original))

    self.assertEqual(restored.request_id, original.request_id)
    self.assertEqual(restored.status, original.status)
    self.assertEqual(restored.metadata, original.metadata)
    self.assertIsNone(restored.error)
    self.assertIsNotNone(restored.payload)
    self.assertEqual(restored.payload.prompt_id, original.payload.prompt_id)
    self.assertEqual(restored.payload.group_index, original.payload.group_index)
    self.assertEqual(
        restored.payload.policy_version, original.payload.policy_version
    )
    self.assertEqual(
        restored.payload.traj.reward, original.payload.traj.reward
    )
    np.testing.assert_array_equal(
        restored.payload.prompt_tokens, original.payload.prompt_tokens
    )
    np.testing.assert_array_equal(
        restored.payload.completion_tokens, original.payload.completion_tokens
    )
    np.testing.assert_array_equal(
        restored.payload.action_mask, original.payload.action_mask
    )

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
    self.assertIsNone(restored.payload)

  def test_token_segment_enforces_shapes(self):
    with self.assertRaisesRegex(
        ValueError, "loss_mask shape .* != tokens shape"
    ):
      datatypes.TokenSegment(
          source="env",
          tokens=np.array([1, 2]),
          loss_mask=np.array([1]),
      )

    with self.assertRaisesRegex(ValueError, "logps shape .* != tokens shape"):
      datatypes.TokenSegment(
          source="assistant",
          tokens=np.array([1, 2]),
          loss_mask=np.array([1, 1]),
          logps=np.array([0.5]),
      )


  def test_trajectory_item_group_id_and_pair_index_aliases(self):
    # Test setting prompt_id and group_index synchronizes aliases
    item1 = datatypes.TrajectoryItem(prompt_id="p1", group_index=3)
    self.assertEqual(item1.group_id, "p1")
    self.assertEqual(item1.pair_index, 3)

    # Test setting legacy group_id and pair_index synchronizes standardized names
    item2 = datatypes.TrajectoryItem(group_id="p2", pair_index=4)
    self.assertEqual(item2.prompt_id, "p2")
    self.assertEqual(item2.group_index, 4)

  def test_trajectory_item_to_and_from_dict(self):
    traj = {
        "reward": 2.5,
        "status": datatypes.TrajectoryStatus.SUCCEEDED,
        "prompt_tokens": np.array([1, 2], dtype=np.int32),
        "completion_tokens": np.array([3, 4], dtype=np.int32),
        "action_mask": np.array([1.0, 1.0], dtype=np.float32),
        "policy_version": 3,
    }
    item = datatypes.TrajectoryItem(
        prompt_id="task_42",
        group_index=1,
        start_step=0,
        traj=traj,
        metadata={"key": "val"},
    )
    d = item.to_dict()
    self.assertEqual(d["prompt_id"], "task_42")
    self.assertEqual(d["group_index"], 1)
    self.assertEqual(d["traj"]["policy_version"], 3)
    self.assertEqual(d["metadata"], {"key": "val"})

    restored = datatypes.TrajectoryItem.from_dict(d)
    self.assertEqual(restored.prompt_id, "task_42")
    self.assertEqual(restored.group_index, 1)
    self.assertEqual(restored.policy_version, 3)
    self.assertEqual(restored.metadata, {"key": "val"})
    np.testing.assert_array_equal(restored.prompt_tokens, item.prompt_tokens)
    np.testing.assert_array_equal(
        restored.completion_tokens, item.completion_tokens
    )

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

    # TrajectoryItem traj_id
    item = datatypes.TrajectoryItem(prompt_id="42", group_index=2, traj={})
    self.assertEqual(item.traj_id, "traj_42_g2")


  def test_trajectory_item_fields(self):
    item = datatypes.TrajectoryItem(prompt_id="prompt_88", group_index=4, traj={})
    self.assertEqual(item.prompt_id, "prompt_88")
    self.assertEqual(item.group_index, 4)

    item_default = datatypes.TrajectoryItem(prompt_id="prompt_99", traj={})
    self.assertEqual(item_default.prompt_id, "prompt_99")
    self.assertEqual(item_default.group_index, 0)


class RLTrainerPayloadTest(absltest.TestCase):

  def test_rl_trainer_payload_fields_and_replace(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([3, 4], dtype=np.int32),
        completion_mask=np.ones(2, dtype=np.float32),
        advantages=np.array([1.5, 1.5], dtype=np.float32),
        num_segments=3,
        metadata={"step": 1},
    )
    self.assertEqual(payload.num_segments, 3)
    self.assertEqual(payload.metadata, {"step": 1})

    replaced = payload.replace(num_segments=4)
    self.assertEqual(replaced.num_segments, 4)
    np.testing.assert_array_equal(replaced.prompt_ids, payload.prompt_ids)

  def test_rl_trainer_payload_pytree_structure(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([1, 2], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([3, 4], dtype=np.int32),
        completion_mask=np.ones(2, dtype=np.float32),
        advantages=np.array([1.0, 2.0], dtype=np.float32),
        num_segments=5,
        metadata={"key": "value"},
    )
    leaves, treedef = jax.tree_util.tree_flatten(payload)
    # num_segments and metadata are pytree_node=False, so they are not in leaves
    for leaf in leaves:
      if leaf is not None:
        self.assertIsInstance(leaf, np.ndarray)

    # Tree unflatten restores full object
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    self.assertEqual(restored.num_segments, 5)
    self.assertEqual(restored.metadata, {"key": "value"})
    np.testing.assert_allclose(restored.advantages, [1.0, 2.0])

  def test_rl_trainer_payload_cloudpickle_roundtrip(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([10, 20], dtype=np.int32),
        prompt_mask=np.ones(2, dtype=np.float32),
        completion_ids=np.array([30, 40], dtype=np.int32),
        completion_mask=np.ones(2, dtype=np.float32),
        advantages=np.array([0.5, -0.5], dtype=np.float32),
        segment_ids=np.array([1, 1], dtype=np.int32),
        segment_positions=np.array([0, 1], dtype=np.int32),
        num_segments=2,
        metadata={"tag": "eval"},
    )
    restored = cloudpickle.loads(cloudpickle.dumps(payload))
    self.assertEqual(restored.num_segments, 2)
    self.assertEqual(restored.metadata, {"tag": "eval"})
    np.testing.assert_array_equal(restored.prompt_ids, payload.prompt_ids)
    np.testing.assert_allclose(restored.advantages, payload.advantages)


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


class GenerationArgsTest(absltest.TestCase):

  def test_generation_args_as_kwargs_with_max_response_length(self):
    args = datatypes.GenerationArgs(
        max_generation_steps=128,
        max_response_length=512,
        temperature=0.7,
        top_p=0.95,
        return_logprobs=True,
    )
    expected = {
        "max_generation_steps": 128,
        "max_response_length": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "return_logprobs": True,
    }
    self.assertEqual(args.as_kwargs(), expected)


if __name__ == "__main__":
  absltest.main()
