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


class CompactRoutedExpertsTest(absltest.TestCase):

  def _compact(self):
    valid = np.array(
        [[True, True, False, False], [False, True, True, True]], dtype=bool
    )
    values = np.arange(5 * 3 * 2, dtype=np.int16).reshape(5, 3, 2)
    return datatypes.CompactRoutedExperts(values=values, valid=valid)

  def test_materialize_scatters_values_and_unsets_the_rest(self):
    compact = self._compact()
    dense = compact.materialize()
    self.assertEqual(compact.shape, (2, 4, 3, 2))
    self.assertEqual(dense.shape, (2, 4, 3, 2))
    self.assertEqual(dense.dtype, np.int16)
    np.testing.assert_array_equal(dense[compact.valid], compact.values)
    np.testing.assert_array_equal(
        dense[~compact.valid],
        np.full((3, 3, 2), datatypes.UNSET_ROUTED_EXPERT, dtype=np.int16),
    )
    # Row-major order: row 0's two tokens come first, then row 1's three.
    np.testing.assert_array_equal(dense[1, 1], compact.values[2])

  def test_nbytes_counts_only_what_is_stored(self):
    compact = self._compact()
    self.assertEqual(compact.nbytes, 5 * 3 * 2 * 2 + 2 * 4)

  def test_cloudpickle_roundtrip_inside_payload(self):
    compact = self._compact()
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.zeros((2, 0), dtype=np.int32),
        prompt_mask=np.zeros((2, 0), dtype=np.float32),
        completion_ids=np.zeros((2, 4), dtype=np.int32),
        completion_mask=np.ones((2, 4), dtype=np.float32),
        advantages=np.zeros((2, 4), dtype=np.float32),
        routed_experts=compact,
    )
    restored = cloudpickle.loads(cloudpickle.dumps(payload))
    self.assertIsInstance(
        restored.routed_experts, datatypes.CompactRoutedExperts
    )
    np.testing.assert_array_equal(
        restored.routed_experts.materialize(), compact.materialize()
    )

  def test_uint10_and_uint8_pack_strip_unrouted_and_roundtrip_exactly(self):
    valid = np.array(
        [[True, True, True, False], [True, True, False, False]], dtype=bool
    )
    # 5 valid tokens: 2 unrouted prompt tokens (all -1) + 3 routed tokens
    # with expert IDs up to 511 (needs uint10) or 200 (fits uint8).
    unrouted = np.full((2, 4, 2), datatypes.UNSET_ROUTED_EXPERT, dtype=np.int16)
    routed_511 = np.array(
        [
            [[0, 511], [12, 345], [511, -1], [7, 8]],
            [[510, 1], [0, 0], [255, 256], [511, 511]],
            [[-1, 511], [300, 400], [100, 200], [1, 2]],
        ],
        dtype=np.int16,
    )
    values = np.concatenate([unrouted, routed_511], axis=0)
    baseline = datatypes.CompactRoutedExperts(values=values, valid=valid)

    packed_u10 = datatypes.CompactRoutedExperts.pack(
        values, valid, mode="uint10", strip_unrouted=True
    )
    self.assertEqual(packed_u10.encoding, "uint10")
    self.assertIsNone(packed_u10._values)
    # 2 unrouted prompt tokens were stripped from valid, leaving 3 routed rows.
    self.assertEqual(int(np.count_nonzero(packed_u10.valid)), 3)
    # 3 * 4 * 2 = 24 values -> 6 groups of 4 -> 30 bytes vs 5 * 4 * 2 * 2 = 80 bytes!
    self.assertEqual(packed_u10.packed_values.nbytes, 30)
    self.assertLess(packed_u10.nbytes, baseline.nbytes)
    np.testing.assert_array_equal(
        packed_u10.materialize(), baseline.materialize()
    )
    restored_u10 = cloudpickle.loads(cloudpickle.dumps(packed_u10))
    np.testing.assert_array_equal(
        restored_u10.materialize(), baseline.materialize()
    )

    routed_200 = np.minimum(routed_511, np.int16(200))
    values_200 = np.concatenate([unrouted, routed_200], axis=0)
    baseline_200 = datatypes.CompactRoutedExperts(
        values=values_200, valid=valid
    )
    packed_u8 = datatypes.CompactRoutedExperts.pack(
        values_200, valid, mode="auto", strip_unrouted=True
    )
    self.assertEqual(packed_u8.encoding, "uint8")
    self.assertEqual(packed_u8.packed_values.nbytes, 24)
    np.testing.assert_array_equal(
        packed_u8.materialize(), baseline_200.materialize()
    )

  def test_rejects_non_int16_values(self):
    with self.assertRaisesRegex(TypeError, "values must be an int16"):
      datatypes.CompactRoutedExperts(
          values=np.zeros((1, 3, 2), dtype=np.int32),
          valid=np.array([[True]]),
      )

  def test_rejects_wrong_values_rank(self):
    with self.assertRaisesRegex(ValueError, r"\[N, num_layers, top_k\]"):
      datatypes.CompactRoutedExperts(
          values=np.zeros((1, 3), dtype=np.int16),
          valid=np.array([[True]]),
      )

  def test_rejects_non_bool_valid(self):
    with self.assertRaisesRegex(TypeError, "valid must be a bool"):
      datatypes.CompactRoutedExperts(
          values=np.zeros((1, 3, 2), dtype=np.int16),
          valid=np.array([[1]], dtype=np.int32),
      )

  def test_rejects_wrong_valid_rank(self):
    with self.assertRaisesRegex(ValueError, r"valid must be \[B, T\]"):
      datatypes.CompactRoutedExperts(
          values=np.zeros((1, 3, 2), dtype=np.int16),
          valid=np.array([True]),
      )

  def test_rejects_count_mismatch(self):
    with self.assertRaisesRegex(
        ValueError, "2 routing rows but valid marks 1 positions"
    ):
      datatypes.CompactRoutedExperts(
          values=np.zeros((2, 3, 2), dtype=np.int16),
          valid=np.array([[True, False]]),
      )


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

  def test_generation_args_as_kwargs(self):
    args = datatypes.GenerationArgs(
        max_generation_steps=128,
        temperature=0.7,
        top_p=0.95,
        return_logprobs=True,
        return_routed_experts=True,
    )
    expected = {
        "max_generation_steps": 128,
        "temperature": 0.7,
        "top_p": 0.95,
        "return_logprobs": True,
        "return_routed_experts": True,
    }
    self.assertEqual(args.as_kwargs(), expected)


class RolloutFaultToleranceConfigTest(absltest.TestCase):

  def test_defaults_enable_evict_and_retry(self):
    cfg = datatypes.RolloutFaultToleranceConfig()
    self.assertTrue(cfg.enabled)
    self.assertTrue(cfg.evict_on_failure)
    self.assertTrue(cfg.retry_on_worker_failure)
    self.assertEqual(cfg.max_task_retries, 3)
    self.assertIsNone(cfg.max_in_flight_per_worker)
    self.assertIsNone(cfg.task_timeout_s)
    self.assertEqual(cfg.max_zero_worker_wait_s, 600.0)
    self.assertTrue(cfg.retry_weight_sync_on_eviction)
    self.assertFalse(cfg.recover_unknown_transfer_state)

  def test_with_overrides_without_arguments_returns_same_config(self):
    cfg = datatypes.RolloutFaultToleranceConfig(max_in_flight_per_worker=4)
    self.assertIs(cfg.with_overrides(), cfg)

  def test_with_overrides_replaces_only_given_fields(self):
    cfg = datatypes.RolloutFaultToleranceConfig(
        max_in_flight_per_worker=4,
        task_timeout_s=30.0,
        max_zero_worker_wait_s=10.0,
        max_task_retries=2,
    )
    merged = cfg.with_overrides(max_in_flight_per_worker=8, task_timeout_s=45.0)
    self.assertEqual(merged.max_in_flight_per_worker, 8)
    self.assertEqual(merged.task_timeout_s, 45.0)
    self.assertEqual(merged.max_zero_worker_wait_s, 10.0)
    self.assertEqual(merged.max_task_retries, 2)
    # The original config is left untouched.
    self.assertEqual(cfg.max_in_flight_per_worker, 4)
    self.assertEqual(cfg.task_timeout_s, 30.0)

  def test_with_overrides_validates_merged_config(self):
    cfg = datatypes.RolloutFaultToleranceConfig()
    with self.assertRaisesRegex(ValueError, "must be positive"):
      cfg.with_overrides(max_in_flight_per_worker=0)
    with self.assertRaisesRegex(ValueError, "must be positive"):
      cfg.with_overrides(task_timeout_s=-1.0)


if __name__ == "__main__":
  absltest.main()
