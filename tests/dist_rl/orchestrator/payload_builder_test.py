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

"""Unit tests for trajectory-to-payload conversion (`payload_builder`)."""

import dataclasses

from absl.testing import absltest
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import payload_builder
from tunix.rl import advantages as rl_advantages
from tunix.rl import algorithm_config
from tunix.rl import function_registry
from tunix.rl import packing

_ROUTING_LAYERS = 2
_ROUTING_TOP_K = 2


def _routing(length, fill):
  """`[length, num_layers, top_k]` routing where every slot holds `fill`."""
  shape = (length, _ROUTING_LAYERS, _ROUTING_TOP_K)
  return np.full(shape, fill, dtype=np.int32)


class CreateTrainerPayloadsTest(absltest.TestCase):
  """Tests for create_trainer_payloads and TrajectoryItem extraction helpers."""

  def test_importing_module_registers_builtin_estimators(self):
    registered = set(function_registry.list_advantage_estimators())
    self.assertContainsSubset(
        {"grpo", "grpo-loo", "rloo", "drgrpo"}, registered
    )

  def test_grpo_create_trainer_payloads(self):
    cfg = algorithm_config.GRPOConfig(num_generations=2)
    item1 = datatypes.TrajectoryItem(
        group_index=0,
        prompt_id="g1",
        start_step=0,
        traj={
            "prompt_tokens": np.array([1, 2], dtype=np.int32),
            "conversation_tokens": np.array([3, 4], dtype=np.int32),
            "conversation_masks": np.array([1, 1], dtype=np.float32),
        },
    )
    item2 = datatypes.TrajectoryItem(
        group_index=1,
        prompt_id="g1",
        start_step=0,
        traj={
            "prompt_tokens": np.array([1, 2], dtype=np.int32),
            "conversation_tokens": np.array([5, 6], dtype=np.int32),
            "conversation_masks": np.array([1, 1], dtype=np.float32),
        },
    )

    payloads = payload_builder.create_trainer_payloads(
        [item1, item2], rewards=[1.0, 2.0], algo_config=cfg
    )
    self.assertLen(payloads, 2)
    self.assertIsInstance(payloads[0], datatypes.RLTrainerPayload)
    np.testing.assert_array_equal(payloads[0].prompt_ids, [1, 2])
    np.testing.assert_array_equal(payloads[0].prompt_mask, [1.0, 1.0])
    np.testing.assert_array_equal(payloads[0].completion_ids, [3, 4])
    np.testing.assert_array_equal(payloads[0].completion_mask, [1.0, 1.0])
    self.assertLen(payloads[0].advantages, 1)
    self.assertLess(payloads[0].advantages[0], 0.0)

    np.testing.assert_array_equal(payloads[1].prompt_ids, [1, 2])
    np.testing.assert_array_equal(payloads[1].prompt_mask, [1.0, 1.0])
    np.testing.assert_array_equal(payloads[1].completion_ids, [5, 6])
    np.testing.assert_array_equal(payloads[1].completion_mask, [1.0, 1.0])
    self.assertLen(payloads[1].advantages, 1)
    self.assertGreater(payloads[1].advantages[0], 0.0)

  def test_grpo_create_trainer_payloads_with_rollout_logprobs(self):
    cfg = algorithm_config.GRPOConfig(num_generations=2)
    item1 = datatypes.TrajectoryItem(
        group_index=0,
        prompt_id="g1",
        start_step=0,
        traj={
            "prompt_tokens": np.array([1, 2], dtype=np.int32),
            "conversation_tokens": np.array([3, 4], dtype=np.int32),
            "conversation_masks": np.array([1, 1], dtype=np.float32),
            "old_logprobs": np.array([-0.5, -0.2], dtype=np.float32),
        },
    )
    item2 = datatypes.TrajectoryItem(
        group_index=1,
        prompt_id="g1",
        start_step=0,
        traj={
            "prompt_tokens": np.array([1, 2], dtype=np.int32),
            "conversation_tokens": np.array([5, 6], dtype=np.int32),
            "conversation_masks": np.array([1, 1], dtype=np.float32),
            "old_logprobs": None,
        },
    )

    payloads = payload_builder.create_trainer_payloads(
        [item1, item2], rewards=[1.0, 2.0], algo_config=cfg
    )
    self.assertLen(payloads, 2)
    np.testing.assert_allclose(
        payloads[0].old_per_token_logps,
        np.array([-0.5, -0.2], dtype=np.float32),
    )
    self.assertIsNone(payloads[1].old_per_token_logps)

  def test_grpo_create_trainer_payloads_with_disabled_rollout_logps(self):
    cfg = algorithm_config.GRPOConfig(
        num_generations=2, use_rollout_logps=False
    )
    item1 = datatypes.TrajectoryItem(
        group_index=0,
        prompt_id="g1",
        start_step=0,
        traj={
            "prompt_tokens": np.array([1, 2], dtype=np.int32),
            "conversation_tokens": np.array([3, 4], dtype=np.int32),
            "conversation_masks": np.array([1, 1], dtype=np.float32),
            "old_logprobs": np.array([-0.5, -0.2], dtype=np.float32),
        },
    )
    item2 = datatypes.TrajectoryItem(
        group_index=1,
        prompt_id="g1",
        start_step=0,
        traj={
            "prompt_tokens": np.array([1, 2], dtype=np.int32),
            "conversation_tokens": np.array([5, 6], dtype=np.int32),
            "conversation_masks": np.array([1, 1], dtype=np.float32),
            "old_logprobs": np.array([-0.1, -0.4], dtype=np.float32),
        },
    )
    payloads = payload_builder.create_trainer_payloads(
        [item1, item2], rewards=[1.0, 2.0], algo_config=cfg
    )
    self.assertLen(payloads, 2)
    self.assertIsNone(payloads[0].old_per_token_logps)
    self.assertIsNone(payloads[1].old_per_token_logps)

  def test_grpo_create_trainer_payloads_with_mismatched_logps_length(self):
    cfg = algorithm_config.GRPOConfig(num_generations=2)
    item1 = datatypes.TrajectoryItem(
        group_index=0,
        prompt_id="g1",
        start_step=0,
        traj={
            "prompt_tokens": np.array([1, 2], dtype=np.int32),
            "conversation_tokens": np.array([3, 4], dtype=np.int32),
            "conversation_masks": np.array([1, 1], dtype=np.float32),
            "old_logprobs": np.array([-0.5, -0.2, -0.1], dtype=np.float32),
        },
    )
    item2 = datatypes.TrajectoryItem(
        group_index=1,
        prompt_id="g1",
        start_step=0,
        traj={
            "prompt_tokens": np.array([1, 2], dtype=np.int32),
            "conversation_tokens": np.array([5, 6], dtype=np.int32),
            "conversation_masks": np.array([1, 1], dtype=np.float32),
            "old_logprobs": np.array([-0.3, -0.4], dtype=np.float32),
        },
    )
    with self.assertRaises(ValueError):
      payload_builder.create_trainer_payloads(
          [item1, item2], rewards=[1.0, 2.0], algo_config=cfg
      )

  def test_ppo_advantages_and_trainer_payloads(self):
    cfg = algorithm_config.PPOConfig()
    item = datatypes.TrajectoryItem(
        group_index=0,
        prompt_id="g1",
        start_step=0,
        traj={
            "prompt_tokens": np.array([10], dtype=np.int32),
            "conversation_tokens": np.array([20], dtype=np.int32),
            "conversation_masks": np.array([1.0], dtype=np.float32),
        },
    )

    payloads = payload_builder.create_trainer_payloads(
        [item], rewards=[2.0], algo_config=cfg
    )
    self.assertLen(payloads, 1)
    np.testing.assert_array_equal(payloads[0].prompt_ids, [10])
    np.testing.assert_array_equal(payloads[0].prompt_mask, [1.0])
    np.testing.assert_array_equal(payloads[0].completion_ids, [20])
    np.testing.assert_array_equal(payloads[0].completion_mask, [1.0])
    # Without a critic the one-step estimate is advantage = returns = reward.
    np.testing.assert_allclose(payloads[0].advantages, [2.0])
    np.testing.assert_allclose(payloads[0].returns, [2.0])

  def test_create_trainer_payloads_rejects_rewards_length_mismatch(self):
    cfg = algorithm_config.GRPOConfig(num_generations=2)
    items = [
        datatypes.TrajectoryItem(
            group_index=i,
            prompt_id="g1",
            traj={
                "prompt_tokens": np.array([1, 2], dtype=np.int32),
                "conversation_tokens": np.array([3, 4], dtype=np.int32),
                "conversation_masks": np.array([1.0, 1.0], dtype=np.float32),
            },
        )
        for i in range(2)
    ]
    with self.assertRaisesRegex(ValueError, "one entry per trajectory"):
      payload_builder.create_trainer_payloads(
          items, rewards=[1.0], algo_config=cfg
      )

  def test_create_trainer_payloads_custom_registered_advantage_estimator(self):
    # pylint: disable=protected-access
    estimators = function_registry.default_registry._registry[
        function_registry._ADVANTAGE_ESTIMATOR_CATEGORY
    ]
    # pylint: enable=protected-access
    self.addCleanup(estimators.pop, "test_custom_irt")

    @function_registry.register_advantage_estimator("test_custom_irt")
    def _custom_adv(rewards, num_generations, valid_mask=None):
      self.assertEqual(num_generations, 2)
      np.testing.assert_allclose(rewards, [1.0, 2.0])
      np.testing.assert_array_equal(valid_mask, [True, True])
      return np.array([42.0, -42.0], dtype=np.float32)

    cfg = algorithm_config.GRPOConfig(
        num_generations=2,
        advantage_estimator="test_custom_irt",
    )
    items = [
        datatypes.TrajectoryItem(
            group_index=i,
            prompt_id="g1",
            traj={
                "prompt_tokens": np.array([1, 2], dtype=np.int32),
                "conversation_tokens": np.array([3, 4], dtype=np.int32),
                "conversation_masks": np.array([1.0, 1.0], dtype=np.float32),
            },
        )
        for i in range(2)
    ]

    payloads = payload_builder.create_trainer_payloads(
        items,
        rewards=[1.0, 2.0],
        algo_config=cfg,
    )
    np.testing.assert_allclose(payloads[0].advantages, [42.0])
    np.testing.assert_allclose(payloads[1].advantages, [-42.0])

  def test_ppo_masks_invalid_trajectory_and_attaches_routed_experts(self):
    cfg = algorithm_config.PPOConfig(num_generations=2)
    routed = np.array([[[0, 1]], [[2, 3]], [[4, 5]]], dtype=np.int32)
    valid_item = datatypes.TrajectoryItem(
        group_index=0,
        prompt_id="g1",
        traj={
            "status": datatypes.TrajectoryStatus.SUCCEEDED,
            "prompt_tokens": np.array([1, 2], dtype=np.int32),
            "conversation_tokens": np.array([3, 4], dtype=np.int32),
            "conversation_masks": np.array([1.0, 1.0], dtype=np.float32),
            "routed_experts": routed,
        },
    )
    invalid_item = datatypes.TrajectoryItem(
        group_index=1,
        prompt_id="g1",
        traj={
            "status": datatypes.TrajectoryStatus.FAILED,
            "prompt_tokens": np.array([1, 2], dtype=np.int32),
            "conversation_tokens": np.array([5, 6], dtype=np.int32),
            "conversation_masks": np.array([1.0, 1.0], dtype=np.float32),
        },
    )
    payloads = payload_builder.create_trainer_payloads(
        [valid_item, invalid_item],
        rewards=[3.0, 5.0],
        algo_config=cfg,
    )
    np.testing.assert_allclose(payloads[0].advantages, [3.0])
    np.testing.assert_allclose(payloads[0].returns, [3.0])
    self.assertIsNotNone(payloads[0].routed_experts)
    np.testing.assert_array_equal(payloads[1].completion_mask, [0.0, 0.0])
    np.testing.assert_array_equal(payloads[1].advantages, [0.0])
    np.testing.assert_array_equal(payloads[1].returns, [0.0])

  def test_ppo_masks_nonfinite_reward_trajectories(self):
    cfg = algorithm_config.PPOConfig(num_generations=3)
    items = [
        datatypes.TrajectoryItem(
            group_index=i,
            prompt_id="g1",
            traj={
                "status": datatypes.TrajectoryStatus.SUCCEEDED,
                "prompt_tokens": np.array([1, 2], dtype=np.int32),
                "conversation_tokens": np.array([3, 4], dtype=np.int32),
                "conversation_masks": np.array([1.0, 1.0], dtype=np.float32),
            },
        )
        for i in range(3)
    ]
    payloads = payload_builder.create_trainer_payloads(
        items,
        rewards=[2.5, np.nan, np.inf],
        algo_config=cfg,
    )
    np.testing.assert_allclose(payloads[0].advantages, [2.5])
    np.testing.assert_allclose(payloads[0].returns, [2.5])
    np.testing.assert_array_equal(payloads[0].completion_mask, [1.0, 1.0])
    for bad_idx in (1, 2):
      with self.subTest(group_index=bad_idx):
        np.testing.assert_array_equal(
            payloads[bad_idx].completion_mask, [0.0, 0.0]
        )
        np.testing.assert_array_equal(payloads[bad_idx].advantages, [0.0])
        np.testing.assert_array_equal(payloads[bad_idx].returns, [0.0])

  def test_sequence_scalars_expand_identically_in_batch_assemblers(self):
    cfg = algorithm_config.PPOConfig(num_generations=2, use_rollout_logps=False)
    items = [
        datatypes.TrajectoryItem(
            group_index=i,
            prompt_id="g1",
            traj={
                "prompt_tokens": np.array([1, 2], dtype=np.int32),
                "conversation_tokens": np.arange(3 + i, dtype=np.int32),
                "conversation_masks": np.ones(3 + i, dtype=np.float32),
            },
        )
        for i in range(2)
    ]
    compact = payload_builder.create_trainer_payloads(
        items, rewards=[2.0, -1.0], algo_config=cfg
    )
    expanded = [
        dataclasses.replace(
            payload,
            advantages=np.full(
                len(payload.completion_ids),
                payload.advantages[0],
                dtype=np.float32,
            ),
            returns=np.full(
                len(payload.completion_ids),
                payload.returns[0],
                dtype=np.float32,
            ),
        )
        for payload in compact
    ]

    def padded_assembler():
      return batch_assembly.PaddedBatchAssembler(
          batch_size=2,
          max_prompt_length=4,
          max_response_length=5,
          pad_id=0,
          num_generations=2,
          mini_batch_size=1,
      )

    def packed_assembler():
      return batch_assembly.SequencePackedBatchAssembler(
          batch_size=1,
          num_generations=2,
          mini_batch_size=1,
          max_packed_len=16,
      )

    for name, assembler_factory in (
        ("padded", padded_assembler),
        ("packed", packed_assembler),
    ):
      with self.subTest(assembler=name):
        actual = list(assembler_factory().feed(compact))[0].payload
        expected = list(assembler_factory().feed(expanded))[0].payload
        np.testing.assert_array_equal(
            actual.completion_ids, expected.completion_ids
        )
        np.testing.assert_array_equal(
            actual.completion_mask, expected.completion_mask
        )
        np.testing.assert_array_equal(actual.advantages, expected.advantages)
        np.testing.assert_array_equal(actual.returns, expected.returns)

  def test_extract_tokens_and_masks_with_failed_item(self):
    failed_item = datatypes.TrajectoryItem(
        prompt_id="g1",
        group_index=0,
        traj={
            "prompt_tokens": np.zeros(0, dtype=np.int32),
            "conversation_tokens": np.zeros(0, dtype=np.int32),
            "conversation_masks": np.zeros(0, dtype=np.float32),
            "status": datatypes.TrajectoryStatus.FAILED.name,
            "trajectory_reward": 0.0,
        },
    )

    p_arr, c_arr, act_arr = payload_builder._extract_tokens_and_masks(
        failed_item
    )

    self.assertEmpty(p_arr)
    self.assertEmpty(c_arr)
    self.assertEmpty(act_arr)

  def test_extract_tokens_and_masks_flattens_2d_arrays(self):
    item = datatypes.TrajectoryItem(
        prompt_id="g1",
        group_index=0,
        traj={
            "prompt_tokens": np.array([[10, 20, 30]], dtype=np.int32),
            "conversation_tokens": np.array([[40, 50]], dtype=np.int32),
            "conversation_masks": np.array([[1.0, 0.0]], dtype=np.float32),
        },
    )

    p_arr, c_arr, act_arr = payload_builder._extract_tokens_and_masks(item)

    self.assertEqual(p_arr.shape, (3,))
    np.testing.assert_array_equal(p_arr, [10, 20, 30])
    self.assertEqual(c_arr.shape, (2,))
    np.testing.assert_array_equal(c_arr, [40, 50])
    self.assertEqual(act_arr.shape, (2,))
    np.testing.assert_array_equal(act_arr, [1.0, 0.0])

  def test_create_trainer_payloads_unpads_prompt_tokens_using_prompt_length(
      self,
  ):
    cfg = algorithm_config.GRPOConfig(num_generations=2)
    item1 = datatypes.TrajectoryItem(
        group_index=0,
        prompt_id="g1",
        start_step=0,
        traj={
            "prompt_tokens": np.array([0, 0, 0, 10, 20], dtype=np.int32),
            "prompt_length": 3,
            "conversation_tokens": np.array([30, 40], dtype=np.int32),
            "conversation_masks": np.array([1, 1], dtype=np.float32),
        },
    )
    item2 = datatypes.TrajectoryItem(
        group_index=1,
        prompt_id="g1",
        start_step=0,
        traj={
            "prompt_tokens": np.array([0, 0, 0, 10, 20], dtype=np.int32),
            "prompt_length": 3,
            "conversation_tokens": np.array([50, 60], dtype=np.int32),
            "conversation_masks": np.array([1, 1], dtype=np.float32),
        },
    )
    payloads = payload_builder.create_trainer_payloads(
        [item1, item2], rewards=[1.0, 2.0], algo_config=cfg
    )
    np.testing.assert_array_equal(payloads[0].prompt_ids, [0, 10, 20])
    np.testing.assert_array_equal(payloads[0].prompt_mask, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(payloads[1].prompt_ids, [0, 10, 20])

  def test_create_trainer_payloads_encodes_compact_routed_experts(self):
    cfg = algorithm_config.GRPOConfig(num_generations=2)
    routed_u8 = np.array(
        [[[0, 1, 2, 3]], [[4, 5, 6, 7]], [[8, 9, 10, 11]]], dtype=np.int32
    )
    routed_u10 = np.array(
        [[[0, 300, 511, 1022]], [[4, 5, 6, 7]], [[8, 9, 10, 11]]],
        dtype=np.int32,
    )
    item1 = datatypes.TrajectoryItem(
        group_index=0,
        prompt_id="g1",
        start_step=0,
        traj={
            "prompt_tokens": np.array([1, 2], dtype=np.int32),
            "conversation_tokens": np.array([3, 4], dtype=np.int32),
            "conversation_masks": np.array([1.0, 1.0], dtype=np.float32),
            "routed_experts": routed_u8,
        },
    )
    item2 = datatypes.TrajectoryItem(
        group_index=1,
        prompt_id="g1",
        start_step=0,
        traj={
            "prompt_tokens": np.array([1, 2], dtype=np.int32),
            "conversation_tokens": np.array([5, 6], dtype=np.int32),
            "conversation_masks": np.array([1.0, 1.0], dtype=np.float32),
            "routed_experts": routed_u10,
        },
    )
    payloads = payload_builder.create_trainer_payloads(
        [item1, item2], rewards=[1.0, 2.0], algo_config=cfg
    )
    self.assertEqual(payloads[0].routed_experts.dtype, np.uint8)
    self.assertEqual(payloads[0].routed_experts.shape, (3, 1, 4))
    np.testing.assert_array_equal(
        packing.decode_routed_experts(
            payloads[0].routed_experts, batched=False
        ),
        routed_u8.astype(np.int16),
    )
    self.assertEqual(payloads[1].routed_experts.dtype, np.uint8)
    self.assertEqual(payloads[1].routed_experts.shape, (3, 1, 1, 5))
    np.testing.assert_array_equal(
        packing.decode_routed_experts(
            payloads[1].routed_experts, batched=False
        ),
        routed_u10.astype(np.int16),
    )

  def test_empty_tokens_handling(self):
    for cfg in [
        algorithm_config.GRPOConfig(num_generations=2),
        algorithm_config.PPOConfig(num_generations=1),
    ]:
      g = cfg.num_generations
      rewards = [float(i + 1) for i in range(g)]

      items = [
          datatypes.TrajectoryItem(
              group_index=0,
              prompt_id="g1",
              start_step=0,
              traj={
                  "prompt_tokens": np.zeros(0, dtype=np.int32),
                  "conversation_tokens": np.zeros(0, dtype=np.int32),
                  "conversation_masks": np.zeros(0, dtype=np.float32),
              },
          )
          for _ in range(g)
      ]
      payloads = payload_builder.create_trainer_payloads(
          items, rewards=rewards, algo_config=cfg
      )
      self.assertLen(payloads, g)
      for payload in payloads:
        self.assertEqual(payload.prompt_ids.shape, (0,))
        self.assertEqual(payload.completion_ids.shape, (0,))
        self.assertEqual(payload.prompt_mask.shape, (0,))
        self.assertEqual(payload.completion_mask.shape, (0,))
        self.assertEqual(payload.advantages.shape, (0,))
        if cfg.advantage_estimator == "gae":
          self.assertEqual(payload.returns.shape, (0,))


class RoutedExpertsForItemTest(absltest.TestCase):
  """`_routed_experts_for` returns a prefix view capped at `seq_len`."""

  def _align(self, routed, seq_len=8):
    item = datatypes.TrajectoryItem(traj={"routed_experts": routed})
    return payload_builder._routed_experts_for(item, seq_len)  # pylint: disable=protected-access

  def test_returns_none_without_capture(self):
    self.assertIsNone(self._align(None))

  def test_prefix_capture_missing_final_token_returns_zero_copy_view(self):
    src = _routing(7, 3).astype(np.int16)
    out = self._align(src, seq_len=8)
    self.assertEqual(out.dtype, np.int16)
    self.assertEqual(out.shape, (7, _ROUTING_LAYERS, _ROUTING_TOP_K))
    np.testing.assert_array_equal(out, 3)
    self.assertTrue(np.shares_memory(out, src))

  def test_rejects_capture_shorter_than_seq_len_minus_one(self):
    with self.assertRaisesRegex(
        ValueError, "routed_experts length must be >= 7"
    ):
      self._align(_routing(5, 3), seq_len=8)

  def test_overlong_capture_truncates_to_seq_len_as_zero_copy_view(self):
    src = _routing(10, 4).astype(np.int16)
    out = self._align(src, seq_len=8)
    self.assertEqual(out.dtype, np.int16)
    self.assertEqual(out.shape, (8, _ROUTING_LAYERS, _ROUTING_TOP_K))
    np.testing.assert_array_equal(out, 4)
    self.assertTrue(np.shares_memory(out, src))

  def test_wrong_rank_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "length, num_layers, top_k"):
      self._align(np.zeros((8, _ROUTING_TOP_K), dtype=np.int32))


class ExcludeMaskedFromAdvantageTest(absltest.TestCase):
  """Masked / unhealthy rollouts must not shape the group baseline."""

  def _item(self, group_index, *, masked=False, failed=False):
    if failed:
      return datatypes.TrajectoryItem(
          prompt_id="g1",
          group_index=group_index,
          traj={
              "status": datatypes.TrajectoryStatus.FAILED,
              "trajectory_reward": 0.0,
              "prompt_tokens": np.array([], dtype=np.int32),
              "conversation_tokens": np.array([], dtype=np.int32),
              "conversation_masks": np.array([], dtype=np.float32),
          },
      )
    return datatypes.TrajectoryItem(
        prompt_id="g1",
        group_index=group_index,
        start_step=0,
        traj={
            "status": datatypes.TrajectoryStatus.SUCCEEDED,
            "prompt_tokens": np.array([1, 2], dtype=np.int32),
            "conversation_tokens": np.array([3, 4], dtype=np.int32),
            "conversation_masks": (
                np.zeros(2, dtype=np.float32)
                if masked
                else np.ones(2, dtype=np.float32)
            ),
        },
    )

  def _cfg(self):
    return algorithm_config.GRPOConfig(num_generations=4)

  def test_masked_item_is_excluded_and_zeroed(self):
    group = [self._item(i) for i in range(3)] + [self._item(3, masked=True)]
    payloads = payload_builder.create_trainer_payloads(
        group, rewards=[1.0, 2.0, 3.0, 0.0], algo_config=self._cfg()
    )

    np.testing.assert_allclose(
        [float(p.advantages[0]) for p in payloads],
        [-1.0, 0.0, 1.0, 0.0],
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_array_equal(payloads[3].completion_mask, [0.0, 0.0])
    np.testing.assert_array_equal(payloads[3].advantages, [0.0])
    self.assertLen(payloads[3].completion_ids, 2)
    np.testing.assert_array_equal(payloads[0].completion_mask, [1.0, 1.0])

  def test_unmasked_overlong_item_is_included_when_overlong_filter_false(self):
    overlong_item = datatypes.TrajectoryItem(
        prompt_id="p0",
        group_index=3,
        traj={
            "status": datatypes.TrajectoryStatus.MAX_STEPS_REACHED,
            "prompt_tokens": np.array([1, 2], dtype=np.int32),
            "conversation_tokens": np.array([3, 4], dtype=np.int32),
            "conversation_masks": np.ones(2, dtype=np.float32),
        },
    )
    group = [self._item(i) for i in range(3)] + [overlong_item]
    rewards = [1.0, 2.0, 3.0, 0.0]

    payloads = payload_builder.create_trainer_payloads(
        group, rewards=rewards, algo_config=self._cfg()
    )

    np.testing.assert_allclose(
        [float(p.advantages[0]) for p in payloads],
        rl_advantages.compute_advantages(
            np.array(rewards, dtype=np.float32), num_generations=4
        ),
        rtol=1e-5,
    )
    np.testing.assert_array_equal(payloads[3].completion_mask, [1.0, 1.0])

  def test_failed_item_is_excluded_without_raising(self):
    group = [self._item(i) for i in range(3)] + [self._item(3, failed=True)]
    payloads = payload_builder.create_trainer_payloads(
        group, rewards=[1.0, 2.0, 3.0, 0.0], algo_config=self._cfg()
    )

    np.testing.assert_allclose(
        [
            float(np.mean(p.advantages)) if p.advantages.size else 0.0
            for p in payloads
        ],
        [-1.0, 0.0, 1.0, 0.0],
        rtol=1e-4,
        atol=1e-4,
    )
    self.assertEmpty(payloads[3].completion_ids)
    self.assertEmpty(payloads[3].advantages)

  def test_degenerate_group_zeroes_every_member(self):
    group = [self._item(0)] + [self._item(i, masked=True) for i in range(1, 4)]
    payloads = payload_builder.create_trainer_payloads(
        group, rewards=[5.0, 0.0, 0.0, 0.0], algo_config=self._cfg()
    )

    for i, payload in enumerate(payloads):
      with self.subTest(group_index=i):
        np.testing.assert_array_equal(payload.completion_mask, [0.0, 0.0])
        np.testing.assert_array_equal(payload.advantages, [0.0])

  def test_nonfinite_reward_item_is_excluded_and_zeroed(self):
    group = [self._item(i) for i in range(4)]
    payloads = payload_builder.create_trainer_payloads(
        group, rewards=[1.0, 2.0, 3.0, np.nan], algo_config=self._cfg()
    )

    np.testing.assert_allclose(
        [float(p.advantages[0]) for p in payloads],
        [-1.0, 0.0, 1.0, 0.0],
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_array_equal(payloads[3].completion_mask, [0.0, 0.0])
    np.testing.assert_array_equal(payloads[3].advantages, [0.0])
    np.testing.assert_array_equal(payloads[0].completion_mask, [1.0, 1.0])

  def test_nonfinite_rewards_leaving_single_valid_peer_zeroes_group(self):
    group = [self._item(i) for i in range(4)]
    payloads = payload_builder.create_trainer_payloads(
        group,
        rewards=[5.0, np.nan, np.inf, -np.inf],
        algo_config=self._cfg(),
    )

    for i, payload in enumerate(payloads):
      with self.subTest(group_index=i):
        np.testing.assert_array_equal(payload.completion_mask, [0.0, 0.0])
        np.testing.assert_array_equal(payload.advantages, [0.0])

  def test_all_valid_group_matches_legacy(self):
    group = [self._item(i) for i in range(4)]
    rewards = [1.0, 2.0, 3.0, 4.0]
    payloads = payload_builder.create_trainer_payloads(
        group, rewards=rewards, algo_config=self._cfg()
    )

    np.testing.assert_allclose(
        [float(p.advantages[0]) for p in payloads],
        rl_advantages.compute_advantages(
            np.array(rewards, dtype=np.float32), num_generations=4
        ),
        rtol=1e-5,
    )

  def test_routed_experts_ref_propagated_only_for_valid_trajectories(self):
    ref0 = datatypes.RoutedExpertsRef(worker_id="w0", ref_id=10)
    ref1 = datatypes.RoutedExpertsRef(worker_id="w0", ref_id=11)
    ref2 = datatypes.RoutedExpertsRef(worker_id="w1", ref_id=12)
    ref3 = datatypes.RoutedExpertsRef(worker_id="w1", ref_id=13)

    items = [
        dataclasses.replace(self._item(0), routed_experts_ref=ref0),
        dataclasses.replace(self._item(1), routed_experts_ref=ref1),
        dataclasses.replace(
            self._item(2, masked=True), routed_experts_ref=ref2
        ),
        dataclasses.replace(self._item(3), routed_experts_ref=ref3),
    ]
    payloads = payload_builder.create_trainer_payloads(
        items, rewards=[1.0, 3.0, 2.0, np.nan], algo_config=self._cfg()
    )
    self.assertEqual(payloads[0].routed_experts_ref, ref0)
    self.assertEqual(payloads[1].routed_experts_ref, ref1)
    self.assertIsNone(payloads[2].routed_experts_ref)
    self.assertIsNone(payloads[3].routed_experts_ref)
    for p in payloads:
      self.assertIsNone(p.routed_experts)

    # Degenerate group (< 2 valid peers) clears routed_experts_ref on the lone
    # valid survivor so critique_stage can release it immediately.
    degenerate_payloads = payload_builder.create_trainer_payloads(
        items, rewards=[1.0, np.nan, 2.0, np.inf], algo_config=self._cfg()
    )
    for p in degenerate_payloads:
      self.assertIsNone(p.routed_experts_ref)

    # Inline non-Relay mode preserves routed_experts on invalid/zero-mask items
    # so chunks without RoutedExpertsSpec retain static pytree structure.
    inline_items = []
    for idx, masked in enumerate((False, False, True, False)):
      base = self._item(idx, masked=masked)
      routed_traj = {**base.traj, "routed_experts": _routing(4, idx + 1)}
      inline_items.append(dataclasses.replace(base, traj=routed_traj))
    inline_payloads = payload_builder.create_trainer_payloads(
        inline_items, rewards=[1.0, 3.0, 2.0, np.nan], algo_config=self._cfg()
    )
    for idx, p in enumerate(inline_payloads):
      self.assertIsNone(p.routed_experts_ref)
      self.assertIsNotNone(p.routed_experts)
      decoded = packing.decode_routed_experts(p.routed_experts, batched=False)
      np.testing.assert_array_equal(decoded, idx + 1)


if __name__ == "__main__":
  absltest.main()
