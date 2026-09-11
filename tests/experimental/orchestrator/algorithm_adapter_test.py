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

import types
from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.rl import algo_core
from tunix.rl import algorithm_config


class AlgorithmAdapterTest(absltest.TestCase):

  def test_grpo_advantage_normalization(self):
    adapter = algorithm_adapter.GRPOAdapter(
        algo_config=algorithm_config.GRPOConfig(num_generations=4)
    )
    rewards = [1.0, 2.0, 3.0, 4.0]
    advs = adapter.compute_advantages(rewards, num_generations=4)

    self.assertLen(advs, 4)
    # Mean should be 0.0
    self.assertAlmostEqual(float(np.mean(advs)), 0.0, places=4)
    # Std should be 1.0
    self.assertAlmostEqual(float(np.std(advs, ddof=1)), 1.0, places=4)
    np.testing.assert_allclose(
        advs,
        algo_core.compute_advantages(
            np.array(rewards, dtype=np.float32), num_generations=4
        ),
    )

  def test_grpo_advantage_normalization_zero_variance(self):
    adapter = algorithm_adapter.GRPOAdapter(
        algo_config=algorithm_config.GRPOConfig(num_generations=4)
    )
    rewards = [1.0, 1.0, 1.0, 1.0]
    advs = adapter.compute_advantages(rewards, num_generations=4)
    self.assertLen(advs, 4)
    np.testing.assert_allclose(advs, np.zeros(4, dtype=np.float32), atol=1e-5)

  def test_grpo_invalid_group_size(self):
    with self.assertRaises(ValueError):
      algorithm_config.GRPOConfig(num_generations=1)
    with self.assertRaises(ValueError):
      algorithm_config.GRPOConfig(num_generations=0)

  def test_grpo_create_trainer_payloads(self):
    adapter = algorithm_adapter.GRPOAdapter(
        algo_config=algorithm_config.GRPOConfig(num_generations=2)
    )
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

    payloads = adapter.create_trainer_payloads(
        [item1, item2], rewards=[1.0, 2.0]
    )
    self.assertLen(payloads, 2)
    self.assertIsInstance(payloads[0], datatypes.RLTrainerPayload)
    np.testing.assert_array_equal(payloads[0].prompt_ids, [1, 2])
    np.testing.assert_array_equal(payloads[0].prompt_mask, [1.0, 1.0])
    np.testing.assert_array_equal(payloads[0].completion_ids, [3, 4])
    np.testing.assert_array_equal(payloads[0].completion_mask, [1.0, 1.0])
    self.assertLen(payloads[0].advantages, 2)
    self.assertLess(payloads[0].advantages[0], 0.0)

    np.testing.assert_array_equal(payloads[1].prompt_ids, [1, 2])
    np.testing.assert_array_equal(payloads[1].prompt_mask, [1.0, 1.0])
    np.testing.assert_array_equal(payloads[1].completion_ids, [5, 6])
    np.testing.assert_array_equal(payloads[1].completion_mask, [1.0, 1.0])
    self.assertLen(payloads[1].advantages, 2)
    self.assertGreater(payloads[1].advantages[0], 0.0)
    self.assertEqual(adapter.loss_fn(), algo_core.grpo_loss_fn)

  def test_grpo_create_trainer_payloads_with_rollout_logprobs(self):
    adapter = algorithm_adapter.GRPOAdapter(group_size=2)
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

    payloads = adapter.create_trainer_payloads(
        [item1, item2], rewards=[1.0, 2.0]
    )
    self.assertLen(payloads, 2)
    np.testing.assert_allclose(
        payloads[0].old_per_token_logps,
        np.array([-0.5, -0.2], dtype=np.float32),
    )
    self.assertIsNone(payloads[1].old_per_token_logps)

  def test_grpo_create_trainer_payloads_with_disabled_rollout_logps(self):
    adapter = algorithm_adapter.GRPOAdapter(
        algo_config=algorithm_config.GRPOConfig(
            num_generations=2, use_rollout_logps=False
        )
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
    payloads = adapter.create_trainer_payloads(
        [item1, item2], rewards=[1.0, 2.0]
    )
    self.assertLen(payloads, 2)
    self.assertIsNone(payloads[0].old_per_token_logps)
    self.assertIsNone(payloads[1].old_per_token_logps)

  def test_grpo_create_trainer_payloads_with_mismatched_logps_length(self):
    adapter = algorithm_adapter.GRPOAdapter(group_size=2)
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
      adapter.create_trainer_payloads([item1, item2], rewards=[1.0, 2.0])

  def test_ppo_advantages_and_trainer_payloads(self):
    adapter = algorithm_adapter.PPOAdapter(group_size=2, gamma=0.99, lam=0.95)
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

    payloads = adapter.create_trainer_payloads(
        [item], rewards=[2.0], values=[1.0]
    )
    self.assertLen(payloads, 1)
    np.testing.assert_array_equal(payloads[0].prompt_ids, [10])
    np.testing.assert_array_equal(payloads[0].prompt_mask, [1.0])
    np.testing.assert_array_equal(payloads[0].completion_ids, [20])
    np.testing.assert_array_equal(payloads[0].completion_mask, [1.0])
    self.assertLen(payloads[0].advantages, 1)
    self.assertLen(payloads[0].returns, 2)
    self.assertAlmostEqual(payloads[0].advantages[0], 1.0)
    self.assertAlmostEqual(payloads[0].returns[0], 2.0)
    self.assertEqual(adapter.loss_fn(), algo_core.ppo_policy_loss_fn)

  def test_grpo_build_gen_model_input_fn(self):
    algo_config = algorithm_config.GRPOConfig(
        num_generations=4,
        epsilon=0.25,
        beta=0.05,
        temperature=0.8,
        loss_agg_mode="token-mean",
        kl_loss_mode="kld",
        kl_clamp_value=1.5,
    )
    adapter = algorithm_adapter.GRPOAdapter(algo_config=algo_config)
    gen_fn = adapter.build_gen_model_input_fn(pad_id=10, eos_id=20)
    self.assertTrue(callable(gen_fn))

    fake_example = {"mock_payload": "data"}
    model_inputs = gen_fn(fake_example)

    self.assertIs(model_inputs["train_example"], fake_example)
    self.assertEqual(model_inputs["pad_id"], 10)
    self.assertEqual(model_inputs["eos_id"], 20)

    cfg = model_inputs["algo_config"]
    self.assertEqual(cfg.beta, 0.05)
    self.assertEqual(cfg.epsilon, 0.25)
    self.assertEqual(cfg.epsilon_high, 0.25)
    self.assertEqual(cfg.loss_algo, "grpo")
    self.assertEqual(cfg.loss_agg_mode, "token-mean")
    self.assertEqual(cfg.temperature, 0.8)
    self.assertEqual(cfg.kl_loss_mode, "kld")
    self.assertEqual(cfg.kl_clamp_value, 1.5)

  def test_grpo_build_gen_model_input_fn_fails_without_temperature(self):
    """Verifies build_gen_model_input_fn raises ValueError if temperature is unset."""
    adapter = algorithm_adapter.GRPOAdapter(
        algo_config=algorithm_config.GRPOConfig(num_generations=2)
    )
    with self.assertRaisesRegex(
        ValueError, "Trainer temperature must be explicitly set"
    ):
      adapter.build_gen_model_input_fn(pad_id=0, eos_id=1)

  def test_grpo_custom_algo_config(self):
    config = algorithm_config.GRPOConfig(
        num_generations=4,
        epsilon=0.2,
        epsilon_high=0.3,
        temperature=1.0,
        loss_algo="gspo-token",
        policy_loss_fn="grpo",
        advantage_estimator="drgrpo",
        kl_loss_mode="mse_kl",
    )
    adapter = algorithm_adapter.GRPOAdapter(algo_config=config)
    self.assertEqual(adapter.algo_config.kl_loss_mode, "mse_kl")
    self.assertEqual(adapter.algo_config.epsilon_high, 0.3)
    self.assertEqual(adapter.algo_config.loss_algo, "gspo-token")
    self.assertEqual(adapter.algo_config.policy_loss_fn, "grpo")
    self.assertEqual(adapter.algo_config.advantage_estimator, "drgrpo")

    rewards = [1.0, 2.0, 3.0, 4.0]
    advs = adapter.compute_advantages(rewards, num_generations=4)
    np.testing.assert_allclose(
        advs,
        algo_core.compute_drgrpo_advantages(
            np.array(rewards, dtype=np.float32), num_generations=4
        ),
    )

    gen_fn = adapter.build_gen_model_input_fn(pad_id=0, eos_id=1)
    model_inputs = gen_fn({})
    algo_config = model_inputs["algo_config"]
    self.assertEqual(algo_config.epsilon, 0.2)
    self.assertEqual(algo_config.epsilon_high, 0.3)
    self.assertEqual(algo_config.loss_algo, "gspo-token")
    self.assertEqual(algo_config.kl_loss_mode, "mse_kl")

  def test_ppo_build_gen_model_input_fn(self):
    adapter = algorithm_adapter.PPOAdapter(
        clip_epsilon=0.3,
        gamma=0.98,
        lam=0.92,
        entropy_coef=0.01,
    )
    gen_fn = adapter.build_gen_model_input_fn(pad_id=5, eos_id=6)
    self.assertTrue(callable(gen_fn))

    fake_example = {"mock_payload": "data"}
    model_inputs = gen_fn(fake_example)

    self.assertIs(model_inputs["train_example"], fake_example)
    self.assertEqual(model_inputs["pad_id"], 5)
    self.assertEqual(model_inputs["eos_id"], 6)

    algo_config = model_inputs["algo_config"]
    self.assertEqual(algo_config.epsilon_low, 0.3)
    self.assertEqual(algo_config.epsilon_high, 0.3)
    self.assertEqual(algo_config.entropy_coef, 0.01)
    self.assertEqual(algo_config.gamma, 0.98)
    self.assertEqual(algo_config.lam, 0.92)

  def test_grpo_with_ref_logps(self):
    adapter = algorithm_adapter.GRPOAdapter(
        algo_config=algorithm_config.GRPOConfig(num_generations=2)
    )
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
    ref_logps = [
        np.array([-0.1, -0.2], dtype=np.float32),
        np.array([-0.3, -0.4], dtype=np.float32),
    ]
    payloads = adapter.create_trainer_payloads(
        [item1, item2], rewards=[1.0, 2.0], ref_logps=ref_logps
    )
    np.testing.assert_allclose(payloads[0].ref_per_token_logps, [-0.1, -0.2])
    np.testing.assert_allclose(payloads[1].ref_per_token_logps, [-0.3, -0.4])

  def test_ppo_with_ref_and_old_logps(self):
    adapter = algorithm_adapter.PPOAdapter(group_size=1)
    item = datatypes.TrajectoryItem(
        group_index=0,
        prompt_id="g1",
        start_step=0,
        traj={
            "prompt_tokens": np.array([10], dtype=np.int32),
            "conversation_tokens": np.array([20], dtype=np.int32),
            "conversation_masks": np.array([0.0], dtype=np.float32),
        },
    )
    payloads = adapter.create_trainer_payloads(
        [item],
        rewards=[2.0],
        values=[1.0],
        ref_logps=[np.array([-0.5], dtype=np.float32)],
        old_logps=[np.array([-0.6], dtype=np.float32)],
    )
    np.testing.assert_array_equal(payloads[0].prompt_ids, [10])
    np.testing.assert_array_equal(payloads[0].prompt_mask, [1.0])
    np.testing.assert_array_equal(payloads[0].completion_ids, [20])
    np.testing.assert_array_equal(payloads[0].completion_mask, [0.0])
    np.testing.assert_allclose(payloads[0].ref_per_token_logps, [-0.5])
    np.testing.assert_allclose(payloads[0].old_per_token_logps, [-0.6])

  def test_empty_tokens_handling(self):
    for adapter in [
        algorithm_adapter.GRPOAdapter(group_size=2),
        algorithm_adapter.PPOAdapter(group_size=1),
    ]:
      g = adapter.group_size
      rewards = [float(i + 1) for i in range(g)]

      # 1. Both prompt_tokens and conversation_tokens are empty.
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
      payloads = adapter.create_trainer_payloads(items, rewards=rewards)
      self.assertLen(payloads, g)
      for payload in payloads:
        self.assertEqual(payload.prompt_ids.shape, (0,))
        self.assertEqual(payload.completion_ids.shape, (0,))
        self.assertEqual(payload.prompt_mask.shape, (0,))
        self.assertEqual(payload.completion_mask.shape, (0,))
        self.assertEqual(payload.advantages.shape, (0,))
        if adapter.has_critic:
          self.assertEqual(payload.returns.shape, (0,))

      # 2. prompt_tokens provided, conversation_tokens is empty.
      items_prompt_only = [
          datatypes.TrajectoryItem(
              group_index=0,
              prompt_id="g1",
              start_step=0,
              traj={
                  "prompt_tokens": np.array([1, 2], dtype=np.int32),
                  "conversation_tokens": np.zeros(0, dtype=np.int32),
                  "conversation_masks": np.zeros(0, dtype=np.float32),
              },
          )
          for _ in range(g)
      ]
      payloads = adapter.create_trainer_payloads(
          items_prompt_only, rewards=rewards
      )
      self.assertLen(payloads, g)
      for payload in payloads:
        np.testing.assert_array_equal(payload.prompt_ids, [1, 2])
        np.testing.assert_array_equal(payload.prompt_mask, [1.0, 1.0])
        self.assertEqual(payload.completion_ids.shape, (0,))
        self.assertEqual(payload.completion_mask.shape, (0,))
        self.assertEqual(payload.advantages.shape, (0,))

      # 3. prompt_tokens is empty, conversation_tokens provided.
      items_completion_only = [
          datatypes.TrajectoryItem(
              group_index=0,
              prompt_id="g1",
              start_step=0,
              traj={
                  "prompt_tokens": np.zeros(0, dtype=np.int32),
                  "conversation_tokens": np.array([3, 4], dtype=np.int32),
                  "conversation_masks": np.ones(2, dtype=np.float32),
              },
          )
          for _ in range(g)
      ]
      payloads = adapter.create_trainer_payloads(
          items_completion_only, rewards=rewards
      )
      self.assertLen(payloads, g)
      for payload in payloads:
        self.assertEqual(payload.prompt_ids.shape, (0,))
        self.assertEqual(payload.prompt_mask.shape, (0,))
        np.testing.assert_array_equal(payload.completion_ids, [3, 4])
        self.assertLen(payload.advantages, 2)

  def test_grpo_group_size_guard(self):
    """Verifies Discrepancy D7: num_generations <= 1 must raise ValueError."""
    with self.assertRaisesRegex(
        ValueError, "num_generations must be greater than 1"
    ):
      algorithm_config.GRPOConfig(num_generations=1)

    with self.assertRaisesRegex(
        ValueError, "num_generations must be greater than 1"
    ):
      algorithm_config.GRPOConfig(num_generations=0)

  def test_grpo_wraps_canonical_config(self):
    """Verifies GRPOAdapter directly wraps and preserves canonical GRPOConfig."""
    canonical_config = algorithm_config.GRPOConfig(
        num_generations=4,
        beta=0.03,
        epsilon=0.15,
        temperature=1.0,
        loss_agg_mode="token-mean",
        kl_loss_mode="low_var_kl",
        kl_clamp_value=10.0,
    )
    adapter = algorithm_adapter.GRPOAdapter(
        algo_config=canonical_config,
    )
    self.assertIs(adapter.algo_config, canonical_config)
    self.assertEqual(adapter.group_size, 4)
    self.assertEqual(adapter.algo_config.epsilon, 0.15)
    self.assertEqual(adapter.algo_config.beta, 0.03)
    self.assertEqual(adapter.algo_config.temperature, 1.0)
    self.assertEqual(adapter.algo_config.loss_agg_mode, "token-mean")
    self.assertEqual(adapter.algo_config.kl_loss_mode, "low_var_kl")
    self.assertEqual(adapter.algo_config.kl_clamp_value, 10.0)
    gen_fn = adapter.build_gen_model_input_fn(pad_id=0, eos_id=1)
    model_inputs = gen_fn({"fake": "batch"})
    self.assertIs(model_inputs["algo_config"], canonical_config)

  def test_grpo_canonical_defaults(self):
    """Verifies Discrepancy D9 and D10: canonical defaults kl_loss_mode='kl' and epsilon_high=epsilon."""
    adapter = algorithm_adapter.GRPOAdapter()
    self.assertIsInstance(adapter.algo_config, algorithm_config.GRPOConfig)
    self.assertEqual(adapter.algo_config.num_generations, 2)
    self.assertEqual(adapter.group_size, 2)
    self.assertEqual(adapter.algo_config.kl_loss_mode, "kl")
    self.assertEqual(adapter.algo_config.epsilon, 0.2)
    self.assertEqual(adapter.algo_config.epsilon_high, 0.2)
    self.assertEqual(adapter.algo_config.beta, 0.04)
    self.assertTrue(adapter.requires_reference_kl)

  def test_grpo_asymmetric_and_dual_clipping(self):
    """Verifies Discrepancy D10: config with epsilon_high and epsilon_c is propagated."""
    config = algorithm_config.GRPOConfig(
        num_generations=2,
        epsilon=0.2,
        epsilon_high=0.28,
        epsilon_c=0.1,
        temperature=1.0,
    )
    adapter = algorithm_adapter.GRPOAdapter(algo_config=config)
    self.assertEqual(adapter.algo_config.epsilon, 0.2)
    self.assertEqual(adapter.algo_config.epsilon_high, 0.28)
    self.assertEqual(adapter.algo_config.epsilon_c, 0.1)

    gen_fn = adapter.build_gen_model_input_fn(pad_id=0, eos_id=1)
    model_inputs = gen_fn({"fake": "batch"})
    algo_cfg = model_inputs["algo_config"]
    self.assertEqual(algo_cfg.epsilon_high, 0.28)
    self.assertEqual(algo_cfg.epsilon_c, 0.1)

  def test_grpo_algo_config_is_single_source_of_truth(self):
    """Verifies algo_config carries temperature/use_rollout_logps unmodified."""
    config = algorithm_config.GRPOConfig(
        num_generations=2,
        temperature=0.8,
        use_rollout_logps=False,
    )
    adapter = algorithm_adapter.GRPOAdapter(algo_config=config)
    # The adapter wraps the caller's config as-is, without copying or mutating.
    self.assertIs(adapter.algo_config, config)
    self.assertEqual(adapter.algo_config.temperature, 0.8)
    self.assertFalse(adapter.use_rollout_logps)
    gen_fn = adapter.build_gen_model_input_fn(pad_id=0, eos_id=1)
    model_inputs = gen_fn({"fake": "batch"})
    self.assertEqual(model_inputs["algo_config"].temperature, 0.8)



_ROUTING_LAYERS = 2
_ROUTING_TOP_K = 2


def _routing(length, fill):
  """`[length, num_layers, top_k]` routing where every slot holds `fill`."""
  shape = (length, _ROUTING_LAYERS, _ROUTING_TOP_K)
  return np.full(shape, fill, dtype=np.int32)


class RoutedExpertsForItemTest(absltest.TestCase):
  """`_routed_experts_for` must match the payload's sequence length exactly."""

  def _align(self, routed, seq_len=8):
    item = datatypes.TrajectoryItem(routed_experts=routed, traj={})
    return algorithm_adapter._routed_experts_for(item, seq_len)  # pylint: disable=protected-access

  def test_returns_none_without_capture(self):
    self.assertIsNone(self._align(None))

  def test_short_capture_is_padded_as_unset(self):
    """Missing tail rows must fall back to the gate, not replay expert 0."""
    out = self._align(_routing(5, 3))
    self.assertEqual(out.shape, (8, _ROUTING_LAYERS, _ROUTING_TOP_K))
    np.testing.assert_array_equal(out[:5], 3)
    np.testing.assert_array_equal(out[5:], datatypes.UNSET_ROUTED_EXPERT)

  def test_wrong_rank_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "length, num_layers, top_k"):
      self._align(np.zeros((8, _ROUTING_TOP_K), dtype=np.int32))


if __name__ == "__main__":
  absltest.main()
