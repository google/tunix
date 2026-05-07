# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for RLOO (REINFORCE Leave-One-Out) learner."""

import os
from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl.grpo import rloo_learner as rloo_lib
from tunix.tests import grpo_test_base
from tunix.tests import test_common as tc

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=2'


class RLOOConfigTest(parameterized.TestCase):
  """Tests for RLOOConfig defaults and validation."""

  def test_defaults(self):
    config = rloo_lib.RLOOConfig(num_generations=4)
    self.assertEqual(config.algo_variant, 'rloo')
    self.assertEqual(config.advantage_estimator, 'rloo')
    self.assertEqual(config.loss_agg_mode, 'token-mean')
    self.assertEqual(config.policy_loss_fn, 'grpo')

  def test_custom_params_preserve_fixed_fields(self):
    config = rloo_lib.RLOOConfig(
        num_generations=8,
        num_iterations=2,
        beta=0.1,
        epsilon=0.3,
        loss_agg_mode='sequence-mean-token-mean',
    )
    self.assertEqual(config.num_generations, 8)
    self.assertEqual(config.num_iterations, 2)
    self.assertEqual(config.beta, 0.1)
    self.assertEqual(config.epsilon, 0.3)
    self.assertEqual(config.loss_agg_mode, 'sequence-mean-token-mean')
    # init=False fields cannot be overridden by callers.
    self.assertEqual(config.algo_variant, 'rloo')
    self.assertEqual(config.advantage_estimator, 'rloo')

  def test_rejects_single_generation(self):
    with self.assertRaisesRegex(ValueError, 'num_generations'):
      rloo_lib.RLOOConfig(num_generations=1)


class RLOOAdvantageTest(parameterized.TestCase):
  """Tests for the RLOO advantage estimator function."""

  def test_basic(self):
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # Group 1: [1, 2, 3]; LOO baselines: [2.5, 2.0, 1.5].
    # Group 2: [4, 5, 6]; LOO baselines: [5.5, 5.0, 4.5].
    advantages = rloo_lib.compute_rloo_advantages(rewards, num_generations=3)
    expected = np.array([-1.5, 0.0, 1.5, -1.5, 0.0, 1.5])
    np.testing.assert_allclose(advantages, expected)

  def test_single_generation_returns_zeros(self):
    rewards = np.array([1.0, 2.0])
    advantages = rloo_lib.compute_rloo_advantages(rewards, num_generations=1)
    np.testing.assert_allclose(advantages, np.zeros_like(rewards))

  def test_two_generations(self):
    # With G=2 the LOO baseline for each sample is just the other one.
    rewards = np.array([3.0, 7.0, 1.0, 5.0])
    advantages = rloo_lib.compute_rloo_advantages(rewards, num_generations=2)
    expected = np.array([-4.0, 4.0, -4.0, 4.0])
    np.testing.assert_allclose(advantages, expected)

  def test_equal_rewards_yield_zero_advantages(self):
    rewards = np.array([5.0, 5.0, 5.0, 5.0])
    advantages = rloo_lib.compute_rloo_advantages(rewards, num_generations=2)
    np.testing.assert_allclose(advantages, np.zeros(4))

  def test_advantages_sum_to_zero_per_group(self):
    rng = np.random.default_rng(42)
    num_generations = 4
    num_groups = 3
    rewards = rng.standard_normal(num_groups * num_generations)
    advantages = rloo_lib.compute_rloo_advantages(
        rewards, num_generations=num_generations
    )
    reshaped = advantages.reshape(-1, num_generations)
    np.testing.assert_allclose(
        reshaped.sum(axis=-1), np.zeros(num_groups), atol=1e-6
    )

  def test_registered_in_advantage_registry(self):
    """RLOOLearner relies on advantage_estimator='rloo' being registered."""
    from tunix.rl import function_registry
    fn = function_registry.get_advantage_estimator('rloo')
    self.assertIs(fn, rloo_lib.compute_rloo_advantages)


class RLOOLearnerTest(parameterized.TestCase):
  """End-to-end tests for the RLOO learner training loop."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    num_cpus = int(os.environ.get('DEVICE_COUNTS', 2))
    chex.set_n_cpu_devices(num_cpus)

  @parameterized.named_parameters(
      dict(
          testcase_name='single_reward_fn',
          reward_fns=grpo_test_base.reward_1,
      ),
      dict(
          testcase_name='multiple_reward_fns',
          reward_fns=[grpo_test_base.reward_1, grpo_test_base.reward_2],
      ),
  )
  def test_trains_and_updates_params(self, reward_fns):
    rl_cluster, model, original_variables = grpo_test_base.setup(
        {'eval_every_n_steps': 2}
    )
    rl_cluster.with_external_metrics_logger(print)
    learner = rloo_lib.RLOOLearner(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        algo_config=rloo_lib.RLOOConfig(num_generations=2, num_iterations=1),
        metric_fns=[lambda **kwargs: {'test_metric': (1.0, np.mean)}],
    )
    self.assertFalse(learner.should_sync_weights)
    self.assertFalse(learner.can_enable_async_rollout)

    train_ds = grpo_test_base.dummy_dataset(
        grpo_test_base.MySource(repeat=10), batch_size=2
    )
    eval_ds = grpo_test_base.dummy_dataset(batch_size=1)
    learner.train(train_ds, eval_ds)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)
    self.assertEqual(learner._iter_steps, 10)

  def test_trains_with_kl_disabled(self):
    """beta=0.0 should skip reference inference and still update params."""
    rl_cluster, model, original_variables = grpo_test_base.setup(
        {'eval_every_n_steps': 2}
    )
    learner = rloo_lib.RLOOLearner(
        rl_cluster=rl_cluster,
        reward_fns=grpo_test_base.reward_1,
        algo_config=rloo_lib.RLOOConfig(
            num_generations=2, num_iterations=1, beta=0.0
        ),
    )
    train_ds = grpo_test_base.dummy_dataset(
        grpo_test_base.MySource(repeat=10), batch_size=2
    )
    learner.train(train_ds)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)

  def test_trains_with_multiple_iterations(self):
    rl_cluster, model, original_variables = grpo_test_base.setup(
        {'eval_every_n_steps': 5}
    )
    learner = rloo_lib.RLOOLearner(
        rl_cluster=rl_cluster,
        reward_fns=grpo_test_base.reward_1,
        algo_config=rloo_lib.RLOOConfig(num_generations=2, num_iterations=2),
    )
    train_ds = grpo_test_base.dummy_dataset(
        grpo_test_base.MySource(repeat=10), batch_size=2
    )
    learner.train(train_ds)

    variables = nnx.state(model, nnx.Param)
    jax.tree.map_with_path(tc.assert_not_equal, original_variables, variables)


if __name__ == '__main__':
  absltest.main()
