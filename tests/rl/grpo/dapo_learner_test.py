# Copyright 2025 Google LLC
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

import dataclasses
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax.numpy as jnp
from tunix.rl import function_registry as fr
from tunix.rl.grpo import dapo_learner as dapo_lib
from tunix.rl.grpo import grpo_learner as grpo_lib
from tunix.tests import test_common as tc


class DAPOlearnerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_model = mock.MagicMock()
    self.pad_id = 0
    self.eos_id = 1

    # Common data shapes
    self.batch_size = 2
    self.seq_len = 4
    self.prompt_ids = jnp.zeros(
        (self.batch_size, self.seq_len), dtype=jnp.int32
    )
    self.completion_ids = jnp.ones(
        (self.batch_size, self.seq_len), dtype=jnp.int32
    )
    self.completion_mask = jnp.array(
        [[1, 1, 1, 0], [1, 1, 0, 0]], dtype=jnp.float32
    )
    self.advantages = jnp.array([0.5, -0.2], dtype=jnp.float32)
    self.ref_per_token_logps = (
        jnp.ones_like(self.completion_ids, dtype=jnp.float32) * -0.2
    )
    self.old_per_token_logps = (
        jnp.ones_like(self.completion_ids, dtype=jnp.float32) * -0.15
    )

  def create_train_example(self):
    example = mock.MagicMock()
    example.prompt_ids = self.prompt_ids
    example.completion_ids = self.completion_ids
    example.completion_mask = self.completion_mask
    example.advantages = self.advantages
    example.ref_per_token_logps = self.ref_per_token_logps
    example.old_per_token_logps = self.old_per_token_logps
    example.segment_ids = None
    example.segment_positions = None
    example.sampler_is_weights = None
    example.routed_experts = None
    return example

  def test_diff_loss(self):
    dapo_config = dapo_lib.DAPOConfig()
    grpo_config = grpo_lib.GRPOConfig()
    dapo_config.temperature = 1.0
    grpo_config.temperature = 1.0

    dapo_loss_fn_impl = fr.default_registry.get(
        "policy_loss_fn", dapo_config.policy_loss_fn
    )
    grpo_loss_fn_impl = fr.default_registry.get(
        "policy_loss_fn", grpo_config.policy_loss_fn
    )

    # Test that the functions is same
    self.assertEqual(dapo_loss_fn_impl, grpo_loss_fn_impl)

    # Create the same input for both functions
    train_example = self.create_train_example()
    pad_id = self.pad_id
    eos_id = self.eos_id
    vocab = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )

    # Call DAPO loss function (DAPO sets ref_per_token_logps to None as it
    # doesn't fetch it).
    dapo_train_example = self.create_train_example()
    dapo_train_example.ref_per_token_logps = None
    dapo_loss_output = dapo_loss_fn_impl(
        model, dapo_train_example, dapo_config, pad_id, eos_id
    )
    dapo_loss = dapo_loss_output.primary_loss.compute()
    dapo_aux = dapo_loss_output.aux_metrics

    # Call GRPO loss function
    grpo_loss_output = grpo_loss_fn_impl(
        model, train_example, grpo_config, pad_id, eos_id
    )
    grpo_loss = grpo_loss_output.primary_loss.compute()
    grpo_aux = grpo_loss_output.aux_metrics

    # Assert that the loss values are different
    self.assertNotEqual(
        dapo_loss,
        grpo_loss,
        msg=(
            "DAPO and GRPO loss values should be different for the same input"
            " due to different loss aggregation logics."
        ),
    )

    self.assertIn("kl", dapo_aux)
    self.assertIn("kl", grpo_aux)
    self.assertEqual(
        dapo_aux["kl"].compute(), 0.0
    )  # DAPO does not have KL term.


class TestDAPOConfigPostInit(parameterized.TestCase):

  def test_valid_default(self):
    """Tests that default values pass validation."""
    try:
      dapo_lib.DAPOConfig()
    except ValueError as e:
      self.fail(f"DAPOConfig raised ValueError on default initialization: {e}")

  def test_drop_zero_variance_groups_defaults_on_and_is_tunable(self):
    self.assertTrue(dapo_lib.DAPOConfig().drop_zero_variance_groups)
    self.assertIsNone(dapo_lib.DAPOConfig().max_consecutive_filtered_groups)
    config = dapo_lib.DAPOConfig(
        drop_zero_variance_groups=False, max_consecutive_filtered_groups=16
    )
    self.assertFalse(config.drop_zero_variance_groups)
    self.assertEqual(config.max_consecutive_filtered_groups, 16)
    with self.assertRaisesRegex(ValueError, "positive int or None"):
      dapo_lib.DAPOConfig(max_consecutive_filtered_groups=0)

  def test_dynamic_sampling_deprecated_alias(self):
    with self.assertWarnsRegex(DeprecationWarning, "dynamic_sampling"):
      cfg = dapo_lib.DAPOConfig(dynamic_sampling=False)
    self.assertFalse(cfg.drop_zero_variance_groups)
    with self.assertWarnsRegex(DeprecationWarning, "dynamic_sampling"):
      self.assertFalse(cfg.dynamic_sampling)
    with self.assertWarnsRegex(DeprecationWarning, "dynamic_sampling"):
      self.assertTrue(dapo_lib.DAPOConfig().dynamic_sampling)

    replaced_epsilon = dataclasses.replace(dapo_lib.DAPOConfig(), epsilon=0.1)
    self.assertEqual(replaced_epsilon.epsilon, 0.1)
    self.assertTrue(replaced_epsilon.drop_zero_variance_groups)
    replaced_flag = dataclasses.replace(
        dapo_lib.DAPOConfig(), drop_zero_variance_groups=False
    )
    self.assertFalse(replaced_flag.drop_zero_variance_groups)

    with self.assertRaisesRegex(
        ValueError,
        "Cannot specify both 'drop_zero_variance_groups' and"
        " 'dynamic_sampling'",
    ):
      dapo_lib.DAPOConfig(
          drop_zero_variance_groups=True, dynamic_sampling=False
      )

  def test_dapo_learner_warns_instead_of_raising_on_default_config(self):
    vocab = tc.MockVocab()
    toy_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )
    mock_engine = mock.MagicMock()
    mock_engine.actor_trainer.model = toy_model
    mock_engine.rollout.model.return_value = toy_model
    mock_engine.get_rollout_config.return_value.temperature = 1.0
    mock_engine.cluster_config.rollout_engine = "vanilla"
    mock_engine.cluster_config.training_config.max_steps = 1
    mock_engine.cluster_config.training_config.eval_every_n_steps = 1
    mock_engine.cluster_config.training_config.mini_batch_size = 1
    for algo_config in (
        dapo_lib.DAPOConfig(),
        dapo_lib.DAPOConfig(algo_variant="grpo"),
    ):
      with self.subTest(algo_variant=algo_config.algo_variant):
        with self.assertLogs(level="WARNING") as cm:
          dapo_lib.DAPOLearner(
              rl_engine=mock_engine,
              algo_config=algo_config,
              reward_fns=lambda prompts, completions, **_: [1.0] * len(prompts),
          )
        self.assertTrue(
            any(
                "drop_zero_variance_groups=True is not yet implemented" in msg
                for msg in cm.output
            )
        )
    with self.assertRaisesRegex(
        ValueError, "only supported by the distributed StandardRLProgram"
    ):
      grpo_lib.GRPOLearner(
          rl_engine=mock_engine,
          algo_config=dapo_lib.DAPOConfig(),
          reward_fns=lambda prompts, completions, **_: [1.0] * len(prompts),
      )

  def test_beta_defaults_to_zero(self):
    """DAPO has no KL term; beta must be 0.0 so learners skip the ref model."""
    self.assertEqual(dapo_lib.DAPOConfig().beta, 0.0)

  @parameterized.named_parameters(
      dict(testcase_name="custom_epsilons", epsilon=0.1, epsilon_high=0.15),
      dict(testcase_name="epsilons_equal", epsilon=0.1, epsilon_high=0.1),
      dict(
          testcase_name="buffer_disabled",
          overlong_buffer={"enable": False},
      ),
      dict(testcase_name="buffer_none", overlong_buffer=None),
      dict(
          testcase_name="valid_buffer",
          overlong_buffer={
              "enable": True,
              "overlong_buffer_length": 2000,
              "overlong_buffer_penalty": 0.5,
              "max_response_length": 10000,
          },
      ),
  )
  def test_valid_configurations(self, **kwargs):
    """Tests various valid custom configurations."""
    try:
      dapo_lib.DAPOConfig(**kwargs)
    except ValueError as e:
      self.fail(f"DAPOConfig raised ValueError for valid case {kwargs}: {e}")

  @parameterized.named_parameters(
      dict(
          testcase_name="nonzero_beta",
          config_kwargs=dict(beta=0.1),
          expected_regex="beta must be 0.0",
      ),
      dict(
          testcase_name="base_validation_num_generations",
          config_kwargs=dict(num_generations=1),
          expected_regex="num_generations must be greater than 1",
      ),
      dict(
          testcase_name="invalid_epsilon_high",
          config_kwargs=dict(epsilon=0.2, epsilon_high=0.1),
          expected_regex=(
              "epsilon_high must be greater than or equal to epsilon."
          ),
      ),
      dict(
          testcase_name="buffer_missing_length",
          config_kwargs=dict(
              overlong_buffer={
                  "enable": True,
                  "overlong_buffer_penalty": 1.0,
                  "max_response_length": 20480,
              }
          ),
          expected_regex=(
              "overlong_buffer is enabled but missing.*overlong_buffer_length.*"
          ),
      ),
      dict(
          testcase_name="buffer_missing_penalty",
          config_kwargs=dict(
              overlong_buffer={
                  "enable": True,
                  "overlong_buffer_length": 4096,
                  "max_response_length": 20480,
              }
          ),
          expected_regex=(
              "overlong_buffer is enabled but missing"
              ".*overlong_buffer_penalty.*"
          ),
      ),
      dict(
          testcase_name="buffer_missing_max_length",
          config_kwargs=dict(
              overlong_buffer={
                  "enable": True,
                  "overlong_buffer_length": 4096,
                  "overlong_buffer_penalty": 1.0,
              }
          ),
          expected_regex=(
              "overlong_buffer is enabled but missing.*max_response_length.*"
          ),
      ),
      dict(
          testcase_name="buffer_length_is_none",
          config_kwargs=dict(
              overlong_buffer={
                  "enable": True,
                  "overlong_buffer_length": None,
                  "overlong_buffer_penalty": 1.0,
                  "max_response_length": 20480,
              }
          ),
          expected_regex=(
              "overlong_buffer is enabled but missing.*overlong_buffer_length.*"
          ),
      ),
      dict(
          testcase_name="negative_penalty",
          config_kwargs=dict(
              overlong_buffer={
                  "enable": True,
                  "overlong_buffer_length": 4096,
                  "overlong_buffer_penalty": -0.5,
                  "max_response_length": 20480,
              }
          ),
          expected_regex="overlong_buffer_penalty must be non-negative",
      ),
  )
  def test_invalid_configurations(self, config_kwargs, expected_regex):
    """Tests various invalid configurations that should raise ValueError."""
    with self.assertRaisesRegex(ValueError, expected_regex):
      dapo_lib.DAPOConfig(**config_kwargs)


class RewardShapingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_cluster = mock.MagicMock()

  def test_raises_error_on_none_buffer(self):
    with self.assertRaisesRegex(
        ValueError, "reward_shaping is called but with empty overlong_buffer."
    ):

      dapo_lib.reward_shaping(
          prompts=["test prompt"],
          completions=["test completion"],
          mode=self.mock_cluster.Mode,
          overlong_buffer=None,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="under_length",
          lengths=[70],
          expected_scores=[0.0],
      ),
      dict(
          testcase_name="at_expected_length",
          lengths=[80],
          expected_scores=[0.0],
      ),
      dict(
          testcase_name="in_buffer_zone",
          lengths=[90],
          expected_scores=[-5.0],
      ),
      dict(
          testcase_name="at_max_length",
          lengths=[100],
          expected_scores=[-10.0],
      ),
      dict(
          testcase_name="over_max_length",
          lengths=[110],
          expected_scores=[-15.0],
      ),
      dict(
          testcase_name="mixed_lengths",
          lengths=[70, 80, 90, 100, 110],
          expected_scores=[0.0, 0.0, -5.0, -10.0, -15.0],
      ),
      dict(
          testcase_name="zero_penalty",
          lengths=[110],
          expected_scores=[0.0],
          penalty=0,
      ),
  )
  def test_reward_scores(self, lengths, expected_scores, penalty=10):
    completions = ["a" * length for length in lengths]
    overlong_buffer = {
        "overlong_buffer_length": 20,
        "overlong_buffer_penalty": penalty,
        "max_response_length": 100,
    }
    # expected_response_length = 100 - 20 = 80

    scores = dapo_lib.reward_shaping(
        prompts=[""] * len(completions),
        completions=completions,
        mode=self.mock_cluster.Mode,
        overlong_buffer=overlong_buffer,
    )

    self.assertSequenceAlmostEqual(expected_scores, scores, places=4)


if __name__ == "__main__":
  absltest.main()
