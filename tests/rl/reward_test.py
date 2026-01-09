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
import inspect
import os
from typing import Any, List
from absl import logging
from absl.testing import absltest
import chex
from flax import nnx
import jax
import jax.numpy as jnp
import mock
import numpy as np
import numpy.testing as npt
import optax
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import reward
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.tests import test_common as tc
from GOOGLE_INTERNAL_PACKAGE_PATH.testing.pybase import parameterized

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

Mesh = jax.sharding.Mesh


# --- Test Reward Functions ---
def len_reward(
    prompts: List[str], completions: List[str], **kwargs: Any
) -> List[float]:
  del prompts, kwargs  # Unused
  res = [float(len(c)) for c in completions]
  return res


len_reward.__name__ = "len_reward"


def prompt_len_reward(
    prompts: List[str],
    completions: List[str],
    custom_param: float = 1.0,
    **kwargs: Any,
) -> List[float]:
  del completions, kwargs  # Unused
  res = [custom_param * len(p) for p in prompts]
  return res


prompt_len_reward.__name__ = "prompt_len_reward"


def nan_reward(
    prompts: List[str], completions: List[str], **kwargs: Any
) -> List[float]:
  del completions, kwargs  # Unused
  return [np.nan] * len(prompts)


nan_reward.__name__ = "nan_reward"


@dataclasses.dataclass(slots=True, kw_only=True)
class TestAlgoConfig(algo_config_lib.AlgorithmConfig):
  """Test Algorithm Config."""

  reward_manager: str = "sequence-level"
  custom_param: float = 2.0


# --- Test Class ---
class SequenceRewardManagerTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    cls.num_cpus = int(os.environ.get("DEVICE_COUNTS", 4))
    chex.set_n_cpu_devices(cls.num_cpus)
    print(f"Setting up test with {cls.num_cpus} CPU devices before JAX init")
    cls.device_count = jax.device_count()

  def set_up_rl_cluster(self):
    """Sets up the RL cluster for testing."""
    split_index = self.device_count // 2

    actor_mesh = Mesh(
        np.array(jax.devices()[:split_index]).reshape(split_index, 1),
        ("fsdp", "tp"),
    )
    rollout_mesh = Mesh(
        np.array(jax.devices()[split_index:]).reshape(1, split_index),
        ("fsdp", "tp"),
    )
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: actor_mesh,
            rl_cluster_lib.Role.REFERENCE: actor_mesh,
            rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optax.sgd(1e-3),
            eval_every_n_steps=1,
            max_steps=10,
            gradient_accumulation_steps=None,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=10,
            max_prompt_length=256,
            kv_cache_size=1024,
            data_type=jnp.bfloat16,
        ),
    )

    vocab = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )
    ref_model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()), rngs=nnx.Rngs(0)
    )

    self.test_rl_cluster = rl_cluster_lib.RLCluster(
        actor=model,
        reference=ref_model,
        tokenizer=vocab,
        cluster_config=cluster_config,
    )

  def setUp(self):
    super().setUp()
    self.set_up_rl_cluster()
    self.test_algo_config = TestAlgoConfig()
    self.prompts = ["p1", "p22"]
    self.completions = ["c1_long", "c2"]
    self.reward_output = {}

  def test_initialization(self):
    manager = reward.SequenceRewardManager(
        reward_fns=len_reward,
        algo_config=self.test_algo_config,
        rl_cluster=self.test_rl_cluster,
    )
    self.assertEqual(manager.reward_fns, [len_reward])
    self.assertEqual(manager.algo_config, self.test_algo_config)
    self.assertEqual(manager.rl_cluster, self.test_rl_cluster)

  def test_single_reward_fn(self):
    manager = reward.SequenceRewardManager(
        reward_fns=[len_reward],
        algo_config=self.test_algo_config,
        rl_cluster=self.test_rl_cluster,
    )
    manager(
        self.prompts,
        self.completions,
        rl_cluster_lib.Mode.TRAIN,
        self.reward_output,
    )

    expected_rewards = np.array([float(len("c1_long")), float(len("c2"))])
    np.testing.assert_array_equal(
        self.reward_output["sequence-level-reward"], expected_rewards
    )
    self.assertLen(manager.rl_cluster._buffered_train_metrics, 1)

  def test_multiple_reward_fns(self):
    manager = reward.SequenceRewardManager(
        reward_fns=[len_reward, prompt_len_reward],
        algo_config=self.test_algo_config,
        rl_cluster=self.test_rl_cluster,
    )
    manager(
        self.prompts,
        self.completions,
        rl_cluster_lib.Mode.TRAIN,
        self.reward_output,
    )

    # custom_param is 2.0 from test_algo_config
    r1 = np.array(len_reward(self.prompts, self.completions))
    r2 = np.array(
        prompt_len_reward(self.prompts, self.completions, custom_param=2.0)
    )
    expected_rewards = r1 + r2
    rewards_matrix = np.array([r1, r2])
    np.testing.assert_array_almost_equal(
        self.reward_output["sequence-level-reward"], expected_rewards
    )
    test_metrics = manager.rl_cluster._buffered_train_metrics[0].metrics
    for metric_name, v in test_metrics.items():
      if metric_name.startswith("rewards/"):
        self.assertLen(v[0], 2)
    npt.assert_allclose(
        test_metrics["rewards/sum"][0],
        expected_rewards,
        err_msg="rewards/sum mismatch",
    )
    npt.assert_allclose(
        test_metrics["rewards/len_reward"][0],
        r1,
        err_msg="rewards/len_reward mismatch",
    )
    npt.assert_allclose(
        test_metrics["rewards/prompt_len_reward"][0],
        r2,
        err_msg="rewards/prompt_len_reward mismatch",
    )
    for col_idx in range(rewards_matrix.shape[0]):
      npt.assert_allclose(
          test_metrics["rewards/min"][0][col_idx],
          np.min(rewards_matrix[:, col_idx]),
      )
      npt.assert_allclose(
          test_metrics["rewards/max"][0][col_idx],
          np.max(rewards_matrix[:, col_idx]),
      )

  def test_algo_config_param_passing(self):
    # Mock the reward function to spy on its call arguments
    mock_fn = mock.Mock(wraps=prompt_len_reward)
    mock_fn.__name__ = prompt_len_reward.__name__
    # Restore the signature for introspection
    mock_fn.__signature__ = inspect.signature(prompt_len_reward)

    manager = reward.SequenceRewardManager(
        reward_fns=[mock_fn],
        algo_config=self.test_algo_config,
        rl_cluster=self.test_rl_cluster,
    )
    manager(
        self.prompts,
        self.completions,
        rl_cluster_lib.Mode.TRAIN,
        self.reward_output,
    )

    mock_fn.assert_called_once()
    _, kwargs = mock_fn.call_args
    self.assertEqual(kwargs["custom_param"], 2.0)
    self.assertNotIn(
        "another_param", kwargs
    )  # Not in prompt_len_reward signature

  def test_nan_handling(self):
    manager = reward.SequenceRewardManager(
        reward_fns=[len_reward, nan_reward],
        algo_config=self.test_algo_config,
        rl_cluster=self.test_rl_cluster,
    )
    manager(
        self.prompts,
        self.completions,
        rl_cluster_lib.Mode.TRAIN,
        self.reward_output,
    )
    # np.nansum should treat nan as 0 for summation
    expected_rewards = np.array([float(len(c)) for c in self.completions])
    np.testing.assert_array_almost_equal(
        self.reward_output["sequence-level-reward"], expected_rewards
    )
    # Check logged metrics for NaN
    logging.info(
        "self.test_rl_cluster._buffered_train_metrics: %s",
        self.test_rl_cluster._buffered_train_metrics,
    )
    test_metrics = manager.rl_cluster._buffered_train_metrics[0].metrics
    self.assertTrue(np.isnan(test_metrics["rewards/nan_reward"][0]).all())
    np.testing.assert_allclose(
        test_metrics["rewards/sum"][0],
        expected_rewards,
        err_msg="rewards/sum mismatch",
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="reward_fn_returns_none",
          reward_fns=[lambda prompts, completions, **kw: None],
          expected_regex="Failed to obtain result.*Result is None",
          error_type=RuntimeError,
      ),
      dict(
          testcase_name="reward_fn_bad_length",
          reward_fns=[
              lambda prompts, completions, **kw: [1.0] * (len(prompts) + 1)
          ],
          expected_regex="Length mismatch",
          error_type=RuntimeError,
      ),
  )
  def test_errors(
      self, expected_regex, error_type, kwargs=None, reward_fns=None
  ):
    if reward_fns is None:
      reward_fns = [len_reward]
    for i, fn in enumerate(reward_fns):
      if not hasattr(fn, "__name__"):
        fn.__name__ = f"test_fn_{i}"

    manager = reward.SequenceRewardManager(
        reward_fns=reward_fns,
        algo_config=self.test_algo_config,
        rl_cluster=self.test_rl_cluster,
    )
    with self.assertRaisesRegex(error_type, expected_regex):
      manager(
          self.prompts,
          self.completions,
          rl_cluster_lib.Mode.TRAIN,
          self.reward_output,
          **(kwargs or {}),
      )


def test_no_reward_fns_raises_error(self):
  with self.assertRaisesRegex(ValueError, "reward_fns cannot be empty"):
    reward.SequenceRewardManager(
        reward_fns=[],
        algo_config=self.test_algo_config,
        rl_cluster=self.test_rl_cluster,
    )


if __name__ == "__main__":
  absltest.main()
