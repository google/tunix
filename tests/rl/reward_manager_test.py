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
from typing import Any, List
from unittest import mock
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import numpy.testing as npt
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import reward_manager


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

  def setUp(self):
    super().setUp()
    self.test_algo_config = TestAlgoConfig()
    self.prompts = ["p1", "p22"]
    self.completions = ["c1_long", "c2"]

  def test_initialization(self):
    manager = reward_manager.SequenceRewardManager(
        reward_fns=len_reward,
        algo_config=self.test_algo_config,
    )
    self.assertEqual(manager.reward_fns, [len_reward])
    self.assertEqual(manager.algo_config, self.test_algo_config)

  def test_single_reward_fn(self):
    manager = reward_manager.SequenceRewardManager(
        reward_fns=[len_reward],
        algo_config=self.test_algo_config,
    )
    rewards_info = manager(
        self.prompts,
        self.completions,
    )

    expected_rewards = np.array([float(len("c1_long")), float(len("c2"))])
    np.testing.assert_array_equal(rewards_info["rewards"], expected_rewards)
    self.assertLen(rewards_info["log_metrics"], 6)

  def test_multiple_reward_fns(self):
    manager = reward_manager.SequenceRewardManager(
        reward_fns=[len_reward, prompt_len_reward],
        algo_config=self.test_algo_config,
    )
    rewards_info = manager(
        self.prompts,
        self.completions,
    )

    # custom_param is 2.0 from test_algo_config
    r1 = np.array(len_reward(self.prompts, self.completions))
    r2 = np.array(
        prompt_len_reward(self.prompts, self.completions, custom_param=2.0)
    )
    expected_rewards = r1 + r2
    rewards_matrix = np.array([r1, r2])
    np.testing.assert_array_almost_equal(
        rewards_info["rewards"], expected_rewards
    )
    test_metrics = rewards_info["log_metrics"]
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

    manager = reward_manager.SequenceRewardManager(
        reward_fns=[mock_fn],
        algo_config=self.test_algo_config,
    )
    manager(
        self.prompts,
        self.completions,
    )

    mock_fn.assert_called_once()
    _, kwargs = mock_fn.call_args
    self.assertEqual(kwargs["custom_param"], 2.0)
    self.assertNotIn(
        "another_param", kwargs
    )  # Not in prompt_len_reward signature

  def test_nan_handling(self):
    manager = reward_manager.SequenceRewardManager(
        reward_fns=[len_reward, nan_reward],
        algo_config=self.test_algo_config,
    )
    rewards_info = manager(
        self.prompts,
        self.completions,
    )
    # np.nansum should treat nan as 0 for summation
    expected_rewards = np.array([float(len(c)) for c in self.completions])
    np.testing.assert_array_almost_equal(
        rewards_info["rewards"], expected_rewards
    )
    # Check logged metrics for NaN
    test_metrics = rewards_info["log_metrics"]
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

    manager = reward_manager.SequenceRewardManager(
        reward_fns=reward_fns,
        algo_config=self.test_algo_config,
    )
    with self.assertRaisesRegex(error_type, expected_regex):
      manager(
          self.prompts,
          self.completions,
          **(kwargs or {}),
      )

  def test_no_reward_fns_raises_error(self):
    with self.assertRaisesRegex(ValueError, "reward_fns cannot be empty"):
      reward_manager.SequenceRewardManager(
          reward_fns=[],
          algo_config=self.test_algo_config,
      )


if __name__ == "__main__":
  absltest.main()
