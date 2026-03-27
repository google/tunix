import os
import unittest
from unittest import mock

from absl import logging
import numpy as np
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import reward_manager


def reward_fn1(prompts, completions, **kwargs):
  del kwargs
  return [1.0] * len(prompts)


def reward_fn2(prompts, completions, **kwargs):
  del kwargs
  return [2.0] * len(prompts)


class RewardManagerTest(unittest.TestCase):

  @mock.patch("tunix.rl.reward_manager.asdict")
  def test_prepare_log_metrics_and_log_one_example(self, mock_asdict):
    mock_asdict.return_value = {}
    reward_fns = [reward_fn1, reward_fn2]
    algo_config = mock.MagicMock(spec=algo_config_lib.AlgorithmConfig)
    reward_manager_instance = reward_manager.SequenceRewardManager(
        reward_fns=reward_fns, algo_config=algo_config
    )
    prompts = ["prompt1", "prompt2"]
    completions = ["completion1", "completion2"]

    with (
        mock.patch.dict(os.environ, {"TUNIX_DEBUG_REWARDS": "1"}),
        mock.patch("absl.logging.info") as mock_log_info,
    ):
      rewards_info = reward_manager_instance(prompts, completions)
      log_metrics = rewards_info["log_metrics"]

      self.assertIn("prompts", log_metrics)
      self.assertIn("completions", log_metrics)
      self.assertIn("rewards/sum", log_metrics)
      np.testing.assert_allclose(log_metrics["rewards/sum"][0], [3.0, 3.0])
      self.assertIn("rewards/reward_fn1", log_metrics)
      np.testing.assert_allclose(
          log_metrics["rewards/reward_fn1"][0], [1.0, 1.0]
      )
      self.assertIn("rewards/reward_fn2", log_metrics)
      np.testing.assert_allclose(
          log_metrics["rewards/reward_fn2"][0], [2.0, 2.0]
      )

      # Check that _log_one_example was called and logged correctly
      mock_log_info.assert_any_call("======= example rewards =======")
      mock_log_info.assert_any_call("%s:\t%s", "prompts", "prompt1")
      mock_log_info.assert_any_call("%s:\t%s", "completions", "completion1")
      mock_log_info.assert_any_call("%s:\t%s", "rewards/sum", "3.0")
      mock_log_info.assert_any_call("%s:\t%s", "rewards/mean", "1.5")
      mock_log_info.assert_any_call("%s:\t%s", "rewards/min", "1.0")
      mock_log_info.assert_any_call("%s:\t%s", "rewards/max", "2.0")
      mock_log_info.assert_any_call("%s:\t%s", "rewards/reward_fn1", "1.0")
      mock_log_info.assert_any_call("%s:\t%s", "rewards/reward_fn2", "2.0")
      mock_log_info.assert_any_call("=======================")


if __name__ == "__main__":
  unittest.main()
