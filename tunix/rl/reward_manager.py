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

"""Reward output for RL."""

import abc
from dataclasses import asdict
import inspect
from typing import Any, Callable, Dict, List, Sequence
from absl import logging
import numpy as np
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import function_registry

RewardFn = Callable[..., Any]


class AbstractRewardManager(abc.ABC):
  """Abstract base class for managing and orchestrating multiple reward function outputs."""

  def __init__(
      self,
      reward_fns: RewardFn | List[RewardFn],
      algo_config: algo_config_lib.AlgorithmConfig,
  ):
    """Initializes the manager with a list of callable reward function objects.

    Args:
        reward_fns: A list of reward functions or models.
        algo_config: The algorithm config to use for reward function
          configuration.
    """
    self.reward_fns = (
        [reward_fns] if not isinstance(reward_fns, Sequence) else reward_fns
    )

    if not self.reward_fns:
      raise ValueError(
          "reward_fns cannot be empty. You must provide at least one reward"
          " function."
      )
    self.algo_config = algo_config

  @abc.abstractmethod
  def __call__(
      self,
      prompts: List[str],
      completions: List[str],
      **kwargs,
  ) -> Dict[str, Any]:
    """Computes the rewards for completions using the provided reward functions.

    Args:
        prompts: A list of input prompts.
        completions: A list of generated text completions.
        mode: The mode to use for logging metrics.
        **kwargs: Additional keyword arguments passed to the reward functions.

    Returns:
        A dictionary of rewards information, including the final rewards for
        advantage computation and intermediate rewards for logging.
    """
    pass


@function_registry.register_reward_manager("sequence-level")
class SequenceRewardManager(AbstractRewardManager):
  """Reward manager for sequence-level rewards only."""

  def __init__(
      self,
      reward_fns: RewardFn | List[RewardFn],
      algo_config: algo_config_lib.AlgorithmConfig,
      **kwargs,
  ):
    """Initializes the manager with a list of callable reward function objects."""
    super().__init__(reward_fns, algo_config)

  def __call__(
      self,
      prompts: List[str],
      completions: List[str],
      **kwargs,
  ) -> Dict[str, Any]:
    """Computes the rewards for completions using the provided reward function, and return the sequence-level rewards information for advantage computationand logging."""
    return self._compute_rewards(prompts, completions, **kwargs)

  def _compute_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      **kwargs,
  ) -> Dict[str, Any]:
    """Computes the rewards for completions using the provided reward functions."""

    algo_config_params = asdict(self.algo_config)
    base_kwargs = kwargs.copy()

    num_prompts = len(prompts)
    num_reward_fns = len(self.reward_fns)
    rewards = np.zeros((num_prompts, num_reward_fns))

    # Compute all rewards for each prompt-completion pair.
    for i, reward_fn in enumerate(self.reward_fns):
      # Update the kwargs with the algo_config parameters.
      signature = inspect.signature(reward_fn)
      reward_fn_config_params = {}
      # Iterate over the function's expected parameters
      for name, _ in signature.parameters.items():
        # Skip standard parameters that are always passed (self, prompts, completions, kwargs)
        if name in ["self", "prompts", "completions", "kwargs"]:
          continue

        # Check if the parameter name matches a key in the algo_config dict. If
        # so, set the value to the algo_config parameter value, otherwise respect the value in the base_kwargs.
        if name in algo_config_params and name not in base_kwargs:
          reward_fn_config_params[name] = algo_config_params[name]

      call_kwargs = base_kwargs.copy()
      call_kwargs.update(reward_fn_config_params)

      r = reward_fn(prompts=prompts, completions=completions, **call_kwargs)

      if r is None:
        raise RuntimeError(
            f"Failed to obtain result from {reward_fn.__name__}. Result is"
            " None."
        )
      if isinstance(r, list) and len(r) != len(prompts):
        raise RuntimeError(
            f"Length mismatch after {reward_fn.__name__}: "
            f"len(r)={len(r)}, len(prompts)={num_prompts}. "
            f"Content of r: {r}"
        )

      rewards[:, i] = np.array(r)

    # Sum rewards across all reward functions for each prompt.
    sum_rewards = np.nansum(rewards, axis=1)
    print(f"Rewards for prompts: {sum_rewards}")

    # Prepare metrics for logging.
    log_metrics = self._prepare_log_metrics(
        prompts,
        completions,
        rewards,
        sum_rewards,
    )
    rewards_info = {
        "rewards": sum_rewards,
        "log_metrics": log_metrics,
    }
    return rewards_info

  def _prepare_log_metrics(
      self,
      prompts: List[str],
      completions: List[str],
      rewards: np.ndarray,  # (num_prompts, num_reward_fns)
      sum_rewards: np.ndarray,  # (num_prompts,)
  ) -> Dict[str, Any]:
    """Logs individual and summed rewards, along with prompts/completions, for each trajectory."""
    # Assuming self.reward_fns and self.rl_cluster are accessible instance attributes
    metrics_to_log = {}

    # Log prompts and completions.
    metrics_to_log["prompts"] = (prompts, None)
    metrics_to_log["completions"] = (completions, None)

    # Log the sum rewards for each prompt-completion pair.
    metrics_to_log["rewards/sum"] = (sum_rewards, np.mean)

    # Log the min and max rewards for the prompt-completion pair.
    metrics_to_log["rewards/min"] = (np.min(rewards, axis=1), np.min)
    metrics_to_log["rewards/max"] = (np.max(rewards, axis=1), np.max)

    # Log individual rewards for this trajectory
    for i, reward_fn in enumerate(self.reward_fns):
      metric_name = f"rewards/{reward_fn.__name__}"
      metrics_to_log[metric_name] = (rewards[:, i], np.mean)

    return metrics_to_log
