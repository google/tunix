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
"""Helper functions for GDPO Trainer."""

from typing import Callable, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner as grpo_learner_lib

RewardFn = Callable[..., List[float]]

MetricFn = Callable[..., rl_cluster_lib.MetricsT]


class GDPOConfig(grpo_learner_lib.GRPOConfig):
  """Configuration for GDPO.

  Attributes:
   algo_variant: The core algorithm variant to use.
   advantage_estimator: The advantage estimator to use.
   References: - GDPO:
     https://arxiv.org/abs/2601.05242
  """

  algo_variant: str = "gdpo"
  advantage_estimator: str = "gdpo"


class GDPOLearner(grpo_learner_lib.GrpoLearner[GDPOConfig]):
  """GDPO learner."""

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      algo_config: GDPOConfig,
      reward_fns: RewardFn | List[RewardFn],
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    """Initializes the `GDPOLearner`."""
    super().__init__(
        rl_cluster=rl_cluster,
        algo_config=algo_config,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
    )

  def _compute_rewards(
      self,
      prompts: List[str],
      completions: List[str],
      mode: rl_cluster_lib.Mode,
      step: int | None = None,
      **kwargs,
  ) -> np.ndarray:
    """Computes the rewards for completions using the provided reward functions.

    Args:
      prompts: A list of input prompts.
      completions: A list of generated text completions.
      mode: The mode to use for logging metrics.
      step: The current training step.
      **kwargs: Additional keyword arguments passed to the reward functions.

    Returns:
      A numpy array (shape `[B]`) of scalar rewards for
      each prompt-completion pair. The rewards are the sum across all the
      provided reward functions.

    Raises:
        RuntimeError: If 'r' reward is None, indicating a failure to obtain the
        result, or if the length of 'r' reward does not match the length of
        'prompts'.
    """
    if "mode" in kwargs:
      raise ValueError(f"kwargs already contains mode as a key: {kwargs}")
    kwargs["mode"] = str(mode)

    num_prompts = len(prompts)
    num_reward_fns = len(self.reward_fns)
    rewards = np.zeros((num_prompts, num_reward_fns))

    # Compute all rewards for each prompt-completion pair.
    for i, reward_fn in enumerate(self.reward_fns):
      r = reward_fn(prompts=prompts, completions=completions, **kwargs)

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

    # Log all metrics in a single loop
    for j, (prompt, completion) in enumerate(zip(prompts, completions)):
      metrics_to_log = {}

      # Log prompts and completions.
      metrics_to_log["prompts"] = (prompt, None)
      metrics_to_log["completions"] = (completion, None)

      # Log the summed rewards for this trajectory.
      trajectory_sum = sum_rewards[j]
      metrics_to_log["rewards/sum"] = (trajectory_sum, np.mean)
      metrics_to_log["rewards/min"] = (np.min(rewards[j]), np.min)
      metrics_to_log["rewards/max"] = (np.max(rewards[j]), np.max)

      # Log individual rewards for this trajectory
      for i, reward_fn in enumerate(self.reward_fns):
        metric_name = f"rewards/{reward_fn.__name__}"
        metrics_to_log[metric_name] = (rewards[j, i], np.mean)

      # Log all metrics for this trajectory in one call
      if step is not None:
        self.rl_cluster.buffer_metrics_async(
            metrics_to_log, mode=mode, step=step
        )
      else:
        self.rl_cluster.buffer_metrics(metrics_to_log, mode=mode)

    return jnp.array(rewards)


@function_registry.register_advantage_estimator("gdpo")
def compute_advantages(rewards: jax.Array, num_generations: int) -> jax.Array:
  """Compute group reward decoupled normalization advantages.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.

  Returns:
    Group reward decoupled normalization advantages.
  """
  rewards_per_func = jnp.nan_to_num(rewards)
  all_reward_advantage = []
  for reward_index in range(rewards.shape[-1]):
    reward_for_index = rewards_per_func[:, reward_index]
    each_reward_mean_grouped = reward_for_index.reshape(
        -1, num_generations
    ).mean(axis=1)
    each_reward_std_grouped = reward_for_index.reshape(-1, num_generations).std(
        axis=1
    )
    each_reward_mean_grouped = each_reward_mean_grouped.repeat(num_generations)
    each_reward_std_grouped = each_reward_std_grouped.repeat(num_generations)
    each_reward_advantage = reward_for_index - each_reward_mean_grouped
    each_reward_advantage = each_reward_advantage / (
        each_reward_std_grouped + 1e-4
    )
    all_reward_advantage.append(each_reward_advantage)

  combined_reward_advantage = jnp.stack(all_reward_advantage, axis=1)
  pre_bn_advantages = jnp.nansum(combined_reward_advantage, axis=1)

  bn_advantages_mean = pre_bn_advantages.mean()
  bn_advantages_std = pre_bn_advantages.std()
  advantages = (pre_bn_advantages - bn_advantages_mean) / (
      bn_advantages_std + 1e-4
  )
  return advantages
