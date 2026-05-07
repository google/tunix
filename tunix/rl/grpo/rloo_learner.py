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

"""RLOO (REINFORCE Leave-One-Out) learner.

RLOO is a variance-reduced policy gradient method that uses leave-one-out
baselines. For each completion in a group, the baseline is the mean reward
of all *other* completions to the same prompt. This yields an unbiased
advantage estimator with lower variance than the group-mean baseline used
in standard GRPO, without requiring a separate critic network.

Reference:
  - Ahmadian et al., "Back to Basics: Revisiting REINFORCE-Style Optimization
    for Learning from Human Feedback in LLMs", 2024.
    https://arxiv.org/abs/2402.14740
"""

import dataclasses
from typing import List, Sequence

import jax
import jax.numpy as jnp
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner
from tunix.rl.grpo import grpo_learner as grpo_learner_lib

TrainingInputT = rl_learner.TrainingInputT
RewardFn = rl_learner.RewardFn
MetricFn = rl_learner.MetricFn


@function_registry.register_advantage_estimator("rloo")
def compute_rloo_advantages(
    rewards: jax.Array, num_generations: int
) -> jax.Array:
  """Compute RLOO (REINFORCE Leave-One-Out) advantages.

  For each completion, the baseline is the mean reward of all *other*
  completions in the same generation group. The advantage is the completion's
  reward minus this leave-one-out baseline:

      baseline_i = (sum(rewards in group) - reward_i) / (G - 1)
      advantage_i = reward_i - baseline_i

  This produces unbiased advantage estimates with lower variance than the
  group-mean baseline used in standard GRPO.

  Args:
    rewards: Flat reward array of shape ``[B * G]`` where ``B`` is the number
      of unique prompts and ``G`` is ``num_generations``. Rewards for the same
      prompt must be contiguous.
    num_generations: Number of generations per prompt (``G``). Must be >= 2;
      otherwise zeros are returned (no baseline can be computed).

  Returns:
    Flat advantage array of the same shape as ``rewards``. Within each group,
    the advantages sum to zero.
  """
  if num_generations < 2:
    # RLOO requires at least 2 samples to calculate a baseline.
    return jnp.zeros_like(rewards)

  reshaped_rewards = rewards.reshape(-1, num_generations)
  loo_mean = (
      reshaped_rewards.sum(axis=-1, keepdims=True) - reshaped_rewards
  ) / (num_generations - 1)
  rloo_advantages = reshaped_rewards - loo_mean

  return rloo_advantages.flatten()


@dataclasses.dataclass(slots=True, kw_only=True)
class RLOOConfig(grpo_learner_lib.GRPOConfig):
  """Configuration for RLOO (REINFORCE Leave-One-Out).

  RLOO replaces GRPO's group-mean advantage estimator with a leave-one-out
  baseline, which provides lower variance while remaining unbiased. The
  advantage for each completion is computed as its reward minus the mean
  reward of all other completions to the same prompt.

  This is particularly effective when ``num_generations`` is moderate (4-16),
  as the leave-one-out baseline becomes more accurate with more samples.

  Attributes:
    algo_variant: The algorithm variant identifier. Always ``rloo``.
    advantage_estimator: The advantage estimator to use. Always ``rloo``.
    loss_agg_mode: The aggregation mode for the loss function. Defaults to
      ``token-mean`` following the RLOO paper's recommendation for
      token-level normalization.

  References:
    - RLOO: https://arxiv.org/abs/2402.14740
  """

  algo_variant: str = dataclasses.field(default="rloo", init=False)
  advantage_estimator: str = dataclasses.field(default="rloo", init=False)
  loss_agg_mode: str = "token-mean"


class RLOOLearner(grpo_learner_lib.GrpoLearner[RLOOConfig]):
  """RLOO (REINFORCE Leave-One-Out) learner.

  RLOO is a reinforcement learning algorithm that improves upon GRPO by using
  leave-one-out baselines for advantage estimation. For each completion in a
  group, the baseline is the average reward of all *other* completions to
  the same prompt, rather than the group mean (which includes the completion
  itself).

  This approach:
    - Produces unbiased advantage estimates (like GRPO).
    - Achieves lower variance than the standard group-mean baseline.
    - Requires no additional model parameters (unlike PPO's critic).
    - Is most effective with moderate group sizes (4-16 generations).

  The learner inherits the full GRPO training loop, including clipped
  importance sampling, optional KL penalty, and multi-iteration support.
  Only the advantage computation differs.

  Example usage::

      rloo_config = RLOOConfig(
          num_generations=8,
          beta=0.04,
          epsilon=0.2,
      )
      learner = RLOOLearner(
          rl_cluster=rl_cluster,
          algo_config=rloo_config,
          reward_fns=reward_fn,
      )
      learner.train(train_ds)
  """

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      algo_config: RLOOConfig,
      reward_fns: RewardFn | List[RewardFn],
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    """Initializes the ``RLOOLearner``.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      algo_config: An instance of ``RLOOConfig`` containing all
        training-specific configuration options.
      reward_fns: A single callable or a list of callables that compute a
        scalar reward for given prompts and completions. Each function should
        accept ``prompts``, ``completions`` and optional keyword arguments,
        and return a list of float rewards.
      metric_fns: A sequence of callables that compute metrics for the
        completions.
      data_shuffle_seed: The seed used to shuffle the training data.
    """
    super().__init__(
        rl_cluster=rl_cluster,
        algo_config=algo_config,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
    )
