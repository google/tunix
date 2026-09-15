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

import dataclasses
from typing import Callable, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner
from tunix.rl.grpo import grpo_learner as grpo_learner_lib

TrainingInputT = rl_learner.TrainingInputT
RewardFn = Callable[..., List[float]]
MetricFn = Callable[..., rl_cluster_lib.MetricsT]


@dataclasses.dataclass(slots=True, kw_only=True)
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

  def _compute_gdpo_specific_rewards(
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

  def _generate_and_compute_advantage(
      self,
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> grpo_learner_lib.TrainExample:
    """Generate text completions and compute the advantages for GRPO training.

    Args:
      training_input: A dictionary containing the training input data,
        containing the key 'prompts'.
      mode: The mode to use for logging metrics.

    Returns:
      A `TrainExample` instance containing the processed input data, including
      prompt IDs, completion IDs, masks, advantages, and per-token log
      probabilities from the reference and policy models.
    """
    training_input["prompts"] = list(training_input["prompts"])
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()
    rollout_output = self.rl_cluster.generate(
        prompts=training_input["prompts"],
        mode=mode,
        micro_batch_size=(
            self._rollout_micro_batch_size * self.algo_config.num_generations
        ),
    )
    completion_ids = rollout_output.tokens
    prompt_ids = jnp.array(rollout_output.left_padded_prompt_tokens)
    completion_text = rollout_output.text

    # Assemble masks
    completion_padding_mask = np.not_equal(completion_ids, pad_value)
    completion_mask = common.np_make_completion_mask(
        completion_ids, eos_tok=eos_value
    )
    # Apply the padding mask to the completion mask.
    completion_mask = completion_mask * completion_padding_mask

    # Convert completion_ids and completion_mask to jax arrays
    jax_completion_ids = jnp.array(completion_ids)

    if self.algo_config.beta != 0.0:
      devices = self.rl_cluster.r2m[rl_cluster_lib.Role.REFERENCE].devices
      # TODO(yangmu): use function decorator to trace this part, same below.
      with self.rl_cluster.perf.span("refer_inference", devices) as interval:
        ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
            prompt_tokens=prompt_ids,
            completion_tokens=jax_completion_ids,
            pad_id=pad_value,
            eos_id=eos_value,
            micro_batch_size=(
                self._compute_logps_micro_batch_size
                * self.algo_config.num_generations
            ),
        )
        interval.device_end([ref_per_token_logps])
    else:
      ref_per_token_logps = None
    if self.algo_config.num_iterations > 1:
      devices = self.rl_cluster.r2m[rl_cluster_lib.Role.ACTOR].devices
      with self.rl_cluster.perf.span(
          "old_actor_inference", devices
      ) as interval:
        old_per_token_logps = self.rl_cluster.get_old_per_token_logps(
            prompt_tokens=prompt_ids,
            completion_tokens=jax_completion_ids,
            micro_batch_size=(
                self._compute_logps_micro_batch_size
                * self.algo_config.num_generations
            ),
        )
        interval.device_end([old_per_token_logps])
    else:
      old_per_token_logps = None

    with self.rl_cluster.perf.span("advantage_computation"):
      # Compute rewards and advantages
      rewards = self._compute_gdpo_specific_rewards(
          prompts=training_input["prompts"],
          completions=completion_text,
          mode=mode,
          **{k: v for k, v in training_input.items() if k != "prompts"},
      )
      advantage_estimator = function_registry.get_advantage_estimator(
          self.algo_config.advantage_estimator
      )
      advantages = advantage_estimator(
          rewards=rewards, num_generations=self.algo_config.num_generations
      )

    # Log completion lengths.
    agg_completion_mask = completion_mask.sum(axis=-1)
    self.rl_cluster.buffer_metrics(
        {
            "completions/mean_length": (
                np.mean(agg_completion_mask),
                np.mean,
            ),
            "completions/max_length": (
                np.max(agg_completion_mask),
                np.max,
            ),
            "completions/min_length": (
                np.min(agg_completion_mask),
                np.min,
            ),
        },
        mode=mode,
    )
    for m_fn in self.metric_fns:
      user_defined_metric = m_fn(
          prompts=training_input["prompts"],
          completions=completion_text,
          advances=advantages,
          rewards=rewards,
          **{k: v for k, v in training_input.items() if k != "prompts"},
      )
      self.rl_cluster.buffer_metrics(user_defined_metric, mode=mode)


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
  # Reshape to group by generations.
  # Shape: (num_prompts, num_generations, num_reward_fns)
  grouped_rewards = rewards_per_func.reshape(
      -1, num_generations, rewards.shape[-1]
  )

  # Compute mean and std per group for each reward function,
  # Use keepdims for broadcasting.
  # Shape: (num_prompts, 1, num_reward_fns)
  mean_grouped = grouped_rewards.mean(axis=1, keepdims=True)
  std_grouped = grouped_rewards.std(axis=1, keepdims=True)

  # Normalize within each group and reshape back.
  normalized_advantages = (grouped_rewards - mean_grouped) / (
      std_grouped + 1e-4
  )
  combined_reward_advantage = normalized_advantages.reshape(
      rewards_per_func.shape
  )

  pre_bn_advantages = jnp.nansum(combined_reward_advantage, axis=1)
  bn_advantages_mean = pre_bn_advantages.mean()
  bn_advantages_std = pre_bn_advantages.std()
  advantages = (pre_bn_advantages - bn_advantages_mean) / (
      bn_advantages_std + 1e-4
  )
  return advantages
