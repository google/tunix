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

"""PPO learner."""

from __future__ import annotations

import dataclasses
from typing import Iterable, List, Sequence

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import rl_learner
from tunix.rl.ppo import ppo_helpers

TrainingInputT = rl_learner.TrainingInputT
RewardFn = rl_learner.RewardFn
MetricFn = rl_learner.MetricFn


@flax.struct.dataclass(frozen=True)
class TrainExample(common.TrainExample):
  returns: jax.Array
  old_values: jax.Array


@dataclasses.dataclass(slots=True, kw_only=True)
class PPOConfig:
  """Configuration for PPO learner.

  Parameters:
    num_ppo_epochs: The number of optimization epochs per batch of rollouts.
    mini_batch_size: The batch size on which the actual model updates happen.
      The rollout phase (`generate_and_compute_advantages`) happen on a larger
      batch, which is then split into "mini-batches".
    gamma: The discount factor for future rewards in GAE.
    gae_lambda: The lambda parameter for Generalized Advantage Estimation (GAE).
    beta: The coefficient for the KL divergence penalty.
    epsilon: Epsilon value for clipping the ratio for the policy objective.
    epsilon_low: Lower bound for clipping the ratio for the policy objective.
      Set to `epsilon` if not provided.
    epsilon_high: Upper bound for clipping the ratio for the policy objective.
      Set to `epsilon` if not provided.
    epsilon_c: Lower bound for clipping for dual-clip PPO. If not provided, we
      don't do dual-clip PPO.
      Reference: https://arxiv.org/abs/1912.09729.
    entropy_coef: Entropy coefficient for the policy loss. Set to `None` or
      `0.0` to disable entropy regularization.
    clip_range_value: The range for clipping the value function loss.
    kl_method: The method for computing KL divergence. Must be one of
      `["low_var_kl", "kl", "mse_kl"]`.
  """

  num_ppo_epochs: int = 1

  # PPO loss and advantage computation configs.
  gamma: float = 1.0
  gae_lambda: float = 0.95
  beta: float = 0.04
  epsilon: float = 0.2
  epsilon_low: float | None = None
  epsilon_high: float | None = None
  epsilon_c: float | None = None
  entropy_coef: float | None = None
  clip_range_value: float = 0.2
  kl_method: str = "low_var_kl"

  def __post_init__(self):
    self.epsilon_low = self.epsilon_low if self.epsilon_low else self.epsilon
    self.epsilon_high = self.epsilon_high if self.epsilon_high else self.epsilon
    self.epsilon = self.epsilon

    if self.epsilon_c is not None and self.epsilon_c <= 1.0:
      raise ValueError(
          f"`epsilon_c` must be greater than 1. Received: {self.epsilon_c}."
      )

    if self.kl_method not in ["kl", "mse_kl", "low_var_kl"]:
      raise ValueError(
          f"Invalid KL method: {self.kl_method}. Must be one of"
          " ['low_var_kl', 'kl', 'mse_kl']."
      )


class PPOLearner(rl_learner.RLLearner):
  """PPO (Proximal Policy Optimization) learner.

  PPO is a reinforcement learning algorithm that fine-tunes models using an
  actor-critic architecture. It optimizes a clipped surrogate objective function
  to ensure stable policy updates, preventing large, destructive changes. The
  actor (policy model) learns what actions to take, while the critic (value
  model) estimates the value of states to help calculate advantages. This
  approach balances exploration and exploitation, making it a robust choice for
  a wide range of RL tasks.

  References:
  - https://arxiv.org/abs/1707.06347
  """

  def __init__(
      self,
      rl_cluster: rl_cluster_lib.RLCluster,
      ppo_config: PPOConfig,
      reward_fns: RewardFn | List[RewardFn] | None = None,
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    """Initializes the `PPOLearner`.

    Args:
      rl_cluster: RL cluster containing actor, reference and reward models.
      ppo_config: An instance of `PPOConfig` containing all training-specific
        configuration options.
      reward_fns: A single callable or a list of callables that compute a scalar
        reward for given prompts and completions. Each function should accept
        `prompts`, `completions` and optional keyword arguments, and return a
        list of float rewards.
      metric_fns: A sequence of callables that compute metrics for the
        completions. Each callable should accept `prompts`, `completions`,
        `rewards`, `advantages` and optional keyword arguments, and return a
        dictionary of metric names to tuples of (metric_value, aggregation_fn):
        >>> def metric_fn(prompts, completions, rewards, advantages, **kargs):
        ...    return { ...        "prompt_min_len": (min(len(p) for p in
        prompts), np.min), ...        ... ...    }
      data_shuffle_seed: The seed for shuffling the data.
    """
    self.ppo_config = ppo_config
    super().__init__(
        rl_cluster=rl_cluster,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
    )

    # ===== RlCluster should have `reward` and `critic` models =====
    if bool(reward_fns) == bool(
        self.rl_cluster.inference_worker._models.get("reward", None)
    ):
      raise ValueError(
          "PPO requires one of `reward_fns` or `rl_cluster.reward` to be set. "
          f"Received: reward_fn={reward_fns}, "
          "rl_cluster.reward="
          f"{self.rl_cluster.inference_worker._models['reward']}"
      )
    if not self.rl_cluster.inference_worker._models["critic"]:
      raise ValueError(
          "PPO requires a critic model. Please pass the correct `critic` to "
          "`RlCluster`."
      )
    self._use_reward_model = bool(
        self.rl_cluster.inference_worker._models.get("reward", None)
    )

    # ===== Configure the actor (policy) trainer =====
    self.rl_cluster.actor_trainer.with_loss_fn(ppo_policy_loss_fn, has_aux=True)
    self.rl_cluster.actor_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "epsilon_low": self.ppo_config.epsilon_low,
            "epsilon_high": self.ppo_config.epsilon_high,
            "epsilon_c": self.ppo_config.epsilon_c,
            "entropy_coef": self.ppo_config.entropy_coef,
            "pad_id": self.rl_cluster.rollout.pad_id(),
            "eos_id": self.rl_cluster.rollout.eos_id(),
        }
    )

    # ===== Configure the critic (value) trainer =====
    self.rl_cluster.critic_trainer.with_loss_fn(ppo_value_loss_fn, has_aux=True)
    self.rl_cluster.critic_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "clip_range_value": self.ppo_config.clip_range_value,
            "pad_id": self.rl_cluster.rollout.pad_id(),
            "eos_id": self.rl_cluster.rollout.eos_id(),
        }
    )

    # ===== Configure the metrics logger =====
    # We just log the metrics returned in `aux`. All other metrics are logged
    # by `RLCluster` itself.
    actor_rl_metrics_to_log = {"pg_clipfrac": np.mean}
    if self.ppo_config.epsilon_c is not None:
      actor_rl_metrics_to_log["pg_clipfrac_lower"] = np.mean
    if (
        self.ppo_config.entropy_coef is not None
        and self.ppo_config.entropy_coef > 0.0
    ):
      actor_rl_metrics_to_log["loss/entropy"] = np.mean
    self.rl_cluster.actor_trainer.with_rl_metrics_to_log(
        actor_rl_metrics_to_log
    )

    self.rl_cluster.critic_trainer.with_rl_metrics_to_log({
        "vpred_mean": np.mean,
        "vf_clipfrac": np.mean,
    })

  def _generate_and_compute_advantage(
      self,
      training_input: TrainingInputT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
  ) -> TrainExample:
    """Generates completions and computes advantages for PPO training.

    Args:
      training_input: A dictionary containing the training input data, including
        the key 'prompts'.
      mode: The mode to use for logging metrics.

    Returns:
      A `TrainExample` instance containing the processed input data for PPO.
    """
    pad_value = self.rl_cluster.rollout.pad_id()
    eos_value = self.rl_cluster.rollout.eos_id()

    # TODO(abheesht): Other RL libraries may allow specifying different micro batch sizes for
    # computing log probs, values, rewards, etc. We can do that here.

    # ===== Generation ======
    # Generate. We use `model`, i.e., the policy model for generating the
    # "experiences".
    completion_output = self.rl_cluster.generate(
        prompts=training_input["prompts"],
    )
    completion_ids = completion_output.tokens
    prompt_ids = completion_output.left_padded_prompt_tokens

    batch_size = completion_ids.shape[0]
    logits_to_keep = completion_ids.shape[1]
    prompt_mask = (prompt_ids != pad_value).astype("int32")
    completion_mask = common.make_completion_mask(
        completion_ids, eos_tok=eos_value
    )
    eos_idx = jnp.max(
        common.build_positions_from_mask(completion_mask),
        axis=-1,
    )

    # ===== Compute log probs ======
    # Compute log probs from the reference model. Shape = `[B, T]`.
    if self.ppo_config.beta != 0.0:
      ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
      )
    else:
      ref_per_token_logps = None

    # Get log probs from the policy before model weights are updated. We use
    # the policy model here. Shape = `[B, T]`.
    # TODO(abheesht): Do we do this only when `self.num_ppo_epochs > 1`? Don't
    # see this condition in other RL libraries.
    old_per_token_logps = self.rl_cluster.get_old_per_token_logps(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
    )

    # ===== Value computation ======
    # Get values from the value model before model weights are updated.
    values = self.rl_cluster.get_values(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
        pad_id=pad_value,
        eos_id=eos_value,
    )
    # `values` start from the last *prompt* token. Shape: `[B, T]`.
    values = values[:, -logits_to_keep - 1 : -1]
    values = values * completion_mask

    # ===== Reward computation ======
    # Get rewards from the reward model. Eventual shape: `[B, T]`.
    if self._use_reward_model:
      scores = self.rl_cluster.get_rewards(
          prompt_tokens=prompt_ids,
          completion_tokens=completion_ids,
          pad_id=pad_value,
          eos_id=eos_value,
      )[:, -logits_to_keep:]
      # We use the score corresponding to the last non-padding token.
      last_token_scores = scores[jnp.arange(batch_size), eos_idx]
    else:
      last_token_scores = self._compute_rewards(
          prompts=training_input["prompts"],
          completions=completion_output.text,
          mode=mode,
          **{k: v for k, v in training_input.items() if k != "prompts"},
      )

    # Reward computation is in accordance with other RL libraries
    # batch reward manager (token-level rewards).
    # 1. Set all rewards (i.e., for every token) to 0s.
    # 2. A positive reward is given only at the final timestep, so we add that
    # to the tensor of zeros.
    # 3. Subtract KL divergence from the reward tensor.
    rewards = jnp.zeros_like(completion_ids)
    rewards = rewards.at[jnp.arange(batch_size), eos_idx].add(last_token_scores)
    if self.ppo_config.beta != 0.0:
      # TODO(abheesht): Add a toggle - KL can either be added directly to
      # rewards or computed in the loss function.
      kl = common.compute_kl_divergence(
          old_per_token_logps,
          ref_per_token_logps,
          method=self.ppo_config.kl_method,
      )
      kl = kl * completion_mask
      rewards = rewards - self.ppo_config.beta * kl

    # ===== Compute advantages using Generalised Advantage Estimation ======
    advantages, returns = ppo_helpers.compute_gae_advantages(
        rewards=rewards,
        values=values,
        completion_mask=completion_mask,
        gamma=self.ppo_config.gamma,
        gae_lambda=self.ppo_config.gae_lambda,
    )

    # ===== Metric logging ======
    # Log raw scores from the reward model fn
    self.rl_cluster.buffer_metrics(
        {
            "score/mean": (np.mean(last_token_scores), np.mean),
            "score/max": (np.max(last_token_scores), np.max),
            "score/min": (np.min(last_token_scores), np.min),
        },
        mode=mode,
    )

    # Log final rewards (scores + KL penalty)
    sequence_rewards = rewards.sum(-1)
    self.rl_cluster.buffer_metrics(
        {
            "reward/mean": (np.mean(sequence_rewards), np.mean),
            "reward/max": (np.max(sequence_rewards), np.max),
            "reward/min": (np.min(sequence_rewards), np.min),
        },
        mode=mode,
    )
    if self.ppo_config.beta != 0.0:
      # Average of the per-sequence mean KL
      per_sequence_mean_kl = ppo_helpers.masked_mean(
          kl, completion_mask, axis=-1  # pylint: disable=undefined-variable
      )
      self.rl_cluster.buffer_metrics(
          {
              "reward_kl_penalty": (
                  per_sequence_mean_kl.mean(),
                  np.mean,
              ),
          },
          mode=mode,
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

    # Log advantages.
    valid_advantages = np.ma.masked_array(
        advantages, mask=np.logical_not(completion_mask)
    )
    self.rl_cluster.buffer_metrics(
        {
            "advantages/mean": (valid_advantages.mean(), np.mean),
            "advantages/max": (valid_advantages.max(), np.max),
            "advantages/min": (valid_advantages.min(), np.min),
        },
        mode=mode,
    )

    # Log returns.
    valid_returns = np.ma.masked_array(
        returns, mask=np.logical_not(completion_mask)
    )
    self.rl_cluster.buffer_metrics(
        {
            "returns/mean": (valid_returns.mean(), np.mean),
            "returns/max": (valid_returns.max(), np.max),
            "returns/min": (valid_returns.min(), np.min),
        },
        mode=mode,
    )

    # Log values.
    valid_values = np.ma.masked_array(
        values, mask=np.logical_not(completion_mask)
    )
    self.rl_cluster.buffer_metrics(
        {
            "old_values/mean": (valid_values.mean(), np.mean),
            "old_values/max": (valid_values.max(), np.max),
            "old_values/min": (valid_values.min(), np.min),
        },
        mode=mode,
    )

    return TrainExample(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        ref_per_token_logps=ref_per_token_logps,
        advantages=advantages,
        returns=returns,
        old_per_token_logps=old_per_token_logps,
        old_values=values,
    )

  def _compute_trajectory_ids(
      self, example: TrainingInputT, steps: int
  ) -> List[str]:
    """Computes the trajectory ID for each prompt in the batch.

    Trajectory id is same as the offset of the example in the data source.

    Args:
      example: The training input data.
      steps: The number of steps taken so far.

    Returns:
      A list of trajectory IDs, one for each prompt in the batch.
    """
    batch_size = len(example["prompts"]) // self._num_generations()
    row_offset = steps * batch_size
    row_offsets = np.arange(row_offset, row_offset + batch_size)
    return row_offsets.astype(str).tolist()

  def _num_iterations(self) -> int:
    return self.ppo_config.num_ppo_epochs

  def _num_generations(self) -> int:
    return 1

  def train(  # pylint: disable=useless-parent-delegation
      self,
      train_ds: Iterable[TrainingInputT],
      eval_ds: Iterable[TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
    """PPO training loop."""
    super().train(train_ds, eval_ds, skip_jit)


def ppo_value_loss_fn(
    model: nnx.Module,
    train_example: TrainExample,
    clip_range_value: float | None,
    pad_id: int,
    eos_id: int,
):
  """Computes the value loss for PPO."""

  prompt_ids, completion_ids, completion_mask = (
      train_example.prompt_ids,
      train_example.completion_ids,
      train_example.completion_mask,
  )
  logits_to_keep = completion_ids.shape[1]

  # ====== Loss ======
  values = train_example.old_values
  returns = train_example.returns
  # Get new values.
  vpreds = common.compute_score(
      model,
      prompt_ids,
      completion_ids,
      pad_id,
      eos_id,
      stop_gradient=False,
  )
  vpreds = vpreds[:, -logits_to_keep - 1 : -1]
  vpred_clipped = jnp.clip(
      vpreds, values - clip_range_value, values + clip_range_value
  )
  vf_losses1 = jnp.square(vpreds - returns)
  vf_losses2 = jnp.square(vpred_clipped - returns)

  clipped_vf_losses = jnp.maximum(vf_losses1, vf_losses2)
  # "token mean" style of normalisation.
  vf_loss = ppo_helpers.masked_mean(clipped_vf_losses, completion_mask)
  vf_loss = 0.5 * vf_loss

  aux = {
      "vpred_mean": ppo_helpers.masked_mean(vpreds, completion_mask),
      "vf_clipfrac": ppo_helpers.masked_mean(
          (vf_losses2 > vf_losses1).astype(jnp.float32), completion_mask
      ),
  }
  return vf_loss, aux


def ppo_policy_loss_fn(
    model: nnx.Module,
    train_example: TrainExample,
    epsilon_low: float,
    epsilon_high: float,
    epsilon_c: float | None,
    entropy_coef: float | None,
    pad_id: int,
    eos_id: int,
):
  """Computes the policy loss for PPO."""

  prompt_ids, completion_ids, completion_mask = (
      train_example.prompt_ids,
      train_example.completion_ids,
      train_example.completion_mask,
  )
  use_dual_clip_ppo = epsilon_c is not None

  # Get log probs.
  per_token_logps, logits = common.compute_per_token_logps(
      model,
      prompt_tokens=prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_id,
      eos_id=eos_id,
      stop_gradient=False,
      return_logits=True,
  )

  advantages = train_example.advantages

  # Compute ratio.
  old_per_token_logps = train_example.old_per_token_logps
  ratio = jnp.exp(per_token_logps - old_per_token_logps)
  ratio_clipped = jnp.clip(ratio, 1 - epsilon_low, 1 + epsilon_high)

  # Vanilla PPO loss
  pg_losses_1 = -ratio * advantages
  pg_losses_2 = -ratio_clipped * advantages
  clip_pg_losses_1 = jnp.maximum(pg_losses_1, pg_losses_2)

  # Dual-clip PPO to avoid negative-advantage policy updates
  pg_losses = clip_pg_losses_1
  if use_dual_clip_ppo:
    pg_losses_3 = -epsilon_c * advantages
    clip_pg_losses_2 = jnp.minimum(pg_losses_3, clip_pg_losses_1)

    pg_losses = jnp.where(advantages < 0.0, clip_pg_losses_2, clip_pg_losses_1)

    # For logging.
    unreduced_pg_clipfrac_lower = (
        (clip_pg_losses_1 > pg_losses_3) & (advantages < 0.0)
    ).astype(jnp.float32)
    pg_clipfrac_lower = ppo_helpers.masked_mean(
        unreduced_pg_clipfrac_lower, completion_mask
    )

  # Logging
  aux = {
      "pg_clipfrac": ppo_helpers.masked_mean(
          (pg_losses_2 > pg_losses_1).astype(jnp.float32), completion_mask
      ),
  }
  if use_dual_clip_ppo:
    aux["pg_clipfrac_lower"] = pg_clipfrac_lower  # pylint: disable=undefined-variable

  # "token mean" style of normalisation
  policy_loss = ppo_helpers.masked_mean(pg_losses, completion_mask)

  # Compute entropy loss.
  if entropy_coef is not None and entropy_coef > 0.0:
    token_entropy = ppo_helpers.compute_entropy_from_logits(logits)
    # "token mean" style of normalisation.
    entropy_loss = ppo_helpers.masked_mean(token_entropy, completion_mask)
    policy_loss -= entropy_coef * entropy_loss

    # Logging
    aux["loss/entropy"] = entropy_loss

  return policy_loss, aux


PpoConfig = PPOConfig
PpoLearner = PPOLearner
