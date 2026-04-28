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

"""GRPO core algorithm implementations."""

from flax import nnx
import jax
import jax.numpy as jnp

from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl.algo_core import utils


@function_registry.register_policy_loss_fn("grpo")
@function_registry.register_policy_loss_fn("agentic_grpo")
def grpo_loss_fn(
    model,
    train_example,
    algo_config,
    pad_id,
    eos_id,
    **kwargs,
):
  """GRPO loss function.

  The loss aims to maximize the expected advantage of the chosen actions while
  constraining the policy updates to stay within a certain range of the
  reference policy.

  Args:
    model: The policy model to be trained.
    train_example: A `TrainExample` instance containing the processed input
      data, including prompt IDs, completion IDs, masks, advantages, and
      per-token log probabilities from the reference and policy models.
    algo_config: The algorithm config.
    pad_id: The pad ID from tokenizer.
    eos_id: The eos ID from.

  Returns:
    A tuple containing the loss and an aux dictionary.
  """
  beta = algo_config.beta
  epsilon = algo_config.epsilon
  loss_algo = algo_config.loss_algo
  epsilon_high = (
      algo_config.epsilon_high
      if hasattr(algo_config, "epsilon_high")
      else epsilon
  )
  epsilon_c = (
      algo_config.epsilon_c
      if hasattr(algo_config, "epsilon_c")
      else 3.0
  )
  loss_aggregation_mode = algo_config.loss_agg_mode

  completion_ids, completion_mask = (
      train_example.completion_ids,
      train_example.completion_mask,
  )

  # TODO(tsbao): split can be avoided with updated peft_trainer model handling.
  graphdef, state = nnx.split(model)
  per_token_logps, logits = common.compute_per_token_logps(
      graphdef,
      state,
      prompt_tokens=train_example.prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_id,
      eos_id=eos_id,
      completion_mask=completion_mask,
      stop_gradient=False,
      return_logits=True,
      segment_ids=getattr(train_example, "segment_ids", None),
      segment_positions=getattr(train_example, "segment_positions", None),
  )
  per_token_logps = jnp.astype(per_token_logps, jnp.float32)
  # TODO(tsbao): We should handle token level advantages.
  advantages = jnp.astype(train_example.advantages, jnp.float32)

  if train_example.old_per_token_logps is None:
    old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
  else:
    old_per_token_logps = jnp.astype(
        train_example.old_per_token_logps, jnp.float32
    )

  seq_importance_ratio = per_token_logps - old_per_token_logps
  # Record KL divergence before clipping.
  ppo_kl = utils.masked_mean(-seq_importance_ratio, completion_mask)

  seq_importance_ratio = jnp.clip(seq_importance_ratio, max=20.0, min=-20.0)

  # TODO(sizhi): Refactor this to a separate function.
  if loss_algo == "gspo-token":
    seq_importance_ratio = (seq_importance_ratio * completion_mask).sum(
        axis=-1
    ) / jnp.clip(completion_mask.sum(-1), min=1)
    seq_importance_ratio = (
        per_token_logps
        - jax.lax.stop_gradient(per_token_logps)
        + jnp.expand_dims(jax.lax.stop_gradient(seq_importance_ratio), axis=-1)
    )
    seq_importance_ratio = jnp.clip(seq_importance_ratio, max=10.0)

  is_ratio = jnp.exp(seq_importance_ratio)

  # Advantages must be broadcast against seq_length.
  # When sequence packing is used, advantages are already 2D [B, seq_length].
  # When unpacked, they are 1D [B].
  adv = advantages if advantages.ndim == 2 else jnp.expand_dims(advantages, 1)

  pg_loss_1 = -adv * is_ratio
  pg_loss_2 = -adv * jnp.clip(is_ratio, 1 - epsilon, 1 + epsilon_high)

  per_token_loss = jnp.maximum(pg_loss_1, pg_loss_2).astype(jnp.float32)

  clipped_fraction = utils.masked_mean(
      jnp.greater(pg_loss_2, pg_loss_1), completion_mask
  )

  # dual-clip ppo loss
  pg_loss_3 = -epsilon_c * adv

  # pg_clipfrac_lower measures how often dual-clip ppo kicks in.
  # It kicks in when the standard clipped loss is larger than pg_loss_3
  # for instances with negative advantages.
  unreduced_pg_clipfrac_lower = (
      (per_token_loss > pg_loss_3) & (adv < 0.0)
  ).astype(jnp.float32)
  pg_clipfrac_lower = common.aggregate_loss(
      unreduced_pg_clipfrac_lower, completion_mask, loss_aggregation_mode
  )

  pg_loss_clipped_dual = jnp.minimum(pg_loss_3, per_token_loss)
  per_token_loss = jnp.where(adv < 0.0, pg_loss_clipped_dual, per_token_loss)
  loss = common.aggregate_loss(
      per_token_loss, completion_mask, loss_aggregation_mode
  )
  aux = {
      "kl": 0.0,
      "kl_loss": 0.0,
      "pg_loss": loss,
      "pg_clipfrac": clipped_fraction,
      "ppo_kl": ppo_kl,
      "pg_clipfrac_lower": pg_clipfrac_lower,
  }
  # We do not alwayscompute KL divergence (e.g. when beta is 0.0 unless
  # force_compute_kl is True).
  if train_example.ref_per_token_logps is not None:
    # Use algo_config.kl_loss_mode if it exists, otherwise "kl" for base algo config
    kl_loss_mode = getattr(algo_config, "kl_loss_mode", "kl")
    kl = common.compute_kl_divergence(
        per_token_logps,
        train_example.ref_per_token_logps,
        kl_loss_mode,
    )
    # Log mean KL.
    aux["kl"] = jnp.astype(
        (kl * completion_mask).sum() / jnp.clip(completion_mask.sum(), min=1),
        jnp.float32,
    )
    kl_loss = common.aggregate_loss(
        kl, completion_mask, loss_aggregation_mode
    )
    aux["kl_loss"] = kl_loss
    if beta is not None and beta != 0.0:
      loss = loss + beta * kl_loss

  token_entropy = utils.compute_entropy_from_logits(logits)
  entropy_loss = common.aggregate_loss(
      token_entropy, completion_mask, loss_aggregation_mode
  )
  aux["entropy"] = entropy_loss

  return loss, aux


@function_registry.register_advantage_estimator("grpo")
@function_registry.register_advantage_estimator("agentic_grpo")
def compute_advantages(rewards: jax.Array, num_generations: int) -> jax.Array:
  """Compute group relative advantages.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.

  Returns:
    Group relative advantages.
  """
  rewards = jnp.astype(rewards, jnp.float32)
  mean_grouped_rewards = rewards.reshape(-1, num_generations).mean(axis=-1)
  std_grouped_rewards = rewards.reshape(-1, num_generations).std(
      axis=-1, ddof=1
  )

  mean_grouped_rewards = mean_grouped_rewards.repeat(num_generations)
  std_grouped_rewards = std_grouped_rewards.repeat(num_generations)
  return (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-6)


@function_registry.register_advantage_estimator("agentic_rloo")
def compute_rloo_advantages(
    rewards: jax.Array, num_generations: int
) -> jax.Array:
  """Compute RLOO (REINFORCE Leave-One-Out) advantages.

  RLOO computes a baseline for each completion by averaging the rewards of all
  other completions to the same prompt.

  Args:
    rewards: reward functions output.
    num_generations: Number of generations.

  Returns:
    RLOO advantages.
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
