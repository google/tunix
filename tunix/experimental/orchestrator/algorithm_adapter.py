# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Algorithm Adapter (Layer 3) for encapsulating RL algorithm math and batch assembly.

Isolates GRPO, GSPO, and PPO advantage estimation math, sequence packing, and
metric aggregation from driver and training loop execution.
"""

import abc
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import algo_core  # pylint: disable=unused-import
from tunix.rl import algorithm_config
from tunix.rl import function_registry
from tunix.rl.agentic import agentic_rl_learner

TrainExample = agentic_rl_learner.TrainExample


class AlgorithmAdapter(abc.ABC):
  """Abstract base adapter encapsulating algorithm math and batch assembly."""

  @abc.abstractmethod
  def compute_advantages(
      self, rewards: np.ndarray | jax.Array, **kwargs: Any
  ) -> jax.Array:
    """Computes advantages from scalar rewards and optional value estimates."""
    ...

  @abc.abstractmethod
  def assemble_train_example(
      self,
      prompt_ids: jax.Array | np.ndarray,
      prompt_mask: jax.Array | np.ndarray,
      completion_ids: jax.Array | np.ndarray,
      completion_mask: jax.Array | np.ndarray,
      advantages: jax.Array | np.ndarray,
      ref_per_token_logps: jax.Array | np.ndarray | None = None,
      old_per_token_logps: jax.Array | np.ndarray | None = None,
      policy_version: int | np.ndarray = 0,
      sampler_is_weights: jax.Array | np.ndarray | None = None,
      **kwargs: Any,
  ) -> TrainExample:
    """Assembles packed training examples for model updates."""
    ...

  @abc.abstractmethod
  def get_metrics(
      self,
      rewards: np.ndarray | jax.Array,
      advantages: np.ndarray | jax.Array,
      **kwargs: Any,
  ) -> dict[str, tuple[float, Any]]:
    """Computes summary metrics for logging."""
    ...

  @abc.abstractmethod
  def get_loss_fn(self) -> Any:
    """Returns the policy loss function for model training."""
    ...


class GRPOAdapter(AlgorithmAdapter):
  """Algorithm adapter for Group Relative Policy Optimization (GRPO)."""

  def __init__(self, config: algorithm_config.AlgorithmConfig):
    self.config = config
    self._advantage_fn = function_registry.get_advantage_estimator(
        getattr(config, "advantage_estimator", "grpo")
    )

  def compute_advantages(
      self,
      rewards: np.ndarray | jax.Array,
      num_generations: int = 1,
      **kwargs: Any,
  ) -> jax.Array:
    return jnp.asarray(
        self._advantage_fn(rewards=rewards, num_generations=num_generations)
    )

  def assemble_train_example(
      self,
      prompt_ids: jax.Array | np.ndarray,
      prompt_mask: jax.Array | np.ndarray,
      completion_ids: jax.Array | np.ndarray,
      completion_mask: jax.Array | np.ndarray,
      advantages: jax.Array | np.ndarray,
      ref_per_token_logps: jax.Array | np.ndarray | None = None,
      old_per_token_logps: jax.Array | np.ndarray | None = None,
      policy_version: int | np.ndarray = 0,
      sampler_is_weights: jax.Array | np.ndarray | None = None,
      **kwargs: Any,
  ) -> TrainExample:
    if isinstance(policy_version, int):
      policy_versions = np.full(
          len(prompt_ids), policy_version, dtype=np.int32
      )
    else:
      policy_versions = np.asarray(policy_version, dtype=np.int32)

    return TrainExample(
        prompt_ids=jnp.asarray(prompt_ids),
        prompt_mask=jnp.asarray(prompt_mask),
        completion_ids=jnp.asarray(completion_ids),
        completion_mask=jnp.asarray(completion_mask),
        ref_per_token_logps=(
            jnp.asarray(ref_per_token_logps)
            if ref_per_token_logps is not None
            else None
        ),
        advantages=jnp.asarray(advantages),
        old_per_token_logps=(
            jnp.asarray(old_per_token_logps)
            if old_per_token_logps is not None
            else None
        ),
        policy_version=policy_versions,
        sampler_is_weights=(
            jnp.asarray(sampler_is_weights)
            if sampler_is_weights is not None
            else None
        ),
    )

  def get_metrics(
      self,
      rewards: np.ndarray | jax.Array,
      advantages: np.ndarray | jax.Array,
      **kwargs: Any,
  ) -> dict[str, tuple[float, Any]]:
    rewards_np = np.asarray(rewards)
    advantages_np = np.asarray(advantages)
    return {
        "rewards/mean": (float(rewards_np.mean()), np.mean),
        "rewards/std": (float(rewards_np.std()), np.mean),
        "rewards/max": (float(rewards_np.max()), np.max),
        "rewards/min": (float(rewards_np.min()), np.min),
        "rewards/advantage/mean": (float(advantages_np.mean()), np.mean),
        "rewards/advantage/std": (float(advantages_np.std()), np.mean),
        "rewards/advantage/max": (float(advantages_np.max()), np.max),
        "rewards/advantage/min": (float(advantages_np.min()), np.min),
    }

  def get_loss_fn(self) -> Any:
    policy_loss_name = getattr(self.config, "policy_loss_fn", "grpo")
    return function_registry.get_policy_loss_fn(policy_loss_name)


class PPOAdapter(AlgorithmAdapter):
  """Algorithm adapter for Proximal Policy Optimization (PPO)."""

  def __init__(self, config: algorithm_config.AlgorithmConfig):
    self.config = config
    try:
      self._advantage_fn = function_registry.get_advantage_estimator(
          getattr(config, "advantage_estimator", "gae")
      )
    except LookupError:
      self._advantage_fn = None

  def compute_advantages(
      self,
      rewards: np.ndarray | jax.Array,
      values: np.ndarray | jax.Array | None = None,
      **kwargs: Any,
  ) -> jax.Array:
    """Computes GAE / PPO advantages from rewards and value estimates."""
    if self._advantage_fn is not None:
      return jnp.asarray(
          self._advantage_fn(rewards=rewards, values=values, **kwargs)
      )
    rewards_arr = jnp.asarray(rewards)
    return (rewards_arr - rewards_arr.mean()) / (rewards_arr.std() + 1e-6)

  def assemble_train_example(
      self,
      prompt_ids: jax.Array | np.ndarray,
      prompt_mask: jax.Array | np.ndarray,
      completion_ids: jax.Array | np.ndarray,
      completion_mask: jax.Array | np.ndarray,
      advantages: jax.Array | np.ndarray,
      ref_per_token_logps: jax.Array | np.ndarray | None = None,
      old_per_token_logps: jax.Array | np.ndarray | None = None,
      policy_version: int | np.ndarray = 0,
      sampler_is_weights: jax.Array | np.ndarray | None = None,
      **kwargs: Any,
  ) -> TrainExample:
    if isinstance(policy_version, int):
      policy_versions = np.full(
          len(prompt_ids), policy_version, dtype=np.int32
      )
    else:
      policy_versions = np.asarray(policy_version, dtype=np.int32)

    return TrainExample(
        prompt_ids=jnp.asarray(prompt_ids),
        prompt_mask=jnp.asarray(prompt_mask),
        completion_ids=jnp.asarray(completion_ids),
        completion_mask=jnp.asarray(completion_mask),
        ref_per_token_logps=(
            jnp.asarray(ref_per_token_logps)
            if ref_per_token_logps is not None
            else None
        ),
        advantages=jnp.asarray(advantages),
        old_per_token_logps=(
            jnp.asarray(old_per_token_logps)
            if old_per_token_logps is not None
            else None
        ),
        policy_version=policy_versions,
        sampler_is_weights=(
            jnp.asarray(sampler_is_weights)
            if sampler_is_weights is not None
            else None
        ),
    )

  def get_metrics(
      self,
      rewards: np.ndarray | jax.Array,
      advantages: np.ndarray | jax.Array,
      **kwargs: Any,
  ) -> dict[str, tuple[float, Any]]:
    rewards_np = np.asarray(rewards)
    advantages_np = np.asarray(advantages)
    return {
        "ppo/reward_mean": (float(rewards_np.mean()), np.mean),
        "ppo/reward_std": (float(rewards_np.std()), np.mean),
        "ppo/advantage_mean": (float(advantages_np.mean()), np.mean),
        "ppo/advantage_std": (float(advantages_np.std()), np.mean),
    }

  def get_loss_fn(self) -> Any:
    policy_loss_name = getattr(self.config, "policy_loss_fn", "ppo")
    return function_registry.get_policy_loss_fn(policy_loss_name)


def get_algorithm_adapter(
    config: algorithm_config.AlgorithmConfig,
) -> AlgorithmAdapter:
  """Factory retrieving the appropriate AlgorithmAdapter for a config."""
  algo_variant = getattr(config, "algo_variant", "grpo").lower()
  if algo_variant in ("grpo", "gspo-token", "dapo"):
    return GRPOAdapter(config)
  elif algo_variant == "ppo":
    return PPOAdapter(config)
  else:
    return GRPOAdapter(config)
