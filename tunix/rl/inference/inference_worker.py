# Copyright 2025 Google LLC
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

"""Inference worker for RL."""

from flax import nnx
import jax
import jaxtyping
from tunix.rl import common


class InferenceWorker:
  """Inference worker hosting critic, reference and reward models."""

  def __init__(self, models: dict[str, nnx.Module]):
    for k in models.keys():
      if k not in ["critic", "reference"] and not k.startswith("reward"):
        raise ValueError(
            f"Model role {k} is not supported. Supported models are critic,"
            " reference and reward (with optional suffix, e.g., reward_model1)."
        )
    self._models = models

  def get_rewards(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
      reward_model_name: str = "reward",
  ) -> jax.Array:
    reward_model = self._models.get(reward_model_name)
    if reward_model is None:
      # Try to find any reward model if the specified one doesn't exist
      reward_models = {k: v for k, v in self._models.items() if k.startswith("reward")}
      if not reward_models:
        raise ValueError("No reward model is available.")
      if len(reward_models) == 1:
        reward_model = next(iter(reward_models.values()))
      else:
        available_models = ", ".join(reward_models.keys())
        raise ValueError(
            f"Reward model '{reward_model_name}' is not available. "
            f"Available reward models: {available_models}"
        )
    return common.compute_score(
        reward_model, prompt_tokens, completion_tokens, pad_id, eos_id
    )

  def get_all_rewards(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
  ) -> dict[str, jax.Array]:
    """Get rewards from all available reward models.
    
    Returns:
      A dictionary mapping reward model names to their computed rewards.
    """
    reward_models = {k: v for k, v in self._models.items() if k.startswith("reward")}
    if not reward_models:
      raise ValueError("No reward models are available.")
    
    rewards = {}
    for model_name, model in reward_models.items():
      rewards[model_name] = common.compute_score(
          model, prompt_tokens, completion_tokens, pad_id, eos_id
      )
    return rewards

  def get_available_reward_models(self) -> list[str]:
    """Get the names of all available reward models."""
    return [k for k in self._models.keys() if k.startswith("reward")]

  def get_ref_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
      completion_mask: jax.Array | None = None,
  ):
    ref_model = self._models.get("reference")
    if ref_model is None:
      raise ValueError("Reference model is not available.")
    return common.compute_per_token_logps(
        ref_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=pad_id,
        eos_id=eos_id,
        completion_mask=completion_mask,
    )[0]

  def get_values(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    critic_model = self._models.get("critic")
    if critic_model is None:
      raise ValueError("Critic model is not available.")
    return common.compute_score(
        critic_model,
        prompt_tokens,
        completion_tokens,
        pad_id,
        eos_id,
        completion_mask=completion_mask,
    )

  def get_model(self, role: str) -> nnx.Module:
    if role not in self._models:
      raise ValueError(f"Model role {role} is not available.")
    return self._models[role]

  def update_model(self, role: str, params: jaxtyping.PyTree):
    if role not in self._models:
      raise ValueError(f"Model role {role} is not available.")
    nnx.update(self._models[role], params)
