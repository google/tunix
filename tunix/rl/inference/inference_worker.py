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
import jax.numpy as jnp

class InferenceWorker:
  """Inference worker hosting critic, reference and reward models."""

  def __init__(self, models: dict[str, nnx.Module]):
    for k in models.keys():
      if k not in ["critic", "reference", "reward"]:
        raise ValueError(
            f"Model role {k} is not supported. Supported models are critic,"
            " reference and reward."
        )
    self._models = models
    # TODO(tsbao): support multiple reward models.

  def get_rewards(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
  ) -> jax.Array:
    reward_model = self._models.get("reward")
    if reward_model is None:
      raise ValueError("Reward model is not available.")
    return common.compute_score(
        reward_model, prompt_tokens, completion_tokens, pad_id, eos_id
    )

  def get_ref_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
  ):
    print('begin ref logps compute')
    ref_model = self._models.get("reference")
    if ref_model is None:
      raise ValueError("Reference model is not available.")
    return common.compute_per_token_logps(
        ref_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=pad_id,
        eos_id=eos_id,
    )

 
  

  # # This is the function from your inference_worker that needs to be changed.
  # def get_ref_per_token_logps(
  #     self,
  #     prompt_tokens: jax.Array,
  #     completion_tokens: jax.Array,
  #     pad_id: int,
  #     eos_id: int,
  #     # Add a parameter for micro-batch size with a safe default
  #     micro_batch_size: int = 4,
  # ):
  #   """
  #   Computes reference log-probabilities using micro-batching to avoid OOM errors.
  #   """
  #   print('Begin ref logps compute with micro-batching.')
  #   ref_model = self._models.get("reference")
  #   if ref_model is None:
  #       raise ValueError("Reference model is not available.")

  #   # Get the total number of samples to process.
  #   total_batch_size = prompt_tokens.shape[0]
  #   if total_batch_size == 0:
  #       return jnp.array([])

  #   all_logps = []
  #   print(
  #       f"Processing {total_batch_size} samples in micro-batches of"
  #       f" {micro_batch_size}..."
  #   )

  #   # Loop through the entire batch in smaller chunks (micro-batches).
  #   for i in range(0, total_batch_size, micro_batch_size):
  #       # Create slices for the current micro-batch.
  #       start_index = i
  #       end_index = i + micro_batch_size
  #       print(f"  -> Processing batch slice [{start_index}:{end_index}]")

  #       batch_prompt_tokens = prompt_tokens[start_index:end_index]
  #       batch_completion_tokens = completion_tokens[start_index:end_index]

  #       # Run the expensive computation only on the small chunk.
  #       batch_logps = common.compute_per_token_logps(
  #           ref_model,
  #           prompt_tokens=batch_prompt_tokens,
  #           completion_tokens=batch_completion_tokens,
  #           pad_id=pad_id,
  #           eos_id=eos_id,
  #       )
  #       all_logps.append(batch_logps)

  #   # Concatenate the results from all micro-batches into a single tensor.
  #   return jnp.concatenate(all_logps, axis=0)

  def get_values(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
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
    )

  def get_model(self, role: str) -> nnx.Module:
    if role not in self._models:
      raise ValueError(f"Model role {role} is not available.")
    return self._models[role]

  def update_model(self, role: str, params: jaxtyping.PyTree):
    if role not in self._models:
      raise ValueError(f"Model role {role} is not available.")
    nnx.update(self._models[role], params)
