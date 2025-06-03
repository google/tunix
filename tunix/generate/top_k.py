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

"""Top-k sampling for LLM generation."""

import jax
import jax.numpy as jnp

def sample_top_k(
    logits: jnp.ndarray, # Shape: (batch_size, vocab_size)
    key: jax.Array,
    top_k: int,
    temperature: float = 1.0,
) -> jnp.ndarray: # Shape: (batch_size,)
  """Samples a token using top-k sampling.

  Args:
    logits: The logits from the model.
    key: A JAX PRNG key.
    top_k: The number of top candidates to consider.
    temperature: The temperature for sampling. Values closer to 0 make the
      sampling more deterministic, while values closer to 1 make it more random.

  Returns:
    The sampled token IDs.
  """
  if top_k <= 0:
    raise ValueError("top_k must be greater than 0.")

  # Apply temperature
  logits = logits / temperature

  # Get top-k logits and their indices
  # Ensure top_k is not larger than the vocabulary size
  vocab_size = logits.shape[-1]
  actual_top_k = min(top_k, vocab_size)

  top_k_logits, top_k_indices = jax.lax.top_k(logits, k=actual_top_k)

  # Convert to probabilities
  top_k_probs = jax.nn.softmax(top_k_logits, axis=-1)

  # Sample from the top-k probabilities
  # Using jnp.log on probabilities for categorical sampling as it expects log-probabilities
  next_token_indices = jax.random.categorical(key, logits=jnp.log(top_k_probs))

  # Gather the actual token IDs corresponding to the sampled indices
  next_token = jnp.take_along_axis(
      top_k_indices, next_token_indices[..., None], axis=-1
  )
  return jnp.squeeze(next_token, axis=-1)
