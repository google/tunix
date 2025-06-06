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

"""Contrastive search for LLM generation (https://arxiv.org/pdf/2202.06417)."""

import functools
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from tunix.generate import utils


def expand_dims(
    cache: dict[str, dict[str, jax.Array]],
    step_positions: jax.Array,
    attention_mask: jax.Array,
    top_k: int,
) -> dict[str, Any]:
  """Expand the batch dimension of the inputs by top_k.

  Expected input shapes: [B, ...], expected output shapes: [B*top_k, ...]

  Args:
    cache: The current cache after computing the logits.
    step_positions: The current step positions.
    attention_mask: The current attention mask.
    top_k: The number of candidates to expand.

  Returns:
    A dictionary of expanded inputs.
  """
  caches = jax.tree.map(lambda x: jnp.repeat(x, top_k, axis=0), cache)
  return {
      'cache': caches,
      'step_positions': jnp.repeat(step_positions, top_k, axis=0),
      'attention_mask': jnp.repeat(attention_mask, top_k, axis=0),
  }


def contrastive_search_step(
    transformer: nnx.Module,
    logits: jax.Array,
    token_buffer: jax.Array,
    positions: jax.Array,
    decoding_step: int,
    cache: dict[str, dict[str, jax.Array]],
    hidden_states_buffer: jax.Array,
    top_k: int,
    alpha: float,
    pad_id: int,
    cache_size: int,
) -> tuple[jax.Array, jax.Array]:
  """Contrastive search step.

  The contrastive search step is used to generate the next token candidate
  based on the current token logits and the previous token hidden states.
  The algorithm works as follows:
  1. Calculate the top_k token candidates with the highest logits.
  2. For each candidate, run inference with the candidate as the current token
     to get the hidden states of the candidate.
  3. Rank the candidates based on the cosine similarity between the previous
     token hidden states and the candidate hidden states. The lower the
     cosine similarity, the better.
  4. Returns the candidate with the highest score.

  Args:
    transformer: The transformer model.
    logits: The current token logits. [B, 1, V]
    token_buffer: The previuos token buffer w/o current token. [B, L]
    positions: The positions for generating the current token, so it's a
      previous position + 1. [B, L]
    decoding_step: The current decoding step.
    cache: The current cache after computing the logits.
    hidden_states_buffer: The previous buffer to hold previous token's hidden
      states. [B, L, D]
    top_k: The number of candidates to consider.
    alpha: The penalty alpha for the search algorithm.
    pad_id: The pad id for the tokenizer.
    cache_size: The cache size for the transformer.

  Returns:
    next_token_candidate: [B]
    next_hidden: [B, D]
  """
  batch_size = logits.shape[0]
  logits = logits.squeeze(axis=1)  # B, 1, V -> B, V
  logits = jax.nn.softmax(logits, axis=-1)
  topk_probs, topk_indices = jax.lax.top_k(logits, k=top_k)  # [B, top_k]
  topk_tokens = topk_indices.reshape((batch_size * top_k), 1)  # [B * top_k, 1]
  # inference with all top_k tokens to get the hidden states. The way
  # to do is to collapse the top_k dimension into the batch dimension.
  # The input cache, step_position, and attention_mask need to be updated
  # accordingly.
  step_positions = jnp.expand_dims(positions[:, decoding_step + 1], -1)

  input_mask = token_buffer == pad_id
  current_token_mask = topk_indices[:, 0] == pad_id
  input_mask = input_mask.at[:, decoding_step + 1].set(current_token_mask)

  attention_mask = utils.compute_attention_masks(
      decoding_step + 1, cache_size, input_mask
  )
  updated_components = expand_dims(cache, step_positions, attention_mask, top_k)
  updated_cache = updated_components['cache']
  updated_step_positions = updated_components['step_positions']
  updated_attention_mask = updated_components['attention_mask']

  transformer(
      topk_tokens,
      updated_step_positions,
      updated_cache,
      updated_attention_mask,
      output_hidden_states=True,
  )
  assert hasattr(
      transformer, 'all_hidden_states'
  ), 'Missing all_hidden_states, set output_hidden_states to True.'
  assert len(transformer.all_hidden_states.value) == 1
  next_hidden = transformer.all_hidden_states.value[0]
  next_hidden = next_hidden.reshape(
      (batch_size, top_k, 1, next_hidden.shape[-1])
  )  # [B, top_k, 1, D]
  assert hidden_states_buffer is not None  # make pyright happy
  context_hidden = hidden_states_buffer.repeat(top_k, axis=0).reshape((
      batch_size,
      top_k,
      hidden_states_buffer.shape[1],
      hidden_states_buffer.shape[-1],
  ))  # [B, top_k, L, D]

  next_token_candidate, next_hidden = jax.vmap(
      ranking, in_axes=(0, 0, 0, 0, None, None)
  )(
      context_hidden,
      next_hidden,
      topk_indices,
      topk_probs,
      decoding_step,
      alpha,
  )
  return next_token_candidate, next_hidden


@functools.partial(jax.jit, static_argnames=['alpha'])
def ranking(
    context_hidden: jax.Array,
    next_hidden: jax.Array,
    next_top_k_ids: jax.Array,
    next_top_k_probs: jax.Array,
    context_len: int,
    alpha: float,
) -> tuple[jax.Array, jax.Array]:
  """Ranking the next top_k candidates and returns the highest ranked one.

  The ranking is based on the cosine similarity between the context hidden
  states (already generated sequences) and the next hidden states (new token).
  The lower the cosine similarity, the more likely the next token is the best
  candidate. The goal is to reduce duplication in the generated text.

  Args:
    context_hidden: [top_k, L, embed_dim]
    next_hidden: [top_k, 1, embed_dim]
    next_top_k_ids: [top_k, 1]
    next_top_k_probs: [top_k, 1]
    context_len: the length of the context (already generated sequences).
    alpha: penalty alpha for the search algorithm, the larger the alpha, the
      more likely the model will consider the similarity effect for ranking. In
      other words, if the alpha is 0, the model will only consider the logits of
      next token and totally ignore the similarity effect.

  Returns:
    The highest ranked next token id and the corresponding hidden state.
  """
  epsilon = 1e-8
  top_k, full_len, embed_dim = context_hidden.shape
  context_hidden = context_hidden.reshape(
      (top_k, full_len, embed_dim)
  )  # [top_k, L, embed_dim]
  next_hidden = next_hidden.reshape((top_k, 1, embed_dim))
  # [top_k, 1, embed_dim]
  norm_context_hidden = context_hidden / (
      jnp.linalg.norm(context_hidden, axis=-1, keepdims=True) + epsilon
  )
  norm_next_hidden = next_hidden / (
      jnp.linalg.norm(next_hidden, axis=-1, keepdims=True) + epsilon
  )
  cosine_matrix = jnp.matmul(
      norm_context_hidden, norm_next_hidden.transpose(0, 2, 1)
  ).squeeze(
      -1
  )  # [top_k, L]
  indices = jnp.arange(cosine_matrix.shape[1])
  mask = indices < context_len
  cosine_matrix = jnp.where(mask[None, :], cosine_matrix, -jnp.inf)

  scores = jnp.max(cosine_matrix, axis=-1)  # [top_k]
  next_top_k_probs = next_top_k_probs.reshape((-1))  # [top_k]
  scores = (1.0 - alpha) * next_top_k_probs - alpha * scores  # [top_k]
  selected_indices = jnp.argmax(scores)
  return next_top_k_ids[selected_indices], next_hidden[
      selected_indices
  ].squeeze(0)
