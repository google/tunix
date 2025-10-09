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


"""Beam search for LLM generation related utils."""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import jaxtyping


@flax.struct.dataclass
class _BeamSearchSamplingState:
  """The state used by beam search.

  This is intented to be included by the main sampling state.
  """

  # Accumulated scores (log sum probabilities) for each beam. [B, beam_size]
  scores: jnp.ndarray
  # flag to indicate whether the state is initialized or not
  initialized: bool = flax.struct.field(pytree_node=False)


def init_batched_beam_state(
    logits: jax.Array,
    input_token_buffer: jax.Array,
    initial_cache: dict[str, dict[str, jaxtyping.Array]],
    done: jax.Array,
    positions: jax.Array,
    logits_buffer: jax.Array | None,
    beam_size: int,
) -> tuple[_BeamSearchSamplingState, dict[str, Any]]:
  """Intializes the beam search sampling state.

  In order to support beam search, we need to expand the input other states to
  support beam search by repeating the batch size by beam size.

  Args:
    logits: The logits of the current step. [B, vocab_size]
    input_token_buffer: The input token buffer with max length and padding if
      needed. [B, L]
    initial_cache: The initial cache for original batch size.
    done: The done array for original batch size. [B, 1]
    positions: The original positions of the input tokens. [B, L]
    logits_buffer: The logits buffer for original batch size. [B, L, V]
    beam_size: The number of beams.

  Returns:
    The beam search sampling state.
  """
  batch_size = input_token_buffer.shape[0]

  caches = jax.tree.map(
      lambda x: jnp.repeat(x, beam_size, axis=0), initial_cache
  )

  return _BeamSearchSamplingState(
      scores=jnp.zeros((batch_size, beam_size), dtype=jnp.float32),
      initialized=False,
  ), {
      "logits": jnp.repeat(logits, beam_size, axis=0),
      "token_buffer": jnp.repeat(input_token_buffer, beam_size, axis=0),
      "cache": caches,
      "done": jnp.repeat(done, beam_size, axis=0),
      "positions": jnp.repeat(positions, beam_size, axis=0),
      "logits_buffer": (
          jnp.repeat(logits_buffer, beam_size, axis=0)
          if logits_buffer is not None
          else None
      ),
  }


def beam_search_step(
    logits: jax.Array,
    done: jax.Array,
    token_buffer: jax.Array,
    cache: dict[str, dict[str, jaxtyping.Array]],
    logits_buffer: jax.Array | None,
    state: _BeamSearchSamplingState,
    pad_token_id: jax.Array,
    decoding_step: int,
) -> tuple[_BeamSearchSamplingState, dict[str, Any]]:
  """Beam search step.

  In beam search, at each step, we generate possible next tokens for each of
  previous candidates. This gives us beam_size_prev * vocab_size potential new
  sequences. Then we evaluate these new sequences scores and select the top
  beam_size sequences to continue with in the next step. The input B (batch
  size)
  is actually the original batch size * beam_size.

  Args:
    logits: The logits of the current step. [B, vocab_size]
    done: The done array of the previous step. [B]
    token_buffer: The token buffer of the previous step. [B, L]
    cache: The cache of the current step.
    logits_buffer: The logits buffer of the current step. [B, L, vocab_size]
    state: The previous beam search sampling state, which contains the last step
      scores and some constant parameters.
    pad_token_id: The pad token id used for end of sequence padding.
    decoding_step: The previous decoding step.

  Returns:
    The new beam search sampling state along with the updated token buffer,
    cache and done array because beam selection has changed the history with the
    new logits calcualtion. The returned results are sorted.
  """
  batch_size, beam_size = state.scores.shape
  vocab_size = logits.shape[-1]
  logits = logits.reshape((batch_size, beam_size, -1))
  # Calculate candidate scores
  log_probs = jax.nn.log_softmax(
      logits, axis=-1
  )  # (batch_size, beam_size, vocab_size)
  if not state.initialized:
    # the beam information is duplicated, just select the first one from every
    # batch to dedup.
    candidate_scores_per_item = log_probs[:, 0, :].reshape(
        batch_size, vocab_size
    )
  else:
    # Add current beam scores to the new log probabilities
    candidate_scores = (
        state.scores[:, :, None] + log_probs
    )  # (batch_size, beam_size, vocab_size)

    # For finished beams, set the candidate scores to -inf
    candidate_scores = jnp.where(
        done.reshape(batch_size, beam_size)[:, :, None],
        -jnp.inf,
        candidate_scores,
    )

    candidate_scores_per_item = candidate_scores.reshape(
        batch_size, beam_size * vocab_size
    )

  # don't forget finished beams
  scores_of_finished_beams = jnp.where(
      done.reshape(batch_size, beam_size), state.scores, -jnp.inf
  )
  # (batch_size, beam_size * vocab_size + beam_size)
  combined_scores = jnp.concatenate(
      [candidate_scores_per_item, scores_of_finished_beams], axis=1
  )

  # top_k_flat_indices in shape of (batch_size, beam_size)
  new_scores, top_k_flat_indices = jax.lax.top_k(combined_scores, k=beam_size)

  # Map flat indices (local to beam_size*vocab_size + finished_beam_size)
  # back to(source_beam_idx, next_token_idx). It's used to idenfity the picked
  # beam and new token. Given the current candidates also include the finished
  # beams, so additional check is needed here:
  # If indices < candidate_scores_per_item.shape[-1], candidates coming from
  # new extension candidates, otherwise, candidates comfing from
  # finished beams.
  is_extension_candidate = (
      top_k_flat_indices < candidate_scores_per_item.shape[-1]
  )
  ext_source_beam_indices = top_k_flat_indices // vocab_size
  ext_next_token_indices = top_k_flat_indices % vocab_size

  done_source_beam_indices = top_k_flat_indices - (
      candidate_scores_per_item.shape[-1]
  )
  done_next_token_indices = jnp.full(
      (batch_size, beam_size), pad_token_id, dtype=jnp.int32
  )  # Use PAD as placeholder

  source_beam_indices = jnp.where(
      is_extension_candidate, ext_source_beam_indices, done_source_beam_indices
  )

  next_token_indices = jnp.where(
      is_extension_candidate, ext_next_token_indices, done_next_token_indices
  )
  # Construct the new beam search state
  batch_indices = jnp.arange(batch_size)[:, None].repeat(
      beam_size, axis=1
  )  # (batch_size, beam_size)
  gather_indices = jnp.stack(
      [batch_indices, source_beam_indices], axis=-1
  )  # (batch_size, beam_size, 2)
  # [..., 0] is batch index, [..., 1] is beam index

  # Gather function for arrays with shape (batch_size, beam_size_prev, ...)
  def gather_beams_by_indices(arr, indices):
    # expected arr shape (batch_size * beam_size, ...)
    # indices shape (batch_size, beam_size, 2)
    # output shape (batch_size *beam_size, ...) with selected beam index
    batch_indices = indices[:, :, 0]
    beam_indices = indices[:, :, 1]
    flat_indices = (batch_indices * beam_size + beam_indices).ravel()
    return arr[flat_indices]

  # Gather tokens, lengths, and finished status
  source_beams_tokens = gather_beams_by_indices(
      token_buffer, gather_indices
  )  # (batch_size * beam_size, max_total_len)

  source_logits_buffer = (
      gather_beams_by_indices(logits_buffer, gather_indices)
      if logits_buffer is not None
      else None
  )  # (batch_size * beam_size, max_total_len, vocab_size)

  source_beams_done = gather_beams_by_indices(done, gather_indices)

  new_cache = jax.tree.map(
      lambda x: gather_beams_by_indices(x, gather_indices), cache
  )

  source_beams_tokens = source_beams_tokens.at[:, decoding_step + 1].set(
      next_token_indices.reshape(batch_size * beam_size)
  )

  return _BeamSearchSamplingState(
      scores=new_scores,
      initialized=True,
  ), {
      "token_buffer": source_beams_tokens,
      "cache": new_cache,
      "done": source_beams_done,
      "logits_buffer": source_logits_buffer,
  }


def finalize_beam_search_state(
    beam_search_state: _BeamSearchSamplingState,
    token_buffer: jax.Array,
    logits_buffer: jax.Array | None,
) -> dict[str, Any]:
  """Finalize the beam search sampling state.

  This function is called when decoding is done. Beams are sorted. So the
  final output is the first beam of each batch.

  Args:
    beam_search_state: The beam search sampling state. It contains the useful
      information for this function, for example, batch_size, beam_size, etc.
    token_buffer: The token buffer of the previous step. [B, L]
    logits_buffer: The logits buffer of the current step. [B, L, vocab_size]

  Returns:
    The updated token buffer and logits buffer by choosing the top beam result.
  """
  batch_size, beam_size = beam_search_state.scores.shape
  token_buffer = token_buffer.reshape((batch_size, beam_size, -1))
  logits_buffer = (
      logits_buffer.reshape((batch_size, beam_size, -1))
      if logits_buffer is not None
      else None
  )
  return {
      "token_buffer": token_buffer[:, 0, ...],
      "logits_buffer": (
          logits_buffer[:, 0, ...] if logits_buffer is not None else None
      ),
  }
