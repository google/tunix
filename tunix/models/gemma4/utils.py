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

"""Utility functions for Gemma 4."""

from __future__ import annotations

import jax.numpy as jnp
import jaxtyping

_PADDING_ID = 0


def get_attention_mask(
    tokens: jaxtyping.ArrayLike,  # (B, L)
    *,
    inputs_mask: jaxtyping.ArrayLike | None = None,  # (B, L, L')
    token_placeholder_id: int | None = None,
):
  """Returns the attention mask for the transformer."""
  if inputs_mask is None:
    inputs_mask = tokens != _PADDING_ID

  bidirectional_mask = None
  if token_placeholder_id is not None:
    bidirectional_mask = tokens == token_placeholder_id
  attention_mask = make_causal_bidirectional_attention_mask(
      inputs_mask,
      bidirectional_mask=bidirectional_mask,
  )

  return attention_mask


def make_causal_bidirectional_attention_mask(
    causal_mask: jaxtyping.ArrayLike,  # (B, L)
    *,
    bidirectional_mask: jaxtyping.ArrayLike | None = None,  # (B, L)
) -> jaxtyping.ArrayLike:
  """Make the attention mask for the transformer."""

  attention_mask = _make_causal_mask(causal_mask)

  if bidirectional_mask is not None:
    attention_mask = _add_bidirectional_mask(attention_mask, bidirectional_mask)

  return attention_mask


def _make_causal_mask(
    input_mask: jaxtyping.ArrayLike,  # (B, L)
) -> jaxtyping.ArrayLike:
  """Makes causal attention mask."""
  if len(input_mask.shape) != 2:  # pytype: disable=attribute-error
    raise ValueError(
        f'Input mask must be 2D (shape [B, L]), but got {input_mask.shape}.'  # pytype: disable=attribute-error
    )
  seq_len = input_mask.shape[-1]  # pytype: disable=attribute-error
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
  attn_mask = input_mask[..., None, :]
  attn_mask *= causal_mask[None, ...]
  return attn_mask


def _make_block_mask_indices(
    bidirectional_mask: jaxtyping.ArrayLike,  # (B, L)
) -> jaxtyping.ArrayLike:
  """Creates block mask identifying segments based on a bidirectional mask."""
  padded_mask = jnp.pad(bidirectional_mask, [(0, 0), (1, 0)], constant_values=0)
  boundary = padded_mask[..., 1:] > padded_mask[..., :-1]
  numbered_boundary = jnp.cumsum(boundary, axis=-1)
  return bidirectional_mask * numbered_boundary


def _add_bidirectional_mask(
    attn_mask: jaxtyping.ArrayLike,  # (B, L, L)
    bidirectional_mask: jaxtyping.ArrayLike,  # (B, L)
) -> jaxtyping.ArrayLike:
  """Adds bidirectional mask to the attention mask."""
  q_block_indices = _make_block_mask_indices(bidirectional_mask)
  kv_block_indices = q_block_indices
  attn_mask = attn_mask | (
      (kv_block_indices[:, None, :] == q_block_indices[..., None])
      & (q_block_indices[..., None] > 0)
  )
  return attn_mask
