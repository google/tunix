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


"""Utility functions for sampler."""

import functools
import jax
from jax import lax
import jax.numpy as jnp


def compute_attention_masks(
    time_step: int, seq_len: int, input_mask: jax.Array
) -> jax.Array:
  """Computes causal attention mask."""
  batch_size = input_mask.shape[0]
  batch_time_step = jnp.full((batch_size, 1), time_step, dtype=jnp.uint32)
  causal_padding = jnp.greater(
      jnp.expand_dims(jnp.arange(seq_len), 0), batch_time_step
  )
  max_seq_len = min(input_mask.shape[-1], seq_len)
  input_mask = jax.lax.dynamic_slice(
      input_mask,
      (0, jnp.maximum(time_step - seq_len + 1, 0)),
      (batch_size, max_seq_len),
  )
  input_mask = (
      jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)
      .at[:, :max_seq_len]
      .set(input_mask)
  )

  causal_padding = jnp.logical_or(causal_padding, input_mask)
  attention_mask = causal_padding[:, jnp.newaxis, :].astype(jnp.bool_)

  return ~attention_mask


def make_causal_attn_mask(input_mask: jax.Array, cache_size: int) -> jax.Array:
  """Create causal attention mask for prefill.

  The causal attention mask during prefill phase is having shape
  (B, T, CACHE_SIZE).

  Args:
    input_mask: Mask for the input
    cache_size: KV cache size

  Returns:
    Attention mask.
  """
  seq_len = input_mask.shape[-1]
  attn_mask = input_mask[..., None, :]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  attn_mask *= causal_mask[None, ...]
  padding = cache_size - seq_len
  assert padding >= 0
  attn_mask = jnp.pad(
      attn_mask, (*((0, 0) for _ in range(attn_mask.ndim - 1)), (0, padding))
  )
  return attn_mask


def next_power_of_2(x: int) -> int:
  """Returns the next power of 2 that is not smaller than x."""
  if x == 0:
    return 1
  return int(2 ** int(jnp.ceil(jnp.log2(x))))


def pad_to_length(
    x: jax.Array,
    target_length: int,
    pad_value: int = 0,
    left=False,
    axis: int = 0,
) -> jax.Array:
  """Pads a JAX array to a specified target length along a given axis.

  Args:
      x: The JAX array to pad.
      target_length: The desired length of the padded array.
      pad_value: The value to use for padding (default: 0).
      left: If True, add padding tokens to the left of the array.
      axis: The axis along which to pad (default: 0).

  Returns:
      A new JAX array that is padded to the target length along the specified
      axis. Return original array if it is already longer than the target
      length.
  """
  length = x.shape[axis]
  if length >= target_length:
    return x

  padding_shape = list(x.shape)
  padding_shape[axis] = target_length - length
  padding = jnp.full(padding_shape, pad_value, dtype=x.dtype)

  if left:
    return jnp.concatenate([padding, x], axis=axis)
  else:
    return jnp.concatenate([x, padding], axis=axis)


def find_first_non_pad_idx(ids, pad_id):
  """Finds the index of the first non-pad token."""
  assert ids.ndim == 1, f'ids should be a 1d array. Got: {ids.shape}'
  mask = ids != pad_id

  return lax.cond(
      jnp.any(mask),
      lambda operands: jnp.argmax(operands[0]),
      lambda operands: 0,
      (mask,),
  )


def find_first_eos_idx(ids, eos_id):
  """Finds the index of the first EOS token."""
  assert ids.ndim == 1, f'ids should be a 1d array. Got: {ids.shape}'
  mask = ids == eos_id

  return lax.cond(
      jnp.any(mask),
      lambda operands: jnp.argmax(operands[0]),
      lambda operands: operands[1].shape[0],
      (mask, ids),
  )


def find_last_non_pad_idx(ids, pad_id):
  """Finds the index of the last non-pad token."""
  assert ids.ndim == 1, f'ids should be a 1d array. Got: {ids.shape}'
  mask = ids != pad_id
  reversed_mask = jnp.flip(mask, axis=-1)

  return jax.lax.cond(
      jnp.any(reversed_mask),
      lambda operands: operands[1].shape[-1] - jnp.argmax(operands[0]) - 1,
      lambda operands: operands[1].shape[-1],
      (reversed_mask, ids),
  )


@functools.partial(
    jax.jit,
    static_argnames=[
        'return_logits',
        'echo',
        'pad_value',
        'eos_value',
        'max_prompt_length',
        'max_total_length',
    ],
)
def padded_fill_tokens_and_logits(
    token_buffers: jax.Array,
    logits_buffers: jax.Array | None,
    return_logits: bool,
    echo: bool,
    pad_value: int,
    eos_value: int,
    max_prompt_length: int,
    max_total_length: int,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
  """Truncates the token_buffers and logits_buffers to the valid output.

  For the token_buffers, find the valid output tokens from the start_idx to the
  end_idx. Then pad the valid output tokens to the max_total_length. Similar
  operation for the logits_buffers if return_logits is True.

  Args:
    token_buffers: The token buffers from the sampler. [B, L2]
    logits_buffers: The logits buffers from the sampler. [B, L2, V]
    return_logits: Whether to return the logits.
    echo: Whether to echo the input prompt in the output.
    pad_value: The value to use for padding.
    eos_value: The value to use for EOS.
    max_prompt_length: The maximum length of the input prompt.
    max_total_length: The maximum total length of the output.

  Returns:
    The shape of the valid output tokens, the output tokens and the output
    logits.
  """
  return jax.vmap(
      single_padded_fill_tokens_and_logits,
      in_axes=(0, 0, None, None, None, None, None, None),
      out_axes=(0, 0, 0),
  )(
      token_buffers,
      logits_buffers,
      return_logits,
      echo,
      pad_value,
      eos_value,
      max_prompt_length,
      max_total_length,
  )


def single_padded_fill_tokens_and_logits(
    token_buffer: jax.Array,
    logits_buffer: jax.Array | None,
    return_logits: bool,
    echo: bool,
    pad_value: int,
    eos_value: int,
    max_prompt_length: int,
    max_total_length: int,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
  """Generates tokens and logits from the input token_buffer and logits_buffer."""
  start_idx = (
      find_first_non_pad_idx(token_buffer, pad_value)
      if echo
      else max_prompt_length
  )
  end_idx = (
      find_first_eos_idx(token_buffer[max_prompt_length:], eos_value)
      + max_prompt_length
  )
  length = end_idx - start_idx
  mask = jnp.arange(max_total_length) < length
  padded_token_buffer = jnp.pad(
      token_buffer, (0, max_total_length), constant_values=pad_value
  )
  output_token = lax.dynamic_slice(
      padded_token_buffer, (start_idx,), (max_total_length,)
  )
  output_token = jnp.where(mask, output_token, pad_value)

  output_logit = None
  if return_logits:
    assert logits_buffer is not None
    dim = logits_buffer.shape[-1]
    padded_logits_buffer = jnp.pad(
        logits_buffer, ((0, max_total_length), (0, 0)), constant_values=0
    )
    output_logit = lax.dynamic_slice(
        padded_logits_buffer, (start_idx, 0), (max_total_length, dim)
    )
    mask = mask[:, None]
    output_logit = jnp.where(mask, output_logit, 0)
  return jnp.array(length), output_token, output_logit


def build_positions_from_mask(input_mask: jax.Array) -> jax.Array:
  """Computes the `positions` from the `input_mask`.

  Args:
    input_mask: The tokens `input_mask`, True for non-padded tokens only.

  Returns:
    The indices to use for RoPE and absolute position encodings for the given
    input mask.
  """
  positions = jnp.cumsum(input_mask, axis=-1)
  # Subtract one for all positions from the first valid one as they are
  # 0-indexed
  return positions - (positions >= 1)


def check_sampling_mode_conflict(
    original_sampling_mode: list[
        str | None
    ],  # pass in as list to modify in place
    new_sampling_mode: str,
) -> None:
  """Checks if the new sampling mode conflicts with the original sampling mode."""

  if original_sampling_mode[0] is not None:
    raise ValueError(
        'Conflicts setting sampling_mode, the current set sampling_mode is'
        f' {original_sampling_mode[0]} but trying to override to'
        f' {new_sampling_mode}. The rules are\n: 1. If top_p is provided,'
        ' top_p will be used. 2. If beam_size is provided,beam_search will be'
        ' used 3. If none of the above, greedy will be used.'
    )
  else:
    original_sampling_mode[0] = new_sampling_mode
