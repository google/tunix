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
"""Common RL helper classes and functions."""

import dataclasses
from typing import Any, Iterable

import flax
from flax import nnx
import jax
from jax import numpy as jnp


class RepeatIterable(Iterable[Any]):
  """A simple wrapper on top of an example to repeat it N times.

  This iterable wraps a list of data and allows iterating over it multiple
  times. It also provides an option to shuffle the contents of each yielded
  batch.

  Attributes:
    data: The list of data to repeat.
    repeat: The number of times to repeat the data.
    shuffle: Whether to shuffle the data.
    key: The key to use for shuffling.
  """

  def __init__(
      self,
      data: list[Any],
      repeat: int = 1,
      shuffle: bool = False,
      key: jnp.ndarray | None = None,
  ):
    self._data = data
    self._data_len = len(data)
    self._total_count = repeat * self._data_len
    self._itr_cnt = 0

    self.shuffle = shuffle
    self.key = key

  def __iter__(self):
    self._itr_cnt = 0
    return self

  def __next__(self):
    if self._itr_cnt >= self._total_count:
      raise StopIteration
    output = self._data[self._itr_cnt % self._data_len]
    self._itr_cnt += 1

    if self.shuffle:
      is_dict = True
      if not isinstance(output, dict):
        is_dict = False
        data_type = type(output)
        output = dataclasses.asdict(output)

      fields = list(output.keys())
      batch_size = output[fields[0]].shape[0]

      self.key, subkey = jax.random.split(self.key)
      shuffled_indices = jax.random.permutation(subkey, batch_size)

      new_output = {}
      for k, v in output.items():
        new_output[k] = v[shuffled_indices]
      if not is_dict:
        new_output = data_type(**new_output)  # pylint: disable=undefined-variable
      return new_output
    return output


@flax.struct.dataclass(frozen=True)
class TrainExample:
  prompt_ids: jax.Array
  prompt_mask: jax.Array
  completion_ids: jax.Array
  completion_mask: jax.Array
  advantages: jax.Array
  ref_per_token_logps: jax.Array | None
  old_per_token_logps: jax.Array | None


def compute_kl_divergence(
    per_token_logps: jax.Array, ref_per_token_logps: jax.Array
) -> jax.Array:
  """Compute per token KL divergence between trained and reference policy.

  Args:
    per_token_logps: Per token log probabilities from the trained policy.
    ref_per_token_logps: Per token log probabilities from the reference policy.

  Returns:
    KL divergence.
  """
  per_token_kl = (
      jnp.exp(ref_per_token_logps - per_token_logps)
      - (ref_per_token_logps - per_token_logps)
      - 1
  )
  return per_token_kl


def selective_log_softmax(logits: jax.Array, input_ids: jax.Array) -> jax.Array:
  """Compute the log probablity based on the input ids.

  Args:
    logits: Logits from the model.
    input_ids: Input ids to get logits.

  Returns:
    Selected log probabilities.
  """
  logps = jax.nn.log_softmax(logits, axis=-1)
  per_token_logps = jnp.take_along_axis(logps, input_ids[..., None], axis=-1)
  return per_token_logps[..., 0]


# TODO(tsbao): remove this once old callsite is cleaned up.
@nnx.jit(static_argnums=(4,))
def get_per_token_logps(
    model: nnx.Module,
    input_tokens: jax.Array,
    positions: jax.Array,
    attn_mask: jax.Array,
    logits_to_keep: int,
) -> jax.Array:
  """Computes the per-token log probabilities."""
  logits, _ = model(
      input_tokens, positions=positions, attention_mask=attn_mask, cache=None
  )
  logits = logits[:, -logits_to_keep - 1 : -1, :]
  input_tokens = input_tokens[:, -logits_to_keep:]
  return selective_log_softmax(logits, input_tokens)


# TODO(abheesht): This is computed 4 times - twice in `compute_per_token_logps`
# and twice in `compute_score`. We can factor this out and compute it just once.
@nnx.jit(static_argnames=('pad_id', 'eos_id'))
def process_ids(
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
):
  """Processes prompt and completion ids."""

  prompt_completion_ids = jnp.concat([prompt_tokens, completion_tokens], axis=1)
  prompt_mask = prompt_tokens != pad_id
  completion_mask = make_completion_mask(completion_tokens, eos_tok=eos_id)
  prompt_completion_mask = jnp.concatenate(
      [prompt_mask, completion_mask], axis=-1
  )
  positions = build_positions_from_mask(prompt_completion_mask)
  attn_mask = make_causal_attn_mask(prompt_completion_mask)

  return prompt_completion_ids, positions, attn_mask


@nnx.jit(static_argnames=('pad_id', 'eos_id', 'stop_gradient'))
def compute_per_token_logps(
    model: nnx.Module,
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    stop_gradient: bool = True,
) -> jax.Array:
  """Computes the per-token log probabilities."""
  prompt_completion_ids, positions, attn_mask = process_ids(
      prompt_tokens, completion_tokens, pad_id, eos_id
  )
  per_token_logps = get_per_token_logps(
      model,
      input_tokens=prompt_completion_ids,
      positions=positions,
      attn_mask=attn_mask,
      logits_to_keep=completion_tokens.shape[1],
  )
  if stop_gradient:
    per_token_logps = jax.lax.stop_gradient(per_token_logps)
  return per_token_logps


@nnx.jit(static_argnames=('pad_id', 'eos_id', 'stop_gradient'))
def compute_score(
    model,
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    stop_gradient: bool = True,
):
  """Computes reward using the provided model."""
  prompt_completion_ids, positions, attn_mask = process_ids(
      prompt_tokens, completion_tokens, pad_id, eos_id
  )

  per_token_scores = model.score(
      prompt_completion_ids,
      positions=positions,
      attention_mask=attn_mask,
  )
  # The model returns a tensor of shape [B, T, 1]. We squeeze the last
  # dimension to get a tensor of shape [B, T].
  per_token_scores = jnp.squeeze(per_token_scores, axis=-1)

  if stop_gradient:
    per_token_scores = jax.lax.stop_gradient(per_token_scores)

  return per_token_scores


def make_completion_mask(completion_ids, eos_tok: int = 0):
  """Create completion mask based on the EOS token.

  Args:
    completion_ids: Completion ids.
    eos_tok: EOS token id.

  Returns:
    Completion mask.
  """
  is_eos = completion_ids == eos_tok
  eos_idx = jnp.full((is_eos.shape[0],), is_eos.shape[1], dtype=jnp.int32)

  any_eos = jnp.any(is_eos, axis=1)
  eos_idx = jax.lax.select(any_eos, jnp.argmax(is_eos, axis=1), eos_idx)

  sequence_indices = jnp.arange(is_eos.shape[1])[None, :]
  sequence_indices = jnp.broadcast_to(
      sequence_indices, (is_eos.shape[0], is_eos.shape[1])
  )
  completion_mask = (sequence_indices <= eos_idx[:, None]).astype(jnp.int32)

  return completion_mask


def make_causal_attn_mask(input_mask: jax.Array) -> jax.Array:
  """Create causal attention mask.

  Args:
    input_mask: Mask for the input

  Returns:
    Attention mask.
  """
  seq_len = input_mask.shape[-1]
  attn_mask = input_mask[..., None, :]
  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
  # Prefixes can be attended by all tokens
  attn_mask *= causal_mask[None, ...]
  return attn_mask


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


def build_positions_from_mask(input_mask: jax.Array) -> jax.Array:
  """Computes `positions` from the `input_mask`.

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
