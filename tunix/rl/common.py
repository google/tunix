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
"""Common RL helper functions."""

from flax import nnx
import gc
import jax
from jax import numpy as jnp
import functools


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


@nnx.jit
def masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
  """Compute mean of ``x`` over True ``mask`` elements."""
  total = jnp.sum(x * mask)
  count = jnp.sum(mask)
  return total / jnp.maximum(count, 1)


@nnx.jit
def masked_whiten(x: jax.Array, mask: jax.Array, shift_mean: bool = True) -> jax.Array:
  """Whitens ``x`` considering the ``mask``."""
  mean = masked_mean(x, mask) if shift_mean else 0.0
  var = masked_mean((x - mean) ** 2, mask)
  return (x - mean) / (jnp.sqrt(var) + 1e-8)


@nnx.jit
def generalized_advantage_estimation(
    rewards: jax.Array,
    values: jax.Array,
    mask: jax.Array,
    gamma: float,
    lam: float,
) -> tuple[jax.Array, jax.Array]:
  """Compute advantages and returns using GAE."""
  seq_len = rewards.shape[1]
  adv = jnp.zeros_like(rewards)
  lastgaelam = jnp.zeros(rewards.shape[0])
  for t in range(seq_len - 1, -1, -1):
    next_value = jnp.where(t < seq_len - 1, values[:, t + 1], 0.0)
    delta = rewards[:, t] + gamma * next_value - values[:, t]
    lastgaelam = delta + gamma * lam * lastgaelam
    adv = adv.at[:, t].set(lastgaelam)
  returns = adv + values
  adv = jnp.where(mask, adv, 0.0)
  returns = jnp.where(mask, returns, 0.0)
  return adv, returns


def clear_memory() -> None:
  """Attempt to free JAX and Python memory."""
  if hasattr(jax, "clear_caches"):
    jax.clear_caches()
  gc.collect()



def pad_inputs(
    inputs: list[jax.Array],
    target_length: int,
    pad_value: int,
    left: bool,
):
  """Pads provided list of JAX arrays to the same length along the last axis.

  Args:
    inputs: A list of JAX arrays to be padded.
    target_length: The desired length of each padded array along the last axis.
    pad_value: The value to use for padding the arrays.
    left: A boolean indicating whether to pad on the left side of the array.

  Returns:
    A JAX array where each original input array has been padded to
    `target_length` along the last axis.
  """
  padded_inputs = []

  for s in inputs:
    padded_s = common.pad_to_length(
        jnp.array(s),
        target_length=target_length,
        pad_value=pad_value,
        left=left,
        axis=-1,
    )
    padded_s = padded_s[..., -target_length:]
    padded_inputs.append(padded_s)
  return jnp.array(padded_inputs)




@functools.partial(jax.jit, static_argnames=["pad_value", "eos_value"])
def process_ids(
    prompt_ids: jax.Array,
    completion_ids: jax.Array,
    pad_value: int,
    eos_value: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Process prompt and completion ids."""
  prompt_completion_ids = jnp.concat([prompt_ids, completion_ids], axis=1)

  # Compute masks. For prompt, this is just the padding mask. For completion,
  # we do an and of the padding mask and the completion mask (computed using
  # the eos token).
  prompt_mask = (prompt_ids != pad_value).astype("int32")

  completion_padding_mask = jnp.not_equal(completion_ids, pad_value).astype(
      "int32"
  )
  completion_mask = common.make_completion_mask(
      completion_ids, eos_tok=eos_value
  )
  completion_mask = completion_mask * completion_padding_mask

  prompt_completion_mask = jnp.concatenate(
      [prompt_mask, completion_mask], axis=-1
  )

  # Get positions for the concatenated prompt and completion ids.
  positions = common.build_positions_from_mask(prompt_completion_mask)
  prompt_completion_causal_mask = common.make_causal_attn_mask(
      prompt_completion_mask
  )
  return (
      positions,
      prompt_completion_ids,
      completion_mask,
      prompt_completion_mask,
      prompt_completion_causal_mask,
  )

