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

from typing import Any, Iterable

import flax
from flax import nnx
import jax
from jax import numpy as jnp
import jax.tree_util as jtu
from tunix.sft import utils

make_causal_attn_mask = utils.make_causal_attn_mask
build_positions_from_mask = utils.build_positions_from_mask


class RepeatIterable(Iterable[Any]):
  """An iterable that processes a list of rollout batches.

  For each rollout batch, it shuffles its contents, slices it into mini-batches,
  and yields them sequentially before moving to the next rollout batch. This
  entire process is repeated for a specified number of epochs.
  """

  def __init__(
      self,
      data: list[Any],
      repeat: int,
      mini_batch_size: int | None = None,
      shuffle: bool = False,
      key: jnp.ndarray | None = None,
  ):
    self._data = data

    self.repeat = repeat
    self.mini_batch_size = mini_batch_size

    self.shuffle = shuffle
    self.key = key if key is not None else jax.random.PRNGKey(0)

    # Maintain a private, mutable `mini_batch_size`, for simpler code.
    self._mini_batch_size = mini_batch_size

  def _shuffle_and_slice_one_batch(self, rollout_batch: Any):
    """A generator that shuffles and slices a single rollout batch."""
    leaves, _ = jtu.tree_flatten(rollout_batch)
    rollout_batch_size = leaves[0].shape[0]

    if self.mini_batch_size is None:
      self._mini_batch_size = rollout_batch_size

    if rollout_batch_size % self._mini_batch_size != 0:
      raise ValueError(
          "Each rollout batch's size must be divisible by `mini_batch_size`."
      )
    num_mini_batches = rollout_batch_size // self._mini_batch_size

    # Shuffle indices.
    if self.shuffle:
      self.key, _ = jax.random.split(self.key)
      shuffled_indices = jax.random.permutation(self.key, rollout_batch_size)
    else:
      shuffled_indices = jnp.arange(rollout_batch_size)

    # Slice the rollout batch into mini-batches.
    for i in range(num_mini_batches):
      start = i * self._mini_batch_size
      end = start + self._mini_batch_size
      batch_indices = shuffled_indices[start:end]

      mini_batch = jtu.tree_map(
          lambda leaf, indices=batch_indices: leaf[indices], rollout_batch
      )
      yield mini_batch

  def __iter__(self):
    """The main generator for the iterable."""
    for _ in range(self.repeat):
      for rollout_batch in self._data:
        yield from self._shuffle_and_slice_one_batch(rollout_batch)


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
    per_token_logps: jax.Array,
    ref_per_token_logps: jax.Array,
    method: str = "low_var_kl",
) -> jax.Array:
  """Compute per token KL divergence between trained and reference policy.

  Based on `method`, we compute one of three kinds of KL divergence:
  - "kl": Unbiased, high-variance estimator. Simple Forward KL:
    `logp - ref_logp`.
  - "mse_kl": Biased, low-variance estimator. Squared log-difference:
    `0.5 * (logp - ref_logp)^2`.
  - "low_var_kl": Unbiased, low-variance estimator. J. Schulman low-variance
    approx: `(r - 1) - log r`, where `r = q/p = exp(ref_logp - logp)`.

  Args:
    per_token_logps: Per token log probabilities from the trained policy.
    ref_per_token_logps: Per token log probabilities from the reference policy.
    method: KL penalty method. Defaults to "low_var_kl".

  Returns:
    KL divergence.
  """
  if method == "kl":
    return per_token_logps - ref_per_token_logps
  elif method == "mse_kl":
    return 0.5 * jnp.square(per_token_logps - ref_per_token_logps)
  elif method == "low_var_kl":
    kl = ref_per_token_logps - per_token_logps
    return jnp.exp(kl) - (kl) - 1
  else:
    raise ValueError(
        "`method` must be one of 'kl', 'mse_kl', 'low_var_kl'. Received:"
        f" {method}"
    )


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
) -> tuple[jax.Array, jax.Array]:
  """Computes the per-token log probabilities."""
  logits, _ = model(
      input_tokens, positions=positions, attention_mask=attn_mask, cache=None
  )
  logits = logits[:, -logits_to_keep - 1 : -1, :]
  input_tokens = input_tokens[:, -logits_to_keep:]
  per_token_logps = selective_log_softmax(logits, input_tokens)
  return per_token_logps, logits


# TODO(abheesht): This is computed 4 times - twice in `compute_per_token_logps`
# and twice in `compute_score`. We can factor this out and compute it just once.
@nnx.jit(static_argnames=("pad_id", "eos_id"))
def process_ids(
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    completion_mask: jax.Array | None = None,
):
  """Processes prompt and completion ids."""

  prompt_completion_ids = jnp.concat([prompt_tokens, completion_tokens], axis=1)
  prompt_mask = prompt_tokens != pad_id

  if completion_mask is None:
    completion_mask = make_completion_mask(completion_tokens, eos_tok=eos_id)

  prompt_completion_mask = jnp.concatenate(
      [prompt_mask, completion_mask], axis=-1
  )
  positions = build_positions_from_mask(prompt_completion_mask)
  attn_mask = make_causal_attn_mask(prompt_completion_mask)

  return prompt_completion_ids, positions, attn_mask


@nnx.jit(static_argnames=("pad_id", "eos_id", "stop_gradient"))
def compute_per_token_logps(
    model: nnx.Module,
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    completion_mask: jax.Array | None = None,
    stop_gradient: bool = True,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Computes the per-token log probabilities."""
  prompt_completion_ids, positions, attn_mask = process_ids(
      prompt_tokens, completion_tokens, pad_id, eos_id, completion_mask
  )
  per_token_logps, logits = get_per_token_logps(
      model,
      input_tokens=prompt_completion_ids,
      positions=positions,
      attn_mask=attn_mask,
      logits_to_keep=completion_tokens.shape[1],
  )
  if stop_gradient:
    per_token_logps = jax.lax.stop_gradient(per_token_logps)
    logits = jax.lax.stop_gradient(logits)
  return per_token_logps, logits


@nnx.jit(static_argnames=("pad_id", "eos_id", "stop_gradient"))
def compute_score(
    model,
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    completion_mask: jax.Array | None = None,
    stop_gradient: bool = True,
):
  """Computes reward using the provided model."""
  prompt_completion_ids, positions, attn_mask = process_ids(
      prompt_tokens, completion_tokens, pad_id, eos_id, completion_mask
  )

  out = model(
      prompt_completion_ids,
      positions=positions,
      cache=None,
      attention_mask=attn_mask,
  )
  per_token_scores = out[0] if isinstance(out, tuple) else out
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
