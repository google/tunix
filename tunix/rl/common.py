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

from functools import partial  # pylint: disable=g-importing-member
import inspect
from typing import Any, Iterable

import flax
from flax import nnx
import jax
from jax import numpy as jnp
import jax.tree_util as jtu
import numpy as np
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
      start = i * self._mini_batch_size  # pyrefly: ignore[unsupported-operation]
      end = start + self._mini_batch_size  # pyrefly: ignore[unsupported-operation]
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
  segment_ids: jax.Array | None = None
  segment_positions: jax.Array | None = None
  # Truncated importance-sampling correction weights for off-policy
  # correction between the rollout sampler and the trainer. Per-token,
  # detached, multiplied into the policy-gradient loss BEFORE aggregation
  # to dampen positions where the trainer's recomputed log-probability
  # diverges from the rollout sampler's. ``None`` disables the correction.
  sampler_is_weights: jax.Array | None = None


def compute_kl_divergence(
    per_token_logps: jax.Array,
    ref_per_token_logps: jax.Array,
    method: str = "low_var_kl",
    clamp_value: float | None = None,
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
    clamp_value: Optional symmetric clamp applied to the returned KL, i.e.
      `clip(kl, -clamp_value, +clamp_value)`. `None` (default) disables the
      clamp and preserves prior behavior. Set to a positive float (e.g.
      `10000.0`) to bound rare outliers — useful when the trained policy briefly
      drifts far from the reference and the `low_var_kl` estimator's `exp(diff)`
      term can overflow fp32 / saturate bf16.

  Returns:
    KL divergence.
  """
  per_token_logps = per_token_logps.astype(jnp.float32)
  if ref_per_token_logps is not None:
    ref_per_token_logps = ref_per_token_logps.astype(jnp.float32)

  if method == "kl":
    kl = per_token_logps - ref_per_token_logps
  elif method == "mse_kl":
    kl = 0.5 * jnp.square(per_token_logps - ref_per_token_logps)
  elif method == "low_var_kl":
    diff = ref_per_token_logps - per_token_logps
    kl = jnp.exp(diff) - diff - 1
  else:
    raise ValueError(
        "`method` must be one of 'kl', 'mse_kl', 'low_var_kl'. Received:"
        f" {method}"
    )

  if clamp_value is not None:
    kl = jnp.clip(kl, -clamp_value, clamp_value)
  return kl


def selective_log_softmax(logits: jax.Array, input_ids: jax.Array) -> jax.Array:
  """Compute the log probablity based on the input ids.

  Args:
    logits: Logits from the model.
    input_ids: Input ids to get logits.

  Returns:
    Selected log probabilities.
  """
  target_logits = (
      jnp.take_along_axis(logits, input_ids[..., None], axis=-1)
      .squeeze(-1)
      .astype(jnp.float32)
  )
  normalizer = jax.nn.logsumexp(logits.astype(jnp.float32), axis=-1)
  return target_logits - normalizer


# TODO(tsbao): remove this once old callsite is cleaned up.
@nnx.jit(static_argnames=("logits_to_keep",))
def get_per_token_logps(
    model: nnx.Module,
    input_tokens: jax.Array,
    positions: jax.Array,
    attn_mask: jax.Array,
    logits_to_keep: int,
    images: jax.Array | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Computes the per-token log probabilities."""
  kwargs = {} if images is None else {"images": images}
  logits, _ = model(
      input_tokens,
      positions=positions,
      attention_mask=attn_mask,
      cache=None,
      **kwargs,
  )
  logits = logits[:, -logits_to_keep - 1 : -1, :]
  input_tokens = input_tokens[:, -logits_to_keep:]
  per_token_logps = selective_log_softmax(logits, input_tokens)
  return per_token_logps


# TODO(abheesht): This is computed 4 times - twice in `compute_per_token_logps`
# and twice in `compute_score`. We can factor this out and compute it just once.
@partial(jax.jit, static_argnames=("pad_id", "eos_id"))
def process_ids(
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    segment_ids: jax.Array | None = None,
    segment_positions: jax.Array | None = None,
):
  """Processes prompt and completion ids.

  Args:
    prompt_tokens: jax.Array token IDs for prompt. If sequence packing is
      enabled, prompt_tokens will be empty (shape [B, 0]), because prompts and
      completions are already concatenated into completion_tokens.
    completion_tokens: jax.Array token IDs for completion. If sequence packing
      is enabled, completion_tokens functions as a unified 1D buffer holding
      pre-concatenated mixed prompt and completion boundaries padded
      sequentially.
    pad_id: pad token identifier.
    eos_id: end of sequence identifier.
    completion_mask: optional attention weights mapping completion sequences.
    segment_ids: optional 1D sequential document identifiers used for packing.
    segment_positions: optional 1D local position indices used for packing.
  """
  prompt_completion_ids = jnp.concat([prompt_tokens, completion_tokens], axis=1)

  if segment_ids is not None:
    # Positions are either provided or must be computed correctly (assumed
    # provided for packed).
    if segment_positions is None:
      raise ValueError(
          "segment_positions must be explicitly provided for packed sequences. "
      )
    attn_mask = None  # Relies on segment_ids inside the model
    # Packed callers supply their own segment_ids that already separate
    # distinct documents in the buffer; no need for an additional padding
    # mask here.
    return prompt_completion_ids, segment_positions, attn_mask, None

  prompt_mask = prompt_tokens != pad_id
  completion_mask = completion_tokens != pad_id

  prompt_completion_mask = jnp.concatenate(
      [prompt_mask, completion_mask], axis=-1
  )
  positions = build_positions_from_mask(prompt_completion_mask)
  attn_mask = make_causal_attn_mask(prompt_completion_mask)

  # 1-D per-position non-pad mask for the full prompt+completion sequence.
  # Used as ``segment_ids`` by attention kernels that cannot consume the 2-D
  # ``attn_mask`` directly (e.g. pallas splash attention takes only a causal
  # mask kernel-side and respects per-position segment ids to suppress
  # cross-segment attention). With pad=0 and real=1, a real position never
  # attends to a pad position regardless of where padding lives in the
  # sequence (typically left-padded for prompt-side alignment).
  input_seg_ids = prompt_completion_mask.astype(jnp.int32)

  return prompt_completion_ids, positions, attn_mask, input_seg_ids


@partial(
    jax.jit,
    static_argnames=(
        "pad_id",
        "eos_id",
        "stop_gradient",
        "return_entropy",
        "temperature",
        "chunk_size",
    ),
)
def compute_per_token_logps(
    graphdef,
    state,
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    images: jax.Array | None = None,
    stop_gradient: bool = True,
    return_entropy: bool = False,
    segment_ids: jax.Array | None = None,
    segment_positions: jax.Array | None = None,
    temperature: float = 1.0,
    chunk_size: int = 0,
) -> jax.Array | tuple[jax.Array, jax.Array]:
  """Computes the per-token log probabilities.

  Args:
    graphdef: Flax NNX GraphDef.
    state: Flax NNX State.
    prompt_tokens: jax.Array token IDs for prompt. If sequence packing is
      enabled, prompt_tokens will be empty (shape [B, 0]), because prompts and
      completions are already concatenated into completion_tokens.
    completion_tokens: jax.Array token IDs for completion. If sequence packing
      is enabled, completion_tokens functions as a unified 1D buffer holding
      pre-concatenated mixed prompt and completion boundaries padded
      sequentially.
    pad_id: pad token identifier.
    eos_id: end of sequence identifier.
    images: optional images array.
    stop_gradient: whether to stop gradient.
    return_entropy: whether to return per token entropy.
    segment_ids: optional 1D sequential document identifiers used for packing.
    segment_positions: optional 1D local position indices used for packing.
    temperature: temperature used for rollout.
    chunk_size: If not 0, computes the log probabilities in sequence chunks of
      this size.

  Returns:
    per_token_logps: jax.Array token-level logarithmic values.
      Without sequence packing, returns log probs for completion tokens only,
      with shape `[B, completion_len]`. With sequence packing, returns log
      probs for the full packed sequence padded out (since prompts and
      completions of multiple sequences are concatenated), with shape `[B,
      FullSeqLen]`.
    entropy: optional per-token entropy jax.Array of shape matches per_token_logps,
      returned if return_entropy is True.
  """
  model = nnx.merge(graphdef, state)

  input_tokens, calculated_positions, attn_mask, input_seg_ids = process_ids(
      prompt_tokens,
      completion_tokens,
      pad_id,
      eos_id,
      segment_ids,
      segment_positions,
  )

  model_kwargs = {
      "positions": calculated_positions,
      "cache": None,
      "attention_mask": attn_mask,
  }
  if chunk_size > 0:
    sig = inspect.signature(model.__call__)
    if "skip_lm_head" not in sig.parameters:
      raise ValueError("Model __call__ must accept skip_lm_head.")
    if not hasattr(model, "compute_final_logits") or not callable(
        getattr(model, "compute_final_logits")
    ):
      raise ValueError(
          "Model must contain a function called compute_final_logits."
      )
    model_kwargs["skip_lm_head"] = True

  # Pass through any segment ids so the model's attention kernel can respect
  # them only if the model signature accepts it: caller-provided packing ids take
  # precedence; otherwise we pass the per-position non-pad mask derived in
  # ``process_ids`` so flash-attention variants that lack a separate
  # padding-mask input still skip pad positions.
  try:
    sig = inspect.signature(model.__call__)
    has_segment_ids = ("segment_ids" in sig.parameters) or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
  except Exception:
    has_segment_ids = False

  if has_segment_ids:
    if segment_ids is not None:
      model_kwargs["segment_ids"] = segment_ids
    elif input_seg_ids is not None:
      model_kwargs["segment_ids"] = input_seg_ids
  if images is not None:
    model_kwargs["images"] = images

  outputs, _ = model(input_tokens, **model_kwargs)

  if segment_ids is not None:
    # Packed Mode: Evaluate the full sequence (mixed prompts + completions).
    # Since predicting token[i] requires logit[i-1], we skip the first token.
    # This shrinks the output shape to [Batch, FullSeqLen - 1]
    logits_to_keep = input_tokens.shape[1] - 1
  else:
    logits_to_keep = completion_tokens.shape[1]

  input_tokens_to_keep = input_tokens[:, -logits_to_keep:]

  if chunk_size > 0:
    hidden_state = outputs[:, -logits_to_keep - 1 : -1, :]
    out = compute_chunked_logps(
        model,
        hidden_state,
        input_tokens_to_keep,
        temperature,
        chunk_size,
        return_entropy,
    )
    if return_entropy:
      per_token_logps, per_token_entropy = out
    else:
      per_token_logps = out

    if segment_ids is not None:
      per_token_logps = jnp.pad(
          per_token_logps, ((0, 0), (1, 0)), constant_values=0.0
      )
      if return_entropy:
        per_token_entropy = jnp.pad(
            per_token_entropy, ((0, 0), (1, 0)), constant_values=0.0  # pyrefly: ignore[unbound-name]
        )

    if stop_gradient:
      per_token_logps = jax.lax.stop_gradient(per_token_logps)
      if return_entropy:
        per_token_entropy = jax.lax.stop_gradient(per_token_entropy)  # pyrefly: ignore[unbound-name]

    if return_entropy:
      return per_token_logps, per_token_entropy  # pyrefly: ignore[unbound-name]
    return per_token_logps
  else:
    logits = outputs[:, -logits_to_keep - 1 : -1, :]
    if temperature != 0.0 and temperature != 1.0:
      logits /= temperature

    per_token_logps = selective_log_softmax(logits, input_tokens_to_keep)

    if segment_ids is not None:
      # Pad the front with 0.0 to make shape back to [Batch, FullSeqLen]. This
      # aligns indices (logp[i] matches token[i]) and avoids mask slicing downstream.
      per_token_logps = jnp.pad(
          per_token_logps, ((0, 0), (1, 0)), constant_values=0.0
      )
      logits = jnp.pad(logits, ((0, 0), (1, 0), (0, 0)), constant_values=0.0)

    if stop_gradient:
      per_token_logps = jax.lax.stop_gradient(per_token_logps)
      logits = jax.lax.stop_gradient(logits)

    if return_entropy:
      entropy = compute_entropy_from_logits(logits)
      return per_token_logps, entropy
    return per_token_logps


def compute_chunked_logps(
    model,
    hidden_states,
    target_ids,
    temperature,
    chunk_size,
    return_entropy,
):
  """Computes per-token log probabilities in sequence chunks to save VRAM.

  Args:
      model: The actor model (needs to expose `lm_head`)
      hidden_states: [Batch, SeqLen, HiddenDim]
      target_ids:    [Batch, SeqLen]
      chunk_size:    Number of tokens to process at a time per sequence.

  Returns:
     per_token_logps: [Batch, SeqLen]
  """
  batch_size, seq_len, hidden_dim = hidden_states.shape

  # 1. Pad the sequence dimension if it's not perfectly divisible by chunk_size
  pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
  if pad_len > 0:
    # Pad sequence dimension (axis 1)
    hidden_states = jnp.pad(hidden_states, ((0, 0), (0, pad_len), (0, 0)))
    target_ids = jnp.pad(target_ids, ((0, 0), (0, pad_len)))

  padded_seq_len = seq_len + pad_len
  num_chunks = padded_seq_len // chunk_size

  # 2. Reshape into chunks: [Batch, NumChunks, ChunkSize, ...]
  hs_reshaped = hidden_states.reshape(
      batch_size, num_chunks, chunk_size, hidden_dim
  )
  ids_reshaped = target_ids.reshape(batch_size, num_chunks, chunk_size)

  # 3. Swap B, T axes to make it time-major for jax.lax.scan
  hs_scannable = jnp.swapaxes(hs_reshaped, 0, 1)
  ids_scannable = jnp.swapaxes(ids_reshaped, 0, 1)

  @nnx.remat
  def logp_step(carry, xs):
    hs_chunk, ids_chunk = xs

    # Project to vocabulary for just this chunk
    # Peak memory: [Batch, ChunkSize, VocabSize]
    logits_chunk = model.compute_final_logits(hs_chunk).astype(jnp.float32)

    if temperature != 0.0 and temperature != 1.0:
      logits_chunk /= temperature

    logps_chunk = selective_log_softmax(logits_chunk, ids_chunk)

    if return_entropy:
      entropy_chunk = compute_entropy_from_logits(logits_chunk)
      return None, (logps_chunk, entropy_chunk)
    else:
      return None, logps_chunk

  # 4. Scan over the NumChunks dimension
  _, scanned_out = jax.lax.scan(
      logp_step, init=None, xs=(hs_scannable, ids_scannable)
  )

  if return_entropy:
    logps_chunked, entropy_chunked = scanned_out
    # 5. Swap back to batch-major and flatten the sequence dimension
    logps_reshaped = jnp.swapaxes(logps_chunked, 0, 1)
    entropy_reshaped = jnp.swapaxes(entropy_chunked, 0, 1)
    # 6. Slice off any padding we added initially.
    per_token_logps = logps_reshaped.reshape(batch_size, padded_seq_len)[
        :, :seq_len
    ]
    per_token_entropy = entropy_reshaped.reshape(batch_size, padded_seq_len)[
        :, :seq_len
    ]
    return per_token_logps, per_token_entropy
  else:
    logps_chunked = scanned_out
    # 5. Swap back to batch-major and flatten the sequence dimension
    logps_reshaped = jnp.swapaxes(logps_chunked, 0, 1)
    # 6. Slice off any padding we added initially.
    per_token_logps = logps_reshaped.reshape(batch_size, padded_seq_len)[
        :, :seq_len
    ]
    return per_token_logps





@nnx.jit(static_argnames=("pad_id", "eos_id", "stop_gradient"))
def compute_score(
    model,
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    stop_gradient: bool = True,
    segment_ids: jax.Array | None = None,
    segment_positions: jax.Array | None = None,
):
  """Computes reward using the provided model."""
  (
      prompt_completion_ids,
      calculated_positions,
      attn_mask,
      input_seg_ids,
  ) = process_ids(
      prompt_tokens,
      completion_tokens,
      pad_id,
      eos_id,
      segment_ids,
      segment_positions,
  )

  model_kwargs = {"positions": calculated_positions, "cache": None}
  if segment_ids is not None:
    model_kwargs["segment_ids"] = segment_ids
  else:
    model_kwargs["attention_mask"] = attn_mask
    if input_seg_ids is not None:
      model_kwargs["segment_ids"] = input_seg_ids

  out = model(prompt_completion_ids, **model_kwargs)
  per_token_scores = out[0] if isinstance(out, tuple) else out
  # The model returns a tensor of shape [B, T, 1]. We squeeze the last
  # dimension to get a tensor of shape [B, T].
  per_token_scores = jnp.squeeze(per_token_scores, axis=-1)

  if stop_gradient:
    per_token_scores = jax.lax.stop_gradient(per_token_scores)

  return per_token_scores


def np_make_completion_mask(
    completion_ids: np.ndarray, eos_tok: int = 0
) -> np.ndarray:
  """Numpy version of make_completion_mask which executes on CPU.

  Args:
    completion_ids: Completion ids with shape [B, T].
    eos_tok: EOS token id.

  Returns:
    Completion mask.
  """
  is_eos = completion_ids == eos_tok
  seq_len = is_eos.shape[1]

  first_eos_idx = np.argmax(is_eos, axis=1)
  any_eos = np.any(is_eos, axis=1)
  eos_idx = np.where(any_eos, first_eos_idx, seq_len)
  sequence_indices = np.arange(seq_len)

  return (sequence_indices < eos_idx[:, None] + 1).astype(np.int32)


def make_completion_mask(
    completion_ids: jax.Array, eos_tok: int = 0
) -> jax.Array:
  """Create completion mask based on the EOS token.

  Args:
    completion_ids: Completion ids with shape [B, T].
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
  return (sequence_indices <= eos_idx[:, None]).astype(jnp.int32)


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


def aggregate_loss(
    per_token_loss: jax.Array,
    completion_mask: jax.Array,
    loss_agg_mode: str,
    **kwargs: Any,
) -> utils.WeightedMetric:
  """Aggregate loss based on the loss aggregation mode.

  Args:
      per_token_loss: Per token loss.[batch_size, sequence_len]
      completion_mask: Completion mask.[batch_size, sequence_len]
      loss_agg_mode: Loss aggregation mode.

  Returns:
      Aggregated loss.
  """

  per_token_loss = per_token_loss.astype(jnp.float32)

  if loss_agg_mode == "token-mean":
    # sum all the token loss, and average by total number of completion tokens
    # in the batch
    unreduced_sum = (per_token_loss * completion_mask).sum()
    denominator = completion_mask.sum()
    min_denom = 1.0
  elif loss_agg_mode == "sequence-mean-token-mean":
    seq_mask = completion_mask.sum(axis=-1)  # per-sequence token count
    seq_loss = ((per_token_loss * completion_mask).sum(axis=-1)) / jnp.clip(
        seq_mask, min=1.0
    )
    unreduced_sum = seq_loss.sum()
    denominator = per_token_loss.shape[0]
    min_denom = 1.0
  elif loss_agg_mode == "sequence-mean-token-scale":
    # Look up custom normalization factor, default to max response length.
    norm = _check_get_norm(kwargs, per_token_loss.shape[-1])

    # Scale by maximum response length instead of actual response length.
    seq_loss = (per_token_loss * completion_mask).sum(axis=-1) / jnp.clip(
        norm, min=1e-6
    )
    unreduced_sum = seq_loss.sum()
    denominator = per_token_loss.shape[0]
    min_denom = 1.0
  elif loss_agg_mode == "seq-mean-token-sum":
    # 1) sum token losses within each sequence
    # 2) average only across sequences that have at least one valid token
    seq_loss = (per_token_loss * completion_mask).sum(axis=-1)
    seq_mask = (completion_mask.sum(axis=-1) > 0).astype(jnp.float32)
    unreduced_sum = (seq_loss * seq_mask).sum()
    denominator = seq_mask.sum()
    min_denom = 1e-6
  elif loss_agg_mode == "sequence-mean-token-sum-norm":
    # Get custom normalization factor from kwargs, default to batch size.
    norm = _check_get_norm(kwargs, per_token_loss.shape[0])
    unreduced_sum = (per_token_loss * completion_mask).sum()
    denominator = norm
    min_denom = 1e-6
  else:
    raise ValueError(
        f"Unsupported loss aggregation mode: {loss_agg_mode}. Supported modes:"
        " 'token-mean', 'sequence-mean-token-mean',"
        " 'sequence-mean-token-scale', 'seq-mean-token-sum',"
        " 'sequence-mean-token-sum-norm'."
    )
  return utils.WeightedMetric(
      jnp.asarray(unreduced_sum, dtype=jnp.float32),
      jnp.asarray(denominator, dtype=jnp.float32),
      min_denom=min_denom,
  )


def compute_entropy_from_logits(logits: jax.Array) -> jax.Array:
  """Computes the entropy of a distribution given its logits.

  Args:
    logits: Logits as returned by the model. Of shape `[batch_size, seq_len,
      emb_dim]`.

  Returns:
    A JAX array of shape `[batch_size, seq_len]`, containing the entropy values.
  """
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  probs = jax.nn.softmax(log_probs)
  return -jnp.sum(probs * log_probs, axis=-1)


def _check_get_norm(
  arguments: dict[str, Any], default: float | int
) -> float:
  """Get custom normalization factor from kwargs with a default value.

  Args:
      arguments: The arguments dictionary.
      default: The default value to use if no 'norm' key is found.

  Returns:
      The normalization factor.

  Raises:
      ValueError: If the 'norm' key is present but has an invalid value or type.
  """
  norm = arguments.get("norm", float(default))
  if not isinstance(norm, (int, float)) or norm <= 0:
    raise ValueError(
        f"Invalid 'norm' value: {norm}. Must be a positive number."
    )
  return norm  # pyrefly: ignore[bad-return]
