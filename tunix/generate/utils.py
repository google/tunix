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

from collections import abc
import functools
import gc
from absl import logging
import math
import re
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from flax import nnx
from flax import traverse_util
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


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
    x: np.ndarray,
    target_length: int,
    pad_value: int = 0,
    left=False,
    axis: int = 0,
) -> np.ndarray:
  """Pads a numpy array to a specified target length along a given axis.

  Args:
      x: The numpy array to pad.
      target_length: The desired length of the padded array.
      pad_value: The value to use for padding (default: 0).
      left: If True, add padding tokens to the left of the array.
      axis: The axis along which to pad (default: 0).

  Returns:
      A new numpy array that is padded to the target length along the specified
      axis. Returns original array if it is already longer than the target
      length.
  """
  length = x.shape[axis]
  if length >= target_length:
    return x

  padding_shape = list(x.shape)
  padding_shape[axis] = target_length - length
  padding = np.full(padding_shape, pad_value, dtype=x.dtype)

  if left:
    return np.concatenate([padding, x], axis=axis)
  else:
    return np.concatenate([x, padding], axis=axis)


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


def find_first_eos_idx(ids, eos_id: int | jax.Array):
  """Finds the index of the first EOS token."""
  assert ids.ndim == 1, f'ids should be a 1d array. Got: {ids.shape}'
  if isinstance(eos_id, int):
    eos_id = jnp.array([eos_id])
  mask = jnp.isin(ids, eos_id)
  first_idx = jnp.argmax(mask)
  is_eos_present = mask[first_idx]
  return jnp.where(is_eos_present, first_idx, ids.shape[0])


def np_find_first_non_pad_idx(ids: np.ndarray, pad_id: int) -> int:
  """Numpy version of find_first_non_pad_idx. Works on CPU arrays."""
  assert ids.ndim == 1, f'ids should be a 1d array. Got: {ids.shape}'
  mask = ids != pad_id
  return int(np.argmax(mask)) if mask.any() else 0


def np_find_first_eos_idx(
    ids: np.ndarray, eos_id: int | jax.Array,
) -> int:
  """Numpy version of find_first_eos_idx. Works on CPU arrays."""
  assert ids.ndim == 1, f'ids should be a 1d array. Got: {ids.shape}'
  if isinstance(eos_id, int):
    eos_id = np.array([eos_id])  # pyrefly: ignore[bad-assignment]
  mask = np.isin(ids, eos_id)
  return int(np.argmax(mask)) if mask.any() else len(ids)


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
    static_argnames=(
        'return_logits',
        'echo',
        'pad_value',
        'max_prompt_length',
        'max_total_length',
    ),
)
def padded_fill_tokens_and_logits(
    token_buffers: jax.Array,
    logits_buffers: jax.Array | None,
    return_logits: bool,
    echo: bool,
    pad_value: int,
    eos_value: int | jax.Array,
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
    eos_value: int | jax.Array,
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


def get_logprobs_from_vllm_output(
    token_ids: List[int],
    logprobs: List[Optional[Dict[int, Any]]],
) -> List[float]:
  """Extracts the log probs from the vLLM output."""
  if not logprobs or logprobs[0] is None:
    logging.debug('Logprobs are missing')
    return []

  assert len(logprobs) == len(token_ids), (
      f'log probs has {len(logprobs)} number of items !='
      f' {len(token_ids)} token ids'
  )

  extracted = []
  for tok_id, tok_logprobs in zip(token_ids, logprobs):
    if tok_id in tok_logprobs:  # pyrefly: ignore[not-iterable]
      extracted.append(tok_logprobs[tok_id].logprob)  # pyrefly: ignore[unsupported-operation]
    else:
      raise ValueError(
          f'The selected token id {tok_id} not in the return log probs list'
          f' {tok_logprobs}'
      )
  return extracted


def build_flat_dict(
    flat_state: Iterator[tuple[tuple[str, ...], nnx.State]],
    mappings: Dict[str, tuple[str, tuple[int, ...]]],
):
  """Build a new flat dictionary from the flat state using the provided mappings.

  Args:
    flat_state: A list of tuples, where each tuple contains the nested keys and
      the corresponding value.
    mappings: A dictionary defining how to map keys from the source state to the
      target state. The keys of the dictionary are the source keys, and the
      values are tuples containing the target key and the sharding information.

  Returns:
    A new flat dictionary with the mapped keys and values.
  """
  new_flat_dict = {}
  compiled_mappings = []

  # PRE-COMPILE MAPPINGS
  # Convert target string patterns into Python Regex objects for fast matching.
  for src, (tgt, sharding) in mappings.items():
    # Scenario A: The mapping already contains regex special characters (manual
    # filtering). The assumption is that `src` does not contain regex
    # characters like `()`; only `tgt` can contain them.
    # Example: 'layers.(0|2|4).*' used to select only even layers for MoE
    # interleaving.
    if any(char in tgt for char in ['|', '(', ')']):
      pattern = '^' + tgt + '$'
    else:
      # Scenario B: Standard wildcard mapping.
      # We escape special dots and replace '.*' with a capturing group '(\d+)'
      # to extract the layer index from the path.
      pattern = '^' + re.escape(tgt).replace('\\.\\*', r'\.(\d+)') + '$'
    compiled_mappings.append((src, re.compile(pattern), sharding))

  # ITERATE THROUGH ACTUAL PARAMETERS
  unmapped_paths = []
  for keys, v in flat_state:
    # Convert key tuple ('model', 'layers', '0') to string 'model.layers.0'
    path = '.'.join(str(key) for key in keys)
    mapped = False
    for src, regex, sharding in compiled_mappings:
      matched = regex.match(path)
      if matched:
        # Extract wildcards if any
        wildcards = matched.groups()

        # Reconstruct the internal name by filling '*' in the source string
        # with the captured wildcards from the external path.
        src_parts = []
        wc_index = 0
        for part in src.split('.'):
          if part == '*':
            src_parts.append(wildcards[wc_index])
            wc_index += 1
          else:
            src_parts.append(part)
        actual_src = '.'.join(src_parts)

        # HANDLE SCANNED VS REGULAR PARAMS
        # Scanned parameters have 'layer' in their sharding spec. This means we
        # stack multiple individual layer weights into one big array.
        if sharding and 'layer' in sharding:
          if actual_src not in new_flat_dict:
            new_flat_dict[actual_src] = ([], [], sharding)

          # Extract layer index from regex match for correct sorting.
          layer_number = int(wildcards[0]) if wildcards else 0
          new_flat_dict[actual_src][0].append((layer_number, v))
          new_flat_dict[actual_src][1].append((layer_number, path))
        else:
          # Regular (non-scanned) parameter
          new_flat_dict[actual_src] = v, path, sharding

        mapped = True
        break
    # There are no mappings for rng related params.
    if not mapped:
      unmapped_paths.append(path)

  if unmapped_paths:
    logging.warning('!!! No mapping for flat states: %s', unmapped_paths)

  # Sort layers based on layer index to ensure correct order.
  for key, (layers, paths, sharding) in new_flat_dict.items():
    if isinstance(layers, list):
      layers.sort(key=lambda x: x[0])
      paths.sort(key=lambda x: x[0])
      values = [v for _, v in layers]
      paths = [p for _, p in paths]
      new_flat_dict[key] = (values, paths, sharding)

  return new_flat_dict


class ShapeMismatchError(ValueError):
  """Raised when source and target shapes are incompatible."""

  pass


class MappingError(ValueError):
  """Raised when key mappings are invalid or missing."""

  pass


def _get_layer_axis_from_sharding_spec(sharding_spec) -> Optional[int]:
  """Returns index of the 'layer' axis in sharding_spec, or None if not found."""
  if isinstance(sharding_spec, (list, tuple)):
    for i, spec in enumerate(sharding_spec):
      if spec == 'layer':
        return i
  return None


def _unroll_scanned_layers(
    src_state: Any,
    src_to_tgt_map: Dict,
) -> Dict[Tuple[str, str], Tuple[Any, Any]]:
  """Unroll scanned layers from source state and map to target keys.

  Args:
      src_state: Source state to unroll.
      src_to_tgt_map: Mapping from flat source keys to (target_param,
        target_path, sharding_spec).

  Returns:
      Dictionary mapping (src_key, tgt_key) to (value, target_param).
  """

  unscanned_flat = {}

  for src_keys, src_val in src_state.flat_state():
    src_key = '.'.join(str(k) for k in src_keys)

    # Skip RNG parameters silently
    if 'rng' in src_key:
      logging.debug('Skipping RNG parameter: %s', src_key)
      continue

    # Validate mapping exists
    if src_key not in src_to_tgt_map:
      logging.error('No mapping for source key: %s', src_key)
      continue

    tgt_param, tgt_path, sharding_spec = src_to_tgt_map[src_key]

    # Check if this is a scanned layer that needs unrolling
    layer_axis = _get_layer_axis_from_sharding_spec(sharding_spec)

    if layer_axis is not None:
      # Unroll the scanned layer dimension
      num_layers = src_val.value.shape[layer_axis]
      for i in range(num_layers):
        idx = [slice(None)] * src_val.value.ndim
        idx[layer_axis] = i
        layer_val = src_val.value[tuple(idx)]
        layer_key = tgt_path[i]
        unscanned_flat[(src_key, layer_key)] = (layer_val, tgt_param[i])
    else:
      # No unrolling needed
      unscanned_flat[(src_key, tgt_path)] = (src_val.value, tgt_param)

  return unscanned_flat


def _apply_transpose(
    val: jnp.ndarray,
    src_key: str,
    transpose_keys: Optional[Dict[str, Tuple[int, ...]]],
    rollout_engine: Optional[str],
) -> jnp.ndarray:
  """Apply transpose operation if configured for this key."""
  if not transpose_keys:
    return val

  last_key = src_key.split('.')[-1]
  all_key = src_key
  target_key = ''
  if last_key in transpose_keys and 'lora' not in last_key:
    target_key = last_key
  elif all_key in transpose_keys and 'lora' not in all_key:
    target_key = all_key
  else:
    for k, _ in transpose_keys.items():
      if '*' in k:
        pattern = '^' + re.escape(k).replace('\\*', '.*') + '$'
        if re.match(pattern, all_key):
          target_key = k
          break

  # For LoRA
  # Note: The following codes takes effect in SGLangJAx rollout, and may not take effect in other rollout engine.

  if rollout_engine == 'sglang_jax' and 'lora' in all_key:
    for r_key in transpose_keys:
      if re.compile(rf'{r_key}').match(all_key):
        logging.debug('Applying LoRA transpose on %s', src_key)
        return jnp.transpose(val[None, :, :], transpose_keys[r_key])

  if target_key != '':
    logging.debug('Applying transpose on %s', src_key)
    return jnp.transpose(val, transpose_keys[target_key])

  return val


def _align_shape(
    val: jnp.ndarray,
    tgt_shape: Tuple[int, ...],
    src_key: str,
    rollout_engine: Optional[str] = None,
    **kwargs,
) -> jnp.ndarray:
  """Align source value shape to target shape through padding or repeating.

  This function attempts to align the shape of a source JAX array (`val`) to a
  target shape (`tgt_shape`). It supports alignment by:
  1.  Reshaping: If the product of dimensions matches, especially for attention
      biases and projections.
  2.  Padding/Repeating: For attention-related weights, it can pad the head
      dimension or repeat along the number of heads dimension.
  3.  Special Handling: Includes specific logic for 1-D KV biases in
      'sglang_jax' rollout.

  Args:
      val: Source value.
      tgt_shape: Target shape.
      src_key: Source key for error messages.
      rollout_engine: Optional string indicating the rollout engine, used for
        special-casing certain alignments (e.g., 'sglang_jax').
      **kwargs: Additional keyword arguments, potentially containing metadata
        like 'num_kv_heads' and 'head_dim' for specific alignment logic.

  Returns:
      Shape-aligned value.

  Raises:
      ShapeMismatchError: If shapes cannot be aligned.
  """
  if val.shape == tgt_shape:
    return val

  additional_reshape = False
  new_tgt_shape = tgt_shape
  # Handle rank mismatch
  if len(val.shape) != len(tgt_shape):
    if re.compile(r'layers\..*\.attn\.(q|k|v)_bias').match(src_key):
      if math.prod(tgt_shape) == math.prod(val.shape):
        new_shape = (tgt_shape[0], val.shape[0] // tgt_shape[0])
        logging.debug(
            'Reshaping attention bias on %s: %s -> %s',
            src_key,
            val.shape,
            new_shape,
        )
        return jnp.reshape(val, new_shape)
      else:
        # If target pads number of heads, we need to reshape and then pad, we
        # don't consider padding head dimensions here.
        # example cases: (256,) -> (8, 128)
        assert (
            val.shape[0] == kwargs['num_kv_heads'] * kwargs['head_dim']
            and tgt_shape[0] % kwargs['num_kv_heads'] == 0
            and tgt_shape[1] == kwargs['head_dim']
        ), (
            f'Unexpected attention bias shape: {val.shape} and target shape:'
            f' {tgt_shape}'
        )
        val = jnp.reshape(val, (kwargs['num_kv_heads'], kwargs['head_dim']))
        new_tgt_shape = tgt_shape

    elif re.compile(r'layers\..*\.attn\.(q|k|v|o)_proj').match(src_key):
      if math.prod(tgt_shape) == math.prod(val.shape):
        logging.debug(
            'Reshaping attention proj on %s: %s -> %s',
            src_key,
            val.shape,
            tgt_shape,
        )
        return jnp.reshape(val, tgt_shape)
      else:
        # need to reshape and then align each dim
        additional_reshape = True
        # Handle cases of mapping from (model_dim, num_head, head_dim) or
        # (model_dim, head_dim, num_head) to
        # (model_dim, num_head_dim * head_dim).
        assert len(val.shape) == 3 and len(tgt_shape) == 2, (
            f'Unexpected attention proj shape: {val.shape} and target shape:'
            f' {tgt_shape}'
        )
        if 'o_proj' in src_key:
          # for output proj, head dim is dim(-2)
          padded_dim = (val.shape[-2] + 127) // 128 * 128
          repeated_dim = tgt_shape[-1] // padded_dim
          new_tgt_shape = tgt_shape[:-1] + (padded_dim, repeated_dim)
        else:
          # for q/k/v proj, head dim is dim(-1)
          padded_dim = (val.shape[-1] + 127) // 128 * 128
          repeated_dim = tgt_shape[-1] // padded_dim
          new_tgt_shape = tgt_shape[:-1] + (repeated_dim, padded_dim)
    elif re.compile(r'layers\..*\.moe\.gating_einsum').match(src_key):
      tp_size = kwargs['tp_size']
      num_experts, expert_dim, embed_dim = val.shape[0], val.shape[2], val.shape[3]
      gate_chunks, up_chunks = val[:, 0, :, :], val[:, 1, :, :]
      chunk_size = expert_dim // tp_size
      padded_expert_chunk_dim = ((chunk_size + 127)//128)*128
      pad_amount = padded_expert_chunk_dim - chunk_size
      gate_chunks = gate_chunks.reshape(num_experts, tp_size, -1, embed_dim)
      up_chunks = up_chunks.reshape(num_experts, tp_size, -1, embed_dim)
      if pad_amount > 0:
        gate_chunks = jnp.pad(gate_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
        up_chunks = jnp.pad(up_chunks, ((0, 0), (0, 0), (0, pad_amount), (0, 0)))
      val_chunks = jnp.stack([gate_chunks, up_chunks], axis=2)
      val_chunks = val_chunks.reshape(num_experts, -1, embed_dim)
      val_chunks = val_chunks.transpose(0, 2, 1)
      return val_chunks
    else:
      raise ShapeMismatchError(
          f'Rank mismatch for {src_key}: {val.shape} vs {tgt_shape}'
      )
  elif re.compile(r'layers\..*\.attn\.(k|v)_bias').match(src_key):
    logging.debug(
        'Handling 1-D KV bias for %s in SGLangJAX rollout.', src_key
    )
    assert tgt_shape[0] > val.shape[0] and tgt_shape[0] % val.shape[0] == 0, (
        f'Unexpected attention bias shape: {val.shape} and target shape:'
        f' {tgt_shape}'
    )
    repeat_factor = tgt_shape[0] // val.shape[0]
    logging.debug(
        'Replicating 1-D KV bias on %s: %s -> %s (repeat x%d per head)',
        src_key,
        val.shape,
        tgt_shape,
        repeat_factor,
    )
    val_2d = jnp.reshape(val, (kwargs['num_kv_heads'], kwargs['head_dim']))
    val_2d = jnp.repeat(val_2d, repeat_factor, axis=0)
    return jnp.reshape(val_2d, tgt_shape)

  attention_patterns = [
      r'.*(q|k|v|o)_proj.*',
      r'.*(q|k|v|o)_bias.*',
      r'.*(key|query|value|output).*',
  ]
  if not any(re.match(pattern, src_key) for pattern in attention_patterns):
    raise ShapeMismatchError(
        f'Shape mismatch for non-attention weight {src_key}: '
        f'{val.shape} vs {tgt_shape}. Padding/repetition only supported '
        'for attention weights.'
    )

  original_shape = val.shape
  # Check if this is an attention weight that can be padded/repeated and
  # align on each dimension.
  pad_width = []
  repeat_ops = []
  for i, (src_dim, tgt_dim) in enumerate(zip(val.shape, new_tgt_shape)):
    if src_dim < tgt_dim:
      # For QKV, H is dim(-1); For O, H is dim(-2), same for Tunix and vLLM
      if ('o_proj' not in src_key and i == len(val.shape) - 1) or (
          'o_proj' in src_key and i == len(val.shape) - 2
      ):
        # Head dimension: pad with zeros
        pad_width.append((0, tgt_dim - src_dim))
      else:
        # Num heads dimension: repeat weights
        repeat_factor = tgt_dim // src_dim
        if tgt_dim % src_dim != 0:
          raise ShapeMismatchError(
              f'Target dimension {tgt_dim} is not divisible by source '
              f'dimension {src_dim} for {src_key}'
          )
        repeat_ops.append((i, repeat_factor))
        pad_width.append((0, 0))
    elif src_dim > tgt_dim:
      raise ShapeMismatchError(
          f'Cannot shrink dimension {i} for {src_key}: {src_dim} -> {tgt_dim}'
      )
    else:
      pad_width.append((0, 0))

  logging.info(
      'Resolved shape mismatch on %s: %s -> %s',
      src_key,
      original_shape,
      tgt_shape,
  )

  for axis, repeat_factor in repeat_ops:
    val = jnp.repeat(val, repeat_factor, axis=axis)
  val = jnp.pad(val, pad_width)

  if additional_reshape:
    assert math.prod(val.shape) == math.prod(
        tgt_shape
    ), f'After align, shape mismatch on {src_key}: {val.shape} vs {tgt_shape}'
    val = jnp.reshape(val, tgt_shape)
  return val


def _apply_dtype_cast(
    val: jax.Array | np.ndarray, tgt_dtype: jnp.dtype, src_key: str
) -> jax.Array | np.ndarray:
  if val.dtype != tgt_dtype:
    logging.log_first_n(
        logging.WARNING,
        'Type mismatch on %s: %s -> %s',
        1,
        src_key,
        val.dtype,
        tgt_dtype,
    )
    return val.astype(tgt_dtype)
  return val


def _sync_tied_lm_head_if_needed(
    tgt_flat_list: List[Tuple[Tuple[str, ...], Any]],
    transferred_target_keys: set[str],
) -> None:
  """Mirrors embed weights into lm_head when the target implies a tied head.

  Some JAX/vLLM state layouts materialize `lm_head` as a separate destination
  leaf even when the module graph ties it to `embed.embedding`. If the mapping
  updates only `embed.embedding`, keep `lm_head` in sync unless `lm_head` was
  actually transferred from the source state.

  Args:
    tgt_flat_list: A list of tuples, where each tuple contains the nested keys
      and the corresponding target parameter.
    transferred_target_keys: Target keys that were actually written during the
      transfer loop.
  """
  if any(key.endswith('lm_head') for key in transferred_target_keys):
    return

  embed_param = None
  lm_head_param = None
  for flat_key, tgt_param in tgt_flat_list:
    path = '.'.join(str(k) for k in flat_key)
    if path.endswith(('embedding', 'embed_tokens.weight')):
      embed_param = tgt_param
    elif path.endswith(('lm_head', 'lm_head.weight')):
      lm_head_param = tgt_param

  if embed_param is None or lm_head_param is None:
    return
  if not hasattr(embed_param, 'value') or not hasattr(lm_head_param, 'value'):
    return
  if embed_param.value.shape != lm_head_param.value.shape:
    return

  lm_head_param.value = embed_param.value


def resolve_parallelism_sizes(
    mesh: jax.sharding.Mesh,
    tensor_parallel_size: int = -1,
    data_parallel_size: int = -1,
    expert_parallel_size: int = 1,
) -> tuple[int, int, int]:
  """Resolves tensor, data, and expert parallelism sizes from the mesh.

  Any size passed as -1 is inferred from the total number of mesh devices and
  the other sizes. Raises ValueError if the mesh size is not divisible by
  expert_parallel_size.

  Args:
    mesh: The JAX device mesh.
    tensor_parallel_size: Desired tensor parallelism degree, or -1 to infer.
    data_parallel_size: Desired data parallelism degree, or -1 to infer.
    expert_parallel_size: Desired expert parallelism degree.

  Returns:
    A tuple of (tensor_parallel_size, data_parallel_size, expert_parallel_size).
  """
  total_mesh_devices = math.prod(mesh.shape.values())

  if total_mesh_devices % expert_parallel_size != 0:
    raise ValueError(
        f"Total mesh devices ({total_mesh_devices}) must be divisible by"
        f" expert_parallel_size ({expert_parallel_size})."
    )

  if tensor_parallel_size == -1 and data_parallel_size == -1:
    tensor_parallel_size = total_mesh_devices // expert_parallel_size
    data_parallel_size = 1
  elif tensor_parallel_size == -1:
    tensor_parallel_size = (
        total_mesh_devices // (data_parallel_size * expert_parallel_size)
    )
  elif data_parallel_size == -1:
    data_parallel_size = (
        total_mesh_devices // (tensor_parallel_size * expert_parallel_size)
    )

  return tensor_parallel_size, data_parallel_size, expert_parallel_size


def verify_state_closeness(golden_state, state, atol=1e-2):
  """Check if the golden NNX state is close to the other NNX state.

  Args:
    golden_state: The golden NNX state.
    state: The NNX state to compare with the golden state.
    atol: The absolute tolerance value for comparing weights.

  Returns:
    True if all weights have the same values within the specified tolerance
  """
  golden_state_flatten = {
      '.'.join(str(key) for key in keys): v
      for keys, v in golden_state.flat_state()
  }

  state_flatten = {
      '.'.join(str(key) for key in keys): v for keys, v in state.flat_state()
  }

  # Check that keys match
  if golden_state_flatten.keys() != state_flatten.keys():
    missing_keys = set(golden_state_flatten.keys()) - set(state_flatten.keys())
    extra_keys = set(state_flatten.keys()) - set(golden_state_flatten.keys())
    logging.info('Keys do not match.')
    logging.info('Missing keys: %s', missing_keys)
    logging.info('Extra keys: %s', extra_keys)
    return False

  # Check that weights match
  matched = True
  for key in golden_state_flatten.keys():

    if golden_state_flatten[key].value.shape != state_flatten[key].value.shape:
      logging.info(
          'Shape mismatch for key %s: golden %s, loaded %s',
          key,
          golden_state_flatten[key].value.shape,
          state_flatten[key].value.shape,
      )
      matched = False
      continue

    if not jax.numpy.allclose(
        golden_state_flatten[key].value, state_flatten[key].value, atol=atol
    ):
      logging.info('Weights for key %s do not match.', key)
      logging.info(
          'Golden state: %s', golden_state_flatten[key].value.ravel()[:10]
      )
      logging.info('Loaded state: %s', state_flatten[key].value.ravel()[:10])
      matched = False
  return matched

def _collect_src_buffer_ids(
    src_flat: Mapping[Tuple[str, ...], jax.Array | np.ndarray | nnx.Variable],
) -> Optional[set[int]]:
  """Collects physical device buffer pointers for arrays in src_flat.

  Used to detect when a target jax.Array shares its underlying buffer with a
  source array — Python identity (`is`) is insufficient because two distinct
  jax.Array wrappers can back the same physical shard (e.g. when source slices
  come from a scanned tensor that also backs another spec entry).
  """
  ids: set[int] = set()
  for v in src_flat.values():
    arr = v.value if hasattr(v, 'value') else v
    if not hasattr(arr, 'addressable_shards'):
      continue
    for shard in arr.addressable_shards:
      try:
        ids.add(shard.data.unsafe_buffer_pointer())
      except jax.errors.JaxRuntimeError as e:
        if "PjRt-compatible backend only" in str(e):
          # Backend doesn't support unsafe pointers (e.g., disaggregated Pathways setup).
          # Fast-fail and return None to signal that aliasing checks should be skipped.
          return None
        raise e  # Do not swallow unrelated JAX runtime errors
      except Exception:  # pylint: disable=broad-except
        # Fallback for non-JAX runtime errors from the original implementation
        pass
  return ids


def _delete_target_buffers(
    spec_flat: Mapping[Tuple[str, ...], jax.Array | np.ndarray | nnx.Variable],
    src_flat: Mapping[Tuple[str, ...], jax.Array | np.ndarray | nnx.Variable],
) -> None:
  """Deletes target arrays in spec_flat that don't alias any source shard."""
  # This will be a set() of IDs, or None if we are on Pathways
  src_buffer_ids = _collect_src_buffer_ids(src_flat)
  for tgt_val in spec_flat.values():
    tgt_arr = tgt_val.value if hasattr(tgt_val, 'value') else tgt_val
    # Skip if the array cannot be deleted or is already deleted
    if not hasattr(tgt_arr, 'delete') or getattr(
        tgt_arr, 'is_deleted', lambda: False
    )():
      continue
    # Check for aliasing only if the backend supports buffer pointer tracking
    if src_buffer_ids is not None and hasattr(tgt_arr, 'addressable_shards'):
      aliases_source = any(
          shard.data.unsafe_buffer_pointer() in src_buffer_ids
          for shard in tgt_arr.addressable_shards
      )
      if aliases_source:
        continue
    tgt_arr.delete()


def _snapshot_dst_sharding(
    arr: jax.Array | np.ndarray,
) -> jax.sharding.Sharding:
  """Snapshots a destination sharding leaf for reshard_fn's target tree.

  Captured *before* any potential `.delete()` on `arr` so the caller never
  needs to dereference a deleted jax.Array later. `reshard_pytree`'s
  `_get_dst_sharding` accepts `NamedSharding` / `SingleDeviceSharding` leaves
  directly, so for those we return the existing sharding object (no rebuild).
  """
  if isinstance(
      arr, (jax.sharding.NamedSharding, jax.sharding.SingleDeviceSharding)
  ):
    return arr

  assert hasattr(arr, 'sharding'), f'Expected array with sharding, got {type(arr)}'
  s = arr.sharding

  if isinstance(
      s, (jax.sharding.NamedSharding, jax.sharding.SingleDeviceSharding)
  ):
    return s
  return jax.sharding.NamedSharding(s.mesh, s.spec, memory_kind=s.memory_kind)  # type: ignore[attr-defined]


def _reshard_in_chunks(
    src_flat: Dict[Tuple[str, ...], jax.Array | np.ndarray],
    spec_flat: Dict[Tuple[str, ...], jax.Array | np.ndarray | nnx.Variable],
    reshard_fn: Callable[..., Mapping[str, Any]],
    chunk_size: int,
    delete_spec_buffers: bool = False,
) -> Dict[Tuple[str, ...], jax.Array | np.ndarray]:
  """Reshards a flat weight dict in sequential chunks to reduce peak HBM pressure.

  Instead of issuing one large jax.device_put for the entire model, this helper
  splits the flat key-value dict into groups of `chunk_size` keys and reshards
  each group independently. Between groups it calls jax.block_until_ready() so
  that the XLA allocator can reclaim the source buffers before committing the
  next chunk, keeping the peak contiguous allocation requirement proportional to
  chunk_size rather than the full model size.

  Args:
    src_flat: Flat dict mapping key tuples to source JAX arrays.
    spec_flat: Flat dict mapping the same key tuples to target-sharded arrays
      (used by reshard_fn to determine destination shardings).
    reshard_fn: Callable with the same signature as reshard_pytree, i.e.
      reshard_fn(source=<nested dict>, target=<nested dict>).
    chunk_size: Maximum number of flat keys to process per reshard call.
    delete_spec_buffers: Whether to delete buffers in the destination spec
      immediately before they are overwritten by resharded chunks.

  Returns:
    A flat dict with the same keys as src_flat, containing resharded arrays.
  """
  keys = list(src_flat.keys())
  resharded: Dict[Tuple[str, ...], jax.Array | np.ndarray] = {}
  for start in range(0, len(keys), chunk_size):
    chunk_keys = keys[start : start + chunk_size]
    chunk_src_flat = {}
    chunk_spec_flat = {}
    chunk_dst_shardings_flat = {}
    for k in chunk_keys:
      src_val = src_flat.pop(k)
      tgt_val = spec_flat[k]
      chunk_src_flat[k] = src_val
      chunk_spec_flat[k] = tgt_val
      tgt_arr = tgt_val.value if hasattr(tgt_val, 'value') else tgt_val
      chunk_dst_shardings_flat[k] = _snapshot_dst_sharding(tgt_arr)

    if delete_spec_buffers:
      _delete_target_buffers(chunk_spec_flat, chunk_src_flat)

    chunk_src = traverse_util.unflatten_dict(chunk_src_flat)
    chunk_dst_shardings = traverse_util.unflatten_dict(chunk_dst_shardings_flat)
    chunk_resharded = reshard_fn(source=chunk_src, target=chunk_dst_shardings)
    jax.block_until_ready(chunk_resharded)
    resharded.update(traverse_util.flatten_dict(chunk_resharded))

    del (  # pyrefly: ignore[unsupported-delete]
        chunk_src,
        chunk_dst_shardings,
        chunk_resharded,
        chunk_src_flat,
        chunk_spec_flat,
        chunk_dst_shardings_flat,
    )
  return resharded

