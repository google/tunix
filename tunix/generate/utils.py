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
import gc
import logging
import math
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from flax import nnx
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
    if tok_id in tok_logprobs:
      extracted.append(tok_logprobs[tok_id].logprob)
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
  for keys, v in flat_state:
    path = '.'.join(str(key) for key in keys)
    mapped = False
    for src, (tgt, sharding) in mappings.items():
      regex = '^' + re.escape(tgt).replace('\\.\\*', r'\.(\d+)') + '$'
      matched = re.match(regex, path)
      if matched:
        # Extract wildcards if any
        wildcards = matched.groups()
        src_parts = []
        wc_index = 0
        for part in src.split('.'):
          if part == '*':
            src_parts.append(wildcards[wc_index])
            wc_index += 1
          else:
            src_parts.append(part)
        actual_src = '.'.join(src_parts)
        # Check if this is a scanned parameter (has 'layer' in sharding spec)
        if sharding and 'layer' in sharding:
          if actual_src not in new_flat_dict:
            new_flat_dict[actual_src] = ([], [], sharding)
          layer_number = int(matched.groups()[0])
          new_flat_dict[actual_src][0].append((layer_number, v))
          new_flat_dict[actual_src][1].append((layer_number, path))
        else:
          # Regular (non-scanned) parameter
          new_flat_dict[actual_src] = v, path, sharding

        mapped = True
        break
    # There are no mappings for rng related params.
    if not mapped:
      logging.warning('!!! No mapping for flat state: %s', path)

  # Sort layers
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
  if target_key != '':
    logging.debug('Applying transpose on %s', src_key)
    return jnp.transpose(val, transpose_keys[target_key])
  return val


def _reshape_attention(
    val: jnp.ndarray, tgt_shape: Tuple[int, ...], src_key: str
) -> jnp.ndarray:
  """Reshape attention tensors with special handling.

  Args:
      val: Value to reshape.
      tgt_shape: Target shape.
      src_key: Source key for error messages.

  Returns:
      Reshaped value.

  Raises:
      ShapeMismatchError: If reshaping is not possible.
  """
  if re.compile(r'layers\..*\.attn\.(q|k|v)_bias').match(src_key):
    new_shape = (tgt_shape[0], val.shape[0] // tgt_shape[0])
    logging.debug(
        'Reshaping attention bias on %s: %s -> %s',
        src_key,
        val.shape,
        new_shape,
    )
    return jnp.reshape(val, new_shape)
  ## here exists tgt_shape is (4096, 1024), but the val shape is (4096, 8, 128) case, so needs reshape
  if re.compile(r'layers\..*\.attn\.(q|k|v|o)_proj').match(
      src_key
  ) and math.prod(tgt_shape) == math.prod(val.shape):
    logging.debug(
        'Reshaping attention proj on %s: %s -> %s',
        src_key,
        val.shape,
        tgt_shape,
    )
    return jnp.reshape(val, tgt_shape)
  raise ShapeMismatchError(
      f'Rank mismatch for {src_key}: {val.shape} vs {tgt_shape}'
  )


def _align_shape(
    val: jnp.ndarray, tgt_shape: Tuple[int, ...], src_key: str
) -> jnp.ndarray:
  """Align source value shape to target shape through padding or repeating.

  Args:
      val: Source value.
      tgt_shape: Target shape.
      src_key: Source key for error messages.

  Returns:
      Shape-aligned value.

  Raises:
      ShapeMismatchError: If shapes cannot be aligned.
  """
  if val.shape == tgt_shape:
    return val

  # Handle rank mismatch
  if len(val.shape) != len(tgt_shape):
    return _reshape_attention(val, tgt_shape, src_key)

  original_shape = val.shape
  # Check if this is an attention weight that can be padded/repeated
  attention_patterns = [r'.*(q|k|v|o)_proj.*', r'.*(key|query|value|output).*']
  if not any(re.match(pattern, src_key) for pattern in attention_patterns):
    raise ShapeMismatchError(
        f'Shape mismatch for non-attention weight {src_key}: '
        f'{val.shape} vs {tgt_shape}. Padding/repetition only supported '
        'for attention weights.'
    )
  # Align each dimension
  pad_width = []
  repeat_ops = []
  for i, (src_dim, tgt_dim) in enumerate(zip(val.shape, tgt_shape)):
    if src_dim < tgt_dim:
      # For QKV, H is dim(-1); For O, H is dim(-2), same for Tunix and vLLM
      if i == len(val.shape) - 1 or (
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
  return jnp.pad(val, pad_width)


def _apply_dtype_cast(
    val: jnp.ndarray, tgt_dtype: jnp.dtype, src_key: str
) -> jnp.ndarray:

  if val.dtype != tgt_dtype:
    logging.warning(
        'Type mismatch on %s: %s -> %s',
        src_key,
        val.dtype,
        tgt_dtype,
    )
    return val.astype(tgt_dtype)
  return val


def transfer_state_with_mappings(
    src_state,
    dst_state,
    key_mappings,
    key_mapping_hook_fns=None,
    transpose_keys=None,
    reshard_fn=None,
):
  """Transfer state using mappings, with optional transpose and shard logic.

  Args:
    src_state: The source state to transfer from.
    dst_state: The destination state to transfer to.
    key_mappings: A dictionary defining how to map keys from the source state to
      the target state. The keys of the dictionary are the source keys, and the
      values are tuples containing the target key and the sharding information.
    key_mapping_hook_fns: A dictionary mapping keys to hook functions that
      modify the values before assignment. The hook fn will be called after the
      transpose operation if transpose were to be applied.
    transpose_keys: A dictionary defining which keys to transpose and the
      corresponding axes to transpose.
    reshard_fn: A function to shard the value.

  Returns:
    The target state with the transferred values.
  """
  # Get flat target state
  tgt_flat_list = dst_state.flat_state()
  # Build sharding dictionary if resharding is needed
  sharding_dict = None
  if reshard_fn:
    sharding_dict = {
        key: (
            tgt_params.value.sharding
            if hasattr(tgt_params, 'value')
            else tgt_params.sharding
        )
        for key, tgt_params in tgt_flat_list
    }

  # Build source-to-target mapping
  src_to_tgt_map = build_flat_dict(tgt_flat_list, key_mappings)

  # Unroll scanned layers and flatten source state
  unscanned_src_to_tgt_flat = _unroll_scanned_layers(src_state, src_to_tgt_map)

  # Transfer values with transformations
  for (flat_src_key, tgt_key), (
      val,
      tgt_param,
  ) in unscanned_src_to_tgt_flat.items():
    # Apply transpose if configured
    val = _apply_transpose(val, flat_src_key, transpose_keys)

    # Apply optional hook function
    if key_mapping_hook_fns and flat_src_key in key_mapping_hook_fns:
      val = key_mapping_hook_fns[flat_src_key](val)

    # Align shapes (padding/repeating as needed)
    val = _align_shape(val, tgt_param.value.shape, flat_src_key)

    # Cast to target dtype
    val = _apply_dtype_cast(val, tgt_param.value.dtype, flat_src_key)

    # Assign transformed value
    tgt_param.value = val

  # Clean up memory
  del unscanned_src_to_tgt_flat
  gc.collect()

  # Batch reshard and assign if resharding is configured
  if reshard_fn:
    tgt_flat_dict = {
        key: tgt_params.value if hasattr(tgt_params, 'value') else tgt_params
        for key, tgt_params in tgt_flat_list
    }
    resharded_values_flat_dict = reshard_fn(tgt_flat_dict, sharding_dict)

    for tgt_key, tgt_param in tgt_flat_list:
      assert (
          tgt_key in resharded_values_flat_dict
      ), f'Key {tgt_key} not in resharded values'
      if hasattr(tgt_param, 'value'):
        tgt_param.value = resharded_values_flat_dict[tgt_key]
      else:
        tgt_param = resharded_values_flat_dict[tgt_key]

  return dst_state.from_flat_path(tgt_flat_list)


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
