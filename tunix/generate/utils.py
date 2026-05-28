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
  if target_key != '':
    logging.debug('Applying transpose on %s', src_key)
    return jnp.transpose(val, transpose_keys[target_key])

  # For LoRA
  # Note: The following codes takes effect in SGLangJAx rollout, and may not take effect in other rollout engine.

  if rollout_engine == 'sglang_jax' and 'lora' in all_key:
    for r_key in transpose_keys:
      if re.compile(rf'{r_key}').match(all_key):
        logging.debug('Applying LoRA transpose on %s', src_key)
        return jnp.transpose(val[None, :, :], transpose_keys[r_key])
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

    elif (
        re.compile(r'.*(per_layer_input_embedding|per_layer_model_projection\.w)').match(src_key)
        and len(val.shape) == 3
        and len(tgt_shape) == 2
        and math.prod(tgt_shape) == math.prod(val.shape)
    ):
      logging.debug(
          'Reshaping 3D tensor on %s: %s -> %s',
          src_key,
          val.shape,
          tgt_shape,
      )
      return jnp.reshape(val, tgt_shape)

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


def transfer_state_with_mappings(
    src_state,
    dst_state,
    key_mappings,
    key_mapping_hook_fns=None,
    transpose_keys=None,
    reshard_fn=None,
    rollout_engine=None,
    **kwargs,
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
    rollout_engine: The name of the rollout engine being used.
    **kwargs: Additional keyword arguments.

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
  transferred_target_keys = set()

  # Transfer values with transformations
  for (flat_src_key, flat_tgt_key), (
      val,
      tgt_param,
  ) in unscanned_src_to_tgt_flat.items():
    # Apply transpose if configured
    val = _apply_transpose(val, flat_src_key, transpose_keys, rollout_engine)

    # Apply optional hook function
    if key_mapping_hook_fns and flat_src_key in key_mapping_hook_fns:
      val = key_mapping_hook_fns[flat_src_key](val)

    # Align shapes (padding/repeating as needed)
    val = _align_shape(
        val, tgt_param.value.shape, flat_src_key, rollout_engine, **kwargs
    )

    # Cast to target dtype
    val = _apply_dtype_cast(val, tgt_param.value.dtype, flat_src_key)

    # Assign transformed value
    tgt_param.value = val
    transferred_target_keys.add(flat_tgt_key)

  # Target rollout engine might have different implementation and have materialized lm_head
  _sync_tied_lm_head_if_needed(tgt_flat_list, transferred_target_keys)

  # Clean up memory
  del unscanned_src_to_tgt_flat
  gc.collect()

  # Batch reshard and assign if resharding is configured
  if reshard_fn:
    tgt_flat_dict = {
        key: tgt_params.value if hasattr(tgt_params, 'value') else tgt_params
        for key, tgt_params in tgt_flat_list
    }
    if kwargs.get('reshard_chunk_size', None) is not None:
      resharded_values_flat_dict = _reshard_in_chunks(
          src_flat=tgt_flat_dict,
          spec_flat=sharding_dict,
          reshard_fn=reshard_fn,
          chunk_size=kwargs['reshard_chunk_size'],
          delete_spec_buffers=kwargs.get('delete_dst_buffers', False),
      )
    else:
      if kwargs.get('delete_dst_buffers', False):
        _delete_target_buffers(sharding_dict, tgt_flat_dict)
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


def _shapes_are_repeatable(
    candidate_shape: tuple[int, ...],
    tgt_shape: tuple[int, ...],
) -> bool:
  """Returns True if candidate_shape can be repeated to match tgt_shape."""
  if len(candidate_shape) != len(tgt_shape):
    return False

  for s, t in zip(candidate_shape, tgt_shape):
    if s > t or t % s != 0:
      return False
  return True


def _unstack_scanned_param(
    src_val: jax.Array | np.ndarray,
    tgt_val: jax.Array | np.ndarray,
    key_path: str,
    scan_axis: Optional[int] = None,
) -> Tuple[jax.Array | np.ndarray, ...]:
  """Unstacks a scanned parameter by moving the scan axis to 0.

  This helper unstacks a scanned array at the specified scan_axis. When scan_axis
  is provided, it transposes that axis to position 0 and unstacks it. This is used
  when transferring weights from a scanned representation (e.g., MaxText) to an
  unrolled one (e.g., vLLM).

  Args:
    src_val: The source array (scanned) to slice from.
    tgt_val: The target array whose shape we want to match.
    key_path: The dot-separated path to the parameter for debugging.
    scan_axis: The axis containing the scanned dimension. If None, attempts to
      auto-detect it for backward compatibility.

  Returns:
      A tuple of unstacked arrays, or a tuple containing just the original src_val
      if unstacking fails or is unnecessary.
  """
  if not (hasattr(src_val, 'shape') and hasattr(tgt_val, 'shape')):
    return (src_val,)

  src_shape = src_val.shape
  tgt_shape = tgt_val.shape

  if src_shape == tgt_shape:
    return (src_val,)

  if len(src_shape) == len(tgt_shape) + 1:
    # If scan_axis not provided, try to detect it
    if scan_axis is None:
      for i in range(len(src_shape)):
        candidate = src_shape[:i] + src_shape[i + 1 :]
        if _shapes_are_repeatable(candidate, tgt_shape):
          scan_axis = i
          break
    
    if scan_axis is not None:
      # Transpose the scanned axis to the 0th position
      if scan_axis != 0:
        perm = (scan_axis,) + tuple(i for i in range(len(src_shape)) if i != scan_axis)
        if hasattr(src_val, 'transpose'):
          src_val = src_val.transpose(perm)
        elif isinstance(src_val, np.ndarray):
          src_val = np.transpose(src_val, perm)

      # Unstack along the 0th axis
      # Handling JAX version differences where unstack might be under jnp
      try:
        if hasattr(jax, 'unstack'):
          return jax.unstack(src_val)
        elif hasattr(jnp, 'unstack'):
          return jnp.unstack(src_val)
        else:
           # Fallback for older JAX versions
          return [src_val[i] for i in range(src_val.shape[0])]
      except Exception as e:
        logging.debug(
            "Failed to unstack parameter '%s'. Error: %s. Using original.",
            key_path, e
        )
        return (src_val,)
    else:
      logging.warning(
          "Shape mismatch in scanned param '%s'. Src: %s, Tgt: %s. Cannot"
          ' determine scan axis.',
          key_path, src_shape, tgt_shape,
      )

  return (src_val,)


_MOE_MLP_WEIGHTS = frozenset({'wi', 'wi_0', 'wi_1'})


def _partition_size(
    partition: Optional[str | Tuple[str, ...]],
    mesh: jax.sharding.Mesh,
) -> int:
  """Total mesh-axis size used to shard a single tensor axis."""
  if partition is None:
    return 1
  names = (partition,) if isinstance(partition, str) else tuple(partition)
  size = 1
  for n in names:
    size *= mesh.shape[n]
  return size


def _spec_at_axis(
    sharding: Optional[jax.sharding.Sharding],
    axis: int,
) -> Optional[str | Tuple[str, ...]]:
  """Returns the PartitionSpec entry at the given axis, or None if absent."""
  if not isinstance(sharding, jax.sharding.NamedSharding):
    return None
  spec = sharding.spec
  return spec[axis] if axis < len(spec) else None


def _get_n_shards(arr: jax.Array | np.ndarray, axis: int) -> int:
  """Returns the number of shards for a given axis of an array."""
  sharding = getattr(arr, 'sharding', None)
  if isinstance(sharding, jax.sharding.NamedSharding):
    return _partition_size(_spec_at_axis(sharding, axis), sharding.mesh)
  return 1


@functools.partial(jax.jit, static_argnames=('pad_specs',))
def _jit_zero_pad_axes(arr, pad_specs):
  """Shard-aware tail-pad on multiple axes within a single JIT.

  `pad_specs` is a tuple of `(axis, n_shards, per_shard_extra)`; entries
  with `per_shard_extra == 0` are no-ops. Composing every padded axis
  inside one trace lets XLA fuse them with the surrounding reshape ops
  rather than launching each as a separate eager primitive.
  """
  out = arr
  for axis, n_shards, per_shard_extra in pad_specs:
    if per_shard_extra <= 0:
      continue
    src_dim = out.shape[axis]
    src_chunk_size = src_dim // n_shards
    split_shape = list(out.shape)
    split_shape.insert(axis + 1, src_chunk_size)
    split_shape[axis] = n_shards
    arr_split = out.reshape(split_shape)
    pad_width = [(0, 0)] * arr_split.ndim
    pad_width[axis + 1] = (0, per_shard_extra)
    arr_padded = jnp.pad(arr_split, pad_width)
    final_shape = list(out.shape)
    final_shape[axis] = src_dim + per_shard_extra * n_shards
    out = arr_padded.reshape(final_shape)
  return out


@functools.partial(jax.jit, static_argnames=('repeats',))
def _jit_repeat_axes(arr, repeats):
  """Apply `jnp.repeat` on multiple axes within a single JIT.

  `repeats` is a tuple of `(axis, count)`. One trace covers every
  repeated axis so XLA can fuse them rather than dispatching per axis.
  """
  out = arr
  for axis, count in repeats:
    out = jnp.repeat(out, count, axis=axis)
  return out


def _align_per_axis(
    arr: jax.Array | np.ndarray,
    tgt_shape: Tuple[int, ...],
    tgt_sharding: Optional[jax.sharding.Sharding],
    key_path: str,
) -> jax.Array | np.ndarray:
  """Aligns `arr` to `tgt_shape` via either pure-repeat or pure-zero_pad.

  Each tensor needs exactly one transform mode in practice:
    * MoE linear weights (`wi`, `wi_0`, `wi_1`, `wo`) → `zero_pad` on
      every mismatched axis (TPU-lane / GMM_v2 alignment).
    * Anything else (attention QKV/O projections, biases, …) → `repeat`
      on every mismatched axis (e.g. KV-head expansion).
  KV-head expansion and MoE-dim padding never co-occur on the same
  tensor, so we classify by leaf key once and dispatch the whole
  transform to a single JIT compiled helper. That collapses what used
  to be N eager primitive launches into one fused XLA program — the hot
  path here is bulk alignment of scanned MoE weights, where eager
  dispatch was costing tens of seconds per tensor.
  """
  if not hasattr(arr, 'shape'):
    return arr
  if arr.shape == tgt_shape:
    return arr
  if len(arr.shape) != len(tgt_shape):
    raise ShapeMismatchError(
        f"Rank mismatch for {key_path}: src={arr.shape} vs tgt={tgt_shape}"
    )

  mismatches = []
  for axis, (s, t) in enumerate(zip(arr.shape, tgt_shape)):
    if s == t:
      continue
    if t < s:
      raise ShapeMismatchError(
          f"Cannot shrink axis {axis} for {key_path}: src={s} -> tgt={t}"
      )
    mismatches.append((axis, s, t))
  if not mismatches:
    return arr

  last_key = key_path.split('.')[-1]
  if last_key in _MOE_MLP_WEIGHTS:
    if isinstance(tgt_sharding, jax.sharding.NamedSharding):
      mesh = tgt_sharding.mesh
      pad_specs = []
      for axis, s, t in mismatches:
        n_shards = _partition_size(_spec_at_axis(tgt_sharding, axis), mesh)
        if t % n_shards != 0:
          raise ValueError(
              f"Target dimension {t} on axis {axis} for {key_path} is not "
              f"divisible by n_shards={n_shards}; the target shape itself "
              f"is misconfigured for the requested sharding."
          )
        if (t - s) % n_shards != 0 or s % n_shards != 0:
          raise ValueError(
              f"Cannot interleave pad axis {axis} for {key_path}: src_dim "
              f"({s}) or extra ({t - s}) is not cleanly divisible by "
              f"n_shards ({n_shards}). Ensure the source tensor is evenly "
              f"partitionable."
          )
        pad_specs.append((axis, n_shards, (t - s) // n_shards))
    else:
      pad_specs = [(axis, 1, t - s) for axis, s, t in mismatches]
    return _jit_zero_pad_axes(arr, tuple(pad_specs))

  repeats = []
  for axis, s, t in mismatches:
    if t % s != 0:
      raise ShapeMismatchError(
          f"Cannot align axis {axis} for {key_path}: src={s} -> tgt={t} "
          f"is not an integer multiple and the key is not a recognized "
          f"MoE pattern."
      )
    repeats.append((axis, t // s))
  return _jit_repeat_axes(arr, tuple(repeats))


def _interleave_moe_weights(
    wi_0: jax.Array | np.ndarray,
    wi_1: jax.Array | np.ndarray,
    tgt_shape: Tuple[int, ...],
    n_shards: int,
    axis: Optional[int] = None,
) -> jax.Array | np.ndarray:
  """Interleaves wi_0 and wi_1 per-shard into a single tensor."""
  if axis is None:
    axis = len(tgt_shape) - 1
    
  target_half_dim = tgt_shape[axis] // 2

  def _pad_and_chunk(arr):
    current_total_size = arr.shape[axis]
    chunk_size = current_total_size // n_shards
    target_chunk_size = target_half_dim // n_shards

    # Safely reshape to expose per-shard chunk without assuming the last axis
    new_shape = list(arr.shape)
    new_shape[axis] = n_shards
    new_shape.insert(axis + 1, chunk_size)
    arr_reshaped = arr.reshape(new_shape)

    pad_amount = target_chunk_size - chunk_size
    if pad_amount > 0:
      pad_widths = [(0, 0)] * arr_reshaped.ndim
      pad_widths[axis + 1] = (0, pad_amount)
      arr_reshaped = jnp.pad(arr_reshaped, pad_widths)
    return arr_reshaped

  p_wi_0 = _pad_and_chunk(wi_0)
  p_wi_1 = _pad_and_chunk(wi_1)

  # Interleave along the chunked dimension
  combined = jnp.concatenate([p_wi_0, p_wi_1], axis=axis + 1)
  return combined.reshape(tgt_shape)


def _align_to_model_shape(
    src_val: jax.Array | np.ndarray,
    tgt_val: jax.Array | np.ndarray,
    key_path: str,
) -> jax.Array | np.ndarray:
  """Aligns src_val to tgt_val's shape via per-axis repeat / zero-pad.

  Thin wrapper around `_align_per_axis` that pulls the target's sharding off
  the target leaf. The per-axis helper is what actually decides repeat vs
  zero-pad per axis and chooses between per-shard and global padding.
  """
  if not (hasattr(src_val, 'shape') and hasattr(tgt_val, 'shape')):
    return src_val
  if src_val.shape == tgt_val.shape:
    return src_val

  tgt_sharding = getattr(tgt_val, 'sharding', None)
  return _align_per_axis(src_val, tgt_val.shape, tgt_sharding, key_path)


def _bulk_align_and_unstack(
    arr: jax.Array | np.ndarray,
    scan_axis: int,
    per_layer_tgt_val: jax.Array | np.ndarray,
    key_path: str,
) -> Tuple[jax.Array | np.ndarray, ...]:
  """Applies per-axis alignment on a scanned tensor, then unstacks.

  Operates on the FULL scanned tensor (one bulk repeat / zero-pad per axis,
  performed under JIT so XLA can fuse the pad with surrounding ops), then
  emits `num_layers` per-layer slices. Replaces the prior pattern of unstack
  → align-each-layer, which created N intermediate allocations and N small
  ops.

  The per-layer target's logical axes map to the scanned tensor's axes by
  inserting the scan axis: `scan_padded_axis = a if a < scan_axis else a + 1`.

  Args:
    arr: The full scanned source tensor (shape includes `num_layers` at
      `scan_axis`).
    scan_axis: Axis at which `num_layers` lives in `arr`.
    per_layer_tgt_val: A target leaf — its `.shape`, `.sharding`, and dtype
      drive the alignment policy.
    key_path: Dot-separated source path for diagnostics.

  Returns:
    A tuple of `num_layers` per-layer arrays at the per-layer target shape.
  """
  per_layer_shape = per_layer_tgt_val.shape
  scanned_tgt_shape = (
      per_layer_shape[:scan_axis]
      + (arr.shape[scan_axis],)
      + per_layer_shape[scan_axis:]
  )
  scanned_tgt_sharding = _scanned_sharding_from_per_layer(
      getattr(per_layer_tgt_val, 'sharding', None), scan_axis
  )

  if arr.shape == scanned_tgt_shape:
    return tuple(jnp.unstack(arr, axis=scan_axis))

  aligned = _align_per_axis(
      arr, scanned_tgt_shape, scanned_tgt_sharding, key_path
  )
  return tuple(jnp.unstack(aligned, axis=scan_axis))


def _scanned_sharding_from_per_layer(
    per_layer_sharding: Optional[jax.sharding.Sharding],
    scan_axis: int,
) -> Optional[jax.sharding.NamedSharding]:
  """Builds a scanned `NamedSharding` from a per-layer one by inserting
  `PartitionSpec(None)` at `scan_axis`. Returns None if input isn't a
  NamedSharding."""
  if not isinstance(per_layer_sharding, jax.sharding.NamedSharding):
    return None
  spec = list(per_layer_sharding.spec)
  spec.insert(scan_axis, None)
  return jax.sharding.NamedSharding(
      per_layer_sharding.mesh,
      jax.sharding.PartitionSpec(*spec),
      memory_kind=per_layer_sharding.memory_kind,
  )


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def _jit_fuse_and_unstack_moe(
    wi_0: jax.Array | np.ndarray,
    wi_1: jax.Array | np.ndarray,
    scan_axis: int,
    num_layers: int,
    n_shards: int,
    tgt_shape: Tuple[int, ...],
    scan_padded_axis: int,
    tgt_padded_axis: int,
) -> Tuple[jax.Array | np.ndarray, ...]:
  """Fuses wi_0/wi_1 along the padded axis, then unstacks along scan_axis.

  By combining concat and unstack under jax.jit, XLA fuses both ops and
  avoids materializing the full concatenated intermediate on device.

  Args:
    wi_0: First MoE gate weight; per-layer layout matches `tgt_shape`
      (with the padded dim halved), with `num_layers` inserted at `scan_axis`.
    wi_1: Second MoE gate weight, same layout as `wi_0`.
    scan_axis: Axis at which `num_layers` is stacked in `wi_0` / `wi_1`.
    num_layers: Number of layers (== `wi_0.shape[scan_axis]`).
    n_shards: Mesh shards along the per-layer fused/padded axis.
    tgt_shape: Per-layer fused target shape (fused mlp dim on `tgt_padded_axis`).
    scan_padded_axis: Position of the padded axis in the scanned layout.
    tgt_padded_axis: Position of the padded axis in the per-layer layout.

  Returns:
    Tuple of `num_layers` per-layer arrays, each with shape `tgt_shape`.
  """
  del num_layers  # Only used to make this a static arg for JIT cache keying.
  fused_shape = list(wi_0.shape)
  fused_shape[scan_padded_axis] = tgt_shape[tgt_padded_axis]

  fused = _interleave_moe_weights(
      wi_0, wi_1, tuple(fused_shape), n_shards, axis=scan_padded_axis
  )
  return jnp.unstack(fused, axis=scan_axis)


def _fuse_moe_weights(
    src_flat: Dict[Tuple[str, ...], jax.Array | np.ndarray],
    tgt_flat: Dict[Tuple[str, ...], jax.Array | np.ndarray],
) -> Dict[Tuple[str, ...], jax.Array | np.ndarray]:
  """Fuses unscanned wi_0/wi_1 into wi for unscanned-fused targets.

  Only catches the case where source and target share the same prefix (e.g.
  src `('layers', 'wi_0')` + `('layers', 'wi_1')` → tgt `('layers', 'wi')`,
  or src `('layers_0', 'wi_0')` + `('layers_0', 'wi_1')` →
  tgt `('layers_0', 'wi')`). The scanned-source / unrolled-target case is
  handled in `intersect_trees` via `_jit_fuse_and_unstack_moe`.

  Args:
    src_flat: Flat dict of source key tuples to JAX arrays.
    tgt_flat: Flat dict of target key tuples to target leaves.

  Returns:
    A new flat dict with wi_0/wi_1 fused into wi at matching prefixes. Any
    remaining shape mismatch on non-fused axes is left for the per-target
    `_align_to_model_shape` call to handle (it composes repeat + zero-pad).
  """
  new_src_flat = dict(src_flat)
  for tgt_key in tgt_flat.keys():
    if not tgt_key or tgt_key[-1] != 'wi':
      continue
    wi_0_key = tgt_key[:-1] + ('wi_0',)
    wi_1_key = tgt_key[:-1] + ('wi_1',)
    if wi_0_key not in new_src_flat or wi_1_key not in new_src_flat:
      continue
    wi_0 = new_src_flat.pop(wi_0_key)
    wi_1 = new_src_flat.pop(wi_1_key)
    tgt_val = tgt_flat[tgt_key]
    # Pick the fused axis as the last axis where src and tgt differ. For the
    # canonical wi_0/wi_1 -> wi case this is the last axis (the mlp dim).
    mismatched_axes = [
        i for i, (s, t) in enumerate(zip(wi_0.shape, tgt_val.shape)) if s != t
    ]
    axis = mismatched_axes[-1] if mismatched_axes else len(tgt_val.shape) - 1
    n_shards = _get_n_shards(tgt_val, axis)
    logging.info(
        'Fusing MoE %s: wi_0=%s, wi_1=%s -> %s on axis %d',
        '.'.join(str(k) for k in tgt_key),
        wi_0.shape, wi_1.shape, tgt_val.shape, axis,
    )
    new_src_flat[tgt_key] = _interleave_moe_weights(
        wi_0, wi_1, tgt_val.shape, n_shards, axis=axis
    )
    del wi_0, wi_1  # Free memory immediately after fusion.
  return new_src_flat


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

    del (
        chunk_src,
        chunk_dst_shardings,
        chunk_resharded,
        chunk_src_flat,
        chunk_spec_flat,
        chunk_dst_shardings_flat,
    )
  return resharded


def transfer_state_directly(
    src_state: Mapping[str, Any],
    dst_state: Mapping[str, Any],
    reshard_fn: Callable[..., Mapping[str, Any]],
    scan_axis: int = 1,
    delete_dst_buffers: bool = False,
    reshard_chunk_size: Optional[int] = None,
) -> None:
  """Transfers state directly by matching structure, stripping wrappers.

  This handles the logic for syncing weights where no explicit mapping is provided,
  common in MaxText -> MaxText workflows. This method should work for all MaxText models.
  It automatically unwraps common containers present in MaxText models like 'base'
  (MaxText TrainState) and nested 'model' keys (vLLM wrappers). Additionally, it handles
  multiple mapping types including dicts, nnx.State, and nnx.Dict. Mismatches in keys are
  logged for debugging and handled by intersecting the source and target trees.

  Args:
    src_state: The source state to transfer from.
    dst_state: The destination state to transfer to.
    reshard_fn: A function to shard the values.
    scan_axis: The axis along which to unroll scanned layers, if needed.
    delete_dst_buffers: Whether to delete buffers in the destination state
      before resharding to save HBM. Buffers that physically alias a source
      shard are preserved automatically (see `_delete_target_buffers`).
    reshard_chunk_size: When set, the final reshard is split into sequential
      groups of this many flat keys instead of one monolithic call. This
      reduces peak contiguous HBM pressure, which prevents XLA allocator
      fragmentation on large models. The unit is *number of flat keys per
      chunk* — per-layer key counts vary by architecture (MQA vs GQA, biases
      on/off, dense vs MoE, fused vs split MoE gates), so as a rule of thumb
      start with roughly `10 * num_layers` for a dense transformer and tune
      downward if you still see fragmentation. When None (default) the
      original single-call reshard behavior is preserved.
  """
  def safe_has_key(obj: Mapping[str, Any], key: str) -> bool:
    if isinstance(obj, dict):
      return key in obj

    return hasattr(obj, key)

  # Unwrap Source (Remove 'base' wrapper from MaxText)
  if isinstance(src_state, abc.Mapping) and safe_has_key(
      src_state, 'base'
  ):
    logging.info("Unwrapping 'base' key from source state.")
    src_state = src_state['base']

  # Unwrap Target (Remove nested 'model' wrappers from vLLM)
  while isinstance(dst_state, abc.Mapping) and safe_has_key(
      dst_state, 'model'
  ):
    logging.info("Unwrapping nested 'model' key from target state.")
    dst_state = dst_state['model']

  # Helper: Convert Target Spec to Pure Dict (Strip NNX Params)
  # JAX needs a spec tree of pure NamedShardings, not Param(NamedSharding).
  def to_pure_spec(node: Any) -> Any:
    # Unwrap NNX containers
    if hasattr(node, 'to_pure_dict'):
      node = node.to_pure_dict()

    # Recurse into dicts
    if isinstance(node, abc.Mapping):
      return {k: to_pure_spec(v) for k, v in node.items()}

    # Unwrap Variables
    if isinstance(node, nnx.Variable):
      return to_pure_spec(node[...])
    if hasattr(node, 'value'):
      return node.value

    return node

  def intersect_trees(
      src: Mapping[str, Any],
      tgt_spec: Mapping[str, Any],
  ) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Optimized intersection (Handle KVCache/RNG mismatches and Scanned Layers).

    Uses flat dictionary traversal for efficiency.
    """
    # Fast path for non-dict inputs (leaves)
    if not isinstance(src, abc.Mapping) or not isinstance(tgt_spec, abc.Mapping):
      return src, tgt_spec

    # Flatten both structures to (path_tuple) -> value
    # usage of sep='/' is optional, but tuples are faster for manipulation
    src_flat = traverse_util.flatten_dict(src)
    tgt_flat = traverse_util.flatten_dict(tgt_spec)

    src_flat = _fuse_moe_weights(src_flat, tgt_flat)

    filtered_src_flat = {}
    filtered_tgt_flat = {}

    # Cache to store unstacked scanned arrays to avoid repeated work
    unstacked_cache = {}

    layer_pattern = re.compile(r'^layers_(\d+)$')

    for key_tuple, tgt_val in tgt_flat.items():
      path_str = '.'.join(str(k) for k in key_tuple)
      # Try Direct Match
      if key_tuple in src_flat:
        src_val = src_flat[key_tuple]
        src_val = _apply_dtype_cast(src_val, tgt_val.dtype, path_str)
        src_val = _align_to_model_shape(src_val, tgt_val, path_str)
        filtered_src_flat[key_tuple] = src_val
        filtered_tgt_flat[key_tuple] = tgt_val
        continue

      # Try Scanned Layer Mapping
      # We look for 'layers_X' in the path and try to map it to 'layers' (MaxText)
      # or remove it (GPT-OSS / implicit stack).

      # Locate which part of the path is 'layers_X'
      layer_idx = -1
      match_index = -1

      for i, part in enumerate(key_tuple):
        # Optimization: Only check strings that look like layers
        if isinstance(part, str) and part.startswith('layers_'):
          m = layer_pattern.match(part)
          if m:
            layer_idx = int(m.group(1))
            match_index = i
            break

      if match_index != -1:
        # Check different candidate path formats for scanned layers
        # Candidate A: Replace 'layers_X' with 'layers' (Standard MaxText)
        candidate_a = list(key_tuple)
        candidate_a[match_index] = 'layers'

        # Candidate B: Remove 'layers_X' (Implicit Container / GPT-OSS)
        candidate_b = list(key_tuple)
        candidate_b.pop(match_index)

        found_candidate = None
        for cand in [tuple(candidate_a), tuple(candidate_b)]:
          if cand in src_flat:
            found_candidate = cand
            break

        if found_candidate:
          # Cache key includes per-layer target shape so distinct unrolled
          # targets with different padded shapes don't collide on the same
          # scanned source.
          cache_key = (found_candidate, tgt_val.shape, 'aligned')
          if cache_key not in unstacked_cache:
            src_val = src_flat[found_candidate]
            candidate_path = '.'.join(str(k) for k in found_candidate)
            # Cast the bulk tensor once before unstacking.
            src_val = _apply_dtype_cast(src_val, tgt_val.dtype, candidate_path)
            scanned_per_layer_shape = (
                src_val.shape[:scan_axis] + src_val.shape[scan_axis + 1:]
            )
            if scanned_per_layer_shape == tgt_val.shape:
              # Pure unstack — no alignment needed.
              unstacked_cache[cache_key] = _unstack_scanned_param(
                  src_val, tgt_val, candidate_path, scan_axis=scan_axis
              )
            else:
              # Bulk align (repeat / zero-pad per axis) on the full scanned
              # tensor, then unstack. Replaces N per-layer alignments with
              # one bulk op + free unstack.
              logging.info(
                  'Bulk-aligning scanned %s: %s -> per-layer %s',
                  candidate_path, src_val.shape, tgt_val.shape,
              )
              unstacked_cache[cache_key] = _bulk_align_and_unstack(
                  src_val, scan_axis, tgt_val, candidate_path
              )

          # Extract the layer_idx-th element from the unstacked cache.
          sliced_val = unstacked_cache[cache_key][layer_idx]
          sliced_val = _align_to_model_shape(sliced_val, tgt_val, path_str)
          filtered_src_flat[key_tuple] = sliced_val
          filtered_tgt_flat[key_tuple] = tgt_val
          continue

        # MoE fusion case: target has 'layers_X/.../wi' but source has scanned
        # 'layers/.../wi_0' and 'layers/.../wi_1'. Fuse the full stacked
        # tensors first, then unstack once via a JIT-compiled helper — avoids
        # N per-layer jnp.concatenate dispatches and 2N intermediate device
        # allocations that cause compilation pressure and memory fragmentation.
        if key_tuple and key_tuple[-1] == 'wi':
          scanned_prefix = (
              key_tuple[:match_index] + ('layers',) + key_tuple[match_index + 1:-1]
          )
          wi_0_key = scanned_prefix + ('wi_0',)
          wi_1_key = scanned_prefix + ('wi_1',)

          if wi_0_key in src_flat and wi_1_key in src_flat:
            fused_scanned_key = scanned_prefix + ('wi_fused',)
            if fused_scanned_key not in unstacked_cache:
              scanned_prefix_path = '.'.join(str(k) for k in scanned_prefix)
              logging.info('Fusing scanned MoE weights for %s', scanned_prefix_path)
              wi_0_full = _apply_dtype_cast(
                  src_flat[wi_0_key], tgt_val.dtype, '.'.join(str(k) for k in wi_0_key)
              )
              wi_1_full = _apply_dtype_cast(
                  src_flat[wi_1_key], tgt_val.dtype, '.'.join(str(k) for k in wi_1_key)
              )
              num_layers = src_flat[wi_0_key].shape[scan_axis]
              
              # Remove scan_axis to compare against the per-layer tgt_val shape
              wi_0_single_shape = wi_0_full.shape[:scan_axis] + wi_0_full.shape[scan_axis + 1:]
              mismatched_axes = [
                  i for i, (s, t) in enumerate(zip(wi_0_single_shape, tgt_val.shape)) if s != t
              ]
              tgt_axis = mismatched_axes[-1] if mismatched_axes else len(tgt_val.shape) - 1
              n_shards = _get_n_shards(tgt_val, tgt_axis)
              
              # Offset the axis by 1 if it falls after the scan axis in the bulk tensor
              scan_padded_axis = tgt_axis if tgt_axis < scan_axis else tgt_axis + 1

              unstacked_cache[fused_scanned_key] = _jit_fuse_and_unstack_moe(
                  wi_0_full, wi_1_full, scan_axis, num_layers, n_shards, tgt_val.shape, scan_padded_axis, tgt_axis
              )
              del wi_0_full, wi_1_full

            sliced_val = unstacked_cache[fused_scanned_key][layer_idx]
            sliced_val = _align_to_model_shape(sliced_val, tgt_val, path_str)

            filtered_src_flat[key_tuple] = sliced_val
            filtered_tgt_flat[key_tuple] = tgt_val
            continue

    # Unflatten back to nested structure
    return (
        traverse_util.unflatten_dict(filtered_src_flat),
        traverse_util.unflatten_dict(filtered_tgt_flat),
    )

  # Prepare clean source and target specs
  full_source_dict = to_pure_spec(src_state)
  full_target_spec = to_pure_spec(dst_state)

  # Filter both to their intersection / mapping
  final_source, final_spec = intersect_trees(full_source_dict, full_target_spec)

  # Reshard and Update
  if reshard_chunk_size is not None:
    # Chunked path: split the flat weight dict into groups of reshard_chunk_size
    # entries and reshard each group independently. This keeps peak contiguous
    # HBM allocation proportional to chunk_size, avoiding XLA fragmentation
    # errors on large models without needing to clear the compilation cache.
    src_flat = traverse_util.flatten_dict(final_source)
    spec_flat = traverse_util.flatten_dict(final_spec)
    del final_source, final_spec
    resharded_flat = _reshard_in_chunks(
        src_flat,
        spec_flat,
        reshard_fn,
        reshard_chunk_size,
        delete_dst_buffers,
    )
    resharded_weights = traverse_util.unflatten_dict(resharded_flat)
  else:
    src_flat = traverse_util.flatten_dict(final_source)
    spec_flat = traverse_util.flatten_dict(final_spec)

    # Snapshot dst shardings before any deletion so reshard_fn never has to
    # touch a deleted jax.Array. reshard_pytree's _get_dst_sharding accepts
    # NamedSharding leaves directly, so this is a drop-in substitute for
    # passing the array objects.
    dst_shardings_flat = {
        k: _snapshot_dst_sharding(
            tgt_val.value if hasattr(tgt_val, 'value') else tgt_val
        )
        for k, tgt_val in spec_flat.items()
    }

    if delete_dst_buffers:
      _delete_target_buffers(spec_flat, src_flat)

    del final_spec
    resharded_weights = reshard_fn(
        source=final_source,
        target=traverse_util.unflatten_dict(dst_shardings_flat),
    )
  nnx.update(dst_state, resharded_weights)


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
