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

def is_fused_path(path):
  if re.compile(r"vllm_model.language_model.model.layers\.\d+\.self_attn\.qkv_proj\.weight").match(path):
    return True
  if re.compile(r"vllm_model.language_model.model.layers\.\d+\.mlp\.gate_up_proj.weight").match(path):
    return True

def build_flat_dict(
    flat_state: Iterator[tuple[tuple[str, ...], nnx.State]],
    mappings: Dict[str, tuple[str, tuple[int, ...]]],
    fused_tgt_map: Dict[str, tuple[str, tuple[int, ...]]],
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
  fused_compiled_mappings = {}

  # PRE-COMPILE MAPPINGS
  # Convert target string patterns into Python Regex objects for fast matching.
  for src, (tgt, sharding) in mappings.items():
    print(f"Compiling mapping from source '{src}' to target pattern '{tgt}' with sharding {sharding}")
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
  for keys, v in flat_state:
    # Convert key tuple ('model', 'layers', '0') to string 'model.layers.0'
    path = keys if isinstance(keys, str) else '.'.join(str(key) for key in keys)
    mapped = False
    for src, regex, sharding in compiled_mappings:
      print(f"Trying to match path '{path}' against pattern '{regex.pattern}' for source '{src}' with sharding {sharding}")
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
            print()
            wc_index += 1
          else:
            src_parts.append(part)
        actual_src = '.'.join(src_parts)
        print(f"{actual_src=}")

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
          
          if is_fused_path(path):
            if path in fused_tgt_map:
              fused_tgt_map[path].append(actual_src)
              print(f"Adding to existing fused mapping for target '{path}': source '{actual_src}' with original source pattern '{src}'")
            else:
              fused_tgt_map[path] = [actual_src]
            print(f"Mapping fused parameter '{path}' to source '{actual_src}' with original source pattern '{src}' and sharding {sharding}")

        mapped = True
        if not is_fused_path(path):
          break
        
    # There are no mappings for rng related params.
    if not mapped:
      logging.warning('!!! No mapping for flat state: %s', path)

  # Sort layers based on layer index to ensure correct order.
  for key, (layers, paths, sharding) in new_flat_dict.items():
    # print(f"Build flat dict key: {key} with {len(layers)} layers and sharding {sharding} for path {paths}")
    if isinstance(layers, list):
      layers.sort(key=lambda x: x[0])
      paths.sort(key=lambda x: x[0])
      values = [v for _, v in layers]
      paths = [p for _, p in paths]
      new_flat_dict[key] = (values, paths, sharding)

  return new_flat_dict, fused_tgt_map


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


def fuse_src_to_same_tgt_params(src_val, src_key, fuse_sources, tgt_path, tp_size):
  print(f"tp_size: {tp_size}")
  if tgt_path in fuse_sources:
    fuse_sources[tgt_path].append((src_key, src_val))
  else:
    fuse_sources[tgt_path] = [(src_key, src_val)]
  if re.compile(r"vllm_model.language_model.model.layers\.\d+\.self_attn\.qkv_proj\.weight").match(tgt_path) and len(fuse_sources[tgt_path]) == 2:
    if 'kv_einsum' in fuse_sources[tgt_path][0][0]:
      q = fuse_sources[tgt_path][1][1]
      k = fuse_sources[tgt_path][0][1][0]
      v = fuse_sources[tgt_path][0][1][1]
    elif 'kv_einsum' in fuse_sources[tgt_path][1][0]:
      q = fuse_sources[tgt_path][0][1]
      k = fuse_sources[tgt_path][1][1][0]
      v = fuse_sources[tgt_path][1][1][1]
    elif 'q_einsum' in fuse_sources[tgt_path][0][0]:
      q = fuse_sources[tgt_path][0][1]
      k = fuse_sources[tgt_path][1][1]
      v = k
    elif 'q_einsum' in fuse_sources[tgt_path][1][0]:
      q = fuse_sources[tgt_path][1][1]
      k = fuse_sources[tgt_path][0][1]
      v = k
    else:
      raise MappingError(f"Neither of the source keys for target '{tgt_path}' contains 'q_einsum' or 'kv_einsum'. Source keys: {[k for k, v in fuse_sources[tgt_path]]}")
    tp = min(tp_size, k.shape[0])
    kv_per_tp = k.shape[0] // tp
    q_per_tp = q.shape[0] // tp
    # (num_heads, d_model, head_dim) -> (d_model, num_heads, head_dim)
    q, k, v = q.transpose(1, 0, 2), k.transpose(1, 0, 2), v.transpose(1, 0, 2)
    head_dim = q.shape[2]
    d_model = q.shape[0]
    q_by_tp = q.reshape(d_model, tp, q_per_tp, head_dim)
    k_by_tp = k.reshape(d_model, tp, kv_per_tp, head_dim)
    v_by_tp = v.reshape(d_model, tp, kv_per_tp, head_dim)
    qkv_by_tp = jnp.concatenate([q_by_tp, k_by_tp, v_by_tp], axis=2)
    qkv = qkv_by_tp.reshape(d_model, -1)
    qkv = qkv.transpose(1, 0)
    match = re.search(r"layers\.(\d+)\.attn\.(q|k|kv)_einsum\.w", fuse_sources[tgt_path][0][0])
    assert match, f"Source key '{fuse_sources[tgt_path][0][0]}' does not match expected pattern for QKV fusion."
    layer_idx = match.group(1)
    fused_src_key = f"layers.{layer_idx}.attn.qkv_fused"
    fuse_sources[tgt_path] = (fused_src_key, qkv)
  elif re.compile(r"vllm_model.language_model.model.layers\.\d+\.mlp\.gate_up_proj.weight").match(tgt_path) and len(fuse_sources[tgt_path]) == 2:
    if 'gate_proj' in fuse_sources[tgt_path][0][0]:
      gate = fuse_sources[tgt_path][0][1]
      up = fuse_sources[tgt_path][1][1]
    else:
      gate = fuse_sources[tgt_path][1][1]
      up = fuse_sources[tgt_path][0][1]
    gate, up = gate.T, up.T
    hidden_dim = gate.shape[0]
    chunk_size = hidden_dim // tp_size
    # padded_chunk_size = ((chunk_size + 127)//128)*128
    # pad_amount = padded_chunk_size - chunk_size
    gate_chunks = gate.reshape(tp_size, chunk_size, gate.shape[1])
    up_chunks = up.reshape(tp_size, chunk_size, up.shape[1])
    # if pad_amount > 0:
    #   gate_chunks = jnp.pad(gate_chunks, ((0, 0), (0, pad_amount), (0, 0)))
    #   up_chunks = jnp.pad(up_chunks, ((0, 0), (0, pad_amount), (0, 0)))
    gate_up = jnp.stack([gate_chunks, up_chunks], axis=1)
    # if pad_amount > 0:
    #   gate_up = gate_up.reshape(2 * padded_chunk_size * tp_size, gate.shape[1])
    # else:
    gate_up = gate_up.reshape(2 * hidden_dim, gate.shape[1])
    match = re.search(r"layers\.(\d+)\.mlp\.gate_proj\.kernel", fuse_sources[tgt_path][0][0])
    assert match, f"Source key '{fuse_sources[tgt_path][0][0]}' does not match expected pattern for QKV fusion."
    layer_idx = match.group(1)
    fused_src_key = f"layers.{layer_idx}.mlp.gate_up_fused"
    fuse_sources[tgt_path] = (fused_src_key, gate_up)
    
  return fuse_sources


def _unroll_scanned_layers(
    src_state: Any,
    src_to_tgt_map: Dict,
    fused_tgt_map: Dict,
    tp_size: int,
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

  fuse_sources = {}
  for src_keys, src_val in src_state.flat_state():
    src_key = '.'.join(str(k) for k in src_keys)
    print(f"_unroll_scanned_layers source key '{src_key}', value: {src_val}")

    # Skip RNG parameters silently
    if 'rng' in src_key:
      logging.debug('Skipping RNG parameter: %s', src_key)
      continue

    # Validate mapping exists
    if src_key not in src_to_tgt_map:
      logging.error('No mapping for source key: %s', src_key)
      continue
    tgt_param, tgt_path, sharding_spec = src_to_tgt_map[src_key]
    print(f'Processing source key "{src_key}" with target path "{tgt_path}" with sharding spec "{sharding_spec}"')

    # Check if this is a scanned layer that needs unrolling
    layer_axis = _get_layer_axis_from_sharding_spec(sharding_spec)
    print(f'Layer axis for key "{src_key}": {layer_axis}')

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
      print(f'Processing source key "{src_key}" with value shape {src_val.value.shape if hasattr(src_val, "value") else type(src_val)}')
      if tgt_path in fused_tgt_map:
        assert src_key in fused_tgt_map[tgt_path], f"Source key '{src_key}' should be part of the fused mapping for target '{tgt_path}' but it's not. Fused mapping keys: {fused_tgt_map[tgt_path]}"
        print(f"{src_key=}, {tgt_path=}, {tgt_param.sharding=}, {src_val.value.sharding=}")
        fuse_sources = fuse_src_to_same_tgt_params(src_val, src_key, fuse_sources, tgt_path, tp_size)
        print(f"fuse_sources for target '{tgt_path}': {[k for k, v in fuse_sources.items()]}")
        if isinstance(fuse_sources[tgt_path], tuple):
          unscanned_flat[(fuse_sources[tgt_path][0], tgt_path)] = (fuse_sources[tgt_path][1], tgt_param)
          print(f"Fused parameter for target '{tgt_path}' from sources '{fuse_sources[tgt_path][0]}' with shape {fuse_sources[tgt_path][1].shape} and sharding {tgt_param.sharding}")
      else:
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
  last_three_keys = '.'.join(src_key.split('.')[-3:])
  print(f"Checking if transpose is needed for {src_key} with last key {last_key} and last three keys {last_three_keys}")
  all_key = src_key
  target_key = ''
  if last_key in transpose_keys and 'lora' not in last_key:
    target_key = last_key
  elif all_key in transpose_keys and 'lora' not in all_key:
    target_key = all_key
  elif last_three_keys in transpose_keys and 'lora' not in last_three_keys:
    target_key = last_three_keys
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

    elif src_key == 'embedder.per_layer_input_embedding': 
      print(f"Reshaping per_layer_input_embedding on {src_key}: {val.shape} -> {tgt_shape}, val type: {type(val)}")
      return jnp.reshape(val, (val.shape[0], -1))
    elif src_key == 'embedder.per_layer_model_projection.w':
      print(f"Reshaping per_layer_model_projection on {src_key}: {val.shape} -> {tgt_shape}, val type: {type(val)}")
      val = jnp.reshape(val, (val.shape[0], -1))
      return val.T
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
    elif re.compile(r'layers\..*\.attn_vec_einsum\.w').match(src_key):
      # reshape from (num_head, head_dim, model_dim) to (model_dim, num_head * head_dim) for vec_einsum.
      print(f"Reshaping attention vec_einsum on {src_key}: {val.shape} -> {tgt_shape}")
      return val.reshape((val.shape[0] * val.shape[1], val.shape[2])).T
    elif re.compile(r'layers\..*\.moe\.gating_einsum').match(src_key):
      print(f"Reshaping moe.gating_einsum on {src_key}: {val.shape} -> {tgt_shape}")
      tp_size = kwargs["tp_size"]
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
      print(f"Reshaping moe.gating_einsum on {src_key}: {val_chunks.shape} -> {tgt_shape}")
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
  elif re.compile(r'layers\..*\.per_layer_input_gate\.w').match(src_key) or re.compile(r'layers\..*\.per_layer_projection\.w').match(src_key) or re.compile(r'layers\..*\.moe\.router_logits').match(src_key):
    return val.T
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
    val: jnp.ndarray, tgt_dtype: jnp.dtype, src_key: str
) -> jnp.ndarray:
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
    if flat_key[-1:] == ('embedding',):
      embed_param = tgt_param
    elif flat_key[-1:] == ('lm_head',):
      lm_head_param = tgt_param

  if embed_param is None or lm_head_param is None:
    return
  if not hasattr(embed_param, 'value') or not hasattr(lm_head_param, 'value'):
    return
  if embed_param.value.shape != lm_head_param.value.shape:
    return

  lm_head_param.value = embed_param.value

def flatten_to_tuples(d):
    items = []
    key_idx_mapping = {}
    i = 0
    for k, v in d.items():
      # If it's a leaf node, add the (path, value) tuple
      items.append((k, v))
      key_idx_mapping[k] = i
      i+= 1
    return items, key_idx_mapping
  

def unflatten_from_tuples(flat_list, dst_state):
  for path, value in flat_list:
    print(f"Processing path: {path} with value = {value }")
    dst_state[path] = value
  return dst_state

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
  if isinstance(dst_state, dict):
    # If it's already a dict, perhaps you don't need to flatten it,
    # or you need to use a different flattening utility.
    tgt_flat_list, tgt_key_idx_mapping = flatten_to_tuples(dst_state)
  else:
    tgt_flat_list = dst_state.flat_state()
    tgt_key_idx_mapping = None
  for k, v in tgt_flat_list:
    print(f"Target flat key: {k}, value shape: {v.shape}, value: {v}")

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
  # {src_key: (tgt_param, tgt_path, sharding_spec)}
  # {fused_tgt_key: (src_key, src_val, sharding_spec)}
  fused_tgt_map = {}
  src_to_tgt_map, fused_tgt_map = build_flat_dict(tgt_flat_list, key_mappings, fused_tgt_map)
  for tgt, src_list in fused_tgt_map.items():
    if len(src_list) > 1:
      print(f"Fused target key '{tgt}' is mapped from multiple source keys: {src_list}. This requires special handling.")

  # Unroll scanned layers and flatten source state
  unscanned_src_to_tgt_flat = _unroll_scanned_layers(src_state, src_to_tgt_map, fused_tgt_map, kwargs['tp_size'])
  transferred_target_keys = set()

  # Transfer values with transformations
  for (flat_src_key, flat_tgt_key), (
      val,
      tgt_param,
  ) in unscanned_src_to_tgt_flat.items():
    # Apply transpose if configured
    print(f'Processing unscanned_src_to_tgt_flat: {flat_src_key} -> {flat_tgt_key} with initial shape {val.shape}')
    val = _apply_transpose(val, flat_src_key, transpose_keys, rollout_engine)
    if flat_src_key == "embedder.input_embedding":
      print(f"src val {val}, tgt_param {tgt_param} ")

    # Apply optional hook function
    if key_mapping_hook_fns and flat_src_key in key_mapping_hook_fns:
      val = key_mapping_hook_fns[flat_src_key](val)

    # Align shapes (padding/repeating as needed)
    print(f'Aligning shape for {flat_src_key} -> {flat_tgt_key}: {val.shape} -> {tgt_param.shape}')
    tgt_shape = tgt_param.value.shape if hasattr(tgt_param, 'value') else tgt_param.shape
    tgt_dtype = tgt_param.value.dtype if hasattr(tgt_param, 'value') else tgt_param.dtype
    val = _align_shape(
        val, tgt_shape, flat_src_key, rollout_engine, **kwargs
    )

    # Cast to target dtype
    val = _apply_dtype_cast(val, tgt_dtype, flat_src_key)

    # Assign transformed value
    if hasattr(tgt_param, 'value'):
      tgt_param.value = val
    else:
      tgt_flat_list[tgt_key_idx_mapping[flat_tgt_key]]  = (flat_tgt_key, val)
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
    for k, v in tgt_flat_dict.items():
      print(f"tgt_flat_dict key: {k}, value {v}")
    resharded_values_flat_dict = reshard_fn(tgt_flat_dict, sharding_dict)

    for tgt_key, tgt_param in tgt_flat_list:
      assert (
          tgt_key in resharded_values_flat_dict
      ), f'Key {tgt_key} not in resharded values'
      if hasattr(tgt_param, 'value'):
        tgt_param.value = resharded_values_flat_dict[tgt_key]
      else:
        tgt_flat_list[tgt_key_idx_mapping[tgt_key]] = (tgt_key, resharded_values_flat_dict[tgt_key])
        print(f"After resharding, assigned {tgt_key} with shape {tgt_param.shape} and value {tgt_param}")  

  if isinstance(dst_state, dict):
    return unflatten_from_tuples(tgt_flat_list, dst_state)
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
    src_val: jax.Array | np.ndarray | Any,
    tgt_val: jax.Array | np.ndarray | Any,
    key_path: str,
    scan_axis: Optional[int] = None,
) -> Tuple[jax.Array | np.ndarray | Any]:
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


def _repeat_to_model_shape(
    src_val: jax.Array | np.ndarray | Any,
    tgt_val: jax.Array | np.ndarray | Any,
    key_path: str,
) -> jax.Array | np.ndarray | Any:
  """Repeats src_val to match tgt_val's shape if shapes are compatible multiples.

  This is used to broadcast KV heads (or other dimensions) from a model with
  fewer heads to one with more heads, e.g., when transferring GQA weights.

  Args:
      src_val: The source array to repeat.
      tgt_val: The target array whose shape we want to match.
      key_path: Path string for debug logging.

  Returns:
      A repeated version of src_val matching tgt_val's shape, or src_val
      unchanged if shapes already match or repeating is not possible.
  """
  if not (hasattr(src_val, 'shape') and hasattr(tgt_val, 'shape')):
    return src_val

  src_shape = src_val.shape
  tgt_shape = tgt_val.shape

  if src_shape == tgt_shape:
    return src_val

  if len(src_shape) != len(tgt_shape):
    return src_val

  for src_dim, tgt_dim in zip(src_shape, tgt_shape):
    if src_dim > tgt_dim or tgt_dim % src_dim != 0:
      return src_val

  logging.info(
      "Repeating '%s' from %s to %s.",
      key_path, src_shape, tgt_shape,
  )
  result = src_val
  for axis, (src_dim, tgt_dim) in enumerate(zip(src_shape, tgt_shape)):
    if tgt_dim != src_dim:
      result = jnp.repeat(result, tgt_dim // src_dim, axis=axis)

  return result


def _delete_pytree_buffers(pytree: Any) -> None:
  """Deletes buffers of jax.Arrays in a pytree to save memory."""
  logging.info('Deleting pytree buffers.')

  def _delete_buffers(x):
    if isinstance(x, nnx.Variable) and isinstance(x.value, jax.Array):
      if not x.value.is_deleted():
        x.value.delete()
    elif isinstance(x, jax.Array):
      if not x.is_deleted():
        x.delete()
    return x

  jax.tree_util.tree_map(_delete_buffers, pytree)


def transfer_state_directly(
    src_state: Mapping[str, Any],
    dst_state: Mapping[str, Any],
    reshard_fn: Callable[..., Mapping[str, Any]],
    scan_axis: int = 1,
    delete_dst_buffers: bool = False,
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
    delete_dst_buffers: Whether to delete buffers in the destination state after transfer to save memory.
  """

  if delete_dst_buffers:
    _delete_pytree_buffers(dst_state)
    gc.collect()

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

    filtered_src_flat = {}
    filtered_tgt_flat = {}

    # Cache to store unstacked scanned arrays to avoid repeated work
    unstacked_cache = {}

    layer_pattern = re.compile(r'^layers_(\d+)$')

    # Cache to store unstacked scanned arrays to avoid repeated work
    unstacked_cache = {}

    for key_tuple, tgt_val in tgt_flat.items():
      # Try Direct Match
      if key_tuple in src_flat:
        src_val = src_flat[key_tuple]
        src_val = _apply_dtype_cast(src_val, tgt_val.dtype, str(key_tuple))
        src_val = _repeat_to_model_shape(src_val, tgt_val, str(key_tuple))
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
          # Apply the dtype cast and the repeating *before* unstacking
          if found_candidate not in unstacked_cache:
            src_val = src_flat[found_candidate]
            
            # Cast the bulk tensor once
            src_val = _apply_dtype_cast(src_val, tgt_val.dtype, str(found_candidate))
            
            # Predict the stacked target shape and repeat the bulk tensor once
            src_shape = getattr(src_val, 'shape', None)
            tgt_shape = getattr(tgt_val, 'shape', None)
            
            if src_shape and tgt_shape and len(src_shape) == len(tgt_shape) + 1:
              # Construct the 3D target shape (e.g., [layers, global_heads, dim])
              stacked_tgt_shape = tgt_shape[:scan_axis] + (src_shape[scan_axis],) + tgt_shape[scan_axis:]
              
              # Mock a target array purely to pass the shape to our repeat helper
              class _MockTarget:
                shape = stacked_tgt_shape
                
              src_val = _repeat_to_model_shape(src_val, _MockTarget(), str(found_candidate))
            
            # Unstack the already casted and repeated tensor using the provided scan_axis
            unstacked_cache[found_candidate] = _unstack_scanned_param(
                src_val, tgt_val, str(found_candidate), scan_axis=scan_axis
            )
          
          # Extract the layer_idx-th element from the unstacked cache
          sliced_val = unstacked_cache[found_candidate][layer_idx]

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
  resharded_weights = reshard_fn(
      source=final_source,
      target=final_spec,
  )
  nnx.update(dst_state, resharded_weights)

  # Explicitly free memory
  gc.collect()


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
