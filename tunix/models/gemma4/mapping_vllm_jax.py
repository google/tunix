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

"""vLLM JAX backend mappings for Gemma4 models."""

from __future__ import annotations

from typing import Any, Dict, Tuple
from flax import nnx
import jax
import jax.numpy as jnp

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]


TO_HF_MAPPINGS = {
    'embedder.input_embedding': ('model.embed_tokens.weight', ('model', None)),
    'layers.*.pre_attention_norm.scale': (
        'model.layers.*.input_layernorm.weight',
        (None,),
    ),
    # GLOBAL layers with `attention_k_eq_v` keep separate (unfused) Q/K
    # kernels (no V; V==K); `preprocess_src_state` leaves those layers'
    # q_einsum/k_einsum untouched, so these two entries only ever match them.
    'layers.*.attn.q_einsum.w': (
        'model.layers.*.self_attn.q_proj.weight',
        (None, 'model', None),
    ),
    'layers.*.attn.k_einsum.w': (
        'model.layers.*.self_attn.k_proj.weight',
        (None, 'model', None),
    ),
    # Every other layer fuses Q, K, V into a single qkv_proj kernel.
    # `preprocess_src_state` combines that layer's q_einsum + kv_einsum into
    # this synthetic, already TP-interleaved `qkv_fused` entry.
    'layers.*.attn.qkv_fused.w': (
        'model.layers.*.self_attn.qkv_proj.weight',
        (None, 'model'),
    ),
    'layers.*.attn._query_norm.scale': (
        'model.layers.*.self_attn.q_norm.weight',
        (None,),
    ),
    'layers.*.attn._key_norm.scale': (
        'model.layers.*.self_attn.k_norm.weight',
        (None,),
    ),
    'layers.*.attn.attn_vec_einsum.w': (
        'model.layers.*.self_attn.o_proj.weight',
        ('model', None, None),
    ),
    'layers.*.post_attention_norm.scale': (
        'model.layers.*.post_attention_layernorm.weight',
        (None,),
    ),
    'layers.*.pre_ffw_norm.scale': (
        'model.layers.*.pre_feedforward_layernorm.weight',
        (None,),
    ),
    # Gemma4MLP always fuses gate_proj/up_proj into one gate_up_proj kernel;
    # `preprocess_src_state` combines them into this synthetic,
    # TP-interleaved `gate_up_fused` entry.
    'layers.*.mlp.gate_up_fused.kernel': (
        'model.layers.*.mlp.gate_up_proj.weight',
        (None, 'model'),
    ),
    'layers.*.mlp.down_proj.kernel': (
        'model.layers.*.mlp.down_proj.weight',
        ('model', None),
    ),
    'layers.*.post_ffw_norm.scale': (
        'model.layers.*.post_feedforward_layernorm.weight',
        (None,),
    ),
    'layers.*.skip_scale': (
        'model.layers.*.layer_scalar',
        (None,),
    ),
    'final_norm.scale': ('model.norm.weight', (None,)),
    'layers.*.moe_pre_ffw_norm.scale': (
        'model.layers.*.pre_feedforward_layernorm_2.weight',
        (None,),
    ),
    'layers.*.moe.router_logits': (
        'model.layers.*.router.proj.weight',
        (None, 'model'),
    ),
    'layers.*.moe.router_scale': (
        'model.layers.*.router.scale',
        (None,),
    ),
    'layers.*.moe.per_expert_scale': (
        'model.layers.*.router.per_expert_scale',
        (None,),
    ),
    'layers.*.moe.gating_einsum': (
        'model.layers.*.experts.kernel_gating_upproj_EDF',
        (None, None, 'model'),
    ),
    'layers.*.moe.linear': (
        'model.layers.*.experts.kernel_down_proj_EFD',
        ('model', None, None),
    ),
    'layers.*.dense_post_ffw_norm.scale': (
        'model.layers.*.post_feedforward_layernorm_1.weight',
        (None,),
    ),
    'layers.*.moe_post_ffw_norm.scale': (
        'model.layers.*.post_feedforward_layernorm_2.weight',
        (None,),
    ),
    'embedder.per_layer_input_embedding': (
        'model.embed_tokens_per_layer.weight',
        ('model', None),
    ),
    'embedder.per_layer_model_projection.w': (
        'model.per_layer_model_projection.weight',
        (None, 'model'),
    ),
    'embedder.per_layer_projection_norm.scale': (
        'model.per_layer_projection_norm.weight',
        (None,),
    ),
    'layers.*.per_layer_input_gate.w': (
        'model.layers.*.per_layer_input_gate.weight',
        (None, 'model'),
    ),
    'layers.*.per_layer_projection.w': (
        'model.layers.*.per_layer_projection.weight',
        ('model', None),
    ),
    'layers.*.post_per_layer_input_norm.scale': (
        'model.layers.*.post_per_layer_input_norm.weight',
        (None,),
    ),
}

LORA_TO_HF_MAPPINGS: Dict[str, MappingEntry] = {}

TO_HF_TRANSPOSE_KEYS = {
    'layers.*.attn.q_einsum.w': (1, 0, 2),
    'layers.*.attn.k_einsum.w': (1, 0, 2),
}


def _reorder_for_tp_sharding(concatenated, split_sizes, tp_size):
  """Reorders parts concatenated on the last axis into a per-shard layout.

  `concatenated` holds `split_sizes` parts back to back on its last axis
  (`[part0, part1, ...]`). This reorders it so each of the `tp_size` equal
  contiguous chunks of the last axis holds a slice of every part
  (`[part0_shard_i, part1_shard_i, ...]`), which is the on-device layout
  tpu_inference's fused linear layers (`JaxQKVParallelLinear`,
  `JaxMergedColumnParallelLinear`) expect from their kernel. Mirrors
  `tpu_inference.layers.common.utils.reorder_concatenated_tensor_for_sharding`.
  """
  if tp_size <= 1:
    return concatenated
  lead_shape = concatenated.shape[:-1]
  parts = []
  offset = 0
  for size in split_sizes:
    if size % tp_size != 0:
      raise ValueError(f'Part size {size} is not divisible by tp_size {tp_size}.')
    part = jax.lax.slice_in_dim(concatenated, offset, offset + size, axis=-1)
    parts.append(part.reshape(lead_shape + (tp_size, size // tp_size)))
    offset += size
  reordered = jnp.concatenate(parts, axis=-1)
  return reordered.reshape(lead_shape + (sum(split_sizes),))


def preprocess_src_state(src_state: Any, tp_size: int = 1) -> Any:
  """Fuses Q/K/V and gate/up projections to match tpu_inference's kernels.

  tpu_inference's Gemma4Attention fuses Q, K, V into a single `qkv_proj`
  kernel on every layer except GLOBAL layers with `attention_k_eq_v` (which
  keep separate `q_proj`/`k_proj`; V is not materialized, V==K).
  Gemma4MLP always fuses `gate_proj`/`up_proj` into a single `gate_up_proj`
  kernel. This combines the corresponding tunix-native params
  (`q_einsum`+`kv_einsum`, `gate_proj`+`up_proj`) into synthetic
  `qkv_fused`/`gate_up_fused` entries that TO_HF_MAPPINGS targets 1:1, and
  TP-interleaves them into the layout the fused kernel expects on-device.
  """
  if not hasattr(src_state, 'flat_state'):
    return src_state

  by_key = {
      '.'.join(str(k) for k in keys): (keys, param)
      for keys, param in src_state.flat_state()
  }

  def value_of(param):
    return param.value if hasattr(param, 'value') else param

  def wrap_like(param, val):
    return nnx.Param(val) if hasattr(param, 'value') else val

  consumed = set()
  fused_entries = []

  for src_key, (keys, param) in by_key.items():
    if src_key.endswith('attn.kv_einsum.w'):
      q_key = src_key[: -len('kv_einsum.w')] + 'q_einsum.w'
      if q_key not in by_key:
        raise ValueError(f'Missing {q_key} to fuse with {src_key}.')
      q_val = value_of(by_key[q_key][1])
      kv_val = value_of(param)
      k_val, v_val = kv_val[0], kv_val[1]
      # (N,D,H) -> (D,N,H) -> (D,N*H); same for K/V with num_kv_heads.
      q_t = jnp.transpose(q_val, (1, 0, 2)).reshape(q_val.shape[1], -1)
      k_t = jnp.transpose(k_val, (1, 0, 2)).reshape(k_val.shape[1], -1)
      v_t = jnp.transpose(v_val, (1, 0, 2)).reshape(v_val.shape[1], -1)
      concatenated = jnp.concatenate([q_t, k_t, v_t], axis=-1)
      fused_val = _reorder_for_tp_sharding(
          concatenated, [q_t.shape[-1], k_t.shape[-1], v_t.shape[-1]], tp_size
      )
      fused_keys = keys[:-2] + ('qkv_fused', 'w')
      fused_entries.append((fused_keys, wrap_like(param, fused_val)))
      consumed.add(src_key)
      consumed.add(q_key)
    elif src_key.endswith('mlp.gate_proj.kernel'):
      up_key = src_key[: -len('gate_proj.kernel')] + 'up_proj.kernel'
      if up_key not in by_key:
        raise ValueError(f'Missing {up_key} to fuse with {src_key}.')
      gate_val = value_of(param)
      up_val = value_of(by_key[up_key][1])
      concatenated = jnp.concatenate([gate_val, up_val], axis=-1)
      fused_val = _reorder_for_tp_sharding(
          concatenated, [gate_val.shape[-1], up_val.shape[-1]], tp_size
      )
      fused_keys = keys[:-2] + ('gate_up_fused', 'kernel')
      fused_entries.append((fused_keys, wrap_like(param, fused_val)))
      consumed.add(src_key)
      consumed.add(up_key)

  new_flat_state = [
      (keys, param)
      for src_key, (keys, param) in by_key.items()
      if src_key not in consumed
  ] + fused_entries

  return src_state.from_flat_path(new_flat_state)


VLLM_JAX_MAPPING: Dict[str, Any] = {
    'to_hf_mappings': TO_HF_MAPPINGS,
    'lora_to_hf_mappings': LORA_TO_HF_MAPPINGS,
    'to_hf_transpose_keys': TO_HF_TRANSPOSE_KEYS,
    'preprocess_src_state': preprocess_src_state,
}

__all__ = [
    'VLLM_JAX_MAPPING',
]
