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

import os
from typing import Any, Dict, Tuple

from absl import logging
from flax import nnx
import jax.numpy as jnp

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]


TO_HF_MAPPINGS = {
    'lm_head.w': (
        r'(?:(?:model|language_model)\.)*lm_head\.(?:weight|kernel)',
        (None, 'model'),
    ),
    'embedder.input_embedding': (
        r'(?:(?:model|language_model)\.)*embed_tokens\.(?:weight|kernel)',
        ('model', None),
    ),
    'layers.*.pre_attention_norm.scale': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.input_layernorm\.(?:weight|kernel)',
        (None,),
    ),
    'layers.*.attn.q_einsum.w': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.self_attn\.q_proj\.(?:weight|kernel)',
        (None, 'model', None),
    ),
    # `k_einsum` exists natively on k_eq_v (GLOBAL) layers; for the other
    # layers it is split out of `kv_einsum` by `preprocess_src_state`.
    'layers.*.attn.k_einsum.w': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.self_attn\.k_proj\.(?:weight|kernel)',
        (None, 'model', None),
    ),
    'layers.*.attn.v_einsum.w': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.self_attn\.v_proj\.(?:weight|kernel)',
        (None, 'model', None),
    ),
    'layers.*.attn._query_norm.scale': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.self_attn\.q_norm\.(?:weight|kernel)',
        (None,),
    ),
    'layers.*.attn._key_norm.scale': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.self_attn\.k_norm\.(?:weight|kernel)',
        (None,),
    ),
    'layers.*.attn.attn_vec_einsum.w': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.self_attn\.o_proj\.(?:weight|kernel)',
        ('model', None, None),
    ),
    'layers.*.post_attention_norm.scale': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.post_attention_layernorm\.(?:weight|kernel)',
        (None,),
    ),
    'layers.*.pre_ffw_norm.scale': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.pre_feedforward_layernorm\.(?:weight|kernel)',
        (None,),
    ),
    'layers.*.mlp.gate_up_proj.kernel': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.mlp\.gate_up_proj\.(?:weight|kernel)',
        (None, 'model'),
    ),
    'layers.*.mlp.down_proj.kernel': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.mlp\.down_proj\.(?:weight|kernel)',
        ('model', None),
    ),
    'layers.*.post_ffw_norm.scale': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.post_feedforward_layernorm\.(?:weight|kernel)',
        (None,),
    ),
    'layers.*.skip_scale': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.layer_scalar',
        (None,),
    ),
    'final_norm.scale': (
        r'(?:(?:model|language_model)\.)*norm\.(?:weight|kernel)',
        (None,),
    ),
    'layers.*.moe_pre_ffw_norm.scale': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.pre_feedforward_layernorm_2\.(?:weight|kernel)',
        (None,),
    ),
    'layers.*.moe.router_logits': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.router\.proj\.(?:weight|kernel)',
        (None, 'model'),
    ),
    'layers.*.moe.router_scale': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.router\.scale',
        (None,),
    ),
    'layers.*.moe.per_expert_scale': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.router\.per_expert_scale',
        (None,),
    ),
    'layers.*.moe.gating_einsum': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.experts\.kernel_gating_upproj_EDF',
        (None, None, 'model'),
    ),
    'layers.*.moe.linear': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.experts\.kernel_down_proj_EFD',
        ('model', None, None),
    ),
    'layers.*.dense_post_ffw_norm.scale': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.post_feedforward_layernorm_1\.(?:weight|kernel)',
        (None,),
    ),
    'layers.*.moe_post_ffw_norm.scale': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.post_feedforward_layernorm_2\.(?:weight|kernel)',
        (None,),
    ),
    'embedder.per_layer_input_embedding': (
        r'(?:(?:model|language_model)\.)*embed_tokens_per_layer\.(?:weight|kernel)',
        ('model', None),
    ),
    'embedder.per_layer_model_projection.w': (
        r'(?:(?:model|language_model)\.)*per_layer_model_projection\.(?:weight|kernel)',
        (None, 'model'),
    ),
    'embedder.per_layer_projection_norm.scale': (
        r'(?:(?:model|language_model)\.)*per_layer_projection_norm\.(?:weight|kernel)',
        (None,),
    ),
    'layers.*.per_layer_input_gate.w': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.per_layer_input_gate\.(?:weight|kernel)',
        (None, 'model'),
    ),
    'layers.*.per_layer_projection.w': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.per_layer_projection\.(?:weight|kernel)',
        ('model', None),
    ),
    'layers.*.post_per_layer_input_norm.scale': (
        r'(?:(?:model|language_model)\.)*layers\.(\d+)\.post_per_layer_input_norm\.(?:weight|kernel)',
        (None,),
    ),
    'vision_encoder.entry.pos_emb': (
        r'(?:model\.)?vision_tower\.patch_embedder\.position_embedding_table',
        (None, None, 'model'),
    ),
    'vision_encoder.entry.input_projection.w': (
        r'(?:model\.)?vision_tower\.patch_embedder\.input_proj\.(?:weight|kernel|w)',
        (None, 'model'),
    ),
    'vision_encoder.layers.*.pre_attention_norm.scale': (
        r'(?:model\.)?vision_tower\.(?:encoder\.)?layers\.(\d+)\.input_layernorm\.(?:weight|kernel|scale)',
        (None,),
    ),
    'vision_encoder.layers.*.attn.q_einsum.w': (
        r'(?:model\.)?vision_tower\.(?:encoder\.)?layers\.(\d+)\.self_attn\.q_proj(?:\.linear)?\.(?:weight|kernel|w)',
        None,
    ),
    'vision_encoder.layers.*.attn.k_einsum.w': (
        r'(?:model\.)?vision_tower\.(?:encoder\.)?layers\.(\d+)\.self_attn\.k_proj(?:\.linear)?\.(?:weight|kernel|w)',
        None,
    ),
    'vision_encoder.layers.*.attn.v_einsum.w': (
        r'(?:model\.)?vision_tower\.(?:encoder\.)?layers\.(\d+)\.self_attn\.v_proj(?:\.linear)?\.(?:weight|kernel|w)',
        None,
    ),
    'vision_encoder.layers.*.attn.query_norm.scale': (
        r'(?:model\.)?vision_tower\.(?:encoder\.)?layers\.(\d+)\.self_attn\.q_norm\.(?:weight|kernel|scale)',
        (None,),
    ),
    'vision_encoder.layers.*.attn.key_norm.scale': (
        r'(?:model\.)?vision_tower\.(?:encoder\.)?layers\.(\d+)\.self_attn\.k_norm\.(?:weight|kernel|scale)',
        (None,),
    ),
    'vision_encoder.layers.*.attn.attn_vec_einsum.w': (
        r'(?:model\.)?vision_tower\.(?:encoder\.)?layers\.(\d+)\.self_attn\.o_proj(?:\.linear)?\.(?:weight|kernel|w)',
        None,
    ),
    'vision_encoder.layers.*.post_attention_norm.scale': (
        r'(?:model\.)?vision_tower\.(?:encoder\.)?layers\.(\d+)\.post_attention_layernorm\.(?:weight|kernel|scale)',
        (None,),
    ),
    'vision_encoder.layers.*.pre_ffw_norm.scale': (
        r'(?:model\.)?vision_tower\.(?:encoder\.)?layers\.(\d+)\.pre_feedforward_layernorm\.(?:weight|kernel|scale)',
        (None,),
    ),
    'vision_encoder.layers.*.mlp.linear.w': (
        r'(?:model\.)?vision_tower\.(?:encoder\.)?layers\.(\d+)\.mlp\.down_proj(?:\.linear)?\.(?:weight|kernel|w)',
        None,
    ),
    'vision_encoder.layers.*.mlp.gate_proj.kernel': (
        r'(?:model\.)?vision_tower\.(?:encoder\.)?layers\.(\d+)\.mlp\.gate_proj(?:\.linear)?\.(?:weight|kernel|w)',
        None,
    ),
    'vision_encoder.layers.*.mlp.up_proj.kernel': (
        r'(?:model\.)?vision_tower\.(?:encoder\.)?layers\.(\d+)\.mlp\.up_proj(?:\.linear)?\.(?:weight|kernel|w)',
        None,
    ),
    'vision_encoder.layers.*.post_ffw_norm.scale': (
        r'(?:model\.)?vision_tower\.(?:encoder\.)?layers\.(\d+)\.post_feedforward_layernorm\.(?:weight|kernel|scale)',
        (None,),
    ),
    'vision_encoder.standardize.scale': (
        r'(?:model\.)?vision_tower\.(?:standardize\.|std_)?scale(?:\.value)?',
        (None,),
    ),
    'vision_encoder.standardize.bias': (
        r'(?:model\.)?vision_tower\.(?:standardize\.|std_)?bias(?:\.value)?',
        (None,),
    ),
    'embedder.mm_input_projection.w': (
        r'(?:model\.)?embed_vision\.(?:embedding_projection|linear)\.(?:weight|kernel|w)',
        None,
    ),
}

LORA_TO_HF_MAPPINGS: Dict[str, MappingEntry] = {}

TO_HF_TRANSPOSE_KEYS = {
    'layers.*.attn.q_einsum.w': (1, 0, 2),
    'layers.*.attn.k_einsum.w': (1, 0, 2),
    'layers.*.attn.v_einsum.w': (1, 0, 2),
}


def _wrap_like(param: Any, val: Any) -> Any:
  """Wraps `val` in `nnx.Param` iff `param` is a variable-like object."""
  return nnx.Param(val) if hasattr(param, 'value') else val


def preprocess_src_state(
    src_state: Any, tp_size: int | None = None
) -> Any:
  """Reshapes the Tunix state to match the vLLM (tpu-inference) pytree."""
  env_shards = os.environ.get('TP_SIZE', os.environ.get('VLLM_TP_SIZE'))
  if tp_size is not None:
    n_shards = max(1, int(tp_size))
  elif env_shards is not None:
    n_shards = max(1, int(env_shards))
  else:
    n_shards = 1
  if hasattr(src_state, 'flat_state'):
    # --- On-the-fly Streaming LoRA Fusion ---
    params = {tuple(keys): param for keys, param in src_state.flat_state()}

    rank = None
    for keys, param in params.items():
      if isinstance(keys[-1], str) and keys[-1] in (
          'w_lora_a',
          'w_lora_b',
          'kernel_lora_a',
          'kernel_lora_b',
      ):
        val = param.value if hasattr(param, 'value') else param
        rank = (
            int(val.shape[-1])
            if keys[-1] in ('w_lora_a', 'kernel_lora_a')
            else int(val.shape[0])
        )
        break

    alpha_str = os.environ.get('LORA_ALPHA')
    scale = 1.0
    if rank is not None:
      alpha = float(alpha_str) if alpha_str is not None else float(rank)
      scale = alpha / rank

    total_lora_a = sum(
        1
        for k in params
        if isinstance(k[-1], str) and k[-1] in ('w_lora_a', 'kernel_lora_a')
    )
    total_lora_b = sum(
        1
        for k in params
        if isinstance(k[-1], str) and k[-1] in ('w_lora_b', 'kernel_lora_b')
    )
    if total_lora_a != total_lora_b:
      raise RuntimeError(
          f'LoRA fusion incomplete: mismatched LoRA adapter counts '
          f'(lora_a={total_lora_a}, lora_b={total_lora_b}) in preprocess_src_state!'
      )
    fused_lora_count = 0

    def _get_fused_val(base_key: tuple[Any, ...], pop_consumed: bool = True) -> Any:
      nonlocal fused_lora_count
      param_base = params.pop(base_key) if pop_consumed else params[base_key]
      base_val = param_base.value if hasattr(param_base, 'value') else param_base
      if rank is None:
        return base_val
      base_leaf = base_key[-1]
      a_leaf = 'w_lora_a' if base_leaf == 'w' else 'kernel_lora_a'
      b_leaf = 'w_lora_b' if base_leaf == 'w' else 'kernel_lora_b'
      a_key = base_key[:-1] + (a_leaf,)
      b_key = base_key[:-1] + (b_leaf,)
      has_a = a_key in params
      has_b = b_key in params
      if has_a != has_b:
        raise RuntimeError(
            f'LoRA fusion incomplete: asymmetric LoRA adapter pair for {base_key} '
            f'(has_a={has_a}, has_b={has_b}) in preprocess_src_state!'
        )
      if not has_a:
        return base_val
      param_a = params.pop(a_key) if pop_consumed else params[a_key]
      param_b = params.pop(b_key) if pop_consumed else params[b_key]
      if pop_consumed:
        fused_lora_count += 1
      a_val = param_a.value if hasattr(param_a, 'value') else param_a
      b_val = param_b.value if hasattr(param_b, 'value') else param_b
      adapter_rank = int(a_val.shape[-1])
      adapter_alpha = (
          float(alpha_str) if alpha_str is not None else float(adapter_rank)
      )
      adapter_scale = adapter_alpha / adapter_rank
      if base_key[-2] in ('q_einsum', 'k_einsum'):
        delta = jnp.einsum('dr,rnh->ndh', a_val, b_val) * adapter_scale
      elif base_key[-2] == 'kv_einsum':
        delta = jnp.einsum('dr,rckh->ckdh', a_val, b_val) * adapter_scale
      elif base_key[-2] == 'gating_einsum':
        delta = jnp.einsum('dr,rcf->cfd', a_val, b_val) * adapter_scale
      elif base_key[-2] == 'attn_vec_einsum':
        delta = jnp.einsum('nhr,rd->nhd', a_val, b_val) * adapter_scale
      else:
        delta = (a_val @ b_val) * adapter_scale
      return base_val + delta
    # --- End LoRA Fusion Helper ---

    new_flat_state = []
    layers_gate = {}
    layers_up = {}

    for keys in list(params.keys()):
      if isinstance(keys[-1], str) and keys[-1] in (
          'w_lora_a',
          'w_lora_b',
          'kernel_lora_a',
          'kernel_lora_b',
      ):
        continue
      param = params.get(keys)
      if param is None:
        continue

      if len(keys) > 0 and keys[0] == 'vision_encoder':
        leaf_key = keys[-2:] if len(keys) >= 2 else ()
        param_val = _get_fused_val(tuple(keys))

        if leaf_key == ('q_einsum', 'w'):
          # Flax: (num_heads, features, head_dim) -> vLLM: (features, num_heads, head_dim)
          q_t = jnp.transpose(param_val, (1, 0, 2))
          new_flat_state.append((keys, _wrap_like(param, q_t)))
        elif leaf_key == ('kv_einsum', 'w'):
          # Flax: (2, num_kv_heads, features, head_dim)
          # Split into k and v, transpose (1, 0, 2) to (features, num_kv_heads, head_dim)
          k_t = jnp.transpose(param_val[0], (1, 0, 2))
          v_t = jnp.transpose(param_val[1], (1, 0, 2))
          k_keys = keys[:-2] + ('k_einsum', 'w')
          v_keys = keys[:-2] + ('v_einsum', 'w')
          new_flat_state.append((k_keys, _wrap_like(param, k_t)))
          new_flat_state.append((v_keys, _wrap_like(param, v_t)))
        elif leaf_key == ('gating_einsum', 'w'):
          # Flax: (2, hidden_dim, features)
          # Split into gate and up, transpose (1, 0) to (features, hidden_dim)
          gate_t = jnp.transpose(param_val[0], (1, 0))
          up_t = jnp.transpose(param_val[1], (1, 0))
          gate_keys = keys[:-2] + ('gate_proj', 'kernel')
          up_keys = keys[:-2] + ('up_proj', 'kernel')
          new_flat_state.append((gate_keys, _wrap_like(param, gate_t)))
          new_flat_state.append((up_keys, _wrap_like(param, up_t)))
        else:
          new_flat_state.append((keys, _wrap_like(param, param_val)))
        continue

      layers_pos = [i for i, k in enumerate(keys) if k == 'layers']
      layer_idx = (
          keys[layers_pos[0] + 1]
          if layers_pos and layers_pos[0] + 1 < len(keys)
          else None
      )
      leaf_key = keys[-2:] if len(keys) >= 2 else ()
      if leaf_key == ('kv_einsum', 'w'):
        kv_val = _get_fused_val(tuple(keys))
        new_flat_state.append(
            (keys[:-2] + ('k_einsum', 'w'), _wrap_like(param, kv_val[0]))
        )
        new_flat_state.append(
            (keys[:-2] + ('v_einsum', 'w'), _wrap_like(param, kv_val[1]))
        )
      elif leaf_key == ('gate_proj', 'kernel') and layer_idx is not None:
        layers_gate[layer_idx] = (keys, param)
      elif leaf_key == ('up_proj', 'kernel') and layer_idx is not None:
        layers_up[layer_idx] = (keys, param)
      else:
        fused_val = _get_fused_val(tuple(keys))
        new_flat_state.append((keys, _wrap_like(param, fused_val)))
        if leaf_key == ('embedder', 'input_embedding') or (
            len(keys) >= 2 and keys[-2:] == ('embedder', 'input_embedding')
        ) or (len(keys) == 1 and keys[0] == 'input_embedding'):
          lm_head_val = jnp.transpose(fused_val, (1, 0))
          lm_head_keys = (
              keys[:-2] + ('lm_head', 'w')
              if len(keys) >= 2
              else ('lm_head', 'w')
          )
          new_flat_state.append((lm_head_keys, _wrap_like(param, lm_head_val)))

    for layer_idx, (gate_keys, gate_param) in layers_gate.items():
      if layer_idx not in layers_up:
        gate_val = _get_fused_val(tuple(gate_keys))
        new_flat_state.append((gate_keys, _wrap_like(gate_param, gate_val)))
        continue
      up_keys, _ = layers_up[layer_idx]

      gate_val = _get_fused_val(tuple(gate_keys))
      up_val = _get_fused_val(tuple(up_keys))

      if n_shards > 1:
        d, f = gate_val.shape
        if f % n_shards:
          raise ValueError(
              f'gate/up width {f} is not divisible by tensor_parallel_size'
              f' {n_shards}; cannot build the TP-interleaved gate_up_proj'
              ' layout.'
          )
        gate_val = gate_val.reshape(d, n_shards, f // n_shards)
        up_val = up_val.reshape(d, n_shards, f // n_shards)
        gate_up_val = jnp.concatenate([gate_val, up_val], axis=-1).reshape(
            d, 2 * f
        )
      else:
        gate_up_val = jnp.concatenate([gate_val, up_val], axis=-1)

      gate_up_keys = gate_keys[:-2] + ('gate_up_proj', 'kernel')
      new_flat_state.append((gate_up_keys, _wrap_like(gate_param, gate_up_val)))
    if total_lora_a > 0:
      logging.info(
          'LoRA fusion complete: fused %d / %d adapter pairs (rank=%s, scale=%.3f).',
          fused_lora_count,
          total_lora_a,
          rank,
          scale,
      )
      if fused_lora_count != total_lora_a:
        raise RuntimeError(
            f'LoRA fusion incomplete: fused {fused_lora_count} of {total_lora_a}'
            ' LoRA adapter pairs in preprocess_src_state! Refusing to evaluate'
            ' or train with partially unfused LoRA weights.'
        )
    params.clear()
    src_state = src_state.from_flat_path(new_flat_state)
  return src_state


VLLM_JAX_MAPPING: Dict[str, Any] = {
    'to_hf_mappings': TO_HF_MAPPINGS,
    'lora_to_hf_mappings': LORA_TO_HF_MAPPINGS,
    'to_hf_transpose_keys': TO_HF_TRANSPOSE_KEYS,
    'preprocess_src_state': preprocess_src_state,
}

__all__ = [
    'VLLM_JAX_MAPPING',
]
