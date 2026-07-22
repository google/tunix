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
import jax.numpy as jnp

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]


TO_HF_MAPPINGS = {
    'embedder.input_embedding': ('model.embed_tokens.weight', ('model', None)),
    'layers.*.pre_attention_norm.scale': (
        'model.layers.*.input_layernorm.weight',
        (None,),
    ),
    'layers.*.attn.q_einsum.w': (
        'model.layers.*.self_attn.q_proj.weight',
        (None, 'model', None),
    ),
    'layers.*.attn.k_einsum.w': (
        'model.layers.*.self_attn.k_proj.weight',
        (None, 'model', None),
    ),
    'layers.*.attn.qkv_einsum.w': (
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
    'layers.*.mlp.gate_up_proj.kernel': (
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


def preprocess_src_state(src_state: Any) -> Any:
  """Fuses Q/K/V and MLP gate/up projections in the source state."""
  if hasattr(src_state, 'flat_state'):
    flat_state = list(src_state.flat_state())
    new_flat_state = []

    layers_q = {}
    layers_k = {}
    layers_kv = {}
    layers_gate = {}
    layers_up = {}

    for keys, param in flat_state:
      src_key = '.'.join(str(k) for k in keys)
      if 'attn.q_einsum.w' in src_key:
        layer_idx = keys[1]
        layers_q[layer_idx] = (keys, param)
      elif 'attn.k_einsum.w' in src_key:
        layer_idx = keys[1]
        layers_k[layer_idx] = (keys, param)
      elif 'attn.kv_einsum.w' in src_key:
        layer_idx = keys[1]
        layers_kv[layer_idx] = (keys, param)
      elif 'mlp.gate_proj.kernel' in src_key:
        layer_idx = keys[1]
        layers_gate[layer_idx] = (keys, param)
      elif 'mlp.up_proj.kernel' in src_key:
        layer_idx = keys[1]
        layers_up[layer_idx] = (keys, param)
      else:
        new_flat_state.append((keys, param))

    sample_kv_val = None
    if layers_kv:
      sample_kv_val = next(iter(layers_kv.values()))[1]
      if hasattr(sample_kv_val, 'value'):
        sample_kv_val = sample_kv_val.value

    for layer_idx in layers_q:
      q_keys, q_param = layers_q[layer_idx]
      q_val = q_param.value if hasattr(q_param, 'value') else q_param
      hidden_size = q_val.shape[1]
      q_val_t = jnp.reshape(jnp.transpose(q_val, (1, 0, 2)), (hidden_size, -1))

      if layer_idx in layers_kv:
        _, kv_param = layers_kv[layer_idx]
        kv_val = kv_param.value if hasattr(kv_param, 'value') else kv_param
        k_val = kv_val[0]
        v_val = kv_val[1]

        k_val_t = jnp.reshape(
            jnp.transpose(k_val, (1, 0, 2)), (hidden_size, -1)
        )
        v_val_t = jnp.reshape(
            jnp.transpose(v_val, (1, 0, 2)), (hidden_size, -1)
        )

        qkv_val = jnp.concatenate([q_val_t, k_val_t, v_val_t], axis=-1)
        qkv_keys = q_keys[:-2] + ('qkv_einsum', 'w')
        if hasattr(q_param, 'value'):
          new_flat_state.append((qkv_keys, nnx.Param(qkv_val)))
        else:
          new_flat_state.append((qkv_keys, qkv_val))
      elif layer_idx in layers_k:
        k_keys, k_param = layers_k[layer_idx]
        new_flat_state.append((q_keys, q_param))
        new_flat_state.append((k_keys, k_param))
      else:
        # KV-shared layer
        k_val = jnp.zeros_like(sample_kv_val[0])
        v_val = jnp.zeros_like(sample_kv_val[1])
        k_val_t = jnp.reshape(
            jnp.transpose(k_val, (1, 0, 2)), (hidden_size, -1)
        )
        v_val_t = jnp.reshape(
            jnp.transpose(v_val, (1, 0, 2)), (hidden_size, -1)
        )

        qkv_val = jnp.concatenate([q_val_t, k_val_t, v_val_t], axis=-1)
        qkv_keys = q_keys[:-2] + ('qkv_einsum', 'w')
        if hasattr(q_param, 'value'):
          new_flat_state.append((qkv_keys, nnx.Param(qkv_val)))
        else:
          new_flat_state.append((qkv_keys, qkv_val))

    for layer_idx in layers_gate:
      gate_keys, gate_param = layers_gate[layer_idx]
      _, up_param = layers_up[layer_idx]

      gate_val = (
          gate_param.value if hasattr(gate_param, 'value') else gate_param
      )
      up_val = up_param.value if hasattr(up_param, 'value') else up_param

      gate_up_val = jnp.concatenate([gate_val, up_val], axis=-1)

      gate_up_keys = gate_keys[:-2] + ('gate_up_proj', 'kernel')
      if hasattr(gate_param, 'value'):
        new_flat_state.append((gate_up_keys, nnx.Param(gate_up_val)))
      else:
        new_flat_state.append((gate_up_keys, gate_up_val))
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
