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
    'embedder.input_embedding': ('language_model.embed_tokens.weight', ('model', None)),
    'layers.*.pre_attention_norm.scale': (
        'language_model.layers.*.input_layernorm.weight',
        (None,),
    ),
    'layers.*.attn.q_einsum.w': (
        'language_model.layers.*.self_attn.q_proj.weight',
        (None, 'model', None),
    ),
    # `k_einsum` exists natively on k_eq_v (GLOBAL) layers; for the other
    # layers it is split out of `kv_einsum` by `preprocess_src_state`.
    'layers.*.attn.k_einsum.w': (
        'language_model.layers.*.self_attn.k_proj.weight',
        (None, 'model', None),
    ),
    'layers.*.attn.v_einsum.w': (
        'language_model.layers.*.self_attn.v_proj.weight',
        (None, 'model', None),
    ),
    'layers.*.attn._query_norm.scale': (
        'language_model.layers.*.self_attn.q_norm.weight',
        (None,),
    ),
    'layers.*.attn._key_norm.scale': (
        'language_model.layers.*.self_attn.k_norm.weight',
        (None,),
    ),
    'layers.*.attn.attn_vec_einsum.w': (
        'language_model.layers.*.self_attn.o_proj.weight',
        ('model', None, None),
    ),
    'layers.*.post_attention_norm.scale': (
        'language_model.layers.*.post_attention_layernorm.weight',
        (None,),
    ),
    'layers.*.pre_ffw_norm.scale': (
        'language_model.layers.*.pre_feedforward_layernorm.weight',
        (None,),
    ),
    'layers.*.mlp.gate_up_proj.kernel': (
        'language_model.layers.*.mlp.gate_up_proj.weight',
        (None, 'model'),
    ),
    'layers.*.mlp.down_proj.kernel': (
        'language_model.layers.*.mlp.down_proj.weight',
        ('model', None),
    ),
    'layers.*.post_ffw_norm.scale': (
        'language_model.layers.*.post_feedforward_layernorm.weight',
        (None,),
    ),
    'layers.*.skip_scale': (
        'language_model.layers.*.layer_scalar',
        (None,),
    ),
    'final_norm.scale': ('language_model.norm.weight', (None,)),
    'layers.*.moe_pre_ffw_norm.scale': (
        'language_model.layers.*.pre_feedforward_layernorm_2.weight',
        (None,),
    ),
    'layers.*.moe.router_logits': (
        'language_model.layers.*.router.proj.weight',
        (None, 'model'),
    ),
    'layers.*.moe.router_scale': (
        'language_model.layers.*.router.scale',
        (None,),
    ),
    'layers.*.moe.per_expert_scale': (
        'language_model.layers.*.router.per_expert_scale',
        (None,),
    ),
    'layers.*.moe.gating_einsum': (
        'language_model.layers.*.experts.kernel_gating_upproj_EDF',
        (None, None, 'model'),
    ),
    'layers.*.moe.linear': (
        'language_model.layers.*.experts.kernel_down_proj_EFD',
        ('model', None, None),
    ),
    'layers.*.dense_post_ffw_norm.scale': (
        'language_model.layers.*.post_feedforward_layernorm_1.weight',
        (None,),
    ),
    'layers.*.moe_post_ffw_norm.scale': (
        'language_model.layers.*.post_feedforward_layernorm_2.weight',
        (None,),
    ),
    'embedder.per_layer_input_embedding': (
        'language_model.embed_tokens_per_layer.weight',
        ('model', None),
    ),
    'embedder.per_layer_model_projection.w': (
        'language_model.per_layer_model_projection.weight',
        (None, 'model'),
    ),
    'embedder.per_layer_projection_norm.scale': (
        'language_model.per_layer_projection_norm.weight',
        (None,),
    ),
    'layers.*.per_layer_input_gate.w': (
        'language_model.layers.*.per_layer_input_gate.weight',
        (None, 'model'),
    ),
    'layers.*.per_layer_projection.w': (
        'language_model.layers.*.per_layer_projection.weight',
        ('model', None),
    ),
    'layers.*.post_per_layer_input_norm.scale': (
        'language_model.layers.*.post_per_layer_input_norm.weight',
        (None,),
    ),
}

LORA_TO_HF_MAPPINGS: Dict[str, MappingEntry] = {}

TO_HF_TRANSPOSE_KEYS = {
    'layers.*.attn.q_einsum.w': (1, 0, 2),
    'layers.*.attn.k_einsum.w': (1, 0, 2),
    'layers.*.attn.v_einsum.w': (1, 0, 2),
}


def preprocess_src_state(src_state: Any, tp_size: int = 1) -> Any:
  """Reshapes the Tunix state to match the vLLM (tpu-inference) pytree.

  * `attn.kv_einsum.w` (2, K, D, H) is split into `attn.k_einsum.w` and
    `attn.v_einsum.w` (K, D, H) each. tpu-inference >= 0.28 keeps separate
    `q_proj` / `k_proj` / `v_proj` params (the fused `qkv_proj` was removed in
    vllm-project/tpu-inference#3376). k_eq_v layers only carry `k_einsum` in
    Tunix and only `k_proj` in vLLM, so they pass through untouched.
  * `mlp.gate_proj.kernel` and `mlp.up_proj.kernel` are concatenated into
    `mlp.gate_up_proj.kernel` in vLLM's TP-interleaved `[gate_0, up_0, ...]`
    layout (matching `tpu_inference`'s `JaxMergedColumnParallelLinear`).
  """
  if not hasattr(src_state, 'flat_state'):
    return src_state

  new_flat_state = []
  layers_gate = {}
  layers_up = {}

  for keys, param in src_state.flat_state():
    src_key = '.'.join(str(k) for k in keys)
    if 'attn.kv_einsum.w' in src_key:
      val = param.value if hasattr(param, 'value') else param
      new_flat_state.append(
          (keys[:-2] + ('k_einsum', 'w'), _wrap_like(param, val[0]))
      )
      new_flat_state.append(
          (keys[:-2] + ('v_einsum', 'w'), _wrap_like(param, val[1]))
      )
    elif 'mlp.gate_proj.kernel' in src_key:
      layers_gate[keys[1]] = (keys, param)
    elif 'mlp.up_proj.kernel' in src_key:
      layers_up[keys[1]] = (keys, param)
    else:
      new_flat_state.append((keys, param))

  for layer_idx, (gate_keys, gate_param) in layers_gate.items():
    if layer_idx not in layers_up:
      new_flat_state.append((gate_keys, gate_param))
      continue
    _, up_param = layers_up[layer_idx]
    gate_val = gate_param.value if hasattr(gate_param, 'value') else gate_param
    up_val = up_param.value if hasattr(up_param, 'value') else up_param
    if tp_size > 1:
      d, f = gate_val.shape
      if f % tp_size:
        raise ValueError(
            f'gate/up width {f} is not divisible by tensor_parallel_size'
            f' {tp_size}; cannot build the TP-interleaved gate_up_proj layout.'
        )
      gate_val = gate_val.reshape(d, tp_size, f // tp_size)
      up_val = up_val.reshape(d, tp_size, f // tp_size)
      gate_up_val = jnp.concatenate([gate_val, up_val], axis=-1).reshape(
          d, 2 * f
      )
    else:
      gate_up_val = jnp.concatenate([gate_val, up_val], axis=-1)
    new_flat_state.append((
        gate_keys[:-2] + ('gate_up_proj', 'kernel'),
        _wrap_like(gate_param, gate_up_val),
    ))

  return src_state.from_flat_path(new_flat_state)


def _wrap_like(param: Any, val: Any) -> Any:
  """Wraps `val` in `nnx.Param` iff `param` is a variable-like object."""
  return nnx.Param(val) if hasattr(param, 'value') else val


VLLM_JAX_MAPPING: Dict[str, Any] = {
    'to_hf_mappings': TO_HF_MAPPINGS,
    'lora_to_hf_mappings': LORA_TO_HF_MAPPINGS,
    'to_hf_transpose_keys': TO_HF_TRANSPOSE_KEYS,
    'preprocess_src_state': preprocess_src_state,
}

__all__ = [
    'VLLM_JAX_MAPPING',
]
