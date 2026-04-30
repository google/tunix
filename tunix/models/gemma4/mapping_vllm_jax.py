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

import re
import sys
from typing import Any, Dict, Tuple
import jax.numpy as jnp

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]


class WildcardDict(dict):
  """Custom dictionary that supports wildcard/regex key matching."""

  def __contains__(self, key: Any) -> bool:
    for k in self.keys():
      pattern = '^' + re.escape(k).replace('\\*', '.*') + '$'
      if re.match(pattern, key):
        return True
    return False

  def __getitem__(self, key: Any) -> Any:
    for k, v in self.items():
      pattern = '^' + re.escape(k).replace('\\*', '.*') + '$'
      if re.match(pattern, key):
        return v
    raise KeyError(key)


class DynamicMappingsDict(dict):
  """Custom dictionary that generates exact regex patterns based on source keys."""

  def items(self) -> list[tuple[str, Any]]:
    frame = sys._getframe(1)
    src_state = frame.f_locals.get('src_state')
    if src_state is not None:
      src_keys = {
          '.'.join(str(k) for k in keys) for keys, _ in src_state.flat_state()
      }
      res = []
      for k, v in super().items():
        tgt, sharding = v
        if '*' in k:
          # Extract matching layer indices from src_keys
          pattern = '^' + re.escape(k).replace('\\*', r'(\d+)') + '$'
          layers = []
          for sk in src_keys:
            m = re.match(pattern, sk)
            if m:
              layers.append(m.group(1))
          if layers:
            layers_pat = '(' + '|'.join(layers) + ')'
            new_tgt = tgt.replace('.', r'\.').replace('*', layers_pat)
            res.append((k, (new_tgt, sharding)))
          else:
            res.append((k, v))
        else:
          res.append((k, v))
      return res
    return list(super().items())


def _kv_hook(val: jnp.ndarray) -> jnp.ndarray:
  # val has shape (2, num_kv_heads, embed_dim, head_dim)
  k_val = jnp.transpose(val[0], (1, 0, 2))
  v_val = jnp.transpose(val[1], (1, 0, 2))

  frame = sys._getframe(1)
  dst_state = frame.f_locals['dst_state']
  flat_src_key = frame.f_locals['flat_src_key']

  match = re.match(r'layers\.(\d+)\.attn\.kv_einsum\.w', flat_src_key)
  if match:
    layer_idx = int(match.group(1))
    for k, v in dst_state.flat_state():
      if (
          len(k) >= 6
          and k[-6:-4] == ('layers', str(layer_idx))
          and 'v_proj' in k
      ):
        v.value = v_val
        break

  return k_val


def _moe_gate_up_hook(val: jnp.ndarray) -> jnp.ndarray:
  gate_val = val[:, 0, :, :]
  up_val = val[:, 1, :, :]

  frame = sys._getframe(1)
  dst_state = frame.f_locals['dst_state']
  flat_src_key = frame.f_locals['flat_src_key']

  match = re.match(r'layers\.(\d+)\.moe\.gating_einsum', flat_src_key)
  if match:
    layer_idx = int(match.group(1))
    for k, v in dst_state.flat_state():
      if (
          len(k) >= 6
          and k[-6:-4] == ('layers', str(layer_idx))
          and 'kernel_up_proj_EDF' in k
      ):
        v.value = up_val
        break

  return gate_val


def _moe_down_hook(val: jnp.ndarray) -> jnp.ndarray:
  return jnp.transpose(val, (0, 2, 1))


TO_HF_MAPPINGS = DynamicMappingsDict({
    'embedder.input_embedding': ('model.embed_tokens.weight', ('model', None)),
    'layers.*.pre_attention_norm.scale': (
        'model.layers.*.input_layernorm.weight',
        (None,),
    ),
    'layers.*.attn.q_einsum.w': (
        'model.layers.*.self_attn.q_proj.weight',
        (None, 'model', None),
    ),
    'layers.*.attn._query_norm.scale': (
        'model.layers.*.self_attn.q_norm.weight',
        (None,),
    ),
    'layers.*.attn.k_einsum.w': (
        'model.layers.*.self_attn.k_proj.weight',
        (None, 'model', None),
    ),
    'layers.*.attn.kv_einsum.w': (
        'model.layers.*.self_attn.k_proj.weight',
        (None, 'model', None),
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
    'layers.*.mlp.gate_proj.kernel': (
        'model.layers.*.mlp.gate_proj.weight',
        (None, 'model'),
    ),
    'layers.*.mlp.up_proj.kernel': (
        'model.layers.*.mlp.up_proj.weight',
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
        'model.layers.*.experts.kernel_gating_EDF',
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
})

LORA_TO_HF_MAPPINGS: Dict[str, MappingEntry] = {}

TO_HF_HOOK_FNS = WildcardDict({
    'layers.*.attn.kv_einsum.w': _kv_hook,
    'layers.*.moe.gating_einsum': _moe_gate_up_hook,
    'layers.*.moe.linear': _moe_down_hook,
})

TO_HF_TRANSPOSE_KEYS = WildcardDict({
    'layers.*.attn.q_einsum.w': (1, 0, 2),
    'layers.*.attn.k_einsum.w': (1, 0, 2),
})

VLLM_JAX_MAPPING: Dict[str, Any] = {
    'to_hf_mappings': TO_HF_MAPPINGS,
    'lora_to_hf_mappings': LORA_TO_HF_MAPPINGS,
    'to_hf_transpose_keys': TO_HF_TRANSPOSE_KEYS,
    'to_hf_hook_fns': TO_HF_HOOK_FNS,
}

__all__ = [
    'VLLM_JAX_MAPPING',
]
