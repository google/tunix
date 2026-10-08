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

"""Gemma4 dense text weights for the SGL-JAX backend.

Targets the 31B text decoder with a text-only actor and Gemma4ForCausalLM
rollout model. Other variants and multimodal weights are not covered.
"""

from flax import nnx

TO_HF_MAPPINGS = {
    'embedder.input_embedding': (
        'model.embed_tokens.embedding',
        ('tensor', None),
    ),
    'final_norm.scale': ('model.norm.weight', (None,)),
}
for src, dst in (
    ('pre_attention_norm.scale', 'input_layernorm.weight'),
    ('post_attention_norm.scale', 'post_attention_layernorm.weight'),
    ('pre_ffw_norm.scale', 'pre_feedforward_layernorm.weight'),
    ('post_ffw_norm.scale', 'post_feedforward_layernorm.weight'),
    ('skip_scale', 'layer_scalar'),
    ('attn._query_norm.scale', 'self_attn.q_norm.weight'),
    ('attn._key_norm.scale', 'self_attn.k_norm.weight'),
):
  TO_HF_MAPPINGS[f'layers.*.{src}'] = (f'model.layers.*.{dst}', (None,))
for src, dst, axes in (
    ('attn.q_einsum.w', 'self_attn.q_proj.weight', (None, 'tensor', None)),
    ('attn.k_einsum.w', 'self_attn.k_proj.weight', (None, 'tensor', None)),
    ('attn.v_einsum.w', 'self_attn.v_proj.weight', (None, 'tensor', None)),
    (
        'attn.attn_vec_einsum.w',
        'self_attn.o_proj.weight',
        ('tensor', None, None),
    ),
    ('mlp.gate_proj.kernel', 'mlp.gate_proj.weight', (None, 'tensor')),
    ('mlp.up_proj.kernel', 'mlp.up_proj.weight', (None, 'tensor')),
    ('mlp.down_proj.kernel', 'mlp.down_proj.weight', ('tensor', None)),
):
  TO_HF_MAPPINGS[f'layers.*.{src}'] = (f'model.layers.*.{dst}', axes)


def preprocess_src_state(src_state, tp_size=1):
  """Split fused KV without modifying the trainer's state or its buffers."""
  del tp_size
  if not hasattr(src_state, 'flat_state'):
    raise TypeError('Gemma4 SGL-JAX sync requires an NNX state.')
  flat = []
  for keys, param in src_state.flat_state():
    if keys[-2:] == ('kv_einsum', 'w'):
      value = param.value if hasattr(param, 'value') else param
      if value.ndim != 4 or value.shape[0] != 2:
        raise ValueError(f'Invalid Gemma4 KV shape at {keys}: {value.shape}')
      for name, part in zip(('k_einsum', 'v_einsum'), (value[0], value[1])):
        flat.append((keys[:-2] + (name, 'w'), nnx.Param(part)))
    else:
      flat.append((keys, param))
  return src_state.from_flat_path(flat)


SGLANG_JAX_MAPPING = {
    'to_hf_mappings': TO_HF_MAPPINGS,
    'to_hf_transpose_keys': {
        f'layers.*.attn.{name}.w': (1, 0, 2)
        for name in ('q_einsum', 'k_einsum', 'v_einsum')
    },
    'preprocess_src_state': preprocess_src_state,
}
