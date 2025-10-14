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

"""vLLM JAX backend mappings for Qwen2 models."""

from __future__ import annotations

from typing import Any, Dict, Tuple

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]


TO_HF_MAPPINGS: Dict[str, MappingEntry] = {
    'embedder.input_embedding': ('embed.embedding', ('model', None)),
    'layers.*.input_layernorm.w': (
        'model.layers.*.input_layernorm.scale',
        (None,),
    ),
    'layers.*.mlp.down_proj.kernel': (
        'model.layers.*.mlp.down_proj.kernel',
        ('model', None),
    ),
    'layers.*.mlp.gate_proj.kernel': (
        'model.layers.*.mlp.gate_proj.kernel',
        (None, 'model'),
    ),
    'layers.*.mlp.up_proj.kernel': (
        'model.layers.*.mlp.up_proj.kernel',
        (None, 'model'),
    ),
    'layers.*.post_attention_layernorm.w': (
        'model.layers.*.post_attention_layernorm.scale',
        (None,),
    ),
    'layers.*.attn.k_proj.w': (
        'model.layers.*.self_attn.k_proj.kernel',
        (None, 'model', None),
    ),
    'layers.*.attn.o_proj.w': (
        'model.layers.*.self_attn.o_proj.kernel',
        ('model', None, None),
    ),
    'layers.*.attn.q_proj.w': (
        'model.layers.*.self_attn.q_proj.kernel',
        (None, 'model', None),
    ),
    'layers.*.attn.v_proj.w': (
        'model.layers.*.self_attn.v_proj.kernel',
        (None, 'model', None),
    ),
    'layers.*.attn.q_bias': (
        'model.layers.*.self_attn.q_proj.bias',
        ('model', None),
    ),
    'layers.*.attn.k_bias': (
        'model.layers.*.self_attn.k_proj.bias',
        ('model', None),
    ),
    'layers.*.attn.v_bias': (
        'model.layers.*.self_attn.v_proj.bias',
        ('model', None),
    ),
    'final_norm.w': ('model.norm.scale', (None,)),
    'lm_head.w': ('lm_head', (None, 'model')),
}


LORA_TO_HF_MAPPINGS: Dict[str, MappingEntry] = {
    'layers.*.mlp.gate_proj.kernel_lora_a': (
        'model.layers.*.mlp.gate_proj.kernel_lora_a',
        (None, None),
    ),
    'layers.*.mlp.gate_proj.kernel_lora_b': (
        'model.layers.*.mlp.gate_proj.kernel_lora_b',
        (None, 'model'),
    ),
    'layers.*.mlp.up_proj.kernel_lora_a': (
        'model.layers.*.mlp.up_proj.kernel_lora_a',
        (None, None),
    ),
    'layers.*.mlp.up_proj.kernel_lora_b': (
        'model.layers.*.mlp.up_proj.kernel_lora_b',
        (None, 'model'),
    ),
    'layers.*.mlp.down_proj.kernel_lora_a': (
        'model.layers.*.mlp.down_proj.kernel_lora_a',
        ('model', None),
    ),
    'layers.*.mlp.down_proj.kernel_lora_b': (
        'model.layers.*.mlp.down_proj.kernel_lora_b',
        (None, None),
    ),
    'layers.*.attn.q_proj.w_lora_a': (
        'layers.*.self_attn.q_proj.kernel_lora_a',
        ('model', None),
    ),
    'layers.*.attn.q_proj.w_lora_b': (
        'layers.*.self_attn.q_proj.kernel_lora_b',
        (None, None),
    ),
    'layers.*.attn.k_proj.w_lora_a': (
        'layers.*.self_attn.k_proj.kernel_lora_a',
        ('model', None),
    ),
    'layers.*.attn.k_proj.w_lora_b': (
        'layers.*.self_attn.k_proj.kernel_lora_b',
        (None, None),
    ),
    'layers.*.attn.v_proj.w_lora_a': (
        'layers.*.self_attn.v_proj.kernel_lora_a',
        ('model', None),
    ),
    'layers.*.attn.v_proj.w_lora_b': (
        'layers.*.self_attn.v_proj.kernel_lora_b',
        (None, None),
    ),
    'layers.*.attn.o_proj.w_lora_a': (
        'layers.*.self_attn.o_proj.kernel_lora_a',
        ('model', None),
    ),
    'layers.*.attn.o_proj.w_lora_b': (
        'layers.*.self_attn.o_proj.kernel_lora_b',
        (None, None),
    ),
    'layers.*.attn.q_bias_lora_a': (
        'model.layers.*.self_attn.q_proj.bias_lora_a',
        ('model', None),
    ),
    'layers.*.attn.q_bias_lora_b': (
        'model.layers.*.self_attn.q_proj.bias_lora_b',
        ('model', None),
    ),
    'layers.*.attn.k_bias_lora_a': (
        'model.layers.*.self_attn.k_proj.bias_lora_a',
        ('model', None),
    ),
    'layers.*.attn.k_bias_lora_b': (
        'model.layers.*.self_attn.k_proj.bias_lora_b',
        ('model', None),
    ),
    'layers.*.attn.v_bias_lora_a': (
        'model.layers.*.self_attn.v_proj.bias_lora_a',
        ('model', None),
    ),
    'layers.*.attn.v_bias_lora_b': (
        'model.layers.*.self_attn.v_proj.bias_lora_b',
        ('model', None),
    ),
}


VLLM_JAX_MAPPING: Dict[str, Any] = {
    'to_hf_mappings': TO_HF_MAPPINGS,
    'lora_to_hf_mappings': LORA_TO_HF_MAPPINGS,
    'to_hf_transpose_keys': {'embedding': (1, 0)},
    'to_hf_hook_fns': None,
}

__all__ = [
    'VLLM_JAX_MAPPING',
]
