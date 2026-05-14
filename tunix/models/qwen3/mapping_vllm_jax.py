# Copyright 2026 Google LLC
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

"""vLLM JAX backend mappings for Qwen3 models."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from tunix.generate import mappings as mappings_lib
from tunix.generate import param_mapping as param_mapping_lib

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]


def _rule(
    source: str,
    target: str,
    transforms: Tuple[param_mapping_lib.Transform, ...] = (),
) -> param_mapping_lib.OperationRule:
  return param_mapping_lib.OperationRule(
      name=f'qwen3_vllm::{source}',
      source_patterns=(source,),
      target_patterns=(target,),
      transforms=transforms,
  )


VLLM_JAX_WEIGHT_MAPPINGS: Dict[str, MappingEntry] = {
  'embedder.input_embedding': ('model.embed_tokens.weight', ('model', None)),
  'layers.*.input_layernorm.w': (
    'model.layers.*.input_layernorm.weight',
    (None,),
  ),
  'layers.*.mlp.down_proj.kernel': (
    'model.layers.*.mlp.down_proj.weight',
    ('model', None),
  ),
  'layers.*.mlp.gate_proj.kernel': (
    'model.layers.*.mlp.gate_proj.weight',
    (None, 'model'),
  ),
  'layers.*.mlp.up_proj.kernel': (
    'model.layers.*.mlp.up_proj.weight',
    (None, 'model'),
  ),
  'layers.*.post_attention_layernorm.w': (
    'model.layers.*.post_attention_layernorm.weight',
    (None,),
  ),
  'layers.*.attn.k_proj.w': (
    'model.layers.*.self_attn.k_proj.weight',
    (None, 'model', None),
  ),
  'layers.*.attn.k_norm.w': (
    'model.layers.*.self_attn.k_norm.weight',
    (None, 'model', None),
  ),
  'layers.*.attn.o_proj.w': (
    'model.layers.*.self_attn.o_proj.weight',
    ('model', None, None),
  ),
  'layers.*.attn.q_proj.w': (
    'model.layers.*.self_attn.q_proj.weight',
    (None, 'model', None),
  ),
  'layers.*.attn.q_norm.w': (
    'model.layers.*.self_attn.q_norm.weight',
    (None, 'model', None),
  ),
  'layers.*.attn.v_proj.w': (
    'model.layers.*.self_attn.v_proj.weight',
    (None, 'model', None),
  ),
  'final_norm.w': ('model.norm.weight', (None,)),
  'lm_head.w': ('lm_head.weight', (None, 'model')),
}


VLLM_JAX_LORA_MAPPINGS: Dict[str, MappingEntry] = {
  'layers.*.mlp.gate_proj.kernel_lora_a': (
    'model.layers.*.mlp.gate_proj.weight_lora_a',
    (None, None),
  ),
  'layers.*.mlp.gate_proj.kernel_lora_b': (
    'model.layers.*.mlp.gate_proj.weight_lora_b',
    (None, 'model'),
  ),
  'layers.*.mlp.up_proj.kernel_lora_a': (
    'model.layers.*.mlp.up_proj.weight_lora_a',
    (None, None),
  ),
  'layers.*.mlp.up_proj.kernel_lora_b': (
    'model.layers.*.mlp.up_proj.weight_lora_b',
    (None, 'model'),
  ),
  'layers.*.mlp.down_proj.kernel_lora_a': (
    'model.layers.*.mlp.down_proj.weight_lora_a',
    ('model', None),
  ),
  'layers.*.mlp.down_proj.kernel_lora_b': (
    'model.layers.*.mlp.down_proj.weight_lora_b',
    (None, None),
  ),
  'layers.*.attn.q_proj.w_lora_a': (
    'model.layers.*.self_attn.q_proj.weight_lora_a',
    ('model', None),
  ),
  'layers.*.attn.q_proj.w_lora_b': (
    'model.layers.*.self_attn.q_proj.weight_lora_b',
    (None, None),
  ),
  'layers.*.attn.k_proj.w_lora_a': (
    'model.layers.*.self_attn.k_proj.weight_lora_a',
    ('model', None),
  ),
  'layers.*.attn.k_proj.w_lora_b': (
    'model.layers.*.self_attn.k_proj.weight_lora_b',
    (None, None),
  ),
  'layers.*.attn.v_proj.w_lora_a': (
    'model.layers.*.self_attn.v_proj.weight_lora_a',
    ('model', None),
  ),
  'layers.*.attn.v_proj.w_lora_b': (
    'model.layers.*.self_attn.v_proj.weight_lora_b',
    (None, None),
  ),
  'layers.*.attn.o_proj.w_lora_a': (
    'model.layers.*.self_attn.o_proj.weight_lora_a',
    ('model', None),
  ),
  'layers.*.attn.o_proj.w_lora_b': (
    'model.layers.*.self_attn.o_proj.weight_lora_b',
    (None, None),
  ),
}


VLLM_JAX_WEIGHT_RULES: Tuple[param_mapping_lib.OperationRule, ...] = (
  tuple(_rule(source, target) for source, (target, _) in VLLM_JAX_WEIGHT_MAPPINGS.items())
)


VLLM_JAX_LORA_RULES: Tuple[param_mapping_lib.OperationRule, ...] = (
  tuple(_rule(source, target) for source, (target, _) in VLLM_JAX_LORA_MAPPINGS.items())
)


def _mapping_table(
  mapping_table: Dict[str, MappingEntry],
) -> Dict[str, MappingEntry]:
  return dict(mapping_table)


def _transpose_table(
    rules: Tuple[param_mapping_lib.OperationRule, ...],
) -> Dict[str, Tuple[int, ...]] | None:
  transpose_rules = {}
  for rule in rules:
    for transform in rule.transforms:
      if transform.kind == 'transpose':
        transpose_rules[rule.source_patterns[0]] = tuple(transform.args['axes'])
  return transpose_rules or None


def to_hf_mappings() -> Dict[str, MappingEntry]:
  """Returns the vanilla-to-vLLM key mappings for Qwen3."""
  return _mapping_table(VLLM_JAX_WEIGHT_MAPPINGS)


def lora_to_hf_mappings() -> Dict[str, MappingEntry]:
  """Returns the LoRA-to-vLLM key mappings for Qwen3."""
  return _mapping_table(VLLM_JAX_LORA_MAPPINGS)


def to_hf_transpose_keys() -> Dict[str, Tuple[int, ...]] | None:
  """Returns transpose rules used during Qwen3 weight sync to vLLM."""
  return _transpose_table(VLLM_JAX_WEIGHT_RULES)


def to_hf_hook_fns() -> Dict[str, Any] | None:
  """Returns optional hook functions used during Qwen3 weight sync."""
  return None


def to_hf_operation_rules() -> Tuple[param_mapping_lib.OperationRule, ...]:
  """Returns planner-ready explicit rules for Qwen3 vLLM sync."""
  return VLLM_JAX_WEIGHT_RULES


def make_vllm_jax_mapping_spec() -> param_mapping_lib.MappingSpec:
  """Returns the default DSL-first MappingSpec for Qwen3 to vLLM transfer."""
  return param_mapping_lib.MappingSpec(
      model_type='qwen3_vllm_jax',
      operation_rules=to_hf_operation_rules(),
  )


def make_vllm_jax_mapping_config() -> mappings_lib.MappingConfig:
  """Returns the legacy runtime MappingConfig derived from the DSL rules."""
  return mappings_lib.MappingConfig.build(
      {
          'to_hf_mappings': to_hf_mappings,
          'lora_to_hf_mappings': lora_to_hf_mappings,
          'to_hf_transpose_keys': to_hf_transpose_keys,
          'to_hf_hook_fns': to_hf_hook_fns,
      }
  )


VLLM_JAX_MAPPING: Dict[str, Any] = {
    'to_hf_mappings': to_hf_mappings(),
    'lora_to_hf_mappings': lora_to_hf_mappings(),
    'to_hf_transpose_keys': to_hf_transpose_keys(),
    'to_hf_hook_fns': to_hf_hook_fns(),
}


__all__ = [
    'make_vllm_jax_mapping_config',
    'make_vllm_jax_mapping_spec',
    'VLLM_JAX_LORA_RULES',
    'VLLM_JAX_MAPPING',
    'VLLM_JAX_WEIGHT_RULES',
    'lora_to_hf_mappings',
    'to_hf_hook_fns',
    'to_hf_mappings',
    'to_hf_operation_rules',
    'to_hf_transpose_keys',
]