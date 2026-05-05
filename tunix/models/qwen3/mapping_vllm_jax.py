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

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from tunix.generate import mappings as mappings_lib
from tunix.generate import utils as transfer_utils

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]


@dataclass(frozen=True)
class MappingRuleDef:
  """Declarative Qwen3-to-vLLM mapping rule."""

  source: str
  target: str
  sharding: Sharding
  transforms: Tuple[transfer_utils.Transform, ...] = ()


VLLM_JAX_WEIGHT_RULE_DEFS: Tuple[MappingRuleDef, ...] = (
    MappingRuleDef(
        source='embedder.input_embedding',
        target='model.embed_tokens.weight',
        sharding=('model', None),
    ),
    MappingRuleDef(
        source='layers.*.input_layernorm.w',
        target='model.layers.*.input_layernorm.weight',
        sharding=(None,),
    ),
    MappingRuleDef(
        source='layers.*.mlp.down_proj.kernel',
        target='model.layers.*.mlp.down_proj.weight',
        sharding=('model', None),
    ),
    MappingRuleDef(
        source='layers.*.mlp.gate_proj.kernel',
        target='model.layers.*.mlp.gate_proj.weight',
        sharding=(None, 'model'),
    ),
    MappingRuleDef(
        source='layers.*.mlp.up_proj.kernel',
        target='model.layers.*.mlp.up_proj.weight',
        sharding=(None, 'model'),
    ),
    MappingRuleDef(
        source='layers.*.post_attention_layernorm.w',
        target='model.layers.*.post_attention_layernorm.weight',
        sharding=(None,),
    ),
    MappingRuleDef(
        source='layers.*.attn.k_proj.w',
        target='model.layers.*.self_attn.k_proj.weight',
        sharding=(None, 'model', None),
    ),
    MappingRuleDef(
        source='layers.*.attn.k_norm.w',
        target='model.layers.*.self_attn.k_norm.weight',
        sharding=(None, 'model', None),
    ),
    MappingRuleDef(
        source='layers.*.attn.o_proj.w',
        target='model.layers.*.self_attn.o_proj.weight',
        sharding=('model', None, None),
    ),
    MappingRuleDef(
        source='layers.*.attn.q_proj.w',
        target='model.layers.*.self_attn.q_proj.weight',
        sharding=(None, 'model', None),
    ),
    MappingRuleDef(
        source='layers.*.attn.q_norm.w',
        target='model.layers.*.self_attn.q_norm.weight',
        sharding=(None, 'model', None),
    ),
    MappingRuleDef(
        source='layers.*.attn.v_proj.w',
        target='model.layers.*.self_attn.v_proj.weight',
        sharding=(None, 'model', None),
    ),
    MappingRuleDef(
        source='final_norm.w',
        target='model.norm.weight',
        sharding=(None,),
    ),
    MappingRuleDef(
        source='lm_head.w',
        target='lm_head.weight',
        sharding=(None, 'model'),
    ),
)


VLLM_JAX_LORA_RULE_DEFS: Tuple[MappingRuleDef, ...] = (
    MappingRuleDef(
        source='layers.*.mlp.gate_proj.kernel_lora_a',
        target='model.layers.*.mlp.gate_proj.weight_lora_a',
        sharding=(None, None),
    ),
    MappingRuleDef(
        source='layers.*.mlp.gate_proj.kernel_lora_b',
        target='model.layers.*.mlp.gate_proj.weight_lora_b',
        sharding=(None, 'model'),
    ),
    MappingRuleDef(
        source='layers.*.mlp.up_proj.kernel_lora_a',
        target='model.layers.*.mlp.up_proj.weight_lora_a',
        sharding=(None, None),
    ),
    MappingRuleDef(
        source='layers.*.mlp.up_proj.kernel_lora_b',
        target='model.layers.*.mlp.up_proj.weight_lora_b',
        sharding=(None, 'model'),
    ),
    MappingRuleDef(
        source='layers.*.mlp.down_proj.kernel_lora_a',
        target='model.layers.*.mlp.down_proj.weight_lora_a',
        sharding=('model', None),
    ),
    MappingRuleDef(
        source='layers.*.mlp.down_proj.kernel_lora_b',
        target='model.layers.*.mlp.down_proj.weight_lora_b',
        sharding=(None, None),
    ),
    MappingRuleDef(
        source='layers.*.attn.q_proj.w_lora_a',
        target='model.layers.*.self_attn.q_proj.weight_lora_a',
        sharding=('model', None),
    ),
    MappingRuleDef(
        source='layers.*.attn.q_proj.w_lora_b',
        target='model.layers.*.self_attn.q_proj.weight_lora_b',
        sharding=(None, None),
    ),
    MappingRuleDef(
        source='layers.*.attn.k_proj.w_lora_a',
        target='model.layers.*.self_attn.k_proj.weight_lora_a',
        sharding=('model', None),
    ),
    MappingRuleDef(
        source='layers.*.attn.k_proj.w_lora_b',
        target='model.layers.*.self_attn.k_proj.weight_lora_b',
        sharding=(None, None),
    ),
    MappingRuleDef(
        source='layers.*.attn.v_proj.w_lora_a',
        target='model.layers.*.self_attn.v_proj.weight_lora_a',
        sharding=('model', None),
    ),
    MappingRuleDef(
        source='layers.*.attn.v_proj.w_lora_b',
        target='model.layers.*.self_attn.v_proj.weight_lora_b',
        sharding=(None, None),
    ),
    MappingRuleDef(
        source='layers.*.attn.o_proj.w_lora_a',
        target='model.layers.*.self_attn.o_proj.weight_lora_a',
        sharding=('model', None),
    ),
    MappingRuleDef(
        source='layers.*.attn.o_proj.w_lora_b',
        target='model.layers.*.self_attn.o_proj.weight_lora_b',
        sharding=(None, None),
    ),
)


def _mapping_table(
    rule_defs: Tuple[MappingRuleDef, ...],
) -> Dict[str, MappingEntry]:
  return {
      rule.source: (rule.target, rule.sharding)
      for rule in rule_defs
  }


def _transpose_table(
    rule_defs: Tuple[MappingRuleDef, ...],
) -> Dict[str, Tuple[int, ...]] | None:
  transpose_rules = {}
  for rule in rule_defs:
    for transform in rule.transforms:
      if transform.kind == 'transpose':
        transpose_rules[rule.source] = tuple(transform.args['axes'])
  return transpose_rules or None


def to_hf_mappings() -> Dict[str, MappingEntry]:
  """Returns the vanilla-to-vLLM key mappings for Qwen3."""
  return _mapping_table(VLLM_JAX_WEIGHT_RULE_DEFS)


def lora_to_hf_mappings() -> Dict[str, MappingEntry]:
  """Returns the LoRA-to-vLLM key mappings for Qwen3."""
  return _mapping_table(VLLM_JAX_LORA_RULE_DEFS)


def to_hf_transpose_keys() -> Dict[str, Tuple[int, ...]] | None:
  """Returns transpose rules used during Qwen3 weight sync to vLLM."""
  return _transpose_table(VLLM_JAX_WEIGHT_RULE_DEFS)


def to_hf_hook_fns() -> Dict[str, Any] | None:
  """Returns optional hook functions used during Qwen3 weight sync."""
  return None


def to_hf_weight_rules() -> Tuple[transfer_utils.WeightRule, ...]:
  """Returns planner-ready explicit rules for Qwen3 vLLM sync."""
  return tuple(
      transfer_utils.WeightRule(
          name=f'qwen3_vllm::{rule.source}',
          source=rule.source,
          target=rule.target,
          transforms=rule.transforms,
      )
      for rule in VLLM_JAX_WEIGHT_RULE_DEFS
  )


def make_vllm_jax_mapping_spec() -> transfer_utils.MappingSpec:
  """Returns the default DSL-first MappingSpec for Qwen3 to vLLM transfer."""
  return transfer_utils.MappingSpec(
      model_type='qwen3_vllm_jax',
      weight_rules=to_hf_weight_rules(),
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
    'MappingRuleDef',
    'VLLM_JAX_LORA_RULE_DEFS',
    'VLLM_JAX_MAPPING',
    'VLLM_JAX_WEIGHT_RULE_DEFS',
    'lora_to_hf_mappings',
    'to_hf_hook_fns',
    'to_hf_mappings',
    'to_hf_transpose_keys',
    'to_hf_weight_rules',
]