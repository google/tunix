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

from typing import Any, Dict, Protocol, Tuple, runtime_checkable

from tunix.models import hf_mapping

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]


@runtime_checkable
class Qwen3HfConfig(Protocol):
  """Minimal Hugging Face Qwen3 config contract."""

  vocab_size: int
  hidden_size: int
  intermediate_size: int
  num_hidden_layers: int
  num_attention_heads: int
  num_key_value_heads: int | None
  head_dim: int | None
  tie_word_embeddings: bool
  attention_bias: bool


TO_HF_MAPPINGS: Dict[str, MappingEntry] = {
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
    'final_norm.w': ('model.norm.weight', (None,)),
    'lm_head.w': ('lm_head.weight', (None, 'model')),
}


LORA_TO_HF_MAPPINGS: Dict[str, MappingEntry] = {
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


VLLM_JAX_MAPPING: Dict[str, Any] = {
    'to_hf_mappings': TO_HF_MAPPINGS,
    'lora_to_hf_mappings': LORA_TO_HF_MAPPINGS,
    'to_hf_transpose_keys': {'embedding': (1, 0)},
    'to_hf_hook_fns': None,
}


def _validate_positive_int(name: str, value: int) -> int:
  """Validates that `value` is a strictly positive int (not bool)."""
  if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise ValueError(f'{name} must be a positive int, got {value!r}')
  return value


def _head_repeat_factor(heads: int, sharding_size: int) -> int:
  """Computes head replication factor matching `tpu_inference` sharding."""
  if heads >= sharding_size:
    if heads % sharding_size != 0:
      raise ValueError(
          f'Number of heads ({heads}) must be divisible by sharding_size'
          f' ({sharding_size}).'
      )
    return 1
  if sharding_size % heads != 0:
    raise ValueError(
        f'sharding_size ({sharding_size}) must be divisible by number of heads'
        f' ({heads}).'
    )
  return sharding_size // heads


def _qkv_weight_ops(
    heads: int,
    head_dim: int,
    d_model: int,
    pad_h: int,
    rep: int,
) -> tuple[dict[str, dict[str, Any]], ...]:
  """Builds 12-Op steps for HF `[H*D_h, D]` -> vLLM-JAX `[D, H*rep, H_pad]`."""
  ops: list[dict[str, dict[str, Any]]] = [{
      'reshape': {
          'src_shape': [heads * head_dim, d_model],
          'dst_shape': [heads, head_dim, d_model],
      }
  }]
  if pad_h > 0:
    ops.append({'pad': {'axis': 1, 'sizes': [0, pad_h]}})
  ops.append({'transpose': {'axes': [2, 0, 1]}})
  if rep > 1:
    ops.append({'repeat': {'axis': 1, 'times': rep}})
  return tuple(ops)


def _o_proj_weight_ops(
    heads: int,
    head_dim: int,
    d_model: int,
    pad_h: int,
    rep: int,
) -> tuple[dict[str, dict[str, Any]], ...]:
  """Builds 12-Op steps for HF `[D, H*D_h]` -> vLLM-JAX `[H*rep, H_pad, D]`."""
  ops: list[dict[str, dict[str, Any]]] = [{
      'reshape': {
          'src_shape': [d_model, heads * head_dim],
          'dst_shape': [d_model, heads, head_dim],
      }
  }]
  if pad_h > 0:
    ops.append({'pad': {'axis': 2, 'sizes': [0, pad_h]}})
  ops.append({'transpose': {'axes': [1, 2, 0]}})
  if rep > 1:
    ops.append({'repeat': {'axis': 0, 'times': rep}})
  return tuple(ops)


def build_vllm_jax_from_hf_manifest(
    hf_config: Qwen3HfConfig,
    *,
    tp_size: int = 1,
    head_sharding_size: int | None = None,
    vocab_size: int | None = None,
) -> hf_mapping.WeightMappingSpec:
  """Builds the declarative `HF -> vLLM-JAX` manifest for Qwen3."""
  if hf_config.attention_bias:
    raise ValueError(
        'tpu_inference Qwen3Attention does not support attention_bias=True.'
    )
  _validate_positive_int('tp_size', tp_size)
  sharding_size = tp_size if head_sharding_size is None else head_sharding_size
  _validate_positive_int('head_sharding_size', sharding_size)

  v_size = _validate_positive_int(
      'vocab_size',
      hf_config.vocab_size if vocab_size is None else vocab_size,
  )
  d_model = _validate_positive_int('hidden_size', hf_config.hidden_size)
  _validate_positive_int('intermediate_size', hf_config.intermediate_size)
  num_layers = _validate_positive_int(
      'num_hidden_layers', hf_config.num_hidden_layers
  )
  num_heads = _validate_positive_int(
      'num_attention_heads', hf_config.num_attention_heads
  )
  num_kv_heads = _validate_positive_int(
      'num_key_value_heads',
      num_heads
      if hf_config.num_key_value_heads is None
      else hf_config.num_key_value_heads,
  )
  if hf_config.head_dim is None:
    if d_model % num_heads != 0:
      raise ValueError(
          f'hidden_size ({d_model}) must be divisible by num_attention_heads'
          f' ({num_heads}) when head_dim is None.'
      )
    head_dim = d_model // num_heads
  else:
    head_dim = hf_config.head_dim
  _validate_positive_int('head_dim', head_dim)

  v_pad = ((v_size + tp_size - 1) // tp_size) * tp_size
  pad_v = v_pad - v_size
  h_pad = 64 if head_dim == 64 else ((head_dim + 127) // 128) * 128
  pad_h = h_pad - head_dim
  rep_q = _head_repeat_factor(num_heads, sharding_size)
  rep_kv = _head_repeat_factor(num_kv_heads, sharding_size)

  rules: list[hf_mapping.ParamRule] = [
      hf_mapping.ParamRule(
          source_keys=('model.embed_tokens.weight',),
          target_keys=('model.embed_tokens.weight',),
          ops=({'pad': {'axis': 0, 'sizes': [0, pad_v]}},)
          if pad_v > 0
          else (),
          sharding=('model', None),
      )
  ]

  norm_pad_ops = (
      ({'pad': {'axis': 0, 'sizes': [0, pad_h]}},) if pad_h > 0 else ()
  )
  transpose_2d = ({'transpose': {'axes': [1, 0]}},)

  for layer_idx in range(num_layers):
    prefix = f'model.layers.{layer_idx}'
    rules.extend([
        hf_mapping.ParamRule(
            source_keys=(f'{prefix}.input_layernorm.weight',),
            target_keys=(f'{prefix}.input_layernorm.weight',),
            ops=(),
            sharding=(None,),
        ),
        hf_mapping.ParamRule(
            source_keys=(f'{prefix}.post_attention_layernorm.weight',),
            target_keys=(f'{prefix}.post_attention_layernorm.weight',),
            ops=(),
            sharding=(None,),
        ),
        hf_mapping.ParamRule(
            source_keys=(f'{prefix}.self_attn.q_norm.weight',),
            target_keys=(f'{prefix}.self_attn.q_norm.weight',),
            ops=norm_pad_ops,
            sharding=(None,),
        ),
        hf_mapping.ParamRule(
            source_keys=(f'{prefix}.self_attn.k_norm.weight',),
            target_keys=(f'{prefix}.self_attn.k_norm.weight',),
            ops=norm_pad_ops,
            sharding=(None,),
        ),
        hf_mapping.ParamRule(
            source_keys=(f'{prefix}.self_attn.q_proj.weight',),
            target_keys=(f'{prefix}.self_attn.q_proj.weight',),
            ops=_qkv_weight_ops(num_heads, head_dim, d_model, pad_h, rep_q),
            sharding=(None, 'model', None),
        ),
        hf_mapping.ParamRule(
            source_keys=(f'{prefix}.self_attn.k_proj.weight',),
            target_keys=(f'{prefix}.self_attn.k_proj.weight',),
            ops=_qkv_weight_ops(
                num_kv_heads, head_dim, d_model, pad_h, rep_kv
            ),
            sharding=(None, 'model', None),
        ),
        hf_mapping.ParamRule(
            source_keys=(f'{prefix}.self_attn.v_proj.weight',),
            target_keys=(f'{prefix}.self_attn.v_proj.weight',),
            ops=_qkv_weight_ops(
                num_kv_heads, head_dim, d_model, pad_h, rep_kv
            ),
            sharding=(None, 'model', None),
        ),
        hf_mapping.ParamRule(
            source_keys=(f'{prefix}.self_attn.o_proj.weight',),
            target_keys=(f'{prefix}.self_attn.o_proj.weight',),
            ops=_o_proj_weight_ops(num_heads, head_dim, d_model, pad_h, rep_q),
            sharding=('model', None, None),
        ),
        hf_mapping.ParamRule(
            source_keys=(f'{prefix}.mlp.gate_proj.weight',),
            target_keys=(f'{prefix}.mlp.gate_proj.weight',),
            ops=transpose_2d,
            sharding=(None, 'model'),
        ),
        hf_mapping.ParamRule(
            source_keys=(f'{prefix}.mlp.up_proj.weight',),
            target_keys=(f'{prefix}.mlp.up_proj.weight',),
            ops=transpose_2d,
            sharding=(None, 'model'),
        ),
        hf_mapping.ParamRule(
            source_keys=(f'{prefix}.mlp.down_proj.weight',),
            target_keys=(f'{prefix}.mlp.down_proj.weight',),
            ops=transpose_2d,
            sharding=('model', None),
        ),
    ])

  rules.append(
      hf_mapping.ParamRule(
          source_keys=('model.norm.weight',),
          target_keys=('model.norm.weight',),
          ops=(),
          sharding=(None,),
      )
  )

  if bool(hf_config.tie_word_embeddings):
    dropped_keys: tuple[str, ...] = ('lm_head.weight',)
  else:
    dropped_keys = ()
    lm_head_ops = (
        ({'pad': {'axis': 0, 'sizes': [0, pad_v]}},) if pad_v > 0 else ()
    ) + transpose_2d
    rules.append(
        hf_mapping.ParamRule(
            source_keys=('lm_head.weight',),
            target_keys=('lm_head.weight',),
            ops=lm_head_ops,
            sharding=(None, None),
        )
    )

  return hf_mapping.WeightMappingSpec(
      rules=tuple(rules),
      tied_keys=(),
      dropped_keys=dropped_keys,
      model_type='qwen3',
  )


__all__ = [
    'TO_HF_MAPPINGS',
    'LORA_TO_HF_MAPPINGS',
    'VLLM_JAX_MAPPING',
    'Qwen3HfConfig',
    'build_vllm_jax_from_hf_manifest',
]
