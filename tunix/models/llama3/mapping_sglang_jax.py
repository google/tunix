"""Mappings for converting Llama3 weights to the Sglang-jax JAX backend."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from tunix.utils.env_utils import SGLANG_JAX_TP_AXIS_NAME

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]


def _to_sglang_jax_mappings() -> Dict[str, MappingEntry]:
  return {
      'lm_head.w': ('lm_head.embedding', (None, SGLANG_JAX_TP_AXIS_NAME)),
      'embedder.input_embedding': (
          'model.embed_tokens.embedding',
          (SGLANG_JAX_TP_AXIS_NAME, None),
      ),
      'layers.*.input_layernorm.w': (
          'model.layers.*.input_layernorm.scale',
          (None,),
      ),
      'layers.*.mlp.down_proj.kernel': (
          'model.layers.*.mlp.down_proj.weight',
          (SGLANG_JAX_TP_AXIS_NAME, None),
      ),
      'layers.*.mlp.gate_proj.kernel': (
          'model.layers.*.mlp.gate_proj.weight',
          (None, SGLANG_JAX_TP_AXIS_NAME),
      ),
      'layers.*.mlp.up_proj.kernel': (
          'model.layers.*.mlp.up_proj.weight',
          (None, SGLANG_JAX_TP_AXIS_NAME),
      ),
      'layers.*.post_attention_layernorm.w': (
          'model.layers.*.post_attention_layernorm.scale',
          (None,),
      ),
      'layers.*.attn.k_proj.w': (
          'model.layers.*.self_attn.k_proj.weight',
          (None, SGLANG_JAX_TP_AXIS_NAME, None),
      ),
      'layers.*.attn.o_proj.w': (
          'model.layers.*.self_attn.o_proj.weight',
          (SGLANG_JAX_TP_AXIS_NAME, None, None),
      ),
      'layers.*.attn.q_proj.w': (
          'model.layers.*.self_attn.q_proj.weight',
          (None, SGLANG_JAX_TP_AXIS_NAME, None),
      ),
      'layers.*.attn.v_proj.w': (
          'model.layers.*.self_attn.v_proj.weight',
          (None, SGLANG_JAX_TP_AXIS_NAME, None),
      ),
      'final_norm.w': ('model.norm.scale', (None,)),
  }


def _lora_to_sglang_jax_mappings() -> Dict[str, MappingEntry] | None:
  """The lora parameter key mapping between Tunix vanilla model and Sglang-jax Jax backend"""
  return {
      'layers.*.mlp.gate_proj.kernel_lora_a': (
          'model.layers.*.mlp.gate_proj.A_buffer',
          (None, None, None),
      ),
      'layers.*.mlp.gate_proj.kernel_lora_b': (
          'model.layers.*.mlp.gate_proj.B_buffer',
          (None, SGLANG_JAX_TP_AXIS_NAME, None),
      ),
      'layers.*.mlp.up_proj.kernel_lora_a': (
          'model.layers.*.mlp.up_proj.A_buffer',
          (None, None, None),
      ),
      'layers.*.mlp.up_proj.kernel_lora_b': (
          'model.layers.*.mlp.up_proj.B_buffer',
          (None, SGLANG_JAX_TP_AXIS_NAME, None),
      ),
      'layers.*.mlp.down_proj.kernel_lora_a': (
          'model.layers.*.mlp.down_proj.A_buffer',
          (None, None, SGLANG_JAX_TP_AXIS_NAME),
      ),
      'layers.*.mlp.down_proj.kernel_lora_b': (
          'model.layers.*.mlp.down_proj.B_buffer',
          (None, None, None),
      ),
      'layers.*.attn.q_proj.w_lora_a': (
          'model.layers.*.self_attn.q_proj.A_buffer',
          (None, None, None),
      ),
      'layers.*.attn.q_proj.w_lora_b': (
          'model.layers.*.self_attn.q_proj.B_buffer',
          (None, SGLANG_JAX_TP_AXIS_NAME, None),
      ),
      'layers.*.attn.k_proj.w_lora_a': (
          'model.layers.*.self_attn.k_proj.A_buffer',
          (None, None, None),
      ),
      'layers.*.attn.k_proj.w_lora_b': (
          'model.layers.*.self_attn.k_proj.B_buffer',
          (None, SGLANG_JAX_TP_AXIS_NAME, None),
      ),
      'layers.*.attn.v_proj.w_lora_a': (
          'model.layers.*.self_attn.v_proj.A_buffer',
          (None, None, None),
      ),
      'layers.*.attn.v_proj.w_lora_b': (
          'model.layers.*.self_attn.v_proj.B_buffer',
          (None, SGLANG_JAX_TP_AXIS_NAME, None),
      ),
      'layers.*.attn.o_proj.w_lora_a': (
          'model.layers.*.self_attn.o_proj.A_buffer',
          (None, None, SGLANG_JAX_TP_AXIS_NAME),
      ),
      'layers.*.attn.o_proj.w_lora_b': (
          'model.layers.*.self_attn.o_proj.B_buffer',
          (None, None, None),
      ),
  }


def _to_sglang_jax_transpose_keys():
  return {
      'lm_head.w': (1, 0),
  }


def _to_sglang_jax_lora_transpose_keys():
  """
  Tunix -> SGLangJax:
  gate_lora_a: (hidden_size, max_lora_rank) -> (1, max_lora_rank, hidden_size)
  gate_lora_b: (max_lora_rank, intermediate_size) -> (1, intermediate_size, max_lora_rank)
  up_lora_a: (hidden_size, max_lora_rank) -> (1, max_lora_rank, hidden_size)
  up_lora_b: (max_lora_rank, intermediate_size) -> (1, intermediate_size, max_lora_rank)
  down_lora_a: (intermediate_size, max_lora_rank) -> (1, max_lora_rank, intermediate_size)
  down_lora_b: (max_lora_rank, hidden_size) -> (1, hidden_size, max_lora_rank)
  q_lora_a: (hidden_size, max_lora_rank) -> (1, max_lora_rank, hidden_size)
  q_lora_b: (max_lora_rank, num_attention_heads, head_dim) -> (1, hidden_size, max_lora_rank)
  k_lora_a: (hidden_size, max_lora_rank) -> (1, max_lora_rank, hidden_size)
  k_lora_b: (max_lora_rank, num_key_value_heads, head_dim) -> (1, num_key_value_heads*head_dim, max_lora_rank)
  v_lora_a: (hidden_size, max_lora_rank) -> (1, max_lora_rank, hidden_size)
  v_lora_b: (max_lora_rank, num_key_value_heads, head_dim) -> (1, num_key_value_heads*head_dim, max_lora_rank)
  o_lora_a: (num_attention_heads, head_dim, max_lora_rank) -> (1, max_lora_rank, hidden_size)
  o_lora_b: (max_lora_rank, hidden_size) -> (1, hidden_size, max_lora_rank)
  """
  return {
      'layers.*.mlp.gate_proj.kernel_lora_a': (0, 2, 1),
      'layers.*.mlp.gate_proj.kernel_lora_b': (0, 2, 1),
      'layers.*.mlp.up_proj.kernel_lora_a': (0, 2, 1),
      'layers.*.mlp.up_proj.kernel_lora_b': (0, 2, 1),
      'layers.*.mlp.down_proj.kernel_lora_a': (0, 2, 1),
      'layers.*.mlp.down_proj.kernel_lora_b': (0, 2, 1),
      'layers.*.attn.q_proj.w_lora_a': (0, 2, 1),
      'layers.*.attn.q_proj.w_lora_b': (0, 2, 3, 1),
      'layers.*.attn.k_proj.w_lora_a': (0, 2, 1),
      'layers.*.attn.k_proj.w_lora_b': (0, 2, 3, 1),
      'layers.*.attn.v_proj.w_lora_a': (0, 2, 1),
      'layers.*.attn.v_proj.w_lora_b': (0, 2, 3, 1),
      'layers.*.attn.o_proj.w_lora_a': (0, 3, 2, 1),
      'layers.*.attn.o_proj.w_lora_b': (0, 2, 1),
  }


def _to_sglang_jax_hook_fns() -> Dict[str, Any] | None:
  """Additional parameter manipulation hook between Tunix vanilla model and Sglang Jax backend"""
  return None


SGLANG_JAX_MAPPING: Dict[str, Any] = {
    'to_hf_mappings': _to_sglang_jax_mappings(),
    'lora_to_hf_mappings': _lora_to_sglang_jax_mappings(),
    'to_hf_transpose_keys': _to_sglang_jax_transpose_keys(),
    'lora_to_hf_transpose_keys': _to_sglang_jax_lora_transpose_keys(),
    'to_hf_hook_fns': _to_sglang_jax_hook_fns(),
}

__all__ = [
    'SGLANG_JAX_MAPPING',
]
