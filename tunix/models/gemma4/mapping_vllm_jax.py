"""vLLM JAX backend mappings for Gemma4 models."""

from __future__ import annotations

from typing import Any, Dict, Tuple
import re
from flax.traverse_util import flatten_dict, unflatten_dict
import jax.numpy as jnp

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]

# Following mappings are only for the torchax implementation of Gemma4.
# The jax implementation needs different mappings.
TO_HF_MAPPINGS: Dict[str, MappingEntry] = {
    'embedder.input_embedding': (
        'vllm_model.language_model.model.embed_tokens.weight',
        ('model', None),
    ),
    'layers.*.pre_attention_norm.scale': (
        'vllm_model.language_model.model.layers.*.input_layernorm.weight',
        (None,),
    ),
    # Q, K, V are fused into one matrix in vLLM.
    'layers.*.attn.q_einsum.w': (
        'vllm_model.language_model.model.layers.*.self_attn.qkv_proj.weight',
        (None, 'model'),
    ),
    'layers.*.attn.k_einsum.w': (
        'vllm_model.language_model.model.layers.*.self_attn.qkv_proj.weight',
        (None, 'model'),
    ),
    'layers.*.attn.kv_einsum.w': (
        'vllm_model.language_model.model.layers.*.self_attn.qkv_proj.weight',
        (None, 'model'),
    ),
    'layers.*.attn.attn_vec_einsum.w': (
        'vllm_model.language_model.model.layers.*.self_attn.o_proj.weight',
        ('model', None, None),
    ),
    'layers.*.post_attention_norm.scale': (
        'vllm_model.language_model.model.layers.*.post_attention_layernorm.weight',
        (None,),
    ),
    'layers.*.pre_ffw_norm.scale': (
        'vllm_model.language_model.model.layers.*.pre_feedforward_layernorm.weight',
        (None,),
    ),
    # Gate/Up are fused into one matrix in vLLM.
    'layers.*.mlp.gate_proj.kernel': (
        'vllm_model.language_model.model.layers.*.mlp.gate_up_proj.weight',
        (None, None),
    ),
    'layers.*.mlp.up_proj.kernel': (
        'vllm_model.language_model.model.layers.*.mlp.gate_up_proj.weight',
        (None, None),
    ),
    'layers.*.mlp.down_proj.kernel': (
        'vllm_model.language_model.model.layers.*.mlp.down_proj.weight',
        ('model', None),
    ),
    'layers.*.post_ffw_norm.scale': (
        'vllm_model.language_model.model.layers.*.post_feedforward_layernorm.weight',
        (None,),
    ),
    'final_norm.scale': (
        'vllm_model.language_model.model.norm.weight',
        (None,),
    ),
    'layers.*.attn._query_norm.scale': (
        'vllm_model.language_model.model.layers.*.self_attn.q_norm.weight',
        (None,),
    ),
    'layers.*.attn._key_norm.scale': (
        'vllm_model.language_model.model.layers.*.self_attn.k_norm.weight',
        (None,),
    ),
    'layers.*.skip_scale': (
        'vllm_model.language_model.model.layers.*.layer_scalar',
        (None,),
    ),
}

# Add per-layer mappings (used in some Gemma4 variants)
TO_HF_MAPPINGS.update({
    'embedder.per_layer_input_embedding': (
        'vllm_model.language_model.model.embed_tokens_per_layer.weight',
        ('model', None, None),
    ),
    'embedder.per_layer_model_projection.w': (
        'vllm_model.language_model.model.per_layer_model_projection.weight',
        (None, None, 'model'),
    ),
    'embedder.per_layer_projection_norm.scale': (
        'vllm_model.language_model.model.per_layer_projection_norm.weight',
        (None,),
    ),
    'layers.*.per_layer_input_gate.w': (
        'vllm_model.language_model.model.layers.*.per_layer_input_gate.weight',
        (None, 'model'),
    ),
    'layers.*.per_layer_projection.w': (
        'vllm_model.language_model.model.layers.*.per_layer_projection.weight',
        ('model', None),
    ),
    'layers.*.post_per_layer_input_norm.scale': (
        'vllm_model.language_model.model.layers.*.post_per_layer_input_norm.weight',
        (None,),
    ),
})

# Add MoE mappings (used in some Gemma4 variants)
TO_HF_MAPPINGS.update({
    'layers.*.moe.router_logits': (
        'vllm_model.language_model.model.layers.*.router.proj.weight',
        (None, 'model'),
    ),
    'layers.*.moe.per_expert_scale': (
        'vllm_model.language_model.model.layers.*.router.per_expert_scale',
        (None,),
    ),
    'layers.*.moe.router_scale': (
        'vllm_model.language_model.model.layers.*.router.scale',
        (None,),
    ),
    'layers.*.moe.gating_einsum': (
        'vllm_model.language_model.model.layers.*.moe.experts.w13_weight',
        (None, None, None, 'model'),
    ),
    'layers.*.moe.linear': (
        'vllm_model.language_model.model.layers.*.moe.experts.w2_weight',
        (None, 'model', None),
    ),
    'layers.*.moe_post_ffw_norm.scale': (
        'vllm_model.language_model.model.layers.*.post_feedforward_layernorm_2.weight',
        (None,),
    ),
    'layers.*.moe_pre_ffw_norm.scale': (
        'vllm_model.language_model.model.layers.*.pre_feedforward_layernorm_2.weight',
        (None,),
    ),
    'layers.*.dense_post_ffw_norm.scale': (
        'vllm_model.language_model.model.layers.*.post_feedforward_layernorm_1.weight',
        (None,),
    ),
})

VLLM_JAX_MAPPING: Dict[str, Any] = {
    'to_hf_mappings': TO_HF_MAPPINGS,
    'lora_to_hf_mappings': {},
    'to_hf_transpose_keys': {'embedding': (1, 0), 'mlp.down_proj.kernel': (1, 0)},
    'to_hf_hook_fns': None,
    # 'to_hf_preprocess_fn': preprocess_gemma4_state,
}

__all__ = [
    'VLLM_JAX_MAPPING',
]
