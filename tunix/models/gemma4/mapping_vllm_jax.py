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
        'vllm_model.language_model.model.layers.*.moe.experts.down_proj.weight',
        (None, 'model', None),
    ),
})


# def preprocess_gemma4_state(src_state):
#     """Preprocess JAX state to fuse QKV and MLP for vLLM."""
#     print("Preprocessing Gemma4 state for vLLM: fusing QKV and MLP weights")
#     def _to_dict(x):
#         if hasattr(x, 'items'):
#             return {k: _to_dict(v) for k, v in x.items()}
#         return x
        
#     state_dict = _to_dict(src_state)
#     flat_state_tuples = flatten_dict(state_dict)
    
#     flat_state = {}
#     for k, v in flat_state_tuples.items():
#         str_key = '.'.join(str(x) for x in k)
#         flat_state[str_key] = v
        
#     new_flat_state = dict(flat_state)
    
#     layer_indices = set()
#     for key in flat_state.keys():
#         match = re.match(r"layers\.(\d+)\.", key)
#         if match:
#             layer_indices.add(int(match.group(1)))
            
#     for i in layer_indices:
#         # 1. Fuse QKV
#         q_key = f"layers.{i}.attn.q_einsum.w"
#         kv_key = f"layers.{i}.attn.kv_einsum.w"
#         k_key = f"layers.{i}.attn.k_einsum.w"
        
#         q = flat_state.get(q_key)
#         kv = flat_state.get(kv_key)
#         k = flat_state.get(k_key)
        
#         if q is not None:
#             if kv is not None:
#                 k_tensor, v_tensor = kv[0], kv[1]
#                 qkv = jnp.concatenate([q, k_tensor, v_tensor], axis=0)
#                 qkv = qkv.reshape(-1, qkv.shape[-1])
#                 print(f"Fusing QKV for layer {i}: q shape {q.shape}, k shape {k_tensor.shape}, v shape {v_tensor.shape}, fused shape {qkv.shape}")
#                 new_flat_state[f"layers.{i}.attn.qkv_fused"] = qkv
#                 del new_flat_state[q_key]
#                 del new_flat_state[kv_key]
#             elif k is not None:
#                 qkv = jnp.concatenate([q, k, k], axis=0)
#                 qkv = qkv.reshape(-1, qkv.shape[-1])
#                 new_flat_state[f"layers.{i}.attn.qkv_fused"] = qkv
#                 del new_flat_state[q_key]
#                 del new_flat_state[k_key]
                
#         # 2. Fuse Gate/Up
#         gate_key = f"layers.{i}.mlp.gate_proj.kernel"
#         up_key = f"layers.{i}.mlp.up_proj.kernel"
        
#         gate = flat_state.get(gate_key)
#         up = flat_state.get(up_key)
        
#         if gate is not None and up is not None:
#             gate_up = jnp.concatenate([gate, up], axis=1)
#             gate_up = gate_up.T
#             new_flat_state[f"layers.{i}.mlp.gate_up_fused"] = gate_up
#             del new_flat_state[gate_key]
#             del new_flat_state[up_key]
            
#     return unflatten_dict(new_flat_state, sep='.')


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
