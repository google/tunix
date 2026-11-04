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

"""vLLM (tpu-inference) backend mappings for dense Qwen3.5 models.

tpu-inference has no JAX-native Qwen3.5 implementation, so vLLM runs its
PyTorch `Qwen3_5ForCausalLM` through torchax. The runner state is then a flat
dict keyed by the torch parameter names (prefixed with `vllm_model.`), with
every linear weight stored as `[in, out]` and these projections fused along
the output dim:

* `self_attn.qkv_proj`        = [q_proj (incl. output gate), k_proj, v_proj]
* `mlp.gate_up_proj`          = [gate_proj, up_proj]
* `linear_attn.in_proj_qkvz`  = [in_proj_qkv, in_proj_z]
* `linear_attn.in_proj_ba`    = [in_proj_b, in_proj_a]

`preprocess_src_state` builds those fused tensors from the Tunix state so that
every mapping below is one-to-one. Serve the model with
`hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]}` (text-only) and
`load_format="dummy"`; the multimodal wrapper is not supported.

Only tensor_parallel_size=1 is supported: with TP>1 the fused weights are laid
out per TP shard and the 0.8B/2B models have fewer KV heads than typical TP
sizes.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from flax import nnx
import jax.numpy as jnp

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]

_P = 'vllm_model.model'

TO_HF_MAPPINGS: Dict[str, MappingEntry] = {
    'embedder.input_embedding': (
        f'{_P}.embed_tokens.weight',
        ('model', None),
    ),
    'final_norm.w': (f'{_P}.norm.weight', (None,)),
    'layers.*.input_layernorm.w': (
        f'{_P}.layers.*.input_layernorm.weight',
        (None,),
    ),
    'layers.*.post_attention_layernorm.w': (
        f'{_P}.layers.*.post_attention_layernorm.weight',
        (None,),
    ),
    # Dense MLP.
    'layers.*.mlp.gate_up_proj.kernel': (
        f'{_P}.layers.*.mlp.gate_up_proj.weight',
        (None, 'model'),
    ),
    'layers.*.mlp.down_proj.kernel': (
        f'{_P}.layers.*.mlp.down_proj.weight',
        ('model', None),
    ),
    # Full (gated) attention layers.
    'layers.*.attn.qkv_proj.w': (
        f'{_P}.layers.*.self_attn.qkv_proj.weight',
        (None, 'model'),
    ),
    'layers.*.attn.o_proj.w': (
        f'{_P}.layers.*.self_attn.o_proj.weight',
        ('model', None),
    ),
    'layers.*.attn.q_norm.w': (
        f'{_P}.layers.*.self_attn.q_norm.weight',
        (None,),
    ),
    'layers.*.attn.k_norm.w': (
        f'{_P}.layers.*.self_attn.k_norm.weight',
        (None,),
    ),
    # Gated DeltaNet linear attention layers.
    'layers.*.linear_attn.in_proj_qkvz.kernel': (
        f'{_P}.layers.*.linear_attn.in_proj_qkvz.weight',
        (None, 'model'),
    ),
    'layers.*.linear_attn.in_proj_ba.kernel': (
        f'{_P}.layers.*.linear_attn.in_proj_ba.weight',
        (None, 'model'),
    ),
    'layers.*.linear_attn.conv1d.kernel': (
        f'{_P}.layers.*.linear_attn.conv1d.weight',
        (None, None, None),
    ),
    'layers.*.linear_attn.out_proj.kernel': (
        f'{_P}.layers.*.linear_attn.out_proj.weight',
        ('model', None),
    ),
    'layers.*.linear_attn.norm.w': (
        f'{_P}.layers.*.linear_attn.norm.weight',
        (None,),
    ),
    'layers.*.linear_attn.A_log': (
        f'{_P}.layers.*.linear_attn.A_log',
        (None,),
    ),
    'layers.*.linear_attn.dt_bias': (
        f'{_P}.layers.*.linear_attn.dt_bias',
        (None,),
    ),
}

LORA_TO_HF_MAPPINGS: Dict[str, MappingEntry] = {}

TO_HF_TRANSPOSE_KEYS = {
    # Tunix (K, 1, C) -> torch Conv1d (C, 1, K).
    'layers.*.linear_attn.conv1d.kernel': (2, 1, 0),
}

# (fused target name, source names in output-dim order) per module.
_FUSIONS = {
    'attn': (('qkv_proj', 'w'), (('q_proj', 'w'), ('k_proj', 'w'),
                                 ('v_proj', 'w'))),
    'mlp': (('gate_up_proj', 'kernel'), (('gate_proj', 'kernel'),
                                         ('up_proj', 'kernel'))),
    'linear_attn': (
        ('in_proj_qkvz', 'kernel'),
        (('in_proj_qkv', 'kernel'), ('in_proj_z', 'kernel')),
    ),
    'linear_attn_ba': (
        ('in_proj_ba', 'kernel'),
        (('in_proj_b', 'kernel'), ('in_proj_a', 'kernel')),
    ),
}


def _value(param: Any) -> Any:
  return param.value if hasattr(param, 'value') else param


def _wrap_like(param: Any, val: Any) -> Any:
  return nnx.Param(val) if hasattr(param, 'value') else val


def preprocess_src_state(src_state: Any, tp_size: int = 1) -> Any:
  """Fuses/reshapes Tunix Qwen3.5 params into vLLM's torch parameter layout."""
  if not hasattr(src_state, 'flat_state'):
    return src_state
  if tp_size != 1:
    raise NotImplementedError(
        'Qwen3.5 vLLM weight sync only supports tensor_parallel_size=1, got'
        f' {tp_size}.'
    )

  # (layer path prefix, fusion key) -> {source leaf name: (keys, param)}
  pending: Dict[Tuple[Tuple[Any, ...], str], Dict[Tuple[str, str], Any]] = {}
  source_to_fusion = {}
  for fusion_key, (_, sources) in _FUSIONS.items():
    module = 'linear_attn' if fusion_key == 'linear_attn_ba' else fusion_key
    for src in sources:
      source_to_fusion[(module,) + src] = fusion_key

  new_flat_state = []
  for keys, param in src_state.flat_state():
    tail = tuple(str(k) for k in keys[-3:])
    fusion_key = source_to_fusion.get(tail)
    if fusion_key is not None:
      pending.setdefault((tuple(keys[:-3]), fusion_key), {})[tail[1:]] = (
          keys,
          param,
      )
    elif tail[-3:] == ('attn', 'o_proj', 'w'):
      val = _value(param)  # (H, head_dim, D) -> (H * head_dim, D)
      new_flat_state.append(
          (keys, _wrap_like(param, val.reshape(-1, val.shape[-1])))
      )
    else:
      new_flat_state.append((keys, param))

  for (prefix, fusion_key), parts in pending.items():
    target, sources = _FUSIONS[fusion_key]
    if set(parts) != set(sources):
      raise ValueError(
          f'Incomplete Qwen3.5 fusion {fusion_key} under {prefix}: have'
          f' {sorted(parts)}, need {sources}.'
      )
    vals = []
    for src in sources:
      val = _value(parts[src][1])
      vals.append(val.reshape(val.shape[0], -1))  # (D, ...) -> (D, out)
    module = 'linear_attn' if fusion_key == 'linear_attn_ba' else fusion_key
    first_param = parts[sources[0]][1]
    new_flat_state.append((
        prefix + (module,) + target,
        _wrap_like(first_param, jnp.concatenate(vals, axis=-1)),
    ))

  return src_state.from_flat_path(new_flat_state)


VLLM_JAX_MAPPING: Dict[str, Any] = {
    'to_hf_mappings': TO_HF_MAPPINGS,
    'lora_to_hf_mappings': LORA_TO_HF_MAPPINGS,
    'to_hf_transpose_keys': TO_HF_TRANSPOSE_KEYS,
    'preprocess_src_state': preprocess_src_state,
}

__all__ = ['VLLM_JAX_MAPPING']
