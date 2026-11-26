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

"""vLLM JAX (tpu-inference VllmModelWrapper) weight mappings for Qwen3.5.

Targets `VllmModelWrapper`'s internal `runner.state` (`dict[str, jax.Array]`
with 2D linear weights stored as `(in_features, out_features)`), as consumed by
`VllmSampler._update_params_jax` and `RaidenSynchronizer` at
`tensor_parallel_size=1`.

Precondition: Target parameter names (`vllm_model.model.*` and
`vllm_model.lm_head.weight`) correspond to vLLM's causal LM class
`Qwen3_5ForCausalLM`. Because official Qwen3.5 Hugging Face checkpoints declare
`Qwen3_5ForConditionalGeneration` (which nests language weights under
`language_model.*`), callers initializing vLLM from those checkpoints must pass
`hf_overrides={'architectures': ['Qwen3_5ForCausalLM']}`.
"""

from __future__ import annotations

import collections
from collections.abc import Callable
import dataclasses
from typing import TypeAlias, TypedDict

from flax import nnx
import jax
import jax.numpy as jnp

Sharding: TypeAlias = tuple[str | None, ...]
MappingEntry: TypeAlias = tuple[str, Sharding]
StateLeaf: TypeAlias = nnx.Variable | jax.Array
FlatKey: TypeAlias = tuple[str | int, ...]


class VllmJaxMapping(TypedDict):
  to_hf_mappings: dict[str, MappingEntry]
  lora_to_hf_mappings: dict[str, MappingEntry]
  to_hf_transpose_keys: dict[str, tuple[int, ...]]
  to_hf_hook_fns: dict[str, Callable[[jax.Array], jax.Array]] | None
  preprocess_src_state: Callable[..., nnx.State]


# In tpu-inference, Qwen3.5 runs through VllmModelWrapper, whose internal
# runner.state is a TP-dependent flat dict keyed by `vllm_model.*` paths
# (`Qwen3_5ForCausalLM`) with 2D linear weights stored in
# (in_features, out_features) layout. Callers loading official multimodal
# Qwen3.5 checkpoints must set
# `hf_overrides={'architectures': ['Qwen3_5ForCausalLM']}`.
TO_HF_MAPPINGS: dict[str, MappingEntry] = {
    'embedder.input_embedding': (
        'vllm_model.model.embed_tokens.weight',
        ('model', None),
    ),
    'lm_head.w': (
        'vllm_model.lm_head.weight',
        ('model', None),
    ),
    'final_norm.w': (
        'vllm_model.model.norm.weight',
        (None,),
    ),
    'layers.*.input_layernorm.w': (
        'vllm_model.model.layers.*.input_layernorm.weight',
        (None,),
    ),
    'layers.*.post_attention_layernorm.w': (
        'vllm_model.model.layers.*.post_attention_layernorm.weight',
        (None,),
    ),
    'layers.*.mlp.gate_up_proj.kernel': (
        'vllm_model.model.layers.*.mlp.gate_up_proj.weight',
        (None, 'model'),
    ),
    'layers.*.mlp.down_proj.kernel': (
        'vllm_model.model.layers.*.mlp.down_proj.weight',
        ('model', None),
    ),
    'layers.*.attn.qkv_proj.w': (
        'vllm_model.model.layers.*.self_attn.qkv_proj.weight',
        (None, 'model'),
    ),
    'layers.*.attn.o_proj.w': (
        'vllm_model.model.layers.*.self_attn.o_proj.weight',
        ('model', None),
    ),
    'layers.*.attn.q_norm.w': (
        'vllm_model.model.layers.*.self_attn.q_norm.weight',
        (None,),
    ),
    'layers.*.attn.k_norm.w': (
        'vllm_model.model.layers.*.self_attn.k_norm.weight',
        (None,),
    ),
    'layers.*.linear_attn.in_proj_qkvz.kernel': (
        'vllm_model.model.layers.*.linear_attn.in_proj_qkvz.weight',
        (None, 'model'),
    ),
    'layers.*.linear_attn.in_proj_ba.kernel': (
        'vllm_model.model.layers.*.linear_attn.in_proj_ba.weight',
        (None, 'model'),
    ),
    'layers.*.linear_attn.conv1d.kernel': (
        'vllm_model.model.layers.*.linear_attn.conv1d.weight',
        ('model', None, None),
    ),
    'layers.*.linear_attn.dt_bias': (
        'vllm_model.model.layers.*.linear_attn.dt_bias',
        ('model',),
    ),
    'layers.*.linear_attn.A_log': (
        'vllm_model.model.layers.*.linear_attn.A_log',
        ('model',),
    ),
    'layers.*.linear_attn.norm.w': (
        'vllm_model.model.layers.*.linear_attn.norm.weight',
        (None,),
    ),
    'layers.*.linear_attn.out_proj.kernel': (
        'vllm_model.model.layers.*.linear_attn.out_proj.weight',
        ('model', None),
    ),
}

LORA_TO_HF_MAPPINGS: dict[str, MappingEntry] = {}

# Tunix conv1d.kernel is (K, 1, conv_dim) while vLLM conv1d.weight is
# (conv_dim, 1, K). Untied Tunix lm_head.w is (D, V) while vLLM ParallelLMHead
# keeps (V, D).
TO_HF_TRANSPOSE_KEYS: dict[str, tuple[int, ...]] = {
    'layers.*.linear_attn.conv1d.kernel': (2, 1, 0),
    'lm_head.w': (1, 0),
}


@dataclasses.dataclass(frozen=True)
class _LayerFusionSpec:
  """Describes a per-layer concatenation of Tunix projections into vLLM form."""

  suffixes: tuple[str, ...]
  target_tail: tuple[str, ...]
  flatten_3d_inputs: bool = False

  @property
  def suffix_depth(self) -> int:
    return len(self.suffixes[0].split('.'))


_LAYER_FUSION_SPECS: tuple[_LayerFusionSpec, ...] = (
    # Output-gated GQA: q_proj has width 2 * N * H (query + sigmoid gate).
    _LayerFusionSpec(
        suffixes=('attn.q_proj.w', 'attn.k_proj.w', 'attn.v_proj.w'),
        target_tail=('attn', 'qkv_proj', 'w'),
        flatten_3d_inputs=True,
    ),
    _LayerFusionSpec(
        suffixes=('mlp.gate_proj.kernel', 'mlp.up_proj.kernel'),
        target_tail=('mlp', 'gate_up_proj', 'kernel'),
    ),
    # QwenGatedDeltaNetAttention sets gqa_interleaved_layout=False, using
    # contiguous [qkv, z] and [b, a] concatenation for TP=1.
    _LayerFusionSpec(
        suffixes=(
            'linear_attn.in_proj_qkv.kernel',
            'linear_attn.in_proj_z.kernel',
        ),
        target_tail=('linear_attn', 'in_proj_qkvz', 'kernel'),
    ),
    _LayerFusionSpec(
        suffixes=(
            'linear_attn.in_proj_b.kernel',
            'linear_attn.in_proj_a.kernel',
        ),
        target_tail=('linear_attn', 'in_proj_ba', 'kernel'),
    ),
)


def _build_suffix_to_spec() -> dict[str, _LayerFusionSpec]:
  mapping: dict[str, _LayerFusionSpec] = {}
  for spec in _LAYER_FUSION_SPECS:
    depths = {len(s.split('.')) for s in spec.suffixes}
    if len(depths) != 1:
      raise ValueError(
          f'Inconsistent suffix depths in {spec.suffixes}: {depths}'
      )
    for suffix in spec.suffixes:
      mapping[suffix] = spec
  return mapping


_SUFFIX_TO_SPEC: dict[str, _LayerFusionSpec] = _build_suffix_to_spec()


def _unwrap_array(param: StateLeaf) -> jax.Array:
  return param.get_value() if isinstance(param, nnx.Variable) else param


def _wrap_like(template: StateLeaf, val: jax.Array) -> StateLeaf:
  # Construct a fresh Variable of the same subtype without copying pre-reshape
  # 3D sharding metadata (e.g. q_proj's `('fsdp', 'tp', None)`) onto 2D fused
  # tensors.
  return type(template)(val) if isinstance(template, nnx.Variable) else val


def _match_fusion_suffix(src_key: str) -> tuple[_LayerFusionSpec, str] | None:
  for suffix, spec in _SUFFIX_TO_SPEC.items():
    if src_key.endswith('.' + suffix):
      return spec, suffix
  return None


def _align_operand_shardings(arrays: list[jax.Array]) -> list[jax.Array]:
  """Aligns operand shardings for `axis=-1` projection concatenation.

  Reshards operands to `PartitionSpec(row_spec, None)` whenever operand
  shardings differ or when an `AxisType.Explicit` mesh shards the concatenation
  axis (`axis=1`), even if all operands share the same sharding.

  Args:
    arrays: 2D projection operands to concatenate along `axis=-1`.

  Returns:
    Operands aligned to a common concatenation-compatible sharding.

  Raises:
    ValueError: If `NamedSharding` operands belong to distinct JAX meshes.
  """
  named = [
      arr.sharding
      for arr in arrays
      if isinstance(arr.sharding, jax.sharding.NamedSharding)
  ]
  if not named:
    return arrays
  ref_shd = named[0]
  if any(shd.mesh != ref_shd.mesh for shd in named[1:]):
    raise ValueError(
        'Cannot fuse Qwen3.5 projections across distinct JAX meshes:'
        f' {[shd.mesh for shd in named]}.'
    )
  explicit_mesh = jax.sharding.AxisType.Explicit in ref_shd.mesh.axis_types
  concat_axis_sharded = len(ref_shd.spec) > 1 and ref_shd.spec[1] is not None
  if all(arr.sharding == arrays[0].sharding for arr in arrays[1:]) and not (
      explicit_mesh and concat_axis_sharded
  ):
    return arrays
  row_spec = ref_shd.spec[0] if ref_shd.spec else None
  target_shd = jax.sharding.NamedSharding(
      ref_shd.mesh, jax.sharding.PartitionSpec(row_spec, None)
  )
  aligned: list[jax.Array] = []
  for arr in arrays:
    if arr.sharding == target_shd:
      aligned.append(arr)
    elif explicit_mesh and isinstance(arr.sharding, jax.sharding.NamedSharding):
      aligned.append(jax.sharding.reshard(arr, target_shd))
    else:
      aligned.append(jax.device_put(arr, target_shd))
  return aligned


def _fuse_layer_group(
    spec: _LayerFusionSpec,
    layer_prefix: FlatKey,
    parts: dict[str, tuple[FlatKey, StateLeaf]],
) -> tuple[FlatKey, StateLeaf]:
  """Concatenates one layer's projection leaves according to `spec`."""
  if len(parts) != len(spec.suffixes):
    raise ValueError(
        f'Incomplete Qwen3.5 fusion group for layer {layer_prefix!r}: expected'
        f' {spec.suffixes}, got {tuple(sorted(parts))}.'
    )
  arrays: list[jax.Array] = []
  for suffix in spec.suffixes:
    arr = _unwrap_array(parts[suffix][1])
    if spec.flatten_3d_inputs:
      arr = arr.reshape(arr.shape[0], -1)
    arrays.append(arr)
  _, first_param = parts[spec.suffixes[0]]
  fused = jnp.concatenate(_align_operand_shardings(arrays), axis=-1)
  return layer_prefix + spec.target_tail, _wrap_like(first_param, fused)


def preprocess_src_state(src_state: nnx.State, tp_size: int = 1) -> nnx.State:
  """Fuses Tunix Qwen3.5 projections to match vLLM's VllmModelWrapper state.

  Args:
    src_state: Tunix Qwen3.5 `nnx.State`.
    tp_size: Rollout tensor-parallel degree. Only `tp_size == 1` is currently
      supported; `tp_size > 1` requires TP-interleaved packing for fused
      projections and is rejected explicitly.

  Returns:
    A new `nnx.State` with fused QKV, MLP gate/up, and GDN projections.

  Raises:
    NotImplementedError: If `tp_size != 1` or if LoRA adapter leaves are
      present in `src_state`.
    ValueError: If any layer has a partially populated fusion group or if
      operands belong to distinct JAX meshes.
  """
  if tp_size != 1:
    raise NotImplementedError(
        'Qwen3.5 vLLM weight sync currently supports tensor_parallel_size=1'
        f' only; got tp_size={tp_size}.'
    )

  new_flat_state: list[tuple[FlatKey, StateLeaf]] = []
  grouped: dict[
      tuple[_LayerFusionSpec, FlatKey], dict[str, tuple[FlatKey, StateLeaf]]
  ] = collections.defaultdict(dict)

  for keys, param in nnx.to_flat_state(src_state):
    src_key = '.'.join(str(k) for k in keys)
    if src_key.endswith(('_lora_a', '_lora_b', '.lora_a', '.lora_b')):
      raise NotImplementedError(
          'Qwen3.5 vLLM weight sync does not yet support LoRA adapter'
          f' weights (found {src_key!r}).'
      )
    matched = _match_fusion_suffix(src_key)
    if matched is not None:
      spec, suffix = matched
      grouped[(spec, keys[: -spec.suffix_depth])][suffix] = (keys, param)
    elif src_key.endswith('.attn.o_proj.w'):
      arr = _unwrap_array(param)
      new_flat_state.append(
          (keys, _wrap_like(param, arr.reshape(-1, arr.shape[-1])))
      )
    else:
      new_flat_state.append((keys, param))

  for (spec, layer_prefix), parts in grouped.items():
    new_flat_state.append(_fuse_layer_group(spec, layer_prefix, parts))

  return nnx.from_flat_state(new_flat_state)


VLLM_JAX_MAPPING: VllmJaxMapping = {
    'to_hf_mappings': TO_HF_MAPPINGS,
    'lora_to_hf_mappings': LORA_TO_HF_MAPPINGS,
    'to_hf_transpose_keys': TO_HF_TRANSPOSE_KEYS,
    'to_hf_hook_fns': None,
    'preprocess_src_state': preprocess_src_state,
}

__all__ = [
    'LORA_TO_HF_MAPPINGS',
    'TO_HF_MAPPINGS',
    'TO_HF_TRANSPOSE_KEYS',
    'VLLM_JAX_MAPPING',
    'preprocess_src_state',
]
