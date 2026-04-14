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

"""Loads Gemma 4 parameters from safetensors files."""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
from tunix.models import safetensors_loader
from tunix.models.gemma4 import model as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  """Mapping of torch_keys to (nnx_keys, (permute_rule, reshape_rule))."""
  pattern = (
      cfg.attention_pattern
      if cfg.attention_pattern
      else model_lib.GEMMA4_ATTENTION_PATTERN
  )
  global_layers = []
  local_layers = []
  for i in range(cfg.num_layers):
    if pattern[i % len(pattern)] == model_lib.AttentionType.GLOBAL:
      global_layers.append(str(i))
    else:
      local_layers.append(str(i))

  global_pat = "|".join(global_layers)
  local_pat = "|".join(local_layers)

  mapping = {
      r"(?:model\.language_model\.)?embed_tokens\.weight": (
          "embedder.input_embedding",
          None,
      ),
      r"(?:model\.language_model\.)?embed_tokens_per_layer\.weight": (
          "embedder.per_layer_input_embedding.value",
          (None, (cfg.num_embed, cfg.num_layers, cfg.per_layer_input_dim)),
      ),
      r"(?:model\.language_model\.)?per_layer_model_projection\.weight": (
          "embedder.per_layer_model_projection.w",
          ((1, 0), (cfg.embed_dim, cfg.num_layers, cfg.per_layer_input_dim)),
      ),
      r"(?:model\.language_model\.)?per_layer_projection_norm\.weight": (
          "embedder.per_layer_projection_norm.scale",
          None,
      ),
      # Local attention layers
      r"(?:model\.language_model\.)?layers\.({})\.self_attn\.q_proj\.weight"
      .format(
          local_pat
      ): (
          r"tmp.layers.\1.attn.q",
          ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim)),
      ),
      r"(?:model\.language_model\.)?layers\.({})\.self_attn\.k_proj\.weight"
      .format(
          local_pat
      ): (
          r"tmp.layers.\1.attn.k",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"(?:model\.language_model\.)?layers\.({})\.self_attn\.v_proj\.weight"
      .format(
          local_pat
      ): (
          r"tmp.layers.\1.attn.v",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"(?:model\.language_model\.)?layers\.({})\.self_attn\.o_proj\.weight"
      .format(
          local_pat
      ): (
          r"layers.\1.attn.attn_vec_einsum.w",
          ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
      ),
      # Global attention layers
      r"(?:model\.language_model\.)?layers\.({})\.self_attn\.q_proj\.weight"
      .format(
          global_pat
      ): (
          r"tmp.layers.\1.attn.q",
          (
              (1, 0),
              (
                  cfg.embed_dim,
                  cfg.num_heads,
                  cfg.global_key_size
                  if cfg.global_key_size is not None
                  else cfg.head_dim,
              ),
          ),
      ),
      r"(?:model\.language_model\.)?layers\.({})\.self_attn\.k_proj\.weight"
      .format(
          global_pat
      ): (
          r"tmp.layers.\1.attn.k",
          (
              (1, 0),
              (
                  cfg.embed_dim,
                  cfg.num_global_kv_heads
                  if cfg.num_global_kv_heads is not None
                  else cfg.num_kv_heads,
                  cfg.global_key_size
                  if cfg.global_key_size is not None
                  else cfg.head_dim,
              ),
          ),
      ),
      r"(?:model\.language_model\.)?layers\.({})\.self_attn\.v_proj\.weight"
      .format(
          global_pat
      ): (
          r"tmp.layers.\1.attn.v",
          (
              (1, 0),
              (
                  cfg.embed_dim,
                  cfg.num_global_kv_heads
                  if cfg.num_global_kv_heads is not None
                  else cfg.num_kv_heads,
                  cfg.global_key_size
                  if cfg.global_key_size is not None
                  else cfg.head_dim,
              ),
          ),
      ),
      r"(?:model\.language_model\.)?layers\.({})\.self_attn\.o_proj\.weight"
      .format(
          global_pat
      ): (
          r"layers.\1.attn.attn_vec_einsum.w",
          (
              (1, 0),
              (
                  cfg.num_heads,
                  cfg.global_key_size
                  if cfg.global_key_size is not None
                  else cfg.head_dim,
                  cfg.embed_dim,
              ),
          ),
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
          r"layers.\1.mlp.gate_proj.kernel",
          ((1, 0), None),
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.mlp\.up_proj\.weight": (
          r"layers.\1.mlp.up_proj.kernel",
          ((1, 0), None),
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.mlp\.down_proj\.weight": (
          r"layers.\1.mlp.down_proj.kernel",
          ((1, 0), None),
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.input_layernorm\.weight": (
          r"layers.\1.pre_attention_norm.scale",
          None,
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.post_attention_layernorm\.weight": (
          r"layers.\1.post_attention_norm.scale",
          None,
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.pre_feedforward_layernorm\.weight": (
          r"layers.\1.pre_ffw_norm.scale",
          None,
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.post_feedforward_layernorm\.weight": (
          r"layers.\1.post_ffw_norm.scale",
          None,
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.post_feedforward_layernorm_2\.weight": (
          r"layers.\1.moe_post_ffw_norm.scale",
          None,
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.post_feedforward_layernorm_1\.weight": (
          r"layers.\1.dense_post_ffw_norm.scale",
          None,
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.pre_feedforward_layernorm_2\.weight": (
          r"layers.\1.moe_pre_ffw_norm.scale",
          None,
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.layer_scalar": (
          r"layers.\1.skip_scale.value",
          None,
      ),
      r"(?:model\.language_model\.)?norm\.weight": ("final_norm.scale", None),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.self_attn\.q_norm\.weight": (
          r"layers.\1.attn._query_norm.scale",
          None,
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.self_attn\.k_norm\.weight": (
          r"layers.\1.attn._key_norm.scale",
          None,
      ),
      # Per-layer and secondary norm mappings
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.per_layer_input_gate\.weight": (
          r"layers.\1.per_layer_input_gate.w",
          ((1, 0), None),
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.per_layer_projection\.weight": (
          r"layers.\1.per_layer_projection.w",
          ((1, 0), None),
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.post_per_layer_input_norm\.weight": (
          r"layers.\1.post_per_layer_input_norm.scale",
          None,
      ),
      # MoE Router and Experts
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.router\.proj\.weight": (
          r"layers.\1.moe.router_logits.value",
          ((1, 0), None),
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.router\.per_expert_scale": (
          r"layers.\1.moe.per_expert_scale.value",
          None,
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.router\.scale": (
          r"layers.\1.moe.router_scale.value",
          None,
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.experts\.gate_up_proj(?:\.weight)?": (
          r"layers.\1.moe.gating_einsum.value",
          (None, (cfg.num_experts, 2, cfg.expert_dim, cfg.embed_dim)),
      ),
      r"(?:model\.language_model\.)?layers\.([0-9]+)\.experts\.down_proj(?:\.weight)?": (
          r"layers.\1.moe.linear.value",
          ((0, 2, 1), None),
      ),
  }

  return mapping


def _make_preprocess_fn(cfg: model_lib.ModelConfig):
  """Creates a tensor preprocessing function for Gemma 4 safetensors."""
  q_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.q$")
  k_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.k$")
  v_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.v$")

  pending: dict[str, dict[str, jnp.ndarray]] = {}

  def _to_ndh(q: jnp.ndarray, head_dim: int) -> jnp.ndarray:
    if q.shape == (cfg.num_heads, cfg.embed_dim, head_dim):
      return q
    if q.shape == (cfg.embed_dim, cfg.num_heads, head_dim):
      return jnp.transpose(q, (1, 0, 2))
    if q.shape == (cfg.num_heads, head_dim, cfg.embed_dim):
      return jnp.transpose(q, (0, 2, 1))
    raise ValueError(f"Unexpected q shape: {q.shape}")

  def _to_kdh(x: jnp.ndarray, head_dim: int, num_kv_heads: int) -> jnp.ndarray:
    # 2D shape handling (Heads * HeadDim, Hidden) or (Hidden, Heads * HeadDim)
    if x.ndim == 2:
      if x.shape == (num_kv_heads * head_dim, cfg.embed_dim):
        x = jnp.reshape(x, (num_kv_heads, head_dim, cfg.embed_dim))
        return jnp.transpose(x, (0, 2, 1))
      if x.shape == (cfg.embed_dim, num_kv_heads * head_dim):
        x = jnp.reshape(x, (cfg.embed_dim, num_kv_heads, head_dim))
        return jnp.transpose(x, (1, 0, 2))
      raise ValueError(
          f"Unexpected 2D kv shape: {x.shape}, expected dims to contain {num_kv_heads * head_dim} and {cfg.embed_dim}"
      )

    # 3D shape handling
    if cfg.embed_dim in x.shape and head_dim in x.shape:
      f_axis = x.shape.index(cfg.embed_dim)
      k_axis = x.shape.index(head_dim)
      all_axes = {0, 1, 2}
      h_axis = list(all_axes - {f_axis, k_axis})[0]

      x = jnp.transpose(x, (h_axis, f_axis, k_axis))
      return x[:num_kv_heads]
    raise ValueError(
        f"Unexpected kv shape: {x.shape}, expected dims to contain {cfg.embed_dim} and {head_dim}"
    )

  def preprocess(tensors: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    out = dict(tensors)

    for key in list(out):
      m = q_pat.fullmatch(key) or k_pat.fullmatch(key) or v_pat.fullmatch(key)
      if not m:
        continue
      layer_id = m.group(1)
      arr = out.pop(key)
      slot = "q" if key.endswith(".q") else ("k" if key.endswith(".k") else "v")
      pending.setdefault(layer_id, {})[slot] = arr

    # Resolve attention type for each layer using the pattern
    pattern = (
        cfg.attention_pattern
        if cfg.attention_pattern
        else model_lib.GEMMA4_ATTENTION_PATTERN
    )

    for layer_id_str, slots in list(pending.items()):
      layer_id = int(layer_id_str)
      q = slots.get("q")
      k = slots.get("k")
      v = slots.get("v")

      attn_type = pattern[layer_id % len(pattern)]

      effective_head_dim = cfg.head_dim
      effective_num_kv_heads = cfg.num_kv_heads
      if attn_type == model_lib.AttentionType.GLOBAL:
        if cfg.global_key_size is not None:
          effective_head_dim = cfg.global_key_size
        if cfg.num_global_kv_heads is not None:
          effective_num_kv_heads = cfg.num_global_kv_heads

      if q is not None:
        q = _to_ndh(q, effective_head_dim)
        out[f"layers.{layer_id_str}.attn.q_einsum.w"] = q
        slots.pop("q", None)

      k_eq_v_active = (
          cfg.k_eq_v_global
          if attn_type == model_lib.AttentionType.GLOBAL
          else False
      )

      if k_eq_v_active and (k is not None):
        k = _to_kdh(k, effective_head_dim, effective_num_kv_heads)
        out[f"layers.{layer_id_str}.attn.k_einsum.w"] = k
        slots.pop("k", None)
      elif (not k_eq_v_active) and (k is not None) and (v is not None):
        k = _to_kdh(k, effective_head_dim, effective_num_kv_heads)
        v = _to_kdh(v, effective_head_dim, effective_num_kv_heads)
        out[f"layers.{layer_id_str}.attn.kv_einsum.w"] = jnp.stack(
            [k, v], axis=0
        )
        slots.pop("k", None)
        slots.pop("v", None)

    return out

  return preprocess


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
):
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Gemma4,
      config=config,
      key_mapping=_get_key_and_transform_mapping,
      mesh=mesh,
      preprocess_fn=_make_preprocess_fn(config),
      dtype=dtype,
  )
