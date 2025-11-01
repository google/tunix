# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0

"""Loads Gemma3 parameters from safetensors files."""

import re

import jax
import jax.numpy as jnp
from tunix.models import safetensors_loader
from tunix.models.gemma3 import model as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  """Mapping of torch_keys to (nnx_keys, (permute_rule, reshape_rule))."""
  return {
      r"(language_model\.)?model\.embed_tokens\.weight": (
          "embedder.input_embedding",
          None,
      ),
      r"(language_model\.)?model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
          r"tmp.layers.\2.attn.q",
          ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim)),
      ),
      r"(language_model\.)?model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
          r"tmp.layers.\2.attn.k",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"(language_model\.)?model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
          r"tmp.layers.\2.attn.v",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"(language_model\.)?model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
          r"layers.\2.attn.attn_vec_einsum.w",
          ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
      ),
      r"(language_model\.)?model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
          r"layers.\2.mlp.gate_proj.kernel",
          ((1, 0), None),
      ),
      r"(language_model\.)?model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
          r"layers.\2.mlp.up_proj.kernel",
          ((1, 0), None),
      ),
      r"(language_model\.)?model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
          r"layers.\2.mlp.down_proj.kernel",
          ((1, 0), None),
      ),
      r"(language_model\.)?model\.layers\.([0-9]+)\.input_layernorm\.weight": (
          r"layers.\2.pre_attention_norm.scale",
          None,
      ),
      r"(language_model\.)?model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
          r"layers.\2.post_attention_norm.scale",
          None,
      ),
      r"(language_model\.)?model\.layers\.([0-9]+)\.(post_feedforward_layernorm|post_ffn_layernorm|post_ffw_layernorm)\.weight": (
          r"layers.\2.post_ffw_norm.scale",
          None,
      ),
      r"(language_model\.)?model\.norm\.weight": ("final_norm.scale", None),
      r"(language_model\.)?model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": (
          r"layers.\2.attn._query_norm.scale",
          None,
      ),
      r"(language_model\.)?model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": (
          r"layers.\2.attn._key_norm.scale",
          None,
      ),
      r"lm_head\.weight": ("unused.lm_head.weight", None),
      r"lm_head\.bias": ("unused.lm_head.bias", None),
      r"(language_model\.)?model\.layers\.([0-9]+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.bias": (
          r"unused.layers.\2.attn.\3.bias",
          None,
      ),
      r"(language_model\.)?model\.layers\.([0-9]+)\.input_layernorm\.bias": (
          r"unused.layers.\2.input_layernorm.bias",
          None,
      ),
      r"(language_model\.)?model\.layers\.([0-9]+)\.post_attention_layernorm\.bias": (
          r"unused.layers.\2.post_attention_layernorm.bias",
          None,
      ),
      r"(language_model\.)?model\.rotary_emb\..*": ("unused.rotary_emb", None),
      r"(language_model\.)?model\.layers\.([0-9]+)\.self_attn\.rotary_emb\..*": (
          r"unused.layers.\2.attn.rotary_emb",
          None,
      ),
      r"(language_model\.)?model\.layers\.([0-9]+)\.self_attn\.qkv_proj\.weight": (
          r"unused.layers.\2.attn.qkv_proj.weight",
          None,
      ),
      r"(language_model\.)?model\.layers\.([0-9]+)\.pre_feedforward_layernorm\.weight": (
          r"layers.\2.pre_ffw_norm.scale",
          None,
      ),
      r"(language_model\.)?model\.layers\.([0-9]+)\.(pre_ffn_layernorm|pre_ffw_layernorm)\.weight": (
          r"layers.\2.pre_ffw_norm.scale",
          None,
      ),
      r"multi_modal_projector.mm_input_projection_weight": (
          r"projector.mm_input_projection.kernel",
          None,
      ),
      r"multi_modal_projector.mm_soft_emb_norm.weight": (
          r"projector.mm_soft_emb_norm.scale",
          None,
      ),
      r"vision_tower\.vision_model\.embeddings\.patch_embedding\.bias": (
          r"siglip.patch.proj.bias",
          None,
      ),
      r"vision_tower\.vision_model\.embeddings\.patch_embedding\.weight": (
          r"siglip.patch.proj.kernel",
          ((2, 3, 1, 0), None),
      ),
      r"vision_tower\.vision_model\.embeddings\.position_embedding\.weight": (
          r"siglip.pos_embed",
          (None, (1, -1, 1152)),
      ),
      r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.layer_norm1\.bias": (
          r"siglip.blocks.\1.ln1.bias",
          None,
      ),
      r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.layer_norm1\.weight": (
          r"siglip.blocks.\1.ln1.scale",
          None,
      ),
      r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.layer_norm2\.bias": (
          r"siglip.blocks.\1.ln2.bias",
          None,
      ),
      r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.layer_norm2\.weight": (
          r"siglip.blocks.\1.ln2.scale",
          None,
      ),
      r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.mlp\.fc1\.bias": (
          r"siglip.blocks.\1.mlp.fc1.bias",
          None,
      ),
      r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.mlp\.fc1\.weight": (
          r"siglip.blocks.\1.mlp.fc1.kernel",
          ((1, 0), None),
      ),
      r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.mlp\.fc2\.bias": (
          r"siglip.blocks.\1.mlp.fc2.bias",
          None,
      ),
      r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.mlp\.fc2\.weight": (
          r"siglip.blocks.\1.mlp.fc2.kernel",
          ((1, 0), None),
      ),
      r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.([qkv])_proj\.bias": (
          r"siglip.blocks.\1.attn.\2.bias",
          None,
      ),
      r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.([qkv])_proj\.weight": (
          r"siglip.blocks.\1.attn.\2.kernel",
          ((1, 0), None),
      ),
      r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.bias": (
          r"siglip.blocks.\1.attn.o.bias",
          None,
      ),
      r"vision_tower\.vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.weight": (
          r"siglip.blocks.\1.attn.o.kernel",
          ((1, 0), None),
      ),
      r"vision_tower\.vision_model\.post_layernorm\.bias": (
          r"siglip.norm.bias",
          None,
      ),
      r"vision_tower\.vision_model\.post_layernorm\.weight": (
          r"siglip.norm.scale",
          None,
      ),
  }


def _make_preprocess_fn(cfg: model_lib.ModelConfig):
  """Creates a tensor preprocessing function for Gemma3 safetensors, fusing q, k, and v projections."""
  q_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.q$")
  k_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.k$")
  v_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.v$")

  fused_qkv = cfg.num_heads == cfg.num_kv_heads
  pending: dict[str, dict[str, jnp.ndarray]] = {}

  if cfg.head_dim % 2 != 0:
    raise ValueError(
        f"Gemma3 head_dim must be even for RoPE, got {cfg.head_dim}"
    )

  def _to_ndh(q: jnp.ndarray) -> jnp.ndarray:
    # Expected shape: (N, D, H)
    if q.shape == (cfg.num_heads, cfg.embed_dim, cfg.head_dim):
      return q
    if q.shape == (cfg.embed_dim, cfg.num_heads, cfg.head_dim):  # D,N,H
      return jnp.transpose(q, (1, 0, 2))  # -> N,D,H
    if q.shape == (cfg.num_heads, cfg.head_dim, cfg.embed_dim):  # N,H,D
      return jnp.transpose(q, (0, 2, 1))  # -> N,D,H
    raise ValueError(f"[gemma3 preprocess] unexpected q shape: {q.shape}")

  def _to_kdh(x: jnp.ndarray) -> jnp.ndarray:
    if x.shape == (cfg.num_kv_heads, cfg.embed_dim, cfg.head_dim):
      return x
    if x.shape == (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim):  # D,K,H
      return jnp.transpose(x, (1, 0, 2))  # -> K,D,H
    if x.shape == (cfg.num_kv_heads, cfg.head_dim, cfg.embed_dim):  # K,H,D
      return jnp.transpose(x, (0, 2, 1))  # -> K,D,H
    raise ValueError(f"[gemma3 preprocess] unexpected kv shape: {x.shape}")

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

    for layer_id, slots in list(pending.items()):
      q = slots.get("q")
      k = slots.get("k")
      v = slots.get("v")

      if fused_qkv:
        if (q is not None) and (k is not None) and (v is not None):
          q = _to_ndh(q)
          k = _to_kdh(k)
          v = _to_kdh(v)
          exp = (cfg.num_heads, cfg.embed_dim, cfg.head_dim)
          if not (q.shape == k.shape == v.shape == exp):
            raise ValueError(
                f"[gemma3 preprocess] layer {layer_id}: fused q/k/v shape"
                f" mismatch: q={getattr(q, 'shape', None)},"
                f" k={getattr(k, 'shape', None)},"
                f" v={getattr(v, 'shape', None)}, expected={exp}"
            )
          out[f"layers.{layer_id}.attn.qkv_einsum.w"] = jnp.stack(
              [q, k, v], axis=0
          )  # (3,N,D,H)
          del pending[layer_id]
      else:
        wrote = False
        if q is not None:
          q = _to_ndh(q)
          out[f"layers.{layer_id}.attn.q_einsum.w"] = q
          slots.pop("q", None)
          wrote = True

        if (k is not None) and (v is not None):
          k = _to_kdh(k)
          v = _to_kdh(v)
          out[f"layers.{layer_id}.attn.kv_einsum.w"] = jnp.stack(
              [k, v], axis=0
          )  # (2,K,D,H)
          slots.pop("k", None)
          slots.pop("v", None)
          wrote = True

        if wrote and not slots:
          del pending[layer_id]

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
      model_class=model_lib.Gemma3,
      config=config,
      key_mapping=_get_key_and_transform_mapping,
      mesh=mesh,
      preprocess_fn=_make_preprocess_fn(config),
      dtype=dtype,
  )
