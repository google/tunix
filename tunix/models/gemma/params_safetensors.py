# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0

"""Loads Gemma2 parameters from safetensors files."""

import dataclasses
import os
import re

import jax
import jax.numpy as jnp
import safetensors
from tunix.models import safetensors_loader
from tunix.models.gemma import model as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  """Mapping of torch_keys to (nnx_keys, (permute_rule, reshape_rule))."""
  mapping = {
      r"model\.embed_tokens\.weight": ("embedder.input_embedding", None),
      r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
          r"tmp.layers.\1.attn.q",
          ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
          r"tmp.layers.\1.attn.k",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
          r"tmp.layers.\1.attn.v",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
          r"layers.\1.attn.attn_vec_einsum.w",
          ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
      ),
      r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
          r"layers.\1.mlp.gate_proj.kernel",
          ((1, 0), None),
      ),
      r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
          r"layers.\1.mlp.up_proj.kernel",
          ((1, 0), None),
      ),
      r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
          r"layers.\1.mlp.down_proj.kernel",
          ((1, 0), None),
      ),
      r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (
          r"layers.\1.pre_attention_norm.scale",
          None,
      ),
      r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
          r"layers.\1.post_attn_norm.scale",
          None,
      ),
      r"model\.layers\.([0-9]+)\.pre_feedforward_layernorm\.weight": (
          r"layers.\1.pre_ffw_norm.scale",
          None,
      ),
      r"model\.layers\.([0-9]+)\.post_feedforward_layernorm\.weight": (
          r"layers.\1.post_ffw_norm.scale",
          None,
      ),
      r"model\.norm\.weight": ("final_norm.scale", None),
      r"lm_head\.weight": ("unused.lm_head.weight", None),
      r"lm_head\.bias": ("unused.lm_head.bias", None),
      r"model\.layers\.([0-9]+)\.self_attn\.rotary_emb\..*": (
          r"unused.rotary.\1",
          None,
      ),
      r".*\.bias": (r"unused.bias", None),
  }
  return mapping


def _make_preprocess_fn(cfg: model_lib.ModelConfig):
  """Creates a preprocess function to reshape and stack Q, K, and V tensors for Gemma safetensors."""
  q_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.q$")
  k_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.k$")
  v_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.v$")
  fused_qkv = cfg.num_heads == cfg.num_kv_heads
  pending: dict[str, dict[str, jnp.ndarray]] = {}
  if cfg.head_dim % 2 != 0:
    raise ValueError(
        f"Gemma2 head_dim must be even for RoPE, got {cfg.head_dim}"
    )

  def _to_ndh(q: jnp.ndarray) -> jnp.ndarray:
    if q.shape == (cfg.num_heads, cfg.embed_dim, cfg.head_dim):
      return q
    if q.shape == (cfg.embed_dim, cfg.num_heads, cfg.head_dim):
      return jnp.transpose(q, (1, 0, 2))
    if q.shape == (cfg.num_heads, cfg.head_dim, cfg.embed_dim):
      return jnp.transpose(q, (0, 2, 1))
    raise ValueError(f"[gemma2 preprocess] unexpected q shape: {q.shape}")

  def _to_kdh(x: jnp.ndarray) -> jnp.ndarray:
    if x.shape == (cfg.num_kv_heads, cfg.embed_dim, cfg.head_dim):
      return x
    if x.shape == (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim):
      return jnp.transpose(x, (1, 0, 2))
    if x.shape == (cfg.num_kv_heads, cfg.head_dim, cfg.embed_dim):
      return jnp.transpose(x, (0, 2, 1))
    raise ValueError(f"[gemma2 preprocess] unexpected kv shape: {x.shape}")

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
                f"[gemma2 preprocess] layer {layer_id}: fused q/k/v shape"
                f" mismatch: q={getattr(q, 'shape', None)},"
                f" k={getattr(k, 'shape', None)},"
                f" v={getattr(v, 'shape', None)}, expected={exp}"
            )
          out[f"layers.{layer_id}.attn.qkv_einsum.w"] = jnp.stack(
              [q, k, v], axis=0
          )
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
          out[f"layers.{layer_id}.attn.kv_einsum.w"] = jnp.stack([k, v], axis=0)
          slots.pop("k", None)
          slots.pop("v", None)
          wrote = True
        if wrote and not slots:
          del pending[layer_id]
    return out
  return preprocess


def _peek_vocab_size_from_safetensors(file_dir: str) -> int:
  for fn in os.listdir(file_dir):
    if fn.endswith(".safetensors"):
      path = os.path.join(file_dir, fn)
      with safetensors.safe_open(path, framework="jax") as f:
        if "model.embed_tokens.weight" in f.keys():
          shape = f.get_tensor("model.embed_tokens.weight").shape
          return shape[0]
  raise FileNotFoundError("No .safetensors found to peek vocab size")


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
) -> model_lib.Gemma:
  v_ckpt = _peek_vocab_size_from_safetensors(file_dir)
  if v_ckpt != config.num_embed:
    config = dataclasses.replace(config, num_embed=v_ckpt)
    print(f"[gemma2] override num_embed -> {v_ckpt} from checkpoint")
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Gemma,
      config=config,
      key_mapping=_get_key_and_transform_mapping,
      mesh=mesh,
      preprocess_fn=_make_preprocess_fn(config),
      dtype=dtype,
  )
