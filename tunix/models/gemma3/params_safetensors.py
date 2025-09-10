# Copyright 2025 Google
# Licensed under the Apache License, Version 2.0

"""Loads Gemma3 parameters from safetensors files."""

import re
import jax
import jax.numpy as jnp
from tunix.models import safetensors_loader
from tunix.models.gemma3 import model as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.Gemma3Config):
  # Mapping of torch_keys -> (nnx_keys, (permute_rule, reshape_rule)).
  return {
      r"model\.embed_tokens\.weight": ("embedder.input_embedding", None),
      r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
          r"tmp.layers.\1.attn.q",
          ((1, 0), (cfg.num_heads, cfg.embed_dim, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
          r"tmp.layers.\1.attn.k",
          ((1, 0), (cfg.num_kv_heads, cfg.embed_dim, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
          r"tmp.layers.\1.attn.v",
          ((1, 0), (cfg.num_kv_heads, cfg.embed_dim, cfg.head_dim)),
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
          r"layers.\1.post_attention_norm.scale",
          None,
      ),
      r"model\.layers\.([0-9]+)\.(post_feedforward_layernorm|post_ffn_layernorm|post_ffw_layernorm)\.weight": (
          r"layers.\1.post_ffw_norm.scale",
          None,
      ),
      r"model\.norm\.weight": ("final_norm.scale", None),
      r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": (
          r"layers.\1.attn._query_norm.scale",
          None,
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": (
          r"layers.\1.attn._key_norm.scale",
          None,
      ),
      r"lm_head\.weight": ("unused.lm_head.weight", None),
      r"lm_head\.bias": ("unused.lm_head.bias", None),
      r"model\.layers\.([0-9]+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.bias": (
          r"unused.layers.\1.attn.\2.bias",
          None,
      ),
      r"model\.layers\.([0-9]+)\.input_layernorm\.bias": (
          r"unused.layers.\1.input_layernorm.bias",
          None,
      ),
      r"model\.layers\.([0-9]+)\.post_attention_layernorm\.bias": (
          r"unused.layers.\1.post_attention_layernorm.bias",
          None,
      ),
      r"model\.rotary_emb\..*": ("unused.rotary_emb", None),
      r"model\.layers\.([0-9]+)\.self_attn\.rotary_emb\..*": (
          r"unused.layers.\1.attn.rotary_emb",
          None,
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.qkv_proj\.weight": (
          r"unused.layers.\1.attn.qkv_proj.weight",
          None,
      ),
      r"model\.layers\.([0-9]+)\.pre_feedforward_layernorm\.weight": (
          r"layers.\1.pre_ffw_norm.scale",
          None,
      ),
      r"model\.layers\.([0-9]+)\.(pre_ffn_layernorm|pre_ffw_layernorm)\.weight": (
          r"layers.\1.pre_ffw_norm.scale",
          None,
      ),
  }


def _make_preprocess_fn(cfg: model_lib.Gemma3Config):
  """Creates a preprocess function to fuse 'q', 'k', and 'v' tensors.

  Args:
    cfg: The Gemma3 model configuration.

  Returns:
    A function that takes a dictionary of tensors and returns a new dictionary
    with 'q', 'k', and 'v' tensors potentially fused.
  """
  q_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.q$")
  k_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.k$")
  v_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.v$")

  fused_qkv = cfg.num_heads == cfg.num_kv_heads
  pending: dict[str, dict[str, jnp.ndarray]] = {}

  if cfg.head_dim % 2 != 0:
    raise ValueError(
        f"Gemma3 head_dim must be even for RoPE, got {cfg.head_dim}"
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

    for layer_id, slots in list(pending.items()):
      q = slots.get("q")
      k = slots.get("k")
      v = slots.get("v")

      if fused_qkv:
        if (q is not None) and (k is not None) and (v is not None):
          exp_qkv = (cfg.num_heads, cfg.embed_dim, cfg.head_dim)
          if not (q.shape == k.shape == v.shape == exp_qkv):
            raise ValueError(
                f"[gemma3 preprocess] layer {layer_id}: q/k/v shape mismatch:"
                f" q={getattr(q, 'shape', None)},"
                f" k={getattr(k, 'shape', None)},"
                f" v={getattr(v, 'shape', None)},"
                f" expected={exp_qkv}"
            )
          out[f"layers.{layer_id}.attn.qkv_einsum.w"] = jnp.stack(
              [q, k, v], axis=0
          )
          del pending[layer_id]
      else:
        wrote = False
        if q is not None:
          exp_q = (cfg.num_heads, cfg.embed_dim, cfg.head_dim)
          if q.shape != exp_q:
            raise ValueError(
                f"[gemma3 preprocess] layer {layer_id}: q shape mismatch:"
                f" got {q.shape}, expected {exp_q}"
            )
          out[f"layers.{layer_id}.attn.q_einsum.w"] = q
          slots.pop("q")
          wrote = True

        if (k is not None) and (v is not None):
          exp_kv = (cfg.num_kv_heads, cfg.embed_dim, cfg.head_dim)
          if not (k.shape == v.shape == exp_kv):
            raise ValueError(
                f"[gemma3 preprocess] layer {layer_id}: k/v shape mismatch:"
                f" k={getattr(k, 'shape', None)},"
                f" v={getattr(v, 'shape', None)}, expected={exp_kv}"
            )
          out[f"layers.{layer_id}.attn.kv_einsum.w"] = jnp.stack([k, v], axis=0)
          slots.pop("k", None)
          slots.pop("v", None)
          wrote = True

        if wrote and not slots:
          del pending[layer_id]

    return out

  return preprocess


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.Gemma3Config,
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
