# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0

"""Loads Gemma2 parameters from safetensors files."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from tunix.models import hf_mapping
from tunix.models import safetensors_loader
from tunix.models.gemma import model as model_lib

from_hf_impl = hf_mapping.from_hf_impl
to_hf_impl = hf_mapping.to_hf_impl


_FROM_HF_BASE_JSON = """[
  {
    "source_pattern": ["model.embed_tokens.weight"],
    "target_pattern": ["embedder.input_embedding"],
    "ops": []
  },
  {
    "source_pattern": ["model.norm.weight"],
    "target_pattern": ["final_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": ["model.layers.{i}.input_layernorm.weight"],
    "target_pattern": ["layers.{i}.pre_attention_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": ["model.layers.{i}.self_attn.o_proj.weight"],
    "target_pattern": ["layers.{i}.attn.attn_vec_einsum.w"],
    "ops": [
      {"transpose": {"axes": [1, 0]}},
      {"reshape": {"src_shape": [$nh, $d], "dst_shape": [$n, $h, $d]}}
    ]
  },
  {
    "source_pattern": ["model.layers.{i}.mlp.gate_proj.weight"],
    "target_pattern": ["layers.{i}.mlp.gate_proj.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["model.layers.{i}.mlp.up_proj.weight"],
    "target_pattern": ["layers.{i}.mlp.up_proj.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["model.layers.{i}.mlp.down_proj.weight"],
    "target_pattern": ["layers.{i}.mlp.down_proj.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  }
]"""

# Gemma 1 / 1.1 (`use_post_attn_norm=False`) names its pre-MLP RMSNorm
# `post_attention_layernorm`; Gemma 2 (`use_post_attn_norm=True`) uses both
# `post_attention_layernorm` (post-attn) and `pre_feedforward_layernorm`.
_FROM_HF_GEMMA1_PRE_FFW_NORM_JSON = """[
  {
    "source_pattern": ["model.layers.{i}.post_attention_layernorm.weight"],
    "target_pattern": ["layers.{i}.pre_ffw_norm.scale"],
    "ops": []
  }
]"""

_FROM_HF_GEMMA2_ATTN_FFW_NORMS_JSON = """[
  {
    "source_pattern": ["model.layers.{i}.post_attention_layernorm.weight"],
    "target_pattern": ["layers.{i}.post_attn_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": ["model.layers.{i}.pre_feedforward_layernorm.weight"],
    "target_pattern": ["layers.{i}.pre_ffw_norm.scale"],
    "ops": []
  }
]"""

_FROM_HF_POST_FFW_NORM_JSON = """[
  {
    "source_pattern": ["model.layers.{i}.post_feedforward_layernorm.weight"],
    "target_pattern": ["layers.{i}.post_ffw_norm.scale"],
    "ops": []
  }
]"""

_FROM_HF_GQA_JSON = """[
  {
    "source_pattern": ["model.layers.{i}.self_attn.q_proj.weight"],
    "target_pattern": ["layers.{i}.attn.q_einsum.w"],
    "ops": [
      {"reshape": {"src_shape": [$nh, $d], "dst_shape": [$n, $h, $d]}},
      {"transpose": {"axes": [0, 2, 1]}}
    ]
  },
  {
    "source_pattern": [
      "model.layers.{i}.self_attn.k_proj.weight",
      "model.layers.{i}.self_attn.v_proj.weight"
    ],
    "target_pattern": ["layers.{i}.attn.kv_einsum.w"],
    "ops": [
      {"reshape": {"src_shape": [$kh, $d], "dst_shape": [$k, $h, $d]}},
      {"transpose": {"axes": [0, 2, 1]}},
      {"stack": {"axis": 0}}
    ]
  }
]"""

_FROM_HF_MHA_JSON = """[
  {
    "source_pattern": [
      "model.layers.{i}.self_attn.q_proj.weight",
      "model.layers.{i}.self_attn.k_proj.weight",
      "model.layers.{i}.self_attn.v_proj.weight"
    ],
    "target_pattern": ["layers.{i}.attn.qkv_einsum.w"],
    "ops": [
      {"reshape": {"src_shape": [$nh, $d], "dst_shape": [$n, $h, $d]}},
      {"transpose": {"axes": [0, 2, 1]}},
      {"stack": {"axis": 0}}
    ]
  }
]"""


def build_from_hf_manifest(cfg: model_lib.ModelConfig) -> dict[str, Any]:
  """Builds the declarative PrimOp `HF -> Tunix` manifest for `cfg`."""
  if cfg.head_dim % 2 != 0:
    raise ValueError(
        f"Gemma2 head_dim must be even for RoPE, got {cfg.head_dim}"
    )
  d, n, k, h = cfg.embed_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
  templates = hf_mapping.render_json_templates(
      _FROM_HF_BASE_JSON,
      (
          _FROM_HF_GEMMA2_ATTN_FFW_NORMS_JSON
          if cfg.use_post_attn_norm
          else _FROM_HF_GEMMA1_PRE_FFW_NORM_JSON
      ),
      _FROM_HF_POST_FFW_NORM_JSON if cfg.use_post_ffw_norm else "",
      _FROM_HF_MHA_JSON if n == k else _FROM_HF_GQA_JSON,
      d=d,
      n=n,
      k=k,
      h=h,
      nh=n * h,
      kh=k * h,
  )
  return hf_mapping.generate_manifest(
      model_type="gemma",
      template_rules=templates,
      axes={"i": cfg.num_layers},
      dropped_keys=["lm_head.weight"],
  )


def build_to_hf_manifest(cfg: model_lib.ModelConfig) -> dict[str, Any]:
  """Builds the declarative PrimOp `Tunix -> HF` manifest for `cfg`."""
  return hf_mapping.invert_manifest(build_from_hf_manifest(cfg))


def export_to_hf_state_dict(
    model: model_lib.Gemma, cfg: model_lib.ModelConfig
) -> dict[str, jax.Array]:
  """Exports a Tunix `Gemma` model into a HuggingFace state dict."""
  flat_state = safetensors_loader.flatten_model_state(model)
  return hf_mapping.to_hf_impl(build_to_hf_manifest(cfg), flat_state)


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
    mode: str = "auto",
) -> model_lib.Gemma:
  """Loads HuggingFace safetensors checkpoints into a Gemma model."""
  local_dir = safetensors_loader.resolve_checkpoint_dir(file_dir)
  headers = safetensors_loader.read_safetensors_headers(local_dir)
  embed_key = "model.embed_tokens.weight"
  if embed_key not in headers:
    raise FileNotFoundError(
        f"{embed_key} not found in safetensors at {local_dir}"
    )
  v_ckpt = int(headers[embed_key]["shape"][0])
  if v_ckpt != config.num_embed:
    config = dataclasses.replace(config, num_embed=v_ckpt)
  return safetensors_loader.load_and_create_model(
      file_dir=local_dir,
      model_class=model_lib.Gemma,
      config=config,
      mesh=mesh,
      dtype=dtype,
      mode=mode,
      manifest=build_from_hf_manifest(config),
  )
