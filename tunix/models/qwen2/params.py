# Copyright 2025 Google LLC
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

"""Utils for loading and converting Qwen2 PT weights."""

from typing import Any

import jax
import jax.numpy as jnp
from tunix.models import hf_mapping
from tunix.models import safetensors_loader
from tunix.models.qwen2 import model as model_lib

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
    "target_pattern": ["final_norm.w"],
    "ops": []
  },
  {
    "source_pattern": ["model.layers.{i}.input_layernorm.weight"],
    "target_pattern": ["layers.{i}.input_layernorm.w"],
    "ops": []
  },
  {
    "source_pattern": ["model.layers.{i}.post_attention_layernorm.weight"],
    "target_pattern": ["layers.{i}.post_attention_layernorm.w"],
    "ops": []
  },
  {
    "source_pattern": ["model.layers.{i}.self_attn.q_proj.weight"],
    "target_pattern": ["layers.{i}.attn.q_proj.w"],
    "ops": [
      {"transpose": {"axes": [1, 0]}},
      {"reshape": {"src_shape": [$d, $nh], "dst_shape": [$d, $n, $h]}}
    ]
  },
  {
    "source_pattern": ["model.layers.{i}.self_attn.k_proj.weight"],
    "target_pattern": ["layers.{i}.attn.k_proj.w"],
    "ops": [
      {"transpose": {"axes": [1, 0]}},
      {"reshape": {"src_shape": [$d, $kh], "dst_shape": [$d, $k, $h]}}
    ]
  },
  {
    "source_pattern": ["model.layers.{i}.self_attn.v_proj.weight"],
    "target_pattern": ["layers.{i}.attn.v_proj.w"],
    "ops": [
      {"transpose": {"axes": [1, 0]}},
      {"reshape": {"src_shape": [$d, $kh], "dst_shape": [$d, $k, $h]}}
    ]
  },
  {
    "source_pattern": ["model.layers.{i}.self_attn.o_proj.weight"],
    "target_pattern": ["layers.{i}.attn.o_proj.w"],
    "ops": [
      {"transpose": {"axes": [1, 0]}},
      {"reshape": {"src_shape": [$nh, $d], "dst_shape": [$n, $h, $d]}}
    ]
  },
  {
    "source_pattern": ["model.layers.{i}.self_attn.q_proj.bias"],
    "target_pattern": ["layers.{i}.attn.q_bias"],
    "ops": []
  },
  {
    "source_pattern": ["model.layers.{i}.self_attn.k_proj.bias"],
    "target_pattern": ["layers.{i}.attn.k_bias"],
    "ops": []
  },
  {
    "source_pattern": ["model.layers.{i}.self_attn.v_proj.bias"],
    "target_pattern": ["layers.{i}.attn.v_bias"],
    "ops": []
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

_FROM_HF_LM_HEAD_JSON = """[
  {
    "source_pattern": ["lm_head.weight"],
    "target_pattern": ["lm_head.w"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  }
]"""


def build_from_hf_manifest(cfg: model_lib.ModelConfig) -> dict[str, Any]:
  """Builds the declarative PrimOp `HF -> Tunix` manifest for `cfg`."""
  d, n, k, h = cfg.embed_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
  templates = hf_mapping.render_json_templates(
      _FROM_HF_BASE_JSON,
      "" if cfg.use_tied_embedding else _FROM_HF_LM_HEAD_JSON,
      d=d,
      n=n,
      k=k,
      h=h,
      nh=n * h,
      kh=k * h,
  )
  return hf_mapping.generate_manifest(
      model_type="qwen2",
      template_rules=templates,
      axes={"i": cfg.num_layers},
      dropped_keys=["lm_head.weight"] if cfg.use_tied_embedding else [],
  )


def build_to_hf_manifest(cfg: model_lib.ModelConfig) -> dict[str, Any]:
  """Builds the declarative PrimOp `Tunix -> HF` manifest for `cfg`."""
  return hf_mapping.invert_manifest(build_from_hf_manifest(cfg))


def export_to_hf_state_dict(
    model: model_lib.Qwen2, cfg: model_lib.ModelConfig
) -> dict[str, jax.Array]:
  """Exports a Tunix `Qwen2` model into a HuggingFace state dict."""
  flat_state = safetensors_loader.flatten_model_state(model)
  return hf_mapping.to_hf_impl(build_to_hf_manifest(cfg), flat_state)


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
    mode: str = "auto",
) -> model_lib.Qwen2:
  """Load tensors from the safetensors file and create a Qwen2 model."""
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Qwen2,
      config=config,
      mesh=mesh,
      dtype=dtype,
      mode=mode,
      manifest=build_from_hf_manifest(config),
  )
