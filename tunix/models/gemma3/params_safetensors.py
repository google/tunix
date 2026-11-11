# Copyright 2025 Google LLC
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

"""Loads Gemma3 parameters from safetensors files."""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from tunix.models import hf_mapping
from tunix.models import safetensors_loader
from tunix.models.gemma3 import model as model_lib

HfVariant = model_lib.HfVariant
from_hf_impl = hf_mapping.from_hf_impl
to_hf_impl = hf_mapping.to_hf_impl


_FROM_HF_BASE_JSON = """[
  {
    "source_pattern": ["${lm}embed_tokens.weight"],
    "target_pattern": ["embedder.input_embedding"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}norm.weight"],
    "target_pattern": ["final_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}layers.{i}.input_layernorm.weight"],
    "target_pattern": ["layers.{i}.pre_attention_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}layers.{i}.post_attention_layernorm.weight"],
    "target_pattern": ["layers.{i}.post_attention_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}layers.{i}.pre_feedforward_layernorm.weight"],
    "target_pattern": ["layers.{i}.pre_ffw_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}layers.{i}.post_feedforward_layernorm.weight"],
    "target_pattern": ["layers.{i}.post_ffw_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}layers.{i}.self_attn.q_norm.weight"],
    "target_pattern": ["layers.{i}.attn._query_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}layers.{i}.self_attn.k_norm.weight"],
    "target_pattern": ["layers.{i}.attn._key_norm.scale"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}layers.{i}.self_attn.o_proj.weight"],
    "target_pattern": ["layers.{i}.attn.attn_vec_einsum.w"],
    "ops": [
      {"transpose": {"axes": [1, 0]}},
      {"reshape": {"src_shape": [$nh, $d], "dst_shape": [$n, $h, $d]}}
    ]
  },
  {
    "source_pattern": ["${lm}layers.{i}.mlp.gate_proj.weight"],
    "target_pattern": ["layers.{i}.mlp.gate_proj.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["${lm}layers.{i}.mlp.up_proj.weight"],
    "target_pattern": ["layers.{i}.mlp.up_proj.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["${lm}layers.{i}.mlp.down_proj.weight"],
    "target_pattern": ["layers.{i}.mlp.down_proj.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  }
]"""

_FROM_HF_GQA_JSON = """[
  {
    "source_pattern": ["${lm}layers.{i}.self_attn.q_proj.weight"],
    "target_pattern": ["layers.{i}.attn.q_einsum.w"],
    "ops": [
      {"reshape": {"src_shape": [$nh, $d], "dst_shape": [$n, $h, $d]}},
      {"transpose": {"axes": [0, 2, 1]}}
    ]
  },
  {
    "source_pattern": [
      "${lm}layers.{i}.self_attn.k_proj.weight",
      "${lm}layers.{i}.self_attn.v_proj.weight"
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
      "${lm}layers.{i}.self_attn.q_proj.weight",
      "${lm}layers.{i}.self_attn.k_proj.weight",
      "${lm}layers.{i}.self_attn.v_proj.weight"
    ],
    "target_pattern": ["layers.{i}.attn.qkv_einsum.w"],
    "ops": [
      {"reshape": {"src_shape": [$nh, $d], "dst_shape": [$n, $h, $d]}},
      {"transpose": {"axes": [0, 2, 1]}},
      {"stack": {"axis": 0}}
    ]
  }
]"""

_FROM_HF_VISION_JSON = """[
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.embeddings.patch_embedding.weight"
    ],
    "target_pattern": ["vision_encoder.siglip_encoder.embedding.kernel"],
    "ops": [{"transpose": {"axes": [2, 3, 1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.embeddings.patch_embedding.bias"
    ],
    "target_pattern": ["vision_encoder.siglip_encoder.embedding.bias"],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.embeddings.position_embedding.weight"
    ],
    "target_pattern": ["vision_encoder.siglip_encoder.pos_embedding"],
    "ops": [{"reshape": {"src_shape": [$vp, $vw], "dst_shape": [1, $vp, $vw]}}]
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.encoder.layers.{v}.self_attn.q_proj.weight"
    ],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.blocks.{v}.attn.query_proj.kernel"
    ],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.encoder.layers.{v}.self_attn.k_proj.weight"
    ],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.blocks.{v}.attn.key_proj.kernel"
    ],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.encoder.layers.{v}.self_attn.v_proj.weight"
    ],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.blocks.{v}.attn.value_proj.kernel"
    ],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.encoder.layers.{v}.self_attn.out_proj.weight"
    ],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.blocks.{v}.attn.out_proj.kernel"
    ],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.encoder.layers.{v}.self_attn.q_proj.bias"
    ],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.blocks.{v}.attn.query_proj.bias"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.encoder.layers.{v}.self_attn.k_proj.bias"
    ],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.blocks.{v}.attn.key_proj.bias"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.encoder.layers.{v}.self_attn.v_proj.bias"
    ],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.blocks.{v}.attn.value_proj.bias"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.encoder.layers.{v}.self_attn.out_proj.bias"
    ],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.blocks.{v}.attn.out_proj.bias"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.encoder.layers.{v}.layer_norm1.weight"
    ],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.blocks.{v}.ln1.scale"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.encoder.layers.{v}.layer_norm1.bias"
    ],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.blocks.{v}.ln1.bias"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.encoder.layers.{v}.layer_norm2.weight"
    ],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.blocks.{v}.ln2.scale"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.encoder.layers.{v}.layer_norm2.bias"
    ],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.blocks.{v}.ln2.bias"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.encoder.layers.{v}.mlp.fc1.weight"
    ],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.blocks.{v}.mlp.fc1.kernel"
    ],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.encoder.layers.{v}.mlp.fc2.weight"
    ],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.blocks.{v}.mlp.fc2.kernel"
    ],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.encoder.layers.{v}.mlp.fc1.bias"
    ],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.blocks.{v}.mlp.fc1.bias"
    ],
    "ops": []
  },
  {
    "source_pattern": [
      "${mm}vision_tower.vision_model.encoder.layers.{v}.mlp.fc2.bias"
    ],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.blocks.{v}.mlp.fc2.bias"
    ],
    "ops": []
  },
  {
    "source_pattern": ["${mm}vision_tower.vision_model.post_layernorm.weight"],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.encoder_norm.scale"
    ],
    "ops": []
  },
  {
    "source_pattern": ["${mm}vision_tower.vision_model.post_layernorm.bias"],
    "target_pattern": [
      "vision_encoder.siglip_encoder.transformer.encoder_norm.bias"
    ],
    "ops": []
  },
  {
    "source_pattern": ["${mm}multi_modal_projector.mm_input_projection_weight"],
    "target_pattern": ["embedder.mm_input_projection.w"],
    "ops": []
  },
  {
    "source_pattern": ["${mm}multi_modal_projector.mm_soft_emb_norm.weight"],
    "target_pattern": ["embedder.mm_soft_embedding_norm.scale"],
    "ops": []
  }
]"""


# Template placeholder values per variant: `lm` prefixes each text decoder key,
# `mm` each vision tower / projector key, and `head` the tied `lm_head.weight`.
_HF_AFFIXES: Mapping[HfVariant, Mapping[str, str]] = {
    HfVariant.CAUSAL_LM: {"lm": "model.", "head": ""},
    HfVariant.CONDITIONAL_GENERATION: {
        "lm": "model.language_model.",
        "mm": "model.",
        "head": "",
    },
    HfVariant.CONDITIONAL_GENERATION_LEGACY: {
        "lm": "language_model.model.",
        "mm": "",
        "head": "language_model.",
    },
}


def build_from_hf_manifest(
    cfg: model_lib.ModelConfig,
    hf_variant: HfVariant | None = None,
) -> dict[str, Any]:
  """Builds the `HF -> Tunix` manifest for `cfg`."""
  if not isinstance(hf_variant, HfVariant):
    raise ValueError(
        f"Pass `hf_variant` (HfVariant) explicitly; got {hf_variant!r}."
    )
  if cfg.head_dim % 2 != 0:
    raise ValueError(
        f"Gemma3 head_dim must be even for RoPE, got {cfg.head_dim}"
    )
  d, n, k, h = cfg.embed_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
  v_cfg = cfg.vision_config
  v_layers, vp, vw = 0, 1, 1
  if v_cfg is not None:
    v_layers = v_cfg.depth
    vp = (v_cfg.image_height // v_cfg.patch_size[0]) * (
        v_cfg.image_width // v_cfg.patch_size[1]
    )
    vw = v_cfg.width
  affixes = _HF_AFFIXES[hf_variant]
  if v_layers > 0 and "mm" not in affixes:
    raise ValueError(
        "Gemma3 vision config requires a CONDITIONAL_GENERATION hf_variant, got"
        " CAUSAL_LM."
    )
  templates = hf_mapping.render_json_templates(
      _FROM_HF_BASE_JSON,
      _FROM_HF_MHA_JSON if n == k else _FROM_HF_GQA_JSON,
      _FROM_HF_VISION_JSON if v_layers > 0 else "",
      **affixes,
      d=d,
      n=n,
      k=k,
      h=h,
      nh=n * h,
      kh=k * h,
      vp=vp,
      vw=vw,
  )
  return hf_mapping.generate_manifest(
      model_type=f"gemma3/{hf_variant.value}",
      template_rules=templates,
      axes={"i": cfg.num_layers, "v": v_layers},
      dropped_keys=[f"{affixes['head']}lm_head.weight"],
  )


def build_to_hf_manifest(
    cfg: model_lib.ModelConfig,
    hf_variant: HfVariant | None = None,
) -> dict[str, Any]:
  """Builds the `Tunix -> HF` manifest for `cfg`."""
  return hf_mapping.invert_manifest(build_from_hf_manifest(cfg, hf_variant))


def export_to_hf_state_dict(
    model: model_lib.Gemma3,
    cfg: model_lib.ModelConfig,
    *,
    hf_variant: HfVariant | None = None,
) -> dict[str, jax.Array]:
  """Exports a Tunix `Gemma3` model into an HF state dict."""
  resolved = model._hf_variant if hf_variant is None else hf_variant  # pylint: disable=protected-access
  manifest = build_to_hf_manifest(cfg, resolved)
  return hf_mapping.to_hf_impl(
      manifest, safetensors_loader.flatten_model_state(model)
  )


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
    mode: str = "auto",
    *,
    hf_variant: HfVariant | None = None,
):
  """Loads a safetensors checkpoint into a Gemma3 model."""
  if hf_variant is None:
    local_dir = safetensors_loader.resolve_checkpoint_dir(file_dir)
    resolved = hf_mapping.resolve_hf_variant_from_keys(
        safetensors_loader.read_checkpoint_keys(local_dir), _HF_AFFIXES
    )
  else:
    if not isinstance(hf_variant, HfVariant):
      raise ValueError(
          f"Pass `hf_variant` (HfVariant) explicitly; got {hf_variant!r}."
      )
    local_dir = file_dir
    resolved = hf_variant
  model = safetensors_loader.load_and_create_model(
      file_dir=local_dir,
      model_class=model_lib.Gemma3,
      config=config,
      mesh=mesh,
      dtype=dtype,
      mode=mode,
      manifest=build_from_hf_manifest(config, resolved),
  )
  model._hf_variant = resolved  # pylint: disable=protected-access
  return model
