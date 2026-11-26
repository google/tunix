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

"""Utils for loading and saving Qwen3.5 model weights."""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from tunix.models import hf_mapping
from tunix.models import safetensors_loader
from tunix.models import safetensors_saver
from tunix.models.qwen3p5 import model as model_lib

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
    "target_pattern": ["final_norm.w"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}layers.{i}.input_layernorm.weight"],
    "target_pattern": ["layers.{i}.input_layernorm.w"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}layers.{i}.post_attention_layernorm.weight"],
    "target_pattern": ["layers.{i}.post_attention_layernorm.w"],
    "ops": []
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
  },
  {
    "source_pattern": ["${lm}layers.{f}.self_attn.q_norm.weight"],
    "target_pattern": ["layers.{f}.attn.q_norm.w"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}layers.{f}.self_attn.k_norm.weight"],
    "target_pattern": ["layers.{f}.attn.k_norm.w"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}layers.{f}.self_attn.q_proj.weight"],
    "target_pattern": ["layers.{f}.attn.q_proj.w"],
    "ops": [
      {"transpose": {"axes": [1, 0]}},
      {"reshape": {"src_shape": [$d, $nq], "dst_shape": [$d, $n, $q_dim]}}
    ]
  },
  {
    "source_pattern": ["${lm}layers.{f}.self_attn.k_proj.weight"],
    "target_pattern": ["layers.{f}.attn.k_proj.w"],
    "ops": [
      {"transpose": {"axes": [1, 0]}},
      {"reshape": {"src_shape": [$d, $kh], "dst_shape": [$d, $k, $h]}}
    ]
  },
  {
    "source_pattern": ["${lm}layers.{f}.self_attn.v_proj.weight"],
    "target_pattern": ["layers.{f}.attn.v_proj.w"],
    "ops": [
      {"transpose": {"axes": [1, 0]}},
      {"reshape": {"src_shape": [$d, $kh], "dst_shape": [$d, $k, $h]}}
    ]
  },
  {
    "source_pattern": ["${lm}layers.{f}.self_attn.o_proj.weight"],
    "target_pattern": ["layers.{f}.attn.o_proj.w"],
    "ops": [
      {"transpose": {"axes": [1, 0]}},
      {"reshape": {"src_shape": [$nh, $d], "dst_shape": [$n, $h, $d]}}
    ]
  },
  {
    "source_pattern": ["${lm}layers.{l}.linear_attn.in_proj_qkv.weight"],
    "target_pattern": ["layers.{l}.linear_attn.in_proj_qkv.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["${lm}layers.{l}.linear_attn.in_proj_z.weight"],
    "target_pattern": ["layers.{l}.linear_attn.in_proj_z.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["${lm}layers.{l}.linear_attn.in_proj_b.weight"],
    "target_pattern": ["layers.{l}.linear_attn.in_proj_b.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["${lm}layers.{l}.linear_attn.in_proj_a.weight"],
    "target_pattern": ["layers.{l}.linear_attn.in_proj_a.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["${lm}layers.{l}.linear_attn.out_proj.weight"],
    "target_pattern": ["layers.{l}.linear_attn.out_proj.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["${lm}layers.{l}.linear_attn.conv1d.weight"],
    "target_pattern": ["layers.{l}.linear_attn.conv1d.kernel"],
    "ops": [{"transpose": {"axes": [2, 1, 0]}}]
  },
  {
    "source_pattern": ["${lm}layers.{l}.linear_attn.dt_bias"],
    "target_pattern": ["layers.{l}.linear_attn.dt_bias"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}layers.{l}.linear_attn.A_log"],
    "target_pattern": ["layers.{l}.linear_attn.A_log"],
    "ops": []
  },
  {
    "source_pattern": ["${lm}layers.{l}.linear_attn.norm.weight"],
    "target_pattern": ["layers.{l}.linear_attn.norm.w"],
    "ops": []
  }
]"""

_FROM_HF_LM_HEAD_JSON = """[
  {
    "source_pattern": ["lm_head.weight"],
    "target_pattern": ["lm_head.w"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  }
]"""


# Template placeholder values per variant: `lm` prefixes each text decoder key.
_HF_AFFIXES: Mapping[HfVariant, Mapping[str, str]] = {
    HfVariant.CAUSAL_LM: {"lm": "model."},
    HfVariant.CONDITIONAL_GENERATION: {"lm": "model.language_model."},
}


def build_from_hf_manifest(
    cfg: model_lib.ModelConfig,
    hf_variant: HfVariant | None = None,
) -> dict[str, Any]:
  """Builds the `HF -> Tunix` manifest for `cfg`."""
  if cfg.num_experts is not None:
    raise NotImplementedError(
        "MoE layers (num_experts != None) are not yet supported in native"
        " Qwen3p5; use ModelSource.MAXTEXT."
    )
  if not isinstance(hf_variant, HfVariant):
    raise ValueError(
        "Pass `hf_variant` (HfVariant) explicitly; models not loaded via"
        f" `create_model_from_safe_tensors` carry none. Got {hf_variant!r}."
    )
  variant = hf_variant
  d, n, k, h = cfg.embed_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
  q_dim = h * (2 if cfg.attn_output_gate else 1)
  assert cfg.layer_types is not None
  full = [
      i
      for i, lt in enumerate(cfg.layer_types)
      if lt == model_lib.LayerType.FULL_ATTENTION
  ]
  linear = sorted(set(range(cfg.num_layers)) - set(full))
  templates = hf_mapping.render_json_templates(
      _FROM_HF_BASE_JSON,
      "" if cfg.use_tied_embedding else _FROM_HF_LM_HEAD_JSON,
      **_HF_AFFIXES[variant],
      d=d,
      n=n,
      k=k,
      h=h,
      q_dim=q_dim,
      nq=n * q_dim,
      nh=n * h,
      kh=k * h,
  )
  return hf_mapping.generate_manifest(
      model_type=f"qwen3p5/{variant.value}",
      template_rules=templates,
      axes={"i": cfg.num_layers, "f": full, "l": linear},
      dropped_keys=["lm_head.weight"] if cfg.use_tied_embedding else [],
  )


def build_to_hf_manifest(
    cfg: model_lib.ModelConfig,
    hf_variant: HfVariant | None = None,
) -> dict[str, Any]:
  """Builds the `Tunix -> HF` manifest for `cfg`."""
  return hf_mapping.invert_manifest(build_from_hf_manifest(cfg, hf_variant))


def export_to_hf_state_dict(
    model: model_lib.Qwen3p5,
    cfg: model_lib.ModelConfig,
    *,
    hf_variant: HfVariant | None = None,
) -> dict[str, jax.Array]:
  """Exports a Tunix `Qwen3p5` model into an HF state dict.

  Args:
    model: Qwen3.5 model to export.
    cfg: Config of `model`.
    hf_variant: Target HF key layout. Defaults to the layout `model` was loaded
      from; required for fresh-init or Orbax-restored models.

  Returns:
    The HF state dict keyed by HF tensor names.
  """
  resolved = model._hf_variant if hf_variant is None else hf_variant  # pylint: disable=protected-access
  manifest = build_to_hf_manifest(cfg, resolved)
  return hf_mapping.to_hf_impl(
      manifest, safetensors_loader.flatten_model_state(model)
  )


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: Mesh | None = None,
    dtype: jnp.dtype | None = None,
    mode: str = "auto",
    *,
    hf_variant: HfVariant | None = None,
) -> model_lib.Qwen3p5:
  """Loads a safetensors checkpoint into a Qwen3.5 model.

  Args:
    file_dir: Local or `gs://` checkpoint directory.
    config: Model config.
    mesh: Optional mesh for sharded loading.
    dtype: Parameter dtype; defaults to `config.param_dtype`.
    mode: Loader mode forwarded to `load_and_create_model`.
    hf_variant: Checkpoint key layout; inferred from checkpoint keys if `None`.

  Returns:
    The loaded model, stamped with its resolved `HfVariant`.
  """
  if hf_variant is None:
    local_dir = safetensors_loader.resolve_checkpoint_dir(file_dir)
    resolved = hf_mapping.resolve_hf_variant_from_keys(
        safetensors_loader.read_checkpoint_keys(local_dir), _HF_AFFIXES
    )
  elif isinstance(hf_variant, HfVariant):
    local_dir, resolved = file_dir, hf_variant
  else:
    raise ValueError(
        f"Expected `hf_variant` of type HfVariant, got {hf_variant!r}."
    )
  resolved_dtype = (
      jnp.dtype(config.param_dtype) if dtype is None else dtype
  )
  model = safetensors_loader.load_and_create_model(
      file_dir=local_dir,
      model_class=model_lib.Qwen3p5,
      config=config,
      mesh=mesh,
      dtype=resolved_dtype,
      mode=mode,
      manifest=build_from_hf_manifest(config, resolved),
  )
  model._hf_variant = resolved  # pylint: disable=protected-access
  return model


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: model_lib.Qwen3p5,
    rank: int,
    alpha: float,
    *,
    hf_variant: HfVariant | None = None,
):
  """Saves a Qwen3.5 model with merged LoRA weights in safetensors format.

  Args:
    local_model_path: Path to the base model safetensors checkpoint directory.
    output_dir: Directory where the merged model will be saved.
    lora_model: Qwen3.5 model instance with LoRA weights.
    rank: LoRA rank used during training.
    alpha: LoRA alpha used during training.
    hf_variant: Key layout of the base checkpoint at `local_model_path`.
      Defaults to the layout `lora_model` was loaded from.
  """
  resolved = lora_model._hf_variant if hf_variant is None else hf_variant  # pylint: disable=protected-access
  manifest = build_to_hf_manifest(lora_model.config, resolved)
  local_dir = safetensors_loader.resolve_checkpoint_dir(local_model_path)
  key_map, transpose_rules = hf_mapping.lora_mapping_from_manifest(manifest)
  safetensors_saver.save_lora_merged_model_as_safetensors(
      local_model_path=local_dir,
      output_dir=output_dir,
      lora_model=lora_model,
      rank=rank,
      alpha=alpha,
      state_key_transform_fn=key_map.__getitem__,
      transpose_rules=transpose_rules,
  )
