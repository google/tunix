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

import functools
import os

import jax.numpy as jnp
from jax.sharding import Mesh
import safetensors.flax as safe_flax
from tunix.models import safetensors_loader
from tunix.models import safetensors_saver
from tunix.models.qwen3p5 import model as model_lib


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  """Mapping of torch_keys to (nnx_keys, (permute_rule, reshape_rule)).

  Supports both standalone text checkpoints (`model.layers.*`) and multimodal
  wrapper checkpoints (`model.language_model.layers.*` as in
  `Qwen/Qwen3.5-0.8B`).

  Args:
    cfg: Qwen3.5 model configuration.

  Returns:
    A dictionary mapping regex patterns of HuggingFace safetensors keys to
    `(nnx_key, (permute_rule, reshape_rule))` tuples.
  """
  q_mult = 2 if cfg.attn_output_gate else 1
  prefix = r"model(?:\.language_model)?"
  layer_prefix = rf"{prefix}\.layers\.([0-9]+)"
  return {
      rf"{prefix}\.embed_tokens\.weight": (
          "embedder.input_embedding",
          None,
      ),
      # Full attention projection weights (layers 3, 7, 11, 15, 19, 23 in 0.8B)
      rf"{layer_prefix}\.self_attn\.q_proj\.weight": (
          r"layers.\1.attn.q_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim * q_mult)),
      ),
      rf"{layer_prefix}\.self_attn\.k_proj\.weight": (
          r"layers.\1.attn.k_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      rf"{layer_prefix}\.self_attn\.v_proj\.weight": (
          r"layers.\1.attn.v_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      rf"{layer_prefix}\.self_attn\.o_proj\.weight": (
          r"layers.\1.attn.o_proj.w",
          ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
      ),
      rf"{layer_prefix}\.self_attn\.q_norm\.weight": (
          r"layers.\1.attn.q_norm.w",
          None,
      ),
      rf"{layer_prefix}\.self_attn\.k_norm\.weight": (
          r"layers.\1.attn.k_norm.w",
          None,
      ),
      # Gated DeltaNet linear attention weights (18 layers in 0.8B)
      rf"{layer_prefix}\.linear_attn\.in_proj_qkv\.weight": (
          r"layers.\1.linear_attn.in_proj_qkv.kernel",
          ((1, 0), None),
      ),
      rf"{layer_prefix}\.linear_attn\.in_proj_z\.weight": (
          r"layers.\1.linear_attn.in_proj_z.kernel",
          ((1, 0), None),
      ),
      rf"{layer_prefix}\.linear_attn\.in_proj_b\.weight": (
          r"layers.\1.linear_attn.in_proj_b.kernel",
          ((1, 0), None),
      ),
      rf"{layer_prefix}\.linear_attn\.in_proj_a\.weight": (
          r"layers.\1.linear_attn.in_proj_a.kernel",
          ((1, 0), None),
      ),
      rf"{layer_prefix}\.linear_attn\.conv1d\.weight": (
          r"layers.\1.linear_attn.conv1d.kernel",
          ((2, 1, 0), None),
      ),
      rf"{layer_prefix}\.linear_attn\.dt_bias": (
          r"layers.\1.linear_attn.dt_bias",
          None,
      ),
      rf"{layer_prefix}\.linear_attn\.A_log": (
          r"layers.\1.linear_attn.A_log",
          None,
      ),
      rf"{layer_prefix}\.linear_attn\.norm\.weight": (
          r"layers.\1.linear_attn.norm.w",
          None,
      ),
      rf"{layer_prefix}\.linear_attn\.out_proj\.weight": (
          r"layers.\1.linear_attn.out_proj.kernel",
          ((1, 0), None),
      ),
      # Dense MLP
      rf"{layer_prefix}\.mlp\.gate_proj\.weight": (
          r"layers.\1.mlp.gate_proj.kernel",
          ((1, 0), None),
      ),
      rf"{layer_prefix}\.mlp\.up_proj\.weight": (
          r"layers.\1.mlp.up_proj.kernel",
          ((1, 0), None),
      ),
      rf"{layer_prefix}\.mlp\.down_proj\.weight": (
          r"layers.\1.mlp.down_proj.kernel",
          ((1, 0), None),
      ),
      # Norms and readout
      rf"{layer_prefix}\.input_layernorm\.weight": (
          r"layers.\1.input_layernorm.w",
          None,
      ),
      rf"{layer_prefix}\.post_attention_layernorm\.weight": (
          r"layers.\1.post_attention_layernorm.w",
          None,
      ),
      rf"{prefix}\.norm\.weight": ("final_norm.w", None),
      r"(?:model\.language_model\.)?lm_head\.weight": (
          "lm_head.w",
          ((1, 0), None),
      ),
  }


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: Mesh | None = None,
    dtype: jnp.dtype | None = None,
    mode: str = "auto",
) -> model_lib.Qwen3p5:
  """Load tensors from the safetensors file and create a Qwen3.5 model."""
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Qwen3p5,
      config=config,
      key_mapping=_get_key_and_transform_mapping,
      mesh=mesh,
      preprocess_fn=None,
      dtype=dtype or jnp.dtype(config.param_dtype),
      mode=mode,
  )


def _has_language_model_prefix(local_model_path: str) -> bool:
  """Returns whether checkpoint uses the `model.language_model.` key prefix."""
  safetensors_path = os.path.join(local_model_path, "model.safetensors")
  with safe_flax.safe_open(safetensors_path, framework="numpy") as sf:
    return any(k.startswith("model.language_model.") for k in sf.keys())


def _qwen3p5_state_key_to_safetensors_key(
    lora_name: str,
    *,
    use_language_model_prefix: bool = True,
) -> str:
  """Transform Qwen3.5 layer path to safetensors state dict key."""
  prefix = "model.language_model" if use_language_model_prefix else "model"
  return f"{prefix}.{lora_name}.weight".replace(".attn.", ".self_attn.")


_QWEN3P5_HUGGINGFACE_TRANSPOSE_RULES: dict[str, tuple[int, ...]] = {
    "q_proj": (1, 0),
    "k_proj": (1, 0),
    "v_proj": (1, 0),
    "o_proj": (1, 0),
    "in_proj_qkv": (1, 0),
    "in_proj_z": (1, 0),
    "in_proj_b": (1, 0),
    "in_proj_a": (1, 0),
    "out_proj": (1, 0),
    "up_proj": (1, 0),
    "down_proj": (1, 0),
    "gate_proj": (1, 0),
}


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: model_lib.Qwen3p5,
    rank: int,
    alpha: float,
):
  """Saves a Qwen3.5 model with merged LoRA weights in safetensors format."""
  use_language_model_prefix = _has_language_model_prefix(local_model_path)
  safetensors_saver.save_lora_merged_model_as_safetensors(
      local_model_path=local_model_path,
      output_dir=output_dir,
      lora_model=lora_model,
      rank=rank,
      alpha=alpha,
      state_key_transform_fn=functools.partial(
          _qwen3p5_state_key_to_safetensors_key,
          use_language_model_prefix=use_language_model_prefix,
      ),
      transpose_rules=_QWEN3P5_HUGGINGFACE_TRANSPOSE_RULES,
  )
