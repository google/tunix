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

"""Utils for loading and converting Qwen3 PT weights."""

import re

import jax
import jax.numpy as jnp
from tunix.models import safetensors_loader
from tunix.models import safetensors_saver
from tunix.models.qwen3 import model as model_lib


def _stack_experts(params: dict[str, jax.Array]):
  """Stack experts in the loaded pytorch params."""
  key_fn = lambda x: int(re.match(r"(.*?)experts\.([0-9]+)\..*", x).group(2))  # pytype: disable=attribute-error
  updated_dict = dict(params).copy()
  for kw in ["gate", "up", "down"]:
    pattern = r"(.*?)experts\.(.*?)\.{}_proj\.(.*)".format(kw)
    keys = [k for k in params.keys() if re.match(pattern, k)]
    prefix_groups = set([re.match(pattern, k).group(1) for k in keys])  # pytype: disable=attribute-error
    for prefix in prefix_groups:
      keys_to_merge = list(
          sorted([k for k in keys if k.startswith(prefix)], key=key_fn)
      )
      for k in keys_to_merge:
        del updated_dict[k]
      with jax.default_device(jax.devices("cpu")[0]):
        updated_dict[f"{prefix}{kw}_proj"] = jnp.stack(
            [params[k] for k in keys_to_merge], 0
        )
  return updated_dict


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  # Mapping of torch_keys -> (nnx_keys, (permute_rule, reshape_rule)).
  return {
      r"model\.embed_tokens\.weight": ("embedder.input_embedding", None),
      # attention projection weights
      r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
          r"layers.\1.attn.q_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
          r"layers.\1.attn.k_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
          r"layers.\1.attn.v_proj.w",
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
          r"layers.\1.attn.o_proj.w",
          ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
      ),
      # mlp
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
      # MoE router/gate
      r"model\.layers\.([0-9]+)\.mlp\.gate\.weight": (
          r"layers.\1.mlp.router.kernel",
          ((1, 0), None),
      ),
      # MoE experts.
      r"model\.layers\.([0-9]+)\.mlp\.experts\.([0-9]+)\.(gate|up|down)_proj\.weight": (
          r"layers.\1.mlp.experts.\2.\3_proj.kernel",
          ((1, 0), None),
      ),
      # norms
      r"model\.norm\.weight": ("final_norm.w", None),
      r"model\.layers\.([0-9]+)\.self_attn\.q_norm\.weight": (
          r"layers.\1.attn.q_norm.w",
          None,
      ),
      r"model\.layers\.([0-9]+)\.self_attn\.k_norm\.weight": (
          r"layers.\1.attn.k_norm.w",
          None,
      ),
      # layer norms (pre/post attention)
      r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (
          r"layers.\1.input_layernorm.w",
          None,
      ),
      r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
          r"layers.\1.post_attention_layernorm.w",
          None,
      ),
      r"lm_head\.weight": ("lm_head.w", ((1, 0), None)),
  }


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
) -> model_lib.Qwen3:
  """Load tensors from the safetensors file and create a Qwen3 model."""
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Qwen3,
      config=config,
      key_mapping=_get_key_and_transform_mapping,
      mesh=mesh,
      preprocess_fn=_stack_experts,
      dtype=dtype,
  )


def _qwen3_state_key_to_safetensors_key(lora_name: str) -> str:
  """Transform Qwen3 layer path to safetensors state dict key.

  Args:
    lora_name: Internal layer path (e.g., 'layers.0.attn.q_proj').

  Returns:
    Safetensors state dict key (e.g., 'model.layers.0.self_attn.q_proj.weight').
  """
  return f"model.{lora_name}.weight".replace(".attn.", ".self_attn.")


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: model_lib.Qwen3,
    rank: int,
    alpha: float,
):
  """Saves a Qwen3 model with LoRA weights merged in safetensors format.

  Args:
    local_model_path: Path to the base model safetensors checkpoint directory.
    output_dir: Directory where the merged model will be saved.
    lora_model: Qwen3 model instance with LoRA weights.
    rank: LoRA rank used during training.
    alpha: LoRA alpha used during training.
  """
  safetensors_saver.save_lora_merged_model_as_safetensors(
      local_model_path=local_model_path,
      output_dir=output_dir,
      lora_model=lora_model,
      rank=rank,
      alpha=alpha,
      state_key_transform_fn=_qwen3_state_key_to_safetensors_key,
      field_patterns=(
          "q_proj",
          "k_proj",
          "v_proj",
          "o_proj",
          "gate_proj",
          "up_proj",
          "down_proj",
      ),
  )
