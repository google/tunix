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
from tunix.models.qwen3vl import model as model_lib


def _stack_experts(params: dict[str, jax.Array]):
  """Stack experts in the loaded pytorch params."""
  key_fn = lambda x: int(re.match(r'(.*?)experts\.([0-9]+)\..*', x).group(2))  # pytype: disable=attribute-error
  updated_dict = dict(params).copy()
  for kw in ['gate', 'up', 'down']:
    pattern = r'(.*?)experts\.(.*?)\.{}_proj\.(.*)'.format(kw)
    keys = [k for k in params.keys() if re.match(pattern, k)]
    prefix_groups = set([re.match(pattern, k).group(1) for k in keys])  # pytype: disable=attribute-error
    for prefix in prefix_groups:
      keys_to_merge = list(
          sorted([k for k in keys if k.startswith(prefix)], key=key_fn)
      )
      for k in keys_to_merge:
        del updated_dict[k]
      with jax.default_device(jax.devices('cpu')[0]):
        updated_dict[f'{prefix}{kw}_proj'] = jnp.stack(
            [params[k] for k in keys_to_merge], 0
        )
  return updated_dict


def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
  # Mapping of torch_keys -> (nnx_keys, (permute_rule, reshape_rule)).
  v_cfg = cfg.vision_config
  pixel_volume = (
      v_cfg.temporal_patch_size * v_cfg.patch_size**2 * v_cfg.in_channels
  )
  return {
      # vision: patch embed
      r'model\.visual\.patch_embed.proj.weight': (
          r'visual.patch_embed.proj.kernel',
          ((1, 2, 3, 4, 0), (pixel_volume, cfg.vision_config.hidden_size)),
      ),
      r'model\.visual\.patch_embed.proj.bias': (
          r'visual.patch_embed.proj.bias',
          None,
      ),
      # vision: pos embed
      r'model\.visual\.pos_embed.weight': (
          r'visual.pos_embed.embedding',
          None,  # TODO: (2304, 1024) - transpose?
      ),
      # vision: attention
      r'model\.visual\.blocks\.([0-9]+)\.attn.qkv.weight': (
          r'visual.blocks.\1.attn.qkv_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.visual\.blocks\.([0-9]+)\.attn.qkv.bias': (
          r'visual.blocks.\1.attn.qkv_proj.bias',
          None,
      ),
      r'model\.visual\.blocks\.([0-9]+)\.attn.proj.weight': (
          r'visual.blocks.\1.attn.out_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.visual\.blocks\.([0-9]+)\.attn.proj.bias': (
          r'visual.blocks.\1.attn.out_proj.bias',
          None,
      ),
      # vision: mlp
      r'model\.visual\.blocks\.([0-9]+)\.mlp.linear_fc1.weight': (
          r'visual.blocks.\1.mlp.linear1.kernel',
          ((1, 0), None),
      ),
      r'model\.visual\.blocks\.([0-9]+)\.mlp.linear_fc1.bias': (
          r'visual.blocks.\1.mlp.linear1.bias',
          None,
      ),
      r'model\.visual\.blocks\.([0-9]+)\.mlp.linear_fc2.weight': (
          r'visual.blocks.\1.mlp.linear2.kernel',
          ((1, 0), None),
      ),
      r'model\.visual\.blocks\.([0-9]+)\.mlp.linear_fc2.bias': (
          r'visual.blocks.\1.mlp.linear2.bias',
          None,
      ),
      r'model\.visual\.blocks\.([0-9]+)\.mlp.linear_fc3.weight': (
          r'visual.blocks.\1.mlp.linear3.kernel',
          ((1, 0), None),
      ),
      r'model\.visual\.blocks\.([0-9]+)\.mlp.linear_fc3.bias': (
          r'visual.blocks.\1.mlp.linear3.bias',
          None,
      ),
      # vision: norm
      r'model\.visual\.blocks\.([0-9]+)\.norm1.weight': (
          r'visual.blocks.\1.norm1.scale',
          None,
      ),
      r'model\.visual\.blocks\.([0-9]+)\.norm1.bias': (
          r'visual.blocks.\1.norm1.bias',
          None,
      ),
      r'model\.visual\.blocks\.([0-9]+)\.norm2.weight': (
          r'visual.blocks.\1.norm2.scale',
          None,
      ),
      r'model\.visual\.blocks\.([0-9]+)\.norm2.bias': (
          r'visual.blocks.\1.norm2.bias',
          None,
      ),
      # vision: deepstack mergers
      r'model\.visual\.deepstack_merger_list\.([0-9]+)\.linear_fc1.weight': (
          r'visual.deepstack_mergers.\1.linear_fc1.kernel',
          ((1, 0), None),
      ),
      r'model\.visual\.deepstack_merger_list\.([0-9]+)\.linear_fc1.bias': (
          r'visual.deepstack_mergers.\1.linear_fc1.bias',
          None,
      ),
      r'model\.visual\.deepstack_merger_list\.([0-9]+)\.linear_fc2.weight': (
          r'visual.deepstack_mergers.\1.linear_fc2.kernel',
          ((1, 0), None),
      ),
      r'model\.visual\.deepstack_merger_list\.([0-9]+)\.linear_fc2.bias': (
          r'visual.deepstack_mergers.\1.linear_fc2.bias',
          None,
      ),
      r'model\.visual\.deepstack_merger_list\.([0-9]+)\.norm.weight': (
          r'visual.deepstack_mergers.\1.norm.scale',
          None,
      ),
      r'model\.visual\.deepstack_merger_list\.([0-9]+)\.norm.bias': (
          r'visual.deepstack_mergers.\1.norm.bias',
          None,
      ),
      # vision: mergers
      r'model\.visual\.merger\.linear_fc1.weight': (
          r'visual.merger.linear_fc1.kernel',
          ((1, 0), None),
      ),
      r'model\.visual\.merger\.linear_fc1.bias': (
          r'visual.merger.linear_fc1.bias',
          None,
      ),
      r'model\.visual\.merger\.linear_fc2.weight': (
          r'visual.merger.linear_fc2.kernel',
          ((1, 0), None),
      ),
      r'model\.visual\.merger\.linear_fc2.bias': (
          r'visual.merger.linear_fc2.bias',
          None,
      ),
      r'model\.visual\.merger\.norm.weight': (
          r'visual.merger.norm.scale',
          None,
      ),
      r'model\.visual\.merger\.norm.bias': (
          r'visual.merger.norm.bias',
          None,
      ),
      # text encoder
      r'model\.(?:language_model\.)?embed_tokens\.weight': (
          'embedder.input_embedding',
          None,
      ),
      # attention projection weights
      r'model\.(?:language_model\.)?layers\.([0-9]+)\.self_attn\.q_proj\.weight': (
          r'layers.\1.attn.q_proj.w',
          ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim)),
      ),
      r'model\.(?:language_model\.)?layers\.([0-9]+)\.self_attn\.k_proj\.weight': (
          r'layers.\1.attn.k_proj.w',
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r'model\.(?:language_model\.)?layers\.([0-9]+)\.self_attn\.v_proj\.weight': (
          r'layers.\1.attn.v_proj.w',
          ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
      ),
      r'model\.(?:language_model\.)?layers\.([0-9]+)\.self_attn\.o_proj\.weight': (
          r'layers.\1.attn.o_proj.w',
          ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
      ),
      # mlp
      r'model\.(?:language_model\.)?layers\.([0-9]+)\.mlp\.gate_proj\.weight': (
          r'layers.\1.mlp.gate_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.(?:language_model\.)?layers\.([0-9]+)\.mlp\.up_proj\.weight': (
          r'layers.\1.mlp.up_proj.kernel',
          ((1, 0), None),
      ),
      r'model\.(?:language_model\.)?layers\.([0-9]+)\.mlp\.down_proj\.weight': (
          r'layers.\1.mlp.down_proj.kernel',
          ((1, 0), None),
      ),
      # MoE router/gate
      r'model\.(?:language_model\.)?layers\.([0-9]+)\.mlp\.gate\.weight': (
          r'layers.\1.mlp.router.kernel',
          ((1, 0), None),
      ),
      # MoE experts.
      r'model\.(?:language_model\.)?layers\.([0-9]+)\.mlp\.experts\.([0-9]+)\.(gate|up|down)_proj\.weight': (
          r'layers.\1.mlp.experts.\2.\3_proj.kernel',
          ((1, 0), None),
      ),
      # norms
      r'model\.(?:language_model\.)?norm\.weight': ('final_norm.w', None),
      r'model\.(?:language_model\.)?layers\.([0-9]+)\.self_attn\.q_norm\.weight': (
          r'layers.\1.attn.q_norm.w',
          None,
      ),
      r'model\.(?:language_model\.)?layers\.([0-9]+)\.self_attn\.k_norm\.weight': (
          r'layers.\1.attn.k_norm.w',
          None,
      ),
      # layer norms (pre/post attention)
      r'model\.(?:language_model\.)?layers\.([0-9]+)\.input_layernorm\.weight': (
          r'layers.\1.input_layernorm.w',
          None,
      ),
      r'model\.(?:language_model\.)?layers\.([0-9]+)\.post_attention_layernorm\.weight': (
          r'layers.\1.post_attention_layernorm.w',
          None,
      ),
      r'lm_head\.weight': ('lm_head.w', ((1, 0), None)),
  }


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
) -> model_lib.Qwen3VL:
  """Load tensors from the safetensors file and create a Qwen3 model."""
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Qwen3VL,
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
  return f'model.{lora_name}.weight'.replace('.attn.', '.self_attn.')


_QWEN3_HUGGINGFACE_TRANSPOSE_RULES = {
    'q_proj': (1, 0),
    'k_proj': (1, 0),
    'v_proj': (1, 0),
    'o_proj': (1, 0),
    'up_proj': (1, 0),
    'down_proj': (1, 0),
    'gate_proj': (1, 0),
    'gate': (1, 0),
}


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: model_lib.Qwen3VL,
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
      transpose_rules=_QWEN3_HUGGINGFACE_TRANSPOSE_RULES,
  )
