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

"""Gemma3 model parameters.

This provides a mapping from the upstream checkpoints[1] to our implementation.

[1] https://github.com/google-deepmind/gemma
"""

from typing import Any

from etils import epath
import flax
from flax import nnx
import jax
from jax import numpy as jnp
from orbax import checkpoint as ocp
from tunix.models import hf_mapping
from tunix.models import safetensors_loader
from tunix.models import safetensors_saver
from tunix.models.gemma3 import model as model_lib
from tunix.models.gemma3 import params_safetensors

import sentencepiece as spm

HfVariant = model_lib.HfVariant
from_hf_impl = params_safetensors.from_hf_impl
to_hf_impl = params_safetensors.to_hf_impl
build_from_hf_manifest = params_safetensors.build_from_hf_manifest
build_to_hf_manifest = params_safetensors.build_to_hf_manifest
export_to_hf_state_dict = params_safetensors.export_to_hf_state_dict
create_model_from_safe_tensors = (
    params_safetensors.create_model_from_safe_tensors
)

# Pretrained
GEMMA3_270M_PT = 'gs://gemma-data/checkpoints/gemma3-270m-pt'
GEMMA3_1B_PT = 'gs://gemma-data/checkpoints/gemma3-1b-pt'
GEMMA3_4B_PT = 'gs://gemma-data/checkpoints/gemma3-4b-pt'
GEMMA3_12B_PT = 'gs://gemma-data/checkpoints/gemma3-12b-pt'
GEMMA3_27B_PT = 'gs://gemma-data/checkpoints/gemma3-27b-pt'
# Instruction Tuned
GEMMA3_270M_IT = 'gs://gemma-data/checkpoints/gemma3-270m-it'
GEMMA3_1B_IT = 'gs://gemma-data/checkpoints/gemma3-1b-it'
GEMMA3_4B_IT = 'gs://gemma-data/checkpoints/gemma3-4b-it'
GEMMA3_12B_IT = 'gs://gemma-data/checkpoints/gemma3-12b-it'
GEMMA3_27B_IT = 'gs://gemma-data/checkpoints/gemma3-27b-it'
# Tokenizer
GEMMA3_TOKENIZER = 'gs://gemma-data/tokenizers/tokenizer_gemma3.model'


def _get_param_dtype(
    path: tuple[Any, ...],
    default_dtype: jnp.dtype,
) -> jnp.dtype:
  """Get dtype for a parameter based on its path.

  All vision components are necessarily kept in fp32.

  Args:
    path: Weight path.
    default_dtype: Default dtype. Everything except vision components will be
      converted to this dtype.

  Returns:
    Dtype of the parameter.
  """
  keys = [k.key for k in path if isinstance(k, jax.tree_util.DictKey)]
  if not keys:
    return default_dtype

  if keys[0] == 'vision_encoder':
    return jnp.float32  # pyrefly: ignore[bad-return]

  if keys[0] == 'embedder':
    # mm_input_projection and mm_soft_embedding_norm
    if len(keys) > 1 and str(keys[1]).startswith('mm_'):
      return jnp.float32  # pyrefly: ignore[bad-return]

  return default_dtype


def create_model_from_checkpoint(
    checkpoint_path: str,
    model_config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype = jnp.bfloat16,  # pyrefly: ignore[bad-function-definition]
) -> model_lib.Gemma3:
  """Load a Gemma3 model from a checkpoint."""
  abs_model = nnx.eval_shape(
      lambda: model_lib.Gemma3(model_config, rngs=nnx.Rngs(0))
  )
  params = ocp.StandardCheckpointer().restore(checkpoint_path)
  params = map_from_upstream_checkpoint(
      params, text_only=model_config.vision_config is None
  )

  if mesh is not None:
    params = jax.tree_util.tree_map_with_path(
        lambda p, x, s: jnp.asarray(
            x, device=s, dtype=_get_param_dtype(p, dtype)
        ),
        params,
        nnx.to_pure_dict(nnx.get_named_sharding(nnx.state(abs_model), mesh)),
    )
  else:
    params = jax.tree_util.tree_map_with_path(
        lambda p, x: jnp.asarray(x, dtype=_get_param_dtype(p, dtype)),
        params,
    )
  nnx.update(abs_model, params)
  return abs_model


PROMPT_TEMPLATE = """\
<start_of_turn>user
{}<end_of_turn>
<start_of_turn>model
"""


def create_tokenizer(
    path: str = GEMMA3_TOKENIZER,
) -> spm.SentencePieceProcessor:
  spm_processor = spm.SentencePieceProcessor()
  model_proto = epath.Path(path).read_bytes()
  spm_processor.LoadFromSerializedProto(model_proto)
  return spm_processor


def map_from_upstream_checkpoint(
    params, model_type: str = 'gemma3', *, text_only: bool = True
):
  """Map from upstream checkpoint to our implementation."""
  # From:
  #
  # ('transformer/embedder', 'input_embedding') (262144, 1152)
  # ('transformer/final_norm', 'scale') (1152,)
  # ('transformer/layer_0/attn/_key_norm', 'scale') (256,)
  # ('transformer/layer_0/attn/_query_norm', 'scale') (256,)
  # ('transformer/layer_0/attn/attn_vec_einsum', 'w') (4, 256, 1152)
  # ('transformer/layer_0/attn/kv_einsum', 'w') (2, 1, 1152, 256)
  # ('transformer/layer_0/attn/q_einsum', 'w') (4, 1152, 256)
  # ('transformer/layer_0/mlp/gating_einsum', 'w') (2, 6912, 1152)
  # ('transformer/layer_0/mlp/linear', 'w') (6912, 1152)
  # ('transformer/layer_0/post_attention_norm', 'scale') (1152,)
  # ('transformer/layer_0/post_ffw_norm', 'scale') (1152,)
  # ('transformer/layer_0/pre_attention_norm', 'scale') (1152,)
  # ('transformer/layer_0/pre_ffw_norm', 'scale') (1152,)
  # ======== Vision ========
  # ('transformer/embedder/mm_input_projection', 'w') (1152, 2560)
  # ('transformer/embedder/mm_soft_embedding_norm', 'scale') (1152,)
  # === SigLipFromPatches_0/siglip_encoder ===
  # ('pos_embedding') (1, 4096, 1152)
  # ('embedding', 'kernel') (14, 14, 3, 1152)
  # ('embedding', 'bias') (1152,)
  # ('Transformer/encoder_norm', 'scale') (1152,)
  # ('Transformer/encoder_norm', 'bias') (1152,)
  # == Transformer/encoderblock_* ==
  # ('LayerNorm_0', 'scale') (1152,)
  # ('LayerNorm_0', 'bias') (1152,)
  # ('LayerNorm_1', 'scale') (1152,)
  # ('LayerNorm_1', 'bias') (1152,)
  # ('MlpBlock_0/Dense_0', 'kernel') (1152, 4304)
  # ('MlpBlock_0/Dense_0', 'bias') (4304,)
  # ('MlpBlock_0/Dense_1', 'kernel') (4304, 1152)
  # ('MlpBlock_0/Dense_1', 'bias') (1152,)
  # ('MultiHeadDotProductAttention_0/key', 'kernel') (1152, 16, 72)
  # ('MultiHeadDotProductAttention_0/key', 'bias') (16, 72)
  # ('MultiHeadDotProductAttention_0/query', 'kernel') (1152, 16, 72)
  # ('MultiHeadDotProductAttention_0/query', 'bias') (16, 72)
  # ('MultiHeadDotProductAttention_0/value', 'kernel') (1152, 16, 72)
  # ('MultiHeadDotProductAttention_0/value', 'bias') (16, 72)
  # ('MultiHeadDotProductAttention_0/out', 'kernel') (16, 72, 1152)
  # ('MultiHeadDotProductAttention_0/out', 'bias') (1152,)
  #
  # To:
  #
  # ('embedder', 'input_embedding') (262144, 1152)
  # ('final_norm', 'scale') (1152,)
  # ('layers', 0, 'attn', '_key_norm', 'scale') (256,)
  # ('layers', 0, 'attn', '_query_norm', 'scale') (256,)
  # ('layers', 0, 'attn', 'attn_vec_einsum', 'w') (4, 256, 1152)
  # ('layers', 0, 'attn', 'kv_einsum', 'w') (2, 1, 1152, 256)
  # ('layers', 0, 'attn', 'q_einsum', 'w') (4, 1152, 256)
  # ('layers', 0, 'mlp', 'down_proj', 'kernel') (6912, 1152)
  # ('layers', 0, 'mlp', 'gate_proj', 'kernel') (1152, 6912)
  # ('layers', 0, 'mlp', 'up_proj', 'kernel') (1152, 6912)
  # ('layers', 0, 'post_attn_norm', 'scale') (1152,)
  # ('layers', 0, 'post_ffw_norm', 'scale') (1152,)
  # ('layers', 0, 'pre_attention_norm', 'scale') (1152,)
  # ('layers', 0, 'pre_ffw_norm', 'scale') (1152,)
  # ======== Vision ========
  # ('embedder', 'mm_input_projection', 'w')
  # ('embedder', 'mm_soft_embedding_norm', 'scale')
  # === vision_encoder/siglip_encoder ===
  # ('pos_embedding') (1, 4096, 1152)
  # ('embedding', 'kernel') (14, 14, 3, 1152)
  # ('embedding', 'bias') (1152,)
  # ('transformer', 'encoder_norm', 'scale') (1152,)
  # ('transformer', 'encoder_norm', 'bias') (1152,)
  # ('transformer', 'blocks', *, 'ln1', 'scale') (1152,)
  # ('transformer', 'blocks', *, 'ln1', 'bias') (1152,)
  # ('transformer', 'blocks', *, 'ln2', 'scale') (1152,)
  # ('transformer', 'blocks', *, 'ln2', 'bias') (1152,)
  # ('transformer', 'blocks', *, 'mlp', 'fc1', 'kernel') (1152, 4304)
  # ('transformer', 'blocks', *, 'mlp', 'fc1', 'bias') (4304,)
  # ('transformer', 'blocks', *, 'mlp', 'fc2', 'kernel') (4304, 1152)
  # ('transformer', 'blocks', *, 'mlp', 'fc2', 'bias') (1152,)
  # ('transformer', 'blocks', *, 'attn', 'key', 'kernel') (1152, 1152)
  # ('transformer', 'blocks', *, 'attn', 'key', 'bias') (1152,)
  # ('transformer', 'blocks', *, 'attn', 'query', 'kernel') (1152, 1152)
  # ('transformer', 'blocks', *, 'attn', 'query', 'bias') (1152,)
  # ('transformer', 'blocks', *, 'attn', 'value', 'kernel') (1152, 1152)
  # ('transformer', 'blocks', *, 'attn', 'value', 'bias') (1152,)
  # ('transformer', 'blocks', *, 'attn', 'out', 'kernel') (1152, 1152)
  # ('transformer', 'blocks', *, 'attn', 'out', 'bias') (1152,)

  new_params = {}
  for key_path, value in flax.traverse_util.flatten_dict(params).items():
    module_path, param_name = key_path
    # Remove the leading 'transformer'/`SigLIPFromPatches`
    module_path = module_path.split('/')[1:]

    # SigLIP encoder
    if module_path[0] == 'siglip_encoder':
      if text_only:
        continue

      target_prefix = ('vision_encoder', 'siglip_encoder')
      # Patch embedding
      if module_path[-1] == 'embedding':
        target_prefix += ('embedding',)
        new_params[(*target_prefix, param_name)] = value
        continue

      # Positional embedding
      if param_name == 'pos_embedding':
        new_params[(*target_prefix, 'pos_embedding')] = value
        continue

      # Transformer
      target_prefix += ('transformer',)
      if module_path[-1] == 'encoder_norm':
        new_params[(*target_prefix, 'encoder_norm', param_name)] = value
        continue

      # All remaining paths are of the form
      # `siglip_encoder/Transformer/encoderblock_{idx}/*`.
      block_idx = int(module_path[2].split('_')[-1])
      layer_name = module_path[3:]
      target_prefix += ('blocks', block_idx)

      if 'LayerNorm' in layer_name[0]:
        idx = int(layer_name[0].split('_')[-1])
        new_params[(*target_prefix, f'ln{idx+1}', param_name)] = value
      elif 'MlpBlock' in layer_name[0]:
        idx = int(layer_name[-1].split('_')[-1])
        new_params[(*target_prefix, 'mlp', f'fc{idx+1}', param_name)] = value
      elif 'MultiHeadDotProductAttention' in layer_name[0]:
        if param_name == 'kernel':
          if layer_name[-1] == 'out':
            value = value.reshape(-1, value.shape[-1])
          else:
            value = value.reshape(value.shape[0], -1)
        elif param_name == 'bias':
          value = value.reshape(-1)
        new_params[
            (*target_prefix, 'attn', f'{layer_name[-1]}_proj', param_name)
        ] = value
      continue

    if module_path[0] == 'embedder':
      # `mm_input_projection` and `mm_soft_embedding_norm`.
      if len(module_path) > 1 and module_path[1].startswith('mm_'):
        if text_only:
          continue
        new_params[tuple(module_path + [param_name])] = value
      else:
        new_params[('embedder', param_name)] = value
      continue

    # Final layer norm
    if module_path[0] == 'final_norm':
      new_params[('final_norm', param_name)] = value
      continue

    # module_path should now look like ('layer_0', 'attn', '_key_norm')
    layer_idx = ('layers', int(module_path[0].removeprefix('layer_')))
    if module_path[1:] == ['mlp', 'gating_einsum']:
      new_params[(*layer_idx, 'mlp', 'gate_proj', 'kernel')] = value[0].T
      new_params[(*layer_idx, 'mlp', 'up_proj', 'kernel')] = value[1].T
    elif module_path[1:] == ['mlp', 'linear']:
      new_params[(*layer_idx, 'mlp', 'down_proj', 'kernel')] = value
    elif module_path[1:] == ['post_attention_norm'] and model_type != 'gemma3':
      new_params[(*layer_idx, 'post_attn_norm', 'scale')] = value
    else:
      new_params[(*layer_idx, *module_path[1:], param_name)] = value
  return flax.traverse_util.unflatten_dict(new_params)


def _extract_gemma3_lora_layers(
    lora_layers: dict[str, list[Any]],
) -> dict[str, list[Any]]:
  """Splits stacked `kv_einsum` / `qkv_einsum` LoRA B tensors per projection."""
  updated_lora_layers: dict[str, list[Any]] = {}
  for k, v in lora_layers.items():
    if k.endswith(('.kv_einsum', '.qkv_einsum')):
      for idx in range(v[1].shape[1]):
        updated_lora_layers[f'{k}.{idx}'] = [v[0], v[1][:, idx]]
    else:
      updated_lora_layers[k] = v
  return updated_lora_layers


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: model_lib.Gemma3,
    rank: int,
    alpha: float,
    *,
    hf_variant: HfVariant | None = None,
):
  """Saves a Gemma3 model with LoRA weights merged in safetensors format.

  Args:
    local_model_path: Path to the base model safetensors checkpoint directory.
    output_dir: Directory where the merged model will be saved.
    lora_model: Gemma3 model instance with LoRA weights.
    rank: LoRA rank used during training.
    alpha: LoRA alpha used during training.
    hf_variant: Key layout of the base checkpoint at `local_model_path`.
      Defaults to the variant recorded on `lora_model` when loaded from
      safetensors; required if `lora_model` was not loaded from safetensors.
  """
  resolved = lora_model._hf_variant if hf_variant is None else hf_variant  # pylint: disable=protected-access
  manifest = params_safetensors.build_to_hf_manifest(
      lora_model.config, resolved
  )
  local_dir = safetensors_loader.resolve_checkpoint_dir(local_model_path)
  key_map, transpose_rules = hf_mapping.lora_mapping_from_manifest(manifest)
  safetensors_saver.save_lora_merged_model_as_safetensors(
      local_model_path=local_dir,
      output_dir=output_dir,
      lora_model=lora_model,
      rank=rank,
      alpha=alpha,
      state_key_transform_fn=key_map.__getitem__,
      custom_layer_extractor_fn=_extract_gemma3_lora_layers,
      transpose_rules=transpose_rules,
  )
