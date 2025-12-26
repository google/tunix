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

import functools
from typing import Any

from etils import epath
import flax
from flax import nnx
import jax
from jax import numpy as jnp
from orbax import checkpoint as ocp
from tunix.models import safetensors_saver
from tunix.models.gemma3 import model as model_lib

# Keep the import below for google internal lint.
import sentencepiece as spm  # isort:skip  # pylint: disable=line-too-long


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


def create_model_from_checkpoint(
    checkpoint_path: str,
    model_config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype = jnp.bfloat16,
) -> model_lib.Gemma3:
  """Load a Gemma3 model from a checkpoint."""
  abs_model = nnx.eval_shape(
      lambda: model_lib.Gemma3(model_config, rngs=nnx.Rngs(0))
  )
  params = ocp.StandardCheckpointer().restore(checkpoint_path)
  params = map_from_upstream_checkpoint(params)
  if mesh is not None:
    params = jax.tree.map(
        lambda x, shd: jnp.asarray(x, device=shd, dtype=dtype),
        params,
        nnx.to_pure_dict(nnx.get_named_sharding(nnx.state(abs_model), mesh)),
    )
  else:
    params = jax.tree.map(functools.partial(jnp.asarray, dtype=dtype), params)
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


def map_from_upstream_checkpoint(params, model_type: str = 'gemma3'):
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
  new_params = {}
  for key_path, value in flax.traverse_util.flatten_dict(params).items():
    module_path, param_name = key_path
    module_path = module_path.split('/')[1:]  # Remove the leading 'transformer'
    if module_path[0] == 'siglip_encoder':
      continue  # We don't support MM input yet.
    if module_path[0] == 'embedder':
      if len(module_path) > 1 and module_path[1].startswith('mm_'):
        continue  # We don't support MM input yet.
    if module_path[0] in ('embedder', 'final_norm'):
      new_params[(module_path[0], param_name)] = value
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


def _extract_gemma3_lora_layers(layer: Any) -> dict[str, tuple[Any, Any]]:
  """Extract LoRA layers from a Gemma3 model.

  Args:
    layer: Gemma3 model layer with possible LoRA weights.

  Returns:
    Dict mapping custom extracted layer paths to (lora_a, lora_b) tuples.
  """
  if hasattr(layer.attn, 'kv_einsum'):
    proj = layer.attn.kv_einsum
    path = safetensors_saver.qwix_path_to_str(proj.qwix_path)
    return {
        path.replace('kv_einsum', 'k_einsum'): (
            proj.w_lora_a,
            proj.w_lora_b[:, 0],
        ),
        path.replace('kv_einsum', 'v_einsum'): (
            proj.w_lora_a,
            proj.w_lora_b[:, 1],
        ),
    }
  return {}


def _gemma3_state_key_to_safetensors_key(lora_name: str) -> str:
  """Transform Gemma3 layer path to safetensors state dict key.

  Args:
    lora_name: Internal layer path (e.g., 'layers.0.attn.q_einsum').

  Returns:
    Safetensors state dict key (e.g., 'model.layers.0.self_attn.q_proj.weight').
  """
  return (
      f'model.{lora_name}.weight'.replace('.attn.', '.self_attn.')
      .replace('q_einsum', 'q_proj')
      .replace('k_einsum', 'k_proj')
      .replace('v_einsum', 'v_proj')
      .replace('attn_vec_einsum', 'o_proj')
  )


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: model_lib.Gemma3,
    rank: int,
    alpha: float,
):
  """Saves a Gemma3 model with LoRA weights merged in safetensors format.

  Args:
    local_model_path: Path to the base model safetensors checkpoint directory.
    output_dir: Directory where the merged model will be saved.
    lora_model: Gemma3 model instance with LoRA weights.
    rank: LoRA rank used during training.
    alpha: LoRA alpha used during training.
  """
  safetensors_saver.save_lora_merged_model_as_safetensors(
      local_model_path=local_model_path,
      output_dir=output_dir,
      lora_model=lora_model,
      rank=rank,
      alpha=alpha,
      state_key_transform_fn=_gemma3_state_key_to_safetensors_key,
      field_patterns=(
          'q_einsum',
          'attn_vec_einsum',
          'gate_proj',
          'up_proj',
          'down_proj',
      ),
      custom_layer_extractor_fn=_extract_gemma3_lora_layers,
  )
