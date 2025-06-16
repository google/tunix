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

from etils import epath
import flax
from flax import nnx
import jax
from jax import numpy as jnp
from orbax import checkpoint as ocp
from orbax.checkpoint.google import pathways_type_handlers  # GOOGLE-INTERNAL
from tunix.models.gemma3 import model as model_lib

from google3.learning.deepmind.jax.ocean import remote_python as rp  # pylint: disable=line-too-long  # GOOGLE-INTERNAL
from google3.learning.pathways.jax import pathways as pw  # GOOGLE-INTERNAL
import sentencepiece as spm

# Pretrained
GEMMA3_1B_PT = 'gs://gemma-data/checkpoints/gemma3-1b-pt'
GEMMA3_4B_PT = 'gs://gemma-data/checkpoints/gemma3-4b-pt'
GEMMA3_12B_PT = 'gs://gemma-data/checkpoints/gemma3-12b-pt'
GEMMA3_27B_PT = 'gs://gemma-data/checkpoints/gemma3-27b-pt'
# Instruction Tuned
GEMMA3_1B_IT = 'gs://gemma-data/checkpoints/gemma3-1b-it'
GEMMA3_4B_IT = 'gs://gemma-data/checkpoints/gemma3-4b-it'
GEMMA3_12B_IT = 'gs://gemma-data/checkpoints/gemma3-12b-it'
GEMMA3_27B_IT = 'gs://gemma-data/checkpoints/gemma3-27b-it'
# Tokenizer
GEMMA3_TOKENIZER = 'gs://gemma-data/tokenizers/tokenizer_gemma3.model'


def create_model_from_checkpoint(
    checkpoint_path: str,
    model_config: model_lib.Gemma3Config,
    mesh: jax.sharding.Mesh | None = None,
) -> model_lib.Gemma3:
  """Load a Gemma3 model from a checkpoint."""
  abs_model = nnx.eval_shape(
      lambda: model_lib.Gemma3(model_config, rngs=nnx.Rngs(0))
  )
  _, abs_state = nnx.split(abs_model)
  if mesh is not None:
    abs_state = jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.float32, sharding=s),
        abs_state,
        nnx.get_named_sharding(abs_state, mesh),
    )

  abs_state_dict = _map_to_downstream_checkpoint(abs_state.to_pure_dict())

  # BEGIN GOOGLE-INTERNAL
  if pw.is_pathways_backend() and rp.available():
    pathways_type_handlers.register_pathways_handlers()
  # END GOOGLE-INTERNAL
  restored_state = jax.tree_util.tree_map(
      lambda x: ocp.type_handlers.ArrayRestoreArgs(
          sharding=x.sharding,
          global_shape=x.shape,
          dtype=x.dtype,
      ),
      abs_state_dict,
  )
  params = ocp.PyTreeCheckpointer().restore(
      checkpoint_path,
      item=abs_state_dict,
      partial_restore=True,
      restore_args=restored_state,
  )
  params = _map_from_upstream_checkpoint(params)
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


def _map_from_upstream_checkpoint(params):
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
    else:
      new_params[(*layer_idx, *module_path[1:], param_name)] = value
  return flax.traverse_util.unflatten_dict(new_params)


def _map_to_downstream_checkpoint(params):
  """Map to downstream checkpoint from our implementation."""
  # From:
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
  # To:
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
  new_params = {}
  gate_proj = {}
  for key_path, value in flax.traverse_util.flatten_dict(params).items():
    if key_path[0] in ('embedder', 'final_norm'):
      new_params[(f'transformer/{key_path[0]}', key_path[1])] = value
      continue
    layer_idx = str(key_path[1])
    if [key_path[2], key_path[3]] == ['mlp', 'gate_proj']:
      gate_proj[layer_idx] = value
    elif [key_path[2], key_path[3]] == ['mlp', 'up_proj']:
      continue
    elif [key_path[2], key_path[3]] == ['mlp', 'down_proj']:
      new_params[(f'transformer/layer_{layer_idx}/mlp/linear', 'w')] = value
    elif len(key_path) == 4:
      new_params[(
          f'transformer/layer_{layer_idx}/{key_path[2]}',
          key_path[3],
      )] = value
    else:
      new_params[(
          f'transformer/layer_{layer_idx}/{key_path[2]}/{key_path[3]}',
          key_path[4],
      )] = value
  for layer_idx, gate_proj_value in gate_proj.items():
    new_params[(f'transformer/layer_{layer_idx}/mlp/gating_einsum', 'w')] = (
        jax.ShapeDtypeStruct(
            [2, *gate_proj_value.shape[::-1]],
            jnp.float32,
            sharding=gate_proj_value.sharding,
        )
    )
  return flax.traverse_util.unflatten_dict(new_params)
