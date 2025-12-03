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
"""Utilities for creating and managing models in Tunix CLI."""

import enum
import gc
import importlib
import os
from typing import Any, Tuple
from absl import logging
import flax
from flax import nnx
import jax
import jax.numpy as jnp
from orbax import checkpoint as ocp
import qwix
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models import naming
from tunix.rl import reshard


_BASE_MODULE_PATH = 'tunix.models'  # pylint: disable=invalid-name


class ModelModule(enum.Enum):
  """Specifies the type of model module to import."""

  MODEL = 'model'
  PARAMS = 'params'


def get_model_module(model_name: str, module_type: ModelModule) -> Any:
  """Dynamically imports a model module (e.g., 'model' or 'params')."""
  model_config_category = naming.get_model_config_category(model_name)
  module_path = (
      f'{_BASE_MODULE_PATH}.{model_config_category}.{module_type.value}'
  )
  try:
    logging.info('Attempting to import: %s', module_path)
    model_lib_module = importlib.import_module(module_path)
    return model_lib_module
  except ImportError as exc:
    raise ImportError(
        'Could not import module for model config category: '
        f'{model_config_category} at path: {module_path}. Please check '
        'BASE_MODULE_PATH and ensure the module exists and is a dependency.'
    ) from exc


def create_model_dynamically(
    model_name: str, file_dir: str, model_config: Any, mesh: jax.sharding.Mesh
) -> Any:
  """Dynamically imports the correct module and calls `create_model_from_safe_tensors` based on the model_name.

  Args:
      model_name: The name of the model (e.g., "qwen2.5-0.5b", "llama3.2-3b").
      file_dir: Directory containing the safe tensors.
      model_config: Model configuration object.
      mesh: Mesh object for device layout.

  Returns:
      The result of the create_model_from_safe_tensors call.

  Raises:
      ValueError: If the model_name is invalid.
      ImportError: If the required model module cannot be found.
      AttributeError: If create_model_from_safe_tensors is not in the module.
  """
  params_module = get_model_module(model_name, ModelModule.PARAMS)

  try:
    create_fn = getattr(params_module, 'create_model_from_safe_tensors')
  except AttributeError as exc:
    raise AttributeError(
        "'create_model_from_safe_tensors' not found in module "
        f'{params_module.__name__} for model {model_name}'
    ) from exc

  logging.info(
      'Calling %s.create_model_from_safe_tensors', params_module.__name__
  )
  return create_fn(file_dir=file_dir, config=model_config, mesh=mesh)


def obtain_model_params(model_name: str) -> Any:
  """Dynamically calls a configuration function based on the model_string.

  The routing to the correct module/class instance is based on the longest
  matching prefix of model_name found in CONFIG_MAP.
  Hyphens and dots in the model_name are converted to underscores
  to form the function name.

  Args:
      model_name: The string indicating which model config function to call
        (e.g., "gemma-2b", "llama3.1-8b", "qwen2.5-0.5b").

  Returns:
      The result from calling the dynamically determined function.

  Raises:
      ValueError: If the model_string doesn't match any known prefix.
      AttributeError: If the derived function name does not exist in the target
      object.
      TypeError: If the attribute found on the target object is not callable.
  """
  config_id = naming.get_model_config_id(model_name)
  model_lib_module = get_model_module(model_name, ModelModule.MODEL)
  target_obj = model_lib_module.ModelConfig

  if not hasattr(target_obj, config_id):
    raise AttributeError(
        f"Error: Function '{config_id}' not found on the target object "
        f"for model '{model_name}'. Target object type: {type(target_obj)}"
    )

  method_to_call = getattr(target_obj, config_id)

  if not callable(method_to_call):
    raise TypeError(
        f"Error: Attribute '{config_id}' on the target object is not callable."
    )

  logging.info(
      'Attempting to call: %s() on object of type %s',
      config_id,
      type(target_obj),
  )
  return method_to_call()


def _get_gemma_base_model(
    model_config: dict[str, Any], mesh: jax.sharding.Mesh
):
  """Get the base model from the intermediate checkpoint."""
  model_params = obtain_model_params(model_config['model_name'])
  model_lib_module = get_model_module(
      model_config['model_name'], ModelModule.MODEL
  )
  abs_model: nnx.Module = nnx.eval_shape(
      lambda: model_lib_module.Gemma(
          model_params, rngs=nnx.Rngs(model_config.get('rng_seed', 0))
      )
  )
  abs_state = nnx.state(abs_model)
  abs_state = jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
      abs_state,
      nnx.get_named_sharding(abs_state, mesh),
  )
  checkpointer = ocp.StandardCheckpointer()
  restored_params = checkpointer.restore(
      os.path.join(model_config['intermediate_ckpt_dir'], 'state'),
      target=abs_state,
  )

  graph_def, _ = nnx.split(abs_model)
  model = nnx.merge(graph_def, restored_params)
  return model, model_params


def apply_lora_to_model(base_model, mesh, lora_config):
  """Apply Lora to the base model if given lora config."""
  logging.info('lora_config %r', lora_config)
  # Basic keyword arguments for LoraProvider
  lora_kwargs = {
      'module_path': lora_config['module_path'],
      'rank': lora_config['rank'],
      'alpha': lora_config['alpha'],
  }
  has_tile_size = 'tile_size' in lora_config
  has_weight_qtype = 'weight_qtype' in lora_config
  if has_tile_size:
    lora_kwargs['tile_size'] = lora_config['tile_size']
  if has_weight_qtype:
    lora_kwargs['weight_qtype'] = lora_config['weight_qtype']
    logging.info('Qlora is applied')
  else:
    logging.info('Lora is applied')

  try:
    lora_provider = qwix.LoraProvider(**lora_kwargs)
  except TypeError as e:
    logging.error(
        'Error initializing qwix.LoraProvider: %s. Kwargs: %s', e, lora_kwargs
    )
    # Depending on desired behavior, you might re-raise or return base_model
    raise

  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )
  if mesh is not None:
    lora_model = reshard.reshard_model_to_mesh(lora_model, mesh)
  return lora_model


def _gemma_conversion(
    model_config: dict[str, Any], gemma: nnx.Module, params, mesh
):
  """Convert the Gemma model to NNX format."""
  checkpointer = ocp.StandardCheckpointer()
  _, state = nnx.split(gemma)
  checkpointer.save(
      os.path.join(model_config['intermediate_ckpt_dir'], 'state'),
      state,
      force=True,
  )
  checkpointer.wait_until_finished()
  # Delete the intermediate model to save memory
  del params
  del gemma
  del state
  gc.collect()

  # Reload the model
  return _get_gemma_base_model(model_config, mesh)


# TODO(b/451662153): make gemma3 and gemma2 loading logic more consistent.
# Currently, gemma2 uses _create_gemma_model_from_params while gemma3 uses
# _create_gemma3_model_from_checkpoint.
def _create_gemma3_model_from_checkpoint(
    ckpt_path: str, model_name: str, mesh: jax.sharding.Mesh
) -> Tuple[nnx.Module, Any]:
  """Creates a Gemma3 model from a checkpoint.

  Args:
      ckpt_path: The path to the checkpoint.
      model_name: The name of the model (e.g., "qwen2.5-0.5b", "llama3.2-3b").
      mesh: Mesh object for device layout.

  Returns:
      A tuple containing:
          - model: The loaded and potentially LoRA-applied nnx.Module.
          - model_params: The model parameters.
  """
  model_params = obtain_model_params(model_name)
  params_lib = get_model_module(model_name, ModelModule.PARAMS)
  model = params_lib.create_model_from_checkpoint(ckpt_path, model_params, mesh)
  return model, model_params


def _create_gemma_model_from_params(
    params_path: str, model_name: str
) -> Tuple[nnx.Module, Any]:
  """Loads Gemma params and creates a model."""
  params_lib = get_model_module(model_name, ModelModule.PARAMS)
  model_params = params_lib.load_and_format_params(params_path)
  model_module_lib = get_model_module(model_name, ModelModule.MODEL)
  model_family, version = naming.split(model_name)
  # TODO(b/451662153): have gemma2 version handling done better in naming.py
  if model_family == 'gemma2':
    version = f'2-{version}'
  model = model_module_lib.Gemma.from_params(model_params, version=version)
  return model, model_params


def create_tokenizer(tokenizer_config, tokenizer_path: str | None):
  if not tokenizer_path:
    tokenizer_path = tokenizer_config['toknenizer_path']
  tokenizer_type, add_bos, add_eos = (
      tokenizer_config['tokenizer_type'],
      tokenizer_config['add_bos'],
      tokenizer_config['add_eos'],
  )

  return tokenizer_lib.Tokenizer(
      tokenizer_type,
      tokenizer_path,
      add_bos,
      add_eos,
      os.environ.get('HF_TOKEN'),
  )


def create_model(
    model_config: dict[str, Any],
    tokenizer_config: dict[str, Any],
    mesh: jax.sharding.Mesh,
) -> Tuple[nnx.Module, str]:
  """Creates a model and determines the tokenizer path based on the model config.

  This function handles model loading from various sources (GCS, Kaggle, HF)
  and applies LoRA if specified in the config.

  Args:
      model_config: A dictionary containing model configuration, including
        'model_name', 'model_source', 'model_id', 'model_download_path',
        'intermediate_ckpt_dir', and optionally 'lora_config'.
      tokenizer_config: A dictionary containing tokenizer configuration,
        including 'tokenizer_path'.
      mesh: The JAX sharding Mesh object.

  Returns:
      A tuple containing:
          - model: The loaded and potentially LoRA-applied nnx.Module.
          - tokenizer_path: The determined path to the tokenizer model.
  """
  model: nnx.Module = None
  model_params: Any = None
  tokenizer_path: str = tokenizer_config['tokenizer_path']
  model_name = model_config['model_name']
  model_source = model_config['model_source']

  if model_name.startswith('gemma3') and model_source == 'gcs':

    ckpt_path = model_config['model_id']
    model, model_params = _create_gemma3_model_from_checkpoint(
        ckpt_path, model_name, mesh
    )
    tokenizer_path = 'gs://gemma-data/tokenizers/tokenizer_gemma3.model'

  # TODO(sizhi): Remove gemma conversion logic once load safetensors for
  # gemma is ready.
  elif model_name.startswith('gemma') and model_source == 'kaggle':
    from tunix.oss import utils as oss_utils

    # Download model from Kaggle requires NNX conversion and can takes long
    # time. It is recommended to save the NNX converted model for later runs.
    ckpt_path = oss_utils.kaggle_pipeline(model_config)
    intermediate_ckpt_dir = model_config['intermediate_ckpt_dir']
    skip_nnx_conversion: bool = os.path.exists(intermediate_ckpt_dir)

    def nnx_conversion():
      # Load the model and save to checkpoint locally, then reload the model
      # sharded. This is a workaround, as the checkpoints on Kaggle don't
      # work with NNX. This takes a long time. Skip if conversion is not
      # needed.
      if model_name.startswith('gemma2'):
        params_path = os.path.join(ckpt_path, model_name)
      else:  # gemma
        suffix = '-'.join(model_name.split('-')[1:])
        params_path = os.path.join(ckpt_path, suffix)

      model, params = _create_gemma_model_from_params(params_path, model_name)
      return _gemma_conversion(model_config, model, params, mesh)

    if skip_nnx_conversion:
      try:
        model, model_params = _get_gemma_base_model(model_config, mesh)
      except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.warning(
            'Failed to load from intermediate_ckpt_dir %s: %s. '
            'Falling back to NNX conversion.',
            intermediate_ckpt_dir,
            e,
        )
        model, model_params = nnx_conversion()

    else:
      model, model_params = nnx_conversion()
    tokenizer_path = os.path.join(ckpt_path, 'tokenizer.model')

  elif model_source == 'huggingface':
    from tunix.oss import utils as oss_utils
    # for all other model
    oss_utils.hf_pipeline(model_config)
  else:
    logging.error(
        'Unsupported workflow: from %s to download %s.',
        model_source,
        model_name,
    )

  if not model_params:
    # pick corresponding config based on model version
    model_params = obtain_model_params(model_name)

    with mesh:
      model = create_model_dynamically(
          model_name,
          model_config['model_download_path'],
          model_params,
          mesh,
      )

  if model_config.get('lora_config'):
    # Apply Lora to model if given lora config
    model = apply_lora_to_model(model, mesh, model_config['lora_config'])
  else:
    logging.info('Training with Full Weight')

  if model_config['model_display']:
    nnx.display(model)

  return model, tokenizer_path
