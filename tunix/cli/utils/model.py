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

import gc
import importlib
import os
import re
from typing import Any, Tuple
from absl import logging
from flax import nnx
import huggingface_hub as hf
import jax
import jax.numpy as jnp
import kagglehub
from orbax import checkpoint as ocp
import qwix
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma import gemma as gemma_lib
from tunix.models.gemma import params as gemma_params_lib
from tunix.models.gemma3 import model as gemma3_lib
from tunix.models.gemma3 import params as gemma3_params_lib
from tunix.models.llama3 import model as llama3_lib
from tunix.models.qwen2 import model as qwen2_lib
from tunix.models.qwen3 import model as qwen3_lib


# Map prefixes to the target object containing the methods.
CONFIG_MAP = {
    'gemma': gemma_lib.ModelConfig,
    'gemma2': gemma_lib.ModelConfig,
    'gemma3': gemma3_lib.ModelConfig,
    'llama3.1': llama3_lib.ModelConfig,
    'llama3.2': llama3_lib.ModelConfig,
    'qwen2.5': qwen2_lib.ModelConfig,
    'qwen3': qwen3_lib.ModelConfig,
}

_BASE_MODULE_PATH = 'tunix.models'  # pylint: disable=invalid-name


def get_model_module(model_name: str) -> Any:
  """Dynamically imports the parameter module based on the model name."""
  # Extract the base model type (e.g., "qwen2", "llama3")
  match = re.match(r'^[a-zA-Z0-9]+', model_name)
  if not match:
    raise ValueError(f'Invalid model name format: {model_name}')
  model_type = match.group(0)
  # Construct the full module path, e.g.,.path.to.your.models.qwen2.params
  if model_name.startswith('gemma2'):
    model_type = 'gemma'
  module_path = f'{_BASE_MODULE_PATH}.{model_type}.params'
  try:
    print(f'Attempting to import: {module_path}')
    model_module = importlib.import_module(module_path)
    return model_module
  except ImportError as exc:  # Capture the original exeception as 'exc'
    raise ImportError(
        f'Could not import module for model type: {model_type} '
        f'at path: {module_path}. Please check BASE_MODULE_PATH '
        'and ensure the module exists and is a dependency.'
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
  model_module = get_model_module(model_name)

  try:
    create_fn = getattr(model_module, 'create_model_from_safe_tensors')
  except AttributeError as exc:
    raise AttributeError(
        "'create_model_from_safe_tensors' not found in module "
        f'{model_module.__name__} for model {model_name}'
    ) from exc

  logging.info(
      'Calling %s.create_model_from_safe_tensors', model_module.__name__
  )
  return create_fn(file_dir=file_dir, config=model_config, mesh=mesh)


def _get_core_version(model_name: str, matched_prefix: str) -> str:
  """Extracts the core version string from the model name."""
  if not model_name.startswith(matched_prefix):
    return ''

  suffix = model_name[len(matched_prefix) :]

  # Remove leading separator (- or .) if present
  if suffix.startswith('-') or suffix.startswith('.'):
    suffix = suffix[1:]

  if not suffix:
    return ''

  # The core version is the part before the first hyphen (e.g., in "-it")
  core_version = suffix.split('-')[0]
  return core_version.replace('.', '_')


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
  target_obj = None
  matched_prefix = ''

  # Find the longest matching prefix
  for prefix, obj in CONFIG_MAP.items():
    if model_name.startswith(prefix):
      if len(prefix) > len(matched_prefix):
        matched_prefix = prefix
        target_obj = obj

  if not target_obj:
    raise ValueError(f'Unsupported model string prefix for: {model_name}')

  logging.info('Routing %s using prefix %s', model_name, matched_prefix)

  family_snake = matched_prefix.replace('-', '_').replace('.', '_')
  core_version = _get_core_version(model_name, matched_prefix)

  if not core_version:
    raise ValueError(
        f"Could not extract core version from '{model_name}' "
        f"for prefix '{matched_prefix}'."
    )

  function_name = f'{family_snake}_{core_version}'

  if not hasattr(target_obj, function_name):
    raise AttributeError(
        f"Error: Function '{function_name}' not found on the target object "
        f"for prefix '{matched_prefix}'. Target object type: {type(target_obj)}"
    )

  method_to_call = getattr(target_obj, function_name)

  if not callable(method_to_call):
    raise TypeError(
        f"Error: Attribute '{function_name}' on the target object is not"
        ' callable.'
    )

  logging.info(
      'Attempting to call: %s() on object of type %s',
      function_name,
      type(target_obj),
  )
  return method_to_call()


def _get_base_model(model_config: dict[str, Any], mesh: jax.sharding.Mesh):
  """Get the base model from the intermediate checkpoint."""
  model_params = obtain_model_params(model_config['model_name'])
  abs_model: nnx.Module = nnx.eval_shape(
      lambda: gemma_lib.Transformer(
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

  with mesh:
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

  return lora_model


def _kaggle_pipeline(model_config: dict[str, Any]):
  if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
    kagglehub.login()
  os.environ['KAGGLEHUB_CACHE'] = model_config['model_download_path']
  return kagglehub.model_download(model_config['model_id'])


def _hf_pipeline(model_config: dict[str, Any]):
  if 'HF_TOKEN' not in os.environ:
    hf.login()
  all_files = hf.list_repo_files(model_config['model_id'])
  filtered_files = [f for f in all_files if not f.startswith('original/')]
  for filename in filtered_files:
    hf.hf_hub_download(
        repo_id=model_config['model_id'],
        filename=filename,
        local_dir=model_config['model_download_path'],
    )
  logging.info(
      'Downloaded %s to: %s',
      filtered_files,
      model_config['model_download_path'],
  )


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
  return _get_base_model(model_config, mesh)


def _get_model_version_suffix(model_name: str) -> str:
  """Extracts the version/variant suffix from a model name string.

  The function is based on the following examples:
  - "gemma2-2b-it" -> "2-2b-it"
  - "gemma2-2b" -> "2-2b"
  - "gemma-2b" -> "2b"

  Args:
      model_name: The full model name string.

  Returns:
      The version/variant suffix string.

  Raises:
      ValueError: If the model_name does not match a known pattern or
                  unsupported model family.
  """
  if model_name.startswith('gemma'):
    # Pattern 1: Matches names like "gemma2-2b-it", "gemma7b", etc.
    # Captures the part starting with the first digit after "gemma".
    match = re.match(r'^gemma(\d.*)$', model_name)
    if match:
      return match.group(1)

    # Pattern 2: Matches names like "gemma-2b", "gemma-7b-it", etc.
    # Captures the part after "gemma-".
    match = re.match(r'^gemma-(.+)$', model_name)
    if match:
      return match.group(1)

    # If neither pattern matches
    raise ValueError(f'Unrecognized gemma model format: {model_name}')
  else:
    # This part can be extended for other model families like "llama", etc.
    raise ValueError(f'Unsupported model family for: {model_name}')


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
    model_params = obtain_model_params(model_name)
    model = gemma3_params_lib.create_model_from_checkpoint(
        ckpt_path, model_params, mesh
    )
    tokenizer_path = 'gs://gemma-data/tokenizers/tokenizer_gemma3.model'

  # TODO(sizhi): Remove gemma conversion logic once load safetensors for
  # gemma is ready.
  elif model_name.startswith('gemma') and model_source == 'kaggle':

    # Download model from Kaggle requires NNX conversion and can takes long time.
    # It is recommended to save the NNX converted model for later runs.
    ckpt_path = _kaggle_pipeline(model_config)
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

      params = gemma_params_lib.load_and_format_params(params_path)
      model = gemma_lib.Transformer.from_params(
          params, version=_get_model_version_suffix(model_name)
      )
      return _gemma_conversion(model_config, model, params, mesh)

    if skip_nnx_conversion:
      try:
        model, model_params = _get_base_model(model_config, mesh)
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
    # for all other model
    _hf_pipeline(model_config)

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
