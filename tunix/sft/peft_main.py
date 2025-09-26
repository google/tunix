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

"""Main entry point for PEFT training."""
from collections.abc import Callable
import gc
import importlib
import os
import re
import time
from typing import Any

from absl import app
from absl import logging
from flax import nnx
import huggingface_hub as hf
import jax
import jax.numpy as jnp
import kagglehub
from orbax import checkpoint as ocp
import qwix
from tunix.examples.data import translation_dataset as data_lib
from tunix.models.gemma import gemma as gemma_lib
from tunix.models.gemma import params as gemma_params_lib
from tunix.models.gemma3 import model as gemma3_lib
from tunix.models.gemma3 import params as gemma3_params_lib
from tunix.models.llama3 import model as llama3_lib
from tunix.models.qwen2 import model as qwen2_lib
from tunix.models.qwen3 import model as qwen3_lib
from tunix.sft import utils
from tunix.sft import config
from tunix.sft import peft_trainer

# Map prefixes to the target object containing the methods.
CONFIG_MAP = {
    'gemma': gemma_lib.TransformerConfig,
    'gemma2': gemma_lib.TransformerConfig,
    'gemma3': gemma3_lib.Gemma3Config,
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
    model_name: str, file_dir: str, model_config: Any, mesh: Any
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


def obtain_model_config(model_name: str):
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
  function_name = model_name.replace('-', '_').replace('.', '_')

  if not hasattr(target_obj, function_name):
    raise AttributeError(
        f"Error: Function '{function_name}' not found on the target object for"
        f" prefix '{matched_prefix}'."
    )

  method_to_call = getattr(target_obj, function_name)

  if not callable(method_to_call):
    raise TypeError(
        f"Error: Attribute '{function_name}' on the target object is not"
        ' callable.'
    )

  logging.info('Attempting to call: %s()', function_name)
  return method_to_call()


def _get_base_model(hyperparms: config.HyperParameters, mesh):
  """Get the base model from the intermediate checkpoint."""
  model_config = obtain_model_config(hyperparms.config['model_name'])
  abs_model: nnx.Module = nnx.eval_shape(
      lambda: gemma_lib.Transformer(
          model_config, rngs=nnx.Rngs(hyperparms.config['rng_seed'])
      )
  )
  _, abs_state = nnx.state(abs_model)
  abs_state = jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
      abs_state,
      nnx.get_named_sharding(abs_state, mesh),
  )
  checkpointer = ocp.StandardCheckpointer()
  restored_params = checkpointer.restore(
      os.path.join(hyperparms.config['intermediate_ckpt_dir'], 'state'),
      target=abs_state,
  )

  graph_def, _ = nnx.split(abs_model)
  model = nnx.merge(graph_def, restored_params)
  return model


def _apply_lora_to_model(base_model, mesh, lora_config):
  """Apply Lora to the base model if given lora config."""
  logging.info('lora_config %r', lora_config)
  # Basic keyword arguments for LoraProvider
  lora_kwargs = {
      'module_path': '.*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj',
      'rank': lora_config['rank'],
      'alpha': lora_config['alpha'],
      'tile_size': lora_config['tile_size'],
  }
  has_weight_qtype = 'weight_qtype' in lora_config

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


def _source_third_party(source: str):
  if source == 'kaggle' or source == 'huggingface':
    return True
  else:
    return False


def _kaggle_pipeline(hyperparms: config.HyperParameters):
  if 'T_KAGGLE_USERNAME' not in os.environ or 'T_KAGGLE_KEY' not in os.environ:
    kagglehub.login()
  ckpt_path = kagglehub.model_download(hyperparms.config['model_id'])
  return ckpt_path


def _hf_pipeline(hyperparms: config.HyperParameters):
  if 'T_HF_TOKEN' not in os.environ:
    hf.login()
  all_files = hf.list_repo_files(hyperparms.config['model_id'])
  filtered_files = [f for f in all_files if not f.startswith('original/')]
  for filename in filtered_files:
    hf.hf_hub_download(
        repo_id=hyperparms.config['model_id'],
        filename=filename,
        local_dir=hyperparms.config['hf_cp_base_model_directory'],
    )
  logging.info(
      'Downloaded %s to: %s',
      filtered_files,
      hyperparms.config['hf_cp_base_model_directory'],
  )


def _gemma_conversion(
    hyperparms: config.HyperParameters, gemma: nnx.Module, params, mesh
):
  """Convert the Gemma model to NNX format."""
  checkpointer = ocp.StandardCheckpointer()
  _, state = nnx.split(gemma)
  checkpointer.save(
      os.path.join(hyperparms.config['intermediate_ckpt_dir'], 'state'), state
  )
  # Wait for ckpt to save successfully
  time.sleep(200)

  # Delete the intermediate model to save memory
  del params
  del gemma
  del state
  gc.collect()

  # Reload the model
  gemma = _get_base_model(hyperparms, mesh)
  return gemma


def _is_gemma(model_name: str):
  # Returns True if model starts with gemma
  if model_name.startswith('gemma'):
    return True
  return False


def run_peft_trainer(hyperparms: config.HyperParameters):
  """Run the PEFT trainer."""
  model: nnx.Module | None = None
  mesh: jax.sharding.Mesh | None = None
  tokenizer: Any | None = None
  my_gen_model_input_fn: (
      Callable[[peft_trainer.TrainingInput], dict[str, Any]] | None
  ) = None

  model_name = hyperparms.config['model_name']
  model_source = hyperparms.config['model_source']

  mesh = jax.make_mesh(*hyperparms.mesh)
  # TODO(sizhi): Remove gemma conversion logic once load safetensors for
  # gemma is ready.
  if _is_gemma(model_name):
    if model_source == 'kaggle':
      ckpt_path = _kaggle_pipeline(hyperparms)
    else:
      ckpt_path = hyperparms.config['model_id']

    model_version = model_name.split('-')[0]
    model_size = model_name.split('-')[1]

    logging.info('model_version %s', model_version)
    logging.info('model_size %s', model_size)

    if model_version == 'gemma' or model_version == 'gemma2':
      params_path = os.path.join(ckpt_path, model_size)
      params = gemma_params_lib.load_and_format_params(params_path)
      model = gemma_lib.Transformer.from_params(params, version=model_size)
      if _source_third_party(model_source):
        # Load the model and save to checkpoint locally, then reload the model
        # sharded. This is a workaround, as the checkpoint on 3rd party don't
        # work with NNX. This takes a long time. Skip if conversion is not
        # needed.
        model = _gemma_conversion(hyperparms, model, params, mesh)
      tokenizer = data_lib.GemmaTokenizer(
          os.path.join(ckpt_path, 'tokenizer.model')
      )

    else:
      model_config = obtain_model_config(model_name)
      model = gemma3_params_lib.create_model_from_checkpoint(
          ckpt_path, model_config, mesh
      )
      tokenizer = data_lib.GemmaTokenizer(hyperparms.config['tokenizer_path'])

  elif model_source == 'huggingface':
    _hf_pipeline(hyperparms)
    # pick corresponding config based on model version
    model_config = obtain_model_config(model_name)
    with mesh:
      model = create_model_dynamically(
          model_name,
          hyperparms.config['hf_cp_base_model_directory'],
          model_config,
          mesh,
      )

    tokenizer = data_lib.HFTokenizer(
        hyperparms.config['tokenizer_path'],
        add_bos=True,
        add_eos=True,
        hf_access_token=os.environ.get('T_HF_TOKEN'),
    )

  if hyperparms.config['lora_config']:
    # Apply Lora to model if given lora config
    model = _apply_lora_to_model(model, mesh, hyperparms.config['lora_config'])
  else:
    logging.info('Training with Full Weight')

  if hyperparms.config['visualize_model']:
    nnx.display(model)

  trainer = peft_trainer.PeftTrainer(
      model, hyperparms.optimizer, hyperparms.training_config
  )

  def gen_model_input_fn(x: peft_trainer.TrainingInput):
    pad_mask = x.input_tokens != 0

    positions = utils.build_positions_from_mask(pad_mask)
    attention_mask = utils.make_causal_attn_mask(pad_mask)
    return {
        'input_tokens': x.input_tokens,
        'input_mask': x.input_mask,
        'positions': positions,
        'attention_mask': attention_mask,
    }

  my_gen_model_input_fn = gen_model_input_fn
  trainer = trainer.with_gen_model_input_fn(my_gen_model_input_fn)

  train_ds, validation_ds = data_lib.create_datasets(
      dataset_name=hyperparms.config['dataset_name'],
      global_batch_size=hyperparms.config['batch_size'],
      max_target_length=hyperparms.config['max_target_length'],
      num_train_epochs=hyperparms.config['num_train_epochs'],
      tokenizer=tokenizer,
  )

  with mesh:
    trainer.train(train_ds, validation_ds)


def main(argv, **kwargs):
  hp = config.initialize(argv, **kwargs)
  run_peft_trainer(hp)


if __name__ == '__main__':
  app.run(main)
