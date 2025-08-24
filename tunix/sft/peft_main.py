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
import os
import sys
import time
from typing import Any
from absl import app, logging
from flax import nnx
import huggingface_hub as hf
import jax
import jax.numpy as jnp
import kagglehub
from orbax import checkpoint as ocp
import qwix
import transformers
from tunix.models.gemma import data as gemma_data_lib
from tunix.models.gemma import gemma as gemma_lib
from tunix.models.gemma import params as gemma_params_lib
from tunix.models.gemma3 import model as gemma3_lib
from tunix.models.llama3 import model as llama3_lib
from tunix.models.llama3 import params as llama3_params_lib
from tunix.models.qwen2 import model as qwen2_lib
from tunix.models.qwen3 import model as qwen3_lib
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


def get_base_model(hyperparms: config.HyperParameters):
  model_config = obtain_model_config(hyperparms.config['model_name'])
  mesh = jax.make_mesh(*hyperparms.mesh)
  abs_gemma: nnx.Module = nnx.eval_shape(
      lambda: gemma_lib.Transformer(model_config, rngs=nnx.Rngs(params=0))
  )
  abs_state = nnx.state(abs_gemma)
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
  jax.block_until_ready(restored_params)

  graph_def, _ = nnx.split(abs_gemma)
  gemma = nnx.merge(graph_def, restored_params)
  return gemma, mesh


def _apply_lora_to_model(base_model, mesh, lora_config):
  """Apply Lora to the base model if given lora config."""
  lora_provider = qwix.LoraProvider(
      module_path='.*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj',
      rank=lora_config['rank'],
      alpha=lora_config['alpha'],
      weight_qtype=lora_config['weight_qtype'],
      tile_size=lora_config['tile_size'],
  )
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
  if source == 'kaggle' or source == 'hf':
    return True
  else:
    return False


def _kaggle_pipeline(hyperparms: config.HyperParameters):
  if 'T_KAGGLE_USERNAME' not in os.environ or 'T_KAGGLE_KEY' not in os.environ:
    kagglehub.login()
  ckpt_path = kagglehub.model_download(hyperparms.config['ckpt_dir'])
  return ckpt_path


def _hf_pipeline(hyperparms: config.HyperParameters):
  if 'T_HF_TOKEN' not in os.environ:
    hf.login(token='T_HF_TOKEN')
  all_files = hf.list_repo_files(hyperparms.config['ckpt_dir'])
  filtered_files = [f for f in all_files if not f.startswith('original/')]
  for filename in filtered_files:
    hf.hf_hub_download(
        repo_id=hyperparms.config['ckpt_dir'],
        filename=filename,
        local_dir=hyperparms.config['hf_cp_base_model_directory'],
    )
  logging.info(
      f'Downloaded {filtered_files} to:'
      f" {hyperparms.config['hf_cp_base_model_directory']}"
  )


def _gemma_conversion(
    hyperparms: config.HyperParameters, gemma: nnx.Module, params
):
  """Convert the Gemma model to NNX format."""
  checkpointer = ocp.StandardCheckpointer()
  _, state = nnx.split(gemma)
  checkpointer.save(
      os.path.join(hyperparms.config['intermediate_ckpt_dir'], 'state'), state
  )
  # Wait for ckpt to save successfully
  time.sleep(300)

  # Delete the intermediate model to save memory
  del params
  del gemma
  del state
  gc.collect()

  # Reload the model
  gemma, mesh = get_base_model(hyperparms)
  return gemma, mesh


def _is_gemma(model_name: str):
  # Returns True if model starts with gemma
  if model_name.startswith('gemma'):
    if model_name.startswith(('gemma2', 'gemma3')):
      return False
    return True
  return False


def _validate_current_workflow(model_name: str, ckpt_source: str):
  if _is_gemma(model_name):
    if _source_third_party(ckpt_source):
      if ckpt_source != 'kaggle':
        raise ValueError(
            f"unsupported workflow: '{model_name}' from third party source must"
            f" use 'kaggle', got '{ckpt_source}'"
        )
      # else: gemma, third party, kaggle -> OK
    # else: gemma, first party -> OK
    logging.info(
        f"OK: Valid config for '{model_name}' with source '{ckpt_source}'"
    )

  elif model_name.startswith('llama3.1'):
    if ckpt_source != 'huggingface':
      raise ValueError(
          f"unsupported workflow: '{model_name}' must use 'hf' source, got"
          f" '{ckpt_source}'"
      )
    # else: llama3.1, hf -> OK
    logging.info(
        f"OK: Valid config for '{model_name}' with source '{ckpt_source}'"
    )

  else:
    # Any model_name not matching the above patterns is unsupported.
    raise ValueError(
        'unsupported workflow: Validation rules not defined for model'
        f" '{model_name}'"
    )


def run_peft_trainer(hyperparms: config.HyperParameters):
  """Run the PEFT trainer."""
  jax.config.update('jax_debug_nans', True)

  model: nnx.Module | None = None
  mesh: jax.sharding.Mesh | None = None
  tokenizer: Any | None = None
  gen_model_input_fn: Callable | None = None

  model_name = hyperparms.config['model_name']
  ckpt_source = hyperparms.config['ckpt_source']

  # Currently, we only support limited workflow.
  _validate_current_workflow(model_name, ckpt_source)

  if _is_gemma(model_name):
    if ckpt_source == 'kaggle':
      ckpt_path = _kaggle_pipeline(hyperparms)
    else:
      ckpt_path = hyperparms.config['ckpt_dir']

    model_version = model_name.split('-')[1]
    # Only gemma is verified, block other workflow for now.
    params = gemma_params_lib.load_and_format_params(
        os.path.join(ckpt_path, model_version)
    )

    # This is necessary to prevent NaN and Inf in input tensor.
    jax.block_until_ready(params)

    model = gemma_lib.Transformer.from_params(params, version=model_version)
    if _source_third_party(ckpt_source):
      # Load the model and save to checkpoint locally, then reload the model
      # sharded. This is a workaround, as the checkpoint on 3rd party don't work
      # with NNX. This takes a long time. Skip if conversion is not needed.
      model, mesh = _gemma_conversion(hyperparms, model, params)
    else:
      mesh = jax.make_mesh(*hyperparms.mesh)

    tokenizer = gemma_data_lib.GemmaTokenizer(
        os.path.join(ckpt_path, 'tokenizer.model')
    )

    def gen_model_input_fn(x: peft_trainer.TrainingInput):
      pad_mask = x.input_tokens != tokenizer.pad_id()
      positions = gemma_lib.build_positions_from_mask(pad_mask)
      attention_mask = gemma_lib.make_causal_attn_mask(pad_mask)
      return {
          'input_tokens': x.input_tokens,
          'input_mask': x.input_mask,
          'positions': positions,
          'attention_mask': attention_mask,
      }

    train_ds, validation_ds = gemma_data_lib.create_datasets(
      dataset_name=hyperparms.config['dataset_name'],
      global_batch_size=hyperparms.config['batch_size'],
      max_target_length=hyperparms.config['max_target_length'],
      num_train_epochs=hyperparms.config['num_train_epochs'],
      tokenizer=tokenizer)
    
  elif model_name.startswith('llama3.1') and ckpt_source == 'huggingface':
    model_cp_path = hyperparms.config['hf_cp_base_model_directory']
    _hf_pipeline(hyperparms)
    mesh = jax.make_mesh(*hyperparms.mesh)
    # pick corresponding config based on model version
    model_config = obtain_model_config(model_name)
    model = llama3_params_lib.create_model_from_safe_tensors(
        model_cp_path, model_config, mesh
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_cp_path)
    gen_model_input_fn = model.get_model_input()
    
    train_ds, validation_ds = gemma_data_lib.create_datasets(
    dataset_name=hyperparms.config['dataset_name'],
    global_batch_size=hyperparms.config['batch_size'],
    max_target_length=hyperparms.config['max_target_length'],
    num_train_epochs=hyperparms.config['num_train_epochs'],
    tokenizer=tokenizer,
    )

  if hyperparms.config['visualize_model']:
    nnx.display(model)

  if hyperparms.config['lora_config']:
    # Apply Lora to model if given lora config
    model = _apply_lora_to_model(model, mesh, hyperparms.config['lora_config'])
    if hyperparms.config['visualize_model']:
      nnx.display(model)



  trainer = peft_trainer.PeftTrainer(
      model, hyperparms.optimizer, hyperparms.training_config
  )
  trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)
  with mesh:
    trainer.train(train_ds, validation_ds)


def main(argv, **kwargs):
  hp = config.initialize(argv, **kwargs)
  run_peft_trainer(hp)


if __name__ == '__main__':
  app.run(main)