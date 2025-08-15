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
import os
import sys
import time
from absl import app, logging
from flax import nnx
import jax
import jax.numpy as jnp
import kagglehub
from orbax import checkpoint as ocp
import qwix
from tunix.models.gemma import data as data_lib
from tunix.models.gemma import gemma as gemma_lib
from tunix.models.gemma import params as params_lib
from tunix.sft import config
from tunix.sft import peft_trainer


def get_base_model(hyperparms: config.HyperParameters):
  model_config = gemma_lib.TransformerConfig.gemma_2b()
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

  graph_def, _ = nnx.split(abs_gemma)
  gemma = nnx.merge(graph_def, restored_params)
  return gemma, mesh


def get_lora_model(base_model, mesh, lora_config):
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


def run_peft_trainer(hyperparms: config.HyperParameters):
  """Run the PEFT trainer."""
  # only gemma model is supported for now.
  if hyperparms.config['ckpt_source'] == 'kaggle':
    if (
        'T_KAGGLE_USERNAME' not in os.environ
        or 'T_KAGGLE_KEY' not in os.environ
    ):
      kagglehub.login()
    kaggle_ckpt_path = kagglehub.model_download('google/gemma/flax/2b')
    ckpt_path = os.path.join(
        kaggle_ckpt_path, hyperparms.config['model_version']
    )
    params = params_lib.load_and_format_params(ckpt_path)
    gemma = gemma_lib.Transformer.from_params(
        params, hyperparms.config['model_version']
    )
    # Load the model and save to checkpoint locally, then reload the model sharded. This is a workaround, the checkpoint on Kaggle don't work with NNX.
    checkpointer = ocp.StandardCheckpointer()
    graphdef, state = nnx.split(gemma)
    checkpointer.save(
        os.path.join(hyperparms.config['intermediate_ckpt_dir'], 'state'), state
    )
    # wait for ckpt to save successfully
    time.sleep(300)
    del params
    del gemma
    del state
    gc.collect()
    gemma, mesh = get_base_model(hyperparms)
    gemma_tokenizer = data_lib.GemmaTokenizer(
        os.path.join(kaggle_ckpt_path, 'tokenizer.model')
    )
  else:
    ckpt_path = os.path.join(
        hyperparms.config['ckpt_dir'], hyperparms.config['model_version']
    )
    params = params_lib.load_and_format_params(ckpt_path)
    gemma = gemma_lib.Transformer.from_params(
        params, hyperparms.config['model_version']
    )
    mesh = jax.make_mesh(*hyperparms.mesh)
    gemma_tokenizer = data_lib.GemmaTokenizer()

  if hyperparms.config['lora_config']:
    gemma = get_lora_model(gemma, mesh, hyperparms.config['lora_config'])

  train_ds, validation_ds = data_lib.create_datasets(
      dataset_name=hyperparms.config['dataset_name'],
      global_batch_size=hyperparms.config['batch_size'],
      max_target_length=hyperparms.config['max_target_length'],
      num_train_epochs=hyperparms.config['num_train_epochs'],
      tokenizer=gemma_tokenizer,
  )

  def gen_model_input_fn(x: peft_trainer.TrainingInput):
    pad_mask = x.input_tokens != gemma_tokenizer.pad_id()
    positions = gemma_lib.build_positions_from_mask(pad_mask)
    attention_mask = gemma_lib.make_causal_attn_mask(pad_mask)
    return {
        'input_tokens': x.input_tokens,
        'input_mask': x.input_mask,
        'positions': positions,
        'attention_mask': attention_mask,
    }

  trainer = peft_trainer.PeftTrainer(
      gemma, hyperparms.optimizer, hyperparms.training_config
  )
  trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)
  with mesh:
    trainer.train(train_ds, validation_ds)


def main(argv, **kwargs):
  hp = config.initialize(argv, **kwargs)
  run_peft_trainer(hp)


if __name__ == '__main__':
  app.run(main)
