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
from typing import Any
from absl import app
from absl import flags
from absl import logging
from flax import nnx
import jax
from tunix.cli import config
from tunix.cli.utils import model as model_lib
from tunix.examples.data import translation_dataset as data_lib
from tunix.sft import peft_trainer
from tunix.sft import utils
from tunix.utils import mesh as mesh_lib

_PATHWAYS_BNS = flags.DEFINE_string(
    'pathways_bns', None, 'BNS address of the Pathways server.'
)


class PeftPipeline(config.HyperParameters):
  """Class for running the Peft trainer."""

  def run_peft_trainer(self):
    """Run the PEFT trainer."""
    axis_shapes, axis_names = self.parse_mesh_config('model_config')
    mesh: jax.sharding.Mesh = mesh_lib.create_mesh(axis_shapes, axis_names)
    model: nnx.Module | None = None
    tokenizer: Any | None = None
    my_gen_model_input_fn: (
        Callable[[peft_trainer.TrainingInput], dict[str, Any]] | None
    ) = None
    model, tokenizer_path = model_lib.create_model(
        self.config['model_config'], self.config['tokenizer_config'], mesh
    )
    if model is None:
      raise ValueError('model is None')
    tokenizer = model_lib.create_tokenizer(
        self.config['tokenizer_config'], tokenizer_path
    )
    optimizer = self.create_optimizer('optimizer_config')
    trainer = peft_trainer.PeftTrainer(
        model,
        optimizer,
        peft_trainer.TrainingConfig(
            **self.obtain_training_config_dict('training_config')
        ),
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

    train_ds, eval_ds = data_lib.create_datasets(
        dataset_name=self.config['dataset_name'],
        global_batch_size=self.config['batch_size'],
        max_target_length=self.config['max_target_length'],
        num_train_epochs=self.config['num_train_epochs'],
        tokenizer=tokenizer,
        tfds_download=self.config['tfds_download'],
    )

    with mesh:
      trainer.train(train_ds, eval_ds)


def _setup_jax_pathways(pathways_bns: str):
  """Sets up Jax with Pathways."""
  flags.FLAGS.pathways_ifrt = True
  jax.config.update('jax_xla_backend', 'pathways')
  jax.config.update('jax_backend_target', pathways_bns)


def _init_multi_controller_jax():
  """Initializes multi-controller JAX before any device is accessed.

  Under multi-controller JAX every host runs this same program and
  `jax.distributed.initialize()` stitches the per-host devices into one global
  device set / mesh. On a single host (and under Pathways, which uses a single
  controller) this must be a no-op so single-host behavior stays byte-for-byte
  identical.

  With no arguments `jax.distributed.initialize()` auto-detects the cluster from
  the environment (e.g. GKE / Cloud TPU pod env vars). On a single host with no
  such environment, auto-detection finds no coordinator and `jax` raises
  ``ValueError('coordinator_address should be defined.')``. We treat ONLY that
  specific case as "nothing to initialize" and proceed with the local devices.
  Any other failure (e.g. an unreachable coordinator on a real multi-host pod)
  is a genuine error and is allowed to propagate.
  """
  try:
    jax.distributed.initialize()
  except ValueError as e:
    # On a single host with no cluster env vars, auto-detection cannot find a
    # coordinator and raises this exact ValueError. That is the legitimate
    # single-controller no-op case. Anything else (including a real multi-host
    # coordinator connection failure) must NOT be swallowed.
    if 'coordinator_address should be defined' not in str(e):
      raise
    logging.info(
        'No multi-host cluster detected; running single-controller. (%s)', e
    )

  logging.info(
      'JAX runtime initialized: process_index=%d process_count=%d '
      'local_device_count=%d global_device_count=%d',
      jax.process_index(),
      jax.process_count(),
      jax.local_device_count(),
      jax.device_count(),
  )


def main(argv, **kwargs):
  if _PATHWAYS_BNS.value:
    _setup_jax_pathways(_PATHWAYS_BNS.value)
  else:
    # Pathways is a single-controller backend, so multi-controller init only
    # applies to the non-Pathways path.
    _init_multi_controller_jax()
  pipeline = PeftPipeline(argv, **kwargs)
  logging.info(
      '--- Launching PEFT pipeline with following config ---\n'
      '%r\n--------------------------',
      pipeline.config,
  )
  pipeline.run_peft_trainer()


if __name__ == '__main__':
  app.run(main)
