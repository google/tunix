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
from flax import nnx
import jax
from tunix.cli import config
from tunix.cli.utils import model as model_lib
from tunix.examples.data import translation_dataset as data_lib
from tunix.sft import peft_trainer
from tunix.sft import utils


class PeftPipeline(config.HyperParameters):
  """Class for running the Peft trainer."""

  def run_peft_trainer(self):
    """Run the PEFT trainer."""
    mesh: jax.sharding.Mesh = self.create_mesh('model_config')
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
    )

    with mesh:
      trainer.train(train_ds, eval_ds)


def main(argv, **kwargs):
  pipeline = PeftPipeline(argv, **kwargs)
  pipeline.run_peft_trainer()


if __name__ == '__main__':
  app.run(main)
