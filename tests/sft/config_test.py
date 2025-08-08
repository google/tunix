# Copyright 2025 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from flax import nnx
import optax
from tunix.sft import config
from tunix.sft import peft_trainer
from tunix.tests import test_common as tc


def run_peft_trainer(hyper_parameters: config.HyperParameters):
  rngs = nnx.Rngs(0)
  model = tc.ToyTransformer(rngs=rngs)
  peft_trainer.PeftTrainer(
      model, optax.sgd(1e-3), hyper_parameters.training_config
  )


class ConfigTest(absltest.TestCase):

  def test_config_from_yaml(self):
    non_existent_argv = ["", "nonexistent_training_config.yaml"]
    self.assertRaises(ValueError, config.initialize, non_existent_argv)

    existing_argv = ["", "sft_training_config.yaml"]
    config.initialize(existing_argv)

  def test_override_config_simple(self):
    argv = [
        "",
        "sft_training_config.yaml",
        "max_steps=150",
        "data_sharding_axis=['fsdp','dp']",
    ]
    hp = config.initialize(argv)
    self.assertEqual(hp._config["max_steps"], 150)
    self.assertEqual(hp._config["data_sharding_axis"], ["fsdp", "dp"])
    run_peft_trainer(hp)

  def test_override_config_complex(self):
    argv = [
        "",
        "sft_training_config.yaml",
        "profiler_options.log_dir=/tmp/profiler_log_dir",
        "profiler_options.skip_first_n_steps=1",
        "profiler_options.profiler_steps=5",
    ]
    run_peft_trainer(config.initialize(argv))


if __name__ == "__main__":
  absltest.main()
