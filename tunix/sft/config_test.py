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


class ConfigTest(absltest.TestCase):

  def test_config_from_yaml(self):
    non_existent_argv = ["", "nonexistent_config.yaml"]
    self.assertRaises(ValueError, config.initialize, non_existent_argv)

    existing_argv = ["", "base_config.yaml"]
    config.initialize(existing_aconfigrgv)

  def run_test_peft_trainer(self, hp):
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(rngs=rngs)
    peft_trainer.PeftTrainer(model, optax.sgd(1e-3), hp.training_config)

  def test_override_training_config_simple(self):
    argv = [
        "",
        "base_config.yaml",
        "training_config.max_steps=150",
        "training_config.data_sharding_axis=['fsdp','dp']",
    ]
    hp = config.initialize(argv)
    self.assertEqual(hp.config["max_steps"], 150)
    self.assertEqual(hp.config["data_sharding_axis"], ["fsdp", "dp"])
    self.run_test_peft_trainer(hp)

  def test_override_training_config_complex(self):
    argv = [
        "",
        "base_config.yaml",
        "training_config.profiler_options.log_dir=/tmp/profiler_log_dir",
        "training_config.profiler_options.skip_first_n_steps=1",
        "training_config.profiler_options.profiler_steps=5",
    ]
    self.run_test_peft_trainer(config.initialize(argv))

  def test_valid_kaggle_with_intermediate_ckpt(self):
    argv = [
        "",
        "base_config.yaml",
        "ckpt_source=kaggle",
        "intermediate_ckpt=/path/to/ckpt",
    ]
    config.initialize(argv)

  def test_valid_huggingface_with_intermediate_ckpt(self):
    argv = [
        "",
        "base_config.yaml",
        "ckpt_source=huggingface",
        "intermediate_ckpt=/path/to/ckpt",
    ]
    config.initialize(argv)

  def test_valid_none_ckpt_source(self):
    argv = ["", "base_config.yaml", "ckpt_source=None"]
    config.initialize(argv)

  def test_invalid_kaggle_without_intermediate_ckpt(self):
    argv = ["", "base_config.yaml", "ckpt_source=kaggle", "intermediate_ckpt="]
    with self.assertRaises(ValueError):
      config.initialize(argv)

  def test_invalid_huggingface_without_intermediate_ckpt(self):
    argv = [
        "",
        "base_config.yaml",
        "ckpt_source=huggingface",
        "intermediate_ckpt=",
    ]
    with self.assertRaises(ValueError):
      config.initialize(argv)

  def test_invalid_ckpt_source(self):
    argv = ["", "base_config.yaml", "ckpt_source=invalid_source"]
    with self.assertRaises(ValueError):
      config.initialize(argv)


if __name__ == "__main__":
  absltest.main()
