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

from typing import Any, Dict, List
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import optax
from tunix.sft import config
from tunix.sft import peft_trainer
from tunix.tests import test_common as tc


class ConfigTest(parameterized.TestCase):
  TEST_ARGV = [
      "",
      "base_config.yaml",
  ]

  def initialize_config(self, configs: list[str]):
    """Helper to build argv and initialize config."""
    argv = self.TEST_ARGV + configs
    return config.initialize(argv)

  def convert_nested_dict_to_list(
      self,
      data_dict: Dict[str, Dict[str, Any]],
  ) -> List[str]:
    """Converts a nested dictionary to a list of strings.

    The function processes a dictionary where each value is expected to be
    another dictionary. The format for each inner item is
    "{outer_key}.{inner_key}={inner_value}".

    Args:
      data_dict: The input nested dictionary (e.g., {"mesh": {"shape": "..."}}).

    Returns:
      A list of formatted strings.

    Raises:
      TypeError: If any value in the outer dictionary is not a dictionary.
    """
    result_list = []
    for outer_key, inner_dict in data_dict.items():
      if isinstance(inner_dict, dict):
        for inner_key, inner_value in inner_dict.items():
          # Using f-string for concise string formatting
          result_list.append(f"{outer_key}.{inner_key}={inner_value}")
      else:
        raise TypeError(
            f"Expected a dictionary for key '{outer_key}', but got"
            f" {type(inner_dict).__name__}"
        )
    return result_list

  def run_test_peft_trainer(self, hp):
    rngs = nnx.Rngs(hp.config["rng_seed"])
    model = tc.ToyTransformer(rngs=rngs)
    peft_trainer.PeftTrainer(model, hp.optimizer, hp.training_config)

  def test_config_from_yaml(self):
    non_existent_argv = ["", "nonexistent_config.yaml"]
    self.assertRaises(ValueError, config.initialize, non_existent_argv)

    self.initialize_config([])

  def test_override_training_config_simple(self):
    argv = [
        "",
        "base_config.yaml",
        "training_config.max_steps=150",
        "training_config.data_sharding_axis=['fsdp','dp']",
    ]
    hp = config.initialize(argv)
    self.assertEqual(hp.config["training_config"]["max_steps"], 150)
    self.assertEqual(
        hp.config["training_config"]["data_sharding_axis"], ["fsdp", "dp"]
    )
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

  @parameterized.named_parameters(
      dict(
          testcase_name="kaggle_with_ckpt",
          overrides=[
              "model_source=kaggle",
              "intermediate_ckpt_dir=/path/to/ckpt",
          ],
      ),
      dict(
          testcase_name="huggingface_with_ckpt",
          overrides=[
              "model_source=huggingface",
              "intermediate_ckpt_dir=/path/to/ckpt",
          ],
      ),
      dict(
          testcase_name="gcs_ckpt_source",
          overrides=["model_source=gcs"],
      ),
  )
  def test_valid_configs(self, overrides):
    argv = ["", "base_config.yaml"] + overrides
    try:
      config.initialize(argv)
    except ValueError as e:
      self.fail(f"Initialization failed for valid config {overrides}: {e}")

  @parameterized.named_parameters(
      dict(
          testcase_name="kaggle_no_ckpt",
          overrides=["ckpt_source=kaggle", "intermediate_ckpt_dir="],
          expected_error=ValueError,
      ),
      dict(
          testcase_name="huggingface_no_ckpt",
          overrides=["model_source=huggingface", "intermediate_ckpt_dir="],
          expected_error=ValueError,
      ),
      dict(
          testcase_name="invalid_model_source",
          overrides=["model_source=invalid_source"],
          expected_error=ValueError,
      ),
  )
  def test_invalid_configs(self, overrides, expected_error):
    argv = ["", "base_config.yaml"] + overrides
    with self.assertRaises(expected_error):
      config.initialize(argv)

  # --- Tests for create_optimizer ---

  @parameterized.named_parameters(
      dict(
          testcase_name="sgd_simple",
          overrides=[
              "optimizer_config.opt_type=sgd",
              "optimizer_config.learning_rate=0.01",
          ],
          expected_type=optax.GradientTransformation,
      ),
  )
  def test_create_optimizer_valid(self, overrides, expected_type):
    """Tests valid optimizer configurations."""
    hp = self.initialize_config(overrides)
    self.assertIsNotNone(hp.optimizer)
    self.assertIsInstance(hp.optimizer, expected_type)

  @parameterized.named_parameters(
      dict(
          testcase_name="unknown_name",
          overrides=[
              "optimizer_config.opt_type=unknown",
              "optimizer_config.learning_rate=0.01",
          ],
          expected_error=ValueError,
          error_regex="Optimizer unknown not found in dict_keys",
      ),
  )
  def test_create_optimizer_invalid(
      self, overrides, expected_error, error_regex
  ):
    """Tests invalid optimizer configurations."""
    with self.assertRaisesRegex(expected_error, error_regex):
      self.initialize_config(overrides)
      # raw_keys = omegaconf.OmegaConf.from_cli(overrides)
      # hp._create_optimizer(raw_keys)

  # --- Tests for create_mesh ---
  # NOTE: These tests might depend on the available JAX devices.
  # Mocking jax.device_count() and jax.devices() is best for hermetic tests.
  @parameterized.named_parameters(
      dict(
          testcase_name="valid_1d",
          raw_keys={"mesh": {"shape": "(4,)", "axis_names": "('data',)"}},
          mock_num_devices=4,
          expected=((4,), ("data",)),
      ),
      dict(
          testcase_name="valid_2d",
          raw_keys={
              "mesh": {"shape": "(2, 4)", "axis_names": "('data', 'model')"}
          },
          mock_num_devices=8,
          expected=((2, 4), ("data", "model")),
      ),
      dict(
          testcase_name="devices_equal_prod",
          raw_keys={"mesh": {"shape": "(8,)", "axis_names": "('a',)"}},
          mock_num_devices=8,
          expected=((8,), ("a",)),
      ),
      dict(
          testcase_name="devices_more_than_prod",
          raw_keys={"mesh": {"shape": "(2, 2)", "axis_names": "('x', 'y')"}},
          mock_num_devices=5,
          expected=((2, 2), ("x", "y")),
      ),
  )
  @mock.patch("jax.device_count")
  def test_create_mesh_valid(
      self, mock_device_count_fn, raw_keys, mock_num_devices, expected
  ):
    mock_device_count_fn.return_value = mock_num_devices
    hp = self.initialize_config(self.convert_nested_dict_to_list(raw_keys))
    self.assertEqual(hp.mesh, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="shape_invalid_literal",
          raw_keys={"mesh": {"shape": "(1,a)", "axis_names": "('data',)"}},
          mock_num_devices=4,
          error_regex="Invalid 'shape' key in 'mesh' configuration",
      ),
      dict(
          testcase_name="shape_not_tuple",
          raw_keys={"mesh": {"shape": "1", "axis_names": "('data',)"}},
          mock_num_devices=4,
          error_regex="Invalid 'shape' key in 'mesh' configuration",
      ),
      dict(
          testcase_name="shape_not_int",
          raw_keys={"mesh": {"shape": "(1, '2')", "axis_names": "('a', 'b')"}},
          mock_num_devices=4,
          error_regex="All elements in mesh.shape must be integers",
      ),
      dict(
          testcase_name="axis_names_not_tuple",
          raw_keys={"mesh": {"shape": "(1,)", "axis_names": "'data'"}},
          mock_num_devices=4,
          error_regex="Invalid 'axis_names' key in 'mesh' configuration",
      ),
      dict(
          testcase_name="axis_names_not_str",
          raw_keys={"mesh": {"shape": "(1,)", "axis_names": "(1,)"}},
          mock_num_devices=4,
          error_regex="All elements in mesh.axis_names must be strings",
      ),
      dict(
          testcase_name="length_mismatch",
          raw_keys={"mesh": {"shape": "(1, 2)", "axis_names": "('data',)"}},
          mock_num_devices=4,
          error_regex="must have the same length",
      ),
      dict(
          testcase_name="too_many_devices_required",
          raw_keys={"mesh": {"shape": "(2, 3)", "axis_names": "('a', 'b')"}},
          mock_num_devices=5,
          error_regex="requires 6 devices, but found 5",
      ),
  )
  @mock.patch("jax.device_count")
  def test_create_mesh_invalid(
      self,
      mock_device_count_fn,
      raw_keys,
      mock_num_devices,
      error_regex,
  ):
    mock_device_count_fn.return_value = mock_num_devices
    with self.assertRaisesRegex(ValueError, error_regex):
      self.initialize_config(self.convert_nested_dict_to_list(raw_keys))


if __name__ == "__main__":
  absltest.main()
