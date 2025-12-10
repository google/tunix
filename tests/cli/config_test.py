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
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, cast
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import optax
from tunix.cli import config
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

  def convert_nested_dict_to_list(self, data_dict: Dict[str, Any]) -> List[str]:
    """Converts a potentially deeply nested dictionary to a list of strings.

    Each string in the list represents a path from the root to a non-dictionary
    value, formatted as "key1.key2.keyN=value".

    Args:
      data_dict: The input nested dictionary.

    Returns:
      A list of formatted strings representing the paths and values.
    """
    result_list = []
    self._flatten_dict_recursive(data_dict, [], result_list)
    return result_list

  def _flatten_dict_recursive(
      self,
      current_dict: Dict[str, Any],
      current_path: List[str],
      result_list: List[str],
  ) -> None:
    """Helper function to recursively traverse the dictionary.

    Args:
      current_dict: The dictionary node currently being processed.
      current_path: A list of keys representing the path from the root to the
        current_dict.
      result_list: The list to accumulate the formatted strings.
    """
    for key, value in current_dict.items():
      new_path = current_path + [str(key)]  # Ensure key is string
      if isinstance(value, dict):
        self._flatten_dict_recursive(value, new_path, result_list)
      else:
        path_str = ".".join(new_path)
        result_list.append(f"{path_str}={value}")

  def run_test_peft_trainer(self, hp):
    """Helper to run a test PeftTrainer."""
    config_dict = cast(Dict[str, Any], hp.config)
    rngs = nnx.Rngs(config_dict["model_config"]["rng_seed"])
    model = tc.ToyTransformer(
        config=tc.ModelConfig(),
        rngs=rngs,
    )
    optimizer = hp.create_optimizer("optimizer_config")
    training_config = peft_trainer.TrainingConfig(
        **hp.obtain_training_config_dict("training_config")
    )

    peft_trainer.PeftTrainer(model, optimizer, training_config)

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
        "training_config.eval_every_n_steps=10",
    ]
    hp = config.initialize(argv)

    config_dict = cast(Dict[str, Any], hp.config)

    self.assertEqual(config_dict["training_config"]["max_steps"], 150)
    self.assertEqual(
        config_dict["training_config"]["data_sharding_axis"], ["fsdp", "dp"]
    )
    self.run_test_peft_trainer(hp)

  def test_override_training_config_complex(self):
    argv = [
        "",
        "base_config.yaml",
        "training_config.profiler_options.log_dir=/tmp/profiler_log_dir",
        "training_config.profiler_options.skip_first_n_steps=1",
        "training_config.profiler_options.profiler_steps=5",
        "training_config.eval_every_n_steps=10",
    ]
    self.run_test_peft_trainer(config.initialize(argv))

  @parameterized.named_parameters(
      dict(
          testcase_name="kaggle_with_ckpt",
          overrides=[
              "model_name=gemma-2b",
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
          overrides=["model_name=gemma3-1b", "model_source=gcs"],
      ),
  )
  def test_valid_configs(self, overrides):
    prefix = "model_config."
    overrides = [f"{prefix}{item}" for item in overrides]
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
    optimizer = hp.create_optimizer("optimizer_config")
    self.assertIsNotNone(optimizer)
    self.assertIsInstance(optimizer, expected_type)

  @parameterized.named_parameters(
      dict(
          testcase_name="unknown_name",
          overrides=[
              "optimizer_config.opt_type=unknown",
              "optimizer_config.learning_rate=0.01",
          ],
          expected_error=AttributeError,
          error_regex="module 'optax' has no attribute 'unknown'",
      ),
  )
  def test_create_optimizer_invalid(
      self, overrides, expected_error, error_regex
  ):
    """Tests invalid optimizer configurations."""
    with self.assertRaisesRegex(expected_error, error_regex):
      hp = self.initialize_config(overrides)
      hp.create_optimizer("optimizer_config")

  #  --- Tests for learning_rate_schedule ---
  @parameterized.named_parameters(
      dict(
          testcase_name="constant_lr",
          overrides=[
              "optimizer_config.schedule_type=constant_schedule",
              "optimizer_config.value=1e-5",
          ],
      ),
      dict(
          testcase_name="exponential_decay",
          overrides=[
              "optimizer_config.schedule_type=exponential_decay",
              "optimizer_config.init_value=0.01",
              "optimizer_config.transition_steps=10",
              "optimizer_config.decay_rate=0.95",
          ],
      ),
      dict(
          testcase_name="warmup_cosine_decay",
          overrides=[
              "optimizer_config.schedule_type=warmup_cosine_decay_schedule",
              "optimizer_config.init_value=0.0",
              "optimizer_config.peak_value=0.001",
              "optimizer_config.warmup_steps=100",
              "optimizer_config.decay_steps=1000",
              "optimizer_config.end_value=0.0001",
              "optimizer_config.alpha=0.5",
          ],
      ),
  )
  def test_learning_rate_schedule_valid(self, overrides):
    hp = self.initialize_config(overrides)
    lr_schedule = hp._create_learning_rate(
        hp.config["optimizer_config"], "test_config_path"
    )
    self.assertIsNotNone(lr_schedule)
    self.assertTrue(callable(lr_schedule), "lr_schedule should be callable")

  # --- Tests for create_mesh ---
  @parameterized.named_parameters(
      dict(
          testcase_name="valid_1d",
          raw_keys={
              "model_config": {
                  "mesh": {"shape": "(4,)", "axis_names": "('data',)"}
              }
          },
          mock_num_devices=4,
          expected=((4,), ("data",)),
      ),
      dict(
          testcase_name="valid_2d",
          raw_keys={
              "model_config": {
                  "mesh": {"shape": "(2, 4)", "axis_names": "('data', 'model')"}
              }
          },
          mock_num_devices=8,
          expected=((2, 4), ("data", "model")),
      ),
      dict(
          testcase_name="devices_equal_prod",
          raw_keys={
              "model_config": {
                  "mesh": {"shape": "(8,)", "axis_names": "('a',)"}
              }
          },
          mock_num_devices=8,
          expected=((8,), ("a",)),
      ),
      dict(
          testcase_name="devices_more_than_prod",
          raw_keys={
              "model_config": {
                  "mesh": {"shape": "(2, 2)", "axis_names": "('x', 'y')"}
              }
          },
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
    mesh = hp.create_mesh("model_config")
    self.assertEqual(
        mesh,
        jax.make_mesh(
            expected[0],
            expected[1],
            axis_types=(jax.sharding.AxisType.Auto,) * len(expected[1]),
        ),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="shape_invalid_literal",
          raw_keys={
              "model_config": {
                  "mesh": {"shape": "(1,a)", "axis_names": "('data',)"}
              }
          },
          mock_num_devices=4,
          error_regex="Invalid 'shape' key in 'mesh' configuration",
      ),
      dict(
          testcase_name="shape_not_tuple",
          raw_keys={
              "model_config": {
                  "mesh": {"shape": "1", "axis_names": "('data',)"}
              }
          },
          mock_num_devices=4,
          error_regex="Invalid 'shape' key in 'mesh' configuration",
      ),
      dict(
          testcase_name="shape_not_int",
          raw_keys={
              "model_config": {
                  "mesh": {"shape": "(1, '2')", "axis_names": "('a', 'b')"}
              }
          },
          mock_num_devices=4,
          error_regex="All elements in mesh.shape must be integers",
      ),
      dict(
          testcase_name="axis_names_not_tuple",
          raw_keys={
              "model_config": {
                  "mesh": {"shape": "(1,)", "axis_names": "'data'"}
              }
          },
          mock_num_devices=4,
          error_regex="Invalid 'axis_names' key in 'mesh' configuration",
      ),
      dict(
          testcase_name="axis_names_not_str",
          raw_keys={
              "model_config": {"mesh": {"shape": "(1,)", "axis_names": "(1,)"}}
          },
          mock_num_devices=4,
          error_regex="All elements in mesh.axis_names must be strings",
      ),
      dict(
          testcase_name="length_mismatch",
          raw_keys={
              "model_config": {
                  "mesh": {"shape": "(1, 2)", "axis_names": "('data',)"}
              }
          },
          mock_num_devices=4,
          error_regex="must have the same length",
      ),
      dict(
          testcase_name="too_many_devices_required",
          raw_keys={
              "model_config": {
                  "mesh": {"shape": "(2, 3)", "axis_names": "('a', 'b')"}
              }
          },
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
      nested_dict = self.convert_nested_dict_to_list(raw_keys)
      hp = self.initialize_config(nested_dict)
      hp.create_mesh("model_config")

  @parameterized.named_parameters(
      dict(
          testcase_name="reward_fn_from_module",
          overrides=[
              "reward_functions=['tunix/cli/reward_fn/gsm8k.py']",
              "verl_compatible=False",
          ],
          expected_reward_fn_len=4,
          expected_reward_fn_names=[
              "match_format_exactly",
              "match_format_approximately",
              "check_answer",
              "check_numbers",
          ],
      ),
      dict(
          testcase_name="reward_fn_from_module_verl_compatible",
          overrides=[
              "reward_functions=['tunix/cli/reward_fn/gsm8k_verl.py']",
              "verl_compatible=True",
          ],
          expected_reward_fn_len=1,
          expected_reward_fn_names=[
              "reward_fn",
          ],
      ),
  )
  def test_get_reward_fns(
      self, overrides, expected_reward_fn_len, expected_reward_fn_names
  ):
    hp = self.initialize_config(overrides)
    reward_fns = hp.obtain_reward_fn()
    self.assertLen(reward_fns, expected_reward_fn_len)
    actual_names = [fn.__name__ for fn in reward_fns]
    self.assertCountEqual(actual_names, expected_reward_fn_names)

  def test_obtain_reward_fn_relative_path(self):
    hp = self.initialize_config([])

    with tempfile.TemporaryDirectory() as tmp_dir_str:
      root = Path(tmp_dir_str)

    reward_dir = root / "tunix" / "cli" / "reward_fn"
    reward_dir.mkdir(parents=True)
    dummy_file = reward_dir / "dummy.py"

    dummy_file.write_text("def reward_fn(x): return x")

    run_dir = root / "examples" / "rl" / "grpo"
    run_dir.mkdir(parents=True)

    hp.config["reward_functions"] = ["tunix/cli/reward_fn/dummy.py"]

    with mock.patch("tunix.cli.config.get_project_root", return_value=root):
      original_cwd = os.getcwd()
      try:
        os.chdir(run_dir)
        reward_fns = hp.obtain_reward_fn()
        self.assertLen(reward_fns, 1)
        self.assertEqual(reward_fns[0].__name__, "reward_fn")

      finally:
        os.chdir(original_cwd)

  def test_obtain_reward_fn_file_not_found(self):
    hp = self.initialize_config(
        ["reward_functions=['tunix/cli/reward_fn/non_existent.py']"]
    )
    with self.assertRaisesRegex(
        ImportError, "Failed to execute module non_existent"
    ):
      hp.obtain_reward_fn()


if __name__ == "__main__":
  if "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = "TestToken"
  absltest.main()
