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
import pathlib
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
from tunix.utils import env_utils
from tunix.utils import mesh as mesh_lib

_BASE_YAML = pathlib.Path(config.__file__).parent / "base_config.yaml"


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
    with tempfile.TemporaryDirectory() as log_dir:
      argv = [
          "",
          "base_config.yaml",
          f"training_config.profiler_options.log_dir={log_dir}",
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
          overrides=["model_name=gemma-3-1b-pt", "model_source=gcs"],
      ),
      dict(
          testcase_name="maxtext_source",
          overrides=["model_name=qwen2.5-0.5b", "model_source=maxtext"],
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
          expected_error=ValueError,
          error_regex="Optimizer type 'unknown' not supported",
      ),
  )
  def test_create_optimizer_invalid(
      self, overrides, expected_error, error_regex
  ):
    """Tests invalid optimizer configurations."""
    with self.assertRaisesRegex(expected_error, error_regex):
      hp = self.initialize_config(overrides)
      hp.create_optimizer("optimizer_config")

  def test_create_optimizer_prefers_schedule_over_scalar_learning_rate(self):
    """A configured schedule wins over a scalar `learning_rate` in the block."""
    hp = self.initialize_config([
        "optimizer_config.opt_type=adamw",
        "optimizer_config.learning_rate=1e-5",
        "optimizer_config.schedule_type=warmup_cosine_decay_schedule",
        "optimizer_config.init_value=0.0",
        "optimizer_config.peak_value=3e-5",
        "optimizer_config.end_value=0.0",
        "optimizer_config.warmup_steps=5",
        "optimizer_config.decay_steps=10",
    ])
    optimizer = hp.create_optimizer("optimizer_config")

    state = optimizer.init({"w": jax.numpy.zeros((2,))})
    # `inject_hyperparams` resolves the schedule at the current step, so step 0
    # reports the schedule's `init_value` rather than the scalar learning rate.
    self.assertEqual(float(state.hyperparams["learning_rate"]), 0.0)

  def test_create_optimizer_builds_non_optimizer_transformation(self):
    """`opt_type` resolves any optax factory, not just optimizers.

    This is what lets callers build gradient clipping with the same helper and
    `optax.chain` it around an optimizer, instead of a bespoke code path.
    """
    clipping = config.create_optimizer(
        {"opt_type": "clip_by_global_norm", "max_norm": 1.0},
        "test_config_path",
    )

    params = {"w": jax.numpy.zeros((2,))}
    # A gradient with global norm 100, far above the configured limit of 1.0.
    grads = {"w": jax.numpy.array([60.0, 80.0])}
    updates, _ = clipping.update(grads, clipping.init(params), params)
    self.assertAlmostEqual(float(optax.global_norm(updates)), 1.0, places=4)

  def test_create_optimizer_reports_missing_transformation_kwarg(self):
    # `max_norm` has no default, so omitting it must be reported rather than
    # falling through to a bare TypeError.
    with self.assertRaisesRegex(ValueError, "max_norm"):
      config.create_optimizer(
          {"opt_type": "clip_by_global_norm"}, "test_config_path"
      )

  def test_create_optimizer_chains_opt_chain_type(self):
    """`opt_chain_type` is built from `chain_kwargs` and chained first."""
    optimizer = config.create_optimizer(
        {
            "opt_type": "sgd",
            "learning_rate": 1.0,
            "opt_chain_type": "clip_by_global_norm",
            "chain_kwargs": {"max_norm": 1.0},
        },
        "test_config_path",
    )

    params = {"w": jax.numpy.zeros((2,))}
    # A gradient with global norm 100, far above the configured limit of 1.0.
    grads = {"w": jax.numpy.array([60.0, 80.0])}
    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    # sgd(learning_rate=1.0) negates the clipped gradient, so the update norm
    # equals the clipped gradient norm.
    self.assertAlmostEqual(float(optax.global_norm(updates)), 1.0, places=4)

  def test_create_optimizer_gradient_transformation_bfloat16(self):
    """Verifies that plain transformations created by create_optimizer preserve bfloat16."""
    clipping = config.create_optimizer(
        {"opt_type": "clip_by_global_norm", "max_norm": 1.0},
        "test_config_path",
    )
    params = {"w": jax.numpy.zeros((2,), dtype=jax.numpy.bfloat16)}
    grads = {"w": jax.numpy.array([60.0, 80.0], dtype=jax.numpy.bfloat16)}
    updates, _ = clipping.update(grads, clipping.init(params), params)
    self.assertAlmostEqual(float(optax.global_norm(updates)), 1.0, places=2)
    self.assertEqual(optax.tree.dtype(updates), jax.numpy.bfloat16)

  def test_create_optimizer_chains_opt_chain_type_bfloat16(self):
    """Verifies that chained transformations work with bfloat16 gradients."""
    optimizer = config.create_optimizer(
        {
            "opt_type": "sgd",
            "learning_rate": 1.0,
            "opt_chain_type": "clip_by_global_norm",
            "chain_kwargs": {"max_norm": 1.0},
        },
        "test_config_path",
    )
    params = {"w": jax.numpy.zeros((2,), dtype=jax.numpy.bfloat16)}
    grads = {"w": jax.numpy.array([60.0, 80.0], dtype=jax.numpy.bfloat16)}
    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    self.assertAlmostEqual(float(optax.global_norm(updates)), 1.0, places=2)

  def test_create_optimizer_without_opt_chain_type_is_unchained(self):
    """`opt_chain_type` defaults to None, which leaves gradients untouched."""
    optimizer = config.create_optimizer(
        {"opt_type": "sgd", "learning_rate": 1.0}, "test_config_path"
    )

    params = {"w": jax.numpy.zeros((2,))}
    grads = {"w": jax.numpy.array([60.0, 80.0])}
    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    self.assertAlmostEqual(float(optax.global_norm(updates)), 100.0, places=4)

  def test_create_optimizer_rejects_unknown_opt_chain_type(self):
    with self.assertRaisesRegex(ValueError, "not_a_transformation"):
      config.create_optimizer(
          {
              "opt_type": "sgd",
              "learning_rate": 1.0,
              "opt_chain_type": "not_a_transformation",
          },
          "test_config_path",
      )

  def test_create_optimizer_reports_missing_chain_kwarg(self):
    # `max_norm` has no default, so omitting `chain_kwargs` must be reported.
    with self.assertRaisesRegex(ValueError, "max_norm"):
      config.create_optimizer(
          {
              "opt_type": "sgd",
              "learning_rate": 1.0,
              "opt_chain_type": "clip_by_global_norm",
          },
          "test_config_path",
      )

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
    lr_schedule = config.create_learning_rate(
        hp.config["optimizer_config"], "test_config_path"
    )
    self.assertIsNotNone(lr_schedule)
    self.assertTrue(callable(lr_schedule), "lr_schedule should be callable")

    # --- Tests for mesh config parsing and mesh creation ---
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
  @mock.patch.object(jax, "device_count")
  def test_create_mesh_valid(
      self, mock_device_count_fn, raw_keys, mock_num_devices, expected
  ):
    mock_device_count_fn.return_value = mock_num_devices
    hp = self.initialize_config(self.convert_nested_dict_to_list(raw_keys))
    axis_shapes, axis_names = hp.parse_mesh_config("model_config")
    expected_mesh = object()

    with mock.patch.object(
        jax, "make_mesh", return_value=expected_mesh
    ) as make_mesh_mock:
      mesh = mesh_lib.create_mesh(axis_shapes, axis_names)

    make_mesh_mock.assert_called_once_with(
        expected[0],
        expected[1],
        axis_types=(jax.sharding.AxisType.Auto,) * len(expected[1]),
    )
    self.assertIs(mesh, expected_mesh)

  def test_create_mesh_with_assigned_devices(self):
    raw_keys = {
        "model_config": {
            "mesh": {"shape": "(2, 2)", "axis_names": "('x', 'y')"}
        }
    }
    hp = self.initialize_config(self.convert_nested_dict_to_list(raw_keys))
    axis_shapes, axis_names = hp.parse_mesh_config("model_config")
    assigned_devices = ["d0", "d1", "d2", "d3"]

    class FakeMesh:

      def __init__(self, devices, axis_names, axis_types=None):
        self.devices = devices
        self.axis_names = axis_names
        self.axis_types = axis_types

    with mock.patch.object(jax.sharding, "Mesh", side_effect=FakeMesh):
      mesh = mesh_lib.create_mesh(
          axis_shapes,
          axis_names,
          devices=assigned_devices,
      )

    self.assertEqual(mesh.devices.shape, (2, 2))
    self.assertSequenceEqual(mesh.devices.flatten().tolist(), assigned_devices)
    self.assertEqual(mesh.axis_names, ("x", "y"))

  def test_parse_mesh_allocation_policy_defaults_to_compact(self):
    raw_keys = {
        "model_config": {
            "mesh": {"shape": "(2, 2)", "axis_names": "('x', 'y')"}
        }
    }
    hp = self.initialize_config(self.convert_nested_dict_to_list(raw_keys))

    self.assertEqual(
        hp._parse_mesh_allocation_policy("model_config"),
        mesh_lib.normalize_allocation_policy(None),
    )

  def test_parse_mesh_allocation_policy_validates_explicit_value(self):
    raw_keys = {
        "model_config": {
            "mesh": {
                "shape": "(2, 2)",
                "axis_names": "('x', 'y')",
                "allocation_policy": "performance",
            }
        }
    }
    hp = self.initialize_config(self.convert_nested_dict_to_list(raw_keys))

    self.assertEqual(
        hp._parse_mesh_allocation_policy("model_config"),
        "PERFORMANCE",
    )

  def test_parse_mesh_allocation_policy_rejects_invalid_value(self):
    raw_keys = {
        "model_config": {
            "mesh": {
                "shape": "(2, 2)",
                "axis_names": "('x', 'y')",
                "allocation_policy": "fastest",
            }
        }
    }
    hp = self.initialize_config(self.convert_nested_dict_to_list(raw_keys))

    with self.assertRaisesRegex(ValueError, "allocation_policy must be one of"):
      hp._parse_mesh_allocation_policy("model_config")

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
  @mock.patch.object(jax, "device_count")
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
      axis_shapes, axis_names = hp.parse_mesh_config("model_config")
      mesh_lib.create_mesh(axis_shapes, axis_names)

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
      root = pathlib.Path(tmp_dir_str)

    reward_dir = root / "tunix" / "cli" / "reward_fn"
    reward_dir.mkdir(parents=True)
    dummy_file = reward_dir / "dummy.py"

    dummy_file.write_text("def reward_fn(x): return x")

    run_dir = root / "examples" / "rl" / "grpo"
    run_dir.mkdir(parents=True)

    hp.config["reward_functions"] = ["tunix/cli/reward_fn/dummy.py"]

    with mock.patch.object(config, "get_project_root", return_value=root):
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

  def test_obtain_reward_fn_absolute_path_outside_project(self):
    """Tests loading a reward function from an absolute FILE path outside the project root."""
    hp = self.initialize_config([])
    with tempfile.TemporaryDirectory() as tmp_dir_str:
      external_root = pathlib.Path(tmp_dir_str) / "some_other_project"
      external_root.mkdir()
      external_module_file = external_root / "external_reward.py"
      module_content = "def external_reward_func(val): return val * 10"
      external_module_file.write_text(module_content)
      abs_path = str(external_module_file.resolve())
      hp.config["reward_functions"] = [abs_path]
      hp.config["verl_compatible"] = False

      with mock.patch.object(
          config,
          "get_project_root",
          return_value=pathlib.Path("/irrelevant/tunix/project"),
      ):
        reward_fns = hp.obtain_reward_fn()
        self.assertLen(reward_fns, 1)
      self.assertEqual(reward_fns[0].__name__, "external_reward_func")
      self.assertEqual(reward_fns[0](5), 50)

  def test_model_config_inheritance(self):
    # Test that actor_model_config inherits from model_config
    # Scenario 1: Override model_config.model_name, actor should inherit
    hp = self.initialize_config([
        "model_config.model_name=gemma-2-2b-it",
        "model_config.model_source=kaggle",
    ])
    self.assertEqual(hp.config["model_config"]["model_name"], "gemma-2-2b-it")
    self.assertEqual(
        hp.config["actor_model_config"]["model_name"], "gemma-2-2b-it"
    )

    # Scenario 2: Override actor, reference and rollout model name to different
    # value, it should keep override
    hp2 = self.initialize_config([
        "actor_model_config.model_name=gemma-2-2b-it",
        "actor_model_config.model_source=kaggle",
        "reference_model_config.model_name=gemma-3-4b",
        "reference_model_config.model_source=gcs",
        "rollout_model_config.model_name=gemma-3-1b-it",
        "rollout_model_config.model_source=gcs",
    ])
    self.assertEqual(hp2.config["model_config"]["model_name"], "llama-3.1-8b")
    self.assertEqual(
        hp2.config["actor_model_config"]["model_name"], "gemma-2-2b-it"
    )
    self.assertEqual(
        hp2.config["reference_model_config"]["model_name"], "gemma-3-4b"
    )
    self.assertEqual(
        hp2.config["rollout_model_config"]["model_name"], "gemma-3-1b-it"
    )

    # Scenario 3: Override actor_model_config and reference_model_config have
    # higher priority than model_config, it should keep override
    hp3 = self.initialize_config([
        "model_config.model_name=gemma-3-12b",
        "model_config.model_source=gcs",
        "actor_model_config.model_name=gemma-2-2b-it",
        "actor_model_config.model_source=kaggle",
        "reference_model_config.model_name=gemma-3-4b",
        "reference_model_config.model_source=gcs",
    ])
    self.assertEqual(hp3.config["model_config"]["model_name"], "gemma-3-12b")
    self.assertEqual(
        hp3.config["actor_model_config"]["model_name"], "gemma-2-2b-it"
    )
    self.assertEqual(
        hp3.config["reference_model_config"]["model_name"], "gemma-3-4b"
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="grpo_enabled",
          main_command="grpo_main",
          overrides=[
              "training_config.perf_metrics_options.log_dir={log_dir}",
          ],
      ),
      dict(
          testcase_name="peft_enabled_fails",
          main_command="peft_main",
          overrides=[
              "training_config.perf_metrics_options.log_dir={log_dir}",
          ],
          expected_error=ValueError,
          error_regex=(
              "Perf metrics are currently only supported for GRPO training"
          ),
      ),
      dict(
          testcase_name="peft_disabled",
          main_command="peft_main",
          overrides=[],
      ),
      dict(
          testcase_name="invalid_custom_export_fn_path",
          main_command="grpo_main",
          overrides=[
              "training_config.perf_metrics_options.custom_export_fn_path=invalid.path",
          ],
          expected_error=ValueError,
          error_regex="Could not load custom export function from invalid.path",
      ),
      dict(
          testcase_name="invalid_custom_export_fn_path_v2",
          main_command="grpo_main",
          overrides=[
              "training_config.perf_metrics_options.custom_export_fn_path_v2=invalid.path",
          ],
          expected_error=ValueError,
          error_regex=(
              "Could not load custom export function v2 from invalid.path"
          ),
      ),
  )
  def test_perf_metrics_validation(
      self,
      main_command,
      overrides,
      expected_error=None,
      error_regex=None,
  ):
    with tempfile.TemporaryDirectory() as log_dir:
      overrides = [o.format(log_dir=log_dir) for o in overrides]
      argv = [main_command, "base_config.yaml"] + overrides
      if expected_error:
        with self.assertRaisesRegex(expected_error, error_regex):  # pyrefly: ignore[bad-argument-type]
          config.initialize(argv)
      else:
        config.initialize(argv)

  def test_dict_to_cli_args_with_none(self):
    d = {"a": 1, "b": None, "c": {"d": None, "e": 2}}
    expected = ["a=1", "b=null", "c.d=null", "c.e=2"]
    got = list(config._dict_to_cli_args(d))
    self.assertEqual(expected, got)

  def test_obtain_training_config_dict_checkpointing(self):
    hp = self.initialize_config([])

    # Valid options
    hp.config["training_config"] = {
        "checkpointing_options": {
            "save_interval_steps": 10,
            "max_to_keep": 5,
            "enable_async_checkpointing": True,
            "timeout_secs": 60,
        }
    }
    result = hp.obtain_training_config_dict("training_config")
    self.assertIn("checkpointing_options", result)
    opts = result["checkpointing_options"]
    self.assertEqual(opts.enable_async_checkpointing, True)
    self.assertEqual(opts.async_options.timeout_secs, 60)
    self.assertIsNotNone(opts.save_decision_policy)
    self.assertIsNotNone(opts.preservation_policy)

  def test_obtain_training_config_dict_checkpointing_invalid_options(self):
    hp = self.initialize_config([])
    hp.config["training_config"] = {
        "checkpointing_options": "not a dict"
    }
    with self.assertRaisesRegex(
        ValueError, "Expected dictionary for checkpointing_options"
    ):
      hp.obtain_training_config_dict("training_config")

    hp.config["training_config"] = {
        "checkpointing_options": {
            "invalid_key": 10,
        }
    }
    with self.assertRaisesRegex(ValueError, "Invalid checkpointing options"):
      hp.obtain_training_config_dict("training_config")

    hp.config["training_config"] = {
        "checkpointing_options": ["save_interval_steps"]
    }
    with self.assertRaisesRegex(
        ValueError, "Expected dictionary for checkpointing_options"
    ):
      hp.obtain_training_config_dict("training_config")

  def _initialize_with_flags(
      self,
      overrides: list[str],
      *,
      merge_replace_keys: bool = True,
      validate_nested: bool = True,
  ) -> config.HyperParameters:
    """Initializes from the default base YAML with the opt-in flags."""
    return config.initialize(
        ["resolve_config", *overrides],
        default_base_config=_BASE_YAML,
        merge_replace_keys=merge_replace_keys,
        validate_nested=validate_nested,
    )

  def test_default_base_config_and_normalize_argv(self):
    hp = config.initialize(
        [
            "resolve_config",
            "model_config.model_name=qwen2.5-0.5b",
            "model_config.model_source=maxtext",
        ],
        default_base_config=_BASE_YAML,
    )
    self.assertEqual(
        hp.config_mapping("model_config")["model_name"], "qwen2.5-0.5b"
    )
    self.assertEqual(
        hp.config_mapping("actor_model_config")["model_name"], "qwen2.5-0.5b"
    )

  @parameterized.named_parameters(
      dict(testcase_name="flags_off", flags_on=False),
      dict(testcase_name="flags_on", flags_on=True),
  )
  def test_default_base_config_matches_explicit_base(self, flags_on: bool):
    ov = self.create_tempdir().create_file(
        "ov.yaml", content="rollout_config:\n  temperature: 0.5\n"
    )
    overrides = [
        f"override_config_file={ov.full_path}",
        "model_config.model_name=qwen2.5-0.5b",
        "model_config.model_source=maxtext",
    ]
    explicit = config.initialize(
        ["resolve_config", str(_BASE_YAML), *overrides],
        merge_replace_keys=flags_on,
        validate_nested=flags_on,
    )
    default = config.initialize(
        ["resolve_config", *overrides],
        default_base_config=_BASE_YAML,
        merge_replace_keys=flags_on,
        validate_nested=flags_on,
    )
    self.assertEqual(explicit.config, default.config)

  def test_merge_replace_keys_and_multiple_override_files(self):
    tmp_dir = self.create_tempdir()
    ov1 = tmp_dir.create_file(
        "ov1.yaml", content="rl_training_config:\n  max_steps: 25\n"
    )
    ov2 = tmp_dir.create_file(
        "ov2.yaml", content="rl_training_config:\n  eval_every_n_steps: 5\n"
    )
    hp = self._initialize_with_flags([
        f"override_config_file={ov1.full_path}",
        f"override_config_file={ov2.full_path}",
    ])
    rl_train = hp.config_mapping("rl_training_config")
    self.assertEqual(rl_train["max_steps"], 25)
    self.assertEqual(rl_train["eval_every_n_steps"], 5)
    self.assertEqual(rl_train["actor_optimizer_config"]["opt_type"], "adamw")

  @parameterized.named_parameters(
      dict(testcase_name="files_only", cli=[], expected=12),
      dict(
          testcase_name="files_then_cli",
          cli=["rl_training_config.max_steps=13"],
          expected=13,
      ),
  )
  def test_merge_mode_later_overrides_win(self, cli: list[str], expected: int):
    tmp_dir = self.create_tempdir()
    ov1 = tmp_dir.create_file(
        "ov1.yaml",
        content=(
            "rl_training_config:\n  max_steps: 11\n  eval_every_n_steps: 3\n"
        ),
    )
    ov2 = tmp_dir.create_file(
        "ov2.yaml", content="rl_training_config:\n  max_steps: 12\n"
    )
    hp = self._initialize_with_flags([
        f"override_config_file={ov1.full_path}",
        f"override_config_file={ov2.full_path}",
        *cli,
    ])
    rl_train = hp.config_mapping("rl_training_config")
    self.assertEqual(rl_train["max_steps"], expected)
    self.assertEqual(rl_train["eval_every_n_steps"], 3)

  def test_merge_mode_keeps_siblings_of_overridden_key(self):
    hp = self._initialize_with_flags(
        ["rl_training_config.actor_optimizer_config.b1=0.5"]
    )
    rl_train = hp.config_mapping("rl_training_config")
    self.assertEqual(rl_train["max_steps"], 10)
    optimizer = rl_train["actor_optimizer_config"]
    self.assertEqual(optimizer["b1"], 0.5)
    self.assertEqual(optimizer["b2"], 0.99)
    self.assertEqual(optimizer["opt_type"], "adamw")

  def test_default_path_replaces_section_of_overridden_key(self):
    hp = self.initialize_config(
        ["rl_training_config.actor_optimizer_config.b1=0.5"]
    )
    self.assertEqual(
        hp.config_mapping("rl_training_config"),
        {"actor_optimizer_config": {"b1": 0.5}},
    )

  def test_validate_nested_rejects_unknown_nested_key(self):
    with self.assertRaisesRegex(
        ValueError,
        "Unknown configuration key 'rollout_config.unknown_nested_key'",
    ):
      self._initialize_with_flags(["rollout_config.unknown_nested_key=1"])

  def test_validate_nested_rejects_unknown_root_key(self):
    with self.assertRaisesRegex(
        ValueError, "Unknown configuration key 'not_a_section'"
    ):
      self._initialize_with_flags(["not_a_section.key=1"])

  def test_validate_nested_rejects_unknown_key_in_override_file(self):
    ov = self.create_tempdir().create_file(
        "ov.yaml", content="rollout_config:\n  not_a_real_key: 1\n"
    )
    with self.assertRaisesRegex(
        ValueError, "Unknown configuration key 'rollout_config.not_a_real_key'"
    ):
      self._initialize_with_flags(
          [f"override_config_file={ov.full_path}"], merge_replace_keys=False
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="float_leaf",
          override="rollout_config.temperature.value=0.7",
          dotpath="rollout_config.temperature",
      ),
      dict(
          testcase_name="string_leaf",
          override="model_config.model_name.foo=1",
          dotpath="model_config.model_name",
      ),
  )
  def test_validate_nested_rejects_keys_under_scalar(
      self, override: str, dotpath: str
  ):
    with self.assertRaisesRegex(
        ValueError, f"'{dotpath}' is a .* in the base config"
    ):
      self._initialize_with_flags([override])

  def test_validate_nested_accepts_open_and_empty_sections(self):
    hp = self._initialize_with_flags([
        # `lora_config` is listed in `_OPEN_DICT_KEYS`.
        "model_config.lora_config.rank=8",
        # `vllm_config` is `{}` in the base YAML, which counts as open.
        "vllm_config.some_key=1",
    ])
    self.assertEqual(
        hp.config_mapping("model_config")["lora_config"], {"rank": 8}
    )
    self.assertEqual(hp.config_mapping("vllm_config"), {"some_key": 1})

  def test_validate_nested_accepts_mapping_under_null_entry(self):
    hp = self._initialize_with_flags(
        ["critic_model_config.model_name=qwen2.5-0.5b"]
    )
    self.assertEqual(
        hp.config_mapping("critic_model_config")["model_name"], "qwen2.5-0.5b"
    )

  def test_validate_nested_rejects_bare_token(self):
    with self.assertRaisesRegex(
        ValueError, "Invalid CLI override argument 'oops'"
    ):
      self._initialize_with_flags(["rollout_config.temperature=0.5", "oops"])

  def test_merge_mode_explicit_role_overrides_survive(self):
    hp = self._initialize_with_flags([
        "model_config.model_name=qwen2.5-0.5b",
        "model_config.model_source=maxtext",
        "model_config.model_path=/gs/base_ckpt",
        "actor_model_config.model_name=llama-3.1-8b",
        "actor_model_config.model_source=huggingface",
        'actor_model_config.model_path=""',
    ])
    actor = hp.config_mapping("actor_model_config")
    self.assertEqual(actor["model_name"], "llama-3.1-8b")
    self.assertEqual(actor["model_path"], "")
    rollout = hp.config_mapping("rollout_model_config")
    self.assertEqual(rollout["model_name"], "qwen2.5-0.5b")
    self.assertEqual(rollout["model_path"], "/gs/base_ckpt")

  def test_merge_mode_propagates_nested_sub_keys(self):
    hp = self._initialize_with_flags([
        "model_config.lora_config.rank=8",
        "model_config.mesh.allocation_policy=PERFORMANCE",
        "actor_model_config.lora_config.alpha=2.0",
    ])
    self.assertEqual(
        hp.config_mapping("actor_model_config")["lora_config"],
        {"rank": 8, "alpha": 2.0},
    )
    self.assertEqual(
        hp.config_mapping("rollout_model_config")["lora_config"], {"rank": 8}
    )
    self.assertEqual(
        hp.config_mapping("rollout_model_config")["mesh"]["allocation_policy"],
        "PERFORMANCE",
    )

  @parameterized.named_parameters(
      dict(testcase_name="cli", via_file=False),
      dict(testcase_name="override_file", via_file=True),
  )
  def test_merge_mode_whole_section_role_override_is_kept(self, via_file: bool):
    model_lora = [
        "model_config.lora_config.rank=16",
        "model_config.lora_config.alpha=2.0",
    ]
    if via_file:
      ov = self.create_tempdir().create_file(
          "ov.yaml",
          content=(
              "rollout_model_config:\n  lora_config: {}\n"
              "critic_model_config:\n  lora_config: {}\n"
          ),
      )
      role_overrides = [f"override_config_file={ov.full_path}"]
    else:
      role_overrides = [
          "rollout_model_config.lora_config={}",
          "critic_model_config.lora_config={}",
      ]
    hp = self._initialize_with_flags([*role_overrides, *model_lora])
    self.assertEqual(
        hp.config_mapping("actor_model_config")["lora_config"],
        {"rank": 16, "alpha": 2.0},
    )
    self.assertEqual(
        hp.config_mapping("rollout_model_config")["lora_config"], {}
    )
    # The critic is null in the base YAML; it still inherits the other keys.
    critic = hp.config_mapping("critic_model_config")
    self.assertEqual(critic["lora_config"], {})
    self.assertEqual(critic["model_name"], "llama-3.1-8b")

  def test_merge_mode_explicit_null_role_key_is_kept(self):
    hp = self._initialize_with_flags([
        "model_config.lora_config.rank=16",
        "rollout_model_config.lora_config=null",
    ])
    self.assertIsNone(hp.config_mapping("rollout_model_config")["lora_config"])

  def test_merge_mode_depth_four_role_override_is_kept(self):
    hp = self._initialize_with_flags(
        [
            "model_config.kwargs.rope.factor=4",
            "model_config.kwargs.rope.type=linear",
            "actor_model_config.kwargs.rope.factor=2",
        ],
        validate_nested=False,
    )
    self.assertEqual(
        hp.config_mapping("actor_model_config")["kwargs"],
        {"rope": {"factor": 2}},
    )
    self.assertEqual(
        hp.config_mapping("rollout_model_config")["kwargs"],
        {"rope": {"factor": 4, "type": "linear"}},
    )

  def test_merge_mode_mesh_pair_different_rank(self):
    hp = self._initialize_with_flags([
        'model_config.mesh.shape="(8,)"',
        "model_config.mesh.axis_names=\"('fsdp',)\"",
        'rollout_model_config.mesh.shape="(2,4)"',
    ])
    self.assertEqual(
        hp.parse_mesh_config("rollout_model_config"),
        ((2, 4), ("fsdp", "tp")),
    )
    self.assertEqual(
        hp.parse_mesh_config("actor_model_config"), ((8,), ("fsdp",))
    )

  def test_merge_mode_mesh_pair_same_rank(self):
    hp = self._initialize_with_flags([
        "model_config.mesh.axis_names=\"('dp','tp')\"",
        "model_config.mesh.allocation_policy=PERFORMANCE",
        'actor_model_config.mesh.shape="(4,1)"',
    ])
    self.assertEqual(
        hp.parse_mesh_config("actor_model_config"), ((4, 1), ("fsdp", "tp"))
    )
    self.assertEqual(
        hp.mesh_axes("actor_model_config"),
        {"fsdp": 4, "tp": 1, "expert": 1, "dp": 1},
    )
    # Mesh sub-keys outside the shape/axis_names pair are still inherited.
    self.assertEqual(
        hp.config_mapping("actor_model_config")["mesh"]["allocation_policy"],
        "PERFORMANCE",
    )
    self.assertEqual(
        hp.mesh_axes("rollout_model_config"),
        {"fsdp": 1, "tp": 2, "expert": 1, "dp": 2},
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="absent_axes_are_one",
          shape="(4,)",
          axis_names="('fsdp',)",
          expected={"fsdp": 4, "tp": 1, "expert": 1, "dp": 1},
      ),
      dict(
          testcase_name="all_axes",
          shape="(1,2,3,4)",
          axis_names="('dp','fsdp','tp','expert')",
          expected={"fsdp": 2, "tp": 3, "expert": 4, "dp": 1},
      ),
  )
  def test_mesh_axes(
      self, shape: str, axis_names: str, expected: dict[str, int]
  ):
    hp = self.initialize_config([
        f'model_config.mesh.shape="{shape}"',
        f'model_config.mesh.axis_names="{axis_names}"',
    ])
    self.assertEqual(hp.mesh_axes("model_config"), expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="unknown",
          axis_names="('fsdp','sp')",
          error=r"unknown axis names \['sp'\]",
      ),
      dict(
          testcase_name="duplicate",
          axis_names="('fsdp','fsdp')",
          error=r"repeats \['fsdp'\]",
      ),
  )
  def test_mesh_axes_rejects_invalid_axis_names(
      self, axis_names: str, error: str
  ):
    hp = self.initialize_config([
        'model_config.mesh.shape="(2,4)"',
        f'model_config.mesh.axis_names="{axis_names}"',
    ])
    with self.assertRaisesRegex(ValueError, error):
      hp.mesh_axes("model_config")

  def test_config_mapping_rejects_missing_null_and_scalar_sections(self):
    hp = self.initialize_config([])
    with self.assertRaisesRegex(KeyError, "not_a_section"):
      hp.config_mapping("not_a_section")
    with self.assertRaisesRegex(TypeError, "'critic_model_config' is null"):
      hp.config_mapping("critic_model_config")
    with self.assertRaisesRegex(TypeError, "'batch_size' to be a mapping"):
      hp.config_mapping("batch_size")

  def test_config_mapping_returns_deep_copy(self):
    hp = self.initialize_config([])
    rollout = hp.config_mapping("rollout_model_config")
    rollout["mesh"]["shape"] = "(1,)"
    self.assertEqual(
        hp.config_mapping("rollout_model_config")["mesh"]["shape"], "(2,2)"
    )
    hp.config["training_config"] = {"nested": {"value": 1}}
    training = hp.config_mapping("training_config")
    training["nested"]["value"] = 2
    self.assertEqual(hp.config["training_config"]["nested"]["value"], 1)

  def test_default_path_partial_role_mesh_override_is_not_merged(self):
    # A role that overrides only part of `mesh` keeps the rest of its own
    # mesh; it must not pick up `axis_names` from a differently shaped
    # `model_config.mesh`.
    hp = self.initialize_config([
        'model_config.mesh.shape="(8,)"',
        "model_config.mesh.axis_names=\"('fsdp',)\"",
        'rollout_model_config.mesh.shape="(2,4)"',
    ])
    self.assertEqual(
        hp.parse_mesh_config("rollout_model_config"),
        ((2, 4), ("fsdp", "tp")),
    )
    self.assertEqual(
        hp.parse_mesh_config("actor_model_config"), ((8,), ("fsdp",))
    )

  def test_default_path_partial_role_lora_override_is_not_merged(self):
    hp = self.initialize_config([
        "model_config.lora_config.rank=16",
        "model_config.lora_config.alpha=2.0",
        'model_config.lora_config.module_path=".*q_einsum"',
        "actor_model_config.lora_config.rank=8",
    ])
    self.assertEqual(
        hp.config_mapping("actor_model_config")["lora_config"], {"rank": 8}
    )
    self.assertEqual(
        hp.config_mapping("rollout_model_config")["lora_config"],
        {"rank": 16, "alpha": 2.0, "module_path": ".*q_einsum"},
    )

  def test_merge_path_explicit_role_override_equal_to_base_is_kept(self):
    # `llama-3.1-8b` is the base `model_config.model_name`. With
    # `merge_replace_keys=True`, an explicit role override with that value is
    # not replaced by the updated `model_config.model_name`.
    hp = self._initialize_with_flags(
        [
            "model_config.model_name=gemma-3-12b",
            "model_config.model_source=gcs",
            "actor_model_config.model_name=llama-3.1-8b",
        ],
        validate_nested=False,
    )
    self.assertEqual(
        hp.config_mapping("actor_model_config")["model_name"], "llama-3.1-8b"
    )
    self.assertEqual(
        hp.config_mapping("rollout_model_config")["model_name"], "gemma-3-12b"
    )

  def test_default_path_rejects_multiple_override_files(self):
    with self.assertRaisesRegex(
        ValueError, "Only one override_config_file argument is allowed"
    ):
      self.initialize_config(
          ["override_config_file=a.yaml", "override_config_file=b.yaml"]
      )

  def test_empty_override_config_file_path_raises(self):
    with self.assertRaisesRegex(
        ValueError, "Empty path in override_config_file"
    ):
      self.initialize_config(["override_config_file="])


if __name__ == "__main__":
  if "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = "TestToken"
  absltest.main()
