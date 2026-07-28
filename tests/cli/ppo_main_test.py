# Copyright 2026 Google LLC
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

"""Tests that ppo_main dispatches correctly.

Also tests that KV cache / PPOConfig computation is correct.
"""

import os
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
import omegaconf
from tunix.cli import base_rl_pipeline
from tunix.cli import ppo_main
from tunix.rl import rl_cluster as rl_cluster_lib


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _make_pipeline(extra_yaml: str) -> ppo_main.PpoPipeline:
  """Write a minimal valid YAML and instantiate PpoPipeline against it."""
  base = """
model_config:
  model_name: "test_model"
  model_id: "test/model"
  model_source: "huggingface"
  model_display: false
  rng_seed: 0
  intermediate_ckpt_dir: "/tmp/ckpt"

actor_model_config:
  mesh:
    shape: "(2,1)"
    axis_names: "('fsdp','tp')"

critic_model_config:
  mesh: null
  same_mesh_as: "actor"

reference_model_config:
  mesh:
    shape: "(2,1)"
    axis_names: "('fsdp','tp')"

rollout_model_config:
  mesh:
    shape: "(2,1)"
    axis_names: "('fsdp','tp')"

tokenizer_config:
  tokenizer_type: "huggingface"
  tokenizer_path: "test/model"
  add_bos: false
  add_eos: false

rollout_engine: "vanilla"
offload_to_cpu: false

rollout_config:
  max_prompt_length: 256
  total_generation_steps: 512
  temperature: 1.0
  top_p: null
  top_k: null

rl_training_config:
  max_steps: 1
  eval_every_n_steps: 1
  mini_batch_size: 1
  train_micro_batch_size: 1
  actor_optimizer_config:
    opt_type: "adamw"
    learning_rate: 1.0e-6
    schedule_type: "warmup_cosine_decay_schedule"
    init_value: 0.0
    end_value: 0.0
    warmup_ratio: 0.1
    b1: 0.9
    b2: 0.99
    weight_decay: 0.01
    max_grad_norm: 1.0
  critic_optimizer_config:
    opt_type: "adamw"
    learning_rate: 1.0e-6
    schedule_type: "warmup_cosine_decay_schedule"
    init_value: 0.0
    end_value: 0.0
    warmup_ratio: 0.1
    b1: 0.9
    b2: 0.99
    weight_decay: 0.01
    max_grad_norm: 1.0
  metrics_logging_options:
    log_dir: "/tmp/tb_test"
    flush_every_n_steps: 1
  checkpointing_options:
    save_interval_steps: 100
    max_to_keep: 1
  checkpoint_root_directory: "/tmp/ckpt_test"

batch_size: 1
num_batches: 1
num_train_epochs: 1
train_fraction: 1.0
"""
  with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    f.write(base + extra_yaml)
    path = f.name

  # Patch HF_TOKEN so tokenizer validation passes
  with mock.patch.dict(os.environ, {"HF_TOKEN": "fake"}):
    pipeline = ppo_main.PpoPipeline(["", path])
  os.unlink(path)
  return pipeline


def _make_pipeline_with_cli_args(
    extra_yaml: str, cli_args: list[str]
) -> ppo_main.PpoPipeline:
  """Write a minimal valid YAML and instantiate PpoPipeline with CLI args."""
  base = """
model_config:
  model_name: "test_model"
  model_id: "test/model"
  model_source: "huggingface"
  model_display: false
  rng_seed: 0
  intermediate_ckpt_dir: "/tmp/ckpt"

actor_model_config:
  mesh:
    shape: "(2,1)"
    axis_names: "('fsdp','tp')"

critic_model_config:
  mesh: null
  same_mesh_as: "actor"

reference_model_config:
  mesh:
    shape: "(2,1)"
    axis_names: "('fsdp','tp')"

rollout_model_config:
  mesh:
    shape: "(2,1)"
    axis_names: "('fsdp','tp')"

tokenizer_config:
  tokenizer_type: "huggingface"
  tokenizer_path: "test/model"
  add_bos: false
  add_eos: false

rollout_engine: "vanilla"
offload_to_cpu: false

rollout_config:
  max_prompt_length: 256
  total_generation_steps: 512
  temperature: 1.0
  top_p: null
  top_k: null

rl_training_config:
  max_steps: 1
  eval_every_n_steps: 1
  mini_batch_size: 1
  train_micro_batch_size: 1
  actor_optimizer_config:
    opt_type: "adamw"
    learning_rate: 1.0e-6
    schedule_type: "warmup_cosine_decay_schedule"
    init_value: 0.0
    end_value: 0.0
    warmup_ratio: 0.1
    b1: 0.9
    b2: 0.99
    weight_decay: 0.01
    max_grad_norm: 1.0
  critic_optimizer_config:
    opt_type: "adamw"
    learning_rate: 1.0e-6
    schedule_type: "warmup_cosine_decay_schedule"
    init_value: 0.0
    end_value: 0.0
    warmup_ratio: 0.1
    b1: 0.9
    b2: 0.99
    weight_decay: 0.01
    max_grad_norm: 1.0
  metrics_logging_options:
    log_dir: "/tmp/tb_test"
    flush_every_n_steps: 1
  checkpointing_options:
    save_interval_steps: 100
    max_to_keep: 1
  checkpoint_root_directory: "/tmp/ckpt_test"

batch_size: 1
num_batches: 1
num_train_epochs: 1
train_fraction: 1.0
"""
  with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    f.write(base + extra_yaml)
    path = f.name

  with mock.patch.dict(os.environ, {"HF_TOKEN": "fake"}):
    pipeline = ppo_main.PpoPipeline(["", path, *cli_args])
  os.unlink(path)
  return pipeline


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


class DispatchTest(absltest.TestCase):

  def test_data_module_receives_data_config_for_raw_dataset(self):
    extra = """
training_mode: "ppo"
data_module: "tunix.cli.recipes.deepscaler_data"
apply_chat_template_to_dataset: false
data_config:
  train_data_path: "gs://fake/train.json"
  eval_data_path: "gs://fake/eval.parquet"
prompt_key: "prompts"
reward_functions: []
verl_compatible: false
chat_parser_config:
  type: "default"
ppo_config:
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
"""
    pipeline = _make_pipeline(extra)
    fake_module = mock.Mock(batch_fn=mock.sentinel.batch_fn)

    with mock.patch.object(
        pipeline, "_get_data_module", return_value=fake_module
    ):
      with mock.patch.object(
          pipeline,
          "_get_dataset",
          return_value=mock.sentinel.dataset,
      ) as get_dataset:
        dataset, batch_fn = pipeline._load_raw_dataset(mock.sentinel.tokenizer)

    self.assertIs(dataset, mock.sentinel.dataset)
    self.assertIs(batch_fn, mock.sentinel.batch_fn)
    get_dataset.assert_called_once_with(
        mock.sentinel.tokenizer,
    )

  def test_standard_ppo_dispatches_to_standard(self):
    extra = """
ppo_config:
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
data_source: "tfds"
dataset_name: "gsm8k"
tfds_download: false
reward_functions: []
verl_compatible: false
"""
    pipeline = _make_pipeline(extra)
    self.assertEqual(pipeline.config.get("training_mode", "ppo"), "ppo")
    with mock.patch.object(pipeline, "_run") as mock_run:
      pipeline.run_trainer()
      mock_run.assert_called_once_with(mode="ppo")

  def test_unknown_mode_raises(self):
    extra = """
ppo_config:
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
data_source: "tfds"
dataset_name: "gsm8k"
tfds_download: false
reward_functions: []
verl_compatible: false
"""
    pipeline = _make_pipeline(extra)
    pipeline.config["training_mode"] = "bad_mode"
    raw_dataset = mock.Mock()
    raw_dataset.__len__ = mock.Mock(return_value=1)
    with mock.patch.object(pipeline, "_setup_kubernetes"):
      with mock.patch.object(
          pipeline, "_get_tokenizer", return_value=mock.sentinel.tokenizer
      ):
        with mock.patch.object(
            pipeline,
            "_create_chat_parser",
            return_value=mock.sentinel.chat_parser,
        ):
          with mock.patch.object(
              pipeline,
              "_load_raw_dataset",
              return_value=(raw_dataset, None),
          ):
            with mock.patch.object(pipeline, "compute_params"):
              with mock.patch.object(
                  ppo_main.data_lib,
                  "post_init_dataset",
                  return_value=(mock.sentinel.dataset, None),
              ):
                with mock.patch.object(
                    pipeline,
                    "create_rl_cluster",
                    return_value=mock.sentinel.rl_cluster,
                ):
                  with self.assertRaisesRegex(
                      ValueError, "Unsupported training_mode 'bad_mode'"
                  ):
                    pipeline.run_trainer()


# ---------------------------------------------------------------------------
# KV cache formula
# ---------------------------------------------------------------------------


class RolloutConfigTest(absltest.TestCase):

  def test_standard_ppo_kv_cache(self):
    extra = """
ppo_config:
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
data_source: "tfds"
dataset_name: "gsm8k"
tfds_download: false
reward_functions: []
verl_compatible: false
"""
    p = _make_pipeline(extra)
    cfg = p.create_rollout_config()
    self.assertEqual(cfg.kv_cache_size, 256 + 512 + 256)

  def test_vllm_submission_threshold_passed_through(self):
    extra = """
vllm_config:
  hbm_utilization: 0.4
  server_mode: true
  server_mode_submission_threshold: 3840
  server_mode_submission_timeout_s: 1.5
"""
    p = _make_pipeline_with_cli_args(extra, ["rollout_engine=vllm"])
    role_to_mesh = {
        rl_cluster_lib.Role.ROLLOUT: mock.Mock(
            devices=mock.Mock(shape=(2, 1))
        )
    }
    cfg = p.create_rollout_config(role_to_mesh=role_to_mesh)
    self.assertEqual(cfg.rollout_vllm_server_mode_submission_threshold, 3840)
    self.assertEqual(cfg.rollout_vllm_server_mode_submission_timeout_s, 1.5)


# ---------------------------------------------------------------------------
# Mesh configuration
# ---------------------------------------------------------------------------


class SplitMeshConfigTest(absltest.TestCase):

  def test_split_mesh_uses_explicit_role_meshes(self):
    extra = """
training_mode: "ppo"
ppo_config:
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
data_source: "tfds"
dataset_name: "gsm8k"
tfds_download: false
reward_functions: []
verl_compatible: false
"""
    pipeline = _make_pipeline(extra)
    actor_model_config = pipeline.config["actor_model_config"]
    if isinstance(actor_model_config, omegaconf.dictconfig.DictConfig):
      actor_model_config["mesh"] = {
          "shape": "(2,1)",
          "axis_names": "('fsdp','tp')",
      }
    pipeline.config["critic_model_config"] = {
        "mesh": None,
        "same_mesh_as": "actor",
    }
    pipeline.config["reference_model_config"] = {"same_mesh_as": "actor"}
    rollout_model_config = pipeline.config["rollout_model_config"]
    if isinstance(rollout_model_config, omegaconf.dictconfig.DictConfig):
      rollout_model_config["mesh"] = {
          "shape": "(1,2)",
          "axis_names": "('fsdp','tp')",
      }

    class FakeDevice:

      def __init__(self, device_id, coords):
        self.id = device_id
        self.coords = coords
        self.process_index = 0
        self.slice_index = 0
        self.device_kind = "TPU v5e"

    fake_devices = [
        FakeDevice(0, (0, 0)),
        FakeDevice(1, (1, 0)),
        FakeDevice(2, (0, 1)),
        FakeDevice(3, (1, 1)),
    ]

    class FakeMesh:

      def __init__(self, devices, axis_names, axis_types=None):
        self.devices = devices
        self.axis_names = axis_names
        self.axis_types = axis_types

    with mock.patch.object(
        base_rl_pipeline.jax, "devices", return_value=fake_devices
    ):
      with mock.patch.object(
          base_rl_pipeline.jax.sharding, "Mesh", side_effect=FakeMesh
      ):
        role_to_mesh = pipeline.create_role_to_mesh()

    self.assertSequenceEqual(
        [
            device.id
            for device in (
                role_to_mesh[rl_cluster_lib.Role.ACTOR]
                .devices.flatten()
                .tolist()
            )
        ],
        [0, 1],
    )
    self.assertSequenceEqual(
        [
            device.id
            for device in (
                role_to_mesh[rl_cluster_lib.Role.ROLLOUT]
                .devices.flatten()
                .tolist()
            )
        ],
        [2, 3],
    )
    self.assertEqual(
        role_to_mesh[rl_cluster_lib.Role.ACTOR].devices.shape,
        (2, 1),
    )
    self.assertEqual(
        role_to_mesh[rl_cluster_lib.Role.ROLLOUT].devices.shape,
        (1, 2),
    )
    self.assertIs(
        role_to_mesh[rl_cluster_lib.Role.REFERENCE],
        role_to_mesh[rl_cluster_lib.Role.ACTOR],
    )
    self.assertIs(
        role_to_mesh[rl_cluster_lib.Role.CRITIC],
        role_to_mesh[rl_cluster_lib.Role.ACTOR],
    )

  def test_create_role_to_mesh_passes_configured_allocation_policy(self):
    extra = """
training_mode: "ppo"
ppo_config:
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
data_source: "tfds"
dataset_name: "gsm8k"
tfds_download: false
reward_functions: []
verl_compatible: false
"""
    pipeline = _make_pipeline(extra)
    actor_model_config = pipeline.config["actor_model_config"]
    if isinstance(actor_model_config, omegaconf.dictconfig.DictConfig):
      actor_model_config["mesh"] = {
          "shape": "(2,1)",
          "axis_names": "('fsdp','tp')",
          "allocation_policy": "PERFORMANCE",
      }
    pipeline.config["critic_model_config"] = {
        "mesh": None,
        "same_mesh_as": "actor",
    }
    pipeline.config["reference_model_config"] = {"same_mesh_as": "actor"}
    rollout_model_config = pipeline.config["rollout_model_config"]
    if isinstance(rollout_model_config, omegaconf.dictconfig.DictConfig):
      rollout_model_config["mesh"] = {
          "shape": "(1,2)",
          "axis_names": "('fsdp','tp')",
          "allocation_policy": "PERFORMANCE",
      }

    fake_devices = list(range(4))

    with mock.patch.object(
        base_rl_pipeline.jax, "devices", return_value=fake_devices
    ):
      with mock.patch.object(
          base_rl_pipeline.mesh_lib,
          "allocate_named_mesh_device_slices",
          return_value={
              "actor_model_config": [0, 1],
              "rollout_model_config": [2, 3],
          },
      ) as allocate_mock, mock.patch.object(
          base_rl_pipeline.mesh_lib,
          "create_mesh",
          side_effect=[object(), object()],
      ):
        pipeline.create_role_to_mesh()

    allocate_mock.assert_called_once_with(
        [("actor_model_config", 2), ("rollout_model_config", 2)],
        devices=fake_devices,
        allocation_policy="PERFORMANCE",
    )


if __name__ == "__main__":
  absltest.main()
