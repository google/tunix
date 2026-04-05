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

"""Tests that grpo_main dispatches correctly for both training modes
and that KV cache / GRPOConfig computation is correct."""

import dataclasses
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from absl.testing import absltest
from tunix.cli import grpo_main
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _make_pipeline(extra_yaml: str) -> grpo_main.GrpoPipeline:
  """Write a minimal valid YAML and instantiate GrpoPipeline against it."""
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
    shape: "(1,1)"
    axis_names: "('fsdp','tp')"

reference_model_config:
  mesh:
    shape: "(1,1)"
    axis_names: "('fsdp','tp')"

rollout_model_config:
  mesh:
    shape: "(1,1)"
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
  with tempfile.NamedTemporaryFile(
      mode="w", suffix=".yaml", delete=False
  ) as f:
    f.write(base + extra_yaml)
    path = f.name

  # Patch HF_TOKEN so tokenizer validation passes
  with mock.patch.dict(os.environ, {"HF_TOKEN": "fake"}):
    pipeline = grpo_main.GrpoPipeline(["", path])
  os.unlink(path)
  return pipeline


def _make_pipeline_with_cli_args(
    extra_yaml: str, cli_args: list[str]
) -> grpo_main.GrpoPipeline:
  """Write a minimal valid YAML and instantiate GrpoPipeline with CLI args."""
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
    shape: "(1,1)"
    axis_names: "('fsdp','tp')"

reference_model_config:
  mesh:
    shape: "(1,1)"
    axis_names: "('fsdp','tp')"

rollout_model_config:
  mesh:
    shape: "(1,1)"
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
  with tempfile.NamedTemporaryFile(
      mode="w", suffix=".yaml", delete=False
  ) as f:
    f.write(base + extra_yaml)
    path = f.name

  with mock.patch.dict(os.environ, {"HF_TOKEN": "fake"}):
    pipeline = grpo_main.GrpoPipeline(["", path, *cli_args])
  os.unlink(path)
  return pipeline


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


class DispatchTest(absltest.TestCase):

  def test_standard_grpo_dispatches_to_standard(self):
    extra = """
grpo_config:
  num_generations: 2
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
    self.assertEqual(pipeline.config.get("training_mode", "grpo"), "grpo")
    # _run_standard_grpo should be called; we verify no AttributeError on dispatch
    with mock.patch.object(pipeline, "_run_standard_grpo") as mock_std:
      pipeline.run_grpo_trainer()
      mock_std.assert_called_once()

  def test_agentic_grpo_dispatches_to_agentic(self):
    extra = """
training_mode: "agentic_grpo"
data_module: "tunix.cli.recipes.deepscaler_data"
data_config:
  train_data_path: "gs://fake/train.json"
  eval_data_path: "gs://fake/eval.parquet"
prompt_key: "prompts"
reward_functions:
  - "tunix/utils/math_rewards.py"
verl_compatible: false
chat_parser_config:
  type: "default"
agent_class_path: null
agent_kwargs: {}
env_class_path: null
env_kwargs: {}
kubernetes_config: null
agentic_grpo_config:
  num_generations: 2
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
  epsilon_high: 0.28
  system_prompt: ""
  max_concurrency: 1
  off_policy_steps: 0
  max_turns: 1
  context_ratio: 1
sglang_jax_config:
  mem_fraction_static: 0.8
vllm_config:
  hbm_utilization: 0.4
"""
    pipeline = _make_pipeline(extra)
    self.assertEqual(pipeline.config["training_mode"], "agentic_grpo")
    with mock.patch.object(pipeline, "_run_agentic_grpo") as mock_ag:
      pipeline.run_grpo_trainer()
      mock_ag.assert_called_once()

  def test_unknown_mode_raises(self):
    # Build pipeline with standard config then manually set bad mode
    extra = """
grpo_config:
  num_generations: 2
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
    with self.assertRaisesRegex(ValueError, "Unknown training_mode"):
      pipeline.run_grpo_trainer()


# ---------------------------------------------------------------------------
# KV cache formula
# ---------------------------------------------------------------------------


class RolloutConfigTest(absltest.TestCase):

  def _make_agentic_pipeline(self, max_turns, context_ratio):
    extra = f"""
training_mode: "agentic_grpo"
data_module: "tunix.cli.recipes.deepscaler_data"
data_config:
  train_data_path: "gs://fake/train.json"
  eval_data_path: "gs://fake/eval.parquet"
prompt_key: "prompts"
reward_functions: []
verl_compatible: false
chat_parser_config:
  type: "default"
agent_class_path: null
agent_kwargs: {{}}
env_class_path: null
env_kwargs: {{}}
kubernetes_config: null
agentic_grpo_config:
  num_generations: 2
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
  epsilon_high: 0.28
  system_prompt: ""
  max_concurrency: 1
  off_policy_steps: 0
  max_turns: {max_turns}
  context_ratio: {context_ratio}
sglang_jax_config:
  mem_fraction_static: 0.8
vllm_config:
  hbm_utilization: 0.4
"""
    return _make_pipeline(extra)

  def test_single_turn_kv_cache(self):
    p = self._make_agentic_pipeline(max_turns=1, context_ratio=1)
    cfg = p.create_rollout_config()
    # max_prompt=256, max_response=512, single-turn → +256
    self.assertEqual(cfg.kv_cache_size, 256 + 512 + 256)

  def test_multi_turn_kv_cache(self):
    p = self._make_agentic_pipeline(max_turns=20, context_ratio=2)
    cfg = p.create_rollout_config()
    # max_prompt=256, max_response=512, 20 turns * ratio 2
    self.assertEqual(cfg.kv_cache_size, 256 + 512 * 2 * 20)

  def test_standard_grpo_kv_cache(self):
    extra = """
grpo_config:
  num_generations: 2
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


# ---------------------------------------------------------------------------
# GRPOConfig construction
# ---------------------------------------------------------------------------


class AgenticConfigTest(absltest.TestCase):

  def _base_extra(
      self, agentic_overrides="", system_prompt='""'
  ):
    return f"""
training_mode: "agentic_grpo"
data_module: "tunix.cli.recipes.deepscaler_data"
data_config:
  train_data_path: "gs://fake/train.json"
  eval_data_path: "gs://fake/eval.parquet"
prompt_key: "prompts"
reward_functions: []
verl_compatible: false
chat_parser_config:
  type: "default"
agent_class_path: null
agent_kwargs: {{}}
env_class_path: null
env_kwargs: {{}}
kubernetes_config: null
agentic_grpo_config:
  num_generations: 2
  num_iterations: 1
  beta: 0.001
  epsilon: 0.2
  epsilon_high: 0.28
  system_prompt: {system_prompt}
  max_concurrency: 1
  off_policy_steps: 0
  {agentic_overrides}
sglang_jax_config:
  mem_fraction_static: 0.8
vllm_config:
  hbm_utilization: 0.4
"""

  def test_episode_timeout_computed(self):
    p = _make_pipeline(
        self._base_extra("max_turns: 20\n  per_turn_timeout_secs: 300")
    )
    algo = p._create_agentic_grpo_config()
    self.assertEqual(algo.episode_timeout, 300 * 20)

  def test_max_response_length_from_rollout(self):
    p = _make_pipeline(self._base_extra("max_turns: 1"))
    algo = p._create_agentic_grpo_config()
    # rollout_config.total_generation_steps = 512
    self.assertEqual(algo.max_response_length, 512)

  def test_num_generations_passed_through(self):
    p = _make_pipeline(self._base_extra("max_turns: 1"))
    algo = p._create_agentic_grpo_config()
    self.assertEqual(algo.num_generations, 2)

  def test_cli_empty_system_prompt_stays_empty_string(self):
    p = _make_pipeline_with_cli_args(
        self._base_extra("max_turns: 1", system_prompt='"base"'),
        ['agentic_grpo_config.system_prompt=""'],
    )
    self.assertEqual(p.config["agentic_grpo_config"]["system_prompt"], "")


class SplitMeshConfigTest(absltest.TestCase):

  def test_split_mesh_uses_explicit_role_meshes(self):
    extra = """
training_mode: "agentic_grpo"
data_module: "tunix.cli.recipes.deepscaler_data"
data_config:
  train_data_path: "gs://fake/train.json"
  eval_data_path: "gs://fake/eval.parquet"
prompt_key: "prompts"
reward_functions: []
verl_compatible: false
chat_parser_config:
  type: "default"
agent_class_path: null
agent_kwargs: {}
env_class_path: null
env_kwargs: {}
kubernetes_config: null
agentic_grpo_config:
  num_generations: 2
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
  epsilon_high: 0.28
  system_prompt: ""
  max_concurrency: 1
  off_policy_steps: 0
  max_turns: 1
  context_ratio: 1
sglang_jax_config:
  mem_fraction_static: 0.8
vllm_config:
  hbm_utilization: 0.4
"""
    pipeline = _make_pipeline(extra)
    pipeline.config["actor_model_config"]["mesh"] = {
        "shape": "(2,1)",
        "axis_names": "('fsdp','tp')",
    }
    pipeline.config["reference_model_config"] = {"same_mesh_as": "actor"}
    pipeline.config["rollout_model_config"]["mesh"] = {
        "shape": "(1,2)",
        "axis_names": "('fsdp','tp')",
    }

    fake_devices = list(range(4))

    class FakeMesh:

      def __init__(self, devices, axis_names, axis_types=None):
        self.devices = devices
        self.axis_names = axis_names
        self.axis_types = axis_types

    with mock.patch.object(grpo_main.jax, "devices", return_value=fake_devices):
      with mock.patch.object(grpo_main.jax.sharding, "Mesh", side_effect=FakeMesh):
        role_to_mesh = pipeline.create_role_to_mesh()

    self.assertSequenceEqual(
        role_to_mesh[rl_cluster_lib.Role.ACTOR].devices.flatten().tolist(),
        [0, 1],
    )
    self.assertSequenceEqual(
        role_to_mesh[rl_cluster_lib.Role.ROLLOUT].devices.flatten().tolist(),
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


if __name__ == "__main__":
  absltest.main()
