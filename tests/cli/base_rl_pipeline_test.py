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

"""Tests that grpo_main dispatches correctly for both training modes.

Also tests that KV cache / GRPOConfig computation is correct.
"""
import dataclasses
import os
import pathlib
import tempfile
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import omegaconf
from tunix.cli import base_rl_pipeline
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclasses.dataclass(kw_only=True)
class DummyConfig(algo_config_lib.AlgorithmConfig):
  """Dummy RL algorithm config."""
  num_generations: int = 2
  num_iterations: int = 1
  episode_timeout: int = 1
  max_response_length: int = 1000


class DummyPipeline(base_rl_pipeline.BasePipeline):
  def __init__(self, argv: list[str], **kwargs):
    self.data_module: types.ModuleType | None = None
    super().__init__(argv, **kwargs)

  def _is_agentic_mode(self, mode: str) -> bool:
    return mode == "agentic_dummy"

  @property
  def _default_training_mode(self):
    return "dummy"

  def _create_agentic_dummy_config(self):
    cfg = dict(self._config_mapping("agentic_dummy_config"))

    # episode_timeout = per_turn_timeout_secs * max_turns when not explicit
    if "episode_timeout" not in cfg:
      per_turn = cfg.pop("per_turn_timeout_secs", None)
      max_turns = cfg.get("max_turns", 1)
      if per_turn is not None:
        cfg["episode_timeout"] = per_turn * max_turns

    # max_response_length mirrors rollout_config.total_generation_steps
    if "max_response_length" not in cfg:
      cfg["max_response_length"] = self._config_mapping("rollout_config").get(
          "total_generation_steps", 8192
      )

    # Strip helper keys that are not GRPOConfig fields
    valid = {f.name for f in dataclasses.fields(DummyConfig) if f.init}
    cfg.pop("max_turns", None)

    return DummyConfig(**{k: v for k, v in cfg.items() if k in valid})

  def create_rl_engine(self, tokenizer: Any):
    pass

  def compute_params(self, dataset: Any):
    pass

  def _run(self, mode: str):
    pass


def _make_pipeline(extra_yaml: str) -> DummyPipeline:
  """Write a minimal valid YAML and instantiate DummyPipeline against it."""
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
  with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    f.write(base + extra_yaml)
    path = f.name

  # Patch HF_TOKEN so tokenizer validation passes
  with mock.patch.dict(os.environ, {"HF_TOKEN": "fake"}):
    pipeline = DummyPipeline(["", path])
  os.unlink(path)
  return pipeline


def _make_pipeline_with_cli_args(
    extra_yaml: str, cli_args: list[str]
) -> DummyPipeline:
  """Write a minimal valid YAML and instantiate DummyPipeline with CLI args."""
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
  with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    f.write(base + extra_yaml)
    path = f.name

  with mock.patch.dict(os.environ, {"HF_TOKEN": "fake"}):
    pipeline = DummyPipeline(["", path, *cli_args])
  os.unlink(path)
  return pipeline


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


class DispatchTest(absltest.TestCase):

  def test_agentic_data_module_receives_data_config_for_raw_dataset(self):
    extra = """
training_mode: "agentic_dummy"
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
agent_class_path: null
agent_kwargs: {}
env_class_path: null
env_kwargs: {}
kubernetes_config: null
agentic_dummy_config:
  num_generations: 2
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
  epsilon_high: 0.28
  system_prompt: ""
  max_concurrency: 1
  off_policy_steps: 0
  max_turns: 1
vllm_config:
  hbm_utilization: 0.4
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

  def test_agentic_nullable_string_can_be_overridden_from_cli(self):
    extra = """
training_mode: "agentic_dummy"
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
agent_class_path: null
agent_kwargs: {}
env_class_path: null
env_kwargs: {}
kubernetes_config: null
agentic_dummy_config:
  num_generations: 2
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
  epsilon_high: 0.28
  system_prompt: ""
  max_concurrency: 1
  off_policy_steps: 0
  max_turns: 1
vllm_config:
  hbm_utilization: 0.4
"""
    pipeline = _make_pipeline_with_cli_args(
        extra,
        ["agent_class_path=examples.deepswe.swe_agent.SWEAgent"],
    )

    self.assertEqual(
        pipeline.config["agent_class_path"],
        "examples.deepswe.swe_agent.SWEAgent",
    )

  def test_agentic_nullable_dict_can_be_overridden_from_cli(self):
    extra = """
training_mode: "agentic_dummy"
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
agent_class_path: null
agent_kwargs: {}
env_class_path: null
env_kwargs: {}
kubernetes_config: null
agentic_dummy_config:
  num_generations: 2
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
  epsilon_high: 0.28
  system_prompt: ""
  max_concurrency: 1
  off_policy_steps: 0
  max_turns: 1
vllm_config:
  hbm_utilization: 0.4
"""
    pipeline = _make_pipeline_with_cli_args(
        extra,
        [
            "kubernetes_config.node_selector_key=cloud.google.com/gke-nodepool",
            "kubernetes_config.node_selector_val=deepswe-cpu-pool",
        ],
    )

    self.assertEqual(
        pipeline.config["kubernetes_config"],
        {
            "node_selector_key": "cloud.google.com/gke-nodepool",
            "node_selector_val": "deepswe-cpu-pool",
        },
    )

  def test_agentic_nullable_string_can_be_overridden_from_env(self):
    extra = """
training_mode: "agentic_dummy"
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
agent_class_path: null
agent_kwargs: {}
env_class_path: null
env_kwargs: {}
kubernetes_config: null
agentic_dummy_config:
  num_generations: 2
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
  epsilon_high: 0.28
  system_prompt: ""
  max_concurrency: 1
  off_policy_steps: 0
  max_turns: 1
vllm_config:
  hbm_utilization: 0.4
"""
    with mock.patch.dict(
        os.environ,
        {"T_AGENT_CLASS_PATH": "examples.deepswe.swe_agent.SWEAgent"},
    ):
      pipeline = _make_pipeline_with_cli_args(extra, [])

    self.assertEqual(
        pipeline.config["agent_class_path"],
        "examples.deepswe.swe_agent.SWEAgent",
    )

  def test_standard_dummy_dispatches_to_standard(self):
    extra = """
dummy_config:
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
    self.assertEqual(pipeline.config.get("training_mode", "dummy"), "dummy")
    with mock.patch.object(pipeline, "_run") as mock_run:
      pipeline.run_trainer()
      mock_run.assert_called_once_with(mode="dummy")

  def test_agentic_dummy_dispatches_to_agentic(self):
    extra = """
training_mode: "agentic_dummy"
data_module: "tunix.cli.recipes.deepscaler_data"
apply_chat_template_to_dataset: false
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
agentic_dummy_config:
  num_generations: 2
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
  epsilon_high: 0.28
  system_prompt: ""
  max_concurrency: 1
  off_policy_steps: 0
  max_turns: 1
vllm_config:
  hbm_utilization: 0.4
"""
    pipeline = _make_pipeline(extra)
    self.assertEqual(pipeline.config["training_mode"], "agentic_dummy")
    with mock.patch.object(pipeline, "_run") as mock_run:
      pipeline.run_trainer()
      mock_run.assert_called_once_with(mode="agentic_dummy")

# ---------------------------------------------------------------------------
# KV cache formula
# ---------------------------------------------------------------------------


class RolloutConfigTest(absltest.TestCase):

  def _make_agentic_pipeline(self, max_turns):
    extra = f"""
training_mode: "agentic_dummy"
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
agent_class_path: null
agent_kwargs: {{}}
env_class_path: null
env_kwargs: {{}}
kubernetes_config: null
agentic_dummy_config:
  num_generations: 2
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
  epsilon_high: 0.28
  system_prompt: ""
  max_concurrency: 1
  off_policy_steps: 0
  max_turns: {max_turns}
vllm_config:
  hbm_utilization: 0.4
"""
    return _make_pipeline(extra)

  def test_single_turn_kv_cache(self):
    p = self._make_agentic_pipeline(max_turns=1)
    cfg = p.create_rollout_config()
    # max_prompt=256, max_response=512, single-turn → +256
    self.assertEqual(cfg.kv_cache_size, 256 + 512 + 256)

  def test_multi_turn_kv_cache(self):
    p = self._make_agentic_pipeline(max_turns=20)
    cfg = p.create_rollout_config()
    self.assertEqual(cfg.kv_cache_size, 256 + 512 + 256)

  def test_standard_dummy_kv_cache(self):
    extra = """
dummy_config:
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
        rl_engine_lib.Role.ROLLOUT: mock.Mock(
            devices=mock.Mock(shape=(1, 1))
        )
    }
    cfg = p.create_rollout_config(role_to_mesh=role_to_mesh)
    self.assertEqual(cfg.rollout_vllm_server_mode_submission_threshold, 3840)
    self.assertEqual(cfg.rollout_vllm_server_mode_submission_timeout_s, 1.5)

# ---------------------------------------------------------------------------
# GRPOConfig construction
# ---------------------------------------------------------------------------


class AgenticConfigTest(absltest.TestCase):

  def _base_extra(self, agentic_overrides="", system_prompt='""'):
    return f"""
training_mode: "agentic_dummy"
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
agent_class_path: null
agent_kwargs: {{}}
env_class_path: null
env_kwargs: {{}}
kubernetes_config: null
agentic_dummy_config:
  num_generations: 2
  num_iterations: 1
  beta: 0.001
  epsilon: 0.2
  epsilon_high: 0.28
  system_prompt: {system_prompt}
  max_concurrency: 1
  off_policy_steps: 0
  {agentic_overrides}
vllm_config:
  hbm_utilization: 0.4
"""

  def test_episode_timeout_computed(self):
    p = _make_pipeline(
        self._base_extra("max_turns: 20\n  per_turn_timeout_secs: 300")
    )
    algo = p._create_agentic_dummy_config()
    self.assertEqual(algo.episode_timeout, 300 * 20)

  def test_max_response_length_from_rollout(self):
    p = _make_pipeline(self._base_extra("max_turns: 1"))
    algo = p._create_agentic_dummy_config()
    # rollout_config.total_generation_steps = 512
    self.assertEqual(algo.max_response_length, 512)

  def test_num_generations_passed_through(self):
    p = _make_pipeline(self._base_extra("max_turns: 1"))
    algo = p._create_agentic_dummy_config()
    self.assertEqual(algo.num_generations, 2)

  def test_cli_empty_system_prompt_stays_empty_string(self):
    p = _make_pipeline_with_cli_args(
        self._base_extra("max_turns: 1", system_prompt='"base"'),
        ['agentic_dummy_config.system_prompt=""'],
    )
    self.assertEqual(p.config["agentic_dummy_config"]["system_prompt"], "")


class SplitMeshConfigTest(absltest.TestCase):

  def test_split_mesh_uses_explicit_role_meshes(self):
    extra = """
training_mode: "agentic_dummy"
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
agent_class_path: null
agent_kwargs: {}
env_class_path: null
env_kwargs: {}
kubernetes_config: null
agentic_dummy_config:
  num_generations: 2
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
  epsilon_high: 0.28
  system_prompt: ""
  max_concurrency: 1
  off_policy_steps: 0
  max_turns: 1
vllm_config:
  hbm_utilization: 0.4
"""
    pipeline = _make_pipeline(extra)
    actor_model_config = pipeline.config["actor_model_config"]
    if isinstance(actor_model_config, omegaconf.dictconfig.DictConfig):
      actor_model_config["mesh"] = {
          "shape": "(2,1)",
          "axis_names": "('fsdp','tp')",
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
                role_to_mesh[rl_engine_lib.Role.ACTOR]
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
                role_to_mesh[rl_engine_lib.Role.ROLLOUT]
                .devices.flatten()
                .tolist()
            )
        ],
        [2, 3],
    )
    self.assertEqual(
        role_to_mesh[rl_engine_lib.Role.ACTOR].devices.shape,
        (2, 1),
    )
    self.assertEqual(
        role_to_mesh[rl_engine_lib.Role.ROLLOUT].devices.shape,
        (1, 2),
    )
    self.assertIs(
        role_to_mesh[rl_engine_lib.Role.REFERENCE],
        role_to_mesh[rl_engine_lib.Role.ACTOR],
    )

  def test_create_role_to_mesh_passes_configured_allocation_policy(self):
    extra = """
training_mode: "agentic_dummy"
verl_compatible: false
chat_parser_config:
  type: "default"
agent_class_path: null
agent_kwargs: {}
env_class_path: null
env_kwargs: {}
kubernetes_config: null
agentic_dummy_config:
  num_generations: 2
  num_iterations: 1
  beta: 0.0
  epsilon: 0.2
  epsilon_high: 0.28
  system_prompt: ""
  max_concurrency: 1
  off_policy_steps: 0
  max_turns: 1
vllm_config:
  hbm_utilization: 0.4
"""
    pipeline = _make_pipeline(extra)
    actor_model_config = pipeline.config["actor_model_config"]
    if isinstance(actor_model_config, omegaconf.dictconfig.DictConfig):
      actor_model_config["mesh"] = {
          "shape": "(2,1)",
          "axis_names": "('fsdp','tp')",
          "allocation_policy": "PERFORMANCE",
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
        base_rl_pipeline.jax, "devices",
        return_value=fake_devices
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

  def test_score_centering_auto_derives_and_validates_num_logprobs(self):
    sc_yaml = """
grpo_config:
  num_generations: 2
  num_iterations: 1
  score_centering: true
  score_centering_top_k: 64
"""
    pipeline = _make_pipeline(sc_yaml)
    self.assertEqual(pipeline.create_rollout_config().num_logprobs, 64)

    top1_pipeline = _make_pipeline_with_cli_args(
        sc_yaml, ["grpo_config.score_centering_top_k=1"]
    )
    self.assertEqual(top1_pipeline.create_rollout_config().num_logprobs, 2)

    bad_pipeline = _make_pipeline_with_cli_args(
        sc_yaml, ["rollout_config.num_logprobs=16"]
    )
    with self.assertRaisesRegex(ValueError, "num_logprobs.*score_centering"):
      bad_pipeline.create_rollout_config()


class ChatParserConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "qwen_null_defaults_true",
          "qwen",
          None,
          [],
          chat_parser_lib.QwenChatTemplateParser,
          True,
      ),
      (
          "qwen_cli_false_disables",
          "qwen",
          None,
          ["chat_parser_config.enable_thinking=false"],
          chat_parser_lib.QwenChatTemplateParser,
          False,
      ),
      (
          "qwen_quoted_string_false_disables",
          "qwen",
          "false",
          [],
          chat_parser_lib.QwenChatTemplateParser,
          False,
      ),
      (
          "qwen_quoted_string_true_enables",
          "qwen",
          "true",
          [],
          chat_parser_lib.QwenChatTemplateParser,
          True,
      ),
      (
          "qwen3p5_null_defaults_true",
          "qwen3.5",
          None,
          [],
          chat_parser_lib.Qwen35ChatTemplateParser,
          True,
      ),
      (
          "qwen3p5_cli_false_disables",
          "qwen3.5",
          None,
          ["chat_parser_config.enable_thinking=false"],
          chat_parser_lib.Qwen35ChatTemplateParser,
          False,
      ),
      (
          "gemma4_null_defaults_false",
          "gemma4",
          None,
          [],
          chat_parser_lib.Gemma4ChatTemplateParser,
          False,
      ),
      (
          "gemma4_cli_true_enables",
          "gemma4",
          None,
          ["chat_parser_config.enable_thinking=true"],
          chat_parser_lib.Gemma4ChatTemplateParser,
          True,
      ),
  )
  def test_chat_parser_enable_thinking_resolution(
      self,
      parser_type,
      raw_value,
      cli_args,
      expected_cls,
      expected_thinking,
  ):
    tokenizer = mock.MagicMock(bos_token="<bos>", eos_token="<eos>")
    pipeline = _make_pipeline_with_cli_args(
        f'chat_parser_config:\n  type: "{parser_type}"\n  enable_thinking:'
        " null\n",
        cli_args,
    )
    if raw_value is not None:
      pipeline.config["chat_parser_config"]["enable_thinking"] = raw_value
    parser = pipeline._create_chat_parser(tokenizer)
    self.assertIsInstance(parser, expected_cls)
    self.assertIs(parser.enable_thinking, expected_thinking)

  @parameterized.named_parameters(
      ("invalid_string", "invalid", r"chat_parser_config\.enable_thinking"),
      ("non_bool_int", 123, "must be a bool, bool string"),
  )
  def test_chat_parser_rejects_invalid_enable_thinking(
      self, bad_value, expected_regex
  ):
    tokenizer = mock.MagicMock(bos_token="<bos>", eos_token="<eos>")
    pipeline = _make_pipeline(
        'chat_parser_config:\n  type: "qwen"\n  enable_thinking: null\n'
    )
    pipeline.config["chat_parser_config"]["enable_thinking"] = bad_value
    with self.assertRaisesRegex(ValueError, expected_regex):
      pipeline._create_chat_parser(tokenizer)


class Qwen3p5VllmRolloutConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "tp_gt_1",
          (1, 2),
          "",
          [],
          None,
          NotImplementedError,
          "tensor_parallel_size=1 only",
      ),
      (
          "inferred_tp_gt_1_from_minus_one_sentinel",
          (1, 2),
          "  tensor_parallel_size: -1\n",
          [],
          None,
          NotImplementedError,
          "got tensor_parallel_size=2",
      ),
      (
          "inferred_tp_gt_1_with_dp_minus_one_and_ep_2",
          (2, 2),
          "  tensor_parallel_size: -1\n  data_parallel_size: -1\n",
          ["rollout_config.expert_parallel_size=2"],
          None,
          NotImplementedError,
          "got tensor_parallel_size=2",
      ),
      (
          "lora_enabled",
          (1, 1),
          "",
          [],
          {"rank": 8},
          NotImplementedError,
          "Qwen3.5 LoRA",
      ),
      (
          "moe_variant",
          (1, 1),
          "",
          ["actor_model_config.model_id=Qwen/Qwen3.5-35B-A3B"],
          None,
          NotImplementedError,
          "Qwen3.5 MoE",
      ),
      (
          "disabled_random_weight_init",
          (1, 1),
          "",
          ["rollout_config.rollout_vllm_init_with_random_weights=false"],
          None,
          ValueError,
          "requires init_with_random_weights=True",
      ),
  )
  def test_qwen3p5_vllm_rollout_rejects_unsupported_configs(
      self,
      mesh_shape,
      extra_vllm_yaml,
      extra_cli,
      lora_cfg,
      exc_type,
      expected_regex,
  ):
    pipeline = _make_pipeline_with_cli_args(
        f"vllm_config:\n  hbm_utilization: 0.4\n{extra_vllm_yaml}",
        [
            "rollout_engine=vllm",
            "actor_model_config.model_id=Qwen/Qwen3.5-0.8B",
            *extra_cli,
        ],
    )
    if lora_cfg is not None:
      pipeline.config["actor_model_config"]["lora_config"] = lora_cfg
    role_to_mesh = {
        rl_engine_lib.Role.ROLLOUT: mock.Mock(
            devices=mock.Mock(shape=mesh_shape)
        )
    }
    with self.assertRaisesRegex(exc_type, expected_regex):
      pipeline.create_rollout_config(role_to_mesh=role_to_mesh)

  @parameterized.named_parameters(
      (
          "missing_hf_overrides_sets_default",
          omegaconf.MISSING,
          {"architectures": ["Qwen3_5ForCausalLM"]},
      ),
      (
          "null_hf_overrides_sets_default",
          None,
          {"architectures": ["Qwen3_5ForCausalLM"]},
      ),
      (
          "caller_extra_keys_merge_with_default_architecture",
          {"max_position_embeddings": 8192},
          {
              "max_position_embeddings": 8192,
              "architectures": ["Qwen3_5ForCausalLM"],
          },
      ),
      (
          "caller_architecture_is_preserved",
          {"architectures": ["CustomQwen3_5ForCausalLM"]},
          {"architectures": ["CustomQwen3_5ForCausalLM"]},
      ),
  )
  def test_qwen3p5_vllm_rollout_merges_hf_overrides_without_mutating_config(
      self, initial_hf_overrides, expected_hf_overrides
  ):
    pipeline = _make_pipeline_with_cli_args(
        "vllm_config:\n  hbm_utilization: 0.4\n  tensor_parallel_size: -1\n",
        [
            "rollout_engine=vllm",
            "actor_model_config.model_id=Qwen/Qwen3.5-0.8B",
        ],
    )
    if initial_hf_overrides is not omegaconf.MISSING:
      caller_dict = (
          dict(initial_hf_overrides)
          if isinstance(initial_hf_overrides, dict)
          else None
      )
      pipeline.config["vllm_config"]["kwargs"] = {"hf_overrides": caller_dict}
    role_to_mesh = {
        rl_engine_lib.Role.ROLLOUT: mock.Mock(devices=mock.Mock(shape=(1, 1)))
    }
    rollout_cfg = pipeline.create_rollout_config(role_to_mesh=role_to_mesh)
    self.assertEqual(rollout_cfg.tensor_parallel_size, -1)
    self.assertEqual(
        rollout_cfg.rollout_vllm_kwargs["hf_overrides"], expected_hf_overrides
    )
    if isinstance(initial_hf_overrides, dict):
      self.assertEqual(
          pipeline.config["vllm_config"]["kwargs"]["hf_overrides"],
          initial_hf_overrides,
      )

  def test_qwen3p5_vllm_detection_keys_on_actor_model_not_local_model_version(
      self,
  ):
    def _mesh(shape):
      return {
          rl_engine_lib.Role.ROLLOUT: mock.Mock(devices=mock.Mock(shape=shape))
      }

    # Local `vllm_config.model_version` without family token still triggers
    # Qwen3.5 validation and `hf_overrides` from `actor_model_config.model_id`.
    qwen3p5_pipeline = _make_pipeline_with_cli_args(
        'vllm_config:\n  model_version: "/scratch/ckpts/local_ckpt"\n'
        "  hbm_utilization: 0.4\n",
        [
            "rollout_engine=vllm",
            "actor_model_config.model_id=Qwen/Qwen3.5-0.8B",
        ],
    )
    with self.assertRaisesRegex(
        NotImplementedError, "tensor_parallel_size=1 only"
    ):
      qwen3p5_pipeline.create_rollout_config(role_to_mesh=_mesh((1, 2)))
    self.assertEqual(
        qwen3p5_pipeline.create_rollout_config(
            role_to_mesh=_mesh((1, 1))
        ).rollout_vllm_kwargs["hf_overrides"],
        {"architectures": ["Qwen3_5ForCausalLM"]},
    )

    # A `qwen3.5` parent directory does not misclassify an actor `Qwen3-8B`
    # checkpoint, and `model_source="maxtext"` bypasses Tunix-only guards.
    qwen3_cfg = _make_pipeline_with_cli_args(
        'vllm_config:\n  model_version: "/scratch/qwen3.5_exp/my-finetune"\n'
        "  hbm_utilization: 0.4\n",
        [
            "rollout_engine=vllm",
            "actor_model_config.model_id=/ckpt/qwen3.5-baselines/Qwen3-8B",
        ],
    ).create_rollout_config(role_to_mesh=_mesh((1, 4)))
    self.assertEqual(qwen3_cfg.tensor_parallel_size, 4)
    self.assertNotIn("hf_overrides", qwen3_cfg.rollout_vllm_kwargs)

    maxtext_cfg = _make_pipeline_with_cli_args(
        "vllm_config:\n  hbm_utilization: 0.4\n",
        [
            "rollout_engine=vllm",
            "actor_model_config.model_source=maxtext",
            "actor_model_config.model_id=Qwen/Qwen3.5-35B-A3B",
        ],
    ).create_rollout_config(role_to_mesh=_mesh((1, 4)))
    self.assertEqual(maxtext_cfg.tensor_parallel_size, 4)

  def test_qwen3p5_frozenlake_yaml_loads_and_builds_configs(self):
    cfg_path = _REPO_ROOT / "examples/frozenlake/configs/qwen3p5_0p8b.yaml"
    self.assertTrue(cfg_path.exists(), f"Missing config file: {cfg_path}")

    tokenizer = mock.MagicMock(bos_token="<bos>", eos_token="<eos>")
    with mock.patch.dict(os.environ, {"HF_TOKEN": "fake"}):
      pipeline = DummyPipeline([
          "",
          "base_agentic_config.yaml",
          f"override_config_file={cfg_path}",
      ])
    self.assertEqual(
        pipeline.config["actor_model_config"]["load_dtype"], "float32"
    )
    self.assertEqual(
        pipeline.config["reference_model_config"]["load_dtype"], "float32"
    )
    self.assertEqual(pipeline.config["actor_model_config"]["dtype"], "bfloat16")
    chat_parser = pipeline._create_chat_parser(tokenizer)
    self.assertIsInstance(chat_parser, chat_parser_lib.QwenChatTemplateParser)
    self.assertIs(chat_parser.enable_thinking, False)
    base_rl_pipeline.BasePipeline.compute_params(pipeline, list(range(320)))
    training_cfg = pipeline.create_rl_training_config()
    self.assertEqual(training_cfg.max_steps, 5)
    self.assertIsNotNone(training_cfg.actor_optimizer)
    role_to_mesh = {
        rl_engine_lib.Role.ROLLOUT: mock.Mock(devices=mock.Mock(shape=(1, 1)))
    }
    rollout_cfg = pipeline.create_rollout_config(role_to_mesh=role_to_mesh)
    self.assertEqual(rollout_cfg.tensor_parallel_size, 1)
    self.assertEqual(
        rollout_cfg.rollout_vllm_kwargs["hf_overrides"],
        {"architectures": ["Qwen3_5ForCausalLM"]},
    )


if __name__ == "__main__":
  absltest.main()
