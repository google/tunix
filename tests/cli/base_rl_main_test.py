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
from unittest import mock

from absl.testing import absltest
import omegaconf
from tunix.cli import base_rl_main
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import rl_cluster as rl_cluster_lib

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

@dataclasses.dataclass(kw_only=True)
class DummyConfig(algo_config_lib.AlgorithmConfig):
  """Dummy RL algorithm config"""
  num_generations: int = 2
  num_iterations: int = 1
  episode_timeout: int = 1
  max_response_length: int = 1000

class DummyPipeline(base_rl_main.BasePipeline):
  def __init__(self, argv: list[str], **kwargs):
    self.data_module: types.ModuleType | None = None
    super().__init__(argv, **kwargs)

  def _is_agentic_mode(self, mode:str) -> bool:
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
    valid = {f.name for f in dataclasses.fields(DummyConfig)}
    cfg.pop("max_turns", None)
    
    return DummyConfig(**{k: v for k, v in cfg.items() if k in valid})
  
  def create_rl_cluster(self):
    pass

  def compute_params(self):
    pass

  def _run(self):
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
sglang_jax_config:
  mem_fraction_static: 0.8
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
sglang_jax_config:
  mem_fraction_static: 0.8
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
sglang_jax_config:
  mem_fraction_static: 0.8
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
sglang_jax_config:
  mem_fraction_static: 0.8
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
sglang_jax_config:
  mem_fraction_static: 0.8
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
sglang_jax_config:
  mem_fraction_static: 0.8
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
sglang_jax_config:
  mem_fraction_static: 0.8
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
sglang_jax_config:
  mem_fraction_static: 0.8
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

    with mock.patch.object(base_rl_main.jax, "devices", return_value=fake_devices):
      with mock.patch.object(
          base_rl_main.jax.sharding, "Mesh", side_effect=FakeMesh
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
sglang_jax_config:
  mem_fraction_static: 0.8
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

    with mock.patch.object(base_rl_main.jax, "devices", return_value=fake_devices):
      with mock.patch.object(
          base_rl_main.mesh_lib,
          "allocate_named_mesh_device_slices",
          return_value={
              "actor_model_config": [0, 1],
              "rollout_model_config": [2, 3],
          },
      ) as allocate_mock, mock.patch.object(
          base_rl_main.mesh_lib,
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
