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

"""Tests for the shared distributed RL orchestrator node."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import contextlib
import dataclasses
import importlib
import pickle
import signal
import sys
import threading
from unittest import mock

from absl.testing import absltest
import grain
import jax
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import otel_setup
from tunix.dist_rl.common import timeouts
from tunix.dist_rl.distributed.runtime import context as runtime_context
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.launch.nodes import orchestrator
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import health_monitor
from tunix.dist_rl.orchestrator import orchestrator as cluster_orchestrator
from tunix.dist_rl.orchestrator import rl_program
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.weight_sync import weight_sync_coordinator
from tunix.dist_rl.worker import remote_execution
from tunix.rl import reward_manager
from tunix.sft import metrics_logger as metrics_logger_lib

# A task `data_module`: `create_dataset` takes the base `data_config` keys
# (`split`, `shuffle`, `seed`) plus `num_records`, and keeps the record order.
_DATA_MODULE = """
import grain

def create_dataset(split, shuffle, seed, num_records=2):
  del shuffle
  return grain.MapDataset.source([
      {"prompt": f"{split}-{i}", "metadata": {"env_config": {"seed": seed}}}
      for i in range(num_records)
  ])
"""


def _records_module(records: str) -> str:
  """Returns a `data_module` source whose dataset holds `records`."""
  return (
      "import grain\n\n"
      "def create_dataset(**kwargs):\n"
      "  del kwargs\n"
      f"  return grain.MapDataset.source({records})\n"
  )


# A `reward_functions` module: each public function is a batch reward fn.
_REWARD_MODULE = """
def length_reward(prompts, completions, **kwargs):
  del prompts, kwargs
  return [float(len(completion)) for completion in completions]
"""


def _config(*overrides: str) -> resolve_config.DistHyperParameters:
  """Returns the base config with `overrides` applied."""
  return resolve_config.initialize(list(overrides))


def _prompt(items: grain.MapDataset[dict[str, object]], index: int) -> object:
  """Returns the `prompt` of prompt item `index`."""
  item = items[index]
  assert item is not None  # Only a `filter` yields None elements.
  return item["prompt"]


class _FakeDiscovery(runtime_context.DiscoveryContext):
  """Registers the given workers as soon as the orchestrator subscribes.

  Every worker sends `config_sha256` as its config fingerprint, or no
  fingerprint when it is None.
  """

  def __init__(
      self,
      service_types: Sequence[str] = (),
      config_sha256: str | None = "",
  ) -> None:
    self._service_types = service_types
    self._config_sha256 = config_sha256

  def on_register(self, callback: Callable[[str, int, bytes], None]) -> None:
    for i, service_type in enumerate(self._service_types):
      metadata: dict[str, object] = {
          "service_type": service_type,
          "service_port": 50000 + i,
          "worker_id": f"{service_type}-{i}",
      }
      if self._config_sha256 is not None:
        metadata[node_utils.CONFIG_SHA256_METADATA_KEY] = self._config_sha256
      callback("localhost", 0, pickle.dumps(metadata))


class _ThreadedDiscovery(_FakeDiscovery):
  """Registers from another thread and drops callback errors, like gRPC."""

  def on_register(self, callback: Callable[[str, int, bytes], None]) -> None:
    def _register() -> None:
      try:
        _FakeDiscovery.on_register(self, callback)
      except ValueError:
        pass  # gRPC returns the error to the registering worker.

    threading.Thread(target=_register, daemon=True).start()


class _FakeIpc(runtime_context.IpcContext):

  def __init__(self, discovery: runtime_context.DiscoveryContext) -> None:
    self._discovery = discovery

  @property
  def discovery(self) -> runtime_context.DiscoveryContext:
    return self._discovery


class _FakeProcessContext(runtime_context.ProcessContext):

  def __init__(self, discovery: runtime_context.DiscoveryContext) -> None:
    self._ipc = _FakeIpc(discovery)

  @property
  def ipc(self) -> runtime_context.IpcContext:
    return self._ipc


@dataclasses.dataclass
class _FakeTokenizer:
  """A `node_utils.HfTokenizer` with fixed special-token ids."""

  pad_token_id: int | None
  eos_token_id: int | None
  pad_token: str | None = None
  eos_token: str | None = None

  def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
    del text, add_special_tokens  # Unused.
    return []


class _RecordingMonitor(health_monitor.HealthMonitor):
  """Records `close` in the owning `_FakeCluster`'s call log."""

  def __init__(
      self, registry: worker_registry.WorkerRegistry, calls: list[str]
  ) -> None:
    super().__init__(registry)
    self._calls = calls

  def close(self) -> None:
    self._calls.append("monitor.close")
    super().close()


class _FakeCluster(cluster_orchestrator.ClusterOrchestrator):
  """Records the lifecycle calls made on it; starts and runs no workers."""

  def __init__(
      self,
      run_error: BaseException | None = None,
      *,
      trajectory_store_config: dict[str, object] | None = None,
      on_run: Callable[[], None] | None = None,
  ) -> None:
    self.calls: list[str] = []
    registry = worker_registry.WorkerRegistry()
    super().__init__(
        registry=registry,
        monitor=_RecordingMonitor(registry, self.calls),
        trajectory_store_config=trajectory_store_config,
    )
    self.programs: list[rl_program.RLProgram] = []
    self.run_kwargs: dict[str, object] = {}
    self._run_error = run_error
    self._on_run = on_run

  def bring_up_workers(self, dummy_data: object = None) -> None:
    del dummy_data  # Unused.
    self.calls.append("bring_up_workers")

  def run(
      self,
      program: rl_program.RLProgram,
      bring_up: bool = True,
      dummy_data: object = None,
      **kwargs: object,
  ) -> None:
    del dummy_data  # Unused.
    self.calls.append("run")
    self.programs.append(program)
    self.run_kwargs = {"bring_up": bring_up, **kwargs}
    if self._on_run is not None:
      self._on_run()
    if self._run_error is not None:
      raise self._run_error

  def shutdown(self, *, abort: bool = False) -> None:
    # `ClusterOrchestrator.shutdown` without the `stop` RPCs to fake workers.
    self.calls.append("shutdown(abort=True)" if abort else "shutdown")
    _close_keeping_workers(self)


def _close_keeping_workers(
    cluster: cluster_orchestrator.ClusterOrchestrator,
) -> None:
  """Releases a test cluster without sending `stop` to its fake workers."""
  cluster.registry.close()
  cluster.monitor.close()
  if cluster.trajectory_store is not None:
    cluster.trajectory_store.close()


class BuildersTest(absltest.TestCase):

  def test_grpo_config_forwards_agentic_grpo_config(self):
    # The base config scores with the environment rewards.
    self.assertEqual(
        orchestrator.build_grpo_config(_config()).reward_mode, "env"
    )
    config = orchestrator.build_grpo_config(
        _config(
            "agentic_grpo_config.num_generations=4",
            "agentic_grpo_config.beta=0.0",
            "agentic_grpo_config.epsilon=0.1",
            "agentic_grpo_config.epsilon_high=0.3",
            "agentic_grpo_config.loss_algo=gspo-token",
            "agentic_grpo_config.loss_agg_mode=token-mean",
            "agentic_grpo_config.kl_loss_mode=kl",
            "agentic_grpo_config.advantage_estimator=rloo",
            "agentic_grpo_config.sampler_is=null",
            "agentic_grpo_config.sampler_is_threshold=3.0",
            "agentic_grpo_config.reward_mode=reward_fn",
            "reward_functions=['tunix/cli/reward_fn/gsm8k.py']",
            "agentic_grpo_config.use_rollout_logps=false",
        )
    )
    self.assertEqual(config.num_generations, 4)
    self.assertEqual(config.beta, 0.0)
    self.assertEqual(config.epsilon, 0.1)
    self.assertEqual(config.epsilon_high, 0.3)
    self.assertEqual(config.loss_algo, "gspo-token")
    self.assertEqual(config.loss_agg_mode, "token-mean")
    self.assertEqual(config.kl_loss_mode, "kl")
    self.assertEqual(config.advantage_estimator, "rloo")
    self.assertIsNone(config.sampler_is)
    self.assertEqual(config.sampler_is_threshold, 3.0)
    self.assertEqual(config.reward_mode, "reward_fn")
    self.assertFalse(config.use_rollout_logps)

  def test_grpo_config_forwards_typed_routed_experts_settings(self):
    config = orchestrator.build_grpo_config(
        _config(
            "vllm_config.routed_experts_encode_mode=uint10",
            "vllm_config.routed_experts_num_experts=256",
        )
    )
    self.assertEqual(config.routed_experts_encode_mode, "uint10")
    self.assertEqual(config.routed_experts_num_experts, 256)

  def test_batch_config(self):
    config = orchestrator.build_batch_config(
        _config(
            "rl_training_config.mini_batch_size=4",
            "rl_training_config.train_micro_batch_size=2",
            "rollout_config.max_prompt_length=64",
            "rollout_config.total_generation_steps=32",
            "rl_training_config.max_segments_per_packed_row=3",
            "dist_training_config.segment_alignment_boundary=64",
        ),
        pad_id=7,
        eos_id=9,
    )
    self.assertEqual(
        config,
        batch_assembly.BatchConfig(
            pad_id=7,
            eos_id=9,
            max_prompt_length=64,
            max_response_length=32,
            max_seq_token_per_tpu=None,
            max_segments_per_packed_row=3,
            segment_alignment_boundary=64,
            num_data_shards=1,
            mini_batch_size=4,
            train_micro_batch_size=2,
        ),
    )

  def test_batch_config_packed_len_follows_sequence_packing_budget(self):
    hp = _config("rl_training_config.max_seq_tokens_per_device=4096")
    config = orchestrator.build_batch_config(hp, pad_id=0, eos_id=1)
    self.assertEqual(config.max_seq_tokens_per_device, 4096)
    self.assertEqual(config.max_seq_token_per_tpu, 4096)

    hp_zero = _config("rl_training_config.max_seq_tokens_per_device=0")
    config_zero = orchestrator.build_batch_config(hp_zero, pad_id=0, eos_id=1)
    self.assertIsNone(config_zero.max_seq_tokens_per_device)
    self.assertIsNone(config_zero.max_seq_token_per_tpu)

  def test_batch_config_shards_over_trainer_data_parallel_axes(self):
    hp = _config(
        "actor_model_config.mesh.shape=(4,2,2)",
        "actor_model_config.mesh.axis_names=('fsdp','tp','expert')",
    )
    config = orchestrator.build_batch_config(hp, pad_id=0, eos_id=1)
    self.assertEqual(config.num_data_shards, 8)

  def test_packing_on_an_unsharded_trainer_packs_one_row_per_batch(self):
    # A TP-only trainer mesh has one data shard. Without `num_data_shards`, the
    # assembler would pack `train_micro_batch_size` rows per batch instead.
    config = orchestrator.build_batch_config(
        _config(
            "actor_model_config.mesh.shape=(1,4)",
            "rl_training_config.train_micro_batch_size=4",
            "rl_training_config.max_seq_token_per_tpu=4096",
        ),
        pad_id=0,
        eos_id=1,
    )
    self.assertEqual(config.num_data_shards, 1)
    assembler = batch_assembly.create_batch_assembler(
        num_generations=8, batch_config=config
    )
    assert isinstance(assembler, batch_assembly.SequencePackedBatchAssembler)
    self.assertEqual(assembler.batch_size, 1)

  def test_generation_args(self):
    args = orchestrator.build_generation_args(
        _config(
            "rollout_config.temperature=0.5",
            "rollout_config.top_p=0.9",
            "rollout_config.top_k=-1",
        )
    )
    self.assertEqual(
        args,
        datatypes.GenerationArgs(
            temperature=0.5, top_p=0.9, top_k=0, return_routed_experts=False
        ),
    )
    args = orchestrator.build_generation_args(
        _config(
            "rollout_config.top_k=20", "vllm_config.return_routed_experts=true"
        )
    )
    self.assertEqual(args.top_k, 20)
    self.assertTrue(args.return_routed_experts)

  def test_metrics_logging_options(self):
    options = orchestrator.build_metrics_logging_options(
        _config(
            "rl_training_config.metrics_logging_options.log_dir=/tmp/metrics",
            "rl_training_config.metrics_logging_options.project_name=proj",
            "batch_size=8",
        )
    )
    assert isinstance(options, metrics_logger_lib.MetricsLoggerOptions)
    self.assertEqual(options.log_dir, "/tmp/metrics")
    self.assertEqual(options.project_name, "proj")
    wandb_kwargs = options.backend_kwargs["wandb"]
    assert isinstance(wandb_kwargs, dict)  # Narrows the type for pytype.
    logged = wandb_kwargs["config"]
    self.assertEqual(logged["batch_size"], 8)
    self.assertEqual(logged["rl_training_config"]["mini_batch_size"], 8)

  def test_base_weight_sync_timeouts_match_coordinator_defaults(self):
    # `episode_timeout=600` derives the default `PhaseTimeouts`, and the
    # explicit 1800s RPC deadline covers them.
    phase_timeouts, rpc_timeout_s = orchestrator.resolve_weight_sync_timeouts(
        _config()
    )
    self.assertEqual(phase_timeouts, weight_sync_coordinator.PhaseTimeouts())
    self.assertEqual(rpc_timeout_s, 1800.0)

  def test_resolve_weight_sync_timeouts_derives_null_rpc_timeout(self):
    phase_timeouts, rpc_timeout_s = orchestrator.resolve_weight_sync_timeouts(
        _config(
            "cluster_config.rpc_timeout_s=null",
            "agentic_grpo_config.episode_timeout=600",
        )
    )
    expected = weight_sync_coordinator.PhaseTimeouts.for_drain_timeout(
        timeouts.drain_timeout_for(600)
    )
    self.assertEqual(phase_timeouts, expected)
    self.assertEqual(
        rpc_timeout_s,
        max(orchestrator.MIN_DERIVED_RPC_TIMEOUT_S, expected.max_rpc_phase_s()),
    )

    phase_timeouts, rpc_timeout_s = orchestrator.resolve_weight_sync_timeouts(
        _config(
            "cluster_config.rpc_timeout_s=null",
            "agentic_grpo_config.episode_timeout=1800",
        )
    )
    self.assertGreater(phase_timeouts.pre, 1800.0)
    self.assertEqual(rpc_timeout_s, phase_timeouts.max_rpc_phase_s())

  def test_resolve_weight_sync_timeouts_validates_explicit_rpc_timeout(self):
    with self.assertRaisesRegex(ValueError, "rpc_timeout_s"):
      orchestrator.resolve_weight_sync_timeouts(
          _config(
              "cluster_config.rpc_timeout_s=1800",
              "agentic_grpo_config.episode_timeout=1800",
          )
      )
    for bad_rpc in ("0", "-1"):
      with self.subTest(rpc_timeout_s=bad_rpc):
        with self.assertRaisesRegex(ValueError, "rpc_timeout_s"):
          orchestrator.resolve_weight_sync_timeouts(
              _config(f"cluster_config.rpc_timeout_s={bad_rpc}")
          )
    _, rpc_timeout_s = orchestrator.resolve_weight_sync_timeouts(
        _config(
            "cluster_config.rpc_timeout_s=99999",
            "agentic_grpo_config.episode_timeout=600",
        )
    )
    self.assertEqual(rpc_timeout_s, 99999.0)

  def test_required_workers_adds_reference_only_for_kl_penalty(self):
    role = datatypes.Role
    self.assertEqual(
        orchestrator.required_workers(_config()),
        {role.ACTOR: 1, role.ROLLOUT: 1, role.REFERENCE: 1},
    )
    self.assertEqual(
        orchestrator.required_workers(_config("agentic_grpo_config.beta=0.0")),
        {role.ACTOR: 1, role.ROLLOUT: 1, role.REFERENCE: 0},
    )

  def test_required_workers_waits_for_every_rollout_replica(self):
    role = datatypes.Role
    for target_overrides in (
        ("cluster_config.target=k8s",),
        (
            "cluster_config.target=external",
            "cluster_config.artifact_root=/artifacts/run",
        ),
    ):
      with self.subTest(target=target_overrides[0]):
        hp = _config(
            *target_overrides,
            "agentic_grpo_config.beta=0.0",
            "cluster_config.rollout_replicas=3",
        )
        self.assertEqual(
            orchestrator.required_workers(hp),
            {role.ACTOR: 1, role.ROLLOUT: 3, role.REFERENCE: 0},
        )


class LoadPromptItemsTest(absltest.TestCase):

  def _data_module(
      self, source: str = _DATA_MODULE, name: str = "task_data"
  ) -> str:
    return self.create_tempfile(f"{name}.py", content=source).full_path

  def test_maps_records_and_repeats(self):
    hp = _config(
        f"data_module={self._data_module()}",
        "task_name=gsm8k",
        "data_config.num_records=3",
        "agentic_grpo_config.max_turns=3",
        "agentic_grpo_config.episode_timeout=60",
        "agentic_grpo_config.overlong_filter=true",
    )
    items = orchestrator.load_prompt_items(hp)
    # Repeats indefinitely; `StandardRLProgram` stops after `max_steps`.
    self.assertLen(items, sys.maxsize)
    self.assertEqual(
        [_prompt(items, i) for i in range(5)],
        ["train-0", "train-1", "train-2", "train-0", "train-1"],
    )
    # `StandardRLProgram` stamps `prompt_id` and `max_response_length`.
    self.assertEqual(
        items[3],
        {
            "prompt": "train-0",
            "task_name": "gsm8k",
            "max_turns": 3,
            "metadata": {
                "env_config": {"seed": 42},
                "episode_timeout": 60.0,
                "overlong_filter": True,
            },
        },
    )

  def test_reshuffles_each_epoch(self):
    data_module = self._data_module(
        "import grain\n\n"
        "def create_dataset(split, shuffle, seed):\n"
        "  del split, shuffle\n"
        "  records = [{'prompt': str(i)} for i in range(8)]\n"
        "  return grain.MapDataset.source(records).shuffle(seed=seed)\n"
    )
    items = orchestrator.load_prompt_items(
        _config(f"data_module={data_module}", "task_name=gsm8k")
    )
    first_epoch = [_prompt(items, i) for i in range(8)]
    second_epoch = [_prompt(items, i) for i in range(8, 16)]
    self.assertCountEqual(first_epoch, [str(i) for i in range(8)])
    self.assertCountEqual(second_epoch, first_epoch)
    self.assertNotEqual(second_epoch, first_epoch)

  def test_requires_data_module(self):
    with self.assertRaisesRegex(ValueError, "data_module must name"):
      orchestrator.load_prompt_items(_config())

  def test_rejects_dataset_chat_template(self):
    hp = _config(
        f"data_module={self._data_module()}",
        "apply_chat_template_to_dataset=true",
    )
    with self.assertRaisesRegex(ValueError, "apply_chat_template_to_dataset"):
      orchestrator.load_prompt_items(hp)

  def test_requires_grain_map_dataset(self):
    for i, returned in enumerate(
        ("[{'prompt': 'q'}]", "iter([{'prompt': 'q'}])")
    ):
      with self.subTest(returned=returned):
        data_module = self._data_module(
            "def create_dataset(**kwargs):\n  del kwargs\n  return"
            f" {returned}\n",
            name=f"data_{i}",
        )
        with self.assertRaisesRegex(
            TypeError, "must return a grain.MapDataset"
        ):
          orchestrator.load_prompt_items(_config(f"data_module={data_module}"))

  def test_rejects_empty_dataset(self):
    hp = _config(
        f"data_module={self._data_module()}", "data_config.num_records=0"
    )
    with self.assertRaisesRegex(ValueError, "empty dataset"):
      orchestrator.load_prompt_items(hp)

  def test_rejects_malformed_first_record_before_iteration(self):
    for i, (records, error, regex) in enumerate((
        # `tunix.cli` data modules emit flat columns, which this path would
        # drop.
        (
            "[{'prompts': 'q', 'answer': 'a'}]",
            ValueError,
            r"unexpected keys \['answer', 'prompts'\]",
        ),
        (
            "[{'prompt': 'q', 'generation_kwargs': {}}]",
            ValueError,
            r"unexpected keys \['generation_kwargs'\]",
        ),
        ("[{'metadata': {}}]", ValueError, "no 'prompt'"),
        ("['q']", TypeError, "must be mappings, got str"),
        ("[{'prompt': 'q', 'metadata': 'x'}]", TypeError, "'metadata' must be"),
        (
            "[{'prompt': 'q', 'metadata': {'episode_timeout': 5}}]",
            ValueError,
            r"\['episode_timeout'\]",
        ),
        (
            "[{'prompt': 'q', 'task_name': 123}]",
            TypeError,
            "'task_name' must be a str",
        ),
        (
            "[{'prompt': 'q', 'task_name': ''}]",
            ValueError,
            "'task_name' must be non-empty",
        ),
        (
            "[{'prompt': 'q', 'task_name': '   '}]",
            ValueError,
            "'task_name' must be non-empty",
        ),
    )):
      with self.subTest(records=records):
        data_module = self._data_module(
            _records_module(records), name=f"data_{i}"
        )
        with self.assertRaisesRegex(error, regex):
          orchestrator.load_prompt_items(
              _config(f"data_module={data_module}", "task_name=gsm8k")
          )

  def test_checks_later_records_as_they_are_read(self):
    data_module = self._data_module(
        _records_module("[{'prompt': 'q0'}, {'prompts': 'q1'}]")
    )
    items = orchestrator.load_prompt_items(
        _config(f"data_module={data_module}", "task_name=gsm8k")
    )
    self.assertEqual(_prompt(items, 0), "q0")
    with self.assertRaisesRegex(ValueError, "unexpected keys"):
      _ = items[1]

  def test_forwards_record_task_name_and_falls_back_to_config_task_name(self):
    data_module = self._data_module(
        _records_module(
            "[{'prompt': 'q0', 'task_name': 'deepswe'}, {'prompt': 'q1'}]"
        ),
        name="multi_task_data",
    )
    items = orchestrator.load_prompt_items(
        _config(f"data_module={data_module}", "task_name=gsm8k")
    )
    first_item = items[0]
    second_item = items[1]
    assert first_item is not None
    assert second_item is not None
    self.assertEqual(first_item["task_name"], "deepswe")
    self.assertEqual(second_item["task_name"], "gsm8k")

    # When every record provides `task_name`, config `task_name` can stay unset.
    labeled_module = self._data_module(
        _records_module("[{'prompt': 'q0', 'task_name': 'deepswe'}]"),
        name="labeled_data",
    )
    labeled_items = orchestrator.load_prompt_items(
        _config(f"data_module={labeled_module}")
    )
    labeled_first = labeled_items[0]
    assert labeled_first is not None
    self.assertEqual(labeled_first["task_name"], "deepswe")

  def test_requires_task_name_when_record_omits_task_name(self):
    data_module = self._data_module(
        _records_module("[{'prompt': 'q0'}]"),
        name="unlabeled_data",
    )
    for task_override in ("task_name=", "task_name='   '"):
      with self.subTest(task_override=task_override):
        with self.assertRaisesRegex(ValueError, "task_name must be non-empty"):
          orchestrator.load_prompt_items(
              _config(f"data_module={data_module}", task_override)
          )


class ClusterTest(absltest.TestCase):

  def _cluster(
      self, hp: resolve_config.DistHyperParameters
  ) -> cluster_orchestrator.ClusterOrchestrator:
    cluster = orchestrator.create_cluster(
        hp, weight_sync_timeouts=weight_sync_coordinator.PhaseTimeouts()
    )
    self.addCleanup(_close_keeping_workers, cluster)
    return cluster

  def test_create_cluster_forwards_settings(self):
    hp = _config("weight_sync_config.mode=in_memory")
    phase_timeouts = weight_sync_coordinator.PhaseTimeouts()
    with mock.patch.object(
        cluster_orchestrator,
        "ClusterOrchestrator",
        wraps=cluster_orchestrator.ClusterOrchestrator,
    ) as cluster_cls:
      cluster = orchestrator.create_cluster(
          hp, weight_sync_timeouts=phase_timeouts
      )
    self.addCleanup(_close_keeping_workers, cluster)
    cluster_cls.assert_called_once_with(
        weight_sync_mode="in_memory",
        trajectory_store_config=None,
        weight_sync_timeouts=phase_timeouts,
        fault_tolerance_config=datatypes.FaultToleranceConfig(enabled=True),
    )
    self.assertTrue(cluster.fault_tolerance_manager.config.enabled)
    # Workers register later, in `run_program`.
    self.assertEmpty(cluster.registry.infos())

    disabled_cluster = self._cluster(
        _config("cluster_config.rollout_fault_tolerance=false")
    )
    self.assertFalse(disabled_cluster.fault_tolerance_manager.config.enabled)

  def test_wait_for_workers_registers_with_rpc_timeout(self):
    hp = _config("agentic_grpo_config.beta=0.0")
    sha = resolve_config.config_fingerprint(hp)
    cluster = self._cluster(hp)
    orchestrator.wait_for_workers(
        hp,
        cluster,
        _FakeDiscovery(["trainer", "rollout"], config_sha256=sha),
        rpc_timeout_s=42.0,
        config_sha256=sha,
    )
    self.assertEqual(
        [info.worker_id for info in cluster.registry.infos()],
        ["rollout-1", "trainer-0"],
    )
    (handle,) = cluster.registry.handles(datatypes.Role.ACTOR, statuses=None)
    assert isinstance(handle, remote_execution.GrpcRemoteActorHandle)
    self.assertEqual(handle._rpc_timeout_s, 42.0)  # pylint: disable=protected-access

  def test_wait_for_workers_waits_for_every_rollout_replica(self):
    hp = _config(
        "cluster_config.target=k8s",
        "agentic_grpo_config.beta=0.0",
        "cluster_config.rollout_replicas=2",
        "cluster_config.init_timeout_s=0.5",
    )

    def wait(
        service_types: Sequence[str],
    ) -> cluster_orchestrator.ClusterOrchestrator:
      cluster = self._cluster(hp)
      sha = resolve_config.config_fingerprint(hp)
      orchestrator.wait_for_workers(
          hp,
          cluster,
          _FakeDiscovery(service_types, config_sha256=sha),
          rpc_timeout_s=1.0,
          config_sha256=sha,
      )
      return cluster

    with self.assertRaisesRegex(
        TimeoutError,
        r"did not register within cluster_config.init_timeout_s=0\.5s\."
        r" Required: .*'rollout'>: 2.*, registered: .*'rollout'>: 1",
    ):
      wait(["trainer", "rollout"])
    cluster = wait(["trainer", "rollout", "rollout"])
    self.assertLen(
        cluster.registry.handles(datatypes.Role.ROLLOUT, statuses=None), 2
    )

  def test_wait_for_workers_rejects_a_worker_with_another_config(self):
    hp = _config(
        "agentic_grpo_config.beta=0.0", "cluster_config.init_timeout_s=30"
    )
    sha = resolve_config.config_fingerprint(hp)
    other_sha = "0" * 64
    for discovery in (
        _FakeDiscovery(["trainer", "rollout"], config_sha256=other_sha),
        # Registered off the waiting thread, as discovery does: the wait
        # still raises the error, instead of timing out.
        _ThreadedDiscovery(["trainer", "rollout"], config_sha256=other_sha),
    ):
      with self.subTest(type(discovery).__name__):
        cluster = self._cluster(hp)
        with self.assertRaisesRegex(
            ValueError,
            "trainer worker 'trainer-0' at localhost resolved a different"
            f" config than the orchestrator \\(config_sha256 {other_sha} vs"
            f" {sha}\\)",
        ):
          orchestrator.wait_for_workers(
              hp, cluster, discovery, rpc_timeout_s=1.0, config_sha256=sha
          )
        self.assertEmpty(cluster.registry.infos())
    # A worker that sends no fingerprint is rejected too.
    with self.assertRaisesRegex(
        ValueError, "registered without a config_sha256"
    ):
      orchestrator.wait_for_workers(
          hp,
          self._cluster(hp),
          _FakeDiscovery(["trainer"], config_sha256=None),
          rpc_timeout_s=1.0,
          config_sha256=sha,
      )

  def test_wait_for_workers_raises_a_rejection_even_past_the_deadline(self):
    hp = _config(
        "agentic_grpo_config.beta=0.0", "cluster_config.init_timeout_s=0.2"
    )
    sha = resolve_config.config_fingerprint(hp)
    discovery = _FakeDiscovery()
    cluster = self._cluster(hp)
    poll = cluster.registry.wait_for_workers

    def reject_during_last_poll(
        *, min_workers: dict[datatypes.Role | str, int], timeout: float | None
    ) -> None:
      # A worker with another config registers (on discovery's thread) while
      # the only poll waits; the poll then times out at the deadline.
      (register,) = on_register.call_args.args
      with self.assertRaises(ValueError):
        register(
            "localhost",
            0,
            pickle.dumps({
                "service_type": "trainer",
                "service_port": 50000,
                "worker_id": "trainer-0",
                node_utils.CONFIG_SHA256_METADATA_KEY: "0" * 64,
            }),
        )
      poll(min_workers=min_workers, timeout=timeout)

    with (
        mock.patch.object(
            discovery, "on_register", wraps=discovery.on_register
        ) as on_register,
        mock.patch.object(
            cluster.registry,
            "wait_for_workers",
            side_effect=reject_during_last_poll,
        ),
        self.assertRaisesRegex(ValueError, "resolved a different config"),
    ):
      orchestrator.wait_for_workers(
          hp, cluster, discovery, rpc_timeout_s=1.0, config_sha256=sha
      )

  def test_wait_for_workers_rejects_a_late_worker_with_another_config(self):
    hp = _config(
        "agentic_grpo_config.beta=0.0", "cluster_config.init_timeout_s=30"
    )
    sha = resolve_config.config_fingerprint(hp)
    discovery = _FakeDiscovery(["trainer", "rollout"], config_sha256=sha)
    cluster = self._cluster(hp)
    with mock.patch.object(
        discovery, "on_register", wraps=discovery.on_register
    ) as on_register:
      orchestrator.wait_for_workers(
          hp, cluster, discovery, rpc_timeout_s=1.0, config_sha256=sha
      )
    registered = cluster.registry.infos()
    # Discovery keeps calling the callback for workers that join later.
    (register,) = on_register.call_args.args
    late_worker = pickle.dumps({
        "service_type": "rollout",
        "service_port": 50009,
        "worker_id": "rollout-9",
        node_utils.CONFIG_SHA256_METADATA_KEY: "0" * 64,
    })
    with self.assertLogs(level="WARNING") as logs:
      with self.assertRaisesRegex(
          ValueError, "rollout worker 'rollout-9' at late-host resolved"
      ):
        register("late-host", 0, late_worker)
    self.assertIn("Rejected a worker registration", logs.output[0])
    self.assertEqual(cluster.registry.infos(), registered)


class ProgramTest(absltest.TestCase):

  def _program_config(
      self, *overrides: str
  ) -> resolve_config.DistHyperParameters:
    """Returns a config whose metrics are logged to a temporary directory."""
    log_dir = self.create_tempdir().full_path
    return _config(
        f"rl_training_config.metrics_logging_options.log_dir={log_dir}",
        *overrides,
    )

  def _build_program(
      self,
      hp: resolve_config.DistHyperParameters,
      reward_fns: Sequence[reward_manager.RewardFn],
  ) -> rl_program.StandardRLProgram:
    return orchestrator.build_program(
        hp,
        algo_config=orchestrator.build_grpo_config(hp),
        reward_fns=reward_fns,
        pad_id=0,
        eos_id=1,
        prompt_items=grain.MapDataset.source([]),
        trajectory_store=None,
        metrics_logging_options=orchestrator.build_metrics_logging_options(hp),
    )

  def test_build_program(self):
    hp = self._program_config(
        "batch_size=4",
        "rl_training_config.mini_batch_size=2",
        "rl_training_config.max_steps=3",
        "agentic_grpo_config.num_generations=4",
        "agentic_grpo_config.off_policy_steps=1",
        "agentic_grpo_config.trajectory_group_order=prompt_arrival",
        "weight_sync_config.mode=in_memory",
        "dist_training_config.precompile_trainer=false",
        "vllm_config.return_routed_experts=true",
    )
    program = self._build_program(hp, reward_fns=[])
    self.addCleanup(program.close)
    self.assertEqual(program.batch_size, 4)
    self.assertEqual(program.mini_batch_size, 2)
    self.assertEqual(program.num_generations, 4)
    self.assertEqual(program.max_steps, 3)
    self.assertEqual(program.max_staleness, 1)
    self.assertIsNone(program.max_version_span)
    self.assertEqual(program.group_order, rl_program.GroupOrder.PROMPT_ARRIVAL)
    self.assertTrue(program.sync_weights)
    self.assertFalse(program.precompile_trainer)
    self.assertTrue(program.generation_args.return_routed_experts)

    hp_span = self._program_config(
        "agentic_grpo_config.off_policy_steps=1",
        "agentic_grpo_config.max_version_span=2",
        "agentic_grpo_config.trajectory_group_order=trajectory_completion",
    )
    program_span = self._build_program(hp_span, reward_fns=[])
    self.addCleanup(program_span.close)
    self.assertEqual(program_span.max_version_span, 2)
    self.assertEqual(
        program_span.group_order, rl_program.GroupOrder.TRAJECTORY_COMPLETION
    )

  def test_trajectory_logs_follow_the_metrics_logging_options(self):
    artifact_root = self.create_tempdir().full_path
    # By default, the metrics and the trajectories go under the artifact root.
    hp = _config(f"cluster_config.artifact_root={artifact_root}")
    options = orchestrator.build_metrics_logging_options(hp)
    assert options is not None  # Narrows the type for pytype.
    self.assertEqual(options.log_dir, f"{artifact_root}/metrics")
    program = self._build_program(hp, reward_fns=[])
    self.addCleanup(program.close)
    self.assertIsNotNone(program.trajectory_logger)
    # Null metrics logging options log neither, ...
    no_metrics = (
        f"cluster_config.artifact_root={artifact_root}",
        "rl_training_config.metrics_logging_options=null",
    )
    hp = _config(*no_metrics)
    self.assertIsNone(orchestrator.build_metrics_logging_options(hp))
    program = self._build_program(hp, reward_fns=[])
    self.addCleanup(program.close)
    self.assertIsNone(program.trajectory_logger)
    # ... unless a trajectory directory is set.
    hp = _config(
        *no_metrics,
        f"dist_training_config.trajectory_log_dir={artifact_root}/trajectories",
    )
    program = self._build_program(hp, reward_fns=[])
    self.addCleanup(program.close)
    self.assertIsNotNone(program.trajectory_logger)

  def test_env_reward_mode_runs_without_reward_functions(self):
    hp = self._program_config()
    # The base config sets no orchestrator-side reward functions.
    self.assertEqual(hp.obtain_reward_fn(), [])
    program = self._build_program(hp, reward_fns=hp.obtain_reward_fn())
    self.addCleanup(program.close)
    rewards = program.reward_manager(
        prompts=["p", "p"],
        completions=["a", "bbb"],
        trajectory_rewards=[1.0, 0.0],
    )["rewards"]
    self.assertEqual(list(rewards), [1.0, 0.0])

  def test_reward_fn_mode_scores_with_reward_functions(self):
    reward_module = self.create_tempfile(
        "task_rewards.py", content=_REWARD_MODULE
    )
    hp = self._program_config(
        "agentic_grpo_config.reward_mode=reward_fn",
        f"reward_functions=['{reward_module.full_path}']",
    )
    program = self._build_program(hp, reward_fns=hp.obtain_reward_fn())
    self.addCleanup(program.close)
    rewards = program.reward_manager(
        prompts=["p", "p"], completions=["a", "bbb"]
    )["rewards"]
    self.assertEqual(list(rewards), [1.0, 3.0])

  def _run_program(
      self,
      cluster: _FakeCluster,
      service_types: Sequence[str],
      *overrides: str,
  ) -> tuple[rl_program.StandardRLProgram, mock.MagicMock]:
    """Runs a real program on `cluster`; returns it and its `close` spy."""
    hp = self._program_config("agentic_grpo_config.beta=0.0", *overrides)
    sha = resolve_config.config_fingerprint(hp)
    program = self._build_program(hp, reward_fns=[])
    self.addCleanup(_close_keeping_workers, cluster)
    with mock.patch.object(program, "close", wraps=program.close) as close:
      try:
        self.assertIsNone(
            orchestrator.run_program(
                hp,
                cluster,
                program,
                discovery=_FakeDiscovery(service_types, config_sha256=sha),
                rpc_timeout_s=1.0,
                config_sha256=sha,
            )
        )
      finally:
        close.assert_called_once()
    return program, close

  def test_run_program_waits_brings_up_and_runs(self):
    cluster = _FakeCluster()
    program, _ = self._run_program(cluster, ["trainer", "rollout"])
    self.assertLen(cluster.registry.infos(), 2)
    # The caller tears the cluster down (`managed_cluster_teardown`).
    self.assertEqual(cluster.calls, ["bring_up_workers", "run"])
    self.assertEqual(cluster.programs, [program])
    # `StandardRLProgram` stops after its own `max_steps`.
    self.assertEqual(cluster.run_kwargs, {"bring_up": False})

  def test_run_program_closes_program_when_workers_do_not_register(self):
    cluster = _FakeCluster()
    with self.assertRaises(TimeoutError):
      self._run_program(cluster, [], "cluster_config.init_timeout_s=0.1")
    self.assertEmpty(cluster.calls)

  def test_run_program_closes_program_on_failure(self):
    cluster = _FakeCluster(run_error=RuntimeError("worker lost"))
    with self.assertRaisesRegex(RuntimeError, "worker lost"):
      self._run_program(cluster, ["trainer", "rollout"])
    self.assertEqual(cluster.calls, ["bring_up_workers", "run"])


class MainTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # `main` defaults an unset JAX platform to CPU.
    self.addCleanup(
        jax.config.update, "jax_platforms", jax.config.jax_platforms
    )
    jax.config.update("jax_platforms", None)

  def test_base_config_selects_this_orchestrator(self):
    main_path = _config().config_mapping("cluster_config")["orchestrator_main"]
    module_name, attr = main_path.rsplit(".", 1)
    self.assertIs(
        getattr(importlib.import_module(module_name), attr), orchestrator.main
    )

  def test_requires_discovery_context(self):
    with self.assertRaisesRegex(RuntimeError, "discovery"):
      orchestrator.main([], None)

  def _recipe(self) -> str:
    """Writes a task recipe with its data and reward modules; returns its path."""
    tmp_dir = self.create_tempdir()
    data_module = tmp_dir.create_file("task_data.py", content=_DATA_MODULE)
    reward_module = tmp_dir.create_file(
        "task_rewards.py", content=_REWARD_MODULE
    )
    return tmp_dir.create_file(
        "recipe.yaml",
        content=f"""\
data_module: "{data_module.full_path}"
task_name: "gsm8k"
data_config:
  num_records: 3
reward_functions:
  - "{reward_module.full_path}"
agentic_grpo_config:
  beta: 0.0
  reward_mode: "reward_fn"
rl_training_config:
  metrics_logging_options:
    log_dir: "{tmp_dir.mkdir('metrics').full_path}"
""",
    ).full_path

  def test_runs_configured_task(self):
    argv = [
        f"override_config_file={self._recipe()}",
        "cluster_config.run_id=run_1",
    ]
    # The workers resolved the same config as the orchestrator.
    sha = resolve_config.config_fingerprint(resolve_config.initialize(argv))
    discovery = _FakeDiscovery(["trainer", "rollout"], config_sha256=sha)
    cluster = _FakeCluster()
    with (
        mock.patch.object(
            node_utils,
            "load_hf_tokenizer",
            return_value=_FakeTokenizer(pad_token_id=3, eos_token_id=7),
        ),
        mock.patch.object(
            orchestrator, "create_cluster", return_value=cluster
        ) as create_cluster,
        mock.patch.object(
            orchestrator,
            "wait_for_workers",
            wraps=orchestrator.wait_for_workers,
        ) as wait_for_workers,
        mock.patch.object(
            otel_setup, "setup_metrics", return_value=True
        ) as setup_metrics,
    ):
      orchestrator.main(argv, _FakeProcessContext(discovery))

    setup_metrics.assert_called_once_with(
        default_service_name="tunix-orchestrator",
        resource_attributes={
            "tunix.role": "orchestrator",
            "tunix.run_id": "run_1",
        },
    )

    self.assertEqual(jax.config.jax_platforms, "cpu")
    _, create_kwargs = create_cluster.call_args
    self.assertEqual(
        create_kwargs,
        {"weight_sync_timeouts": weight_sync_coordinator.PhaseTimeouts()},
    )
    (_, cluster_arg, discovery_arg), wait_kwargs = wait_for_workers.call_args
    self.assertIs(cluster_arg, cluster)
    self.assertIs(discovery_arg, discovery)
    self.assertEqual(
        wait_kwargs, {"rpc_timeout_s": 1800.0, "config_sha256": sha}
    )
    self.assertEqual(
        cluster.calls, ["bring_up_workers", "run", "shutdown", "monitor.close"]
    )
    (program,) = cluster.programs
    assert isinstance(program, rl_program.StandardRLProgram)
    self.assertEqual(program.batch_config.pad_id, 3)
    self.assertEqual(program.batch_config.eos_id, 7)
    self.assertEqual(program.algo_config.reward_mode, "reward_fn")
    # Scores with the `reward_functions` module.
    rewards = program.reward_manager(prompts=["p"], completions=["abcd"])[
        "rewards"
    ]
    self.assertEqual(list(rewards), [4.0])
    assert isinstance(program.dataset, grain.MapDataset)
    self.assertEqual(
        [_prompt(program.dataset, i) for i in range(4)],
        ["train-0", "train-1", "train-2", "train-0"],
    )

  def test_config_errors_fail_before_the_tokenizer_and_the_cluster(self):
    flat_records = self.create_tempfile(
        "flat_records.py",
        content=_records_module("[{'prompts': 'q', 'answer': 'a'}]"),
    )
    for argv, regex in (
        ([], "data_module must name"),
        ([f"data_module={flat_records.full_path}"], "unexpected keys"),
    ):
      with self.subTest(argv=argv):
        with (
            mock.patch.object(
                node_utils, "load_hf_tokenizer"
            ) as load_tokenizer,
            mock.patch.object(orchestrator, "create_cluster") as create_cluster,
            self.assertRaisesRegex(ValueError, regex),
        ):
          orchestrator.main(argv, _FakeProcessContext(_FakeDiscovery()))
        load_tokenizer.assert_not_called()
        create_cluster.assert_not_called()

  def test_keeps_workers_running_and_closes_local_resources_on_clean_exit(self):
    recipe = self._recipe()
    for with_store in (False, True):
      with self.subTest(with_store=with_store):
        argv = [
            f"override_config_file={recipe}",
            "cluster_config.run_id=run_1",
            "cluster_config.stop_workers_on_exit=false",
        ]
        store_cfg: dict[str, object] | None = (
            {"backend": "memory"} if with_store else None
        )
        sha = resolve_config.config_fingerprint(resolve_config.initialize(argv))
        cluster = _FakeCluster(trajectory_store_config=store_cfg)
        store = cluster.trajectory_store
        with (
            contextlib.ExitStack() as stack,
            mock.patch.object(
                node_utils,
                "load_hf_tokenizer",
                return_value=_FakeTokenizer(pad_token_id=0, eos_token_id=1),
            ),
            mock.patch.object(
                orchestrator, "create_cluster", return_value=cluster
            ),
            mock.patch.object(otel_setup, "setup_metrics", return_value=False),
        ):
          if store is not None:
            stack.enter_context(
                mock.patch.object(
                    store,
                    "close",
                    side_effect=lambda rc=store.close, calls=cluster.calls: (
                        calls.append("trajectory_store.close"),
                        rc(),
                    ),
                )
            )
          orchestrator.main(
              argv,
              _FakeProcessContext(
                  _FakeDiscovery(["trainer", "rollout"], config_sha256=sha)
              ),
          )
        expected = ["bring_up_workers", "run", "monitor.close"]
        if with_store:
          expected.append("trajectory_store.close")
        self.assertEqual(cluster.calls, expected)

  def test_aborts_cluster_on_startup_or_run_failure(self):
    recipe = self._recipe()
    for (
        name,
        extra_argv,
        service_types,
        build_error,
        run_error,
        expected_error,
        expected_prefix,
    ) in (
        # The program is built before waiting for the workers.
        (
            "program build fails",
            [],
            ["trainer", "rollout"],
            ValueError("bad program"),
            None,
            ValueError,
            [],
        ),
        (
            "workers do not register",
            ["cluster_config.init_timeout_s=0.1"],
            [],
            None,
            None,
            TimeoutError,
            [],
        ),
        (
            "program run fails",
            [],
            ["trainer", "rollout"],
            None,
            RuntimeError("worker lost"),
            RuntimeError,
            ["bring_up_workers", "run"],
        ),
    ):
      for stop_on_exit in (True, False):
        with self.subTest(name=name, stop_workers_on_exit=stop_on_exit):
          argv = [
              f"override_config_file={recipe}",
              "cluster_config.run_id=run_1",
              f"cluster_config.stop_workers_on_exit={str(stop_on_exit).lower()}",
              *extra_argv,
          ]
          sha = resolve_config.config_fingerprint(
              resolve_config.initialize(argv)
          )
          store_cfg: dict[str, object] | None = (
              None if stop_on_exit else {"backend": "memory"}
          )
          cluster = _FakeCluster(
              run_error=run_error, trajectory_store_config=store_cfg
          )
          store = cluster.trajectory_store
          with (
              contextlib.ExitStack() as stack,
              mock.patch.object(
                  node_utils,
                  "load_hf_tokenizer",
                  return_value=_FakeTokenizer(pad_token_id=0, eos_token_id=1),
              ),
              mock.patch.object(
                  orchestrator, "create_cluster", return_value=cluster
              ),
              mock.patch.object(
                  orchestrator,
                  "build_program",
                  wraps=orchestrator.build_program,
                  side_effect=build_error,
              ),
              mock.patch.object(
                  otel_setup, "setup_metrics", return_value=False
              ),
              self.assertRaises(expected_error),
          ):
            if store is not None:
              stack.enter_context(
                  mock.patch.object(
                      store,
                      "close",
                      side_effect=lambda rc=store.close, calls=cluster.calls: (
                          calls.append("trajectory_store.close"),
                          rc(),
                      ),
                  )
              )
            orchestrator.main(
                argv,
                _FakeProcessContext(
                    _FakeDiscovery(service_types, config_sha256=sha)
                ),
            )
          expected_teardown = (
              ["shutdown(abort=True)", "monitor.close"]
              if stop_on_exit
              else ["monitor.close", "trajectory_store.close"]
          )
          self.assertEqual(
              cluster.calls, [*expected_prefix, *expected_teardown]
          )

  def test_sigterm_during_run_program_aborts_cluster_and_exits_143(self):
    argv = [
        f"override_config_file={self._recipe()}",
        "cluster_config.run_id=run_1",
    ]
    sha = resolve_config.config_fingerprint(resolve_config.initialize(argv))
    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    cluster = _FakeCluster(
        on_run=lambda: signal.raise_signal(signal.SIGTERM)
    )
    with (
        mock.patch.object(
            node_utils,
            "load_hf_tokenizer",
            return_value=_FakeTokenizer(pad_token_id=0, eos_token_id=1),
        ),
        mock.patch.object(orchestrator, "create_cluster", return_value=cluster),
        mock.patch.object(otel_setup, "setup_metrics", return_value=False),
        self.assertRaises(SystemExit) as cm,
    ):
      orchestrator.main(
          argv,
          _FakeProcessContext(
              _FakeDiscovery(["trainer", "rollout"], config_sha256=sha)
          ),
      )
    self.assertEqual(cm.exception.code, 128 + signal.SIGTERM)
    self.assertEqual(
        cluster.calls,
        ["bring_up_workers", "run", "shutdown(abort=True)", "monitor.close"],
    )
    self.assertEqual(signal.getsignal(signal.SIGINT), prev_sigint)
    self.assertEqual(signal.getsignal(signal.SIGTERM), prev_sigterm)


if __name__ == "__main__":
  absltest.main()
