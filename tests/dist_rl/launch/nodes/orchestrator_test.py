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
import dataclasses
import importlib
import pickle
import sys
from unittest import mock

from absl.testing import absltest
import grain
import jax
from tunix.dist_rl.common import datatypes
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
  """Registers the given workers as soon as the orchestrator subscribes."""

  def __init__(self, service_types: Sequence[str] = ()) -> None:
    self._service_types = service_types

  def on_register(self, callback: Callable[[str, int, bytes], None]) -> None:
    for i, service_type in enumerate(self._service_types):
      metadata = {
          "service_type": service_type,
          "service_port": 50000 + i,
          "worker_id": f"{service_type}-{i}",
      }
      callback("localhost", 0, pickle.dumps(metadata))


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

  def __init__(self, run_error: Exception | None = None) -> None:
    self.calls: list[str] = []
    registry = worker_registry.WorkerRegistry()
    super().__init__(
        registry=registry, monitor=_RecordingMonitor(registry, self.calls)
    )
    self.programs: list[rl_program.RLProgram] = []
    self.run_kwargs: dict[str, object] = {}
    self._run_error = run_error

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
    if self._run_error is not None:
      raise self._run_error

  def shutdown(self) -> None:
    # `ClusterOrchestrator.shutdown` without the `stop` RPCs to fake workers.
    self.calls.append("shutdown")
    _close_keeping_workers(self)


def _close_keeping_workers(
    cluster: cluster_orchestrator.ClusterOrchestrator,
) -> None:
  """Releases a test cluster without sending `stop` to its fake workers."""
  cluster.registry.close()
  cluster.monitor.close()


class BuildersTest(absltest.TestCase):

  def test_grpo_config_forwards_agentic_grpo_config(self):
    # The base config scores with the environment rewards.
    self.assertEqual(
        orchestrator.build_grpo_config(_config()).reward_mode, "env"
    )
    config = orchestrator.build_grpo_config(
        _config(
            "agentic_grpo_config.num_generations=4",
            "agentic_grpo_config.num_iterations=2",
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
    self.assertEqual(config.num_iterations, 2)
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

  def test_algo_uses_mini_batch_and_length_budgets(self):
    algo = orchestrator.build_algo(
        _config(
            "batch_size=8",
            "rl_training_config.mini_batch_size=4",
            "rl_training_config.train_micro_batch_size=2",
            "rollout_config.max_prompt_length=100",
            "rollout_config.total_generation_steps=28",
            "agentic_grpo_config.max_turns=3",
        )
    )
    self.assertEqual(algo.mini_batch_size, 4)
    self.assertEqual(algo.train_micro_batch_size, 2)
    self.assertEqual(algo.max_turns, 3)
    self.assertEqual(algo.max_packed_len, 128)
    self.assertEqual(algo.max_response_length, 28)

  def test_algo_packed_len_follows_sequence_packing_budget(self):
    hp = _config("rl_training_config.max_seq_token_per_tpu=4096")
    self.assertEqual(orchestrator.build_algo(hp).max_packed_len, 4096)

  def test_batch_config(self):
    config = orchestrator.build_batch_config(
        _config(
            "rollout_config.max_prompt_length=64",
            "rollout_config.total_generation_steps=32",
            "rl_training_config.max_segments_per_packed_row=3",
            "dist_training_config.segment_alignment_boundary=64",
        ),
        pad_id=7,
    )
    self.assertEqual(
        config,
        batch_assembly.BatchConfig(
            pad_id=7,
            max_prompt_length=64,
            max_response_length=32,
            max_seq_token_per_tpu=None,
            max_segments_per_packed_row=3,
            segment_alignment_boundary=64,
            num_data_shards=None,
        ),
    )

  def test_batch_config_shards_over_trainer_data_parallel_axes(self):
    hp = _config(
        "actor_model_config.mesh.shape=(4,2,2)",
        "actor_model_config.mesh.axis_names=('fsdp','tp','expert')",
    )
    self.assertEqual(
        orchestrator.build_batch_config(hp, pad_id=0).num_data_shards, 8
    )

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
    self.assertIsInstance(options, metrics_logger_lib.MetricsLoggerOptions)
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


class LoadPromptItemsTest(absltest.TestCase):

  def _data_module(
      self, source: str = _DATA_MODULE, name: str = "task_data"
  ) -> str:
    return self.create_tempfile(f"{name}.py", content=source).full_path

  def test_maps_records_and_repeats(self):
    hp = _config(
        f"data_module={self._data_module()}",
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
        _config(f"data_module={data_module}")
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
    )):
      with self.subTest(records=records):
        data_module = self._data_module(
            _records_module(records), name=f"data_{i}"
        )
        with self.assertRaisesRegex(error, regex):
          orchestrator.load_prompt_items(_config(f"data_module={data_module}"))

  def test_checks_later_records_as_they_are_read(self):
    data_module = self._data_module(
        _records_module("[{'prompt': 'q0'}, {'prompts': 'q1'}]")
    )
    items = orchestrator.load_prompt_items(
        _config(f"data_module={data_module}")
    )
    self.assertEqual(_prompt(items, 0), "q0")
    with self.assertRaisesRegex(ValueError, "unexpected keys"):
      _ = items[1]


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
    hp = _config("weight_sync_config.mode=fallback")
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
        weight_sync_mode="fallback",
        trajectory_store_config=None,
        weight_sync_timeouts=phase_timeouts,
    )
    # Workers register later, in `run_program`.
    self.assertEmpty(cluster.registry.infos())

  def test_wait_for_workers_registers_with_rpc_timeout(self):
    hp = _config("agentic_grpo_config.beta=0.0")
    cluster = self._cluster(hp)
    orchestrator.wait_for_workers(
        hp, cluster, _FakeDiscovery(["trainer", "rollout"]), rpc_timeout_s=42.0
    )
    self.assertEqual(
        [info.worker_id for info in cluster.registry.infos()],
        ["rollout-1", "trainer-0"],
    )
    (handle,) = cluster.registry.handles(datatypes.Role.ACTOR)
    assert isinstance(handle, remote_execution.GrpcRemoteActorHandle)
    self.assertEqual(handle._rpc_timeout_s, 42.0)  # pylint: disable=protected-access

  def test_wait_for_workers_needs_rollout_replicas_on_k8s_only(self):
    overrides = (
        "agentic_grpo_config.beta=0.0",
        "cluster_config.k8s.rollout.replicas=2",
        "cluster_config.init_timeout_s=0.5",
    )

    def wait(
        hp: resolve_config.DistHyperParameters, service_types: Sequence[str]
    ) -> cluster_orchestrator.ClusterOrchestrator:
      cluster = self._cluster(hp)
      orchestrator.wait_for_workers(
          hp, cluster, _FakeDiscovery(service_types), rpc_timeout_s=1.0
      )
      return cluster

    # `target: local` launches a single rollout node.
    local = wait(_config(*overrides), ["trainer", "rollout"])
    self.assertLen(local.registry.handles(datatypes.Role.ROLLOUT), 1)

    k8s_hp = _config(*overrides, "cluster_config.target=k8s")
    with self.assertRaises(TimeoutError):
      wait(k8s_hp, ["trainer", "rollout"])
    k8s = wait(k8s_hp, ["trainer", "rollout", "rollout"])
    self.assertLen(k8s.registry.handles(datatypes.Role.ROLLOUT), 2)

  def test_close_cluster_stops_workers_on_exit(self):
    cluster = _FakeCluster()
    orchestrator.close_cluster(_config(), cluster)
    self.assertEqual(cluster.calls, ["shutdown", "monitor.close"])

  def test_close_cluster_keeps_workers_running(self):
    cluster = _FakeCluster()
    orchestrator.close_cluster(
        _config("cluster_config.stop_workers_on_exit=false"), cluster
    )
    self.assertEqual(cluster.calls, ["monitor.close"])


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
        algo=orchestrator.build_algo(hp),
        reward_fns=reward_fns,
        pad_id=0,
        prompt_items=grain.MapDataset.source([]),
        trajectory_store=None,
    )

  def test_build_program(self):
    hp = self._program_config(
        "batch_size=4",
        "rl_training_config.mini_batch_size=2",
        "rl_training_config.max_steps=3",
        "agentic_grpo_config.num_generations=4",
        "agentic_grpo_config.off_policy_steps=1",
        "agentic_grpo_config.trajectory_group_order=prompt_arrival",
        "weight_sync_config.mode=fallback",
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
    self.assertEqual(program.group_order, rl_program.GroupOrder.PROMPT_ARRIVAL)
    self.assertTrue(program.sync_weights)
    self.assertFalse(program.precompile_trainer)
    self.assertTrue(program.generation_args.return_routed_experts)

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
    program = self._build_program(hp, reward_fns=[])
    self.addCleanup(_close_keeping_workers, cluster)
    with mock.patch.object(program, "close", wraps=program.close) as close:
      try:
        self.assertIsNone(
            orchestrator.run_program(
                hp,
                cluster,
                program,
                discovery=_FakeDiscovery(service_types),
                rpc_timeout_s=1.0,
            )
        )
      finally:
        close.assert_called_once()
    return program, close

  def test_run_program_waits_brings_up_and_runs(self):
    cluster = _FakeCluster()
    program, _ = self._run_program(cluster, ["trainer", "rollout"])
    self.assertLen(cluster.registry.infos(), 2)
    # The caller tears the cluster down (`close_cluster`).
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
    discovery = _FakeDiscovery(["trainer", "rollout"])
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
    ):
      orchestrator.main(
          [f"override_config_file={self._recipe()}"],
          _FakeProcessContext(discovery),
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
    self.assertEqual(wait_kwargs, {"rpc_timeout_s": 1800.0})
    self.assertEqual(
        cluster.calls, ["bring_up_workers", "run", "shutdown", "monitor.close"]
    )
    (program,) = cluster.programs
    assert isinstance(program, rl_program.StandardRLProgram)
    self.assertEqual(program.batch_config.pad_id, 3)
    self.assertEqual(program.algo.algo_config.reward_mode, "reward_fn")
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

  def test_tears_down_the_cluster_when_startup_fails(self):
    recipe = self._recipe()
    for name, extra_argv, service_types, build_error, error in (
        # The program is built before waiting for the workers.
        (
            "program build fails",
            [],
            ["trainer", "rollout"],
            ValueError,
            ValueError,
        ),
        (
            "workers do not register",
            ["cluster_config.init_timeout_s=0.1"],
            [],
            None,
            TimeoutError,
        ),
    ):
      with self.subTest(name):
        cluster = _FakeCluster()
        with (
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
            self.assertRaises(error),
        ):
          orchestrator.main(
              [f"override_config_file={recipe}", *extra_argv],
              _FakeProcessContext(_FakeDiscovery(service_types)),
          )
        self.assertEqual(cluster.calls, ["shutdown", "monitor.close"])


if __name__ == "__main__":
  absltest.main()
