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

"""CPU orchestrator node shared by every distributed RL task.

The TPU worker nodes in this directory (trainer, rollout, and the optional
reference inference node) host the models. This process registers them with
`ClusterOrchestrator` and runs `StandardRLProgram`, built from the same resolved
config (`resolve_config.DistHyperParameters`) as the workers.

There is no per-task code here. A task is selected by config keys shared with
the agentic `tunix.cli` path:

* `data_module` / `data_config`: the task records, from
  `create_dataset(**data_config) -> grain.MapDataset` as in `tunix.cli`. The
  record format is specific to this path (`{"prompt", "metadata"}`, not
  `tunix.cli`'s flat columns); see `load_prompt_items`.
* `reward_functions` and `agentic_grpo_config.reward_mode`: `reward_fn` scores
  completions with `reward_functions`; `env` uses the environment rewards, plus
  any `reward_functions`.
* `env_class_path` / `agent_class_path` and their kwargs: read by the rollout
  node, which imports `cluster_config.registry_module` to register them.

Importing this module has no side effects. `main` pins the control-plane JAX
platform to CPU (unless `JAX_PLATFORMS` is set) before any JAX use.

Run via the distributed runtime, e.g.
  python -m tunix.dist_rl.distributed.runtime.main --discovery_id=... \\
    --process_main=tunix.dist_rl.launch.nodes.orchestrator.main \\
    override_config_file=<task>.yaml
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import logging

import grain
import jax
from tunix.cli.utils import data as data_lib
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import timeouts
from tunix.dist_rl.distributed.runtime import context as runtime_context
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.orchestrator import algorithm_adapter
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import orchestrator as cluster_orchestrator
from tunix.dist_rl.orchestrator import rl_program
from tunix.dist_rl.trajectory import store as trajectory_store_lib
from tunix.dist_rl.weight_sync import weight_sync
from tunix.dist_rl.weight_sync import weight_sync_coordinator
from tunix.rl import algorithm_config
from tunix.rl import reward_manager
from tunix.sft import metrics_logger as metrics_logger_lib

# Lower bound for the worker RPC deadline derived from the weight-sync phase
# timeouts when `cluster_config.rpc_timeout_s` is null.
MIN_DERIVED_RPC_TIMEOUT_S = 1800.0

# Keys a `data_module` record may hold (see `load_prompt_items`).
_RECORD_KEYS = frozenset({"prompt", "metadata"})
# Prompt-item metadata stamped from the run configuration; records must not set
# it.
_RUN_WIDE_METADATA_KEYS = frozenset({"episode_timeout", "overlong_filter"})


def build_grpo_config(
    hp: resolve_config.DistHyperParameters,
) -> algorithm_config.GRPOConfig:
  """Builds `GRPOConfig` from `agentic_grpo_config` and `vllm_config`.

  Every `agentic_grpo_config` key that is a `GRPOConfig` field is forwarded, as
  in `tunix.cli.grpo_main`; the other keys (e.g. `max_turns`,
  `off_policy_steps`) configure the program and the rollout workers. The
  routing-replay encoding comes from the typed `vllm_config` keys shared with
  the rollout node.

  Args:
    hp: Resolved distributed RL configuration.

  Returns:
    The validated `GRPOConfig`.
  """
  init_fields = {
      field.name
      for field in dataclasses.fields(algorithm_config.GRPOConfig)
      if field.init
  }
  grpo = hp.config_mapping("agentic_grpo_config")
  vllm = hp.config_mapping("vllm_config")
  return algorithm_config.GRPOConfig(
      **{key: grpo[key] for key in init_fields & grpo.keys()},
      routed_experts_encode_mode=vllm["routed_experts_encode_mode"],
      routed_experts_num_experts=vllm["routed_experts_num_experts"],
  )


def build_algo(
    hp: resolve_config.DistHyperParameters,
) -> algorithm_adapter.GRPOAdapter:
  """Builds the GRPO algorithm adapter."""
  rl_train = hp.config_mapping("rl_training_config")
  # `create_rollout_config()` needs worker meshes for `rollout_engine=vllm`; the
  # CPU orchestrator only needs the length budgets.
  rollout = hp.config_mapping("rollout_config")
  max_response_length = int(rollout["total_generation_steps"])
  max_seq_token_per_tpu = rl_train["max_seq_token_per_tpu"]
  return algorithm_adapter.GRPOAdapter(
      algo_config=build_grpo_config(hp),
      mini_batch_size=int(rl_train["mini_batch_size"]),
      train_micro_batch_size=int(rl_train["train_micro_batch_size"]),
      max_turns=int(hp.config_mapping("agentic_grpo_config")["max_turns"]),
      max_packed_len=(
          int(max_seq_token_per_tpu)
          if max_seq_token_per_tpu is not None
          else int(rollout["max_prompt_length"]) + max_response_length
      ),
      max_response_length=max_response_length,
  )


def _num_data_shards(hp: resolve_config.DistHyperParameters) -> int | None:
  """Returns the trainer data-parallel shard count, or None when unsharded."""
  axes = hp.mesh_axes("actor_model_config")
  shards = axes["fsdp"] * axes["dp"] * axes["expert"]
  return shards if shards > 1 else None


def build_batch_config(
    hp: resolve_config.DistHyperParameters, *, pad_id: int
) -> batch_assembly.BatchConfig:
  """Builds the batch-assembly config (padding and sequence packing)."""
  rl_train = hp.config_mapping("rl_training_config")
  # See `build_algo` for why the rollout section is read directly.
  rollout = hp.config_mapping("rollout_config")
  max_seq_token_per_tpu = rl_train["max_seq_token_per_tpu"]
  max_segments_per_packed_row = rl_train["max_segments_per_packed_row"]
  return batch_assembly.BatchConfig(
      pad_id=pad_id,
      max_prompt_length=int(rollout["max_prompt_length"]),
      max_response_length=int(rollout["total_generation_steps"]),
      max_seq_token_per_tpu=(
          None if max_seq_token_per_tpu is None else int(max_seq_token_per_tpu)
      ),
      max_segments_per_packed_row=(
          None
          if max_segments_per_packed_row is None
          else int(max_segments_per_packed_row)
      ),
      segment_alignment_boundary=int(
          hp.config_mapping("dist_training_config")[
              "segment_alignment_boundary"
          ]
      ),
      num_data_shards=_num_data_shards(hp),
  )


def build_generation_args(
    hp: resolve_config.DistHyperParameters,
) -> datatypes.GenerationArgs:
  """Builds the sampling arguments sent with every rollout request.

  `return_logprobs` is left unset: `StandardRLProgram` sets it from
  `GRPOConfig.use_rollout_logps`.

  Args:
    hp: Resolved distributed RL configuration.

  Returns:
    The generation arguments.
  """
  # See `build_algo` for why the rollout section is read directly.
  # TODO(noghabi): Share the sampling settings (temperature, top_p, top_k) with
  # the rollout node's `RolloutConfig` through one `node_utils` helper, so both
  # nodes use the same disabled-`top_k` convention.
  rollout = hp.config_mapping("rollout_config")
  top_k = int(rollout["top_k"])
  return datatypes.GenerationArgs(
      temperature=float(rollout["temperature"]),
      top_p=float(rollout["top_p"]),
      # The schema disables top-k with a negative value, the rollout worker
      # with 0.
      top_k=0 if top_k < 0 else top_k,
      # `StandardRLProgram` also enables routing replay on the trainer from it.
      return_routed_experts=bool(
          hp.config_mapping("vllm_config")["return_routed_experts"]
      ),
  )


def _logged_config(hp: resolve_config.DistHyperParameters) -> dict[str, object]:
  """Returns the resolved configuration as plain containers for W&B."""
  return {
      key: hp.config_mapping(key) if isinstance(value, Mapping) else value
      for key, value in hp.config.items()
  }


def build_metrics_logging_options(
    hp: resolve_config.DistHyperParameters,
) -> metrics_logger_lib.MetricsLoggerOptions:
  """Builds `MetricsLoggerOptions` from `rl_training_config`.

  Uses `obtain_training_config_dict` rather than `create_rl_training_config()`:
  the distributed `train_micro_batch_size` counts trajectories, which
  `RLTrainingConfig` validation does not accept.

  Args:
    hp: Resolved distributed RL configuration.

  Returns:
    The metrics logger options, with the resolved config logged to W&B.
  """
  options = hp.obtain_training_config_dict("rl_training_config")[
      "metrics_logging_options"
  ]
  if not isinstance(options, metrics_logger_lib.MetricsLoggerOptions):
    raise TypeError(
        "rl_training_config.metrics_logging_options must be set, got"
        f" {options!r}."
    )
  return dataclasses.replace(
      options, backend_kwargs={"wandb": {"config": _logged_config(hp)}}
  )


def _trajectory_log_dir(hp: resolve_config.DistHyperParameters) -> str | None:
  """Returns the per-step trajectory log directory, or None when unset."""
  log_dir = str(hp.config_mapping("dist_training_config")["trajectory_log_dir"])
  return log_dir or None


def _weight_sync_mode(
    hp: resolve_config.DistHyperParameters,
) -> weight_sync.WeightSyncMode:
  """Returns `weight_sync_config.mode`."""
  return weight_sync.WeightSyncMode(
      str(hp.config_mapping("weight_sync_config")["mode"])
  )


def resolve_weight_sync_timeouts(
    hp: resolve_config.DistHyperParameters,
) -> tuple[weight_sync_coordinator.PhaseTimeouts, float]:
  """Derives weight-sync phase timeouts and the worker RPC deadline.

  The rollout drain window scales with `agentic_grpo_config.episode_timeout`.
  A null `cluster_config.rpc_timeout_s` is derived from the longest RPC-backed
  phase (at least `MIN_DERIVED_RPC_TIMEOUT_S`); an explicit value must cover it.

  Args:
    hp: Resolved distributed RL configuration.

  Returns:
    `(phase_timeouts, rpc_timeout_s)`.
  """
  episode_timeout = hp.config_mapping("agentic_grpo_config")["episode_timeout"]
  phase_timeouts = weight_sync_coordinator.PhaseTimeouts.for_drain_timeout(
      timeouts.drain_timeout_for(episode_timeout)
  )
  min_rpc_s = phase_timeouts.max_rpc_phase_s()
  raw_rpc = hp.config_mapping("cluster_config")["rpc_timeout_s"]
  if raw_rpc is None:
    return phase_timeouts, max(MIN_DERIVED_RPC_TIMEOUT_S, min_rpc_s)
  rpc_timeout_s = timeouts.validate_positive_finite(
      raw_rpc, "cluster_config.rpc_timeout_s"
  )
  if rpc_timeout_s < min_rpc_s:
    raise ValueError(
        f"cluster_config.rpc_timeout_s ({rpc_timeout_s}) must be >="
        f" weight_sync_timeouts.max_rpc_phase_s() ({min_rpc_s})."
    )
  return phase_timeouts, rpc_timeout_s


def load_prompt_items(
    hp: resolve_config.DistHyperParameters,
) -> grain.MapDataset[dict[str, object]]:
  """Loads the task records from `data_module` as endless rollout prompt items.

  `data_module` provides `create_dataset(**data_config) -> grain.MapDataset`, as
  in the agentic `tunix.cli` path. Each record is a mapping with a `prompt` and
  an optional `metadata`; other keys are rejected instead of being dropped by
  `DistributedRLEngine`. Reward functions get their inputs from `metadata`, the
  environment its settings from `metadata.env_config`, and the agent from
  `metadata.agent_config`. Environments build the model's first turn from
  `metadata.env_config`, so `prompt` only fills `RolloutRequest.prompt` (`""`
  is fine).

  The dataset repeats indefinitely, and an upstream `shuffle` reshuffles each
  epoch. `StandardRLProgram` stops dispatching after `max_steps` and stamps each
  item's `prompt_id` and `max_response_length`. The dataset and its first record
  are checked here, so a missing module or a malformed dataset fails before the
  orchestrator waits for workers.

  Args:
    hp: Resolved distributed RL configuration.

  Returns:
    The prompt items, repeated indefinitely.

  Raises:
    ValueError: If `data_module` is empty, `apply_chat_template_to_dataset` is
      set, the dataset is empty, or its first record is malformed.
    TypeError: If `create_dataset` does not return a `grain.MapDataset`, or the
      first record or its `metadata` is not a mapping.
  """
  data_module = str(hp.config["data_module"])
  if not data_module:
    raise ValueError(
        "data_module must name the task's dataset module, which provides"
        " create_dataset(**data_config) -> grain.MapDataset."
    )
  if hp.config["apply_chat_template_to_dataset"]:
    raise ValueError(
        "apply_chat_template_to_dataset must be false: the rollout workers"
        " apply the chat template (chat_parser_config)."
    )
  dataset = data_lib.get_dataset_from_module(
      data_module, **hp.config_mapping("data_config")
  )
  if not isinstance(dataset, grain.MapDataset):
    raise TypeError(
        f"data_module {data_module!r} returned a {type(dataset).__name__};"
        " create_dataset must return a grain.MapDataset."
    )
  if not len(dataset):
    raise ValueError(f"data_module {data_module!r} returned an empty dataset.")
  logging.info("Loaded %d task records from %s.", len(dataset), data_module)
  grpo = hp.config_mapping("agentic_grpo_config")
  max_turns = int(grpo["max_turns"])
  episode_timeout = float(grpo["episode_timeout"])
  overlong_filter = bool(grpo["overlong_filter"])

  # TODO(noghabi): Unify the rollout env construction
  # (`RolloutManager`: `env_cls(**metadata["env_config"])`) with the agentic
  # learner's (`AgenticRLLearner`: `env_class(single_example, **env_kwargs)`) so
  # `tunix.cli` data modules work unchanged, and drop `prompt` from records if
  # `RolloutRequest.prompt` stays unused. Also have `StandardRLProgram` stamp
  # `max_turns` (from `GRPOAdapter.max_turns`) and the rollout worker take
  # `episode_timeout` / `overlong_filter` from its config, so prompt items carry
  # only task data.
  def to_prompt_item(record: object) -> dict[str, object]:
    if not isinstance(record, Mapping):
      raise TypeError(
          f"Dataset records must be mappings, got {type(record).__name__}."
      )
    unexpected = sorted(set(record) - _RECORD_KEYS)
    if unexpected:
      raise ValueError(
          f"Dataset record has unexpected keys {unexpected}; a record holds"
          " only 'prompt' and 'metadata'. Put reward function inputs in"
          " 'metadata' and environment settings in 'metadata.env_config'."
      )
    if "prompt" not in record:
      raise ValueError("Dataset record has no 'prompt'.")
    metadata = record["metadata"] if "metadata" in record else {}
    if not isinstance(metadata, Mapping):
      raise TypeError(
          "Dataset record 'metadata' must be a mapping, got"
          f" {type(metadata).__name__}."
      )
    run_wide = sorted(_RUN_WIDE_METADATA_KEYS & set(metadata))
    if run_wide:
      raise ValueError(
          f"Dataset record metadata sets {run_wide}, which come from the run"
          " configuration."
      )
    return {
        "prompt": record["prompt"],
        "max_turns": max_turns,
        "metadata": {
            **metadata,
            "episode_timeout": episode_timeout,
            "overlong_filter": overlong_filter,
        },
    }

  prompt_items = dataset.map(to_prompt_item)
  _ = prompt_items[0]  # Checks the record format before the run starts.
  return prompt_items.repeat()


def required_workers(
    hp: resolve_config.DistHyperParameters,
) -> dict[datatypes.Role | str, int]:
  """Returns the number of workers per role to wait for.

  `cluster_config.target: k8s` launches `cluster_config.k8s.rollout.replicas`
  rollout nodes; `target: local` launches one. The reference inference worker
  is only needed for a KL penalty (`agentic_grpo_config.beta != 0`).

  Args:
    hp: Resolved distributed RL configuration.

  Returns:
    The minimum worker count per role.
  """
  cluster = hp.config_mapping("cluster_config")
  num_rollouts = (
      int(cluster["k8s"]["rollout"]["replicas"])
      if cluster["target"] == "k8s"
      else 1
  )
  beta = float(hp.config_mapping("agentic_grpo_config")["beta"])
  return {
      datatypes.Role.ACTOR: 1,
      datatypes.Role.ROLLOUT: num_rollouts,
      datatypes.Role.REFERENCE: 1 if beta != 0.0 else 0,
  }


def create_cluster(
    hp: resolve_config.DistHyperParameters,
    *,
    weight_sync_timeouts: weight_sync_coordinator.PhaseTimeouts,
) -> cluster_orchestrator.ClusterOrchestrator:
  """Creates the `ClusterOrchestrator`; the workers register in `run_program`."""
  return cluster_orchestrator.ClusterOrchestrator(
      weight_sync_mode=_weight_sync_mode(hp).value,
      trajectory_store_config=node_utils.trajectory_store_config(hp),
      weight_sync_timeouts=weight_sync_timeouts,
  )


def wait_for_workers(
    hp: resolve_config.DistHyperParameters,
    cluster: cluster_orchestrator.ClusterOrchestrator,
    discovery: runtime_context.DiscoveryContext,
    *,
    rpc_timeout_s: float,
) -> None:
  """Registers the workers announced through `discovery` until all are up.

  Args:
    hp: Resolved distributed RL configuration.
    cluster: Cluster orchestrator to register the workers with.
    discovery: Discovery API the workers register through.
    rpc_timeout_s: Deadline for worker RPCs.

  Raises:
    TimeoutError: If `required_workers(hp)` do not register within
      `cluster_config.init_timeout_s`.
  """

  def register(hostname: str, port: int, metadata: bytes) -> None:
    cluster.registry.register_from_hostname(
        hostname, port, metadata, rpc_timeout_s=rpc_timeout_s
    )

  discovery.on_register(register)
  min_workers = required_workers(hp)
  init_timeout_s = hp.config_mapping("cluster_config")["init_timeout_s"]
  logging.info("Waiting for workers to register: %s", min_workers)
  cluster.registry.wait_for_workers(
      min_workers=min_workers,
      timeout=None if init_timeout_s is None else float(init_timeout_s),
  )
  logging.info("Registered workers: %s", cluster.registry.infos())


def build_program(
    hp: resolve_config.DistHyperParameters,
    *,
    algo: algorithm_adapter.GRPOAdapter,
    reward_fns: Sequence[reward_manager.RewardFn],
    pad_id: int,
    prompt_items: grain.MapDataset[dict[str, object]],
    trajectory_store: trajectory_store_lib.TrajectoryStore | None,
) -> rl_program.StandardRLProgram:
  """Builds the `StandardRLProgram` from the configuration.

  Args:
    hp: Resolved distributed RL configuration.
    algo: GRPO algorithm adapter (see `build_algo`).
    reward_fns: Orchestrator-side reward functions (`reward_functions`).
    pad_id: Tokenizer padding id for batch assembly.
    prompt_items: Prompt items to dispatch (see `load_prompt_items`).
    trajectory_store: The cluster's trajectory store, if enabled.

  Returns:
    The RL program.
  """
  grpo = hp.config_mapping("agentic_grpo_config")
  return rl_program.StandardRLProgram(
      algo=algo,
      dataset=prompt_items,
      max_steps=int(hp.config_mapping("rl_training_config")["max_steps"]),
      reward_fns=reward_fns,
      generation_args=build_generation_args(hp),
      batch_size=int(hp.config["batch_size"]),
      batch_config=build_batch_config(hp, pad_id=pad_id),
      metrics_logging_options=build_metrics_logging_options(hp),
      trajectory_log_dir=_trajectory_log_dir(hp),
      trajectory_store=trajectory_store,
      max_staleness=int(grpo["off_policy_steps"]),
      group_order=rl_program.GroupOrder(str(grpo["trajectory_group_order"])),
      sync_weights=_weight_sync_mode(hp) != weight_sync.WeightSyncMode.NONE,
      precompile_trainer=bool(
          hp.config_mapping("dist_training_config")["precompile_trainer"]
      ),
  )


def run_program(
    hp: resolve_config.DistHyperParameters,
    cluster: cluster_orchestrator.ClusterOrchestrator,
    program: rl_program.StandardRLProgram,
    *,
    discovery: runtime_context.DiscoveryContext,
    rpc_timeout_s: float,
) -> rl_program.RLStepResult | None:
  """Waits for the workers, brings them up, and runs `program`.

  Always closes `program`; the caller tears the cluster down (`close_cluster`).

  Args:
    hp: Resolved distributed RL configuration.
    cluster: Cluster orchestrator to run on.
    program: The RL program to run.
    discovery: Discovery API the workers register through.
    rpc_timeout_s: Deadline for worker RPCs.

  Returns:
    The final step result, or None when no step completed.
  """
  try:
    wait_for_workers(hp, cluster, discovery, rpc_timeout_s=rpc_timeout_s)
    logging.info("Bringing up remote workers...")
    cluster.bring_up_workers(dummy_data=None)
    logging.info(
        "Workers ready: %s. Running StandardRLProgram...",
        [info.worker_id for info in cluster.registry.infos()],
    )
    # `StandardRLProgram` stops after its own `max_steps`; `run` takes no step
    # count.
    cluster.run(program=program, bring_up=False)
  except BaseException:
    logging.exception("The orchestrator run failed.")
    raise
  finally:
    program.close()
  return program.last_step_result


def close_cluster(
    hp: resolve_config.DistHyperParameters,
    cluster: cluster_orchestrator.ClusterOrchestrator,
) -> None:
  """Tears the cluster down.

  With `cluster_config.stop_workers_on_exit: false` the workers keep running;
  only this process's monitor and trajectory store are closed.

  Args:
    hp: Resolved distributed RL configuration.
    cluster: Cluster orchestrator to tear down.
  """
  if bool(hp.config_mapping("cluster_config")["stop_workers_on_exit"]):
    logging.info("Shutting down cluster workers...")
    cluster.shutdown()
  else:
    cluster.monitor.close()
    if cluster.trajectory_store is not None:
      cluster.trajectory_store.close()


def _pin_control_plane_to_cpu() -> None:
  """Defaults the control-plane JAX platform to CPU.

  The orchestrator runs small JAX computations (`GRPOAdapter.compute_advantages`
  returns `jax.numpy` arrays), and its imports load JAX anyway. On
  `cluster_config.target: local` it shares the host with the trainer and rollout
  nodes, so an unpinned JAX would claim the TPUs those nodes own.

  JAX reads `JAX_PLATFORMS` when it is imported, so an explicitly set value is
  already in `jax.config` and is kept. Must run before any JAX backend use.
  """
  # TODO(noghabi): Remove the orchestrator's JAX dependency (NumPy advantages,
  # no JAX-importing modules on the control plane), then drop this pin.
  if not jax.config.jax_platforms:
    jax.config.update("jax_platforms", "cpu")


def main(
    argv: list[str], context: runtime_context.ProcessContext | None = None
) -> None:
  """Runs the distributed GRPO orchestrator for the configured task.

  Args:
    argv: `override_config_file=...` and `section.key=value` overrides.
    context: Runtime process context providing the discovery API.
  """
  _pin_control_plane_to_cpu()
  discovery = node_utils.require_discovery_context(context).ipc.discovery
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [Orchestrator] %(message)s",
      force=True,
  )
  hp = resolve_config.initialize(argv)
  if hp.config_mapping("cluster_config")["debug"]:
    # Enables debug logging, including full sampler responses.
    logging.getLogger().setLevel(logging.DEBUG)
  logging.info("Resolved config: %s", hp.config)
  logging.info("Control-plane JAX backend: %s", jax.default_backend())

  # Everything that can fail on the config runs before waiting for the
  # workers: the checks that need no download first, then the dataset and the
  # tokenizer, then the program itself.
  weight_sync_timeouts, rpc_timeout_s = resolve_weight_sync_timeouts(hp)
  algo = build_algo(hp)
  reward_fns = hp.obtain_reward_fn()
  prompt_items = load_prompt_items(hp)
  pad_id, _ = node_utils.pad_and_eos_ids(node_utils.load_hf_tokenizer(hp))

  cluster = create_cluster(hp, weight_sync_timeouts=weight_sync_timeouts)
  try:
    program = build_program(
        hp,
        algo=algo,
        reward_fns=reward_fns,
        pad_id=pad_id,
        prompt_items=prompt_items,
        trajectory_store=cluster.trajectory_store,
    )
    result = run_program(
        hp, cluster, program, discovery=discovery, rpc_timeout_s=rpc_timeout_s
    )
  finally:
    close_cluster(hp, cluster)

  if result is None:
    logging.info("=== GRPO run finished without a completed step ===")
    return
  logging.info(
      "=== GRPO run finished ===\n"
      "  Final step: %d\n"
      "  Final policy version: %d\n"
      "  Rollouts in final step: %d\n"
      "  Microbatches in final step: %d\n"
      "  Final reward: mean=%.4f std=%.4f",
      result.step,
      result.policy_version,
      result.num_rollouts,
      result.num_microbatches,
      result.reward_mean,
      result.reward_std,
  )
