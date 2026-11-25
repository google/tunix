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

"""Distributed RL configuration resolver built on `tunix.cli.config.HyperParameters`.

Provides:
1. `DistHyperParameters` / `initialize(overrides)`: Validated configuration
   loaded from `base_dist_config.yaml`, task/recipe override files, and
   `section.key=value` overrides (no program name; see `_dist_argv`). Used
   directly by the bash launchers, worker nodes, and orchestrator
   (lightweight; no training imports).
2. `render_bash_exports(hp)`: Emits shell environment variables for `local.sh`
   and `k8s.sh` (ports, device IDs, K8s JobSet parameters, and artifact/model
   paths).
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
import datetime
import getpass
import hashlib
import itertools
import json
import math
import os
import pathlib
import re
import shlex
import sys
import types
from typing import Any, Final, Literal

import omegaconf
from tunix.cli import config as cli_config
from tunix.common import configs as common_configs
from tunix.common import datatypes as common_datatypes
from tunix.dist_rl.common import datatypes as dist_rl_datatypes
from tunix.dist_rl.common import execution_target
from tunix.dist_rl.common import gcs_cache
from tunix.dist_rl.common import otel_setup
from tunix.dist_rl.distributed.deployment import yaml_generator

LAUNCH_DIR = pathlib.Path(__file__).parent
REPO_ROOT = LAUNCH_DIR.parents[3]
DEFAULT_BASE_CONFIG_PATH = LAUNCH_DIR / "base_dist_config.yaml"


# Program name (`argv[0]`) that `DistHyperParameters` passes to
# `HyperParameters`; dist callers pass overrides only.
_DIST_PROG: Final[str] = "dist_rl"

_MODEL_SECTIONS: Final[tuple[str, ...]] = (
    "model_config",
    "actor_model_config",
    "reference_model_config",
    "rollout_model_config",
)
ProcessRole = Literal["orchestrator", "trainer", "rollout", "inference"]
PROCESS_ROLES: Final[tuple[ProcessRole, ...]] = (
    "orchestrator",
    "trainer",
    "rollout",
    "inference",
)
_NOT_READ: Final[str] = "the distributed RL nodes never read it, so remove it"
# Keys that earlier versions of `base_dist_config.yaml` had, mapped to what to
# do instead ("moved to <key>" for a key that only changed its path). An
# override that sets one fails with that hint instead of the unknown-key error.
# A whole section is listed by its name.
_RETIRED_KEYS: Final[Mapping[str, str]] = types.MappingProxyType({
    **dict.fromkeys(
        (
            "agentic_ppo_config",
            "critic_model_config",
            "data_directory",
            "data_source",
            "dataset_name",
            "grpo_config",
            "max_target_length",
            "num_batches",
            "offload_to_cpu",
            "ppo_config",
            "prompt_key",
            "rl_training_config.critic_optimizer_config",
            "rl_training_config.data_sharding_axis",
            "rl_training_config.gradient_accumulation_steps",
            "rl_training_config.max_inflight_computations",
            "rl_training_config.perf_metrics_options",
            "rl_training_config.rollout_micro_batch_size",
            "tfds_download",
            "tokenizer_config.add_bos",
            "tokenizer_config.add_eos",
            "training_mode",
            "vllm_config.model_version",
        ),
        _NOT_READ,
    ),
    **{
        f"{section}.{key}": _NOT_READ
        for section, key in itertools.product(
            _MODEL_SECTIONS,
            ("model_display", "mesh.allocation_policy"),
        )
    },
    "training_config": (
        "the distributed RL nodes never read it; set rl_training_config"
        " instead"
    ),
    "optimizer_config": (
        "the distributed RL nodes never read it; set"
        " rl_training_config.actor_optimizer_config instead"
    ),
    "apply_chat_template_to_dataset": (
        "the rollout workers apply the chat template (chat_parser_config), so"
        " remove it"
    ),
    "rollout_config.return_logprobs": (
        "derived from agentic_grpo_config.use_rollout_logps, so remove it"
    ),
    "vllm_config.server_mode": (
        "the vLLM rollout engines always run in server mode, so remove it"
    ),
    "vllm_config.tensor_parallel_size": (
        "it is the 'tp' axis of rollout_model_config.mesh, so remove it"
    ),
    "vllm_config.data_parallel_size": (
        "it is the 'fsdp' axis of rollout_model_config.mesh, so remove it"
    ),
    "cluster_config.registry_module": "moved to registry_module",
    **{
        f"cluster_config.{role}_main": (
            f"moved to cluster_config.roles.{role}.main"
        )
        for role in PROCESS_ROLES
    },
    **{
        f"cluster_config.ports.{role}": (
            f"moved to cluster_config.roles.{role}.port"
        )
        for role in PROCESS_ROLES
    },
    "cluster_config.role_env": "moved to cluster_config.roles.<role>.env",
    "cluster_config.k8s.user_container_memory": (
        "moved to cluster_config.roles.trainer.memory.request and"
        " cluster_config.roles.rollout.memory.request"
    ),
    "cluster_config.k8s.cpu_memory": (
        "moved to cluster_config.roles.orchestrator.memory.request"
    ),
    "cluster_config.k8s.user_container_memory_limit": (
        "moved to cluster_config.roles.trainer.memory.limit and"
        " cluster_config.roles.rollout.memory.limit"
    ),
    **{
        f"cluster_config.{key}": f"moved to cluster_config.local.{key}"
        for key in (
            "wait_timeout_secs",
            "wait_poll_secs",
            "wait_debug_every_polls",
            "wait_log_tail_lines",
        )
    },
    "cluster_config.rollout_fault_tolerance": (
        "moved to cluster_config.fault_tolerance.enabled"
    ),
    **{
        f"cluster_config.k8s.pathways_{old}": (
            f"moved to cluster_config.pathways.{new}"
        )
        for old, new in (
            ("server_image", "server_image"),
            ("proxy_image", "proxy_image"),
            ("proxy_memory", "proxy_memory.request"),
            ("proxy_memory_limit", "proxy_memory.limit"),
            ("rm_memory", "rm_memory.request"),
            ("worker_memory", "server_memory.request"),
        )
    },
    "cluster_config.k8s.gcs_scratch_location": (
        "moved to cluster_config.pathways.gcs_scratch_location"
    ),
    "cluster_config.k8s.enable_pathways_persistence": (
        "moved to cluster_config.pathways.trainer.persistence"
    ),
    "weight_sync_config.raiden_use_ffi": (
        "moved to weight_sync_config.raiden.use_ffi"
    ),
    "weight_sync_config.raiden_data_nics": (
        "moved to weight_sync_config.raiden.data_nics"
    ),
    **{
        f"cluster_config.{key}": (
            f"moved to cluster_config.fault_tolerance.{key}"
        )
        for key in (
            "max_task_retries",
            "poll_retry_budget_s",
            "min_active_rollout_ratio",
            "min_active_rollout_grace_s",
        )
    },
})
_NO_EVALUATION: Final[str] = "the distributed RL path runs no evaluation yet"
# Keys the distributed RL path does not implement yet, with the only value it
# accepts and why. The keys stay in `base_dist_config.yaml` (each with a TODO)
# so that setting one fails loudly instead of being ignored.
_UNIMPLEMENTED_KEYS: Final[Mapping[str, tuple[int | None, str]]] = (
    types.MappingProxyType({
        **{
            f"{section}.rng_seed": (
                None,
                (
                    "LoRA always initializes with seed 0, and rollout sampling"
                    " is not seeded"
                ),
            )
            for section in _MODEL_SECTIONS
        },
        "num_train_epochs": (
            None,
            (
                "the dataset repeats and rl_training_config.max_steps sets the"
                " training length"
            ),
        ),
        "train_fraction": (None, _NO_EVALUATION),
        "num_test_batches": (None, _NO_EVALUATION),
        "rl_training_config.eval_every_n_steps": (None, _NO_EVALUATION),
        "agentic_grpo_config.num_iterations": (
            1,
            "each rollout batch gets one optimizer pass",
        ),
    })
)


def _dist_argv(overrides: Sequence[str]) -> list[str]:
  """Returns the `HyperParameters` argv for the dist `overrides`.

  The agentic CLI passes `[prog, base_yaml, *overrides]` from its command line.
  Dist callers pass overrides only, so the program name (`_DIST_PROG`) and the
  base YAML (`DEFAULT_BASE_CONFIG_PATH`) are prepended here.

  Args:
    overrides: `section.key=value` and `override_config_file=path` arguments,
      without a program name or base YAML.

  Returns:
    `[_DIST_PROG, DEFAULT_BASE_CONFIG_PATH, *overrides]`.

  Raises:
    ValueError: If an argument has no `=` (a program name or base YAML).
  """
  bare = [arg for arg in overrides if "=" not in arg]
  if bare:
    raise ValueError(
        "Expected only 'section.key=value' or 'override_config_file=path'"
        f" arguments (no program name or base YAML), got {bare}."
    )
  return [_DIST_PROG, str(DEFAULT_BASE_CONFIG_PATH), *overrides]


def finalize_dist_config(
    hp: cli_config.HyperParameters,
) -> execution_target.ClusterTargets:
  """Applies the distributed RL derived defaults, then validates invariants.

  Called at the end of `DistHyperParameters.__init__`, after
  `HyperParameters.__init__`.

  Args:
    hp: Loaded configuration; updated in place.

  Returns:
    Resolved `ClusterTargets` for the cluster run.
  """
  _finalize_dist_defaults(hp)
  _validate_local_settings(hp)
  _validate_dist_invariants(hp)
  cluster_targets = resolve_cluster_targets(hp)
  cluster = _mutable_section(hp, "cluster_config")
  local = cluster["local"]
  if (
      cluster["target"] == "local"
      and cluster_targets.placement == execution_target.Placement.COLOCATED
  ):
    local["rollout_devices"] = local["trainer_devices"]
    local["inference_devices"] = local["trainer_devices"]
  return cluster_targets


class DistHyperParameters(cli_config.HyperParameters):
  """`tunix.cli.config.HyperParameters` extended for distributed RL.

  Takes overrides only (see `_dist_argv`). Override files and `key=value`
  arguments deep-merge (`merge_replace_keys`), and every override key must be
  in the base schema (`validate_nested`). A key in `_RETIRED_KEYS` fails with
  what to do instead.
  """

  cluster_targets: execution_target.ClusterTargets

  def __init__(self, overrides: Sequence[str]) -> None:
    try:
      super().__init__(
          _dist_argv(overrides), merge_replace_keys=True, validate_nested=True
      )
    except cli_config.UnknownConfigKeyError as e:
      if e.key not in _RETIRED_KEYS:
        raise
      raise ValueError(
          f"{e.key} is no longer a config key: {_RETIRED_KEYS[e.key]}."
      ) from e
    self.cluster_targets = finalize_dist_config(self)


def resolve_cluster_targets(
    hp: cli_config.HyperParameters,
) -> execution_target.ClusterTargets:
  """Resolves `execution_targets` and returns validated `ClusterTargets`."""
  exec_section = _mutable_section(hp, "execution_targets")
  for key in ("target", "trainer", "rollout", "inference", "placement"):
    if not isinstance(exec_section[key], str):
      raise TypeError(
          f"execution_targets.{key} must be a str, got {exec_section[key]!r}."
      )
  default_target = execution_target.ExecutionTarget.parse(
      exec_section["target"] or str(execution_target.JAX_TPU)
  )
  exec_section["target"] = str(default_target)
  for role in ("trainer", "rollout", "inference"):
    raw = exec_section[role].strip()
    exec_section[role] = str(
        execution_target.ExecutionTarget.parse(raw) if raw else default_target
    )
  placement_raw = (
      exec_section["placement"].strip().lower()
      or execution_target.Placement.DISAGGREGATED.value
  )
  _check_choice(
      "execution_targets.placement",
      placement_raw,
      tuple(p.value for p in execution_target.Placement),
  )
  exec_section["placement"] = placement_raw
  return execution_target.ClusterTargets(
      trainer=execution_target.ExecutionTarget.parse(exec_section["trainer"]),
      rollout=execution_target.ExecutionTarget.parse(exec_section["rollout"]),
      inference=execution_target.ExecutionTarget.parse(
          exec_section["inference"]
      ),
      placement=execution_target.Placement(placement_raw),
      weight_sync_mode=common_datatypes.WeightSyncMode(
          str(hp.config_mapping("weight_sync_config")["mode"])
      ),
  )


def _sync_packed_token_budget(section: MutableMapping[str, Any]) -> None:
  """Validates and syncs `max_seq_tokens_per_device` and `max_seq_token_per_tpu`."""
  resolved = common_configs.resolve_seq_tokens_budget(
      section["max_seq_tokens_per_device"],
      section["max_seq_token_per_tpu"],
  )
  section["max_seq_tokens_per_device"] = resolved
  section["max_seq_token_per_tpu"] = resolved


def _mutable_section(
    hp: cli_config.HyperParameters, key: str
) -> MutableMapping[str, Any]:
  """Returns config section `key` for in-place updates."""
  section = hp.config[key]
  if not isinstance(section, MutableMapping):
    raise TypeError(
        f"Expected config section {key!r} to be a mutable mapping, got"
        f" {type(section).__name__}."
    )
  return section


def _resolve_sandbox_namespace(
    hp: cli_config.HyperParameters, target: str
) -> None:
  """Resolves `kubernetes_config.namespace: null` to the GKE namespace.

  On `cluster_config.target=k8s`, a null sandbox namespace means the namespace
  the run's JobSets use (`cluster_config.k8s.namespace`), so the agent
  sandboxes run next to the RL pods.

  Args:
    hp: Loaded configuration; updated in place.
    target: The validated `cluster_config.target`.

  Raises:
    ValueError: If the namespace is null on any other target, which has no GKE
      namespace to follow.
  """
  sandbox_k8s = hp.config["kubernetes_config"]
  if not isinstance(sandbox_k8s, MutableMapping):
    return
  if "namespace" not in sandbox_k8s or sandbox_k8s["namespace"] is not None:
    return
  if target != "k8s":
    raise ValueError(
        "kubernetes_config.namespace=null follows cluster_config.k8s.namespace"
        f" and needs cluster_config.target=k8s, got {target!r}; set the"
        " sandbox namespace explicitly."
    )
  sandbox_k8s["namespace"] = str(
      hp.config_mapping("cluster_config")["k8s"]["namespace"]
  )


def _finalize_dist_defaults(hp: cli_config.HyperParameters) -> None:
  """Resolves derived cluster, model, tokenizer, and optimizer defaults.

  The output directories (checkpoints, metrics, eval) default to
  subdirectories of `cluster_config.artifact_root`.

  Args:
    hp: Loaded configuration updated in place.
  """
  cluster = _mutable_section(hp, "cluster_config")
  target = str(cluster["target"])
  _check_choice("cluster_config.target", target, _CLUSTER_TARGETS)

  if not cluster["artifact_root"]:
    raise ValueError("cluster_config.artifact_root must not be empty.")
  # `pathlib` turns `gs://bucket/run` into the relative path `gs:/bucket/run`,
  # so the derived output paths would silently land on local disk.
  if "://" in str(cluster["artifact_root"]):
    raise ValueError(
        "cluster_config.artifact_root must be a filesystem path, got"
        f" {str(cluster['artifact_root'])!r}. To write the outputs to a URI,"
        " set rl_training_config.checkpoint_root_directory,"
        " rl_training_config.metrics_logging_options.log_dir,"
        " trajectory_store_config.dir and (for a MaxText trainer)"
        " actor_model_config.kwargs.maxtext_output_dir instead."
    )
  artifact_root = pathlib.Path(str(cluster["artifact_root"]))
  if target == "external" and not artifact_root.is_absolute():
    raise ValueError(
        "cluster_config.target=external requires an absolute"
        f" cluster_config.artifact_root, got {str(artifact_root)!r}: each"
        " process would resolve a relative path against its own working"
        " directory."
    )
  if target == "local" and not artifact_root.is_absolute():
    artifact_root = REPO_ROOT / artifact_root
  cluster["artifact_root"] = str(artifact_root)

  _resolve_sandbox_namespace(hp, target)

  if not cluster["run_id"]:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    cluster["run_id"] = f"run_{ts}_{os.getpid()}"

  for role_key in (
      "model_config",
      "actor_model_config",
      "reference_model_config",
      "rollout_model_config",
  ):
    role_cfg = _mutable_section(hp, role_key)
    if not role_cfg["model_download_path"]:
      role_cfg["model_download_path"] = str(
          artifact_root / "models" / str(role_cfg["model_name"])
      )

  rl_train = _mutable_section(hp, "rl_training_config")
  _sync_packed_token_budget(rl_train)
  if rl_train["mini_batch_size"] is None:
    rl_train["mini_batch_size"] = int(hp.config["batch_size"])
  if not rl_train["checkpoint_root_directory"]:
    rl_train["checkpoint_root_directory"] = str(artifact_root / "checkpoints")
  metrics = rl_train["metrics_logging_options"]
  # Null logs no metrics (see `orchestrator.build_metrics_logging_options`).
  if metrics is not None:
    if not isinstance(metrics, MutableMapping):
      raise TypeError(
          "rl_training_config.metrics_logging_options must be a mapping or"
          f" null, got {type(metrics).__name__}."
      )
    if not metrics["log_dir"]:
      # The orchestrator also logs trajectories under `<log_dir>/trajectories`
      # unless `dist_training_config.trajectory_log_dir` is set or
      # `dist_training_config.trajectory_logging` is false.
      metrics["log_dir"] = str(artifact_root / "metrics")

  traj_store = _mutable_section(hp, "trajectory_store_config")
  if traj_store["backend"] == "file" and not traj_store["dir"]:
    traj_store["dir"] = str(artifact_root / "trajectory_store")

  eval_config = _mutable_section(hp, "eval_config")
  if not eval_config["output_dir"]:
    eval_config["output_dir"] = str(artifact_root / "eval")

  opt = rl_train["actor_optimizer_config"]
  if isinstance(opt, MutableMapping):
    if opt["peak_value"] is None:
      opt["peak_value"] = float(opt["learning_rate"])
    if opt["warmup_steps"] is None and opt["decay_steps"] is not None:
      warmup_ratio = float(opt["warmup_ratio"])
      if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError(
            "rl_training_config.actor_optimizer_config.warmup_ratio must be in"
            f" [0, 1], got {warmup_ratio}."
        )
      # Round before ceil so float noise (e.g. 50.000000001) doesn't add a step.
      opt["warmup_steps"] = math.ceil(
          round(int(opt["decay_steps"]) * warmup_ratio, 9)
      )

  grpo = _mutable_section(hp, "agentic_grpo_config")
  if cluster["run_inference_node"] is None:
    cluster["run_inference_node"] = float(grpo["beta"]) > 0.0
  # String keys resolve YAML null to "" (`tunix.cli.config`), while
  # `GRPOConfig` spells "no sampler importance correction" as None.
  if grpo["sampler_is"] == "":
    grpo["sampler_is"] = None

  k8s = cluster["k8s"]
  if isinstance(k8s, MutableMapping):
    user = (
        os.environ["USER"]
        if "USER" in os.environ and os.environ["USER"]
        else getpass.getuser()
    )
    if not k8s["orchestrator_id"]:
      k8s["orchestrator_id"] = f"{user}-orch"
    if not k8s["trainer_id"]:
      k8s["trainer_id"] = f"{user}-train"
    if not k8s["rollout_id"]:
      k8s["rollout_id"] = f"{user}-roll"
    if not k8s["inference_id"]:
      k8s["inference_id"] = f"{user}-inf"
    if isinstance(k8s["otel_id"], str) and not k8s["otel_id"]:
      k8s["otel_id"] = f"{user}-otel"

  _finalize_otel_config(hp, target)


def _validate_otel_cluster_settings(
    hp: cli_config.HyperParameters, target: str
) -> tuple[int, str]:
  """Validates `cluster_config.ports.otel` and `cluster_config.k8s.otel_id`."""
  cluster = hp.config_mapping("cluster_config")
  otel_port = cluster["ports"]["otel"]
  if isinstance(otel_port, bool) or not isinstance(otel_port, int):
    raise TypeError(
        f"cluster_config.ports.otel must be an int, got {otel_port!r}."
    )
  if not 1 <= otel_port <= 65535:
    raise ValueError(
        f"cluster_config.ports.otel must be in 1..65535, got {otel_port}."
    )
  otel_id = cluster["k8s"]["otel_id"]
  if target == "k8s":
    if not isinstance(otel_id, str):
      raise TypeError(
          f"cluster_config.k8s.otel_id must be a str, got {otel_id!r}."
      )
    if not otel_id or any(ch.isspace() for ch in otel_id):
      raise ValueError(
          "cluster_config.k8s.otel_id must be non-empty without whitespace,"
          f" got {otel_id!r}."
      )
  return otel_port, str(otel_id)


def _validate_otel_scalars(otel: Mapping[str, Any]) -> tuple[bool, int, float]:
  """Checks types and numeric ranges of `otel_config` scalar fields."""
  enable = otel["enable"]
  if not isinstance(enable, bool):
    raise TypeError(f"otel_config.enable must be a bool, got {enable!r}.")
  googlecloud = otel["googlecloud"]
  if not isinstance(googlecloud, bool):
    raise TypeError(
        f"otel_config.googlecloud must be a bool, got {googlecloud!r}."
    )
  for key in ("endpoint", "dashboard_id", "dashboard_project"):
    if not isinstance(otel[key], str):
      raise TypeError(
          f"otel_config.{key} must be a str, got {otel[key]!r} (quote numeric"
          " IDs on the command line)."
      )
  interval_ms = otel["export_interval_ms"]
  if isinstance(interval_ms, bool) or not isinstance(interval_ms, int):
    raise TypeError(
        f"otel_config.export_interval_ms must be an int, got {interval_ms!r}."
    )
  if interval_ms <= 0:
    raise ValueError(
        f"otel_config.export_interval_ms must be > 0, got {interval_ms}."
    )
  timeout_s = otel["export_timeout_s"]
  if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
    raise TypeError(
        f"otel_config.export_timeout_s must be a number, got {timeout_s!r}."
    )
  if not (0.0 < float(timeout_s) < math.inf):
    raise ValueError(
        "otel_config.export_timeout_s must be finite and > 0, got"
        f" {timeout_s!r}."
    )
  return enable, interval_ms, float(timeout_s)


def _finalize_otel_config(hp: cli_config.HyperParameters, target: str) -> None:
  """Validates `otel_config` and derives a default collector endpoint when enabled."""
  otel_port, otel_id = _validate_otel_cluster_settings(hp, target)
  otel = _mutable_section(hp, "otel_config")
  enable, interval_ms, timeout_s = _validate_otel_scalars(otel)
  endpoint = otel["endpoint"].strip()
  dash_id = otel["dashboard_id"].strip()
  dash_proj = otel["dashboard_project"].strip()
  otel["export_timeout_s"] = timeout_s
  if not enable:
    ignored = [
        name
        for name, is_set in (
            ("endpoint", bool(endpoint)),
            ("dashboard_id", bool(dash_id)),
            ("dashboard_project", bool(dash_proj)),
            (
                "export_interval_ms",
                interval_ms != otel_setup.DEFAULT_EXPORT_INTERVAL_MS,
            ),
            (
                "export_timeout_s",
                timeout_s != otel_setup.DEFAULT_EXPORT_TIMEOUT_S,
            ),
        )
        if is_set
    ]
    if ignored:
      raise ValueError(
          f"otel_config sets {ignored}, which only take effect with"
          " otel_config.enable=true. Set otel_config.enable=true or clear them."
      )
    otel["endpoint"] = ""
    otel["dashboard_id"] = ""
    otel["dashboard_project"] = ""
    return

  if bool(dash_id) != bool(dash_proj):
    raise ValueError(
        "otel_config.dashboard_id and otel_config.dashboard_project must both"
        " be set or both be empty."
    )
  otel["dashboard_id"] = dash_id
  otel["dashboard_project"] = dash_proj
  if not endpoint:
    if target == "k8s":
      endpoint = f"http://{otel_id}:{otel_port}"
    elif target == "local":
      endpoint = f"http://localhost:{otel_port}"
    else:
      raise ValueError(
          "otel_config.enable=true on cluster_config.target='external'"
          " requires an explicit otel_config.endpoint."
      )
  otel["endpoint"] = otel_setup.normalize_endpoint(endpoint)


def _validate_rollout_cache_settings(hp: cli_config.HyperParameters) -> None:
  """Checks the settings that `rollout_cache_params` and the cache URI read.

  Args:
    hp: Loaded configuration.

  Raises:
    TypeError: If `cluster_config.k8s.jax_cache.disable` or `save` is not a
      bool, which `k8s.sh` and `render_bash_exports` could read differently
      (e.g. `save: null` would still save on the pods).
    ValueError: If `cluster_config.k8s.jax_cache.local_dir` is not a non-empty
      absolute path, or if `rollout_model_config.lora_config` is set without
      `rank`.
  """
  jax_cache = hp.config_mapping("cluster_config")["k8s"]["jax_cache"]
  for key in ("disable", "save"):
    if not isinstance(jax_cache[key], bool):
      raise TypeError(
          f"cluster_config.k8s.jax_cache.{key} must be a bool, got"
          f" {jax_cache[key]!r}."
      )
  local_dir = jax_cache["local_dir"]
  if not isinstance(local_dir, str) or not os.path.isabs(local_dir):
    raise ValueError(
        "cluster_config.k8s.jax_cache.local_dir must be a non-empty absolute"
        f" path, got {local_dir!r}."
    )
  lora_config = hp.config_mapping("rollout_model_config")["lora_config"]
  if lora_config and "rank" not in lora_config:
    raise ValueError("rollout_model_config.lora_config is missing 'rank'.")


_GCS_URI_RE: Final[re.Pattern[str]] = re.compile(
    r"^gs://[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?(?:/\S*)?\Z"
)


def _validate_k8s_jax_cache_gcs(hp: cli_config.HyperParameters) -> None:
  """Checks the GCS location settings in `cluster_config.k8s.jax_cache`.

  Args:
    hp: Loaded configuration.

  Raises:
    TypeError: If `rollout_gcs_dir`, `gcs_dir`, or `bucket` is not a `str`.
    ValueError: If a non-empty `rollout_gcs_dir` or `gcs_dir` is not a
      `gs://<bucket>[/...]` URI without whitespace, or a non-empty `bucket` is
      not a GCS bucket name or `gs://<bucket>[/...]` URI without whitespace.
  """
  jax_cache = hp.config_mapping("cluster_config")["k8s"]["jax_cache"]
  for key in ("rollout_gcs_dir", "gcs_dir", "bucket"):
    if not isinstance(jax_cache[key], str):
      raise TypeError(
          f"cluster_config.k8s.jax_cache.{key} must be a str, got"
          f" {jax_cache[key]!r}."
      )
  for key in ("rollout_gcs_dir", "gcs_dir"):
    value = jax_cache[key]
    if value and not _GCS_URI_RE.match(value):
      raise ValueError(
          f"cluster_config.k8s.jax_cache.{key} must start with"
          f" 'gs://<bucket>', got {value!r}."
      )
  bucket = jax_cache["bucket"]
  bucket_uri = bucket if bucket.startswith("gs://") else f"gs://{bucket}"
  if bucket and not _GCS_URI_RE.match(bucket_uri):
    raise ValueError(
        "cluster_config.k8s.jax_cache.bucket must be a GCS bucket name or"
        f" 'gs://<bucket>' URI, got {bucket!r}."
    )


_K8S_BOOL_KEYS: Final[tuple[str, ...]] = (
    "preemptible",
    "fail_fast",
    "teardown_gang_on_exit",
    "use_dynamic_slicing",
)
_K8S_STR_KEYS: Final[tuple[str, ...]] = (
    "priority_class",
    "service_account",
    "cpu_nodepool",
    "head_nodepool",
)
_K8S_SCALAR_TYPES: Final[tuple[tuple[str, type[bool | str]], ...]] = (
    *((key, bool) for key in _K8S_BOOL_KEYS),
    *((key, str) for key in _K8S_STR_KEYS),
)
_K8S_JOBSET_ONLY_KEYS: Final[tuple[str, ...]] = (
    "fail_fast",
    "teardown_gang_on_exit",
    "use_dynamic_slicing",
)


def _validate_k8s_scalars(hp: cli_config.HyperParameters) -> None:
  """Validates scalar types and ranges of `cluster_config.k8s` settings."""
  cluster = hp.config_mapping("cluster_config")
  if cluster["target"] != "k8s":
    return
  k8s = cluster["k8s"]
  for key, expected_type in _K8S_SCALAR_TYPES:
    if not isinstance(k8s[key], expected_type):
      raise TypeError(
          f"cluster_config.k8s.{key} must be a {expected_type.__name__}, got"
          f" {k8s[key]!r}."
      )
  retries = k8s["startup_retries"]
  if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
    raise ValueError(
        "cluster_config.k8s.startup_retries must be an int >= 0, got"
        f" {retries!r}."
    )
  grace = k8s["termination_grace_seconds"]
  if grace is not None and (
      isinstance(grace, bool) or not isinstance(grace, int) or grace < 0
  ):
    raise ValueError(
        "cluster_config.k8s.termination_grace_seconds must be an int >= 0 or"
        f" null, got {grace!r}."
    )


def _head_excluded_nodepools(k8s: Mapping[str, object]) -> str:
  """Returns `cluster_config.k8s.head_excluded_nodepools` as a flag value.

  Args:
    k8s: The `cluster_config.k8s` mapping.

  Returns:
    The node pools joined with commas, as `--head_excluded_nodepools` of the
    YAML generator takes them.

  Raises:
    TypeError: If the setting is not a list of str.
  """
  pools = k8s["head_excluded_nodepools"]
  if (
      isinstance(pools, str)
      or not isinstance(pools, Sequence)
      or not all(isinstance(pool, str) for pool in pools)
  ):
    raise TypeError(
        "cluster_config.k8s.head_excluded_nodepools must be a list of str, got"
        f" {pools!r}."
    )
  return ",".join(pools)


def _validate_k8s_worker_jobsets(hp: cli_config.HyperParameters) -> None:
  """Rejects `cluster_config.k8s` settings the worker templates cannot render.

  `k8s.sh` applies the orchestrator JobSet before it renders the trainer and
  rollout ones, so a setting the YAML generator rejects or drops for a worker
  template would leave a partial deploy.

  Args:
    hp: Loaded configuration.

  Raises:
    TypeError: If `head_excluded_nodepools` is not a list of str.
    ValueError: If `target` is "k8s" and `run_id` is not a valid K8s label
      value when `teardown_gang_on_exit` is enabled, `head_nodepool` is set
      without `use_dynamic_slicing`, `head_excluded_nodepools` is set with
      `use_dynamic_slicing` or names an invalid node pool, or a JobSet-only
      setting is enabled with a `leaderworkerset.*.yaml` worker template.
  """
  cluster = hp.config_mapping("cluster_config")
  if cluster["target"] != "k8s":
    return
  k8s = cluster["k8s"]
  if k8s["teardown_gang_on_exit"]:
    try:
      yaml_generator.validate_gang_id(str(cluster["run_id"]))
    except ValueError as e:
      raise ValueError(
          "cluster_config.run_id must be a valid Kubernetes label value when"
          f" cluster_config.k8s.teardown_gang_on_exit=true: {e}"
      ) from e
  if k8s["head_nodepool"] and not k8s["use_dynamic_slicing"]:
    raise ValueError(
        "cluster_config.k8s.head_nodepool requires"
        " cluster_config.k8s.use_dynamic_slicing=true."
    )
  excluded = _head_excluded_nodepools(k8s)
  if excluded and k8s["use_dynamic_slicing"]:
    raise ValueError(
        "cluster_config.k8s.head_excluded_nodepools requires"
        " cluster_config.k8s.use_dynamic_slicing=false; with it, set"
        " cluster_config.k8s.head_nodepool."
    )
  try:
    yaml_generator.parse_head_excluded_nodepools(excluded)
  except ValueError as e:
    raise ValueError(f"cluster_config.k8s.head_excluded_nodepools: {e}") from e
  enabled_jobset_keys = [
      f"cluster_config.k8s.{k}" for k in _K8S_JOBSET_ONLY_KEYS if k8s[k]
  ]
  lws_templates = [
      f"cluster_config.k8s.{role}.jobset_yaml={k8s[role]['jobset_yaml']}"
      for role in ("trainer", "rollout")
      if str(k8s[role]["jobset_yaml"]).startswith("leaderworkerset.")
  ]
  if enabled_jobset_keys and lws_templates:
    raise ValueError(
        f"{', '.join(enabled_jobset_keys)} need JobSet worker templates"
        f" (jobset.*.yaml), got {', '.join(lws_templates)}."
    )


def _validate_rollout_replicas(hp: cli_config.HyperParameters) -> None:
  """Checks `cluster_config.{rollout_replicas,min_rollout_replicas}`.

  Args:
    hp: Loaded configuration.

  Raises:
    ValueError: If `rollout_replicas` is not an int >= 1 (or != 1 on
      `cluster_config.target: local`), or `min_rollout_replicas` is neither
      null nor an int in `1..rollout_replicas`.
  """
  cluster = hp.config_mapping("cluster_config")
  replicas = cluster["rollout_replicas"]
  if (
      isinstance(replicas, bool)
      or not isinstance(replicas, int)
      or replicas < 1
  ):
    raise ValueError(
        "cluster_config.rollout_replicas must be an int >= 1, got"
        f" {replicas!r}."
    )
  if cluster["target"] == "local" and replicas != 1:
    raise ValueError(
        "cluster_config.target=local runs one rollout (local.sh starts one), so"
        f" cluster_config.rollout_replicas must be 1, got {replicas}."
    )
  min_replicas = cluster["min_rollout_replicas"]
  if min_replicas is None:
    return
  if (
      isinstance(min_replicas, bool)
      or not isinstance(min_replicas, int)
      or min_replicas < 1
  ):
    raise ValueError(
        "cluster_config.min_rollout_replicas must be null or an int >= 1, got"
        f" {min_replicas!r}."
    )
  if min_replicas > replicas:
    raise ValueError(
        f"cluster_config.min_rollout_replicas ({min_replicas}) must be <="
        f" cluster_config.rollout_replicas ({replicas})."
    )


def _validate_eval_config(hp: cli_config.HyperParameters) -> None:
  """Checks that `eval_config.attempts_per_task` is an int >= 1."""
  attempts = hp.config_mapping("eval_config")["attempts_per_task"]
  if (
      isinstance(attempts, bool)
      or not isinstance(attempts, int)
      or attempts < 1
  ):
    raise ValueError(
        "eval_config.attempts_per_task must be an int >= 1, got"
        f" {attempts!r}."
    )


def _validate_trajectory_logging(hp: cli_config.HyperParameters) -> None:
  """Checks `dist_training_config.trajectory_logging`.

  `trajectory_log_dir` names where the trajectories go, so it must be empty
  when `trajectory_logging` is false.

  Args:
    hp: Loaded configuration.

  Raises:
    TypeError: If `trajectory_logging` is not a bool.
    ValueError: If `trajectory_logging` is false and `trajectory_log_dir` is
      not empty.
  """
  dist_train = hp.config_mapping("dist_training_config")
  enabled = dist_train["trajectory_logging"]
  if not isinstance(enabled, bool):
    raise TypeError(
        "dist_training_config.trajectory_logging must be a bool, got"
        f" {enabled!r}."
    )
  log_dir = dist_train["trajectory_log_dir"]
  if not enabled and log_dir:
    raise ValueError(
        "dist_training_config.trajectory_logging=false logs no trajectories,"
        " but dist_training_config.trajectory_log_dir is set"
        f" ({log_dir!r}). Clear trajectory_log_dir or set"
        " trajectory_logging=true."
    )


# `agentic_grpo_config` keys that only take effect with `use_rollout_logps:
# true`. `sampler_is` is left out: it defaults to "token", which runs with
# `use_rollout_logps: false` keep.
_ROLLOUT_LOGPS_KEYS: Final[tuple[str, ...]] = (
    "sampler_rs",
    "seq_logprob_error_threshold",
)


def _validate_rollout_logps_settings(hp: cli_config.HyperParameters) -> None:
  """Checks the settings that need `agentic_grpo_config.use_rollout_logps`.

  With `use_rollout_logps: false`, the trainer doesn't compare its
  log-probabilities with the rollout's, so it would skip rejection sampling
  and the sequence log-prob error check without a word.

  Args:
    hp: Loaded configuration.

  Raises:
    TypeError: If `use_rollout_logps` is not a bool.
    ValueError: If `use_rollout_logps` is false and a `_ROLLOUT_LOGPS_KEYS`
      key is set.
  """
  grpo = hp.config_mapping("agentic_grpo_config")
  use_rollout_logps = grpo["use_rollout_logps"]
  if not isinstance(use_rollout_logps, bool):
    raise TypeError(
        "agentic_grpo_config.use_rollout_logps must be a bool, got"
        f" {use_rollout_logps!r}."
    )
  if use_rollout_logps:
    return
  ignored = [key for key in _ROLLOUT_LOGPS_KEYS if grpo[key] is not None]
  if ignored:
    raise ValueError(
        f"agentic_grpo_config sets {ignored}, which only take effect with"
        " agentic_grpo_config.use_rollout_logps=true: with false, the trainer"
        " doesn't compare its log-probabilities with the rollout's. Set"
        " use_rollout_logps=true or clear them."
    )


def _validate_fault_tolerance(hp: cli_config.HyperParameters) -> None:
  """Checks the `cluster_config.fault_tolerance` settings.

  Args:
    hp: Loaded configuration.

  Raises:
    TypeError: If `enabled` is not a bool, `max_task_retries` is not an int, or
      `poll_retry_budget_s`, `min_active_rollout_ratio`, or
      `min_active_rollout_grace_s` is not a number (the first two may be null).
    ValueError: If a setting is non-finite or out of range.
  """
  ft = hp.config_mapping("cluster_config")["fault_tolerance"]
  enabled = ft["enabled"]
  if not isinstance(enabled, bool):
    raise TypeError(
        "cluster_config.fault_tolerance.enabled must be a bool, got"
        f" {enabled!r}."
    )
  nullable_number = (int, float, type(None))
  for key, allowed, type_desc in (
      ("max_task_retries", int, "an int"),
      ("poll_retry_budget_s", nullable_number, "a number or null"),
      ("min_active_rollout_ratio", nullable_number, "a number or null"),
      ("min_active_rollout_grace_s", (int, float), "a number"),
  ):
    value = ft[key]
    if isinstance(value, bool) or not isinstance(value, allowed):
      raise TypeError(
          f"cluster_config.fault_tolerance.{key} must be {type_desc}, got"
          f" {value!r}."
      )
  retries = ft["max_task_retries"]
  if retries < 0:
    raise ValueError(
        "cluster_config.fault_tolerance.max_task_retries must be >= 0, got"
        f" {retries}."
    )
  budget = ft["poll_retry_budget_s"]
  if budget is not None and not (0.0 <= budget < math.inf):
    raise ValueError(
        "cluster_config.fault_tolerance.poll_retry_budget_s must be finite and"
        f" >= 0, got {budget}."
    )
  ratio = ft["min_active_rollout_ratio"]
  if ratio is not None and not (0.0 < ratio <= 1.0):
    raise ValueError(
        "cluster_config.fault_tolerance.min_active_rollout_ratio must be in"
        f" (0.0, 1.0], got {ratio}."
    )
  grace = ft["min_active_rollout_grace_s"]
  if not (0.0 <= grace < math.inf):
    raise ValueError(
        "cluster_config.fault_tolerance.min_active_rollout_grace_s must be"
        f" finite and >= 0, got {grace}."
    )


# Environment variables `node_utils.apply_library_env` sets from the config,
# and the config key each comes from. Launchers reject them in user-supplied
# environments so a value can only come from the config.
LIBRARY_ENV_CONFIG_KEYS: Final[Mapping[str, str]] = types.MappingProxyType({
    "SKIP_JAX_PRECOMPILE": "vllm_config.skip_jax_precompile",
    "TPU_RAIDEN_DATA_NICS": "weight_sync_config.raiden.data_nics",
    "VERIFY_WEIGHTS": "weight_sync_config.verify_weights",
})
# Legacy OpenTelemetry env vars removed in favor of `otel_config`, mapped to
# user-facing replacement hints. Shared by `resolve_config` and `xm_launch`.
OTEL_ENV_CONFIG_HINTS: Final[Mapping[str, str]] = types.MappingProxyType({
    "OTEL_EXPORTER_OTLP_ENDPOINT": (
        "set otel_config.enable=true (and otel_config.endpoint if needed)"
        " instead"
    ),
    "OTEL_METRIC_EXPORT_INTERVAL": (
        "set otel_config.enable=true and otel_config.export_interval_ms instead"
    ),
    "OTEL_EXPORTER_OTLP_TIMEOUT": (
        "set otel_config.enable=true and otel_config.export_timeout_s instead"
    ),
})
_FORBIDDEN_ROLE_ENVS: Final[Mapping[str, str]] = types.MappingProxyType({
    **{k: f"set {v} instead" for k, v in LIBRARY_ENV_CONFIG_KEYS.items()},
    **OTEL_ENV_CONFIG_HINTS,
    "ENABLE_PATHWAYS_PERSISTENCE": (
        "set cluster_config.pathways.trainer.persistence instead"
    ),
    "DIST_RL_WORKER_PORT": "launcher-managed worker identity",
    "DIST_RL_ROLLOUT_REPLICA": "launcher-managed worker identity",
})
_ROLE_ENV_PREFIX_REASONS: Final[tuple[tuple[str, str], ...]] = (
    ("T_", "Tunix config override"),
    ("M_", "MaxText config override"),
)
# The manifest generator's rule, so a name the config accepts also renders.
_ENV_VAR_NAME_RE: Final[re.Pattern[str]] = yaml_generator.ENV_VAR_NAME_RE


def role_env(
    hp: cli_config.HyperParameters,
) -> dict[ProcessRole, dict[str, str]]:
  """Validates and returns `cluster_config.roles.<role>.env` per process."""
  roles = hp.config_mapping("cluster_config")["roles"]
  if not isinstance(roles, Mapping) or set(roles) != set(PROCESS_ROLES):
    raise ValueError(
        f"cluster_config.roles must map {PROCESS_ROLES}, got {roles!r}."
    )
  result: dict[ProcessRole, dict[str, str]] = {}
  for role in PROCESS_ROLES:
    env = roles[role]["env"]
    if not isinstance(env, Mapping):
      raise TypeError(
          f"cluster_config.roles.{role}.env must be a mapping, got {env!r}."
      )
    if not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env.items()
    ):
      raise TypeError(
          f"cluster_config.roles.{role}.env keys and values must be strings,"
          f" got {dict(env)!r}."
      )
    result[role] = dict(env)
  return result


def _validate_role_env(hp: cli_config.HyperParameters) -> None:
  """Checks `cluster_config.roles.<role>.env` names and active roles."""
  parsed = role_env(hp)
  cluster = hp.config_mapping("cluster_config")
  if cluster["target"] == "external" and any(parsed.values()):
    raise ValueError(
        "cluster_config.roles.<role>.env is set, but"
        " cluster_config.target=external"
        " runs the processes outside local.sh and k8s.sh, which apply it."
    )
  if (
      parsed["inference"]
      and not hp.config_mapping("cluster_config")["run_inference_node"]
  ):
    raise ValueError(
        "cluster_config.roles.inference.env is set, but"
        " cluster_config.run_inference_node resolved to false."
    )
  pairs: list[tuple[str, str]] = []
  for role, env in parsed.items():
    pairs.extend((role, k) for k in env)
  for role, name in pairs:
    default_reason = (
        _FORBIDDEN_ROLE_ENVS[name] if name in _FORBIDDEN_ROLE_ENVS else ""
    )
    reason = next(
        (msg for pfx, msg in _ROLE_ENV_PREFIX_REASONS if name.startswith(pfx)),
        default_reason,
    )
    if reason or not _ENV_VAR_NAME_RE.match(name):
      raise ValueError(
          f"cluster_config.roles.{role}.env rejects {name!r}"
          f" ({reason or 'invalid variable name'})."
      )


def _reject_unimplemented_settings(hp: cli_config.HyperParameters) -> None:
  """Fails on a `_UNIMPLEMENTED_KEYS` key set to a value other than its own.

  Args:
    hp: Loaded configuration.

  Raises:
    NotImplementedError: If such a key has another value.
  """
  for key, (accepted, reason) in _UNIMPLEMENTED_KEYS.items():
    section, _, name = key.rpartition(".")
    value = hp.config_mapping(section)[name] if section else hp.config[name]
    # `True == 1 == 1.0`, so require the exact type too.
    if type(value) is type(accepted) and value == accepted:
      continue
    raise NotImplementedError(
        f"{key}={value!r} is not supported yet ({reason}); leave it at"
        f" {accepted!r}."
    )


def validate_memory(key: str, memory: object) -> None:
  """Checks a `{request, limit}` memory setting.

  Args:
    key: Dotted config key of the setting, for the error message.
    memory: The setting's value.

  Raises:
    TypeError: If `memory` is not a mapping of `request` and `limit` to str.
  """
  if not isinstance(memory, Mapping) or set(memory) != {"request", "limit"}:
    raise TypeError(
        f"{key} must be a mapping of `request` and `limit` (Kubernetes"
        f" quantities such as '48G', or empty for none), got {memory!r}."
    )
  for field in ("request", "limit"):
    if not isinstance(memory[field], str):
      raise TypeError(
          f"{key}.{field} must be a str (a Kubernetes quantity such as"
          f" '48G', or empty for none), got {memory[field]!r}."
      )


def _validate_role_memory(hp: cli_config.HyperParameters) -> None:
  """Checks each `cluster_config.roles.<role>.memory`."""
  roles = hp.config_mapping("cluster_config")["roles"]
  for role in ("orchestrator", "trainer", "rollout"):
    validate_memory(
        f"cluster_config.roles.{role}.memory", roles[role]["memory"]
    )


# The largest value the IFRT proxy client parses (a positive int64).
_MAX_IFRT_THRESHOLD_BYTES: Final[int] = 2**63 - 1


def _is_int(value: object) -> bool:
  """Returns whether `value` is an int and not a bool."""
  return isinstance(value, int) and not isinstance(value, bool)


def pathways_worker_env_json(hp: cli_config.HyperParameters) -> str:
  """Validates `cluster_config.pathways.worker_env`; returns it as JSON.

  `k8s.sh` passes the JSON to the manifest generator's
  `--pathways_worker_env_json`.

  Args:
    hp: Loaded configuration.

  Returns:
    The variables as a JSON object of strings.

  Raises:
    TypeError: If the section is not a mapping of strings to strings.
    ValueError: If a name is not a variable name or is one the Pathways
      template sets itself.
  """
  env = hp.config_mapping("cluster_config")["pathways"]["worker_env"]
  if not isinstance(env, Mapping):
    raise TypeError(
        f"cluster_config.pathways.worker_env must be a mapping, got {env!r}."
    )
  for name, value in env.items():
    if not isinstance(name, str) or not _ENV_VAR_NAME_RE.match(name):
      raise ValueError(
          f"cluster_config.pathways.worker_env has the invalid variable name"
          f" {name!r}."
      )
    if not isinstance(value, str):
      raise TypeError(
          f"cluster_config.pathways.worker_env.{name} must be a string, got"
          f" {value!r}."
      )
  env_json = json.dumps(dict(env))
  try:
    yaml_generator.parse_pathways_worker_env_json(env_json)
  except ValueError as e:
    raise ValueError(f"cluster_config.pathways.worker_env: {e}") from e
  return env_json


def _validate_launcher_settings(hp: cli_config.HyperParameters) -> None:
  """Checks the types of the Pathways, Raiden and reservation settings.

  These used to be environment variables of the launcher, so a malformed value
  would otherwise fail only when `k8s.sh` renders the manifests.

  Args:
    hp: Loaded configuration.

  Raises:
    TypeError: If a setting has the wrong type.
    ValueError: If a number is out of range, or a
      `cluster_config.pathways.worker_env` name is not a variable name.
  """
  cluster = hp.config_mapping("cluster_config")
  pathways = cluster["pathways"]
  raiden = hp.config_mapping("weight_sync_config")["raiden"]
  sidecar = pathways["sidecar"]
  pathways_trainer = pathways["trainer"]
  ckpt_gcsfuse = pathways_trainer["checkpoint_gcsfuse"]
  ckpt_gcsfuse_key = "cluster_config.pathways.trainer.checkpoint_gcsfuse"
  for key, value, expected in (
      *(
          (f"{ckpt_gcsfuse_key}.{name}", ckpt_gcsfuse[name], str)
          for name in ("bucket", "mount_path", "mount_options")
      ),
      (
          "cluster_config.pathways.trainer.persistence",
          pathways_trainer["persistence"],
          bool,
      ),
      (
          "cluster_config.pathways.proxy_extra_args",
          pathways["proxy_extra_args"],
          str,
      ),
      ("cluster_config.pathways.sidecar.image", sidecar["image"], str),
      ("cluster_config.pathways.sidecar.shm", sidecar["shm"], bool),
      (
          "cluster_config.k8s.tpu_reservation",
          cluster["k8s"]["tpu_reservation"],
          str,
      ),
      ("weight_sync_config.raiden.multi_numa", raiden["multi_numa"], bool),
  ):
    if not isinstance(value, expected):
      raise TypeError(
          f"{key} must be a {expected.__name__}, got {value!r}."
      )
  for key, memory in (
      ("proxy_memory", pathways["proxy_memory"]),
      ("rm_memory", pathways["rm_memory"]),
      ("server_memory", pathways["server_memory"]),
      ("sidecar.memory", sidecar["memory"]),
  ):
    validate_memory(f"cluster_config.pathways.{key}", memory)
  try:
    yaml_generator.parse_ckpt_gcsfuse_bucket(ckpt_gcsfuse["bucket"])
    yaml_generator.validate_ckpt_gcsfuse_mount_path(ckpt_gcsfuse["mount_path"])
  except ValueError as e:
    raise ValueError(f"{ckpt_gcsfuse_key}: {e}") from e
  pathways_worker_env_json(hp)
  use_ffi = raiden["use_ffi"]
  if use_ffi is not None and not isinstance(use_ffi, bool):
    raise TypeError(
        "weight_sync_config.raiden.use_ffi must be a bool or null, got"
        f" {use_ffi!r}."
    )
  broadcast_k = raiden["broadcast_k"]
  if not _is_int(broadcast_k) or broadcast_k <= 0:
    raise ValueError(
        "weight_sync_config.raiden.broadcast_k must be a positive int, got"
        f" {broadcast_k!r}."
    )
  threshold = pathways_trainer["ifrt_proxy_large_transfer_threshold_bytes"]
  if threshold is not None and not (
      _is_int(threshold) and 0 < threshold <= _MAX_IFRT_THRESHOLD_BYTES
  ):
    raise ValueError(
        "cluster_config.pathways.trainer.ifrt_proxy_large_transfer_threshold"
        "_bytes must be null or an int in [1, 2**63 - 1], got"
        f" {threshold!r}."
    )


def _validate_local_settings(hp: cli_config.HyperParameters) -> None:
  """Validates and normalizes `cluster_config.local` settings."""
  local = _mutable_section(hp, "cluster_config")["local"]
  if not isinstance(local, MutableMapping):
    raise TypeError(
        f"cluster_config.local must be a mapping, got {type(local).__name__}."
    )
  for bool_key in (
      "auto_detect_devices",
      "download_model_weights",
  ):
    if not isinstance(local[bool_key], bool):
      raise TypeError(
          f"cluster_config.local.{bool_key} must be a bool, got"
          f" {local[bool_key]!r}."
      )
  for device_key in ("trainer_devices", "rollout_devices", "inference_devices"):
    val = local[device_key]
    if val is None:
      local[device_key] = ""
    elif not isinstance(val, bool) and isinstance(val, int):
      if val < 0:
        raise ValueError(
            f"cluster_config.local.{device_key} must be non-negative, got"
            f" {val}."
        )
      local[device_key] = str(val)
    elif isinstance(val, str):
      local[device_key] = val.strip()
    else:
      raise TypeError(
          f"cluster_config.local.{device_key} must be a str or non-negative"
          f" int, got {val!r}."
      )


def _validate_dist_invariants(hp: cli_config.HyperParameters) -> None:
  """Validates cross-field invariants across trainer and rollout configs."""
  # TODO(noghabi): Check which of these checks can be consolidated with the
  # `tunix.cli` validations or left to the constructors that consume them.
  _validate_rollout_cache_settings(hp)
  _validate_k8s_jax_cache_gcs(hp)
  _validate_k8s_scalars(hp)
  _validate_k8s_worker_jobsets(hp)
  _validate_rollout_replicas(hp)
  _validate_eval_config(hp)
  _validate_role_env(hp)
  _reject_unimplemented_settings(hp)
  _validate_role_memory(hp)
  _validate_launcher_settings(hp)
  _validate_trajectory_logging(hp)
  _validate_rollout_logps_settings(hp)
  _validate_fault_tolerance(hp)
  batch_size = int(hp.config["batch_size"])
  rl_train = hp.config_mapping("rl_training_config")
  mini_batch_size = int(rl_train["mini_batch_size"])
  if batch_size <= 0 or mini_batch_size <= 0:
    raise ValueError(
        "batch_size and rl_training_config.mini_batch_size must be > 0."
    )
  if batch_size % mini_batch_size != 0:
    raise ValueError(
        f"batch_size ({batch_size}) must be divisible by"
        f" rl_training_config.mini_batch_size ({mini_batch_size})."
    )

  actor_cfg = hp.config_mapping("actor_model_config")
  actor_axes = hp.mesh_axes("actor_model_config")
  # Parses the rollout mesh, so a malformed one fails here.
  hp.mesh_axes("rollout_model_config")
  if actor_cfg["model_source"] == "maxtext":
    if not actor_cfg["model_path"]:
      raise ValueError(
          "actor_model_config.model_source='maxtext' requires"
          " actor_model_config.model_path (or model_config.model_path)."
      )
    if not rl_train["max_seq_tokens_per_device"]:
      micro_batch = int(rl_train["train_micro_batch_size"])
      if micro_batch % actor_axes["fsdp"] != 0:
        raise ValueError(
            f"rl_training_config.train_micro_batch_size ({micro_batch}) must"
            " be divisible by actor_model_config.mesh fsdp"
            f" ({actor_axes['fsdp']}) when model_source='maxtext'."
        )

  _validate_typed_vllm_settings(hp)

  for role_key in (
      "actor_model_config",
      "reference_model_config",
      "rollout_model_config",
  ):
    role_cfg = hp.config_mapping(role_key)
    for dtype_key in ("dtype", "load_dtype"):
      _check_choice(
          f"{role_key}.{dtype_key}", role_cfg[dtype_key], _MODEL_DTYPES
      )
  _check_choice(
      "actor_model_config.remat_config",
      str(actor_cfg["remat_config"]).lower(),
      _REMAT_CONFIGS,
  )
  _check_choice("rollout_engine", hp.config["rollout_engine"], _ROLLOUT_ENGINES)
  _check_choice(
      "chat_parser_config.type",
      hp.config_mapping("chat_parser_config")["type"],
      _CHAT_PARSER_TYPES,
  )
  grpo = hp.config_mapping("agentic_grpo_config")
  _check_choice(
      "agentic_grpo_config.trajectory_group_order",
      grpo["trajectory_group_order"],
      tuple(order.value for order in dist_rl_datatypes.GroupOrder),
  )
  max_version_span = grpo["max_version_span"]
  if max_version_span is not None:
    if int(max_version_span) < 0:
      raise ValueError(
          "agentic_grpo_config.max_version_span must be non-negative, got"
          f" {max_version_span}."
      )
    if (
        grpo["trajectory_group_order"]
        == dist_rl_datatypes.GroupOrder.PROMPT_ARRIVAL.value
    ):
      raise ValueError(
          "agentic_grpo_config.max_version_span is only supported with"
          " trajectory_group_order='trajectory_completion'."
      )
  _check_choice(
      "weight_sync_config.mode",
      hp.config_mapping("weight_sync_config")["mode"],
      tuple(mode.value for mode in common_datatypes.WeightSyncMode),
  )
  _validate_maxtext_pairing(hp)
  _validate_maxtext_model_names(hp)
  _validate_maxtext_extra_config(hp)
  _validate_maxtext_skip_step_kwargs(hp)
  _validate_freeze_router(hp)
  _validate_rollout_weights(hp)
  _validate_library_settings(hp)
  if grpo["reward_mode"] == "reward_fn" and not hp.config["reward_functions"]:
    raise ValueError(
        "agentic_grpo_config.reward_mode='reward_fn' needs at least one entry"
        " in reward_functions; use reward_mode='env' to score with the"
        " environment rewards."
    )
  # TODO(noghabi): Check `cluster_config.rpc_timeout_s` against the weight-sync
  # phase timeouts here (now done by the orchestrator at startup) once
  # `PhaseTimeouts` moves out of `weight_sync_coordinator`, which imports gRPC
  # and Raiden, into `dist_rl/common/timeouts.py`.
  num_generations = int(grpo["num_generations"])
  if num_generations <= 1:
    raise ValueError(
        "agentic_grpo_config.num_generations must be > 1 for GRPO, got"
        f" {num_generations}."
    )
  micro_batch = int(rl_train["train_micro_batch_size"])
  if micro_batch <= 0:
    raise ValueError(
        "rl_training_config.train_micro_batch_size must be > 0, got"
        f" {micro_batch}."
    )
  # `train_micro_batch_size` counts trajectories in the distributed trainer:
  # gradient accumulation = mini_batch_size * num_generations /
  # train_micro_batch_size (not applicable with sequence packing).
  if not rl_train["max_seq_tokens_per_device"]:
    trajectories = mini_batch_size * num_generations
    if trajectories % micro_batch != 0:
      raise ValueError(
          "rl_training_config.mini_batch_size *"
          " agentic_grpo_config.num_generations"
          f" ({trajectories}) must be divisible by"
          f" rl_training_config.train_micro_batch_size ({micro_batch})."
      )


def _validate_maxtext_pairing(hp: cli_config.HyperParameters) -> None:
  """Requires a MaxText trainer and a MaxText rollout, or neither.

  `model_source: maxtext` selects MaxText's model implementation, and with it
  MaxText's parameter names; other sources load the same checkpoint into Tunix
  or vLLM implementations, whose names differ. Weight sync pairs trainer and
  rollout parameters by name, so both mixes pair nothing: a MaxText trainer
  with a non-MaxText rollout (e.g. only `actor_model_config.model_source` set),
  and a Tunix trainer with vLLM's `MaxTextForCausalLM`. Set
  `model_config.model_source` to set both roles.

  Args:
    hp: Loaded configuration.

  Raises:
    ValueError: If weight sync is on and exactly one of
      `actor_model_config.model_source` and `rollout_model_config.model_source`
      is "maxtext".
  """
  mode = hp.config_mapping("weight_sync_config")["mode"]
  if mode in (
      common_datatypes.WeightSyncMode.NONE.value,
      common_datatypes.WeightSyncMode.NOOP.value,
  ):
    return
  actor_source = hp.config_mapping("actor_model_config")["model_source"]
  rollout_source = hp.config_mapping("rollout_model_config")["model_source"]
  if (actor_source == "maxtext") != (rollout_source == "maxtext"):
    raise ValueError(
        f"weight_sync_config.mode={mode!r} pairs trainer and rollout tensors by"
        " name, so actor_model_config.model_source and"
        " rollout_model_config.model_source must both be 'maxtext' or both"
        f" not; got {actor_source!r} and {rollout_source!r}. Set"
        " model_config.model_source to set both."
    )


def rollout_init_with_random_weights(hp: cli_config.HyperParameters) -> bool:
  """Returns the effective `vllm_config.init_with_random_weights`.

  True makes the `inprocess_vllm` rollout start from random weights (vLLM's
  `load_format="dummy"`) instead of loading the checkpoint. Only Raiden weight
  sync replaces them: its bootstrap sync loads the trainer's weights before
  step 0, whereas `none` skips the sync.

  Args:
    hp: Loaded configuration.

  Returns:
    `vllm_config.init_with_random_weights`, or, when it is null, whether
    `weight_sync_config.mode` is "raiden".

  Raises:
    TypeError: If `vllm_config.init_with_random_weights` is not a bool or null.
  """
  value = hp.config_mapping("vllm_config")["init_with_random_weights"]
  if value is None:
    mode = hp.config_mapping("weight_sync_config")["mode"]
    return mode == common_datatypes.WeightSyncMode.RAIDEN.value
  if not isinstance(value, bool):
    raise TypeError(
        "vllm_config.init_with_random_weights must be a bool or null, got"
        f" {value!r}."
    )
  return value


def _validate_rollout_weights(hp: cli_config.HyperParameters) -> None:
  """Rejects a vLLM rollout that would never get real weights.

  Without Raiden weight sync, the rollout keeps the weights it starts from for
  the whole run. vLLM's `MaxTextForCausalLM` always starts from random weights:
  it loads weights only from `maxtext_config.load_parameters_path`, which the
  rollout does not set (`maxtext_utils.build_vllm_maxtext_additional_config`).

  Args:
    hp: Loaded configuration.

  Raises:
    TypeError: If `vllm_config.init_with_random_weights` is not a bool or null.
    ValueError: If `weight_sync_config.mode` is not "raiden" and either a vLLM
      `rollout_engine` serves `rollout_model_config.model_source: maxtext`, or
      `rollout_engine` is "inprocess_vllm" with
      `vllm_config.init_with_random_weights` true.
  """
  random_init = rollout_init_with_random_weights(hp)
  mode = hp.config_mapping("weight_sync_config")["mode"]
  if mode == common_datatypes.WeightSyncMode.RAIDEN.value:
    return
  rollout_engine = hp.config["rollout_engine"]
  rollout_source = hp.config_mapping("rollout_model_config")["model_source"]
  if rollout_source == "maxtext" and rollout_engine != "vanilla":
    raise ValueError(
        f"A MaxText {rollout_engine} rollout"
        " (rollout_model_config.model_source='maxtext') needs"
        f" weight_sync_config.mode=raiden, got {mode!r}. vLLM's"
        " MaxTextForCausalLM loads weights only from"
        " maxtext_config.load_parameters_path, which the rollout does not set,"
        " so the rollout would sample from random weights for the whole run."
        " Use weight_sync_config.mode=raiden with a MaxText trainer"
        " (model_config.model_source=maxtext sets both roles)."
    )
  if random_init and rollout_engine == "inprocess_vllm":
    raise ValueError(
        "vllm_config.init_with_random_weights=true starts the inprocess_vllm"
        f" rollout from random weights, and weight_sync_config.mode={mode!r}"
        " never loads the trainer's weights into it, so the rollout would"
        " sample from random weights for the whole run. Set"
        " vllm_config.init_with_random_weights to false or null to load the"
        " checkpoint, or use weight_sync_config.mode=raiden."
    )


def _validate_typed_vllm_settings(hp: cli_config.HyperParameters) -> None:
  """Validates the typed `vllm_config` rollout settings."""
  vllm = hp.config_mapping("vllm_config")
  for key in ("return_routed_experts", "detokenize"):
    if not isinstance(vllm[key], bool):
      raise TypeError(f"vllm_config.{key} must be a bool, got {vllm[key]!r}.")
  common_configs.validate_routed_experts_encode_config(
      vllm["routed_experts_encode_mode"], vllm["routed_experts_num_experts"]
  )
  if vllm["return_routed_experts"] and hp.config["rollout_engine"] == "vanilla":
    raise ValueError(
        "vllm_config.return_routed_experts needs a vLLM rollout_engine; the"
        " vanilla sampler does not capture routed experts."
    )


def _validate_maxtext_model_names(hp: cli_config.HyperParameters) -> None:
  """Requires one `kwargs.maxtext_model_name` for the MaxText trainer and rollout.

  The trainer, the rollout (vLLM's `MaxTextForCausalLM`), and the Hugging Face
  to Orbax conversion in `local.sh` all take the MaxText model name from
  `kwargs.maxtext_model_name`; nothing falls back to `model_name`.

  Args:
    hp: Loaded configuration.

  Raises:
    TypeError: If a MaxText role's `kwargs` is not a mapping.
    ValueError: If the actor or rollout has `model_source: maxtext` but its
      `kwargs.maxtext_model_name` is missing, not a string, empty, or has
      surrounding spaces, or if weight sync is on and the two name different
      MaxText models.
  """
  names: dict[str, str] = {}
  for role_key in ("actor_model_config", "rollout_model_config"):
    role_cfg = hp.config_mapping(role_key)
    if role_cfg["model_source"] != "maxtext":
      continue
    kwargs = role_cfg["kwargs"]
    if not isinstance(kwargs, Mapping):
      raise TypeError(
          f"{role_key}.kwargs must be a mapping when {role_key}.model_source"
          f" is 'maxtext' (it holds maxtext_model_name), got {kwargs!r}."
      )
    if "maxtext_model_name" not in kwargs:
      raise ValueError(
          f"{role_key}.model_source='maxtext' requires"
          f" {role_key}.kwargs.maxtext_model_name (or"
          " model_config.kwargs.maxtext_model_name), e.g. 'qwen3-0.6b'."
      )
    name = kwargs["maxtext_model_name"]
    if not isinstance(name, str) or not name or name != name.strip():
      raise ValueError(
          f"{role_key}.kwargs.maxtext_model_name must be a non-empty string"
          f" without surrounding spaces, e.g. 'qwen3-0.6b'; got {name!r}."
      )
    names[role_key] = name
  mode = hp.config_mapping("weight_sync_config")["mode"]
  if (
      mode
      not in (
          common_datatypes.WeightSyncMode.NONE.value,
          common_datatypes.WeightSyncMode.NOOP.value,
      )
      and len(names) == 2
      and names["actor_model_config"] != names["rollout_model_config"]
  ):
    raise ValueError(
        f"weight_sync_config.mode={mode!r} syncs the trainer's weights into the"
        " rollout, so actor_model_config.kwargs.maxtext_model_name"
        f" ({names['actor_model_config']!r}) and"
        " rollout_model_config.kwargs.maxtext_model_name"
        f" ({names['rollout_model_config']!r}) must match."
    )


# MaxText config keys that the trainer builds from this config, mapped to a
# hint naming the config keys they come from. Only the error message below uses
# the hints; the values themselves are computed in
# `trainer._create_maxtext_trainer_factory` and
# `maxtext_utils.build_maxtext_config`. `maxtext_extra_config` is appended last,
# so an entry for one of these keys would silently override the config value.
# The fixed settings the builder adds (e.g. `scan_layers`, `grad_dtype`,
# `use_tokamax_gmm`) have no config key and may still be overridden.
# `trainer_test.MaxTextExtraConfigTest` checks this key set against the builder.
_MAXTEXT_KEYS_FROM_CONFIG: Final[Mapping[str, str]] = {
    "model_name": "actor_model_config.kwargs.maxtext_model_name",
    "load_parameters_path": "actor_model_config.model_path",
    "base_output_directory": (
        "actor_model_config.kwargs.maxtext_output_dir (default:"
        " <cluster_config.artifact_root>/maxtext)"
    ),
    "weight_dtype": "actor_model_config.load_dtype",
    "attention": "actor_model_config.kwargs.attention",
    "padded_base_moe_mlp_dim": (
        "actor_model_config.kwargs.padded_moe_mlp_dim (when 0, derived from"
        " the 'tp' axis of rollout_model_config.mesh)"
    ),
    "base_num_kv_heads": (
        "actor_model_config.kwargs.base_num_kv_heads (replicated up to the"
        " 'tp' axis of rollout_model_config.mesh)"
    ),
    "prefuse_moe_weights": "actor_model_config.kwargs.prefuse_moe_weights",
    "ici_fsdp_parallelism": "the 'fsdp' axis of actor_model_config.mesh",
    "ici_tensor_parallelism": "the 'tp' axis of actor_model_config.mesh",
    "ici_expert_parallelism": "the 'expert' axis of actor_model_config.mesh",
    "ici_context_parallelism": "actor_model_config.kwargs.mesh_context",
    "rollout_tensor_parallelism": "the 'tp' axis of rollout_model_config.mesh",
    "per_device_batch_size": (
        "rl_training_config.train_micro_batch_size, or with sequence packing"
        " the 'fsdp' and 'expert' axes of actor_model_config.mesh, per device"
    ),
    "gradient_accumulation_steps": (
        "rl_training_config.mini_batch_size,"
        " agentic_grpo_config.num_generations and"
        " rl_training_config.train_micro_batch_size (1 with sequence packing)"
    ),
    "max_target_length": (
        "rollout_config.max_prompt_length and total_generation_steps, or"
        " rl_training_config.max_seq_tokens_per_device"
    ),
    "learning_rate": "rl_training_config.actor_optimizer_config.learning_rate",
    "warmup_steps_fraction": "actor_model_config.kwargs.warmup_steps_fraction",
    "adam_b1": "rl_training_config.actor_optimizer_config.b1",
    "adam_b2": "rl_training_config.actor_optimizer_config.b2",
    "adam_eps": "rl_training_config.actor_optimizer_config.eps",
    "adam_weight_decay": (
        "rl_training_config.actor_optimizer_config.weight_decay"
    ),
    "gradient_clipping_threshold": (
        "rl_training_config.actor_optimizer_config.chain_kwargs.max_norm"
    ),
    "enable_checkpointing": (
        "rl_training_config.checkpointing_options.save_interval_steps"
    ),
    "checkpoint_period": (
        "rl_training_config.checkpointing_options.save_interval_steps"
    ),
    "max_num_checkpoints_to_keep": (
        "rl_training_config.checkpointing_options.max_to_keep"
    ),
    "checkpoint_storage_device_host_concurrent_gb": (
        "dist_training_config.ckpt_d2h_concurrent_gb"
    ),
    "use_weight_converter": "weight_sync_config.use_weight_converter",
    "profiler": "rl_training_config.profiler_options",
    "profiler_steps": "rl_training_config.profiler_options",
    "skip_first_n_steps_for_profiler": "rl_training_config.profiler_options",
    "profile_periodically_period": "dist_training_config.profiler_period",
    "upload_all_profiler_results": (
        "actor_model_config.kwargs.upload_all_profiler_results"
    ),
    "enable_tpu_profiling_options": (
        "actor_model_config.kwargs.enable_tpu_profiling_options"
    ),
    "tpu_num_chips_to_profile_per_task": (
        "actor_model_config.kwargs.tpu_num_chips_to_profile_per_task"
    ),
    "tpu_num_sparse_cores_to_trace": (
        "actor_model_config.kwargs.tpu_num_sparse_cores_to_trace"
    ),
    "tpu_num_sparse_core_tiles_to_trace": (
        "actor_model_config.kwargs.tpu_num_sparse_core_tiles_to_trace"
    ),
    "skip_step_on_spikes": "actor_model_config.kwargs.skip_step_on_spikes",
    "skip_step_on_nan": "actor_model_config.kwargs.skip_step_on_nan",
    "skip_step_interval": "actor_model_config.kwargs.skip_step_interval",
    "skip_step_scaling_factor": (
        "actor_model_config.kwargs.skip_step_scaling_factor"
    ),
}


# MaxText config names are Python identifiers (fields of its config class).
_MAXTEXT_CONFIG_NAME_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*"
)


def _validate_maxtext_skip_step_kwargs(hp: cli_config.HyperParameters) -> None:
  """Validates `actor_model_config.kwargs.skip_step_*` types and backend."""
  actor = hp.config_mapping("actor_model_config")
  kwargs = actor["kwargs"]
  if not isinstance(kwargs, Mapping):
    return
  keys = (
      "skip_step_on_spikes",
      "skip_step_on_nan",
      "skip_step_interval",
      "skip_step_scaling_factor",
  )
  present = [k for k in keys if k in kwargs and kwargs[k] is not None]
  if not present:
    return
  if actor["model_source"] != "maxtext":
    raise ValueError(
        "actor_model_config.kwargs."
        + ", actor_model_config.kwargs.".join(present)
        + " require actor_model_config.model_source='maxtext', got"
        f" {actor['model_source']!r}."
    )
  for bool_key in ("skip_step_on_spikes", "skip_step_on_nan"):
    if bool_key in kwargs and kwargs[bool_key] is not None:
      if not isinstance(kwargs[bool_key], bool):
        raise ValueError(
            f"actor_model_config.kwargs.{bool_key} must be a bool, got"
            f" {kwargs[bool_key]!r}."
        )
  if "skip_step_interval" in kwargs and kwargs["skip_step_interval"] is not None:
    interval = kwargs["skip_step_interval"]
    if (
        isinstance(interval, bool)
        or not isinstance(interval, int)
        or interval <= 0
    ):
      raise ValueError(
          "actor_model_config.kwargs.skip_step_interval must be a positive int,"
          f" got {interval!r}."
      )
  if (
      "skip_step_scaling_factor" in kwargs
      and kwargs["skip_step_scaling_factor"] is not None
  ):
    factor = kwargs["skip_step_scaling_factor"]
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not (0 < factor < float("inf"))
    ):
      raise ValueError(
          "actor_model_config.kwargs.skip_step_scaling_factor must be a"
          f" positive finite number, got {factor!r}."
      )


def _validate_maxtext_extra_config(hp: cli_config.HyperParameters) -> None:
  """Checks `dist_training_config.maxtext_extra_config` before any job starts.

  The entries are extra MaxText config keys, which only a MaxText trainer
  reads (`maxtext_utils.build_maxtext_config`); a Tunix trainer would silently
  ignore them. The trainer passes each one to MaxText as `key=value`, so a
  value must be a scalar; anything else would otherwise fail only inside the
  trainer, after the launcher has submitted the jobs. An entry for a key the
  trainer builds from this config would silently override that config value.

  Args:
    hp: Loaded configuration.

  Raises:
    TypeError: If the section is not a mapping, or a value is not a str, int,
      float or bool.
    ValueError: If entries are set and `actor_model_config.model_source` is not
      "maxtext", a key is not a MaxText config name, or an entry sets a key in
      `_MAXTEXT_KEYS_FROM_CONFIG`.
  """
  extra = hp.config_mapping("dist_training_config")["maxtext_extra_config"]
  if not isinstance(extra, Mapping):
    raise TypeError(
        "dist_training_config.maxtext_extra_config must be a mapping of"
        f" MaxText config keys to values, got {extra!r}."
    )
  if not extra:
    return
  actor_source = hp.config_mapping("actor_model_config")["model_source"]
  if actor_source != "maxtext":
    raise ValueError(
        "dist_training_config.maxtext_extra_config configures a MaxText"
        f" trainer, but actor_model_config.model_source is {actor_source!r}."
        " Set model_config.model_source=maxtext or drop the entries."
    )
  # A key such as "learning_rate=1e-6" would become the MaxText argument
  # `learning_rate=1e-6=1` and slip past `_MAXTEXT_KEYS_FROM_CONFIG`.
  bad_keys = [
      key
      for key in extra
      if not isinstance(key, str) or not _MAXTEXT_CONFIG_NAME_RE.fullmatch(key)
  ]
  if bad_keys:
    raise ValueError(
        "dist_training_config.maxtext_extra_config keys must be MaxText config"
        f" names (letters, digits and underscores), got {bad_keys!r}."
    )
  not_scalar = {
      key: value
      for key, value in extra.items()
      if not isinstance(value, (str, int, float, bool))
  }
  if not_scalar:
    raise TypeError(
        "dist_training_config.maxtext_extra_config values must be a str, int,"
        f" float or bool, got {not_scalar!r}."
    )
  from_config = [
      f"{key} (set {_MAXTEXT_KEYS_FROM_CONFIG[key]} instead)"
      for key in extra
      if key in _MAXTEXT_KEYS_FROM_CONFIG
  ]
  if from_config:
    raise ValueError(
        "dist_training_config.maxtext_extra_config must not set MaxText keys"
        " that the trainer builds from the config: "
        + ", ".join(from_config)
        + "."
    )


def _validate_freeze_router(hp: cli_config.HyperParameters) -> None:
  """Checks `actor_model_config.freeze_router` type and trainer backend."""
  actor = hp.config_mapping("actor_model_config")
  freeze_router = actor["freeze_router"]
  if not isinstance(freeze_router, bool):
    raise TypeError(
        "actor_model_config.freeze_router must be a bool, got"
        f" {freeze_router!r}."
    )
  if freeze_router and actor["model_source"] == "maxtext":
    raise ValueError(
        "actor_model_config.freeze_router is not supported by the MaxText"
        " trainer (actor_model_config.model_source='maxtext')."
    )


# `launch.sh` starts the `local` processes (through `local.sh`) and the `k8s`
# ones (through `k8s.sh`); another launcher starts the `external` ones, e.g. a
# job scheduler that starts each process separately.
_CLUSTER_TARGETS = ("local", "k8s", "external")

_MODEL_DTYPES = ("bfloat16", "float32")
_REMAT_CONFIGS = ("none", "block", "decoder")
_ROLLOUT_ENGINES = ("inprocess_vllm", "vllm", "vanilla")
_CHAT_PARSER_TYPES = ("auto", "raw")


def _check_choice(key: str, value: object, choices: Sequence[str]) -> None:
  if value not in choices:
    raise ValueError(f"{key} must be one of {choices}, got {value!r}.")


# Settings the worker nodes hand to the libraries as environment variables
# (`node_utils.apply_library_env`), as (section, key).
_LIBRARY_BOOL_SETTINGS = (
    ("weight_sync_config", "verify_weights"),
    ("vllm_config", "skip_jax_precompile"),
)


def _validate_library_settings(hp: cli_config.HyperParameters) -> None:
  """Checks the types of the settings the nodes hand to the libraries.

  The resolver checks that a key exists, not its type: a `section.key=value`
  override takes whatever type its value parses to. Without this check a
  malformed value would be misread or fail only inside a node, after its TPUs
  are allocated: e.g. `weight_sync_config.verify_weights='false'` is a
  non-empty string, so `node_utils.apply_library_env` would export
  `VERIFY_WEIGHTS=true`, and `dist_training_config.ckpt_d2h_concurrent_gb=8.5`
  would fail inside MaxText or Orbax on the trainer.

  Args:
    hp: Loaded configuration.

  Raises:
    TypeError: If a `_LIBRARY_BOOL_SETTINGS` value is not a bool,
      `weight_sync_config.raiden.data_nics` is not a string, or
      `dist_training_config.ckpt_d2h_concurrent_gb` is neither an int nor
      null.
    ValueError: If `dist_training_config.ckpt_d2h_concurrent_gb` is not
      positive, or is set for the Tunix trainer.
  """
  for section, key in _LIBRARY_BOOL_SETTINGS:
    value = hp.config_mapping(section)[key]
    if not isinstance(value, bool):
      raise TypeError(f"{section}.{key} must be a bool, got {value!r}.")
  nics = hp.config_mapping("weight_sync_config")["raiden"]["data_nics"]
  if not isinstance(nics, str):
    raise TypeError(
        "weight_sync_config.raiden.data_nics must be a string of"
        f" comma-separated NIC names, or empty, got {nics!r}."
    )
  gb = hp.config_mapping("dist_training_config")["ckpt_d2h_concurrent_gb"]
  if gb is None:
    return
  if isinstance(gb, bool) or not isinstance(gb, int):
    raise TypeError(
        "dist_training_config.ckpt_d2h_concurrent_gb must be an int or null,"
        f" got {gb!r}."
    )
  if gb <= 0:
    raise ValueError(
        "dist_training_config.ckpt_d2h_concurrent_gb must be positive, got"
        f" {gb}."
    )
  # TODO(noghabi): Cap the Tunix trainer's checkpoint D2H copies too
  # (`TunixCheckpointingOptions.device_host_concurrent_gb` takes the value)
  # once step-time data on a checkpointing step shows async saves still
  # overlap training: Orbax may block on earlier writes past the cap.
  if hp.config_mapping("actor_model_config")["model_source"] != "maxtext":
    raise ValueError(
        "dist_training_config.ckpt_d2h_concurrent_gb only applies to the"
        " MaxText trainer (actor_model_config.model_source='maxtext'); the"
        f" Tunix trainer has no D2H cap yet, so leave it null (got {gb})."
    )


def initialize(overrides: Sequence[str]) -> DistHyperParameters:
  """Initializes `DistHyperParameters` from overrides only (no program name)."""
  return DistHyperParameters(overrides)


# Top-level keys that name the task a run trains on. `base_dist_config.yaml`
# leaves them empty; each task YAML sets them.
_TASK_KEYS: Final[tuple[str, ...]] = (
    "data_module",
    "env_class_path",
    "agent_class_path",
    "registry_module",
)


def task_config_error(hp: cli_config.HyperParameters) -> str:
  """Returns why the config cannot start a run, or "" when it names a task.

  A run needs the task's dataset (`data_module`), the registered names of its
  environment and agent (`env_class_path`, `agent_class_path`), and the module
  that registers them (`registry_module`). `initialize` does not
  check this: some callers resolve a config without starting a run, e.g.
  `launch.sh stop`.

  Args:
    hp: Resolved configuration.
  """
  missing = [key for key in _TASK_KEYS if not hp.config[key]]
  if not missing:
    return ""
  return (
      f"The config names no task: {', '.join(missing)} must be set. Pass the"
      " task's YAML as override_config_file=<path> (e.g."
      " tunix/dist_rl/examples/math_gsm8k_dist/gsm8k.yaml)."
  )


def validate_task(hp: cli_config.HyperParameters) -> None:
  """Checks that the config names a task (see `task_config_error`).

  Args:
    hp: Resolved configuration.

  Raises:
    ValueError: If a task key is empty.
  """
  error = task_config_error(hp)
  if error:
    raise ValueError(error)


def plain_config(hp: cli_config.HyperParameters) -> dict[str, object]:
  """Returns the resolved config as plain, JSON-serializable containers.

  Args:
    hp: Resolved configuration.

  Returns:
    A deep copy of `hp.config` in which OmegaConf containers (e.g. the
    top-level `reward_functions` list), at any depth, are plain dicts and
    lists, and every mapping key is a string.

  Raises:
    omegaconf.errors.MissingMandatoryValue: If a value is still OmegaConf's
      mandatory `???`.
  """
  return {str(key): _plain(value) for key, value in hp.config.items()}


def _plain(value: object) -> object:
  """Returns a deep copy of `value` with OmegaConf containers made plain."""
  if isinstance(value, (omegaconf.DictConfig, omegaconf.ListConfig)):
    value = omegaconf.OmegaConf.to_container(
        value, resolve=True, throw_on_missing=True
    )
  if isinstance(value, Mapping):
    return {str(key): _plain(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_plain(item) for item in value]
  return value


def config_fingerprint(hp: cli_config.HyperParameters) -> str:
  """Returns the sha256 hex digest of the resolved config's canonical JSON.

  Two processes have the same fingerprint exactly when they resolved the same
  config, so comparing the nodes' fingerprints shows whether they got the same
  overrides.

  Args:
    hp: Resolved configuration.

  Raises:
    TypeError: If the config holds a value that is not JSON-serializable.
  """
  canonical = json.dumps(
      plain_config(hp), sort_keys=True, separators=(",", ":")
  )
  return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def rollout_cache_params(
    hp: cli_config.HyperParameters,
) -> gcs_cache.RolloutCacheKeyParams:
  """Returns the rollout's compilation-cache key, from the config alone.

  On k8s, the launcher derives the rollout pods' cache URI from it
  (`render_bash_exports`). The rollout node keys its cache with it when it gets
  no URI (e.g. local runs, where `cluster_config.k8s.rollout.tpu_slice` still
  names the slice). Key fields that no config key sets keep their defaults;
  JAX's own cache key still tells apart the programs they change (e.g. through
  `LIBTPU_INIT_ARGS`).

  Args:
    hp: Resolved configuration.
  """
  # TODO(noghabi): Add `float32_logits` and the vLLM quantization to the key
  # once they are schema keys.
  rollout_model = hp.config_mapping("rollout_model_config")
  rollout = hp.config_mapping("rollout_config")
  vllm = hp.config_mapping("vllm_config")
  axes = hp.mesh_axes("rollout_model_config")
  lora_config = rollout_model["lora_config"]
  max_model_len = vllm["max_model_len"]
  return gcs_cache.RolloutCacheKeyParams(
      # TODO(noghabi): Name the local rollout's slice from the local devices or
      # config instead of `cluster_config.k8s.rollout.tpu_slice`.
      tpu_slice=str(
          hp.config_mapping("cluster_config")["k8s"]["rollout"]["tpu_slice"]
      ),
      model_name=str(
          rollout_model["kwargs"]["maxtext_model_name"]
          if rollout_model["model_source"] == "maxtext"
          else (rollout_model["model_name"] or rollout_model["model_id"])
      ),
      sampler=str(hp.config["rollout_engine"]),
      max_prompt_length=int(rollout["max_prompt_length"]),
      max_response_length=int(rollout["total_generation_steps"]),
      vllm_max_model_len=None if max_model_len is None else int(max_model_len),
      mesh_tp=axes["tp"],
      mesh_fsdp=axes["fsdp"],
      mesh_expert=axes["expert"],
      use_lora=bool(lora_config),
      # `cache_fingerprint` hashes the rank only with LoRA.
      lora_rank=int(lora_config["rank"]) if lora_config else 0,
      # In-flight weight updates force prefix caching on, as in
      # `rollout._common_vllm_engine_kwargs`.
      enable_prefix_caching=bool(
          vllm["enable_prefix_caching"]
          or hp.config_mapping("weight_sync_config")["in_flight_weight_updates"]
      ),
      vllm_async_scheduling=bool(vllm["async_scheduling"]),
  )


def _bash_quote(val: object) -> str:
  if isinstance(val, bool):
    return "true" if val else "false"
  if val is None:
    return "''"
  return shlex.quote(str(val))


def _shell_env_words(env: Mapping[str, str]) -> str:
  """Renders `env` as space-separated `KEY=VALUE` words, each value quoted."""
  return " ".join(f"{name}={shlex.quote(value)}" for name, value in env.items())


# TODO(noghabi): Consider auto-flattening `cluster_config.{local,k8s}` into
# `UPPER_SNAKE_CASE` exports (and mapping only the non-cluster keys by hand),
# which would shrink this list and the drift check in `launch_test.py`.
def render_bash_exports(hp: DistHyperParameters) -> str:
  """Renders cluster/deployment shell exports for `local.sh` and `k8s.sh`."""
  cluster = hp.config_mapping("cluster_config")
  roles = {
      role: cli_config.to_str_dict(cluster["roles"][role])
      for role in PROCESS_ROLES
  }
  memory = {
      role: cli_config.to_str_dict(roles[role]["memory"])
      for role in ("orchestrator", "trainer", "rollout")
  }
  local = cli_config.to_str_dict(cluster["local"])
  k8s = cli_config.to_str_dict(cluster["k8s"])
  jax_cache = cli_config.to_str_dict(k8s["jax_cache"])
  k8s_trainer = cli_config.to_str_dict(k8s["trainer"])
  k8s_rollout = cli_config.to_str_dict(k8s["rollout"])
  pathways = cli_config.to_str_dict(cluster["pathways"])
  sidecar = cli_config.to_str_dict(pathways["sidecar"])
  pathways_trainer = cli_config.to_str_dict(pathways["trainer"])
  proxy_memory = cli_config.to_str_dict(pathways["proxy_memory"])
  rm_memory = cli_config.to_str_dict(pathways["rm_memory"])
  server_memory = cli_config.to_str_dict(pathways["server_memory"])
  sidecar_memory = cli_config.to_str_dict(sidecar["memory"])
  ckpt_gcsfuse = cli_config.to_str_dict(pathways_trainer["checkpoint_gcsfuse"])
  ifrt_threshold = pathways_trainer["ifrt_proxy_large_transfer_threshold_bytes"]
  actor = hp.config_mapping("actor_model_config")
  rollout_model = hp.config_mapping("rollout_model_config")
  ws = hp.config_mapping("weight_sync_config")
  raiden = cli_config.to_str_dict(ws["raiden"])
  otel = hp.config_mapping("otel_config")
  sandbox_k8s = (
      hp.config["kubernetes_config"]
      if isinstance(hp.config["kubernetes_config"], Mapping)
      else {}
  )
  trainer_backend = "maxtext" if actor["model_source"] == "maxtext" else "tunix"
  cluster_targets = hp.cluster_targets
  # The rollout pods' GCS compilation-cache URI, from the config alone and never
  # the launcher's environment: `gcs_cache` reads the settings from this
  # mapping, in its usual order (`rollout_gcs_dir`, then `gcs_dir`, `bucket`).
  rollout_cache_uri: str | None = None
  if not jax_cache["disable"]:
    rollout_cache_uri = gcs_cache.derive_rollout_cache_uri(
        params=rollout_cache_params(hp),
        env={
            "ROLLOUT_JAX_CACHE_GCS_DIR": str(jax_cache["rollout_gcs_dir"]),
            "JAX_CACHE_GCS_DIR": str(jax_cache["gcs_dir"]),
            "JAX_CACHE_BUCKET": str(jax_cache["bucket"]),
        },
    )

  exports: dict[str, object] = {
      "CLUSTER_TARGET": cluster["target"],
      # Empty when the config names a task (`task_config_error`).
      "TASK_CONFIG_ERROR": task_config_error(hp),
      "EXECUTION_PLACEMENT": cluster_targets.placement.value,
      "TRAINER_FRAMEWORK": cluster_targets.trainer.framework.value,
      "TRAINER_ACCELERATOR": cluster_targets.trainer.accelerator.value,
      "ROLLOUT_FRAMEWORK": cluster_targets.rollout.framework.value,
      "ROLLOUT_ACCELERATOR": cluster_targets.rollout.accelerator.value,
      "INFERENCE_FRAMEWORK": cluster_targets.inference.framework.value,
      "INFERENCE_ACCELERATOR": cluster_targets.inference.accelerator.value,
      "ORCHESTRATOR_MAIN": roles["orchestrator"]["main"],
      "TRAINER_MAIN": roles["trainer"]["main"],
      "ROLLOUT_MAIN": roles["rollout"]["main"],
      "INFERENCE_MAIN": roles["inference"]["main"],
      "ARTIFACT_ROOT": cluster["artifact_root"],
      "RUN_ID": cluster["run_id"],
      "DRY_RUN": cluster["dry_run"],
      "WAIT_TIMEOUT_SECS": local["wait_timeout_secs"],
      "WAIT_POLL_SECS": local["wait_poll_secs"],
      "WAIT_DEBUG_EVERY_POLLS": local["wait_debug_every_polls"],
      "WAIT_LOG_TAIL_LINES": local["wait_log_tail_lines"],
      "RUN_INFERENCE_NODE": cluster["run_inference_node"],
      "ROLLOUT_REPLICAS": cluster["rollout_replicas"],
      "ORCHESTRATOR_PORT": roles["orchestrator"]["port"],
      "TRAINER_PORT": roles["trainer"]["port"],
      "ROLLOUT_PORT": roles["rollout"]["port"],
      "INFERENCE_PORT": roles["inference"]["port"],
      "OTEL_PORT": cluster["ports"]["otel"],
      "OTEL_GOOGLECLOUD": otel["googlecloud"],
      "ACTOR_MAXTEXT_MODEL_NAME": (
          actor["kwargs"]["maxtext_model_name"]
          if actor["model_source"] == "maxtext"
          else ""
      ),
      "ACTOR_MODEL_ID": actor["model_id"],
      "ACTOR_MODEL_DIR": actor["model_download_path"],
      "ACTOR_TRAINER_BACKEND": trainer_backend,
      "ACTOR_MAXTEXT_CKPT_PATH": actor["model_path"],
      "ROLLOUT_MODEL_ID": rollout_model["model_id"],
      "ROLLOUT_MODEL_DIR": rollout_model["model_download_path"],
      "WEIGHT_SYNC_MODE": ws["mode"],
      # Empty when null.
      "WEIGHT_SYNC_RAIDEN_USE_FFI": (
          "" if raiden["use_ffi"] is None else raiden["use_ffi"]
      ),
      "WEIGHT_SYNC_RAIDEN_DATA_NICS": raiden["data_nics"],
      "WEIGHT_SYNC_RAIDEN_BROADCAST_K": raiden["broadcast_k"],
      "WEIGHT_SYNC_RAIDEN_MULTI_NUMA": raiden["multi_numa"],
      "LOCAL_AUTO_DETECT_DEVICES": local["auto_detect_devices"],
      "LOCAL_DOWNLOAD_MODEL_WEIGHTS": local["download_model_weights"],
      "LOCAL_TRAINER_DEVICES": local["trainer_devices"],
      "LOCAL_ROLLOUT_DEVICES": local["rollout_devices"],
      "LOCAL_INFERENCE_DEVICES": local["inference_devices"],
      "LOCAL_TPU_CHIPS_PER_HOST_BOUNDS": local["tpu_chips_per_host_bounds"],
      "LOCAL_TPU_HOST_BOUNDS": local["tpu_host_bounds"],
      "LOCAL_ALLOW_MULTIPLE_LIBTPU_LOAD": local["allow_multiple_libtpu_load"],
      "LOCAL_TRAINER_PATHWAYS": local["trainer_pathways"],
      "LOCAL_JAX_BACKEND_TARGET": local["jax_backend_target"],
      "LOCAL_WORKER_SHUTDOWN_GRACE_SECS": local["worker_shutdown_grace_secs"],
      "LOCAL_TRAINER_SHUTDOWN_GRACE_SECS": local["trainer_shutdown_grace_secs"],
      "LOCAL_FORCE_ARM_SECS": local["force_arm_secs"],
      "K8S_NAMESPACE": k8s["namespace"],
      "K8S_KUEUE_QUEUE_NAME": k8s["kueue_queue_name"],
      "K8S_PREEMPTIBLE": k8s["preemptible"],
      "K8S_PRIORITY_CLASS": k8s["priority_class"],
      "K8S_SERVICE_ACCOUNT": k8s["service_account"],
      "K8S_TERMINATION_GRACE_SECONDS": k8s["termination_grace_seconds"],
      "K8S_ORCHESTRATOR_ID": k8s["orchestrator_id"],
      "K8S_TRAINER_ID": k8s["trainer_id"],
      "K8S_ROLLOUT_ID": k8s["rollout_id"],
      "K8S_INFERENCE_ID": k8s["inference_id"],
      "K8S_OTEL_ID": k8s["otel_id"],
      "K8S_CPU_MACHINE": k8s["cpu_machine"],
      "K8S_CPU_NODEPOOL": k8s["cpu_nodepool"],
      "K8S_CPU_MEMORY": memory["orchestrator"]["request"],
      "K8S_CPU_MEMORY_LIMIT": memory["orchestrator"]["limit"],
      "K8S_GCS_SCRATCH_LOCATION": pathways["gcs_scratch_location"],
      "K8S_ENABLE_PATHWAYS_PERSISTENCE": pathways_trainer["persistence"],
      "K8S_TUNIX_IMAGE": k8s["tunix_image"],
      "K8S_TPU_RESERVATION": k8s["tpu_reservation"],
      "K8S_PATHWAYS_SERVER_IMAGE": pathways["server_image"],
      "K8S_PATHWAYS_PROXY_IMAGE": pathways["proxy_image"],
      "K8S_PATHWAYS_PROXY_MEMORY": proxy_memory["request"],
      "K8S_PATHWAYS_PROXY_MEMORY_LIMIT": proxy_memory["limit"],
      "K8S_PATHWAYS_RM_MEMORY": rm_memory["request"],
      "K8S_PATHWAYS_RM_MEMORY_LIMIT": rm_memory["limit"],
      "K8S_PATHWAYS_SERVER_MEMORY": server_memory["request"],
      "K8S_PATHWAYS_SERVER_MEMORY_LIMIT": server_memory["limit"],
      # Empty when null.
      "K8S_PATHWAYS_IFRT_THRESHOLD_BYTES": (
          "" if ifrt_threshold is None else ifrt_threshold
      ),
      "K8S_PATHWAYS_WORKER_ENV_JSON": pathways_worker_env_json(hp),
      "K8S_PATHWAYS_PROXY_EXTRA_ARGS": pathways["proxy_extra_args"],
      "K8S_PATHWAYS_SIDECAR_IMAGE": sidecar["image"],
      "K8S_PATHWAYS_SIDECAR_MEMORY": sidecar_memory["request"],
      "K8S_PATHWAYS_SIDECAR_MEMORY_LIMIT": sidecar_memory["limit"],
      "K8S_PATHWAYS_SIDECAR_SHM": sidecar["shm"],
      "K8S_PATHWAYS_CKPT_GCSFUSE_BUCKET": ckpt_gcsfuse["bucket"],
      "K8S_PATHWAYS_CKPT_GCSFUSE_MOUNT_PATH": ckpt_gcsfuse["mount_path"],
      "K8S_PATHWAYS_CKPT_GCSFUSE_MOUNT_OPTIONS": ckpt_gcsfuse["mount_options"],
      "K8S_TRAINER_MEMORY": memory["trainer"]["request"],
      "K8S_TRAINER_MEMORY_LIMIT": memory["trainer"]["limit"],
      "K8S_ROLLOUT_MEMORY": memory["rollout"]["request"],
      "K8S_ROLLOUT_MEMORY_LIMIT": memory["rollout"]["limit"],
      "K8S_FAIL_FAST": k8s["fail_fast"],
      "K8S_STARTUP_RETRIES": k8s["startup_retries"],
      "K8S_TEARDOWN_GANG_ON_EXIT": k8s["teardown_gang_on_exit"],
      "K8S_USE_DYNAMIC_SLICING": k8s["use_dynamic_slicing"],
      "K8S_HEAD_NODEPOOL": k8s["head_nodepool"],
      "K8S_HEAD_EXCLUDED_NODEPOOLS": _head_excluded_nodepools(k8s),
      "K8S_TRAINER_JOBSET_YAML": k8s_trainer["jobset_yaml"],
      "K8S_TRAINER_TPU_SLICE": k8s_trainer["tpu_slice"],
      "K8S_ROLLOUT_JOBSET_YAML": k8s_rollout["jobset_yaml"],
      "K8S_ROLLOUT_TPU_SLICE": k8s_rollout["tpu_slice"],
      "K8S_JAX_CACHE_DISABLE": jax_cache["disable"],
      "K8S_JAX_CACHE_LOCAL_DIR": jax_cache["local_dir"],
      # Empty without a GCS cache.
      "K8S_JAX_CACHE_ROLLOUT_GCS_DIR": rollout_cache_uri or "",
      "K8S_JAX_CACHE_SAVE": jax_cache["save"],
      **{
          f"ROLE_ENV_{role.upper()}": _shell_env_words(env)
          for role, env in role_env(hp).items()
      },
      "SANDBOX_NAMESPACE": (
          sandbox_k8s["namespace"] if "namespace" in sandbox_k8s else ""
      ),
      "SANDBOX_NODE_SELECTOR_KEY": (
          sandbox_k8s["node_selector_key"]
          if "node_selector_key" in sandbox_k8s
          else ""
      ),
      "SANDBOX_NODE_SELECTOR_VAL": (
          sandbox_k8s["node_selector_val"]
          if "node_selector_val" in sandbox_k8s
          else ""
      ),
  }

  lines = [f"export {k}={_bash_quote(v)}" for k, v in exports.items()]
  return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
  """Prints the bash exports for the config given on the command line.

  Args:
    argv: Full command line (`sys.argv` when None): the program name, then the
      overrides.

  Returns:
    The process exit status.

  Raises:
    ValueError: If `argv` is empty or starts with an override instead of the
      program name.
  """
  args = sys.argv if argv is None else argv
  if not args or "=" in args[0]:
    raise ValueError(
        f"argv must start with the program name, got {list(args)!r}."
    )
  hp = initialize(args[1:])
  sys.stdout.write(render_bash_exports(hp))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
