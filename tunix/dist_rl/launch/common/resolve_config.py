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
   and `k8s.sh` (ports, TPU chip IDs, K8s JobSet parameters, and artifact/model
   paths).
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
import datetime
import getpass
import hashlib
import json
import math
import os
import pathlib
import re
import shlex
import sys
from typing import Any, Final

import omegaconf
from tunix.cli import config as cli_config
from tunix.common import datatypes as common_datatypes
from tunix.dist_rl.common import gcs_cache
from tunix.dist_rl.distributed.deployment import yaml_generator
from tunix.dist_rl.queue_manager import trajectory_queue_manager
from tunix.dist_rl.weight_sync import weight_sync

LAUNCH_DIR = pathlib.Path(__file__).parent
REPO_ROOT = LAUNCH_DIR.parents[3]
DEFAULT_BASE_CONFIG_PATH = LAUNCH_DIR / "base_dist_config.yaml"


# Program name (`argv[0]`) that `DistHyperParameters` passes to
# `HyperParameters`; dist callers pass overrides only.
_DIST_PROG: Final[str] = "dist_rl"


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


def finalize_dist_config(hp: cli_config.HyperParameters) -> None:
  """Applies the distributed RL derived defaults, then validates invariants.

  Called at the end of `DistHyperParameters.__init__`, after
  `HyperParameters.__init__`.

  Args:
    hp: Loaded configuration; updated in place.
  """
  _finalize_dist_defaults(hp)
  _validate_dist_invariants(hp)


class DistHyperParameters(cli_config.HyperParameters):
  """`tunix.cli.config.HyperParameters` extended for distributed RL.

  Takes overrides only (see `_dist_argv`). Override files and `key=value`
  arguments deep-merge (`merge_replace_keys`), and every override key must be
  in the base schema (`validate_nested`).
  """

  def __init__(self, overrides: Sequence[str]) -> None:
    super().__init__(
        _dist_argv(overrides), merge_replace_keys=True, validate_nested=True
    )
    finalize_dist_config(self)


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

  The output directories (checkpoints, metrics) default to subdirectories of
  `cluster_config.artifact_root`.

  Args:
    hp: Loaded configuration updated in place.
  """
  cluster = _mutable_section(hp, "cluster_config")
  target = str(cluster["target"])
  _check_choice("cluster_config.target", target, _CLUSTER_TARGETS)

  if not cluster["artifact_root"]:
    raise ValueError("cluster_config.artifact_root must not be empty.")
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
      # unless `dist_training_config.trajectory_log_dir` is set.
      metrics["log_dir"] = str(artifact_root / "metrics")

  traj_store = _mutable_section(hp, "trajectory_store_config")
  if traj_store["backend"] == "file" and not traj_store["dir"]:
    traj_store["dir"] = str(artifact_root / "trajectory_store")

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
    if not k8s["otel_id"]:
      k8s["otel_id"] = f"{user}-otel"


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
    "cpu_memory",
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


def _validate_k8s_worker_jobsets(hp: cli_config.HyperParameters) -> None:
  """Rejects `cluster_config.k8s` settings the worker templates cannot render.

  `k8s.sh` applies the orchestrator JobSet before it renders the trainer and
  rollout ones, so a setting the YAML generator rejects or drops for a worker
  template would leave a partial deploy.

  Args:
    hp: Loaded configuration.

  Raises:
    ValueError: If `target` is "k8s" and `run_id` is not a valid K8s label
      value when `teardown_gang_on_exit` is enabled, `head_nodepool` is set
      without `use_dynamic_slicing`, or a JobSet-only setting is enabled with a
      `leaderworkerset.*.yaml` worker template.
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
  """Checks `cluster_config.rollout_replicas`, the rollouts to wait for.

  Args:
    hp: Loaded configuration.

  Raises:
    ValueError: If the count is not an int >= 1, or is not 1 on
      `cluster_config.target: local`, where `local.sh` starts one rollout.
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


def _validate_dist_invariants(hp: cli_config.HyperParameters) -> None:
  """Validates cross-field invariants across trainer and rollout configs."""
  # TODO(noghabi): Check which of these checks can be consolidated with the
  # `tunix.cli` validations or left to the constructors that consume them.
  _validate_rollout_cache_settings(hp)
  _validate_k8s_jax_cache_gcs(hp)
  _validate_k8s_scalars(hp)
  _validate_k8s_worker_jobsets(hp)
  _validate_rollout_replicas(hp)
  rollout_ft = hp.config_mapping("cluster_config")["rollout_fault_tolerance"]
  if not isinstance(rollout_ft, bool):
    raise TypeError(
        "cluster_config.rollout_fault_tolerance must be a bool, got"
        f" {rollout_ft!r}."
    )
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
  rollout_axes = hp.mesh_axes("rollout_model_config")
  if actor_cfg["model_source"] == "maxtext":
    if not actor_cfg["model_path"]:
      raise ValueError(
          "actor_model_config.model_source='maxtext' requires"
          " actor_model_config.model_path (or model_config.model_path)."
      )
    if rl_train["max_seq_token_per_tpu"] is None:
      micro_batch = int(rl_train["train_micro_batch_size"])
      if micro_batch % actor_axes["fsdp"] != 0:
        raise ValueError(
            f"rl_training_config.train_micro_batch_size ({micro_batch}) must"
            " be divisible by actor_model_config.mesh fsdp"
            f" ({actor_axes['fsdp']}) when model_source='maxtext'."
        )

  vllm = hp.config_mapping("vllm_config")
  vllm_tp = vllm["tensor_parallel_size"]
  if vllm_tp is not None and int(vllm_tp) != rollout_axes["tp"]:
    raise ValueError(
        f"vllm_config.tensor_parallel_size ({vllm_tp}) must match"
        f" rollout_model_config.mesh tp ({rollout_axes['tp']})."
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
  # TODO(noghabi): Move `GroupOrder` into a dependency-free module (e.g. under
  # `dist_rl/common/`) so the resolver doesn't depend on `queue_manager`, and
  # through it on flax and the agentic types. (`weight_sync.py` has no deps.)
  _check_choice(
      "agentic_grpo_config.trajectory_group_order",
      grpo["trajectory_group_order"],
      tuple(order.value for order in trajectory_queue_manager.GroupOrder),
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
        == trajectory_queue_manager.GroupOrder.PROMPT_ARRIVAL.value
    ):
      raise ValueError(
          "agentic_grpo_config.max_version_span is only supported with"
          " trajectory_group_order='trajectory_completion'."
      )
  _check_choice(
      "weight_sync_config.mode",
      hp.config_mapping("weight_sync_config")["mode"],
      tuple(mode.value for mode in weight_sync.WeightSyncMode),
  )
  _validate_maxtext_pairing(hp)
  _validate_maxtext_model_names(hp)
  _validate_maxtext_extra_flags(hp)
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
  if grpo["num_iterations"] != 1:
    raise ValueError(
        "agentic_grpo_config.num_iterations must be 1, got"
        f" {grpo['num_iterations']!r}: StandardRLProgram runs one optimizer"
        " pass per rollout batch."
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
  if not rl_train["max_seq_token_per_tpu"]:
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
  if mode == weight_sync.WeightSyncMode.NONE.value:
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
  step 0. `fallback` moves no weights, and `none` skips the sync.

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
    return mode == weight_sync.WeightSyncMode.RAIDEN.value
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
  if mode == weight_sync.WeightSyncMode.RAIDEN.value:
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


# Rollout settings promoted from the free-form `vllm_config.kwargs` to typed
# `vllm_config` keys.
_TYPED_VLLM_KEYS = (
    "return_routed_experts",
    "routed_experts_encode_mode",
    "routed_experts_num_experts",
    "detokenize",
)


def _validate_typed_vllm_settings(hp: cli_config.HyperParameters) -> None:
  """Validates the typed `vllm_config` rollout settings."""
  vllm = hp.config_mapping("vllm_config")
  promoted = sorted(set(_TYPED_VLLM_KEYS) & set(vllm["kwargs"]))
  if promoted:
    raise ValueError(
        f"vllm_config.kwargs sets {promoted}, which are typed keys now; set"
        f" vllm_config.<key> instead (e.g. vllm_config.{promoted[0]})."
    )
  for key in ("return_routed_experts", "detokenize"):
    if not isinstance(vllm[key], bool):
      raise TypeError(f"vllm_config.{key} must be a bool, got {vllm[key]!r}.")
  if vllm["server_mode"] is not None and not isinstance(
      vllm["server_mode"], bool
  ):
    raise TypeError(
        "vllm_config.server_mode must be a bool or null, got"
        f" {vllm['server_mode']!r}."
    )
  common_datatypes.validate_routed_experts_encode_config(
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
      mode != weight_sync.WeightSyncMode.NONE.value
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


def _validate_maxtext_extra_flags(hp: cli_config.HyperParameters) -> None:
  """Checks `dist_training_config.maxtext_extra_flags` before any job starts.

  The flags are extra MaxText config arguments, which only a MaxText trainer
  reads (`maxtext_utils.build_maxtext_config`); a Tunix trainer would silently
  ignore them. A malformed entry would otherwise fail only inside the trainer,
  after the launcher has submitted the jobs.

  Args:
    hp: Loaded configuration.

  Raises:
    TypeError: If the flags are not a string.
    ValueError: If the flags are set and `actor_model_config.model_source` is
      not "maxtext", or an entry is not a `key=value` pair.
  """
  flags = hp.config_mapping("dist_training_config")["maxtext_extra_flags"]
  if not isinstance(flags, str):
    raise TypeError(
        "dist_training_config.maxtext_extra_flags must be a string of"
        f" space-separated key=value pairs, got {flags!r}."
    )
  entries = flags.split()
  if not entries:
    return
  actor_source = hp.config_mapping("actor_model_config")["model_source"]
  if actor_source != "maxtext":
    raise ValueError(
        "dist_training_config.maxtext_extra_flags configures a MaxText trainer,"
        f" but actor_model_config.model_source is {actor_source!r}. Set"
        " model_config.model_source=maxtext or drop the flags."
    )
  malformed = [e for e in entries if "=" not in e or e.startswith("=")]
  if malformed:
    raise ValueError(
        "dist_training_config.maxtext_extra_flags entries must be key=value"
        f" pairs, got {malformed!r}."
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
      `weight_sync_config.raiden_data_nics` is not a string, or
      `dist_training_config.ckpt_d2h_concurrent_gb` is neither an int nor
      null.
    ValueError: If `dist_training_config.ckpt_d2h_concurrent_gb` is not
      positive, or is set for the Tunix trainer.
  """
  for section, key in _LIBRARY_BOOL_SETTINGS:
    value = hp.config_mapping(section)[key]
    if not isinstance(value, bool):
      raise TypeError(f"{section}.{key} must be a bool, got {value!r}.")
  nics = hp.config_mapping("weight_sync_config")["raiden_data_nics"]
  if not isinstance(nics, str):
    raise TypeError(
        "weight_sync_config.raiden_data_nics must be a string of"
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


# TODO(noghabi): Consider auto-flattening `cluster_config.{local,k8s}` into
# `UPPER_SNAKE_CASE` exports (and mapping only the non-cluster keys by hand),
# which would shrink this list and the drift check in `launch_test.py`.
def render_bash_exports(hp: cli_config.HyperParameters) -> str:
  """Renders cluster/deployment shell exports for `local.sh` and `k8s.sh`."""
  cluster = hp.config_mapping("cluster_config")
  ports = cli_config.to_str_dict(cluster["ports"])
  local = cli_config.to_str_dict(cluster["local"])
  k8s = cli_config.to_str_dict(cluster["k8s"])
  jax_cache = cli_config.to_str_dict(k8s["jax_cache"])
  k8s_trainer = cli_config.to_str_dict(k8s["trainer"])
  k8s_rollout = cli_config.to_str_dict(k8s["rollout"])
  actor = hp.config_mapping("actor_model_config")
  rollout_model = hp.config_mapping("rollout_model_config")
  ws = hp.config_mapping("weight_sync_config")
  vllm = hp.config_mapping("vllm_config")
  sandbox_k8s = (
      hp.config["kubernetes_config"]
      if isinstance(hp.config["kubernetes_config"], Mapping)
      else {}
  )
  trainer_backend = "maxtext" if actor["model_source"] == "maxtext" else "tunix"
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
      "ORCHESTRATOR_MAIN": cluster["orchestrator_main"],
      "TRAINER_MAIN": cluster["trainer_main"],
      "ROLLOUT_MAIN": cluster["rollout_main"],
      "INFERENCE_MAIN": cluster["inference_main"],
      "ARTIFACT_ROOT": cluster["artifact_root"],
      "RUN_ID": cluster["run_id"],
      "DRY_RUN": cluster["dry_run"],
      "WAIT_TIMEOUT_SECS": cluster["wait_timeout_secs"],
      "WAIT_POLL_SECS": cluster["wait_poll_secs"],
      "WAIT_DEBUG_EVERY_POLLS": cluster["wait_debug_every_polls"],
      "WAIT_LOG_TAIL_LINES": cluster["wait_log_tail_lines"],
      "RUN_INFERENCE_NODE": cluster["run_inference_node"],
      "ROLLOUT_REPLICAS": cluster["rollout_replicas"],
      "ORCHESTRATOR_PORT": ports["orchestrator"],
      "TRAINER_PORT": ports["trainer"],
      "ROLLOUT_PORT": ports["rollout"],
      "INFERENCE_PORT": ports["inference"],
      "OTEL_PORT": ports["otel"],
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
      "WEIGHT_SYNC_RAIDEN_USE_FFI": (
          "" if ws["raiden_use_ffi"] is None else ws["raiden_use_ffi"]
      ),
      "VLLM_ADDITIONAL_CONFIG_JSON": (
          json.dumps(vllm["kwargs"], sort_keys=True) if vllm["kwargs"] else ""
      ),
      "LOCAL_AUTO_DETECT_PCI_TPUS": local["auto_detect_pci_tpus"],
      "LOCAL_TRAINER_TPU_CHIPS": local["trainer_tpu_chips"],
      "LOCAL_ROLLOUT_TPU_CHIPS": local["rollout_tpu_chips"],
      "LOCAL_INFERENCE_TPU_CHIPS": local["inference_tpu_chips"],
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
      "K8S_CPU_MEMORY": k8s["cpu_memory"],
      "K8S_GCS_SCRATCH_LOCATION": k8s["gcs_scratch_location"],
      "K8S_ENABLE_PATHWAYS_PERSISTENCE": k8s["enable_pathways_persistence"],
      "K8S_TRAINER_EXTRA_ENV": k8s["trainer_extra_env"],
      "K8S_TUNIX_IMAGE": k8s["tunix_image"],
      "K8S_PATHWAYS_SERVER_IMAGE": k8s["pathways_server_image"],
      "K8S_PATHWAYS_PROXY_IMAGE": k8s["pathways_proxy_image"],
      "K8S_PATHWAYS_PROXY_MEMORY_LIMIT": k8s["pathways_proxy_memory_limit"],
      "K8S_PATHWAYS_PROXY_MEMORY": k8s["pathways_proxy_memory"],
      "K8S_PATHWAYS_RM_MEMORY": k8s["pathways_rm_memory"],
      "K8S_USER_CONTAINER_MEMORY": k8s["user_container_memory"],
      "K8S_USER_CONTAINER_MEMORY_LIMIT": k8s["user_container_memory_limit"],
      "K8S_PATHWAYS_WORKER_MEMORY": k8s["pathways_worker_memory"],
      "K8S_FAIL_FAST": k8s["fail_fast"],
      "K8S_STARTUP_RETRIES": k8s["startup_retries"],
      "K8S_TEARDOWN_GANG_ON_EXIT": k8s["teardown_gang_on_exit"],
      "K8S_USE_DYNAMIC_SLICING": k8s["use_dynamic_slicing"],
      "K8S_HEAD_NODEPOOL": k8s["head_nodepool"],
      "K8S_TRAINER_JOBSET_YAML": k8s_trainer["jobset_yaml"],
      "K8S_TRAINER_TPU_SLICE": k8s_trainer["tpu_slice"],
      "K8S_ROLLOUT_JOBSET_YAML": k8s_rollout["jobset_yaml"],
      "K8S_ROLLOUT_TPU_SLICE": k8s_rollout["tpu_slice"],
      "K8S_JAX_CACHE_DISABLE": jax_cache["disable"],
      "K8S_JAX_CACHE_LOCAL_DIR": jax_cache["local_dir"],
      # Empty without a GCS cache.
      "K8S_JAX_CACHE_ROLLOUT_GCS_DIR": rollout_cache_uri or "",
      "K8S_JAX_CACHE_SAVE": jax_cache["save"],
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
