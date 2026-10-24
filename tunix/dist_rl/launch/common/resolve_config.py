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
1. `DistHyperParameters` / `initialize(argv)`: Validated configuration loaded
   from `base_dist_config.yaml`, task/recipe override files, and
   `section.key=value` overrides. Lightweight (no training imports); the node
   and orchestrator mains use `dist_pipeline.DistPipeline`, which adds the
   `tunix.cli` pipeline builders. Both load with `DIST_INIT_KWARGS` and then
   call `finalize_dist_config` (derived defaults + dist checks).
2. `render_bash_exports(hp)`: Emits shell environment variables for `local.sh`
   and `k8s.sh` (ports, TPU chip IDs, K8s JobSet parameters, and artifact/model
   paths).
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
import datetime
import getpass
import json
import math
import os
import pathlib
import shlex
import sys
from typing import Any, Final, TypedDict

from tunix.cli import config as cli_config

LAUNCH_DIR = pathlib.Path(__file__).parent
REPO_ROOT = LAUNCH_DIR.parents[3]
DEFAULT_BASE_CONFIG_PATH = LAUNCH_DIR / "base_dist_config.yaml"


class DistInitKwargs(TypedDict):
  """`HyperParameters.__init__` options shared by the dist config classes."""

  default_base_config: pathlib.Path
  merge_replace_keys: bool
  validate_nested: bool


DIST_INIT_KWARGS: Final[DistInitKwargs] = {
    "default_base_config": DEFAULT_BASE_CONFIG_PATH,
    "merge_replace_keys": True,
    "validate_nested": True,
}


def finalize_dist_config(hp: cli_config.HyperParameters) -> None:
  """Applies the distributed RL derived defaults, then validates invariants.

  Called at the end of `DistHyperParameters.__init__` and
  `dist_pipeline.DistPipeline.__init__`, after `HyperParameters.__init__` with
  `DIST_INIT_KWARGS`.

  Args:
    hp: Loaded configuration; updated in place.
  """
  _finalize_dist_defaults(hp)
  _validate_dist_invariants(hp)


class DistHyperParameters(cli_config.HyperParameters):
  """`tunix.cli.config.HyperParameters` extended for distributed RL.

  Lightweight (no training imports) for `render_bash_exports`; the node and
  orchestrator mains use `dist_pipeline.DistPipeline` instead.
  """

  def __init__(self, argv: Sequence[str]) -> None:
    super().__init__(argv, **DIST_INIT_KWARGS)
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


def _finalize_dist_defaults(hp: cli_config.HyperParameters) -> None:
  """Resolves derived cluster, model, tokenizer, and optimizer defaults."""
  cluster = _mutable_section(hp, "cluster_config")
  target = str(cluster["target"])
  if target not in ("local", "k8s"):
    raise ValueError(
        f"cluster_config.target must be 'local' or 'k8s', got {target!r}."
    )

  if not cluster["artifact_root"]:
    raise ValueError("cluster_config.artifact_root must not be empty.")
  artifact_root = pathlib.Path(str(cluster["artifact_root"]))
  if target == "local" and not artifact_root.is_absolute():
    artifact_root = REPO_ROOT / artifact_root
  cluster["artifact_root"] = str(artifact_root)

  if not cluster["run_id"]:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )
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
    rl_train["checkpoint_root_directory"] = (
        str(REPO_ROOT / "checkpoints") if target == "local" else "checkpoints"
    )

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

  grpo = hp.config_mapping("agentic_grpo_config")
  if cluster["run_inference_node"] is None:
    cluster["run_inference_node"] = (
        float(grpo["beta"]) > 0.0 and not cluster["inference_addr"]
    )

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


def _validate_dist_invariants(hp: cli_config.HyperParameters) -> None:
  """Validates cross-field invariants across trainer and rollout configs."""
  # TODO(noghabi): Check which of these checks can be consolidated with the
  # `tunix.cli` validations or left to the constructors that consume them.
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
            f" be divisible by actor_model_config.mesh fsdp"
            f" ({actor_axes['fsdp']}) when model_source='maxtext'."
        )

  vllm_tp = hp.config_mapping("vllm_config")["tensor_parallel_size"]
  if vllm_tp is not None and int(vllm_tp) != rollout_axes["tp"]:
    raise ValueError(
        f"vllm_config.tensor_parallel_size ({vllm_tp}) must match"
        f" rollout_model_config.mesh tp ({rollout_axes['tp']})."
    )

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
  for key, value in (
      ("agentic_grpo_config.num_generations", grpo["num_generations"]),
      (
          "rl_training_config.train_micro_batch_size",
          rl_train["train_micro_batch_size"],
      ),
  ):
    if int(value) <= 0:
      raise ValueError(f"{key} must be > 0, got {value}.")
  # `train_micro_batch_size` counts trajectories in the distributed trainer:
  # gradient accumulation = mini_batch_size * num_generations /
  # train_micro_batch_size (not applicable with sequence packing).
  if not rl_train["max_seq_token_per_tpu"]:
    trajectories = mini_batch_size * int(grpo["num_generations"])
    micro_batch = int(rl_train["train_micro_batch_size"])
    if trajectories % micro_batch != 0:
      raise ValueError(
          "rl_training_config.mini_batch_size *"
          " agentic_grpo_config.num_generations"
          f" ({trajectories}) must be divisible by"
          f" rl_training_config.train_micro_batch_size ({micro_batch})."
      )


_MODEL_DTYPES = ("bfloat16", "float32")
_REMAT_CONFIGS = ("none", "block", "decoder")
_ROLLOUT_ENGINES = ("inprocess_vllm", "vllm", "vanilla")
_CHAT_PARSER_TYPES = ("auto", "raw")


def _check_choice(key: str, value: object, choices: Sequence[str]) -> None:
  if value not in choices:
    raise ValueError(f"{key} must be one of {choices}, got {value!r}.")


def initialize(argv: Sequence[str]) -> DistHyperParameters:
  """Initializes `DistHyperParameters` from `argv` (mirrors `config.initialize`)."""
  return DistHyperParameters(argv)


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
  dist_training = hp.config_mapping("dist_training_config")
  sandbox_k8s = (
      hp.config["kubernetes_config"]
      if isinstance(hp.config["kubernetes_config"], Mapping)
      else {}
  )
  trainer_backend = (
      "maxtext" if actor["model_source"] == "maxtext" else "tunix"
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
      "INFERENCE_ADDR": cluster["inference_addr"],
      "ORCHESTRATOR_PORT": ports["orchestrator"],
      "TRAINER_PORT": ports["trainer"],
      "ROLLOUT_PORT": ports["rollout"],
      "INFERENCE_PORT": ports["inference"],
      "ACTOR_MODEL_NAME": actor["model_name"],
      "ACTOR_MODEL_ID": actor["model_id"],
      "ACTOR_MODEL_DIR": actor["model_download_path"],
      "ACTOR_TRAINER_BACKEND": trainer_backend,
      "ACTOR_MAXTEXT_CKPT_PATH": actor["model_path"],
      "ROLLOUT_MODEL_ID": rollout_model["model_id"],
      "ROLLOUT_MODEL_DIR": rollout_model["model_download_path"],
      "WEIGHT_SYNC_MODE": ws["mode"],
      "WEIGHT_SYNC_VERIFY_WEIGHTS": ws["verify_weights"],
      "WEIGHT_SYNC_RAIDEN_USE_FFI": (
          "" if ws["raiden_use_ffi"] is None else ws["raiden_use_ffi"]
      ),
      "CKPT_D2H_CONCURRENT_GB": dist_training["ckpt_d2h_concurrent_gb"],
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
      "K8S_ORCHESTRATOR_ID": k8s["orchestrator_id"],
      "K8S_TRAINER_ID": k8s["trainer_id"],
      "K8S_ROLLOUT_ID": k8s["rollout_id"],
      "K8S_CPU_MACHINE": k8s["cpu_machine"],
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
      "K8S_TRAINER_JOBSET_YAML": k8s_trainer["jobset_yaml"],
      "K8S_TRAINER_TPU_SLICE": k8s_trainer["tpu_slice"],
      "K8S_ROLLOUT_JOBSET_YAML": k8s_rollout["jobset_yaml"],
      "K8S_ROLLOUT_TPU_SLICE": k8s_rollout["tpu_slice"],
      "K8S_ROLLOUT_REPLICAS": k8s_rollout["replicas"],
      "K8S_JAX_CACHE_DISABLE": jax_cache["disable"],
      "K8S_JAX_CACHE_LOCAL_DIR": jax_cache["local_dir"],
      "K8S_JAX_CACHE_GCS_DIR": jax_cache["gcs_dir"],
      "K8S_JAX_CACHE_ROLLOUT_GCS_DIR": jax_cache["rollout_gcs_dir"],
      "K8S_JAX_CACHE_BUCKET": jax_cache["bucket"],
      "K8S_JAX_CACHE_SAVE": jax_cache["save"],
      "K8S_ROLLOUT_SKIP_JAX_PRECOMPILE": jax_cache[
          "rollout_skip_jax_precompile"
      ],
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
  hp = initialize(sys.argv if argv is None else argv)
  sys.stdout.write(render_bash_exports(hp))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
