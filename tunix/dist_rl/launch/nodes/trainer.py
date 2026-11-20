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

"""Trainer worker node for distributed RL.

Launched as
`python -m tunix.dist_rl.distributed.runtime.main --discovery_addrs=...
--process_main=tunix.dist_rl.launch.nodes.trainer.main <config overrides>`.
The overrides are resolved once by `resolve_config.initialize`, and every
runtime object (optimizer, checkpointing options, mesh, trainer) is built
directly from that validated config.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
import dataclasses
import functools
import logging
import os
import re
import sys
from typing import Final, TypeVar

from flax import nnx
import jax
from jax.sharding import Mesh
import optax
from orbax import checkpoint as ocp
from tunix.cli import config as cli_config
from tunix.cli.utils import model as model_utils
from tunix.common import datatypes as common_datatypes
from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.train import abstract_trainer
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.worker import trainer_worker
from tunix.models import automodel
from tunix.utils import maxtext_utils

LOCAL_WORKER_ID = "trainer-0"

_ACTOR_KEY = "actor_model_config"
_RL_TRAINING_KEY = "rl_training_config"
_ACTOR_OPTIMIZER_KEY = "actor_optimizer_config"

# Default LoRA target pattern across attention and MLP projections when
# `actor_model_config.lora_config.module_path` is not set.
_LORA_MODULE_PATH = (
    ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj"
)

TrainerFactory = Callable[[], abstract_trainer.AbstractTrainer]
# Value type of the open `*_model_config.kwargs` override dict.
KwargValue = int | float | str | bool
_T = TypeVar("_T", int, float, str, bool)


def _sequence_packing_enabled(
    hp: resolve_config.DistHyperParameters,
) -> bool:
  """Returns whether `rl_training_config.max_seq_tokens_per_device > 0`.

  `common_configs.is_sequence_packing_enabled` needs a built TrainingConfig:
  `RLTrainingConfig` cannot be built for dist configs (its micro-batch is
  prompt-level, the dist one is trajectory-level), and the trainer's
  `TrainingConfig` deliberately leaves `max_seq_tokens_per_device` unset.

  Args:
    hp: Validated distributed RL configuration.
  """
  budget = hp.config_mapping(_RL_TRAINING_KEY)["max_seq_tokens_per_device"]
  return budget is not None and int(budget) > 0


def _gradient_accumulation_steps(
    hp: resolve_config.DistHyperParameters,
) -> int:
  """Returns micro-steps per optimizer update.

  `mini_batch_size` counts prompt groups and `train_micro_batch_size` counts
  trajectories, so one update spans `mini_batch_size * num_generations`
  trajectories (divisibility is validated by `DistHyperParameters`). This
  differs from `RLTrainingConfig`'s prompt-level
  `mini_batch_size // train_micro_batch_size`.

  Args:
    hp: Validated distributed RL configuration.

  Returns:
    1 with sequence packing (the micro-batch count is then dynamic and losses
    are token-normalized), otherwise the static micro-step count.
  """
  if _sequence_packing_enabled(hp):
    return 1
  rl_train = hp.config_mapping(_RL_TRAINING_KEY)
  num_generations = int(
      hp.config_mapping("agentic_grpo_config")["num_generations"]
  )
  update_trajectories = int(rl_train["mini_batch_size"]) * num_generations
  return update_trajectories // int(rl_train["train_micro_batch_size"])


def _save_interval_steps(hp: resolve_config.DistHyperParameters) -> int:
  """Returns the checkpoint save interval; 0 means "never save".

  Args:
    hp: Validated distributed RL configuration.

  Raises:
    ValueError: If the interval is negative.
  """
  rl_train = hp.config_mapping(_RL_TRAINING_KEY)
  interval = int(rl_train["checkpointing_options"]["save_interval_steps"])
  if interval < 0:
    raise ValueError(
        "rl_training_config.checkpointing_options.save_interval_steps must be"
        f" non-negative, got {interval}."
    )
  return interval


def _checkpoint_root_directory(
    hp: resolve_config.DistHyperParameters,
) -> str | None:
  """Returns the checkpoint root, or None when saving is disabled.

  Tunix's `CheckpointManager` has no read-only mode, so keeping a root with
  `save_interval_steps=0` makes `save_checkpoint()` hit `step % 0` and
  `close()` force-save anyway. Without a root there is no checkpointer at all;
  the orchestrator-driven restore path is unaffected.

  Args:
    hp: Validated distributed RL configuration.
  """
  if _save_interval_steps(hp) > 0:
    return str(hp.config_mapping(_RL_TRAINING_KEY)["checkpoint_root_directory"])
  logging.info(
      "save_interval_steps=0; checkpoint saving is disabled and"
      " checkpoint_root_directory is withheld."
  )
  return None


def _ckpt_d2h_concurrent_gb(
    hp: resolve_config.DistHyperParameters,
) -> int | None:
  """Returns `dist_training_config.ckpt_d2h_concurrent_gb`, or None if null.

  The MaxText trainer caps the GB a checkpoint save copies from device to host
  at once by it; None keeps the backend's default. The resolver only allows a
  value for the MaxText trainer, so the Tunix trainer always gets None (no cap)
  for now.

  Args:
    hp: Validated distributed RL configuration.
  """
  gb = hp.config_mapping("dist_training_config")["ckpt_d2h_concurrent_gb"]
  return None if gb is None else int(gb)


def _tunix_training_config(
    hp: resolve_config.DistHyperParameters, *, gradient_accumulation_steps: int
) -> peft_trainer_v2.TrainingConfig:
  """Builds the PeftTrainer v2 config for the orchestrator-driven trainer.

  The checkpointing options come from the shared
  `obtain_training_config_dict` builder, with the device-to-host cap of
  `dist_training_config.ckpt_d2h_concurrent_gb`.
  `hp.create_rl_training_config()` is not used: `RLTrainingConfig` rejects
  trajectory-level micro-batches (see `_gradient_accumulation_steps`). Metrics
  logging and profiling stay with the orchestrator, so they are not set here.

  Args:
    hp: Validated distributed RL configuration.
    gradient_accumulation_steps: Micro-steps per optimizer update.

  Returns:
    The trainer config.
  """
  rl_train = hp.config_mapping(_RL_TRAINING_KEY)
  built = hp.obtain_training_config_dict(_RL_TRAINING_KEY)
  return peft_trainer_v2.TrainingConfig(
      eval_every_n_steps=int(rl_train["eval_every_n_steps"]),
      gradient_accumulation_steps=gradient_accumulation_steps,
      metrics_prefix="actor",
      pbar_description="Actor Training",
      # Matches the batch axis of `node_utils.MESH_AXIS_NAMES`.
      data_sharding_axis=("fsdp",),
      checkpointing_options=dataclasses.replace(
          built["checkpointing_options"],
          device_host_concurrent_gb=_ckpt_d2h_concurrent_gb(hp),
      ),
      checkpoint_root_directory=_checkpoint_root_directory(hp),
      # No max_seq_tokens_per_device here. Under the orchestrator the trainer
      # never packs: the SequencePackedBatchAssembler does, and the weighting
      # that packing needs rides in on the payload (`_fwd_bwd_step` accumulates
      # with `denom=aux.primary_loss.denominator`). The only thing the field
      # would change on this path is `_is_single_microstep()`, and the
      # orchestrator drives `fwd_bwd` + `update` rather than the fused step it
      # gates. The orchestrator owns resume: it calls restore_checkpoint()
      # explicitly and realigns its step/policy_version from the returned
      # metadata.
      resume_from_checkpoint_on_init=False,
  )


def _create_optimizer(
    hp: resolve_config.DistHyperParameters,
) -> optax.GradientTransformation:
  """Builds the actor optimizer from `rl_training_config.actor_optimizer_config`.

  `opt_type` picks the `optax` optimizer, `schedule_type` the
  `optax.schedules` function, and `opt_chain_type` + `chain_kwargs` (e.g.
  `clip_by_global_norm` + `{max_norm: 1.0}`) a transformation chained ahead of
  it, all via the shared `tunix.cli.config.create_optimizer`.

  Args:
    hp: Validated distributed RL configuration.
  """
  return hp.create_optimizer(_RL_TRAINING_KEY, _ACTOR_OPTIMIZER_KEY)


def _lora_config(actor: Mapping[str, object]) -> dict[str, object] | None:
  """Returns the qwix LoRA config, or None when `lora_config` is empty.

  Args:
    actor: The `actor_model_config` section.

  Raises:
    ValueError: If `lora_config` is set without `rank` and `alpha`.
  """
  lora_cfg = actor["lora_config"]
  if not isinstance(lora_cfg, Mapping):
    raise TypeError(
        f"actor_model_config.lora_config must be a mapping, got {lora_cfg!r}."
    )
  if not lora_cfg:
    return None
  missing = {"rank", "alpha"} - set(lora_cfg)
  if missing:
    raise ValueError(
        f"actor_model_config.lora_config is missing {sorted(missing)}."
    )
  module_path = (
      str(lora_cfg["module_path"])
      if "module_path" in lora_cfg and lora_cfg["module_path"]
      else _LORA_MODULE_PATH
  )
  config: dict[str, object] = {
      "module_path": module_path,
      "rank": int(lora_cfg["rank"]),
      "alpha": float(lora_cfg["alpha"]),
  }
  if "weight_qtype" in lora_cfg and lora_cfg["weight_qtype"] is not None:
    config["weight_qtype"] = str(lora_cfg["weight_qtype"])
  if "tile_size" in lora_cfg and lora_cfg["tile_size"] is not None:
    config["tile_size"] = int(lora_cfg["tile_size"])
  return config


# TODO(noghabi): Load through `tunix.cli.utils.model.create_model` (which also
# honors `model_source`) once `node_utils.ensure_local_safetensors` is retired;
# see the TODO there.
def _load_actor_model(
    hp: resolve_config.DistHyperParameters, model_dir: str, mesh: Mesh
) -> nnx.Module:
  """Loads the actor from local safetensors, applying LoRA when configured.

  Args:
    hp: Validated distributed RL configuration.
    model_dir: Local directory with direct safetensors shards.
    mesh: Trainer mesh.

  Returns:
    The actor `nnx.Module`.
  """
  actor = hp.config_mapping(_ACTOR_KEY)
  block_size = actor["flash_attention_block_size"]
  model, _ = automodel.AutoModel.from_pretrained(
      model_id=str(actor["model_name"]),
      mesh=mesh,
      model_path=model_dir,
      dtype=str(actor["dtype"]),
      load_dtype=str(actor["load_dtype"]),
      remat_config=str(actor["remat_config"]).lower(),
      use_flash_attention=bool(actor["use_flash_attention"]),
      flash_attention_block_size=(
          int(block_size) if block_size is not None else None
      ),
  )
  lora_config = _lora_config(actor)
  if lora_config is None:
    return model
  return model_utils.apply_lora_to_model(
      model, mesh=mesh, lora_config=lora_config
  )


def _create_tunix_trainer_factory(
    hp: resolve_config.DistHyperParameters,
    worker_id: str = LOCAL_WORKER_ID,
) -> tuple[TrainerFactory, Mesh]:
  """Creates the PeftTrainer v2 factory and the trainer mesh."""
  logging.info("Trainer backend: Tunix's PeftTrainer.")
  actor = hp.config_mapping(_ACTOR_KEY)
  rl_train = hp.config_mapping(_RL_TRAINING_KEY)
  grad_accumulation_steps = _gradient_accumulation_steps(hp)

  model_dir = node_utils.ensure_local_safetensors(
      str(actor["model_download_path"]),
      str(actor["model_id"]),
      model_key=_ACTOR_KEY,
  )
  logging.info("Prepared trainer safetensors directory: %s", model_dir)

  is_ici = (
      hp.cluster_targets.weight_sync_mode
      == common_datatypes.WeightSyncMode.ICI
  )
  logging.info("Creating trainer mesh...")
  trainer_devices = _ici_trainer_devices(hp) if is_ici else None
  if trainer_devices is not None and len(trainer_devices) > len(
      jax.local_devices()
  ):
    axes = hp.mesh_axes(_ACTOR_KEY)
    shape = (axes["fsdp"], axes["tp"])
    if shape[0] * shape[1] != len(trainer_devices):
      raise ValueError(
          "Trainer mesh dimensions must multiply to ICI trainer device count."
          f" Got {_ACTOR_KEY}.mesh shape={shape},"
          f" devices={len(trainer_devices)}."
      )
    from jax.experimental import mesh_utils  # pylint: disable=g-import-not-at-top

    mesh = Mesh(
        mesh_utils.create_device_mesh(
            shape, trainer_devices, allow_split_physical_axes=True
        ),
        axis_names=node_utils.MESH_AXIS_NAMES,
    )
  else:
    mesh = node_utils.create_mesh(hp, _ACTOR_KEY, "Trainer")
  logging.info("Trainer mesh: %s", mesh)

  training_config = _tunix_training_config(
      hp, gradient_accumulation_steps=grad_accumulation_steps
  )
  optimizer = _create_optimizer(hp)
  num_generations = int(
      hp.config_mapping("agentic_grpo_config")["num_generations"]
  )
  if _sequence_packing_enabled(hp):
    logging.info(
        "Trainer gradient accumulation is dynamic via sequence packing"
        " (max_seq_tokens_per_device=%s, mini_batch_size=%s prompt groups,"
        " num_generations=%d).",
        rl_train["max_seq_tokens_per_device"],
        rl_train["mini_batch_size"],
        num_generations,
    )
  else:
    logging.info(
        "Trainer gradient_accumulation_steps=%d (mini_batch_size=%s prompt"
        " groups, num_generations=%d, train_micro_batch_size=%s).",
        grad_accumulation_steps,
        rl_train["mini_batch_size"],
        num_generations,
        rl_train["train_micro_batch_size"],
    )
  # `sampler_type` and `rollout_tp_size` describe the weight-sync destination,
  # not the training recipe, so they ride on the trainer's weight-sync surface
  # rather than on TrainingConfig.
  sampler_type = str(hp.config["rollout_engine"])
  rollout_tp_size = hp.mesh_axes("rollout_model_config")["tp"]

  # Raiden is the trainer's default source transport; ICI swaps in its own.
  transport_kwargs = {}
  if is_ici:
    from tunix.dist_rl.weight_sync import ici_weight_sync_delegate  # pylint: disable=g-import-not-at-top

    transport_kwargs["weight_sync_worker_factory"] = functools.partial(
        ici_weight_sync_delegate.IciSourceSynchronizer, worker_id
    )

  def _factory() -> abstract_trainer.AbstractTrainer:
    logging.info("Loading actor model...")
    actor_model = _load_actor_model(hp, model_dir, mesh)
    return peft_trainer_v2.PeftTrainer(
        actor_model,
        optimizer,
        training_config,
        sampler_type=sampler_type,
        rollout_tp_size=rollout_tp_size,
        **transport_kwargs,
    )

  return _factory, mesh


def _maxtext_checkpointing_options(
    hp: resolve_config.DistHyperParameters,
) -> ocp.CheckpointManagerOptions:
  """Builds the Orbax v0 options `maxtext_utils.build_maxtext_config` reads.

  The shared `checkpointing_options_from_dict` builder returns
  `TunixCheckpointingOptions`, which has no `save_interval_steps` /
  `max_to_keep`; MaxText reads exactly those, so the v0 options are built here.

  Args:
    hp: Validated distributed RL configuration.

  Returns:
    Read-only options when `save_interval_steps=0` (MaxText then disables
    saving but still restores `load_parameters_path`), otherwise interval and
    retention options.
  """
  interval = _save_interval_steps(hp)
  if interval == 0:
    logging.info(
        "save_interval_steps=0; checkpoint saving is disabled (restore is"
        " unaffected)."
    )
    return ocp.CheckpointManagerOptions(read_only=True)
  ckpt = hp.config_mapping(_RL_TRAINING_KEY)["checkpointing_options"]
  return ocp.CheckpointManagerOptions(
      save_interval_steps=interval, max_to_keep=int(ckpt["max_to_keep"])
  )


def _maxtext_adamw_options(
    hp: resolve_config.DistHyperParameters,
) -> maxtext_utils.AdamWOptions:
  """Maps `actor_optimizer_config` onto MaxText's AdamW configuration.

  Only AdamW, optionally behind global-norm clipping, is mapped. Any other
  optimizer or chained transformation fails here instead of silently training
  with MaxText's defaults. The scalar hyperparameters are read directly: the
  shared `create_optimizer` builder returns an opaque `optax` transformation.

  Args:
    hp: Validated distributed RL configuration.

  Returns:
    The AdamW options for `maxtext_utils.build_maxtext_config`.

  Raises:
    ValueError: If the config asks for an unmapped optimizer or chained
      transformation, or clips without a positive `max_norm`.
  """
  opt = hp.config_mapping(_RL_TRAINING_KEY)[_ACTOR_OPTIMIZER_KEY]
  prefix = f"{_RL_TRAINING_KEY}.{_ACTOR_OPTIMIZER_KEY}"
  hint = (
      " For other MaxText optimizer settings, drop this setting and set them in"
      " dist_training_config.maxtext_extra_config instead (e.g."
      " dist_training_config.maxtext_extra_config.opt_type=sgd)."
  )
  opt_type = str(opt["opt_type"])
  if opt_type.lower() != "adamw":
    raise ValueError(
        f"{prefix}.opt_type={opt_type} is not supported by the maxtext"
        " backend, which maps only adamw."
        + hint
    )
  chain = opt["opt_chain_type"]
  if not chain:
    # Unchained, as in the tunix backend.
    clipping_threshold = 0.0
  elif str(chain).lower() == "clip_by_global_norm":
    chain_kwargs = opt["chain_kwargs"]
    if "max_norm" not in chain_kwargs or chain_kwargs["max_norm"] is None:
      raise ValueError(
          f"{prefix}.opt_chain_type={chain} needs a max_norm in"
          f" {prefix}.chain_kwargs."
      )
    clipping_threshold = float(chain_kwargs["max_norm"])
    # MaxText reads 0 as "do not clip", where optax would zero every update.
    if clipping_threshold <= 0:
      raise ValueError(f"max_norm must be positive, got {clipping_threshold}.")
  else:
    raise ValueError(
        f"{prefix}.opt_chain_type={chain} is not supported by the maxtext"
        " backend, which maps only clip_by_global_norm."
        + hint
    )
  return maxtext_utils.AdamWOptions(
      b1=float(opt["b1"]),
      b2=float(opt["b2"]),
      eps=float(opt["eps"]),
      weight_decay=float(opt["weight_decay"]),
      gradient_clipping_threshold=clipping_threshold,
  )


def _maxtext_profiling_options(
    hp: resolve_config.DistHyperParameters,
) -> maxtext_utils.ProfilerOptions | None:
  """Builds MaxText's profiler options, or None when profiling is off.

  The step window comes from the shared `profiler_options` builder; the
  period is `dist_training_config.profiler_period`, and the TPU tracing knobs
  are optional `actor_model_config.kwargs` overrides.

  Args:
    hp: Validated distributed RL configuration.
  """
  profiler_options = hp.obtain_training_config_dict(_RL_TRAINING_KEY)[
      "profiler_options"
  ]
  if profiler_options is None or profiler_options.profiler_steps <= 0:
    return None
  kwargs = hp.config_mapping(_ACTOR_KEY)["kwargs"]
  return maxtext_utils.ProfilerOptions(
      skip_first_n_steps=int(profiler_options.skip_first_n_steps),
      profiler_steps=int(profiler_options.profiler_steps),
      profiler_period=int(
          hp.config_mapping("dist_training_config")["profiler_period"]
      ),
      enable_tpu_profiling_options=_kwarg(
          kwargs, "enable_tpu_profiling_options", True
      ),
      tpu_num_chips_to_profile_per_task=_kwarg(
          kwargs, "tpu_num_chips_to_profile_per_task", 1
      ),
      tpu_num_sparse_cores_to_trace=_kwarg(
          kwargs, "tpu_num_sparse_cores_to_trace", 1
      ),
      tpu_num_sparse_core_tiles_to_trace=_kwarg(
          kwargs, "tpu_num_sparse_core_tiles_to_trace", 1
      ),
      upload_all_profiler_results=_kwarg(
          kwargs, "upload_all_profiler_results", False
      ),
  )


# Strings that YAML reads back unchanged without quotes: a word that is not a
# YAML boolean or null keyword (numbers and `${...}` start with another
# character). The checked keywords are lowercased, so some casings are quoted
# needlessly.
_PLAIN_YAML_STRING_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z_][A-Za-z0-9_./-]*"
)
_YAML_KEYWORDS: Final[frozenset[str]] = frozenset(
    {"true", "false", "yes", "no", "on", "off", "null"}
)


def _maxtext_extra_flag(key: str, value: str | int | float | bool) -> str:
  """Returns the MaxText `key=value` argument for one extra config entry.

  MaxText reads its arguments with `OmegaConf.from_cli`, which parses each value
  as YAML. A string YAML would read as something else (`""`, `"0"`, `"true"`,
  `"a: b"`, ...) is written double-quoted by `cli_config.format_cli_arg`. Plain
  words such as `custom` stay unquoted, so the logged flags read like the
  hand-written `MAXTEXT_EXTRA_FLAGS` of the old launchers.

  Args:
    key: MaxText config name.
    value: The value; `DistHyperParameters` checked its type.
  """
  if isinstance(value, str) and (
      not _PLAIN_YAML_STRING_RE.fullmatch(value)
      or value.lower() in _YAML_KEYWORDS
  ):
    return cli_config.format_cli_arg(key, value)
  return f"{key}={value}"


def _maxtext_extra_flags(hp: resolve_config.DistHyperParameters) -> list[str]:
  """Formats `dist_training_config.maxtext_extra_config` as `key=value` args.

  `DistHyperParameters` checked that each key is a MaxText config name the
  trainer does not build from the config, and that each value is a str, int,
  float or bool.

  Args:
    hp: Validated distributed RL configuration.

  Returns:
    One MaxText config argument per entry, in config order.
  """
  extra = hp.config_mapping("dist_training_config")["maxtext_extra_config"]
  return [_maxtext_extra_flag(key, value) for key, value in extra.items()]


# `actor_model_config.kwargs` is an open dict of optional MaxText overrides; an
# absent or `null` key keeps the `maxtext_utils.build_maxtext_config` default
# passed here as `unset`, whose type is also the type the value is converted to.
def _kwarg(kwargs: Mapping[str, KwargValue | None], key: str, unset: _T) -> _T:
  value = kwargs[key] if key in kwargs else None
  return unset if value is None else type(unset)(value)


def _discover_trainer_process_indices(
    hp: resolve_config.DistHyperParameters,
) -> list[int]:
  """Resolves the actual `jax.process_index()` set for all trainer hosts.

  On GKE TPU slices, `jax.distributed.initialize()` orders `jax.process_index()`
  by physical torus coordinate rather than Kubernetes `POD_INDEX`, so multi-host
  trainers must NOT assume their ranks equal `0..trainer_replicas-1`.

  Resolution order:
  1. Explicit `ICI_TRAINER_PROCESS_INDICES` env override (if set).
  2. When `cluster_config.trainer_replicas > 1`: dynamic rendezvous through the
     shared JAX distributed coordinator KV store so every trainer replica
     exchanges its actual `jax.process_index()`.
  3. Otherwise (single-host trainer): `[jax.process_index()]`.
  """
  raw = os.environ.get("ICI_TRAINER_PROCESS_INDICES", "").strip()
  if raw:
    try:
      process_indices = sorted(
          {int(p.strip()) for p in raw.split(",") if p.strip()}
      )
    except ValueError as exc:
      raise ValueError(
          "ICI_TRAINER_PROCESS_INDICES must be comma-separated ints, got"
          f" {raw!r}."
      ) from exc
    if not process_indices:
      raise ValueError("ICI_TRAINER_PROCESS_INDICES must not be empty.")
    return process_indices

  cluster = hp.config_mapping("cluster_config")
  trainer_replicas = int(cluster.get("trainer_replicas", 1))
  if trainer_replicas <= 1:
    return [int(jax.process_index())]

  raw_replica = os.environ.get(node_utils.TRAINER_REPLICA_ENV, "").strip()
  if not raw_replica.isdecimal():
    raise ValueError(
        f"{node_utils.TRAINER_REPLICA_ENV} must be a non-negative integer when"
        f" cluster_config.trainer_replicas={trainer_replicas}, got"
        f" {raw_replica!r}."
    )
  replica = int(raw_replica)
  if not 0 <= replica < trainer_replicas:
    raise ValueError(
        f"{node_utils.TRAINER_REPLICA_ENV}={replica} is out of range for"
        f" cluster_config.trainer_replicas={trainer_replicas}."
    )

  from orbax.checkpoint._src.multihost import multihost as ocp_multihost  # pylint: disable=g-import-not-at-top

  client = ocp_multihost.get_jax_distributed_client()
  run_id = str(cluster.get("run_id") or "default")
  prefix = f"tunix_ici_trainer_proc/{run_id}"
  timeout_ms = int(float(cluster.get("rpc_timeout_s") or 600.0) * 1000)
  my_proc = int(jax.process_index())
  client.key_value_set(f"{prefix}/{replica}", str(my_proc))
  discovered: list[int] = []
  for idx in range(trainer_replicas):
    val = client.blocking_key_value_get(f"{prefix}/{idx}", timeout_ms)
    discovered.append(int(val))
  process_indices = sorted(set(discovered))
  if len(process_indices) != trainer_replicas:
    raise RuntimeError(
        f"Expected {trainer_replicas} distinct trainer JAX processes, got"
        f" {discovered}."
    )
  logging.info(
      "Dynamically discovered trainer JAX process indices via coordinator KV"
      " store: %s (this replica=%d -> process_index=%d).",
      process_indices,
      replica,
      my_proc,
  )
  return process_indices


def _ici_trainer_devices(
    hp: resolve_config.DistHyperParameters,
) -> list[jax.Device] | None:
  """Returns the trainer's device subset inside a shared multi-process ICI world.

  Returns None when running single-process (every visible device belongs to the
  trainer). In a multi-process ICI world, resolves the actual `jax.process_index()`
  of every trainer process (dynamically via the JAX distributed coordinator KV
  store for multi-host trainers, or `ICI_TRAINER_PROCESS_INDICES` if overridden)
  and returns their sorted devices.
  """
  if jax.process_count() <= 1:
    return None
  from tunix.dist_rl.weight_sync import ici_transport  # pylint: disable=g-import-not-at-top

  process_indices = _discover_trainer_process_indices(hp)
  world = {d.process_index for d in jax.devices()}
  missing = [p for p in process_indices if p not in world]
  if missing:
    raise ValueError(
        f"Trainer process indices {missing} are outside this JAX world"
        f" {sorted(world)}."
    )
  if jax.process_index() not in process_indices:
    raise ValueError(
        f"This trainer process ({jax.process_index()}) is not in trainer"
        f" process indices {process_indices}."
    )
  return [d for p in process_indices for d in ici_transport.local_devices(p)]


def _create_maxtext_trainer_factory(
    hp: resolve_config.DistHyperParameters, worker_id: str
) -> tuple[TrainerFactory, Mesh]:
  """Creates the MaxTextTrainingEngine factory and its mesh.

  Args:
    hp: Validated distributed RL configuration.
    worker_id: Worker id, used as the MaxText run name.

  Returns:
    The trainer factory and the MaxText device mesh.
  """
  logging.info("Trainer backend: MaxText's MaxTextTrainingEngine.")
  actor = hp.config_mapping(_ACTOR_KEY)
  kwargs = actor["kwargs"]
  # `DistHyperParameters` requires it (non-empty) for a MaxText actor.
  maxtext_model_name = str(kwargs["maxtext_model_name"])
  rl_train = hp.config_mapping(_RL_TRAINING_KEY)
  rollout = hp.config_mapping("rollout_config")
  opt = rl_train[_ACTOR_OPTIMIZER_KEY]
  axes = hp.mesh_axes(_ACTOR_KEY)
  is_ici = (
      hp.cluster_targets.weight_sync_mode
      == common_datatypes.WeightSyncMode.ICI
  )
  trainer_devices = _ici_trainer_devices(hp) if is_ici else None
  num_devices = (
      len(trainer_devices) if trainer_devices is not None else jax.device_count()
  )

  adamw_options = _maxtext_adamw_options(hp)
  pad_id, _ = node_utils.pad_and_eos_ids(node_utils.load_hf_tokenizer(hp))
  if opt["schedule_type"]:
    logging.warning(
        "%s.%s.schedule_type=%s is ignored by the maxtext backend, which"
        " builds its own learning-rate schedule from learning_rate and"
        " actor_model_config.kwargs.warmup_steps_fraction.",
        _RL_TRAINING_KEY,
        _ACTOR_OPTIMIZER_KEY,
        opt["schedule_type"],
    )
  # All multi-host trainer replicas share one Orbax CheckpointManager directory
  # (`{base_output_directory}/{run_name}/checkpoints`). Using the per-replica
  # `worker_id` (`...-0`, `...-1`, ...) would split shards across separate
  # directories and fail multi-host checkpoint save/restore.
  cluster = hp.config_mapping("cluster_config")
  if int(cluster.get("trainer_replicas", 1)) > 1:
    maxtext_run_name = (
        str(cluster["k8s"]["trainer_id"])
        if cluster["target"] == "k8s"
        else LOCAL_WORKER_ID
    )
  else:
    maxtext_run_name = worker_id

  max_seq_tokens_per_device = rl_train["max_seq_tokens_per_device"]
  maxtext_config = maxtext_utils.build_maxtext_config(
      model_name=maxtext_model_name,
      worker_id=maxtext_run_name,
      train_micro_batch_size=int(rl_train["train_micro_batch_size"]),
      mesh_fsdp=axes["fsdp"],
      mesh_tp=axes["tp"],
      mesh_expert=axes["expert"],
      mesh_context=_kwarg(kwargs, "mesh_context", 1),
      num_devices=num_devices,
      max_prompt_length=int(rollout["max_prompt_length"]),
      max_response_length=int(rollout["total_generation_steps"]),
      learning_rate=float(opt["learning_rate"]),
      warmup_steps_fraction=_kwarg(kwargs, "warmup_steps_fraction", 0.0),
      adamw_options=adamw_options,
      load_parameters_path=str(actor["model_path"]),
      padded_moe_mlp_dim=_kwarg(kwargs, "padded_moe_mlp_dim", 0),
      base_output_directory=_kwarg(
          kwargs,
          "maxtext_output_dir",
          os.path.join(
              str(hp.config_mapping("cluster_config")["artifact_root"]),
              "maxtext",
          ),
      ),
      gradient_accumulation_steps=_gradient_accumulation_steps(hp),
      checkpointing_options=_maxtext_checkpointing_options(hp),
      profiling_options=_maxtext_profiling_options(hp),
      base_num_kv_heads=_kwarg(kwargs, "base_num_kv_heads", 0),
      rollout_mesh_tp=hp.mesh_axes("rollout_model_config")["tp"],
      # Off for the trainer by default; the rollout side reads
      # weight_sync_config.prefuse_moe_weights.
      prefuse_moe_weights=_kwarg(kwargs, "prefuse_moe_weights", False),
      use_weight_converter=bool(
          hp.config_mapping("weight_sync_config")["use_weight_converter"]
      ),
      max_seq_token_per_tpu=(
          int(max_seq_tokens_per_device)
          if max_seq_tokens_per_device is not None
          else 0
      ),
      weight_dtype=str(actor["load_dtype"]),
      attention=_kwarg(kwargs, "attention", ""),
      device_host_concurrent_gb=_ckpt_d2h_concurrent_gb(hp),
      extra_flags=_maxtext_extra_flags(hp),
      skip_step_on_spikes=bool(
          node_utils.optional_bool(kwargs, "skip_step_on_spikes")
      ),
      skip_step_on_nan=node_utils.optional_bool(kwargs, "skip_step_on_nan"),
      skip_step_interval=_kwarg(kwargs, "skip_step_interval", 128),
      skip_step_scaling_factor=_kwarg(kwargs, "skip_step_scaling_factor", 6.0),
  )
  logging.info("Creating MaxText device mesh...")
  if trainer_devices is None:
    mesh = maxtext_utils.create_maxtext_mesh(maxtext_config)
  else:
    mesh = maxtext_utils.create_maxtext_mesh(
        maxtext_config, devices=trainer_devices
    )
  logging.info("Trainer mesh: %s", mesh)

  def _factory() -> abstract_trainer.AbstractTrainer:
    engine = maxtext_utils.create_maxtext_engine(
        maxtext_config,
        mesh=mesh,
        tokenizer_pad_id=pad_id,
        wrap_with_tunix_adapter=True,
    )
    if is_ici:
      from tunix.dist_rl.weight_sync import ici_weight_sync_delegate  # pylint: disable=g-import-not-at-top

      return ici_weight_sync_delegate.IciMaxTextTrainerAdapter(
          engine, job_name=worker_id
      )
    return engine

  return _factory, mesh


def _create_trainer_factory(
    hp: resolve_config.DistHyperParameters, worker_id: str
) -> tuple[TrainerFactory, Mesh]:
  """Dispatches on `actor_model_config.model_source` (maxtext vs tunix)."""
  if hp.config_mapping(_ACTOR_KEY)["model_source"] == "maxtext":
    return _create_maxtext_trainer_factory(hp, worker_id)
  return _create_tunix_trainer_factory(hp, worker_id)


def main(
    argv: list[str], context: context_lib.ProcessContext | None = None
) -> None:
  """Serves a TrainerWorker over gRPC until SIGINT/SIGTERM.

  Args:
    argv: `override_config_file=...` and `section.key=value` overrides.
    context: Runtime process context providing JAX init and discovery.
  """
  process_context = node_utils.require_discovery_context(context)
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [TrainerNode] %(message)s",
      force=True,
  )
  hp = resolve_config.initialize(argv)
  node_utils.apply_library_env(hp, "trainer")
  config_sha256 = node_utils.log_resolved_config(hp, "trainer")
  worker_id = node_utils.worker_id(hp, "trainer", LOCAL_WORKER_ID)
  port = node_utils.service_port(hp, "trainer")
  # Before the JAX initialization, which can block on the other processes, so
  # a malformed port or OpenTelemetry setting fails fast.
  node_utils.setup_otel_metrics(hp, "trainer", worker_id)

  process_context.initialize_runtime(hp.cluster_targets.trainer)

  rl_train = hp.config_mapping(_RL_TRAINING_KEY)

  logging.info("Creating generic TrainerWorker and gRPC server...")
  trainer_factory, mesh = _create_trainer_factory(hp, worker_id)
  worker_service = trainer_worker.TrainerWorker(
      trainer_factory=trainer_factory,
      worker_id=worker_id,
      logps_chunk_size=int(rl_train["compute_logps_chunk_size"]),
      logps_micro_batch_size=int(rl_train["compute_logps_micro_batch_size"]),
      execution_context=mesh,
  )

  # On SIGTERM/SIGINT, stop the gRPC server first: `stop_serving` cancels the
  # queued RPCs and waits for the one running on the worker thread (e.g.
  # `fwd_bwd`, `update`, or `save_checkpoint`). Only then close the worker
  # (`TrainerWorker.stop()` -> `PeftTrainerV2.close()`), which blocks until the
  # async checkpoint saves are finished.
  def _close() -> None:
    worker_service.stop()

  asyncio.run(
      node_utils.serve_worker(
          worker_service,
          process_context=process_context,
          role="trainer",
          worker_id_str=worker_id,
          port=port,
          config_sha256=config_sha256,
          close_fn=_close,
      )
  )


if __name__ == "__main__":
  main(sys.argv[1:])
