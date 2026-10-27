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
import logging
import os
import sys
from typing import TypeVar

from flax import nnx
import jax
from jax.sharding import Mesh
import optax
from orbax import checkpoint as ocp
from tunix.cli.utils import model as model_utils
from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.train import abstract_trainer
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.weight_sync import raiden_preload
from tunix.dist_rl.worker import trainer_worker
from tunix.models import automodel
from tunix.utils import maxtext_utils

# Import Raiden before any other libraries to ensure correct JAX compilation.
raiden_preload.import_raiden()

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


def _sequence_packing_enabled(hp: resolve_config.DistHyperParameters) -> bool:
  """Returns whether `rl_training_config.max_seq_token_per_tpu > 0`.

  `common_configs.is_sequence_packing_enabled` needs a built TrainingConfig:
  `RLTrainingConfig` cannot be built for dist configs (its micro-batch is
  prompt-level, the dist one is trajectory-level), and the trainer's
  `TrainingConfig` deliberately leaves `max_seq_token_per_tpu` unset.

  Args:
    hp: Validated distributed RL configuration.
  """
  budget = hp.config_mapping(_RL_TRAINING_KEY)["max_seq_token_per_tpu"]
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


def _tunix_training_config(
    hp: resolve_config.DistHyperParameters, *, gradient_accumulation_steps: int
) -> peft_trainer_v2.TrainingConfig:
  """Builds the PeftTrainer v2 config for the orchestrator-driven trainer.

  The checkpointing options come from the shared
  `obtain_training_config_dict` builder. `hp.create_rl_training_config()` is
  not used: `RLTrainingConfig` rejects trajectory-level micro-batches (see
  `_gradient_accumulation_steps`). Metrics logging and profiling stay with the
  orchestrator, so they are not set here.

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
      checkpointing_options=built["checkpointing_options"],
      checkpoint_root_directory=_checkpoint_root_directory(hp),
      # No max_seq_token_per_tpu here. Under the orchestrator the trainer never
      # packs: the SequencePackedBatchAssembler does, and the weighting that
      # packing needs rides in on the payload (`_fwd_bwd_step` accumulates with
      # `denom=aux.primary_loss.denominator`). The only thing the field would
      # change on this path is `_is_single_microstep()`, and the orchestrator
      # drives `fwd_bwd` + `update` rather than the fused step it gates.
      # The orchestrator owns resume: it calls restore_checkpoint() explicitly
      # and realigns its step/policy_version from the returned metadata.
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

  logging.info("Creating trainer mesh...")
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
        " (max_seq_token_per_tpu=%s, mini_batch_size=%s prompt groups,"
        " num_generations=%d).",
        rl_train["max_seq_token_per_tpu"],
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

  def _factory() -> abstract_trainer.AbstractTrainer:
    logging.info("Loading actor model...")
    actor_model = _load_actor_model(hp, model_dir, mesh)
    return peft_trainer_v2.PeftTrainer(
        actor_model,
        optimizer,
        training_config,
        sampler_type=sampler_type,
        rollout_tp_size=rollout_tp_size,
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
      ' MAXTEXT_EXTRA_FLAGS instead (e.g. MAXTEXT_EXTRA_FLAGS="opt_type=sgd").'
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


# `actor_model_config.kwargs` is an open dict of optional MaxText overrides; an
# absent or `null` key keeps the `maxtext_utils.build_maxtext_config` default
# passed here as `unset`, whose type is also the type the value is converted to.
def _kwarg(kwargs: Mapping[str, KwargValue | None], key: str, unset: _T) -> _T:
  value = kwargs[key] if key in kwargs else None
  return unset if value is None else type(unset)(value)


def _create_maxtext_trainer_factory(
    hp: resolve_config.DistHyperParameters, worker_id: str
) -> tuple[TrainerFactory, Mesh]:
  """Creates the MaxTextTrainingEngine factory and its mesh.

  Args:
    hp: Validated distributed RL configuration.
    worker_id: Worker id, used as the MaxText run name.

  Returns:
    The trainer factory and the MaxText device mesh.

  Raises:
    ValueError: If `actor_model_config.kwargs.maxtext_model_name` is unset,
      null, or empty.
  """
  logging.info("Trainer backend: MaxText's MaxTextTrainingEngine.")
  actor = hp.config_mapping(_ACTOR_KEY)
  kwargs = actor["kwargs"]
  maxtext_model_name = _kwarg(kwargs, "maxtext_model_name", "")
  if not maxtext_model_name:
    raise ValueError(
        "actor_model_config.model_source='maxtext' requires"
        " actor_model_config.kwargs.maxtext_model_name (or"
        " model_config.kwargs.maxtext_model_name)."
    )
  rl_train = hp.config_mapping(_RL_TRAINING_KEY)
  rollout = hp.config_mapping("rollout_config")
  opt = rl_train[_ACTOR_OPTIMIZER_KEY]
  axes = hp.mesh_axes(_ACTOR_KEY)

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
  max_seq_token_per_tpu = rl_train["max_seq_token_per_tpu"]
  maxtext_config = maxtext_utils.build_maxtext_config(
      model_name=maxtext_model_name,
      worker_id=worker_id,
      train_micro_batch_size=int(rl_train["train_micro_batch_size"]),
      mesh_fsdp=axes["fsdp"],
      mesh_tp=axes["tp"],
      mesh_expert=axes["expert"],
      mesh_context=_kwarg(kwargs, "mesh_context", 1),
      num_devices=jax.device_count(),
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
          int(max_seq_token_per_tpu) if max_seq_token_per_tpu is not None else 0
      ),
      weight_dtype=str(actor["load_dtype"]),
      attention=_kwarg(kwargs, "attention", ""),
  )
  logging.info("Creating MaxText device mesh...")
  mesh = maxtext_utils.create_maxtext_mesh(maxtext_config)
  logging.info("Trainer mesh: %s", mesh)

  def _factory() -> abstract_trainer.AbstractTrainer:
    return maxtext_utils.create_maxtext_engine(
        maxtext_config,
        mesh=mesh,
        tokenizer_pad_id=pad_id,
        wrap_with_tunix_adapter=True,
    )

  return _factory, mesh


def _create_trainer_factory(
    hp: resolve_config.DistHyperParameters, worker_id: str
) -> tuple[TrainerFactory, Mesh]:
  """Dispatches on `actor_model_config.model_source` (maxtext vs tunix)."""
  if hp.config_mapping(_ACTOR_KEY)["model_source"] == "maxtext":
    return _create_maxtext_trainer_factory(hp, worker_id)
  return _create_tunix_trainer_factory(hp)


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
  logging.info("Resolved config: %s", hp.config)

  process_context.jax.initialize()

  worker_id = node_utils.worker_id(hp, "trainer", LOCAL_WORKER_ID)
  port = node_utils.service_port(hp, "trainer")
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

  # Drain on SIGTERM/SIGINT so that TrainerWorker.stop() -> PeftTrainerV2.close()
  # runs and blocks until every in-flight async op is finished.
  def _drain() -> None:
    worker_service.stop()

  asyncio.run(
      node_utils.serve_worker(
          worker_service,
          process_context=process_context,
          role="trainer",
          worker_id_str=worker_id,
          port=port,
          drain_fn=_drain,
      )
  )


if __name__ == "__main__":
  main(sys.argv[1:])
