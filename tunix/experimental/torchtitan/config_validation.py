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

"""TorchTitan configuration validation rules."""

from __future__ import annotations

from tunix.common import datatypes as common_datatypes
from tunix.dist_rl.common import execution_target
from tunix.dist_rl.launch.common import resolve_config


def validate_torchtitan_config(hp: resolve_config.DistHyperParameters) -> None:
  """Validates all TorchTitan invariants for distributed training.

  Args:
    hp: Validated DistHyperParameters tree.

  Raises:
    ValueError: If any invariant is violated.
  """
  if hp.cluster_targets.trainer != execution_target.TORCH_TPU:
    raise ValueError(
        "TorchTitan trainer requires execution_targets.trainer='torch:tpu',"
        f" got '{hp.cluster_targets.trainer}'."
    )
  actor_cfg = hp.config_mapping("actor_model_config")
  if actor_cfg["model_source"] == "maxtext":
    raise ValueError(
        "TorchTitan trainer (execution_targets.trainer='torch:tpu') requires"
        " Hugging Face safetensors weights"
        " (actor_model_config.model_source != 'maxtext'), got 'maxtext'."
    )

  actor_axes = hp.mesh_axes("actor_model_config")
  if (
      actor_axes["tp"] != 1
      or actor_axes["dp"] != 1
      or actor_axes["expert"] != 1
  ):
    raise ValueError(
        "TorchTitan trainer (execution_targets.trainer='torch:tpu') requires"
        f" 1D FSDP (tp=1, dp=1, expert=1), got mesh axes: {actor_axes}."
    )

  rl_train = hp.config_mapping("rl_training_config")
  if rl_train["max_seq_tokens_per_device"] is not None:
    raise ValueError(
        "TorchTitan trainer (execution_targets.trainer='torch:tpu') does not"
        " support sequence packing"
        " (rl_training_config.max_seq_tokens_per_device must be null)."
    )

  micro_batch = int(rl_train["train_micro_batch_size"])
  if micro_batch % actor_axes["fsdp"] != 0:
    raise ValueError(
        f"rl_training_config.train_micro_batch_size ({micro_batch}) must"
        " be divisible by actor_model_config.mesh fsdp"
        f" ({actor_axes['fsdp']}) for TorchTitan trainer"
        " (execution_targets.trainer='torch:tpu')."
    )

  batch_size = int(hp.config["batch_size"])
  mini_batch = int(rl_train["mini_batch_size"])
  grpo_cfg = hp.config_mapping("agentic_grpo_config")
  if int(grpo_cfg["num_iterations"]) == 1 and mini_batch >= batch_size:
    raise ValueError(
        "TorchTitan trainer does not yet support fused in-loss"
        " sampler-trainer agreement (triggered when"
        f" rl_training_config.mini_batch_size ({mini_batch}) >="
        f" batch_size ({batch_size}) with"
        " agentic_grpo_config.num_iterations=1); set"
        " rl_training_config.mini_batch_size < batch_size."
    )

  actor_master_dtype = str(actor_cfg["load_dtype"])
  if actor_master_dtype not in ("bfloat16", "float32"):
    raise ValueError(
        "TorchTitan trainer (execution_targets.trainer='torch:tpu') requires"
        " actor_model_config.load_dtype to be 'bfloat16' or 'float32', got"
        f" {actor_master_dtype!r}."
    )

  actor_export_dtype = str(actor_cfg["dtype"])
  if actor_export_dtype not in ("bfloat16", "float32"):
    raise ValueError(
        "TorchTitan trainer (execution_targets.trainer='torch:tpu') requires"
        " actor_model_config.dtype to be 'bfloat16' or 'float32', got"
        f" {actor_export_dtype!r}."
    )

  if "remat_config" in actor_cfg and actor_cfg["remat_config"] not in (
      None,
      "",
      "none",
  ):
    raise ValueError(
        "TorchTitan trainer (execution_targets.trainer='torch:tpu') does not"
        f" yet support remat_config={actor_cfg['remat_config']!r} (must be"
        " 'none', '', or null)."
    )

  ws_mode = hp.config_mapping("weight_sync_config")["mode"]
  allowed_ws_modes = (
      common_datatypes.WeightSyncMode.BASIC.value,
      common_datatypes.WeightSyncMode.NONE.value,
      common_datatypes.WeightSyncMode.NOOP.value,
  )
  if ws_mode not in allowed_ws_modes:
    raise ValueError(
        "TorchTitan trainer (execution_targets.trainer='torch:tpu') requires"
        f" weight_sync_config.mode to be one of {allowed_ws_modes},"
        f" got {ws_mode!r}."
    )

  if ws_mode == common_datatypes.WeightSyncMode.BASIC.value:
    vllm_dtype = str(hp.config_mapping("vllm_config")["dtype"])
    if actor_export_dtype != vllm_dtype:
      raise ValueError(
          "TorchTitan export dtype"
          f" (actor_model_config.dtype={actor_export_dtype!r}) must match"
          f" explicit vllm_config.dtype ({vllm_dtype!r}; 'auto' is not"
          " supported) when weight_sync_config.mode='basic'."
      )

  opt = rl_train["actor_optimizer_config"]
  if str(opt["opt_type"]).lower() != "adamw":
    raise ValueError(
        "TorchTitan trainer only supports AdamW optimizer, got:"
        f" {opt['opt_type']}"
    )

  schedule_type = str(opt["schedule_type"]) if opt["schedule_type"] else ""
  if schedule_type not in (
      "",
      "constant",
      "constant_schedule",
      "warmup_cosine_decay_schedule",
  ):
    raise ValueError(
        f"Unsupported schedule_type for TorchTitan: {schedule_type}"
    )

  if schedule_type == "warmup_cosine_decay_schedule":
    if opt["init_value"] is None or opt["end_value"] is None:
      raise ValueError(
          "warmup_cosine_decay_schedule requires non-null init_value and"
          " end_value"
      )
    if opt["decay_steps"] is None or int(opt["decay_steps"]) <= 0:
      raise ValueError(
          "warmup_cosine_decay_schedule requires decay_steps > 0, got:"
          f" {opt['decay_steps']}"
      )
    if opt["warmup_steps"] is None or int(opt["warmup_steps"]) < 0:
      raise ValueError(
          "warmup_cosine_decay_schedule requires warmup_steps >= 0, got:"
          f" {opt['warmup_steps']}"
      )

  opt_chain = opt["opt_chain_type"]
  if opt_chain not in (None, "", "clip_by_global_norm"):
    raise ValueError(f"Unsupported opt_chain_type for TorchTitan: {opt_chain}")

  if opt_chain == "clip_by_global_norm":
    chain_kwargs = opt["chain_kwargs"]
    if (
        not chain_kwargs
        or "max_norm" not in chain_kwargs
        or float(chain_kwargs["max_norm"]) <= 0
    ):
      raise ValueError("clip_by_global_norm requires chain_kwargs.max_norm > 0")
