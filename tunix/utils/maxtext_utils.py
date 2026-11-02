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

"""MaxText model configuration and runtime utilities."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import logging
import os
import types
from typing import Any


@dataclasses.dataclass(frozen=True)
class ProfilerOptions:
  """Options for configuring the profiler."""
  # Number of steps to skip before profiling.
  skip_first_n_steps: int
  # Number of steps to profile.
  profiler_steps: int
  # If positive, profile every N steps.
  profiler_period: int = -1
  # TPU advanced profiling options, meant to bound XSpace proto size and
  # extend the SparseCore trace window across multi-second training micro-steps.
  # MaxText's training engine profiler (MicroStepProfiler) does not read these
  # or upload_all_profiler_results yet: it honors only the step settings above
  # and traces only host 0.
  enable_tpu_profiling_options: bool = True
  tpu_num_chips_to_profile_per_task: int = 1
  tpu_num_sparse_cores_to_trace: int = 1
  tpu_num_sparse_core_tiles_to_trace: int = 1
  upload_all_profiler_results: bool = False


@dataclasses.dataclass(frozen=True)
class AdamWOptions:
  """AdamW hyperparameters that replace MaxText's base.yml defaults."""
  b1: float
  b2: float
  eps: float
  weight_decay: float
  # Global-norm clipping threshold; 0 disables clipping.
  gradient_clipping_threshold: float


def maxtext_modules():
  """Imports MaxText lazily; some installs nest it under maxtext.src.maxtext."""
  from maxtext.configs import pyconfig  # pylint: disable=g-import-not-at-top
  from maxtext.training_engine import maxtext_engine  # pylint: disable=g-import-not-at-top
  from maxtext.utils import maxtext_utils  # pylint: disable=g-import-not-at-top
  return pyconfig, maxtext_engine, maxtext_utils


# TODO(tunix-dev): [legacy-maxtext-a2cf3bc8b] Once pinned MaxText container
# images predating `rollout_weight_dtype` (e.g. `--maxtext-commit=a2cf3bc8b` /
# `68792f55`) are retired, remove all legacy-MaxText compatibility shims:
#   1. Delete `_maxtext_get_weight_dtype()`, `_pyconfig_supports_field()`,
#      `_ROLLOUT_DTYPE_MODULE_ALIASES`, and the top-level `Mapping` and `types`
#      imports.
#   2. In `build_maxtext_config()`, remove `supports_rollout_weight_dtype`
#      gating (and its fallback `logging.info`) and emit
#      `f"rollout_weight_dtype={DEFAULT_ROLLOUT_WEIGHT_DTYPE}"` unconditionally.
#   3. Delete `_configure_target_free_rollout_dtypes()` and remove its call from
#      `create_maxtext_engine()`.
#   4. In `maxtext_utils_test.py`, replace the `[legacy-maxtext-a2cf3bc8b]`
#      compatibility tests (and remove `jax`/`jnp` test deps in `utils/BUILD` if
#      unused) with a direct assertion that `build_maxtext_config()` emits
#      `rollout_weight_dtype=bfloat16`.
def _maxtext_get_weight_dtype():
  """Returns MaxText's `common_types.get_weight_dtype` helper when available."""
  try:
    from maxtext.common import common_types  # pylint: disable=g-import-not-at-top

    return getattr(common_types, "get_weight_dtype", None)
  except (ImportError, ModuleNotFoundError):
    return None


def _pyconfig_supports_field(pyconfig: Any, field_name: str) -> bool:
  """Returns whether `pyconfig`'s `MaxTextConfig` schema declares `field_name`.

  Inspects `pyconfig.types.MaxTextConfig.model_fields` (the schema dictionary
  `pyconfig` validates CLI keys against) so older pinned MaxText installs that
  predate `field_name` skip it up front without triggering
  `pyconfig._handle_config_exception` (which writes `/dev/termination-log` and
  calls `sys.exit(2)` outside test runners). Returns `True` when `pyconfig` is
  a mock or pre-pydantic install without a `Mapping` `model_fields`.

  Args:
    pyconfig: Imported MaxText `pyconfig` module (or test double).
    field_name: Name of the `MaxTextConfig` field to check.

  Returns:
    `True` if `field_name` is declared on `MaxTextConfig` (or if `model_fields`
    is unavailable), `False` otherwise.
  """
  cfg_types = getattr(pyconfig, "types", None)
  cfg_cls = getattr(cfg_types, "MaxTextConfig", None)
  model_fields = getattr(cfg_cls, "model_fields", None)
  if isinstance(model_fields, Mapping):
    return field_name in model_fields
  return True


def get_tokenizer_pad_id(
    model_id: str,
    tokenizer_path: str = "",
    model_dir: str = "",
) -> int:
  """Resolves the pad token id the MaxText adapter masks with."""
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

  path = tokenizer_path or model_dir or model_id
  tokenizer: Any = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
  if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
    tokenizer.pad_token = tokenizer.eos_token
  pad_id = getattr(tokenizer, "pad_token_id", None)
  return int(pad_id) if pad_id is not None else 0


# vLLM instantiates a model class by the HF `architectures` entry, so this is
# what selects maxtext_vllm_adapter's `MaxTextForCausalLM` over the stock
# vLLM implementation. Pass it as the engine's `hf_overrides`.
VLLM_MAXTEXT_HF_OVERRIDES = {"architectures": ["MaxTextForCausalLM"]}

DEFAULT_TRAINER_WEIGHT_DTYPE = "float32"
DEFAULT_ROLLOUT_WEIGHT_DTYPE = "bfloat16"

# TODO(tunix-dev): [legacy-maxtext-a2cf3bc8b] Delete with
# `_configure_target_free_rollout_dtypes` once pinned MaxText images predating
# `rollout_weight_dtype` are retired.
# Maps source tree path segments onto the canonical module names that MaxText
# passes to `common_types.get_weight_dtype(config, module_name)` when building
# the rollout model (`decoder_norm`, `query_norm`, and `key_norm` read
# `config.weight_dtype` directly and are intentionally omitted).
_ROLLOUT_DTYPE_MODULE_ALIASES = {
    "norm": "norm",
    "input_layernorm": "norm",
    "post_attention_layernorm": "norm",
    "gate": "gate",
    "shared_expert_gate": "shared_expert_gate",
    "A_log": "A_log",
    "dt_bias": "dt_bias",
    "conv1d": "conv1d",
    "logits_dense": "logits_dense",
}


def build_vllm_maxtext_additional_config(
    model_name: str,
    *,
    attention: str = "",
    prefuse_moe_weights: bool | None = None,
    return_routed_experts: bool = False,
    float32_gate_logits: bool | None = None,
    float32_logits: bool | None = None,
) -> dict[str, Any]:
  """Builds the vLLM `additional_config` a MaxText rollout model reads.

  `MaxTextForCausalLM` builds its own MaxText config from
  `additional_config["maxtext_config"]`; these are the inference-side overrides
  that make it match the trainer's model. Kept here so the in-process and
  server-mode samplers cannot drift apart.

  Args:
    model_name: MaxText model name, e.g. `qwen3-1.7b`.
    attention: MaxText attention kernel override; MaxText's default is used when
      empty.
    prefuse_moe_weights: Whether the rollout expects w0/w1 pre-fused into the
      TPU GMM layout. None leaves MaxText's default in place.
    return_routed_experts: Whether the MaxText MoE layers return the experts
      each token was routed to, for router replay. Only written when True, so
      MaxText versions that predate this key keep working by default.
    float32_gate_logits: Whether MaxText runs the MoE gate (router) in float32.
      None leaves the model YAML default in place.
    float32_logits: Whether dot-product attention casts its softmax inputs to
      float32. None leaves the model YAML default in place.

  Returns:
    The `additional_config` mapping to hand to the vLLM engine.
  """
  overrides: dict[str, Any] = {
      "model_name": model_name,
      "model_call_mode": "inference",
      "scan_layers": False,
      "enable_dp_attention": False,
      "allow_split_physical_axes": True,
      "log_config": False,
      "weight_dtype": DEFAULT_ROLLOUT_WEIGHT_DTYPE,
  }
  if float32_gate_logits is not None:
    overrides["float32_gate_logits"] = float32_gate_logits
  if float32_logits is not None:
    overrides["float32_logits"] = float32_logits
  if prefuse_moe_weights is not None:
    overrides["prefuse_moe_weights"] = prefuse_moe_weights
  if attention:
    overrides["attention"] = attention
  if return_routed_experts:
    overrides["return_routed_experts"] = True
  return {"maxtext_config": overrides}


def build_maxtext_config(
    model_name: str,
    worker_id: str = "",
    train_micro_batch_size: int = 1,
    mesh_fsdp: int = 1,
    mesh_tp: int = 1,
    mesh_expert: int = 1,
    num_devices: int = 1,
    max_prompt_length: int = 512,
    max_response_length: int = 128,
    learning_rate: float = 1e-5,
    warmup_steps_fraction: float = 0.0,
    load_parameters_path: str = "",
    padded_moe_mlp_dim: int = 0,
    base_output_directory: str = "",
    gradient_accumulation_steps: int = 1,
    checkpointing_options: Any = None,
    profiling_options: ProfilerOptions | None = None,
    *,
    mesh_context: int = 1,
    base_num_kv_heads: int = 0,
    kv_tp_size: int = 0,
    moe_mlp_tp_size: int = 0,
    rollout_mesh_tp: int = 0,
    prefuse_moe_weights: bool = False,
    use_weight_converter: bool = True,
    max_seq_token_per_tpu: int | None = 0,
    adamw_options: AdamWOptions | None = None,
    weight_dtype: str = DEFAULT_TRAINER_WEIGHT_DTYPE,
    attention: str = "",
) -> Any:
  """Builds the MaxText HyperParameters the training engine runs on.

  Note on keyword-only options:
    mesh_context: ICI context-parallelism degree; when > 1, disables
      `context_parallel_load_balance` because Tunix RL feeds batches in natural
      token order rather than MaxText's grain `DUAL_CHUNK_SWAP` order.
  """
  pyconfig, _, _ = maxtext_modules()

  # Backward compatibility: if rollout_mesh_tp was provided, default kv_tp_size and moe_mlp_tp_size
  if rollout_mesh_tp > 0:
    if kv_tp_size == 0:
      logging.info(
          "Overriding kv_tp_size from 0 to rollout_mesh_tp=%d", rollout_mesh_tp
      )
      kv_tp_size = rollout_mesh_tp
    if moe_mlp_tp_size == 0:
      logging.info(
          "Overriding moe_mlp_tp_size from 0 to rollout_mesh_tp=%d",
          rollout_mesh_tp,
      )
      moe_mlp_tp_size = rollout_mesh_tp

  if mesh_context <= 0:
    raise ValueError(f"mesh_context must be positive, got {mesh_context}")
  if padded_moe_mlp_dim < 0:
    raise ValueError(
        f"padded_moe_mlp_dim must be non-negative, got {padded_moe_mlp_dim}"
    )
  if base_num_kv_heads < 0:
    raise ValueError(
        f"base_num_kv_heads must be non-negative, got {base_num_kv_heads}"
    )
  if kv_tp_size < 0:
    raise ValueError(f"kv_tp_size must be non-negative, got {kv_tp_size}")
  if moe_mlp_tp_size < 0:
    raise ValueError(
        f"moe_mlp_tp_size must be non-negative, got {moe_mlp_tp_size}"
    )
  if rollout_mesh_tp < 0:
    raise ValueError(
        f"rollout_mesh_tp must be non-negative, got {rollout_mesh_tp}"
    )
  if max_seq_token_per_tpu is not None and max_seq_token_per_tpu < 0:
    raise ValueError(
        "max_seq_token_per_tpu must be non-negative, got"
        f" {max_seq_token_per_tpu}"
    )

  num_data_shards = mesh_fsdp * mesh_expert
  if max_seq_token_per_tpu is not None and max_seq_token_per_tpu > 0:
    # When sequence packing is enabled, the batch assembler feeds 1 packed row
    # per data shard (global batch size = mesh_fsdp * mesh_expert), independent
    # of train_micro_batch_size.
    # TODO(maxtext-team): verify this is the desired behavior.
    per_device_batch_size = num_data_shards / num_devices
  else:
    # MaxText shards the batch dimension over the mesh axes behind
    # `activation_embed_and_logits_batch`; of those, this config sets fsdp and
    # expert. A micro-batch that does not divide their product is replicated
    # across them instead, and every device computes all of it.
    if train_micro_batch_size % num_data_shards:
      raise ValueError(
          f"train_micro_batch_size={train_micro_batch_size} must be a multiple"
          f" of mesh_fsdp * mesh_expert = {mesh_fsdp} * {mesh_expert} ="
          f" {num_data_shards}; MaxText shards the batch dimension across both"
          " axes."
      )
    per_device_batch_size = train_micro_batch_size / num_devices

  base_yml = os.path.join(
      os.path.dirname(os.path.abspath(pyconfig.__file__)), "base.yml"
  )
  if not os.path.exists(base_yml):
    raise FileNotFoundError(f"MaxText base.yml not found at {base_yml}")

  effective_kv_heads = base_num_kv_heads
  effective_padded_moe_mlp_dim = padded_moe_mlp_dim

  # Determine if we need to inspect the model's YAML configuration
  needs_model_yml = (effective_kv_heads <= 0 and kv_tp_size > 0) or (
      not effective_padded_moe_mlp_dim and moe_mlp_tp_size > 0
  )
  model_data = None
  if needs_model_yml:
    models_dir = os.path.join(os.path.dirname(base_yml), "models")
    model_yml = os.path.join(models_dir, f"{model_name}.yml")
    if os.path.exists(model_yml):
      try:
        import yaml

        with open(model_yml, "r") as f:
          data = yaml.safe_load(f)
        if isinstance(data, dict):
          model_data = data
        else:
          logging.warning(
              "Expected dict in model config %s, got %s",
              model_yml,
              type(data).__name__,
          )
      except Exception as e:
        logging.warning("Failed to load model config from %s: %s", model_yml, e)
    else:
      logging.warning("Model config file not found at %s", model_yml)

  # 1. Resolve KV head replication:
  yaml_kv_heads = 0
  if effective_kv_heads <= 0 and kv_tp_size > 0:
    if model_data:
      yaml_kv_heads = int(model_data.get("base_num_kv_heads") or 0)
      effective_kv_heads = yaml_kv_heads
    if effective_kv_heads <= 0:
      raise ValueError(
          f"kv_tp_size ({kv_tp_size}) requires base_num_kv_heads > 0, but could"
          f" not determine base_num_kv_heads from config or {model_name}.yml."
          " Please specify --maxtext_base_num_kv_heads."
      )

  kv_heads_replicated = False
  if effective_kv_heads > 0 and kv_tp_size > effective_kv_heads:
    if kv_tp_size % effective_kv_heads != 0:
      raise ValueError(
          f"kv_tp_size ({kv_tp_size}) must be cleanly divisible by "
          f"base_num_kv_heads ({effective_kv_heads})."
      )
    effective_kv_heads = kv_tp_size
    kv_heads_replicated = True

  # Emit `base_num_kv_heads` on the CLI only when the caller explicitly passed
  # `base_num_kv_heads > 0` or when KV replication increased the head count
  # above the model YAML's `base_num_kv_heads`. When `base_num_kv_heads == 0`
  # and `kv_tp_size <= yaml_kv_heads`, the model YAML value is already
  # sufficient and omitting the CLI override keeps MaxText's
  # `validate_no_keys_overridden_twice` guard intact for all other keys.
  emit_kv_heads_override = base_num_kv_heads > 0 or kv_heads_replicated
  if emit_kv_heads_override:
    logging.info(
        "Enabling override_model_config=True for base_num_kv_heads=%d"
        " (explicit=%d, replicated=%s).",
        effective_kv_heads,
        base_num_kv_heads,
        kv_heads_replicated,
    )

  # 2. Resolve padded MoE MLP dimension before pyconfig initialization:
  if not effective_padded_moe_mlp_dim and moe_mlp_tp_size > 0:
    compute_padded_moe_mlp_dim = None
    try:
      from maxtext.integration.vllm.convert_utils import compute_padded_moe_mlp_dim
    except (ImportError, ModuleNotFoundError) as e:
      logging.warning(
          "Could not import compute_padded_moe_mlp_dim: %s. Skipping automatic"
          " MoE dimension padding.",
          e,
      )

    if compute_padded_moe_mlp_dim is not None and model_data:
      base_dim = model_data.get("base_moe_mlp_dim") or model_data.get(
          "moe_intermediate_size"
      )
      if base_dim:
        try:
          effective_padded_moe_mlp_dim = compute_padded_moe_mlp_dim(
              base_dim, moe_mlp_tp_size
          )
          logging.info(
              "Auto-computed padded_base_moe_mlp_dim=%d for moe_mlp_tp_size=%d",
              effective_padded_moe_mlp_dim,
              moe_mlp_tp_size,
          )
        except Exception as e:
          raise RuntimeError(
              "Failed to auto-compute padded_base_moe_mlp_dim for"
              f" moe_mlp_tp_size={moe_mlp_tp_size}: {e}"
          ) from e

  output_dir = base_output_directory or "/tmp/maxtext"
  argv = [
      "maxtext_trainer",
      base_yml,
      f"model_name={model_name}",
      f"run_name={worker_id or 'tunix_maxtext'}",
      f"base_output_directory={output_dir}",
  ]
  if load_parameters_path:
    argv.append(f"load_parameters_path={load_parameters_path}")
  # Checkpointing configs. `save_interval_steps=0` means "never save"
  save_interval_steps = int(
      getattr(checkpointing_options, "save_interval_steps", 0) or 0
  )
  if checkpointing_options is not None and save_interval_steps < 0:
    raise ValueError(
        "checkpoint save_interval_steps must be non-negative, got"
        f" {save_interval_steps}."
    )
  if checkpointing_options is not None and save_interval_steps > 0:
    argv.extend([
        "enable_checkpointing=True",
        f"checkpoint_period={save_interval_steps}",
        f"max_num_checkpoints_to_keep={checkpointing_options.max_to_keep}",
    ])
  elif checkpointing_options is not None:
    # `enable_checkpointing=False` still warm starts from `load_parameters_path`:
    # MaxText restores it through its own `ocp.Checkpointer`, not the
    # CheckpointManager that this flag gates.
    logging.info(
        "checkpoint save_interval_steps=0; disabling checkpoint saving "
        "(load_parameters_path still restores)."
    )
    argv.append("enable_checkpointing=False")
  elif load_parameters_path:
    argv.append("enable_checkpointing=True")
  else:
    argv.append("enable_checkpointing=False")
  # max_target_length is the row width this config declares. A packed row holds
  # several trajectories end to end, so it is wider than any single one, and
  # max_prompt+max_response describes one trajectory. MaxText takes its actual
  # shapes from the batch it is handed, so a wider row still runs -- but
  # everything MaxText derives from max_target_length is then computed for a row
  # narrower than the ones the trainer is fed: per-device TFLOPs, and the
  # divisibility checks MaxTextConfig runs against it (num_vocab_tiling,
  # context parallelism, num_moe_token_chunks). Declare the real width instead.
  max_target_length = max_prompt_length + max_response_length
  if (
      max_seq_token_per_tpu is not None
      and max_seq_token_per_tpu > max_target_length
  ):
    logging.info(
        "Raising max_target_length %d -> %d: with sequence packing the rows"
        " the trainer is fed are max_seq_token_per_tpu wide.",
        max_target_length,
        max_seq_token_per_tpu,
    )
    max_target_length = max_seq_token_per_tpu
  elif (
      max_seq_token_per_tpu is not None
      and 0 < max_seq_token_per_tpu < max_target_length
  ):
    logging.warning(
        "max_seq_token_per_tpu=%d is smaller than max_prompt_length + "
        "max_response_length (%d + %d = %d), which is not a legal packing "
        "budget -- validate_packing_budget rejects it on the learner. Keeping "
        "max_target_length=%d.",
        max_seq_token_per_tpu,
        max_prompt_length,
        max_response_length,
        max_target_length,
        max_target_length,
    )

  if profiling_options is not None:
    argv.extend([
        "profiler=xplane",
        f"profiler_steps={profiling_options.profiler_steps}",
        f"skip_first_n_steps_for_profiler={profiling_options.skip_first_n_steps}",
        f"profile_periodically_period={profiling_options.profiler_period}",
        f"upload_all_profiler_results={str(profiling_options.upload_all_profiler_results).lower()}",
    ])
    if profiling_options.enable_tpu_profiling_options:
      argv.extend([
          "enable_tpu_profiling_options=true",
          f"tpu_num_chips_to_profile_per_task={profiling_options.tpu_num_chips_to_profile_per_task}",
          f"tpu_num_sparse_cores_to_trace={profiling_options.tpu_num_sparse_cores_to_trace}",
          f"tpu_num_sparse_core_tiles_to_trace={profiling_options.tpu_num_sparse_core_tiles_to_trace}",
      ])
    else:
      argv.append("enable_tpu_profiling_options=false")
  else:
    argv.extend([
        "profiler=''",
        "profiler_steps=0",
        "skip_first_n_steps_for_profiler=-1",
    ])

  # Without options, MaxText trains with its base.yml optimizer defaults.
  # opt_type is left at MaxText's default, adamw (callers validate the flag),
  # so an M_OPT_TYPE env override of it still works.
  if adamw_options is not None:
    argv.extend([
        f"adam_b1={adamw_options.b1}",
        f"adam_b2={adamw_options.b2}",
        f"adam_eps={adamw_options.eps}",
        f"adam_weight_decay={adamw_options.weight_decay}",
        "gradient_clipping_threshold="
        f"{adamw_options.gradient_clipping_threshold}",
    ])

  # TODO(tunix-dev): [legacy-maxtext-a2cf3bc8b] Remove
  # `_pyconfig_supports_field` gating and emit `rollout_weight_dtype`
  # unconditionally once legacy pinned MaxText images are retired.
  supports_rollout_weight_dtype = _pyconfig_supports_field(
      pyconfig, "rollout_weight_dtype"
  )
  argv.extend([
      "scan_layers=True",
      "convert_checkpoint_if_possible=False",
      "skip_jax_distributed_system=True",
      "allow_split_physical_axes=True",
      f"per_device_batch_size={per_device_batch_size}",
      f"gradient_accumulation_steps={gradient_accumulation_steps}",
      f"max_target_length={max_target_length}",
      f"attention={attention or 'dot_product'}",
      "use_tokamax_gmm=true",
      "use_gmm_v2=true",
      f"ici_fsdp_parallelism={mesh_fsdp}",
      *(
          [f"padded_base_moe_mlp_dim={effective_padded_moe_mlp_dim}"]
          if effective_padded_moe_mlp_dim
          else []
      ),
      # The vLLM rollout replicates KV heads up to kv_tp_size (tp*ep) when the
      # model has fewer -- see maxtext_vllm_adapter. Weight sync pairs by name,
      # so the trainer must build the same shape. Prefer attention DP on the
      # rollout instead, which avoids the replication entirely; this is the
      # fallback when that is not available.
      *(
          [f"base_num_kv_heads={effective_kv_heads}"]
          if emit_kv_heads_override
          else []
      ),
      # MaxText's `pyconfig.validate_no_keys_overridden_twice` rejects any CLI
      # key that also appears in `{model_name}.yml` (including
      # `base_num_kv_heads`) unless `override_model_config=True` is set.
      *(["override_model_config=True"] if emit_kv_heads_override else []),
      f"ici_tensor_parallelism={mesh_tp}",
      f"ici_expert_parallelism={mesh_expert}",
      f"ici_context_parallelism={mesh_context}",
      # MaxText's context_parallel_load_balance=True assumes the training batch
      # arrives in DUAL_CHUNK_SWAP token order from MaxText's grain pipeline;
      # Tunix RL feeds batches in natural token order.
      *(["context_parallel_load_balance=False"] if mesh_context > 1 else []),
      f"learning_rate={learning_rate}",
      f"warmup_steps_fraction={warmup_steps_fraction}",
      "dtype=bfloat16",
      f"weight_dtype={weight_dtype}",
      *(
          [f"rollout_weight_dtype={DEFAULT_ROLLOUT_WEIGHT_DTYPE}"]
          if supports_rollout_weight_dtype
          else []
      ),
      "grad_dtype=float32",
      "enable_tensorboard=False",
      "record_internal_nn_metrics=False",
      "init_weights_seed=42",
      f"prefuse_moe_weights={prefuse_moe_weights}",
      f"use_weight_converter={use_weight_converter}",
      *(
          [
              f"rollout_tensor_parallelism={rollout_mesh_tp or kv_tp_size or moe_mlp_tp_size}"
          ]
          if (rollout_mesh_tp or kv_tp_size or moe_mlp_tp_size) > 0
          else []
      ),
  ])
  # Pathways persistence: let the TPU workers write the checkpoint themselves
  # The persistence handler rejects the OCDBT/zarr3 layout MaxText writes by
  # default (see maxtext/common/checkpoint_context.py), so both must be off
  if os.environ.get("ENABLE_PATHWAYS_PERSISTENCE", "") == "1":
    logging.info(
        "ENABLE_PATHWAYS_PERSISTENCE=1; disabling OCDBT/zarr3 so the Pathways "
        "persistence handler can save directly from the TPU workers."
    )
    argv.extend([
        "checkpoint_storage_use_ocdbt=false",
        "checkpoint_storage_use_zarr3=false",
    ])

  _d2h_gb = os.environ.get("CKPT_D2H_CONCURRENT_GB", "").strip()
  if _d2h_gb:
    logging.info(
        "CKPT_D2H_CONCURRENT_GB=%s; overriding "
        "checkpoint_storage_device_host_concurrent_gb.",
        _d2h_gb,
    )
    argv.append(f"checkpoint_storage_device_host_concurrent_gb={_d2h_gb}")

  _extra = os.environ.get("MAXTEXT_EXTRA_FLAGS", "").strip()
  if _extra:
    _pairs = [tok for tok in _extra.split() if tok]
    _bad = [tok for tok in _pairs if "=" not in tok]
    if _bad:
      raise ValueError(
          "MAXTEXT_EXTRA_FLAGS entries must be key=value, got: "
          f"{' '.join(_bad)}"
      )
    logging.info(
        "MAXTEXT_EXTRA_FLAGS adding %d flag(s): %s", len(_pairs), _pairs
    )
    argv.extend(_pairs)

  if not supports_rollout_weight_dtype:
    logging.info(
        "Installed MaxText does not declare 'rollout_weight_dtype'; omitting"
        " CLI override."
    )
  logging.info("MaxText config argv: %s", argv)
  return pyconfig.initialize(argv)


def create_maxtext_mesh(maxtext_config: Any) -> Any:
  """Builds the JAX device Mesh with axis names from MaxText config."""
  from jax.sharding import Mesh  # pylint: disable=g-import-not-at-top

  _, _, m_utils = maxtext_modules()
  devices = m_utils.create_device_mesh(maxtext_config)
  return Mesh(devices, maxtext_config.mesh_axes)


def log_param_shapes(model: Any) -> None:
  """Logs parameter shapes as a sanity check that weights loaded correctly."""
  from flax import nnx  # pylint: disable=g-import-not-at-top

  flat = nnx.to_pure_dict(nnx.state(model, nnx.Param))

  def walk(node, path=""):
    if isinstance(node, dict):
      for key, value in node.items():
        yield from walk(value, f"{path}.{key}" if path else str(key))
    elif hasattr(node, "shape"):
      yield path, node.shape

  shapes = dict(walk(flat))
  for name, shape in shapes.items():
    if "wi_0" in name or "query" in name:
      logging.info("MaxText param %s shape=%s", name, shape)
  logging.info("MaxText model has %d parameter arrays.", len(shapes))


def _configure_target_free_rollout_dtypes(
    engine: Any, maxtext_config: Any
) -> None:
  """Aligns target-free weight conversion dtypes with the installed MaxText.

  Trainer `weight_dtype=float32` keeps master weights and AdamW moments in FP32
  while the vLLM rollout model uses `DEFAULT_ROLLOUT_WEIGHT_DTYPE` (`bfloat16`).
  Older pinned MaxText container images (predating `rollout_weight_dtype`, such
  as images built for the `a2cf3bc8b` 35B recipe) default
  `MaxTextToMaxTextConverter.target_dtype` to the trainer's `weight_dtype`,
  unconditionally retain raw FP32 source dtypes on `"gate"`/`"router"` paths in
  `_execute_group_target_free` even when `float32_gate_logits=False`, and retain
  the `float32_gate_logits` FP32 exceptions in `common_types.get_weight_dtype`.

  We probe `common_types.get_weight_dtype` once at engine creation time (relying
  on the recipe invariant that trainer `maxtext_config.float32_gate_logits`
  matches the rollout node's `--maxtext_float32_gate_logits`). On upstream
  MaxText `main`, where `maxtext_config` declares `rollout_weight_dtype`,
  `fp32_modules` is empty, and `MaxTextToMaxTextConverter` already casts via
  `_resolve_target_dtype`, this function is a strict no-op.

  TODO(tunix-dev): [legacy-maxtext-a2cf3bc8b] Delete this shim (see cleanup
  manifest above `_maxtext_get_weight_dtype`) once older pinned MaxText images
  predating `rollout_weight_dtype` are retired.

  Args:
    engine: Initialized `MaxTextTrainingEngine` instance.
    maxtext_config: Trainer `HyperParameters` configuration.
  """
  # pylint: disable=protected-access
  direct_converter: Any = getattr(
      getattr(engine, "_weight_converter", None), "_direct", None
  )
  if direct_converter is None:
    return

  orig_exec_target_free = getattr(
      direct_converter, "_execute_group_target_free", None
  )
  if not callable(orig_exec_target_free):
    logging.warning(
        "MaxText direct weight converter has no callable"
        " '_execute_group_target_free'; skipping target-free rollout dtype"
        " shim."
    )
    return

  import jax  # pylint: disable=g-import-not-at-top
  from jax import numpy as jnp  # pylint: disable=g-import-not-at-top

  # Explicit type check avoids passing auto-created `MagicMock` attributes into
  # `jnp.dtype(...)` in unit tests that pass a bare `MagicMock` config.
  cfg_rollout_dtype = getattr(maxtext_config, "rollout_weight_dtype", None)
  has_rollout_weight_dtype = (
      isinstance(cfg_rollout_dtype, jnp.dtype)
      or (isinstance(cfg_rollout_dtype, str) and bool(cfg_rollout_dtype))
      or (
          isinstance(cfg_rollout_dtype, type)
          and issubclass(cfg_rollout_dtype, jnp.generic)
      )
  )
  if has_rollout_weight_dtype:
    rollout_dtype = jnp.dtype(cfg_rollout_dtype)
  else:
    # Mirrors `build_vllm_maxtext_additional_config`'s rollout `weight_dtype`.
    rollout_dtype = jnp.dtype(DEFAULT_ROLLOUT_WEIGHT_DTYPE)
  direct_converter.target_dtype = rollout_dtype

  get_weight_dtype = _maxtext_get_weight_dtype()
  if not callable(get_weight_dtype):
    return

  # Invariant: trainer `maxtext_config.float32_gate_logits` must match the
  # rollout worker's `--maxtext_float32_gate_logits` flag so both sides agree on
  # per-module storage dtypes during target-free Raiden weight sync.
  fp32_gate_logits = bool(
      getattr(maxtext_config, "float32_gate_logits", False)
  )
  rollout_probe = types.SimpleNamespace(
      weight_dtype=rollout_dtype,
      float32_gate_logits=fp32_gate_logits,
  )
  fp32_modules = frozenset(
      module_name
      for module_name in set(_ROLLOUT_DTYPE_MODULE_ALIASES.values())
      if jnp.dtype(get_weight_dtype(rollout_probe, module_name))
      != rollout_dtype
  )
  if has_rollout_weight_dtype and not fp32_modules:
    return

  logging.info(
      "Installing target-free rollout dtype shim (rollout_dtype=%s,"
      " float32_gate_logits=%s, fp32_modules=%s).",
      rollout_dtype,
      fp32_gate_logits,
      sorted(fp32_modules),
  )

  def _exec_target_free_rollout_dtypes(group: Any, src_flat: Any) -> Any:
    source_keys = getattr(group, "source_keys", ())
    keep_fp32 = (
        bool(fp32_modules)
        and bool(source_keys)
        and any(
            _ROLLOUT_DTYPE_MODULE_ALIASES.get(str(part)) in fp32_modules
            for part in tuple(source_keys[0])
        )
    )
    prev_target_dtype = direct_converter.target_dtype
    desired_dtype = jnp.float32 if keep_fp32 else jnp.dtype(prev_target_dtype)
    direct_converter.target_dtype = (
        jnp.float32 if keep_fp32 else prev_target_dtype
    )
    try:
      outs = orig_exec_target_free(group, src_flat)
    finally:
      direct_converter.target_dtype = prev_target_dtype
    if getattr(group, "quantize_to_fp8", False):
      return outs
    fixed = []
    for tgt_key, val in outs:
      if (
          hasattr(val, "dtype")
          and jnp.issubdtype(val.dtype, jnp.floating)
          and val.dtype != desired_dtype
      ):
        if isinstance(val, jax.ShapeDtypeStruct):
          val = jax.ShapeDtypeStruct(
              val.shape,
              desired_dtype,
              sharding=getattr(val, "sharding", None),
          )
        else:
          val = val.astype(desired_dtype)
      fixed.append((tgt_key, val))
    return fixed

  direct_converter._execute_group_target_free = _exec_target_free_rollout_dtypes
  # pylint: enable=protected-access


def create_maxtext_engine(
    maxtext_config: Any,
    mesh: Any,
    tokenizer_pad_id: int = 0,
    wrap_with_tunix_adapter: bool = True,
    log_shapes: bool = True,
) -> Any:
  """Builds and initializes a MaxTextTrainingEngine within the given mesh."""
  from flax.core.spmd import logical_axis_rules  # pylint: disable=g-import-not-at-top

  _, maxtext_engine, _ = maxtext_modules()

  with mesh, logical_axis_rules(maxtext_config.logical_axis_rules):
    engine = maxtext_engine.MaxTextTrainingEngine(
        maxtext_config,
        mesh=mesh,
        wrap_with_tunix_adapter=wrap_with_tunix_adapter,
        tokenizer_pad_id=tokenizer_pad_id,
    )

  # TODO(tunix-dev): [legacy-maxtext-a2cf3bc8b] Remove once legacy pinned
  # MaxText images predating `rollout_weight_dtype` are retired.
  _configure_target_free_rollout_dtypes(engine, maxtext_config)

  model_type = type(engine.model).__name__
  logging.info(
      "MaxText engine model: %s (pad_id=%d)", model_type, tokenizer_pad_id
  )
  if wrap_with_tunix_adapter and model_type != "TunixMaxTextAdapter":
    raise RuntimeError(
        f"Expected the engine's model to be TunixMaxTextAdapter, got {model_type}."
    )
  if log_shapes:
    log_param_shapes(engine.model)

  return engine


def _iter_leaves(tree, jax_mod):
  tree_util = getattr(jax_mod, "tree_util", None)
  if tree_util is not None and hasattr(tree_util, "tree_leaves"):
    yield from tree_util.tree_leaves(tree)
    return
  if isinstance(tree, dict):
    for v in tree.values():
      yield from _iter_leaves(v, jax_mod)
  elif isinstance(tree, (list, tuple)):
    for v in tree:
      yield from _iter_leaves(v, jax_mod)
  else:
    yield tree


def _delete_pytree_buffers(tree, jax_mod, keep_tree=None):
  array_cls = getattr(jax_mod, "Array", ())
  if not isinstance(array_cls, type):
    return
  keep_ids = set()
  if keep_tree is not None:
    for leaf in _iter_leaves(keep_tree, jax_mod):
      val = getattr(leaf, "value", leaf)
      keep_ids.add(id(val))
  for leaf in _iter_leaves(tree, jax_mod):
    val = getattr(leaf, "value", leaf)
    if id(val) in keep_ids:
      continue
    if isinstance(val, array_cls) and not getattr(
        val, "is_deleted", lambda: False
    )():
      try:
        val.delete()
      except Exception:  # pylint: disable=broad-exception-caught
        pass


def load_and_convert_scanned_checkpoint(
    path: str | os.PathLike,
    sampler: Any,
    *,
    mesh_tp: int = 1,
    ckpt_prefuse_moe: bool = False,
    maxtext_config_overrides: dict[str, Any] | None = None,
    base_config_path: str | os.PathLike | None = None,
) -> None:
  """Restores a scanned MaxText checkpoint and converts it into vLLM's unscanned state in-memory."""
  import gc  # pylint: disable=import-outside-toplevel
  from flax import nnx  # pylint: disable=import-outside-toplevel
  import jax  # pylint: disable=import-outside-toplevel
  from maxtext.common.common_types import MODEL_MODE_AUTOREGRESSIVE  # pylint: disable=import-outside-toplevel
  from maxtext.configs import pyconfig  # pylint: disable=import-outside-toplevel
  from maxtext.integration.vllm.weight_converter import MaxTextToMaxTextConverter  # pylint: disable=import-outside-toplevel
  from maxtext.utils import model_creation_utils  # pylint: disable=import-outside-toplevel
  from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR  # pylint: disable=import-outside-toplevel

  logging.info(
      "Restoring scanned MaxText checkpoint from %s and converting to unscanned"
      " vLLM weights in-memory...",
      path,
  )
  vllm_sampler = getattr(sampler, "vllm_sampler", sampler)
  if vllm_sampler is None:
    if hasattr(sampler, "_get_underlying_sampler"):
      vllm_sampler = sampler._get_underlying_sampler()
    elif hasattr(sampler, "initialize"):
      sampler.initialize()
      vllm_sampler = getattr(sampler, "vllm_sampler", sampler)
  if vllm_sampler is None:
    vllm_sampler = sampler
  reinit_needed = False
  if hasattr(vllm_sampler, "delete_cache"):
    vllm_sampler.delete_cache()
    reinit_needed = True
  _delete_pytree_buffers(
      getattr(vllm_sampler, "transformer_state", None), jax
  )
  try:
    if base_config_path is None:
      base_config_path = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
    scanned_overrides = dict(maxtext_config_overrides or {})
    scanned_overrides.update({
        "load_parameters_path": str(path),
        "scan_layers": True,
        "prefuse_moe_weights": ckpt_prefuse_moe,
        "attention": "dot_product",
        "model_call_mode": "",
    })
    scanned_cfg = pyconfig.initialize(
        ["", str(base_config_path)], **scanned_overrides
    )
    scanned_model, scanned_mesh = model_creation_utils.from_pretrained(
        scanned_cfg,
        devices=jax.devices(),
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
    )
    scanned_state = nnx.state(scanned_model, nnx.Param)
    del scanned_model, scanned_mesh
    gc.collect()

    converter = MaxTextToMaxTextConverter(
        config=scanned_cfg,
        tp=mesh_tp,
        prefuse_moe_weights=True,
        target_dtype=None,
    )
    orig_exec_group = getattr(converter, "_execute_group", None)
    prefused_moe_sources = {}
    orig_build_plan = getattr(converter, "_build_plan", None)
    if ckpt_prefuse_moe and mesh_tp > 1 and callable(orig_build_plan):

      def _split_prefused_moe_and_build_plan(
          src_flat, tgt_flat, skip_paths=frozenset()
      ):
        for k in list(src_flat.keys()):
          if "layers" in k and k[-1] == "wi":
            wi = src_flat.pop(k)
            prefix = k[:-1]
            if callable(orig_exec_group):
              prefused_moe_sources[prefix] = wi
              src_flat[prefix + ("wi_0",)] = wi
              src_flat[prefix + ("wi_1",)] = wi
            else:
              half = wi.shape[-1] // 2
              src_flat[prefix + ("wi_0",)] = wi[..., :half]
              src_flat[prefix + ("wi_1",)] = wi[..., half:]
        return orig_build_plan(src_flat, tgt_flat, skip_paths)

      converter._build_plan = _split_prefused_moe_and_build_plan

    if callable(orig_exec_group):

      def _exec_and_free_group(group, src_flat, tgt_flat):
        prefused_wi = None
        if group.op == "fuse_moe" and group.source_keys:
          prefused_wi = prefused_moe_sources.pop(
              group.source_keys[0][:-1], None
          )
          if prefused_wi is not None:
            half = prefused_wi.shape[-1] // 2
            wi_0 = prefused_wi[..., :half]
            wi_1 = prefused_wi[..., half:]
            src_flat[group.source_keys[0]] = wi_0
            src_flat[group.source_keys[1]] = wi_1
        outs = orig_exec_group(group, src_flat, tgt_flat)
        out_arrays = [out for _, out in outs]
        if hasattr(jax, "block_until_ready"):
          jax.block_until_ready(out_arrays)
        _delete_pytree_buffers(
            [prefused_wi] + [src_flat.get(k) for k in group.source_keys],
            jax,
            keep_tree=out_arrays,
        )
        aligned_outs = []
        for tgt_key, out in outs:
          tgt_sharding = getattr(tgt_flat.get(tgt_key), "sharding", None)
          out_sharding = getattr(out, "sharding", None)
          if (
              tgt_sharding is not None
              and out_sharding is not None
              and out_sharding != tgt_sharding
              and hasattr(jax, "device_put")
          ):
            resharded_out = jax.device_put(out, tgt_sharding)
            if hasattr(jax, "block_until_ready"):
              jax.block_until_ready(resharded_out)
            _delete_pytree_buffers(out, jax, keep_tree=resharded_out)
            out = resharded_out
          aligned_outs.append((tgt_key, out))
        return aligned_outs

      converter._execute_group = _exec_and_free_group

    converted_state = converter.convert(
        scanned_state,
        target_state=vllm_sampler.transformer_state,
    )
    _delete_pytree_buffers(scanned_state, jax, keep_tree=converted_state)
    del scanned_state
    gc.collect()

    while isinstance(converted_state, dict) and "model" in converted_state:
      converted_state = converted_state["model"]
    sampler_cfg = getattr(vllm_sampler, "config", None)
    orig_free_kv = getattr(
        sampler_cfg, "free_kv_cache_during_weight_sync", None
    )
    if sampler_cfg is not None and orig_free_kv is not None:
      sampler_cfg.free_kv_cache_during_weight_sync = False
    reshard_mod = None
    orig_reshard_pytree = None
    try:
      from tunix.rl import reshard as reshard_mod  # pylint: disable=import-outside-toplevel

      orig_reshard_pytree = getattr(reshard_mod, "reshard_pytree", None)
    except Exception:  # pylint: disable=broad-exception-caught
      reshard_mod = None
    if (
        reshard_mod is not None
        and orig_reshard_pytree is not None
        and hasattr(jax, "tree_util")
        and hasattr(jax, "device_put")
    ):

      def _inplace_or_chunked_reshard(source, target, **kwargs):
        del kwargs

        def _put_or_keep(x, dst):
          sharding = getattr(dst, "sharding", dst)
          if sharding is None or getattr(x, "sharding", None) == sharding:
            return x
          out = jax.device_put(x, sharding)
          if hasattr(jax, "block_until_ready"):
            jax.block_until_ready(out)
          _delete_pytree_buffers(x, jax, keep_tree=out)
          return out

        return jax.tree_util.tree_map(_put_or_keep, source, target)

      reshard_mod.reshard_pytree = _inplace_or_chunked_reshard
    try:
      vllm_sampler.update_params(converted_state)
    finally:
      if reshard_mod is not None and orig_reshard_pytree is not None:
        reshard_mod.reshard_pytree = orig_reshard_pytree
      if sampler_cfg is not None and orig_free_kv is not None:
        sampler_cfg.free_kv_cache_during_weight_sync = orig_free_kv
    _delete_pytree_buffers(
        converted_state,
        jax,
        keep_tree=getattr(vllm_sampler, "transformer_state", None),
    )
    del converted_state
    gc.collect()
    jax.clear_caches()
    if hasattr(jax, "effects_barrier"):
      jax.effects_barrier()
  finally:
    if reinit_needed and hasattr(vllm_sampler, "reinitialize_cache"):
      vllm_sampler.reinitialize_cache()
  logging.info("Completed in-memory scanned-to-unscanned weight conversion.")
