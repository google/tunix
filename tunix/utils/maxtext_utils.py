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

import dataclasses
import logging
import os
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


def maxtext_modules():
  """Imports MaxText lazily; some installs nest it under maxtext.src.maxtext."""
  from maxtext.configs import pyconfig  # pylint: disable=g-import-not-at-top
  from maxtext.training_engine import maxtext_engine  # pylint: disable=g-import-not-at-top
  from maxtext.utils import maxtext_utils  # pylint: disable=g-import-not-at-top
  return pyconfig, maxtext_engine, maxtext_utils


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


def _resolve_bool_config(
    value: bool | None, env_var: str, default: bool | None = None
) -> bool | None:
  """Resolves an optional boolean parameter from an explicit arg or env var."""
  if value is not None:
    return bool(value)
  raw = os.environ.get(env_var, "").strip().lower()
  if raw in ("1", "true", "yes"):
    return True
  if raw in ("0", "false", "no"):
    return False
  return default


def build_vllm_maxtext_additional_config(
    model_name: str,
    *,
    attention: str = "",
    prefuse_moe_weights: bool | None = None,
    return_routed_experts: bool | None = None,
    float32_gate_logits: bool | None = None,
    float32_logits: bool | None = None,
) -> dict[str, Any]:
  """Builds the vLLM `additional_config` a MaxText rollout model reads."""
  effective_float32_gate_logits = _resolve_bool_config(
      float32_gate_logits, "FLOAT32_GATE_LOGITS", default=True
  )
  effective_float32_logits = _resolve_bool_config(
      float32_logits, "FLOAT32_LOGITS", default=None
  )
  overrides: dict[str, Any] = {
      "model_name": model_name,
      "model_call_mode": "inference",
      "scan_layers": False,
      "enable_dp_attention": False,
      "allow_split_physical_axes": True,
      "log_config": False,
      "weight_dtype": "bfloat16",
  }
  if effective_float32_gate_logits is not None:
    overrides["float32_gate_logits"] = effective_float32_gate_logits
  if effective_float32_logits is not None:
    overrides["float32_logits"] = effective_float32_logits
  if prefuse_moe_weights is not None:
    overrides["prefuse_moe_weights"] = prefuse_moe_weights
  if return_routed_experts is not None:
    overrides["return_routed_experts"] = return_routed_experts
  if attention:
    overrides["attention"] = attention
  return {"maxtext_config": overrides}


def build_maxtext_config(
    model_name: str,
    worker_id: str = "",
    train_micro_batch_size: int = 1,
    mesh_fsdp: int = 1,
    mesh_tp: int = 1,
    mesh_expert: int = 1,
    mesh_context: int = 1,
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
    base_num_kv_heads: int = 0,
    kv_tp_size: int = 0,
    moe_mlp_tp_size: int = 0,
    rollout_mesh_tp: int = 0,
    prefuse_moe_weights: bool = False,
    use_weight_converter: bool = True,
    max_seq_token_per_tpu: int | None = 0,
    trainable_parameters_mask: list[str] | str | None = None,
    attention: str | None = None,
    remat_policy: str = "",
    learning_rate_final_fraction: float | None = None,
    adam_b1: float | None = None,
    adam_b2: float | None = None,
    adam_eps: float | None = None,
    adam_weight_decay: float | None = None,
    gradient_clipping_threshold: float | None = None,
    skip_step_on_spikes: bool = False,
    skip_step_on_nan: bool = True,
    skip_step_interval: int = 128,
    skip_step_scaling_factor: float = 6.0,
    float32_gate_logits: bool | None = None,
    float32_logits: bool | None = None,
) -> Any:
  """Builds the MaxText HyperParameters the training engine runs on."""
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

  if max_seq_token_per_tpu is not None and max_seq_token_per_tpu > 0:
    # When sequence packing is enabled, the batch assembler feeds 1 packed row
    # per FSDP shard (global batch size = mesh_fsdp), independent of
    # train_micro_batch_size.
    # TODO(maxtext-team): verify this is the desired behavior.
    per_device_batch_size = mesh_fsdp / num_devices
  else:
    if train_micro_batch_size % mesh_fsdp:
      raise ValueError(
          f"train_micro_batch_size={train_micro_batch_size} must be a multiple"
          f" of mesh_fsdp={mesh_fsdp}; MaxText shards the batch dimension"
          " across it."
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
  if effective_kv_heads <= 0 and kv_tp_size > 0:
    if model_data:
      effective_kv_heads = int(model_data.get("base_num_kv_heads") or 0)
    if effective_kv_heads <= 0:
      raise ValueError(
          f"kv_tp_size ({kv_tp_size}) requires base_num_kv_heads > 0, but could"
          f" not determine base_num_kv_heads from config or {model_name}.yml."
          " Please specify --base_num_kv_heads."
      )

  if effective_kv_heads > 0 and kv_tp_size > effective_kv_heads:
    if kv_tp_size % effective_kv_heads != 0:
      raise ValueError(
          f"kv_tp_size ({kv_tp_size}) must be cleanly divisible by "
          f"base_num_kv_heads ({effective_kv_heads})."
      )
    effective_kv_heads = kv_tp_size

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
        f"profiler_steps={profiling_options.profiler_steps}",
        f"skip_first_n_steps_for_profiler={profiling_options.skip_first_n_steps}",
        f"profile_periodically_period={profiling_options.profiler_period}",
    ])

  effective_attention = (
      attention
      or os.environ.get("TRAINER_MAXTEXT_ATTENTION", "").strip()
      or "dot_product"
  )
  effective_float32_gate_logits = _resolve_bool_config(
      float32_gate_logits, "FLOAT32_GATE_LOGITS", default=True
  )
  effective_float32_logits = _resolve_bool_config(
      float32_logits, "FLOAT32_LOGITS", default=None
  )
  argv.extend([
      "scan_layers=True",
      "convert_checkpoint_if_possible=False",
      "skip_jax_distributed_system=True",
      "allow_split_physical_axes=True",
      f"per_device_batch_size={per_device_batch_size}",
      f"gradient_accumulation_steps={gradient_accumulation_steps}",
      f"max_target_length={max_target_length}",
      f"attention={effective_attention}",
      *([f"remat_policy={remat_policy}"] if remat_policy else []),
      *(
          [f"learning_rate_final_fraction={learning_rate_final_fraction}"]
          if learning_rate_final_fraction is not None
          else []
      ),
      "use_tokamax_gmm=true",
      "use_gmm_v2=true",
      f"ici_fsdp_parallelism={mesh_fsdp}",
      *(
          [f"padded_base_moe_mlp_dim={effective_padded_moe_mlp_dim}"]
          if effective_padded_moe_mlp_dim
          else []
      ),
      *(
          [
              f"base_num_kv_heads={effective_kv_heads}",
              "override_model_config=true",
          ]
          if effective_kv_heads
          else []
      ),
      f"ici_tensor_parallelism={mesh_tp}",
      f"ici_expert_parallelism={mesh_expert}",
      f"ici_context_parallelism={mesh_context}",
      *(["context_parallel_load_balance=False"] if mesh_context > 1 else []),
      f"learning_rate={learning_rate}",
      f"warmup_steps_fraction={warmup_steps_fraction}",
      *(
          f"{key}={value}"
          for key, value in (
              ("adam_b1", adam_b1),
              ("adam_b2", adam_b2),
              ("adam_eps", adam_eps),
              ("adam_weight_decay", adam_weight_decay),
              ("gradient_clipping_threshold", gradient_clipping_threshold),
          )
          if value is not None
      ),
      "dtype=bfloat16",
      "weight_dtype=float32",
      "grad_dtype=float32",
      *(
          [f"float32_gate_logits={effective_float32_gate_logits}"]
          if effective_float32_gate_logits is not None
          else []
      ),
      *(
          [f"float32_logits={effective_float32_logits}"]
          if effective_float32_logits is not None
          else []
      ),
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
      *(
          [f"trainable_parameters_mask={trainable_parameters_mask}"]
          if trainable_parameters_mask
          else []
      ),
      *(["skip_step_on_spikes=True"] if skip_step_on_spikes else []),
      *(["skip_step_on_nan=True"] if skip_step_on_nan else ["skip_step_on_nan=False"]),
      *(
          [
              f"skip_step_interval={skip_step_interval}",
              f"skip_step_scaling_factor={skip_step_scaling_factor}",
          ]
          if skip_step_on_spikes
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

  if os.environ.get("OVERRIDE_MODEL_CONFIG", "").lower() in ("1", "true") and "override_model_config=true" not in argv:
    argv.append("override_model_config=true")

  _extra = os.environ.get("MAXTEXT_EXTRA_FLAGS", "").strip()
  if _extra:
    _pairs = [tok for tok in _extra.split() if tok]
    _bad = [tok for tok in _pairs if "=" not in tok]
    if _bad:
      raise ValueError(
          "MAXTEXT_EXTRA_FLAGS entries must be key=value, got: "
          f"{' '.join(_bad)}"
      )
    logging.info("MAXTEXT_EXTRA_FLAGS adding %d flag(s): %s", len(_pairs), _pairs)
    argv.extend(_pairs)

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


def create_maxtext_engine(
    maxtext_config: Any,
    mesh: Any,
    tokenizer_pad_id: int = 0,
    wrap_with_tunix_adapter: bool = True,
    log_shapes: bool = True,
) -> Any:
  """Builds and initializes a MaxTextTrainingEngine within the given mesh."""
  _, maxtext_engine, _ = maxtext_modules()

  with mesh:
    engine = maxtext_engine.MaxTextTrainingEngine(
        maxtext_config,
        mesh=mesh,
        wrap_with_tunix_adapter=wrap_with_tunix_adapter,
        tokenizer_pad_id=tokenizer_pad_id,
    )

  # cl/995291149 sets trainer weight_dtype=float32 so master weights and AdamW
  # moments stay in FP32, while the vLLM rollout model uses weight_dtype=bfloat16
  # (with float32_gate_logits keeping norm/gate/router/GDN/logits_dense modules
  # in FP32). Ensure target-free weight conversion exports exact rollout dtypes.
  direct_converter = getattr(
      getattr(engine, "_weight_converter", None), "_direct", None
  )
  if direct_converter is not None:
    import jax  # pylint: disable=g-import-not-at-top
    from jax import numpy as jnp  # pylint: disable=g-import-not-at-top

    direct_converter.target_dtype = "bfloat16"
    orig_exec_target_free = getattr(
        direct_converter, "_execute_group_target_free", None
    )
    if callable(orig_exec_target_free):
      fp32_gate_logits = bool(
          getattr(maxtext_config, "float32_gate_logits", False)
      )
      fp32_modules = frozenset({
          "norm",
          "input_layernorm",
          "post_attention_layernorm",
          "gate",
          "router",
          "shared_expert_gate",
          "A_log",
          "dt_bias",
          "conv1d",
          "logits_dense",
      })

      def _exec_target_free_rollout_dtypes(group: Any, src_flat: Any) -> Any:
        keep_fp32 = fp32_gate_logits and bool(
            fp32_modules.intersection(
                str(part) for part in group.source_keys[0]
            )
        )
        desired_dtype = jnp.float32 if keep_fp32 else jnp.bfloat16
        prev_target_dtype = direct_converter.target_dtype
        direct_converter.target_dtype = "float32" if keep_fp32 else "bfloat16"
        try:
          outs = orig_exec_target_free(group, src_flat)
        finally:
          direct_converter.target_dtype = prev_target_dtype
        if getattr(group, "quantize_to_fp8", False):
          return outs
        fixed = []
        for tgt_key, val in outs:
          if hasattr(val, "dtype") and val.dtype != desired_dtype:
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

      direct_converter._execute_group_target_free = (
          _exec_target_free_rollout_dtypes
      )

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
