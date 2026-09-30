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

# Mirrors maxtext `pathways_checkpointing_impl`. Duplicated as literals rather than
# imported: this module must stay importable without maxtext installed.
_PERSISTENCE_IMPL = "persistence"
_COLOCATED_PYTHON_IMPL = "colocated_python"
_PATHWAYS_CHECKPOINTING_IMPLS = (_PERSISTENCE_IMPL, _COLOCATED_PYTHON_IMPL)


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
    return_routed_experts: Whether the MaxText rollout model should sow and
      return top-k routed expert indices for router replay.
    float32_gate_logits: Whether to keep gate/router/norm/GDN weights in
      float32. Defaults to True (or `FLOAT32_GATE_LOGITS` env var if set).
    float32_logits: Whether to cast output logits to float32. Defaults to
      `FLOAT32_LOGITS` env var if set, else None.

  Returns:
    The `additional_config` mapping to hand to the vLLM engine.
  """
  rollout_moe_fp8 = bool(
      _resolve_bool_config(None, "ROLLOUT_MOE_FP8", default=False)
  )
  effective_float32_gate_logits = _resolve_bool_config(
      float32_gate_logits, "FLOAT32_GATE_LOGITS", default=True
  )
  effective_float32_logits = _resolve_bool_config(
      float32_logits,
      "FLOAT32_LOGITS",
      default=True if rollout_moe_fp8 else None,
  )
  overrides: dict[str, Any] = {
      "model_name": model_name,
      "model_call_mode": "inference",
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
  # 0 and None mean packing is off. Any positive value is a packing budget, and
  # one too small to hold a maximal sequence is not a degraded mode: the
  # learner's rl_utils.validate_packing_budget rejects it once the assembler is
  # built. Reject it here too, before the mesh and the model are.
  if max_seq_token_per_tpu:
    longest = max_prompt_length + max_response_length
    if max_seq_token_per_tpu < longest:
      raise ValueError(
          f"max_seq_token_per_tpu={max_seq_token_per_tpu} is smaller than the"
          f" longest possible sequence (max_prompt_length {max_prompt_length} +"
          f" max_response_length {max_response_length} = {longest}); packing"
          f" cannot place such a sequence in a row. Set"
          f" max_seq_token_per_tpu >= {longest}."
      )

  if train_micro_batch_size % mesh_fsdp:
    raise ValueError(
        f"train_micro_batch_size={train_micro_batch_size} must be a multiple of "
        f"mesh_fsdp={mesh_fsdp}; MaxText shards the batch dimension across it."
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
  # An undersized budget was rejected above, so a packed row is exactly
  # max_seq_token_per_tpu wide and an unpacked one holds one trajectory.
  if max_seq_token_per_tpu:
    logging.info(
        "Sequence packing: max_target_length=%d (max_seq_token_per_tpu) rather"
        " than %d (max_prompt_length + max_response_length); the rows the"
        " trainer is fed are one packing budget wide.",
        max_seq_token_per_tpu,
        max_prompt_length + max_response_length,
    )
    max_target_length = max_seq_token_per_tpu
  else:
    max_target_length = max_prompt_length + max_response_length

  if profiling_options is not None:
    argv.extend([
        "profiler=xplane",
        f"profiler_steps={profiling_options.profiler_steps}",
        f"skip_first_n_steps_for_profiler={profiling_options.skip_first_n_steps}",
        f"profile_periodically_period={profiling_options.profiler_period}",
    ])
  else:
    argv.extend([
        "profiler=''",
        "profiler_steps=0",
        "skip_first_n_steps_for_profiler=-1",
    ])

  effective_attention = (
      attention
      or os.environ.get("TRAINER_MAXTEXT_ATTENTION")
      or "dot_product"
  )
  rollout_moe_fp8 = bool(
      _resolve_bool_config(None, "ROLLOUT_MOE_FP8", default=False)
  )
  effective_float32_gate_logits = _resolve_bool_config(
      float32_gate_logits, "FLOAT32_GATE_LOGITS", default=True
  )
  effective_float32_logits = _resolve_bool_config(
      float32_logits,
      "FLOAT32_LOGITS",
      default=True if rollout_moe_fp8 else None,
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
      # The vLLM rollout replicates KV heads up to kv_tp_size (tp*ep) when the
      # model has fewer -- see maxtext_vllm_adapter. Weight sync pairs by name,
      # so the trainer must build the same shape. Prefer attention DP on the
      # rollout instead, which avoids the replication entirely; this is the
      # fallback when that is not available.
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
      # Qwen3.5's GatedDeltaNet layers carry a recurrence, so device order is
      # sequence order: device i composes the state device i-1 left behind. The
      # default DUAL_CHUNK_SWAP balancing hands device 0 the first and last
      # chunks, device 1 the second and second-to-last, which composes the
      # segments out of order. Softmax attention tolerates that because it
      # rebuilds the causal mask from positions; a recurrence cannot. MaxText
      # rejects the combination outright, and warns that the run would otherwise
      # still train with the loss falling -- i.e. it fails silently.
      *(["context_parallel_load_balance=False"] if mesh_context > 1 else []),
      f"learning_rate={learning_rate}",
      f"warmup_steps_fraction={warmup_steps_fraction}",
      # AdamW and clipping, each emitted only when given: an unset one keeps
      # base.yml's default (adam_b2=0.95, adam_weight_decay=0.1,
      # gradient_clipping_threshold=1.0).
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
      "weight_dtype=bfloat16",
      "grad_dtype=float32",
      "mu_dtype=float32",
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
    if save_interval_steps > 0 and not output_dir.startswith("gs://"):
      raise ValueError(
          "ENABLE_PATHWAYS_PERSISTENCE=1 with save_interval_steps > 0 "
          "requires a gs:// base_output_directory so all pathways-worker pods "
          f"write to shared GCS storage; got {output_dir!r}. "
          "Set MAXTEXT_OUTPUT_DIR=gs://..."
      )

    impl = os.environ.get("PATHWAYS_CHECKPOINTING_IMPL", "").strip() or _PERSISTENCE_IMPL
    if impl not in _PATHWAYS_CHECKPOINTING_IMPLS:
      raise ValueError(
          f"PATHWAYS_CHECKPOINTING_IMPL={impl!r} is not recognised; "
          f"expected one of {_PATHWAYS_CHECKPOINTING_IMPLS}."
      )
    if impl == _COLOCATED_PYTHON_IMPL and not os.environ.get("COLOCATED_PYTHON_SIDECAR_IMAGE", "").strip():
      raise ValueError(
          "PATHWAYS_CHECKPOINTING_IMPL=colocated_python requires "
          "COLOCATED_PYTHON_SIDECAR_IMAGE to be set so the sidecar container is added to "
          "the pathways-worker pods. Without it Orbax silently falls back to "
          "controller-side host staging, which OOMs the proxy pod at 397B scale."
      )
    argv.append(f"pathways_checkpointing_impl={impl}")

    # Keep OCDBT/zarr3 off in BOTH modes: colocated_python supports them, but matching
    # the persistence layout keeps checkpoints restorable across a mode switch.
    logging.info(
        "ENABLE_PATHWAYS_PERSISTENCE=1 (impl=%s); disabling OCDBT/zarr3 so the Pathways "
        "handler can save directly from the TPU workers and both modes share one layout.",
        impl,
    )
    argv.extend([
        "checkpoint_storage_use_ocdbt=false",
        "checkpoint_storage_use_zarr3=false",
    ])

  _ckpt_async = os.environ.get("CHECKPOINT_ASYNC", "").strip()
  if _ckpt_async:
    argv.append(f"async_checkpointing={_ckpt_async}")

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

  # Generic passthrough, applied last so it wins over anything derived above.
  # MaxText exposes far more knobs than this helper has named parameters for --
  # MoE kernel selection, splash-attention block sizes, GDN tuning, custom mesh
  # rules -- and a run that needs one of them otherwise has nowhere to put it.
  # Space-separated key=value pairs, e.g.
  #   MAXTEXT_EXTRA_FLAGS="use_ring_of_experts=true sa_block_q=512"
  # MaxText rejects unknown keys outright (ValueError listing every valid
  # field), so a typo fails at startup rather than being silently dropped.
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
  try:
    return pyconfig.initialize(argv)
  except ValueError as e:
    if "pathways_checkpointing_impl" in str(e):
      argv = [
          arg
          for arg in argv
          if not arg.startswith("pathways_checkpointing_impl=")
      ]
      logging.warning(
          "Installed MaxText does not recognize 'pathways_checkpointing_impl'. "
          "Retrying initialization without it."
      )
      return pyconfig.initialize(argv)
    raise


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


def _build_fp32_master_optimizer_cls(nnx_mod: Any | None = None) -> type[Any]:
  """Builds an `nnx.Optimizer` subclass that maintains FP32 master weights.

  When `weight_dtype=bfloat16`, standard `nnx.Optimizer` initializes Optax's
  first/second moments (`mu`, `nu`) in `bfloat16` and applies `w + lr * update`
  directly in `bfloat16`. At `lr=1e-6`, the `bfloat16` unit-in-the-last-place
  (`2^-8 * |w|`) is ~3-4 orders of magnitude larger than the per-step update,
  causing parameter updates on `bfloat16` weights to underflow to zero.

  `Fp32MasterOptimizer` keeps model weights in their original dtype (`bfloat16`
  for bulk weights, `float32` for `float32_gate_logits` layers) during forward,
  backward, and weight sync, while storing a `float32` master copy and `float32`
  Optax state in `self.opt_state`. Each `update()` step updates the `float32`
  master weights and casts them back to each model parameter's dtype.
  """
  from flax.nnx.training import optimizer as nnx_opt  # pylint: disable=g-import-not-at-top
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  import optax  # pylint: disable=g-import-not-at-top

  if nnx_mod is None:
    from flax import nnx as nnx_mod  # pylint: disable=g-import-not-at-top

  opt_state_var_cls = getattr(
      nnx_opt, "OptState", getattr(nnx_mod, "Variable", None)
  )
  to_opt_state_fn = getattr(nnx_opt, "to_opt_state", lambda x: x)

  class Fp32MasterOptimizer(nnx_mod.Optimizer):
    """NNX Optimizer that keeps FP32 master weights and FP32 Optax state."""

    def __init__(
        self,
        model: Any,
        tx: optax.GradientTransformation,
        *,
        wrt: Any = nnx_mod.Param,
    ):
      self.step = opt_state_var_cls(jnp.array(0, dtype=jnp.uint32))
      self.tx = tx
      self.wrt = wrt
      params_state = nnx_mod.state(model, wrt)
      # Only allocate a separate float32 master copy for non-float32 (e.g.
      # bfloat16) parameters. Parameters that are already float32 (such as
      # float32_gate_logits layers) use the model parameter directly; storing a
      # second reference via `x.astype(jnp.float32)` would alias the same PJRT
      # buffer inside `TrainStateNNX` and fail `jax.jit(..., donate_argnums=(0,))`.
      master_params = jax.tree.map(
          lambda x: x.astype(jnp.float32) if x.dtype != jnp.float32 else None,
          params_state,
      )
      full_f32_params = jax.tree.map(
          lambda p, mp: p if mp is None else mp,
          params_state,
          master_params,
      )
      inner_state = tx.init(full_f32_params)
      opt_state: dict[str, Any] = {
          "master_params": master_params,
          "inner": inner_state,
      }
      if isinstance(inner_state, dict) and "is_skipped" in inner_state:
        opt_state["is_skipped"] = inner_state["is_skipped"]
      self.opt_state = nnx_mod.data(to_opt_state_fn(opt_state))

    def update(self, model: Any, grads: Any, /, **kwargs: Any) -> Any:
      param_arrays = nnx_mod.as_pure(nnx_mod.state(model, self.wrt))
      grad_arrays = nnx_mod.as_pure(nnx_mod.state(grads, self.wrt))
      opt_state_arrays = nnx_mod.as_pure(self.opt_state)
      kwargs_arrays = nnx_mod.as_pure(kwargs)

      master_params = opt_state_arrays["master_params"]
      inner_state = opt_state_arrays["inner"]
      full_f32_params = jax.tree.map(
          lambda p, mp: p if mp is None else mp,
          param_arrays,
          master_params,
      )
      grads_f32 = jax.tree.map(lambda g: g.astype(jnp.float32), grad_arrays)

      updates, new_inner = self.tx.update(
          grads_f32, inner_state, full_f32_params, **kwargs_arrays
      )
      updated_f32_params = optax.apply_updates(full_f32_params, updates)
      new_master = jax.tree.map(
          lambda p, up: up if p.dtype != jnp.float32 else None,
          param_arrays,
          updated_f32_params,
      )
      new_params = jax.tree.map(
          lambda p, up: up.astype(p.dtype),
          param_arrays,
          updated_f32_params,
      )
      new_opt_state: dict[str, Any] = {
          "master_params": new_master,
          "inner": new_inner,
      }
      if isinstance(new_inner, dict) and "is_skipped" in new_inner:
        new_opt_state["is_skipped"] = new_inner["is_skipped"]

      nnx_mod.update(model, new_params)
      nnx_mod.update(self.opt_state, nnx_mod.state(new_opt_state))
      self.step[...] += 1
      return updates

  return Fp32MasterOptimizer


def create_maxtext_engine(
    maxtext_config: Any,
    mesh: Any,
    tokenizer_pad_id: int = 0,
    wrap_with_tunix_adapter: bool = True,
    log_shapes: bool = True,
) -> Any:
  """Builds and initializes a MaxTextTrainingEngine within the given mesh."""
  _, maxtext_engine, _ = maxtext_modules()

  weight_dtype_str = str(getattr(maxtext_config, "weight_dtype", "bfloat16"))
  use_fp32_master = weight_dtype_str in ("bfloat16", "bf16")

  if use_fp32_master:
    fp32_optimizer_cls = _build_fp32_master_optimizer_cls(
        getattr(maxtext_engine, "nnx", None)
    )

    class _Fp32MasterMaxTextTrainingEngine(
        maxtext_engine.MaxTextTrainingEngine
    ):
      """MaxTextTrainingEngine that uses `Fp32MasterOptimizer` for BF16 weights."""

      def _build_optimizer(self, tx: Any) -> Any:
        orig_optimizer_cls = maxtext_engine.nnx.Optimizer
        maxtext_engine.nnx.Optimizer = fp32_optimizer_cls
        try:
          return super()._build_optimizer(tx)
        finally:
          maxtext_engine.nnx.Optimizer = orig_optimizer_cls

      def _init_state(self) -> None:
        orig_optimizer_cls = maxtext_engine.nnx.Optimizer
        maxtext_engine.nnx.Optimizer = fp32_optimizer_cls
        try:
          super()._init_state()
        finally:
          maxtext_engine.nnx.Optimizer = orig_optimizer_cls

    engine_cls = _Fp32MasterMaxTextTrainingEngine
  else:
    engine_cls = maxtext_engine.MaxTextTrainingEngine

  with mesh:
    engine = engine_cls(
        maxtext_config,
        mesh=mesh,
        wrap_with_tunix_adapter=wrap_with_tunix_adapter,
        tokenizer_pad_id=tokenizer_pad_id,
    )
  engine.checkpoint_dir = maxtext_config.checkpoint_dir

  # When `float32_gate_logits=True` and `weight_dtype=bfloat16`, both the
  # trainer and vLLM rollout models store gate/router/norm/GDN/logits_dense
  # weights in `float32` and all other weights in `bfloat16`. However,
  # `MaxTextToMaxTextConverter` defaults `target_dtype` to `config.weight_dtype`
  # ("bfloat16") and only exempts parameter paths containing "gate" or "router",
  # which downcasts `A_log`, `conv1d`, `dt_bias`, `norm`, and `logits_dense` to
  # `bfloat16` during target-free conversion and fails Raiden's `item_size`
  # preflight check. Clearing `_direct.target_dtype` preserves each parameter's
  # exact dtype during conversion.
  if (
      getattr(maxtext_config, "float32_gate_logits", False)
      and getattr(engine, "_weight_converter", None) is not None
  ):
    direct_converter = getattr(engine._weight_converter, "_direct", None)
    if direct_converter is not None:
      direct_converter.target_dtype = None

  if getattr(engine, "_weight_converter", None) is not None:
    try:
      from tunix.experimental.rollout import moe_fp8_utils  # pylint: disable=g-import-not-at-top

      moe_fp8_utils.patch_trainer_converter_moe_fp8(engine._weight_converter)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning("Failed to patch trainer converter for MoE FP8: %s", e)

  model_type = type(engine.model).__name__
  logging.info(
      "MaxText engine model: %s (pad_id=%d, fp32_master=%s)",
      model_type,
      tokenizer_pad_id,
      use_fp32_master,
  )
  if wrap_with_tunix_adapter and model_type != "TunixMaxTextAdapter":
    raise RuntimeError(
        f"Expected the engine's model to be TunixMaxTextAdapter, got {model_type}."
    )
  if log_shapes:
    log_param_shapes(engine.model)

  return engine
