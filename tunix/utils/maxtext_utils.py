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

import logging
import os
from typing import Any


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
    *,
    base_num_kv_heads: int = 0,
    kv_tp_size: int = 0,
    moe_mlp_tp_size: int = 0,
    rollout_mesh_tp: int = 0,
    prefuse_moe_weights: bool = False,
    use_weight_converter: bool = True,
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
  # Checkpointing configs
  if checkpointing_options:
    argv.extend([
        "enable_checkpointing=True",
        f"checkpoint_period={checkpointing_options.save_interval_steps}",
        f"max_num_checkpoints_to_keep={checkpointing_options.max_to_keep}",
    ])
  elif load_parameters_path:
    argv.append("enable_checkpointing=True")
  else:
    argv.append("enable_checkpointing=False")
  argv.extend([
      "scan_layers=True",
      "convert_checkpoint_if_possible=False",
      "skip_jax_distributed_system=True",
      f"per_device_batch_size={per_device_batch_size}",
      f"gradient_accumulation_steps={gradient_accumulation_steps}",
      f"max_target_length={max_prompt_length + max_response_length}",
      "attention=dot_product",
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
          if effective_kv_heads
          else []
      ),
      f"ici_tensor_parallelism={mesh_tp}",
      f"ici_expert_parallelism={mesh_expert}",
      f"learning_rate={learning_rate}",
      f"warmup_steps_fraction={warmup_steps_fraction}",
      "dtype=bfloat16",
      "weight_dtype=bfloat16",
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
