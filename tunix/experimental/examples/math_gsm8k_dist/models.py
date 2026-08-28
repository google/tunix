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

"""Model construction shared by the demo's trainer and rollout nodes.

Both nodes build the model here so the weight sync source and
destination expose identically named tensors.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
from typing import Any

from jax import numpy as jnp
from jax.sharding import Mesh
from tunix.models import automodel
from tunix.models import naming as naming_lib
from tunix.models.gemma import model as gemma_model_lib
from tunix.models.gemma import params_safetensors as gemma_params_lib
from tunix.models.qwen3 import model as qwen3_model_lib
from tunix.models.qwen3 import params as qwen3_params_lib


SAFETENSOR_MODEL_SOURCES = frozenset(("safetensors", "huggingface", "hf"))
MAXTEXT_MODEL_SOURCE = "maxtext"
MAXTEXT_VLLM_ARCHITECTURE = "MaxTextForCausalLM"


def add_model_source_args(parser: argparse.ArgumentParser) -> None:
  """Adds shared model-source arguments used by all distributed demo nodes."""
  parser.add_argument(
      "--model_source",
      type=str,
      default=os.getenv("MODEL_SOURCE", "safetensors"),
      choices=["safetensors", "huggingface", "hf", "maxtext"],
  )
  parser.add_argument(
      "--maxtext_dtype",
      type=str,
      default=os.getenv("MAXTEXT_DTYPE", "bfloat16"),
  )


def _gemma_config(model_name: str) -> gemma_model_lib.ModelConfig:
  normalized = model_name.lower().replace("_", "-")
  if "gemma-2-2b" in normalized or "gemma2-2b" in normalized:
    return gemma_model_lib.ModelConfig.gemma2_2b()
  if "gemma-2b" in normalized:
    return gemma_model_lib.ModelConfig.gemma_2b()
  raise ValueError(f"Unsupported gemma model_name: {model_name!r}")


def _qwen3_config(model_name: str) -> qwen3_model_lib.ModelConfig:
  normalized = model_name.lower().replace("_", "-")
  if "0.6b" in normalized or "0p6b" in normalized:
    config = qwen3_model_lib.ModelConfig.qwen3_0p6b()
  elif "1.7b" in normalized or "1p7b" in normalized:
    config = qwen3_model_lib.ModelConfig.qwen3_1p7b()
  elif "14b" in normalized:
    config = qwen3_model_lib.ModelConfig.qwen3_14b()
  elif "8b" in normalized:
    config = qwen3_model_lib.ModelConfig.qwen3_8b()
  elif "4b" in normalized:
    config = qwen3_model_lib.ModelConfig.qwen3_4b()
  elif "30b-a3b" in normalized:
    config = qwen3_model_lib.ModelConfig.qwen3_30b_a3b()
  elif "32b" in normalized:
    config = qwen3_model_lib.ModelConfig.qwen3_32b()
  else:
    raise ValueError(f"Unsupported qwen3 model_name: {model_name!r}")
  config.shd_config = qwen3_model_lib.ShardingConfig.get_default_sharding()
  config.remat_config = qwen3_model_lib.RematConfig.NONE
  config.use_flash_attention = True
  config.flash_attention_block_size = 256
  config.dtype = jnp.bfloat16
  config.param_dtype = jnp.float32
  return config


def is_maxtext_source(model_source: str) -> bool:
  return model_source.lower() == MAXTEXT_MODEL_SOURCE


def maxtext_model_kwargs(
    *,
    mesh: Mesh,
    max_prompt_length: int,
    max_response_length: int,
    model_name: str,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
  """Builds MaxText config overrides matching this demo's JAX mesh."""
  mesh_shape = dict(mesh.shape)
  kwargs: dict[str, Any] = {
      "ici_fsdp_parallelism": mesh_shape.get("fsdp", 1),
      "ici_tensor_parallelism": mesh_shape.get("tp", 1),
      "max_target_length": max_prompt_length + max_response_length,
      "max_prefill_predict_length": max_prompt_length,
      "dtype": dtype,
      "remat_policy": "none",
      "scan_layers": False,
      "skip_jax_distributed_system": True,
      "load_checkpoint_only_once": True,
      "use_standalone_converter": False,
      "checkpoint_storage_use_ocdbt": True,
      "checkpoint_storage_use_zarr3": True,
      "log_config": False,
  }
  if "qwen3" in model_name.lower().replace("_", "-"):
    try:
      kwargs["base_emb_dim"] = _qwen3_config(model_name).embed_dim
    except ValueError:
      pass
  if "qwen" in model_name.lower():
    kwargs["sparse_matmul"] = True
  return kwargs


def maxtext_kwargs_from_args(args: Any, mesh: Mesh) -> dict[str, Any]:
  """Builds MaxText overrides from a parsed worker-node argparse namespace."""
  return maxtext_model_kwargs(
      mesh=mesh,
      max_prompt_length=args.max_prompt_length,
      max_response_length=args.max_response_length,
      model_name=args.model_name,
      dtype=args.maxtext_dtype,
  )


def maxtext_vllm_additional_config(
    *,
    mesh: Mesh,
    max_prompt_length: int,
    max_response_length: int,
    model_name: str,
    model_dir: str,
    dtype: str = "bfloat16",
) -> dict[str, Any]:
  """Builds MaxText vLLM overrides matching this demo's rollout mesh."""
  mesh_shape = dict(mesh.shape)
  maxtext_config: dict[str, Any] = {
      "model_name": naming_lib.ModelNaming(model_name=model_name).model_name,
      "model_call_mode": "inference",
      "attention": "vllm_rpa",
      "allow_split_physical_axes": True,
      "log_config": False,
      "weight_dtype": dtype,
      "prefuse_moe_weights": True,
      "remat_policy": "none",
      "enable_dp_attention": False,
      "vllm_hf_overrides": {"architectures": [MAXTEXT_VLLM_ARCHITECTURE]},
      "ici_data_parallelism": mesh_shape.get("fsdp", 1),
      "ici_tensor_parallelism": mesh_shape.get("tp", 1),
      "max_target_length": max_prompt_length + max_response_length,
      "max_prefill_predict_length": max_prompt_length,
      "dtype": dtype,
      "skip_jax_distributed_system": True,
      "use_standalone_converter": False,
      "scan_layers": False,
      "checkpoint_storage_use_ocdbt": True,
      "checkpoint_storage_use_zarr3": True,
  }
  if model_dir:
    maxtext_config["load_parameters_path"] = model_dir
  if "qwen3" in model_name.lower().replace("_", "-"):
    try:
      maxtext_config["base_emb_dim"] = _qwen3_config(model_name).embed_dim
    except ValueError:
      pass
  if "qwen" in model_name.lower():
    maxtext_config["sparse_matmul"] = True
  return {"maxtext_config": maxtext_config}


def create_model(
    model_name: str,
    model_dir: str,
    mesh: Mesh,
    *,
    model_source: str = "safetensors",
    model_id: str = "",
    maxtext_kwargs: Mapping[str, Any] | None = None,
    dtype: jnp.dtype | None = None,
):
  """Builds the demo model on the given mesh.

  Args:
    model_name: Demo model selector, e.g. "gemma-2-2b" or "Qwen3-1.7B".
    model_dir: Directory holding safetensors shards, or a MaxText checkpoint
      path when model_source is "maxtext".
    mesh: Device mesh the parameters are sharded over.
    model_source: "safetensors"/"huggingface" for the native demo loaders, or
      "maxtext" for MaxText checkpoint loading.
    model_id: HF/config model identifier used by AutoModel for MaxText naming.
    maxtext_kwargs: Extra MaxText config overrides.
    dtype: Optional safetensors load dtype. The distributed GSM8K trainer uses
      fp32 actor weights to match `examples/math_gsm8k/qwen3_grpo_demo.py`.

  Returns:
    An nnx module ready for training or serving.
  """
  normalized_source = model_source.lower()
  if is_maxtext_source(normalized_source):
    model, _ = automodel.AutoModel.from_pretrained(
        model_name,
        mesh=mesh,
        model_source=automodel.ModelSource.MAXTEXT,
        model_path=model_dir,
        model_download_path=model_dir,
        **dict(maxtext_kwargs or {}),
    )
    return model

  if normalized_source not in SAFETENSOR_MODEL_SOURCES:
    raise ValueError(f"Unsupported model_source: {model_source!r}")

  normalized = model_name.lower().replace("_", "-")
  if "gemma" in normalized:
    return gemma_params_lib.create_model_from_safe_tensors(
        model_dir, _gemma_config(model_name), mesh=mesh
    )
  if "qwen3" in normalized:
    return qwen3_params_lib.create_model_from_safe_tensors(
        model_dir, _qwen3_config(model_name), mesh, dtype=dtype or jnp.bfloat16
    )
  raise ValueError(f"Unsupported demo model_name: {model_name!r}")
