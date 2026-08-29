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
  from maxtext.src.maxtext.configs import pyconfig  # pylint: disable=g-import-not-at-top
  from maxtext.src.maxtext.training_engine import maxtext_engine  # pylint: disable=g-import-not-at-top
  from maxtext.src.maxtext.utils import maxtext_utils  # pylint: disable=g-import-not-at-top
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
) -> Any:
  """Builds the MaxText HyperParameters the training engine runs on."""
  pyconfig, _, _ = maxtext_modules()

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

  output_dir = base_output_directory or "/tmp/maxtext"
  enable_checkpointing = bool(load_parameters_path)
  argv = [
      "maxtext_trainer",
      base_yml,
      f"model_name={model_name}",
      f"run_name={worker_id or 'tunix_maxtext'}",
      f"base_output_directory={output_dir}",
      f"enable_checkpointing={enable_checkpointing}",
  ]
  if load_parameters_path:
    argv.append(f"load_parameters_path={load_parameters_path}")
  argv.extend([
      "scan_layers=True",
      "convert_checkpoint_if_possible=False",
      "skip_jax_distributed_system=True",
      f"per_device_batch_size={per_device_batch_size}",
      "gradient_accumulation_steps=1",
      f"max_target_length={max_prompt_length + max_response_length}",
      "attention=dot_product",
      "use_tokamax_gmm=true",
      "use_gmm_v2=true",
      f"ici_fsdp_parallelism={mesh_fsdp}",
      *(
          [f"padded_base_moe_mlp_dim={padded_moe_mlp_dim}"]
          if padded_moe_mlp_dim
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
