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

from jax import numpy as jnp
from jax.sharding import Mesh
from tunix.models.gemma import model as gemma_model_lib
from tunix.models.gemma import params_safetensors as gemma_params_lib
from tunix.models.qwen3 import model as qwen3_model_lib
from tunix.models.qwen3 import params as qwen3_params_lib


def _gemma_config(model_name: str) -> gemma_model_lib.ModelConfig:
  normalized = model_name.lower().replace("_", "-")
  if "gemma-2-2b" in normalized or "gemma2-2b" in normalized:
    return gemma_model_lib.ModelConfig.gemma2_2b()
  if "gemma-2b" in normalized:
    return gemma_model_lib.ModelConfig.gemma_2b()
  raise ValueError(f"Unsupported gemma model_name: {model_name!r}")


def _qwen3_config(model_name: str) -> qwen3_model_lib.ModelConfig:
  normalized = model_name.lower().replace("_", "-")
  if "1.7b" in normalized or "1p7b" in normalized:
    config = qwen3_model_lib.ModelConfig.qwen3_1p7b()
  elif "32b" in normalized:
    config = qwen3_model_lib.ModelConfig.qwen3_32b()
  else:
    raise ValueError(f"Unsupported qwen3 model_name: {model_name!r}")
  config.shd_config = qwen3_model_lib.ShardingConfig.get_default_sharding()
  config.dtype = jnp.bfloat16
  config.param_dtype = jnp.float32
  return config


def create_model(model_name: str, model_dir: str, mesh: Mesh):
  """Builds the demo model on the given mesh.

  Args:
    model_name: Demo model selector, e.g. "gemma-2-2b" or "Qwen3-1.7B".
    model_dir: Directory holding the safetensors shards.
    mesh: Device mesh the parameters are sharded over.

  Returns:
    An nnx module ready for training or serving.
  """
  normalized = model_name.lower().replace("_", "-")
  if "gemma" in normalized:
    return gemma_params_lib.create_model_from_safe_tensors(
        model_dir, _gemma_config(model_name), mesh=mesh
    )
  if "qwen3" in normalized:
    return qwen3_params_lib.create_model_from_safe_tensors(
        model_dir, _qwen3_config(model_name), mesh, dtype=jnp.bfloat16
    )
  raise ValueError(f"Unsupported demo model_name: {model_name!r}")
