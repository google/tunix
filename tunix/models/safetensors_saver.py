# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for saving models with merged LoRA weights in safetensors format."""

import os
import shutil
from typing import Any, Callable
from etils import epath
from flax import nnx
import jax.numpy as jnp
import safetensors.numpy as safe_np


def join_path(path) -> str:
  return '.'.join([str(field) for field in path])


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: Any,
    rank: int,
    alpha: float,
    state_key_transform_fn: Callable[[str], str],
    custom_layer_extractor_fn: (
        Callable[[dict[str, list[Any]]], dict[str, list[Any]]] | None
    ) = None,
):
  """Saves a model with LoRA weights merged in safetensors format.

  This is a generic function that can be used for any model architecture.
  Model-specific logic is provided via callback functions.

  Args:
    local_model_path: Path to the base model safetensors checkpoint directory.
    output_dir: Directory where the merged model will be saved.
    lora_model: Model instance with LoRA weights.
    rank: LoRA rank used during training.
    alpha: LoRA alpha used during training.
    state_key_transform_fn: Function that transforms model layer paths to
      safetensors state dict keys.
    custom_layer_extractor_fn: Optional function that updates the extracted LoRA
      layer dictionary; it should accept the current layer dict and return a
      dict of the new/updated LoRA layers' names as strings to a tuple of the
      corresponding lora pair.
  """

  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
  os.makedirs(output_dir)

  lora_layers: dict[str, list[Any]] = {}

  for path, value in nnx.iter_graph(lora_model):
    if isinstance(value, nnx.LoRAParam):
      path_str = join_path(path[:-1])
      if path_str in lora_layers:
        assert (
            'lora_b' in path[-1]
        ), f'Expect second LoRAParam to be lora_b, got {path[-1]}'
        lora_layers[path_str].append(value)
      else:
        assert (
            'lora_a' in path[-1]
        ), f'Expect first LoRAParam to be lora_a, got {path[-1]}'
        lora_layers[path_str] = [value]

  if custom_layer_extractor_fn:
    lora_layers = custom_layer_extractor_fn(lora_layers)

  # Load base model state
  files = list(epath.Path(local_model_path).expanduser().glob('*.safetensors'))

  if not files:
    raise ValueError(f'No safetensors found in {local_model_path}')

  base_state = {}
  for f in files:
    base_state.update(safe_np.load_file(f))

  # Apply LoRA deltas
  for path, (lora_a, lora_b) in lora_layers.items():
    state_key = state_key_transform_fn(path)
    assert state_key in base_state, (
        f'LoRA layer {path} not found in base model state dict'
        f' {base_state.keys()}'
    )

    lora_a_val = jnp.asarray(getattr(lora_a, 'value', lora_a))
    lora_b_val = jnp.asarray(getattr(lora_b, 'value', lora_b))

    # Reshape 3D tensors to 2D if necessary
    if lora_a_val.ndim == 3:
      d0, d1, d2 = lora_a_val.shape
      lora_a_val = lora_a_val.reshape(d0 * d1, d2)
    if lora_b_val.ndim == 3:
      d0, d1, d2 = lora_b_val.shape
      lora_b_val = lora_b_val.reshape(d0, d1 * d2)

    # Compute and apply LoRA delta
    combined_lora = (lora_a_val @ lora_b_val) * (alpha / rank)
    base_state[state_key] += combined_lora.T.astype(base_state[state_key].dtype)

  # Save merged model
  safetensors_path = os.path.join(output_dir, 'model.safetensors')
  safe_np.save_file(base_state, safetensors_path)

  # Copy non-safetensors files (config, tokenizer, etc.)
  for filename in os.listdir(local_model_path):
    if not filename.endswith('.safetensors'):
      src = os.path.join(local_model_path, filename)
      if os.path.isfile(src):
        dst = os.path.join(output_dir, filename)
        shutil.copy(src, dst)
