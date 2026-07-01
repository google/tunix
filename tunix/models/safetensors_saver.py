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
import re
from flax import nnx
import jax.numpy as jnp
import safetensors.numpy as safe_np
from safetensors import safe_open
import json

def join_path(path) -> str:
  return '.'.join([str(field) for field in path])

def load_base_state(local_model_path): # for sharded check points such as gemma-3-4b...
    single_file = os.path.join(local_model_path, "model.safetensors")
    index_file = os.path.join(local_model_path, "model.safetensors.index.json")

    # Case 1: single file
    if os.path.exists(single_file):
        return safe_np.load_file(single_file)

    # Case 2: sharded checkpoint
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            index_data = json.load(f)

        tensors = {}
        shard_files = set(index_data["weight_map"].values())

        for shard in shard_files:
            shard_path = os.path.join(local_model_path, shard)
            with safe_open(shard_path, framework="np") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k)

        return tensors

    raise FileNotFoundError("No safetensors found")


def map_tunix_to_hf_key(state_key: str) -> str:
    m = re.match(r"layers\.(\d+)\.(.*)", state_key)
    if not m:
        return state_key

    layer_id = m.group(1)
    suffix = m.group(2)

    prefix = f"language_model.model.layers.{layer_id}"

    mapping = {
        "attn.q_einsum": f"{prefix}.self_attn.q_proj.weight",
        "attn.k_einsum": f"{prefix}.self_attn.k_proj.weight",
        "attn.v_einsum": f"{prefix}.self_attn.v_proj.weight",
        "attn.attn_vec_einsum": f"{prefix}.self_attn.o_proj.weight",
        "attn.o_einsum": f"{prefix}.self_attn.o_proj.weight",
        "mlp.gate_proj": f"{prefix}.mlp.gate_proj.weight",
        "mlp.up_proj": f"{prefix}.mlp.up_proj.weight",
        "mlp.down_proj": f"{prefix}.mlp.down_proj.weight",
    }

    return mapping.get(suffix, state_key)


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
    transpose_rules: dict[str, tuple[int, ...]] | None = None,
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
  # base_state = safe_np.load_file(local_model_path + '/model.safetensors') original
    base_state = load_base_state(local_model_path)
    

  # Apply LoRA deltas
  for path, (lora_a, lora_b) in lora_layers.items():
    #tunix original as of Apr 1st 2026
    '''state_key = state_key_transform_fn(path)
    assert state_key in base_state, (
        f'LoRA layer {path} not found in base model state dict'
        f' {base_state.keys()}'
    )
    '''
    state_key = state_key_transform_fn(path)
    hf_key = map_tunix_to_hf_key(state_key)

    if hf_key not in base_state:
        print(f"[WARNING] Skipping unmatched key:")
        print(f"  Tunix key: {state_key}")
        print(f"  HF key: {hf_key}")
        continue

    state_key = hf_key

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
    if transpose_rules:
      for t_key, rule in transpose_rules.items():
        if t_key in state_key:
          combined_lora = combined_lora.transpose(rule)
          break

    base_state[state_key] += combined_lora.astype(base_state[state_key].dtype)

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
