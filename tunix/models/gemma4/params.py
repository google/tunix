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

"""Parameter mapping for Gemma 4."""

from __future__ import annotations

import flax
from flax import nnx
import jax
import jax.numpy as jnp


def map_from_upstream_checkpoint(
    params, model_type: str = 'gemma4', *, text_only: bool = True
):
  """Map from upstream checkpoint to our implementation."""
  new_params = {}
  flat_params = flax.traverse_util.flatten_dict(params)
  # Detect if MoE is enabled by checking if 'mlp2' exists in any key
  has_moe = any('mlp2' in k for k in flat_params.keys())

  for key_path, value in flat_params.items():
    if not key_path:
      continue
    # Handle both string paths and tuple paths if flatten_dict behaves differently or if we have mixed input.
    # Usually flatten_dict returns tuple of strings.
    if isinstance(key_path[0], str) and '/' in key_path[0]:
      # If it's a single string with slashes (unlikely for flatten_dict but possible in some formats)
      module_path = key_path[0].split('/')
      param_name = key_path[1] if len(key_path) > 1 else 'w'  # fallback
    else:
      module_path = key_path[:-1]
      param_name = key_path[-1]

    # Normalize module_path to remove leading 'transformer' if present
    if module_path and module_path[0] == 'transformer':
      module_path = module_path[1:]

    if not module_path:
      continue

    if module_path[0] == 'embedder':
      if len(module_path) == 1:
        if param_name == 'input_embedding':
          new_params[('embedder', 'input_embedding')] = value
        elif param_name == 'per_layer_embeddings':
          new_params[('embedder', 'per_layer_input_embedding')] = value
      elif len(module_path) > 1 and module_path[1] == 'per_layer_model_projection':
        new_params[('embedder', 'per_layer_model_projection', 'w')] = value
      elif len(module_path) > 1 and module_path[1] == 'per_layer_projection_norm':
        new_params[('embedder', 'per_layer_projection_norm', param_name)] = value
      else:
        new_params[('embedder', *module_path[1:], param_name)] = value
      continue

    if module_path[0] == 'final_norm':
      new_params[('final_norm', param_name)] = value
      continue

    if module_path[0].startswith('layer_'):
      layer_idx = int(module_path[0].removeprefix('layer_'))
      target_path = ('layers', layer_idx)

      if len(module_path) > 1 and (module_path[1] == 'mlp' or module_path[1] == 'mlp2'):
        if has_moe:
          if module_path[1] == 'mlp2':
            if param_name == 'gating_einsum':
              new_params[(*target_path, 'mlp', 'gate_proj', 'kernel')] = value[0].T
              new_params[(*target_path, 'mlp', 'up_proj', 'kernel')] = value[1].T
            elif param_name == 'linear':
              new_params[(*target_path, 'mlp', 'down_proj', 'kernel')] = value
            else:
              new_params[(*target_path, 'mlp', param_name)] = value
          elif module_path[1] == 'mlp':
            # Upstream mlp is MoE
            if len(module_path) > 2 and module_path[2] == 'router_logits':
              new_params[(*target_path, 'moe', 'router_logits')] = value
            elif param_name == 'w':
              if module_path[-1] == 'gating_einsum':
                new_params[(*target_path, 'moe', 'gating_einsum')] = value
              elif module_path[-1] == 'linear':
                new_params[(*target_path, 'moe', 'linear')] = value
            else:
              new_params[(*target_path, 'moe', module_path[-1])] = value
        else:
          if module_path[1] == 'mlp':
            if param_name == 'gating_einsum':
              # Upstream combines gate and up: (2, hidden_dim, embed_dim)
              new_params[(*target_path, 'mlp', 'gate_proj', 'kernel')] = value[0].T
              new_params[(*target_path, 'mlp', 'up_proj', 'kernel')] = value[1].T
            elif param_name == 'linear':
              new_params[(*target_path, 'mlp', 'down_proj', 'kernel')] = value
            else:
              new_params[(*target_path, 'mlp', param_name)] = value

      elif has_moe and len(module_path) > 1 and module_path[1] in ('pre_ffw2_norm', 'post_ffw2_norm', 'pre_ffw_norm', 'post_ffw1_norm'):
        if module_path[1] == 'pre_ffw2_norm':
          new_params[(*target_path, 'pre_ffw_norm', param_name)] = value
        elif module_path[1] == 'post_ffw2_norm':
          new_params[(*target_path, 'dense_post_ffw_norm', param_name)] = value
        elif module_path[1] == 'pre_ffw_norm':
          new_params[(*target_path, 'moe_pre_ffw_norm', param_name)] = value
        elif module_path[1] == 'post_ffw1_norm':
          new_params[(*target_path, 'moe_post_ffw_norm', param_name)] = value

      elif len(module_path) > 1 and module_path[1] == 'attn':

        # Map attention params
        if len(module_path) > 2 and module_path[2] in ('query_norm', 'key_norm'):
          new_params[(*target_path, 'attn', f'_{module_path[2]}', param_name)] = value
        else:
          new_params[(*target_path, *module_path[1:], param_name)] = value

      else:
        new_params[(*target_path, *module_path[1:], param_name)] = value
      continue

    # Fallback for direct copies
    new_params[tuple(list(module_path) + [param_name])] = value

  return flax.traverse_util.unflatten_dict(new_params)
