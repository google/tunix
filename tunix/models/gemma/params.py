# Copyright 2025 Google LLC
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

"""Utils for loading Gemma params."""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

Params = Mapping[str, Any]


def load_and_format_params(path: str) -> Params:
  """Loads parameters and formats them for compatibility."""
  params = _load_params(path)
  param_state = jax.tree.map(jnp.array, params)
  remapped_params = _param_remapper(param_state)
  nested_params = _unflatten_params(remapped_params)
  return nested_params


def load_metadata(path: str) -> Any | None:
  """Loads metadata from a checkpoint path."""
  checkpointer = ocp.StandardCheckpointer()
  metadata = checkpointer.metadata(path)
  return metadata


def _load_params(path: str) -> Params:
  """Loads parameters from a checkpoint path."""
  checkpointer = ocp.StandardCheckpointer()
  params = checkpointer.restore(path)
  return params


def _param_remapper(orig_params: Params) -> Params:
  """Remaps params to new module layout.

  This is needed here because the model definition  does not have a separate
  `mlp` module.

  Args:
    orig_params: original dict of parameters in Gemma format.

  Returns:
    dict of params with different names.
  """
  new_params = {}
  for k, v in orig_params.items():
    if 'mlp/' in k:
      layer_name, param = k.rsplit('/', maxsplit=1)
      if layer_name not in new_params:
        new_params[layer_name] = {}
      if 'w' in v:
        new_params[layer_name][param] = v['w']
    else:
      new_params[k] = v
  return new_params


def _unflatten_params(params: Params) -> Params:
  """Unflatten keys of params as nested dicts."""
  nested_params = {}
  for path, param in params.items():
    *path, leaf = path.split('/')
    subdict = nested_params
    for key in path:
      subdict = subdict.setdefault(key, {})
    subdict[leaf] = param
  return nested_params
