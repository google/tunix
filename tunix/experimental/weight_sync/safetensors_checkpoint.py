# Copyright 2026 Google LLC
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

"""Utilities for file-backed weight sync artifacts in safetensors format."""

from __future__ import annotations

import os
from typing import Any

import flax
from flax import nnx
import jax
import numpy as np
from safetensors import numpy as safe_np
from tunix.experimental.weight_sync import weight_sync


def _state_to_pure_dict(state: Any) -> Any:
  if isinstance(state, nnx.State):
    return nnx.to_pure_dict(state)
  return state


def _axis_name(axis: Any) -> str:
  if axis is None:
    return ""
  if isinstance(axis, str):
    return axis
  return ",".join(axis)


def _tensor_metadata(
    name: str,
    arr: Any,
    layer_idx: int,
) -> weight_sync.TensorMetadata:
  shape = tuple(getattr(arr, "shape", ()))
  if not shape:
    raise ValueError(f"State leaf {name!r} has no shape; cannot export")
  sharding = getattr(arr, "sharding", None)
  spec = tuple(getattr(sharding, "spec", ()) or ())
  spec = (spec + (None,) * len(shape))[: len(shape)]
  try:
    local = sharding.shard_shape(shape)
    mesh_shape = tuple(g // l for g, l in zip(shape, local))
  except Exception:  # pylint: disable=broad-exception-caught
    mesh_shape = (1,) * len(shape)
  return weight_sync.TensorMetadata(
      name=name,
      shape=shape,
      mesh_shape=mesh_shape,
      layout=tuple(reversed(range(len(shape)))),
      item_size=np.dtype(arr.dtype).itemsize,
      layer_idx=layer_idx,
      sharding_spec=tuple(_axis_name(axis) for axis in spec),
  )


def build_work_unit_metadata(
    state: Any,
    job_name: str,
    *,
    artifact_path: str = "",
    job_replica_id: str = "",
) -> weight_sync.WorkUnitMetadata:
  """Builds transport-neutral manifest metadata for a state tree."""
  pure_state = _state_to_pure_dict(state)
  flat_state = flax.traverse_util.flatten_dict(pure_state)
  variables = tuple(
      _tensor_metadata(
          ".".join(str(part) for part in key),
          value,
          idx,
      )
      for idx, (key, value) in enumerate(flat_state.items())
  )
  return weight_sync.WorkUnitMetadata(
      unit=weight_sync.WorkUnitId(
          job_name=job_name,
          job_replica_id=job_replica_id,
      ),
      artifact_path=artifact_path,
      variables=variables,
  )


def source_artifact_path(sync_request: Any) -> str:
  """Returns the first prepared safetensors artifact carried by a sync request."""
  if sync_request is None:
    return ""
  source_metadata = getattr(sync_request, "source_metadata", ()) or ()
  for metadata in source_metadata:
    artifact_path = getattr(metadata, "artifact_path", "")
    if not artifact_path and isinstance(metadata, dict):
      artifact_path = str(metadata.get("artifact_path", ""))
    if artifact_path:
      return artifact_path
  return ""


def save_state_to_safetensors(state: Any, path: str) -> str:
  """Writes a matching state tree to a single safetensors file."""
  file_path = path
  if os.path.isdir(path) or not path.endswith(".safetensors"):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "model.safetensors")
  else:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

  pure_state = _state_to_pure_dict(state)
  flat_state = flax.traverse_util.flatten_dict(pure_state)
  tensors = {
      ".".join(str(part) for part in key): np.asarray(jax.device_get(value))
      for key, value in flat_state.items()
  }
  safe_np.save_file(tensors, file_path)
  return file_path


def load_state_from_safetensors(path: str, target_state: Any) -> Any:
  """Loads a safetensors artifact into a target-shaped state tree."""
  file_path = path
  if os.path.isdir(path):
    file_path = os.path.join(path, "model.safetensors")

  tensors = safe_np.load_file(file_path)
  pure_target = _state_to_pure_dict(target_state)

  def _load_leaf(tree_path, leaf):
    if leaf is None:
      return None
    key = ".".join(str(part.key if hasattr(part, "key") else part) for part in tree_path)
    loaded = tensors.get(key)
    if loaded is None:
      loaded = tensors.get(f"{key}.value")
    if loaded is None:
      return leaf
    if loaded.shape != getattr(leaf, "shape", None):
      raise ValueError(
          f"Shape mismatch for {key}: got {loaded.shape}, expected"
          f" {getattr(leaf, 'shape', None)}"
      )
    arr = jax.device_put(loaded)
    if hasattr(leaf, "dtype") and arr.dtype != leaf.dtype:
      arr = arr.astype(leaf.dtype)
    return arr

  restored = jax.tree_util.tree_map_with_path(
      _load_leaf,
      pure_target,
      is_leaf=lambda x: x is None,
  )
  return restored
