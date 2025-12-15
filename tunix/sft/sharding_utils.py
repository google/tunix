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

"""Utilities for sharding tensors."""

from typing import Tuple

import jax
from jax.interpreters import pxla
import jax.sharding as shd
import numpy as np


def shard_input(
    input_data: jax.Array, data_sharding_axis: Tuple[str, ...]
) -> jax.Array:
  """Shards the input data across the available devices.

  Args:
    input_data: The input data to be sharded, expected to be a TrainingInput
      dataclass.
    data_sharding_axis: The sharding axis for the input data, e.g. ("fsdp",).

  Returns:
    The sharded TrainingInput.
  """
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty:
    return input_data

  pspec = shd.PartitionSpec(*data_sharding_axis)
  # Per-leaf sharding that is robust to existing global Arrays.
  def _shard_leaf(x):
    # Already a JAX Array
    if isinstance(x, jax.Array):
      sh = getattr(x, "sharding", None)
      # No-op if already on the target mesh/spec
      if (
          hasattr(sh, "mesh")
          and sh.mesh == mesh
          and hasattr(sh, "spec")
          and sh.spec == pspec
      ):
        return x
      # Global but not fully addressable: annotate sharding and let JIT reshard
      if not getattr(x, "is_fully_addressable", True):
        return jax.lax.with_sharding_constraint(x, pspec)
      # Fully addressable JAX Array (process-local): assemble a global Array
      with jax.transfer_guard("allow"):
        return jax.make_array_from_process_local_data(
            get_sharding(x, mesh=mesh, pspec=pspec), jax.device_get(x)
        )
    # NumPy arrays (process-local): assemble a global Array
    if isinstance(x, np.ndarray):
      with jax.transfer_guard("allow"):
        return jax.make_array_from_process_local_data(
            get_sharding(x, mesh=mesh, pspec=pspec), x
        )
    # Other leaves: return as-is
    return x

  return jax.tree.map(_shard_leaf, input_data)


def get_sharding(x: jax.Array, mesh: shd.Mesh, pspec: shd.PartitionSpec):
  """Get a sharding for an tensor given a mesh and partition spec."""
  # Only shard arrays with rank > 0.
  if not isinstance(x, (np.ndarray, jax.Array)) or x.ndim == 0:
    return shd.NamedSharding(mesh, shd.PartitionSpec())  # Replicated

  # Don't shard if rank is not sufficient.
  if x.ndim < len(pspec):
    return shd.NamedSharding(mesh, shd.PartitionSpec())  # Replicated

  # Check for divisibility for all sharded axes.
  for i, axis_name in enumerate(pspec):
    if axis_name is not None:
      axis_names = axis_name if isinstance(axis_name, tuple) else (axis_name,)
      for name in axis_names:
        axis_size = mesh.shape[name]
        if x.shape[i] % axis_size != 0:
          # Replicate if not evenly divisible.
          return shd.NamedSharding(mesh, shd.PartitionSpec())
  return shd.NamedSharding(mesh, pspec)
