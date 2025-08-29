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

import jax
import jax.sharding as shd
import numpy as np


def get_sharding(x, mesh, pspec):
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
