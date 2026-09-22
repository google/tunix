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


"""Utility functions for sampler."""

import math
from typing import Any
import jax
import jax.numpy as jnp


def cdiv(a: int, b: int) -> int:
  """Computes the ceiling division of a by b."""
  return (a + b - 1) // b


def derive_sharding(
    model_config: Any,
    transformer_state: Any,
) -> tuple[str | None, str | None, int, jax.sharding.Mesh | None]:
  """Derives the KV cache sharding axes, data parallel size and mesh.

  Args:
    model_config: The transformer's config, optionally carrying a `shd_config`.
    transformer_state: Transformer parameters, used to recover the mesh the
      model is sharded over.

  Returns:
    A tuple of (dp_axis, tp_axis, dp_size, mesh). `mesh` is None when the model
    has no sharding.
  """
  shd_config = getattr(model_config, 'shd_config', None)
  if shd_config is None:
    return None, None, 1, None

  dp_axis = shd_config.act_btd[0]
  tp_axis = shd_config.act_btnh[2]

  param_0 = jax.tree.leaves(transformer_state)[0]
  sharding = getattr(param_0, 'sharding', None)
  mesh = getattr(sharding, 'mesh', None)

  dp_size = 1
  if dp_axis and mesh is not None:
    dp_size = mesh.shape.get(dp_axis, 1)

  return dp_axis, tp_axis, dp_size, mesh


def model_dtype(
    model_config: Any,
    transformer_state: Any,
) -> jnp.dtype:
  """Returns the dtype a transformer's values are in.

  Args:
    model_config: The transformer's config, which names the dtype when the
      model was built for one in particular.
    transformer_state: Transformer parameters, fallen back on when the config
      names no dtype.

  Returns:
    The dtype callers should compute and cache values in.
  """
  dtype = getattr(model_config, 'dtype', None)
  if dtype is not None:
    return dtype
  return jax.tree.leaves(transformer_state)[0].dtype


def shard(x: jnp.ndarray, s: tuple[str | None, ...]):
  mesh = jax.interpreters.pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return x
  return jax.lax.with_sharding_constraint(
      x, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*s))
  )


def _remove_dp_spec(
    spec: jax.sharding.PartitionSpec,
) -> jax.sharding.PartitionSpec:
  dp_axis = ['dp', 'fsdp']
  new_spec = tuple(None if axis in dp_axis else axis for axis in spec)
  return jax.sharding.PartitionSpec(*new_spec)


def _put_on_target_device(
    tensor: jax.Array, target_tensor: jax.Array
) -> jax.Array:
  if hasattr(target_tensor, 'sharding') and target_tensor.sharding is not None:
    sharding = target_tensor.sharding
    if isinstance(sharding, jax.sharding.NamedSharding):
      safe_spec = _remove_dp_spec(sharding.spec)
      target_sharding = jax.sharding.NamedSharding(sharding.mesh, safe_spec)
      return jax.device_put(tensor, target_sharding)
    elif isinstance(sharding, jax.sharding.SingleDeviceSharding):
      return jax.device_put(tensor, sharding)

  if hasattr(target_tensor, 'devices') and len(target_tensor.devices()) > 0:
    return jax.device_put(tensor, list(target_tensor.devices())[0])

  return tensor


def copy_physical_pages(
    src_pages: jax.Array,
    dst_pages: jax.Array,
    src_idxs: jax.Array,
    dst_idxs: jax.Array,
) -> jax.Array:
  if len(src_idxs) == 0:
    return dst_pages

  src_slice = src_pages[src_idxs]
  src_slice = _put_on_target_device(src_slice, dst_pages)

  dst_pages = dst_pages.at[dst_idxs].set(src_slice)
  return dst_pages


def get_dtype_packing(dtype: jnp.dtype) -> int:
  """Returns the packing factor for a given data type."""
  dtype = jnp.dtype(dtype)
  bits = dtype.itemsize * 8
  return max(1, 32 // bits)


def calculate_pages_for_capacity(
    max_bytes: int,
    logical_sharding: tuple[str | None, ...],
    page_size: int,
    page_subshape: tuple[int, ...],
    dp_size: int,
    dtype: jnp.dtype,
    partition_keys: tuple[str, ...],
) -> int:
  """Calculates the number of pages that fit within the specified byte capacity."""
  if max_bytes <= 0:
    return 0
  num_partitions = len(partition_keys)
  if num_partitions == 0:
    return 0
  elements_per_page = page_size * math.prod(page_subshape)
  bytes_per_page = elements_per_page * jnp.dtype(dtype).itemsize * num_partitions
  if bytes_per_page <= 0:
    return 0
  effective_dp_size = dp_size if (logical_sharding and logical_sharding[0] == 'dp') else 1
  return (max_bytes * effective_dp_size) // bytes_per_page

