# Copyright 2026 The Tunix Authors.
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

"""Utilities for the v2 sampler."""

from collections.abc import Sequence
import math

import jax
import jax.numpy as jnp
from tunix.experimental.generate import configs

_GIB = 1 << 30


def cdiv(a: int, b: int) -> int:
  """Returns the ceiling of `a / b` for integers."""
  return -(-a // b)


def next_power_of_2(x: int) -> int:
  """Returns the next power of 2 that is not smaller than `x`."""
  x = int(x)
  if x <= 1:
    return 1
  return 1 << (x - 1).bit_length()


def get_dtype_packing(dtype: jax.typing.DTypeLike) -> int:
  """Returns the number of elements of `dtype` packed into 32 bits."""
  return max(1, 32 // jax.dtypes.itemsize_bits(dtype))


@jax.jit
def jit_scatter_last_column(
    target: jax.Array,
    source: jax.Array,
    target_idxs: jax.Array,
) -> jax.Array:
  """Writes the last column of `source` into `target` at `target_idxs`.

  Out-of-bounds indices in `target_idxs` are dropped.

  Args:
    target: 1D destination array to update.
    source: 2D array whose last column (`source[:, -1]`) supplies the values.
    target_idxs: 1D index array mapping each row of `source` into `target`.

  Returns:
    A copy of `target` with `source[:, -1]` written at `target_idxs`.
  """
  return target.at[target_idxs].set(source[:, -1], mode="drop")


def derive_kv_geometry(
    config: configs.CacheConfig,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[int, int, int]:
  """Derives the packed shape of a KV cache element.

  When `num_kv_heads < config.tp_size`, the KV head count is padded up to
  `config.tp_size` so that each tensor-parallel shard holds one replicated KV
  head.

  Args:
    config: Cache configuration specifying dtype and tensor-parallel sharding.
    num_kv_heads: Number of KV attention heads before tensor-parallel padding.
    head_dim: Dimension of each attention head.

  Returns:
    A `(packed_kv_dim, kv_packing, head_dim)` tuple representing the packed
    element shape of a single token in the KV cache.

  Raises:
    ValueError: If `num_kv_heads` and `config.tp_size` are not divisible by one
      another, or if `2 * (num_kv_heads // tp_size)` (after padding) is not
      divisible by `kv_packing`.
  """
  tp_size = config.tp_size
  if num_kv_heads < tp_size:
    if tp_size % num_kv_heads != 0:
      raise ValueError(
          f"tp_size ({tp_size}) must be divisible by "
          f"num_kv_heads ({num_kv_heads})."
      )
    num_kv_heads = tp_size
  elif num_kv_heads % tp_size != 0:
    raise ValueError(
        f"num_kv_heads ({num_kv_heads}) must be divisible by "
        f"tp_size ({tp_size})."
    )

  kv_packing = get_dtype_packing(config.dtype)
  if (2 * (num_kv_heads // tp_size)) % kv_packing != 0:
    raise ValueError(
        f"2 * (num_kv_heads // tp_size) ({2 * (num_kv_heads // tp_size)}) must "
        f"be divisible by kv_packing ({kv_packing})."
    )

  packed_kv_dim = (2 * num_kv_heads) // kv_packing
  return (packed_kv_dim, kv_packing, head_dim)


def derive_cache_sharding(
    config: configs.CacheConfig,
) -> jax.sharding.NamedSharding:
  """Derives the device sharding for a KV cache page pool.

  Args:
    config: Cache configuration specifying the mesh, `dp_axis`, and `tp_axis`.

  Returns:
    A `NamedSharding` partitioning pages (`axis=0`) across `config.dp_axis` and
    packed KV heads (`axis=2`) across `config.tp_axis`.
  """
  # The cache has shape:
  #   (num_pages, page_size, packed_kv_dim, kv_packing, head_dim)
  #   i.e. (num_pages, num_elements_per_page, *element_shape)
  # Pages are sharded across DP and KV elements are sharded across TP.
  sharding_spec = jax.sharding.PartitionSpec(
      config.dp_axis, None, config.tp_axis, None, None
  )
  return jax.sharding.NamedSharding(config.mesh, sharding_spec)


def compute_page_limits(
    config: configs.CacheConfig,
    element_shapes: Sequence[tuple[int, int, int]],
    sharding: jax.sharding.Sharding,
) -> tuple[int, int]:
  """Calculates the per-cache device and host page capacities.

  Each cache is assigned the same number of device and host pages.

  Args:
    config: Cache capacity and sharding configuration.
    element_shapes: The packed element shape of every cache, one entry each.
    sharding: The device sharding of every cache.

  Returns:
    A `(num_device_pages, num_host_pages)` tuple giving the device and host
    page capacities per cache.

  Raises:
    ValueError: If `element_shapes` is empty, or if `config.max_device_size_gib`
      cannot hold at least one page per data-parallel replica across all caches.
  """
  if not element_shapes:
    raise ValueError("element_shapes must not be empty.")

  dtype_bytes = jnp.dtype(config.dtype).itemsize

  # Pages are sharded across DP replicas, so they are allocated in groups of
  # `dp_size`, one page per replica. Each device only stores its shard of a
  # group.
  group_shapes = [
      sharding.shard_shape((config.dp_size, config.page_size, *shape))
      for shape in element_shapes
  ]
  bytes_per_device_per_group = dtype_bytes * sum(
      math.prod(shape) for shape in group_shapes
  )

  max_device_bytes = math.floor(config.max_device_size_gib * _GIB)
  num_groups = max_device_bytes // bytes_per_device_per_group
  if num_groups <= 0:
    raise ValueError(
        "Cannot allocate 0 device pages. max_device_size_gib="
        f"{config.max_device_size_gib} ({max_device_bytes} bytes) is smaller "
        f"than the {bytes_per_device_per_group} bytes each device needs to "
        f"hold one page on each of {config.dp_size} DP replica(s)."
    )
  num_device_pages = num_groups * config.dp_size

  # Host pages are not sharded, so the host stores every page whole.
  bytes_per_page = (
      dtype_bytes
      * config.page_size
      * sum(math.prod(shape) for shape in element_shapes)
  )
  max_host_bytes = math.floor(config.max_host_size_gib * _GIB)
  num_host_pages = max_host_bytes // bytes_per_page
  return num_device_pages, num_host_pages
