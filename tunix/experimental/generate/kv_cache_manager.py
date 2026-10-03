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

"""Top-level coordinator for a model's key-value caches.

Dispatches cache operations to underlying `SingleTypeKVCacheManager` instances
grouped by attention geometry (KV heads, head dimension, window size).

Per-Step Lifecycle:
    During an engine step, requests flow through three phases:
    1. Synchronization (`sync_request_state`): Persist computed tokens and trim
       out-of-window pages.
    2. Allocation (`get_computed_pages`, `allocate_slots`): Match cached
       prefixes for new requests and reserve pages for incoming tokens.
    3. Hardware Execution (`get_page_idxs`, `get_physical_pages`): Retrieve page
       mappings for kernel dispatch.

Use `release_request` upon completion, or `reset_kv_cache` to invalidate all state.
"""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import single_type_kv_cache_manager
import tunix.experimental.generate.tiered_page_pool as page_pool_lib
from tunix.generate import utils


@dataclasses.dataclass(kw_only=True)
class CacheConfig:
  """Raw configuration parameters for KV Cache allocation and sharding."""

  # The maximum number of bytes to allocate on each TPU device.
  max_tpu_bytes_per_device: int
  # The maximum number of bytes to allocate on the host CPU.
  max_cpu_bytes: int = 0
  # The number of tokens per page.
  page_size: int = 16
  # The data type of KV cache entries.
  dtype: jax.typing.DTypeLike
  # The mesh axis to shard pages across.
  dp_axis: str | None = None
  # The mesh axis to shard KV heads across.
  tp_axis: str | None = None
  # The number of data parallel replicas.
  dp_size: int = 1
  # The mesh to shard the KV caches over. Required if `dp_axis` or `tp_axis`
  # is set.
  mesh: jax.sharding.Mesh | None = None

  def __post_init__(self):
    positive_checks = {
        "page_size": self.page_size,
        "dp_size": self.dp_size,
        "max_tpu_bytes_per_device": self.max_tpu_bytes_per_device,
    }
    for field_name, value in positive_checks.items():
      if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}.")

    if self.max_cpu_bytes < 0:
      raise ValueError(
          f"max_cpu_bytes cannot be negative, got {self.max_cpu_bytes}."
      )

    if (self.dp_axis or self.tp_axis) and self.mesh is None:
      raise ValueError(
          "mesh is required when dp_axis or tp_axis is set, got "
          f"dp_axis={self.dp_axis!r}, tp_axis={self.tp_axis!r}. Pages are "
          "allocated outside any mesh context, so the mesh cannot be inferred."
      )


@dataclasses.dataclass(frozen=True, kw_only=True)
class CacheGeometry:
  """The shape and reach of one KV cache."""

  num_kv_heads: int
  head_dim: int
  # None means the layers reading this cache attend globally.
  window_size: int | None = None


def _derive_kv_geometry(
    config: CacheConfig,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[int, int, int]:
  """Derives the shape of a KV cache element."""
  kv_packing = utils.get_dtype_packing(config.dtype)
  if (2 * num_kv_heads) % kv_packing != 0:
    raise ValueError(
        f"2 * num_kv_heads ({2 * num_kv_heads}) must be divisible by "
        f"kv_packing ({kv_packing})."
    )

  packed_kv_dim = (2 * num_kv_heads) // kv_packing
  return (packed_kv_dim, kv_packing, head_dim)


def _derive_cache_sharding(
    config: CacheConfig,
) -> jax.sharding.Sharding | None:
  """Derives the cache sharding."""
  if not (config.dp_axis or config.tp_axis):
    return None

  if config.mesh is None:
    raise ValueError(
        "Mesh is required when dp_axis or tp_axis is set, but got None."
    )

  # The cache has shape:
  #   (num_pages, page_size, packed_kv_dim, kv_packing, head_dim)
  #   i.e. (num_pages, num_elements_per_page, *element_shape)
  # Pages are sharded across DP and KV elements are sharded across TP.
  sharding_spec = jax.sharding.PartitionSpec(
      config.dp_axis, None, config.tp_axis, None, None
  )
  return jax.sharding.NamedSharding(config.mesh, sharding_spec)


def _compute_page_limits(
    config: CacheConfig,
    element_shapes: Sequence[tuple[int, int, int]],
    sharding: jax.sharding.Sharding | None,
) -> tuple[int, int]:
  """Calculates the per-cache TPU and CPU page capacities.

  Each cache is assigned the same number of TPU and CPU pages.
  Args:
    config: Cache capacity and sharding configuration.
    element_shapes: The packed element shape of every cache, one entry each.
    sharding: The TPU sharding of every cache.

  Returns:
    A tuple of (TPU pages per cache, CPU pages per cache).
  """
  dtype_bytes = jnp.dtype(config.dtype).itemsize

  # Pages are sharded across DP replicas, so they are allocated in groups of
  # `dp_size`, one page per replica. Each device only stores its shard of a
  # group.
  group_shapes = [
      (config.dp_size, config.page_size, *shape) for shape in element_shapes
  ]
  if sharding is not None:
    group_shapes = [sharding.shard_shape(shape) for shape in group_shapes]
  bytes_per_device_per_group = dtype_bytes * sum(
      math.prod(shape) for shape in group_shapes
  )

  num_groups = config.max_tpu_bytes_per_device // bytes_per_device_per_group
  if num_groups <= 0:
    raise ValueError(
        "Cannot allocate 0 TPU pages. max_tpu_bytes_per_device="
        f"{config.max_tpu_bytes_per_device} is smaller than the "
        f"{bytes_per_device_per_group} bytes each device needs to hold one "
        f"page on each of {config.dp_size} DP replica(s)."
    )
  num_tpu_pages = num_groups * config.dp_size

  # CPU pages are not sharded, so the host stores every page whole.
  bytes_per_page = (
      dtype_bytes
      * config.page_size
      * sum(math.prod(shape) for shape in element_shapes)
  )
  num_cpu_pages = config.max_cpu_bytes // bytes_per_page
  return num_tpu_pages, num_cpu_pages


def _create_kv_cache_group_managers(
    config: CacheConfig,
    cache_geometries: Mapping[str, CacheGeometry],
) -> list[single_type_kv_cache_manager.SingleTypeKVCacheManager]:
  """Creates one `SingleTypeKVCacheManager` per unique cache geometry."""
  if not cache_geometries:
    raise ValueError("At least one cache must be configured.")

  caches_by_geometry: dict[CacheGeometry, list[str]] = (
      collections.defaultdict(list)
  )
  for cache, geometry in cache_geometries.items():
    caches_by_geometry[geometry].append(cache)

  kv_shapes: dict[CacheGeometry, tuple[int, int, int]] = {
      geometry: _derive_kv_geometry(
          config, geometry.num_kv_heads, geometry.head_dim
      )
      for geometry in caches_by_geometry
  }
  sharding = _derive_cache_sharding(config)
  num_tpu_pages, num_cpu_pages = _compute_page_limits(
      config=config,
      element_shapes=[kv_shapes[g] for g in cache_geometries.values()],
      sharding=sharding,
  )

  managers = []
  for geometry, caches in caches_by_geometry.items():
    page_pool_config = page_pool_lib.TieredPagePoolConfig(
        page_size=config.page_size,
        element_shape=kv_shapes[geometry],
        dtype=config.dtype,
        partition_keys=tuple(caches),
        num_tpu_pages=num_tpu_pages,
        num_cpu_pages=num_cpu_pages,
        sharding=sharding,
    )
    managers.append(
        single_type_kv_cache_manager.SingleTypeKVCacheManager(
            page_pool_config=page_pool_config,
            window_size=geometry.window_size,
        )
    )
  return managers


class KVCacheManager:
  """Manages a model's KV caches."""

  def __init__(
      self,
      config: CacheConfig,
      cache_geometries: Mapping[str, CacheGeometry],
  ):
    """Initializes the cache.

    Args:
      config: Cache configuration.
      cache_geometries: The geometry of each named cache.
    """
    self._cache_names = tuple(cache_geometries.keys())

    self._page_size = config.page_size
    self._kv_cache_group_managers = _create_kv_cache_group_managers(
        config, cache_geometries
    )

  @property
  def page_size(self) -> int:
    return self._page_size

  @property
  def cache_names(self) -> tuple[str, ...]:
    """Returns the names of all allocated caches."""
    return self._cache_names

  def get_physical_pages(self) -> dict[str, jax.Array | np.ndarray]:
    """Returns a mapping of cache name to physical TPU pages."""
    physical_pages: dict[str, jax.Array | np.ndarray] = {}
    for manager in self._kv_cache_group_managers:
      physical_pages.update(manager.get_physical_pages())
    return physical_pages

  def update_tpu_pool(self, new_pages: dict[str, Any]) -> None:
    """Updates the physical TPU pages."""
    for manager in self._kv_cache_group_managers:
      manager.update_tpu_pool(new_pages)
