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
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import single_type_kv_cache_manager
import tunix.experimental.generate.tiered_page_pool as page_pool_lib
from tunix.generate import utils


Page = single_type_kv_cache_manager.Page


@dataclasses.dataclass(kw_only=True)
class CacheConfig:
  """Raw configuration parameters for KV Cache allocation and sharding."""

  # The maximum number of bytes to allocate on each device.
  max_device_bytes: int
  # The maximum number of bytes to allocate on the host.
  max_host_bytes: int = 0
  # The number of tokens per page.
  page_size: int = 16
  # Whether pages may be shared between requests with a common prefix.
  enable_prefix_caching: bool = True
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
        "max_device_bytes": self.max_device_bytes,
    }
    for field_name, value in positive_checks.items():
      if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}.")

    if self.max_host_bytes < 0:
      raise ValueError(
          f"max_host_bytes cannot be negative, got {self.max_host_bytes}."
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
  """Calculates the per-cache device and host page capacities.

  Each cache is assigned the same number of device and host pages.
  Args:
    config: Cache capacity and sharding configuration.
    element_shapes: The packed element shape of every cache, one entry each.
    sharding: The device sharding of every cache.

  Returns:
    A tuple of (device pages per cache, host pages per cache).
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

  num_groups = config.max_device_bytes // bytes_per_device_per_group
  if num_groups <= 0:
    raise ValueError(
        "Cannot allocate 0 device pages. max_device_bytes="
        f"{config.max_device_bytes} is smaller than the "
        f"{bytes_per_device_per_group} bytes each device needs to hold one "
        f"page on each of {config.dp_size} DP replica(s)."
    )
  num_device_pages = num_groups * config.dp_size

  # Host pages are not sharded, so the host stores every page whole.
  bytes_per_page = (
      dtype_bytes
      * config.page_size
      * sum(math.prod(shape) for shape in element_shapes)
  )
  num_host_pages = config.max_host_bytes // bytes_per_page
  return num_device_pages, num_host_pages


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
  num_device_pages, num_host_pages = _compute_page_limits(
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
        num_device_pages=num_device_pages,
        num_host_pages=num_host_pages,
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
    self._enable_prefix_caching = config.enable_prefix_caching
    self._request_to_prefix_hashes: dict[str, list[int]] = {}
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

  @property
  def _null_computed_pages(self) -> tuple[tuple[Page | None, ...], ...]:
    """Returns the computed pages of a request with no prefix match."""
    return tuple(() for _ in self._kv_cache_group_managers)

  def _chunk_and_hash(
      self, tokens: list[int], start_hash: int = 0
  ) -> list[int]:
    """Chunks tokens into pages and hashes them."""
    hashes = []
    parent_hash = start_hash
    aligned_n_tokens = (len(tokens) // self._page_size) * self._page_size
    # TODO: Hashing should be deterministic across multiple hosts.
    for i in range(0, aligned_n_tokens, self._page_size):
      chunk = tuple(tokens[i : i + self._page_size])
      parent_hash = hash((parent_hash, chunk))
      hashes.append(parent_hash)
    return hashes

  def _find_longest_cache_hit(
      self,
      page_hashes: Sequence[int],
  ) -> tuple[int, tuple[tuple[Page | None, ...], ...]]:
    """Finds the longest prefix of `page_hashes` cached in every group.

    Groups may hit different prefix lengths. Each round truncates the hashes to
    the shortest hit, and queries the groups again, until they agree. Groups
    must be requeried since a local attention group may miss on a shorter
    prefix, as the window covers different pages.

    Args:
      page_hashes: The prefix hashes of the request's full pages.

    Returns:
      A tuple of
        - The number of computed pages, common to all groups.
        - The computed pages of each group. Entries are `None` for pages
          outside a group's active window.
    """
    while True:
      hits = [
          manager.find_longest_cache_hit(page_hashes)
          for manager in self._kv_cache_group_managers
      ]
      min_n_hit = min(len(h) for h in hits)
      max_n_hit = max(len(h) for h in hits)
      if min_n_hit == max_n_hit:
        prefix_match_len = min_n_hit
        return prefix_match_len, tuple(tuple(h) for h in hits)

      page_hashes = page_hashes[:min_n_hit]

  def get_page_idxs(
      self,
      request_id: str,
  ) -> dict[str, tuple[int, ...]]:
    """Returns a mapping of cache name to device page indices for the request.

    Args:
        request_id: The request identifier to get page indices for.

    Returns:
        A dictionary mapping each cache name to its tuple of device page
        indices. Entries are -1 for pages outside the cache's active window.
    """
    page_idxs: dict[str, tuple[int, ...]] = {}
    for manager in self._kv_cache_group_managers:
      group_page_idxs = tuple(manager.get_page_idxs(request_id))
      for cache in manager.cache_names:
        page_idxs[cache] = group_page_idxs
    return page_idxs

  def get_physical_pages(self) -> dict[str, jax.Array | np.ndarray]:
    """Returns a mapping of cache name to physical device pages."""
    physical_pages: dict[str, jax.Array | np.ndarray] = {}
    for manager in self._kv_cache_group_managers:
      physical_pages.update(manager.get_physical_pages())
    return physical_pages

  def sync_request_state(self, req: request_lib.Request) -> None:
    """Advances the cache's view of a request to its current token count.

    Registers the pages the request has filled since its last schedule, and
    releases the ones that have fallen out of a sliding window.

    Must be called before `get_computed_pages` or `allocate_slots`.

    Args:
        req: The request to sync the internal state for.
    """
    page_hashes: list[int] = []
    if self._enable_prefix_caching:
      page_hashes = self._request_to_prefix_hashes.setdefault(
          req.request_id, []
      )
      n_hashed_tokens = len(page_hashes) * self._page_size

      # Withhold the last token from hashing so that it is not hit in the
      # cache. Otherwise, 0-len prefills may occur if all tokens are stored in
      # the cache.
      unhashed_tokens = req.token_ids[n_hashed_tokens:-1]
      last_page_hash = page_hashes[-1] if page_hashes else 0
      page_hashes.extend(self._chunk_and_hash(unhashed_tokens, last_page_hash))

    for manager in self._kv_cache_group_managers:
      manager.sync_request_state(
          req.request_id, page_hashes, req.num_completed_tokens
      )

  def get_computed_pages(
      self,
      req: request_lib.Request,
  ) -> tuple[int, tuple[tuple[Page | None, ...], ...]]:
    """Returns the computed (prefix matched) pages for a request.

    Looks the request's prompt up against the pages other requests have
    already cached. `sync_request_state` must be called on a request before
    this to ensure it has registered its hashable pages.

    Args:
        req: The request to get the computed pages for.

    Returns:
        A tuple containing:
          - The number of computed pages.
          - The computed pages, to be passed to `allocate_slots`.
    """
    if not self._enable_prefix_caching:
      return 0, self._null_computed_pages

    page_hashes = self._request_to_prefix_hashes.get(req.request_id, [])
    return self._find_longest_cache_hit(page_hashes)

  def allocate_slots(
      self,
      req: request_lib.Request,
      num_new_tokens: int,
      new_computed_pages: Sequence[Sequence[Page | None]] | None = None,
  ) -> bool:
    """Allocates token slots for a request.

    Args:
        req: The request to prepare for the next engine step.
        num_new_tokens: The number of new tokens to be allocated and computed.
        new_computed_pages: The computed pages, as returned by
            `get_computed_pages`. `None` if there are no computed pages.

    Returns:
        True if the allocation succeeded, False if there is insufficient space.
        On failure, no slots are allocated.
    """
    if new_computed_pages is None:
      new_computed_pages = self._null_computed_pages

    if len(new_computed_pages) != len(self._kv_cache_group_managers):
      raise ValueError(
          "new_computed_pages must be the computed pages returned by "
          "get_computed_pages."
      )

    if num_new_tokens <= 0:
      return True

    managers_and_pages = list(
        zip(self._kv_cache_group_managers, new_computed_pages)
    )
    if not all(
        manager.has_sufficient_space(
            req.request_id, num_new_tokens, req.num_completed_tokens, pages
        )
        for manager, pages in managers_and_pages
    ):
      return False

    for manager, pages in managers_and_pages:
      manager.allocate_slots(
          req.request_id, num_new_tokens, req.num_completed_tokens, pages
      )
    return True

  def release_request(self, req: request_lib.Request) -> None:
    """Releases pages for a finished request."""
    for manager in self._kv_cache_group_managers:
      manager.release_request(req.request_id)
    self._request_to_prefix_hashes.pop(req.request_id, None)

  def update_device_pool(self, new_pages: dict[str, Any]) -> None:
    """Updates the physical device pages."""
    for manager in self._kv_cache_group_managers:
      manager.update_device_pool(new_pages)

  def reset_kv_caches(self) -> None:
    """Releases all requests and frees all pages."""
    for manager in self._kv_cache_group_managers:
      manager.reset_kv_caches()
    self._request_to_prefix_hashes.clear()
