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

"""KV Cache Manager coordinating across multiple layer groups."""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import math
import types
from typing import Any

import jax
import jax.numpy as jnp

from tunix.experimental.generate import kv_cache_coordinator
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import single_type_kv_cache_manager
from tunix.experimental.generate import utils
import tunix.experimental.generate.tiered_page_pool as page_pool_lib

Page = single_type_kv_cache_manager.Page


@dataclasses.dataclass(kw_only=True)
class CacheConfig:
  """Raw configuration parameters for KV Cache allocation and sharding."""

  # Capacity limits in bytes.
  max_tpu_bytes: int
  max_cpu_bytes: int = 0

  # Number of tokens per page.
  page_size: int = 16

  # Whether pages may be shared between requests with a common prefix. When
  # disabled, no tokens are hashed and no request ever sees a cache hit.
  enable_prefix_caching: bool = True

  # Data type for KV cache entries.
  dtype: jnp.dtype

  # Parallelism sharding configuration.
  dp_axis: str | None = None
  tp_axis: str | None = None
  dp_size: int = 1

  mesh: jax.sharding.Mesh | None = None

  def __post_init__(self):
    positive_checks = {
        "page_size": self.page_size,
        "dp_size": self.dp_size,
        "max_tpu_bytes": self.max_tpu_bytes,
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


def derive_kv_geometry(
    config: CacheConfig,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[int, int, int]:
  """Derives the shape of a KV cache page."""

  kv_packing = utils.get_dtype_packing(config.dtype)
  if (2 * num_kv_heads) % kv_packing != 0:
    raise ValueError(
        f"2 * num_kv_heads ({2 * num_kv_heads}) must be divisible by "
        f"kv_packing ({kv_packing})."
    )

  packed_kv_dim = (2 * num_kv_heads) // kv_packing
  return (packed_kv_dim, kv_packing, head_dim)


def derive_cache_sharding(
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
  #   (num_pages, page_size, packed_kv_dim, kv_packing, head_dim).
  # Pages are sharded across DP and KV elements are sharded across TP.
  sharding_spec = jax.sharding.PartitionSpec(
      config.dp_axis, None, config.tp_axis, None, None
  )
  sharding = jax.sharding.NamedSharding(config.mesh, sharding_spec)
  return sharding


def compute_page_limits(
    config: CacheConfig,
    element_shapes: Sequence[tuple[int, int, int]],
) -> tuple[int, int]:
  """Calculates the per-cache TPU and CPU page capacities.

  Each cache receives a byte budget proportional to its own per-page cost.
  That weight cancels when the budget is divided back into pages, so every
  cache ends up with the same page count. This is also what the caller wants
  operationally: a request needing `n` pages of context has the same index
  range available in every pool.

  Args:
    config: Cache capacity and sharding configuration.
    element_shapes: The packed element shape of every cache, one entry each.

  Returns:
    A tuple of (TPU pages per cache, CPU pages per cache).
  """
  dtype_bytes = jnp.dtype(config.dtype).itemsize

  # One page, counted across every cache.
  # Elements per token * bytes per element * tokens per page.
  total_bytes_per_page = sum(
      math.prod(shape) * dtype_bytes * config.page_size
      for shape in element_shapes
  )

  total_tpu_bytes_per_page = total_bytes_per_page * config.dp_size
  num_tpu_pages = config.max_tpu_bytes // total_tpu_bytes_per_page
  # Round down to the nearest multiple of dp_size.
  num_tpu_pages = (num_tpu_pages // config.dp_size) * config.dp_size
  if num_tpu_pages <= 0:
    raise ValueError(
        f"Cannot allocate 0 TPU pages. max_tpu_bytes={config.max_tpu_bytes} "
        f"is smaller than the minimum {total_tpu_bytes_per_page} bytes "
        f"required for one page across {config.dp_size} DP replica(s)."
    )

  num_cpu_pages = config.max_cpu_bytes // total_bytes_per_page
  return num_tpu_pages, num_cpu_pages


def setup_pools_and_coordinators(
    config: CacheConfig,
    cache_geometries: Mapping[str, CacheGeometry]
) -> kv_cache_coordinator.KVCacheCoordinator:
    """Initializes TieredPagePoolManager and SingleTypeKVCacheManagers."""
    if not cache_geometries:
      raise ValueError("At least one cache must be configured.")

    # Caches of identical geometry share a pool, so they also share an
    # allocator and one set of page indices.
    caches_by_geometry: dict[CacheGeometry, list[str]] = (
        collections.defaultdict(list)
    )
    for cache, geometry in cache_geometries.items():
      caches_by_geometry[geometry].append(cache)

    cache_kv_shapes: dict[CacheGeometry, tuple[int, int, int]] = {
        geometry: derive_kv_geometry(
            config, geometry.num_kv_heads, geometry.head_dim
        )
        for geometry in caches_by_geometry
    }
    cache_sharding = derive_cache_sharding(config)

    element_shapes = [
        cache_kv_shapes[geometry]
        for geometry in cache_geometries.values()
    ]
    num_tpu_pages, num_cpu_pages = compute_page_limits(
        config=config,
        element_shapes=element_shapes,
    )

    single_type_managers = []
    for geometry, caches in caches_by_geometry.items():
      element_shape = cache_kv_shapes[geometry]
      page_pool_config = page_pool_lib.TieredPagePoolConfig(
          page_size=config.page_size,
          element_shape=element_shape,
          dtype=config.dtype,
          partition_keys=tuple(caches),
          num_tpu_pages=num_tpu_pages,
          num_cpu_pages=num_cpu_pages,
          sharding=cache_sharding,
      )
      manager = single_type_kv_cache_manager.SingleTypeKVCacheManager(
          page_pool_config=page_pool_config,
          window_size=geometry.window_size,
      )
      single_type_managers.append(manager)

    return kv_cache_coordinator.KVCacheCoordinator(
        kv_cache_group_managers=single_type_managers
    )


class KVCacheManager:
  """A manager for a model's KV caches."""

  def __init__(
      self,
      config: CacheConfig,
      cache_geometries: Mapping[str, CacheGeometry]
  ):
    """Initializes the cache.

    Args:
      config: Capacity, page geometry and sharding for the cache.
      cache_geometries: The geometry of each named cache. Caches sharing a
        geometry are pooled together.
    """
    self._cache_geometries = dict(cache_geometries)

    self._page_size = config.page_size
    self._enable_prefix_caching = config.enable_prefix_caching
    self._request_to_prefix_hashes: dict[str, list[int]] = {}
    self._coordinator: kv_cache_coordinator.KVCacheCoordinator = (
        setup_pools_and_coordinators(
            config, cache_geometries
        )
    )

    # PERF-DEBUG: temporary prefix cache counters, read by LLMEngine.step to
    # report a hit rate. Delete along with the other PERF-DEBUG blocks.
    self.perf_prompt_tokens = 0
    self.perf_cached_tokens = 0

  @property
  def page_size(self) -> int:
    return self._page_size

  @property
  def caches(self) -> tuple[str, ...]:
    """Returns the names of all allocated caches."""
    return tuple(self._cache_geometries)

  @property
  def cache_geometries(self) -> Mapping[str, CacheGeometry]:
    """Returns the geometry of each allocated cache."""
    return types.MappingProxyType(self._cache_geometries)

  @property
  def null_computed_pages(self) -> tuple[tuple[Page | None, ...], ...]:
    return tuple(() for _ in range(self._coordinator.num_groups))

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

  def get_page_idxs(
      self,
      request_id: str,
  ) -> dict[str, tuple[int, ...]]:
    """Returns a mapping of layer ID to physical TPU page indices for the request.

    Args:
        request_id: The request identifier to get page indices for.

    Returns:
        A dictionary mapping each layer ID to its tuple of TPU page indices.
    """
    return self._coordinator.get_page_idxs(request_id)

  def get_physical_pages(self) -> dict[str, Any]:
    """Returns a mapping of layer ID to physical TPU pages across caches."""
    return self._coordinator.get_physical_pages()

  def sync_request_state(self, req: request_lib.Request) -> None:
    """Advances the cache's view of a request to its current token count.

    Registers the pages the request has filled since its last schedule, and
    releases the ones that have fallen out of a sliding window.

    Callers must run before calling `get_computed_pages` or
    `allocate_slots`.

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

    self._coordinator.sync_request_state(
        req.request_id,
        page_hashes,
        req.num_completed_tokens,
    )

  def get_computed_pages(
      self,
      req: request_lib.Request,
  ) -> tuple[int, tuple[tuple[Page | None, ...], ...]]:
    """Returns the computed (cached) pages for a request.

    Looks the request's prompt up against the pages other requests have
    already cached. `sync_request_state` must be called on a request before
    this to ensure it has registered its hashable pages.

    Args:
        req: The request to get the computed pages for.

    Returns:
        A tuple containing:
          - The number of computed pages.
          - Pages computed for the request grouped by KV cache groups.
    """
    if not self._enable_prefix_caching:
      return 0, self.null_computed_pages

    page_hashes = self._request_to_prefix_hashes.get(req.request_id, [])
    n_pages_hit, computed_pages = self._coordinator.find_longest_cache_hit(
        page_hashes
    )

    # PERF-DEBUG
    self.perf_prompt_tokens += len(req.token_ids)
    self.perf_cached_tokens += n_pages_hit * self._page_size

    return n_pages_hit, computed_pages

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
        new_computed_pages: Cached pages for the new computed tokens,
            grouped by KV cache groups.

    Returns:
        True if the allocation succeeded, False otherwise.
    """
    if num_new_tokens <= 0:
      return True

    computed_pages: Sequence[Sequence[Page | None]] = (
        self.null_computed_pages
        if new_computed_pages is None
        else new_computed_pages
    )

    can_fit = self._coordinator.has_sufficient_space(
        req.request_id,
        num_new_tokens,
        req.num_completed_tokens,
        computed_pages,
    )
    if not can_fit:
      return False

    self._coordinator.allocate_slots(
        req.request_id,
        num_new_tokens,
        req.num_completed_tokens,
        computed_pages,
    )
    return True

  def release_request(self, req: Any):
    """Releases pages for a finished request."""
    self._coordinator.release_request(req.request_id)
    self._request_to_prefix_hashes.pop(req.request_id, None)

  def update_tpu_pool(self, new_pages: dict[str, Any]):
    """Updates the physical page pool."""
    self._coordinator.update_tpu_pool(new_pages)

  def reset_kv_caches(self) -> None:
    """Resets the KV cache."""
    self._coordinator.reset_kv_caches()


