"""A tiered memory cache manager for device and host memory.

This module provides a `TieredPagePoolManager`.

The `TieredPagePoolManager` abstracts device and host memory logic
by wrapping around an underlying _PartitionedPagePool. Callers interact
exclusively via stable logical page IDs, while the manager tracks physical
indices and handles swapping (load/offload) across memory tiers.

Caller should initialize a TieredPagePoolManager via
TieredPagePoolConfig.create_manager().
A single TieredPagePoolManager coordinates multiple KV caches that share
identical memory geometry (page size, element shape, and data type), with each
cache mapping to a dedicated partition within the underlying pools.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import functools
import importlib
from typing import Generic, Protocol, TypeVar

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
import numpy as np

# The array type of a pool's pages: `jax.Array` on device, `np.ndarray` on host.
_ArrayT = TypeVar("_ArrayT", jax.Array, np.ndarray)


class _RaidenFuture(Protocol):
  """Protocol for a tpu-raiden async transfer future."""

  def Await(self) -> None:  # pylint: disable=invalid-name
    ...


class _RaidenKVCacheManager(Protocol):
  """Protocol for tpu-raiden's JAX KVCacheManager."""

  def d2h(
      self,
      src_offsets: list[int],
      dst_offsets: list[int],
      copy_sizes: list[int] | None = None,
  ) -> _RaidenFuture:
    ...

  def h2d(
      self,
      src_offsets: list[int],
      dst_offsets: list[int],
      copy_sizes: list[int] | None = None,
  ) -> _RaidenFuture:
    ...


def _create_raiden_kv_cache_manager(
    kv_caches: list[jax.Array],
    num_host_pages: int,
) -> _RaidenKVCacheManager:
  """Instantiates tpu-raiden's KVCacheManager for the given device KV caches."""
  jax.block_until_ready(kv_caches)
  raiden_kv_lib = importlib.import_module(
      "tpu_sync.api.jax.kv_cache_manager"
  )
  return raiden_kv_lib.KVCacheManager(
      kv_caches=kv_caches,
      local_control_port=0,
      host_blocks_to_allocate=num_host_pages,
      unsafe_skip_buffer_lock=True,
  )


@dataclasses.dataclass(kw_only=True)
class _PartitionedPagePool(Generic[_ArrayT]):
  """A partitioned page pool."""

  # A mapping of partition names to the pages for that partition.
  partition_pages: dict[str, _ArrayT]
  # A list of available page indices across all partitions.
  _available_page_indices: list[int] = dataclasses.field(
      default_factory=list, init=False
  )
  # A set of allocated pages. This is used to validate and prevent double-free
  # operations.
  _in_use: set[int] = dataclasses.field(default_factory=set, init=False)

  # Store the expected shape/dtype per partition to validate updates.
  _expected_shape: tuple[int, ...] = dataclasses.field(init=False)
  _expected_dtype: jnp.dtype = dataclasses.field(init=False)

  def __post_init__(self):
    if not self.partition_pages:
      raise ValueError("Partition pages cannot be empty.")

    # Derive pool spec from the first partition
    first_arr = next(iter(self.partition_pages.values()))
    self._expected_shape = first_arr.shape
    self._expected_dtype = first_arr.dtype

    # Validate that all partitions conform to the same spec
    for k, v in self.partition_pages.items():
      if v.shape != self._expected_shape or v.dtype != self._expected_dtype:
        raise ValueError(
            f"Partition '{k}' does not match pool spec. Expected "
            f"shape={self._expected_shape}, dtype={self._expected_dtype}; "
            f"got shape={v.shape}, dtype={v.dtype}."
        )

    n_pages = self._expected_shape[0]
    self._available_page_indices = list(range(n_pages))
    self._in_use = set()

  def allocate(self, num_pages: int) -> list[int]:
    """Allocates `num_pages` pages for each partition."""
    if num_pages < 0:
      raise ValueError(
          f"Cannot allocate a negative number of pages: {num_pages}."
      )

    if num_pages > self.num_free_pages:
      raise ValueError(
          f"Cannot allocate {num_pages} pages, "
          f"only {self.num_free_pages} available."
      )

    if num_pages == 0:
      return []

    indices = self._available_page_indices[-num_pages:]
    del self._available_page_indices[-num_pages:]

    self._in_use.update(indices)

    return indices

  def free(self, indices: Sequence[int]):
    """Frees pages with the given indices for each partition."""
    indices_set = set(indices)
    if len(indices_set) != len(indices):
      raise ValueError("Cannot free duplicate page indices.")

    if len(indices_set - self._in_use) > 0:
      raise ValueError(
          f"Cannot free pages {indices_set - self._in_use}. "
          "These pages are not in use."
      )

    for idx in indices:
      self._in_use.remove(idx)
    self._available_page_indices.extend(indices)

  def update_pages(
      self,
      new_pages: Mapping[str, _ArrayT],
  ):
    """Updates the underlying pages for each partition with `new_pages`."""
    for k, new_arr in new_pages.items():
      # Skip partitions not belonging to this pool. Callers broadcast new_pages
      # containing partitions for all pools.
      if k not in self.partition_pages:
        continue

      if (
          new_arr.shape != self._expected_shape
          or new_arr.dtype != self._expected_dtype
      ):
        raise ValueError(
            f"Updated partition '{k}' does not match pool spec. Expected "
            f"shape={self._expected_shape}, dtype={self._expected_dtype}; "
            f"got shape={new_arr.shape}, dtype={new_arr.dtype}."
        )

      self.partition_pages[k] = new_arr

  @property
  def num_pages(self) -> int:
    return self._expected_shape[0]

  @property
  def num_free_pages(self) -> int:
    return len(self._available_page_indices)


@dataclasses.dataclass(frozen=True, kw_only=True)
class TieredPagePoolConfig:
  """Configuration for tiered page pool."""

  # The number of elements in a page.
  page_size: int
  # The shape of an individual element in a page.
  element_shape: tuple[int, ...] = ()
  # The data type of the elements in a page.
  dtype: DTypeLike
  # The names of the pool partitions (e.g. layer1, layer2).
  partition_keys: tuple[str, ...]
  # The number of device pages to allocate.
  num_device_pages: int
  # The number of host pages to allocate.
  num_host_pages: int = 0
  # The device sharding of the page pool tensor.
  # (Page dim, elements dim, element shape dim0, element shape dim1, ...)
  sharding: jax.sharding.Sharding | None = None
  # Whether to use tpu-raiden for host-device page transfers.
  use_raiden: bool = False

  def __post_init__(self):
    if self.num_device_pages <= 0:
      raise ValueError(
          f"num_device_pages must be positive, got {self.num_device_pages}."
      )
    if self.num_host_pages < 0:
      raise ValueError(
          f"num_host_pages cannot be negative, got {self.num_host_pages}."
      )
    if not self.partition_keys:
      raise ValueError("partition_keys cannot be empty.")

    page_shape = self._page_shape()
    if not page_shape:
      raise ValueError("page_shape cannot be empty.")

    for dim in page_shape:
      if dim <= 0:
        raise ValueError(
            f"All dimensions of page_shape must be positive, got {dim} in"
            f" {page_shape}."
        )

  def _page_shape(self, num_pages: int | None = None) -> tuple[int, ...]:
    if num_pages is None:
      num_pages = self.num_device_pages
    return (num_pages, self.page_size, *self.element_shape)

  def _make_device_pool(self) -> _PartitionedPagePool[jax.Array]:
    """Creates the device page pool."""
    page_shape = self._page_shape(self.num_device_pages)
    if self.sharding is not None:
      init_fn = jax.jit(
          lambda: jnp.zeros(page_shape, dtype=self.dtype),
          out_shardings=self.sharding,
      )
      pages = {k: init_fn() for k in self.partition_keys}
    else:
      pages = {
          k: jnp.zeros(page_shape, dtype=self.dtype)
          for k in self.partition_keys
      }
    return _PartitionedPagePool(partition_pages=pages)

  def _make_host_pool(self) -> _PartitionedPagePool[np.ndarray]:
    """Creates the host page pool."""
    page_shape = (
        (self.num_host_pages,)
        if self.use_raiden
        else self._page_shape(self.num_host_pages)
    )
    return _PartitionedPagePool(
        partition_pages={
            k: np.zeros(page_shape, dtype=self.dtype)
            for k in self.partition_keys
        }
    )

  def create_manager(self) -> "TieredPagePoolManager":
    """Initializes a TieredPagePoolManager with the given configuration."""
    return TieredPagePoolManager(
        device_pool=self._make_device_pool(),
        host_pool=self._make_host_pool() if self.num_host_pages > 0 else None,
        use_raiden=self.use_raiden,
    )


@functools.partial(jax.jit, donate_argnames=("device_pages", "slices"))
def _scatter_device_pages(
    device_pages: dict[str, jax.Array],
    indices: jax.Array,
    slices: dict[str, jax.Array],
) -> dict[str, jax.Array]:
  """Scatters pages into the device page pool.

  To optimize memory and prevent XLA from allocating redundant copies, the
  `device_pages` and `slices` buffers are donated. The function is jitted so
  that these scattered updates are executed concurrently across all device
  partitions.

  Args:
    device_pages: The current device page pool.
    indices: The indices of the pages to scatter.
    slices: The slices of the pages to scatter.

  Returns:
    The updated device page pool.
  """
  return {k: device_pages[k].at[indices].set(slices[k]) for k in device_pages}


@jax.jit
def _get_device_slices(
    device_pages: dict[str, jax.Array],
    indices: jax.Array,
) -> dict[str, jax.Array]:
  """Returns the slices of the device pages for the given indices.

  The function is jitted to ensure these slices are executed concurrently across
  all device partitions.

  Args:
    device_pages: The current device page pool.
    indices: The indices of the pages to get slices for.

  Returns:
    The slices of the device pages for the given indices.
  """
  return {layer: device_pages[layer][indices] for layer in device_pages}


def _pad_indices(indices: Sequence[int], max_length: int) -> list[int]:
  """Pads `indices` to the next power of 2, capped at `max_length`.

  Transfers are padded so that the jitted `_scatter_device_pages` and
  `_get_device_slices` only compile O(log N) shapes. Padded entries repeat the
  first index, so they read or write the same page with the same values.

  Args:
    indices: The page indices to pad. Must be non-empty.
    max_length: The maximum padded length.

  Returns:
    The padded page indices.
  """
  padded_length = min(1 << (len(indices) - 1).bit_length(), max_length)
  return [*indices, *[indices[0]] * (padded_length - len(indices))]


class TieredPagePoolManager:
  """Manager for tiered device/host memory."""

  def __init__(
      self,
      device_pool: _PartitionedPagePool[jax.Array],
      # None if host offloading is disabled.
      host_pool: _PartitionedPagePool[np.ndarray] | None,
      use_raiden: bool = False,
  ):
    self._device_pool = device_pool
    self._host_pool = host_pool

    self._next_page_id: int = 0
    self._page_id_to_idx: dict[int, int] = {}
    self._page_location: dict[int, str] = {}

    # The type parameters are not checked at runtime, so check the arrays.
    for k, arr in self._device_pool.partition_pages.items():
      if not isinstance(arr, jax.Array):
        raise ValueError(
            f"Device pool partition '{k}' must be jax.Array, got {type(arr)}."
        )

    if self._host_pool:
      for k, arr in self._host_pool.partition_pages.items():
        if not isinstance(arr, np.ndarray):
          raise ValueError(
              f"Host pool partition '{k}' must be np.ndarray, got {type(arr)}."
          )

    self._raiden_mgr: _RaidenKVCacheManager | None = (
        _create_raiden_kv_cache_manager(
            list(self._device_pool.partition_pages.values()),
            self._host_pool.num_pages,
        )
        if use_raiden and self._host_pool is not None
        else None
    )

  @property
  def num_device_pages(self) -> int:
    return self._device_pool.num_pages

  @property
  def num_free_device_pages(self) -> int:
    return self._device_pool.num_free_pages

  @property
  def num_free_host_pages(self) -> int:
    if self._host_pool:
      return self._host_pool.num_free_pages
    return 0

  @property
  def physical_device_pages(self) -> dict[str, jax.Array]:
    """Returns the underlying device page arrays."""
    return self._device_pool.partition_pages

  def page_location(self, page_id: int) -> str | None:
    return self._page_location.get(page_id)

  def page_idx(self, page_id: int) -> int | None:
    return self._page_id_to_idx.get(page_id)

  def allocate_device_pages(self, num_pages: int) -> list[int]:
    """Allocate logical device pages."""
    if num_pages < 0:
      raise ValueError("Cannot allocate a negative number of pages.")

    if num_pages == 0:
      return []

    if num_pages > self.num_free_device_pages:
      raise ValueError(
          f"Cannot allocate {num_pages} device pages, "
          f"only {self.num_free_device_pages} available."
      )

    allocated_ids = []
    phys_indices = self._device_pool.allocate(num_pages)

    for phys_idx in phys_indices:
      pid = self._next_page_id
      self._next_page_id += 1
      self._page_id_to_idx[pid] = phys_idx
      self._page_location[pid] = "device"

      allocated_ids.append(pid)

    return allocated_ids

  def update_device_pool(self, new_pages: Mapping[str, jax.Array]) -> None:
    """Updates the underlying device pool partition pages with new pages."""
    self._device_pool.update_pages(new_pages)

  @property
  def _device_sharding(self) -> jax.sharding.Sharding:
    first_layer_pages = next(iter(self._device_pool.partition_pages.values()))
    return first_layer_pages.sharding

  @property
  def _transferred_sharding(self) -> jax.sharding.Sharding:
    """Returns the target sharding for transferring to device."""
    device_sharding = self._device_sharding

    # Replicate the page dimension across all devices, so that pages are
    # scattered without cross-device communication.
    if isinstance(device_sharding, jax.sharding.NamedSharding):
      slice_spec = jax.sharding.PartitionSpec(None, *device_sharding.spec[1:])
      return jax.sharding.NamedSharding(device_sharding.mesh, slice_spec)

    return device_sharding

  def load(self, page_ids: Sequence[int]) -> None:
    """Transfers logical pages from host to device."""
    # TODO(yatlas): device_get only handles process local sharding.
    # Support for cross-host sharding needs to be added.
    if not page_ids:
      return

    if self._host_pool is None:
      raise ValueError(
          "Cannot load pages from host to device, host pool is not initialized."
      )

    if len(page_ids) > self.num_free_device_pages:
      raise ValueError(
          f"Cannot load {len(page_ids)} pages, "
          f"only {self.num_free_device_pages} available."
      )

    if len(set(page_ids)) != len(page_ids):
      raise ValueError("Cannot load duplicate pages.")

    for pid in page_ids:
      if self._page_location.get(pid) != "host":
        raise ValueError(
            f"Page ID {pid} is not on host "
            f"(location: {self._page_location.get(pid)})."
        )

    host_idxs = [self._page_id_to_idx[pid] for pid in page_ids]
    device_idxs = self._device_pool.allocate(len(page_ids))

    if self._raiden_mgr is not None:
      self._raiden_mgr.h2d(
          src_offsets=host_idxs,
          dst_offsets=device_idxs,
      ).Await()
    else:
      max_length = self._device_pool.num_pages
      padded_host_idxs = _pad_indices(host_idxs, max_length)
      padded_device_idxs = _pad_indices(device_idxs, max_length)

      # Gather all the pages that need to be transferred to device.
      host_slices = {
          k: self._host_pool.partition_pages[k][padded_host_idxs]
          for k in self._device_pool.partition_pages
      }

      # Transfer pages to the device
      device_slices = jax.device_put(host_slices, self._transferred_sharding)

      # Scatter pages to the device partitions.
      device_indices_arr = jnp.array(padded_device_idxs, dtype=jnp.int32)
      device_partitions = self._device_pool.partition_pages

      # Use a jit-compiled function to avoid replicating page pools when
      # scattering pages across all device partitions.
      updated_device_pages = _scatter_device_pages(
          device_partitions, device_indices_arr, device_slices
      )
      self._device_pool.update_pages(updated_device_pages)

    # Update page state
    self._host_pool.free(host_idxs)
    for pid, p_idx in zip(page_ids, device_idxs):
      self._page_id_to_idx[pid] = p_idx
      self._page_location[pid] = "device"

  def offload(self, page_ids: Sequence[int]) -> None:
    """Moves logical pages from device to host transferring only active ones."""
    if not page_ids:
      return

    if self._host_pool is None:
      raise ValueError(
          "Cannot offload pages to host, host pool is not initialized."
      )

    if len(page_ids) > self.num_free_host_pages:
      raise ValueError(
          f"Cannot offload {len(page_ids)} pages, "
          f"only {self.num_free_host_pages} available."
      )

    if len(set(page_ids)) != len(page_ids):
      raise ValueError("Cannot offload duplicate pages.")

    for pid in page_ids:
      if self._page_location.get(pid) != "device":
        raise ValueError(
            f"Page ID {pid} is not on device "
            f"(location: {self._page_location.get(pid)})."
        )

    physical_device_idxs = [self._page_id_to_idx[pid] for pid in page_ids]
    physical_host_idxs = self._host_pool.allocate(len(page_ids))

    if self._raiden_mgr is not None:
      self._raiden_mgr.d2h(
          src_offsets=physical_device_idxs,
          dst_offsets=physical_host_idxs,
      ).Await()
    else:
      padded_device_idxs = _pad_indices(
          physical_device_idxs, self._device_pool.num_pages
      )
      device_indices_arr = jnp.array(padded_device_idxs, dtype=jnp.int32)

      # Use a jit-compiled function here to concurrently gather slices from all
      # device partitions.
      # TODO(yatlas): Make this not blocking, and overlap page loading with
      # the engine step.
      device_slices = _get_device_slices(
          self._device_pool.partition_pages, device_indices_arr
      )
      host_slices = jax.device_get(device_slices)
      for layer, host_slice in host_slices.items():
        self._host_pool.partition_pages[layer][physical_host_idxs] = host_slice[
            : len(page_ids)
        ]

    self._device_pool.free(physical_device_idxs)
    for pid, p_idx in zip(page_ids, physical_host_idxs):
      self._page_id_to_idx[pid] = p_idx
      self._page_location[pid] = "host"

  def free(self, page_ids: Sequence[int]) -> None:
    """Releases physical allocations in device_pool or host_pool and removes logical IDs."""
    if not page_ids:
      return

    if len(set(page_ids)) != len(page_ids):
      raise ValueError("Cannot free duplicate pages.")

    for pid in page_ids:
      if pid not in self._page_location or pid not in self._page_id_to_idx:
        raise ValueError(f"Attempting to free page {pid} which is not in use.")

    host_idxs_to_free = []
    device_idxs_to_free = []

    for pid in page_ids:
      loc = self._page_location[pid]
      if loc == "host":
        host_idxs_to_free.append(self._page_id_to_idx[pid])
      elif loc == "device":
        device_idxs_to_free.append(self._page_id_to_idx[pid])

    if host_idxs_to_free and self._host_pool:
      self._host_pool.free(host_idxs_to_free)
    if device_idxs_to_free and self._device_pool:
      self._device_pool.free(device_idxs_to_free)

    for pid in page_ids:
      del self._page_location[pid]
      del self._page_id_to_idx[pid]
