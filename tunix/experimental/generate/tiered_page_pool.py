"""A tiered memory cache manager for TPU and CPU memory.

This module acts as the interface for logical page management, abstracting away
the allocation, tracking, and transfer of physical pages across devices.

It is used by the KV cache coordinator to track and manage logical pages for
requests.

Key components:
  TieredPagePoolManager: Provides an interface for page pools, exposing logical
    page IDs instead of physical indices. It handles the allocation, tracking,
    and transfer of low-level physical page indices across devices.
  TieredPagePoolConfig: Defines the layout, sharding, and capacity of the
    underlying physical page pools. It is used to initialize the page pools.
"""

from collections.abc import Sequence
import dataclasses
import functools
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(kw_only=True)
class _PagePool:
  """A pool of pages."""

  # A mapping of partition names to the pages for that partition.
  partition_pages: dict[str, jax.Array | np.ndarray]
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
    """Allocates pages in the pool."""
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

  def free(self, indices: Sequence[int]) -> None:
    """Frees pages in the pool."""
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
      new_pages: dict[str, jax.Array | np.ndarray],
  ) -> None:
    """Updates the underlying page pool partition pages with new pages."""
    for k, new_arr in new_pages.items():
      # Skip partitions that are not in the TPU pool. `new_pages` may contain
      # partitions for other page pools.
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
  dtype: jnp.dtype
  # The names of the pool partitions (e.g. layer1, layer2).
  partition_keys: tuple[str, ...]
  # The number of TPU pages to allocate.
  num_tpu_pages: int
  # The number of CPU pages to allocate.
  num_cpu_pages: int = 0
  # The TPU sharding of the page pool tensor.
  # (Page dim, elements dim, element shape dim0, element shape dim1, ...)
  page_sharding: jax.sharding.PartitionSpec | None = None

  def __post_init__(self):
    if self.num_tpu_pages <= 0:
      raise ValueError(
          f"num_tpu_pages must be positive, got {self.num_tpu_pages}."
      )
    if self.num_cpu_pages < 0:
      raise ValueError(
          f"num_cpu_pages cannot be negative, got {self.num_cpu_pages}."
      )
    if not self.partition_keys:
      raise ValueError("partition_keys cannot be empty.")

    page_shape = self.page_shape()
    if not page_shape:
      raise ValueError("page_shape cannot be empty.")

    for dim in page_shape:
      if dim <= 0:
        raise ValueError(
            f"All dimensions of page_shape must be positive, got {dim} in"
            f" {page_shape}."
        )

  def page_shape(self, num_pages: int | None = None) -> tuple[int, ...]:
    if num_pages is None:
      num_pages = self.num_tpu_pages
    return (num_pages, self.page_size, *self.element_shape)

  def _make_pool(
      self,
      num_pages: int,
      sharding: jax.sharding.PartitionSpec | None = None,
      is_cpu: bool = False,
  ) -> _PagePool:
    """Creates a page pool."""
    if is_cpu and sharding is not None:
      raise ValueError("Cannot shard pages on CPU.")

    pages_dict = {}
    page_shape = self.page_shape(num_pages)

    if sharding is not None:
      init_sharded_fn = jax.jit(
          lambda: jnp.zeros(page_shape, dtype=self.dtype),
          out_shardings=sharding,
      )
      for k in self.partition_keys:
        pages_dict[k] = init_sharded_fn()
    elif is_cpu:
      for k in self.partition_keys:
        pages_dict[k] = np.zeros(page_shape, dtype=self.dtype)
    else:
      for k in self.partition_keys:
        pages_dict[k] = jnp.zeros(page_shape, dtype=self.dtype)

    return _PagePool(
        partition_pages=pages_dict,
    )

  def init(self) -> tuple[_PagePool, _PagePool | None]:
    """Initializes physical page tensors for TPU and CPU."""

    tpu_pool = self._make_pool(
        num_pages=self.num_tpu_pages, sharding=self.page_sharding
    )

    cpu_pool = (
        self._make_pool(num_pages=self.num_cpu_pages, is_cpu=True)
        if self.num_cpu_pages > 0
        else None
    )

    return (tpu_pool, cpu_pool)


@functools.partial(jax.jit, donate_argnames=("tpu_pages", "slices"))
def _scatter_tpu_pages(
    tpu_pages: dict[str, jax.Array],
    indices: jax.Array,
    slices: dict[str, jax.Array],
) -> dict[str, jax.Array]:
  """Scatters pages into the TPU page pool.

  To optimize memory and prevent XLA from allocating redundant copies, the
  `tpu_pages` and `slices` buffers are donated. The function is jitted so that
  these scattered updates are executed concurrently across all TPU partitions.

  Args:
    tpu_pages: The current TPU page pool.
    indices: The indices of the pages to scatter.
    slices: The slices of the pages to scatter.

  Returns:
    The updated TPU page pool.
  """
  return {k: tpu_pages[k].at[indices].set(slices[k]) for k in tpu_pages}


@jax.jit
def _get_tpu_slices(
    tpu_pages: dict[str, jax.Array],
    indices: jax.Array,
) -> dict[str, jax.Array]:
  """Returns the slices of the TPU pages for the given indices.

  The function is jitted to ensure these slices are executed concurrently across
  all TPU partitions.

  Args:
    tpu_pages: The current TPU page pool.
    indices: The indices of the pages to get slices for.

  Returns:
    The slices of the TPU pages for the given indices.
  """
  return {layer: tpu_pages[layer][indices] for layer in tpu_pages}


class TieredPagePoolManager:
  """Manager for tiered TPU/CPU memory."""

  def __init__(
      self,
      tiered_config: TieredPagePoolConfig,
      tpu_pool: _PagePool,
      cpu_pool: _PagePool | None,  # CPU pool is optional.
  ):
    self.config = tiered_config
    self.tpu_pool = tpu_pool
    self.cpu_pool = cpu_pool
    self.page_size = tiered_config.page_size

    self._next_page_id: int = 0
    self._page_id_to_idx: dict[int, int] = {}
    self._page_location: dict[int, str] = {}

  @property
  def num_free_tpu_pages(self) -> int:
    return self.tpu_pool.num_free_pages

  @property
  def num_free_cpu_pages(self) -> int:
    if self.cpu_pool:
      return self.cpu_pool.num_free_pages
    return 0

  def get_page_location(self, page_id: int) -> str | None:
    return self._page_location.get(page_id)

  def get_page_idx(self, page_id: int) -> int | None:
    return self._page_id_to_idx.get(page_id)

  def allocate_tpu_pages(self, num_pages: int) -> list[int]:
    """Allocate logical TPU pages."""
    if num_pages < 0:
      raise ValueError("Cannot allocate a negative number of pages.")

    if num_pages == 0:
      return []

    if num_pages > self.num_free_tpu_pages:
      raise ValueError(
          f"Cannot allocate {num_pages} TPU pages, "
          f"only {self.num_free_tpu_pages} available."
      )

    allocated_ids = []
    phys_indices = self.tpu_pool.allocate(num_pages)

    for phys_idx in phys_indices:
      pid = self._next_page_id
      self._next_page_id += 1
      self._page_id_to_idx[pid] = phys_idx
      self._page_location[pid] = "tpu"

      allocated_ids.append(pid)

    return allocated_ids

  def update_tpu_pool(
      self, new_pages: dict[str, jax.Array | np.ndarray]
  ) -> None:
    """Updates the underlying TPU pool partition pages with new pages."""
    self.tpu_pool.update_pages(new_pages)

  @property
  def _tpu_sharding(self) -> jax.sharding.Sharding | None:
    first_layer_pages = next(iter(self.tpu_pool.partition_pages.values()))
    return getattr(first_layer_pages, "sharding", None)

  @property
  def _transferred_page_sharding(self) -> jax.sharding.Sharding | None:
    """Returns the target sharding for transferring to TPU."""
    tpu_sharding = self._tpu_sharding

    # Replicate the page dimension across all devices, so that pages are
    # scattered without cross-device communication.
    if isinstance(tpu_sharding, jax.sharding.NamedSharding):
      slice_spec = jax.sharding.PartitionSpec(None, *tpu_sharding.spec[1:])
      return jax.sharding.NamedSharding(tpu_sharding.mesh, slice_spec)

    return tpu_sharding

  def load(self, page_ids: Sequence[int]) -> None:
    """Transfers logical pages from CPU to TPU."""
    # TODO(yatlas): device_get only handles process local sharding.
    # Support for cross-host sharding needs to be added.
    if not page_ids:
      return

    if self.cpu_pool is None:
      raise ValueError(
          "Cannot load pages from CPU to TPU, CPU pool is not initialized."
      )

    if len(page_ids) > self.num_free_tpu_pages:
      raise ValueError(
          f"Cannot load {len(page_ids)} pages, "
          f"only {self.num_free_tpu_pages} available."
      )

    if len(set(page_ids)) != len(page_ids):
      raise ValueError("Cannot load duplicate pages.")

    for pid in page_ids:
      if self._page_location.get(pid) != "cpu":
        raise ValueError(
            f"Page ID {pid} is not on CPU "
            f"(location: {self._page_location.get(pid)})."
        )

    cpu_idxs = [self._page_id_to_idx[pid] for pid in page_ids]
    tpu_idxs = self.tpu_pool.allocate(len(page_ids))

    # Gather all the pages that need to be transferred to TPU.
    cpu_slices = {
        k: self.cpu_pool.partition_pages[k][cpu_idxs]
        for k in self.tpu_pool.partition_pages
    }

    # Transfer pages to the TPU
    tpu_slices = jax.device_put(cpu_slices, self._transferred_page_sharding)

    # Scatter pages to the TPU partitions.
    tpu_indices_arr = jnp.array(tpu_idxs, dtype=jnp.int32)
    tpu_partitions = self.tpu_pool.partition_pages

    # Use a jit-compiled function to avoid replicating page pools when
    # scattering pages across all TPU partitions.
    self.tpu_pool.partition_pages = _scatter_tpu_pages(
        tpu_partitions, tpu_indices_arr, tpu_slices
    )

    # Update page state
    self.cpu_pool.free(cpu_idxs)
    for pid, p_idx in zip(page_ids, tpu_idxs):
      self._page_id_to_idx[pid] = p_idx
      self._page_location[pid] = "tpu"

  def offload(self, page_ids: Sequence[int]) -> None:
    """Moves logical pages from TPU to CPU transferring only active pages."""
    if not page_ids:
      return

    if self.cpu_pool is None:
      raise ValueError(
          "Cannot offload pages to CPU, CPU pool is not initialized."
      )

    if len(page_ids) > self.num_free_cpu_pages:
      raise ValueError(
          f"Cannot offload {len(page_ids)} pages, "
          f"only {self.num_free_cpu_pages} available."
      )

    if len(set(page_ids)) != len(page_ids):
      raise ValueError("Cannot offload duplicate pages.")

    for pid in page_ids:
      if self._page_location.get(pid) != "tpu":
        raise ValueError(
            f"Page ID {pid} is not on TPU "
            f"(location: {self._page_location.get(pid)})."
        )

    physical_tpu_idxs = [self._page_id_to_idx[pid] for pid in page_ids]
    physical_cpu_idxs = self.cpu_pool.allocate(len(page_ids))
    tpu_indices_arr = jnp.array(physical_tpu_idxs, dtype=jnp.int32)

    # Use a jit-compiled function here to concurrently gather slices from all
    # TPU partitions.
    tpu_slices = _get_tpu_slices(self.tpu_pool.partition_pages, tpu_indices_arr)
    host_slices = jax.device_get(tpu_slices)
    for layer, host_slice in host_slices.items():
      self.cpu_pool.partition_pages[layer][physical_cpu_idxs] = host_slice

    self.tpu_pool.free(physical_tpu_idxs)
    for pid, p_idx in zip(page_ids, physical_cpu_idxs):
      self._page_id_to_idx[pid] = p_idx
      self._page_location[pid] = "cpu"

  def free(self, page_ids: Sequence[int]) -> None:
    """Releases physical allocations in tpu_pool or cpu_pool and removes logical IDs."""
    if not page_ids:
      return

    if len(set(page_ids)) != len(page_ids):
      raise ValueError("Cannot free duplicate pages.")

    for pid in page_ids:
      if pid not in self._page_location or pid not in self._page_id_to_idx:
        raise ValueError(f"Attempting to free page {pid} which is not in use.")

    cpu_idxs_to_free = []
    tpu_idxs_to_free = []

    for pid in page_ids:
      loc = self._page_location[pid]
      if loc == "cpu":
        cpu_idxs_to_free.append(self._page_id_to_idx[pid])
      elif loc == "tpu":
        tpu_idxs_to_free.append(self._page_id_to_idx[pid])

      del self._page_location[pid]
      del self._page_id_to_idx[pid]

    if cpu_idxs_to_free and self.cpu_pool:
      self.cpu_pool.free(cpu_idxs_to_free)
    if tpu_idxs_to_free and self.tpu_pool:
      self.tpu_pool.free(tpu_idxs_to_free)
