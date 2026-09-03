"""A tiered memory cache and manager."""

import contextlib
import dataclasses
from typing import Dict, List
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import utils


@dataclasses.dataclass(kw_only=True)
class PagePool:
  """A pool of pages."""

  partition_pages: dict[str, jax.Array]
  available_page_indices: List[int]
  allocated_pages: set[int] = dataclasses.field(default_factory=set)

  def allocate(self, num_pages: int) -> list[int]:
    """Allocates pages in the pool."""
    assert num_pages <= self.num_free_pages
    assert num_pages >= 0

    if num_pages == 0:
      return []

    indices = self.available_page_indices[-num_pages:]
    del self.available_page_indices[-num_pages:]

    if __debug__:
      self.allocated_pages.update(indices)

    return indices

  def free(self, indices: list[int]):
    """Frees pages in the pool."""
    if __debug__:
      for idx in indices:
        assert idx in self.allocated_pages
        self.allocated_pages.remove(idx)

    self.available_page_indices.extend(indices)

  @property
  def num_free_pages(self):
    return len(self.available_page_indices)


@dataclasses.dataclass(frozen=True, kw_only=True)
class TieredPagePoolConfig:
  """Configuration for tiered cache."""

  page_size: int
  page_subshape: tuple[int, ...] = ()
  dtype: jnp.dtype
  partition_keys: tuple[str, ...]

  num_tpu_pages: int
  num_cpu_pages: int = 0

  logical_page_sharding: str | None = None
  logical_subsharding: tuple[str | None, ...] = ()

  dp_axis: str | None = None
  tp_axis: str | None = None
  dp_size: int = 1
  tp_size: int = 1

  @property
  def logical_shard_to_physical(
      self,
  ) -> dict[str | None, str | None]:
    return {"dp": self.dp_axis, "tp": self.tp_axis, None: None}

  @property
  def logical_sharding(self) -> tuple[str | None, ...]:
    """Returns the logical sharding of the page pool."""
    logical_page_sharding = self.logical_page_sharding
    logical_subsharding = self.logical_subsharding

    logical_prefix_sharding = (logical_page_sharding, None)

    l_subsharding = len(logical_subsharding)
    l_subshape = len(self.page_subshape)
    assert l_subsharding <= l_subshape

    if l_subsharding < l_subshape:
      num_padding = l_subshape - l_subsharding
      logical_subsharding += (None,) * num_padding

    return logical_prefix_sharding + logical_subsharding

  @property
  def physical_sharding(self) -> tuple[str, ...] | None:
    logical_sharding = self.logical_sharding
    if all(axis is None for axis in logical_sharding):
      return None

    mapping = self.logical_shard_to_physical
    # pytype: disable=bad-return-type
    return tuple(mapping[axis] for axis in logical_sharding)
    # pytype: enable=bad-return-type

  def _make_pool(
      self, num_pages: int, sharding=None, device: jax.Device | None = None
  ) -> PagePool:
    """Creates a page pool."""
    with jax.default_device(device) if device else contextlib.nullcontext():
      shape = (num_pages, self.page_size) + self.page_subshape

      pages_dict = {}
      for k in self.partition_keys:
        arr = jnp.zeros(shape, dtype=self.dtype)
        if sharding is not None:
          arr = utils.shard(arr, sharding)
        pages_dict[k] = arr

    avail_indices = list(range(num_pages))

    return PagePool(
        partition_pages=pages_dict,
        available_page_indices=avail_indices,
    )

  def init(self) -> tuple[PagePool, PagePool | None]:
    """Initializes physical page tensors for TPU and CPU."""

    tpu_pool = self._make_pool(
        num_pages=self.num_tpu_pages, sharding=self.physical_sharding
    )

    cpu_pool = None
    if self.num_cpu_pages > 0:
      cpu_pool = self._make_pool(
          num_pages=self.num_cpu_pages, device=jax.devices("cpu")[0]
      )

    return (tpu_pool, cpu_pool)


class TieredPagePoolManager:
  """Manager for tiered TPU/CPU memory."""

  def __init__(
      self,
      tiered_config: TieredPagePoolConfig,
      tpu_pool: PagePool,
      cpu_pool: PagePool | None,
  ):
    self.config = tiered_config
    self.tpu_pool = tpu_pool
    self.cpu_pool = cpu_pool
    self.page_size = tiered_config.page_size

    self._next_page_id: int = 0
    self._page_id_to_idx: Dict[int, int] = {}
    self._page_location: Dict[int, str] = {}

  @property
  def num_free_tpu_pages(self):
    return self.tpu_pool.num_free_pages

  @property
  def num_free_cpu_pages(self):
    if self.cpu_pool:
      return self.cpu_pool.num_free_pages
    return 0

  def get_page_location(self, page_id):
    return self._page_location.get(page_id)

  def get_page_idx(self, page_id):
    return self._page_id_to_idx.get(page_id)

  def allocate_tpu_pages(self, num_pages: int) -> List[int]:
    """Allocate logical TPU pages."""
    if num_pages == 0:
      return []

    assert num_pages <= self.num_free_tpu_pages

    allocated_ids = []
    phys_indices = self.tpu_pool.allocate(num_pages)

    for phys_idx in phys_indices:
      pid = self._next_page_id
      self._next_page_id += 1
      self._page_id_to_idx[pid] = phys_idx
      self._page_location[pid] = "tpu"

      allocated_ids.append(pid)

    return allocated_ids

  def update_tpu_pool(self, new_pages):
    self.tpu_pool.partition_pages = new_pages

  def load(self, page_ids: List[int]):
    """Moves logical pages from CPU to TPU."""

    if not page_ids:
      return

    assert self.cpu_pool is not None

    # Caller should verify that sufficent pages are available
    assert len(page_ids) <= self.num_free_tpu_pages

    # --- Gather physical CPU idxs ---
    physical_cpu_idxs = []
    for pid in page_ids:
      assert self._page_location.get(pid) == "cpu"

    for pid in page_ids:
      physical_cpu_idxs.append(self._page_id_to_idx[pid])

    # --- Copy CPU pages into new TPU pages ---
    physical_hbm_idxs = self.tpu_pool.allocate(len(page_ids))

    static_size = utils.next_power_of_2(len(page_ids))
    valid_mask = np.zeros(static_size, dtype=np.bool_)
    valid_mask[:len(page_ids)] = True

    cpu_indices_arr = np.array(physical_cpu_idxs, dtype=np.int32)
    cpu_indices_arr = utils.pad_to_length(cpu_indices_arr, static_size)
    
    tpu_indices_arr = np.array(physical_hbm_idxs, dtype=np.int32)
    tpu_indices_arr = utils.pad_to_length(tpu_indices_arr, static_size)

    cpu_device = jax.devices("cpu")[0]
    cpu_indices_arr = jax.device_put(cpu_indices_arr, cpu_device)
    tpu_indices_arr = jnp.array(tpu_indices_arr)
    valid_mask_arr = jnp.array(valid_mask)

    for layer_name in self.tpu_pool.partition_pages.keys():
      self.tpu_pool.partition_pages[layer_name] = utils.copy_physical_pages(
          src_pages=self.cpu_pool.partition_pages[layer_name],
          dst_pages=self.tpu_pool.partition_pages[layer_name],
          src_idxs=cpu_indices_arr,
          dst_idxs=tpu_indices_arr,
          valid_mask=valid_mask_arr,
      )

    # --- Update page states ---
    self.cpu_pool.free(physical_cpu_idxs)
    for pid, p_idx in zip(page_ids, physical_hbm_idxs):
      self._page_id_to_idx[pid] = p_idx
      self._page_location[pid] = "tpu"

  def offload(self, page_ids: List[int]):
    """Moves logical pages from TPU to CPU."""

    if not page_ids:
      return

    # Caller should verify that sufficent pages are available
    assert len(page_ids) <= self.num_free_cpu_pages
    assert self.cpu_pool is not None

    # --- Gather physical TPU idxs ---
    physical_tpu_idxs = []
    for pid in page_ids:
      assert self._page_location.get(pid) == "tpu"

    for pid in page_ids:
      physical_tpu_idxs.append(self._page_id_to_idx[pid])

    physical_cpu_idxs = self.cpu_pool.allocate(len(page_ids))

    # --- Copy TPU pages into new CPU pages ---
    static_size = utils.next_power_of_2(len(page_ids))
    valid_mask = np.zeros(static_size, dtype=np.bool_)
    valid_mask[:len(page_ids)] = True

    cpu_indices_arr = np.array(physical_cpu_idxs, dtype=np.int32)
    cpu_indices_arr = utils.pad_to_length(cpu_indices_arr, static_size)
    
    tpu_indices_arr = np.array(physical_tpu_idxs, dtype=np.int32)
    tpu_indices_arr = utils.pad_to_length(tpu_indices_arr, static_size)

    cpu_device = jax.devices("cpu")[0]
    cpu_indices_arr = jax.device_put(cpu_indices_arr, cpu_device)
    tpu_indices_arr = jnp.array(tpu_indices_arr)
    valid_mask_arr = jnp.array(valid_mask)

    for layer_name in self.tpu_pool.partition_pages.keys():
      self.cpu_pool.partition_pages[layer_name] = utils.copy_physical_pages(
          src_pages=self.tpu_pool.partition_pages[layer_name],
          dst_pages=self.cpu_pool.partition_pages[layer_name],
          src_idxs=tpu_indices_arr,
          dst_idxs=cpu_indices_arr,
          valid_mask=valid_mask_arr,
      )

    # --- Update page states ---
    self.tpu_pool.free(physical_tpu_idxs)
    for pid, p_idx in zip(page_ids, physical_cpu_idxs):
      self._page_id_to_idx[pid] = p_idx
      self._page_location[pid] = "cpu"

  def evict(self, page_ids: List[int]):
    """Releases logical page allocations."""

    cpu_idxs_to_evict = []
    tpu_idxs_to_evict = []

    for pid in page_ids:
      assert pid in self._page_location
      assert pid in self._page_id_to_idx

    for pid in page_ids:
      loc = self._page_location[pid]
      if loc == "cpu":
        cpu_idxs_to_evict.append(self._page_id_to_idx[pid])
      elif loc == "tpu":
        tpu_idxs_to_evict.append(self._page_id_to_idx[pid])

      del self._page_location[pid]
      del self._page_id_to_idx[pid]

    if cpu_idxs_to_evict and self.cpu_pool:
      self.cpu_pool.free(cpu_idxs_to_evict)
    if tpu_idxs_to_evict and self.tpu_pool:
      self.tpu_pool.free(tpu_idxs_to_evict)
