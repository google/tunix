import contextlib
import collections
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Optional
import dataclasses
from tunix.generate import utils
"""A tiered memory cache and manager."""

@dataclasses.dataclass(kw_only=True)
class PagePool:
  partition_pages: dict[str, jax.Array]
  available_page_indices: collections.deque
  num_free_pages: int

  def allocate(self, num_pages: int) -> list[int]:
    """Allocates physical indices from the pool."""
    assert num_pages <= self.num_free_pages, "Not enough physical pages"
    indices = [self.available_page_indices.popleft() for _ in range(num_pages)]
    self.num_free_pages -= num_pages
    return indices

  def free(self, indices: list[int]):
    """Frees physical indices logically."""
    for idx in indices:
        self.available_page_indices.append(idx)
    self.num_free_pages += len(indices)

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
  def logical_shard_to_physical(self) -> dict[str | None, str | None]:
    return {'dp_axis': self.dp_axis, 'tp_axis': self.tp_axis, None: None}

  @property
  def logical_sharding(self) -> tuple:
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
  def physical_sharding(self) -> tuple | None:
    logical_sharding = self.logical_sharding
    if all(axis is None for axis in logical_sharding):
      return None

    mapping = self.logical_shard_to_physical
    return tuple(mapping[axis] for axis in logical_sharding)

  def _make_pool(
    self,
    num_pages: int,
    sharding=None,
    device: jax.Device = None
    ) -> PagePool:
    with jax.default_device(device) if device else contextlib.nullcontext():
      shape = (num_pages, self.page_size) + self.page_subshape

      pages_dict = {}
      for k in self.partition_keys:
          arr = jnp.zeros(shape, dtype=self.dtype)
          if sharding is not None:
              arr = utils.shard(arr, sharding)
          pages_dict[k] = arr

      init_page_indices = range(num_pages)
      avail_indices = collections.deque(init_page_indices)

    return PagePool(
      partition_pages=pages_dict,
      available_page_indices=avail_indices,
      num_free_pages=num_pages,
    )

  def init(self) -> tuple[PagePool, PagePool | None]:
    """Initializes physical page tensors for TPU and CPU."""
        
    tpu_pool = self._make_pool(
        num_pages=self.num_tpu_pages,
        sharding=self.physical_sharding
    )

    cpu_pool = None
    if self.num_cpu_pages > 0:
        cpu_pool = self._make_pool(
          num_pages=self.num_cpu_pages,
          device=jax.devices('cpu')[0]
        )

    return (tpu_pool, cpu_pool)

class TieredPagePoolManager:
  """Manager for tiered TPU/CPU memory."""
  def __init__(
      self,
      tiered_config: TieredMemoryConfig,
      tpu_pool: PagePool,
      cpu_pool: PagePool | None,
      max_num_seqs: int = 256,
  ):
    self.config = tiered_config
    self.tpu_pool = tpu_pool
    self.cpu_pool = cpu_pool
    self.page_size = tiered_config.page_size
    self.max_num_seqs = max_num_seqs

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
    
    assert(num_pages <= self.num_free_tpu_pages)

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
      
    assert(self.cpu_pool is not None)

    # Caller should verify that sufficent pages are available
    assert(len(page_ids) <= self.num_free_tpu_pages)

    physical_cpu_idxs = []
    for pid in page_ids:
      assert(self._page_location.get(pid) == "cpu")
      physical_cpu_idxs.append(self._page_id_to_idx[pid])

    physical_hbm_idxs = self.tpu_pool.allocate(len(page_ids))
    self.cpu_pool.free(physical_cpu_idxs)
    
    cpu_indices_arr = jnp.array(physical_cpu_idxs, dtype=jnp.int32)
    tpu_indices_arr = jnp.array(physical_hbm_idxs, dtype=jnp.int32)

    for layer_name in self.tpu_pool.partition_pages.keys():
       cpu_tensor = self.cpu_pool.partition_pages[layer_name]
       source_data = cpu_tensor[cpu_indices_arr]
       
       tpu_tensor = self.tpu_pool.partition_pages[layer_name]
       self.tpu_pool.partition_pages[layer_name] = tpu_tensor.at[tpu_indices_arr].set(source_data)

    for pid, p_idx in zip(page_ids, physical_hbm_idxs):
      self._page_id_to_idx[pid] = p_idx
      self._page_location[pid] = "tpu"

  def offload(self, page_ids: List[int]):
    """Moves logical pages from TPU to CPU."""
    assert(len(page_ids) <= self.num_free_cpu_pages)

    if not page_ids:
      return

    physical_tpu_idxs = []
    for pid in page_ids:
      assert(self._page_location.get(pid) == "tpu")
      physical_tpu_idxs.append(self._page_id_to_idx[pid])

    physical_cpu_idxs = self.cpu_pool.allocate(len(page_ids))
    self.tpu_pool.free(physical_tpu_idxs)
    
    cpu_indices_arr = jnp.array(physical_cpu_idxs, dtype=jnp.int32)
    tpu_indices_arr = jnp.array(physical_tpu_idxs, dtype=jnp.int32)

    for layer_name in self.tpu_pool.partition_pages.keys():
       tpu_tensor = self.tpu_pool.partition_pages[layer_name]
       source_data = tpu_tensor[tpu_indices_arr]
       
       cpu_tensor = self.cpu_pool.partition_pages[layer_name]
       self.cpu_pool.partition_pages[layer_name] = cpu_tensor.at[cpu_indices_arr].set(source_data)

    for pid, p_idx in zip(page_ids, physical_cpu_idxs):
      self._page_id_to_idx[pid] = p_idx
      self._page_location[pid] = "cpu"

  def evict(self, page_ids: List[int]):
    """Releases the underlying physical allocation in tpu_pool and removes logical IDs."""
    cpu_idxs_to_evict = []
    tpu_idxs_to_evict = []

    for pid in page_ids:
      assert pid in self._page_location
      assert pid in self._page_id_to_idx 

      loc = self._page_location[pid]
      if loc == "cpu":
        cpu_idxs_to_evict.append(self._page_id_to_idx[pid])
      elif loc == "tpu":
        tpu_idxs_to_evict.append(self._page_id_to_idx[pid])

      del self._page_location[pid]
      del self._page_id_to_idx[pid]
