import contextlib
import collections
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Optional
import dataclasses
from tunix.generate import utils

@dataclasses.dataclass(kw_only=True)
class Block:
  partition_pages: dict[str, jax.Array]
  available_page_indices: collections.deque
  num_available_pages: int

  def allocate(self, num_pages: int) -> list[int]:
    """Allocates physical indices from the block."""
    assert num_pages <= self.num_available_pages, "Not enough physical pages"
    indices = [self.available_page_indices.popleft() for _ in range(num_pages)]
    self.num_available_pages -= num_pages
    return indices

  def free(self, indices: list[int]):
    """Frees physical indices logically."""
    for idx in indices:
        self.available_page_indices.append(idx)
    self.num_available_pages += len(indices)

@dataclasses.dataclass(frozen=True, kw_only=True)
class TieredMemoryConfig:
  """Configuration for tiered memory."""
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
    assert l_subsharding <= l_subshape, "Logical subsharding cannot exceed subshape dimensions."

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

  def _make_block(
    self,
    num_pages: int,
    sharding=None,
    device: jax.Device = None
    ) -> Block:
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

    return Block(
      partition_pages=pages_dict,
      available_page_indices=avail_indices,
      num_available_pages=num_pages,
    )

  def init(self) -> tuple[Block, Block | None]:
    """Initializes physical page tensors for TPU and CPU."""
        
    tpu_block = self._make_block(
        num_pages=self.num_tpu_pages,
        sharding=self.physical_sharding
    )

    cpu_block = None
    if self.num_cpu_pages > 0:
        cpu_block = self._make_block(
          num_pages=self.num_cpu_pages,
          device=jax.devices('cpu')[0]
        )

    return (tpu_block, cpu_block)


def init_cache_manager(
    num_tpu_pages: int,
    num_cpu_pages: int,
    cache_config,
    model_config,
    kv_dtype: jnp.dtype,
    dp_axis: str | None = None,
    tp_axis: str | None = None,
    dp_size: int = 1,
    tp_size: int = 1,
) -> 'CacheManager':
    """
    Initializes a CacheManager for the KV Cache.
    It builds the TpuCpuPageManagerConfig statically matching the model architecture.
    """
    num_layers = model_config.num_layers
    num_kv_heads = model_config.num_kv_heads
    head_dim = model_config.head_dim
    
    kv_packing = utils.get_dtype_packing(kv_dtype)
    assert(num_kv_heads % kv_packing == 0)
    packed_kv_dim = 2 * num_kv_heads // kv_packing
    
    partition_keys = tuple(f"layer_{i}" for i in range(num_layers))
    page_subshape = (packed_kv_dim, kv_packing, head_dim)
    

    
    tiered_memory_config = TieredMemoryConfig(
        page_size=cache_config.page_size,
        page_subshape=page_subshape,
        dtype=kv_dtype,
        partition_keys=partition_keys,
        num_tpu_pages=num_tpu_pages,
        num_cpu_pages=num_cpu_pages,
        logical_page_sharding='dp_axis',
        logical_subsharding=('tp_axis', None, None),
        dp_axis=dp_axis,
        tp_axis=tp_axis,
        dp_size=dp_size,
        tp_size=tp_size,
    )
    
    tpu_block, cpu_block = tiered_memory_config.init() 
    return CacheManager(
        tiered_config=tiered_memory_config,
        tpu_block=tpu_block,
        cpu_block=cpu_block,
        max_num_seqs=cache_config.max_num_seqs,
    )

class CacheManager:
  """
    Manages logical page IDs and orchestrates the physical JAX block memory.
    Operates outside of `jax.jit()`, tracking logic in pure Python and calling JAX methods
    when physical state needs to be updated.
    """

  def __init__(
      self,
      tiered_config: TieredMemoryConfig,
      tpu_block: Block,
      cpu_block: Block | None,
      max_num_seqs: int = 256,
  ):
    self.config = tiered_config
    self.tpu_block = tpu_block
    self.cpu_block = cpu_block
    self.page_size = tiered_config.page_size
    self.max_num_seqs = max_num_seqs

    self._next_page_id: int = 0
    self._page_id_to_idx: Dict[int, int] = {}
    self._page_location: Dict[int, str] = {}

    self.available_tpu_pages = tpu_block.num_available_pages
    self.available_cpu_pages = cpu_block.num_available_pages if cpu_block else 0
  
  @property
  def page_manager(self):
    return self.tpu_block

  def allocate_tpu_pages(self, num_pages: int) -> List[int]:
    """Allocates logical pages backing them immediately with TPU physical pages."""
    if num_pages == 0:
      return []
    
    assert(num_pages <= self.available_tpu_pages)

    allocated_ids = []
    phys_indices = self.tpu_block.allocate(num_pages)
    
    for phys_idx in phys_indices:
      pid = self._next_page_id
      self._next_page_id += 1
      self._page_id_to_idx[pid] = phys_idx
      self._page_location[pid] = "tpu"

      allocated_ids.append(pid)

    self.available_tpu_pages -= num_pages

    return allocated_ids

  def load(self, page_ids: List[int]):
    """Moves logical pages from CPU to TPU."""
    if not page_ids:
      return

    if not self.cpu_block:
      raise RuntimeError("No CPU cache configured to load from.")

    if len(page_ids) > self.available_tpu_pages:
      raise RuntimeError("Not enough HBM pages available to perform load.")

    physical_cpu_idxs = []
    for pid in page_ids:
      if self._page_location.get(pid) != "cpu":
        raise RuntimeError("Page is not actually in CPU")
      physical_cpu_idxs.append(self._page_id_to_idx[pid])

    physical_hbm_idxs = self.tpu_block.allocate(len(page_ids))
    self.cpu_block.free(physical_cpu_idxs)
    
    cpu_indices_arr = jnp.array(physical_cpu_idxs, dtype=jnp.int32)
    tpu_indices_arr = jnp.array(physical_hbm_idxs, dtype=jnp.int32)

    for layer_name in self.tpu_block.partition_pages.keys():
       cpu_tensor = self.cpu_block.partition_pages[layer_name]
       source_data = cpu_tensor[cpu_indices_arr]
       
       tpu_tensor = self.tpu_block.partition_pages[layer_name]
       self.tpu_block.partition_pages[layer_name] = tpu_tensor.at[tpu_indices_arr].set(source_data)

    self.available_tpu_pages -= len(page_ids)
    self.available_cpu_pages += len(page_ids)

    for pid, p_idx in zip(page_ids, physical_hbm_idxs):
      self._page_id_to_idx[pid] = p_idx
      self._page_location[pid] = "tpu"

  def offload(self, page_ids: List[int]):
    """Moves logical pages from TPU to CPU."""
    assert(len(page_ids) <= self.available_cpu_pages)

    if not page_ids:
      return

    physical_tpu_idxs = []
    for pid in page_ids:
      assert(self._page_location.get(pid) == "tpu")
      physical_tpu_idxs.append(self._page_id_to_idx[pid])

    physical_cpu_idxs = self.cpu_block.allocate(len(page_ids))
    self.tpu_block.free(physical_tpu_idxs)
    
    cpu_indices_arr = jnp.array(physical_cpu_idxs, dtype=jnp.int32)
    tpu_indices_arr = jnp.array(physical_tpu_idxs, dtype=jnp.int32)

    for layer_name in self.tpu_block.partition_pages.keys():
       tpu_tensor = self.tpu_block.partition_pages[layer_name]
       source_data = tpu_tensor[tpu_indices_arr]
       
       cpu_tensor = self.cpu_block.partition_pages[layer_name]
       self.cpu_block.partition_pages[layer_name] = cpu_tensor.at[cpu_indices_arr].set(source_data)

    self.available_cpu_pages -= len(page_ids)
    self.available_tpu_pages += len(page_ids)

    for pid, p_idx in zip(page_ids, physical_cpu_idxs):
      self._page_id_to_idx[pid] = p_idx
      self._page_location[pid] = "cpu"

  def evict(self, page_ids: List[int]):
    """Releases the underlying physical allocation in tpu_block and removes logical IDs."""
    cpu_idxs_to_evict = []
    tpu_idxs_to_evict = []

    for pid in page_ids:
      loc = self._page_location.get(pid)
      if loc == "cpu":
        cpu_idxs_to_evict.append(self._page_id_to_idx[pid])
      elif loc == "tpu":
        tpu_idxs_to_evict.append(self._page_id_to_idx[pid])

      if pid in self._page_location:
        del self._page_location[pid]
      if pid in self._page_id_to_idx:
        del self._page_id_to_idx[pid]

    self.available_cpu_pages += len(cpu_idxs_to_evict)
    self.available_tpu_pages += len(tpu_idxs_to_evict)
