import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Optional
import dataclasses
from tunix.generate import page_manager as pm_lib
from tunix.generate import utils


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class CacheManagerConfig:
  page_size: int
  page_subshape: tuple[int, ...] = ()
  dtype: jnp.dtype
  max_num_seqs: int
  max_seq_len: int
  max_tpu_bytes: int
  max_cpu_bytes: int = 0
  logical_page_sharding: Optional[str] = None
  logical_subsharding: tuple[Optional[str], ...] = ()
  dp_axis: Optional[str] = None
  tp_axis: Optional[str] = None
  dp_size: int = 1
  tp_size: int = 1

  def init(self) -> 'CacheManager':
    pm_config = pm_lib.TpuCpuPageManagerConfig(
        page_size=self.page_size,
        page_subshape=self.page_subshape,
        dtype=self.dtype,
        max_num_seqs=self.max_num_seqs,
        max_seq_len=self.max_seq_len,
        max_tpu_bytes=self.max_tpu_bytes,
        max_cpu_bytes=self.max_cpu_bytes,
        logical_page_sharding=self.logical_page_sharding,
        logical_subsharding=self.logical_subsharding,
        dp_axis=self.dp_axis,
        tp_axis=self.tp_axis,
        dp_size=self.dp_size,
        tp_size=self.tp_size,
    )
    return CacheManager(pm_config)


class CacheManager:
  """
    Manages logical page IDs and orchestrates the physical JAX TpuCpuPageManager.
    Operates outside of `jax.jit()`, tracking logic in pure Python and calling JAX methods
    when physical state needs to be updated.
    """

  def __init__(
      self,
      config: pm_lib.TpuCpuPageManagerConfig,
  ):
    self.config = config
    self.page_manager = config.init()

    self.max_num_seqs = config.max_num_seqs
    self.max_num_pages_per_seq = utils.cdiv(config.max_seq_len,
                                            config.page_size)
    self.page_size = config.page_size

    self._next_page_id: int = 0
    self._page_id_to_idx: Dict[int, int] = {}
    self._page_location: Dict[int, str] = {}

    self.seq_lens = np.zeros(self.max_num_seqs, dtype=np.int32)

    self.available_tpu_pages = config.num_tpu_pages
    self.available_cpu_pages = config.num_cpu_pages

  def allocate(self, num_pages: int) -> List[int]:
    """Allocates logical pages backing them immediately with TPU physical pages."""
    if num_pages == 0:
      return []

    if num_pages > self.available_tpu_pages:
      raise RuntimeError(
          f"Cannot allocate {num_pages} pages. Only {self.available_tpu_pages} available."
      )

    # TpuCpuPageManager allocate returns (TpuCpuPageManager, padded_allocated_page_data_indices)
    self.page_manager, allocated_indices_padded = self.page_manager.allocate(
        jnp.array(num_pages, dtype=jnp.int32))

    physical_indices = np.array(allocated_indices_padded)[:num_pages].tolist()

    self.available_tpu_pages -= num_pages

    allocated_ids = []
    for phys_idx in physical_indices:
      pid = self._next_page_id
      self._next_page_id += 1
      self._page_id_to_idx[pid] = phys_idx
      self._page_location[pid] = "tpu"
      allocated_ids.append(pid)

    return allocated_ids

  def assign(self, append_page_ids: List[List[int]]):
    """
        Maps logical page_ids to physical page_idxs and natively populates the python 
        sequence arrays for TpuCpuPageManager.
        This behaves iteratively like an append. CacheManager natively syncs seq_lens.
        """
    num_seqs = len(append_page_ids)
    if num_seqs > self.max_num_seqs:
      raise RuntimeError("Exceeded max_num_seqs")

    page_indices_list = []
    lens = np.zeros(self.max_num_seqs, dtype=np.int32)

    for req_idx, page_ids in enumerate(append_page_ids):
      if not page_ids:
        continue

      for pid in page_ids:
        if self._page_location.get(pid) == "tpu":
          page_indices_list.append(self._page_id_to_idx[pid])
          lens[req_idx] += 1
        else:
          raise RuntimeError(
              f"Page {pid} is not strictly in HBM (tpu), cannot assign!")

      self.seq_lens[req_idx] += lens[req_idx]

    if len(page_indices_list) == 0:
      return

    padded_page_indices = np.zeros(self.config.num_tpu_pages, dtype=np.int32)
    padded_page_indices[:len(page_indices_list)] = page_indices_list

    self.page_manager = self.page_manager.assign(
        jnp.array(padded_page_indices, dtype=jnp.int32),
        jnp.array(lens, dtype=jnp.int32))

  def load(self, page_ids: List[int]):
    """Moves logical pages from CPU to TPU."""
    if self.config.num_cpu_pages == 0:
      raise RuntimeError("No CPU cache configured to load from.")

    if len(page_ids) > self.available_tpu_pages:
      raise RuntimeError("Not enough HBM pages available to perform load.")

    if not page_ids:
      return

    physical_cpu_idxs = []
    for pid in page_ids:
      if self._page_location.get(pid) != "cpu":
        raise RuntimeError("Page is not actually in CPU")
      physical_cpu_idxs.append(self._page_id_to_idx[pid])

    padded_cpu_idxs = np.zeros(self.config.num_cpu_pages, dtype=np.int32)
    padded_cpu_idxs[:len(physical_cpu_idxs)] = physical_cpu_idxs

    self.page_manager, padded_hbm_idxs = self.page_manager.load(
        len(page_ids), jnp.array(padded_cpu_idxs, dtype=jnp.int32))

    physical_hbm_idxs = np.array(padded_hbm_idxs)[:len(page_ids)].tolist()

    self.available_tpu_pages -= len(page_ids)
    self.available_cpu_pages += len(page_ids)

    for pid, p_idx in zip(page_ids, physical_hbm_idxs):
      self._page_id_to_idx[pid] = p_idx
      self._page_location[pid] = "tpu"

  def offload(self, page_ids: List[int]):
    """Moves logical pages from TPU to CPU."""
    if self.config.num_cpu_pages == 0:
      raise RuntimeError("No CPU cache configured to offload to.")

    if len(page_ids) > self.available_cpu_pages:
      raise RuntimeError("Not enough CPU pages available to perform offload.")

    if not page_ids:
      return

    physical_tpu_idxs = []
    for pid in page_ids:
      if self._page_location.get(pid) != "tpu":
        raise RuntimeError("Page is not actually in TPU")
      physical_tpu_idxs.append(self._page_id_to_idx[pid])

    padded_tpu_idxs = np.zeros(self.config.num_tpu_pages, dtype=np.int32)
    padded_tpu_idxs[:len(physical_tpu_idxs)] = physical_tpu_idxs

    self.page_manager, padded_cpu_idxs = self.page_manager.offload(
        len(page_ids), jnp.array(padded_tpu_idxs, dtype=jnp.int32))

    physical_cpu_idxs = np.array(padded_cpu_idxs)[:len(page_ids)].tolist()

    self.available_cpu_pages -= len(page_ids)
    self.available_tpu_pages += len(page_ids)

    for pid, p_idx in zip(page_ids, physical_cpu_idxs):
      self._page_id_to_idx[pid] = p_idx
      self._page_location[pid] = "cpu"

  def evict(self, page_ids: List[int]):
    """Releases the underlying physical allocation in page_manager and removes logical IDs."""
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

    if cpu_idxs_to_evict and self.config.num_cpu_pages > 0:
      padded_cpu = np.zeros(self.config.num_cpu_pages, dtype=np.int32)
      padded_cpu[:len(cpu_idxs_to_evict)] = cpu_idxs_to_evict
      new_cpu_block = self.page_manager.cpu_block.release(
          len(cpu_idxs_to_evict), jnp.array(padded_cpu, dtype=jnp.int32))
      self.page_manager = dataclasses.replace(self.page_manager,
                                              cpu_block=new_cpu_block)
      self.available_cpu_pages += len(cpu_idxs_to_evict)

    if tpu_idxs_to_evict:
      padded_tpu = np.zeros(self.config.num_tpu_pages, dtype=np.int32)
      padded_tpu[:len(tpu_idxs_to_evict)] = tpu_idxs_to_evict
      new_tpu_block = self.page_manager.tpu_block.release(
          len(tpu_idxs_to_evict), jnp.array(padded_tpu, dtype=jnp.int32))
      self.page_manager = dataclasses.replace(self.page_manager,
                                              tpu_block=new_tpu_block)
      self.available_tpu_pages += len(tpu_idxs_to_evict)

  def tree_flatten(self):
    children = (self.page_manager,)
    aux_data = {
        'config': self.config,
        'max_num_seqs': self.max_num_seqs,
        'max_num_pages_per_seq': self.max_num_pages_per_seq,
        'page_size': self.page_size,
        '_next_page_id': self._next_page_id,
        '_page_id_to_idx': self._page_id_to_idx,
        '_page_location': self._page_location,
        'seq_lens': self.seq_lens,
        'available_tpu_pages': self.available_tpu_pages,
        'available_cpu_pages': self.available_cpu_pages,
    }
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    obj = cls.__new__(cls)
    obj.page_manager = children[0]
    for k, v in aux_data.items():
      setattr(obj, k, v)
    return obj


jax.tree_util.register_pytree_node(CacheManager, CacheManager.tree_flatten,
                                   CacheManager.tree_unflatten)

from typing import Any
