import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Optional
from tunix.generate import batch_page_manager as batch_page_manager_lib

class CacheManager:
    """
    Manages logical page IDs and orchestrates physical JAX PageManagers (HBM and CPU).
    Operates outside of `jax.jit()`, tracking logic in pure Python and calling JAX methods
    when physical state needs to be updated.
    """
    def __init__(
        self, 
        hbm_page_manager: batch_page_manager_lib.BatchPageManager,
        max_num_seqs: int,
        max_num_pages_per_seq: int,
        page_size: int,
        cpu_block: Optional[jax.Array] = None
    ):
        self.hbm_page_manager = hbm_page_manager
        self.cpu_block = cpu_block
        
        self.max_num_seqs = max_num_seqs
        self.max_num_pages_per_seq = max_num_pages_per_seq
        self.page_size = page_size
        
        self.page_indices = np.full((max_num_seqs, max_num_pages_per_seq), -1, dtype=np.int32)
        self.seq_lens = np.zeros((max_num_seqs,), dtype=np.int32)
        
        self._next_page_id: int = 0
        
        self._page_id_to_idx: Dict[int, int] = {}
        self._page_location: Dict[int, str] = {}
        
        self.available_hbm_pages = int(self.hbm_page_manager.block.num_available_pages)
        self.cpu_free_indices = list(range(cpu_block.shape[0])) if cpu_block is not None else []
        self.available_cpu_pages = len(self.cpu_free_indices)

    def assign(self, sseq_page_ids: List[List[int]]):
        """
        Maps logical page_ids to physical page_idxs and natively populates the python 
        sequence arrays for PageManager.
        """
        num_seqs = len(sseq_page_ids)
        if num_seqs > self.max_num_seqs:
            raise RuntimeError("Exceeded max_num_seqs")
            
        self.page_indices = np.full((self.max_num_seqs, self.max_num_pages_per_seq), -1, dtype=np.int32)
        self.seq_lens = np.zeros((self.max_num_seqs,), dtype=np.int32)
        
        for req_idx, page_ids in enumerate(sseq_page_ids):
            seq_hbm_count = 0
            for i, pid in enumerate(page_ids):
                if self._page_location.get(pid) == "tpu":
                    self.page_indices[req_idx, i] = self._page_id_to_idx[pid]
                    seq_hbm_count += 1
                else:
                    raise RuntimeError(f"Page {pid} is not strictly in HBM, cannot assign!")
            self.seq_lens[req_idx] = seq_hbm_count * self.page_size

    def allocate(self, num_pages: int) -> List[int]:
        """Allocates logical pages backing them immediately with HBM physical pages."""
        if num_pages == 0:
            return []
            
        if num_pages > self.available_hbm_pages:
            raise RuntimeError(f"Cannot allocate {num_pages} pages. Only {self.available_hbm_pages} available.")
        
        self.hbm_page_manager, allocated_indices = self.hbm_page_manager.allocate(jnp.array(num_pages))
        physical_indices = np.array(allocated_indices).tolist()
        self.available_hbm_pages -= num_pages
        
        allocated_ids = []
        for phys_idx in physical_indices:
            pid = self._next_page_id
            self._next_page_id += 1
            self._page_id_to_idx[pid] = phys_idx
            self._page_location[pid] = "tpu"
            allocated_ids.append(pid)
            
        return allocated_ids

    def load(self, page_ids: List[int]):
        """Moves logical pages from CPU to TPU."""
        if self.cpu_block is None:
            raise RuntimeError("No offload cache configured to load from.")
        
        if len(page_ids) > self.available_hbm_pages:
            raise RuntimeError("Not enough HBM pages available to perform load.")
        
        if not page_ids:
            return

        # 1. Allocate equivalent physical HBM pages
        self.hbm_page_manager, allocated_hbm_idxs = self.hbm_page_manager.allocate(jnp.array(len(page_ids)))
        self.available_hbm_pages -= len(page_ids)
        physical_hbm_idxs = np.array(allocated_hbm_idxs).tolist()
        
        # 2. Gather source CPU physical indices
        physical_cpu_idxs = []
        for pid in page_ids:
            if self._page_location.get(pid) != "cpu":
                raise RuntimeError("Page is not actually in CPU")
            physical_cpu_idxs.append(self._page_id_to_idx[pid])

        # 3. Issue batch copy CPU -> HBM
        new_pages = batch_page_manager_lib.copy_physical_pages(
            src_pages=self.cpu_block,
            dst_pages=self.hbm_page_manager.block.pages,
            src_idxs=jnp.array(physical_cpu_idxs),
            dst_idxs=jnp.array(physical_hbm_idxs)
        )
        # Note: replace needs dataclasses
        import dataclasses
        new_block = dataclasses.replace(
            self.hbm_page_manager.block, pages=new_pages
        )
        self.hbm_page_manager = dataclasses.replace(
            self.hbm_page_manager, block=new_block
        )

        # 4. Evict the old CPU physical pages
        for cpu_idx in physical_cpu_idxs:
            self.cpu_free_indices.append(cpu_idx)
            self.available_cpu_pages += 1

        # 5. Re-map logical tracking
        for pid, p_idx in zip(page_ids, physical_hbm_idxs):
            self._page_id_to_idx[pid] = p_idx
            self._page_location[pid] = "tpu" 

    def offload(self, page_ids: List[int]):
        """Moves logical pages from TPU to CPU."""
        if self.cpu_block is None:
            raise RuntimeError("No offload cache configured to offload to.")
            
        if len(page_ids) > self.available_cpu_pages:
            raise RuntimeError("Not enough CPU pages available to perform offload.")

        if not page_ids:
            return

        # 1. Allocate equivalent physical CPU pages from python array
        physical_cpu_idxs = []
        for _ in range(len(page_ids)):
            physical_cpu_idxs.append(self.cpu_free_indices.pop())
            self.available_cpu_pages -= 1
        
        # 2. Gather source TPU physical indices
        physical_tpu_idxs = []
        for pid in page_ids:
            if self._page_location.get(pid) != "tpu":
                raise RuntimeError("Page is not actually in TPU")
            physical_tpu_idxs.append(self._page_id_to_idx[pid])

        # 3. Issue batch copy HBM -> CPU
        self.cpu_block = batch_page_manager_lib.copy_physical_pages(
            src_pages=self.hbm_page_manager.block.pages,
            dst_pages=self.cpu_block,
            src_idxs=jnp.array(physical_tpu_idxs),
            dst_idxs=jnp.array(physical_cpu_idxs)
        )

        # 4. Evict the old TPU physical pages
        self.hbm_page_manager = self.hbm_page_manager.evict_pages(
            page_indices_to_evict=jnp.array(physical_tpu_idxs),
            num_evicted=jnp.array(len(physical_tpu_idxs))
        )
        self.available_hbm_pages += len(physical_tpu_idxs)
        
        # 5. Re-map logical tracking
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
            else:
                pass
            
            if pid in self._page_location:
                del self._page_location[pid]
            if pid in self._page_id_to_idx:
                del self._page_id_to_idx[pid]

        if cpu_idxs_to_evict and self.cpu_block is not None:
            for idx in cpu_idxs_to_evict:
                self.cpu_free_indices.append(idx)
                self.available_cpu_pages += 1
            
        if tpu_idxs_to_evict:
            padded_tpu = np.zeros((self.hbm_page_manager.block.total_num_pages,), dtype=np.int32)
            padded_tpu[:len(tpu_idxs_to_evict)] = tpu_idxs_to_evict
            self.hbm_page_manager = self.hbm_page_manager.evict_pages(
                jnp.array(padded_tpu), 
                jnp.array(len(tpu_idxs_to_evict))
            )
            self.available_hbm_pages += len(tpu_idxs_to_evict)

    def tree_flatten(self):
        children = (
            self.hbm_page_manager,
            self.cpu_block,
            self.page_indices,
            self.seq_lens
        )
        aux_data = {
            'max_num_seqs': self.max_num_seqs,
            'max_num_pages_per_seq': self.max_num_pages_per_seq,
            'page_size': self.page_size,
            '_next_page_id': self._next_page_id,
            '_page_id_to_idx': self._page_id_to_idx,
            '_page_location': self._page_location,
            'available_hbm_pages': self.available_hbm_pages,
            'cpu_free_indices': self.cpu_free_indices,
            'available_cpu_pages': self.available_cpu_pages,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.hbm_page_manager = children[0]
        obj.cpu_block = children[1]
        obj.page_indices = children[2]
        obj.seq_lens = children[3]
        for k, v in aux_data.items():
            setattr(obj, k, v)
        return obj

jax.tree_util.register_pytree_node(CacheManager, CacheManager.tree_flatten, CacheManager.tree_unflatten)
