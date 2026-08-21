import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Optional
from tunix.generate import page_manager as page_manager_lib

class CacheManager:
    """
    Manages logical page IDs and orchestrates physical JAX PageManagers (HBM and CPU).
    Operates outside of `jax.jit()`, tracking logic in pure Python and calling JAX methods
    when physical state needs to be updated.
    """
    def __init__(
        self, 
        hbm_page_manager: page_manager_lib.PageManager,
        offload_page_manager: Optional[page_manager_lib.PageManager] = None
    ):
        self.hbm_page_manager = hbm_page_manager
        self.offload_page_manager = offload_page_manager
        
        self._next_page_id: int = 0
        
        self._page_id_to_idx: Dict[int, int] = {}
        self._page_location: Dict[int, str] = {}
        
        self.available_hbm_pages = int(self.hbm_page_manager.num_available_pages)
        self.available_cpu_pages = int(self.offload_page_manager.num_available_pages) if offload_page_manager else 0

    def assign(self, sseq_page_ids: List[List[int]]):
        """
        Maps logical page_ids to physical page_idxs and calls the underlying 
        JAX PageManager.assign method directly.
        """
        num_seqs = len(sseq_page_ids)
        if num_seqs == 0:
            return

        packed_hbm_idxs = []
        hbm_lens = []
        
        for page_ids in sseq_page_ids:
            seq_hbm_count = 0
            for pid in page_ids:
                if self._page_location.get(pid) == "tpu":
                    packed_hbm_idxs.append(self._page_id_to_idx[pid])
                    seq_hbm_count += 1
                else:
                    raise RuntimeError(f"Page {pid} is not strictly in HBM, cannot assign!")
            hbm_lens.append(seq_hbm_count)

        padded_seq_idxs = np.full((self.hbm_page_manager.batch_size,), fill_value=self.hbm_page_manager.max_num_seqs, dtype=np.int32) 
        padded_seq_idxs[:num_seqs] = list(range(num_seqs))
        
        padded_lens = np.zeros((self.hbm_page_manager.batch_size,), dtype=np.int32)
        padded_lens[:num_seqs] = hbm_lens
        
        max_packed_capacity = self.hbm_page_manager.batch_size * self.hbm_page_manager.max_num_pages_per_seq
        padded_hbm_idxs = np.full((max_packed_capacity,), fill_value=-1, dtype=np.int32)
        
        actual_packed_len = len(packed_hbm_idxs)
        padded_hbm_idxs[:actual_packed_len] = packed_hbm_idxs

        self.hbm_page_manager = self.hbm_page_manager.assign(
            jnp.array(padded_seq_idxs),
            jnp.array(padded_hbm_idxs), 
            jnp.array(padded_lens)
        )

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
        if not self.offload_page_manager:
            raise RuntimeError("No offload cache configured to load from.")
        
        if len(page_ids) > self.available_hbm_pages:
            raise RuntimeError("Not enough HBM pages available to perform load.")
            
        # TODO: Implement physical tensor block transfers between the two page managers
        # 1. Allocate equivalent physical HBM pages
        # 2. Issue a batch copy kernel from CPU -> HBM
        # 3. Evict the old CPU physical pages back into CPU pool
        # 4. Re-map logical tracking 
        pass

    def offload(self, page_ids: List[int]):
        """Moves logical pages from TPU to CPU."""
        if not self.offload_page_manager:
            raise RuntimeError("No offload cache configured to offload to.")
            
        if len(page_ids) > self.available_cpu_pages:
            raise RuntimeError("Not enough CPU pages available to perform offload.")

        # TODO: Implement physical tensor block transfers between the two page managers
        # 1. Allocate equivalent physical CPU pages
        # 2. Issue a batch copy kernel from HBM -> CPU
        # 3. Evict the old HBM physical pages back into HBM pool
        # 4. Re-map logical tracking 
        pass

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

        if cpu_idxs_to_evict and self.offload_page_manager:
            padded_cpu = np.zeros((self.offload_page_manager.total_num_pages,), dtype=np.int32)
            padded_cpu[:len(cpu_idxs_to_evict)] = cpu_idxs_to_evict
            self.offload_page_manager = self.offload_page_manager.evict_pages(
                jnp.array(padded_cpu), 
                jnp.array(len(cpu_idxs_to_evict))
            )
            self.available_cpu_pages += len(cpu_idxs_to_evict)
            
        if tpu_idxs_to_evict:
            padded_tpu = np.zeros((self.hbm_page_manager.total_num_pages,), dtype=np.int32)
            padded_tpu[:len(tpu_idxs_to_evict)] = tpu_idxs_to_evict
            self.hbm_page_manager = self.hbm_page_manager.evict_pages(
                jnp.array(padded_tpu), 
                jnp.array(len(tpu_idxs_to_evict))
            )
            self.available_hbm_pages += len(tpu_idxs_to_evict)
