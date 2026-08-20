import collections
from typing import List, Dict, Tuple
from tunix.generate.cache_manager import CacheManager

class Request:
    def __init__(self, req_id: str, prompt_tokens: List[int]):
        self.req_id = req_id
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = []
        self.page_ids = []
        self.seq_len = len(prompt_tokens)
        self.is_prefill_done = False

class Scheduler:
    """Python-based continuous batching scheduler."""
    def __init__(self, cache_manager: CacheManager, page_size: int, max_num_seqs: int):
        self.cache_manager = cache_manager
        self.page_size = page_size
        self.max_num_seqs = max_num_seqs
        
        self.running_requests = collections.deque()
        self.pending_requests = collections.deque()
        
        self.prefix_hash_to_page_id: Dict[int, int] = {}
        self.page_ref_counts: Dict[int, int] = {}
        self.page_location: Dict[int, str] = {} # "tpu" or "cpu"
        
        # O(1) LRU explicit caches naturally segregated by physical location
        self.unreferenced_tpu_pages = collections.OrderedDict()
        self.unreferenced_cpu_pages = collections.OrderedDict()

    def touch_page(self, page_id: int):
        if page_id not in self.page_ref_counts:
            self.page_ref_counts[page_id] = 1
        else:
            self.page_ref_counts[page_id] += 1
            if page_id in self.unreferenced_tpu_pages:
                del self.unreferenced_tpu_pages[page_id]
            elif page_id in self.unreferenced_cpu_pages:
                del self.unreferenced_cpu_pages[page_id]

    def release_page(self, page_id: int):
        self.page_ref_counts[page_id] -= 1
        if self.page_ref_counts[page_id] == 0:
            if self.page_location.get(page_id) == "tpu":
                self.unreferenced_tpu_pages[page_id] = None
            else:
                self.unreferenced_cpu_pages[page_id] = None

    def _free_up_tpu_space(self, num_pages: int):
        """Strict O(1) lookup of oldest unreferenced TPU pages."""
        if num_pages <= 0: return

        pages_to_offload = []
        for _ in range(num_pages):
            if not self.unreferenced_tpu_pages:
                raise RuntimeError("Out of TPU memory! Not enough unreferenced TPU pages to offload.")
            pid, _ = self.unreferenced_tpu_pages.popitem(last=False)
            pages_to_offload.append(pid)
            self.page_location[pid] = "cpu"
            self.unreferenced_cpu_pages[pid] = None 
            
        if pages_to_offload:
            self.cache_manager.offload(pages_to_offload)

    def _free_up_cpu_space(self, num_pages: int):
        """Strict O(1) eviction of oldest CPU pages."""
        if num_pages <= 0: return
        
        pages_to_evict = []
        for _ in range(num_pages):
            if not self.unreferenced_cpu_pages:
                raise RuntimeError("Out of CPU memory! System completely OOM.")
            pid, _ = self.unreferenced_cpu_pages.popitem(last=False)
            pages_to_evict.append(pid)
            del self.page_location[pid]
            del self.page_ref_counts[pid]
            
        if pages_to_evict:
            self.cache_manager.evict(pages_to_evict)

    def schedule_step(self, new_requests: List[Request]) -> Tuple[List[Request], List[Request]]:
        """
        Determines which requests participate, matches prefixes, and strictly batches 
        all JAX cache manager interactions.
        """
        self._queue_new_requests(new_requests)
        self._make_room_for_allocation()
        self._drain_pending_queue()
        
        total_new_pages_needed = self._calculate_new_pages_needed()
        if total_new_pages_needed > 0:
            assigned_page_ids = self.cache_manager.allocate(total_new_pages_needed)
            self._distribute_allocated_pages(assigned_page_ids)
            
        sseq_ids = [req.req_id for req in self.running_requests]
        sseq_page_ids = [req.page_ids for req in self.running_requests]
        self.cache_manager.assign(sseq_ids, sseq_page_ids)
        
        scheduled_decodes = []
        scheduled_prefills = []
        for req in self.running_requests:
            if req.is_prefill_done:
                scheduled_decodes.append(req)
            else:
                scheduled_prefills.append(req)
                
        return scheduled_decodes, scheduled_prefills

    def _queue_new_requests(self, new_requests: List[Request]):
        for req in new_requests:
            self.pending_requests.append(req)

    def _make_room_for_allocation(self):
        """
        Preempts running decodes if required_pages exceeds boundary limits.
        """
        required_pages = len(self.running_requests)
        free_tpu = self.cache_manager.available_hbm_pages

        while free_tpu + len(self.unreferenced_tpu_pages) < required_pages and self.running_requests:
            preempted = self.running_requests.pop()
            for pid in preempted.page_ids:
                self.release_page(pid)
            self.pending_requests.appendleft(preempted)
            required_pages -= 1

        pages_to_free = max(0, required_pages - free_tpu)
        self._free_up_tpu_space(pages_to_free)

    def _chunk_and_hash(self, tokens: List[int]) -> List[int]:
        """Returns the list of block hashes for a full sequence of tokens."""
        hashes = []
        parent_hash = 0
        for i in range(0, len(tokens), self.page_size):
            chunk = tuple(tokens[i:i+self.page_size])
            parent_hash = hash((parent_hash, chunk))
            hashes.append(parent_hash)
        return hashes

    def _drain_pending_queue(self):
        """
        Admit sequences based on prefix matches and available HBM space.
        """
        from tunix.generate import utils
        free_tpu = self.cache_manager.available_hbm_pages + len(self.unreferenced_tpu_pages)
        pages_to_load = []
        pages_to_load_set = set()

        while self.pending_requests:
            req = self.pending_requests[0]
            req_hashes = self._chunk_and_hash(req.prompt_tokens)
            
            matched_page_ids = []
            for h in req_hashes:
                if h in self.prefix_hash_to_page_id:
                    matched_page_ids.append(self.prefix_hash_to_page_id[h])
                else:
                    break
            
            total_pages_needed = utils.cdiv(len(req.prompt_tokens), self.page_size)
            new_pages_needed = total_pages_needed - len(matched_page_ids)
            
            # CPU pages being recycled will cost HBM space
            cpu_pages_used = 0
            for pid in matched_page_ids:
                if self.page_location.get(pid) == "cpu" and pid not in pages_to_load_set:
                    cpu_pages_used += 1
                    
            total_hbm_cost = new_pages_needed + cpu_pages_used
            
            if free_tpu >= total_hbm_cost:
                self.pending_requests.popleft()
                
                # Reclaim matches
                req.page_ids = []
                for pid in matched_page_ids:
                    if self.page_location.get(pid) == "cpu" and pid not in pages_to_load_set:
                        pages_to_load.append(pid)
                        pages_to_load_set.add(pid)
                    self.touch_page(pid)
                    req.page_ids.append(pid)
                
                self.running_requests.append(req)
                free_tpu -= total_hbm_cost
            else:
                break
                
        # Batch load any matched prefix pages residing physically in CPU
        if pages_to_load:
            # Must free physical TPU space for loads if required
            physically_free = self.cache_manager.available_hbm_pages
            if len(pages_to_load) > physically_free:
                self._free_up_tpu_space(len(pages_to_load) - physically_free)
            self.cache_manager.load(pages_to_load)

    def _calculate_new_pages_needed(self) -> int:
        """Sums up the missing boundary pages for all sequences in `running_requests`."""
        from tunix.generate import utils
        total_missing = 0
        for req in self.running_requests:
            current_capacity = len(req.page_ids) * self.page_size
            total_desired = req.seq_len + 1
            
            if total_desired > current_capacity:
                total_missing += utils.cdiv(total_desired - current_capacity, self.page_size)
                
        return total_missing

    def _distribute_allocated_pages(self, allocated_ids: List[int]):
        """Pops logical page IDs from batch and appends onto requests, updating prefix cache where appropriate."""
        from tunix.generate import utils
        allocated_queue = collections.deque(allocated_ids)
        
        for req in self.running_requests:
            current_capacity = len(req.page_ids) * self.page_size
            total_desired = req.seq_len + 1
            
            if total_desired > current_capacity:
                needed = utils.cdiv(total_desired - current_capacity, self.page_size)
                
                # Retrieve the full hash chain to identify what hashes these new blocks represent
                full_tokens = req.prompt_tokens + req.generated_tokens
                req_hashes = self._chunk_and_hash(full_tokens)
                
                for _ in range(needed):
                    new_pid = allocated_queue.popleft()
                    
                    # Associate this new block ID with its prefix chunk hash
                    chunk_idx = len(req.page_ids)
                    if chunk_idx < len(req_hashes):
                        block_hash = req_hashes[chunk_idx]
                        self.prefix_hash_to_page_id[block_hash] = new_pid
                    
                    req.page_ids.append(new_pid)
                    self.touch_page(new_pid)
