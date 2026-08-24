import dataclasses

import collections
from typing import List, Dict, Tuple
from tunix.generate.cache_manager import CacheManager
from tunix.generate import utils



class Request:
  def __init__(self, req_id: str, prompt_token_ids: List[int]):
    self.request_id = req_id
    self.token_ids = prompt_token_ids
    self.page_ids = []
    
    self.num_completed_tokens = 0
    self.num_in_flight_tokens = 0
    self.is_decode = False
    self.is_chunked_prefill = True 
    self.is_prefill = True

class Scheduler:
  """Python-based continuous batching scheduler."""
  def __init__(self, cache_manager: CacheManager, page_size: int, max_num_seqs: int, max_num_batch_tokens):
    self.cache_manager = cache_manager
    self.page_size = page_size
    self.max_num_seqs = max_num_seqs
    self.max_num_batch_tokens = max_num_batch_tokens
    self.token_budget = max_num_batch_tokens
    
    self.running_requests = collections.deque()
    self.pending_requests = collections.deque()
    
    self.prefix_hash_to_page_id: Dict[int, int] = {}
    self.page_ref_counts: Dict[int, int] = {}
    self.page_location: Dict[int, str] = {} # "tpu" or "cpu"
    
    self.unreferenced_tpu_pages = collections.OrderedDict()
    self.unreferenced_cpu_pages = collections.OrderedDict()

  def _touch_page(self, page_id: int):
    """Mark that a page was referenced."""
    if page_id not in self.page_ref_counts:
      self.page_ref_counts[page_id] = 1
    else:
      self.page_ref_counts[page_id] += 1
      if page_id in self.unreferenced_tpu_pages:
          del self.unreferenced_tpu_pages[page_id]
      elif page_id in self.unreferenced_cpu_pages:
          del self.unreferenced_cpu_pages[page_id]

  def _release_page(self, page_id: int):
    self.page_ref_counts[page_id] -= 1
    
    if self.page_ref_counts[page_id] > 0:  
      return 
    
    if self.page_location.get(page_id) == "tpu":
      self.unreferenced_tpu_pages[page_id] = None
    
    else:
      self.unreferenced_cpu_pages[page_id] = None

  def _free_up_unreferenced_tpu_space(self, num_pages: int):
    if num_pages <= 0: return
    
    n_unref_pages = len(self.unreferenced_tpu_pages)
    if num_pages > n_unref_pages:
      raise RuntimeError(
        f"Scheduler is attempting to free {num_pages} HBM pages, "
        f"but only {n_unref_pages} HBM pages are available."
      )
    
    pages_to_offload = []
    for _ in range(num_pages):
      pid, _ = self.unreferenced_tpu_pages.popitem(last=False)
      pages_to_offload.append(pid)
      self.page_location[pid] = "cpu"
      self.unreferenced_cpu_pages[pid] = None 
        
    if pages_to_offload:
      self.cache_manager.offload(pages_to_offload)

  def _free_up_unreferenced_cpu_space(self, num_pages: int):
    if num_pages <= 0: return
    
    n_unref_pages = len(self.unreferenced_cpu_pages)
    if num_pages > n_unref_pages:
      raise RuntimeError(
        f"Scheduler is attempting to free {num_pages} host pages, "
        f"but only {n_unref_pages} host pages are available."
      )
    
    pages_to_evict = []
    for _ in range(num_pages):
      pid, _ = self.unreferenced_cpu_pages.popitem(last=False)
      pages_to_evict.append(pid)
      del self.page_location[pid]
      del self.page_ref_counts[pid]
        
    self.cache_manager.evict(pages_to_evict)

  def schedule_step(self, new_requests: List[Request]) -> Tuple[List[Request], List[Request]]:
    """
    Determine which requests should be sampled during the next step, and ensure their pages
    are loaded on the TPU.
    """
    
    self.token_budget = self.max_num_batch_tokens
    self._queue_new_requests(new_requests)
    self._make_room_for_step()
    self._drain_pending_queue()
    
    total_new_pages_needed = self._calculate_new_pages_needed()
    if total_new_pages_needed > 0:
        assigned_page_ids = self.cache_manager.allocate(total_new_pages_needed)
        self._distribute_allocated_pages(assigned_page_ids)
    
    max_pages = self.max_num_batch_tokens  

    req_ids = [req.request_id for req in self.running_requests]
    n_new_pages = [len(req.page_ids) - utils.cdiv(req.num_completed_tokens, self.page_size) for req in self.running_requests]
    new_page_ids = [req.page_ids[-n_new_pages[i]:] if n_new_pages[i] > 0 else [] for (i, req) in enumerate(self.running_requests)]
    self.cache_manager.assign(new_page_ids)
    
    return self.running_requests 

  def _queue_new_requests(self, new_requests: List[Request]):
    for req in new_requests:
        self.pending_requests.append(req)
  
  def _preempt(self):
    preempted_request = self.running_requests.pop()
    preempted_request.is_prefill = True
    # Release pages in reverse order so that right-most pages are reused 
    # first (this allows left-most pages to be prefixed-matched).
    for pid in reversed(preempted_request.page_ids):
        self._release_page(pid)
    
    self.pending_requests.appendleft(preempted_request)

  def _make_room_for_step(self):
    """
    Preempt running requests untill sufficent TPU pages are available for a sampling step to take place.
    """
    if len(self.running_requests) == 0:
      return

    # chunked prefill
    n_free_pages = self.cache_manager.available_tpu_pages
    
    n_pages_available = n_free_pages + len(self.unreferenced_tpu_pages)
    n_new_pages_required = 0

    # Admit sequences while space allows    
    n_running_admitted = 0
    while self.token_budget > 0 and n_running_admitted != len(self.running_requests):
      candidate_req = self.running_requests[n_running_admitted]
      
      max_n_tokens_to_compute = (
          1 + len(candidate_req.token_ids) - candidate_req.num_completed_tokens
      )
      n_tokens_to_compute = min(self.token_budget, max_n_tokens_to_compute)
        
      seq_n_pages_required = utils.cdiv(n_tokens_to_compute, self.page_size)
      seq_n_pages_allocated = len(candidate_req.page_ids)
      seq_n_new_pages_required = seq_n_pages_required - seq_n_pages_allocated 
      
      if seq_n_new_pages_required > n_pages_available:
        self._preempt() 
        continue
      
      candidate_req.num_in_flight_tokens = n_tokens_to_compute
      n_pages_available -= seq_n_new_pages_required
      n_new_pages_required += seq_n_new_pages_required
      self.token_budget -= n_tokens_to_compute
      n_running_admitted += 1      
    
    # E.g. Error should be thrown if deadlock occurs 

    pages_to_free = max(0, n_new_pages_required - n_free_pages)
    self._free_up_unreferenced_tpu_space(pages_to_free)

  def _chunk_and_hash(self, tokens: List[int]) -> List[int]:
    """Returns the list of block hashes for a full sequence of tokens."""
    hashes = []
    parent_hash = 0
    for i in range(0, len(tokens), self.page_size):
        chunk = tuple(tokens[i:i+self.page_size])
        parent_hash = hash((parent_hash, chunk))
        hashes.append(parent_hash)
    return hashes


  def _get_matched_pages(self, request: Request):
    req_hashes = self._chunk_and_hash(request.token_ids)
    
   
    matched_page_ids = []
    for h in req_hashes:
      if h not in self.prefix_hash_to_page_id:
        return matched_page_ids
      
      pid = self.prefix_hash_to_page_id[h]
      
      if self.page_location.get(pid) is None:
        # Skip evicted pages
        del self.prefix_hash_to_page_id[h]
        break

      matched_page_ids.append(pid)

    return matched_page_ids

    
  def _drain_pending_queue(self):
    """
    Admit sequences while TPU space is available.
    """
    free_tpu = self.cache_manager.available_tpu_pages + len(self.unreferenced_tpu_pages)
    pages_to_load = set()

    while self.token_budget > 0 and self.pending_requests:
        req = self.pending_requests[0]
        matched_page_ids = self._get_matched_pages(req)

        n_tokens = len(req.token_ids)
        n_matched_tokens = len(matched_page_ids) * self.page_size

        max_n_tokens_to_compute = 1 + n_tokens - n_matched_tokens
        n_tokens_to_compute = min(self.token_budget, max_n_tokens_to_compute)

        n_tokens_to_load = n_tokens_to_compute + n_matched_tokens
        total_pages_needed = utils.cdiv(n_tokens_to_load, self.page_size)

        matched_page_ids = self._get_matched_pages(req)
        new_pages_needed = total_pages_needed - len(matched_page_ids)
        

        # CPU pages will need to be loaded onto the TPU
        cpu_pages_used = 0
        for pid in matched_page_ids:
            if self.page_location.get(pid) == "cpu" and pid not in pages_to_load:
                cpu_pages_used += 1
         
        total_hbm_cost = new_pages_needed + cpu_pages_used
        if free_tpu < total_hbm_cost:
          break

        self.pending_requests.popleft()
        
        req.num_in_flight_tokens = n_tokens_to_load 
        req.page_ids = []
        for pid in matched_page_ids:
            if self.page_location.get(pid) == "cpu":
                pages_to_load.add(pid)

            self._touch_page(pid)
            req.page_ids.append(pid)
        
        self.running_requests.append(req)
        free_tpu -= total_hbm_cost
            
    physically_free = self.cache_manager.available_tpu_pages
    if len(pages_to_load) > physically_free:
        self._free_up_unreferenced_tpu_space(len(pages_to_load) - physically_free)

    self.cache_manager.load(list(pages_to_load))
    for pid in pages_to_load:
        self.page_location[pid] = "tpu"

  def _calculate_new_pages_needed(self) -> int:
    """Sums up the missing boundary pages for all sequences in `running_requests`."""
    total_missing = 0
    for req in self.running_requests:
        current_capacity = len(req.page_ids) * self.page_size
        total_desired = len(req.token_ids) + 1
        
        if total_desired > current_capacity:
            total_missing += utils.cdiv(total_desired - current_capacity, self.page_size)
            
    return total_missing

  def _distribute_allocated_pages(self, allocated_ids: List[int]):
    """Pops logical page IDs from batch and appends onto requests, updating prefix cache where appropriate."""
    allocated_queue = collections.deque(allocated_ids)
    
    for req in self.running_requests:
        current_capacity = len(req.page_ids) * self.page_size
        total_desired = req.num_in_flight_tokens
          
        if total_desired <= current_capacity:
          continue
        
        needed = utils.cdiv(total_desired - current_capacity, self.page_size)
        
        # Retrieve the full hash chain to identify what hashes these new blocks represent
        full_tokens = req.token_ids
        req_hashes = self._chunk_and_hash(full_tokens)
        n_full_pages = len(full_tokens) // self.page_size
        
        for _ in range(needed):
            new_pid = allocated_queue.popleft()
            self.page_location[new_pid] = "tpu"
            
            # Associate this new block ID with its prefix chunk hash
            chunk_idx = len(req.page_ids)
            if chunk_idx < n_full_pages:
                block_hash = req_hashes[chunk_idx]
                self.prefix_hash_to_page_id[block_hash] = new_pid
            
            req.page_ids.append(new_pid)
            self._touch_page(new_pid)
