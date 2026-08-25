# Copyright 2026 The Tunix Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A scheduler for rollout requests in Tunix."""

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
    
    self.last_page_hash = 0 
    self.num_completed_tokens = 0
    self.num_in_flight_tokens = 0
    self.is_decode = False
    self.is_chunked_prefill = True 
    self.is_prefill = True

class Scheduler:
  """A continuous batching scheduler."""
  def __init__(
    self, 
    cache_manager: CacheManager, 
    max_num_batch_tokens: int
  ):
    self.cache_manager = cache_manager
    self.page_size = cache_manager.page_size
    self.max_num_seqs = cache_manager.max_num_seqs
    self.max_num_batch_tokens = max_num_batch_tokens
    
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
    """Release a reference to a page."""
    self.page_ref_counts[page_id] -= 1
    
    if self.page_ref_counts[page_id] > 0:  
      return 
    
    if self.page_location.get(page_id) == "tpu":
      self.unreferenced_tpu_pages[page_id] = None
    
    else:
      self.unreferenced_cpu_pages[page_id] = None

  def _allocate(self, request: Request, num_pages: int):
    full_tokens = request.token_ids
    req_hashes = self._chunk_and_hash(full_tokens)
    n_full_pages = len(full_tokens) // self.page_size
    allocated_pages = self._cache_manager.allocate_tpu_pages(num_pages)

    for pid in allocated_pages:
      self.page_location[] = "tpu"
      
      # Associate this new block ID with its prefix chunk hash
      chunk_idx = len(request.page_ids)
      if chunk_idx < n_full_pages:
        block_hash = req_hashes[chunk_idx]
        self.prefix_hash_to_page_id[block_hash] = new_pid
      
      req.page_ids.append(new_pid)
      self._touch_page(new_pid)

  def _preempt(self):
    """Remove the newest-request from the active batch."""
    preempted_request = self.running_requests.pop()
    preempted_request.is_prefill = True
    preempted_request.last_page_hash = 0 

    # Release pages in reverse order so that right-most pages are reused 
    # first (this allows left-most pages to be prefixed-matched).
    for pid in reversed(preempted_request.page_ids):
        self._release_page(pid)
    
    self.pending_requests.appendleft(preempted_request)

  def _free_up_unreferenced_tpu_space(self, num_pages: int):
    """Free unreferenced TPU pages, offloading to CPU where possible."""
    if num_pages <= 0: 
        return

    assert num_pages <= len(self.unreferenced_tpu_pages)

    cpu_shortfall = num_pages - self.cache_manager.available_cpu_pages
    if cpu_shortfall > 0:
        evictable = min(cpu_shortfall, len(self.unreferenced_cpu_pages))
        self._free_up_unreferenced_cpu_space(evictable)

    pages_to_offload = []
    pages_to_discard = []
    available_cpu = self.cache_manager.available_cpu_pages
    
    for _ in range(num_pages):
        pid, _ = self.unreferenced_tpu_pages.popitem(last=False)
        
        if available_cpu > 0:
            pages_to_offload.append(pid)
            self.page_location[pid] = "cpu"
            self.unreferenced_cpu_pages[pid] = None
            available_cpu -= 1
        else:
          pages_to_discard.append(pid)
          self.page_location.pop(pid, None)

    if pages_to_offload:
        self.cache_manager.offload(pages_to_offload)
        
    if pages_to_discard:
        self.cache_manager.evict(pages_to_discard)

  def _free_up_unreferenced_cpu_space(self, num_pages: int):
    """Free unreferenced cpu pages."""
    if num_pages <= 0: return
    
    n_unref_pages = len(self.unreferenced_cpu_pages)
    assert(num_pages > n_unref_pages)
    
    pages_to_evict = []
    for _ in range(num_pages):
      pid, _ = self.unreferenced_cpu_pages.popitem(last=False)
      pages_to_evict.append(pid)
      del self.page_location[pid]
      del self.page_ref_counts[pid]
        
    self.cache_manager.evict(pages_to_evict)

  def schedule_step(self, new_requests: List[Request]) -> Tuple[List[Request], List[Request]]:
    """
    Select the requests to sample in the next step, and ensure their pages are loaded
    on to the TPU.

    Requests are scheduled in order of arrival untill the batch reaches either 
    `max_num_batch_tokens` compute tokens or `max_num_seqs` requests. 
    """
    
    self._token_budget = self.max_num_batch_tokens
    self._deduplicate_and_cache_full_pages()

    self._queue_new_requests(new_requests)
    self._schedule_running_sequences()
    self._schedule_pending_sequences()
    
    return self.running_requests 

  def _deduplicate_and_cache_full_pages(self) -> int:
    """
    Hashes newly completed pages. If a collision is found in the cache, 
    deduplicates the page and returns the number of freed physical pages.
    """
    for req in self.running_requests:

      n_tokens = len(req.token_ids)
      is_empty = (n_tokens == 0)
      is_partially_full = (n_tokens % self.page_size != 0)
      
      if is_empty or is_partially_full:
          continue
      
      n_full_pages = len(req.token_ids) // self.page_size 
      last_full_idx = n_full_pages - 1
      pid = req.page_ids[last_full_idx]
      
      chunk = tuple(tokens[i:i+self.page_size])
      chunk_hash = hash(request.last_page_hash, chunk)
      request.last_page_hash = chunk_hash 
      
      if chunk_hash in self.prefix_hash_to_page_id:
        cached_pid = self.prefix_hash_to_page_id[chunk_hash]
        if cached_pid != pid:
          # Deduplicate page in case of cache hit
          req.page_ids[last_full_idx] = cached_pid
          self.release_page(pid)
          self.touch_page(cached_pid)
      else:
        self.prefix_hash_to_page_id[block_hash] = pid
            
  def _queue_new_requests(self, new_requests: List[Request]):
    """Insert new requests into the pending queue."""
    for req in new_requests:
        self.pending_requests.append(req)
  
  
  def _schedule_running_sequences(self):
    """
    Schedule current running sequences while TPU pages are available.
    """
    if len(self.running_requests) == 0:
      return

    n_free_pages = self.cache_manager.available_tpu_pages
    
    n_pages_available = n_free_pages + len(self.unreferenced_tpu_pages)
    n_new_pages_required = 0
    
    # TODO: Unit test token budget
    # Admit sequences while space allows    
    n_running_admitted = 0
    while self._token_budget > 0 and n_running_admitted < len(self.running_requests):
      candidate_req = self.running_requests[n_running_admitted]
      
      max_n_tokens_to_compute = (
          1 + len(candidate_req.token_ids) - candidate_req.num_completed_tokens
      )
      n_tokens_to_compute = min(self._token_budget, max_n_tokens_to_compute)
        
      seq_n_pages_required = utils.cdiv(n_tokens_to_compute, self.page_size)
      seq_n_pages_allocated = len(candidate_req.page_ids)
      seq_n_new_pages_required = seq_n_pages_required - seq_n_pages_allocated 
      
      if seq_n_new_pages_required > n_pages_available:
        self._preempt() 
        continue
      
      candidate_req.num_in_flight_tokens = n_tokens_to_compute
      n_pages_available -= seq_n_new_pages_required
      n_new_pages_required += seq_n_new_pages_required
      self._token_budget -= n_tokens_to_compute
      n_running_admitted += 1 

      self._allocate(candidate_req, n_new_pages_required) 
    
    assert(len(self.running_requests) > 0)

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
    """Returns a list of matched prefix pages for a request."""
    req_hashes = self._chunk_and_hash(request.token_ids)
    matched_page_ids = []

    for h in req_hashes:
      if h not in self.prefix_hash_to_page_id:
        break 
      
      pid = self.prefix_hash_to_page_id[h]
      
      if self.page_location.get(pid) is None:
        del self.prefix_hash_to_page_id[h]
        break

      matched_page_ids.append(pid)
      request.last_page_hash = h 

    return matched_page_ids

    
  def _schedule_pending_sequences(self):
    """
    Admit pending sequences while TPU space is available.
    """
    free_tpu = self.cache_manager.available_tpu_pages + len(self.unreferenced_tpu_pages)
    pages_to_load = set()

    while self._token_budget > 0 and self.pending_requests:
        req = self.pending_requests[0]
        matched_page_ids = self._get_matched_pages(req)

        n_tokens = len(req.token_ids)
        n_matched_tokens = len(matched_page_ids) * self.page_size

        max_n_tokens_to_compute = 1 + n_tokens - n_matched_tokens
        n_tokens_to_compute = min(self._token_budget, max_n_tokens_to_compute)

        n_tokens_to_load = n_tokens_to_compute + n_matched_tokens
        total_pages_needed = utils.cdiv(n_tokens_to_load, self.page_size)

        matched_page_ids = self._get_matched_pages(req)
        n_new_pages_needed = total_pages_needed - len(matched_page_ids)
        
        # Any CPU pages used by the request, need to be loaded onto the TPU
        n_cpu_pages_used = 0
        for pid in matched_page_ids:
            if self.page_location.get(pid) == "cpu" and pid not in pages_to_load:
                cpu_pages_used += 1
         
        total_hbm_cost = n_new_pages_needed + n_cpu_pages_used
        if free_tpu < total_hbm_cost:
          break

        self.pending_requests.popleft()
        self._token_budget -= n_tokens_to_compute
        self._allocate(req, n_new_pages_required) 

        req.num_in_flight_tokens = n_tokens_to_load 
        req.page_ids = []
        cache_seq_slot = len(self.running_requests)

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
  
  
