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

import collections
from typing import List, Dict, Tuple
from tunix.generate.tiered_page_pool.py import TieredPagePoolManager
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
    kv_cache_manager: TieredPagePoolManager, 
    max_num_batch_tokens: int,
    max_seqs_per_batch: int,
  ):
    self._kv_cache_manager = kv_cache_manager
    
    self.max_seqs_per_batch = max_seqs_per_batch 
    self.max_num_batch_tokens = max_num_batch_tokens
    
    self._running_requests = collections.deque()
    self._pending_requests = collections.deque()
    
    self._prefix_hash_to_page_id: Dict[int, int] = {}
    self._page_ref_counts: Dict[int, int] = {}
    
    self._unreferenced_tpu_pages = collections.OrderedDict()
    self._unreferenced_cpu_pages = collections.OrderedDict()

  @property
  def _num_free_tpu_pages(self):
    return self._kv_cache_manager.num_free_tpu_pages

  @property
  def _num_free_cpu_pages(self):
    return self._kv_cache_manager.num_free_cpu_pages

  @property
  def _page_size(self):
    return self._kv_cache_manager.page_size

  def _get_page_location(self, page_id):
    return self._kv_cache_manager.get_page_location(page_id)

  def _touch_page(self, page_id: int):
    """Mark that a page was referenced."""
    if page_id not in self._page_ref_counts:
      self._page_ref_counts[page_id] = 1
    else:
      self._page_ref_counts[page_id] += 1
      if page_id in self._unreferenced_tpu_pages:
          del self._unreferenced_tpu_pages[page_id]
      elif page_id in self._unreferenced_cpu_pages:
          del self._unreferenced_cpu_pages[page_id]

  def _release_page(self, page_id: int):
    """Release a reference to a page."""
    self._page_ref_counts[page_id] -= 1
    
    if self._page_ref_counts[page_id] > 0:  
      return 
    
    if self._get_page_location(page_id) == "tpu":
      self._unreferenced_tpu_pages[page_id] = None
    
    else:
      self._unreferenced_cpu_pages[page_id] = None

  def _allocate(self, request: Request, num_pages: int):
    """Allocate TPU pages for a request."""
    full_tokens = request.token_ids
    req_hashes = self._chunk_and_hash(full_tokens)
    n_full_pages = len(full_tokens) // self._page_size
    allocated_pages = self._kv_cache_manager.allocate_tpu_pages(num_pages)

    for pid in allocated_pages:
      chunk_idx = len(request.page_ids)
      
      if chunk_idx < n_full_pages:
        block_hash = req_hashes[chunk_idx]
        self._prefix_hash_to_page_id[block_hash] = pid
      
      request.page_ids.append(pid)
      self._touch_page(pid)

  def _preempt(self):
    """Remove the newest-request from the active batch."""
    preempted_request = self._running_requests.pop()
    preempted_request.is_prefill = True
    preempted_request.last_page_hash = 0 

    # Release pages in reverse order so that right-most pages are reused 
    # first (this allows left-most pages to still be prefixed-matched).
    for pid in reversed(preempted_request.page_ids):
        self._release_page(pid)
    
    self._pending_requests.appendleft(preempted_request)

  def _free_up_unreferenced_tpu_space(self, num_pages: int):
    """Free unreferenced TPU pages, offloading to CPU where possible."""
    if num_pages <= 0: 
        return
    
    assert(num_pages <= len(self._unreferenced_tpu_pages))

    cpu_shortfall = num_pages - self._num_free_cpu_pages
    if cpu_shortfall > 0:
        evictable = min(cpu_shortfall, len(self._unreferenced_cpu_pages))
        self._free_up_unreferenced_cpu_space(evictable)

    pages_to_offload = []
    pages_to_discard = []
    
    for _ in range(num_pages):
        pid, _ = self._unreferenced_tpu_pages.popitem(last=False)
        
        if self._num_free_cpu_pages > 0:
            pages_to_offload.append(pid)
            self._unreferenced_cpu_pages[pid] = None
        else:
          pages_to_discard.append(pid)

    if pages_to_offload:
        self._kv_cache_manager.offload(pages_to_offload)
        
    if pages_to_discard:
        self._kv_cache_manager.evict(pages_to_discard)

  def _free_up_unreferenced_cpu_space(self, num_pages: int):
    """Free unreferenced cpu pages."""
    if num_pages <= 0: return
    
    n_unref_pages = len(self._unreferenced_cpu_pages)
    assert(num_pages <= n_unref_pages)
    
    pages_to_evict = []
    for _ in range(num_pages):
      pid, _ = self._unreferenced_cpu_pages.popitem(last=False)
      pages_to_evict.append(pid)
      del self._page_ref_counts[pid]
        
    self._kv_cache_manager.evict(pages_to_evict)

  def schedule_step(self) -> Tuple[List[Request], List[int]]:
    """
    Select the requests to sample in the next step, and ensure their pages are loaded
    on to the TPU.

    Requests are scheduled in order of arrival untill the batch reaches either 
    `max_num_batch_tokens` compute tokens or `max_seqs_per_batch` requests. 
    """
    
    self._token_budget = self.max_num_batch_tokens
    self._deduplicate_and_cache_full_pages()

    self._queue_new_requests(new_requests)
    self._schedule_running_sequences()
    self._schedule_pending_sequences()

    n_decodes = 0
    n_prefills_completing = 0
    n_chunked_prefills = 0
    
    for r in self._running_requests:
        total_prompt_tokens = len(r.token_ids)
        n_completed = r.num_completed_tokens
        n_in_flight = r.num_in_flight_tokens

        if n_completed >= total_prompt_tokens:
          n_decodes += 1
        elif n_completed + n_in_flight >= total_prompt_tokens:
          n_prefills_completing += 1
        else:
          n_chunked_prefills += 1

    i = n_decodes
    j = i + n_prefills_completing
    k = j + n_chunked_prefills
    distribution = [i, j, k]

    return list(self._running_requests), distribution

  def _deduplicate_and_cache_full_pages(self) -> int:
    """
    Hashes newly completed pages. If a collision is found in the cache, 
    deduplicates the page and returns the number of freed physical pages.
    """

    # TODO: We are not handling duplications that occur during prefill
    # when multiple pages are completed at once. 
    for req in self._running_requests:
      n_tokens = len(req.token_ids)
      is_empty = (n_tokens == 0)
      is_partially_full = (n_tokens % self._page_size != 0)
      
      # We only hash full pages 
      if is_empty or is_partially_full:
          continue
      
      n_full_pages = n_tokens // self._page_size 
      last_full_idx = n_full_pages - 1
      pid = req.page_ids[last_full_idx]
      
      chunk = tuple(req.token_ids[-self._page_size:])
      chunk_hash = hash((req.last_page_hash, chunk))
      req.last_page_hash = chunk_hash 
      
      if chunk_hash in self._prefix_hash_to_page_id:
        cached_pid = self._prefix_hash_to_page_id[chunk_hash]
        if cached_pid != pid:
          # Deduplicate page in case of cache hit
          req.page_ids[last_full_idx] = cached_pid
          self._release_page(pid)
          self._touch_page(cached_pid)
      else:
        self._prefix_hash_to_page_id[chunk_hash] = pid
            
  def _queue_new_requests(self, new_requests: List[Request]):
    """Insert new requests into the pending queue."""
    for req in new_requests:
        self._pending_requests.append(req)
  
  def _schedule_running_sequences(self):
    """Schedule current running sequences while TPU pages are available."""
    # TODO: Unit test token budget
    # TODO: Handle dones here instead of inside engine? 

    if len(self._running_requests) == 0:
      return

    n_pages_available = self._num_free_tpu_pages + len(self._unreferenced_tpu_pages)

    n_running_admitted = 0
    while n_running_admitted < len(self._running_requests):
      if self._token_budget <= 0 or n_running_admitted >= self.max_seqs_per_batch:
        break

      req = self._running_requests[n_running_admitted]
      
      # --- Compute page requirements ---  
      n_tokens_remaining = 1 + len(req.token_ids) - req.num_completed_tokens
      n_tokens_to_compute = min(self._token_budget, n_tokens_remaining)
      
      n_total_pages_needed = utils.cdiv(
          req.num_completed_tokens + n_tokens_to_compute, 
          self._page_size
      )
      n_new_pages_needed = n_total_pages_needed - len(req.page_ids)
      
      # If there is an insufficent amount of available pages,  
      # we preempt the most recently scheduled request, and
      # retry scheduling.
      if n_new_pages_needed > n_pages_available:
        self._preempt() 
        continue
      
      # --- Assign new pages to request ---  
      if n_new_pages_needed > self._num_free_tpu_pages:
        shortfall = n_new_pages_needed - self._num_free_tpu_pages 
        self._free_up_unreferenced_tpu_space(shortfall)

      self._allocate(req, n_new_pages_needed) 
      
      # --- Update state ---
      req.num_in_flight_tokens = n_tokens_to_compute
      n_pages_available -= n_new_pages_needed 
      self._token_budget -= n_tokens_to_compute
      n_running_admitted += 1 
    
    # Verify that deadlock did not occur 
    assert(len(self._running_requests) > 0)

  def _chunk_and_hash(self, tokens: List[int]) -> List[int]:
    """Returns the list of block hashes for a full sequence of tokens."""
    hashes = []
    parent_hash = 0
    for i in range(0, len(tokens), self._page_size):
        chunk = tuple(tokens[i:i+self._page_size])
        parent_hash = hash((parent_hash, chunk))
        hashes.append(parent_hash)
    return hashes


  def _get_matched_pages(self, request: Request):
    """Returns a list of matched prefix pages for a request."""
    req_hashes = self._chunk_and_hash(request.token_ids)
    matched_page_ids = []

    for h in req_hashes:
      if h not in self._prefix_hash_to_page_id:
        break 
      
      pid = self._prefix_hash_to_page_id[h]
      
      if self._get_page_location(pid) is None:
        del self._prefix_hash_to_page_id[h]
        break

      matched_page_ids.append(pid)
      request.last_page_hash = h 

    return matched_page_ids

  def _schedule_pending_sequences(self):
    """
    Admit pending sequences while TPU space is available.
    """
    n_free_tpu = self._num_free_tpu_pages + len(self._unreferenced_tpu_pages)
    pages_to_load = set()

    while self._pending_requests:
      if self._token_budget <= 0 or len(self._running_requests) >= self.max_seqs_per_batch:
        break

      req = self._pending_requests[0]
      
      # --- Compute page requirements ---
      matched_page_ids = self._get_matched_pages(req)

      n_tokens = len(req.token_ids)
      n_matched_tokens = len(matched_page_ids) * self._page_size

      n_tokens_remaining = 1 + n_tokens - n_matched_tokens
      n_tokens_to_compute = min(self._token_budget, n_tokens_remaining)
      
      n_tokens_to_load = n_tokens_to_compute + n_matched_tokens
      n_total_pages_needed = utils.cdiv(n_tokens_to_load, self._page_size)
      n_new_pages_needed = n_total_pages_needed - len(matched_page_ids)

      n_cpu_pages_used = sum(
          1 for pid in matched_page_ids
          if self._get_page_location(pid) == "cpu" and pid not in pages_to_load
      )
      n_total_tpu_cost = n_new_pages_needed + n_cpu_pages_used

      if n_free_tpu < n_total_tpu_cost:
          break
      
      # --- Assign pages to request -- 
      req.page_ids = []
      for pid in matched_page_ids:
          if self._get_page_location(pid) == "cpu":
              pages_to_load.add(pid)

          self._touch_page(pid)
          req.page_ids.append(pid)

      self._allocate(req, n_new_pages_needed)

      # --- Schedule request ---
      self._pending_requests.popleft()
      self._running_requests.append(req)

      self._token_budget -= n_tokens_to_compute
      req.num_in_flight_tokens = n_tokens_to_load

      n_free_tpu -= n_total_tpu_cost

    n_physically_free = self._num_free_tpu_pages
    n_pages_to_load = len(pages_to_load)
    
    if n_pages_to_load > n_physically_free:
        shortfall = n_pages_to_load - n_physically_free
        self._free_up_unreferenced_tpu_space(shortfall)
    
    # Load all matched CPU pages on to the TPU.
    self._kv_cache_manager.load(list(pages_to_load))
