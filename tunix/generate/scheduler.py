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

"""A Scheduler for rollout requests in Tunix."""

import collections
import logging
from typing import Dict, List, Tuple, OrderedDict, Deque
from tunix.generate import utils
from tunix.generate.tiered_page_pool import TieredPagePoolManager

logger = logging.getLogger(__name__)
logging.disable(logging.CRITICAL)

class Request:
  """A request to be scheduled."""

  def __init__(self, req_id: str, prompt_token_ids: List[int]):
    self.request_id = req_id
    self.prompt_length = len(prompt_token_ids)

    self.token_ids = prompt_token_ids
    self.logprobs = []
    self.logits = []

    self.last_page_hash = 0
    self.page_ids = []

    self.num_completed_tokens = 0
    self.num_in_flight_tokens = 0

    self.is_done = False
    self.is_decode = False


class Scheduler:
  """A continuous batching scheduler."""

  def __init__(
      self,
      kv_cache_manager: TieredPagePoolManager,
      max_num_batch_tokens: int,
      max_seqs_per_batch: int,
      max_tokens_to_generate: int,
      eos_token_ids: int,
  ):
    # --- Scheduling parameters --
    self.max_seqs_per_batch: int = max_seqs_per_batch
    self.max_num_batch_tokens: int = max_num_batch_tokens
    self.max_tokens_to_generate: int = max_tokens_to_generate
    self.eos_token_ids: int = eos_token_ids

    # --- Seq state ---
    self._running_requests: Deque[Request] = collections.deque()
    self._pending_requests: Deque[Request] = collections.deque()

    # --- KV cache state ---
    self._kv_cache_manager: TieredPagePoolManager = kv_cache_manager

    self._prefix_hash_to_page_id: Dict[int, int] = {}
    self._page_ref_counts: Dict[int, int] = {}

    self._unreferenced_tpu_page_ids: OrderedDict[int, None] = (
        collections.OrderedDict()
    )
    self._unreferenced_cpu_page_ids: OrderedDict[int, None] = (
        collections.OrderedDict()
    )

  # ----------- Page state getters -----------
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

  # ----------- Page management -----------
  def _touch_page(self, page_id: int):
    """Mark that a page was referenced."""
    logger.info(f"Touched page {page_id}")
    if page_id not in self._page_ref_counts:
      self._page_ref_counts[page_id] = 1
    else:
      self._page_ref_counts[page_id] += 1
      if page_id in self._unreferenced_tpu_page_ids:
        del self._unreferenced_tpu_page_ids[page_id]
      elif page_id in self._unreferenced_cpu_page_ids:
        del self._unreferenced_cpu_page_ids[page_id]

  def _release_page(self, page_id: int):
    """Release a reference to a page."""
    logger.info(f"Released page {page_id}")

    assert page_id in self._page_ref_counts
    assert self._page_ref_counts[page_id] > 0

    self._page_ref_counts[page_id] -= 1

    if self._page_ref_counts[page_id] > 0:
      return

    if self._get_page_location(page_id) == "tpu":
      self._unreferenced_tpu_page_ids[page_id] = None
    else:
      self._unreferenced_cpu_page_ids[page_id] = None

  def _allocate(self, request: Request, num_pages: int):
    """Allocate TPU pages for a request."""
    full_tokens = request.token_ids
    allocated_pages = self._kv_cache_manager.allocate_tpu_pages(num_pages)
    logger.info(f"Allocated {num_pages} pages")


    for pid in allocated_pages:
      chunk_idx = len(request.page_ids)
      request.page_ids.append(pid)
      self._touch_page(pid)

  def _free_up_unreferenced_tpu_space(self, num_pages: int):
    """Free unreferenced TPU pages, offloading to CPU where possible."""
    if num_pages <= 0:
      return
  
    assert num_pages <= len(self._unreferenced_tpu_page_ids)

    cpu_shortfall = num_pages - self._num_free_cpu_pages
    if cpu_shortfall > 0:
      evictable = min(cpu_shortfall, len(self._unreferenced_cpu_page_ids))
      self._free_up_unreferenced_cpu_space(evictable)

    pages_to_offload = []
    pages_to_discard = []

    for _ in range(num_pages):
      pid, _ = self._unreferenced_tpu_page_ids.popitem(last=False)

      if self._num_free_cpu_pages > 0:
        pages_to_offload.append(pid)
        self._unreferenced_cpu_page_ids[pid] = None
      else:
        pages_to_discard.append(pid)

    if pages_to_offload:
      self._kv_cache_manager.offload(pages_to_offload)

    if pages_to_discard:
      self._kv_cache_manager.evict(pages_to_discard)
    
    logger.info(
      f"Freed {num_pages} unreferenced tpu pages."
      f"{len(pages_to_offload)} pages were offloaded to the cpu."
      f"{len(pages_to_discard)} pages were discarded."
    )


  def _free_up_unreferenced_cpu_space(self, num_pages: int):
    """Free unreferenced cpu pages."""
    if num_pages <= 0:
      return

    n_unref_pages = len(self._unreferenced_cpu_page_ids)
    assert num_pages <= n_unref_pages

    pages_to_evict = []
    for _ in range(num_pages):
      pid, _ = self._unreferenced_cpu_page_ids.popitem(last=False)
      pages_to_evict.append(pid)
      del self._page_ref_counts[pid]

    self._kv_cache_manager.evict(pages_to_evict)
    logger.info(
      f"Freed {num_pages} unreferenced cpu pages."
    )

  # ----------- Request helpers -----------
  def _chunk_and_hash(self, tokens: List[int], start_hash: int = 0) -> List[int]:
    """Returns the list of block hashes for a full sequence of tokens."""
    hashes = []
    parent_hash = start_hash
    for i in range(0, len(tokens), self._page_size):
      chunk = tuple(tokens[i : i + self._page_size])
      parent_hash = hash((parent_hash, chunk))
      hashes.append(parent_hash)
    return hashes

  def _get_matched_pages(self, request: Request):
    """Returns a list of matched prefix pages for a request."""
    assert len(request.token_ids) > 0

    req_hashes = self._chunk_and_hash(request.token_ids[:-1])
    matched_page_ids = []

    for h in req_hashes:
      if h not in self._prefix_hash_to_page_id:
        break

      pid = self._prefix_hash_to_page_id[h]

      if self._get_page_location(pid) is None:
        del self._prefix_hash_to_page_id[h]
        break

      matched_page_ids.append(pid)

    logger.info(
      f"Found {len(matched_page_ids)} matched pages for Request {request.request_id}."
    )

    return matched_page_ids

  def _deduplicate_and_cache_full_pages(self, request):
    """Hashes newly completed pages.

    If a collision is found in the cache, deduplicates the page. 
    """

    n_unhashed_tokens = (
      request.num_completed_tokens % self._page_size +
      request.num_inflight_tokens
    )
  
    n_unhashed_full_pages = n_unhashed_tokens // self._page_size
    if n_unhashed_full_pages == 0:
      return 

    n_hashed_full_pages = request.num_completed_tokens // self._page_size
    n_hashed_tokens = n_hashed_full_pages * self._page_size
    
    n_new_tokens_to_hash = n_unhashed_full_pages * self._page_size
    new_tokens_to_hash = request.token_ids[n_hashed_tokens: n_hashed_tokens + n_new_tokens_to_hash]

    start_hash = request.last_page_hash
    hashes = self._chunk_and_hash(new_tokens_to_hash, start_hash)

    for i, chunk_hash in enumerate(hashes):
      p_idx = i + n_hashed_full_pages
      pid = request.page_ids[p_idx]

      if chunk_hash in self._prefix_hash_to_page_id:
        cached_pid = self._prefix_hash_to_page_id[chunk_hash]

        # Deduplicate page in case of cache hit.
        if cached_pid != pid:
          request.page_ids[p_idx] = cached_pid

          self._release_page(pid)
          self._touch_page(cached_pid)
      else:
        self._prefix_hash_to_page_id[chunk_hash] = pid

    request.last_page_hash = hashes[-1]

  def update_from_output(
      self, 
      generated_tokens: list[int], 
      logits: list | None = None, 
      logprobs: list | None = None
  ):
    completed_reqs = []
    surviving_reqs = []

    for i, req in enumerate(self._running_requests):
      new_token = generated_tokens[i]
     
      # TODO: Logprobs and logits should be stored with pages.
      # Otherwise, they may be lost because of prefix matching. 
      if logprobs is not None:
          req.logprobs.append(logprobs[i])
      if logits is not None:
          req.logits.append(logits[i])

      terminated = new_token in self.eos_token_ids       
      truncated = (
          len(req.token_ids) - req.prompt_length
      ) >= self.max_tokens_to_generate

      if terminated or truncated:
        for pid in reversed(req.page_ids):
            self._release_page(pid)
        
        if new_token not in self.eos_token_ids:
          new_token = self.eos_token_ids[0] 
        req.token_ids.append(new_token)

        completed_reqs.append(req)
      else:
        req.token_ids.append(new_token)
        self._deduplicate_and_cache_full_pages(req)
        surviving_reqs.append(req)

      req.num_completed_tokens += req.num_inflight_tokens
      req.num_inflight_tokens = 0

    self._running_requests = surviving_reqs

    return completed_reqs

  # ----------- Scheduling -----------
  def _preempt(self):
    """Remove the newest-request from the active batch."""
    preempted_request = self._running_requests.pop()
    preempted_request.last_page_hash = 0

    # Release pages in reverse order so that right-most pages are reused
    # first (this allows left-most pages to still be prefixed-matched).
    for pid in reversed(preempted_request.page_ids):
      self._release_page(pid)

    self._pending_requests.appendleft(preempted_request)

    logger.info(
      f"Preempted request {preempted_request.request_id}."
    )

  def schedule_step(
      self, new_requests: List[Request]
  ) -> Tuple[List[Request], List[int]]:
    """Select the requests to sample in the next step, and ensure their pages are loaded
    on to the TPU.

    Requests are scheduled in order of arrival untill the batch reaches either
    `max_num_batch_tokens` compute tokens or `max_seqs_per_batch` requests.
    """

    self._num_decodes = 0
    self._num_prefills = 0

    self._token_budget = self.max_num_batch_tokens
    self._deduplicate_and_cache_full_pages()

    self._queue_new_requests(new_requests)

    self._schedule_running_sequences()
    self._schedule_pending_sequences()
    
    decodes = [r for r in self._running_requests if r.is_decode]
    prefills = [r for r in self._running_requests if not r.is_decode]

    # TODO: We may want to support chunked prefill in cases of long prompts.
    i = len(decodes)        # n_decodes 
    j = i + 0               # n_decodes + n_chunked
    k = j + len(prefills)   # n_decodes + n_chunked + n_prefills
    
    # The RPA kernel expects [decodes, prefills]. Genrally, 
    # running_requests will have the same order as ordered_requests.
    # Exceptions can occur if req_{n+1} prefix matched all of its
    # prompt tokens, but req_n has not.
 
    ordered_requests = decodes + prefills
    distribution = [i, j, k]

    return ordered_requests, distribution

  def _queue_new_requests(self, new_requests: List[Request]):
    """Insert new requests into the pending queue."""
    for req in new_requests:
      self._pending_requests.append(req)

  def _schedule_running_sequences(self):
    """Schedule current running sequences while TPU pages are available."""
    # TODO: Unit test token budget

    if len(self._running_requests) == 0:
      return

    n_pages_available = self._num_free_tpu_pages + len(
        self._unreferenced_tpu_page_ids
    )

    n_running_admitted = 0
    while n_running_admitted < len(self._running_requests):
      if (
          self._token_budget <= 0
          or n_running_admitted >= self.max_seqs_per_batch
      ):
        break

      req = self._running_requests[n_running_admitted]

      # --- Compute page requirements ---
      n_tokens_to_compute = len(req.token_ids) - req.num_completed_tokens
      if n_tokens_to_compute > self._token_budget:
        break

      n_total_pages_needed = utils.cdiv(
          req.num_completed_tokens + n_tokens_to_compute, self._page_size
      )
      n_new_pages_needed = n_total_pages_needed - len(req.page_ids)

      # If there is an insufficent amount of available pages,
      # we preempt the most recently scheduled request, and
      # retry Scheduling.
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

      req.is_decode = True 
      logger.info(
        f"Scheduled decode request {req.request_id}."
      )

    # Verify that deadlock did not occur.
    assert n_running_admitted > 0
    
    # We must preempt the remaining unscheduled requests if the token
    # budget is exceeded.
    while n_running_admitted < len(self._running_requests):
      self._preempt()

  def _schedule_pending_sequences(self):
    """Admit pending sequences while TPU space is available."""
    n_free_tpu = self._num_free_tpu_pages + len(self._unreferenced_tpu_page_ids)
    pages_to_load = set()

    while self._pending_requests:
      if (
          self._token_budget <= 0
          or len(self._running_requests) >= self.max_seqs_per_batch
      ):
        break

      req = self._pending_requests[0]

      # --- Compute page requirements ---
      matched_page_ids = self._get_matched_pages(req)

      n_tokens = len(req.token_ids)
      n_matched_tokens = len(matched_page_ids) * self._page_size
      
      n_tokens_to_compute = n_tokens - n_matched_tokens
      if n_tokens_to_compute > self._token_budget:
        break

      n_tokens_to_load = n_tokens_to_compute + n_matched_tokens
      n_total_pages_needed = utils.cdiv(n_tokens_to_load, self._page_size)
      n_new_pages_needed = n_total_pages_needed - len(matched_page_ids)

      n_cpu_pages_used = sum(
          1
          for pid in matched_page_ids
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
      req.num_in_flight_tokens = n_tokens_to_compute
      req.num_completed_tokens = n_matched_tokens

      n_free_tpu -= n_total_tpu_cost
      
 
      if (
        n_matched_tokens >= req.prompt_length and 
        n_tokens_to_compute == 1
      ):
        req.is_decode = True
        logger.info(
          f"Scheduled decode request {req.request_id}."
        )
      else:
        req.is_decode = False 
        logger.info(
          f"Scheduled {n_tokens_to_compute} tokens for prefill request {req.request_id}."
        )

    n_physically_free = self._num_free_tpu_pages
    n_pages_to_load = len(pages_to_load)

    if n_pages_to_load > n_physically_free:
      shortfall = n_pages_to_load - n_physically_free
      self._free_up_unreferenced_tpu_space(shortfall)

    # Load all matched CPU pages on to the TPU.
    self._kv_cache_manager.load(list(pages_to_load))

