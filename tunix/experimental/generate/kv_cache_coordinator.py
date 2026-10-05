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

"""KV cache coordinator for a group of layers with the same window size."""

from __future__ import annotations

import collections
from collections.abc import Sequence
import dataclasses
from typing import Any

from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import utils
import tunix.experimental.generate.tiered_page_pool as page_pool_lib


@dataclasses.dataclass
class Page:
  """A page in the KV cache."""
  page_id: int
  location: str
  ref_count: int = 0
  prefix_hash: int | None = None

  def __hash__(self) -> int:
    return self.page_id

  def __eq__(self, other: Any) -> bool:
    return isinstance(other, Page) and self.page_id == other.page_id


@dataclasses.dataclass
class _RequestState:
  """Tracks per-request page allocations and prefix hashing state."""
  pages: list[Page | None] = dataclasses.field(default_factory=list)
  num_hashed_pages: int = 0
  num_released_pages: int = 0
  last_page_hash: int = 0


class KVCacheCoordinator:
  """Coordinates KV cache page management for a group of layers with the same window size."""

  def __init__(
      self,
      page_manager: page_pool_lib.TieredPagePoolManager,
      window_size: int | None = None,
  ):
    self._page_manager: page_pool_lib.TieredPagePoolManager = page_manager
    self._window_size: int | None = window_size

    self._prefix_hash_to_page: dict[int, Page] = {}
    self._unreferenced_tpu_pages: collections.OrderedDict[Page, None] = (
        collections.OrderedDict()
    )
    self._unreferenced_cpu_pages: collections.OrderedDict[Page, None] = (
        collections.OrderedDict()
    )
    self._requests: dict[str, _RequestState] = {}

  @property
  def _page_size(self) -> int:
    return self._page_manager.page_size

  @property
  def window_size(self) -> int | None:
    return self._window_size

  def _touch_page(self, page: Page | None):
    """Increments ref count and removes page from unreferenced queues."""
    if page is None:
      return

    page.ref_count += 1
    self._unreferenced_tpu_pages.pop(page, None)
    self._unreferenced_cpu_pages.pop(page, None)

  def _release_page(self, page: Page | None):
    """Releases a reference to a page."""
    if page is None:
      return
    if page.ref_count <= 0:
      raise ValueError(f"Cannot release page {page.page_id} with no references.")

    page.ref_count -= 1
    if page.ref_count > 0:
      return

    if page.prefix_hash is None:
      # If a page was never hashed, it cannot be reused and should be freed.
      self._free_pages([page])
    elif page.location == "tpu":
      self._unreferenced_tpu_pages[page] = None
    elif page.location == "cpu":
      self._unreferenced_cpu_pages[page] = None
    else:
      raise ValueError(f"Unknown page location: {page.location}")

  def _free_pages(self, pages: list[Page]):
    """Frees pages from the cache."""
    pids = []
    for page in pages:
      pids.append(page.page_id)
      self._unreferenced_tpu_pages.pop(page, None)
      self._unreferenced_cpu_pages.pop(page, None)

      if page.prefix_hash is not None:
        cached_page = self._prefix_hash_to_page.get(page.prefix_hash)
        if page == cached_page:
          self._prefix_hash_to_page.pop(page.prefix_hash, None)
        page.prefix_hash = None

    self._page_manager.free(pids)

  def _free_unreferenced_tpu_pages(self, num_pages: int):
    """Free num_pages unreferenced TPU pages, offloading to CPU if possible."""
    if num_pages <= 0:
      return
    if num_pages > len(self._unreferenced_tpu_pages):
      raise ValueError(
          f"Cannot free {num_pages} TPU pages, "
          f"only {len(self._unreferenced_tpu_pages)} available."
      )
    cpu_shortfall = num_pages - self._page_manager.num_free_cpu_pages
    if cpu_shortfall > 0:
      n_to_free = min(cpu_shortfall, len(self._unreferenced_cpu_pages))
      self._free_unreferenced_cpu_pages(n_to_free)

    pages_to_offload: list[Page] = []
    pages_to_free: list[Page] = []
    n_free_cpu = self._page_manager.num_free_cpu_pages
    for _ in range(num_pages):
      page, _ = self._unreferenced_tpu_pages.popitem(last=False)
      if n_free_cpu > 0:
        pages_to_offload.append(page)
        n_free_cpu -= 1
      else:
        pages_to_free.append(page)

    if pages_to_offload:
      self._page_manager.offload([p.page_id for p in pages_to_offload])
      for page in pages_to_offload:
        page.location = "cpu"
        self._unreferenced_cpu_pages[page] = None

    if pages_to_free:
      self._free_pages(pages_to_free)

  def _free_unreferenced_cpu_pages(self, num_pages: int):
    """Free num_pages unreferenced CPU pages."""
    if num_pages <= 0:
      return
    if num_pages > len(self._unreferenced_cpu_pages):
      raise ValueError(
          f"Cannot free {num_pages} CPU pages, "
          f"only {len(self._unreferenced_cpu_pages)} available."
      )
    pages = [
        self._unreferenced_cpu_pages.popitem(last=False)[0]
        for _ in range(num_pages)
    ]
    self._free_pages(pages)

  def release_out_of_window_pages(self, req: request_lib.Request):
    """Release pages outside the sliding window for a request."""
    if self._window_size is None:
      return

    state = self._requests.get(req.request_id)
    if not state or not state.pages:
      return
    lowest_needed_token_idx = max(
        0, req.num_completed_tokens - self._window_size
    )
    lowest_needed_page_idx = min(
        lowest_needed_token_idx // self._page_size,
        len(state.pages) - 1,
    )

    idxs_to_release: list[int] = []
    for i in range(lowest_needed_page_idx - 1, -1, -1):
      if state.pages[i] is None:
        break

      idxs_to_release.append(i)

    # Low index pages are released first to ensure they are evicted first.
    for i in reversed(idxs_to_release):
      self._release_page(state.pages[i])
      state.pages[i] = None
    state.num_released_pages += len(idxs_to_release)

  def _chunk_and_hash(
      self, tokens: list[int], start_hash: int = 0
  ) -> list[int]:
    """Chunks tokens into pages and hashes them."""
    hashes = []
    parent_hash = start_hash
    aligned_n_tokens = (len(tokens) // self._page_size) * self._page_size
    # TODO: Hashing should be deterministic across multiple hosts.
    for i in range(0, aligned_n_tokens, self._page_size):
      chunk = tuple(tokens[i : i + self._page_size])
      parent_hash = hash((parent_hash, chunk))
      hashes.append(parent_hash)
    return hashes

  def cache_full_pages(self, req: request_lib.Request):
    """Hashes newly completed full pages and registers them in prefix cache."""
    state = self._requests.get(req.request_id)
    if not state or not state.pages:
      return
    num_full_pages = req.num_completed_tokens // self._page_size
    if num_full_pages <= state.num_hashed_pages:
      return

    prompt_tokens = req.token_ids[
        state.num_hashed_pages * self._page_size:
        num_full_pages * self._page_size
    ]
    hashes = self._chunk_and_hash(
        prompt_tokens, start_hash=state.last_page_hash
    )

    unhashed_full_pages = state.pages[state.num_hashed_pages : num_full_pages]
    if len(unhashed_full_pages) != len(hashes):
      raise ValueError(
          f"Number of unhashed full pages {len(unhashed_full_pages)} does not "
          f"match number of hashes {len(hashes)}."
      )

    for i, (prefix_hash, page) in enumerate(
        zip(hashes, unhashed_full_pages)
    ):
      p_idx = i + state.num_hashed_pages
      if page is None:
        continue

      cached_page = self._prefix_hash_to_page.get(prefix_hash)
      if cached_page is None:
        page.prefix_hash = prefix_hash
        self._prefix_hash_to_page[prefix_hash] = page
      elif cached_page == page:
        continue
      elif cached_page.location == "tpu":
        state.pages[p_idx] = cached_page
        self._release_page(page)
        self._touch_page(cached_page)
      else:
        # If the cached page is on the CPU, it must be unreferenced.
        # Free it and remap the prefix hash to the new page.
        assert cached_page.ref_count == 0, "CPU page must be unreferenced"
        self._free_pages([cached_page])
        page.prefix_hash = prefix_hash
        self._prefix_hash_to_page[prefix_hash] = page

    state.num_hashed_pages = num_full_pages
    state.last_page_hash = hashes[-1]

  def get_cached_pages(
      self,
      req: request_lib.Request,
  ) -> list[Page | None]:
    """Returns candidate cached pages positionally aligned with the prompt tokens.

    Each entry at index `i` corresponds to the logical page spanning token
    offsets `[i * page_size, (i + 1) * page_size)`. Hits return the cached
    `Page`, while misses return `None`. The returned list is truncated after
    the last cache hit.

    This method performs raw hash lookups without enforcing layer-specific
    continuity constraints. `KVCacheManager` is responsible for reconciling
    these cached pages across all coordinators:
    - Full-attention layers require an unbroken prefix of non-None pages from
      index 0 (note that full attention layers will never have `None` entries).
    - Sliding-window layers require either:
      1. An unbroken prefix starting from index 0, or
      2. A contiguous block spanning the entire sliding window ending at the
        candidate cutoff.

    Args:
      req: The request whose prompt tokens to check for cached pages.

    Returns:
      A list where index `i` is the page for chunk `i` or `None` if that chunk
      is not in cache. Returns an empty list if there are no hits or if the
      request is already scheduled.
    """
    if req.num_scheduled_tokens > 0:
      return []

    # Reserve the final token to prevent 0-length prefills
    num_full_pages = (len(req.token_ids) - 1) // self._page_size
    if num_full_pages <= 0:
      return []

    prompt_tokens = req.token_ids[: num_full_pages * self._page_size]
    hashes = self._chunk_and_hash(prompt_tokens)

    last_hit = 0
    matched_pages: list[Page | None] = []
    for i, h in enumerate(hashes):
      page = self._prefix_hash_to_page.get(h)
      matched_pages.append(page)

      if page is not None:
        last_hit = i + 1

    matched_pages = matched_pages[:last_hit]
    return matched_pages

  def _calculate_page_requirements(
      self,
      req: request_lib.Request,
      num_tokens: int,
      matched_prefix_pages: Sequence[Page | None] = (),
  ) -> tuple[int, int]:
    """Calculates the number of TPU pages needed to schedule num_tokens.

    Args:
      req: The request to calculate page requirements for.
      num_tokens: The number of tokens to be computed in the next step.
      matched_prefix_pages: The list of matched prefix pages for the request.
        Entries should be `None` for pages outside the active window.

    Returns:
      A tuple of (n_tpu_pages_to_allocate, n_unreferenced_tpu_pages_to_reclaim).
    """
    matched_cpu = [
        p
        for p in matched_prefix_pages
        if p is not None and p.location == "cpu"
    ]
    n_matched_unreferenced_tpu = sum(
        1
        for p in matched_prefix_pages
        if p is not None and p.location == "tpu" and p.ref_count == 0
    )
    current_pages = self._requests.get(req.request_id, _RequestState()).pages

    total_tokens = (
        req.num_completed_tokens
        + (len(matched_prefix_pages) * self._page_size)
        + num_tokens
    )
    target_total_pages = utils.cdiv(total_tokens, self._page_size)
    n_needed = max(
        0, target_total_pages - len(current_pages) - len(matched_prefix_pages)
    )

    return (n_needed + len(matched_cpu), n_matched_unreferenced_tpu)

  def has_sufficient_space(
      self,
      req: request_lib.Request,
      num_tokens: int,
      matched_prefix_pages: Sequence[Page | None] = (),
  ) -> bool:
    """Checks whether the coordinator can prepare pages for the request.

    Args:
      req: The request to check space for.
      num_tokens: The number of tokens to be computed in the next step.
      matched_prefix_pages: The list of matched prefix pages for the request.
        Entries should be `None` for pages outside the active window.

    Returns:
      True if there is sufficient space in the TPU pool, False otherwise.
    """
    tpu_needed, n_matched_unref_tpu = self._calculate_page_requirements(
        req, num_tokens, matched_prefix_pages=matched_prefix_pages
    )

    # Matched unreferenced pages are being claimed, so they cannot be evicted.
    evictable_tpu = len(self._unreferenced_tpu_pages) - n_matched_unref_tpu
    total_tpu_available = (
        self._page_manager.num_free_tpu_pages + evictable_tpu
    )

    return total_tpu_available >= tpu_needed

  def _is_valid_matched_pages(
      self,
      matched_prefix_pages: Sequence[Page | None],
      req: request_lib.Request,
  ) -> bool:
    """Validates the matched pages."""
    if not matched_prefix_pages:
      return True

    if req.num_scheduled_tokens > 0:
      return False

    # Prefix match: begins at index 0 and must not contain any holes.
    if matched_prefix_pages[0] is not None:
      if None in matched_prefix_pages:
        return False
      if self._window_size is not None:
        # If matched pages extend past the window, out-of-window pages must be
        # masked with None (i.e. a suffix match must be passed).
        n_pages_in_window = utils.cdiv(self._window_size, self._page_size)
        return len(matched_prefix_pages) <= n_pages_in_window
      return True

    # Suffix matches are only permitted when a window size is defined.
    if self._window_size is None:
      return False

    # Suffix match: must end with exactly n_pages_in_window valid pages,
    # and all preceding pages outside the window must be None.
    n_pages_in_window = utils.cdiv(self._window_size, self._page_size)
    if len(matched_prefix_pages) < n_pages_in_window:
      return False

    out_of_window = matched_prefix_pages[:-n_pages_in_window]
    in_window = matched_prefix_pages[-n_pages_in_window:]

    return (
        all(p is None for p in out_of_window)
        and all(p is not None for p in in_window)
    )

  def prepare_request_pages(
      self,
      req: request_lib.Request,
      num_tokens: int,
      matched_prefix_pages: Sequence[Page | None] = (),
  ) -> None:
    """Prepares pages for a request in the KV cache.

    Before preparing pages for a request, the caller should:
    1. Hash any newly completed full pages using cache_full_pages,
    2. Release out-of-window pages using release_out_of_window_pages,
    3. Fetch the matched pages for the request using get_cached_pages,
    4. Verify that there is sufficient space in the TPU pool using
       has_sufficient_space.

    Args:
      req: The request to prepare pages for.
      num_tokens: The number of tokens to be computed in the next step.
      matched_prefix_pages: The list of matched prefix pages for the request.
        Entries should be `None` for pages outside the active window. It is
        expected that matched_prefix_pages contains either a contiguous
        prefix or full-window suffix of pages.
    """
    if not self._is_valid_matched_pages(matched_prefix_pages, req):
      raise ValueError(
          f"Invalid matched pages for request {req.request_id}."
      )

    state = self._requests.setdefault(req.request_id, _RequestState())

    tpu_needed, _ = self._calculate_page_requirements(
        req, num_tokens, matched_prefix_pages=matched_prefix_pages
    )

    n_out_of_window_pages = 0
    matched_cpu = []
    for page in matched_prefix_pages:
      if page is None:
        n_out_of_window_pages += 1
        continue
      elif page.location == "cpu":
        matched_cpu.append(page)

    # Protect matched pages from eviction before making room
    for page in matched_prefix_pages:
      self._touch_page(page)
      state.pages.append(page)
    if matched_prefix_pages:
      state.num_hashed_pages = len(matched_prefix_pages)
      state.num_released_pages = n_out_of_window_pages
      last_page = matched_prefix_pages[-1]
      if last_page is not None and last_page.prefix_hash is not None:
        state.last_page_hash = last_page.prefix_hash

    total_tpu_available = self._page_manager.num_free_tpu_pages + len(
        self._unreferenced_tpu_pages
    )
    if tpu_needed > total_tpu_available:
      raise ValueError(
          f"Cannot schedule request {req.request_id}. "
          f"Need {tpu_needed} TPU pages, but only {total_tpu_available} "
          "are available."
      )

    if tpu_needed > self._page_manager.num_free_tpu_pages:
      shortfall = tpu_needed - self._page_manager.num_free_tpu_pages
      self._free_unreferenced_tpu_pages(shortfall)

    if matched_cpu:
      self._page_manager.load([p.page_id for p in matched_cpu])
      for page in matched_cpu:
        page.location = "tpu"

    n_new_tpu_pages = tpu_needed - len(matched_cpu)
    if n_new_tpu_pages > 0:
      allocated_pids = self._page_manager.allocate_tpu_pages(n_new_tpu_pages)
      for pid in allocated_pids:
        page = Page(page_id=pid, location="tpu")
        state.pages.append(page)
        self._touch_page(page)

  def release_request(self, req: request_lib.Request) -> None:
    """Releases all pages for the given request."""
    state = self._requests.pop(req.request_id, None)
    if not state:
      return

    for page in reversed(state.pages):
      self._release_page(page)

  def get_page_idxs(self, req: request_lib.Request) -> list[int]:
    """Returns the page indices for the given request."""
    state = self._requests.get(req.request_id)
    if not state:
      return []
    active_pages = state.pages[state.num_released_pages :]
    result: list[int] = []
    for page in active_pages:
      if page is None:
        raise ValueError(
            "Cannot get page indices for released pages."
        )
      idx = self._page_manager.get_page_idx(page.page_id)
      if idx is None:
        raise ValueError(f"Page {page.page_id} not found in page manager.")
      result.append(idx)

    return result

  def update_tpu_pool(self, new_pages: dict[str, Any]) -> None:
    """Updates the TPU pool in the page manager with new pages."""
    self._page_manager.update_tpu_pool(new_pages)
