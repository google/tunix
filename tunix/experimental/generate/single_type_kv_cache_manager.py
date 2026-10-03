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

"""Manages KV cache pages for a set of layers sharing a single attention type.

Coordinates prefix caching, sliding-window eviction, and page lifecycles across
associated layers. Physical page allocation is delegated to an underlying
`TieredPagePoolManager`.

Responsibilities:
    - Prefix caching: Identifies matching cached prefixes for incoming requests.
    - Window management: Releases memory for tokens exceeding local attention
      boundaries.
    - Request tracking: Tracks the logical pages assigned to each active 
      request.
    - Page lifecycle: Maintains page reference counts, offloading and evicting
      LRU pages when cache pressure requires it.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
import dataclasses
from typing import Any

import tunix.experimental.generate.tiered_page_pool as page_pool_lib
from tunix.generate import utils


@dataclasses.dataclass
class Page:
  """A page in the KV cache."""
  # The logical ID of the page.
  page_id: int
  # The number of references to this page.
  ref_count: int = 0
  # The prefix hash of the page, or `None` if the page has not been cached.
  prefix_hash: int | None = None
  # Whether the page has been freed. Used to detect double-free.
  is_freed: bool = False

  def __hash__(self) -> int:
    return self.page_id

  def __eq__(self, other: Any) -> bool:
    return isinstance(other, Page) and self.page_id == other.page_id


class SingleTypeKVCacheManager:
  """Manages the KV caches for a group of layers with one attention type."""

  def __init__(
      self,
      page_pool_config: page_pool_lib.TieredPagePoolConfig,
      window_size: int | None = None,
  ):
    """Initializes the KV cache manager.

    Args:
      page_pool_config: The configuration for the KV caches' page pool.
      window_size: The size of the sliding window, or `None` for global
        attention.
    """

    self._page_manager = page_pool_config.create_manager()
    self._request_to_pages: dict[str, list[Page | None]] = {}

    self._page_size = page_pool_config.page_size
    self._window_size: int | None = window_size  # None for global attention.

    self._prefix_hash_to_page: dict[int, Page] = {}
    self._unreferenced_tpu_pages: collections.OrderedDict[Page, None] = (
        collections.OrderedDict()
    )
    self._unreferenced_cpu_pages: collections.OrderedDict[Page, None] = (
        collections.OrderedDict()
    )

  @property
  def _num_pages_in_window(self) -> int:
    if self._window_size is None:
      return 0
    # Add 1 to account for the window sliding and leaking into the next page.
    return utils.cdiv(self._window_size, self._page_size) + 1

  def _touch_page(self, page: Page | None):
    """Increments a page's reference count."""
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
      raise ValueError(
          f"Cannot release page {page.page_id} with no references."
      )

    page.ref_count -= 1
    if page.ref_count > 0:
      return

    location = self._page_manager.page_location(page.page_id)
    if page.prefix_hash is None:
      # If a page was never hashed, it cannot be reused and should be freed
      # immediately.
      self._free_pages([page])
    elif location == "tpu":
      self._unreferenced_tpu_pages[page] = None
    elif location == "cpu":
      self._unreferenced_cpu_pages[page] = None
    else:
      raise ValueError(f"Unknown page location: {location}")

  def _offload_pages(self, pages: list[Page]):
    """Offloads a page to the CPU."""
    if not pages:
      return

    self._page_manager.offload([p.page_id for p in pages])
    for page in pages:
      self._unreferenced_cpu_pages[page] = None

  def _free_pages(self, pages: list[Page]):
    """Frees pages from the cache."""
    pids = []
    for page in pages:
      if page.ref_count > 0:
        raise ValueError(
            f"Cannot free page {page.page_id} with {page.ref_count} references."
        )

      if page.is_freed:
        raise ValueError(f"Attempting to double-free page {page.page_id}.")

      page.is_freed = True

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

    self._offload_pages(pages_to_offload)
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

  def _release_out_of_window(
      self, request_id: str, num_completed_tokens: int
  ):
    """Release pages outside the sliding window for a request."""
    if self._window_size is None:
      return

    pages = self._request_to_pages.get(request_id)
    if not pages:
      return

    lowest_needed_token_idx = max(
        0, num_completed_tokens - self._window_size
    )
    lowest_needed_page_idx = min(
        lowest_needed_token_idx // self._page_size,
        len(pages) - 1,
    )

    idxs_to_release: list[int] = []
    for i in range(lowest_needed_page_idx - 1, -1, -1):
      if pages[i] is None:
        break

      idxs_to_release.append(i)

    # Unreferenced pages are freed lazily in LRU order when space is needed.
    # Low index pages are released first to ensure they are freed first.
    # This allows pages near the window to still prefix match.
    for i in reversed(idxs_to_release):
      self._release_page(pages[i])
      pages[i] = None

  def _cache_full_pages(
      self, request_id: str, page_hashes: Sequence[int]
  ):
    """Caches newly completed full pages."""

    pages = self._request_to_pages.get(request_id)
    if not pages:
      return

    n_hashed = min(len(page_hashes), len(pages))
    for i in range(n_hashed - 1, -1, -1):
      page = pages[i]
      prefix_hash = page_hashes[i]

      if page is None:
        # If this page is released, all previous pages must also be released.
        break

      cached_page = self._prefix_hash_to_page.get(prefix_hash)
      cached_location = (
          self._page_manager.page_location(cached_page.page_id)
          if cached_page is not None
          else None
      )

      if cached_page is None:
        page.prefix_hash = prefix_hash
        self._prefix_hash_to_page[prefix_hash] = page
      elif cached_page == page:
        # If this page is already cached, so are all
        # previous pages.
        break
      elif cached_location == "tpu":
        # On a TPU hit, the cached page cannot be freed as it may be
        # referenced. Reuse the cached page instead. `page` is never hashed,
        # so releasing it frees it immediately.
        pages[i] = cached_page
        self._release_page(page)
        self._touch_page(cached_page)
      elif cached_location == "cpu":
        # On a CPU hit, the cached page must be unreferenced.
        # To avoid an unnecessary transfer, free the cached page,
        # and rebind the hash to `page`.

        assert cached_page.ref_count == 0, "CPU page must be unreferenced"
        # Rebind the prefix first so that the cached CPU page is freed
        # immediately.
        page.prefix_hash = prefix_hash
        self._prefix_hash_to_page[prefix_hash] = page
        self._free_pages([cached_page])
      else:
        raise ValueError(
            f"Unknown cached page location: {cached_location}"
        )

  def sync_request_state(
      self,
      request_id: str,
      page_hashes: Sequence[int],
      num_completed_tokens: int,
  ):
    """Syncs the request state for the given request."""
    # Pages must be cached before release, since uncached pages are freed
    # upon release.

    self._cache_full_pages(request_id, page_hashes)
    self._release_out_of_window(request_id, num_completed_tokens)

  def _longest_cache_hit_full_attention(
      self,
      page_hashes: Sequence[int]
  ) -> list[Page | None]:
    """Finds the longest kv cache hit for a group of full attention layers."""
    cache_hits: list[Page | None] = []
    for h in page_hashes:
      page = self._prefix_hash_to_page.get(h)
      if page is None:
        break

      cache_hits.append(page)

    return cache_hits

  def _longest_cache_hit_local_attention(
      self,
      page_hashes: Sequence[int]
  ) -> list[Page | None]:
    """Finds the longest kv cache hit for a group of local attention layers."""
    # Entries for pages outside the window should be `None`. Otherwise,
    # they will be unnecessarily loaded and referenced.
    cache_hits: list[Page | None] = [None] * len(page_hashes)

    def find_miss(w_start: int, w_end: int) -> int:
      """Scans right to left for a page miss in the window."""
      for i in range(w_end, w_start - 1, -1):
        h = page_hashes[i]
        page = self._prefix_hash_to_page.get(h)
        if page is None:
          return i
      return -1

    w_end = len(cache_hits) - 1
    w_start = 0
    while w_end >= 0:
      w_start = max(0, w_end - self._num_pages_in_window + 1)

      miss_idx = find_miss(w_start, w_end)
      if miss_idx >= 0:
        w_end = miss_idx - 1
        continue

      break

    for i in range(w_start, w_end + 1):
      cache_hits[i] = self._prefix_hash_to_page[page_hashes[i]]

    return cache_hits[:w_end + 1]

  def find_longest_cache_hit(
      self,
      page_hashes: Sequence[int]
  ) -> list[Page | None]:
    """Queries the prefix cache for pages matching page_hashes."""
    if self._window_size is None:
      return self._longest_cache_hit_full_attention(
          page_hashes
      )

    return self._longest_cache_hit_local_attention(
        page_hashes
    )
