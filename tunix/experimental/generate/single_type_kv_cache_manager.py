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

import jax
import numpy as np
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
    self._cache_names = page_pool_config.partition_keys
    self._request_to_pages: dict[str, list[Page | None]] = {}

    self._page_size = page_pool_config.page_size
    self._window_size: int | None = window_size  # None for global attention.

    self._prefix_hash_to_page: dict[int, Page] = {}
    # Ordered dicts are used as ordered sets (the values are always `None`),
    # since the LRU order for unreferenced pages is needed for eviction.
    self._unreferenced_device_pages: collections.OrderedDict[Page, None] = (
        collections.OrderedDict()
    )
    self._unreferenced_host_pages: collections.OrderedDict[Page, None] = (
        collections.OrderedDict()
    )

  @property
  def cache_names(self) -> tuple[str, ...]:
    return self._cache_names

  @property
  def window_size(self) -> int | None:
    return self._window_size

  def get_physical_pages(
      self,
  ) -> dict[str, jax.Array | np.ndarray]:
    """Returns a mapping of cache ID to physical page arrays."""
    return self._page_manager.physical_device_pages

  def get_page_idxs(self, request_id: str) -> list[int]:
    """Returns the page indices for the given request."""
    pages = self._request_to_pages.get(request_id, [])

    result: list[int] = []
    for page in pages:
      if page is None:
        result.append(-1)
        continue

      idx = self._page_manager.page_idx(page.page_id)
      if idx is None:
        raise ValueError(f"Page {page.page_id} not found in page manager.")
      result.append(idx)

    return result

  def update_device_pool(self, new_pages: dict[str, Any]) -> None:
    """Updates the device pool in the page manager with new pages."""
    self._page_manager.update_device_pool(new_pages)

  def reset_kv_caches(self) -> None:
    """Resets all KV caches."""
    req_ids = list(self._request_to_pages.keys())
    for request_id in req_ids:
      self.release_request(request_id)

    self._free_pages(list(self._unreferenced_device_pages.keys()))
    self._free_pages(list(self._unreferenced_host_pages.keys()))

  @property
  def _num_pages_in_window(self) -> int:
    if self._window_size is None:
      return 0
    # Add 1 to account for the window sliding and leaking into the next page.
    return utils.cdiv(self._window_size, self._page_size) + 1

  def _touch_page(self, page: Page | None) -> None:
    """Increments a page's reference count."""
    if page is None:
      return

    if page.is_freed:
      raise ValueError(
          f"Attempting to touch page {page.page_id} which has been freed."
      )

    page.ref_count += 1

    self._unreferenced_device_pages.pop(page, None)
    self._unreferenced_host_pages.pop(page, None)

  def _release_page(self, page: Page | None) -> None:
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
    elif location == "device":
      self._unreferenced_device_pages[page] = None
    elif location == "host":
      self._unreferenced_host_pages[page] = None
    else:
      raise ValueError(f"Unknown page location: {location}")

  def _offload_pages(self, pages: Sequence[Page]) -> None:
    """Offloads pages to the host."""
    if not pages:
      return

    for page in pages:
      if page.ref_count > 0:
        raise ValueError(
            f"Cannot offload page {page.page_id} with {page.ref_count}"
            " references."
        )

    self._page_manager.offload([p.page_id for p in pages])
    for page in pages:
      self._unreferenced_host_pages[page] = None

  def _free_pages(self, pages: Sequence[Page]) -> None:
    """Frees pages from the cache."""
    # Validate every page before mutating any state, so that a failure leaves
    # the cache untouched.
    for page in pages:
      if page.ref_count > 0:
        raise ValueError(
            f"Cannot free page {page.page_id} with {page.ref_count} references."
        )

      if page.is_freed:
        raise ValueError(f"Attempting to double-free page {page.page_id}.")

    self._page_manager.free([p.page_id for p in pages])

    for page in pages:
      page.is_freed = True
      self._unreferenced_device_pages.pop(page, None)
      self._unreferenced_host_pages.pop(page, None)

      if page.prefix_hash is not None:
        cached_page = self._prefix_hash_to_page.get(page.prefix_hash)
        if page == cached_page:
          self._prefix_hash_to_page.pop(page.prefix_hash, None)
        page.prefix_hash = None

  def _free_unreferenced_device_pages(self, num_pages: int) -> None:
    """Free `num_pages` unreferenced device pages, offloading if possible."""
    if num_pages <= 0:
      return

    if num_pages > len(self._unreferenced_device_pages):
      raise ValueError(
          f"Cannot free {num_pages} device pages, "
          f"only {len(self._unreferenced_device_pages)} available."
      )

    host_shortfall = num_pages - self._page_manager.num_free_host_pages
    if host_shortfall > 0:
      n_to_free = min(host_shortfall, len(self._unreferenced_host_pages))
      self._free_unreferenced_host_pages(n_to_free)

    n_to_offload = min(num_pages, self._page_manager.num_free_host_pages)
    n_to_free = num_pages - n_to_offload

    pages = [
        self._unreferenced_device_pages.popitem(last=False)[0]
        for _ in range(num_pages)
    ]
    self._free_pages(pages[:n_to_free])
    self._offload_pages(pages[n_to_free:])

  def _free_unreferenced_host_pages(self, num_pages: int) -> None:
    """Free num_pages unreferenced host pages."""
    if num_pages <= 0:
      return

    if num_pages > len(self._unreferenced_host_pages):
      raise ValueError(
          f"Cannot free {num_pages} host pages, "
          f"only {len(self._unreferenced_host_pages)} available."
      )

    pages = [
        self._unreferenced_host_pages.popitem(last=False)[0]
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
      elif cached_location == "device":
        # On a device hit, the cached page cannot be freed as it may be
        # referenced. Reuse the cached page instead. `page` is never hashed,
        # so releasing it frees it immediately.
        pages[i] = cached_page
        self._release_page(page)
        self._touch_page(cached_page)
      elif cached_location == "host":
        # On a host hit, the cached page must be unreferenced.
        # To avoid an unnecessary transfer, free the cached page,
        # and rebind the hash to `page`.

        assert cached_page.ref_count == 0, "Host page must be unreferenced"
        # Rebind the prefix first so that the cached host page is freed
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
    """Queries the KV cache for a prefix match of the given page hashes."""
    if self._window_size is None:
      return self._longest_cache_hit_full_attention(
          page_hashes
      )

    return self._longest_cache_hit_local_attention(
        page_hashes
    )

  def _calculate_page_requirements(
      self,
      request_id: str,
      num_tokens: int,
      num_completed_tokens: int,
      computed_pages: Sequence[Page | None] = (),
  ) -> tuple[int, int]:
    """Calculates the number of device pages needed to schedule num_tokens.

    Args:
      request_id: The request to calculate page requirements for.
      num_tokens: The number of tokens to be computed in the next step.
      num_completed_tokens: The number of completed tokens for the request.
      computed_pages: The list of already computed (prefix matched) pages for
        the request. Entries should be `None` for pages outside the active
        window.

    Returns:
      A tuple of
        - The number of new device pages that need to be allocated.
        - The number of unreferenced device pages that need to be reclaimed.
    """
    pm = self._page_manager
    matched_host = [
        p
        for p in computed_pages
        if p is not None and pm.page_location(p.page_id) == "host"
    ]

    def is_device_unreferenced(p: Page | None) -> bool:
      return (
          p is not None
          and pm.page_location(p.page_id) == "device"
          and p.ref_count == 0
      )

    n_matched_unreferenced_device = sum(
        1
        for p in computed_pages
        if is_device_unreferenced(p)
    )

    current_pages = self._request_to_pages.get(request_id, [])

    total_tokens = (
        num_completed_tokens
        + (len(computed_pages) * self._page_size)
        + num_tokens
    )
    target_total_pages = utils.cdiv(total_tokens, self._page_size)
    n_needed = max(
        0, target_total_pages - len(current_pages) - len(computed_pages)
    )

    return n_needed + len(matched_host), n_matched_unreferenced_device

  def has_sufficient_space(
      self,
      request_id: str,
      num_tokens: int,
      num_completed_tokens: int,
      computed_pages: Sequence[Page | None] = (),
  ) -> bool:
    """Checks whether `num_tokens` token slots can be allocated for a request.

    Args:
      request_id: The ID of the request to check space for.
      num_tokens: The number of token slots to allocate.
      num_completed_tokens: The number of completed tokens for the request.
      computed_pages: The list of already computed pages for the request.
        Entries should be `None` for pages outside the active window.

    Returns:
      True if there is sufficient space to allocate `num_tokens` token slots,
      False otherwise.
    """
    device_needed, n_computed_unref_device = self._calculate_page_requirements(
        request_id, num_tokens, num_completed_tokens, computed_pages
    )

    # Unreferenced computed device pages will need to be reclaimed. They cannot
    # be counted towards `freeable_device`.
    freeable_device = (
        len(self._unreferenced_device_pages) - n_computed_unref_device
    )
    total_device_available = (
        self._page_manager.num_free_device_pages + freeable_device
    )

    return total_device_available >= device_needed

  def _swap_in_pages(
      self,
      pages: Sequence[Page | None],
  ) -> None:
    """Swap in host pages to the device.

    Args:
      pages: The list of pages to swap in. Each page to swap in must be
        referenced.
    """
    pm = self._page_manager

    # Only pages on the host need to be swapped in.
    pages_to_swap_in = [
        p
        for p in pages
        if p is not None and pm.page_location(p.page_id) == "host"
    ]

    if not pages_to_swap_in:
      return

    # If a page has no references, it may be freed when unreferenced space is
    # reclaimed. Thus, the caller must ensure all pages to be swapped in
    # have at least one reference.
    if not all(p.ref_count > 0 for p in pages_to_swap_in):
      raise ValueError(
          "Cannot swap in page with no references."
      )

    shortfall = len(pages_to_swap_in) - pm.num_free_device_pages
    if shortfall > 0:
      self._free_unreferenced_device_pages(shortfall)

    pm.load([p.page_id for p in pages_to_swap_in])

  def _bind_pages(
      self,
      pages: Sequence[Page | None],
      request_id: str,
  ) -> None:
    """Binds pages to the request state."""
    for page in pages:
      self._touch_page(page)

    if request_id not in self._request_to_pages:
      self._request_to_pages[request_id] = []

    self._request_to_pages[request_id].extend(pages)

  def _allocate_device_pages(
      self,
      n_pages: int,
      request_id: str,
  ) -> None:
    """Allocates `n_pages` device pages and binds them to the request.

    Args:
      n_pages: The number of device pages to allocate.
      request_id: The ID of the request to bind the pages to.
    """
    if n_pages <= 0:
      return

    shortfall = n_pages - self._page_manager.num_free_device_pages
    if shortfall > 0:
      self._free_unreferenced_device_pages(shortfall)

    allocated_pids = self._page_manager.allocate_device_pages(n_pages)
    allocated_pages = [Page(page_id=pid) for pid in allocated_pids]
    self._bind_pages(allocated_pages, request_id)

  def allocate_slots(
      self,
      request_id: str,
      num_tokens: int,
      num_completed_tokens: int,
      computed_pages: Sequence[Page | None] = (),
  ) -> None:
    """Allocates pages so the request has at least `num_tokens` free slots.

    Computed pages are bound to the request, and any on the host are loaded onto
    the device.

    Args:
      request_id: The ID of the request to allocate pages for.
      num_tokens: The number of tokens to be computed in the next step.
      num_completed_tokens: The number of completed tokens for the request.
      computed_pages: The list of computed (prefix matched) pages for the
        request. Entries should be `None` for pages outside the active window.
    """

    if not self.has_sufficient_space(
        request_id, num_tokens, num_completed_tokens, computed_pages
    ):
      raise ValueError(
          f"Cannot allocate {num_tokens} slots for request {request_id}. "
          "Insufficient space."
      )

    # `cache_full_pages` assumes that all computed tokens have their KV values
    # written to pages. When the length of a prefill exceeds the window size,
    # kvs are only written for the last `window_size` tokens. To maintain
    # simple bookkeeping, disallow computing more than the window size of tokens
    # in a single step.
    if self._window_size is not None and num_tokens > self._window_size:
      raise ValueError(
          "Cannot allocate more than window size tokens in a single step."
      )

    # Pages must be referenced before they are swapped in from host to device.
    # So, bind the computed pages to the request state first.
    self._bind_pages(computed_pages, request_id)

    self._swap_in_pages(computed_pages)

    # Must be computed after binding and swapping in, so that pages aren't
    # double counted. The computed pages are now part of the request's pages.
    new_n_completed_tokens = num_completed_tokens + len(
        computed_pages
    ) * self._page_size
    n_new_device_pages, _ = self._calculate_page_requirements(
        request_id, num_tokens, new_n_completed_tokens
    )

    self._allocate_device_pages(n_new_device_pages, request_id)

  def release_request(self, request_id: str) -> None:
    """Releases all pages for the given request."""
    pages = self._request_to_pages.pop(request_id, None)
    if not pages:
      return

    # Unreferenced pages are freed lazily in LRU order when space is needed.
    # Pages are released in reverse order so that right most pages are freed
    # first. This allows the left most pages to be reclaimed for a prefix match.
    for page in reversed(pages):
      self._release_page(page)
