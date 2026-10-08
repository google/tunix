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
    # Ordered dicts are used as ordered sets (the values are always `None`),
    # since the LRU order for unreferenced pages is needed for eviction.
    self._unreferenced_device_pages: collections.OrderedDict[Page, None] = (
        collections.OrderedDict()
    )
    self._unreferenced_host_pages: collections.OrderedDict[Page, None] = (
        collections.OrderedDict()
    )

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
