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

"""KV cache coordinator across multiple KV cache groups."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import numpy as np

from tunix.experimental.generate import single_type_kv_cache_manager

Page = single_type_kv_cache_manager.Page


class KVCacheCoordinator:
  """Coordinates KV cache page management across multiple KV cache groups."""

  def __init__(
      self,
      kv_cache_group_managers: Sequence[
          single_type_kv_cache_manager.SingleTypeKVCacheManager
      ],
  ):
    if not kv_cache_group_managers:
      raise ValueError("kv_cache_group_managers cannot be empty.")
    self._kv_cache_group_managers = kv_cache_group_managers

  @property
  def num_groups(self) -> int:
    return len(self._kv_cache_group_managers)

  def find_longest_cache_hit(
      self,
      page_hashes: Sequence[int],
  ) -> tuple[int, tuple[tuple[Page | None, ...], ...]]:
    """Computes the common prefix cache hit across all KV cache groups.

    A cache hit is determined at the prefix level: if a token prefix matches,
    it is a hit across all layers.

    Args:
      page_hashes: The sequence of prefix hash values for the request.

    Returns:
      tuple[int, tuple[tuple[Page | None, ...], ...]]:
        - num_hit_tokens: The total prefix length matched across all groups.
        - longest_hits: For each cache group, a sequence of length 
          `num_hit_tokens` aligned with the prefix:
            - `Page`: Active KV page cached on device/host.
            - `None`: Out-of-window hit for local/sliding-window attention.
    """
    n_groups = len(self._kv_cache_group_managers)
    longest_hits: list[list[Page | None]] = [
        [] for _ in range(n_groups)
    ]

    # Find the longest prefix match across all cache groups.
    # Note: `longest_hits` is 1:1 with the matched token prefix. For local
    # attention, tokens outside the active sliding window are valid hits
    # represented as `None` (they matched, but require no KVs).
    while True:
      for i, group in enumerate(self._kv_cache_group_managers):
        longest_hits[i] = group.find_longest_cache_hit(page_hashes)

      min_n_hit_tokens = min(len(h) for h in longest_hits)
      max_n_hit_tokens = max(len(h) for h in longest_hits)

      page_hashes = page_hashes[:min_n_hit_tokens]

      if min_n_hit_tokens == max_n_hit_tokens:
        break

    return min_n_hit_tokens, tuple(tuple(h) for h in longest_hits)

  def sync_request_state(
      self,
      request_id: str,
      page_hashes: Sequence[int],
      num_completed_tokens: int,
  ):
    """Syncs the request state for the given request."""
    for group in self._kv_cache_group_managers:
      group.sync_request_state(request_id, page_hashes, num_completed_tokens)

  def has_sufficient_space(
      self,
      request_id: str,
      num_tokens: int,
      num_completed_tokens: int,
      computed_pages: Sequence[Sequence[Page | None]],
  ) -> bool:
    """Checks whether the coordinator can prepare pages for the request.

    Args:
      request_id: The request to check space for.
      num_tokens: The number of token slots to allocate.
      num_completed_tokens: The number of completed tokens for the request.
      computed_pages: The list of matched prefix pages per group. Entries should
        be `None` for pages outside the active window.

    Returns:
      True if there is sufficient space in the TPU pool, False otherwise.
    """
    return all(
        group.has_sufficient_space(
            request_id,
            num_tokens,
            num_completed_tokens,
            computed_pages[i],
        )
        for i, group in enumerate(self._kv_cache_group_managers)
    )

  def get_page_idxs(
      self,
      request_id: str,
  ) -> dict[str, tuple[int, ...]]:
    """Returns a mapping of layer ID to physical TPU page indices."""
    page_idxs: dict[str, tuple[int, ...]] = {}
    for manager in self._kv_cache_group_managers:
      group_page_idxs = tuple(manager.get_page_idxs(request_id))
      for layer in manager.layers:
        page_idxs[layer] = group_page_idxs

    return page_idxs

  def get_physical_pages(
      self,
  ) -> dict[str, jax.Array | np.ndarray]:
    """Returns a mapping of layer ID to physical page arrays."""
    physical_pages: dict[str, jax.Array | np.ndarray] = {}
    for manager in self._kv_cache_group_managers:
      physical_pages.update(manager.get_physical_pages())
    return physical_pages

  def allocate_slots(
      self,
      request_id: str,
      num_tokens: int,
      num_completed_tokens: int,
      computed_pages: Sequence[Sequence[Page | None]],
  ) -> None:
    """Allocates `num_tokens` token slots for a request."""
    for i, manager in enumerate(self._kv_cache_group_managers):
      manager.allocate_slots(
          request_id,
          num_tokens,
          num_completed_tokens,
          computed_pages[i],
      )

  def release_request(self, request_id: str) -> None:
    """Releases all pages for the given request."""
    for group in self._kv_cache_group_managers:
      group.release_request(request_id)

  def update_tpu_pool(self, new_pages: dict[str, Any]) -> None:
    """Updates the TPU pool in the page manager with new pages."""
    for manager in self._kv_cache_group_managers:
      manager.update_tpu_pool(new_pages)

  def reset_kv_caches(self) -> None:
    """Resets all KV caches."""
    for manager in self._kv_cache_group_managers:
      manager.reset_kv_caches()

