# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Manages queues of trajectory items with pluggable grouping and filtering."""

from __future__ import annotations

import asyncio
import collections
from collections.abc import Hashable
import dataclasses
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

from tunix.rl.agentic.agents import agent_types

TrajectoryItem = agent_types.TrajectoryItem

# GroupKeyFn extracts a hashable grouping key from a TrajectoryItem.
GroupKeyFn = Callable[[TrajectoryItem], Hashable]

# CustomGroupFn handles direct custom bucket aggregation and returns a ready group if formed.
CustomGroupFn = Callable[
    [Dict[Hashable, List[List[TrajectoryItem]]], TrajectoryItem, int],
    Optional[List[TrajectoryItem]],
]

# FilterFn takes a candidate group and returns (valid_group, filtered_out_items)
# or just valid_group.
FilterFn = Callable[
    [List[TrajectoryItem]],
    Union[
        List[TrajectoryItem], Tuple[List[TrajectoryItem], List[TrajectoryItem]]
    ],
]


class TrajectoryQueueManager:
  """Manages queues of trajectory items with pluggable grouping and filtering.

  This class collects `TrajectoryItem` instances into candidate groups based on
  a pluggable grouping strategy (defaulting to grouping items by `group_id`
  with distinct `pair_index` values). Once a candidate group reaches
  `group_size`,
  it is passed through a pluggable `filter_fn`. Valid groups are added to
  `_ready_groups` for consumption, while filtered-out items are recorded and
  available via `get_filtered_groups()`.
  """

  def __init__(
      self,
      *,
      group_size: int,
      group_fn: Optional[Callable[..., Any]] = None,
      filter_fn: Optional[FilterFn] = None,
  ):
    """Initializes TrajectoryQueueManager.

    Args:
      group_size: Target number of trajectories in each ready group.
      group_fn: Optional pluggable function for grouping. Can be a key extractor
        `Callable[[TrajectoryItem], Hashable]` or custom grouping logic. If
        None, defaults to extracting `item.group_id`.
      filter_fn: Optional pluggable function to filter candidate groups. Can
        return either a filtered list of valid `TrajectoryItem`s, or a tuple of
        `(valid_items, filtered_out_items)`.
    """
    self.group_size = group_size
    self.group_fn = group_fn
    self.filter_fn = filter_fn

    # Bucket structure: key -> list of sub-buckets (each sub-bucket has distinct pair_indices)
    self._buckets: Dict[Hashable, List[List[TrajectoryItem]]] = (
        collections.defaultdict(list)
    )
    self._ready_groups: Deque[List[TrajectoryItem]] = collections.deque()
    self._filtered_groups: Deque[List[TrajectoryItem]] = collections.deque()
    self._clearing = False
    self._exc: Optional[Exception] = None
    self._lock = asyncio.Lock()
    self._have_ready = asyncio.Event()
    self._batch_buf: List[TrajectoryItem] = []

  async def put_exception(self, exc: Exception):
    self._exc = exc
    self._have_ready.set()

  async def prepare_clear(self):
    self._clearing = True
    self._have_ready.set()

  async def clear(self):
    async with self._lock:
      self._buckets.clear()
      self._ready_groups.clear()
      self._filtered_groups.clear()
      self._batch_buf.clear()
      self._exc = None
      self._clearing = False
      self._have_ready = asyncio.Event()

  def get_filtered_groups(self) -> List[List[TrajectoryItem]]:
    """Returns and clears all groups/items that were filtered out."""
    filtered = list(self._filtered_groups)
    self._filtered_groups.clear()
    return filtered

  def _add_to_default_bucket(
      self, key: Hashable, item: TrajectoryItem
  ) -> Optional[List[TrajectoryItem]]:
    """Adds item to bucket for key, ensuring distinct pair_index in ready group."""
    sub_buckets = self._buckets[key]

    # Find first sub-bucket that doesn't already contain item.pair_index
    target_sub = None
    for sub in sub_buckets:
      existing_pairs = {x.pair_index for x in sub}
      if item.pair_index not in existing_pairs:
        target_sub = sub
        break

    if target_sub is None:
      target_sub = []
      sub_buckets.append(target_sub)

    target_sub.append(item)

    if len(target_sub) == self.group_size:
      ready_group = target_sub.copy()
      sub_buckets.remove(target_sub)
      if not sub_buckets:
        del self._buckets[key]
      return ready_group

    return None

  def _process_grouping(
      self, item: TrajectoryItem
  ) -> Optional[List[TrajectoryItem]]:
    """Applies pluggable or default grouping logic to item."""
    if self.group_fn is None:
      return self._add_to_default_bucket(item.group_id, item)

    fn: Any = self.group_fn
    # 1. Try calling fn as a key extractor: key = fn(item)
    try:
      key = fn(item)
      if isinstance(key, (str, int, float, tuple, bytes)) or key is None:
        return self._add_to_default_bucket(key, item)
    except Exception:
      pass

    # 2. Try calling fn as custom grouping logic: fn(buckets, item, group_size)
    try:
      res = fn(self._buckets, item, self.group_size)
      if res is not None:
        return res
    except Exception:
      pass

    # Fallback default grouping
    return self._add_to_default_bucket(item.group_id, item)

  async def put(self, item: TrajectoryItem):
    """Adds an item, executing pluggable grouping and filtering.

    Args:
      item: The TrajectoryItem to add.

    Raises:
      Exception: If an exception has been set via `put_exception`.
    """
    if self._clearing:
      return
    if self._exc:
      raise self._exc

    async with self._lock:
      if self._clearing:
        return
      if self._exc:
        raise self._exc

      candidate_group = self._process_grouping(item)

      if candidate_group is not None:
        valid_group = candidate_group
        filtered_out = []

        if self.filter_fn is not None:
          fn: Any = self.filter_fn
          filter_res = fn(candidate_group)
          if isinstance(filter_res, tuple) and len(filter_res) == 2:
            valid_group, filtered_out = filter_res
          elif isinstance(filter_res, list):
            valid_group = filter_res
            valid_set = set(id(x) for x in valid_group)
            filtered_out = [
                x for x in candidate_group if id(x) not in valid_set
            ]

        if filtered_out:
          self._filtered_groups.append(filtered_out)

        if valid_group:
          self._ready_groups.append(valid_group)
          self._have_ready.set()

  async def _get_one_ready_group(self) -> List[TrajectoryItem]:
    while True:
      if self._exc:
        raise self._exc
      if self._clearing:
        return []
      if self._ready_groups:
        return self._ready_groups.popleft()
      await self._have_ready.wait()
      self._have_ready.clear()

  async def get_batch(self, batch_size: int) -> List[TrajectoryItem]:
    """Retrieves a batch of TrajectoryItems, waiting until enough are ready.

    Items are taken from `_batch_buf` and then from `_ready_groups`. Excess
    items from groups are buffered in `_batch_buf`.

    Args:
      batch_size: The desired number of TrajectoryItems.

    Returns:
      A list of `TrajectoryItem` instances, up to `batch_size`.
    """
    out = []
    if self._batch_buf:
      take = min(batch_size, len(self._batch_buf))
      out.extend(self._batch_buf[:take])
      self._batch_buf = self._batch_buf[take:]
      if len(out) == batch_size:
        return out
    while len(out) < batch_size:
      group = await self._get_one_ready_group()
      if not group:
        break
      room = batch_size - len(out)
      if len(group) <= room:
        out.extend(group)
      else:
        out.extend(group[:room])
        self._batch_buf.extend(group[room:])
    return out
