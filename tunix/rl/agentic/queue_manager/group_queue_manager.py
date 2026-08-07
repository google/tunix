# Copyright 2025 Google LLC
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

"""Manages queues of items with pluggable grouping and filtering."""

from __future__ import annotations

import asyncio
import collections
from collections.abc import Hashable
import dataclasses
from typing import Callable, Deque, Dict, Generic, List, Optional, Tuple, TypeVar, Union

_T = TypeVar("_T")

# GroupFn takes the internal buckets dictionary and an incoming item,
# returning a ready group of items if formed, else None.
GroupFn = Callable[[Dict[Hashable, List[_T]], _T], Optional[List[_T]]]

# FilterFn takes a candidate group and returns valid items (or tuple of (valid, filtered_out)).
FilterFn = Callable[
    [List[_T]],
    Union[List[_T], Tuple[List[_T], List[_T]]],
]


class GroupQueueManager(Generic[_T]):
  """Manages queues of items with pluggable grouping and filtering.

  This class collects instances into buckets based on a pluggable grouping
  strategy (defaulting to grouping by `item.group_id` / `item.prompt_id`).
  Once a candidate group reaches completion, it passes through an optional
  `filter_fn` before being made available for retrieval in batches.
  """

  def __init__(
      self,
      *,
      group_size: Optional[int] = None,
      group_fn: Optional[GroupFn[_T]] = None,
      filter_fn: Optional[FilterFn[_T]] = None,
  ):
    """Initializes GroupQueueManager.

    Args:
      group_size: Optional target size for default grouping.
      group_fn: Optional custom grouping function `Callable[[buckets, item],
        Optional[List[_T]]]`. If None, `group_size` must be provided.
      filter_fn: Optional filtering function `Callable[[candidate_group],
        valid_items]`.
    """
    if group_fn is None:
      if group_size is None:
        raise ValueError(
            "Must specify either group_size or a custom group_fn for"
            " GroupQueueManager."
        )

      def default_group_fn(
          buckets: Dict[Hashable, List[_T]],
          item: _T,
      ) -> Optional[List[_T]]:
        group_id = getattr(item, "group_id", None)
        prompt_id = getattr(item, "prompt_id", None)
        if group_id is not None and group_id != "":
          key = group_id
        elif prompt_id is not None and prompt_id != "":
          key = prompt_id
        else:
          key = id(item)
        bucket = buckets[key]
        bucket.append(item)
        if len(bucket) == group_size:
          del buckets[key]
          return bucket
        return None

      group_fn = default_group_fn

    self.group_size = group_size
    self.group_fn = group_fn
    self.filter_fn = filter_fn

    self._buckets: Dict[Hashable, List[_T]] = collections.defaultdict(list)
    self._ready_groups: Deque[List[_T]] = collections.deque()
    self._filtered_groups: Deque[List[_T]] = collections.deque()
    self._clearing = False
    self._exc: Optional[Exception] = None
    self._lock = asyncio.Lock()
    self._have_ready = asyncio.Event()

  async def put_exception(self, exc: Exception):
    """Sets an exception on the queue, failing future and pending operations."""
    self._exc = exc
    self._have_ready.set()

  async def prepare_clear(self):
    """Flags the queue manager as clearing."""
    self._clearing = True
    self._have_ready.set()

  async def clear(self):
    """Clears all internal buckets and ready groups."""
    async with self._lock:
      self._buckets.clear()
      self._ready_groups.clear()
      self._filtered_groups.clear()
      self._exc = None
      self._clearing = False
      self._have_ready.clear()

  async def get_filtered_groups(self) -> List[List[_T]]:
    """Returns and clears all candidate groups/items that were filtered out."""
    async with self._lock:
      filtered = list(self._filtered_groups)
      self._filtered_groups.clear()
      return filtered

  async def put(self, item: _T):
    """Adds an item, executing pluggable grouping and filtering.

    Args:
      item: The item to add.

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

      candidate_group = self.group_fn(self._buckets, item)

      if candidate_group is not None:
        valid_group = candidate_group
        filtered_out = []

        if self.filter_fn is not None:
          filter_res = self.filter_fn(candidate_group)
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

  async def _get_one_ready_group(self) -> List[_T]:
    while True:
      if self._exc:
        raise self._exc
      if self._clearing:
        return []
      async with self._lock:
        if self._ready_groups:
          return self._ready_groups.popleft()
      await self._have_ready.wait()
      self._have_ready.clear()

  async def get_batch(self, batch_size: int) -> List[_T]:
    """Retrieves a batch of items, waiting until enough are ready.

    Args:
      batch_size: The desired number of items.

    Returns:
      A list of items up to `batch_size`.
    """
    out = []
    while len(out) < batch_size:
      group = await self._get_one_ready_group()
      if not group:
        break
      async with self._lock:
        room = batch_size - len(out)
        if len(group) <= room:
          out.extend(group)
        else:
          out.extend(group[:room])
          self._ready_groups.appendleft(group[room:])
          self._have_ready.set()
    return out
