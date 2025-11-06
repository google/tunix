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

"""Manages queues of trajectory items, grouping them by group_id and episode_id."""

from __future__ import annotations
import asyncio
import collections
from collections.abc import Hashable
import dataclasses
from typing import Deque, Dict, List, Optional, Tuple
from tunix.rl.agentic.agents import agent_types

Trajectory = agent_types.Trajectory
TrajectoryItem = agent_types.TrajectoryItem
field = dataclasses.field
dataclass = dataclasses.dataclass


class GroupQueueManager:
  """Manages queues of trajectory items, grouping them by group_id and episode_id.

  This class collects `TrajectoryItem` instances into buckets based on their
  `(group_id, episode_id)`. Once a bucket reaches `group_size`, it becomes a
  "ready group" and can be retrieved in batches. It also handles managing the
  number of open buckets and provides mechanisms for clearing and handling
  exceptions.
  """

  def __init__(
      self,
      *,
      group_size: int,
      max_open_buckets: Optional[int] = None,
  ):
    self.group_size = group_size
    self.max_open_buckets = max_open_buckets or 0
    self._buckets: Dict[Tuple[Hashable, int], List[TrajectoryItem]] = {}
    self._ready_groups: Deque[List[TrajectoryItem]] = collections.deque()
    self._clearing = False
    self._exc: Optional[BaseException] = None
    self._lock = asyncio.Lock()
    self._capacity = asyncio.Condition(self._lock)
    self._have_ready = asyncio.Event()
    self._batch_buf: List[TrajectoryItem] = []

  async def put_exception(self, exc: BaseException):
    self._exc = exc
    self._have_ready.set()
    async with self._capacity:
      self._capacity.notify_all()

  async def prepare_clear(self):
    self._clearing = True
    self._have_ready.set()
    async with self._capacity:
      self._capacity.notify_all()

  async def clear(self):
    async with self._lock:
      self._buckets.clear()
      self._ready_groups.clear()
      self._batch_buf.clear()
      self._exc = None
      self._clearing = False
      self._have_ready = asyncio.Event()

  async def put(self, item: TrajectoryItem):
    """Adds an item, grouping by `(group_id, episode_id)`.

    Items are grouped in buckets. When a bucket reaches `self.group_size`, it's
    moved to `_ready_groups`. Waits if `max_open_buckets` is exceeded.

    Args:
      item: The TrajectoryItem to add.

    Raises:
      BaseException: If an exception has been set via `put_exception`.
    """
    if self._clearing:
      return
    if self._exc:
      raise self._exc
    key = (item.group_id, item.episode_id)
    async with self._capacity:
      new_bucket = key not in self._buckets
      while (
          (not self._clearing)
          and (self.max_open_buckets > 0)
          and new_bucket
          and (self._open_bucket_count() >= self.max_open_buckets)
      ):
        await self._capacity.wait()
      if self._clearing:
        return
      if self._exc:
        raise self._exc
      bucket = self._buckets.setdefault(key, [])
      bucket.append(item)
      if len(bucket) == self.group_size:
        self._ready_groups.append(bucket.copy())
        del self._buckets[key]
        self._capacity.notify_all()
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

  def _open_bucket_count(self) -> int:
    return len(self._buckets)
