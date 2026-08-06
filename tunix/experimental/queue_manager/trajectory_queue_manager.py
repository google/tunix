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

"""TrajectoryQueueManager specialization of GroupQueueManager for TrajectoryItem."""

from collections.abc import Callable, Sequence
from typing import Any
from tunix.experimental.common import datatypes
from tunix.rl.agentic.queue_manager import group_queue_manager


class TrajectoryQueueManager(group_queue_manager.GroupQueueManager):
  """Specialized GroupQueueManager holding TrajectoryItems with ACK and abort support."""

  def __init__(
      self,
      group_size: int,
      filter_fn: Callable[
          [Sequence[datatypes.TrajectoryItem]],
          tuple[
              Sequence[datatypes.TrajectoryItem],
              Sequence[datatypes.TrajectoryItem],
          ],
      ]
      | None = None,
      maxsize: int = 0,
  ):
    del maxsize
    super().__init__(group_size=group_size)
    self._filter_fn = filter_fn
    self._uncommitted_groups: list[Any] = []

  def __aiter__(self) -> "TrajectoryQueueManager":
    return self

  async def __anext__(self) -> list[datatypes.TrajectoryItem]:
    group = await self.get_group()
    if not group:
      raise StopAsyncIteration
    return group

  async def get_group(self) -> list[datatypes.TrajectoryItem]:
    """Retrieves a single ready group of TrajectoryItems."""
    return await self._get_one_ready_group()

  async def get_batch(
      self,
      batch_size: int | None = None,
      num_groups: int | None = None,
  ) -> list[datatypes.TrajectoryItem]:
    """Retrieves items by either batch_size or num_groups."""
    if num_groups is not None:
      out: list[datatypes.TrajectoryItem] = []
      for _ in range(num_groups):
        g = await self._get_one_ready_group()
        if not g:
          break
        out.extend(g)
      return out
    actual_batch_size = batch_size if batch_size is not None else self.group_size
    return await super().get_batch(batch_size=actual_batch_size)

  def commit(self, step: int, groups: Sequence[Any] | None = None) -> None:
    """Commits in-flight groups after a successful global step boundary."""
    del step, groups
    self._uncommitted_groups.clear()

  async def abort(self, exc: BaseException) -> None:
    """Aborts queue and unblocks all waiting consumers with the given exception."""
    if isinstance(exc, Exception):
      await self.put_exception(exc)
    await self.prepare_clear()
