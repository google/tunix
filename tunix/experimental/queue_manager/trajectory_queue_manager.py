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

"""Queue managers for TrajectoryItem groups.

Two consume orders are available, one per class:

* `TrajectoryQueueManager.get_group()` yields whichever group completes first.
* `BatchOrderedQueueManager.get_ordered_group()` yields `(batch_idx, group)` in
  prompt-batch order, behind a cursor, so a trainer sees whole prompt batches in
  dataset order.

The reads are deliberately named apart: they return different things, so a
consumer picks a class rather than discovering the shape at runtime.
`GroupOrder` names the two orders for configuration; each class has its own
`create`.
"""

from __future__ import annotations

import collections
from collections.abc import Callable, Hashable, Sequence
import enum
from typing import Any, Deque, Dict, Optional, Tuple

from tunix.experimental.common import datatypes
from tunix.rl.agentic.queue_manager import group_queue_manager

TrajectoryItem = datatypes.TrajectoryItem
GroupFn = group_queue_manager.GroupFn[TrajectoryItem]
FilterFn = group_queue_manager.FilterFn[TrajectoryItem]


class GroupOrder(enum.Enum):
  """Order in which the scored trajectory queue yields ready groups to the trainer.

  Attributes:
    TRAJECTORY_COMPLETION: Yields any completed trajectory group (all
      `num_generations` rollouts for a prompt) in completion/arrival order as
      soon as it finishes scoring, regardless of which dataset prompt batch it
      originated from. Fast prompts can overtake slower earlier prompts across
      batches, and stale groups (`policy_version < current - max_staleness`) are
      dropped via `TrajectoryQueueManager.create`.
    PROMPT_ARRIVAL: Partitions groups by dataset prompt batch (`batch_idx =
      prompt_idx // full_batch_size`) via `BatchOrderedQueueManager` and serves
      only groups belonging to the current batch cursor (`next_batch_idx`),
      withholding groups from later batches until the current batch commits.
      Because `DispatchWindow` bounds in-flight batches to `next_batch +
      max_staleness` and `pre_weight_sync` synchronizes in-flight rollouts at
      each step boundary, every batch delivers all `full_batch_size` groups
      within the staleness window without dropping groups or creating short
      batches.
  """

  # Completion order: any ready group, as soon as it exists.
  TRAJECTORY_COMPLETION = "trajectory_completion"
  # Prompt-arrival order: only groups of the batch the cursor points at.
  PROMPT_ARRIVAL = "prompt_arrival"


class TrajectoryQueueManager(group_queue_manager.GroupQueueManager):
  """Specialized GroupQueueManager holding TrajectoryItems with ACK and abort support."""

  def __init__(
      self,
      *,
      num_generations: Optional[int] = None,
      group_fn: Optional[GroupFn] = None,
      filter_fn: Optional[FilterFn] = None,
      key_fn: Optional[Callable[[datatypes.TrajectoryItem], Hashable]] = None,
      on_group_filtered: (
          Callable[[list[datatypes.TrajectoryItem]], None] | None
      ) = None,
  ):
    """Initializes TrajectoryQueueManager.

    Args:
      num_generations: Optional target number of trajectories per ready group
        when using default grouping.
      group_fn: Optional custom grouping function. If None, `num_generations` must
        be provided.
      filter_fn: Optional pluggable function to filter candidate groups.
      key_fn: Optional function to extract grouping key. Defaults to prompt_id
        fallback.
      on_group_filtered: Optional callback invoked when a candidate group is
        completely filtered out (e.g., due to staleness).
    """
    if key_fn is None and group_fn is None:

      def _default_key_fn(item: datatypes.TrajectoryItem) -> Hashable:
        prompt_id = getattr(item, "prompt_id", None)
        if prompt_id is not None and prompt_id != "":
          return prompt_id
        return id(item)

      key_fn = _default_key_fn

    super().__init__(
        num_generations=num_generations,
        group_fn=group_fn,
        filter_fn=filter_fn,
        key_fn=key_fn,
        on_group_filtered=on_group_filtered,
    )

  @classmethod
  def create(
      cls,
      num_generations: int = 1,
      max_staleness: int = 0,
      current_policy_version: Callable[[], int] | None = None,
      filter_fn: FilterFn | None = None,
      on_group_filtered: (
          Callable[[list[datatypes.TrajectoryItem]], None] | None
      ) = None,
  ) -> "TrajectoryQueueManager":
    """Creates a grouped trajectory queue with optional policy staleness filtering."""
    assert max_staleness >= 0, "max_staleness must be non-negative."
    combined_filter: FilterFn | None = filter_fn
    if max_staleness > 0 and current_policy_version is not None:

      def _item_policy_version(item: datatypes.TrajectoryItem) -> int:
        version = getattr(item, "policy_version", None)
        if version is None and isinstance(item.metadata, dict):
          version = item.metadata.get("policy_version")
        return int(version or 0)

      def _staleness_filter(
          group: list[datatypes.TrajectoryItem],
      ) -> tuple[
          list[datatypes.TrajectoryItem], list[datatypes.TrajectoryItem]
      ]:
        min_allowed = current_policy_version() - max_staleness
        if any(_item_policy_version(item) < min_allowed for item in group):
          return [], list(group)
        if filter_fn is not None:
          res = filter_fn(group)
          if isinstance(res, tuple):
            return res[0], list(res[1])
          valid_ids = {id(x) for x in res}
          return res, [x for x in group if id(x) not in valid_ids]
        return list(group), []

      combined_filter = _staleness_filter

    return cls(
        num_generations=num_generations,
        filter_fn=combined_filter,
        on_group_filtered=on_group_filtered,
    )

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
    # TODO: why do we need both batch_size and num_groups?
    # TODO: should this be in parent class?
    if num_groups is not None:
      out: list[datatypes.TrajectoryItem] = []
      for _ in range(num_groups):
        g = await self._get_one_ready_group()
        if not g:
          break
        out.extend(g)
      return out
    actual_batch_size = (
        batch_size if batch_size is not None else self.num_generations
    )
    return await super().get_batch(batch_size=actual_batch_size)  # pyrefly: ignore[bad-argument-type]

  def commit(self, step: int, groups: Sequence[Any] | None = None) -> None:
    """Commits in-flight groups after a successful global step boundary."""
    # TODO: implement the commit and keep track of uncommited items.
    # might be worth putting in parent class.
    pass

  @property
  def ready_groups_count(self) -> int:
    """Returns the count of ready complete groups in the queue."""
    return len(self._ready_groups)

  @property
  def incomplete_buckets_count(self) -> int:
    """Returns the count of incomplete buckets currently buffering items."""
    return len(self._buckets)

  @property
  def filtered_groups_count(self) -> int:
    """Returns the count of filtered-out groups recorded by the queue."""
    return len(self._filtered_groups)

  async def abort(self, exc: BaseException) -> None:
    """Aborts queue and unblocks all waiting consumers with the given exception."""
    if isinstance(exc, Exception):
      await self.put_exception(exc)
    await self.prepare_clear()


def default_key_fn(item: TrajectoryItem) -> Hashable:
  """Groups items by `prompt_id`, falling back to identity when it is unset."""
  if item.prompt_id:
    return item.prompt_id
  return id(item)


def default_batch_idx_fn(item: TrajectoryItem) -> int:
  """Reads the prompt-batch index the dispatcher tagged the item with."""
  batch_idx = item.metadata.get("batch_idx") if item.metadata else None
  if batch_idx is None:
    raise ValueError(
        "BatchOrderedQueueManager requires a `batch_idx` in item metadata;"
        f" got metadata={item.metadata!r}."
    )
  return int(batch_idx)


class BatchOrderedQueueManager(group_queue_manager.GroupQueueManager):
  """Serves ready groups in prompt-batch order, behind a cursor.

  Ready groups are staged per `batch_idx` (tagged by the dispatcher) instead of
  in one flat FIFO, and `get_ordered_group` / `get_batch` only ever serve up to
  `full_batch_size` groups of the batch at `next_batch_idx` (the first
  untrained batch). Once `full_batch_size` groups have been served (or the queue
  is closed and `next_batch_idx` is drained), reads return `None` / `[]` without
  crossing into `next_batch_idx + 1` until `commit_batch` (or `commit`) advances
  the cursor.

  `skip` passes over batches `[begin, end)` without ever serving them, such as
  the prefix already trained before a checkpoint restore.

  Not thread-safe; like the base class it assumes a single asyncio event loop.
  """

  def __init__(
      self,
      *,
      full_batch_size: int,
      num_generations: Optional[int] = None,
      group_fn: Optional[GroupFn] = None,
      filter_fn: Optional[FilterFn] = None,
      key_fn: Optional[Callable[[TrajectoryItem], Hashable]] = None,
      batch_idx_fn: Optional[Callable[[TrajectoryItem], int]] = None,
  ):
    """Initializes BatchOrderedQueueManager.

    Args:
      full_batch_size: Groups per prompt batch.
      num_generations: Optional target number of trajectories per ready group
        when using default grouping.
      group_fn: Optional custom grouping function. If None, `num_generations`
        must be provided.
      filter_fn: Optional pluggable function to filter candidate groups.
      key_fn: Optional function to extract grouping key. Defaults to prompt_id
        fallback.
      batch_idx_fn: Optional function to read an item's prompt-batch index.
        Defaults to `metadata["batch_idx"]`.

    Raises:
      ValueError: If `full_batch_size` is not positive.
    """
    if full_batch_size <= 0:
      raise ValueError(
          f"full_batch_size must be positive, got {full_batch_size}."
      )

    if key_fn is None and group_fn is None:
      key_fn = default_key_fn

    super().__init__(
        num_generations=num_generations,
        group_fn=group_fn,
        filter_fn=filter_fn,
        key_fn=key_fn,
    )

    self.full_batch_size = full_batch_size
    self._batch_idx_fn = batch_idx_fn or default_batch_idx_fn

    # Ready groups staged per batch, replacing the base `_ready_groups` FIFO.
    self._pending: Dict[int, Deque[list[TrajectoryItem]]] = (
        collections.defaultdict(collections.deque)
    )
    self._next_batch_idx = 0
    self._served_in_batch = 0

  @classmethod
  def create(
      cls,
      num_generations: int = 1,
      full_batch_size: int = 1,
      filter_fn: FilterFn | None = None,
  ) -> "BatchOrderedQueueManager":
    """Creates a prompt-batch-ordered queue.

    Args:
      num_generations: Number of trajectories that form one group.
      full_batch_size: Groups per prompt batch.
      filter_fn: Optional filter applied to candidate groups.

    Returns:
      A queue that yields groups in prompt-batch order.
    """
    return cls(
        num_generations=num_generations,
        full_batch_size=full_batch_size,
        filter_fn=filter_fn,
    )

  async def clear(self) -> None:
    """Clears all internal buckets, staged batches, and resets the batch cursor."""
    async with self._lock:
      self._buckets.clear()
      self._ready_groups.clear()
      self._filtered_groups.clear()
      self._pending.clear()
      self._served_in_batch = 0
      self._next_batch_idx = 0
      self._exc = None
      self._clearing = False
      self._have_ready.clear()

  async def get_batch(self, batch_size: int) -> list[TrajectoryItem]:
    """Retrieves up to `batch_size` items from the active prompt batch.

    Drains whole groups strictly from `next_batch_idx` until `out` reaches
    `batch_size`. Once `full_batch_size` groups of `next_batch_idx` have been
    served (or the queue is closed and `next_batch_idx` is drained), returns
    whatever items were collected (or `[]` on subsequent calls before
    `commit_batch`) without crossing into `next_batch_idx + 1`.

    Args:
      batch_size: Maximum number of TrajectoryItems to return.

    Returns:
      A list of TrajectoryItems from `next_batch_idx`.
    """
    out: list[TrajectoryItem] = []
    while len(out) < batch_size:
      ordered = await self.get_ordered_group()
      if ordered is None:
        break
      _, group = ordered
      out.extend(group)
    return out

  def commit(self, step: int, groups: Sequence[Any] | None = None) -> None:
    """Commits the active prompt batch (`next_batch_idx`) at the step boundary."""
    del step, groups
    self.commit_batch(self._next_batch_idx)

  @property
  def next_batch_idx(self) -> int:
    """First batch not yet trained; advanced by `commit_batch` / `skip`."""
    return self._next_batch_idx

  def skip(self, begin: int, end: int) -> None:
    """Passes over batches `[begin, end)` without ever serving them.

    Used for the prefix already trained before a checkpoint restore.

    Args:
      begin: First batch to pass over.
      end: One past the last batch to pass over.
    """
    for batch_idx in range(begin, end):
      self._pending.pop(batch_idx, None)
    if end > self._next_batch_idx:
      self._next_batch_idx = end
      self._served_in_batch = 0
    self._have_ready.set()

  def commit_batch(self, batch_idx: int) -> None:
    """Releases a batch's state after its optimizer step.

    Named apart from `TrajectoryQueueManager.commit(step, groups)`: that one
    takes a training step and the groups it consumed, this one takes a batch
    index and already knows which groups belong to it.

    Args:
      batch_idx: The batch that was trained.
    """
    self.skip(batch_idx, batch_idx + 1)

  async def get_ordered_group(
      self,
  ) -> Optional[Tuple[int, list[TrajectoryItem]]]:
    """Waits for the next group of the batch at `next_batch_idx`.

    Named apart from `TrajectoryQueueManager.get_group`, which yields a bare
    group in completion order and `[]` at EOF.

    Returns:
      `(next_batch_idx, group)`, or None once `full_batch_size` groups of
      `next_batch_idx` have been served (or the queue is closed and drained).
      Call `commit_batch` / `commit` to advance `next_batch_idx` to the next
      batch.

    Raises:
      Exception: Whatever was set via `put_exception` / `abort`.
    """
    while True:
      if self._exc:
        raise self._exc
      if self._clearing:
        return None

      batch_idx = self._next_batch_idx
      pending = self._pending.get(batch_idx)
      if pending and self._served_in_batch < self.full_batch_size:
        self._served_in_batch += 1
        return batch_idx, pending.popleft()
      if self._served_in_batch >= self.full_batch_size or self._closed:
        return None

      await self._have_ready.wait()
      self._have_ready.clear()

  def _on_candidate_group(
      self,
      valid_group: list[TrajectoryItem],
      filtered_out: list[TrajectoryItem],
  ) -> None:
    """Stages a resolved group under its batch."""
    representative = next(iter(valid_group or filtered_out), None)
    if representative is None:
      return
    batch_idx = self._batch_idx_fn(representative)

    if filtered_out:
      self._filtered_groups.append(filtered_out)

    if batch_idx < self._next_batch_idx:
      return

    if valid_group:
      self._pending[batch_idx].append(valid_group)

    self._have_ready.set()

  @property
  def pending_groups_count(self) -> int:
    """Returns the count of staged groups across all batches."""
    return sum(len(groups) for groups in self._pending.values())

  @property
  def incomplete_buckets_count(self) -> int:
    """Returns the count of incomplete buckets currently buffering items."""
    return len(self._buckets)

  async def abort(self, exc: BaseException) -> None:
    """Aborts queue and unblocks all waiting consumers with the given exception."""
    if isinstance(exc, Exception):
      await self.put_exception(exc)
    await self.prepare_clear()
