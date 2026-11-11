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

"""Queue managers for raw trajectory grouping and scored group staging.

* `TrajectoryQueueManager` buckets incoming `TrajectoryItem`s by `prompt_id` up
  to `num_generations` and filters stale groups for `StandardRLProgram.raw_q`.
* `FifoGroupQueue` and `BatchOrderedGroupQueue` hold pre-formed group items
  (such as `ScoredGroup`) for `StandardRLProgram.scored_q` without redundant
  per-item bucketing:
  - `FifoGroupQueue.get()` yields groups in completion/arrival order (or
    smallest-priority-key-first order with FIFO tie-breaking when `priority_fn`
    is provided; `None` at EOF).
  - `BatchOrderedGroupQueue.get()` yields up to `full_batch_size` groups of the
    batch at `next_batch_idx` in dataset prompt-batch order (`None` at batch
    boundary or EOF).
"""

from __future__ import annotations

import asyncio
import collections
from collections.abc import Callable, Hashable, Sequence
import heapq
import itertools
import logging
from typing import Any, Deque, Dict, Generic, Optional, Tuple, TypeVar

from tunix.dist_rl.common import datatypes
from tunix.rl.agentic.queue_manager import group_queue_manager

_T = TypeVar("_T")
TrajectoryItem = datatypes.TrajectoryItem
GroupOrder = datatypes.GroupOrder
GroupFn = group_queue_manager.GroupFn
FilterFn = group_queue_manager.FilterFn



def has_policy_version(item: TrajectoryItem) -> bool:
  """Returns whether `item.traj` carries a non-empty `policy_versions` list."""
  return isinstance(item.traj, dict) and bool(item.traj.get("policy_versions"))


def item_policy_versions(item: TrajectoryItem) -> Sequence[int]:
  """Returns the non-empty policy version sequence from `item.traj`."""
  versions = (
      item.traj.get("policy_versions") if isinstance(item.traj, dict) else None
  )
  if not versions:
    raise ValueError(
        f"Valid TrajectoryItem {item.traj_id} (prompt_id="
        f"{item.prompt_id!r}, group_index={item.group_index}) is"
        " missing policy_versions."
    )
  return [int(version) for version in versions]


def item_policy_version(item: TrajectoryItem) -> int:
  """Returns the latest policy version that generated tokens for `item`."""
  return item_policy_versions(item)[-1]


def item_version_span(item: TrajectoryItem) -> int:
  """Returns the policy version span (`max - min`) across turns for `item`."""
  versions = item_policy_versions(item)
  return max(versions) - min(versions)


def _is_group_within_version_window(
    valid_items: Sequence[TrajectoryItem],
    *,
    max_staleness: int,
    policy_version_fn: Callable[[], int] | None,
    span_limit: int | None,
) -> bool:
  """Returns whether `valid_items` satisfy monotonicity, span, and staleness bounds."""
  item_versions = [item_policy_versions(item) for item in valid_items]
  if any(
      a > b
      for versions in item_versions
      for a, b in itertools.pairwise(versions)
  ):
    return False
  all_versions = list(itertools.chain.from_iterable(item_versions))
  if (
      span_limit is not None
      and max(all_versions) - min(all_versions) > span_limit
  ):
    return False
  if policy_version_fn is None:
    return True
  current_version = policy_version_fn()
  min_version = current_version - max_staleness
  checked_versions = (
      (max(all_versions),)
      if span_limit is not None
      else (versions[-1] for versions in item_versions)
  )
  return all(min_version <= v <= current_version for v in checked_versions)


class TrajectoryQueueManager(
    group_queue_manager.GroupQueueManager[TrajectoryItem]
):
  """Specialized GroupQueueManager holding grouped TrajectoryItems with ACK and abort support."""

  def __init__(
      self,
      *,
      num_generations: Optional[int] = None,
      group_fn: Optional[GroupFn[TrajectoryItem]] = None,
      filter_fn: Optional[FilterFn[TrajectoryItem]] = None,
      key_fn: Optional[Callable[[TrajectoryItem], Hashable]] = None,
      on_group_filtered: Callable[[list[TrajectoryItem]], None] | None = None,
  ):
    """Initializes TrajectoryQueueManager.

    Args:
      num_generations: Optional target number of trajectories per ready group
        when using default grouping.
      group_fn: Optional custom grouping function. If None, `num_generations`
        must be provided.
      filter_fn: Optional pluggable function to filter candidate groups.
      key_fn: Optional function to extract grouping key. Defaults to prompt_id
        fallback.
      on_group_filtered: Optional callback invoked when a candidate group is
        completely filtered out (e.g., due to staleness).
    """
    if key_fn is None and group_fn is None:
      key_fn = default_key_fn

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
      filter_fn: FilterFn[TrajectoryItem] | None = None,
      on_group_filtered: Callable[[list[TrajectoryItem]], None] | None = None,
      max_version_span: int | None = None,
  ) -> "TrajectoryQueueManager":
    """Creates a grouped trajectory queue with optional staleness and span filtering.

    The window is checked once, when a group completes in `put`, and only
    against the group's valid items: invalid items (error tombstones,
    overlong-masked rollouts) contribute no gradient, so their version does
    not matter and is never substituted with `0`. A group with no valid item
    has nothing to judge and passes through; it trains with zero advantage. A
    valid item without a `policy_version` is a producer bug and makes `put`
    raise `ValueError`.

    Each valid item's per-turn version sequence must be monotonically
    non-decreasing (`v_0 <= v_1 <= ... <= v_k`); any trajectory whose version
    regressed across turns (e.g. after a checkpoint rollback) is dropped. When
    `max_version_span` is set (partial rollout), filtering enforces group-wide
    version span (`max(all_versions) - min(all_versions) <= max_version_span`)
    and group completion window (`max(all_versions) in
    [current_policy_version() - max_staleness, current_policy_version()]`),
    allowing early-finishing rollouts in `num_generations > 1` groups as long
    as the group-wide span is bounded. When `max_version_span` is `None`,
    filtering enforces per-item two-sided window (`item_policy_version(item) in
    [current_policy_version() - max_staleness, current_policy_version()]`)
    across valid items.

    Args:
      num_generations: Number of trajectories per ready group.
      max_staleness: How many policy versions behind `current_policy_version()`
        a group's rollouts may be when the group completes. `0` admits only
        rollouts sampled from the current version.
      current_policy_version: Returns the trainer's current policy version. When
        given, the staleness window is enforced (including for `max_staleness ==
        0`); when `None`, no staleness filtering happens.
      filter_fn: Optional extra filter applied after the staleness check.
      on_group_filtered: Called with each group the filters drop.
      max_version_span: Optional maximum allowed policy-version span (`max -
        min`) across valid items/turns in a group.

    Returns:
      The configured `TrajectoryQueueManager`.

    Raises:
      ValueError: If `max_staleness` or `max_version_span` is negative.
    """
    if max_staleness < 0:
      raise ValueError("max_staleness must be non-negative.")
    if max_version_span is not None and max_version_span < 0:
      raise ValueError(
          f"max_version_span must be non-negative, got {max_version_span}."
      )
    combined_filter: FilterFn[TrajectoryItem] | None = filter_fn
    policy_version_fn = current_policy_version
    span_limit = max_version_span
    if policy_version_fn is not None or span_limit is not None:

      def _version_filter(
          group: list[TrajectoryItem],
      ) -> tuple[list[TrajectoryItem], list[TrajectoryItem]]:
        valid_items = [item for item in group if item.is_valid]
        if valid_items and not _is_group_within_version_window(
            valid_items,
            max_staleness=max_staleness,
            policy_version_fn=policy_version_fn,
            span_limit=span_limit,
        ):
          return [], list(group)
        if filter_fn is not None:
          res = filter_fn(group)
          if isinstance(res, tuple):
            return res[0], list(res[1])
          valid_ids = {id(x) for x in res}
          return res, [x for x in group if id(x) not in valid_ids]
        return list(group), []

      combined_filter = _version_filter

    return cls(
        num_generations=num_generations,
        filter_fn=combined_filter,
        on_group_filtered=on_group_filtered,
    )

  def __aiter__(self) -> "TrajectoryQueueManager":
    return self

  async def __anext__(self) -> list[TrajectoryItem]:
    group = await self.get_group()
    if not group:
      raise StopAsyncIteration
    return group

  async def get_group(self) -> list[TrajectoryItem]:
    """Retrieves a single ready group of items."""
    return await self._get_one_ready_group()

  async def get_batch(
      self,
      batch_size: int | None = None,
      num_groups: int | None = None,
  ) -> list[TrajectoryItem]:
    """Retrieves items by either batch_size or num_groups."""
    # TODO: why do we need both batch_size and num_groups?
    # TODO: should this be in parent class?
    if num_groups is not None:
      out: list[TrajectoryItem] = []
      for _ in range(num_groups):
        g = await self._get_one_ready_group()
        if not g:
          break
        out.extend(g)
      return out
    actual_batch_size = (
        batch_size if batch_size is not None else self.num_generations
    )
    if actual_batch_size is None:
      raise ValueError(
          "batch_size must be provided when num_generations is None."
      )
    return await super().get_batch(batch_size=actual_batch_size)

  def commit(self, step: int, groups: Sequence[Any] | None = None) -> None:
    """Commits in-flight groups after a successful global step boundary."""
    # TODO: implement the commit and keep track of uncommited items.
    # might be worth putting in parent class.
    del step, groups

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
        "BatchOrderedGroupQueue requires a `batch_idx` in item metadata;"
        f" got metadata={item.metadata!r}."
    )
  return int(batch_idx)


class FifoGroupQueue(Generic[_T]):
  """Async closable/abortable completion-order queue of pre-formed groups (`_T`).

  Used for `StandardRLProgram.scored_q` under
  `GroupOrder.TRAJECTORY_COMPLETION`, where each `put` enqueues an already
  completed `ScoredGroup` without redundant per-item bucketing. When
  `priority_fn` is provided, buffered groups are yielded in smallest-key-first
  order with FIFO tie-breaking; otherwise they are yielded in pure FIFO order.
  """

  def __init__(
      self,
      priority_fn: Callable[[_T], int] | None = None,
  ) -> None:
    self._ready: list[tuple[int, int, _T]] = []
    self._seq = itertools.count()
    self._priority_fn = priority_fn
    self._clearing = False
    self._closed = False
    self._exc: Optional[BaseException] = None
    self._lock = asyncio.Lock()
    self._have_ready = asyncio.Event()

  @classmethod
  def create(
      cls,
      priority_fn: Callable[[_T], int] | None = None,
  ) -> "FifoGroupQueue[_T]":
    """Creates a FIFO group queue with optional smallest-key-first priority ordering and FIFO tie-breaking."""
    return cls(priority_fn=priority_fn)

  async def put(self, group: _T) -> None:
    """Enqueues a completed group.

    Args:
      group: The group item to enqueue.

    Raises:
      RuntimeError: If the queue is already closed.
      BaseException: If an exception was set via `abort`.
    """
    if self._closed:
      raise RuntimeError("Cannot put into a closed FifoGroupQueue.")
    if self._clearing:
      return
    if self._exc is not None:
      raise self._exc

    async with self._lock:
      if self._closed:
        raise RuntimeError("Cannot put into a closed FifoGroupQueue.")
      if self._clearing:
        return
      if self._exc is not None:
        raise self._exc
      priority = (
          self._priority_fn(group) if self._priority_fn is not None else 0
      )
      heapq.heappush(self._ready, (priority, next(self._seq), group))
      self._have_ready.set()

  async def get(self) -> Optional[_T]:
    """Dequeues the next ready group in FIFO (or smallest-key-first priority) order, or `None` at EOF/clear.

    When `priority_fn` is set and multiple groups are buffered, selects the
    group with the smallest `priority_fn(group)` value (breaking ties in FIFO
    arrival order via a monotonic sequence counter).

    Returns:
      The next group, or `None` once closed and drained (or cleared).

    Raises:
      BaseException: If an exception was set via `abort`.
    """
    while True:
      if self._exc is not None:
        raise self._exc
      if self._clearing:
        return None
      async with self._lock:
        if self._ready:
          _, _, group = heapq.heappop(self._ready)
          return group
        if self._closed:
          return None
      await self._have_ready.wait()
      self._have_ready.clear()

  async def close(self) -> None:
    """Gracefully marks the queue as closed (EOF)."""
    async with self._lock:
      self._closed = True
      self._have_ready.set()

  async def abort(self, exc: BaseException) -> None:
    """Aborts the queue and unblocks all waiting consumers with `exc`."""
    self._exc = exc
    self._clearing = True
    self._have_ready.set()

  async def clear(self) -> None:
    """Clears all queued groups and resets close/abort flags."""
    async with self._lock:
      self._ready.clear()
      self._seq = itertools.count()
      self._exc = None
      self._clearing = False
      self._closed = False
      self._have_ready.clear()

  def commit(self, step: int = 0, groups: Sequence[Any] | None = None) -> None:
    """No-op commit hook matching `BatchOrderedGroupQueue.commit`."""
    del step, groups

  @property
  def ready_groups_count(self) -> int:
    """Returns the number of groups currently buffered in the queue."""
    return len(self._ready)


class BatchOrderedGroupQueue(Generic[_T]):
  """Serves pre-formed groups (`_T`) in prompt-batch order, behind a cursor.

  Groups are staged per `batch_idx` (extracted via `batch_idx_fn`) instead of in
  one flat FIFO, and `get` / `get_ordered_group` only ever serve up to
  `full_batch_size` groups of the batch at `next_batch_idx` (the first
  untrained batch). Once `full_batch_size` groups have been served (or the queue
  is closed and `next_batch_idx` is drained), reads return `None` without
  crossing into `next_batch_idx + 1` until `commit_batch` (or `commit`) advances
  the cursor.

  `skip` passes over batches `[begin, end)` without ever serving them, such as
  the prefix already trained before a checkpoint restore.

  Not thread-safe; assumes a single asyncio event loop.
  """

  def __init__(
      self,
      *,
      full_batch_size: int,
      batch_idx_fn: Callable[[_T], int],
  ) -> None:
    """Initializes BatchOrderedGroupQueue.

    Args:
      full_batch_size: Groups per prompt batch.
      batch_idx_fn: Function to read a group's prompt-batch index.

    Raises:
      ValueError: If `full_batch_size` is not positive.
    """
    if full_batch_size <= 0:
      raise ValueError(
          f"full_batch_size must be positive, got {full_batch_size}."
      )
    self.full_batch_size = full_batch_size
    self._batch_idx_fn = batch_idx_fn
    self._pending: Dict[int, Deque[_T]] = collections.defaultdict(
        collections.deque
    )
    self._next_batch_idx = 0
    self._served_in_batch = 0
    self._dropped_late_groups = 0
    self._clearing = False
    self._closed = False
    self._exc: Optional[BaseException] = None
    self._lock = asyncio.Lock()
    self._have_ready = asyncio.Event()

  @classmethod
  def create(
      cls,
      full_batch_size: int = 1,
      *,
      batch_idx_fn: Callable[[_T], int],
  ) -> "BatchOrderedGroupQueue[_T]":
    """Creates a prompt-batch-ordered group queue."""
    return cls(
        full_batch_size=full_batch_size,
        batch_idx_fn=batch_idx_fn,
    )

  async def put(self, group: _T) -> None:
    """Stages a completed group under its `batch_idx`.

    Args:
      group: The group item to stage.

    Raises:
      RuntimeError: If the queue is already closed.
      BaseException: If an exception was set via `abort`.
      ValueError: If `batch_idx_fn(group)` fails.
    """
    if self._closed:
      raise RuntimeError("Cannot put into a closed BatchOrderedGroupQueue.")
    if self._clearing:
      return
    if self._exc is not None:
      raise self._exc

    batch_idx = self._batch_idx_fn(group)

    async with self._lock:
      if self._closed:
        raise RuntimeError("Cannot put into a closed BatchOrderedGroupQueue.")
      if self._clearing:
        return
      if self._exc is not None:
        raise self._exc

      if batch_idx < self._next_batch_idx:
        self._dropped_late_groups += 1
        logging.info(
            "BatchOrderedGroupQueue dropping group from already-passed"
            " batch_idx=%d (next_batch_idx=%d, dropped_late_groups=%d).",
            batch_idx,
            self._next_batch_idx,
            self._dropped_late_groups,
        )
        return

      self._pending[batch_idx].append(group)
      self._have_ready.set()

  async def get_ordered_group(self) -> Optional[Tuple[int, _T]]:
    """Waits for the next group of the batch at `next_batch_idx`.

    Returns:
      `(next_batch_idx, group)`, or `None` once `full_batch_size` groups of
      `next_batch_idx` have been served (or the queue is closed and drained).
      Call `commit_batch` / `commit` to advance `next_batch_idx` to the next
      batch.

    Raises:
      BaseException: Whatever was set via `abort`.
    """
    while True:
      if self._exc is not None:
        raise self._exc
      if self._clearing:
        return None

      async with self._lock:
        batch_idx = self._next_batch_idx
        pending = self._pending.get(batch_idx)
        if pending and self._served_in_batch < self.full_batch_size:
          self._served_in_batch += 1
          return batch_idx, pending.popleft()
        if self._served_in_batch >= self.full_batch_size or self._closed:
          return None

      await self._have_ready.wait()
      self._have_ready.clear()

  async def get(self) -> Optional[_T]:
    """Waits for and returns the next group of `next_batch_idx`, or `None` at boundary/EOF."""
    ordered = await self.get_ordered_group()
    if ordered is None:
      return None
    _, group = ordered
    return group

  def commit(self, step: int = 0, groups: Sequence[Any] | None = None) -> None:
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

    Args:
      batch_idx: The batch that was trained.
    """
    self.skip(batch_idx, batch_idx + 1)

  async def close(self) -> None:
    """Gracefully marks the queue as closed (EOF)."""
    async with self._lock:
      self._closed = True
      self._have_ready.set()

  async def abort(self, exc: BaseException) -> None:
    """Aborts queue and unblocks all waiting consumers with `exc`."""
    self._exc = exc
    self._clearing = True
    self._have_ready.set()

  async def clear(self) -> None:
    """Clears all staged batches and resets the batch cursor."""
    async with self._lock:
      self._pending.clear()
      self._served_in_batch = 0
      self._next_batch_idx = 0
      self._dropped_late_groups = 0
      self._exc = None
      self._clearing = False
      self._closed = False
      self._have_ready.clear()

  @property
  def pending_groups_count(self) -> int:
    """Returns the count of staged groups across all batches."""
    return sum(len(groups) for groups in self._pending.values())

  @property
  def dropped_late_groups_count(self) -> int:
    """Returns the count of groups dropped because `batch_idx < next_batch_idx`."""
    return self._dropped_late_groups
