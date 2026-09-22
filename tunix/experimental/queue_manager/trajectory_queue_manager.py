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
from typing import Any, Deque, Dict, Optional, Set, Tuple

from tunix.experimental.common import datatypes
from tunix.rl.agentic.queue_manager import group_queue_manager

TrajectoryItem = datatypes.TrajectoryItem
GroupFn = group_queue_manager.GroupFn[TrajectoryItem]
FilterFn = group_queue_manager.FilterFn[TrajectoryItem]

# Reports that the batch cursor moved forward, with the new `next_batch_idx`.
CursorFn = Callable[[int], None]
# Reports `num_groups` groups dropped upstream for `batch_idx`.
DropFn = Callable[[int, int], None]


class GroupOrder(enum.Enum):
  """Order in which a queue manager yields ready groups."""

  # Completion order: any ready group, as soon as it exists.
  ARRIVAL = "arrival"
  # Prompt-batch order: only groups of the batch the cursor points at.
  PROMPT_BATCH = "prompt_batch"


def default_key_fn(item: TrajectoryItem) -> Hashable:
  """Groups items by `prompt_id`, falling back to identity when it is unset."""
  prompt_id = getattr(item, "prompt_id", None)
  if prompt_id is not None and prompt_id != "":
    return prompt_id
  return id(item)


def default_batch_idx_fn(item: TrajectoryItem) -> int:
  """Reads the prompt-batch index the dispatcher tagged the item with."""
  metadata = getattr(item, "metadata", None) or {}
  batch_idx = metadata.get("batch_idx")
  if batch_idx is None:
    raise ValueError(
        "BatchOrderedQueueManager requires a `batch_idx` in item metadata;"
        f" got metadata={metadata!r}."
    )
  return int(batch_idx)


def build_filter(
    max_staleness: int,
    current_policy_version: Callable[[], int] | None,
    filter_fn: Any | None,
) -> Any | None:
  """Composes the policy-staleness filter with a caller-supplied one.

  Args:
    max_staleness: Reject items produced more than this many policy versions
      behind. 0 disables the staleness check, leaving `filter_fn` alone.
    current_policy_version: Reads the trainer's current policy version.
    filter_fn: Optional filter to apply to whatever survives the staleness
      check.

  Returns:
    A filter returning `(valid, filtered_out)`, or `filter_fn` unchanged when
    no staleness check is wanted.
  """
  assert max_staleness >= 0, "max_staleness must be non-negative."
  if max_staleness <= 0 or current_policy_version is None:
    return filter_fn

  def _staleness_filter(group: Sequence[Any]) -> Any:
    min_allowed = current_policy_version() - max_staleness
    valid = [
        item for item in group if getattr(item, "policy_version", 0) >= min_allowed
    ]
    filtered = [
        item for item in group if getattr(item, "policy_version", 0) < min_allowed
    ]
    if filter_fn is not None:
      res = filter_fn(valid)
      if isinstance(res, tuple):
        return res[0], list(res[1]) + filtered
      return res, filtered
    return valid, filtered

  return _staleness_filter


class TrajectoryQueueManager(group_queue_manager.GroupQueueManager):
  """Specialized GroupQueueManager holding TrajectoryItems with ACK and abort support."""

  def __init__(
      self,
      *,
      num_generations: Optional[int] = None,
      group_fn: Optional[GroupFn] = None,
      filter_fn: Optional[FilterFn] = None,
      key_fn: Optional[Callable[[datatypes.TrajectoryItem], Hashable]] = None,
      batch_idx_fn: Optional[Callable[[datatypes.TrajectoryItem], int]] = None,
      on_drop: DropFn | None = None,
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
      batch_idx_fn: Optional function to read an item's prompt-batch index when
        reporting drops via `on_drop`. Defaults to `metadata["batch_idx"]`.
      on_drop: Optional callback invoked as `on_drop(batch_idx, 1)` whenever
        `filter_fn` discards an entire candidate group.
    """
    if key_fn is None and group_fn is None:
      key_fn = default_key_fn

    super().__init__(
        num_generations=num_generations,
        group_fn=group_fn,
        filter_fn=filter_fn,
        key_fn=key_fn,
    )
    self._batch_idx_fn = batch_idx_fn or default_batch_idx_fn
    self._on_drop = on_drop

  @classmethod
  def create(
      cls,
      num_generations: int = 1,
      max_staleness: int = 0,
      current_policy_version: Callable[[], int] | None = None,
      filter_fn: Any | None = None,
      on_drop: DropFn | None = None,
  ) -> "TrajectoryQueueManager":
    """Creates a grouped trajectory queue with optional policy staleness filtering.

    Args:
      num_generations: Number of trajectories that form one group.
      max_staleness: Reject groups produced more than this many policy versions
        behind. 0 disables the staleness filter.
      current_policy_version: Reads the trainer's current policy version; only
        used when `max_staleness > 0`.
      filter_fn: Optional additional filter applied to candidate groups.
      on_drop: Optional callback invoked as `on_drop(batch_idx, 1)` whenever
        `filter_fn` discards an entire candidate group.

    Returns:
      A queue that yields groups in completion order. For prompt-batch order,
      use `BatchOrderedQueueManager.create` instead.
    """
    return cls(
        num_generations=num_generations,
        filter_fn=build_filter(  # pyrefly: ignore[bad-argument-type]
            max_staleness, current_policy_version, filter_fn
        ),
        on_drop=on_drop,
    )

  def _on_candidate_group(
      self,
      valid_group: list[datatypes.TrajectoryItem],
      filtered_out: list[datatypes.TrajectoryItem],
  ) -> None:
    """Routes a resolved group and reports whole-group drops via `on_drop`."""
    super()._on_candidate_group(valid_group, filtered_out)
    if not valid_group and filtered_out and self._on_drop is not None:
      batch_idx = self._batch_idx_fn(filtered_out[0])
      self._on_drop(batch_idx, 1)

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

  async def get_group_batch(
      self, num_groups: int
  ) -> list[datatypes.TrajectoryItem]:
    """Retrieves a batch of TrajectoryItem groups."""
    out: list[datatypes.TrajectoryItem] = []
    for _ in range(num_groups):
      g = await self._get_one_ready_group()
      if not g:
        break
      out.extend(g)
    return out

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

  async def abort(self, exc: BaseException) -> None:
    """Aborts queue and unblocks all waiting consumers with the given exception."""
    if isinstance(exc, Exception):
      await self.put_exception(exc)
    await self.prepare_clear()


class BatchOrderedQueueManager(group_queue_manager.GroupQueueManager):
  """Serves ready groups in prompt-batch order, behind a cursor.

  Ready groups are staged per `batch_idx` (tagged by the dispatcher) instead of
  in one flat FIFO, and `get_ordered_group` only ever serves the batch the
  cursor points at. A consumer therefore sees whole prompt batches in dataset
  order, and a change in the returned `batch_idx` marks a batch boundary.

  Two indices move forward, and they differ by at most one batch:

  * `next_batch_idx` is the first batch that has not been trained.
    `commit_batch` and `skip` advance it. It is what a checkpoint should persist
    and what a staleness window should measure against.
  * the serving cursor is the batch `get_ordered_group` currently draws from. It
    runs one batch ahead of `next_batch_idx` while the consumer is training a
    batch it has already been handed in full, which is how the consumer learns
    the batch ended.

  A batch that will never be trained must not strand the cursor. `expect`
  declares how many groups a batch should contain; once that many have been
  resolved (admitted or dropped by `filter_fn`) the batch is sealed, and a
  sealed batch with nothing admitted is passed over without ever being served.
  `skip` does the same for batches that were never dispatched, such as the
  prefix already trained before a checkpoint restore.

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
      on_cursor_advance: CursorFn | None = None,
  ):
    """Initializes BatchOrderedQueueManager.

    Args:
      full_batch_size: Groups per prompt batch. Used as the expected group count
        for batches that `expect` was not called for. This is `rl_program`'s
        `full_batch_size`, not its `mini_batch_size`: a prompt batch is the unit
        the dataset cursor and the staleness window advance in, and a full batch
        may hold several optimizer updates.
      num_generations: Optional target number of trajectories per ready group
        when using default grouping.
      group_fn: Optional custom grouping function. If None, `num_generations`
        must be provided.
      filter_fn: Optional pluggable function to filter candidate groups.
      key_fn: Optional function to extract grouping key. Defaults to prompt_id
        fallback.
      batch_idx_fn: Optional function to read an item's prompt-batch index.
        Defaults to `metadata["batch_idx"]`.
      on_cursor_advance: Optional callback invoked with the new
        `next_batch_idx` whenever it moves forward.

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
    self._on_cursor_advance = on_cursor_advance

    # Ready groups staged per batch, replacing the base `_ready_groups` FIFO.
    self._pending: Dict[int, Deque[list[TrajectoryItem]]] = (
        collections.defaultdict(collections.deque)
    )
    # Groups a batch should contain, declared by `expect`.
    self._expected: Dict[int, int] = {}
    # Groups resolved so far, admitted or dropped.
    self._seen: Dict[int, int] = collections.defaultdict(int)
    # Groups admitted so far. A sealed batch with none is a hole.
    self._admitted: Dict[int, int] = collections.defaultdict(int)
    # Batches that will receive nothing further.
    self._sealed: Set[int] = set()

    self._next_batch_idx = 0
    self._serving_batch_idx = 0

  @classmethod
  def create(
      cls,
      num_generations: int = 1,
      full_batch_size: int = 1,
      max_staleness: int = 0,
      current_policy_version: Callable[[], int] | None = None,
      filter_fn: Any | None = None,
      on_cursor_advance: CursorFn | None = None,
  ) -> "BatchOrderedQueueManager":
    """Creates a prompt-batch-ordered queue, mirroring the arrival factory.

    Args:
      num_generations: Number of trajectories that form one group.
      full_batch_size: Groups per prompt batch.
      max_staleness: Reject groups produced more than this many policy versions
        behind. 0 disables the staleness filter.
      current_policy_version: Reads the trainer's current policy version; only
        used when `max_staleness > 0`.
      filter_fn: Optional additional filter applied to candidate groups.
      on_cursor_advance: Called with the new `next_batch_idx` whenever the
        cursor moves forward.

    Returns:
      A queue that yields groups in prompt-batch order.
    """
    return cls(
        num_generations=num_generations,
        full_batch_size=full_batch_size,
        filter_fn=build_filter(
            max_staleness, current_policy_version, filter_fn
        ),
        on_cursor_advance=on_cursor_advance,
    )

  async def get_batch(self, batch_size: int) -> list[TrajectoryItem]:
    """Unsupported here; the base implementation drains `_ready_groups`.

    Args:
      batch_size: Unused.

    Raises:
      NotImplementedError: Always. This class stages groups per batch, so the
        base FIFO is always empty and the inherited implementation would park
        forever. Use `get_ordered_group` instead.
    """
    del batch_size
    raise NotImplementedError(
        "BatchOrderedQueueManager serves groups in prompt-batch order; use"
        " get_ordered_group() instead of get_batch()."
    )

  @property
  def next_batch_idx(self) -> int:
    """First batch not yet trained; advanced by `commit_batch` / `skip`."""
    return self._next_batch_idx

  @property
  def serving_batch_idx(self) -> int:
    """Batch `get_ordered_group` currently draws from."""
    return self._serving_batch_idx

  def next_batch_after(self, batch_idx: int) -> int:
    """Returns the first untrained, non-hole batch index after `batch_idx`."""
    candidate = max(self._next_batch_idx, batch_idx + 1)
    while (
        candidate in self._sealed
        and self._admitted.get(candidate, 0) == 0
    ):
      candidate += 1
    return candidate

  def expect(self, batch_idx: int, num_groups: int) -> None:
    """Declares how many groups a batch should eventually contain.

    Without this a dropped group is indistinguishable from a straggler, and a
    batch that loses every group would strand the cursor forever. The final
    batch of a dataset may declare fewer than `full_batch_size`.

    Call this before the batch's first group is put. A later call that *raises*
    the count re-opens a batch that sealed early, but only while the serving
    cursor still sits at or behind it; past that point the batch has already
    been handed out in full and the extra groups are discarded at
    `commit_batch`.

    Args:
      batch_idx: The prompt batch being announced.
      num_groups: Groups the batch should contain.
    """
    if batch_idx < self._next_batch_idx:
      return
    self._expected[batch_idx] = num_groups
    self._maybe_seal(batch_idx)
    self._advance_over_holes()
    self._have_ready.set()

  def dropped(self, batch_idx: int, num_groups: int = 1) -> None:
    """Records `num_groups` groups dropped upstream for `batch_idx`.

    When `raw_q` filters out a whole group before it reaches `critique_stage`,
    `scored_q` never sees those items in `put()`. Calling `dropped` increments
    the batch's resolved count (`_seen`) so the batch can still seal (or pass
    over as a hole if nothing in it survived).

    Args:
      batch_idx: The prompt batch that lost the group(s).
      num_groups: Number of dropped groups to account for.

    Raises:
      ValueError: If `num_groups` is not positive.
    """
    if num_groups <= 0:
      raise ValueError(f"num_groups must be positive, got {num_groups}.")
    if batch_idx < self._next_batch_idx:
      return
    self._seen[batch_idx] += num_groups
    self._maybe_seal(batch_idx)
    self._advance_over_holes()
    self._have_ready.set()

  def skip(self, begin: int, end: int) -> None:
    """Passes over batches `[begin, end)` without ever serving them.

    Used for the prefix already trained before a checkpoint restore, and for
    batches that lost every group.

    Args:
      begin: First batch to pass over.
      end: One past the last batch to pass over.
    """
    for batch_idx in range(begin, end):
      self._drop_batch_state(batch_idx)
    self._advance_to(end)
    self._advance_over_holes()
    self._have_ready.set()

  def commit_batch(self, batch_idx: int) -> None:
    """Releases a batch's state after its optimizer step.

    Named apart from `TrajectoryQueueManager.commit(step, groups)`: that one
    takes a training step and the groups it consumed, this one takes a batch
    index and already knows which groups belong to it.

    Args:
      batch_idx: The batch that was trained.
    """
    self._drop_batch_state(batch_idx)
    self._advance_to(batch_idx + 1)
    self._advance_over_holes()
    self._have_ready.set()

  async def get_ordered_group(
      self,
      batch_idx: Optional[int] = None,
  ) -> Optional[Tuple[int, list[TrajectoryItem]]]:
    """Waits for the next group of the batch at the cursor.

    Named apart from `TrajectoryQueueManager.get_group`, which yields a bare
    group in completion order and `[]` at EOF.

    Args:
      batch_idx: Optional batch index to restrict consumption to. When set,
        returns `None` as soon as `batch_idx` is sealed and drained (or moved
        past) instead of advancing into `batch_idx + 1`. This lets a consumer
        finish a short batch without waiting for the next batch to produce a
        group.

    Returns:
      `(batch_idx, group)`, or None once the queue is closed or the requested
      `batch_idx` has no more groups to yield. `batch_idx` never decreases, so
      a change in it marks a batch boundary.

    Raises:
      Exception: Whatever was set via `put_exception` / `abort`.
    """
    while True:
      if self._exc:
        raise self._exc
      if self._clearing:
        return None

      if batch_idx is not None:
        pending = self._pending.get(batch_idx)
        if pending:
          return batch_idx, pending.popleft()
        if batch_idx in self._sealed:
          if self._serving_batch_idx == batch_idx:
            self._serving_batch_idx = batch_idx + 1
          return None
        if batch_idx < self._serving_batch_idx or self._closed:
          return None
      else:
        served = self._serve()
        if served is not None:
          return served
        if self._closed:
          return None

      await self._have_ready.wait()
      self._have_ready.clear()

  def _serve(self) -> Optional[Tuple[int, list[TrajectoryItem]]]:
    """Pops the next group at the cursor, passing over sealed empty batches."""
    while True:
      batch_idx = self._serving_batch_idx
      pending = self._pending.get(batch_idx)
      if pending:
        return batch_idx, pending.popleft()

      if batch_idx not in self._sealed:
        return None

      if self._admitted.get(batch_idx, 0) == 0:
        # A hole: nothing was ever admitted, so nothing will ever be trained.
        self._drop_batch_state(batch_idx)
        self._advance_to(batch_idx + 1)
      else:
        # Fully served; `commit_batch` still owes the training-side advance.
        self._serving_batch_idx = batch_idx + 1

  def _on_candidate_group(
      self,
      valid_group: list[TrajectoryItem],
      filtered_out: list[TrajectoryItem],
  ) -> None:
    """Stages a resolved group under its batch and updates drop accounting."""
    representative = next(iter(valid_group or filtered_out), None)
    if representative is None:
      # A filter_fn that returned nothing at all: no batch to attribute this
      # to, and no group to count. The base class's default hook is likewise a
      # no-op here.
      return
    batch_idx = self._batch_idx_fn(representative)

    if filtered_out:
      self._filtered_groups.append(filtered_out)

    if batch_idx < self._next_batch_idx:
      # The batch was trained or skipped already; its accounting is gone.
      return

    self._seen[batch_idx] += 1
    if valid_group:
      self._admitted[batch_idx] += 1
      self._pending[batch_idx].append(valid_group)

    self._maybe_seal(batch_idx)
    self._advance_over_holes()
    self._have_ready.set()

  def _maybe_seal(self, batch_idx: int) -> None:
    """Seals a batch once every group it should contain has been resolved."""
    expected = self._expected.get(batch_idx, self.full_batch_size)
    if self._seen.get(batch_idx, 0) >= expected:
      self._sealed.add(batch_idx)
    elif batch_idx >= self._serving_batch_idx:
      # An `expect` that raises the count un-seals a prematurely sealed batch.
      # Only while the batch is still at or ahead of the serving cursor: that
      # cursor never moves backwards, so un-sealing behind it would withhold a
      # batch nothing can ever serve again.
      self._sealed.discard(batch_idx)

  def _advance_over_holes(self) -> None:
    """Passes over any sealed empty batches sitting at `next_batch_idx`."""
    while (
        self._serving_batch_idx == self._next_batch_idx
        and self._next_batch_idx in self._sealed
        and self._admitted.get(self._next_batch_idx, 0) == 0
    ):
      hole = self._next_batch_idx
      self._drop_batch_state(hole)
      self._advance_to(hole + 1)

  def _advance_to(self, batch_idx: int) -> None:
    """Moves both indices to `batch_idx`, reporting a cursor advance."""
    if batch_idx <= self._next_batch_idx:
      return
    self._next_batch_idx = batch_idx
    if self._serving_batch_idx < batch_idx:
      self._serving_batch_idx = batch_idx
    if self._on_cursor_advance is not None:
      self._on_cursor_advance(batch_idx)

  def _drop_batch_state(self, batch_idx: int) -> None:
    """Forgets everything staged or counted for a batch."""
    self._pending.pop(batch_idx, None)
    self._expected.pop(batch_idx, None)
    self._seen.pop(batch_idx, None)
    self._admitted.pop(batch_idx, None)
    self._sealed.discard(batch_idx)

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

