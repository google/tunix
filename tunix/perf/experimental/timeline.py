# Copyright 2026 Google LLC
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

"""Span based data model."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent import futures
import dataclasses
import threading
import time
from typing import Any, Callable, cast

from absl import logging
import jax
import jaxtyping


@dataclasses.dataclass
class Span:
  """A duration of time with a name, beginning, and end.

  Attributes:
    name: The name of the span.
    begin: The start time of the span.
    end: The end time of the span.
    id: The ID of the span.
    parent_id: The ID of the parent span.
    tags: A mapping of tags associated with the span. Tags are used for post-
      processing and grouping spans. Some well-known tags are defined as
      constants in `tunix.perf.experimental.constants` (e.g.,
      constants.GLOBAL_STEP). Users can also add arbitrary tags.
  """

  name: str
  begin: float
  end: float = dataclasses.field(default=float("inf"), init=False)
  id: int
  parent_id: int | None = None
  tags: Mapping[str, Any] | None = None

  def __post_init__(self) -> None:
    self.tags = dict(self.tags) if self.tags is not None else {}

  def add_tag(self, key: str, value: Any) -> None:
    """Adds a tag to the span.

    Args:
      key: The tag key.
      value: The tag value.
    """
    if key in self.tags:
      logging.warning(
          "Span '%s' (id=%s): Tag %r already exists with value %r."
          " Overwriting with %r.",
          self.name,
          self.id,
          key,
          self.tags[key],
          value,
      )
    self.tags[key] = value

  def _format_relative(self, born_at: float) -> str:
    """Returns a string representation of the span with relative times.

    The times are relative to `born_at`.
    """
    begin = self.begin - born_at
    end = self.end - born_at
    out = f"[{self.id}] {self.name}: {begin:.6f}, {end:.6f}"
    if self.parent_id is not None:
      out += f" (parent={self.parent_id})"
    if self.tags:
      out += f", tags={self.tags}"
    return out

  @property
  def ended(self) -> bool:
    """Whether the span has ended."""
    return self.end != float("inf")

  @property
  def duration(self) -> float:
    """The duration of the span."""
    return self.end - self.begin


# TODO(noghabi): maybe reuse `callback_on_ready` in tunix.rl.
def _async_wait(
    waitlist: jaxtyping.PyTree,
    success: Callable[[], None],
    failure: Callable[[Exception], None],
) -> threading.Thread:
  """Asynchronously waits for all JAX computations to finish.

  Args:
    waitlist: A JAX PyTree containing computations to wait for.
    success: A callable to execute upon successful completion of the waitlist.
    failure: A callable to execute if an exception occurs during
      jax.block_until_ready.

  Returns:
    A threading.Thread object that is executing the wait.
  """
  fut = futures.Future()

  def callback(f: futures.Future[Any]) -> None:
    e = f.exception()
    if e is None:
      success()
    else:
      failure(cast(Exception, e))

  fut.add_done_callback(callback)

  def wait() -> None:
    try:
      jax.block_until_ready(waitlist)
    except Exception as e:  # pylint: disable=broad-exception-caught
      fut.set_exception(e)
    else:
      fut.set_result(waitlist)

  t = threading.Thread(target=wait)
  t.start()
  return t


class Timeline:
  """A thread-safe data structure for recording spans of execution time.

  A timeline represents a chronological sequence of spans or events. It supports
  nested spans by keeping track of the active span stack, allowing parent-child
  relationships to model hierarchical execution.

  The timeline is organized into a series of steps, defined by when `commit_step`
  is called. It is expected that spans do not cross commit boundaries (i.e.,
  all spans started in a step must finish before the step is committed).

  Attributes:
    id: A unique identifier for the timeline.
    born: The time when the timeline was created.
    _spans_stack: Stack of active (uncompleted) synchronous span IDs.
    _cur_step: Dict of active and completed spans accumulated in this step.
    _last_span_id: ID of the last created span.
    _committed_steps: Archive of committed step dictionaries. Each dictionary
      represents a step and contains the spans that were active during that
      step. Once a step is committed, the spans in cur_step are moved to the
      committed steps and cannot be modified. This allows for lock-free read
      access to the committed spans and copy-on-write for updates.
    _lock: A reentrant lock to protect timeline state.
  """

  def __init__(self, id: str, born: float):
    """Initializes the Timeline instance.

    Args:
      id: A unique string identifier for the timeline.
      born: The creation time of the timeline, used as the base for relative
        time calculations.
    """
    self.id = id
    self.born = born
    self._spans_stack: list[int] = []
    self._cur_step: dict[int, Span] = {}
    self._last_span_id = -1
    self._committed_steps: list[dict[int, Span]] = []
    # Multi-consumer bookkeeping for bounded-memory mode. ``_dropped_step_count``
    # is the number of committed steps that have been dropped from
    # ``_committed_steps`` so far; ``_consumer_cursors[name]`` is the absolute
    # count of committed steps each registered consumer has finished
    # processing. Steps are released from memory once every registered consumer
    # has advanced past them. With no consumers registered, the timeline keeps
    # everything (the default; preserves the pre-existing behavior for tests
    # and direct users).
    self._dropped_step_count: int = 0
    self._consumer_cursors: dict[str, int] = {}
    self._lock = threading.RLock()

  @property
  def committed_steps(self) -> list[dict[int, Span]]:
    """The immutable chunks of committed steps lock-free."""
    return self._committed_steps

  @property
  def cur_step(self) -> dict[int, Span]:
    """A copy of the active current step spans."""
    with self._lock:
      return dict(self._cur_step)

  @property
  def all_steps(self) -> list[dict[int, Span]]:
    """A list of all step dictionaries (history and current)."""
    with self._lock:
      return self._committed_steps + [dict(self._cur_step)]

  def start_span(
      self, name: str, begin: float, tags: Mapping[str, Any] | None = None
  ) -> Span:
    """Starts a new span and pushes it to active spans.

    Args:
      name: The name of the span.
      begin: The start time of the span.
      tags: An optional dictionary of tags to associate with the span.

    Returns:
      The newly created Span object.
    """
    with self._lock:
      parent_id = self._spans_stack[-1] if self._spans_stack else None
      self._last_span_id += 1
      span_id = self._last_span_id
      _span = Span(
          name=name, begin=begin, id=span_id, parent_id=parent_id, tags=tags
      )
      self._cur_step[span_id] = _span
      self._spans_stack.append(span_id)
    return _span

  def stop_span(self, end: float) -> None:
    """Ends the current span on the stack.

    Args:
      end: The end time to record for the span.

    Raises:
      ValueError: If there are no active spans to end, or if the end time is
        before the span's begin time.
    """
    with self._lock:
      if not self._spans_stack:
        raise ValueError(f"{self.id}: no more spans to end.")
      span_id = self._spans_stack.pop()
      _span = self._cur_step[span_id]
      if _span.begin > end:
        raise ValueError(
            f"{self.id}: span '{_span.name}' ended at {end:.6f} before it began"
            f" at {_span.begin:.6f}."
        )
      _span.end = end

  def commit_step(self) -> None:
    """Commits current step spans to history, purging any uncompleted/dangling spans."""
    with self._lock:
      to_remove = []
      for sid, span in self._cur_step.items():
        if span.end == float("inf") or sid in self._spans_stack:
          logging.warning(
              "Purging uncompleted span %r crossing step boundary in"
              " timeline %s",
              span.name,
              self.id,
          )
          to_remove.append(sid)

      for sid in to_remove:
        self._cur_step.pop(sid, None)
      self._spans_stack.clear()

      # Archive current step dict and reset via copy-on-write
      self._committed_steps = list(self._committed_steps) + [self._cur_step]

      self._cur_step = {}

  @property
  def dropped_step_count(self) -> int:
    """Number of committed steps that have been dropped from memory so far.

    Together with ``len(committed_steps)``, this lets callers compute the
    absolute count of committed steps observed by this timeline:
    ``absolute_count = dropped_step_count + len(committed_steps)``.
    """
    return self._dropped_step_count

  def register_consumer(
      self, consumer_id: str, *, start_at_current_head: bool = True
  ) -> None:
    """Registers a consumer of this timeline's committed steps.

    A consumer represents a downstream system that reads committed step dicts
    -- e.g. a trace writer, an aggregator that pushes metrics to TensorBoard,
    a registry computer, etc. Once at least one consumer is registered, the
    timeline enters bounded-memory mode: committed steps are retained until
    every registered consumer has advanced its cursor past them, then they
    are released.

    With ``start_at_current_head=True`` (the default), the new consumer
    inherits the current head of the committed history, so it is only
    responsible for steps committed *after* registration. Set this to False
    if the consumer wants to replay everything currently held in memory.

    Args:
      consumer_id: A stable, unique identifier for this consumer. Calling
        ``register_consumer`` with an already-registered id is a no-op (the
        consumer's cursor is unchanged).
      start_at_current_head: When True, the consumer's cursor is initialized
        to the current absolute committed-step count (i.e. it skips anything
        already held). When False, the cursor is initialized to
        ``dropped_step_count`` so the consumer must traverse all currently
        held steps before any can be released.
    """
    with self._lock:
      if consumer_id in self._consumer_cursors:
        return
      if start_at_current_head:
        self._consumer_cursors[consumer_id] = (
            self._dropped_step_count + len(self._committed_steps)
        )
      else:
        self._consumer_cursors[consumer_id] = self._dropped_step_count

  def unregister_consumer(self, consumer_id: str) -> int:
    """Unregisters a previously registered consumer.

    Useful when a consumer shuts down (e.g. trace writer disabled mid-run or
    closed) -- without this, that consumer's stale cursor would pin the
    timeline's memory forever. Removal of a laggard cursor may immediately
    advance the effective min cursor; any steps that become reachable as a
    result are released right away.

    If no consumers remain after this call, the timeline reverts to keeping
    all committed steps.

    Args:
      consumer_id: The id passed to ``register_consumer``. Unknown ids are a
        no-op.

    Returns:
      The number of committed step dicts that were dropped from memory as a
      direct result of removing this consumer (zero in the common case where
      this consumer was not the slowest, or where no consumers remain).
    """
    with self._lock:
      if consumer_id not in self._consumer_cursors:
        return 0
      del self._consumer_cursors[consumer_id]
      if not self._consumer_cursors:
        return 0
      min_cursor = min(self._consumer_cursors.values())
      to_drop = max(0, min_cursor - self._dropped_step_count)
      if to_drop == 0:
        return 0
      self._drop_oldest_committed_steps_locked(to_drop)
      return to_drop

  def advance_consumer(self, consumer_id: str, n: int = 1) -> int:
    """Advances ``consumer_id``'s cursor by ``n`` committed steps.

    Once every registered consumer has advanced past a step, that step is
    released from the in-memory ``committed_steps`` list. This is the path by
    which any downstream consumer (trace writer, aggregator, ...) signals "I
    am done with the next N committed steps; free their memory if no one else
    still needs them".

    Args:
      consumer_id: A consumer previously passed to ``register_consumer``.
      n: How many committed steps the consumer just finished processing. Must
        be non-negative. Advancing past the absolute committed-step count is
        clamped to the current count (so callers don't have to worry about
        racing with new commits).

    Returns:
      The number of committed step dicts that were dropped from memory as a
      result of this call. Zero if some other registered consumer is still
      behind, or if ``n`` is zero.

    Raises:
      ValueError: If ``n`` is negative or ``consumer_id`` was never
        registered.
    """
    if n < 0:
      raise ValueError(f"n must be non-negative, got {n}")
    with self._lock:
      if consumer_id not in self._consumer_cursors:
        raise ValueError(
            f"consumer {consumer_id!r} is not registered on timeline"
            f" {self.id!r}; call register_consumer first."
        )
      if n == 0:
        return 0
      max_cursor = self._dropped_step_count + len(self._committed_steps)
      self._consumer_cursors[consumer_id] = min(
          self._consumer_cursors[consumer_id] + n, max_cursor
      )
      min_cursor = min(self._consumer_cursors.values())
      to_drop = max(0, min_cursor - self._dropped_step_count)
      if to_drop == 0:
        return 0
      self._drop_oldest_committed_steps_locked(to_drop)
      return to_drop

  def _drop_oldest_committed_steps_locked(self, n: int) -> None:
    """Internal: drops the ``n`` oldest committed step dicts.

    Must be called with ``self._lock`` held. ``n`` must be in
    ``[0, len(self._committed_steps)]``. Uses copy-on-write on
    ``_committed_steps`` so concurrent lock-free readers iterating a previous
    snapshot are unaffected.
    """
    if n <= 0:
      return
    self._committed_steps = self._committed_steps[n:]
    self._dropped_step_count += n

  def __repr__(self) -> str:
    parts = [f"Timeline({self.id}, {self.born:.6f})\n"]
    with self._lock:
      if self._cur_step:
        parts.append(f"Current Step -{len(self._committed_steps)}:\n")
        for s in sorted(self._cur_step.values(), key=lambda span: span.id):
          parts.append(f"  {s._format_relative(self.born)}\n")
      for i, step in enumerate(reversed(self._committed_steps)):
        if step:
          parts.append(f"Committed Step -{i}:\n")
          for s in sorted(step.values(), key=lambda span: span.id):
            parts.append(f"  {s._format_relative(self.born)}\n")
    return "".join(parts)


class AsyncTimeline(Timeline):
  """A timeline with asynchronously closing spans."""

  def __init__(self, id: str, born: float):
    """Initializes the AsyncTimeline instance.

    Args:
      id: A unique string identifier for the timeline.
      born: The creation time of the timeline, used as the base for relative
        time calculations.
    """
    super().__init__(id, born)
    self._threads: list[threading.Thread] = []

  def span(
      self,
      name: str,
      thread_span_begin: float,
      waitlist: jaxtyping.PyTree,
      tags: Mapping[str, Any] | None = None,
  ) -> None:
    """Records a new span for a timeline.

    Args:
      name: The name of the span.
      thread_span_begin: The time at which the span was initiated in the calling
        thread (e.g., using time.perf_counter()).
      waitlist: A JAX PyTree (e.g., an array or a list of arrays) that this span
        is waiting on. The span will end once all computations in the waitlist
        are ready.
      tags: An optional dictionary of tags to associate with the span.
    """

    # Capture parent_id at schedule time
    with self._lock:
      # Async spans cannot be parents of other spans (sync or async), so we
      # don't push the new span_id to self._active_spans. They are always
      # "leaves".
      parent_id = self._spans_stack[-1] if self._spans_stack else None
      self._last_span_id += 1
      span_id = self._last_span_id

    def on_success() -> None:
      # Capture time immediately to avoid including lock contention in the span.
      end = time.perf_counter()

      with self._lock:
        # TODO(noghabi): figure out why it used to be Max of when the thread
        # launched it and when the timeline finished the previous op??
        # used to be: begin = max(thread_span_begin, self._last_op_end_time)

        # TODO(noghabi): create the span with inf end time and update it here,
        # then post process such spans instead of creating and adding them here.
        _span = Span(
            name=name,
            begin=thread_span_begin,
            id=span_id,
            parent_id=parent_id,
            tags=tags,
        )
        _span.end = end
        self._cur_step[span_id] = _span

    def on_failure(e: Exception) -> None:
      # TODO(noghabi):Capture the span even if it fails, but add a tag that it
      # failed and process accordingly.
      # Metrics are best effort and should not raise exceptions.
      logging.error(
          "Timeline span '%s' (id=%d) failed: %s", name, span_id, e, exc_info=e
      )

    if not waitlist:
      on_success()
    else:
      t = _async_wait(waitlist=waitlist, success=on_success, failure=on_failure)
      with self._lock:
        self._threads = [t for t in self._threads if t.is_alive()]
        self._threads.append(t)

  def wait_pending_spans(self) -> None:
    """Waits for all current pending spans to finish."""
    with self._lock:
      cur_threads = list(self._threads)
      self._threads.clear()
    for t in cur_threads:
      t.join()


class BatchAsyncTimelines:
  """A helper class to record spans across multiple AsyncTimeline instances."""

  def __init__(self, timelines: list[AsyncTimeline]):
    """Initializes the BatchAsyncTimelines instance.

    Args:
      timelines: A list of AsyncTimeline objects to manage.
    """
    self._timelines = timelines

  def span(
      self,
      name: str,
      thread_span_begin: float,
      waitlist: jaxtyping.PyTree,
      tags: Mapping[str, Any] | None = None,
  ) -> None:
    """Records a new span across multiple timelines.

    Args:
      name: The name of the span.
      thread_span_begin: The time at which the span was initiated in the calling
        thread (e.g., using time.perf_counter()).
      waitlist: A JAX PyTree (e.g., an array or a list of arrays) that this span
        is waiting on. The span will end once all computations in the waitlist
        are ready.
      tags: An optional mapping of tags to associate with the span.
    """
    # TODO(noghabi): This creates N threads all waiting on the same waitlist.
    # Change this to a single thread waiting on the waitlist and then updating
    # all timelines in a single callback/thread.

    for timeline in self._timelines:
      timeline.span(name, thread_span_begin, waitlist, tags=tags)
