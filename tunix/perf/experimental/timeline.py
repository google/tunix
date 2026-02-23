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

from concurrent import futures
import threading
import time
from typing import Any, Callable

from absl import logging
import jax
import jaxtyping

JaxDevice = Any


class Span:
  """Represents a duration of time with a name, beginning, and end.

  Attributes:
    name: The name of the span.
    begin: The start time of the span.
    end: The end time of the span.
    id: The ID of the span.
    parent_id: The ID of the parent span.
    tags: A dictionary of tags associated with the span. Tags are used for post-
      processing and grouping spans. Some well-known tags are defined as
      constants in `tunix.perf.experimental.constants` (e.g.,
      constants.GLOBAL_STEP). Users can also add arbitrary tags.
  """

  name: str
  begin: float
  end: float
  tags: dict[str, Any]
  id: int
  parent_id: int | None

  def __init__(
      self,
      name: str,
      begin: float,
      id: int,
      parent_id: int | None = None,
      tags: dict[str, Any] | None = None,
  ):
    self.name = name
    self.begin = begin
    self.id = id
    self.parent_id = parent_id
    self.tags = tags or {}
    self.end = float("inf")

  def add_tag(self, key: str, value: Any) -> None:
    """Adds a tag to the span.

    Args:
      key: The tag key.
      value: The tag value.
    """
    if key in self.tags:
      logging.warning(
          "Tag '%s' already exists with value '%s'. Overwriting with '%s'.",
          key,
          self.tags[key],
          value,
      )
    self.tags[key] = value

  def __repr__(self, born_at: float = 0.0) -> str:
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
    """Returns True if the span has ended."""

    return self.end != float("inf")

  @property
  def duration(self) -> float:
    """Returns the duration of the span."""
    return self.end - self.begin


class Timeline:
  """Manages a sequence of spans or events."""

  def __init__(self, id: str, born: float):
    self.id = id
    self.born = born
    self.spans: dict[int, Span] = {}
    self._active_spans: list[int] = []  # stack of active span IDs
    self._lock = threading.Lock()
    self._last_span_id = -1

  def start_span(
      self, name: str, begin: float, tags: dict[str, Any] | None = None
  ) -> Span:
    """Starts a new span and pushes it to active spans."""
    with self._lock:
      parent_id = self._active_spans[-1] if self._active_spans else None
      self._last_span_id += 1
      span_id = self._last_span_id
      _span = Span(
          name=name, begin=begin, id=span_id, parent_id=parent_id, tags=tags
      )
      self.spans[span_id] = _span
      self._active_spans.append(span_id)
    return _span

  def stop_span(self, end: float) -> None:
    """Ends the current span on the stack."""
    with self._lock:
      if not self._active_spans:
        raise ValueError(f"{self.id}: no more spans to end.")
      span_id = self._active_spans.pop()
      _span = self.spans[span_id]
      if _span.begin > end:
        # TODO(noghabi): should I raise an error here instead?
        logging.error(
            "%s: span '%s' ended at %.6f before it began at %.6f.",
            self.id,
            _span.name,
            end,
            _span.begin,
        )
      _span.end = end

  def __repr__(self) -> str:
    out = f"Timeline({self.id}, {self.born:.6f})\n"
    for s in sorted(self.spans.values(), key=lambda span: span.id):
      out += f"{s.__repr__(self.born)}\n"
    return out


class AsyncTimeline(Timeline):
  """Manages a timelines with asynchronously closing spans."""

  def __init__(self, id: str, born: float):
    super().__init__(id, born)
    self._threads: list[threading.Thread] = []

  def span(
      self,
      name: str,
      thread_span_begin: float,
      waitlist: jaxtyping.PyTree,
      tags: dict[str, Any] | None = None,
  ) -> None:
    """Record a new span for a timeline."""

    # Capture parent_id at schedule time
    with self._lock:
      parent_id = self._active_spans[-1] if self._active_spans else None
      self._last_span_id += 1
      span_id = self._last_span_id

    def on_success():
      # Capture time immediately to avoid including lock contention in the span.
      end = time.perf_counter()

      with self._lock:
        # TODO (noghabi): figure out why it used to be Max of when the thread
        # launched it and when the timeline finished the previous op??
        # begin = max(thread_span_begin, self._last_op_end_time)

        _span = Span(
            name=name,
            begin=thread_span_begin,
            id=span_id,
            parent_id=parent_id,
            tags=tags,
        )
        _span.end = end
        self.spans[span_id] = _span

    def on_failure(e: Exception):
      raise e

    if not waitlist:
      on_success()
    else:
      t = _async_wait(waitlist=waitlist, success=on_success, failure=on_failure)
      with self._lock:
        self._threads.append(t)

  def wait_pending_spans(self) -> None:
    for t in self._threads:
      t.join()


class BatchAsyncTimelines:

  def __init__(self, timelines: list[AsyncTimeline]):
    self._timelines = timelines

  def span(
      self,
      name: str,
      thread_span_begin: float,
      waitlist: jaxtyping.PyTree,
      tags: dict[str, Any] | None = None,
  ):
    for timeline in self._timelines:
      timeline.span(name, thread_span_begin, waitlist, tags=tags)


# TODO(noghabi): why do I need parent_id at all?
# TODO(noghabi): remove metric items after a while. the list will grow without bound.


# TODO(yangmu): maybe reuse `callback_on_ready` in tunix.rl.
def _async_wait(
    waitlist: jaxtyping.PyTree,
    success: Callable[[], None],
    failure: Callable[[Exception], None],
) -> threading.Thread:
  """Asynchronously wait for all JAX computations to finish."""
  fut = futures.Future()

  def callback(f):
    e = f.exception()
    if e is None:
      success()
    else:
      failure(e)

  fut.add_done_callback(callback)

  def wait():
    try:
      jax.block_until_ready(waitlist)
    except Exception as e:  # pylint: disable=broad-exception-caught
      fut.set_exception(e)
    else:
      fut.set_result(waitlist)

  t = threading.Thread(target=wait)
  t.start()
  return t
