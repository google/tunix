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

"""APIs of performance metrics for RL workflows.

Collect API:

  tracer = PerfTracer(devices, export_fn)

  1. Host span (Synchronous Timeline)

    with tracer.span("data_loading"):
      ...

  2. Device span (Asynchronous Timeline)

    with tracer.span("rollout", mesh.devices) as span:
      ...
      span.async_end(waitlist)
"""

from __future__ import annotations

from concurrent import futures
import contextlib
import threading
import time
from typing import Any, Callable

from absl import logging
import jax
import jaxtyping
import numpy as np
from tunix.perf import metrics
from tunix.perf.experimental import span


JaxDevice = Any
MetricsT = metrics.MetricsT
PerfSpanQuery = metrics.PerfSpanQuery
Span = span.Span


def generate_host_timeline_id() -> str:
  """Generates a string ID for a host timeline."""
  return "host-" + str(threading.get_ident())


def generate_device_timeline_id(id: str | JaxDevice) -> str:
  """Generates a string ID for a device timeline.

  Args:
    id: A string ID or a JAX device object.

  Returns:
    A string representation of the device ID. For a JAX device object, it will
    be the platform name followed by the device ID, e.g., "tpu0".

  Raises:
    ValueError: If the input ID type is not supported. Only string and JAX
    device objects (with platform and id attributes) are supported.
  """

  if isinstance(id, str):
    return id
  elif hasattr(id, "platform") and hasattr(id, "id"):
    # if it's a JAX device object, convert to string
    return getattr(id, "platform") + str(getattr(id, "id"))
  else:
    raise ValueError(f"Unsupport id type: {type(id)}")


def generate_device_timeline_ids(
    devices: list[str | JaxDevice] | np.ndarray | None,
) -> list[str]:
  """Generates a list of string IDs for a list of devices."""
  if devices is None:
    return []
  if isinstance(devices, np.ndarray):
    devices = devices.flatten().tolist()
  return [generate_device_timeline_id(device) for device in devices]


class NoopTracer:
  """An no-op tracer that does nothing."""

  def synchronize(self) -> None:
    pass

  def print(self) -> None:
    pass

  def export(self) -> MetricsT:
    return {}

  @property
  def all_devices(self) -> list[str | JaxDevice]:
    return []

  @contextlib.contextmanager
  def span(
      self,
      name: str,
      devices: list[str | JaxDevice] | np.ndarray | None = None,
      tags: dict[str, Any] | None = None,
  ):
    yield _AsyncWaitlist()


class PerfTracer(NoopTracer):
  """Provides an API to collect events to construct thread and devices timelines."""

  def __init__(
      self,
      devices: list[str | JaxDevice] | np.ndarray | None = None,
      export_fn: Callable[[Any], MetricsT] | None = None,
  ):
    self._export_fn = export_fn

    # align all timelines with the same born time.
    self._born = time.perf_counter()

    self._main_thread_id = generate_host_timeline_id()

    self._host_timelines: dict[str, Timeline] = {
        self._main_thread_id: Timeline(self._main_thread_id, self._born)
    }
    self._device_timelines: dict[str, AsyncTimeline] = {}
    # TODO(noghabi): add support for collecting trace on one device in the mesh
    # instead of all of them.
    if devices:
      for device in devices:
        self._get_or_create_device_timeline(generate_device_timeline_id(device))

  def _get_timelines(self) -> dict[str, Timeline]:
    timelines: dict[str, Timeline] = {}
    for timeline in self._host_timelines.values():
      timelines[timeline.id] = timeline
    for timeline in self._device_timelines.values():
      timelines[timeline.id] = timeline
    return timelines

  def _get_or_create_host_timeline(self, timeline_id: str) -> Timeline:
    if timeline_id not in self._host_timelines:
      self._host_timelines[timeline_id] = Timeline(timeline_id, self._born)
    return self._host_timelines[timeline_id]

  def _get_or_create_device_timeline(self, timeline_id: str) -> AsyncTimeline:
    if timeline_id not in self._device_timelines:
      self._device_timelines[timeline_id] = AsyncTimeline(
          timeline_id, self._born
      )
    return self._device_timelines[timeline_id]

  def _get_or_create_device_timelines(
      self, ids: list[str | JaxDevice] | np.ndarray | None
  ) -> BatchAsyncTimelines:
    return BatchAsyncTimelines([
        self._get_or_create_device_timeline(generate_device_timeline_id(id))
        for id in generate_device_timeline_ids(ids)
    ])

  def synchronize(self) -> None:
    _synchronize_devices()
    for timeline in self._device_timelines.values():
      timeline.wait_pending_spans()

  def print(self) -> None:
    self.synchronize()
    for timeline in self._get_timelines().values():
      print(f"\n[{timeline.id}]")
      print(timeline)

  def export(self) -> MetricsT:
    if self._export_fn is not None:
      return self._export_fn(self._get_timelines())
    else:
      return {}

  @property
  def all_devices(self) -> list[str | JaxDevice]:
    return list(self._device_timelines.keys())

  @contextlib.contextmanager
  def span(
      self,
      name: str,
      devices: list[str | JaxDevice] | np.ndarray | None = None,
      tags: dict[str, Any] | None = None,
  ):
    begin = time.perf_counter()
    host_timeline = self._get_or_create_host_timeline(
        generate_host_timeline_id()
    )
    host_timeline.start_span(name, begin, tags=tags)
    device_waitlist = _AsyncWaitlist()
    try:
      yield device_waitlist
    finally:
      end = time.perf_counter()
      host_timeline.stop_span(end)
      self._get_or_create_device_timelines(devices).span(
          name, begin, device_waitlist._data, tags=tags
      )  # pylint: disable=protected-access


Tracer = PerfTracer | NoopTracer


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


########################################################################
#
# internal only - utility classes and functions
#
########################################################################


class _AsyncWaitlist:
  """Provides an interface to collect waitlist for PerfTracer span()."""

  def __init__(self):
    self._data = []

  def async_end(self, waitlist: jaxtyping.PyTree) -> None:
    self._data.append(waitlist)


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


def _synchronize_devices():
  for device in jax.devices():
    jax.device_put(jax.numpy.array(0.0), device=device).block_until_ready()
