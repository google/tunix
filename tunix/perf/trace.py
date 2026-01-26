# Copyright 2025 Google LLC
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

  1. span group

    with tracer.span_group("global_step"):
      with tracer.span_group("mini_batch"):
        with tracer.span_group("micro_batch"):
          ...

  2. thread span

    with tracer.span("data_loading"):
      ...

  3. device span

    with tracer.span("rollout", mesh.devices) as span:
      ...
      span.device_end(waitlist)
"""

from __future__ import annotations

from concurrent import futures
import contextlib
import logging
import threading
import time
from typing import Any, Callable

import jax
import jaxtyping
import numpy as np
from tunix.perf import metrics
from tunix.perf import span


JaxDevice = Any
MetricsT = metrics.MetricsT
PerfSpanQuery = metrics.PerfSpanQuery
Span = span.Span
SpanGroup = span.SpanGroup


def create_thread_timeline_id() -> str:
  return "thread-" + str(threading.get_ident())


def create_device_timeline_id(id: str | JaxDevice) -> str:
  if isinstance(id, str):
    return id
  elif hasattr(id, "platform") and hasattr(id, "id"):
    # if it's a JAX device object, convert to string
    return getattr(id, "platform") + str(getattr(id, "id"))
  else:
    raise ValueError(f"Unsupport id type: {type(id)}")


def create_device_timeline_ids(
    devices: list[str | JaxDevice] | np.ndarray | None,
) -> list[str]:
  if devices is None:
    return []
  if isinstance(devices, np.ndarray):
    devices = devices.flatten().tolist()
  return [create_device_timeline_id(device) for device in devices]


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
  def span_group(self, name: str):
    yield

  @contextlib.contextmanager
  def span(
      self, name: str, devices: list[str | JaxDevice] | np.ndarray | None = None
  ):
    yield _DeviceWaitlist()


class PerfTracer(NoopTracer):
  """Provides an API to collect events to construct thread and devices timelines."""

  def __init__(
      self,
      devices: list[str | JaxDevice] | np.ndarray | None = None,
      export_fn: Callable[[PerfSpanQuery], MetricsT] | None = None,
  ):
    self._export_fn = export_fn

    # align all timelines with the same born time.
    self._born = time.perf_counter()

    self._main_thread_id = create_thread_timeline_id()

    self._thread_timelines: dict[str, ThreadTimeline] = {
        self._main_thread_id: ThreadTimeline(self._main_thread_id, self._born)
    }
    self._device_timelines: dict[str, DeviceTimeline] = {}
    if devices:
      for device in devices:
        self._get_or_create_device_timeline(device)

  def _get_timelines(self) -> dict[str, Timeline]:
    timelines: dict[str, Timeline] = {}
    for timeline in self._thread_timelines.values():
      timelines[timeline.id] = timeline
    for timeline in self._device_timelines.values():
      timelines[timeline.id] = timeline
    return timelines

  def _get_or_create_thread_timeline(self, id: str) -> ThreadTimeline:
    if id not in self._thread_timelines:
      self._thread_timelines[id] = ThreadTimeline(
          id,
          self._born,
          span.span_group_stack_clone(
              self._thread_timelines[self._main_thread_id].stack
          ),
      )
    return self._thread_timelines[id]

  def _get_or_create_device_timeline(
      self, id: str | JaxDevice
  ) -> DeviceTimeline:
    tid = create_device_timeline_id(id)

    if tid not in self._device_timelines:
      self._device_timelines[tid] = DeviceTimeline(tid, self._born)
    return self._device_timelines[tid]

  def _get_or_create_device_timelines(
      self, ids: list[str | JaxDevice] | np.ndarray | None
  ) -> BatchDeviceTimeline:
    return BatchDeviceTimeline([
        self._get_or_create_device_timeline(id)
        for id in create_device_timeline_ids(ids)
    ])

  def synchronize(self) -> None:
    _synchronize_devices()
    for timeline in self._device_timelines.values():
      timeline.wait_pending_spans()

  def print(self) -> None:
    self.synchronize()
    for timeline in self._get_timelines().values():
      print(f"\n[{timeline.id}]")
      span.span_group_print(timeline.root, self._born)

  def export(self) -> MetricsT:
    if self._export_fn is not None:
      query = PerfSpanQuery(self._get_timelines(), self._main_thread_id)
      return self._export_fn(query)
    else:
      return {}

  @property
  def all_devices(self) -> list[str | JaxDevice]:
    return list(self._device_timelines.keys())

  @contextlib.contextmanager
  def span_group(self, name: str):
    begin = time.perf_counter()
    for timeline in self._get_timelines().values():
      timeline.span_group_begin(name, begin)
    try:
      yield
    finally:
      end = time.perf_counter()
      for timeline in self._get_timelines().values():
        timeline.span_group_end(end)

  @contextlib.contextmanager
  def span(
      self, name: str, devices: list[str | JaxDevice] | np.ndarray | None = None
  ):
    begin = time.perf_counter()
    thread_timeline = self._get_or_create_thread_timeline(
        create_thread_timeline_id()
    )
    thread_timeline.span_begin(name, begin)
    device_waitlist = _DeviceWaitlist()
    try:
      yield device_waitlist
    finally:
      end = time.perf_counter()
      thread_timeline.span_end(end)
      self._get_or_create_device_timelines(devices).span(name, begin, device_waitlist._data)  # pylint: disable=protected-access


Tracer = PerfTracer | NoopTracer


class Timeline:
  """Provides an API to collect events to construct a tree of spans and groups."""

  id: str
  born: float
  root: SpanGroup
  stack: list[SpanGroup]

  def __init__(
      self, id: str, born: float, stack: list[SpanGroup] | None = None
  ):
    self.id = id
    self.born = born
    if stack is None:
      self.root = SpanGroup("root", None)
      self.root.begin = born
      self.stack = [self.root]
    else:
      self.root = stack[0]
      self.stack = stack

    self._last_span: Span | None = None
    # TODO(yangmu): add lock to protect stack and _last_span.

  def _stack_debug(self) -> str:
    out = f"{self.root.name}"
    for group in self.stack[1:]:
      out += f" -> {group.name}"
    return out

  def span_group_begin(self, name: str, begin: float) -> SpanGroup:
    if self._last_span and not self._last_span.ended:
      logging.warning(
          f"{self.id}: last span '{self._last_span.name}' is not ended. current"
          f" group stack: {self._stack_debug()}"
      )
    inner = SpanGroup(name, self.stack[-1])
    inner.begin = begin
    self.stack.append(inner)
    # print(f"{self.id}: begin {name}")
    return inner

  def span_group_end(self, end: float) -> None:
    if len(self.stack) == 1:
      raise ValueError(f"{self.id}: no more span groups to end.")
    if self._last_span and not self._last_span.ended:
      logging.warning(
          f"{self.id}: last span '{self._last_span.name}' is not ended. current"
          f" group stack: {self._stack_debug()}"
      )
    inner = self.stack.pop()
    inner.end = end
    # print(f"{self.id}: end {inner.name}")

  def device_span(self, name: str, thread_begin: float, end: float) -> None:
    if self._last_span and not self._last_span.ended:
      raise ValueError(
          f"{self.id}: last span '{self._last_span.name}' is not ended. current"
          f" group stack: {self._stack_debug()}"
      )

    if self._last_span and self._last_span.end > thread_begin:
      inner = Span(name, self._last_span.end)
    else:
      inner = Span(name, thread_begin)
    inner.end = end
    self.stack[-1].inner.append(inner)

    self._last_span = inner

  def thread_span_begin(self, name: str, begin: float) -> Span:
    if self._last_span and not self._last_span.ended:
      raise ValueError(
          f"{self.id}: last span '{self._last_span.name}' is not ended. current"
          f" group stack: {self._stack_debug()}"
      )

    inner = Span(name, begin)
    self.stack[-1].inner.append(inner)

    self._last_span = inner
    return inner

  def thread_span_end(self, end: float) -> None:
    if self._last_span is None:
      raise ValueError(f"{self.id}: no span to end.")
    if self._last_span.ended:
      raise ValueError(
          f"{self.id}: span '{self._last_span.name}' is already ended."
      )
    self._last_span.end = end


class ThreadTimeline(Timeline):

  def span_begin(self, name: str, begin: float) -> None:
    self.thread_span_begin(name, begin)

  def span_end(self, end: float) -> None:
    self.thread_span_end(end)


class DeviceTimeline(Timeline):
  """Manages a custom-annotated timeline for a device (e.g. TPU)."""

  def __init__(self, id: str, born: float):
    super().__init__(id, born)

    # wait pending data.
    self._threads: list[threading.Thread] = []

  def span(
      self, name: str, thread_span_begin: float, waitlist: jaxtyping.PyTree
  ) -> None:
    """Record a new span for device (e.g. TPU).

    The span begin time is inferred from the thread span begin time (i.e. thread
    launches a computation on the device) and the end time of prevous span
    on the same device, the late one is used.

    The span end time is determined when all JAX computations associated
    with 'waitlist' finish.

    Args:
      name: The name of the span.
      thread_span_begin: The begin time of the span on the thread, used to infer
        the begin time of the span on the device.
      waitlist: The JAX computation to be tracked, used to infer the end time of
        the span on the device.
    """

    def on_success():
      self.device_span(name, thread_span_begin, time.perf_counter())

    def on_failure(e: Exception):
      raise e

    if not waitlist:
      on_success()
    else:
      t = _async_wait(waitlist=waitlist, success=on_success, failure=on_failure)
      self._threads.append(t)

  def wait_pending_spans(self) -> None:
    for t in self._threads:
      t.join()


class BatchDeviceTimeline:

  def __init__(self, timelines: list[DeviceTimeline]):
    self._timelines = timelines

  def span(
      self, name: str, thread_span_begin: float, waitlist: jaxtyping.PyTree
  ):
    for timeline in self._timelines:
      timeline.span(name, thread_span_begin, waitlist)


########################################################################
#
# internal only - utility classes and functions
#
########################################################################


class _DeviceWaitlist:
  """Provides an interface to collect waitlist for PerfTracer span()."""

  def __init__(self):
    self._data = []

  def device_end(self, waitlist: jaxtyping.PyTree) -> None:
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
