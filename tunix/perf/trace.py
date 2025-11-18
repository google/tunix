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

"""Performance tracer."""

from __future__ import annotations

from concurrent import futures
import contextlib
import threading
import time
from typing import Any, Callable

import jax
import jaxtyping
import numpy as np
from tunix.perf import metrics


JaxDevice = Any
MetricsT = metrics.MetricsT
PerfMetricsApi = metrics.PerfMetricsApi
PerfMetricsContext = metrics.PerfMetricsContext
PerfMetricsQuery = metrics.PerfMetricsQuery


def create_timeline_id(id: str | JaxDevice) -> str:
  if isinstance(id, str):
    return id
  elif hasattr(id, "platform") and hasattr(id, "id"):
    # if it's a JAX device object, convert to string
    return getattr(id, "platform") + str(getattr(id, "id"))
  else:
    raise ValueError(f"Unsupport id type: {type(id)}")


class NoopTracer:
  """An no-op tracer that does nothing."""

  def synchronize(self) -> None:
    pass

  def dump(self) -> None:
    pass

  def export(self) -> PerfMetricsApi:
    return PerfMetricsApi({}, "")

  @property
  def all(self) -> list[str]:
    return []

  @contextlib.contextmanager
  def interval(
      self,
      label: str,
      devices: list[str | JaxDevice] | np.ndarray | None = None,
  ):
    try:
      yield _DeviceWaitlist()
    finally:
      pass

  def end_epoch(self, context: PerfMetricsContext | None = None) -> MetricsT:
    return {}


class PerfTracer(NoopTracer):
  """Provides an API to collect events to construct thread and devices timelines.

  Usage:
    1. Collect intervals for thread and devices.

      with tracer.interval("label")
        ...

      with tracer.interval("label", devices=tracer.all) as interval:
        ...
        interval.device_end(data)

    2. Establish epoch boundaries.

      tracer.end_epoch()

    3. Access the collected events.

      tracer.dump()
      api = tracer.export()
  """

  def __init__(
      self,
      devices: list[str | JaxDevice] | np.ndarray | None = None,
      export_fn: (
          Callable[[PerfMetricsQuery, PerfMetricsContext], MetricsT] | None
      ) = None,
  ):
    self._epoch = 0
    self._export_fn = export_fn

    # align all timelines with the same born time.
    self._born = time.perf_counter()

    self._main_thread_id = str(threading.get_ident())

    self._thread_timelines: dict[str, ThreadTimeline] = {}
    self._get_or_create_thread_timeline(self._main_thread_id)
    self._device_timelines: dict[str, DeviceTimeline] = {}
    if devices:
      for device in devices:
        self._get_or_create_device_timeline(device)

  def _get_timelines(self) -> dict[str, BaseTimeline]:
    timelines: dict[str, BaseTimeline] = {}
    for timeline in self._thread_timelines.values():
      timelines[timeline.id] = timeline
    for timeline in self._device_timelines.values():
      timelines[timeline.id] = timeline
    return timelines

  def _get_or_create_thread_timeline(self, id: str) -> ThreadTimeline:
    if id not in self._thread_timelines:
      self._thread_timelines[id] = ThreadTimeline(id, self._born)
    return self._thread_timelines[id]

  def _get_or_create_device_timeline(
      self, id: str | JaxDevice
  ) -> DeviceTimeline:
    tid = create_timeline_id(id)

    if tid not in self._device_timelines:
      self._device_timelines[tid] = DeviceTimeline(tid, self._born)
    return self._device_timelines[tid]

  def _get_or_create_device_timelines(
      self, ids: list[str | JaxDevice] | np.ndarray
  ) -> BatchDeviceTimeline:
    if isinstance(ids, np.ndarray):
      ids = ids.flatten().tolist()
    return BatchDeviceTimeline(
        [self._get_or_create_device_timeline(id) for id in ids]
    )

  def synchronize(self) -> None:
    _synchronize_devices()
    for timeline in self._device_timelines.values():
      timeline.wait_pending_intervals()

  def dump(self) -> None:
    self.synchronize()
    for timeline in self._get_timelines().values():
      print(timeline)

  def export(self) -> PerfMetricsApi:
    self.synchronize()
    return PerfMetricsApi(self._get_timelines(), self._main_thread_id)

  @property
  def all(self) -> list[str]:
    """Returns all device ids.

    To be used to set `devices` in interval().
    """
    return list(self._device_timelines.keys())

  @contextlib.contextmanager
  def interval(
      self,
      label: str,
      devices: list[str | JaxDevice] | np.ndarray | None = None,
  ):
    """Collect an interval for thread and devices.

    Thread interval will always be collected, device intervals will be collected
    for devices listed in `devices`.

    Usage:
      1. Collect an interval for thread only. (thread defaults to "main")

        with tracer.interval("label"):
          ...

        with tracer.interval("label", thread="my_thread"):
          ...

      2. Collect an interval for thread and devices.

        with tracer.interval("label", devices=tracer.all) as interval:
          ...
          interval.device_end(data)

    Args:
      label: The label of the interval.
      devices: The devices to collect the interval for.
    """
    thread = str(threading.get_ident())
    thread_begin = self._get_or_create_thread_timeline(thread).begin(label)
    device_waitlist = _DeviceWaitlist()
    try:
      yield device_waitlist
    finally:
      self._thread_timelines[thread].end()
      if devices is not None:
        self._get_or_create_device_timelines(devices).interval(label, thread_begin, device_waitlist._data)  # pylint: disable=protected-access

  def end_epoch(self, context: PerfMetricsContext | None = None) -> MetricsT:
    # TODO(yangmu): revisit this, may not be needed.
    now = time.perf_counter()
    for thread_timeline in self._thread_timelines.values():
      with thread_timeline.interval("end_epoch"):
        pass
    for device_timeline in self._device_timelines.values():
      device_timeline.interval("end_epoch", now, [])

    if context is None:
      context = PerfMetricsContext()
    context.epoch = self._epoch
    self._epoch += 1

    for timeline in self._get_timelines().values():
      timeline.end_epoch()

    if self._export_fn is not None:
      query = PerfMetricsQuery(
          self._get_timelines(), self._main_thread_id
      ).epoch([context.epoch])
      return self._export_fn(query, context)
    else:
      return {}


class _DeviceWaitlist:
  """Provides an interface to collect waitlist for PerfTracer interval()."""

  def __init__(self):
    self._data = []

  def device_end(self, waitlist: jaxtyping.PyTree) -> None:
    self._data.append(waitlist)


class BaseTimeline:
  """Base class for custom-annotated timelines."""

  id: str
  born: float
  intervals: list[tuple[float, float]]
  labels: list[str]
  epochs: list[int]

  def __init__(self, id: str, born: float):
    self.id = id
    self.born = born
    self.intervals = []
    self.labels = []
    self.epochs = []

  def __eq__(self, other: object) -> bool:
    return (
        isinstance(other, BaseTimeline)
        and self.id == other.id
        and self.born == other.born
        and self.intervals == other.intervals
        and self.labels == other.labels
        and self.epochs == other.epochs
    )

  def __repr__(self) -> str:
    out = f"[{self.id}] "
    epoch_begin = 0
    for epoch_end in self.epochs + [len(self.intervals)]:
      out += f"epoch:[{epoch_begin},{epoch_end}) "
      for i in range(epoch_begin, epoch_end):
        out += (
            f"{self.labels[i]}:({self.intervals[i][0] - self.born:.6f},{self.intervals[i][1] - self.born:.6f}) "
        )
      epoch_begin = epoch_end
    return out

  def __str__(self) -> str:
    return repr(self)

  def end_epoch(self) -> None:
    if len(self.epochs) == 0 or self.epochs[-1] != len(self.intervals):
      self.epochs.append(len(self.intervals))


class ThreadTimeline(BaseTimeline):
  """Manages a custom-annotated timeline for a thread."""

  def __init__(self, id: str, born: float):
    super().__init__(id, born)

    # detect nested call.
    self._label: str | None = None
    self._begin: float | None = None
    self._lock = threading.Lock()

  def begin(self, label: str) -> float:
    begin = time.perf_counter()
    with self._lock:
      if self._label is not None:
        raise ValueError(
            f"{self.id}: interval '{label}' is nested in '{self._label}'."
        )
      self._label = label
      self._begin = begin
    return begin

  def end(self) -> None:
    end = time.perf_counter()
    with self._lock:
      if self._label is None:
        raise ValueError(f"{self.id}: interval is not started.")
      self.intervals.append((self._begin, end))  # type: ignore
      self.labels.append(self._label)
      self._label = None
      self._begin = None
    return

  @contextlib.contextmanager
  def interval(self, label: str):
    begin = self.begin(label)
    try:
      yield begin
    finally:
      self.end()


class DeviceTimeline(BaseTimeline):
  """Manages a custom-annotated timeline for a device (e.g. TPU)."""

  def __init__(self, id: str, born: float):
    super().__init__(id, born)

    # wait pending data.
    self._threads: list[threading.Thread] = []

  def interval(
      self, label: str, thread_begin: float, waitlist: jaxtyping.PyTree
  ) -> None:
    """Record a new interval for device (e.g. TPU).

    The interval begin time is inferred from the thread begin time (i.e. thread
    launches a computation on the device) and the end time of prevous interval
    on the same device, the late one is used.

    The interval end time is determined when all JAX computations associated
    with 'data' finish.

    Args:
      label: The label of the interval.
      thread_begin: The begin time of the interval on the thread, used to infer
        the begin time of the interval on the device.
      waitlist: The JAX computation to be tracked, used to infer the end time of
        the interval on the device.
    """

    def on_success():
      begin = thread_begin
      if len(self.intervals) > 0:
        begin = max(begin, self.intervals[-1][1])
      self.intervals.append((begin, time.perf_counter()))  # type: ignore
      self.labels.append(label)

    def on_failure(e: Exception):
      raise e

    if not waitlist:
      on_success()
    else:
      t = _async_wait(waitlist=waitlist, success=on_success, failure=on_failure)
      self._threads.append(t)

  def wait_pending_intervals(self) -> None:
    for t in self._threads:
      t.join()


class BatchDeviceTimeline:
  """Provides the same API as DeviceTimeline to operate on a batch of devices."""

  def __init__(self, timelines: list[DeviceTimeline]):
    self._timelines = timelines

  def __str__(self) -> str:
    out = ""
    for timeline in self._timelines:
      out += str(timeline) + "\n"
    return out

  def interval(
      self, label: str, thread_begin: float, data: jaxtyping.PyTree
  ) -> None:
    for timeline in self._timelines:
      timeline.interval(label, thread_begin, data)

  def end_epoch(self) -> None:
    for timeline in self._timelines:
      timeline.end_epoch()


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
