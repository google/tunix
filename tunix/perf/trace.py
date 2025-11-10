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
PerfMetricsApi = metrics.PerfMetricsApi


class NoopTracer:
  """An no-op tracer that does nothing."""

  def synchronize(self) -> None:
    pass

  def dump(self) -> None:
    pass

  def export(self) -> PerfMetricsApi:
    return PerfMetricsApi({})

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

  def end_epoch(self) -> None:
    pass


class PerfTracer(NoopTracer):
  """Provides an API to collect events to construct host and devices timelines.

  Usage:
    1. Collect intervals for host and devices.

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

  def __init__(self, devices: list[str | JaxDevice] | np.ndarray | None = None):
    # align all timelines with the same born time.
    self._born = time.perf_counter()

    self._host_timeline: HostTimeline = HostTimeline("host", self._born)
    self._device_timelines: dict[str, DeviceTimeline] = {}
    if devices:
      for device in devices:
        self._get_or_create_device_timeline(device)

  def _get_timelines(self) -> dict[str, BaseTimeline]:
    timelines: dict[str, BaseTimeline] = {
        self._host_timeline.id: self._host_timeline
    }
    for timeline in self._device_timelines.values():
      timelines[timeline.id] = timeline
    return timelines

  def _get_or_create_device_timeline(
      self, id: str | JaxDevice
  ) -> DeviceTimeline:
    # if it's a JAX device object, convert to id
    if hasattr(id, "platform") and hasattr(id, "id"):
      id = getattr(id, "platform") + str(getattr(id, "id"))

    if not isinstance(id, str):
      raise ValueError(f"Unsupport id type: {type(id)}")

    if id not in self._device_timelines:
      self._device_timelines[id] = DeviceTimeline(id, self._born)
    return self._device_timelines[id]

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
    return PerfMetricsApi(self._get_timelines())

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
    """Collect an interval for host and devices.

    Host interval will always be collected, device intervals will be collected
    for devices listed in `devices`.

    Usage:
      1. Collect an interval for host only.

        with tracer.interval("label"):
          ...

      2. Collect an interval for host and devices.

        with tracer.interval("label", devices=tracer.all) as interval:
          ...
          interval.device_end(data)

    Args:
      label: The label of the interval.
      devices: The devices to collect the interval for.
    """

    host_begin = self._host_timeline.begin(label)
    device_waitlist = _DeviceWaitlist()
    try:
      yield device_waitlist
    finally:
      self._host_timeline.end()
      if devices is not None:
        self._get_or_create_device_timelines(devices).interval(label, host_begin, device_waitlist._data)  # pylint: disable=protected-access

  def end_epoch(self) -> None:
    for timeline in self._get_timelines().values():
      timeline.end_epoch()


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


class HostTimeline(BaseTimeline):
  """Manages a custom-annotated timeline for the host."""

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
      self, label: str, host_begin: float, waitlist: jaxtyping.PyTree
  ) -> None:
    """Record a new interval for device (e.g. TPU).

    The interval begin time is inferred from the host begin time (i.e. host
    launches a computation on the device) and the end time of prevous interval
    on the same device, the late one is used.

    The interval end time is determined when all JAX computations associated
    with 'data' finish.

    Args:
      label: The label of the interval.
      host_begin: The begin time of the interval on the host, used to infer the
        begin time of the interval on the device.
      waitlist: The JAX computation to be tracked, used to infer the end time of
        the interval on the device.
    """

    def on_success():
      begin = host_begin
      if len(self.intervals) > 0:
        begin = max(begin, self.intervals[-1][1])
      self.intervals.append((begin, time.perf_counter()))  # type: ignore
      self.labels.append(label)

    def on_failure(e: Exception):
      raise e

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
      self, label: str, host_begin: float, data: jaxtyping.PyTree
  ) -> None:
    for timeline in self._timelines:
      timeline.interval(label, host_begin, data)

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
