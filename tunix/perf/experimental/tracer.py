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

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import threading
import time
from typing import Any

import jax
import jaxtyping
import numpy as np
from tunix.perf import metrics
from tunix.perf.experimental import timeline


JaxDevice = Any
MetricsT = metrics.MetricsT
PerfSpanQuery = metrics.PerfSpanQuery
Span = timeline.Span
Timeline = timeline.Timeline
AsyncTimeline = timeline.AsyncTimeline
BatchAsyncTimelines = timeline.BatchAsyncTimelines


def generate_host_timeline_id() -> str:
  """Generates a string ID for a host timeline."""
  return f"host-{threading.get_ident()}"


def generate_device_timeline_id(device_id: str | JaxDevice) -> str:
  """Generates a string ID for a device timeline.

  Args:
    device_id: A string ID or a JAX device object.

  Returns:
    A string representation of the device ID. For a JAX device object, it will
    be the platform name followed by the device ID, e.g., "tpu0".

  Raises:
    ValueError: If the input device_id type is not supported. Only string and
    JAX
    device objects (with platform and id attributes) are supported.
  """

  if isinstance(device_id, str):
    return device_id
  elif hasattr(device_id, "platform") and hasattr(device_id, "id"):
    # if it's a JAX device object, convert to string
    return f"{device_id.platform}{device_id.id}"
  else:
    raise ValueError(f"Unsupported id type: {type(device_id)}")


def generate_device_timeline_ids(
    devices: Sequence[str | JaxDevice] | np.ndarray | None,
) -> Sequence[str]:
  """Generates a list of string IDs for a list of devices."""
  if devices is None:
    return []
  if isinstance(devices, np.ndarray):
    device_list = devices.flatten().tolist()
  else:
    device_list = devices
  return [generate_device_timeline_id(device) for device in device_list]


def _synchronize_devices() -> None:
  for device in jax.devices():
    jax.device_put(0.0, device=device).block_until_ready()


class AsyncWaitlist:
  """An interface to collect waitlist for PerfTracer span()."""

  def __init__(self) -> None:
    self._waitlist = []

  def async_end(self, waitlist: jaxtyping.PyTree) -> None:
    self._waitlist.append(waitlist)

  @property
  def waitlist(self) -> jaxtyping.PyTree:
    return self._waitlist


class NoopTracer:
  """A no-op tracer that does nothing."""

  def synchronize(self) -> None:
    pass

  def print(self) -> None:
    pass

  def export(self) -> MetricsT:
    return {}

  @property
  def all_devices(self) -> Sequence[str | JaxDevice]:
    return []

  @contextlib.contextmanager
  def span(
      self,
      name: str,
      devices: Sequence[str | JaxDevice] | np.ndarray | None = None,
      tags: Mapping[str, Any] | None = None,
  ) -> Iterator[AsyncWaitlist]:
    yield AsyncWaitlist()


class PerfTracer(NoopTracer):
  """An API to collect performance metrics in the form of timelines.

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

  def __init__(
      self,
      devices: Sequence[str | JaxDevice] | np.ndarray | None = None,
      export_fn: Callable[[Any], MetricsT] | None = None,
      *,
      collect_on_first_device_per_mesh: bool = True,
  ) -> None:
    """Initializes the instance.

    Args:
      devices: A list of devices to collect trace data for.
      export_fn: A function that takes in a dictionary of timelines and returns
        a MetricsT object.
      collect_on_first_device_per_mesh: If True, collect trace data on the first
        device per mesh when calling span().
    """

    self._export_fn = export_fn

    # align all timelines with the same born time.
    self._born = time.perf_counter()

    self._main_thread_id = generate_host_timeline_id()

    self._host_timelines: dict[str, Timeline] = {
        self._main_thread_id: Timeline(self._main_thread_id, self._born)
    }
    self._device_timelines: dict[str, AsyncTimeline] = {}
    if devices is not None:
      for device_id in generate_device_timeline_ids(devices):
        self._get_or_create_device_timeline(device_id)
    self._collect_on_first_device_per_mesh = collect_on_first_device_per_mesh

  def _get_timelines(self) -> Mapping[str, Timeline]:
    timelines: dict[str, Timeline] = {}
    for tl in self._host_timelines.values():
      timelines[tl.id] = tl
    for tl in self._device_timelines.values():
      if tl.id in timelines:
        raise ValueError(f"Timeline ID collision detected: {tl.id}")
      timelines[tl.id] = tl
    return timelines

  def _get_or_create_host_timeline(self, timeline_id: str) -> Timeline:
    host_timeline = self._host_timelines.get(timeline_id)
    if host_timeline is None:
      host_timeline = Timeline(timeline_id, self._born)
      self._host_timelines[timeline_id] = host_timeline
    return host_timeline

  def _get_or_create_device_timeline(self, timeline_id: str) -> AsyncTimeline:
    device_timeline = self._device_timelines.get(timeline_id)
    if device_timeline is None:
      device_timeline = AsyncTimeline(timeline_id, self._born)
      self._device_timelines[timeline_id] = device_timeline
    return device_timeline

  def _get_or_create_device_timelines(
      self, ids: Sequence[str | JaxDevice] | np.ndarray | None
  ) -> BatchAsyncTimelines:
    return BatchAsyncTimelines([
        self._get_or_create_device_timeline(id)
        for id in generate_device_timeline_ids(ids)
    ])

  def synchronize(self) -> None:
    _synchronize_devices()
    for tl in self._device_timelines.values():
      tl.wait_pending_spans()

  def print(self) -> None:
    self.synchronize()
    for tl in self._get_timelines().values():
      print(f"\n[{tl.id}]")
      print(tl)

  def export(self) -> MetricsT:
    if self._export_fn is None:
      return {}
    return self._export_fn(self._get_timelines())

  @property
  def all_devices(self) -> Sequence[str]:
    return list(self._device_timelines.keys())

  @contextlib.contextmanager
  def span(
      self,
      name: str,
      devices: Sequence[str | JaxDevice] | np.ndarray | None = None,
      tags: Mapping[str, Any] | None = None,
  ) -> Iterator[AsyncWaitlist]:
    span_devices = devices
    if self._collect_on_first_device_per_mesh and devices is not None:
      if isinstance(devices, np.ndarray):
        if devices.size > 0:
          span_devices = [devices.flat[0]]
      elif devices:
        span_devices = [devices[0]]

    begin = time.perf_counter()
    host_timeline = None
    device_waitlist = AsyncWaitlist()
    try:
      host_timeline = self._get_or_create_host_timeline(
          generate_host_timeline_id()
      )
      host_timeline.start_span(name, begin, tags=tags)
      yield device_waitlist
    finally:
      if host_timeline is not None:
        end = time.perf_counter()
        host_timeline.stop_span(end)
        self._get_or_create_device_timelines(span_devices).span(
            name, begin, device_waitlist.waitlist, tags=tags
        )  # pylint: disable=protected-access


Tracer = PerfTracer | NoopTracer
