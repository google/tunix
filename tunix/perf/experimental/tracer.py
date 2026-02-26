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

import contextlib
import threading
import time
from typing import Any, Callable

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


def _synchronize_devices():
  for device in jax.devices():
    jax.device_put(jax.numpy.array(0.0), device=device).block_until_ready()


class _AsyncWaitlist:
  """Provides an interface to collect waitlist for PerfTracer span()."""

  def __init__(self):
    self._data = []

  def async_end(self, waitlist: jaxtyping.PyTree) -> None:
    self._data.append(waitlist)


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
