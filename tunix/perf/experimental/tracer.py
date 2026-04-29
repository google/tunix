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
from tunix.perf.experimental import timeline_utils


JaxDevice = timeline_utils.JaxDevice
MetricsT = metrics.MetricsT
PerfSpanQuery = metrics.PerfSpanQuery
Span = timeline.Span
Timeline = timeline.Timeline
AsyncTimeline = timeline.AsyncTimeline
BatchAsyncTimelines = timeline.BatchAsyncTimelines


def _synchronize_devices() -> None:
  for device in jax.devices():
    jax.device_put(0.0, device=device).block_until_ready()


class AsyncWaitlist:
  """An interface to collect waitlist for PerfTracer span()."""

  def __init__(self) -> None:
    self._waitlist = []

  def async_end(self, waitlist: jaxtyping.PyTree) -> None:
    """Adds a JAX PyTree computation to the asynchronous waitlist.

    Args:
      waitlist: A JAX PyTree (e.g., arrays or nested lists of arrays) to wait on
        before marking the span as ended.
    """
    self._waitlist.append(waitlist)

  @property
  def waitlist(self) -> jaxtyping.PyTree:
    """Returns the current accumulated waitlist."""
    return self._waitlist


class NoopTracer:
  """A no-op tracer that does nothing."""

  def synchronize(self) -> None:
    """Synchronizes all devices to ensure pending computations are finished.

    Use this carefully as it is a blocking operation.
    """
    pass

  def print(self) -> None:
    """Prints the captured timelines to standard output."""
    pass

  def export(self) -> MetricsT:
    """Exports the gathered timeline metrics.

    Returns:
      An empty dictionary as no metrics are collected by NoopTracer.
    """
    return {}

  @property
  def all_devices(self) -> Sequence[str | JaxDevice]:
    """Returns a list of all devices tracked by the tracer."""
    return []

  @contextlib.contextmanager
  def span(
      self,
      name: str,
      devices: Sequence[str | JaxDevice] | np.ndarray | None = None,
      tags: Mapping[str, Any] | None = None,
  ) -> Iterator[AsyncWaitlist]:
    """A context manager to trace a span of execution.

    Args:
      name: The name of the span.
      devices: A sequence of devices or a numpy array of devices on which the
        span is executed.
      tags: A mapping of tags associated with the span.

    Yields:
      An AsyncWaitlist for associating asynchronous computations with this span.
    """
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
      concurrent_device_spans: Sequence[str] | str | None = None,
  ) -> None:
    """Initializes the instance.

    Args:
      devices: A list of devices to collect trace data for.
      export_fn: A function that takes in a dictionary of timelines and returns
        a MetricsT object.
      collect_on_first_device_per_mesh: If True, collect trace data on the first
        device per mesh when calling span().
      concurrent_device_spans: A list of span names that are allowed to run
        concurrently (overlap) on a device timeline. These spans will not be
        sequentialized. If a single span from the ignore list is present, the
        entire timeline will not be sequentialized. If "all" is passed, no
        sequentialization will be performed on any device timeline. If None, all
        device timelines will be sequentialized.
    """

    self._export_fn = export_fn

    # align all timelines with the same born time.
    self._born = time.perf_counter()

    self._main_thread_id = timeline_utils.generate_host_timeline_id()
    self._timelines_lock = threading.Lock()

    self._host_timelines: dict[str, Timeline] = {
        self._main_thread_id: Timeline(self._main_thread_id, self._born)
    }
    self._device_timelines: dict[str, AsyncTimeline] = {}
    self._device_queue_timelines: dict[str, Timeline] = {}
    if devices is not None:
      for device_id in timeline_utils.generate_device_timeline_ids(devices):
        self._get_or_create_device_timeline(device_id)
    self._collect_on_first_device_per_mesh = collect_on_first_device_per_mesh
    if (
        isinstance(concurrent_device_spans, str)
        and concurrent_device_spans != "all"
    ):
      self._concurrent_spans = [concurrent_device_spans]
    else:
      self._concurrent_spans = concurrent_device_spans

  def _get_timelines(self) -> Mapping[str, Timeline]:
    # TODO(noghabi): do we need this function anymore?
    timelines_by_id: dict[str, Timeline] = {}
    with self._timelines_lock:
      for tl in self._host_timelines.values():
        timelines_by_id[tl.id] = tl
      for tl in self._device_timelines.values():
        if tl.id in timelines_by_id:
          raise ValueError(f"Timeline ID collision detected: {tl.id!r}")
        timelines_by_id[tl.id] = tl
      for tl in self._device_queue_timelines.values():
        if tl.id in timelines_by_id:
          raise ValueError(f"Timeline ID collision detected: {tl.id!r}")
        timelines_by_id[tl.id] = tl

    return timelines_by_id

  def _get_or_create_host_timeline(self, timeline_id: str) -> Timeline:
    with self._timelines_lock:
      host_timeline = self._host_timelines.get(timeline_id)
      if host_timeline is None:
        host_timeline = Timeline(timeline_id, self._born)
        self._host_timelines[timeline_id] = host_timeline
      return host_timeline

  def _get_or_create_device_timeline(self, timeline_id: str) -> AsyncTimeline:
    with self._timelines_lock:
      device_timeline = self._device_timelines.get(timeline_id)
      if device_timeline is None:
        device_timeline = AsyncTimeline(timeline_id, self._born)
        self._device_timelines[timeline_id] = device_timeline
        queue_tl_id = timeline_utils.generate_queued_timeline_id(timeline_id)
        self._device_queue_timelines[queue_tl_id] = Timeline(
            queue_tl_id, self._born
        )

      return device_timeline

  def _get_or_create_device_timelines(
      self, ids: Sequence[str | JaxDevice] | np.ndarray | None
  ) -> BatchAsyncTimelines:
    return BatchAsyncTimelines([
        self._get_or_create_device_timeline(id)
        for id in timeline_utils.generate_device_timeline_ids(ids)
    ])

  def synchronize(self) -> None:
    """Synchronizes all devices and waits for pending asynchronous spans to finish.

    Use this carefully as it is a blocking operation.
    """
    _synchronize_devices()
    with self._timelines_lock:
      device_timelines = list(self._device_timelines.values())
    for tl in device_timelines:
      tl.wait_pending_spans()

  def print(self) -> None:
    """Synchronizes and prints the captured timelines to standard output."""
    self.synchronize()
    for tl in self._get_timelines().values():
      print(f"\n[{tl.id}]")
      print(tl)

  def _sequentialize_device_timeline(
      self, timeline_id: str, tl: AsyncTimeline
  ) -> None:
    """Sequentializes overlapping spans on a device timeline."""
    with tl._lock:
      ignore_list = self._concurrent_spans or []
      if any(span.name in ignore_list for span in tl._cur_step.values()):
        return

      active_spans, queue_spans = (
          timeline_utils.sequentialize_overlapping_spans(tl._cur_step)
      )

      tl._cur_step = dict(active_spans)

      if queue_spans:
        queue_tl_id = timeline_utils.generate_queued_timeline_id(timeline_id)
        queue_tl = self._device_queue_timelines[queue_tl_id]

        with queue_tl._lock:
          for span_id, span in queue_spans.items():
            queue_tl._cur_step[span_id] = span

  def process_and_commit_timelines(self) -> None:
    """Explicitly commits current steps across all active timelines."""
    with self._timelines_lock:
      # 1. Post process timelines
      # Sequentialize overlapping device timelines, except for those in ignore list.
      if self._concurrent_spans != "all":
        for timeline_id, tl in self._device_timelines.items():
          self._sequentialize_device_timeline(timeline_id, tl)

      # 2. Commit all timelines.
      for tl in self._host_timelines.values():
        tl.commit_step()
      for tl in self._device_timelines.values():
        tl.commit_step()
      for tl in self._device_queue_timelines.values():
        tl.commit_step()

  def export(self) -> MetricsT:
    """Commits timelines and exports the gathered metrics using export_fn.

    Returns:
      The result of the export function applied to the timeline snapshots, or an
      empty dictionary if no export_fn was provided.
    """
    if self._export_fn is None:
      return {}

    self.process_and_commit_timelines()
    return self._export_fn(self._get_timelines())

  @property
  def all_devices(self) -> Sequence[str]:
    """Returns a list of all device IDs currently tracked by the tracer."""
    with self._timelines_lock:
      return list(self._device_timelines.keys())

  @contextlib.contextmanager
  def span(
      self,
      name: str,
      devices: Sequence[str | JaxDevice] | np.ndarray | None = None,
      tags: Mapping[str, Any] | None = None,
  ) -> Iterator[AsyncWaitlist]:
    """A context manager to trace a synchronous or asynchronous span of execution.

    Args:
      name: The name of the span.
      devices: An optional sequence or array of JAX devices to track this span
        on.
      tags: An optional dictionary of tags to associate with the span.

    Yields:
      An AsyncWaitlist for accumulating JAX computations. When the context
      exits,
      the tracker waits on the accumulated waitlist before completing the device
      span.
    """
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
          timeline_utils.generate_host_timeline_id()
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
