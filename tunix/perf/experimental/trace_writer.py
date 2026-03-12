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

"""Trace writer implementations and helper functions."""

from __future__ import annotations

import abc
from collections.abc import Mapping
import os
import time

from absl import logging
from perfetto.trace_builder.proto_builder import TraceProtoBuilder
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import timeline

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackDescriptor
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent


Timeline = timeline.Timeline


class TraceWriter(abc.ABC):
  """An abstract base class for writing traces."""

  @abc.abstractmethod
  def write_timelines(self, timelines: Mapping[str, Timeline]) -> None:
    """Writes timelines to the trace."""


class NoopTraceWriter(TraceWriter):
  """A no-op trace writer that does nothing."""

  def write_timelines(self, timelines: Mapping[str, Timeline]) -> None:
    del self, timelines  # Unused.


class PerfettoTraceWriter(TraceWriter):
  """A writer for Perfetto trace events."""

  def __init__(self, trace_dir: str):
    """Initializes the PerfettoTraceWriter.

    Args:
      trace_dir: The directory to export trace files to. This path must be an
        absolute Linux path with write permissions.
    """
    self._trace_dir = trace_dir
    self._trace_file_path = None
    try:
      os.makedirs(self._trace_dir, exist_ok=True)
      trace_file_name = f"perfetto_trace_v2_{int(time.time())}.pb"
      self._trace_file_path = os.path.join(self._trace_dir, trace_file_name)
      logging.info(
          "Initializing perfetto trace writer at: %s", self._trace_file_path
      )
    except Exception:  # pylint: disable=broad-except
      # Catching broad exceptions to ensure that failures in trace
      # initialization (e.g., due to file system errors, permissions, etc.) do
      # not crash the application. Tracing is best-effort.
      logging.exception(
          "Failed to initialize perfetto trace writer in directory %r. Skipping"
          " trace dumping for this run.",
          self._trace_dir,
      )
      self._trace_file_path = None

  def write(self, builder: TraceProtoBuilder) -> None:
    """Writes the built trace to the file."""
    if self._trace_file_path is None:
      return

    try:
      # TODO(b/480134569): see if file writing is a bottleneck and explore
      # faster alternatives (e.g., keeping in memory and writing at the end).
      # TODO(noghabi): once we empty timeline, we can just append the recent
      # spans to the file without serializing the entire trace.
      with open(self._trace_file_path, "wb") as f:
        f.write(builder.serialize())
    except Exception:  # pylint: disable=broad-except
      # Catching broad exceptions to ensure that failures in trace
      # serialization or writing do not crash the application. Tracing is
      # best-effort.
      logging.exception(
          "Failed to write to trace file: %s", self._trace_file_path
      )

  def write_timelines(self, timelines: Mapping[str, Timeline]) -> None:
    """Writes timelines to the trace file."""
    if not timelines:
      return

    builder = TraceProtoBuilder()

    # Sort timelines by ID to ensure consistent track ordering.
    sorted_ids = sorted(timelines)

    for i, t_id in enumerate(sorted_ids):
      # Assign a unique UUID for the track.
      # We start from 1 because 0 is sometimes reserved or special.
      uuid = i + 1
      timeline = timelines[t_id]

      # Skip empty timelines.
      if not timeline.spans:
        continue

      # Track Descriptor
      packet = builder.add_packet()
      packet.track_descriptor.uuid = uuid
      packet.track_descriptor.name = t_id

      # Sort spans by ID to ensure deterministic output order
      sorted_spans = sorted(timeline.spans.values(), key=lambda s: s.id)

      for s in sorted_spans:
        # Timestamp in nanoseconds, relative to timeline creation (born).
        start_ns = int((s.begin - timeline.born) * 1e9)

        # Slice Begin
        packet_begin = builder.add_packet()
        packet_begin.timestamp = start_ns
        packet_begin.trusted_packet_sequence_id = 1
        packet_begin.track_event.type = TrackEvent.Type.TYPE_SLICE_BEGIN
        packet_begin.track_event.track_uuid = uuid
        packet_begin.track_event.name = (
            f"{s.name} (step={(s.tags or {}).get(perf_constants.GLOBAL_STEP)})"
        )

        if s.ended:
          end_ns = int((s.end - timeline.born) * 1e9)
          # Slice End
          packet_end = builder.add_packet()
          packet_end.timestamp = end_ns
          packet_end.trusted_packet_sequence_id = 1
          packet_end.track_event.type = TrackEvent.Type.TYPE_SLICE_END
          packet_end.track_event.track_uuid = uuid

    self.write(builder)
