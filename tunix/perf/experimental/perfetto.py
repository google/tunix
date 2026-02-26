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

"""Perfetto trace writer and helper functions."""

from __future__ import annotations

import os
import time

from absl import logging
from perfetto.trace_builder.proto_builder import TraceProtoBuilder
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import timeline

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackDescriptor
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent


Timeline = timeline.Timeline


class PerfettoTraceWriter:
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
      # Create/Touch the file to ensure we have permissions and to log the path.
      with open(self._trace_file_path, "ab"):
        pass
      logging.info(
          "Initializing perfetto trace writer at: %s", self._trace_file_path
      )
    except Exception:  # pylint: disable=broad-except
      # Catching broad exceptions to ensure that failures in trace
      # initialization (e.g., due to file system errors, permissions, etc.) do
      # not crash the application. Tracing is best-effort.
      logging.exception(
          "Failed to initialize perfetto trace writer. Skipping trace dumping"
          " for this run.",
      )
      self._trace_file_path = None

  def write(self, builder: TraceProtoBuilder) -> None:
    """Writes the built trace to the file."""
    if self._trace_file_path is None:
      return

    try:
      # TODO(b/480134569): see if file writing is a bottleneck and explore
      # faster alternatives (e.g., keeping in memory and writing at the end).
      # TODO(noghabi): once we empty timline, we can just append the recent
      # spans to the file without serializing the entire trace.
      with open(self._trace_file_path, "wb") as f:
        f.write(builder.serialize())
    except Exception:  # pylint: disable=broad-except
      # Catching broad exceptions to ensure that failures in trace
      # serialization or writing do not crash the application. Tracing is
      # best-effort.
      logging.exception("Failed to write to trace file.")

  def write_timelines(self, timelines: dict[str, Timeline]) -> None:
    """Writes timelines to the trace file."""
    if not timelines:
      return

    builder = TraceProtoBuilder()

    # Sort timelines by ID to ensure consistent track ordering.
    sorted_ids = sorted(timelines.keys())

    for i, t_id in enumerate(sorted_ids):
      # Assign a unique UUID for the track.
      # We start from 1 because 0 is sometimes reserved or special.
      uuid = i + 1
      timeline = timelines[t_id]

      # Track Descriptor
      packet = builder.add_packet()
      packet.track_descriptor.uuid = uuid
      packet.track_descriptor.name = t_id
      # Use EXPLICIT child ordering if we had hierarchy, but here it's flat tracks.

      # Write spans
      # Sort spans by ID to ensure deterministic output order, though not strictly required by Perfetto.
      sorted_spans = sorted(timeline.spans.values(), key=lambda s: s.id)

      for s in sorted_spans:
        # Timestamp in nanoseconds, relative to timeline creation (born).
        # This aligns the start of the trace near 0.
        start_ns = int((s.begin - timeline.born) * 1e9)

        # Slice Begin
        packet_begin = builder.add_packet()
        packet_begin.timestamp = start_ns
        packet_begin.trusted_packet_sequence_id = 1
        packet_begin.track_event.type = TrackEvent.Type.TYPE_SLICE_BEGIN
        packet_begin.track_event.track_uuid = uuid
        packet_begin.track_event.name = (
            f"{s.name} (step={s.tags.get(perf_constants.GLOBAL_STEP)})"
        )

        # Add tags as debug annotations if needed.
        # For now, we skip detailed tag serialization to keep it simple,
        # unless requested.

        if s.ended:
          end_ns = int((s.end - timeline.born) * 1e9)
          # Slice End
          packet_end = builder.add_packet()
          packet_end.timestamp = end_ns
          packet_end.trusted_packet_sequence_id = 1
          packet_end.track_event.type = TrackEvent.Type.TYPE_SLICE_END
          packet_end.track_event.track_uuid = uuid

    self.write(builder)
