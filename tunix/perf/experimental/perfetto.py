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

from collections.abc import Mapping
import os
import time
from typing import Any

from absl import logging
from perfetto.trace_builder.proto_builder import TraceProtoBuilder
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import timeline

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackDescriptor
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent

Timeline = timeline.Timeline


def _create_span_name(name: str, tags: Mapping[str, Any]) -> str:
  """Creates a descriptive name for the span based on its tags."""
  parts = []
  if perf_constants.STEP in tags:
    parts.append(f"step={tags[perf_constants.STEP]}")

  if name == perf_constants.PEFT_TRAIN:
    if perf_constants.ROLE in tags:
      parts.append(f"role={tags[perf_constants.ROLE]}")
    if perf_constants.MINI_BATCH in tags:
      parts.append(f"mini_batch={tags[perf_constants.MINI_BATCH]}")
    if perf_constants.MICRO_BATCH in tags:
      parts.append(f"micro_batch={tags[perf_constants.MICRO_BATCH]}")

  if name in [
      perf_constants.ROLLOUT,
      perf_constants.REFERENCE_INFERENCE,
      perf_constants.OLD_ACTOR_INFERENCE,
      perf_constants.ADVANTAGE_COMPUTATION,
  ]:
    if perf_constants.GROUP_ID in tags:
      parts.append(f"group_id={tags[perf_constants.GROUP_ID]}")

  if name == perf_constants.ROLLOUT:
    if perf_constants.PAIR_INDEX in tags:
      parts.append(f"pair_index={tags[perf_constants.PAIR_INDEX]}")

  if parts:
    return f"{name} ({', '.join(parts)})"
  return name


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

    events = []

    for i, tl_id in enumerate(sorted_ids):
      # Assign a unique UUID for the track.
      # We start from 1 because 0 is sometimes reserved or special.
      # Multiply by a large number to reserve a block of UUIDs for child lanes.
      tl_uuid = (i + 1) * 1000000
      tl = timelines[tl_id]

      # Skip empty timelines.
      if not tl.spans:
        continue

      # Track Descriptor for the timeline group
      packet = builder.add_packet()
      packet.track_descriptor.uuid = tl_uuid
      packet.track_descriptor.name = tl_id

      # Determine lanes for spans to handle overlaps. Perfetto requires spans
      # on the same track to be strictly nested (no arbitrary overlaps).
      # TODO(noghabi): Instead of agnostically splitting into lanes, define
      # proper groupings for spans, e.g., a better way for combining rollouts
      # and overlaps of peft_train and reference_inference.
      sorted_spans = sorted(tl.spans.values(), key=lambda s: (s.begin, s.id))
      lanes_end_times = []
      span_to_lane = {}

      for s in sorted_spans:
        placed = False
        for lane_idx, lane_end in enumerate(lanes_end_times):
          if lane_end <= s.begin:
            lanes_end_times[lane_idx] = s.end
            span_to_lane[s.id] = lane_idx
            placed = True
            break
        if not placed:
          span_to_lane[s.id] = len(lanes_end_times)
          lanes_end_times.append(s.end)

      if len(lanes_end_times) <= 1:
        # Just use the main track, no need for child tracks
        for s in tl.spans.values():
          span_to_lane[s.id] = -1
      else:
        # Emit track descriptors for each lane so they group under the timeline
        for lane_idx in range(len(lanes_end_times)):
          lane_uuid = tl_uuid + lane_idx + 1
          packet = builder.add_packet()
          packet.track_descriptor.uuid = lane_uuid
          packet.track_descriptor.parent_uuid = tl_uuid
          packet.track_descriptor.name = f"Lane {lane_idx}"

      for s in tl.spans.values():
        lane_idx = span_to_lane[s.id]
        lane_uuid = tl_uuid if lane_idx == -1 else (tl_uuid + lane_idx + 1)

        # Timestamp in nanoseconds, relative to timeline creation (born).
        start_ns = int((s.begin - tl.born) * 1e9)
        events.append({
            "timestamp": start_ns,
            "type": TrackEvent.Type.TYPE_SLICE_BEGIN,
            "uuid": lane_uuid,
            "name": _create_span_name(s.name, s.tags),
        })

        if s.ended:
          end_ns = int((s.end - tl.born) * 1e9)
          events.append({
              "timestamp": end_ns,
              "type": TrackEvent.Type.TYPE_SLICE_END,
              "uuid": lane_uuid,
              "name": None,
          })

    # Perfetto trace processor requires events within a sequence to be strictly
    # sorted by timestamp. Out-of-order events can cause spans to be rendered
    # incorrectly (e.g., stretched end times).
    events.sort(
        key=lambda e: (
            e["timestamp"],
            0 if e["type"] == TrackEvent.Type.TYPE_SLICE_END else 1,
        )
    )

    for e in events:
      packet = builder.add_packet()
      packet.timestamp = e["timestamp"]
      packet.trusted_packet_sequence_id = 1
      packet.track_event.type = e["type"]
      packet.track_event.track_uuid = e["uuid"]
      if e["name"] is not None:
        packet.track_event.name = e["name"]

    self.write(builder)
