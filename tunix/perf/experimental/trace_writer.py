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
from collections.abc import Iterable, Mapping
import time
from typing import Any, Tuple

from absl import logging
from etils import epath
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


def _assign_lanes(
    spans: Iterable[timeline.Span],
) -> Tuple[Mapping[int, int], int]:
  """Assigns lanes to spans to handle overlaps.

  Perfetto requires spans on the same track to be strictly nested (no arbitrary
  overlaps). This function assigns a lane index to each span such that spans
  in the same lane do not overlap.

  Args:
    spans: An iterable of spans to assign to lanes.

  Returns:
    A tuple containing:
      - A dictionary mapping span IDs to their assigned lane index. If all spans
        fit in a single lane, the lane index is -1.
      - The total number of lanes required.
  """
  # TODO: noghabi - Instead of agnostically splitting into lanes, define
  # proper groupings for spans, e.g., a better way for combining rollouts
  # and overlaps of peft_train and reference_inference.
  sorted_spans = sorted(spans, key=lambda s: (s.begin, s.id))
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
    for s in spans:
      span_to_lane[s.id] = -1

  return span_to_lane, len(lanes_end_times)


def _is_host_timeline(tl_id: str) -> bool:
  return tl_id.startswith("host-")


def _is_rollout_only_timeline(tl: Timeline) -> bool:
  if not tl.spans:
    return False
  return all(span.name == perf_constants.ROLLOUT for span in tl.spans.values())


def _get_track_name(tl_id: str, role_to_devices: Mapping[str, Any]) -> str:
  """Gets a formatted track name for a timeline.

  Args:
    tl_id: The timeline ID.
    role_to_devices: A mapping from role names to their assigned devices.

  Returns:
    A formatted track name.
  """
  if _is_host_timeline(tl_id):
    return tl_id

  for role, devices in role_to_devices.items():
    for device in devices:
      # TODO: noghabi - this is duplicated from tracer.py. Refactor to a common
      # function.
      device_str = (
          device if isinstance(device, str) else f"{device.platform}{device.id}"
      )
      if device_str == tl_id:
        return f"{role} - {tl_id}"
  return tl_id


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

  def __init__(
      self,
      trace_dir: str,
      role_to_devices: Mapping[str, Any] | None = None,
  ):
    """Initializes the PerfettoTraceWriter.

    Args:
      trace_dir: The directory to export trace files to. This path can be a
        local Linux path or a remote storage path (e.g. gs://).
      role_to_devices: An optional mapping from role names to their assigned
        devices.
    """
    self._trace_dir = trace_dir
    self._role_to_devices = dict(role_to_devices) if role_to_devices else {}
    self._track_names: dict[str, str] = {}
    self._trace_file_path = None
    try:
      trace_dir_path = epath.Path(self._trace_dir)
      trace_dir_path.mkdir(parents=True, exist_ok=True)
      trace_file_name = f"perfetto_trace_v2_{int(time.time())}.pb"
      self._trace_file_path = trace_dir_path / trace_file_name
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
      # TODO: b/480134569 -  see if file writing is a bottleneck and explore
      # faster alternatives (e.g., keeping in memory and writing at the end).
      # TODO: noghabi - once we empty timeline, we can just append the recent
      # spans to the file without serializing the entire trace.
      self._trace_file_path.write_bytes(builder.serialize())
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

    host_main_uuid = 999998
    host_rollout_uuid = 999999

    has_host_main = False
    has_host_rollout = False
    for tl_id in sorted_ids:
      tl = timelines[tl_id]
      if not tl.spans:
        continue
      if _is_host_timeline(tl_id):
        if _is_rollout_only_timeline(tl):
          has_host_rollout = True
        else:
          has_host_main = True

    if has_host_main:
      packet = builder.add_packet()
      packet.track_descriptor.uuid = host_main_uuid
      packet.track_descriptor.name = "Host - Main threads"

    if has_host_rollout:
      packet = builder.add_packet()
      packet.track_descriptor.uuid = host_rollout_uuid
      packet.track_descriptor.name = "Host - Rollout threads"

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

      if tl_id not in self._track_names:
        self._track_names[tl_id] = _get_track_name(tl_id, self._role_to_devices)
      packet.track_descriptor.name = self._track_names[tl_id]

      if _is_host_timeline(tl_id):
        if _is_rollout_only_timeline(tl):
          packet.track_descriptor.parent_uuid = host_rollout_uuid
        else:
          packet.track_descriptor.parent_uuid = host_main_uuid

      span_to_lane, num_lanes = _assign_lanes(tl.spans.values())

      if num_lanes > 1:
        # Emit track descriptors for each lane so they group under the timeline
        for lane_idx in range(num_lanes):
          lane_uuid = tl_uuid + lane_idx + 1
          packet = builder.add_packet()
          packet.track_descriptor.uuid = lane_uuid
          packet.track_descriptor.parent_uuid = tl_uuid
          packet.track_descriptor.name = ""  # empty name for lanes

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
