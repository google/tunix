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
import dataclasses
import time
from typing import Any

from absl import logging
from etils import epath
from perfetto.trace_builder.proto_builder import TraceProtoBuilder
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import timeline
from tunix.perf.experimental import timeline_utils

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackDescriptor
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent


Timeline = timeline.Timeline

_UUID_OFFSET = 100_000  # Offset for lane UUIDs.


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
      perf_constants.ENVIRONMENT,
  ]:
    if perf_constants.GROUP_ID in tags:
      parts.append(f"group_id={tags[perf_constants.GROUP_ID]}")

  if name in [perf_constants.ROLLOUT, perf_constants.ENVIRONMENT]:
    if perf_constants.PAIR_INDEX in tags:
      parts.append(f"pair_index={tags[perf_constants.PAIR_INDEX]}")

  if name == perf_constants.QUEUE:
    if perf_constants.QUEUED_SPAN in tags:
      parts.append(f"queued_span={tags[perf_constants.QUEUED_SPAN]}")

  if parts:
    return f"{name} ({', '.join(parts)})"
  return name


def _assign_lanes(
    spans: Iterable[timeline.Span],
) -> tuple[Mapping[int, int], int]:
  """Assigns lanes to spans to handle overlaps.

  Perfetto requires spans on the same track to be strictly nested (no arbitrary
  overlaps). This function assigns a lane index to each span such that spans
  in the same lane do not overlap.

  Args:
    spans: An iterable of spans to assign to lanes.

  Returns:
    A tuple (`lane_by_span_id`, `num_lanes`), where:
      `lane_by_span_id`: A dictionary mapping span IDs to their assigned lane
        index.
      `num_lanes`: The total number of lanes required.
  """
  sorted_spans = sorted(spans, key=lambda s: (s.begin, s.id))
  lanes_end_times = []
  lane_by_span_id = {}

  for s in sorted_spans:
    placed = False
    for lane_idx, lane_end in enumerate(lanes_end_times):
      if lane_end <= s.begin:
        lanes_end_times[lane_idx] = s.end
        lane_by_span_id[s.id] = lane_idx
        placed = True
        break
    if not placed:
      lane_by_span_id[s.id] = len(lanes_end_times)
      lanes_end_times.append(s.end)

  return lane_by_span_id, len(lanes_end_times)


class TraceWriter(abc.ABC):
  """An abstract base class for writing traces."""

  @abc.abstractmethod
  def write_timelines(self, timelines: Mapping[str, Timeline]) -> None:
    """Writes timelines to the trace."""


class NoopTraceWriter(TraceWriter):
  """A no-op trace writer that does nothing."""

  def write_timelines(self, timelines: Mapping[str, Timeline]) -> None:
    del self, timelines  # Unused.


@dataclasses.dataclass
class TrackInfo:
  """Information about a track.

  Attributes:
    name: The name of the track.
    uuid: The unique identifier for the track in Perfetto.
  """

  name: str
  uuid: int | None = None


class PerfettoTraceWriter(TraceWriter):
  """A writer for Perfetto trace events."""

  def __init__(
      self,
      trace_dir: str,
      role_to_devices: Mapping[str, Any] | None = None,
  ):
    """Initializes the instance.

    Args:
      trace_dir: The directory to export trace files to. This path can be a
        local Linux path or a remote storage path (e.g. gs://).
      role_to_devices: An optional mapping from role names to their assigned
        devices.
    """
    self._trace_dir = trace_dir
    self._role_to_devices = (
        dict(role_to_devices) if role_to_devices is not None else {}
    )
    self._track_info: dict[str, TrackInfo] = {}
    self._timeline_tracks: dict[str, str] = {}
    self._timeline_uuids: dict[str, int] = {}

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

  def _get_device_track_name(self, tl_id: str) -> str | None:
    """Gets a formatted track name for a device timeline.

    Args:
      tl_id: The timeline ID.

    Returns:
      A formatted track name. Returns None if the timeline is a host timeline
      or if the device is not found in the role_to_devices mapping.
    """
    if timeline_utils.is_host_timeline(tl_id):
      return None

    for role, devices in self._role_to_devices.items():
      for device in devices:
        device_str = timeline_utils.generate_device_timeline_id(device)
        if device_str == tl_id:
          camel_role = "".join(word.capitalize() for word in role.split("_"))
          return f"{camel_role} Cluster"
    return None

  def write(self, builder: TraceProtoBuilder) -> None:
    """Writes the built trace to the file."""
    if self._trace_file_path is None:
      return

    try:
      # TODO: b/480134569 -  see if file writing is a bottleneck and explore
      # faster alternatives (e.g., keeping in memory and writing at the end).
      self._trace_file_path.write_bytes(builder.serialize())
    except Exception:  # pylint: disable=broad-except
      # Catching broad exceptions to ensure that failures in trace
      # serialization or writing do not crash the application. Tracing is
      # best-effort.
      logging.exception(
          "Failed to write to trace file: %s", self._trace_file_path
      )

  def _init_tracks(
      self, timelines: Mapping[str, Timeline]
  ) -> None:
    """Initializes track info for timelines.

    Args:
      timelines: A mapping of timeline IDs to timelines.
    """

    for tl_id in sorted(timelines):
      if tl_id in self._timeline_tracks:
        continue

      tl = timelines[tl_id]
      if not tl.spans:
        continue

      if timeline_utils.is_host_timeline(tl_id):
        # Rollout only timelines threads.
        if timeline_utils.is_timeline_only_of_allowed_type(
            tl, [perf_constants.ROLLOUT]
        ):
          self._track_info["host_rollout"] = TrackInfo(
              name="Host - Rollout threads",
              uuid=_UUID_OFFSET + 1,
          )
          self._timeline_tracks[tl_id] = "host_rollout"
        # Main timelines threads.
        else:
          self._track_info["host_main"] = TrackInfo(
              name="Host - Main threads",
              uuid=_UUID_OFFSET,
          )
          self._timeline_tracks[tl_id] = "host_main"
      else:
        track_name = self._get_device_track_name(tl_id)
        if track_name:
          if not self._track_info.get(track_name):
            self._track_info[track_name] = TrackInfo(
                name=track_name,
                uuid=_UUID_OFFSET + (2 + len(self._track_info)),
            )
          self._timeline_tracks[tl_id] = track_name
        else:
          logging.warning("Failed to get track name for timeline ID: %s", tl_id)

  def write_timelines(self, timelines: Mapping[str, Timeline]) -> None:
    """Writes timelines to the trace file."""
    if not timelines:
      return

    timelines_dict = dict(timelines)
    lane_names = {}

    actor_dev_str = None
    if "actor" in self._role_to_devices and self._role_to_devices["actor"]:
      dev = self._role_to_devices["actor"][0]
      actor_dev_str = dev if isinstance(dev, str) else f"{dev.platform}{dev.id}"

    # Create actor execution and queue timelines.
    if actor_dev_str and actor_dev_str in timelines_dict:
      exec_tl, queue_tl = timeline_utils.flatten_overlapping_spans(
          timelines_dict[actor_dev_str]
      )
      exec_idle_tl = timeline_utils.add_idle_spans(exec_tl)
      exec_idle_tl.id = f"{actor_dev_str}_actor_exec_idle"
      queue_tl.id = f"{actor_dev_str}_actor_queue"
      timeline_utils.strip_tags_from_timeline(queue_tl, [perf_constants.STEP])
      lane_names[exec_idle_tl.id] = f"Actor cluster - (Execution)"
      lane_names[queue_tl.id] = f"Actor cluster - (Queued on device)"
      timelines_dict[exec_idle_tl.id] = exec_idle_tl
      timelines_dict[queue_tl.id] = queue_tl

    # Create rollout execution timeline.
    if "rollout" in self._role_to_devices and self._role_to_devices["rollout"]:
      dev = self._role_to_devices["rollout"][0]
      dev_str = dev if isinstance(dev, str) else f"{dev.platform}{dev.id}"
      if dev_str in timelines_dict:
        rollout_tl = timelines_dict[dev_str]

        # Add weight sync spans from actor timeline to rollout timeline.
        if actor_dev_str and actor_dev_str in timelines_dict:
          actor_tl = timelines_dict[actor_dev_str]
          weight_sync_spans = timeline_utils.filter_spans_by_name(
              actor_tl, perf_constants.WEIGHT_SYNC
          )
          rollout_tl = timeline_utils.add_spans_to_timeline(
              rollout_tl, weight_sync_spans
          )

        merged_tl = timeline_utils.merge_overlapping_spans(rollout_tl)
        timeline_utils.strip_tags_from_timeline(
            merged_tl,
            [perf_constants.GROUP_ID, perf_constants.PAIR_INDEX],
        )
        aggregate_rollout_tl = timeline_utils.add_idle_spans(merged_tl)
        aggregate_rollout_tl.id = f"{dev_str}_rollout_merged_idle"
        lane_names[aggregate_rollout_tl.id] = (
            f"Rollout cluster - (Execution)"
        )
        timelines_dict[aggregate_rollout_tl.id] = aggregate_rollout_tl

    builder = TraceProtoBuilder()

    self._init_tracks(timelines)

    # Write track descriptors for parent tracks.
    for track_info in self._track_info.values():
      packet = builder.add_packet()
      packet.track_descriptor.uuid = track_info.uuid
      packet.track_descriptor.name = track_info.name

    # Sort timelines by ID to ensure consistent track ordering.
    sorted_ids = sorted(timelines_dict)

    events = []

    for tl_id in sorted_ids:
      tl = timelines_dict[tl_id]

      if not tl.spans:
        continue

      # Assign a UUID to the timeline if it hasn't been assigned one yet. Offset
      # by 2 to account for track descriptor UUIDs.
      if tl_id not in self._timeline_uuids:
        self._timeline_uuids[tl_id] = _UUID_OFFSET * (
            len(self._timeline_uuids) + 2
        )
      tl_uuid = self._timeline_uuids[tl_id]

      packet = builder.add_packet()
      packet.track_descriptor.uuid = tl_uuid
      packet.track_descriptor.name = lane_names.get(tl_id, tl_id)

      if tl_id in self._timeline_tracks:
        track_info = self._track_info[self._timeline_tracks[tl_id]]
        packet.track_descriptor.parent_uuid = track_info.uuid

      lane_by_span_id, num_lanes = _assign_lanes(tl.spans.values())

      if num_lanes > 1:
        # Emit track descriptors for each lane so they group under the timeline
        for lane_idx in range(num_lanes):
          lane_uuid = tl_uuid + lane_idx + 1
          packet = builder.add_packet()
          packet.track_descriptor.uuid = lane_uuid
          packet.track_descriptor.parent_uuid = tl_uuid
          packet.track_descriptor.name = ""  # empty name for lanes

      for s in tl.spans.values():
        lane_idx = lane_by_span_id[s.id]
        lane_uuid = tl_uuid if num_lanes <= 1 else (tl_uuid + lane_idx + 1)

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
