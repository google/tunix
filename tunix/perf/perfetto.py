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
from tunix.perf import span

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackDescriptor
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent

Span = span.Span
SpanGroup = span.SpanGroup

DEFAULT_EXPORT_DIR = "/tmp/perf_traces"
ROOT_TRACK_UUID = 100


def _add_trace_events(
    *,
    builder: TraceProtoBuilder,
    global_step_group: SpanGroup,
    rollout_spans: list[Span],
    refer_inference_spans: list[Span],
    actor_train_groups: list[SpanGroup],
) -> None:
  """Populates the TraceProtoBuilder with spans and groups.

  Args:
    builder: The TraceProtoBuilder to add events to.
    global_step_group: A SpanGroup representing the global step.
    rollout_spans: A list of Spans for rollout operations.
    refer_inference_spans: A list of Spans for reference inference operations.
    actor_train_groups: A list of SpanGroups for actor training operations.
  """

  packet = builder.add_packet()
  packet.track_descriptor.uuid = ROOT_TRACK_UUID
  packet.track_descriptor.name = "Post-Training Run"
  packet.track_descriptor.child_ordering = (
      TrackDescriptor.ChildTracksOrdering.EXPLICIT
  )

  # Metadata for process names
  # Main: uuid=1, Rollout: uuid=2, Reference: uuid=3, Actor: uuid=4
  for name, uuid in [
      ("Main", 1),
      ("Rollout", 2),
      ("Reference", 3),
      ("Actor", 4),
  ]:
    packet = builder.add_packet()
    packet.track_descriptor.uuid = uuid
    packet.track_descriptor.name = name
    packet.track_descriptor.parent_uuid = ROOT_TRACK_UUID
    packet.track_descriptor.sibling_order_rank = uuid

  def add_span(s: Span | SpanGroup, track_uuid: int):
    # Slice Begin
    packet_begin = builder.add_packet()
    packet_begin.timestamp = int(s.begin * 1e9)
    packet_begin.trusted_packet_sequence_id = 1
    packet_begin.track_event.type = TrackEvent.Type.TYPE_SLICE_BEGIN
    packet_begin.track_event.track_uuid = track_uuid
    packet_begin.track_event.name = s.name

    # Slice End
    packet_end = builder.add_packet()
    packet_end.timestamp = int(s.end * 1e9)
    packet_end.trusted_packet_sequence_id = 1
    packet_end.track_event.type = TrackEvent.Type.TYPE_SLICE_END
    packet_end.track_event.track_uuid = track_uuid

  def add_group(g: SpanGroup, track_uuid: int):
    add_span(g, track_uuid)
    for inner in g.inner:
      if isinstance(inner, SpanGroup):
        add_group(inner, track_uuid)
      elif isinstance(inner, Span):
        add_span(inner, track_uuid)

  # Main
  add_group(global_step_group, 1)

  # Rollout
  for s in rollout_spans:
    add_span(s, 2)

  # Reference
  for s in refer_inference_spans:
    add_span(s, 3)

  # Actor
  for g in actor_train_groups:
    add_group(g, 4)


class PerfettoTraceWriter:
  """A writer for Perfetto trace events."""

  def __init__(self, export_dir: str | None):
    """Initializes the PerfettoTraceWriter.

    Args:
      export_dir: The directory to export trace files to. If None,
        DEFAULT_EXPORT_DIR is used. This path must be an absolute Linux path
        with write permissions.
    """
    self._export_dir = export_dir or DEFAULT_EXPORT_DIR
    self._trace_file_path = None
    try:
      os.makedirs(self._export_dir, exist_ok=True)
      trace_file_name = f"perfetto_trace_{int(time.time())}.pb"
      self._trace_file_path = os.path.join(self._export_dir, trace_file_name)
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
      with open(self._trace_file_path, "ab") as f:
        f.write(builder.serialize())
    except Exception:  # pylint: disable=broad-except
      # Catching broad exceptions to ensure that failures in trace
      # serialization or writing do not crash the application. Tracing is
      # best-effort.
      logging.exception("Failed to write to trace file.")

  def log_trace(
      self,
      global_step_group: SpanGroup,
      rollout_spans: list[Span],
      refer_inference_spans: list[Span],
      actor_train_groups: list[SpanGroup],
  ) -> None:
    """Generates and writes Perfetto trace events to a file.

    Args:
      global_step_group: A SpanGroup representing the global step.
      rollout_spans: A list of Spans for rollout operations.
      refer_inference_spans: A list of Spans for reference inference operations.
      actor_train_groups: A list of SpanGroups for actor training operations.
    """
    builder = TraceProtoBuilder()
    _add_trace_events(
        builder=builder,
        global_step_group=global_step_group,
        rollout_spans=rollout_spans,
        refer_inference_spans=refer_inference_spans,
        actor_train_groups=actor_train_groups,
    )
    self.write(builder)
