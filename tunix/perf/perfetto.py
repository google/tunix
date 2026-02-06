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

"""Perfetto trace writer and helper functions."""

from __future__ import annotations

import logging
import os
import time

from perfetto.trace_builder.proto_builder import TraceProtoBuilder
from tunix.perf import span

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent

Span = span.Span
SpanGroup = span.SpanGroup

DEFAULT_EXPORT_DIR = "/tmp/perf_traces"


class PerfettoTraceWriter:
  """Writes Perfetto trace events to a file."""

  def __init__(self, export_dir: str | None):
    self._export_dir = export_dir or DEFAULT_EXPORT_DIR
    self._file = None
    try:
      os.makedirs(self._export_dir, exist_ok=True)
      trace_file_name = f"perfetto_trace_{int(time.time())}.pb"
      trace_file = os.path.join(self._export_dir, trace_file_name)
      self._file = open(trace_file, "ab")
      logging.info("Initializing perfetto trace writer at: %s", trace_file)
    except Exception as e:  # pylint: disable=broad-except
      logging.error("Failed to initialize perfetto trace writer: %s", e)

  def write(self, builder: TraceProtoBuilder) -> None:
    """Writes the built trace to the file."""
    if self._file is None:
      return
    try:
      self._file.write(builder.serialize())
      self._file.flush()
    except Exception as e:  # pylint: disable=broad-except
      logging.error(
          "Failed to write perfetto trace. Skipping the trace dump: %s", e
      )

  def log_trace(
      self,
      global_step_group: SpanGroup,
      rollout_spans: list[Span],
      refer_inference_spans: list[Span],
      actor_train_groups: list[SpanGroup],
  ) -> None:
    """Generates and writes Perfetto trace events to a file."""
    builder = TraceProtoBuilder()
    _add_trace_events(
        builder,
        global_step_group,
        rollout_spans,
        refer_inference_spans,
        actor_train_groups,
    )
    self.write(builder)


def _add_trace_events(
    builder: TraceProtoBuilder,
    global_step_group: SpanGroup,
    rollout_spans: list[Span],
    refer_inference_spans: list[Span],
    actor_train_groups: list[SpanGroup],
) -> None:
  """Populates the TraceProtoBuilder with spans and groups."""

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
  if global_step_group:
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
