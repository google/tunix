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


import os
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import trace_writer as trace_writer_lib
from tunix.perf.experimental import tracer

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TracePacket
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackDescriptor
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent


class PerfettoTraceWriterTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "basic_with_global_step",
          "my_span",
          {perf_constants.STEP: 10},
          "my_span (step=10)",
      ),
      (
          "peft_train_with_role",
          perf_constants.PEFT_TRAIN,
          {perf_constants.STEP: 20, perf_constants.ROLE: "actor"},
          "peft_train (step=20, role=actor)",
      ),
      (
          "rollout_with_group_id_and_pair_index",
          perf_constants.ROLLOUT,
          {
              perf_constants.GROUP_ID: 5,
              perf_constants.PAIR_INDEX: 3,
              perf_constants.STEP: 100,
          },
          "rollout (step=100, group_id=5, pair_index=3)",
      ),
      (
          "rollout_with_missing_pair_index",
          perf_constants.ROLLOUT,
          {perf_constants.GROUP_ID: 5},
          "rollout (group_id=5)",
      ),
      (
          "unknown_span_with_extra_tags",
          "unknown_span",
          {perf_constants.ROLE: "actor", perf_constants.STEP: 50},
          "unknown_span (step=50)",
      ),
      (
          "no_tags",
          "simple_span",
          {},
          "simple_span",
      ),
  )
  def test_create_span_name(self, name, tags, expected):
    actual_name = trace_writer_lib._create_span_name(name, tags)
    self.assertEqual(actual_name, expected)

  @mock.patch.object(trace_writer_lib, "TraceProtoBuilder", autospec=True)
  def test_write_timelines_content(self, mock_builder_cls):
    mock_builder = mock_builder_cls.return_value
    mock_builder.serialize.return_value = b""
    captured_packets = []

    def add_packet_side_effect():
      p = mock.create_autospec(TracePacket, instance=True)
      p.track_descriptor = mock.create_autospec(TrackDescriptor, instance=True)
      p.track_event = mock.create_autospec(TrackEvent, instance=True)
      captured_packets.append(p)
      return p

    mock_builder.add_packet.side_effect = add_packet_side_effect

    with tempfile.TemporaryDirectory() as tmp_dir:
      writer = trace_writer_lib.PerfettoTraceWriter(trace_dir=tmp_dir)

      t = tracer.Timeline("timeline_test", 1000.0)
      # Create a span with a specific tag
      t.start_span(
          "span_test",
          1001.0,
          tags={
              perf_constants.STEP: 42,
              perf_constants.MINI_BATCH: 10,
          },
      )
      t.stop_span(1002.0)

      writer.write_timelines({"timeline_test": t})

    self.assertLen(captured_packets, 3)
    uuid = captured_packets[0].track_descriptor.uuid
    self.assertEqual(captured_packets[0].track_descriptor.name, "timeline_test")

    actual_events = [
        {
            "timestamp": p.timestamp,
            "type": p.track_event.type,
            "track_uuid": p.track_event.track_uuid,
            "name": (
                p.track_event.name
                if p.track_event.type == TrackEvent.Type.TYPE_SLICE_BEGIN
                else None
            ),
        }
        for p in captured_packets[1:]
    ]
    expected_events = [
        {
            "timestamp": 1_000_000_000,
            "type": TrackEvent.Type.TYPE_SLICE_BEGIN,
            "track_uuid": uuid,
            "name": "span_test (step=42)",
        },
        {
            "timestamp": 2_000_000_000,
            "type": TrackEvent.Type.TYPE_SLICE_END,
            "track_uuid": uuid,
            "name": None,
        },
    ]
    self.assertEqual(actual_events, expected_events)

  @mock.patch.object(trace_writer_lib, "TraceProtoBuilder", autospec=True)
  def test_write_timelines_overlapping_spans(self, mock_builder_cls):
    mock_builder = mock_builder_cls.return_value
    mock_builder.serialize.return_value = b""
    captured_packets = []

    def add_packet_side_effect():
      p = mock.create_autospec(TracePacket, instance=True)
      p.track_descriptor = mock.create_autospec(TrackDescriptor, instance=True)
      p.track_event = mock.create_autospec(TrackEvent, instance=True)
      captured_packets.append(p)
      return p

    mock_builder.add_packet.side_effect = add_packet_side_effect

    with tempfile.TemporaryDirectory() as tmp_dir:
      writer = trace_writer_lib.PerfettoTraceWriter(trace_dir=tmp_dir)

      t = tracer.Timeline("overlap_timeline", 1000.0)

      # Create mock spans manually to simulate overlaps
      span1 = tracer.Span(name="span_1", begin=1001.0, id=1)
      span1.end = 1005.0
      t.spans[1] = span1

      # Overlaps with span 1
      span2 = tracer.Span(name="span_2", begin=1002.0, id=2)
      span2.end = 1006.0
      t.spans[2] = span2

      # Starts exactly when span 2 ends, testing tie-breaker logic
      span3 = tracer.Span(name="span_3", begin=1006.0, id=3)
      span3.end = 1010.0
      t.spans[3] = span3

      writer.write_timelines({"overlap_timeline": t})

    self.assertLen(captured_packets, 9)

    main_uuid = captured_packets[0].track_descriptor.uuid
    lane0_uuid = captured_packets[1].track_descriptor.uuid
    lane1_uuid = captured_packets[2].track_descriptor.uuid

    actual_descriptors = [
        {
            "name": p.track_descriptor.name,
            "parent_uuid": p.track_descriptor.parent_uuid if i > 0 else None,
        }
        for i, p in enumerate(captured_packets[:3])
    ]
    expected_descriptors = [
        {"name": "overlap_timeline", "parent_uuid": None},
        {"name": "Lane 0", "parent_uuid": main_uuid},
        {"name": "Lane 1", "parent_uuid": main_uuid},
    ]
    self.assertEqual(actual_descriptors, expected_descriptors)

    actual_events = [
        {
            "timestamp": p.timestamp,
            "type": p.track_event.type,
            "track_uuid": p.track_event.track_uuid,
            "name": (
                p.track_event.name
                if p.track_event.type == TrackEvent.Type.TYPE_SLICE_BEGIN
                else None
            ),
        }
        for p in captured_packets[3:]
    ]
    # Expected packets:
    # 1. Main Track Descriptor (overlap_timeline)
    # 2. Lane 0 Track Descriptor
    # 3. Lane 1 Track Descriptor
    # 4. span_1 BEGIN (Lane 0, t=1.0)
    # 5. span_2 BEGIN (Lane 1, t=2.0)
    # 6. span_1 END (Lane 0, t=5.0)
    # 7. span_2 END (Lane 1, t=6.0)
    # 8. span_3 BEGIN (Lane 1, t=6.0)  <- Tie-breaker should put END before BEGIN!
    # 9. span_3 END (Lane 1, t=10.0)
    expected_events = [
        {
            "timestamp": 1_000_000_000,
            "type": TrackEvent.Type.TYPE_SLICE_BEGIN,
            "track_uuid": lane0_uuid,
            "name": "span_1",
        },
        {
            "timestamp": 2_000_000_000,
            "type": TrackEvent.Type.TYPE_SLICE_BEGIN,
            "track_uuid": lane1_uuid,
            "name": "span_2",
        },
        {
            "timestamp": 5_000_000_000,
            "type": TrackEvent.Type.TYPE_SLICE_END,
            "track_uuid": lane0_uuid,
            "name": None,
        },
        {
            "timestamp": 6_000_000_000,
            "type": TrackEvent.Type.TYPE_SLICE_END,
            "track_uuid": lane1_uuid,
            "name": None,
        },
        {
            "timestamp": 6_000_000_000,
            "type": TrackEvent.Type.TYPE_SLICE_BEGIN,
            "track_uuid": lane0_uuid,
            "name": (
                "span_3"
            ),  # Tie-breaker puts END before BEGIN, and reuses Lane 0
        },
        {
            "timestamp": 10_000_000_000,
            "type": TrackEvent.Type.TYPE_SLICE_END,
            "track_uuid": lane0_uuid,
            "name": None,
        },
    ]
    self.assertEqual(actual_events, expected_events)

  def test_perfetto_trace_writer_integration(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      writer = trace_writer_lib.PerfettoTraceWriter(trace_dir=tmp_dir)

      # Create multiple timelines with various configs to verify E2E flow.
      t1 = tracer.Timeline("timeline1", 1000.0)
      t1.start_span("span1", 1001.0)
      t1.stop_span(1002.0)

      t2 = tracer.Timeline("timeline_tags", 2000.0)
      t2.start_span(
          "span_tags",
          2001.0,
          tags={perf_constants.STEP: 100},
      )
      t2.stop_span(2002.0)

      timelines = {"timeline1": t1, "timeline_tags": t2}

      writer.write_timelines(timelines)

      # Check if file was created and has content
      files = os.listdir(tmp_dir)
      self.assertLen(files, 1)
      self.assertStartsWith(files[0], "perfetto_trace_v2_")
      self.assertEndsWith(files[0], ".pb")
      self.assertGreater(os.path.getsize(os.path.join(tmp_dir, files[0])), 0)

  def test_perfetto_trace_writer_invalid_dir(self):
    # Use a file path as directory to cause failure
    with tempfile.NamedTemporaryFile() as tmp_file:
      # Expect initialization to fail gracefully (log error, no crash)
      writer = trace_writer_lib.PerfettoTraceWriter(trace_dir=tmp_file.name)

      t = tracer.Timeline("timeline", 1000.0)
      # Expect write to fail gracefully
      writer.write_timelines({"timeline": t})

  def test_perfetto_trace_writer_empty_timelines(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      writer = trace_writer_lib.PerfettoTraceWriter(trace_dir=tmp_dir)
      writer.write_timelines({})
      files = os.listdir(tmp_dir)
      # No content should be written.
      self.assertEmpty(files)


class NoopTraceWriterTest(absltest.TestCase):

  def test_noop_trace_writer_write_timelines(self):
    writer = trace_writer_lib.NoopTraceWriter()
    t = tracer.Timeline("timeline", 1000.0)
    t.start_span("span1", 1001.0)
    # Should not crash and do nothing.
    writer.write_timelines({"timeline": t})


if __name__ == "__main__":
  absltest.main()
