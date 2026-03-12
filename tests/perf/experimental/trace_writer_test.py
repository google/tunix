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
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import trace_writer as trace_writer_lib
from tunix.perf.experimental import tracer

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TracePacket
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackDescriptor
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent


class PerfettoTraceWriterTest(absltest.TestCase):

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
              perf_constants.GLOBAL_STEP: 42,
              perf_constants.MINI_BATCH_STEP: 10,
          },
      )
      t.stop_span(1002.0)

      writer.write_timelines({"timeline_test": t})

    # Expected packets:
    # 1. Track Descriptor
    # 2. Slice Begin
    # 3. Slice End
    with self.subTest("Packet Count"):
      self.assertLen(captured_packets, 3)

    with self.subTest("Track Descriptor Packet"):
      descriptor_packet = captured_packets[0]
      self.assertEqual(descriptor_packet.track_descriptor.name, "timeline_test")
      uuid = descriptor_packet.track_descriptor.uuid

    with self.subTest("Slice Begin Packet"):
      begin_packet = captured_packets[1]
      self.assertEqual(
          begin_packet.track_event.type, TrackEvent.Type.TYPE_SLICE_BEGIN
      )
      self.assertEqual(begin_packet.track_event.track_uuid, uuid)
      # Timestamp is relative to 'born' (1000.0), so 1001.0 -> 1.0s -> 1e9 ns
      self.assertEqual(begin_packet.timestamp, 1_000_000_000)
      self.assertEqual(begin_packet.track_event.name, "span_test (step=42)")

    with self.subTest("Slice End Packet"):
      end_packet = captured_packets[2]
      self.assertEqual(
          end_packet.track_event.type, TrackEvent.Type.TYPE_SLICE_END
      )
      self.assertEqual(end_packet.track_event.track_uuid, uuid)
      # 1002.0 -> 2.0s -> 2e9 ns
      self.assertEqual(end_packet.timestamp, 2_000_000_000)

  def test_perfetto_trace_writer_integration(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      writer = trace_writer_lib.PerfettoTraceWriter(trace_dir=tmp_dir)

      # Create multiple timelines with various configurations to verify E2E flow.
      t1 = tracer.Timeline("timeline1", 1000.0)
      t1.start_span("span1", 1001.0)
      t1.stop_span(1002.0)

      t2 = tracer.Timeline("timeline_tags", 2000.0)
      t2.start_span(
          "span_tags",
          2001.0,
          tags={perf_constants.GLOBAL_STEP: 100},
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
