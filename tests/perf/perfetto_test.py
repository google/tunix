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

import builtins
import os
import time
from unittest import mock

from absl.testing import absltest
from perfetto.trace_builder.proto_builder import TraceProtoBuilder
from tunix.perf import perfetto
from tunix.perf import span

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TracePacket
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackDescriptor
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import TrackEvent


def _create_mock_spans():
  """Creates a nested span group for testing."""
  g_global_step = span.SpanGroup("global_step")
  g_global_step.begin = 0.0
  g_global_step.end = 10.0

  g_mini_batch = span.SpanGroup("mini_batch_step", outer=g_global_step)
  g_mini_batch.begin = 0.0
  g_mini_batch.end = 10.0

  # Rollout: 0-4
  s_rollout = span.Span("rollout", 0.0)
  s_rollout.end = 4.0
  g_mini_batch.inner.append(s_rollout)
  rollout_spans = [s_rollout]

  # Reference Inference: 3-6
  s_refer = span.Span("refer_inference", 4.0)
  s_refer.end = 6.0
  g_mini_batch.inner.append(s_refer)
  refer_inference_spans = [s_refer]

  # Actor Training: 6-9
  g_actor = span.SpanGroup("actor_training", outer=g_mini_batch)
  g_actor.begin = 6.0
  g_actor.end = 9.0
  s_actor = span.Span("peft_train_step", 6.0)
  s_actor.end = 9.0
  g_actor.inner.append(s_actor)
  actor_train_groups = [g_actor]

  return (
      g_global_step,
      rollout_spans,
      refer_inference_spans,
      actor_train_groups,
  )


class PerfettoTraceWriterTest(absltest.TestCase):

  @mock.patch.object(os, "makedirs", autospec=True)
  @mock.patch.object(builtins, "open", autospec=True)
  @mock.patch.object(time, "time", autospec=True)
  def test_init_success(self, mock_time, mock_open, mock_makedirs):
    mock_time.return_value = 12345
    writer = perfetto.PerfettoTraceWriter("/tmp/test_dir")

    mock_makedirs.assert_called_once_with("/tmp/test_dir", exist_ok=True)
    mock_open.assert_called_once_with(
        "/tmp/test_dir/perfetto_trace_12345.pb", "ab"
    )
    self.assertEqual(
        writer._trace_file_path, "/tmp/test_dir/perfetto_trace_12345.pb"
    )

  @mock.patch.object(os, "makedirs", autospec=True)
  def test_init_failure_logs_error(self, mock_makedirs):
    mock_makedirs.side_effect = OSError("Permission denied")
    with self.assertLogs(level="ERROR") as cm:
      writer = perfetto.PerfettoTraceWriter("/tmp/test_dir")
    self.assertIsNone(writer._trace_file_path)
    self.assertLen(cm.output, 1)
    self.assertIn(
        "Failed to initialize perfetto trace writer. Skipping trace dumping"
        " for this run.",
        cm.output[0],
    )
    self.assertIn("Permission denied", cm.output[0])

  @mock.patch.object(os, "makedirs", autospec=True)
  @mock.patch.object(builtins, "open", autospec=True)
  def test_write(self, mock_open, mock_makedirs):
    mock_file = mock_open.return_value
    # Context manager setup
    mock_open.return_value.__enter__.return_value = mock_file

    writer = perfetto.PerfettoTraceWriter(None)

    mock_builder = mock.create_autospec(TraceProtoBuilder, instance=True)
    mock_builder.serialize.return_value = b"bytes"

    writer.write(mock_builder)

    # Check that open was called twice (once in init, once in write)
    self.assertEqual(mock_open.call_count, 2)
    mock_file.write.assert_called_once_with(b"bytes")

  @mock.patch.object(os, "makedirs", autospec=True)
  @mock.patch.object(builtins, "open", autospec=True)
  @mock.patch.object(perfetto, "TraceProtoBuilder", autospec=True)
  def test_log_trace(self, mock_builder_cls, mock_open, mock_makedirs):
    mock_builder = mock_builder_cls.return_value
    mock_file = mock_open.return_value
    mock_open.return_value.__enter__.return_value = mock_file
    captured_packets = []

    def add_packet_side_effect():
      p = mock.create_autospec(TracePacket, instance=True)
      p.track_descriptor = mock.create_autospec(TrackDescriptor, instance=True)
      p.track_event = mock.create_autospec(TrackEvent, instance=True)
      captured_packets.append(p)
      return p

    mock_builder.add_packet.side_effect = add_packet_side_effect

    writer = perfetto.PerfettoTraceWriter(None)

    writer.log_trace(*_create_mock_spans())

    # Metadata (4) + Global (12) + Rollout (2) + Refer (2) + Actor (4) = 24
    self.assertLen(captured_packets, 24)

    # Helper to simplify assertions
    SliceBegin = perfetto.TrackEvent.Type.TYPE_SLICE_BEGIN
    SliceEnd = perfetto.TrackEvent.Type.TYPE_SLICE_END

    def assert_metadata(packet, name, uuid):
      self.assertEqual(packet.track_descriptor.uuid, uuid)
      self.assertEqual(packet.track_descriptor.name, name)

    def assert_slice(packet, type_, uuid, ts, name=None):
      self.assertEqual(packet.track_event.type, type_)
      self.assertEqual(packet.track_event.track_uuid, uuid)
      self.assertEqual(packet.timestamp, ts)
      if name:
        self.assertEqual(packet.track_event.name, name)

    with self.subTest("Metadata"):
      assert_metadata(captured_packets[0], "Main", 1)
      assert_metadata(captured_packets[1], "Rollout", 2)
      assert_metadata(captured_packets[2], "Reference", 3)
      assert_metadata(captured_packets[3], "Actor", 4)

    with self.subTest("Main Track"):
      # global_step
      assert_slice(captured_packets[4], SliceBegin, 1, 0, "global_step")
      assert_slice(captured_packets[5], SliceEnd, 1, 10_000_000_000)
      # mini_batch_step
      assert_slice(captured_packets[6], SliceBegin, 1, 0, "mini_batch_step")
      assert_slice(captured_packets[7], SliceEnd, 1, 10_000_000_000)
      # rollout
      assert_slice(captured_packets[8], SliceBegin, 1, 0, "rollout")
      assert_slice(captured_packets[9], SliceEnd, 1, 4_000_000_000)
      # refer_inference
      assert_slice(
          captured_packets[10], SliceBegin, 1, 4_000_000_000, "refer_inference"
      )
      assert_slice(captured_packets[11], SliceEnd, 1, 6_000_000_000)
      # actor_training
      assert_slice(
          captured_packets[12], SliceBegin, 1, 6_000_000_000, "actor_training"
      )
      assert_slice(captured_packets[13], SliceEnd, 1, 9_000_000_000)
      # peft_train_step
      assert_slice(
          captured_packets[14], SliceBegin, 1, 6_000_000_000, "peft_train_step"
      )
      assert_slice(captured_packets[15], SliceEnd, 1, 9_000_000_000)

    with self.subTest("Rollout Track"):
      assert_slice(captured_packets[16], SliceBegin, 2, 0, "rollout")
      assert_slice(captured_packets[17], SliceEnd, 2, 4_000_000_000)

    with self.subTest("Reference Track"):
      assert_slice(
          captured_packets[18], SliceBegin, 3, 4_000_000_000, "refer_inference"
      )
      assert_slice(captured_packets[19], SliceEnd, 3, 6_000_000_000)

    with self.subTest("Actor Track"):
      # actor_training
      assert_slice(
          captured_packets[20], SliceBegin, 4, 6_000_000_000, "actor_training"
      )
      assert_slice(captured_packets[21], SliceEnd, 4, 9_000_000_000)
      # peft_train_step
      assert_slice(
          captured_packets[22], SliceBegin, 4, 6_000_000_000, "peft_train_step"
      )
      assert_slice(captured_packets[23], SliceEnd, 4, 9_000_000_000)

    mock_builder.serialize.assert_called_once()
    mock_open.return_value.write.assert_called_once()


if __name__ == "__main__":
  absltest.main()
