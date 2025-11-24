# Copyright 2025 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.perf import span
from tunix.perf import trace

Timeline = trace.Timeline
DeviceTimeline = trace.DeviceTimeline
ThreadTimeline = trace.ThreadTimeline
NoopTracer = trace.NoopTracer
PerfTracer = trace.PerfTracer
patch = mock.patch
Mock = mock.Mock


def mock_array():
  class JaxArray:

    def block_until_ready(self):
      pass

  return Mock(spec=JaxArray)


def timeline_tostring(timeline: Timeline) -> str:
  # ignore thread id diffs.
  timeline_id = (
      timeline.id if not timeline.id.startswith("thread-") else "thread"
  )
  return (
      f"{timeline_id}:"
      + f" born={timeline.born:.6f} stack={timeline._stack_debug()}\n"
      + span.span_group_tostring(timeline.root, timeline.born)
  )


def timelines_tostring(tracer: PerfTracer) -> list[str]:
  return [
      timeline_tostring(timeline)
      for timeline in tracer._get_timelines().values()
  ]


class TracerTest(parameterized.TestCase):

  @patch("time.perf_counter")
  def test_host_ok(self, mock_perf_counter):
    mock_perf_counter.side_effect = [0.0, 2.0, 3.0]

    tracer = PerfTracer()
    with tracer.span("x"):
      pass
    tracer.synchronize()

    self.assertListEqual(
        timelines_tostring(tracer),
        [
            "thread: born=0.000000 stack=root\n"
            "- root (0.000000, inf)\n"
            "  - x (2.000000, 3.000000)\n"
        ],
    )

  @patch("time.perf_counter")
  def test_device_ok(self, mock_perf_counter):
    mock_perf_counter.side_effect = [0.0, 2.0, 3.0, 5.0]
    waitlist = mock_array()

    tracer = PerfTracer()

    with tracer.span("x", devices=["tpu0"]) as span:
      span.device_end(waitlist)

    tracer.synchronize()

    waitlist.block_until_ready.assert_called_once()
    self.assertListEqual(
        timelines_tostring(tracer),
        [
            (
                "thread: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - x (2.000000, 3.000000)\n"
            ),
            (
                "tpu0: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - x (2.000000, 5.000000)\n"
            ),
        ],
    )

  @patch("time.perf_counter")
  def test_device_multi_ok(self, mock_perf_counter):
    mock_perf_counter.side_effect = [0.0, 2.0, 3.0, 5.0]
    waitlist = mock_array()

    tracer = PerfTracer(devices=["tpu0", "tpu1"])

    with tracer.span("int", devices=["tpu0"]) as span:
      span.device_end(waitlist)

    tracer.synchronize()

    waitlist.block_until_ready.assert_called_once()
    self.assertListEqual(
        timelines_tostring(tracer),
        [
            (
                "thread: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - int (2.000000, 3.000000)\n"
            ),
            (
                "tpu0: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - int (2.000000, 5.000000)\n"
            ),
            "tpu1: born=0.000000 stack=root\n- root (0.000000, inf)\n",
        ],
    )

  @patch("time.perf_counter")
  def test_device_span_begin_algorithm(self, mock_perf_counter):
    mock_perf_counter.side_effect = [0.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0]
    waitlist = mock_array()

    tracer = PerfTracer()

    with tracer.span("step1", devices=["tpu0"]) as span:
      span.device_end(waitlist)
    with tracer.span("step2", devices=["tpu0"]) as span:
      span.device_end(waitlist)

    tracer.synchronize()

    self.assertEqual(waitlist.block_until_ready.call_count, 2)
    self.assertListEqual(
        timelines_tostring(tracer),
        [
            (
                "thread: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - step1 (2.000000, 3.000000)\n"
                "  - step2 (4.000000, 6.000000)\n"
            ),
            (
                "tpu0: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - step1 (2.000000, 5.000000)\n"
                "  - step2 (5.000000, 7.000000)\n"  # begin is 5, not 4
            ),
        ],
    )

  @patch("time.perf_counter")
  def test_device_all_matcher(self, mock_perf_counter):
    mock_perf_counter.side_effect = [0.0, 2.0, 3.0, 5.0, 5.0]
    waitlist = mock_array()

    tracer = PerfTracer(devices=["tpu0", "tpu1"])

    with tracer.span("x", devices=tracer.all_devices) as span:
      span.device_end(waitlist)

    tracer.synchronize()

    self.assertEqual(waitlist.block_until_ready.call_count, 2)
    self.assertListEqual(
        timelines_tostring(tracer),
        [
            (
                "thread: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - x (2.000000, 3.000000)\n"
            ),
            (
                "tpu0: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - x (2.000000, 5.000000)\n"
            ),
            (
                "tpu1: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - x (2.000000, 5.000000)\n"
            ),
        ],
    )

  @patch("time.perf_counter")
  def test_span_group_ok(self, mock_perf_counter):
    mock_perf_counter.side_effect = [
        0.0,
        1.0,
        2.0,
        3.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
    ]
    waitlist = mock_array()

    tracer = PerfTracer(["tpu0", "tpu1"])

    with tracer.span_group("outer_loop"):
      with tracer.span("step1", devices=["tpu0"]) as span:
        span.device_end(waitlist)
      with tracer.span_group("inner_loop"):
        with tracer.span("step2", devices=["tpu1"]) as span:
          span.device_end(waitlist)
      with tracer.span("step3"):
        pass

    tracer.synchronize()

    self.assertEqual(waitlist.block_until_ready.call_count, 2)
    self.assertListEqual(
        timelines_tostring(tracer),
        [
            (
                "thread: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - outer_loop (1.000000, 13.000000)\n"
                "    - step1 (2.000000, 3.000000)\n"
                "    - inner_loop (6.000000, 10.000000)\n"
                "      - step2 (7.000000, 8.000000)\n"
                "    - step3 (11.000000, 12.000000)\n"
            ),
            (
                "tpu0: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - outer_loop (1.000000, 13.000000)\n"
                "    - step1 (2.000000, 5.000000)\n"
                "    - inner_loop (6.000000, 10.000000)\n"
            ),
            (
                "tpu1: born=0.000000 stack=root\n"
                "- root (0.000000, inf)\n"
                "  - outer_loop (1.000000, 13.000000)\n"
                "    - inner_loop (6.000000, 10.000000)\n"
                "      - step2 (7.000000, 9.000000)\n"
            ),
        ],
    )

  def test_nested_span_raise_exception(self):
    tracer = PerfTracer()
    with tracer.span("outer"):
      with self.assertRaises(ValueError):
        with tracer.span("inner"):
          pass

  def test_noop_interface_is_same(self):
    noop_public_attrs = [
        name for name in dir(NoopTracer()) if not name.startswith("_")
    ]
    perf_public_attrs = [
        name for name in dir(PerfTracer()) if not name.startswith("_")
    ]
    self.assertEqual(noop_public_attrs, perf_public_attrs)


if __name__ == "__main__":
  absltest.main()
