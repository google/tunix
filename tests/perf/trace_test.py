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
from tunix.perf import trace

BaseTimeline = trace.BaseTimeline
DeviceTimeline = trace.DeviceTimeline
HostTimeline = trace.HostTimeline
NoopTracer = trace.NoopTracer
PerfTracer = trace.PerfTracer
patch = mock.patch
Mock = mock.Mock


def mock_array():
  class JaxArray:

    def block_until_ready(self):
      pass

  return Mock(spec=JaxArray)


def get_timelines(tracer: PerfTracer) -> list[BaseTimeline]:
  return list(tracer._get_timelines().values())


def make_host_timeline(
    born: float,
    intervals: list[tuple[float, float]] | None = None,
    labels: list[str] | None = None,
    epochs: list[int] | None = None,
) -> HostTimeline:
  timeline = HostTimeline("host", born)
  timeline.intervals = intervals if intervals else []
  timeline.labels = labels if labels else []
  timeline.epochs = epochs if epochs else []
  return timeline


def make_device_timeline(
    id: str,
    born: float,
    intervals: list[tuple[float, float]] | None = None,
    labels: list[str] | None = None,
    epochs: list[int] | None = None,
) -> DeviceTimeline:
  timeline = DeviceTimeline(id, born)
  timeline.intervals = intervals if intervals else []
  timeline.labels = labels if labels else []
  timeline.epochs = epochs if epochs else []
  return timeline


class TracerTest(parameterized.TestCase):

  @patch("time.perf_counter")
  def test_host_ok(self, mock_perf_counter):
    mock_perf_counter.side_effect = [0.0, 2.0, 3.0]

    tracer = PerfTracer()
    with tracer.interval("x"):
      pass
    tracer.end_epoch()

    tracer.synchronize()

    self.assertListEqual(
        get_timelines(tracer),
        [make_host_timeline(0.0, [(2.0, 3.0)], ["x"], [1])],
    )

  @patch("time.perf_counter")
  def test_device_ok(self, mock_perf_counter):
    mock_perf_counter.side_effect = [0.0, 2.0, 3.0, 5.0]
    waitlist = mock_array()

    tracer = PerfTracer()

    with tracer.interval("x", devices=["tpu0"]) as interval:
      interval.device_end(waitlist)
    tracer.end_epoch()

    tracer.synchronize()

    waitlist.block_until_ready.assert_called_once()
    self.assertListEqual(
        get_timelines(tracer),
        [
            make_host_timeline(0.0, [(2.0, 3.0)], ["x"], [1]),
            make_device_timeline("tpu0", 0.0, [(2.0, 5.0)], ["x"], [1]),
        ],
    )

  @patch("time.perf_counter")
  def test_device_multi_ok(self, mock_perf_counter):
    mock_perf_counter.side_effect = [0.0, 2.0, 3.0, 5.0]
    waitlist = mock_array()

    tracer = PerfTracer(devices=["tpu0", "tpu1"])

    with tracer.interval("int", devices=["tpu0"]) as interval:
      interval.device_end(waitlist)
    tracer.end_epoch()

    tracer.synchronize()

    waitlist.block_until_ready.assert_called_once()
    self.assertListEqual(
        get_timelines(tracer),
        [
            make_host_timeline(0.0, [(2.0, 3.0)], ["int"], [1]),
            make_device_timeline("tpu0", 0.0, [(2.0, 5.0)], ["int"], [1]),
            make_device_timeline("tpu1", 0.0, [], [], [0]),
        ],
    )

  @patch("time.perf_counter")
  def test_device_interval_begin_algorithm(self, mock_perf_counter):
    mock_perf_counter.side_effect = [0.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0]
    waitlist = mock_array()

    tracer = PerfTracer()

    with tracer.interval("step1", devices=["tpu0"]) as interval:
      interval.device_end(waitlist)

    with tracer.interval("step2", devices=["tpu0"]) as interval:
      interval.device_end(waitlist)

    tracer.end_epoch()
    tracer.synchronize()

    self.assertEqual(waitlist.block_until_ready.call_count, 2)
    # "tpu0:step2:begin" is equal to
    # max("host:step2:begin" 4.0, "tpu0:step1:end" 5.0)
    self.assertListEqual(
        get_timelines(tracer),
        [
            make_host_timeline(
                0.0, [(2.0, 3.0), (4.0, 6.0)], ["step1", "step2"], [2]
            ),
            make_device_timeline(
                "tpu0", 0.0, [(2.0, 5.0), (5.0, 7.0)], ["step1", "step2"], [2]
            ),
        ],
    )

  @patch("time.perf_counter")
  def test_device_all_matcher(self, mock_perf_counter):
    mock_perf_counter.side_effect = [0.0, 2.0, 3.0, 5.0, 5.0]
    waitlist = mock_array()

    tracer = PerfTracer(devices=["tpu0", "tpu1"])

    with tracer.interval("x", devices=tracer.all) as interval:
      interval.device_end(waitlist)

    tracer.synchronize()

    self.assertEqual(waitlist.block_until_ready.call_count, 2)
    self.assertListEqual(
        get_timelines(tracer),
        [
            make_host_timeline(0.0, [(2.0, 3.0)], ["x"], []),
            make_device_timeline("tpu0", 0.0, [(2.0, 5.0)], ["x"], []),
            make_device_timeline("tpu1", 0.0, [(2.0, 5.0)], ["x"], []),
        ],
    )

  def test_nested_interval_raise_exception(self):
    tracer = PerfTracer()
    with tracer.interval("outer"):
      with self.assertRaises(ValueError):
        with tracer.interval("inner"):
          pass

  @patch("time.perf_counter")
  def test_end_epoch_is_idempotent(self, mock_perf_counter):
    mock_perf_counter.side_effect = [0.0, 2.0, 3.0, 5.0, 7.0]

    tracer = PerfTracer()
    with tracer.interval("x"):
      pass
    tracer.end_epoch()
    tracer.end_epoch()
    with tracer.interval("y"):
      pass
    tracer.end_epoch()
    tracer.end_epoch()

    tracer.synchronize()

    self.assertListEqual(
        get_timelines(tracer),
        [make_host_timeline(0.0, [(2.0, 3.0), (5.0, 7.0)], ["x", "y"], [1, 2])],
    )

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
