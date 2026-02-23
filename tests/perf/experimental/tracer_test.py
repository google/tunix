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

"""Tests for tracer."""

from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

import numpy as np
from tunix.perf.experimental import tracer


class MockDevice:

  def __init__(self, platform, device_id):
    self.platform = platform
    self.id = device_id

  def __repr__(self):
    return f"{self.platform}{self.id}"


class ControlledAsyncWait:
  """Helper to control the completion of async tasks."""

  def __init__(self):
    self.tasks = []
    self.lock = threading.Lock()

  def __call__(self, waitlist, success, failure):
    event = threading.Event()

    def target():
      event.wait()
      success()

    t = threading.Thread(target=target)
    t.start()
    with self.lock:
      self.tasks.append((event, t))
    return t

  def trigger(self, index):
    with self.lock:
      event, t = self.tasks[index]
    event.set()
    t.join()

  def trigger_all(self):
    with self.lock:
      tasks = list(self.tasks)
    for event, _ in tasks:
      event.set()
    for _, t in tasks:
      t.join()


class TracerUtilsTest(unittest.TestCase):

  def test_generate_host_timeline_id(self):
    tid = tracer.generate_host_timeline_id()
    self.assertTrue(tid.startswith("host-"))
    self.assertIn(str(threading.get_ident()), tid)

  def test_generate_device_timeline_id(self):
    self.assertEqual(tracer.generate_device_timeline_id("tpu0"), "tpu0")

    device = MockDevice("gpu", 7)
    self.assertEqual(tracer.generate_device_timeline_id(device), "gpu7")

    with self.assertRaisesRegex(ValueError, "Unsupport id type"):
      tracer.generate_device_timeline_id(123)

  def test_generate_device_timeline_ids(self):
    self.assertEqual(tracer.generate_device_timeline_ids(None), [])

    devices = ["dev1", MockDevice("tpu", 0)]
    self.assertEqual(
        tracer.generate_device_timeline_ids(devices), ["dev1", "tpu0"]
    )

    arr_devices = np.array([MockDevice("tpu", 0), MockDevice("tpu", 1)])
    self.assertEqual(
        tracer.generate_device_timeline_ids(arr_devices), ["tpu0", "tpu1"]
    )


class TimelineTest(unittest.TestCase):

  def test_basic_span_lifecycle(self):
    t = tracer.Timeline("test_tl", 100.0)
    s = t.start_span("span1", 101.0)
    self.assertEqual(s.name, "span1")
    self.assertEqual(s.begin, 101.0)
    self.assertEqual(s.id, 0)
    self.assertIsNone(s.parent_id)
    self.assertFalse(s.ended)

    t.stop_span(102.0)
    self.assertTrue(s.ended)
    self.assertEqual(s.end, 102.0)

  def test_nested_spans(self):
    t = tracer.Timeline("test_tl", 0.0)
    s1 = t.start_span("root", 1.0)
    s2 = t.start_span("child", 2.0)

    self.assertEqual(s2.parent_id, s1.id)

    t.stop_span(3.0)  # stops s2
    self.assertEqual(s2.end, 3.0)

    t.stop_span(4.0)  # stops s1
    self.assertEqual(s1.end, 4.0)

  def test_stop_span_error_cases(self):
    t = tracer.Timeline("test_tl", 0.0)
    with self.assertRaisesRegex(ValueError, "no more spans to end"):
      t.stop_span(1.0)

    s = t.start_span("s1", 2.0)
    # End before begin
    with mock.patch("absl.logging.error") as mock_log:
      t.stop_span(1.0)
      mock_log.assert_called_once()
    self.assertEqual(s.end, 1.0)

  def test_nested_timeline_with_tags_repr(self):
    born = 1000.0
    t = tracer.Timeline("test_tl", born)

    # Start root
    s_root = t.start_span("root", born + 1.0)
    s_root.add_tag("type", "root_span")

    # Start nested
    s_child = t.start_span("child", born + 2.0)
    s_child.add_tag("iter", 1)

    # Stop nested
    t.stop_span(born + 3.0)

    # Stop root
    t.stop_span(born + 4.0)

    # Check tags are stored correctly
    self.assertEqual(s_root.tags, {"type": "root_span"})
    self.assertEqual(s_child.tags, {"iter": 1})

    # Check full repr string
    expected_repr = (
        f"Timeline(test_tl, {born:.6f})\n"
        f"[0] root: 1.000000, 4.000000, tags={{'type': 'root_span'}}\n"
        f"[1] child: 2.000000, 3.000000 (parent=0), tags={{'iter': 1}}\n"
    )
    self.assertEqual(repr(t), expected_repr)


class AsyncTimelineTest(unittest.TestCase):

  def setUp(self):
    self.patcher = mock.patch("tunix.perf.experimental.tracer._async_wait")
    self.mock_async_wait = self.patcher.start()

    # Setup mock behavior for _async_wait to immediately succeed by default
    def default_wait(waitlist, success, failure):
      success()
      return mock.Mock(spec=threading.Thread)

    self.mock_async_wait.side_effect = default_wait

  def tearDown(self):
    self.patcher.stop()

  def test_span_success(self):
    t = tracer.AsyncTimeline("dev", 0.0)
    waitlist = ["thing"]

    t.span("async_op", 1.0, waitlist)

    self.mock_async_wait.assert_called_once()
    self.assertEqual(len(t.spans), 1)
    s = t.spans[0]
    self.assertEqual(s.name, "async_op")
    self.assertEqual(s.begin, 1.0)
    self.assertTrue(s.ended)  # Ended because mock calls success immediately

  def test_span_with_no_waitlist(self):
    t = tracer.AsyncTimeline("dev", 0.0)
    t.span("immediate", 1.0, [])
    self.mock_async_wait.assert_not_called()
    self.assertEqual(len(t.spans), 1)
    self.assertTrue(t.spans[0].ended)

  def test_delayed_completion(self):
    t = tracer.AsyncTimeline("dev", 0.0)

    # Capture callbacks
    callbacks = {}

    def capture_wait(waitlist, success, failure):
      callbacks["success"] = success
      callbacks["failure"] = failure
      return mock.Mock(spec=threading.Thread)

    self.mock_async_wait.side_effect = capture_wait

    t.span("delayed", 1.0, ["wait"])

    self.assertEqual(len(t.spans), 0)  # Not yet recorded

    # Simulate completion
    with mock.patch("time.perf_counter", return_value=5.0):
      callbacks["success"]()

    self.assertEqual(len(t.spans), 1)
    s = t.spans[0]
    self.assertEqual(s.end, 5.0)

  def test_failure(self):
    t = tracer.AsyncTimeline("dev", 0.0)

    def fail_wait(waitlist, success, failure):
      failure(RuntimeError("failed"))
      return mock.Mock(spec=threading.Thread)

    self.mock_async_wait.side_effect = fail_wait

    with self.assertRaisesRegex(RuntimeError, "failed"):
      t.span("failed", 1.0, ["wait"])


class PerfTracerTest(unittest.TestCase):

  def setUp(self):
    self.mock_async_wait_patcher = mock.patch(
        "tunix.perf.experimental.tracer._async_wait"
    )
    self.mock_async_wait = self.mock_async_wait_patcher.start()
    self.mock_async_wait.return_value = mock.Mock(spec=threading.Thread)

    self.mock_perf_counter_patcher = mock.patch("time.perf_counter")
    self.mock_perf_counter = self.mock_perf_counter_patcher.start()
    self.mock_perf_counter.return_value = 1000.0

  def tearDown(self):
    self.mock_async_wait_patcher.stop()
    self.mock_perf_counter_patcher.stop()

  def test_no_devices_tracer(self):
    t = tracer.PerfTracer()
    self.assertEqual(t.all_devices, [])

    with t.span("host_only"):
      pass

    timelines = t._get_timelines()
    self.assertEqual(len(timelines), 1)  # Just host
    host_id = list(timelines.keys())[0]
    self.assertTrue(host_id.startswith("host-"))
    self.assertEqual(len(timelines[host_id].spans), 1)

  def test_tracer_with_devices_and_tags(self):
    d1 = MockDevice("tpu", 0)
    t = tracer.PerfTracer(devices=[d1])
    self.assertEqual(t.all_devices, ["tpu0"])

    # Simulate async wait success immediately calling back
    def immediate_success(waitlist, success, failure):
      success()
      return mock.Mock(spec=threading.Thread)

    self.mock_async_wait.side_effect = immediate_success
    tags = {"key": "value", "step": 100}
    with t.span("op_with_tags", devices=[d1], tags=tags) as span:
      span.async_end(["future"])
    timelines = t._get_timelines()

    with self.subTest("host_timeline"):
      host_tl = timelines[t._main_thread_id]
      self.assertEqual(len(host_tl.spans), 1)
      self.assertEqual(host_tl.spans[0].name, "op_with_tags")
      self.assertEqual(host_tl.spans[0].tags, tags)

    with self.subTest("device_timeline"):
      self.assertIn("tpu0", timelines)
      dev_tl = timelines["tpu0"]
      self.assertEqual(len(dev_tl.spans), 1)
      self.assertEqual(dev_tl.spans[0].name, "op_with_tags")
      self.assertEqual(dev_tl.spans[0].tags, tags)

  def test_tracer_with_multiple_devices(self):
    d1 = MockDevice("tpu", 0)
    d2 = MockDevice("tpu", 1)
    t = tracer.PerfTracer(devices=[d1, d2])
    self.assertCountEqual(t.all_devices, ["tpu0", "tpu1"])

    def immediate_success(waitlist, success, failure):
      success()
      return mock.Mock(spec=threading.Thread)

    self.mock_async_wait.side_effect = immediate_success

    with t.span("multi_dev_op", devices=[d1, d2]) as span:
      span.async_end(["future"])

    timelines = t._get_timelines()
    self.assertIn("tpu0", timelines)
    self.assertIn("tpu1", timelines)

    tl1 = timelines["tpu0"]
    tl2 = timelines["tpu1"]
    host_tl = timelines[t._main_thread_id]

    # Verify all timelines share the same 'born' time
    self.assertEqual(tl1.born, tl2.born)
    self.assertEqual(tl1.born, host_tl.born)

    self.assertEqual(len(tl1.spans), 1)
    self.assertEqual(len(tl2.spans), 1)
    self.assertEqual(tl1.spans[0].name, "multi_dev_op")
    self.assertEqual(tl2.spans[0].name, "multi_dev_op")

  def test_nested_host_spans_with_devices(self):
    d1 = "gpu0"
    t = tracer.PerfTracer(devices=[d1])

    def immediate_success(waitlist, success, failure):
      success()
      return mock.Mock()

    self.mock_async_wait.side_effect = immediate_success

    with t.span("outer", devices=[d1]) as s1:
      s1.async_end(["w1"])
      with t.span("inner", devices=[d1]) as s2:
        s2.async_end(["w2"])

    timelines = t._get_timelines()
    host_tl = timelines[t._main_thread_id]
    dev_tl = timelines[d1]

    self.assertEqual(len(host_tl.spans), 2)
    self.assertEqual(len(dev_tl.spans), 2)

    # Verify IDs match implicit sequential ordering for device.
    # Because 'inner' context manager exits first, it is recorded first (ID=0).
    self.assertEqual(dev_tl.spans[0].name, "inner")
    self.assertEqual(dev_tl.spans[1].name, "outer")

  def test_interleaved_threads(self):
    # Test multiple threads using the same tracer
    t = tracer.PerfTracer()

    def worker(name):
      with t.span(name):
        time.sleep(0.01)

    t1 = threading.Thread(target=worker, args=("w1",))
    t2 = threading.Thread(target=worker, args=("w2",))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    timelines = t._get_timelines()
    # Main thread + 2 worker threads
    self.assertEqual(len(timelines), 3)

    # Verify each has 1 span
    span_counts = [len(tl.spans) for tl in timelines.values()]
    # Main thread created in init has 0 spans, workers have 1 each
    self.assertEqual(sorted(span_counts), [0, 1, 1])

  def test_synchronize(self):
    t = tracer.PerfTracer()
    with mock.patch(
        "tunix.perf.experimental.tracer._synchronize_devices"
    ) as mock_sync:
      t.synchronize()
      mock_sync.assert_called_once()

  def test_synchronize_slow_fast_devices(self):
    """Tests synchronization with devices having different latencies."""
    t = tracer.PerfTracer()

    with mock.patch("tunix.perf.experimental.tracer.jax") as mock_jax:
      mock_jax.devices.return_value = ["fast_dev", "slow_dev"]

      # Mock device_put returning an object with block_until_ready
      mock_future_fast = mock.Mock()
      mock_future_slow = mock.Mock()

      # Fast one returns immediately
      mock_future_fast.block_until_ready.return_value = None

      # Slow one sleeps
      def slow_block():
        time.sleep(0.1)

      mock_future_slow.block_until_ready.side_effect = slow_block

      # device_put returns fast then slow (or based on device arg)
      def side_effect_put(arr, device):
        if device == "fast_dev":
          return mock_future_fast
        else:
          return mock_future_slow

      mock_jax.device_put.side_effect = side_effect_put

      # Override perf_counter to simulate time passing for the assertion
      self.mock_perf_counter.side_effect = [1000.0, 1000.2]
      start_time = time.perf_counter()
      t.synchronize()
      end_time = time.perf_counter()

      # Ensure both were called
      mock_future_fast.block_until_ready.assert_called_once()
      mock_future_slow.block_until_ready.assert_called_once()

      # Ensure we waited at least for the slow one
      self.assertGreaterEqual(end_time - start_time, 0.1)

  def test_synchronize_waits_for_pending_spans(self):
    d0 = MockDevice("tpu", 0)
    t = tracer.PerfTracer(devices=[d0])

    # We need to mock _synchronize_devices to avoid jax calls,
    # as we are focusing on wait_pending_spans here.
    with mock.patch("tunix.perf.experimental.tracer._synchronize_devices"):

      # Use ControlledAsyncWait to simulate a pending span
      controller = ControlledAsyncWait()
      with mock.patch(
          "tunix.perf.experimental.tracer._async_wait", side_effect=controller
      ):
        with t.span("async_op", devices=[d0]) as s:
          s.async_end(["waitlist"])

        self.assertEqual(len(controller.tasks), 1)

        # Run synchronize in a thread
        sync_thread = threading.Thread(target=t.synchronize)
        sync_thread.start()

        # Wait a bit to ensure it blocks
        time.sleep(0.1)
        self.assertTrue(sync_thread.is_alive())

        # Trigger completion
        controller.trigger(0)

        # Should finish now
        sync_thread.join(timeout=1.0)
        self.assertFalse(sync_thread.is_alive())

  def test_export(self):
    export_mock = mock.Mock(return_value={"a": 1})
    t = tracer.PerfTracer(export_fn=export_mock)

    res = t.export()
    self.assertEqual(res, {"a": 1})
    export_mock.assert_called_once()

    # Check default export
    t2 = tracer.PerfTracer()
    self.assertEqual(t2.export(), {})


class AdvancedPerfTracerTest(unittest.TestCase):

  def setUp(self):
    self.patcher = mock.patch("tunix.perf.experimental.tracer._async_wait")
    self.mock_async_wait = self.patcher.start()
    self.controller = ControlledAsyncWait()
    self.mock_async_wait.side_effect = self.controller
    self.device = MockDevice("tpu", 0)

    # Mock JAX to prevent real calls in synchronize()
    self.jax_patcher = mock.patch("tunix.perf.experimental.tracer.jax")
    self.mock_jax = self.jax_patcher.start()
    self.mock_jax.devices.return_value = ["tpu0"]
    self.mock_jax.device_put.return_value.block_until_ready.return_value = None

  def tearDown(self):
    self.patcher.stop()
    self.jax_patcher.stop()

  def test_interleaved_device_completion(self):
    """Tests that device spans can complete in a different order than started."""
    t = tracer.PerfTracer(devices=[self.device])

    # 1. Start  Async Span A
    with t.span("A", devices=[self.device]) as s:
      s.async_end(["wa"])

    # 2. Start Async Span B
    with t.span("B", devices=[self.device]) as s:
      s.async_end(["wb"])

    self.assertEqual(len(self.controller.tasks), 2)

    # 3. Complete Span B first
    with mock.patch("time.perf_counter", return_value=100.0):
      self.controller.trigger(1)  # Index 1 is B

    dev_timeline = t._get_timelines()["tpu0"]
    self.assertEqual(len(dev_timeline.spans), 1)
    # Verify B is complete and A (ID 0) is NOT complete (still pending)
    self.assertIn(1, dev_timeline.spans)
    self.assertNotIn(0, dev_timeline.spans)
    self.assertEqual(dev_timeline.spans[1].name, "B")
    self.assertEqual(dev_timeline.spans[1].end, 100.0)

    # 4. Complete Span A
    with mock.patch("time.perf_counter", return_value=101.0):
      self.controller.trigger(0)  # Index 0 is A

    self.assertEqual(len(dev_timeline.spans), 2)
    self.assertEqual(dev_timeline.spans[0].name, "A")
    self.assertEqual(dev_timeline.spans[0].end, 101.0)

    # Verify IDs (sequential based on start order)
    self.assertEqual(dev_timeline.spans[0].id, 0)
    self.assertEqual(dev_timeline.spans[1].id, 1)

  def test_host_nesting_and_device_flatness(self):
    """Tests correct nesting on host vs flat structure on device."""
    t = tracer.PerfTracer(devices=[self.device])

    # Host: Root -> Child
    # Device: Root, Child (flat, ordered by start time)

    with t.span("Root", devices=[self.device]) as s_root:
      with t.span("Child", devices=[self.device]) as s_child:
        s_child.async_end(["w_child"])
      s_root.async_end(["w_root"])

    timelines = t._get_timelines()
    host_tl = timelines[t._main_thread_id]
    dev_tl = timelines["tpu0"]

    # Check Span counts & waitlist
    self.assertEqual(len(host_tl.spans), 2)
    self.assertEqual(len(dev_tl.spans), 0)
    self.assertEqual(len(self.controller.tasks), 2)

    self.controller.trigger_all()
    t.synchronize()

    # Check Host Nesting
    host_root = host_tl.spans[0]
    host_child = host_tl.spans[1]
    self.assertEqual(host_root.name, "Root")
    self.assertEqual(host_child.name, "Child")
    self.assertEqual(host_child.parent_id, host_root.id)

    # TODO (noghabi): is this the behavior we want? how can we post process this?
    # Check Device Flatness
    self.assertEqual(len(dev_tl.spans), 2)

    # Submission order:
    # 1. Enter Root (Host Start)
    # 2. Enter Child (Host Start)
    # 3. Exit Child (Host End, Device Submit Child)
    # 4. Exit Root (Host End, Device Submit Root)

    d_first = dev_tl.spans[0]
    d_second = dev_tl.spans[1]

    self.assertEqual(d_first.name, "Child")
    self.assertEqual(d_second.name, "Root")

    # Verify no parent relationship on device timeline
    self.assertIsNone(d_first.parent_id)
    self.assertIsNone(d_second.parent_id)

  def test_concurrent_stress(self):
    """Tests correctness under concurrent submissions via PerfTracer."""
    t = tracer.PerfTracer(devices=[self.device])
    num_threads = 10
    spans_per_thread = 20

    def worker():
      for _ in range(spans_per_thread):
        with t.span(f"span", devices=[self.device]) as s:
          s.async_end(["w"])

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for th in threads:
      th.start()
    for th in threads:
      th.join()

    # All tasks scheduled
    total_spans = num_threads * spans_per_thread
    self.assertEqual(len(self.controller.tasks), total_spans)

    # Trigger all to finish
    self.controller.trigger_all()
    t.synchronize()

    dev_tl = t._get_timelines()["tpu0"]
    self.assertEqual(len(dev_tl.spans), total_spans)

    # Check IDs are sequential 0 to N-1
    ids = sorted(dev_tl.spans.keys())
    self.assertEqual(ids, list(range(total_spans)))


class PrivateHelpersTest(unittest.TestCase):

  def test_async_wait_implementation(self):
    # Test the real _async_wait logic (threading and jax calls) with mocks
    mock_success = mock.Mock()
    mock_failure = mock.Mock()

    # Case 1: Success
    with mock.patch("jax.block_until_ready") as mock_block:
      t = tracer._async_wait("waitlist", mock_success, mock_failure)
      t.join()

      mock_block.assert_called_with("waitlist")
      mock_success.assert_called_once()
      mock_failure.assert_not_called()

    mock_success.reset_mock()
    mock_failure.reset_mock()

    # Case 2: Failure
    with mock.patch("jax.block_until_ready") as mock_block:
      mock_block.side_effect = ValueError("fail")
      t = tracer._async_wait("waitlist", mock_success, mock_failure)
      t.join()

      mock_success.assert_not_called()
      mock_failure.assert_called_once()
      self.assertIsInstance(mock_failure.call_args[0][0], ValueError)

  def test_synchronize_devices(self):
    with mock.patch("tunix.perf.experimental.tracer.jax") as mock_jax:
      mock_jax.devices.return_value = ["d1"]
      mock_put = mock_jax.device_put
      mock_put.return_value.block_until_ready = mock.Mock()

      tracer._synchronize_devices()

      mock_put.assert_called()


if __name__ == "__main__":
  unittest.main()
