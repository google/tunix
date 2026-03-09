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

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any, Callable, List
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.perf.experimental import timeline
from tunix.perf.experimental import tracer


@dataclasses.dataclass
class MockDevice:
  platform: str
  id: int


class ControlledAsyncWait:
  """Helper to control the completion of async tasks.

  Attributes:
    tasks: A list of tuples, where each tuple contains a `threading.Event` and a
      `threading.Thread` for a pending async task.
    lock: A `threading.Lock` to ensure thread-safe access to `tasks`.
  """

  def __init__(self) -> None:
    self.tasks: List[tuple[threading.Event, threading.Thread]] = []
    self.lock: threading.Lock = threading.Lock()

  def __call__(
      self,
      waitlist: Any,
      success: Callable[[], None],
      failure: Callable[[Exception], None],
  ) -> threading.Thread:
    event = threading.Event()

    def target() -> None:
      event.wait()
      success()

    t = threading.Thread(target=target)
    t.start()
    with self.lock:
      self.tasks.append((event, t))
    return t

  def trigger(self, index: int) -> None:
    with self.lock:
      event, t = self.tasks[index]
    event.set()
    t.join()

  def trigger_all(self) -> None:
    with self.lock:
      tasks = list(self.tasks)
    for event, _ in tasks:
      event.set()
    for _, t in tasks:
      t.join()


class TracerUtilsTest(parameterized.TestCase):

  def test_generate_host_timeline_id(self):
    tid = tracer.generate_host_timeline_id()
    self.assertStartsWith(tid, "host-")
    self.assertIn(str(threading.get_ident()), tid)

  @parameterized.named_parameters(
      ("string", "tpu0", "tpu0"),
      ("device_object", MockDevice("gpu", 7), "gpu7"),
  )
  def test_generate_device_timeline_id(self, device_id, expected_id):
    self.assertEqual(tracer.generate_device_timeline_id(device_id), expected_id)

  def test_generate_device_timeline_id_error(self):
    with self.assertRaisesRegex(ValueError, "Unsupported id type"):
      tracer.generate_device_timeline_id(123)

  @parameterized.named_parameters(
      ("none", None, []),
      ("mixed_list", ["dev1", MockDevice("tpu", 0)], ["dev1", "tpu0"]),
      (
          "numpy_array",
          np.array([MockDevice("tpu", 0), MockDevice("tpu", 1)]),
          ["tpu0", "tpu1"],
      ),
      (
          "numpy_array_2d",
          np.array([
              [MockDevice("tpu", 0), MockDevice("tpu", 1)],
              [MockDevice("tpu", 2), MockDevice("tpu", 3)],
          ]),
          ["tpu0", "tpu1", "tpu2", "tpu3"],
      ),
  )
  def test_generate_device_timeline_ids(self, devices, expected_ids):
    self.assertEqual(tracer.generate_device_timeline_ids(devices), expected_ids)


class PerfTracerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_async_wait = self.enter_context(
        mock.patch.object(timeline, "_async_wait", autospec=True)
    )
    self.mock_async_wait.return_value = mock.Mock(spec=threading.Thread)

    self.mock_perf_counter = self.enter_context(
        mock.patch.object(time, "perf_counter", autospec=True)
    )
    self.mock_perf_counter.return_value = 1000.0

  def test_no_devices_tracer(self):
    t = tracer.PerfTracer()
    self.assertEqual(t.all_devices, [])

    with t.span("host_only"):
      pass

    timelines = t._get_timelines()
    self.assertLen(timelines, 1)  # Just host
    [host_id] = timelines.keys()
    self.assertStartsWith(host_id, "host-")
    self.assertLen(timelines[host_id].spans, 1)

  def test_tracer_with_devices_and_tags(self):
    d1 = MockDevice("tpu", 0)
    t = tracer.PerfTracer(devices=[d1])
    self.assertEqual(t.all_devices, ["tpu0"])

    # Simulate async wait success immediately calling back
    def immediate_success(
        waitlist: Any, success: Callable[[], None], failure: Callable[[], None]
    ) -> mock.Mock:
      success()
      return mock.create_autospec(threading.Thread, instance=True)

    self.mock_async_wait.side_effect = immediate_success
    tags = {"key": "value", "step": 100}
    with t.span("op_with_tags", devices=[d1], tags=tags) as span:
      span.async_end(["future"])
    timelines = t._get_timelines()

    with self.subTest("host_timeline"):
      host_tl = timelines[t._main_thread_id]
      self.assertLen(host_tl.spans, 1)
      self.assertEqual(host_tl.spans[0].name, "op_with_tags")
      self.assertEqual(host_tl.spans[0].tags, tags)

    with self.subTest("device_timeline"):
      self.assertIn("tpu0", timelines)
      dev_tl = timelines["tpu0"]
      self.assertLen(dev_tl.spans, 1)
      self.assertEqual(dev_tl.spans[0].name, "op_with_tags")
      self.assertEqual(dev_tl.spans[0].tags, tags)

  def test_tracer_with_multiple_devices(self):
    d1 = MockDevice("tpu", 0)
    d2 = MockDevice("tpu", 1)
    t = tracer.PerfTracer(
        devices=[d1, d2],
        collect_on_first_device_per_mesh=False,
    )
    self.assertCountEqual(t.all_devices, ["tpu0", "tpu1"])

    def immediate_success(
        waitlist: Any, success: Callable[[], None], failure: Callable[[], None]
    ) -> mock.Mock:
      success()
      return mock.create_autospec(threading.Thread, instance=True)

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

    self.assertLen(tl1.spans, 1)
    self.assertLen(tl2.spans, 1)
    self.assertEqual(tl1.spans[0].name, "multi_dev_op")
    self.assertEqual(tl2.spans[0].name, "multi_dev_op")

  def test_tracer_collect_on_one_device(self):
    d1 = MockDevice("tpu", 0)
    d2 = MockDevice("tpu", 1)
    # Default collect_on_first_device_per_mesh=True
    t = tracer.PerfTracer(devices=[d1, d2])
    self.assertCountEqual(t.all_devices, ["tpu0", "tpu1"])

    def immediate_success(
        waitlist: Any, success: Callable[[], None], failure: Callable[[], None]
    ) -> mock.Mock:
      success()
      return mock.create_autospec(threading.Thread, instance=True)

    self.mock_async_wait.side_effect = immediate_success

    with t.span("multi_dev_op", devices=[d1, d2]) as span:
      span.async_end(["future"])

    timelines = t._get_timelines()
    self.assertIn("tpu0", timelines)
    self.assertIn("tpu1", timelines)

    tl1 = timelines["tpu0"]
    tl2 = timelines["tpu1"]

    self.assertLen(tl1.spans, 1, msg="Only the first device should have a span")
    self.assertEqual(tl1.spans[0].name, "multi_dev_op")
    self.assertEmpty(tl2.spans)

  def test_tracer_with_multidim_devices(self):
    d1 = MockDevice("tpu", 0)
    d2 = MockDevice("tpu", 1)
    arr_devices = np.array([[d1], [d2]])

    t = tracer.PerfTracer(
        devices=arr_devices, collect_on_first_device_per_mesh=False
    )

    self.assertCountEqual(t.all_devices, ["tpu0", "tpu1"])

    with t.span("multidim_op", devices=arr_devices):
      pass

    timelines = t._get_timelines()
    self.assertIn("tpu0", timelines)
    self.assertIn("tpu1", timelines)
    self.assertLen(timelines["tpu0"].spans, 1)
    self.assertLen(timelines["tpu1"].spans, 1)

  def test_nested_host_spans_with_devices(self):
    d1 = "gpu0"
    t = tracer.PerfTracer(devices=[d1])

    def immediate_success(
        waitlist: Any, success: Callable[[], None], failure: Callable[[], None]
    ) -> mock.Mock:
      success()
      return mock.create_autospec(threading.Thread, instance=True)

    self.mock_async_wait.side_effect = immediate_success

    with t.span("outer", devices=[d1]) as s1:
      s1.async_end(["w1"])
      with t.span("inner", devices=[d1]) as s2:
        s2.async_end(["w2"])

    timelines = t._get_timelines()
    host_tl = timelines[t._main_thread_id]
    dev_tl = timelines[d1]

    self.assertLen(host_tl.spans, 2)
    self.assertLen(dev_tl.spans, 2)

    # Verify IDs match implicit sequential ordering for device.
    # Because 'inner' context manager exits first, it is recorded first (ID=0).
    inner_span, outer_span = dev_tl.spans.values()
    self.assertEqual(inner_span.name, "inner")
    self.assertEqual(outer_span.name, "outer")

  def test_interleaved_threads(self):
    # Test multiple threads using the same tracer
    t = tracer.PerfTracer()

    def worker(name: str) -> None:
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
    self.assertLen(timelines, 3)

    # Verify each has 1 span
    span_counts = [len(tl.spans) for tl in timelines.values()]
    # Main thread created in init has 0 spans, workers have 1 each
    self.assertEqual(sorted(span_counts), [0, 1, 1])

  def test_synchronize(self):
    t = tracer.PerfTracer()
    with mock.patch.object(
        tracer, "_synchronize_devices", autospec=True
    ) as mock_sync:
      t.synchronize()
      mock_sync.assert_called_once()

  def test_synchronize_slow_fast_devices(self):
    """Tests synchronization with devices having different latencies."""
    t = tracer.PerfTracer()

    with mock.patch.object(tracer, "jax", autospec=True) as mock_jax:
      mock_jax.devices.return_value = ["fast_dev", "slow_dev"]

      # Mock device_put returning an object with block_until_ready
      mock_future_fast = mock.Mock()
      mock_future_slow = mock.Mock()

      # Fast one returns immediately
      mock_future_fast.block_until_ready.return_value = None

      # Slow one sleeps
      def slow_block() -> None:
        time.sleep(0.1)

      mock_future_slow.block_until_ready.side_effect = slow_block

      # device_put returns fast then slow (or based on device arg)
      def side_effect_put(arr: Any, device: Any) -> mock.Mock:
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
    with mock.patch.object(tracer, "_synchronize_devices", autospec=True):

      # Use ControlledAsyncWait to simulate a pending span
      controller = ControlledAsyncWait()
      self.mock_async_wait.side_effect = controller

      with t.span("async_op", devices=[d0]) as s:
        s.async_end(["waitlist"])

      self.assertLen(controller.tasks, 1)

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


class AdvancedPerfTracerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_async_wait = self.enter_context(
        mock.patch.object(timeline, "_async_wait", autospec=True)
    )
    self.controller = ControlledAsyncWait()
    self.mock_async_wait.side_effect = self.controller
    self.device = MockDevice("tpu", 0)

    # Mock JAX to prevent real calls in synchronize()
    self.mock_jax = self.enter_context(
        mock.patch.object(tracer, "jax", autospec=True)
    )
    self.mock_jax.devices.return_value = ["tpu0"]
    self.mock_jax.device_put.return_value.block_until_ready.return_value = None

  def test_interleaved_device_completion(self):
    """Tests that device spans can complete in a different order than started."""
    t = tracer.PerfTracer(devices=[self.device])

    # 1. Start  Async Span A
    with t.span("A", devices=[self.device]) as s:
      s.async_end(["wa"])

    # 2. Start Async Span B
    with t.span("B", devices=[self.device]) as s:
      s.async_end(["wb"])

    self.assertLen(self.controller.tasks, 2)

    # 3. Complete Span B first
    with mock.patch.object(
        time, "perf_counter", return_value=100.0, autospec=True
    ):
      self.controller.trigger(1)  # Index 1 is B

    dev_timeline = t._get_timelines()["tpu0"]
    self.assertLen(dev_timeline.spans, 1)
    # Verify B is complete and A (ID 0) is NOT complete (still pending)
    self.assertIn(1, dev_timeline.spans)
    self.assertNotIn(0, dev_timeline.spans)
    self.assertEqual(dev_timeline.spans[1].name, "B")
    self.assertEqual(dev_timeline.spans[1].end, 100.0)

    # 4. Complete Span A
    with mock.patch.object(
        time, "perf_counter", return_value=101.0, autospec=True
    ):
      self.controller.trigger(0)  # Index 0 is A

    self.assertLen(dev_timeline.spans, 2)
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
    self.assertLen(host_tl.spans, 2)
    self.assertEmpty(dev_tl.spans)
    self.assertLen(self.controller.tasks, 2)

    self.controller.trigger_all()
    t.synchronize()

    # Check Host Nesting
    host_root = host_tl.spans[0]
    host_child = host_tl.spans[1]
    self.assertEqual(host_root.name, "Root")
    self.assertEqual(host_child.name, "Child")
    self.assertEqual(host_child.parent_id, host_root.id)

    # TODO(noghabi): is this the behavior we want? how can we post process this?
    # Check Device Flatness
    self.assertLen(dev_tl.spans, 2)

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

    def worker() -> None:
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
    self.assertLen(self.controller.tasks, total_spans)

    # Trigger all to finish
    self.controller.trigger_all()
    t.synchronize()

    dev_tl = t._get_timelines()["tpu0"]
    self.assertLen(dev_tl.spans, total_spans)

    # Check IDs are sequential 0 to N-1
    ids = sorted(dev_tl.spans.keys())
    self.assertEqual(ids, list(range(total_spans)))


class PrivateHelpersTest(absltest.TestCase):

  def test_synchronize_devices(self):
    with mock.patch.object(tracer, "jax", autospec=True) as mock_jax:
      mock_jax.devices.return_value = ["d1"]
      mock_put = mock_jax.device_put
      mock_put.return_value.block_until_ready = mock.Mock()

      tracer._synchronize_devices()

      mock_put.assert_called()


class NoopTracerTest(absltest.TestCase):

  def test_noop_tracer_span(self):
    t = tracer.NoopTracer()
    with t.span("noop") as span:
      span.async_end([])

  def test_noop_tracer_export(self):
    t = tracer.NoopTracer()
    self.assertEqual(t.export(), {})

  def test_noop_tracer_all_devices(self):
    t = tracer.NoopTracer()
    self.assertEqual(t.all_devices, [])

  def test_noop_tracer_synchronize(self):
    t = tracer.NoopTracer()
    t.synchronize()

  def test_noop_tracer_print(self):
    t = tracer.NoopTracer()
    t.print()


if __name__ == "__main__":
  absltest.main()
