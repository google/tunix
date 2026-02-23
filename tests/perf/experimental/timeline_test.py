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

import threading
from unittest import mock
from absl.testing import absltest
from tunix.perf.experimental import constants
from tunix.perf.experimental import timeline


class SpanTest(absltest.TestCase):

  def test_span(self):
    s = timeline.Span(name="test", begin=1.0, id=0)
    self.assertEqual(s.name, "test")
    self.assertEqual(s.begin, 1.0)
    self.assertEqual(s.end, float("inf"))
    self.assertEqual(s.ended, False)
    self.assertEqual(s.duration, float("inf"))

  def test_span_with_tags(self):
    tags_dict = {constants.GLOBAL_STEP: 1, "custom_tag": "value"}
    s = timeline.Span(name="test_tags", begin=1.0, id=0, tags=tags_dict)
    self.assertEqual(s.tags, tags_dict)
    self.assertIn("tags=", repr(s))
    self.assertIn("global_step", repr(s))

  def test_add_tag(self):
    s = timeline.Span(name="test_add_tag", begin=1.0, id=0)
    s.add_tag("foo", "bar")
    self.assertEqual(s.tags, {"foo": "bar"})
    s.add_tag(constants.GLOBAL_STEP, 100)
    self.assertEqual(s.tags, {"foo": "bar", "global_step": 100})

  def test_add_tag_overwrite_warning(self):
    s = timeline.Span(name="test_add_tag_overwrite", begin=1.0, id=0)
    s.add_tag("foo", "bar")
    with self.assertLogs(level="WARNING") as cm:
      s.add_tag("foo", "baz")
    self.assertEqual(s.tags, {"foo": "baz"})
    self.assertTrue(
        any(
            "Tag 'foo' already exists with value 'bar'. Overwriting with 'baz'."
            in o
            for o in cm.output
        )
    )

  def test_repr_with_born_at(self):
    born_at = 100.0
    s = timeline.Span(name="test_born_at", begin=101.0, id=0)
    s.end = 105.0

    # Check default repr (born_at=0.0)
    expected_default = "[0] test_born_at: 101.000000, 105.000000"
    self.assertEqual(repr(s), expected_default)

    # Check repr with explicit born_at
    expected_adjusted = "[0] test_born_at: 1.000000, 5.000000"
    self.assertEqual(s._format_relative(born_at=born_at), expected_adjusted)


class TimelineTest(absltest.TestCase):

  def test_basic_span_lifecycle(self):
    t = timeline.Timeline("test_tl", 100.0)
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
    t = timeline.Timeline("test_tl", 0.0)
    s1 = t.start_span("root", 1.0)
    s2 = t.start_span("child", 2.0)

    self.assertEqual(s2.parent_id, s1.id)

    t.stop_span(3.0)  # stops s2
    self.assertEqual(s2.end, 3.0)

    t.stop_span(4.0)  # stops s1
    self.assertEqual(s1.end, 4.0)

  def test_stop_span_error_cases(self):
    t = timeline.Timeline("test_tl", 0.0)
    with self.assertRaisesRegex(ValueError, "no more spans to end"):
      t.stop_span(1.0)

    s = t.start_span("s1", 2.0)
    # End before begin
    with self.assertRaisesRegex(ValueError, "ended at .* before it began"):
      t.stop_span(1.0)

  def test_nested_timeline_with_tags_repr(self):
    born = 1000.0
    t = timeline.Timeline("test_tl", born)

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
        "[0] root: 1.000000, 4.000000, tags={'type': 'root_span'}\n"
        "[1] child: 2.000000, 3.000000 (parent=0), tags={'iter': 1}\n"
    )
    self.assertEqual(repr(t), expected_repr)


class AsyncTimelineTest(absltest.TestCase):

  def setUp(self):
    self.patcher = mock.patch("tunix.perf.experimental.timeline._async_wait")
    self.mock_async_wait = self.patcher.start()

    # Setup mock behavior for _async_wait to immediately succeed by default
    def default_wait(waitlist, success, failure):
      success()
      return mock.Mock(spec=threading.Thread)

    self.mock_async_wait.side_effect = default_wait

  def tearDown(self):
    self.patcher.stop()

  def test_span_success(self):
    t = timeline.AsyncTimeline("dev", 0.0)
    waitlist = ["thing"]

    t.span("async_op", 1.0, waitlist)

    self.mock_async_wait.assert_called_once()
    self.assertEqual(len(t.spans), 1)
    s = t.spans[0]
    self.assertEqual(s.name, "async_op")
    self.assertEqual(s.begin, 1.0)
    self.assertTrue(s.ended)  # Ended because mock calls success immediately

  def test_span_with_no_waitlist(self):
    t = timeline.AsyncTimeline("dev", 0.0)
    t.span("immediate", 1.0, [])
    self.mock_async_wait.assert_not_called()
    self.assertEqual(len(t.spans), 1)
    self.assertTrue(t.spans[0].ended)

  def test_delayed_completion(self):
    t = timeline.AsyncTimeline("dev", 0.0)

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
    t = timeline.AsyncTimeline("dev", 0.0)

    def fail_wait(waitlist, success, failure):
      failure(RuntimeError("failed"))
      return mock.Mock(spec=threading.Thread)

    self.mock_async_wait.side_effect = fail_wait

    with self.assertRaisesRegex(RuntimeError, "failed"):
      t.span("failed", 1.0, ["wait"])

  def test_wait_pending_spans_clears_threads(self):
    t = timeline.AsyncTimeline("test_tl", 0.0)
    t.span("s1", 1.0, ["wait"])
    t.span("s2", 2.0, ["wait"])

    self.assertLen(t._threads, 2)
    t.wait_pending_spans()
    self.assertLen(t._threads, 0)

  def test_async_wait_helper(self):
    # Temporarily unpatch _async_wait to test the real implementation
    self.patcher.stop()
    try:
      waitlist = ["data"]
      success_cb = mock.Mock()
      failure_cb = mock.Mock()

      # Mock jax.block_until_ready to avoid actual JAX calls
      with mock.patch("tunix.perf.experimental.timeline.jax.block_until_ready") as mock_block:
        t = timeline._async_wait(waitlist, success_cb, failure_cb)

        # Verify it returned a thread and started it
        self.assertIsInstance(t, threading.Thread)
        self.assertIsNotNone(t.ident)
        t.join()

        mock_block.assert_called_once_with(waitlist)
        success_cb.assert_called_once()
        failure_cb.assert_not_called()
    finally:
      self.patcher.start()


class BatchAsyncTimelinesTest(absltest.TestCase):

  def test_span_broadcast(self):
    t1 = mock.create_autospec(timeline.AsyncTimeline, instance=True)
    t2 = mock.create_autospec(timeline.AsyncTimeline, instance=True)
    batch = timeline.BatchAsyncTimelines([t1, t2])

    waitlist = ["thing"]
    tags = {"foo": "bar"}
    batch.span("test_span", 100.0, waitlist, tags=tags)

    t1.span.assert_called_once_with("test_span", 100.0, waitlist, tags=tags)
    t2.span.assert_called_once_with("test_span", 100.0, waitlist, tags=tags)


if __name__ == "__main__":
  absltest.main()
