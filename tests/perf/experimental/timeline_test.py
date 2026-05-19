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
import time
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
    self.assertFalse(s.ended)
    self.assertEqual(s.duration, float("inf"))

  def test_span_with_tags(self):
    tags_dict = {constants.STEP: 1, "custom_tag": "value"}
    s = timeline.Span(name="test_tags", begin=1.0, id=0, tags=tags_dict)
    self.assertEqual(s.tags, tags_dict)
    self.assertIn("tags=", repr(s))
    self.assertIn("step", repr(s))

  def test_add_tag(self):
    s = timeline.Span(name="test_add_tag", begin=1.0, id=0)
    s.add_tag("foo", "bar")
    self.assertEqual(s.tags, {"foo": "bar"})
    s.add_tag(constants.STEP, 100)
    self.assertEqual(s.tags, {"foo": "bar", "step": 100})

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


class TimelineTest(absltest.TestCase):

  def test_commit_step(self):
    t = timeline.Timeline("test_tl", 0.0)

    # Start and stop a span
    s1 = t.start_span("span1", 1.0)
    t.stop_span(2.0)

    # Start an uncompleted span
    s2 = t.start_span("span2", 3.0)

    with self.subTest("Pre-commit state"):
      self.assertLen(t.cur_step, 2)
      self.assertEmpty(t.committed_steps)

    # Commit step
    with self.assertLogs(level="WARNING") as cm:
      t.commit_step()

    with self.subTest("Post-commit purging"):
      # Verify uncompleted span was purged
      self.assertTrue(
          any("Purging uncompleted span 'span2'" in o for o in cm.output)
      )

    with self.subTest("Committed steps"):
      self.assertLen(t.committed_steps, 1)
      self.assertIn(s1.id, t.committed_steps[0])
      self.assertNotIn(s2.id, t.committed_steps[0])

    with self.subTest("All steps and current step"):
      self.assertEmpty(t.cur_step)
      self.assertLen(t.all_steps, 2)
      self.assertEqual(t.all_steps[0], t.committed_steps[0])
      self.assertEqual(t.all_steps[1], {})

  def test_basic_span_lifecycle(self):
    t = timeline.Timeline("test_tl", 100.0)
    s = t.start_span("span1", 101.0)
    with self.subTest("Span started"):
      self.assertEqual(s.name, "span1")
      self.assertEqual(s.begin, 101.0)
      self.assertEqual(s.id, 0)
      self.assertIsNone(s.parent_id)
      self.assertFalse(s.ended)

    t.stop_span(102.0)
    with self.subTest("Span stopped"):
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

  def _commit_simple_steps(self, tl, n):
    """Helper: commits ``n`` steps with one span each, returns the span ids."""
    span_ids = []
    for i in range(n):
      s = tl.start_span(f"span{i}", float(i))
      tl.stop_span(float(i) + 0.1)
      span_ids.append(s.id)
      tl.commit_step()
    return span_ids

  def test_no_consumer_means_no_drops(self):
    """With no registered consumers, committed_steps grows unboundedly --
    this preserves the pre-existing behavior for direct timeline users."""
    t = timeline.Timeline("test_tl", 0.0)
    self._commit_simple_steps(t, 5)
    self.assertLen(t.committed_steps, 5)
    self.assertEqual(t.dropped_step_count, 0)

  def test_single_consumer_drops_after_advance(self):
    t = timeline.Timeline("test_tl", 0.0)
    self._commit_simple_steps(t, 5)
    t.register_consumer("writer", start_at_current_head=False)
    # Snapshot reference before drop to confirm copy-on-write semantics.
    pre_drop_ref = t.committed_steps

    dropped = t.advance_consumer("writer", 2)
    self.assertEqual(dropped, 2)
    self.assertLen(t.committed_steps, 3)
    self.assertEqual(t.dropped_step_count, 2)
    # The previously captured list is unchanged.
    self.assertLen(pre_drop_ref, 5)

    # Advancing the remaining 3 drains the rest.
    dropped = t.advance_consumer("writer", 3)
    self.assertEqual(dropped, 3)
    self.assertEmpty(t.committed_steps)
    self.assertEqual(t.dropped_step_count, 5)

  def test_slowest_consumer_pins_memory(self):
    """When multiple consumers are registered, the slowest one holds memory."""
    t = timeline.Timeline("test_tl", 0.0)
    self._commit_simple_steps(t, 5)
    t.register_consumer("fast", start_at_current_head=False)
    t.register_consumer("slow", start_at_current_head=False)

    # Fast consumer races ahead; nothing should be dropped because slow is
    # still at 0.
    self.assertEqual(t.advance_consumer("fast", 5), 0)
    self.assertLen(t.committed_steps, 5)

    # Slow consumer advances by 2; min cursor becomes 2 -> drop 2.
    self.assertEqual(t.advance_consumer("slow", 2), 2)
    self.assertLen(t.committed_steps, 3)
    self.assertEqual(t.dropped_step_count, 2)

    # Slow consumer catches up; everything drains.
    self.assertEqual(t.advance_consumer("slow", 3), 3)
    self.assertEmpty(t.committed_steps)

  def test_unregister_consumer_releases_pinned_memory(self):
    """A laggard consumer that goes away must not pin memory forever."""
    t = timeline.Timeline("test_tl", 0.0)
    self._commit_simple_steps(t, 5)
    t.register_consumer("active", start_at_current_head=False)
    t.register_consumer("laggard", start_at_current_head=False)
    # Active races ahead; laggard never moves.
    self.assertEqual(t.advance_consumer("active", 5), 0)
    self.assertLen(t.committed_steps, 5)

    # Removing the laggard immediately drops the steps it was pinning.
    self.assertEqual(t.unregister_consumer("laggard"), 5)
    self.assertEmpty(t.committed_steps)
    self.assertEqual(t.dropped_step_count, 5)

  def test_unregister_unknown_consumer_is_noop(self):
    t = timeline.Timeline("test_tl", 0.0)
    self.assertEqual(t.unregister_consumer("never_registered"), 0)

  def test_unregister_last_consumer_stops_drops(self):
    """If the last consumer is removed, the timeline reverts to keep-all."""
    t = timeline.Timeline("test_tl", 0.0)
    self._commit_simple_steps(t, 3)
    t.register_consumer("only", start_at_current_head=False)
    self.assertEqual(t.advance_consumer("only", 1), 1)
    t.unregister_consumer("only")
    # Further commits should accumulate forever -- no consumer left to drop.
    self._commit_simple_steps(t, 4)
    self.assertLen(t.committed_steps, 2 + 4)  # 2 leftover + 4 new

  def test_register_with_start_at_current_head_skips_existing(self):
    """``start_at_current_head=True`` is about *consumer responsibility*, not
    memory: the late consumer never sees the pre-existing steps as work to
    process, but those steps still drop from memory as soon as the timeline
    catches up (here, immediately on registration since no other consumer is
    holding them)."""
    t = timeline.Timeline("test_tl", 0.0)
    self._commit_simple_steps(t, 3)
    # Pre-existing 3 steps are still in memory (no consumers registered).
    self.assertLen(t.committed_steps, 3)

    t.register_consumer("late", start_at_current_head=True)
    # Late consumer's cursor is at the absolute count (3), so without any
    # advance the min cursor is already 3 -- but drops only fire on advance
    # or unregister calls, so the 3 are still in memory at this exact point.
    self.assertLen(t.committed_steps, 3)

    # Commit 2 more; advance by 2 (the late consumer's "new work").
    self._commit_simple_steps(t, 2)
    self.assertEqual(t.advance_consumer("late", 2), 5)
    # All 5 are now released: 3 because the cursor was already past them at
    # registration, and the new 2 because the cursor just advanced through.
    self.assertEmpty(t.committed_steps)
    self.assertEqual(t.dropped_step_count, 5)

  def test_advance_clamps_to_current_max_cursor(self):
    t = timeline.Timeline("test_tl", 0.0)
    self._commit_simple_steps(t, 3)
    t.register_consumer("c", start_at_current_head=False)
    # Asking to advance past the end is clamped, not an error.
    self.assertEqual(t.advance_consumer("c", 100), 3)
    self.assertEmpty(t.committed_steps)

  def test_advance_unknown_consumer_raises(self):
    t = timeline.Timeline("test_tl", 0.0)
    with self.assertRaisesRegex(ValueError, "is not registered"):
      t.advance_consumer("ghost", 1)

  def test_advance_negative_raises(self):
    t = timeline.Timeline("test_tl", 0.0)
    t.register_consumer("c")
    with self.assertRaisesRegex(ValueError, "n must be non-negative"):
      t.advance_consumer("c", -1)

  def test_register_consumer_idempotent(self):
    t = timeline.Timeline("test_tl", 0.0)
    self._commit_simple_steps(t, 3)
    t.register_consumer("c", start_at_current_head=False)
    self.assertEqual(t.advance_consumer("c", 1), 1)
    # Re-registering must NOT reset the cursor (which would re-pin the dropped
    # step and break monotonicity).
    t.register_consumer("c", start_at_current_head=False)
    self.assertEqual(t.advance_consumer("c", 2), 2)
    self.assertEmpty(t.committed_steps)

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
        "Current Step -0:\n"
        "  [0] root: 1.000000, 4.000000, tags={'type': 'root_span'}\n"
        "  [1] child: 2.000000, 3.000000 (parent=0), tags={'iter': 1}\n"
    )
    self.assertEqual(repr(t), expected_repr)


class AsyncTimelineTest(absltest.TestCase):

  def setUp(self):
    self.patcher = mock.patch.object(timeline, "_async_wait")
    self.mock_async_wait = self.patcher.start()

    # Setup mock behavior for _async_wait to immediately succeed by default
    def default_wait(waitlist, success, failure):
      success()
      return mock.create_autospec(threading.Thread, instance=True)

    self.mock_async_wait.side_effect = default_wait

  def tearDown(self):
    self.patcher.stop()

  def test_span_success(self):
    t = timeline.AsyncTimeline("dev", 0.0)
    waitlist = ["thing"]

    t.span("async_op", 1.0, waitlist)

    self.mock_async_wait.assert_called_once()
    self.assertLen(t.cur_step, 1)
    s = t.cur_step[0]
    self.assertEqual(s.name, "async_op")
    self.assertEqual(s.begin, 1.0)
    self.assertTrue(s.ended)  # Ended because mock calls success immediately

  def test_span_with_no_waitlist(self):
    t = timeline.AsyncTimeline("dev", 0.0)
    t.span("immediate", 1.0, [])
    self.mock_async_wait.assert_not_called()
    self.assertLen(t.cur_step, 1)
    self.assertTrue(t.cur_step[0].ended)

  def test_delayed_completion(self):
    t = timeline.AsyncTimeline("dev", 0.0)

    # Capture callbacks
    callbacks = {}

    def capture_wait(waitlist, success, failure):
      callbacks["success"] = success
      callbacks["failure"] = failure
      return mock.create_autospec(threading.Thread, instance=True)

    self.mock_async_wait.side_effect = capture_wait

    t.span("delayed", 1.0, ["wait"])
    self.assertEmpty(t.cur_step)  # Not yet recorded

    # Simulate completion
    with mock.patch.object(time, "perf_counter", return_value=5.0):
      callbacks["success"]()

    self.assertLen(t.cur_step, 1)
    s = t.cur_step[0]
    self.assertEqual(s.end, 5.0)

  def test_nested_async_span_parent(self):
    t = timeline.AsyncTimeline("dev", 0.0)

    # Start sync spans to populate the stack with multiple items
    s0 = t.start_span("root", 1.0)
    s1 = t.start_span("child1", 2.0)
    s2 = t.start_span("child2", 3.0)

    # Now create an async span
    t.span("async_op", 4.0, ["wait"])

    # Find the async span.
    async_s = None
    for span in t.cur_step.values():
      if span.name == "async_op":
        async_s = span
        break

    self.assertIsNotNone(async_s)
    self.assertEqual(async_s.parent_id, s2.id)

    # Protect against index hardcoding bugs (e.g. self._spans_stack[1])
    self.assertNotEqual(async_s.parent_id, s1.id)

  def test_failure(self):
    t = timeline.AsyncTimeline("dev", 0.0)

    def fail_wait(waitlist, success, failure):
      failure(RuntimeError("failed"))
      return mock.create_autospec(threading.Thread, instance=True)

    self.mock_async_wait.side_effect = fail_wait

    with mock.patch.object(timeline.logging, "error") as mock_log_error:
      t.span("failed_op", 1.0, ["wait"])
      # Exception is caught and logged, no exception is raised to the caller.
      mock_log_error.assert_called_once()
      args, kwargs = mock_log_error.call_args
      format_str, name, span_id, err = args
      self.assertEqual(format_str, "Timeline span '%s' (id=%d) failed: %s")
      self.assertEqual(name, "failed_op")
      self.assertEqual(span_id, 0)
      self.assertIsInstance(err, RuntimeError)
      self.assertEqual(str(err), "failed")
      self.assertIn("exc_info", kwargs)
      self.assertIsInstance(kwargs["exc_info"], RuntimeError)

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
      success_cb = mock.create_autospec(lambda: None)
      failure_cb = mock.create_autospec(lambda e: None)

      # Mock jax.block_until_ready to avoid actual JAX calls
      with mock.patch.object(timeline.jax, "block_until_ready") as mock_block:
        t = timeline._async_wait(waitlist, success_cb, failure_cb)

        # Verify it returned a thread and started it
        with self.subTest("Thread execution"):
          self.assertIsInstance(t, threading.Thread)
          self.assertIsNotNone(t.ident)
        t.join()

        with self.subTest("Callbacks and JAX integration"):
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
