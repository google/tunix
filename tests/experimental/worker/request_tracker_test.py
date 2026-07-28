# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for outgoing-request tracking and response correlation."""

import asyncio

from absl.testing import absltest
from tunix.experimental.worker import remote_execution as remote_lib
from tunix.experimental.worker import request_tracker as tracker_lib


def _response(request_id: str, result="ok") -> remote_lib.ExecutionResponse:
  return remote_lib.ExecutionResponse(result=result, request_id=request_id)


class _Clock:

  def __init__(self):
    self.now = 0.0

  def __call__(self) -> float:
    return self.now


class _Engine:

  async def work(self, tag: str, delay: float = 0.0) -> str:
    await asyncio.sleep(delay)
    return f"done:{tag}"


def _tracked() -> tracker_lib.TrackedActorHandle:
  server = remote_lib.InProcessRemoteExecutionServer(_Engine())
  return tracker_lib.TrackedActorHandle(remote_lib.InProcessActorHandle(server))


class RequestTrackerTest(absltest.TestCase):

  def test_resolves_response_to_its_request(self):
    tracker = tracker_lib.RequestTracker()
    tracker.register("a", "generate")
    record = tracker.resolve(_response("a"))
    self.assertIsNotNone(record)
    self.assertEqual(record.request_id, "a")
    self.assertEqual(record.method_name, "generate")
    self.assertTrue(record.is_complete)
    self.assertEqual(tracker.take("a").unwrap(), "ok")

  def test_matches_out_of_order_responses(self):
    tracker = tracker_lib.RequestTracker()
    for rid in ("a", "b", "c"):
      tracker.register(rid)
    # Responses arrive in completion order, not dispatch order.
    tracker.resolve(_response("c", "third"))
    tracker.resolve(_response("a", "first"))
    self.assertEqual(tracker.take("a").unwrap(), "first")
    self.assertEqual(tracker.take("c").unwrap(), "third")
    self.assertEqual(tracker.pending_ids(), ["b"])

  def test_duplicate_response_is_ignored_first_wins(self):
    tracker = tracker_lib.RequestTracker()
    tracker.register("a")
    self.assertIsNotNone(tracker.resolve(_response("a", "first")))
    self.assertIsNone(tracker.resolve(_response("a", "second")))
    self.assertEqual(tracker.duplicate_count, 1)
    self.assertEqual(tracker.take("a").unwrap(), "first")

  def test_orphan_response_is_reported_not_raised(self):
    tracker = tracker_lib.RequestTracker()
    self.assertIsNone(tracker.resolve(_response("never-sent")))
    self.assertEqual(tracker.orphan_count, 1)

  def test_take_returns_none_until_resolved(self):
    tracker = tracker_lib.RequestTracker()
    tracker.register("a")
    self.assertIsNone(tracker.take("a"))
    self.assertTrue(tracker.is_pending("a"))
    tracker.resolve(_response("a"))
    self.assertFalse(tracker.is_pending("a"))
    self.assertIsNotNone(tracker.take("a"))
    # Taking removes it from the book.
    self.assertIsNone(tracker.take("a"))

  def test_in_flight_and_pending_ids_track_outstanding_work(self):
    tracker = tracker_lib.RequestTracker()
    tracker.register("a")
    tracker.register("b")
    self.assertEqual(tracker.in_flight, 2)
    tracker.resolve(_response("a"))
    self.assertEqual(tracker.in_flight, 1)
    self.assertEqual(tracker.pending_ids(), ["b"])

  def test_overdue_lists_only_stale_in_flight_requests(self):
    clock = _Clock()
    tracker = tracker_lib.RequestTracker(time_fn=clock)
    tracker.register("old")
    clock.now = 5.0
    tracker.register("new")
    clock.now = 6.0
    overdue = tracker.overdue(timeout_s=2.0)
    self.assertEqual([r.request_id for r in overdue], ["old"])
    # A completed request is never overdue.
    tracker.resolve(_response("old"))
    self.assertEmpty(tracker.overdue(timeout_s=2.0))

  def test_re_registering_keeps_the_original_dispatch_time(self):
    clock = _Clock()
    tracker = tracker_lib.RequestTracker(time_fn=clock)
    first = tracker.register("a")
    clock.now = 9.0
    again = tracker.register("a")
    self.assertIs(first, again)
    self.assertEqual(again.dispatched_at, 0.0)

  def test_forget_stops_tracking(self):
    tracker = tracker_lib.RequestTracker()
    tracker.register("a")
    self.assertTrue(tracker.forget("a"))
    self.assertFalse(tracker.forget("a"))
    self.assertEqual(tracker.in_flight, 0)


class TrackedActorHandleTest(absltest.TestCase):

  def test_dispatch_records_request_and_await_returns_its_response(self):
    async def _run():
      tracked = _tracked()
      request_id = await tracked.dispatch("work", "a")
      self.assertTrue(tracked.tracker.is_pending(request_id))

      response = await tracked.await_response(request_id, timeout_s=1.0)
      self.assertIsNotNone(response)
      self.assertEqual(response.request_id, request_id)
      self.assertEqual(response.unwrap(), "done:a")
      self.assertEqual(tracked.tracker.in_flight, 0)

    asyncio.run(_run())

  def test_await_returns_the_right_response_when_another_finishes_first(self):
    async def _run():
      tracked = _tracked()
      slow = await tracked.dispatch("work", "slow", delay=0.10)
      fast = await tracked.dispatch("work", "fast", delay=0.0)

      # Wait on the slow one; the fast response lands first but must not be
      # mistaken for it, and must remain collectable.
      slow_resp = await tracked.await_response(slow, timeout_s=2.0)
      self.assertEqual(slow_resp.unwrap(), "done:slow")

      fast_resp = await tracked.await_response(fast, timeout_s=1.0)
      self.assertEqual(fast_resp.unwrap(), "done:fast")

    asyncio.run(_run())

  def test_drain_matches_all_ready_responses(self):
    async def _run():
      tracked = _tracked()
      ids = [await tracked.dispatch("work", str(i)) for i in range(3)]
      await asyncio.sleep(0.05)  # let them all finish

      matched = await tracked.drain(timeout_s=0.0)
      self.assertEqual(matched, 3)
      self.assertEqual(tracked.tracker.in_flight, 0)
      for rid in ids:
        self.assertIsNotNone(tracked.take(rid))

    asyncio.run(_run())

  def test_await_times_out_and_leaves_request_pending(self):
    async def _run():
      tracked = _tracked()
      request_id = await tracked.dispatch("work", "slow", delay=1.0)
      self.assertIsNone(
          await tracked.await_response(request_id, timeout_s=0.02)
      )
      # Still outstanding, so a timeout/retry policy can see it.
      self.assertTrue(tracked.tracker.is_pending(request_id))
      self.assertEqual(tracked.tracker.pending_ids(), [request_id])

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
