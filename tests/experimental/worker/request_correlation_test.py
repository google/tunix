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

"""Request/response correlation over the fire-and-forget execution path."""

import asyncio

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.worker import remote_execution as remote_lib


class _Engine:
  """Worker whose call latency is caller-controlled, to force out-of-order completion."""

  async def work(self, tag: str, delay: float = 0.0) -> str:
    await asyncio.sleep(delay)
    return f"done:{tag}"

  async def consume(self, req: datatypes.RolloutRequest) -> str:
    return f"consumed:{req.prompt_id}"


def _handle(engine=None) -> remote_lib.InProcessActorHandle:
  server = remote_lib.InProcessRemoteExecutionServer(engine or _Engine())
  return remote_lib.InProcessActorHandle(server)


class RequestCorrelationTest(absltest.TestCase):

  def test_dispatch_ack_is_the_request_id_and_response_echoes_it(self):
    async def _run():
      handle = _handle()
      request_id = await handle.dispatch_task("work", "a")
      response = await handle.poll_responses(timeout_s=1.0)
      self.assertIsNotNone(response)
      # The ack the caller holds is exactly what comes back on the response.
      self.assertEqual(response.request_id, request_id)
      self.assertEqual(response.unwrap(), "done:a")

    asyncio.run(_run())

  def test_request_ids_are_unique_per_dispatch(self):
    async def _run():
      handle = _handle()
      ids = {await handle.dispatch_task("work", str(i)) for i in range(5)}
      self.assertLen(ids, 5)

    asyncio.run(_run())

  def test_domain_request_id_is_reused_as_the_transport_id(self):
    # A domain DTO already carries a request_id; the transport should adopt it
    # rather than mint a competing one.
    async def _run():
      handle = _handle()
      req = datatypes.RolloutRequest(request_id="rid-42", prompt_id="p7")
      request_id = await handle.dispatch_task("consume", req)
      self.assertEqual(request_id, "rid-42")
      response = await handle.poll_responses(timeout_s=1.0, request_id="rid-42")
      self.assertIsNotNone(response)
      self.assertEqual(response.unwrap(), "consumed:p7")

    asyncio.run(_run())

  def test_poll_by_request_id_returns_that_request_response(self):
    # The slow request is dispatched first but finishes last: polling for it by
    # id must not hand back the fast request's response.
    async def _run():
      handle = _handle()
      slow_id = await handle.dispatch_task("work", "slow", delay=0.10)
      fast_id = await handle.dispatch_task("work", "fast", delay=0.0)
      self.assertNotEqual(slow_id, fast_id)

      slow = await handle.poll_responses(timeout_s=2.0, request_id=slow_id)
      self.assertIsNotNone(slow)
      self.assertEqual(slow.request_id, slow_id)
      self.assertEqual(slow.unwrap(), "done:slow")

    asyncio.run(_run())

  def test_responses_for_other_requests_are_parked_not_dropped(self):
    # While waiting on the slow request, the fast response arrives first; it must
    # still be collectable afterwards.
    async def _run():
      handle = _handle()
      slow_id = await handle.dispatch_task("work", "slow", delay=0.10)
      fast_id = await handle.dispatch_task("work", "fast", delay=0.0)

      slow = await handle.poll_responses(timeout_s=2.0, request_id=slow_id)
      self.assertEqual(slow.unwrap(), "done:slow")

      fast = await handle.poll_responses(timeout_s=1.0, request_id=fast_id)
      self.assertIsNotNone(fast)
      self.assertEqual(fast.request_id, fast_id)
      self.assertEqual(fast.unwrap(), "done:fast")

    asyncio.run(_run())

  def test_poll_for_unknown_request_id_times_out(self):
    async def _run():
      handle = _handle()
      await handle.dispatch_task("work", "a")
      unmatched = await handle.poll_responses(
          timeout_s=0.05, request_id="no-such-id"
      )
      self.assertIsNone(unmatched)

    asyncio.run(_run())

  def test_unfiltered_poll_still_returns_next_response(self):
    # Back-compat: callers that do not correlate keep FIFO behavior.
    async def _run():
      handle = _handle()
      await handle.dispatch_task("work", "a")
      response = await handle.poll_responses(timeout_s=1.0)
      self.assertIsNotNone(response)
      self.assertEqual(response.unwrap(), "done:a")

    asyncio.run(_run())

  def test_failed_request_response_still_carries_the_request_id(self):
    async def _run():
      handle = _handle()
      # Unknown method -> error response, which must still be correlatable.
      request_id = await handle.dispatch_task("nope")
      response = await handle.poll_responses(timeout_s=1.0, request_id=request_id)
      self.assertIsNotNone(response)
      self.assertEqual(response.request_id, request_id)
      self.assertIsNotNone(response.error_message)

    asyncio.run(_run())

  def test_execution_request_round_trips_its_id(self):
    req = remote_lib.ExecutionRequest("work", ("a",), {}, request_id="rid-1")
    restored = remote_lib.ExecutionRequest.deserialize(req.serialize())
    self.assertEqual(restored.request_id, "rid-1")

  def test_execution_response_round_trips_its_id(self):
    resp = remote_lib.ExecutionResponse(result=7, request_id="rid-2")
    restored = remote_lib.ExecutionResponse.deserialize(resp.serialize())
    self.assertEqual(restored.request_id, "rid-2")
    self.assertEqual(restored.unwrap(), 7)


if __name__ == "__main__":
  absltest.main()
