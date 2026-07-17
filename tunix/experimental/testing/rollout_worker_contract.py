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

"""Reusable contract test suite for RolloutWorker implementations.

Mix `RolloutWorkerContractSuite` into an `absltest.TestCase` and implement
`make_worker()`; the shared tests then pin the ticket/cursor/ack/cancel and
weight-sync contract. The same suite is run against the fake here and against a
real RolloutWorker implementation in its own plan — that is what keeps the fake
and the real worker behaviorally interchangeable.
"""

import asyncio

from tunix.experimental.common import datatypes
from tunix.experimental.worker import rollout_worker


class RolloutWorkerContractSuite:
  """Contract tests shared across all RolloutWorker implementations."""

  def make_worker(self) -> rollout_worker.RolloutWorker:
    raise NotImplementedError("Subclasses must provide make_worker().")

  def _started_worker(self) -> rollout_worker.RolloutWorker:
    worker = self.make_worker()
    worker.initialize()
    worker.start()
    return worker

  def _requests(self, n: int) -> list[datatypes.TrajectoryRequest]:
    return [
        datatypes.TrajectoryRequest(
            request_id=f"req-{i}",
            prompt_id=f"prompt-{i}",
            prompt_text=f"prompt text {i}",
            sampling_params=datatypes.SamplingParams(max_tokens=8),
        )
        for i in range(n)
    ]

  def test_lifecycle_and_info(self):
    worker = self.make_worker()
    worker.initialize()
    worker.start()
    self.assertEqual(worker.health().state, "READY")
    self.assertIn("rollout", worker.info().roles)
    worker.stop()
    self.assertEqual(worker.health().state, "STOPPED")

  def test_generate_then_cursor_read_returns_all(self):
    worker = self._started_worker()

    async def _run():
      ticket = await worker.generate(self._requests(3))
      return await worker.get_completed(ticket, after_seq=0)

    page = asyncio.run(_run())
    self.assertLen(page.results, 3)
    self.assertTrue(page.done)
    self.assertEqual(
        {r.request_id for r in page.results}, {"req-0", "req-1", "req-2"}
    )
    for result in page.results:
      self.assertEqual(result.status, "SUCCEEDED")

  def test_cursor_read_is_idempotent(self):
    worker = self._started_worker()

    async def _run():
      ticket = await worker.generate(self._requests(2))
      first = await worker.get_completed(ticket, after_seq=0)
      second = await worker.get_completed(ticket, after_seq=0)
      return first, second

    first, second = asyncio.run(_run())
    self.assertEqual(
        [r.request_id for r in first.results],
        [r.request_id for r in second.results],
    )

  def test_paging_by_cursor(self):
    worker = self._started_worker()

    async def _run():
      ticket = await worker.generate(self._requests(3))
      first = await worker.get_completed(ticket, after_seq=0, max_items=2)
      second = await worker.get_completed(
          ticket, after_seq=first.last_seq, max_items=2
      )
      return first, second

    first, second = asyncio.run(_run())
    self.assertLen(first.results, 2)
    self.assertFalse(first.done)
    self.assertLen(second.results, 1)
    self.assertTrue(second.done)

  def test_ack_releases_acked_results(self):
    worker = self._started_worker()

    async def _run():
      ticket = await worker.generate(self._requests(2))
      page = await worker.get_completed(ticket, after_seq=0)
      await worker.ack(ticket, page.last_seq)
      return await worker.get_completed(ticket, after_seq=page.last_seq)

    page = asyncio.run(_run())
    self.assertEmpty(page.results)
    self.assertTrue(page.done)

  def test_cancel_emits_cancelled_results(self):
    worker = self._started_worker()

    async def _run():
      ticket = await worker.generate(self._requests(2))
      await worker.cancel(ticket)
      return await worker.get_completed(ticket, after_seq=0)

    page = asyncio.run(_run())
    self.assertLen(page.results, 2)
    for result in page.results:
      self.assertEqual(result.status, "CANCELLED")

  def test_unknown_ticket_raises(self):
    worker = self._started_worker()
    with self.assertRaises(rollout_worker.TicketNotFound):
      asyncio.run(worker.get_completed("no-such-ticket", after_seq=0))

  def test_sync_weights_advances_version(self):
    worker = self._started_worker()
    version = worker.sync_weights(
        datatypes.WeightSyncMetadata(version=7, method="in_process")
    )
    self.assertEqual(version, 7)
    self.assertEqual(worker.health().policy_version, 7)
