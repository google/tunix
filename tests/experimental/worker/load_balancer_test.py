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

"""Unit tests for capacity-aware pool routing (`load_balancer.py`)."""

import asyncio
from typing import Any, Dict, List, Optional

from absl.testing import absltest
from tunix.experimental.worker import load_balancer as lb_lib
from tunix.experimental.worker import remote_execution as remote_lib


class FakeRolloutHandle(remote_lib.ActorHandle):
  """Worker whose per-request completion is driven by the test.

  Records the high-water mark of concurrently running requests so tests can
  assert the pool never exceeds `max_concurrency` on a single worker.
  """

  def __init__(self, worker_id: str):
    self.worker_id = worker_id
    self.running: List[str] = []
    self.peak_concurrency = 0
    self.accepted: List[str] = []
    self._done: List[remote_lib.ExecutionResponse] = []
    self.dispatch_error: Optional[Exception] = None

  def submit(self, method_name: Optional[str] = None, *args, **kwargs) -> Any:
    raise NotImplementedError()

  async def asubmit(
      self, method_name: Optional[str] = None, *args, **kwargs
  ) -> Any:
    raise NotImplementedError()

  async def dispatch_task(
      self,
      request_id: Optional[str] = None,
      method_name: Optional[str] = None,
      *args,
      **kwargs,
  ) -> str:
    if self.dispatch_error is not None:
      raise self.dispatch_error
    self.accepted.append(request_id)
    self.running.append(request_id)
    self.peak_concurrency = max(self.peak_concurrency, len(self.running))
    return request_id

  async def poll_responses(
      self, timeout_s: float = remote_lib.LONG_POLL_TIMEOUT_S
  ) -> Any:
    while not self._done:
      await asyncio.sleep(0)
    return self._done.pop(0)

  def finish(self, request_id: str, result: Any = None) -> None:
    """Completes an in-flight request, as the worker would."""
    self.running.remove(request_id)
    self._done.append(
        remote_lib.ExecutionResponse(
            request_id=request_id,
            result=result if result is not None else f"{self.worker_id}:done",
        )
    )

  def finish_all(self) -> None:
    for request_id in list(self.running):
      self.finish(request_id)


def _tasks(count: int, start: int = 0) -> List[lb_lib.Task]:
  return [
      lb_lib.Task(request_id=f"req_{i}", method_name="generate", args=(i,))
      for i in range(start, start + count)
  ]


class CapacityRouterTest(absltest.TestCase):

  def test_rejects_invalid_concurrency(self):
    with self.assertRaises(ValueError):
      lb_lib.CapacityRouter(max_concurrency=0)

  def test_fills_every_worker_before_saturating(self):
    router = lb_lib.CapacityRouter(max_concurrency=2)
    actors = [FakeRolloutHandle("a"), FakeRolloutHandle("b")]

    chosen = [router(actors) for _ in range(4)]

    # Two slots each, spread evenly rather than stacked on one worker.
    self.assertCountEqual(chosen, actors * 2)
    self.assertEqual(router.outstanding(actors[0]), 2)
    self.assertEqual(router.outstanding(actors[1]), 2)
    self.assertFalse(router.has_capacity(actors))

  def test_raises_when_all_workers_are_full(self):
    router = lb_lib.CapacityRouter(max_concurrency=1)
    actors = [FakeRolloutHandle("a")]
    router(actors)

    with self.assertRaises(lb_lib.NoCapacityError):
      router(actors)

  def test_release_returns_the_slot_to_that_worker(self):
    router = lb_lib.CapacityRouter(max_concurrency=1)
    actors = [FakeRolloutHandle("a"), FakeRolloutHandle("b")]
    router(actors)
    router(actors)

    router.release(actors[1])

    self.assertEqual(router.select(actors), actors[1])
    self.assertEqual(router.total_outstanding(), 1)

  def test_wait_for_capacity_resumes_on_release(self):
    async def _run():
      router = lb_lib.CapacityRouter(max_concurrency=1)
      actors = [FakeRolloutHandle("a")]
      router(actors)

      waiter = asyncio.create_task(router.wait_for_capacity(actors))
      await asyncio.sleep(0)
      self.assertFalse(waiter.done())

      router.release(actors[0])
      await asyncio.wait_for(waiter, timeout=1.0)

    asyncio.run(_run())


class BalancedDispatcherTest(absltest.TestCase):

  def test_requires_at_least_one_worker(self):
    with self.assertRaises(ValueError):
      lb_lib.BalancedDispatcher([], max_concurrency=1)

  def test_primes_max_concurrency_per_worker_before_any_completion(self):
    """The opening burst is max_concurrency on every worker, and no more."""

    async def _run():
      workers = [FakeRolloutHandle("w0"), FakeRolloutHandle("w1")]
      dispatcher = lb_lib.BalancedDispatcher(workers, max_concurrency=2)
      self.assertEqual(dispatcher.max_in_flight, 4)

      results = []
      stream = dispatcher.run(_tasks(10))
      consumer = asyncio.create_task(stream.__anext__())

      # Nothing has completed yet, so the pool is exactly saturated.
      await asyncio.sleep(0.05)
      self.assertLen(workers[0].running, 2)
      self.assertLen(workers[1].running, 2)
      self.assertFalse(consumer.done())

      workers[0].finish_all()
      workers[1].finish_all()
      results.append(await asyncio.wait_for(consumer, timeout=1.0))
      await stream.aclose()

      self.assertLen(results, 1)

    asyncio.run(_run())

  def test_finished_worker_receives_the_next_task(self):
    """A worker that completes work is refilled while its peer stays busy."""

    async def _run():
      fast, slow = FakeRolloutHandle("fast"), FakeRolloutHandle("slow")
      dispatcher = lb_lib.BalancedDispatcher([fast, slow], max_concurrency=1)

      seen = []

      async def _consume():
        async for request_id, _, exc in dispatcher.run(_tasks(6)):
          self.assertIsNone(exc)
          seen.append(request_id)

      consumer = asyncio.create_task(_consume())
      await asyncio.sleep(0.05)
      self.assertLen(fast.running, 1)
      self.assertLen(slow.running, 1)

      # Only `fast` ever completes; `slow` holds its single slot the whole time.
      for _ in range(4):
        fast.finish_all()
        await asyncio.sleep(0.05)

      self.assertLen(slow.accepted, 1)
      self.assertLen(fast.accepted, 5)
      self.assertEqual(fast.peak_concurrency, 1)

      slow.finish_all()
      fast.finish_all()
      await asyncio.wait_for(consumer, timeout=2.0)
      self.assertLen(seen, 6)

    asyncio.run(_run())

  def test_never_exceeds_max_concurrency_on_a_worker(self):
    async def _run():
      workers = [FakeRolloutHandle(f"w{i}") for i in range(3)]
      dispatcher = lb_lib.BalancedDispatcher(workers, max_concurrency=2)

      completed = []

      async def _consume():
        async for request_id, _, exc in dispatcher.run(_tasks(30)):
          self.assertIsNone(exc)
          completed.append(request_id)

      consumer = asyncio.create_task(_consume())
      for _ in range(40):
        await asyncio.sleep(0.005)
        for worker in workers:
          worker.finish_all()
        if consumer.done():
          break
      await asyncio.wait_for(consumer, timeout=3.0)

      self.assertLen(completed, 30)
      self.assertCountEqual(completed, [f"req_{i}" for i in range(30)])
      for worker in workers:
        self.assertLessEqual(worker.peak_concurrency, 2)

    asyncio.run(_run())

  def test_correlates_results_to_their_request_ids(self):
    async def _run():
      worker = FakeRolloutHandle("w0")
      dispatcher = lb_lib.BalancedDispatcher([worker], max_concurrency=2)

      got: Dict[str, Any] = {}

      async def _consume():
        async for request_id, result, exc in dispatcher.run(_tasks(2)):
          self.assertIsNone(exc)
          got[request_id] = result

      consumer = asyncio.create_task(_consume())
      await asyncio.sleep(0.05)

      # Complete out of submission order.
      worker.finish("req_1", result="second")
      worker.finish("req_0", result="first")
      await asyncio.wait_for(consumer, timeout=2.0)

      self.assertEqual(got, {"req_0": "first", "req_1": "second"})

    asyncio.run(_run())

  def test_failed_task_does_not_abort_the_batch(self):
    async def _run():
      worker = FakeRolloutHandle("w0")
      dispatcher = lb_lib.BalancedDispatcher([worker], max_concurrency=2)

      outcomes = []

      async def _consume():
        async for request_id, result, exc in dispatcher.run(_tasks(2)):
          outcomes.append((request_id, result, exc))

      consumer = asyncio.create_task(_consume())
      await asyncio.sleep(0.05)

      worker.running.remove("req_0")
      worker._done.append(  # pylint: disable=protected-access
          remote_lib.ExecutionResponse(
              request_id="req_0",
              error_message="boom",
              error_type="RuntimeError",
          )
      )
      worker.finish("req_1", result="ok")
      await asyncio.wait_for(consumer, timeout=2.0)

      by_id = {rid: (res, exc) for rid, res, exc in outcomes}
      self.assertLen(by_id, 2)
      self.assertIsNotNone(by_id["req_0"][1])
      self.assertEqual(by_id["req_1"], ("ok", None))
      # The failed task's slot was returned, not leaked.
      self.assertEqual(dispatcher.router.total_outstanding(), 0)

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
