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

"""Unit tests for pooled rollout generation (`rollout_pool.py`)."""

import asyncio
import contextlib
import types
from typing import Any, List, Optional, Sequence

from absl.testing import absltest
import numpy as np
import portpicker
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import rollout_pool
from tunix.experimental.worker import remote_execution as remote_lib


class SpyRolloutWorker:
  """Rollout worker that records concurrency and can stall chosen prompts."""

  def __init__(self, worker_id: str, stalled_prompts: Sequence[str] = ()):
    self.worker_id = worker_id
    self.seen: List[str] = []
    self.in_flight = 0
    self.peak_concurrency = 0
    self._stalled = set(stalled_prompts)
    self.release = asyncio.Event()

  async def generate(
      self, request: datatypes.RolloutRequest
  ) -> datatypes.RolloutResponse:
    self.seen.append(request.prompt_id)
    self.in_flight += 1
    self.peak_concurrency = max(self.peak_concurrency, self.in_flight)
    try:
      if request.prompt_id in self._stalled:
        await self.release.wait()
      else:
        await asyncio.sleep(0)
      return datatypes.RolloutResponse(
          status="SUCCEEDED",
          prompt_tokens=np.asarray([1, 2, 3], dtype=np.int32),
          env_reward=1.0,
          metadata={"served_by": self.worker_id},
      )
    finally:
      self.in_flight -= 1


class ExplodingRolloutWorker:
  """Rollout worker that fails for one specific prompt."""

  def __init__(self, bad_prompt: str):
    self._bad_prompt = bad_prompt

  async def generate(
      self, request: datatypes.RolloutRequest
  ) -> datatypes.RolloutResponse:
    if request.prompt_id == self._bad_prompt:
      raise RuntimeError("sampler exploded")
    return datatypes.RolloutResponse(status="SUCCEEDED")


def _requests(count: int) -> List[datatypes.RolloutRequest]:
  return [
      datatypes.RolloutRequest(prompt=f"prompt {i}", prompt_id=f"p{i}")
      for i in range(count)
  ]


def _in_process_handles(
    workers: Sequence[object],
) -> List[remote_lib.ActorHandle]:
  handles = []
  for worker in workers:
    server = remote_lib.InProcessRemoteExecutionServer(instance=worker)
    handles.append(remote_lib.InProcessActorHandle(server))
  return handles


class GrpcRolloutWorker:
  """Rollout worker served over gRPC; reports which port answered."""

  def __init__(self, worker_id: str, latency_s: float):
    self.worker_id = worker_id
    self._latency_s = latency_s

  async def generate(
      self, request: datatypes.RolloutRequest
  ) -> datatypes.RolloutResponse:
    await asyncio.sleep(self._latency_s)
    return datatypes.RolloutResponse(
        status="SUCCEEDED",
        prompt_tokens=np.asarray([7, 8], dtype=np.int32),
        metadata={"served_by": self.worker_id},
    )


@contextlib.asynccontextmanager
async def _serving(workers: Sequence[Any]):
  """Serves each worker on its own localhost gRPC port."""
  servers, handles = [], []
  try:
    for worker in workers:
      port = portpicker.pick_unused_port()
      server = remote_lib.GrpcRemoteExecutionServer(worker)
      await server.start_serving_async(port=port)
      servers.append(server)
      handles.append(
          remote_lib.GrpcRemoteActorHandle(
              target_address=f"grpc://localhost:{port}"
          )
      )
    yield handles
  finally:
    for handle in handles:
      await handle.close()
    for server in servers:
      await server.stop_serving()


class PooledRolloutWorkerTest(absltest.TestCase):

  def test_returns_responses_in_request_order_despite_out_of_order_completion(
      self,
  ):
    async def _run():
      # p0 stalls on worker 0, so p1 and p2 finish first on worker 1.
      slow = SpyRolloutWorker("slow", stalled_prompts=("p0",))
      fast = SpyRolloutWorker("fast")
      pool = rollout_pool.PooledRolloutWorker(
          _in_process_handles([slow, fast]), max_concurrency=1
      )

      requests = _requests(3)
      streamed: List[Optional[str]] = []
      generate = asyncio.create_task(
          pool.generate(
              requests,
              on_complete=lambda r: streamed.append(r.metadata.get("served_by")),
          )
      )
      await asyncio.sleep(0.05)
      slow.release.set()
      responses = await asyncio.wait_for(generate, timeout=5.0)

      self.assertLen(responses, 3)
      # Order follows the requests, not completion order.
      self.assertEqual(
          [r.request_id for r in responses], ["p0:0", "p1:1", "p2:2"]
      )
      self.assertTrue(all(r.status == "SUCCEEDED" for r in responses))
      # The stalled trajectory was streamed last even though it was submitted
      # first.
      self.assertLen(streamed, 3)
      self.assertEqual(streamed[-1], "slow")

    asyncio.run(_run())

  def test_spreads_work_across_the_pool_and_refills_free_workers(self):
    async def _run():
      # Worker 0 holds one prompt open; the rest of the batch must land on the
      # other worker rather than queueing behind the stall.
      stalling = SpyRolloutWorker("w0", stalled_prompts=("p0",))
      available = SpyRolloutWorker("w1")
      pool = rollout_pool.PooledRolloutWorker(
          _in_process_handles([stalling, available]), max_concurrency=1
      )
      self.assertEqual(pool.max_in_flight, 2)

      generate = asyncio.create_task(pool.generate(_requests(6)))
      await asyncio.sleep(0.1)

      self.assertEqual(stalling.seen, ["p0"])
      self.assertLen(available.seen, 5)
      self.assertEqual(available.peak_concurrency, 1)

      stalling.release.set()
      responses = await asyncio.wait_for(generate, timeout=5.0)
      self.assertLen(responses, 6)

    asyncio.run(_run())

  def test_honors_max_concurrency_per_worker(self):
    async def _run():
      worker = SpyRolloutWorker(
          "w0", stalled_prompts=[f"p{i}" for i in range(8)]
      )
      pool = rollout_pool.PooledRolloutWorker(
          _in_process_handles([worker]), max_concurrency=3
      )

      generate = asyncio.create_task(pool.generate(_requests(8)))
      await asyncio.sleep(0.1)
      self.assertEqual(worker.in_flight, 3)

      worker.release.set()
      responses = await asyncio.wait_for(generate, timeout=5.0)
      self.assertLen(responses, 8)
      self.assertEqual(worker.peak_concurrency, 3)

    asyncio.run(_run())

  def test_failed_request_is_reported_in_band(self):
    async def _run():
      pool = rollout_pool.PooledRolloutWorker(
          _in_process_handles([ExplodingRolloutWorker(bad_prompt="p1")]),
          max_concurrency=2,
      )

      responses = await asyncio.wait_for(
          pool.generate(_requests(3)), timeout=5.0
      )

      self.assertLen(responses, 3)
      failed = responses[1]
      self.assertEqual(failed.status, "FAILED")
      self.assertIsNotNone(failed.error)
      self.assertIn("sampler exploded", failed.error.message)
      # The rest of the batch still succeeded.
      self.assertEqual(responses[0].status, "SUCCEEDED")
      self.assertEqual(responses[2].status, "SUCCEEDED")

    asyncio.run(_run())

  def test_single_request_returns_a_single_response(self):
    async def _run():
      pool = rollout_pool.PooledRolloutWorker(
          _in_process_handles([SpyRolloutWorker("w0")]), max_concurrency=1
      )

      response = await asyncio.wait_for(
          pool.generate(_requests(1)[0]), timeout=5.0
      )

      self.assertIsInstance(response, datatypes.RolloutResponse)
      self.assertEqual(response.status, "SUCCEEDED")

    asyncio.run(_run())

  def test_explicit_request_ids_are_preserved(self):
    async def _run():
      pool = rollout_pool.PooledRolloutWorker(
          _in_process_handles([SpyRolloutWorker("w0")]), max_concurrency=2
      )
      requests = [
          datatypes.RolloutRequest(
              prompt="a", prompt_id="shared_group", request_id="rid_a"
          ),
          datatypes.RolloutRequest(
              prompt="b", prompt_id="shared_group", request_id="rid_b"
          ),
      ]

      responses = await asyncio.wait_for(pool.generate(requests), timeout=5.0)

      self.assertEqual([r.request_id for r in responses], ["rid_a", "rid_b"])

    asyncio.run(_run())

  def test_a_second_caller_waits_for_the_first_to_finish(self):
    """One consumer per worker: the second batch starts only after the first.

    Two consumers polling one worker would take each other's responses, which
    are then discarded as unrecognized, starving whoever was waiting. The pool
    prevents that by never having two batches in flight at once.
    """

    async def _run():
      worker = SpyRolloutWorker("w0", stalled_prompts=("p0",))
      pool = rollout_pool.PooledRolloutWorker(
          _in_process_handles([worker]), max_concurrency=2
      )

      first = asyncio.create_task(pool.generate(_requests(1)))
      await asyncio.sleep(0.05)
      second = asyncio.create_task(pool.generate(_requests(2)))
      await asyncio.sleep(0.05)

      # The first batch is stalled, so the second has not been dispatched.
      self.assertEqual(worker.seen, ["p0"])
      self.assertFalse(second.done())

      worker.release.set()
      await asyncio.wait_for(asyncio.gather(first, second), timeout=10.0)
      self.assertEqual(worker.seen, ["p0", "p0", "p1"])

    asyncio.run(_run())

  def test_concurrent_generate_calls_are_serialized(self):
    """Two callers overlap safely: neither loses a response nor leaks a slot."""

    async def _run():
      worker = SpyRolloutWorker("w0")
      pool = rollout_pool.PooledRolloutWorker(
          _in_process_handles([worker]), max_concurrency=2
      )

      first, second = await asyncio.wait_for(
          asyncio.gather(
              pool.generate(_requests(3)),
              pool.generate(_requests(3)),
          ),
          timeout=10.0,
      )

      self.assertLen(first, 3)
      self.assertLen(second, 3)
      for responses in (first, second):
        self.assertTrue(all(r.status == "SUCCEEDED" for r in responses))
      # Serialized, so the worker never saw two batches at once.
      self.assertEqual(worker.peak_concurrency, 2)
      self.assertEqual(pool.dispatcher.router.total_outstanding(), 0)

    asyncio.run(_run())

  def test_duplicate_request_ids_are_rejected(self):
    """One id for two requests would dispatch one twice and the other never."""

    async def _run():
      pool = rollout_pool.PooledRolloutWorker(
          _in_process_handles([SpyRolloutWorker("w0")]), max_concurrency=2
      )
      requests = [
          datatypes.RolloutRequest(
              prompt="a", prompt_id="p0", request_id="same"
          ),
          datatypes.RolloutRequest(
              prompt="b", prompt_id="p1", request_id="same"
          ),
      ]

      with self.assertRaises(rollout_pool.DuplicateRequestIdError):
        await pool.generate(requests)

    asyncio.run(_run())

  def test_ids_derived_from_prompts_stay_unique_within_a_group(self):
    """A GRPO group repeats prompt_id, so the fallback id adds the position."""

    async def _run():
      worker = SpyRolloutWorker("w0")
      pool = rollout_pool.PooledRolloutWorker(
          _in_process_handles([worker]), max_concurrency=2
      )
      group = [
          datatypes.RolloutRequest(prompt="a", prompt_id="shared")
          for _ in range(3)
      ]

      responses = await asyncio.wait_for(pool.generate(group), timeout=5.0)

      self.assertEqual(
          [r.request_id for r in responses],
          ["shared:0", "shared:1", "shared:2"],
      )

    asyncio.run(_run())

  def test_a_request_that_never_answers_becomes_a_failed_response(self):
    """The batch keeps its shape even if a result goes missing."""
    pool = rollout_pool.PooledRolloutWorker(
        _in_process_handles([SpyRolloutWorker("w0")]), max_concurrency=1
    )

    response = pool._missing_response("req-7")  # pylint: disable=protected-access

    self.assertEqual(response.request_id, "req-7")
    self.assertEqual(response.status, "FAILED")
    self.assertIsNotNone(response.error)
    self.assertIn("req-7", response.error.message)

  def test_pool_accepts_handles_that_wrap_an_actor(self):
    """The fleet's RPC handles expose their transport as `.actor`."""
    server = remote_lib.InProcessRemoteExecutionServer(
        instance=SpyRolloutWorker("w0")
    )
    actor = remote_lib.InProcessActorHandle(server)
    wrapper = types.SimpleNamespace(actor=actor)

    pool = rollout_pool.PooledRolloutWorker.from_workers([wrapper])

    self.assertEqual(pool.dispatcher.actors, (actor,))

  def test_pool_rejects_something_that_is_not_a_worker(self):
    with self.assertRaises(TypeError):
      rollout_pool.PooledRolloutWorker.from_workers([object()])

  def test_a_lost_trajectory_times_out_instead_of_stalling_the_batch(self):
    """Nothing else bounds the wait, so a stuck worker would stall the step."""

    async def _run():
      worker = SpyRolloutWorker("w0", stalled_prompts=("p1",))
      pool = rollout_pool.PooledRolloutWorker(
          _in_process_handles([worker]),
          max_concurrency=2,
          batch_timeout_s=0.2,
      )

      responses = await asyncio.wait_for(
          pool.generate(_requests(2)), timeout=10.0
      )

      self.assertLen(responses, 2)
      self.assertEqual(responses[0].status, "SUCCEEDED")
      # The straggler is reported, not waited on forever.
      self.assertEqual(responses[1].status, "FAILED")
      self.assertIsNotNone(responses[1].error)

      worker.release.set()

    asyncio.run(_run())

  def test_draining_waits_for_outstanding_work_and_holds_off_new_work(self):
    """Weights can only be replaced when nothing is mid-generation."""

    async def _run():
      worker = SpyRolloutWorker("w0", stalled_prompts=("p0",))
      pool = rollout_pool.PooledRolloutWorker(
          _in_process_handles([worker]), max_concurrency=2
      )
      observed = {}

      generating = asyncio.create_task(pool.generate(_requests(1)))
      await asyncio.sleep(0.05)

      async def _fence():
        async with pool.drained():
          # Nothing is in flight by the time the fence is held.
          observed["outstanding"] = (
              pool.dispatcher.router.total_outstanding()
          )
          observed["generate_done"] = generating.done()
          # And nothing new can start while it is held.
          queued = asyncio.create_task(pool.generate(_requests(1)))
          await asyncio.sleep(0.05)
          observed["queued_started"] = queued.done()
          return queued

      fence = asyncio.create_task(_fence())
      await asyncio.sleep(0.05)
      # The fence is still waiting on the stalled request.
      self.assertFalse(fence.done())

      worker.release.set()
      queued = await asyncio.wait_for(fence, timeout=10.0)
      await asyncio.wait_for(generating, timeout=10.0)
      await asyncio.wait_for(queued, timeout=10.0)

      self.assertEqual(observed["outstanding"], 0)
      self.assertTrue(observed["generate_done"])
      self.assertFalse(observed["queued_started"])

    asyncio.run(_run())

  def test_balances_over_real_grpc_workers(self):
    """Fire-and-forget dispatch plus long polling, across two localhost servers."""

    async def _run():
      workers = [
          GrpcRolloutWorker("fast", latency_s=0.01),
          GrpcRolloutWorker("slow", latency_s=0.25),
      ]
      async with _serving(workers) as handles:
        pool = rollout_pool.PooledRolloutWorker(handles, max_concurrency=1)

        responses = await asyncio.wait_for(
            pool.generate(_requests(8)), timeout=60.0
        )

        self.assertLen(responses, 8)
        self.assertTrue(all(r.status == "SUCCEEDED" for r in responses))
        self.assertEqual(
            [r.request_id for r in responses],
            [f"p{i}:{i}" for i in range(8)],
        )
        # Work followed capacity rather than a fixed rotation: the fast server
        # took more of the batch than the slow one.
        served_by = [r.metadata["served_by"] for r in responses]
        self.assertGreater(served_by.count("fast"), served_by.count("slow"))
        self.assertGreaterEqual(served_by.count("slow"), 1)

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
