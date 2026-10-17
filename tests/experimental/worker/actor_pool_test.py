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

"""Unit tests for actor routing pool and execution session (`actor_pool.py`)."""

import asyncio
from typing import Any, Optional
from absl.testing import absltest
from absl.testing import parameterized
from tunix.experimental.worker import actor_pool
from tunix.experimental.worker import remote_execution as remote_lib


class StubWorkerEngine:
  """Mock backend domain instance for verifying dynamic remote method execution."""

  def __init__(self, worker_id: str, latency: float = 0.01):
    self.worker_id = worker_id
    self.latency = latency
    self.call_count = 0
    self.is_paused = False

  async def compute_trajectory(
      self, prompt_id: str = "default_prompt", turns: int = 3
  ) -> str:
    await asyncio.sleep(self.latency)
    self.call_count += 1
    if self.is_paused:
      raise RuntimeError(f"Worker [{self.worker_id}] is currently paused.")
    return (
        f"[{self.worker_id}] Trajectory for prompt {prompt_id} ({turns} turns)"
    )

  async def __call__(
      self, prompt_id: str = "default_prompt", turns: int = 3
  ) -> str:
    return await self.compute_trajectory(prompt_id, turns)

  def pause(self) -> None:
    self.is_paused = True

  def resume(self) -> None:
    self.is_paused = False

  def get_status(self) -> str:
    return (
        f"worker_id={self.worker_id}, count={self.call_count},"
        f" paused={self.is_paused}"
    )

  def kv_cache_aware(self, prompt: str = "test") -> str:
    return f"[{self.worker_id}] KV-cache aware routing for {prompt}"


def create_in_process_handle(
    engine: Any,
) -> remote_lib.InProcessActorHandle:
  """Helper creating an InProcessActorHandle bound to an InProcessRemoteExecutionServer."""
  server = remote_lib.InProcessRemoteExecutionServer(engine)
  return remote_lib.InProcessActorHandle(server)


class ActorPoolTest(parameterized.TestCase):
  """Tests verifying RoutingActorPool and PoolExecutionSession."""

  def test_actor_pool_load_balancing_and_streaming(self):
    async def _run_test():
      engine_a = StubWorkerEngine("worker_A", latency=0.04)
      engine_b = StubWorkerEngine("worker_B", latency=0.01)

      server_a = remote_lib.InProcessRemoteExecutionServer(engine_a)
      server_b = remote_lib.InProcessRemoteExecutionServer(engine_b)

      handle_a = remote_lib.InProcessActorHandle(server_a)
      handle_b = remote_lib.InProcessActorHandle(server_b)

      pool = actor_pool.RoutingActorPool([handle_a, handle_b])

      # Submit batch of tasks across pool and verify out-of-order completion
      # stream
      tasks = [
          ("req_1", "compute_trajectory", ("req_slow_on_A",), {"turns": 1}),
          ("req_2", "compute_trajectory", ("req_fast_on_B",), {"turns": 1}),
      ]

      results = []
      async for res in pool.as_completed_stream(tasks):
        results.append(res)

      # worker_B (latency 0.01) should finish before worker_A (latency 0.04)
      self.assertLen(results, 2)
      self.assertIn(
          "[worker_B] Trajectory for prompt req_fast_on_B", results[0]
      )
      self.assertIn(
          "[worker_A] Trajectory for prompt req_slow_on_A", results[1]
      )

    asyncio.run(_run_test())

  def test_stable_route_hash_is_deterministic_across_processes(self):
    # Golden values, not a self-consistency check: a PYTHONHASHSEED-salted
    # hash would produce different numbers on each interpreter invocation and
    # so could never match a hardcoded constant.
    self.assertEqual(
        actor_pool.stable_route_hash("prompt_7"), 7682816066691324195
    )
    self.assertEqual(
        actor_pool.stable_route_hash("route_key_abc"), 8273245243606849686
    )

  def test_stable_route_hash_passes_integers_through(self):
    # Integer keys are explicit shard indices and must not be hashed.
    for shard in (0, 1, 7, 64):
      self.assertEqual(actor_pool.stable_route_hash(shard), shard)

  def test_stable_route_hash_rejects_negative_integers(self):
    # A negative shard index is always a caller bug: `%` would silently fold it
    # onto a valid actor instead of surfacing the mistake.
    with self.assertRaises(ValueError):
      actor_pool.stable_route_hash(-1)

  def test_stable_route_hash_separates_adjacent_structured_keys(self):
    # Regression guard: CRC32 is linear over GF(2), so keys differing in one
    # trailing character collapse into the same bucket under small moduli.
    for num_actors in (2, 3, 4, 8):
      buckets = {
          actor_pool.stable_route_hash(f"prompt_7#{shard}") % num_actors
          for shard in range(num_actors)
      }
      self.assertGreater(
          len(buckets),
          1,
          f"adjacent route keys all collapsed to one bucket for"
          f" num_actors={num_actors}",
      )

  def test_routing_actor_pool_prompt_affinity_and_custom_router(self):
    async def _run_test():
      engine_0 = StubWorkerEngine("worker_0", latency=0.001)
      engine_1 = StubWorkerEngine("worker_1", latency=0.001)
      handle_0 = remote_lib.InProcessActorHandle(
          remote_lib.InProcessRemoteExecutionServer(engine_0)
      )
      handle_1 = remote_lib.InProcessActorHandle(
          remote_lib.InProcessRemoteExecutionServer(engine_1)
      )

      pool = actor_pool.RoutingActorPool([handle_0, handle_1])

      # Verify sticky routing: identical route_keys consistently route to the
      # same worker without method_name
      res_a1 = await pool.asubmit(
          route_key="prompt_sticky_X",
      )
      res_a2 = await pool.asubmit(
          route_key="prompt_sticky_X",
      )
      res_a3 = await pool.asubmit(
          route_key="prompt_sticky_X",
      )

      worker_prefix = res_a1.split("]")[0] + "]"
      self.assertTrue(res_a2.startswith(worker_prefix))
      self.assertTrue(res_a3.startswith(worker_prefix))

      # Verify custom router callable switching strategies per method_name
      def custom_router(actors, method_name, unused_args, kwargs):
        if method_name == "compute_trajectory":
          route_key = kwargs.get("route_key")
          return (
              actors[hash(route_key) % len(actors)] if route_key else actors[0]
          )
        elif method_name == "kv_cache_aware":
          return actors[1]
        return actors[0]

      smart_pool = actor_pool.RoutingActorPool(
          [handle_0, handle_1], router=custom_router
      )

      # Heavy inference call routes by sticky route_key
      res_traj = await smart_pool.asubmit(
          "compute_trajectory",
          route_key="prompt_X",
      )
      self.assertTrue(res_traj.startswith("[worker_"))

      # Test synchronous submit across the pool
      sync_res = smart_pool.submit("get_status")
      self.assertIn("count=", sync_res)

      # KV-cache aware call routes directly to actors[1]
      res_kv = await smart_pool.asubmit("kv_cache_aware", "prompt_v2")
      self.assertIn("[worker_1] KV-cache aware routing for prompt_v2", res_kv)

      # Verify router object providing a method matching `method_name`
      # (`self.router."method_name"()`)
      class MethodSpecificRouter:

        def kv_cache_aware(self, actors, unused_args, unused_kwargs):
          return actors[1]

      method_pool = actor_pool.RoutingActorPool(
          [handle_0, handle_1], router=MethodSpecificRouter()
      )
      res_method = await method_pool.asubmit("kv_cache_aware", "any_prompt")
      self.assertIn("[worker_1]", res_method)

      # Verify router without the method raises TypeError
      class BadRouter:
        pass

      bad_pool = actor_pool.RoutingActorPool(
          [handle_0, handle_1], router=BadRouter()
      )
      with self.assertRaisesRegex(
          TypeError, "Router object .* must provide a method"
      ):
        await bad_pool.asubmit("kv_cache_aware", "any_prompt")

    asyncio.run(_run_test())

  def test_actor_pool_submit_without_actors_raises(self):
    pool = actor_pool.RoutingActorPool()
    with self.assertRaisesRegex(
        RuntimeError, "RoutingActorPool contains no registered ActorHandles"
    ):
      pool.submit("any")

  def test_actor_pool_stream_without_actors_raises(self):
    pool = actor_pool.RoutingActorPool()

    async def _run_pool_err():
      with self.assertRaisesRegex(
          RuntimeError,
          "RoutingActorPool contains no registered ActorHandles",
      ):
        async for _ in pool.as_completed_stream([("any", (), {})]):
          pass

    asyncio.run(_run_pool_err())

  def test_actor_pool_add_invalid_type_raises(self):
    pool = actor_pool.RoutingActorPool()
    with self.assertRaisesRegex(
        TypeError, "Expected str or ActorHandle, got <class 'int'>"
    ):
      pool.add_actor(123)  # type: ignore

  def test_as_completed_stream_with_none_results(self):
    """Verifies as_completed_stream handles tasks returning None without deadlocking."""

    class NoneWorker:

      def void_task(self) -> None:
        return None

    async def _run():
      handle = create_in_process_handle(NoneWorker())
      pool = actor_pool.RoutingActorPool([handle])

      tasks = [("req_1", "void_task", (), {}), ("req_2", "void_task", (), {})]
      results = []
      async for res in pool.as_completed_stream(tasks):
        results.append(res)

      self.assertEqual(results, [None, None])

    asyncio.run(_run())

  def test_background_task_exception_propagation(self):
    """Verifies exceptions raised during background task execution propagate to client stream."""

    async def _run():
      engine = StubWorkerEngine("crashing_worker", latency=0.01)
      engine.pause()
      handle = create_in_process_handle(engine)
      pool = actor_pool.RoutingActorPool([handle])

      tasks = [("req_1", "compute_trajectory", ("failing_prompt",), {})]
      with self.assertRaisesRegex(RuntimeError, "currently paused"):
        async for _ in pool.as_completed_stream(tasks):
          pass

    asyncio.run(_run())

  def test_pool_execution_session_dynamic_enqueuing_and_fault_isolation(self):
    """Verifies bulk submission, dynamic task enqueuing, and fault isolation in PoolExecutionSession."""

    class DynamicWorker:

      def __init__(self, name: str):
        self.name = name

      def process(self, val: int) -> int:
        if val < 0:
          raise ValueError(f"Negative value {val} not allowed on {self.name}")
        return val * 10

    async def _run():
      w1 = create_in_process_handle(DynamicWorker("w1"))
      w2 = create_in_process_handle(DynamicWorker("w2"))
      pool = actor_pool.RoutingActorPool([w1, w2])

      results = []
      errors = []

      # 1. Client calls execution_session for an initial bulk of tasks
      initial_tasks = [
          ("req_1", "process", (5,), {}),
          ("req_2", "process", (-10,), {}),  # This task will throw an exception!
          ("req_3", "process", (15,), {}),
      ]
      async with pool.execution_session(initial_tasks) as session:
        # 2. Client waits for tasks to finish
        async for result, exc in session.as_completed():
          if exc is not None:
            # 4. If any task throws an exception, it does not affect receiving responses from other tasks
            errors.append(str(exc))
            continue

          results.append(result)
          # 3. Whenever a task is finished, client tries to enqueue a new task
          if result == 50:
            await session.submit("req_4", "process", 2)  # will produce 20
          elif result == 150:
            await session.submit("req_5", "process", 3)  # will produce 30

      self.assertLen(errors, 1)
      self.assertIn("Negative value -10 not allowed", errors[0])
      self.assertCountEqual(results, [50, 150, 20, 30])

    asyncio.run(_run())

  def test_pool_execution_session_with_explicit_request_ids_in_initial_tasks(
      self,
  ):
    class EchoWorker:

      def process(self, val: int) -> int:
        return val * 2

    async def _run():
      w1 = create_in_process_handle(EchoWorker())
      pool = actor_pool.RoutingActorPool([w1])

      initial_tasks = [
          ("custom_id_1", "process", (5,), {}),
          ("custom_id_2", "process", (10,), {}),
      ]
      results = []
      async with pool.execution_session(initial_tasks) as session:
        async for result, exc in session.as_completed():
          self.assertIsNone(exc)
          results.append(result)

      self.assertCountEqual(results, [10, 20])

    asyncio.run(_run())

  def test_pool_execution_session_no_double_decrement_on_dispatch_failure(self):
    """Verifies that dispatch failures in submit() do not double-decrement _in_flight."""

    class FailingDispatchHandle(remote_lib.ActorHandle):

      def submit(
          self, method_name: Optional[str] = None, *args, **kwargs
      ) -> Any:
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
        raise ConnectionError("Network error during dispatch")

      async def poll_responses(
          self, timeout_s: float = remote_lib.LONG_POLL_TIMEOUT_S
      ) -> Any:
        raise ConnectionError("Network error during polling")

    async def _run():
      handle = FailingDispatchHandle()
      pool = actor_pool.RoutingActorPool([handle])
      session = actor_pool.PoolExecutionSession(pool)

      with self.assertRaises(ConnectionError):
        await session.submit("req_fail", "process", 1)

      # Verify that _in_flight is properly 0 and not negative/corrupted
      self.assertEqual(session._in_flight, 0)
      self.assertEmpty(session._dispatched_tasks.get(handle, set()))
      await session.close()

    asyncio.run(_run())

  def test_pool_execution_session_deadlock_when_loop_dies_during_second_dispatch(
      self,
  ):
    """Verifies as_completed() does not hang if polling loop dies while dispatch_task is awaiting."""

    class DropDuringSecondDispatchHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.poll1_started = asyncio.Event()
        self.task2_dispatch_pause = asyncio.Event()
        self.trigger_connection_drop = asyncio.Event()

      def submit(
          self, method_name: Optional[str] = None, *args, **kwargs
      ) -> Any:
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
        if method_name == "task1":
          return "task_1"
        # Task 2 dispatch: pause while poll1 is active
        await self.task2_dispatch_pause.wait()
        return "task_2"

      async def poll_responses(
          self, timeout_s: float = remote_lib.LONG_POLL_TIMEOUT_S
      ) -> Any:
        # First call: long poll for task1
        if not self.poll1_started.is_set():
          self.poll1_started.set()
          await self.trigger_connection_drop.wait()
          raise ConnectionError("Connection dropped during task1 poll")
        # Subsequent calls (for restarted loop)
        raise ConnectionError("Worker dead on second loop poll")

    async def _run():
      handle = DropDuringSecondDispatchHandle()
      pool = actor_pool.RoutingActorPool([handle])
      session = actor_pool.PoolExecutionSession(pool)

      # 1. Dispatch task1 (returns task_1, polling loop starts long-poll for task1)
      await session.submit("req_t1", "task1")
      await handle.poll1_started.wait()

      # 2. Start task2 dispatch in background (pauses in dispatch_task)
      task2_fut = asyncio.create_task(session.submit("req_t2", "task2"))
      await asyncio.sleep(0.01)

      # 3. Connection drops while task1 is polling and task2 is dispatching!
      handle.trigger_connection_drop.set()
      await asyncio.sleep(0.01)

      # 4. Now allow task2 dispatch_task to finish
      handle.task2_dispatch_pause.set()
      await task2_fut

      # 5. Consume as_completed stream; both task1 and task2 errors must be yielded
      results = []

      async def _consume():
        async for item in session.as_completed():
          results.append(item)

      await asyncio.wait_for(_consume(), timeout=1.0)
      await session.close()

      self.assertLen(results, 2)
      self.assertIsNotNone(results[0][1])
      self.assertIsInstance(results[0][1], ConnectionError)
      self.assertIsNotNone(results[1][1])
      self.assertIsInstance(results[1][1], ConnectionError)

    asyncio.run(_run())

  def test_pool_execution_session_removes_matching_request_id_from_dispatched_tasks_out_of_order(
      self,
  ):
    """Verifies that response.request_id removes the matching request ID from session._dispatched_tasks."""

    class OutOfOrderHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.poll_event = asyncio.Event()
        self.responses = [
            remote_lib.ExecutionResponse(
                request_id="req_2", result="res_2"
            ),
            remote_lib.ExecutionResponse(
                request_id="req_1", result="res_1"
            ),
        ]

      def submit(
          self, method_name: Optional[str] = None, *args, **kwargs
      ) -> Any:
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
        return request_id

      async def poll_responses(
          self, timeout_s: float = remote_lib.LONG_POLL_TIMEOUT_S
      ) -> Any:
        await self.poll_event.wait()
        self.poll_event.clear()
        if self.responses:
          return self.responses.pop(0)
        await asyncio.sleep(100)

    async def _run():
      handle = OutOfOrderHandle()
      pool = actor_pool.RoutingActorPool([handle])
      session = actor_pool.PoolExecutionSession(pool)

      await session.submit("req_1", "method")
      await session.submit("req_2", "method")

      self.assertEqual(session._dispatched_tasks[handle], {"req_1", "req_2"})

      # Allow first poll: req_2 arrives first (out of order!)
      handle.poll_event.set()
      it = session.as_completed()
      res1, exc1 = await it.__anext__()
      self.assertEqual(res1, "res_2")
      # req_2 should be removed from dispatched_tasks, leaving only req_1
      self.assertEqual(session._dispatched_tasks[handle], {"req_1"})

      # Allow second poll: req_1 arrives next
      handle.poll_event.set()
      res2, exc2 = await it.__anext__()
      self.assertEqual(res2, "res_1")
      # dispatched_set should now be empty
      self.assertEqual(session._dispatched_tasks[handle], set())

      await session.close()

    asyncio.run(_run())

  def test_pool_execution_session_submit_pops_route_key_after_routing(self):
    dispatched_kwargs = []

    class CaptureHandle(remote_lib.ActorHandle):

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del method_name, args
        dispatched_kwargs.append(dict(kwargs))
        return request_id or ""

      async def poll_responses(self, timeout_s=50.0):
        await asyncio.sleep(timeout_s)
        return None

    async def _run():
      h1, h2 = CaptureHandle(), CaptureHandle()
      pool = actor_pool.RoutingActorPool([h1, h2])
      session = actor_pool.PoolExecutionSession(pool)
      await session.submit("req_1", "generate", route_key="sticky_key", x=42)
      self.assertEqual(dispatched_kwargs, [{"x": 42}])
      await session.close()

    asyncio.run(_run())

  def test_pool_execution_session_poll_completed_drains_batch_and_skips_idle_worker(
      self,
  ):
    class FastBatchHandle(remote_lib.ActorHandle):

      def __init__(self, items):
        self._items = list(items)
        self.poll_calls = 0

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del method_name, args, kwargs
        return request_id or ""

      async def poll_responses(self, timeout_s=50.0):
        self.poll_calls += 1
        if self._items:
          return self._items.pop(0)
        await asyncio.sleep(timeout_s)
        return None

    async def _run():
      busy_handle = FastBatchHandle([
          remote_lib.ExecutionResponse(request_id="req_1", result="r1"),
          remote_lib.ExecutionResponse(request_id="req_2", result="r2"),
      ])
      idle_handle = FastBatchHandle([])
      # Custom router sends all tasks to busy_handle, leaving idle_handle with 0 tasks
      pool = actor_pool.RoutingActorPool(
          [busy_handle, idle_handle],
          router=lambda actors, method, args, kwargs: actors[0],
      )
      session = actor_pool.PoolExecutionSession(pool)

      await session.submit("req_1", "generate")
      await session.submit("req_2", "generate")
      # Let background _poll_worker_loop(busy_handle) enqueue both completions
      await asyncio.sleep(0.02)

      batch = await session.poll_completed(timeout_s=1.0)
      self.assertEqual(batch, [("r1", None), ("r2", None)])
      self.assertEqual(idle_handle.poll_calls, 0)
      self.assertEqual(session._in_flight, 0)

      # Subsequent poll_completed when idle returns [] quickly without blocking 1.0s
      loop = asyncio.get_running_loop()
      t0 = loop.time()
      empty_batch = await session.poll_completed(timeout_s=2.0)
      self.assertEqual(empty_batch, [])
      self.assertLess(loop.time() - t0, 0.5)

      await session.close()

    asyncio.run(_run())

  def test_pool_execution_session_simultaneous_dispatch_and_poll_error_invariant(
      self,
  ):
    class CrashingHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.crash_event = asyncio.Event()
        self.dispatch_count = 0

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del method_name, args, kwargs
        self.dispatch_count += 1
        if self.dispatch_count == 1:
          return request_id or ""
        # Second dispatch suspends until worker crashes
        await self.crash_event.wait()
        raise RuntimeError("dispatch transport error")

      async def poll_responses(self, timeout_s=50.0):
        del timeout_s
        await self.crash_event.wait()
        raise RuntimeError("poll transport error")

    async def _run():
      handle = CrashingHandle()
      pool = actor_pool.RoutingActorPool([handle])
      session = actor_pool.PoolExecutionSession(pool)

      # First task succeeds dispatch and starts _poll_worker_loop
      await session.submit("req_1", "generate")
      # Second task suspends inside dispatch_task while _poll_worker_loop is polling
      submit_task = asyncio.create_task(session.submit("req_2", "generate"))
      await asyncio.sleep(0.01)

      # Trigger simultaneous crash of both poll_responses and dispatch_task
      handle.crash_event.set()
      with self.assertRaisesRegex(RuntimeError, "dispatch transport error"):
        await submit_task
      await asyncio.sleep(0.01)

      # _in_flight must be exactly 0 (not double-decremented) and _dispatched_tasks empty
      self.assertEqual(session._in_flight, 0)
      self.assertEqual(session._dispatched_tasks[handle], set())
      await session.close()

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
