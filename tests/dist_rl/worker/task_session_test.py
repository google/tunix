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

"""Unit tests for TaskSession and DefaultRouter (`task_session.py`)."""

import asyncio
from typing import Any, Optional
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from tunix.dist_rl.worker import remote_execution as remote_lib
from tunix.dist_rl.worker import task_session


class StubWorkerEngine:
  """Mock backend domain instance for verifying remote method execution."""

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
  """Creates an InProcessActorHandle bound to an InProcessRemoteExecutionServer."""
  server = remote_lib.InProcessRemoteExecutionServer(engine)
  return remote_lib.InProcessActorHandle(server)


class GatedEngine:
  """Engine whose `compute` blocks on a per-label gate when one is set."""

  def __init__(self, name: str):
    self.name = name
    self.calls: list[str] = []
    self.gates: dict[str, asyncio.Event] = {}

  async def compute(self, label: str) -> str:
    self.calls.append(label)
    gate = self.gates.get(label)
    if gate is not None:
      await gate.wait()
    return f"[{self.name}] {label}"


class BlipHandle(remote_lib.ActorHandle):
  """Server-backed handle that injects transient dispatch faults.

  Per-request `plan` actions (consumed one per `dispatch_task` call; requests
  without a plan, or with an exhausted plan, follow `fail_all`):
    - "fail": raise `TransientTransportError` before reaching the server.
    - "lose_reply": the server accepts the task, then the ACK is lost.
    - "late_lose_reply": like "lose_reply", but the ACK is lost only after
      the poller had time to deliver the task's response.
    - "gate": wait on `retry_gate`, then forward to the server normally.
    - "ok": forward to the server normally.
  """

  def __init__(
      self,
      engine: GatedEngine,
      plan: Optional[dict[str, list[str]]] = None,
      fail_all: bool = False,
      quick_poll: bool = False,
  ):
    self.engine = engine
    self.quick_poll = quick_poll
    self.poll_calls = 0
    self.poll_blips = 0
    self.poll_timeouts: list[float] = []
    self.server = remote_lib.InProcessRemoteExecutionServer(engine)
    self.plan: dict[str, list[str]] = {
        k: list(v) for k, v in (plan or {}).items()
    }
    self.fail_all = fail_all
    self.dispatch_log: list[str] = []
    self.failed_dispatches = 0
    self.retry_gate = asyncio.Event()
    self.measure = False
    self._inflight = 0
    self.max_inflight_measured = 0

  def submit(self, method_name=None, *args, **kwargs):
    raise NotImplementedError()

  async def asubmit(self, method_name=None, *args, **kwargs):
    raise NotImplementedError()

  async def close(self) -> None:
    pass

  def sync_close(self) -> None:
    pass

  async def dispatch_task(
      self, request_id=None, method_name=None, *args, **kwargs
  ) -> str:
    assert request_id is not None
    self.dispatch_log.append(request_id)
    steps = self.plan.get(request_id)
    action = steps.pop(0) if steps else ("fail" if self.fail_all else "ok")
    self._inflight += 1
    if self.measure:
      self.max_inflight_measured = max(
          self.max_inflight_measured, self._inflight
      )
    try:
      await asyncio.sleep(0.002)
      if action == "fail":
        self.failed_dispatches += 1
        raise remote_lib.TransientTransportError(f"UNAVAILABLE {request_id}")
      if action == "gate":
        await self.retry_gate.wait()
      ack = await self.server.dispatch_task(
          remote_lib.ExecutionRequest(
              request_id=request_id,
              method_name=method_name,
              args=args,
              kwargs=kwargs,
          )
      )
      if action in ("lose_reply", "late_lose_reply"):
        if action == "late_lose_reply":
          await asyncio.sleep(0.05)
        self.failed_dispatches += 1
        raise remote_lib.TransientTransportError(f"reply lost {request_id}")
      return ack
    finally:
      self._inflight -= 1

  async def poll_responses(self, timeout_s=50.0):
    self.poll_calls += 1
    self.poll_timeouts.append(timeout_s)
    if self.poll_blips > 0:
      self.poll_blips -= 1
      raise remote_lib.TransientTransportError("poll blip")
    if self.quick_poll:
      timeout_s = min(timeout_s, 0.005)
    return await self.server.poll_response(timeout_s=timeout_s)


async def wait_until(cond, timeout_s: float = 2.0) -> None:
  """Polls `cond()` on the running loop until it holds or times out."""
  loop = asyncio.get_running_loop()
  deadline = loop.time() + timeout_s
  while not cond():
    if loop.time() > deadline:
      raise AssertionError("Timed out waiting for condition.")
    await asyncio.sleep(0.002)


def ft_session(actors, **kwargs) -> task_session.TaskSession:
  """Builds a fault-tolerant `TaskSession` that evicts into `unreachable`."""
  unreachable: list[remote_lib.ActorHandle] = kwargs.pop("unreachable")
  live: list[remote_lib.ActorHandle] = actors

  def _evict(actor, exc):
    del exc
    unreachable.append(actor)
    if actor in live:
      live.remove(actor)

  kwargs.setdefault("poll_retry_initial_backoff_s", 0.02)
  kwargs.setdefault("poll_retry_max_backoff_s", 0.02)
  return task_session.TaskSession(
      lambda: list(live),
      fault_tolerance_config=task_session.TaskSessionFaultToleranceConfig(
          max_task_retries=kwargs.pop("max_task_retries", 1),
          on_worker_unreachable=_evict,
      ),
      **kwargs,
  )


class TaskSessionTest(parameterized.TestCase):
  """Tests verifying TaskSession and DefaultRouter."""

  def test_task_session_load_balancing_and_out_of_order_streaming(self):
    async def _run_test():
      engine_a = StubWorkerEngine("worker_A", latency=0.04)
      engine_b = StubWorkerEngine("worker_B", latency=0.01)

      handle_a = create_in_process_handle(engine_a)
      handle_b = create_in_process_handle(engine_b)

      results = []
      async with task_session.TaskSession([handle_a, handle_b]) as session:
        await session.submit(
            "req_1", "compute_trajectory", "req_slow_on_A", turns=1
        )
        await session.submit(
            "req_2", "compute_trajectory", "req_fast_on_B", turns=1
        )
        async for res, exc in session.as_completed():
          self.assertIsNone(exc)
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
        task_session.stable_route_hash("prompt_7"), 7682816066691324195
    )
    self.assertEqual(
        task_session.stable_route_hash("route_key_abc"), 8273245243606849686
    )

  def test_stable_route_hash_passes_integers_through(self):
    # Integer keys are explicit shard indices and must not be hashed.
    for shard in (0, 1, 7, 64):
      self.assertEqual(task_session.stable_route_hash(shard), shard)

  def test_stable_route_hash_rejects_negative_integers(self):
    # A negative shard index is always a caller bug: `%` would silently fold it
    # onto a valid actor instead of surfacing the mistake.
    with self.assertRaises(ValueError):
      task_session.stable_route_hash(-1)

  def test_stable_route_hash_separates_adjacent_structured_keys(self):
    # Regression guard: CRC32 is linear over GF(2), so keys differing in one
    # trailing character collapse into the same bucket under small moduli.
    for num_actors in (2, 3, 4, 8):
      buckets = {
          task_session.stable_route_hash(f"prompt_7#{shard}") % num_actors
          for shard in range(num_actors)
      }
      self.assertGreater(
          len(buckets),
          1,
          f"adjacent route keys all collapsed to one bucket for"
          f" num_actors={num_actors}",
      )

  def test_default_router_prompt_affinity_and_custom_route_fn(self):
    async def _run_test():
      engine_0 = StubWorkerEngine("worker_0", latency=0.001)
      engine_1 = StubWorkerEngine("worker_1", latency=0.001)
      handle_0 = create_in_process_handle(engine_0)
      handle_1 = create_in_process_handle(engine_1)

      session = task_session.TaskSession([handle_0, handle_1])

      # Verify sticky routing: identical route_keys consistently route to the
      # same worker
      target_1 = session.route("prompt_sticky_X")
      target_2 = session.route("prompt_sticky_X")
      target_3 = session.route("prompt_sticky_X")
      self.assertIs(target_1, target_2)
      self.assertIs(target_2, target_3)

      # Verify round-robin fallback when route_key is None
      rr_0 = session.route()
      rr_1 = session.route()
      rr_2 = session.route()
      self.assertIs(rr_0, handle_0)
      self.assertIs(rr_1, handle_1)
      self.assertIs(rr_2, handle_0)
      await session.close()

      # Verify custom route_fn
      custom_session = task_session.TaskSession(
          [handle_0, handle_1],
          route_fn=lambda actors, route_key=None: (
              actors[1] if route_key == "pin_to_1" else actors[0]
          ),
      )
      self.assertIs(custom_session.route("pin_to_1"), handle_1)
      self.assertIs(custom_session.route("other"), handle_0)
      await custom_session.close()

    asyncio.run(_run_test())

  def test_task_session_submit_or_route_without_actors_raises(self):
    session = task_session.TaskSession([])
    with self.assertRaisesRegex(
        RuntimeError, "TaskSession has no registered ActorHandles"
    ):
      session.route()

    async def _run_err():
      with self.assertRaisesRegex(
          RuntimeError,
          "TaskSession has no registered ActorHandles",
      ):
        await session.submit("req_1", "any")
      await session.close()

    asyncio.run(_run_err())

  def test_task_session_invalid_actor_type_raises(self):
    bad_actors: Any = [123]
    with self.assertRaisesRegex(
        TypeError, "Expected ActorHandle, got <class 'int'>"
    ):
      task_session.TaskSession(bad_actors)

  def test_as_completed_with_none_results(self):
    """Verifies as_completed handles tasks returning None without deadlocking."""

    class NoneWorker:

      def void_task(self) -> None:
        return None

    async def _run():
      handle = create_in_process_handle(NoneWorker())
      results = []
      async with task_session.TaskSession([handle]) as session:
        await session.submit("req_1", "void_task")
        await session.submit("req_2", "void_task")
        async for res, exc in session.as_completed():
          self.assertIsNone(exc)
          results.append(res)

      self.assertEqual(results, [None, None])

    asyncio.run(_run())

  def test_background_task_exception_propagation(self):
    """Verifies exceptions raised during task execution surface in stream."""

    async def _run():
      engine = StubWorkerEngine("crashing_worker", latency=0.01)
      engine.pause()
      handle = create_in_process_handle(engine)
      async with task_session.TaskSession([handle]) as session:
        await session.submit("req_1", "compute_trajectory", "failing_prompt")
        async for res, exc in session.as_completed():
          self.assertIsNone(res)
          self.assertIsInstance(exc, RuntimeError)
          self.assertIn("currently paused", str(exc))

    asyncio.run(_run())

  def test_task_session_dynamic_enqueuing_and_fault_isolation(self):
    """Verifies dynamic task enqueuing and fault isolation in TaskSession."""

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

      results = []
      errors = []

      async with task_session.TaskSession([w1, w2]) as session:
        await session.submit("req_1", "process", 5)
        await session.submit("req_2", "process", -10)
        await session.submit("req_3", "process", 15)

        async for result, exc in session.as_completed():
          if exc is not None:
            errors.append(str(exc))
            continue

          results.append(result)
          if result == 50:
            await session.submit("req_4", "process", 2)  # will produce 20
          elif result == 150:
            await session.submit("req_5", "process", 3)  # will produce 30

      self.assertLen(errors, 1)
      self.assertIn("Negative value -10 not allowed", errors[0])
      self.assertCountEqual(results, [50, 150, 20, 30])

    asyncio.run(_run())

  def test_task_session_no_double_decrement_on_dispatch_failure(self):
    """Verifies that dispatch failures in submit() do not corrupt _in_flight."""

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
      session = task_session.TaskSession([handle])

      with self.assertRaises(ConnectionError):
        await session.submit("req_fail", "process", 1)

      self.assertEqual(session._in_flight, 0)
      self.assertEmpty(session._dispatched_tasks.get(handle, set()))
      await session.close()

    asyncio.run(_run())

  def test_task_session_deadlock_when_loop_dies_during_second_dispatch(
      self,
  ):
    """Verifies as_completed() does not hang if polling loop dies mid-dispatch."""

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
        await self.task2_dispatch_pause.wait()
        return "task_2"

      async def poll_responses(
          self, timeout_s: float = remote_lib.LONG_POLL_TIMEOUT_S
      ) -> Any:
        if not self.poll1_started.is_set():
          self.poll1_started.set()
          await self.trigger_connection_drop.wait()
          raise ConnectionError("Connection dropped during task1 poll")
        raise ConnectionError("Worker dead on second loop poll")

    async def _run():
      handle = DropDuringSecondDispatchHandle()
      session = task_session.TaskSession([handle])

      await session.submit("req_t1", "task1")
      await handle.poll1_started.wait()

      task2_fut = asyncio.create_task(session.submit("req_t2", "task2"))
      await asyncio.sleep(0.01)

      handle.trigger_connection_drop.set()
      await asyncio.sleep(0.01)

      handle.task2_dispatch_pause.set()
      await task2_fut

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

  def test_task_session_removes_matching_request_id_out_of_order(
      self,
  ):
    """Verifies response.request_id removes the matching ID out of order."""

    class OutOfOrderHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.poll_event = asyncio.Event()
        self.responses = [
            remote_lib.ExecutionResponse(request_id="req_2", result="res_2"),
            remote_lib.ExecutionResponse(request_id="req_1", result="res_1"),
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
        return request_id or ""

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
      session = task_session.TaskSession([handle])

      await session.submit("req_1", "method")
      await session.submit("req_2", "method")

      self.assertEqual(
          set(session._dispatched_tasks[handle]), {"req_1", "req_2"}
      )

      # Allow first poll: req_2 arrives first (out of order!)
      handle.poll_event.set()
      it = session.as_completed()
      res1, unused_exc1 = await it.__anext__()
      self.assertEqual(res1, "res_2")
      self.assertEqual(set(session._dispatched_tasks[handle]), {"req_1"})

      # Allow second poll: req_1 arrives next
      handle.poll_event.set()
      res2, unused_exc2 = await it.__anext__()
      self.assertEqual(res2, "res_1")
      self.assertEqual(session._dispatched_tasks[handle], {})

      await session.close()

    asyncio.run(_run())

  def test_task_session_submit_consumes_route_key_without_forwarding(self):
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
      async with task_session.TaskSession([h1, h2]) as session:
        await session.submit("req_1", "generate", route_key="sticky_key", x=42)
        self.assertEqual(dispatched_kwargs, [{"x": 42}])

    asyncio.run(_run())

  def test_task_session_poll_completed_drains_batch_and_skips_idle_worker(
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
      # Custom route_fn sends all tasks to busy_handle, leaving idle_handle idle
      session = task_session.TaskSession(
          [busy_handle, idle_handle],
          route_fn=lambda actors, route_key=None: actors[0],
      )

      await session.submit("req_1", "generate")
      await session.submit("req_2", "generate")
      await asyncio.sleep(0.02)

      batch = await session.poll_completed(timeout_s=1.0)
      self.assertEqual(batch, [("r1", None), ("r2", None)])
      self.assertEqual(idle_handle.poll_calls, 0)
      self.assertEqual(session._in_flight, 0)

      # Subsequent poll_completed when idle returns [] quickly (<0.5s)
      loop = asyncio.get_running_loop()
      t0 = loop.time()
      empty_batch = await session.poll_completed(timeout_s=2.0)
      self.assertEqual(empty_batch, [])
      self.assertLess(loop.time() - t0, 0.5)

      await session.close()

    asyncio.run(_run())

  def test_task_session_simultaneous_dispatch_and_poll_error_invariant(
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
        await self.crash_event.wait()
        raise RuntimeError("dispatch transport error")

      async def poll_responses(self, timeout_s=50.0):
        del timeout_s
        await self.crash_event.wait()
        raise RuntimeError("poll transport error")

    class ParkedHandle(remote_lib.ActorHandle):

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
        await asyncio.sleep(timeout_s)
        return None

    async def _run():
      handle = CrashingHandle()
      parked = ParkedHandle()
      session = task_session.TaskSession(
          [handle, parked],
          route_fn=lambda actors, key: parked if key == "parked" else handle,
      )

      await session.submit("req_parked", "generate", route_key="parked")
      await session.submit("req_1", "generate")
      submit_task = asyncio.create_task(session.submit("req_2", "generate"))
      await asyncio.sleep(0.01)

      handle.crash_event.set()
      with self.assertRaisesRegex(RuntimeError, "dispatch transport error"):
        await submit_task
      await asyncio.sleep(0.01)

      self.assertEqual(session._in_flight, 1)
      self.assertEqual(session._dispatched_tasks[handle], {})
      await session.close()

    asyncio.run(_run())

  def test_task_session_actors_callable_dynamic_view(self):
    engine_0 = StubWorkerEngine("worker_0", latency=0.001)
    engine_1 = StubWorkerEngine("worker_1", latency=0.001)
    h0 = create_in_process_handle(engine_0)
    h1 = create_in_process_handle(engine_1)

    live_handles: list[remote_lib.ActorHandle] = []
    session = task_session.TaskSession(lambda: list(live_handles))

    # Empty initially
    with self.assertRaisesRegex(
        RuntimeError, "TaskSession has no registered ActorHandles"
    ):
      session.route()

    # Dynamically populate h0, then h1
    live_handles.append(h0)
    self.assertIs(session.route(), h0)

    live_handles.append(h1)
    seen = {session.route(), session.route()}
    self.assertEqual(seen, {h0, h1})

    # Dynamically remove h0 -> only h1 receives calls
    live_handles.remove(h0)
    for _ in range(3):
      self.assertIs(session.route(), h1)

    asyncio.run(session.close())

  def test_task_session_drops_unknown_and_rejects_missing_request_id(self):
    """Unknown IDs are dropped as duplicates; a missing ID is fatal."""

    class CorruptedIdHandle(remote_lib.ActorHandle):

      def __init__(self, response_id: str):
        self._response_id = response_id
        self._script: list[str] = [response_id, "req_1"]

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
        if self._script:
          rid = self._script.pop(0)
          return remote_lib.ExecutionResponse(request_id=rid, result=rid)
        await asyncio.sleep(timeout_s)
        return None

    async def _run():
      # Unknown (e.g. duplicate) request_id: dropped and counted; the real
      # response still completes the task.
      handle = CorruptedIdHandle("unknown_req_id")
      async with task_session.TaskSession([handle]) as session:
        await session.submit("req_1", "generate")
        batch = await session.poll_completed(timeout_s=1.0)
        self.assertEqual(batch, [("req_1", None)])
        self.assertEqual(session.stale_responses_total, 1)

      # Missing request_id is a protocol violation and fails loudly.
      handle = CorruptedIdHandle("")
      async with task_session.TaskSession([handle]) as session:
        await session.submit("req_1", "generate")
        batch = await session.poll_completed(timeout_s=1.0)
        self.assertLen(batch, 1)
        res, exc = batch[0]
        self.assertIsNone(res)
        self.assertIsInstance(exc, RuntimeError)
        self.assertIn("missing request_id", str(exc))

    asyncio.run(_run())

  def test_task_session_invalid_poll_retry_params_raise(self):
    with self.assertRaisesRegex(ValueError, "poll_retry_budget_s"):
      task_session.TaskSession([], poll_retry_budget_s=-1.0)
    with self.assertRaisesRegex(ValueError, "poll_retry_initial_backoff_s"):
      task_session.TaskSession([], poll_retry_initial_backoff_s=0.0)
    with self.assertRaisesRegex(ValueError, "poll_retry_max_backoff_s"):
      task_session.TaskSession(
          [],
          poll_retry_initial_backoff_s=2.0,
          poll_retry_max_backoff_s=1.0,
      )

  def test_task_session_poll_retries_transient_error_and_succeeds(self):
    class FlakyPollHandle(remote_lib.ActorHandle):

      def __init__(self, failures_before_success: int):
        self.failures_remaining = failures_before_success
        self.poll_attempts = 0

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
        del timeout_s
        self.poll_attempts += 1
        if self.failures_remaining > 0:
          self.failures_remaining -= 1
          raise remote_lib.TransientTransportError("transient UNAVAILABLE")
        return remote_lib.ExecutionResponse(
            request_id="req_1", result="recovered_ok"
        )

    async def _run():
      handle = FlakyPollHandle(failures_before_success=3)
      async with task_session.TaskSession(
          [handle],
          poll_retry_budget_s=5.0,
          poll_retry_initial_backoff_s=0.01,
          poll_retry_max_backoff_s=0.02,
      ) as session:
        await session.submit("req_1", "generate")
        batch = await session.poll_completed(timeout_s=2.0)
        self.assertEqual(batch, [("recovered_ok", None)])
        self.assertEqual(handle.poll_attempts, 4)
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_task_session_poll_retry_budget_exhaustion_raises_worker_unreachable(
      self,
  ):
    class AlwaysUnavailableHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.poll_attempts = 0

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
        del timeout_s
        self.poll_attempts += 1
        raise remote_lib.TransientTransportError(
            f"UNAVAILABLE attempt {self.poll_attempts}"
        )

    async def _run():
      handle = AlwaysUnavailableHandle()
      release_healthy = asyncio.Event()

      class HealthyParkedHandle(remote_lib.ActorHandle):

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
          del timeout_s
          await release_healthy.wait()
          return remote_lib.ExecutionResponse(
              request_id="req_healthy", result="healthy_ok"
          )

      healthy = HealthyParkedHandle()
      async with task_session.TaskSession(
          [handle, healthy],
          route_fn=lambda actors, key: actors[0 if key != "healthy" else 1],
          poll_retry_budget_s=0.05,
          poll_retry_initial_backoff_s=0.015,
          poll_retry_max_backoff_s=0.02,
      ) as session:
        await session.submit("req_1", "generate", route_key="fail")
        await session.submit("req_2", "generate", route_key="fail")
        await session.submit("req_healthy", "generate", route_key="healthy")
        self.assertEqual(session._in_flight, 3)

        failed_batch = []
        while len(failed_batch) < 2:
          failed_batch.extend(await session.poll_completed(timeout_s=2.0))
        self.assertLen(failed_batch, 2)
        for res, exc in failed_batch:
          self.assertIsNone(res)
          self.assertIsInstance(exc, task_session.WorkerUnreachableError)
          self.assertIsInstance(
              exc.__cause__, remote_lib.TransientTransportError
          )
        # Failing the 2 tasks on `handle` must decrement _in_flight by exactly 2
        # (from 3 to 1), leaving `healthy`'s in-flight task accounted for.
        self.assertEqual(session._in_flight, 1)
        release_healthy.set()
        healthy_batch = await session.poll_completed(timeout_s=2.0)
        self.assertEqual(healthy_batch, [("healthy_ok", None)])
        self.assertGreaterEqual(handle.poll_attempts, 2)
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_task_session_evicted_actor_fails_immediately_on_transient_error(
      self,
  ):
    class EvictedOnPollHandle(remote_lib.ActorHandle):

      def __init__(self, on_first_poll):
        self._on_first_poll = on_first_poll
        self.poll_attempts = 0

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
        del timeout_s
        self.poll_attempts += 1
        self._on_first_poll()
        raise remote_lib.TransientTransportError("worker evicted mid-poll")

    async def _run():
      live_actors: list[remote_lib.ActorHandle] = []
      handle = EvictedOnPollHandle(on_first_poll=live_actors.clear)
      live_actors.append(handle)
      async with task_session.TaskSession(
          lambda: list(live_actors),
          poll_retry_budget_s=60.0,
          poll_retry_initial_backoff_s=10.0,
          poll_retry_max_backoff_s=10.0,
      ) as session:
        await session.submit("req_1", "generate")
        batch = await session.poll_completed(timeout_s=1.0)
        self.assertLen(batch, 1)
        res, exc = batch[0]
        self.assertIsNone(res)
        self.assertIsInstance(exc, task_session.WorkerUnreachableError)
        self.assertIsInstance(exc.__cause__, remote_lib.TransientTransportError)
        self.assertEqual(handle.poll_attempts, 1)

    asyncio.run(_run())

  def test_task_session_close_during_retry_backoff_exits_promptly(self):
    first_poll_failed = asyncio.Event()

    class SlowBackoffHandle(remote_lib.ActorHandle):

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
        del timeout_s
        first_poll_failed.set()
        raise remote_lib.TransientTransportError("UNAVAILABLE")

    async def _run():
      handle = SlowBackoffHandle()
      session = task_session.TaskSession(
          [handle],
          poll_retry_budget_s=60.0,
          poll_retry_initial_backoff_s=30.0,
          poll_retry_max_backoff_s=30.0,
      )
      await session.submit("req_1", "generate")
      await asyncio.wait_for(first_poll_failed.wait(), timeout=1.0)
      await asyncio.sleep(0.01)
      t0 = asyncio.get_running_loop().time()
      await asyncio.wait_for(session.close(), timeout=1.0)
      self.assertLess(asyncio.get_running_loop().time() - t0, 0.5)

    asyncio.run(_run())

  def test_task_session_successful_poll_resets_outage_budget(self):
    fake_now = [100.0]

    def _fake_monotonic() -> float:
      return fake_now[0]

    class IntermittentHandle(remote_lib.ActorHandle):

      def __init__(self):
        # Burst 1: err1 at t=100.0, err2 at t=102.0 (elapsed=2.0s < 3.0s budget)
        # -> empty poll (None) at t=103.0 resets outage_start.
        # Burst 2: err1 at t=104.0, err2 at t=106.0 (elapsed=2.0s < 3.0s budget;
        # would be 6.0s >= 3.0s budget if None had not reset outage_start!)
        # -> real response succeeds.
        self._script = [
            (0.0, remote_lib.TransientTransportError("burst1_err1")),
            (2.0, remote_lib.TransientTransportError("burst1_err2")),
            (1.0, None),
            (1.0, remote_lib.TransientTransportError("burst2_err1")),
            (2.0, remote_lib.TransientTransportError("burst2_err2")),
            (
                1.0,
                remote_lib.ExecutionResponse(request_id="req_1", result="done"),
            ),
        ]

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
        del timeout_s
        advance_s, item = self._script.pop(0)
        fake_now[0] += advance_s
        if isinstance(item, BaseException):
          raise item
        return item

    async def _run():
      handle = IntermittentHandle()
      with mock.patch.object(
          task_session, "_monotonic", side_effect=_fake_monotonic
      ):
        async with task_session.TaskSession(
            [handle],
            poll_retry_budget_s=3.0,
            poll_retry_initial_backoff_s=0.001,
            poll_retry_max_backoff_s=0.001,
        ) as session:
          await session.submit("req_1", "generate")
          batch = await session.poll_completed(timeout_s=2.0)
          self.assertEqual(batch, [("done", None)])

    asyncio.run(_run())

  def test_task_session_unexpected_cancelled_error_from_handle_fails_tasks(
      self,
  ):
    class SpuriousCancelHandle(remote_lib.ActorHandle):

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
        del timeout_s
        raise asyncio.CancelledError("channel closed underneath poll")

    async def _run():
      handle = SpuriousCancelHandle()
      async with task_session.TaskSession([handle]) as session:
        await session.submit("req_1", "generate")
        batch = await session.poll_completed(timeout_s=1.0)
        self.assertLen(batch, 1)
        res, exc = batch[0]
        self.assertIsNone(res)
        self.assertIsInstance(exc, RuntimeError)
        self.assertIn("unexpected CancelledError", str(exc))
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_task_session_poll_retry_exponent_clamped_and_closed_mid_poll_exits(
      self,
  ):
    # 1. Verify retry_attempts > 1024 does not raise OverflowError when
    # computing 2 ** min(retry_attempts - 1, 60) * initial_backoff_s.
    class HighRetryCountHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.calls = 0

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
        del timeout_s
        self.calls += 1
        if self.calls <= 1030:
          raise remote_lib.TransientTransportError(f"err_{self.calls}")
        return remote_lib.ExecutionResponse(request_id="req_1", result="ok")

    # 2. Verify that if session._closed becomes True while poll_responses is
    # raising TransientTransportError, _poll_worker_loop breaks cleanly without
    # enqueueing a failure into _response_queue.
    class ClosedOnTransientErrorHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.session_ref: Optional[task_session.TaskSession] = None

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
        del timeout_s
        assert self.session_ref is not None
        self.session_ref._closed = True
        raise remote_lib.TransientTransportError("closing transient error")

    async def _run():
      high_retry_handle = HighRetryCountHandle()
      async with task_session.TaskSession(
          [high_retry_handle],
          poll_retry_budget_s=60.0,
          poll_retry_initial_backoff_s=1e-7,
          poll_retry_max_backoff_s=1e-7,
      ) as session:
        await session.submit("req_1", "generate")
        batch = await session.poll_completed(timeout_s=5.0)
        self.assertEqual(batch, [("ok", None)])
        self.assertEqual(high_retry_handle.calls, 1031)

      closed_handle = ClosedOnTransientErrorHandle()
      session2 = task_session.TaskSession([closed_handle])
      closed_handle.session_ref = session2
      await session2.submit("req_2", "generate")
      await asyncio.gather(
          *list(session2._poll_tasks.values()), return_exceptions=True
      )
      self.assertTrue(session2._response_queue.empty())
      await session2.close()

    asyncio.run(_run())

  def test_task_session_invalid_fault_tolerance_args_raise(self):
    h1 = create_in_process_handle(StubWorkerEngine("w1"))
    with self.assertRaisesRegex(ValueError, "max_task_retries"):
      task_session.TaskSessionFaultToleranceConfig(max_task_retries=-1)
    with self.assertRaisesRegex(
        ValueError, "max_task_retries > 0 requires on_worker_unreachable"
    ):
      task_session.TaskSessionFaultToleranceConfig(max_task_retries=1)
    with self.assertRaisesRegex(
        ValueError, "requires a dynamic `actors` callable"
    ):
      task_session.TaskSession(
          [h1],
          fault_tolerance_config=task_session.TaskSessionFaultToleranceConfig(
              max_task_retries=1,
              on_worker_unreachable=lambda actor, exc: None,
          ),
      )
    with self.assertRaisesRegex(
        ValueError, "requires a dynamic `actors` callable"
    ):
      task_session.TaskSession(
          [h1],
          fault_tolerance_config=task_session.TaskSessionFaultToleranceConfig(
              on_worker_unreachable=lambda actor, exc: None,
          ),
      )

  def test_task_session_poll_failure_evicts_worker_and_retries_tasks_on_surviving_worker(
      self,
  ):
    class DeadOnPollHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.dispatched: list[tuple[Optional[str], Optional[str], Any, Any]] = (
            []
        )

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        self.dispatched.append((request_id, method_name, args, kwargs))
        return request_id or ""

      async def poll_responses(self, timeout_s=50.0):
        del timeout_s
        await asyncio.sleep(0.01)
        raise ConnectionError("worker crashed mid-generation")

    async def _run():
      bad_handle = DeadOnPollHandle()
      healthy_engine = StubWorkerEngine("healthy_worker", latency=0.005)
      healthy_handle = create_in_process_handle(healthy_engine)
      live_actors: list[remote_lib.ActorHandle] = [bad_handle, healthy_handle]
      evicted_notifications: list[tuple[remote_lib.ActorHandle, Exception]] = []

      def _on_unreachable(actor, exc):
        evicted_notifications.append((actor, exc))
        if actor in live_actors:
          live_actors.remove(actor)

      async with task_session.TaskSession(
          lambda: list(live_actors),
          route_fn=lambda actors, key: actors[0],
          fault_tolerance_config=task_session.TaskSessionFaultToleranceConfig(
              max_task_retries=3,
              on_worker_unreachable=_on_unreachable,
          ),
      ) as session:
        await session.submit(
            "req_1",
            "compute_trajectory",
            "prompt_A",
            route_key="sticky_A",
            turns=2,
        )
        await session.submit(
            "req_2",
            "compute_trajectory",
            "prompt_B",
            route_key="sticky_B",
            turns=4,
        )

        results = []
        async for res, exc in session.as_completed():
          self.assertIsNone(exc)
          results.append(res)

        self.assertCountEqual(
            results,
            [
                "[healthy_worker] Trajectory for prompt prompt_A (2 turns)",
                "[healthy_worker] Trajectory for prompt prompt_B (4 turns)",
            ],
        )
        self.assertLen(evicted_notifications, 1)
        self.assertIs(evicted_notifications[0][0], bad_handle)
        self.assertIsInstance(evicted_notifications[0][1], ConnectionError)
        self.assertIs(session.route("sticky_A"), healthy_handle)
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_task_session_dispatch_failure_evicts_worker_and_retries_on_surviving_worker(
      self,
  ):
    class DeadOnDispatchHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.dispatch_attempts = 0

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del request_id, method_name, args, kwargs
        self.dispatch_attempts += 1
        raise ConnectionError("connection refused on dispatch")

      async def poll_responses(self, timeout_s=50.0):
        await asyncio.sleep(timeout_s)
        return None

    async def _run():
      bad_handle = DeadOnDispatchHandle()
      healthy_engine = StubWorkerEngine("healthy_worker", latency=0.005)
      healthy_handle = create_in_process_handle(healthy_engine)
      live_actors: list[remote_lib.ActorHandle] = [bad_handle, healthy_handle]
      evicted: list[remote_lib.ActorHandle] = []

      def _on_unreachable(actor, exc):
        del exc
        evicted.append(actor)
        if actor in live_actors:
          live_actors.remove(actor)

      async with task_session.TaskSession(
          lambda: list(live_actors),
          route_fn=lambda actors, key: actors[0],
          fault_tolerance_config=task_session.TaskSessionFaultToleranceConfig(
              max_task_retries=3,
              on_worker_unreachable=_on_unreachable,
          ),
      ) as session:
        req_id = await session.submit(
            "req_send_retry", "compute_trajectory", "prompt_send", turns=1
        )
        self.assertEqual(req_id, "req_send_retry")
        self.assertEqual(bad_handle.dispatch_attempts, 1)
        self.assertEqual(evicted, [bad_handle])

        batch = await session.poll_completed(timeout_s=1.0)
        self.assertEqual(
            batch,
            [(
                "[healthy_worker] Trajectory for prompt prompt_send (1 turns)",
                None,
            )],
        )
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_task_session_default_config_dispatch_failure_does_not_fail_siblings(
      self,
  ):
    class ConditionalDispatchHandle(remote_lib.ActorHandle):

      def __init__(self):
        self._queue: asyncio.Queue[remote_lib.ExecutionResponse] = (
            asyncio.Queue()
        )

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del method_name, args, kwargs
        if request_id == "bad":
          raise ConnectionError("dispatch failed for bad")
        await self._queue.put(
            remote_lib.ExecutionResponse(
                result=f"ok:{request_id}", request_id=request_id
            )
        )
        return request_id or ""

      async def poll_responses(self, timeout_s=50.0):
        return await asyncio.wait_for(self._queue.get(), timeout=timeout_s)

    async def _run():
      handle = ConditionalDispatchHandle()
      async with task_session.TaskSession([handle]) as session:
        await session.submit("good_1", "compute_trajectory", "payload_A")
        # With max_task_retries == 0, payload args/kwargs are cleared after
        # dispatch.
        self.assertEqual(session._dispatched_tasks[handle]["good_1"].args, ())
        self.assertEqual(session._dispatched_tasks[handle]["good_1"].kwargs, {})
        with self.assertRaisesRegex(ConnectionError, "dispatch failed for bad"):
          await session.submit("bad", "compute_trajectory", "payload_B")

        batch = await session.poll_completed(timeout_s=1.0)
        self.assertEqual(batch, [("ok:good_1", None)])
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_task_session_retryable_execution_response_does_not_evict_worker_or_fail_siblings(
      self,
  ):
    class RetryableErrorHandle(remote_lib.ActorHandle):

      def __init__(self):
        self._queue: asyncio.Queue[remote_lib.ExecutionResponse] = (
            asyncio.Queue()
        )

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del method_name, args, kwargs
        if request_id == "req_err":
          await self._queue.put(
              remote_lib.ExecutionResponse(
                  request_id=request_id,
                  error_message="request-level retryable error",
                  error_type="RuntimeError",
                  retryable=True,
              )
          )
        else:
          await self._queue.put(
              remote_lib.ExecutionResponse(
                  request_id=request_id,
                  result=f"ok:{request_id}",
              )
          )
        return request_id or ""

      async def poll_responses(self, timeout_s=50.0):
        return await asyncio.wait_for(self._queue.get(), timeout=timeout_s)

    async def _run():
      handle = RetryableErrorHandle()
      live_actors: list[remote_lib.ActorHandle] = [handle]
      evicted: list[remote_lib.ActorHandle] = []

      async with task_session.TaskSession(
          lambda: list(live_actors),
          fault_tolerance_config=task_session.TaskSessionFaultToleranceConfig(
              max_task_retries=2,
              on_worker_unreachable=lambda actor, exc: evicted.append(actor),
          ),
      ) as session:
        await session.submit("req_err", "compute_trajectory", "p1")
        await session.submit("req_ok", "compute_trajectory", "p2")
        outcomes = []
        async for res, exc in session.as_completed():
          outcomes.append((res, exc))
        self.assertLen(outcomes, 2)
        self.assertIsNone(outcomes[0][0])
        self.assertIsInstance(outcomes[0][1], RuntimeError)
        self.assertEqual(outcomes[1], ("ok:req_ok", None))
        self.assertEmpty(evicted)
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_task_session_exhausting_max_task_retries_fails_terminally(self):
    class AlwaysCrashesOnPollHandle(remote_lib.ActorHandle):

      def __init__(self, name: str):
        self.name = name
        self.dispatched_count = 0

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del method_name, args, kwargs
        self.dispatched_count += 1
        return request_id or ""

      async def poll_responses(self, timeout_s=50.0):
        del timeout_s
        await asyncio.sleep(0.005)
        raise ConnectionError(f"{self.name} crashed")

    async def _run():
      h1 = AlwaysCrashesOnPollHandle("w1")
      h2 = AlwaysCrashesOnPollHandle("w2")
      h3 = AlwaysCrashesOnPollHandle("w3")
      live_actors: list[remote_lib.ActorHandle] = [h1, h2, h3]
      evicted: list[remote_lib.ActorHandle] = []

      def _on_unreachable(actor, exc):
        del exc
        evicted.append(actor)
        if actor in live_actors:
          live_actors.remove(actor)

      async with task_session.TaskSession(
          lambda: list(live_actors),
          route_fn=lambda actors, key: actors[0],
          fault_tolerance_config=task_session.TaskSessionFaultToleranceConfig(
              max_task_retries=1,
              on_worker_unreachable=_on_unreachable,
          ),
      ) as session:
        await session.submit("req_1", "generate")
        batch = await session.poll_completed(timeout_s=1.0)
        self.assertLen(batch, 1)
        res, exc = batch[0]
        self.assertIsNone(res)
        self.assertIsInstance(exc, ConnectionError)
        # Initial attempt on h1 + 1 retry on h2 (h3 is never tried because
        # max_task_retries=1).
        self.assertEqual(h1.dispatched_count, 1)
        self.assertEqual(h2.dispatched_count, 1)
        self.assertEqual(h3.dispatched_count, 0)
        self.assertEqual(evicted, [h1, h2])
        self.assertEqual(session._in_flight, 0)
        self.assertEqual(session.task_retries_total, 1)
        self.assertEqual(session.task_unrecovered_total, 1)

    asyncio.run(_run())

  def test_task_session_concurrent_submits_to_same_dead_worker_retry_cleanly(
      self,
  ):
    class SimultaneousFailOnDispatchHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.fail_event = asyncio.Event()
        self.dispatched_ids: list[str] = []

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del method_name, args, kwargs
        self.dispatched_ids.append(request_id or "")
        await self.fail_event.wait()
        raise ConnectionError("concurrent dispatch failed")

      async def poll_responses(self, timeout_s=50.0):
        await asyncio.sleep(timeout_s)
        return None

    async def _run():
      dead_handle = SimultaneousFailOnDispatchHandle()
      engine_ok = StubWorkerEngine("worker_ok", latency=0.002)
      healthy_handle = create_in_process_handle(engine_ok)
      live_actors: list[remote_lib.ActorHandle] = [dead_handle, healthy_handle]
      evicted: list[remote_lib.ActorHandle] = []

      def _on_unreachable(actor, exc):
        del exc
        evicted.append(actor)
        if actor in live_actors:
          live_actors.remove(actor)

      async with task_session.TaskSession(
          lambda: list(live_actors),
          route_fn=lambda actors, key: actors[0],
          fault_tolerance_config=task_session.TaskSessionFaultToleranceConfig(
              max_task_retries=1,
              on_worker_unreachable=_on_unreachable,
          ),
      ) as session:
        t1 = asyncio.create_task(
            session.submit("req_1", "compute_trajectory", "p1")
        )
        t2 = asyncio.create_task(
            session.submit("req_2", "compute_trajectory", "p2")
        )
        await asyncio.sleep(0.01)
        self.assertEqual(dead_handle.dispatched_ids, ["req_1", "req_2"])

        dead_handle.fail_event.set()
        r1, r2 = await asyncio.gather(t1, t2)
        self.assertEqual((r1, r2), ("req_1", "req_2"))

        outcomes = []
        async for res, exc in session.as_completed():
          self.assertIsNone(exc)
          outcomes.append(res)
        self.assertCountEqual(
            outcomes,
            [
                "[worker_ok] Trajectory for prompt p1 (3 turns)",
                "[worker_ok] Trajectory for prompt p2 (3 turns)",
            ],
        )
        self.assertEqual(engine_ok.call_count, 2)
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_task_session_transient_poll_budget_exhaustion_evicts_and_retries(
      self,
  ):
    class TransientPollFailingHandle(remote_lib.ActorHandle):

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
        del timeout_s
        raise remote_lib.TransientTransportError("UNAVAILABLE: pod deleted")

    async def _run():
      bad_handle = TransientPollFailingHandle()
      engine_ok = StubWorkerEngine("worker_ok", latency=0.002)
      healthy_handle = create_in_process_handle(engine_ok)
      live_actors: list[remote_lib.ActorHandle] = [bad_handle, healthy_handle]
      unreachable_events: list[tuple[remote_lib.ActorHandle, Exception]] = []

      def _on_unreachable(actor, exc):
        unreachable_events.append((actor, exc))
        if actor in live_actors:
          live_actors.remove(actor)

      async with task_session.TaskSession(
          lambda: list(live_actors),
          route_fn=lambda actors, key: actors[0],
          poll_retry_budget_s=0.05,
          poll_retry_initial_backoff_s=0.01,
          poll_retry_max_backoff_s=0.01,
          fault_tolerance_config=task_session.TaskSessionFaultToleranceConfig(
              max_task_retries=1,
              on_worker_unreachable=_on_unreachable,
          ),
      ) as session:
        await session.submit("req_1", "compute_trajectory", "p1")
        outcomes = []
        async for res, exc in session.as_completed():
          self.assertIsNone(exc)
          outcomes.append(res)
        self.assertEqual(
            outcomes,
            ["[worker_ok] Trajectory for prompt p1 (3 turns)"],
        )
        self.assertLen(unreachable_events, 1)
        self.assertIs(unreachable_events[0][0], bad_handle)
        self.assertIsInstance(
            unreachable_events[0][1], task_session.WorkerUnreachableError
        )
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_task_session_does_not_respawn_poller_on_evicted_worker(self):
    class DeadWorkerHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.release_second_dispatch = asyncio.Event()
        self.evicted = False
        self.polls_after_eviction = 0

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del method_name, args, kwargs
        if request_id == "req_1":
          await asyncio.sleep(0.01)
          raise ConnectionError("dispatch req_1 failed")
        await self.release_second_dispatch.wait()
        raise ConnectionError("dispatch req_2 failed")

      async def poll_responses(self, timeout_s=50.0):
        if self.evicted:
          self.polls_after_eviction += 1
        await asyncio.sleep(timeout_s)
        return None

    async def _run():
      dead_handle = DeadWorkerHandle()
      engine_ok = StubWorkerEngine("worker_ok", latency=0.002)
      healthy_handle = create_in_process_handle(engine_ok)
      live_actors: list[remote_lib.ActorHandle] = [dead_handle, healthy_handle]

      def _on_unreachable(actor, exc):
        del exc
        if actor in live_actors:
          live_actors.remove(actor)
          actor.evicted = True

      async with task_session.TaskSession(
          lambda: list(live_actors),
          route_fn=lambda actors, key: actors[0],
          fault_tolerance_config=task_session.TaskSessionFaultToleranceConfig(
              max_task_retries=1,
              on_worker_unreachable=_on_unreachable,
          ),
      ) as session:
        t1 = asyncio.create_task(
            session.submit("req_1", "compute_trajectory", "p1")
        )
        t2 = asyncio.create_task(
            session.submit("req_2", "compute_trajectory", "p2")
        )
        # req_1 fails and evicts dead_handle while req_2 is still dispatching
        # to it; the cancelled poller must not be respawned for dead_handle.
        await asyncio.sleep(0.05)
        self.assertTrue(dead_handle.evicted)
        self.assertNotIn(dead_handle, session._poll_tasks)
        self.assertEqual(dead_handle.polls_after_eviction, 0)

        dead_handle.release_second_dispatch.set()
        self.assertEqual(await asyncio.gather(t1, t2), ["req_1", "req_2"])
        outcomes = [res async for res, _ in session.as_completed()]
        self.assertLen(outcomes, 2)
        self.assertNotIn(dead_handle, session._poll_tasks)
        self.assertEqual(dead_handle.polls_after_eviction, 0)
        self.assertEqual(engine_ok.call_count, 2)
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_task_session_dispatch_cancelled_by_closed_channel_is_retried(self):
    class ChannelClosingDeadHandle(remote_lib.ActorHandle):
      """Mimics grpc.aio: closing the channel cancels in-flight calls."""

      def __init__(self):
        self._pending: set[asyncio.Future[str]] = set()

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del method_name, args, kwargs
        if request_id == "req_1":
          await asyncio.sleep(0.01)
          raise ConnectionError("UNAVAILABLE: connection refused")
        pending = asyncio.get_running_loop().create_future()
        self._pending.add(pending)
        return await pending

      async def poll_responses(self, timeout_s=50.0):
        await asyncio.sleep(timeout_s)
        return None

      async def close(self) -> None:
        for pending in self._pending:
          pending.cancel()

    async def _run():
      dead_handle = ChannelClosingDeadHandle()
      engine_ok = StubWorkerEngine("worker_ok", latency=0.002)
      healthy_handle = create_in_process_handle(engine_ok)
      live_actors: list[remote_lib.ActorHandle] = [dead_handle, healthy_handle]
      unreachable_errors: list[Exception] = []

      def _on_unreachable(actor, exc):
        unreachable_errors.append(exc)
        if actor in live_actors:
          live_actors.remove(actor)
          # Like WorkerRegistry.unregister_handle(): close the evicted handle,
          # which cancels req_2's in-flight dispatch on the same channel.
          asyncio.get_running_loop().create_task(actor.close())

      async with task_session.TaskSession(
          lambda: list(live_actors),
          route_fn=lambda actors, key: actors[0],
          fault_tolerance_config=task_session.TaskSessionFaultToleranceConfig(
              max_task_retries=1,
              on_worker_unreachable=_on_unreachable,
          ),
      ) as session:
        results = await asyncio.gather(
            session.submit("req_1", "compute_trajectory", "p1"),
            session.submit("req_2", "compute_trajectory", "p2"),
        )
        self.assertEqual(results, ["req_1", "req_2"])
        outcomes = []
        async for res, exc in session.as_completed():
          self.assertIsNone(exc)
          outcomes.append(res)
        self.assertCountEqual(
            outcomes,
            [
                "[worker_ok] Trajectory for prompt p1 (3 turns)",
                "[worker_ok] Trajectory for prompt p2 (3 turns)",
            ],
        )
        self.assertTrue(
            any(
                isinstance(exc, task_session.WorkerUnreachableError)
                and isinstance(exc.__cause__, asyncio.CancelledError)
                for exc in unreachable_errors
            )
        )
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_worker_inflight_loads_tracks_active_and_dynamic_actors(self):
    async def _run():
      release = asyncio.Event()

      class _GatedWorker:

        async def compute_trajectory(
            self, prompt_id: str, turns: int = 1
        ) -> str:
          await release.wait()
          return f"{prompt_id}:{turns}"

      handle_0 = create_in_process_handle(_GatedWorker())
      handle_1 = create_in_process_handle(_GatedWorker())
      handle_late = create_in_process_handle(_GatedWorker())
      live_handles = [handle_0, handle_1]

      async with task_session.TaskSession(
          lambda: list(live_handles)
      ) as session:
        self.assertEqual(
            session.worker_inflight_loads(),
            [(handle_0, 0), (handle_1, 0)],
        )
        await session.submit(
            "req_0_a",
            "compute_trajectory",
            "p0",
            turns=1,
            preferred_actor=handle_0,
        )
        await session.submit(
            "req_0_b",
            "compute_trajectory",
            "p1",
            turns=1,
            preferred_actor=handle_0,
        )
        await session.submit(
            "req_1_a",
            "compute_trajectory",
            "p2",
            turns=1,
            preferred_actor=handle_1,
        )
        live_handles.append(handle_late)
        loads_with_late = session.worker_inflight_loads()
        # Evict handle_late (0 tasks) and handle_1 (1 active task still in
        # _dispatched_tasks) and verify both are omitted from live loads.
        live_handles.remove(handle_late)
        live_handles.remove(handle_1)
        loads_after_eviction = session.worker_inflight_loads()
        release.set()

        outcomes = []
        async for res, exc in session.as_completed():
          self.assertIsNone(exc)
          outcomes.append(res)
        loads_after_drain = session.worker_inflight_loads()

      self.assertEqual(
          loads_with_late,
          [(handle_0, 2), (handle_1, 1), (handle_late, 0)],
      )
      self.assertEqual(loads_after_eviction, [(handle_0, 2)])
      self.assertLen(outcomes, 3)
      self.assertEqual(loads_after_drain, [(handle_0, 0)])

    asyncio.run(_run())

  def test_submit_rejects_non_actor_handle_preferred_actor(self):
    async def _run():
      handle_0 = create_in_process_handle(
          StubWorkerEngine("worker_0", latency=0.0)
      )
      async with task_session.TaskSession([handle_0]) as session:
        bad_actor: Any = "not_a_handle"
        with self.assertRaisesRegex(
            TypeError, "Expected ActorHandle for preferred_actor"
        ):
          await session.submit(
              "req_bad",
              "compute_trajectory",
              "p_bad",
              preferred_actor=bad_actor,
          )

    asyncio.run(_run())

  def test_submit_dispatches_to_live_preferred_actor(self):
    async def _run():
      engine_0 = StubWorkerEngine("worker_0", latency=0.0)
      engine_1 = StubWorkerEngine("worker_1", latency=0.0)
      handle_0 = create_in_process_handle(engine_0)
      handle_1 = create_in_process_handle(engine_1)

      async with task_session.TaskSession([handle_0, handle_1]) as session:
        # route_key=0 alone would route to handle_0; preferred_actor=handle_1
        # overrides initial placement while handle_1 is live.
        await session.submit(
            "req_pref",
            "compute_trajectory",
            "p_pref",
            turns=1,
            route_key=0,
            preferred_actor=handle_1,
        )
        outcomes = []
        async for res, exc in session.as_completed():
          self.assertIsNone(exc)
          outcomes.append(res)

      self.assertEqual(
          outcomes, ["[worker_1] Trajectory for prompt p_pref (1 turns)"]
      )
      self.assertEqual(engine_0.call_count, 0)

    asyncio.run(_run())

  def test_submit_falls_back_to_route_key_for_stale_preferred_actor(self):
    async def _run():
      engine_0 = StubWorkerEngine("worker_0", latency=0.0)
      engine_1 = StubWorkerEngine("worker_1", latency=0.0)
      engine_stale = StubWorkerEngine("worker_stale", latency=0.0)
      handle_0 = create_in_process_handle(engine_0)
      handle_1 = create_in_process_handle(engine_1)
      handle_stale = create_in_process_handle(engine_stale)

      async with task_session.TaskSession([handle_0, handle_1]) as session:
        # handle_stale is absent from live actors, so submit falls back to
        # route_key=0 -> handle_0 without touching handle_stale.
        await session.submit(
            "req_stale",
            "compute_trajectory",
            "p_stale",
            turns=1,
            route_key=0,
            preferred_actor=handle_stale,
        )
        outcomes = []
        async for res, exc in session.as_completed():
          self.assertIsNone(exc)
          outcomes.append(res)

      self.assertEqual(
          outcomes, ["[worker_0] Trajectory for prompt p_stale (1 turns)"]
      )
      self.assertEqual(engine_1.call_count, 0)
      self.assertEqual(engine_stale.call_count, 0)

    asyncio.run(_run())

  def test_submit_preferred_actor_retries_via_route_key_on_worker_failure(self):
    async def _run():
      trigger_crash = asyncio.Event()

      class _CrashingPollHandle(remote_lib.ActorHandle):

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
          del timeout_s
          await trigger_crash.wait()
          raise ConnectionError("preferred worker crashed mid-poll")

      bad_handle = _CrashingPollHandle()
      engine_a = StubWorkerEngine("survivor_a", latency=0.0)
      engine_b = StubWorkerEngine("survivor_b", latency=0.0)
      survivor_a = create_in_process_handle(engine_a)
      survivor_b = create_in_process_handle(engine_b)
      live_actors: list[remote_lib.ActorHandle] = [
          bad_handle,
          survivor_a,
          survivor_b,
      ]

      def _evict(actor, exc):
        del exc
        if actor in live_actors:
          live_actors.remove(actor)

      async with task_session.TaskSession(
          lambda: list(live_actors),
          fault_tolerance_config=task_session.TaskSessionFaultToleranceConfig(
              max_task_retries=1,
              on_worker_unreachable=_evict,
          ),
      ) as session:
        for g in range(8):
          await session.submit(
              f"req_g{g}",
              "compute_trajectory",
              f"p_g{g}",
              turns=1,
              route_key=f"traj_p0_g{g}",
              preferred_actor=bad_handle,
          )
        trigger_crash.set()

        outcomes = []
        async for res, exc in session.as_completed():
          self.assertIsNone(exc)
          outcomes.append(res)

      self.assertLen(outcomes, 8)
      self.assertGreater(engine_a.call_count, 0)
      self.assertGreater(engine_b.call_count, 0)
      self.assertEqual(engine_a.call_count + engine_b.call_count, 8)

    asyncio.run(_run())

  def test_task_session_retry_failure_chains_original_worker_error(self):
    class DeadOnPollHandle(remote_lib.ActorHandle):

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
        del timeout_s
        await asyncio.sleep(0.01)
        raise ConnectionError("worker crashed mid-generation")

    async def _run():
      bad_handle = DeadOnPollHandle()
      other_handle = create_in_process_handle(StubWorkerEngine("other"))
      live_actors: list[remote_lib.ActorHandle] = [bad_handle, other_handle]
      route_calls = 0

      def _route(actors, key):
        del key
        nonlocal route_calls
        route_calls += 1
        if route_calls == 1:
          return actors[0]
        # The retry cannot be placed (e.g. the survivors vanished meanwhile).
        raise RuntimeError("no route for retried task")

      def _on_unreachable(actor, exc):
        del exc
        if actor in live_actors:
          live_actors.remove(actor)

      async with task_session.TaskSession(
          lambda: list(live_actors),
          route_fn=_route,
          fault_tolerance_config=task_session.TaskSessionFaultToleranceConfig(
              max_task_retries=1,
              on_worker_unreachable=_on_unreachable,
          ),
      ) as session:
        await session.submit("req_1", "compute_trajectory", "p1")
        batch = await session.poll_completed(timeout_s=1.0)
        self.assertLen(batch, 1)
        res, exc = batch[0]
        self.assertIsNone(res)
        self.assertIsInstance(exc, RuntimeError)
        self.assertIn("no route for retried task", str(exc))
        self.assertIsInstance(exc.__cause__, ConnectionError)
        self.assertEqual(session._in_flight, 0)
        # `route()` raised before any surviving worker was selected for
        # re-dispatch, so no retry was actually re-dispatched.
        self.assertEqual(session.task_retries_total, 0)
        self.assertEqual(session.task_unrecovered_total, 1)

    asyncio.run(_run())

  def test_unrecovered_total_counts_empty_pool(self):
    class DeadOnPollHandle(remote_lib.ActorHandle):

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
        del timeout_s
        await asyncio.sleep(0.01)
        raise ConnectionError("w1 crashed mid-generation")

    async def _run():
      w1 = DeadOnPollHandle()
      w2 = create_in_process_handle(StubWorkerEngine("w2"))
      live_actors: list[remote_lib.ActorHandle] = [w1, w2]

      def _on_unreachable(actor, exc):
        del exc
        if actor in live_actors:
          live_actors.remove(actor)
        # Evict w2 on the next event-loop turn: `_can_retry` still sees `[w2]`
        # and schedules `_retry_or_fail_task`, then `live_actors` becomes empty
        # right before `_retry_or_fail_task` calls `DefaultRouter`.
        asyncio.get_running_loop().call_soon(live_actors.clear)

      async with task_session.TaskSession(
          lambda: list(live_actors),
          fault_tolerance_config=task_session.TaskSessionFaultToleranceConfig(
              max_task_retries=1,
              on_worker_unreachable=_on_unreachable,
          ),
      ) as session:
        # Caller argument error on a non-empty pool is not a worker failure.
        with self.assertRaises(ValueError):
          await session.submit("bad_key", "compute_trajectory", route_key=-1)
        self.assertEqual(session.task_unrecovered_total, 0)

        # Route req_1 deterministically to w1 (`route_key=0`).
        await session.submit("req_1", "compute_trajectory", "p1", route_key=0)
        batch = await session.poll_completed(timeout_s=1.0)
        self.assertLen(batch, 1)
        res, exc = batch[0]
        self.assertIsNone(res)
        self.assertIsInstance(exc, RuntimeError)
        self.assertIn("no registered ActorHandles", str(exc))
        self.assertIsInstance(exc.__cause__, ConnectionError)
        # `live_actors` emptied before `route()` ran on the retry, so zero
        # workers received a re-dispatch.
        self.assertEqual(session.task_retries_total, 0)
        self.assertEqual(session.task_unrecovered_total, 1)

        # Foreground submit onto the now-empty pool also counts as a task
        # failure without a retry.
        with self.assertRaisesRegex(RuntimeError, "no registered ActorHandles"):
          await session.submit("req_2", "compute_trajectory", "p2", route_key=0)
        self.assertEqual(session.task_retries_total, 0)
        self.assertEqual(session.task_unrecovered_total, 2)
        self.assertEqual(session._in_flight, 0)

      # Custom `route_fn` that would otherwise raise `ValueError` on an empty
      # sequence (`min([])`) also gets the empty-pool `RuntimeError` contract.
      async with task_session.TaskSession(
          lambda: [],
          route_fn=lambda actors, key: min(actors, key=id),
      ) as custom_empty_session:
        with self.assertRaisesRegex(RuntimeError, "no registered ActorHandles"):
          await custom_empty_session.submit("req_empty", "compute_trajectory")
        self.assertEqual(custom_empty_session.task_unrecovered_total, 1)
        self.assertEqual(custom_empty_session._in_flight, 0)

    asyncio.run(_run())

  def test_dispatch_blip_on_idle_worker_retries_in_place_without_stall(self):
    """A dispatch that never reached an idle worker is re-sent after backoff."""

    async def _run():
      engine = GatedEngine("w1")
      w1 = BlipHandle(engine, plan={"r1": ["fail"]})
      unreachable: list[remote_lib.ActorHandle] = []
      loop = asyncio.get_running_loop()
      async with ft_session([w1], unreachable=unreachable) as session:
        t0 = loop.time()
        await session.submit("r1", "compute", "r1")
        batch = await session.poll_completed(timeout_s=2.0)
        # No `LONG_POLL_TIMEOUT_S` stall waiting on an empty server queue.
        self.assertLess(loop.time() - t0, 1.0)
        self.assertEqual(batch, [("[w1] r1", None)])
        self.assertEqual(w1.dispatch_log, ["r1", "r1"])
        self.assertEqual(engine.calls, ["r1"])
        self.assertEmpty(unreachable)
        self.assertEmpty(session._retry_states)
        self.assertEqual(session.task_retries_total, 1)
        self.assertEqual(session.stale_responses_total, 0)

    asyncio.run(_run())

  def test_reply_lost_response_delivered_during_backoff_skips_retry_rpc(self):
    """A reply-lost task stays pending, so its response is accepted directly."""

    async def _run():
      engine = GatedEngine("w1")
      engine.gates["r1"] = asyncio.Event()
      w1 = BlipHandle(engine, plan={"r1": ["lose_reply"], "r2": ["gate"]})
      unreachable: list[remote_lib.ActorHandle] = []
      loop = asyncio.get_running_loop()
      async with ft_session(
          [w1],
          unreachable=unreachable,
          poll_retry_initial_backoff_s=0.5,
          poll_retry_max_backoff_s=0.5,
      ) as session:
        # `r2` stays in its first, healthy `dispatch_task` throughout.
        submit_r2 = asyncio.create_task(session.submit("r2", "compute", "r2"))
        await wait_until(lambda: w1.dispatch_log == ["r2"])
        t0 = loop.time()
        submit_r1 = asyncio.create_task(session.submit("r1", "compute", "r1"))
        await wait_until(lambda: w1.failed_dispatches == 1)
        await asyncio.sleep(0.01)
        self.assertFalse(submit_r1.done())
        engine.gates["r1"].set()
        batch = await session.poll_completed(timeout_s=0.3)
        self.assertEqual(batch, [("[w1] r1", None)])
        # Delivery wakes `submit()` immediately via `state.wakeup` without
        # waiting out the 0.5s backoff or sending a retry RPC.
        self.assertEqual(await submit_r1, "r1")
        self.assertLess(loop.time() - t0, 0.35)
        # `r1` was the last retry waiter, so its state is dropped even though
        # `r2` is still dispatching.
        self.assertEmpty(session._retry_states)
        w1.retry_gate.set()
        self.assertEqual(await submit_r2, "r2")
        self.assertEqual(
            await session.poll_completed(timeout_s=1.0), [("[w1] r2", None)]
        )
        self.assertEqual(w1.dispatch_log, ["r2", "r1"])
        self.assertCountEqual(engine.calls, ["r1", "r2"])
        self.assertEqual(session.task_retries_total, 0)
        self.assertEqual(session.stale_responses_total, 0)
        self.assertEmpty(unreachable)
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_retry_landing_after_delivery_drops_duplicate_without_crash(self):
    """A retry that re-executes after delivery yields a dropped duplicate."""

    async def _run():
      engine = GatedEngine("w1")
      engine.gates["r1"] = asyncio.Event()
      w1 = BlipHandle(engine, plan={"r1": ["lose_reply", "gate"]})
      unreachable: list[remote_lib.ActorHandle] = []
      async with ft_session(
          [w1],
          unreachable=unreachable,
          poll_retry_initial_backoff_s=0.01,
          poll_retry_max_backoff_s=0.01,
      ) as session:
        submit_r1 = asyncio.create_task(session.submit("r1", "compute", "r1"))
        # The retry RPC is in flight (held before reaching the server).
        await wait_until(lambda: len(w1.dispatch_log) == 2)
        engine.gates["r1"].set()
        batch = await session.poll_completed(timeout_s=1.0)
        self.assertEqual(batch, [("[w1] r1", None)])
        w1.retry_gate.set()
        self.assertEqual(await submit_r1, "r1")
        # The server keeps no completed-ID state, so the late retry re-ran.
        await wait_until(lambda: len(engine.calls) == 2)
        await asyncio.sleep(0.01)
        # Restart polling with another task; the duplicate must be dropped
        # (not raise), and `r1` is never delivered twice.
        await session.submit("r2", "compute", "r2")
        batch2 = await session.poll_completed(timeout_s=1.0)
        self.assertEqual(batch2, [("[w1] r2", None)])
        self.assertEqual(session.stale_responses_total, 1)
        self.assertEqual(session._in_flight, 0)
        self.assertEmpty(unreachable)

    asyncio.run(_run())

  def test_concurrent_tasks_on_degraded_worker_probe_single_flight(self):
    """At most one follow-up probe dispatch is in flight at a time."""

    async def _run():
      engine = GatedEngine("w1")
      w1 = BlipHandle(engine, fail_all=True)
      unreachable: list[remote_lib.ActorHandle] = []
      async with ft_session(
          [w1],
          unreachable=unreachable,
          poll_retry_budget_s=10.0,
          poll_retry_initial_backoff_s=0.03,
          poll_retry_max_backoff_s=0.03,
      ) as session:
        submits = [
            asyncio.create_task(session.submit(f"r{i}", "compute", f"r{i}"))
            for i in range(16)
        ]
        # First wave: all 16 dispatches were already on the wire.
        await wait_until(lambda: w1.failed_dispatches >= 16)
        w1.measure = True
        # Wait for 1 follow-up probe to fail (`probe_attempts` == 2 < 3); while
        # degraded, at most 1 follow-up probe RPC may be in flight at a time.
        await wait_until(lambda: w1.failed_dispatches >= 17)
        w1.measure = False
        self.assertEqual(w1.max_inflight_measured, 1)
        w1.fail_all = False
        await asyncio.wait_for(asyncio.gather(*submits), timeout=2.0)
        results = []
        while len(results) < 16:
          for res, exc in await session.poll_completed(timeout_s=1.0):
            self.assertIsNone(exc)
            results.append(res)
        self.assertEqual(w1.failed_dispatches, 17)
        self.assertCountEqual(engine.calls, [f"r{i}" for i in range(16)])
        self.assertEmpty(unreachable)
        self.assertEmpty(session._retry_states)

    asyncio.run(_run())

  def test_waiters_resend_concurrently_after_probe_recovers_worker(self):
    """Waiters re-send in parallel after recovery, even past the budget."""

    async def _run():
      engine = GatedEngine("w1")
      labels = [f"r{i}" for i in range(4)]
      w1 = BlipHandle(engine, plan={r: ["fail", "gate"] for r in labels})
      unreachable: list[remote_lib.ActorHandle] = []
      async with ft_session(
          [w1],
          unreachable=unreachable,
          poll_retry_budget_s=0.2,
          poll_retry_initial_backoff_s=0.02,
          poll_retry_max_backoff_s=0.02,
      ) as session:
        submits = [
            asyncio.create_task(session.submit(r, "compute", r)) for r in labels
        ]
        await wait_until(lambda: w1.failed_dispatches == 4)
        w1.measure = True
        # The single probe succeeds only after the outage budget has elapsed.
        await wait_until(lambda: len(w1.dispatch_log) == 5)
        await asyncio.sleep(0.25)
        w1.retry_gate.set()
        await asyncio.wait_for(asyncio.gather(*submits), timeout=1.0)
        results = []
        while len(results) < 4:
          for res, exc in await session.poll_completed(timeout_s=1.0):
            self.assertIsNone(exc)
            results.append(res)
        self.assertCountEqual(results, [f"[w1] {r}" for r in labels])
        # The recovered worker is kept, and the 3 waiters re-sent their
        # dispatches concurrently rather than one at a time.
        self.assertEmpty(unreachable)
        self.assertGreater(w1.max_inflight_measured, 1)
        self.assertLen(w1.dispatch_log, 8)
        self.assertCountEqual(engine.calls, labels)
        self.assertEmpty(session._retry_states)

    asyncio.run(_run())

  def test_delivery_during_backoff_releases_non_probing_waiter(self):
    """A waiter delivered during another task's backoff returns immediately."""

    async def _run():
      engine = GatedEngine("w1")
      engine.gates["b"] = asyncio.Event()
      w1 = BlipHandle(engine, plan={"a": ["fail", "fail"], "b": ["lose_reply"]})
      unreachable: list[remote_lib.ActorHandle] = []
      loop = asyncio.get_running_loop()
      async with ft_session(
          [w1],
          unreachable=unreachable,
          poll_retry_budget_s=10.0,
          poll_retry_initial_backoff_s=0.5,
          poll_retry_max_backoff_s=0.5,
      ) as session:
        # `a` creates the worker's retry state and starts its 0.5s backoff.
        submit_a = asyncio.create_task(session.submit("a", "compute", "a"))
        await wait_until(lambda: w1 in session._retry_states)
        # `b`'s ACK is lost, so it joins the same retry state behind `a`.
        submit_b = asyncio.create_task(session.submit("b", "compute", "b"))
        await wait_until(lambda: w1.failed_dispatches == 2)
        await asyncio.sleep(0.01)
        self.assertEqual(session._retry_states[w1].waiters, 2)
        t0 = loop.time()
        engine.gates["b"].set()
        self.assertEqual(
            await session.poll_completed(timeout_s=0.3), [("[w1] b", None)]
        )
        # `b` returns without waiting out `a`'s backoff and probes.
        self.assertEqual(await asyncio.wait_for(submit_b, timeout=0.2), "b")
        self.assertLess(loop.time() - t0, 0.3)
        self.assertFalse(submit_a.done())
        self.assertEqual(w1.dispatch_log.count("b"), 1)
        self.assertEqual(await asyncio.wait_for(submit_a, timeout=2.0), "a")
        self.assertEqual(
            await session.poll_completed(timeout_s=1.0), [("[w1] a", None)]
        )
        self.assertEqual(w1.dispatch_log.count("a"), 3)
        self.assertEmpty(unreachable)
        self.assertEmpty(session._retry_states)
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_replay_failures_after_recovery_count_toward_task_attempts(self):
    """A waiter whose replays fail after another probe recovers stays bounded."""

    async def _run():
      w_bad = BlipHandle(
          GatedEngine("bad"), plan={"r0": ["fail"], "r1": ["fail"] * 10}
      )
      w_good = BlipHandle(GatedEngine("good"))
      unreachable: list[remote_lib.ActorHandle] = []
      async with ft_session(
          [w_bad, w_good], unreachable=unreachable, poll_retry_budget_s=10.0
      ) as session:
        submit_r0 = asyncio.create_task(
            session.submit("r0", "compute", "r0", route_key=0)
        )
        await wait_until(lambda: w_bad in session._retry_states)
        # `r1` joins `r0`'s outage; `r0`'s probe recovers `w_bad`, but every
        # dispatch of `r1` to `w_bad` keeps failing.
        submit_r1 = asyncio.create_task(
            session.submit("r1", "compute", "r1", route_key=0)
        )
        await asyncio.wait_for(
            asyncio.gather(submit_r0, submit_r1), timeout=2.0
        )
        results = []
        while len(results) < 2:
          for res, exc in await session.poll_completed(timeout_s=1.0):
            self.assertIsNone(exc)
            results.append(res)
        self.assertIn("[good] r1", results)
        # The outage `r1` waited on counts toward its own attempts, so `r1`
        # sends exactly `MAX_IN_PLACE_DISPATCH_ATTEMPTS` dispatches to `w_bad`
        # before `w_bad` is evicted, however the recovering probe interleaves.
        self.assertEqual(
            w_bad.dispatch_log.count("r1"),
            task_session.MAX_IN_PLACE_DISPATCH_ATTEMPTS,
        )
        self.assertEqual(unreachable, [w_bad])
        self.assertEmpty(session._retry_states)
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_reply_lost_after_delivery_creates_no_retry_state(self):
    """A lost ACK for an already delivered task leaves no retry state behind."""

    async def _run():
      engine = GatedEngine("w1")
      w1 = BlipHandle(engine, plan={"r1": ["late_lose_reply"], "r2": ["gate"]})
      unreachable: list[remote_lib.ActorHandle] = []
      async with ft_session([w1], unreachable=unreachable) as session:
        submit_r2 = asyncio.create_task(session.submit("r2", "compute", "r2"))
        await wait_until(lambda: w1.dispatch_log == ["r2"])
        # `r1`'s response is delivered before its ACK is lost, while `r2` is
        # still dispatching on the same worker.
        self.assertEqual(await session.submit("r1", "compute", "r1"), "r1")
        self.assertEqual(
            await session.poll_completed(timeout_s=1.0), [("[w1] r1", None)]
        )
        self.assertNotIn(w1, session._retry_states)
        w1.retry_gate.set()
        self.assertEqual(await submit_r2, "r2")
        self.assertEqual(
            await session.poll_completed(timeout_s=1.0), [("[w1] r2", None)]
        )
        self.assertEqual(w1.dispatch_log, ["r2", "r1"])
        self.assertEqual(session.task_retries_total, 0)
        self.assertEmpty(unreachable)

    asyncio.run(_run())

  def test_ack_after_mid_dispatch_eviction_fails_over_without_reevicting(self):
    """A normal ACK from a worker evicted mid-dispatch fails over once."""

    async def _run():
      w_bad = BlipHandle(GatedEngine("bad"), plan={"r1": ["gate"]})
      w_bad.poll_blips = 1
      w_good = BlipHandle(GatedEngine("good"))
      unreachable: list[remote_lib.ActorHandle] = []
      async with ft_session(
          [w_bad, w_good], unreachable=unreachable, poll_retry_budget_s=0.0
      ) as session:
        submit_r1 = asyncio.create_task(
            session.submit("r1", "compute", "r1", route_key=0)
        )
        # The poller's blip evicts `w_bad` while `r1`'s dispatch is in flight.
        await wait_until(lambda: unreachable == [w_bad])
        w_bad.retry_gate.set()
        self.assertEqual(await submit_r1, "r1")
        self.assertEqual(
            await session.poll_completed(timeout_s=1.0), [("[good] r1", None)]
        )
        self.assertEqual(unreachable, [w_bad])
        self.assertNotIn(w_bad, session._poll_tasks)
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_concurrent_batch_on_dead_worker_evicts_once_after_constant_probes(
      self,
  ):
    """K=16 submits send 2 follow-up probes, evict w_bad once, and rescue orphans."""

    async def _run():
      bad_engine = GatedEngine("bad")
      bad_engine.gates["orphan"] = asyncio.Event()
      w_bad = BlipHandle(bad_engine)
      w_good = BlipHandle(GatedEngine("good"))
      unreachable: list[remote_lib.ActorHandle] = []
      loop = asyncio.get_running_loop()
      async with ft_session(
          [w_bad, w_good],
          unreachable=unreachable,
          poll_retry_budget_s=60.0,
          poll_retry_initial_backoff_s=0.02,
          poll_retry_max_backoff_s=0.5,
      ) as session:
        # Park a non-dispatching in-flight task on `w_bad` before it breaks.
        await session.submit("orphan", "compute", "orphan", route_key=0)
        w_bad.fail_all = True
        t0 = loop.time()
        await asyncio.gather(*(
            session.submit(f"r{i}", "compute", f"r{i}", route_key=0)
            for i in range(16)
        ))
        elapsed_s = loop.time() - t0
        results = []
        while len(results) < 17:
          for res, exc in await session.poll_completed(timeout_s=1.0):
            self.assertIsNone(exc)
            results.append(res)
        self.assertCountEqual(
            results,
            ["[good] orphan"] + [f"[good] r{i}" for i in range(16)],
        )
        # 1 (`orphan`) + 16 concurrent Wave-1 RPCs + 2 serialized follow-up
        # probes before `w_bad` is evicted and all 17 tasks run on `w_good`.
        self.assertLen(
            w_bad.dispatch_log,
            1 + 16 + task_session.MAX_IN_PLACE_DISPATCH_ATTEMPTS - 1,
        )
        self.assertLess(elapsed_s, 0.5)
        self.assertEqual(unreachable, [w_bad])
        self.assertNotIn(w_bad, session._poll_tasks)
        self.assertEmpty(session._retry_states)
        self.assertIs(session.route(0), w_good)
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_dispatch_retry_respects_poll_retry_budget(self):
    """A budget shorter than the backoff still ends in one final probe."""

    async def _run():
      w1 = BlipHandle(GatedEngine("w1"), fail_all=True, quick_poll=True)
      unreachable: list[remote_lib.ActorHandle] = []
      loop = asyncio.get_running_loop()
      async with ft_session(
          [w1],
          unreachable=unreachable,
          poll_retry_budget_s=0.05,
          poll_retry_initial_backoff_s=0.5,
          poll_retry_max_backoff_s=0.5,
      ) as session:
        t0 = loop.time()
        with self.assertRaises(task_session.WorkerUnreachableError) as cm:
          await session.submit("r", "compute", "r")
        self.assertIsInstance(
            cm.exception.__cause__, remote_lib.TransientTransportError
        )
        # The 0.5s backoff is clamped to the 0.05s budget and then probed once
        # more; the budget (not the 3-attempt cap) triggers the eviction.
        self.assertLess(loop.time() - t0, 0.3)
        self.assertEqual(w1.failed_dispatches, 2)
        self.assertEqual(unreachable, [w1])
        self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_cancel_or_close_during_probe_backoff_cleans_up_and_hands_off(self):
    """Cancelling or closing during backoff cleans ledger and retry state."""

    async def _run():
      engine = GatedEngine("w1")
      w1 = BlipHandle(engine, fail_all=True)
      unreachable: list[remote_lib.ActorHandle] = []
      async with ft_session(
          [w1],
          unreachable=unreachable,
          poll_retry_budget_s=10.0,
          poll_retry_initial_backoff_s=0.2,
          poll_retry_max_backoff_s=0.2,
      ) as session:
        # 1. Cancelling the first waiter hands probing off to the next one.
        submit_a = asyncio.create_task(session.submit("a", "compute", "a"))
        submit_b = asyncio.create_task(session.submit("b", "compute", "b"))
        await wait_until(lambda: w1.failed_dispatches >= 2)
        await asyncio.sleep(0.01)
        submit_a.cancel()
        with self.assertRaises(asyncio.CancelledError):
          await submit_a
        self.assertNotIn("a", session._dispatched_tasks.get(w1, {}))
        w1.fail_all = False
        await asyncio.wait_for(submit_b, timeout=1.0)
        batch = await session.poll_completed(timeout_s=1.0)
        self.assertEqual(batch, [("[w1] b", None)])
        self.assertEqual(engine.calls, ["b"])
        self.assertEqual(session._in_flight, 0)
        self.assertEmpty(session._retry_states)
        self.assertEmpty(unreachable)

        # 2. Cancelling the sole dispatching task clears `_retry_states[w1]` so
        # no stale `outage_start` lingers, and stops the idle poller.
        w1.fail_all = True
        submit_c = asyncio.create_task(session.submit("c", "compute", "c"))
        await wait_until(lambda: w1.failed_dispatches >= 3)
        await asyncio.sleep(0.01)
        self.assertIn(w1, session._retry_states)
        self.assertIn(w1, session._poll_tasks)
        submit_c.cancel()
        with self.assertRaises(asyncio.CancelledError):
          await submit_c
        self.assertEmpty(session._retry_states)
        self.assertNotIn(w1, session._poll_tasks)
        self.assertEqual(session._in_flight, 0)

        # 3. Closing the session during a 10s backoff wakes `submit()` in <0.5s
        # via `state.wakeup` and raises `RuntimeError`.
        session._poll_retry_initial_backoff_s = 10.0
        session._poll_retry_max_backoff_s = 10.0
        submit_d = asyncio.create_task(session.submit("d", "compute", "d"))
        await wait_until(lambda: w1.failed_dispatches >= 4)
        await asyncio.sleep(0.01)
        t0 = asyncio.get_running_loop().time()
        await asyncio.wait_for(session.close(), timeout=0.5)
        with self.assertRaisesRegex(RuntimeError, "TaskSession is closed"):
          await asyncio.wait_for(submit_d, timeout=0.5)
        self.assertLess(asyncio.get_running_loop().time() - t0, 0.5)

    asyncio.run(_run())

  def test_poll_recovery_zero_timeout_probe_and_external_eviction(self):
    """Covers timeout_s=0.0 post-blip poll probe, external removal, and non-evicting callback."""

    async def _run():
      # 1. `poll_responses(timeout_s=0.0)` recovery probe resets `outage_start`
      # without blocking 50s on a quiet queue, then external removal from
      # `actors_fn()` evicts the worker and fails over its in-flight task.
      eng1 = GatedEngine("w1")
      eng1.gates["q"] = asyncio.Event()
      w1 = BlipHandle(eng1, quick_poll=True)
      w1.poll_blips = 1
      w2 = BlipHandle(GatedEngine("w2"))
      live: list[remote_lib.ActorHandle] = [w1, w2]
      unreachable: list[remote_lib.ActorHandle] = []
      async with ft_session(
          live,
          unreachable=unreachable,
          poll_retry_budget_s=0.08,
          poll_retry_initial_backoff_s=0.01,
          poll_retry_max_backoff_s=0.01,
      ) as session:
        await session.submit("q", "compute", "q", route_key=0)
        await wait_until(lambda: len(w1.poll_timeouts) >= 3)
        self.assertEqual(
            w1.poll_timeouts[:3],
            [
                task_session.LONG_POLL_TIMEOUT_S,
                0.0,
                task_session.LONG_POLL_TIMEOUT_S,
            ],
        )
        self.assertEmpty(unreachable)
        # External removal from `actors_fn()` aborts the poll loop on the next
        # poll cycle and fails `q` over to `w2`.
        live.remove(w1)
        batch = await session.poll_completed(timeout_s=1.0)
        self.assertEqual(batch, [("[w2] q", None)])
        self.assertEqual(unreachable, [w1])
        eng1.gates["q"].set()

      # 2. If `on_worker_unreachable` leaves `actor` in `actors`, `submit()`
      # fails fast instead of re-routing onto the dead actor.
      w_stuck = BlipHandle(GatedEngine("stuck"), fail_all=True)
      async with task_session.TaskSession(
          lambda: [w_stuck],
          poll_retry_budget_s=0.0,
          fault_tolerance_config=task_session.TaskSessionFaultToleranceConfig(
              max_task_retries=2,
              on_worker_unreachable=lambda a, e: None,
          ),
      ) as s_stuck:
        with self.assertRaises(task_session.WorkerUnreachableError):
          await s_stuck.submit("r_stuck", "compute", "r_stuck")
        self.assertEqual(w_stuck.failed_dispatches, 1)

    asyncio.run(_run())

  def test_abandon_in_flight_drains_queued_responses_and_preserves_reuse(self):
    """Verifies cross-run isolation of `abandon_in_flight()` without closing."""

    async def _run():
      eng1 = GatedEngine("w1")
      eng1.gates["run_a_slow"] = asyncio.Event()
      w1 = BlipHandle(eng1, quick_poll=True)
      live: list[remote_lib.ActorHandle] = [w1]
      unreachable: list[remote_lib.ActorHandle] = []

      session = ft_session(
          live,
          unreachable=unreachable,
          poll_retry_budget_s=0.5,
      )
      # Run A: "run_a_fast" finishes and sits unpolled in `_response_queue`
      # while "run_a_slow" stays in flight on `w1`.
      await session.submit("req_fast", "compute", "run_a_fast")
      await session.submit("req_slow", "compute", "run_a_slow")
      await wait_until(lambda: not session._response_queue.empty())

      await session.abandon_in_flight()
      eng1.gates["run_a_slow"].set()
      await asyncio.sleep(0.02)

      self.assertFalse(session._closed)
      self.assertEqual(session._in_flight, 0)
      self.assertEmpty(unreachable)
      self.assertGreaterEqual(session.stale_responses_total, 1)
      self.assertEqual(await session.poll_completed(timeout_s=0.05), [])

      # Run B on the same TaskSession must see zero Run A leftovers.
      await session.submit("req_b", "compute", "run_b_clean")
      batch = await session.poll_completed(timeout_s=1.0)
      self.assertEqual(batch, [("[w1] run_b_clean", None)])
      self.assertEmpty(unreachable)
      await session.abandon_in_flight()
      return session, live

    reused_session, live_handles = asyncio.run(_run())

    async def _run_on_fresh_event_loop():
      # Run C on a brand-new `asyncio.run()` event loop (with a fresh actor
      # handle bound to the new loop) must not raise "Queue is bound to a
      # different event loop" on the reused `TaskSession`.
      live_handles[:] = [BlipHandle(GatedEngine("w1_loop2"), quick_poll=True)]
      await reused_session.submit("req_c", "compute", "run_c_new_loop")
      batch = await reused_session.poll_completed(timeout_s=1.0)
      self.assertEqual(batch, [("[w1_loop2] run_c_new_loop", None)])
      await reused_session.close()

    asyncio.run(_run_on_fresh_event_loop())

  def test_abandon_in_flight_cancels_poller_spawned_during_gather(self):
    """Cancels and drains any poller spawned concurrently during `gather()`."""
    in_poll = asyncio.Event()
    submitted_during_gather = False

    class YieldOnCancelHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.session: task_session.TaskSession | None = None

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
        nonlocal submitted_during_gather
        del timeout_s
        in_poll.set()
        try:
          await asyncio.Future()
        except asyncio.CancelledError:
          if not submitted_during_gather and self.session is not None:
            submitted_during_gather = True
            await self.session.submit("req_concurrent", "generate")
          raise

    async def _run():
      w0 = YieldOnCancelHandle()
      w1 = BlipHandle(GatedEngine("w1"), quick_poll=True)
      session = task_session.TaskSession([w0, w1])
      w0.session = session
      await session.submit("req_0", "generate", preferred_actor=w0)
      await asyncio.wait_for(in_poll.wait(), timeout=1.0)

      await session.abandon_in_flight()
      self.assertTrue(submitted_during_gather)
      self.assertEmpty(session._poll_tasks)
      self.assertEqual(session._in_flight, 0)
      await session.close()

    asyncio.run(_run())

  def test_abandon_in_flight_wakes_parked_poller_and_leaves_new_queue_empty(
      self,
  ):
    class HangingPollHandle(remote_lib.ActorHandle):

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
        del timeout_s
        await asyncio.Future()

    async def _run():
      w0 = HangingPollHandle()
      session = task_session.TaskSession([w0])
      await session.submit("req_0", "generate", preferred_actor=w0)
      parked_poll = asyncio.create_task(session.poll_completed())
      await asyncio.sleep(0)

      await session.abandon_in_flight()
      res = await asyncio.wait_for(parked_poll, timeout=1.0)
      self.assertEqual(res, [])
      self.assertTrue(session._response_queue.empty())
      await session.close()

    asyncio.run(_run())


if __name__ == "__main__":
  absltest.main()
