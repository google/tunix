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

"""Unit tests for universal Actor Model (`remote_execution.py`)."""

import asyncio
import contextlib
import socket
import threading
import time
from typing import Any, Optional
from unittest import mock
from absl.testing import absltest
import numpy as np
import portpicker
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


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
  """Polls a TCP socket until it is open and accepting connections.

  This prevents race conditions when starting a server in a background thread,
  ensuring the client doesn't attempt to connect before the server is fully
  bound.
  """
  deadline = time.time() + timeout
  while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      s.settimeout(0.2)
      if s.connect_ex((host, port)) == 0:
        return True
    time.sleep(0.05)
  return False


@contextlib.contextmanager
def background_server(engine, port):
  """Starts a gRPC server on an isolated background event loop and thread.

  Because `GrpcRemoteActorHandle.submit()` throws a RuntimeError if called
  from within an active asyncio event loop, tests verifying synchronous
  `submit()`
  must be completely synchronous in the main thread. This context manager runs
  the mocked gRPC server in the background so the main thread remains clean.

  Yields:
    (server, loop) so the caller can drive blocking client calls
    from the main thread while the server runs asynchronously elsewhere.
  """
  server = remote_lib.GrpcRemoteExecutionServer(engine)
  loop = asyncio.new_event_loop()
  ready = threading.Event()

  def _runner():
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.start_serving_async(port))
    loop.call_soon(ready.set)
    loop.run_forever()

  thread = threading.Thread(target=_runner, daemon=True)
  thread.start()
  ready.wait(timeout=10)
  try:
    yield server, loop
  finally:
    asyncio.run_coroutine_threadsafe(server.stop_serving(), loop).result(
        timeout=5
    )
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def create_in_process_handle(
    engine: Any,
) -> remote_lib.InProcessActorHandle:
  """Helper creating an InProcessActorHandle bound to an InProcessRemoteExecutionServer."""
  server = remote_lib.InProcessRemoteExecutionServer(engine)
  return remote_lib.InProcessActorHandle(server)


@contextlib.asynccontextmanager
async def running_grpc_server(
    engine: Any,
    rpc_timeout_s: Optional[float] = remote_lib.RPC_TIMEOUT_S,
):
  """Async context manager managing lifecycle of a GrpcRemoteExecutionServer and handle."""
  port = portpicker.pick_unused_port()
  server = remote_lib.GrpcRemoteExecutionServer(engine)
  await server.start_serving_async(port=port)
  handle = remote_lib.GrpcRemoteActorHandle(
      target_address=f"grpc://localhost:{port}",
      rpc_timeout_s=rpc_timeout_s,
  )
  try:
    yield server, handle
  finally:
    await handle.close()
    await server.stop_serving()


class _LockReturner:

  def get(self):
    return threading.Lock()  # not serializable by (cloud)pickle


class RemoteExecutionTest(absltest.TestCase):
  """Tests verifying ActorHandle and ActorPool dynamic routing."""

  def test_execution_request_serialization(self):
    req = remote_lib.ExecutionRequest(
        request_id="req_100",
        method_name="compute_trajectory",
        args=("prompt_1",),
        kwargs={"turns": 5},
    )
    chunks = list(req.serialize_chunks())
    restored = remote_lib.ExecutionRequest.deserialize_chunks(chunks)
    self.assertEqual(restored.method_name, "compute_trajectory")
    self.assertEqual(restored.args, ("prompt_1",))
    self.assertEqual(restored.kwargs, {"turns": 5})
    self.assertEqual(restored.request_id, "req_100")

    with self.assertRaises(ValueError) as ctx:
      remote_lib.ExecutionRequest(
          method_name="compute",
          args=("data",),
          kwargs={"request_id": "disallowed"},
      )
    self.assertIn("reserved framework parameter", str(ctx.exception))

  def test_execution_request_default_call(self):
    req = remote_lib.ExecutionRequest()
    self.assertEqual(req.method_name, "__call__")

    handle = create_in_process_handle(StubWorkerEngine("actor_call"))

    res = handle.submit()
    self.assertEqual(
        res, "[actor_call] Trajectory for prompt default_prompt (3 turns)"
    )

  def test_actor_handle_sync_and_async_invocation(self):
    async def _run_test():
      engine = StubWorkerEngine("actor_01", latency=0.01)
      handle = create_in_process_handle(engine)

      # Test async execution via asubmit
      res = await handle.asubmit("compute_trajectory", "prompt_a", turns=2)
      self.assertEqual(
          res, "[actor_01] Trajectory for prompt prompt_a (2 turns)"
      )

      # Test sync execution via submit
      status = handle.submit("get_status")
      self.assertIn("count=1", status)

      # Test sync execution of a coroutine from inside an event loop raises
      # RuntimeError
      with self.assertRaisesRegex(
          RuntimeError,
          "submit\\(\\) cannot be called from a running async event loop",
      ):
        handle.submit("compute_trajectory", "prompt_c")

      # Test exception propagation when paused
      await handle.asubmit("pause")
      with self.assertRaises(RuntimeError) as cm:
        await handle.asubmit("compute_trajectory", "prompt_b")
      self.assertIn("currently paused", str(cm.exception))

    asyncio.run(_run_test())

  def test_server_register_instance(self):
    server = remote_lib.InProcessRemoteExecutionServer()
    self.assertIsNone(server.bound_instance)

    engine = StubWorkerEngine("dynamic_worker")
    server.register_instance(engine)
    self.assertIsNotNone(server.bound_instance)

    handle = remote_lib.InProcessActorHandle(server)
    self.assertIn("dynamic_worker", handle.submit("get_status"))

  def test_actor_pool_load_balancing_and_streaming(self):
    async def _run_test():
      engine_a = StubWorkerEngine("worker_A", latency=0.04)
      engine_b = StubWorkerEngine("worker_B", latency=0.01)

      server_a = remote_lib.InProcessRemoteExecutionServer(engine_a)
      server_b = remote_lib.InProcessRemoteExecutionServer(engine_b)

      handle_a = remote_lib.InProcessActorHandle(server_a)
      handle_b = remote_lib.InProcessActorHandle(server_b)

      pool = remote_lib.RoutingActorPool([handle_a, handle_b])

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
        remote_lib.stable_route_hash("prompt_7"), 7682816066691324195
    )
    self.assertEqual(
        remote_lib.stable_route_hash("route_key_abc"), 8273245243606849686
    )

  def test_stable_route_hash_passes_integers_through(self):
    # Integer keys are explicit shard indices and must not be hashed.
    for shard in (0, 1, 7, 64):
      self.assertEqual(remote_lib.stable_route_hash(shard), shard)

  def test_stable_route_hash_rejects_negative_integers(self):
    # A negative shard index is always a caller bug: `%` would silently fold it
    # onto a valid actor instead of surfacing the mistake.
    with self.assertRaises(ValueError):
      remote_lib.stable_route_hash(-1)

  def test_stable_route_hash_separates_adjacent_structured_keys(self):
    # Regression guard: CRC32 is linear over GF(2), so keys differing in one
    # trailing character collapse into the same bucket under small moduli.
    for num_actors in (2, 3, 4, 8):
      buckets = {
          remote_lib.stable_route_hash(f"prompt_7#{shard}") % num_actors
          for shard in range(num_actors)
      }
      self.assertGreater(
          len(buckets),
          1,
          "adjacent route keys all collapsed to one bucket for"
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

      pool = remote_lib.RoutingActorPool([handle_0, handle_1])

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

      smart_pool = remote_lib.RoutingActorPool(
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

      method_pool = remote_lib.RoutingActorPool(
          [handle_0, handle_1], router=MethodSpecificRouter()
      )
      res_method = await method_pool.asubmit("kv_cache_aware", "any_prompt")
      self.assertIn("[worker_1]", res_method)

      # Verify router without the method raises TypeError
      class BadRouter:
        pass

      bad_pool = remote_lib.RoutingActorPool(
          [handle_0, handle_1], router=BadRouter()
      )
      with self.assertRaisesRegex(
          TypeError, "Router object .* must provide a method"
      ):
        await bad_pool.asubmit("kv_cache_aware", "any_prompt")

    asyncio.run(_run_test())

  def test_actor_pool_submit_without_actors_raises(self):
    pool = remote_lib.RoutingActorPool()
    with self.assertRaisesRegex(
        RuntimeError, "RoutingActorPool contains no registered ActorHandles"
    ):
      pool.submit("any")

  def test_actor_pool_stream_without_actors_raises(self):
    pool = remote_lib.RoutingActorPool()

    async def _run_pool_err():
      with self.assertRaisesRegex(
          RuntimeError,
          "RoutingActorPool contains no registered ActorHandles",
      ):
        async for _ in pool.as_completed_stream([("any", (), {})]):
          pass

    asyncio.run(_run_pool_err())

  def test_actor_pool_add_invalid_type_raises(self):
    pool = remote_lib.RoutingActorPool()
    with self.assertRaisesRegex(
        TypeError, "Expected str or ActorHandle, got <class 'int'>"
    ):
      pool.add_actor(123)  # type: ignore

  def test_ray_style_remote_decorators(self):
    """Verifies @remote, @grpc, and @stubby decorators turning classes/funcs into actors."""

    @remote_lib.remote(transport="inprocess")
    class DecoratedWorker:

      def __init__(self, name: str):
        self.name = name

      def greet(self, msg: str) -> str:
        return f"Hello {msg} from {self.name}"

    @remote_lib.remote
    def standalone_task(x: int) -> int:
      return x * 10

    @remote_lib.remote
    async def async_standalone_task(x: int) -> int:
      await asyncio.sleep(0.001)
      return x * 20

    @remote_lib.remote("grpc://fake-pod:50051")
    class GrpcWorker:
      pass

    @remote_lib.remote(address="grpc://fake-pod:50051")
    class ExplicitGrpcWorker:
      pass

    @remote_lib.remote(transport="grpc")
    class DynamicWorker:
      pass

    # Verify class actor factory (inprocess)
    actor_handle = DecoratedWorker.remote("WorkerX")
    self.assertIsInstance(actor_handle, remote_lib.InProcessActorHandle)
    self.assertEqual(
        actor_handle.submit("greet", "World"), "Hello World from WorkerX"
    )

    # Verify standalone function task (inprocess)
    self.assertEqual(standalone_task.remote(5), 50)
    self.assertEqual(async_standalone_task.remote(5), 100)

    # Verify coroutine function task when called inside an active async event
    # loop
    async def _verify_in_loop():
      res = await async_standalone_task.remote(7)
      self.assertEqual(res, 140)

    asyncio.run(_verify_in_loop())

    # Verify grpc class actor factory (remote gRPC target via string address)
    grpc_handle = GrpcWorker.remote()

    self.assertIsInstance(grpc_handle, remote_lib.GrpcRemoteActorHandle)
    self.assertEqual(grpc_handle.target_address, "grpc://fake-pod:50051")

    # Verify grpc class actor factory (remote gRPC target via explicit address
    # kwarg)
    explicit_handle = ExplicitGrpcWorker.remote()
    self.assertIsInstance(explicit_handle, remote_lib.GrpcRemoteActorHandle)
    self.assertEqual(explicit_handle.target_address, "grpc://fake-pod:50051")

    # Verify late address binding (dynamic pod allocation at runtime)
    late_handle = DynamicWorker.remote(address="grpc://allocated-pod:50051")
    self.assertIsInstance(late_handle, remote_lib.GrpcRemoteActorHandle)
    self.assertEqual(late_handle.target_address, "grpc://allocated-pod:50051")

    # Verify standalone function with non-inprocess transport raises
    # NotImplementedError
    with self.assertRaises(NotImplementedError):

      @remote_lib.remote("grpc://fake-pod:50051")
      def remote_grpc_func(x: int) -> int:
        return x * 2

  def test_remote_decorator_invalid_transport_raises(self):
    @remote_lib.remote(transport="invalid")
    class BrokenWorker:
      pass

    with self.assertRaisesRegex(ValueError, "Unsupported transport: invalid"):
      BrokenWorker.remote()

  def test_remote_decorator_invalid_target_type_raises(self):
    with self.assertRaisesRegex(
        TypeError, "@remote expects a class or function"
    ):
      remote_lib.remote(12345)

  def test_remote_actor_handle_submit_not_implemented(self):
    handle = remote_lib.RemoteActorHandle("tcp://dummy")
    with self.assertRaisesRegex(
        NotImplementedError,
        "Remote execution over tcp://dummy not initialized.",
    ):
      handle.submit("method")

  def test_remote_actor_handle_asubmit_not_implemented(self):
    handle = remote_lib.RemoteActorHandle("tcp://dummy")

    async def _run():
      with self.assertRaisesRegex(
          NotImplementedError,
          "Remote execution over tcp://dummy not initialized.",
      ):
        await handle.asubmit("method")

    asyncio.run(_run())

  def test_real_grpc_tcp_execution(self):
    """Verifies GrpcRemoteExecutionServer and GrpcRemoteActorHandle over physical TCP sockets."""

    async def _run_test():
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]

      engine = StubWorkerEngine("grpc_worker_01", latency=0.01)
      server = remote_lib.GrpcRemoteExecutionServer(engine)
      await server.start_serving_async(port=port)

      try:
        handle = remote_lib.ActorHandle.from_address(f"grpc://localhost:{port}")
        self.assertIsInstance(handle, remote_lib.GrpcRemoteActorHandle)

        res = await handle.asubmit("compute_trajectory", "prompt_grpc", turns=4)
        self.assertEqual(
            res, "[grpc_worker_01] Trajectory for prompt prompt_grpc (4 turns)"
        )

        status = await handle.asubmit("get_status")
        self.assertIn("count=1", status)

        # Test GrpcRemoteActorHandle.submit cannot be called inside an event
        # loop
        with self.assertRaisesRegex(
            RuntimeError,
            "GrpcRemoteActorHandle.submit\\(\\) is blocking and cannot be"
            " called from a running event loop",
        ):
          handle.submit("get_status")

        # Test GrpcRemoteExecutionServer.start_serving cannot be called inside
        # an event loop
        with self.assertRaisesRegex(
            RuntimeError,
            "GrpcRemoteExecutionServer.start_serving\\(\\) is blocking and"
            " cannot be called from a running event loop",
        ):
          server.start_serving(port)

        await handle.close()
      finally:
        await server.stop_serving()

    asyncio.run(_run_test())

  def test_server_side_queue_and_polling_in_process(self):
    async def _run():
      engine = StubWorkerEngine("async_worker", latency=0.02)
      handle = create_in_process_handle(engine)

      task_id = await handle.dispatch_task(
          None, "compute_trajectory", "prompt_async", turns=2
      )
      self.assertTrue(task_id.startswith("task_"))

      # Poll response queue asynchronously
      resp = await handle.poll_responses(timeout_s=1.0)
      self.assertIsNotNone(resp)
      self.assertEqual(resp.request_id, task_id)
      self.assertEqual(
          resp.unwrap(),
          "[async_worker] Trajectory for prompt prompt_async (2 turns)",
      )

    asyncio.run(_run())

  def test_dispatch_task_custom_request_id_in_process(self):
    async def _run():
      engine = StubWorkerEngine("custom_id_worker", latency=0.01)
      handle = create_in_process_handle(engine)

      custom_id = "req_rollout_10042"
      req_id = await handle.dispatch_task(
          custom_id, "compute_trajectory", "prompt_custom", turns=1
      )
      self.assertEqual(req_id, custom_id)

      resp = await handle.poll_responses(timeout_s=1.0)
      self.assertIsNotNone(resp)
      self.assertEqual(resp.request_id, custom_id)
      self.assertEqual(
          resp.unwrap(),
          "[custom_id_worker] Trajectory for prompt prompt_custom (1 turns)",
      )

    asyncio.run(_run())

  def test_grpc_dispatch_task_and_long_polling(self):
    """Verifies dispatch_task and poll_responses over real gRPC TCP channels."""

    async def _run_test():
      engine = StubWorkerEngine("grpc_async_worker", latency=0.05)
      async with running_grpc_server(engine) as (_, handle):
        task_id = await handle.dispatch_task(
            None, "compute_trajectory", "prompt_grpc_async", turns=3
        )
        self.assertTrue(task_id.startswith("task_"))

        # Long-poll response queue over gRPC
        resp = await handle.poll_responses(timeout_s=2.0)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.request_id, task_id)
        self.assertEqual(
            resp.unwrap(),
            "[grpc_async_worker] Trajectory for prompt prompt_grpc_async (3"
            " turns)",
        )

    asyncio.run(_run_test())

  def test_grpc_dispatch_task_custom_request_id(self):
    """Verifies client-provided request_id correlates with ExecutionResponse over gRPC."""

    async def _run_test():
      engine = StubWorkerEngine("grpc_custom_worker", latency=0.02)
      async with running_grpc_server(engine) as (_, handle):
        custom_id = "rollout_req_grpc_99"
        req_id = await handle.dispatch_task(
            custom_id, "compute_trajectory", "prompt_grpc_custom", turns=2
        )
        self.assertEqual(req_id, custom_id)

        resp = await handle.poll_responses(timeout_s=2.0)
        self.assertIsNotNone(resp)
        self.assertEqual(resp.request_id, custom_id)
        self.assertEqual(
            resp.unwrap(),
            "[grpc_custom_worker] Trajectory for prompt prompt_grpc_custom (2"
            " turns)",
        )

    asyncio.run(_run_test())

  def test_dispatch_task_method_accepting_domain_request_id(self):
    class WorkerWithRequestIdParam:

      def process(self, data: str, domain_req_id: str) -> str:
        return f"Processed {data} with domain_id={domain_req_id}"

    async def _run():
      handle = create_in_process_handle(WorkerWithRequestIdParam())
      rpc_req_id = await handle.dispatch_task(
          "rpc_req_55", "process", "my_data", domain_req_id="domain_req_99"
      )
      self.assertEqual(rpc_req_id, "rpc_req_55")

      resp = await handle.poll_responses(timeout_s=1.0)
      self.assertIsNotNone(resp)
      self.assertEqual(resp.request_id, "rpc_req_55")
      self.assertEqual(
          resp.unwrap(), "Processed my_data with domain_id=domain_req_99"
      )

    asyncio.run(_run())

  def test_as_completed_stream_with_none_results(self):
    """Verifies as_completed_stream handles tasks returning None without deadlocking."""

    class NoneWorker:

      def void_task(self) -> None:
        return None

    async def _run():
      handle = create_in_process_handle(NoneWorker())
      pool = remote_lib.RoutingActorPool([handle])

      tasks = [("req_1", "void_task", (), {}), ("req_2", "void_task", (), {})]
      results = []
      async for res in pool.as_completed_stream(tasks):
        results.append(res)

      self.assertEqual(results, [None, None])

    asyncio.run(_run())

  def test_polling_timeout_returns_none(self):
    """Verifies poll_responses(timeout_s=0.0) and 0.01 return None when queue is empty."""

    async def _run():
      engine = StubWorkerEngine("slow_worker", latency=2.0)
      # Test in-process handle (QueueEmpty via get_nowait)
      handle_inproc = create_in_process_handle(engine)
      await handle_inproc.dispatch_task(None, "compute_trajectory", "p1")
      self.assertIsNone(await handle_inproc.poll_responses(timeout_s=0.0))
      self.assertIsNone(await handle_inproc.poll_responses(timeout_s=0.01))

      # Test over real gRPC TCP channels
      async with running_grpc_server(engine) as (_, handle_grpc):
        await handle_grpc.dispatch_task(None, "compute_trajectory", "p2")
        self.assertIsNone(await handle_grpc.poll_responses(timeout_s=0.0))
        self.assertIsNone(await handle_grpc.poll_responses(timeout_s=0.01))

    asyncio.run(_run())

  def test_background_task_exception_propagation(self):
    """Verifies exceptions raised during background task execution propagate to client stream."""

    async def _run():
      engine = StubWorkerEngine("crashing_worker", latency=0.01)
      engine.pause()
      handle = create_in_process_handle(engine)
      pool = remote_lib.RoutingActorPool([handle])

      tasks = [("req_1", "compute_trajectory", ("failing_prompt",), {})]
      with self.assertRaisesRegex(RuntimeError, "currently paused"):
        async for _ in pool.as_completed_stream(tasks):
          pass

    asyncio.run(_run())

  def test_execution_response_error_round_trips(self):
    resp = remote_lib.ExecutionResponse(
        error_message="worker 1 is paused!",
        error_type="ValueError",
        traceback="Traceback: ...",
        retryable=True,
    )

    restored = remote_lib.ExecutionResponse.deserialize_chunks(
        resp.serialize_chunks()
    )

    self.assertEqual(restored.error_type, "ValueError")
    self.assertEqual(restored.error_message, "worker 1 is paused!")
    self.assertEqual(restored.traceback, "Traceback: ...")
    self.assertTrue(restored.retryable)
    with self.assertRaises(RuntimeError):
      restored.unwrap()

  def test_execution_response_serialization_success(self):
    res_success = remote_lib.ExecutionResponse(result=42)
    chunks = list(res_success.serialize_chunks())
    restored = remote_lib.ExecutionResponse.deserialize_chunks(chunks)
    self.assertEqual(restored.unwrap(), 42)

  def test_execute_request_captures_traceback(self):
    async def _run():
      engine = StubWorkerEngine("worker_1")
      engine.pause()
      server = remote_lib.InProcessRemoteExecutionServer(engine)
      resp = await server.execute_request(
          remote_lib.ExecutionRequest(
              request_id="req_tb_1", method_name="compute_trajectory"
          )
      )
      self.assertEqual(resp.error_type, "RuntimeError")
      self.assertIn("currently paused", resp.error_message)
      self.assertIsNotNone(resp.traceback)
      self.assertIn("compute_trajectory", resp.traceback)
      self.assertEqual(resp.request_id, "req_tb_1")

    asyncio.run(_run())

  def test_server_error_no_instance_bound_sync(self):
    server = remote_lib.InProcessRemoteExecutionServer()
    req = remote_lib.ExecutionRequest(
        request_id="req_err_1", method_name="any_method"
    )
    resp1 = server.execute_sync_request(req)
    self.assertEqual(resp1.error_type, "InstanceNotBoundError")
    self.assertEqual(
        resp1.error_message, "RemoteExecutionServer has no registered instance."
    )
    self.assertEqual(resp1.request_id, "req_err_1")

  def test_server_error_no_instance_bound_async(self):
    server = remote_lib.InProcessRemoteExecutionServer()
    req = remote_lib.ExecutionRequest(
        request_id="req_err_2", method_name="any_method"
    )

    async def _run_async_err():
      resp6 = await server.execute_request(req)
      self.assertEqual(resp6.error_type, "InstanceNotBoundError")
      self.assertEqual(
          resp6.error_message,
          "RemoteExecutionServer has no registered instance.",
      )
      self.assertEqual(resp6.request_id, "req_err_2")

    asyncio.run(_run_async_err())

  def test_server_error_method_not_found_sync(self):
    server = remote_lib.InProcessRemoteExecutionServer(
        StubWorkerEngine("worker_01")
    )
    resp2 = server.execute_sync_request(
        remote_lib.ExecutionRequest(
            request_id="req_err_3", method_name="missing_method"
        )
    )
    self.assertEqual(resp2.error_type, "AttributeError")
    self.assertEqual(
        resp2.error_message,
        "Method 'missing_method' not found on bound instance.",
    )
    self.assertEqual(resp2.request_id, "req_err_3")

  def test_server_error_method_not_found_async(self):
    server = remote_lib.InProcessRemoteExecutionServer(
        StubWorkerEngine("worker_01")
    )

    async def _run_async_err():
      resp7 = await server.execute_request(
          remote_lib.ExecutionRequest(
              request_id="req_err_4", method_name="missing"
          )
      )
      self.assertEqual(resp7.error_type, "AttributeError")
      self.assertEqual(
          resp7.error_message, "Method 'missing' not found on bound instance."
      )
      self.assertEqual(resp7.request_id, "req_err_4")

    asyncio.run(_run_async_err())

  def test_server_error_exception_during_sync_execution(self):
    class ThrowingWorker:

      def fail(self):
        raise ValueError("Intentional failure")

    server = remote_lib.InProcessRemoteExecutionServer(ThrowingWorker())
    resp4 = server.execute_sync_request(
        remote_lib.ExecutionRequest(request_id="req_err_5", method_name="fail")
    )
    self.assertEqual(resp4.error_type, "ValueError")
    self.assertEqual(resp4.error_message, "Intentional failure")
    self.assertEqual(resp4.request_id, "req_err_5")

  def test_handle_execute_returns_error_for_unserializable_result(self):
    async def _run():
      server = remote_lib.GrpcRemoteExecutionServer(_LockReturner())
      req = remote_lib.ExecutionRequest(
          request_id="req_err_6", method_name="get"
      )

      async def _req_stream():
        for chunk in req.serialize_chunks():
          yield chunk

      resp = await remote_lib.ExecutionResponse.deserialize_async_chunks(
          server._handle_execute(_req_stream(), context=None)
      )
      self.assertIsNotNone(resp)
      self.assertEqual(resp.error_type, "ExecutionResponseSerializationError")
      self.assertIsNotNone(resp.error_message)
      self.assertEqual(resp.request_id, "req_err_6")

    asyncio.run(_run())

  def test_serialize_returns_error_for_unserializable_result(self):
    resp = remote_lib.ExecutionResponse(
        request_id="req_unser_chunks", result=_LockReturner().get()
    )
    chunks = list(resp.serialize_chunks(chunk_size=1024))
    restored = remote_lib.ExecutionResponse.deserialize_chunks(chunks)
    self.assertEqual(restored.request_id, "req_unser_chunks")
    self.assertEqual(restored.error_type, "ExecutionResponseSerializationError")
    self.assertIn("failed to serialize result", restored.error_message)

  def test_handle_poll_responses_returns_error_for_unserializable_result(self):
    async def _run():
      server = remote_lib.GrpcRemoteExecutionServer(_LockReturner())
      request = remote_lib.ExecutionRequest(
          request_id="req_err_7", method_name="get"
      )
      await server.dispatch_task(request)

      resp = await remote_lib.ExecutionResponse.deserialize_async_chunks(
          server._handle_poll_responses(b"", context=None)
      )
      self.assertIsNotNone(resp)
      self.assertEqual(resp.error_type, "ExecutionResponseSerializationError")
      self.assertIsNotNone(resp.error_message)
      self.assertEqual(resp.request_id, "req_err_7")

    asyncio.run(_run())

  def test_grpc_large_payload_round_trips(self):
    async def _run():
      engine = StubWorkerEngine("worker_1")
      async with running_grpc_server(engine) as (_, handle):
        # 8 MiB exceeds gRPC's default ~4 MiB message cap.
        blob = "x" * (8 * 1024 * 1024)
        echoed = await handle.asubmit("kv_cache_aware", blob)
        self.assertTrue(echoed.endswith(blob))
        self.assertGreater(len(echoed), len(blob))

    asyncio.run(_run())

  def test_grpc_asubmit_times_out_on_slow_worker(self):
    async def _run():
      engine = StubWorkerEngine("slow_worker", latency=2.0)
      async with running_grpc_server(engine, rpc_timeout_s=0.3) as (
          _,
          handle,
      ):
        with self.assertRaises(Exception) as cm:
          await handle.asubmit("compute_trajectory", "p", turns=1)
        self.assertIn("deadline", str(cm.exception).lower())

    asyncio.run(_run())

  def test_grpc_sync_submit_survives_repeated_calls(self):
    port = portpicker.pick_unused_port()
    with background_server(StubWorkerEngine("sync_worker"), port):
      handle = remote_lib.GrpcRemoteActorHandle(
          target_address=f"grpc://localhost:{port}"
      )
      try:
        first = handle.submit("compute_trajectory", "p1", turns=1)
        self.assertIn("sync_worker", first)
        # The 2nd blocking call reuses the handle's persistent submit loop and
        # its channel; it must not rebuild the channel or fail on a stale loop.
        second = handle.submit("compute_trajectory", "p2", turns=1)
        self.assertIn("sync_worker", second)
      finally:
        asyncio.run(handle.close())  # tears down the persistent submit loop

  def test_grpc_options_tolerate_client_keepalive_pings(self):
    options = dict(remote_lib._grpc_options())
    self.assertEqual(
        options["grpc.http2.min_recv_ping_interval_without_data_ms"], 5000
    )
    self.assertEqual(options["grpc.http2.max_ping_strikes"], 0)

  def test_grpc_sync_start_serving_actually_serves(self):
    port = portpicker.pick_unused_port()
    server = remote_lib.GrpcRemoteExecutionServer(
        StubWorkerEngine("blocking_worker")
    )
    thread = threading.Thread(
        target=server.start_serving, kwargs={"port": port}, daemon=True
    )
    thread.start()
    try:
      self.assertTrue(
          _wait_for_port("localhost", port),
          "start_serving() never began accepting connections",
      )

      async def _call():
        handle = remote_lib.GrpcRemoteActorHandle(
            target_address=f"grpc://localhost:{port}"
        )
        try:
          return await handle.asubmit("compute_trajectory", "p", turns=1)
        finally:
          await handle.close()

      result = asyncio.run(_call())
      self.assertIn("blocking_worker", result)
    finally:
      serve_loop = server.serve_loop
      if serve_loop is not None:
        asyncio.run_coroutine_threadsafe(
            server.stop_serving(), serve_loop
        ).result(timeout=5)
      thread.join(timeout=5)

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
      pool = remote_lib.RoutingActorPool([w1, w2])

      results = []
      errors = []

      # 1. Client calls execution_session for an initial bulk of tasks
      initial_tasks = [
          ("req_1", "process", (5,), {}),
          (
              "req_2",
              "process",
              (-10,),
              {},
          ),  # This task will throw an exception!
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
      pool = remote_lib.RoutingActorPool([w1])

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
      pool = remote_lib.RoutingActorPool([handle])
      session = remote_lib.PoolExecutionSession(pool)

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
      pool = remote_lib.RoutingActorPool([handle])
      session = remote_lib.PoolExecutionSession(pool)

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
      pool = remote_lib.RoutingActorPool([handle])
      session = remote_lib.PoolExecutionSession(pool)

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
      pool = remote_lib.RoutingActorPool([h1, h2])
      session = remote_lib.PoolExecutionSession(pool)
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
      pool = remote_lib.RoutingActorPool(
          [busy_handle, idle_handle],
          router=lambda actors, method, args, kwargs: actors[0],
      )
      session = remote_lib.PoolExecutionSession(pool)

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
      pool = remote_lib.RoutingActorPool([handle])
      session = remote_lib.PoolExecutionSession(pool)

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

  def test_chunked_serialization_roundtrip_with_numpy_and_empty_buffers(self):
    rng = np.random.default_rng(42)
    packed_tokens = rng.integers(0, 32000, size=(16, 1024), dtype=np.int32)
    empty_prompts = np.zeros((16, 0), dtype=np.int32)
    strided_view = rng.standard_normal((32, 256), dtype=np.float32)[:, ::2]
    advantages = rng.standard_normal((16, 1024), dtype=np.float32)

    payload = {
        "completion_ids": packed_tokens,
        "prompt_ids": empty_prompts,
        "strided_logps": strided_view,
        "advantages": advantages,
        "metadata": {"step": 7, "packed": True},
    }
    req = remote_lib.ExecutionRequest(
        request_id="req_chunked_1",
        method_name="train_step",
        args=(payload,),
        kwargs={"lr": 1e-4},
    )
    chunks = list(req.serialize_chunks(chunk_size=4096))
    self.assertGreater(len(chunks), 10)
    self.assertTrue(all(len(c) <= 4096 for c in chunks))

    restored_req = remote_lib.ExecutionRequest.deserialize_chunks(chunks)
    self.assertEqual(restored_req.request_id, "req_chunked_1")
    self.assertEqual(restored_req.method_name, "train_step")
    self.assertEqual(restored_req.kwargs, {"lr": 1e-4})

    restored_payload = restored_req.args[0]
    np.testing.assert_array_equal(
        restored_payload["completion_ids"], packed_tokens
    )
    self.assertEqual(restored_payload["prompt_ids"].shape, (16, 0))
    np.testing.assert_array_equal(
        restored_payload["strided_logps"], strided_view
    )
    np.testing.assert_array_equal(restored_payload["advantages"], advantages)
    self.assertEqual(restored_payload["metadata"], {"step": 7, "packed": True})
    # Reconstructed arrays backed by released bytearrays must be writable.
    restored_payload["completion_ids"][0, 0] = 999
    self.assertEqual(restored_payload["completion_ids"][0, 0], 999)

  def test_grpc_streaming_payload_exceeds_max_message_bytes(self):
    class PackedBatchWorker:

      def echo_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            "completion_ids": batch["completion_ids"] + 1,
            "prompt_ids": batch["prompt_ids"],
            "advantages": batch["advantages"] * 2.0,
        }

    rng = np.random.default_rng(123)
    # ~4 MiB total payload, while max_message_bytes is capped at 512 KiB and
    # stream_chunk_bytes is 128 KiB. A unary RPC would fail with
    # RESOURCE_EXHAUSTED, whereas chunked streaming succeeds.
    batch = {
        "completion_ids": rng.integers(
            0, 50000, size=(32, 16384), dtype=np.int32
        ),
        "prompt_ids": np.zeros((32, 0), dtype=np.int32),
        "advantages": rng.standard_normal((32, 16384), dtype=np.float32),
    }

    max_msg_bytes = 512 * 1024
    chunk_bytes = 128 * 1024
    port = portpicker.pick_unused_port()
    server = remote_lib.GrpcRemoteExecutionServer(
        PackedBatchWorker(),
        stream_chunk_bytes=chunk_bytes,
        max_message_bytes=max_msg_bytes,
    )

    async def _run_async():
      await server.start_serving_async(port=port)
      handle = remote_lib.GrpcRemoteActorHandle(
          target_address=f"grpc://localhost:{port}",
          stream_chunk_bytes=chunk_bytes,
          max_message_bytes=max_msg_bytes,
      )
      try:
        # 1. Verify bidirectional streaming via asubmit (ExecuteStream)
        out = await handle.asubmit("echo_batch", batch)
        np.testing.assert_array_equal(
            out["completion_ids"], batch["completion_ids"] + 1
        )
        self.assertEqual(out["prompt_ids"].shape, (32, 0))
        np.testing.assert_array_equal(
            out["advantages"], batch["advantages"] * 2.0
        )

        # 2. Verify client-streaming dispatch + server-streaming poll
        ack_id = await handle.dispatch_task("req_stream_1", "echo_batch", batch)
        self.assertEqual(ack_id, "req_stream_1")
        polled = await handle.poll_responses(timeout_s=5.0)
        self.assertIsNotNone(polled)
        self.assertEqual(polled.request_id, "req_stream_1")
        polled_out = polled.unwrap()
        np.testing.assert_array_equal(
            polled_out["completion_ids"], batch["completion_ids"] + 1
        )
        np.testing.assert_array_equal(
            polled_out["advantages"], batch["advantages"] * 2.0
        )
      finally:
        await handle.close()
        await server.stop_serving()

    asyncio.run(_run_async())

  def test_chunk_reassembler_rejects_malformed_truncated_and_overflow_streams(
      self,
  ):
    with self.assertRaisesRegex(
        ValueError, "Cannot deserialize from an empty chunk stream"
    ):
      remote_lib.ExecutionRequest.deserialize_chunks([])

    with self.assertRaisesRegex(ValueError, "Invalid chunk stream manifest"):
      remote_lib.ExecutionRequest.deserialize_chunks([b"not-a-valid-pickle"])

    with self.assertRaisesRegex(
        ValueError, "Invalid header_len in chunk stream manifest"
    ):
      bad_manifest = remote_lib.cloudpickle.dumps((0, ()))
      remote_lib.ExecutionRequest.deserialize_chunks([bad_manifest])

    with self.assertRaisesRegex(
        ValueError, "Invalid buffer_lengths in chunk stream manifest"
    ):
      bad_manifest = remote_lib.cloudpickle.dumps((16, (-4,)))
      remote_lib.ExecutionRequest.deserialize_chunks([bad_manifest])

    req = remote_lib.ExecutionRequest(
        request_id="r1",
        method_name="m",
        args=(np.arange(128, dtype=np.int32),),
    )
    valid_chunks = list(req.serialize_chunks(chunk_size=64))

    # Truncated stream raises ValueError
    with self.assertRaisesRegex(
        ValueError, "Stream ended before all declared buffer bytes"
    ):
      remote_lib.ExecutionRequest.deserialize_chunks(valid_chunks[:-1])

    # Extra trailing bytes raise ValueError
    with self.assertRaisesRegex(
        ValueError, "Received more chunk bytes than declared"
    ):
      remote_lib.ExecutionRequest.deserialize_chunks(valid_chunks + [b"extra"])

  def test_stream_config_validation_rejects_invalid_bounds(self):
    with self.assertRaisesRegex(ValueError, "stream_chunk_bytes must be"):
      remote_lib.GrpcRemoteExecutionServer(stream_chunk_bytes=0)

    with self.assertRaisesRegex(ValueError, "max_message_bytes must be"):
      remote_lib.GrpcRemoteExecutionServer(max_message_bytes=0)

    with self.assertRaisesRegex(
        ValueError, "stream_chunk_bytes .* must not exceed max_message_bytes"
    ):
      remote_lib.GrpcRemoteActorHandle(
          "grpc://localhost:50051",
          stream_chunk_bytes=2 * 1024 * 1024,
          max_message_bytes=1 * 1024 * 1024,
      )

  def test_unserializable_request_arg_raises_immediately_on_client(self):
    async def _run():
      engine = StubWorkerEngine("worker_1")
      async with running_grpc_server(engine) as (_, handle):
        with self.assertRaises(TypeError):
          await handle.asubmit("compute_trajectory", threading.Lock())
        with self.assertRaises(TypeError):
          await handle.dispatch_task(
              "req_bad", "compute_trajectory", threading.Lock()
          )

    asyncio.run(_run())

  def test_interrupted_poll_responses_requeues_completed_response(self):
    async def _run():
      engine = StubWorkerEngine("requeue_worker", latency=0.01)
      server = remote_lib.GrpcRemoteExecutionServer(
          engine,
          stream_chunk_bytes=16,
      )
      await server.dispatch_task(
          remote_lib.ExecutionRequest(
              request_id="req_requeue_1",
              method_name="compute_trajectory",
              args=("prompt_requeue",),
              kwargs={"turns": 2},
          )
      )
      await asyncio.sleep(0.05)

      # Simulate a client disconnecting after reading only the first chunk
      stream = server._handle_poll_responses(b"", context=None)
      first_chunk = await stream.__anext__()
      self.assertNotEmpty(first_chunk)
      await stream.aclose()

      # Subsequent poll must still receive the re-queued response
      recovered = await remote_lib.ExecutionResponse.deserialize_async_chunks(
          server._handle_poll_responses(b"", context=None)
      )
      self.assertIsNotNone(recovered)
      self.assertEqual(recovered.request_id, "req_requeue_1")
      self.assertIn("prompt_requeue", recovered.unwrap())

    asyncio.run(_run())

  def test_grpc_actor_handle_survives_multiple_asyncio_run_loops(self):
    port = portpicker.pick_unused_port()
    with background_server(StubWorkerEngine("multi_loop_worker"), port):
      handle = remote_lib.GrpcRemoteActorHandle(
          target_address=f"grpc://localhost:{port}"
      )
      res1 = asyncio.run(handle.asubmit("compute_trajectory", "p1", turns=1))
      self.assertIn("multi_loop_worker", res1)

      # Second asyncio.run() creates a brand-new event loop; handle must
      # transparently rebind its async channel to the new loop.
      res2 = asyncio.run(handle.asubmit("compute_trajectory", "p2", turns=2))
      self.assertIn("multi_loop_worker", res2)
      asyncio.run(handle.close())

  def test_iter_async_from_sync_chunks_closes_sync_iter_on_early_exit(self):
    closed = False

    def _sync_gen():
      nonlocal closed
      try:
        yield b"chunk_0"
        yield b"chunk_1"
        yield b"chunk_2"
      finally:
        closed = True

    async def _run():
      ait = remote_lib._iter_async_from_sync_chunks(_sync_gen())
      first = await ait.__anext__()
      self.assertEqual(first, b"chunk_0")
      await ait.aclose()
      self.assertTrue(closed)

    asyncio.run(_run())

  def test_iter_serialized_chunks_releases_buffers_on_serialization_failure(
      self,
  ):
    released = []

    class _TrackingBuffer:

      def __init__(self, data: bytes):
        self._mv = memoryview(data)

      def raw(self):
        return self._mv

      def release(self):
        released.append(True)
        self._mv.release()

    def _faulty_dumps(obj, protocol=None, buffer_callback=None):
      if buffer_callback is not None:
        buf = _TrackingBuffer(b"out_of_band_data")
        buffer_callback(buf)
      raise ValueError("Simulated serialization crash")

    with mock.patch.object(
        remote_lib.cloudpickle, "dumps", side_effect=_faulty_dumps
    ):
      with self.assertRaises(ValueError):
        remote_lib._iter_serialized_chunks({"key": "val"})

    self.assertEqual(len(released), 1)

  def test_iter_async_from_sync_chunks_closes_sync_iter_on_cancellation(self):
    closed = False

    def _sync_gen():
      nonlocal closed
      try:
        yield b"chunk_0"
        yield b"chunk_1"
      finally:
        closed = True

    async def _run():
      ait = remote_lib._iter_async_from_sync_chunks(_sync_gen())
      first = await ait.__anext__()
      self.assertEqual(first, b"chunk_0")

      async def _consume():
        await ait.__anext__()

      task = asyncio.create_task(_consume())
      await asyncio.sleep(0.01)
      task.cancel()
      try:
        await task
      except asyncio.CancelledError:
        pass
      try:
        await ait.aclose()
      except Exception:
        pass
      self.assertTrue(closed)

    asyncio.run(_run())

  def test_grpc_remote_actor_handle_closes_old_channel_on_loop_rebind(self):
    async def _run():
      handle = remote_lib.GrpcRemoteActorHandle(
          target_address="grpc://localhost:50051"
      )
      await handle._ensure_async_channel()
      old_channel = handle._channel
      self.assertIsNotNone(old_channel)

      dummy_loop = asyncio.new_event_loop()
      handle._channel_loop = dummy_loop

      await handle._ensure_async_channel()
      new_channel = handle._channel
      self.assertIsNot(new_channel, old_channel)

      await handle.close()

    asyncio.run(_run())




  def test_pool_execution_session_least_loaded_routes_to_idlest_actor(self):
    """least_loaded=True fills actors evenly but keeps route_key affinity."""

    class ParkedHandle(remote_lib.ActorHandle):

      def __init__(self, release: asyncio.Event):
        self.release = release
        self.received = []

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
        assert "route_key" not in kwargs
        self.received.append(request_id)
        return request_id

      async def poll_responses(
          self, timeout_s: float = remote_lib.LONG_POLL_TIMEOUT_S
      ) -> Any:
        await self.release.wait()
        return None

    async def _run():
      release = asyncio.Event()
      handles = [ParkedHandle(release) for _ in range(3)]
      pool = remote_lib.RoutingActorPool(handles)
      session = remote_lib.PoolExecutionSession(pool, least_loaded=True)

      # Keys 0, 3, 6 all hash to actor 0 under plain hash routing.
      await asyncio.gather(
          *(
              session.submit(f"req_{i}", "generate", route_key=3 * i)
              for i in range(9)
          )
      )
      self.assertEqual([len(h.received) for h in handles], [3, 3, 3])

      # Free up actor 2; a new key must go there.
      session._dispatched_tasks[handles[2]].clear()
      await session.submit("req_new", "generate", route_key=100)
      self.assertEqual(handles[2].received[-1], "req_new")

      # A key seen before goes back to its actor, even though it is busier.
      first_actor = next(h for h in handles if "req_0" in h.received)
      await session.submit("req_0_retry", "generate", route_key=0)
      self.assertEqual(first_actor.received[-1], "req_0_retry")

      release.set()
      await session.close()

    asyncio.run(_run())

  def test_routing_actor_pool_membership_operations(self):
    h1 = create_in_process_handle(StubWorkerEngine("w1"))
    h2 = create_in_process_handle(StubWorkerEngine("w2"))
    h3 = create_in_process_handle(StubWorkerEngine("w3"))

    pool = remote_lib.RoutingActorPool([h1, h2])
    self.assertEqual(pool.actors, [h1, h2])

    # Duplicate add_actor is a no-op and returns the handle
    self.assertIs(pool.add_actor(h1), h1)
    self.assertEqual(pool.actors, [h1, h2])

    # add_actor appends new handle
    self.assertIs(pool.add_actor(h3), h3)
    self.assertEqual(pool.actors, [h1, h2, h3])

    # remove_actor returns True when present, False when absent
    self.assertTrue(pool.remove_actor(h1))
    self.assertTrue(pool.remove_actor(h2))
    self.assertFalse(pool.remove_actor(h2))
    self.assertEqual(pool.actors, [h3])

  def test_pool_execution_session_poll_failure_evicts_and_retries_in_flight(
      self,
  ):
    class CrashingPollHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.crash_event = asyncio.Event()

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
        await self.crash_event.wait()
        raise ConnectionError("worker_dead_mid_poll")

    async def _run():
      dead_handle = CrashingPollHandle()
      healthy_handle = create_in_process_handle(
          StubWorkerEngine("healthy_worker", latency=0.01)
      )
      pool = remote_lib.RoutingActorPool([dead_handle, healthy_handle])

      evicted = []
      reentrant_returns = []

      def _on_evicted(actor, exc):
        evicted.append((actor, exc))
        reentrant_returns.append(session.remove_actor(actor))

      session = remote_lib.PoolExecutionSession(
          pool,
          config=remote_lib.PoolSessionConfig(
              evict_on_failure=True,
              retry_on_worker_failure=True,
              max_task_retries=3,
              on_worker_evicted=_on_evicted,
          ),
      )

      # Round-robin sends req_1 to dead_handle and req_2 to healthy_handle
      await session.submit(
          "req_1", "compute_trajectory", "prompt_1", turns=2, route_key=None
      )
      await session.submit(
          "req_2", "compute_trajectory", "prompt_2", turns=2, route_key=None
      )

      # Trigger crash on dead_handle during poll_responses
      dead_handle.crash_event.set()

      results = []
      async for res, exc in session.as_completed():
        self.assertIsNone(exc)
        results.append(res)

      await session.close()

      self.assertEqual(pool.actors, [healthy_handle])
      self.assertLen(evicted, 1)
      self.assertEqual(reentrant_returns, [False])
      self.assertIs(evicted[0][0], dead_handle)
      self.assertIsInstance(evicted[0][1], ConnectionError)
      self.assertEqual(session._in_flight, 0)
      self.assertEmpty(session.pop_failed_tasks())
      self.assertCountEqual(
          results,
          [
              "[healthy_worker] Trajectory for prompt prompt_1 (2 turns)",
              "[healthy_worker] Trajectory for prompt prompt_2 (2 turns)",
          ],
      )

    asyncio.run(_run())

  def test_pool_execution_session_dispatch_failure_evicts_and_retries(self):
    class CrashingDispatchHandle(remote_lib.ActorHandle):

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del request_id, method_name, args, kwargs
        raise ConnectionError("dispatch_failed")

      async def poll_responses(self, timeout_s=50.0):
        await asyncio.sleep(timeout_s)
        return None

    async def _run():
      dead_handle = CrashingDispatchHandle()
      healthy_handle = create_in_process_handle(
          StubWorkerEngine("healthy_worker", latency=0.01)
      )
      pool = remote_lib.RoutingActorPool([dead_handle, healthy_handle])

      evicted = []
      session = remote_lib.PoolExecutionSession(
          pool,
          config=remote_lib.PoolSessionConfig(
              evict_on_failure=True,
              retry_on_worker_failure=True,
              on_worker_evicted=lambda actor, exc: evicted.append((actor, exc)),
          ),
      )

      # First submit hits dead_handle, evicts it, and retries on healthy_handle
      req_id = await session.submit(
          "req_dispatch_retry", "compute_trajectory", "p_retry", turns=1
      )
      self.assertEqual(req_id, "req_dispatch_retry")
      self.assertEqual(pool.actors, [healthy_handle])
      self.assertLen(evicted, 1)
      self.assertIs(evicted[0][0], dead_handle)

      results = []
      async for res, exc in session.as_completed():
        self.assertIsNone(exc)
        results.append(res)

      await session.close()
      self.assertEqual(
          results,
          ["[healthy_worker] Trajectory for prompt p_retry (1 turns)"],
      )
      self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_pool_execution_session_exhausted_retries_populates_failed_tasks(
      self,
  ):
    class AlwaysCrashingPollHandle(remote_lib.ActorHandle):

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
        await asyncio.sleep(0.005)
        raise ConnectionError("poll_crash")

    async def _run():
      h1 = AlwaysCrashingPollHandle()
      h2 = AlwaysCrashingPollHandle()
      pool = remote_lib.RoutingActorPool([h1, h2])
      session = remote_lib.PoolExecutionSession(
          pool,
          config=remote_lib.PoolSessionConfig(
              evict_on_failure=True,
              retry_on_worker_failure=True,
              max_task_retries=3,
          ),
      )

      await session.submit(
          "req_doomed", "compute_trajectory", "p_doom", turns=4, route_key="k1"
      )

      outcomes = []
      async for res, exc in session.as_completed():
        outcomes.append((res, exc))

      await session.close()
      self.assertLen(outcomes, 1)
      self.assertIsNone(outcomes[0][0])
      self.assertIsInstance(outcomes[0][1], ConnectionError)
      self.assertEmpty(pool.actors)
      self.assertEqual(session._in_flight, 0)

      failed = session.pop_failed_tasks()
      self.assertLen(failed, 1)
      req_id, payload, exc = failed[0]
      self.assertEqual(req_id, "req_doomed")
      self.assertEqual(
          payload,
          ("compute_trajectory", ("p_doom",), {"turns": 4, "route_key": "k1"}),
      )
      self.assertIsInstance(exc, ConnectionError)
      self.assertEmpty(session.pop_failed_tasks())

    asyncio.run(_run())

  def test_pool_execution_session_dynamic_add_and_remove_actor(self):
    class SlowHangingHandle(remote_lib.ActorHandle):

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
      slow_handle = SlowHangingHandle()
      pool = remote_lib.RoutingActorPool([slow_handle])
      evicted = []
      session = remote_lib.PoolExecutionSession(
          pool,
          config=remote_lib.PoolSessionConfig(
              evict_on_failure=True,
              retry_on_worker_failure=True,
              on_worker_evicted=lambda actor, exc: evicted.append((actor, exc)),
          ),
      )

      await session.submit("req_1", "compute_trajectory", "p1", turns=1)
      self.assertEqual(session._dispatched_tasks[slow_handle], {"req_1"})

      # Dynamically add a healthy worker and explicitly remove slow_handle
      new_handle = create_in_process_handle(
          StubWorkerEngine("new_worker", latency=0.01)
      )
      session.add_actor(new_handle)
      self.assertTrue(
          session.remove_actor(slow_handle, RuntimeError("manual_eviction"))
      )

      # Re-removing the same handle, or an unknown one, is a no-op: no second
      # on_worker_evicted callback fires and the dead handle is not retained.
      self.assertFalse(session.remove_actor(slow_handle))
      self.assertFalse(session.remove_actor(SlowHangingHandle()))
      self.assertLen(evicted, 1)
      self.assertNotIn(slow_handle, session._known_actors)
      self.assertEqual(session._known_actors, {new_handle})

      # Submit another task which must route to new_handle
      await session.submit("req_2", "compute_trajectory", "p2", turns=1)

      results = []
      async for res, exc in session.as_completed():
        self.assertIsNone(exc)
        results.append(res)

      await session.close()
      self.assertEqual(pool.actors, [new_handle])
      self.assertLen(evicted, 1)
      self.assertIs(evicted[0][0], slow_handle)
      self.assertCountEqual(
          results,
          [
              "[new_worker] Trajectory for prompt p1 (1 turns)",
              "[new_worker] Trajectory for prompt p2 (1 turns)",
          ],
      )
      self.assertEqual(session._in_flight, 0)

    asyncio.run(_run())

  def test_pool_execution_session_max_in_flight_per_worker_limits_concurrency_and_queues(
      self,
  ):
    class GatedHandle(remote_lib.ActorHandle):

      def __init__(self, name: str):
        self.name = name
        self.current_in_flight = 0
        self.peak_in_flight = 0
        self.release_gate = asyncio.Event()
        self._tasks: set[asyncio.Task[None]] = set()
        self._completed: asyncio.Queue[remote_lib.ExecutionResponse] = (
            asyncio.Queue()
        )

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del method_name, kwargs
        self.current_in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.current_in_flight)
        prompt = args[0] if args else ""

        async def _finish():
          await self.release_gate.wait()
          await asyncio.sleep(0.005)
          self.current_in_flight -= 1
          await self._completed.put(
              remote_lib.ExecutionResponse(
                  request_id=request_id or "",
                  result=f"[{self.name}] {prompt}",
              )
          )

        t = asyncio.create_task(_finish())
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return request_id or ""

      async def poll_responses(self, timeout_s=50.0):
        return await asyncio.wait_for(self._completed.get(), timeout=timeout_s)

    async def _run():
      h1 = GatedHandle("w1")
      h2 = GatedHandle("w2")
      pool = remote_lib.RoutingActorPool([h1, h2])
      session = remote_lib.PoolExecutionSession(
          pool,
          config=remote_lib.PoolSessionConfig(max_in_flight_per_worker=2),
      )

      for i in range(6):
        await session.submit(
            f"req_{i}", "compute_trajectory", f"p_{i}", route_key=None
        )

      # Only 2 tasks per worker should be dispatched initially; 2 remain queued.
      self.assertEqual(session._worker_load(h1), 2)
      self.assertEqual(session._worker_load(h2), 2)
      self.assertEqual(session.pending_count, 2)
      self.assertEqual(session.in_flight_count, 6)

      # Release the gate so tasks finish and pending tasks drain.
      h1.release_gate.set()
      h2.release_gate.set()

      results = []
      async for res, exc in session.as_completed():
        self.assertIsNone(exc)
        results.append(res)

      await session.close()
      self.assertLen(results, 6)
      self.assertLessEqual(h1.peak_in_flight, 2)
      self.assertLessEqual(h2.peak_in_flight, 2)
      self.assertEqual(session.pending_count, 0)
      self.assertEqual(session.in_flight_count, 0)

    asyncio.run(_run())

  def test_pool_execution_session_route_key_overflows_to_least_loaded_worker(
      self,
  ):
    async def _run():
      h1 = create_in_process_handle(StubWorkerEngine("w1", latency=0.05))
      h2 = create_in_process_handle(StubWorkerEngine("w2", latency=0.05))
      pool = remote_lib.RoutingActorPool([h1, h2])
      session = remote_lib.PoolExecutionSession(
          pool,
          config=remote_lib.PoolSessionConfig(max_in_flight_per_worker=1),
      )

      # Submit two requests with the exact same route_key while
      # max_in_flight_per_worker=1.
      await session.submit(
          "req_a", "compute_trajectory", "p_a", turns=1, route_key="sticky_key"
      )
      await session.submit(
          "req_b", "compute_trajectory", "p_b", turns=1, route_key="sticky_key"
      )

      # Because the preferred worker is at capacity (1/1), the second request
      # must spill over to the other idle worker rather than queueing.
      self.assertEqual(session._worker_load(h1), 1)
      self.assertEqual(session._worker_load(h2), 1)
      self.assertEqual(session.pending_count, 0)

      results = []
      async for res, exc in session.as_completed():
        self.assertIsNone(exc)
        results.append(res)

      await session.close()
      self.assertLen(results, 2)

    asyncio.run(_run())

  def test_pool_execution_session_worker_failure_respects_survivor_max_in_flight(
      self,
  ):
    class ControlledHandle(remote_lib.ActorHandle):

      def __init__(self, name: str, should_crash: bool = False):
        self.name = name
        self.should_crash = should_crash
        self.current_in_flight = 0
        self.peak_in_flight = 0
        self.trigger = asyncio.Event()
        self._tasks: set[asyncio.Task[None]] = set()
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del method_name, kwargs
        self.current_in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.current_in_flight)
        prompt = args[0] if args else ""

        async def _work():
          await self.trigger.wait()
          await asyncio.sleep(0.005)
          self.current_in_flight -= 1
          if self.should_crash:
            await self._queue.put(ConnectionError("worker_died"))
          else:
            await self._queue.put(
                remote_lib.ExecutionResponse(
                    request_id=request_id or "",
                    result=f"[{self.name}] {prompt}",
                )
            )

        t = asyncio.create_task(_work())
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return request_id or ""

      async def poll_responses(self, timeout_s=50.0):
        item = await asyncio.wait_for(self._queue.get(), timeout=timeout_s)
        if isinstance(item, Exception):
          raise item
        return item

    async def _run():
      dead = ControlledHandle("dead", should_crash=True)
      survivor = ControlledHandle("survivor", should_crash=False)
      pool = remote_lib.RoutingActorPool([dead, survivor])
      session = remote_lib.PoolExecutionSession(
          pool,
          config=remote_lib.PoolSessionConfig(
              evict_on_failure=True,
              retry_on_worker_failure=True,
              max_in_flight_per_worker=1,
          ),
      )

      await session.submit("req_1", "compute_trajectory", "p1", route_key=None)
      await session.submit("req_2", "compute_trajectory", "p2", route_key=None)
      self.assertEqual(session._worker_load(dead), 1)
      self.assertEqual(session._worker_load(survivor), 1)

      # Trigger dead worker crash while survivor is still busy at 1/1 capacity.
      dead.trigger.set()
      await asyncio.sleep(0.02)

      # req_1 must be queued in _pending_queue rather than overloading survivor
      # to 2/1.
      self.assertEqual(session._worker_load(survivor), 1)
      self.assertEqual(session.pending_count, 1)

      # Allow survivor to finish req_2, which drains req_1 onto survivor.
      survivor.trigger.set()

      results = []
      async for res, exc in session.as_completed():
        self.assertIsNone(exc)
        results.append(res)

      await session.close()
      self.assertCountEqual(results, ["[survivor] p1", "[survivor] p2"])
      self.assertEqual(survivor.peak_in_flight, 1)

    asyncio.run(_run())

  def test_pool_execution_session_dynamic_add_actor_drains_pending_queue(self):
    class SlowHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.gate = asyncio.Event()
        self._tasks: set[asyncio.Task[None]] = set()
        self._q: asyncio.Queue[Any] = asyncio.Queue()

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del method_name, kwargs
        prompt = args[0] if args else ""

        async def _work():
          await self.gate.wait()
          await self._q.put(
              remote_lib.ExecutionResponse(
                  request_id=request_id or "",
                  result=f"[slow] {prompt}",
              )
          )

        t = asyncio.create_task(_work())
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return request_id or ""

      async def poll_responses(self, timeout_s=50.0):
        return await asyncio.wait_for(self._q.get(), timeout=timeout_s)

    async def _run():
      slow = SlowHandle()
      pool = remote_lib.RoutingActorPool([slow])
      session = remote_lib.PoolExecutionSession(
          pool,
          config=remote_lib.PoolSessionConfig(max_in_flight_per_worker=1),
      )

      await session.submit("req_1", "compute_trajectory", "p1", route_key=None)
      await session.submit("req_2", "compute_trajectory", "p2", route_key=None)
      self.assertEqual(session._worker_load(slow), 1)
      self.assertEqual(session.pending_count, 1)

      # Add a fast worker while slow is still occupied with req_1.
      fast = create_in_process_handle(StubWorkerEngine("fast", latency=0.01))
      session.add_actor(fast, max_in_flight=1)

      # Wait briefly for the scheduled drain task to dispatch req_2 to fast.
      batch = await session.poll_completed(timeout_s=1.0)
      self.assertLen(batch, 1)
      self.assertEqual(batch[0][0], "[fast] Trajectory for prompt p2 (3 turns)")
      self.assertEqual(session.pending_count, 0)

      slow.gate.set()
      batch2 = await session.poll_completed(timeout_s=1.0)
      self.assertLen(batch2, 1)
      self.assertEqual(batch2[0][0], "[slow] p1")
      await session.close()

    asyncio.run(_run())

  def test_pool_and_session_unified_actor_add_remove_and_pending_workers_hold(
      self,
  ):
    class CrashingPollHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.crash_gate = asyncio.Event()

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
        await self.crash_gate.wait()
        raise ConnectionError("last_worker_crashed")

    async def _run():
      h_dead = CrashingPollHandle()
      pool = remote_lib.RoutingActorPool([h_dead])
      has_pending = True
      session = remote_lib.PoolExecutionSession(
          pool,
          config=remote_lib.PoolSessionConfig(
              evict_on_failure=True,
              retry_on_worker_failure=True,
              has_pending_workers_fn=lambda: has_pending,
          ),
      )

      await session.submit("req_hold", "compute_trajectory", "p_hold", turns=1)
      self.assertEqual(session._worker_load(h_dead), 1)

      # Trigger crash of the ONLY active worker while has_pending_workers_fn()
      # is True.
      h_dead.crash_gate.set()
      batch = await session.poll_completed(timeout_s=0.1)
      self.assertEmpty(batch)
      self.assertEmpty(pool.actors)
      # Task must be held in _pending_queue instead of failing the session
      self.assertEqual(session.pending_count, 1)
      self.assertEqual(session.in_flight_count, 1)

      # Now add replacement worker on `session`.
      h_new = create_in_process_handle(
          StubWorkerEngine("recovered_worker", latency=0.01)
      )
      session.add_actor(h_new, max_in_flight=3)
      self.assertEqual(session._get_worker_limit(h_new), 3)

      batch2 = await session.poll_completed(timeout_s=1.0)
      self.assertLen(batch2, 1)
      self.assertIsNone(batch2[0][1])
      self.assertEqual(
          batch2[0][0],
          "[recovered_worker] Trajectory for prompt p_hold (1 turns)",
      )
      await session.close()

    asyncio.run(_run())

  def test_application_level_task_error_does_not_evict_worker(self):
    class AppErrorHandle(remote_lib.ActorHandle):

      def __init__(self):
        self._q: asyncio.Queue[Any] = asyncio.Queue()

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del method_name, kwargs
        prompt = args[0] if args else ""
        req_id = request_id or ""
        if prompt == "bad_prompt":
          await self._q.put(
              remote_lib.ExecutionResponse(
                  request_id=req_id,
                  error_message="application_level_error",
                  error_type="ValueError",
              )
          )
        else:
          await self._q.put(
              remote_lib.ExecutionResponse(
                  request_id=req_id,
                  result=f"[ok] {prompt}",
              )
          )
        return req_id

      async def poll_responses(self, timeout_s=50.0):
        return await asyncio.wait_for(self._q.get(), timeout=timeout_s)

    async def _run():
      handle = AppErrorHandle()
      pool = remote_lib.RoutingActorPool([handle])
      evicted = []
      session = remote_lib.PoolExecutionSession(
          pool,
          config=remote_lib.PoolSessionConfig(
              evict_on_failure=True,
              retry_on_worker_failure=True,
              on_worker_evicted=lambda h, exc: evicted.append((h, exc)),
          ),
      )

      await session.submit(
          "req_app_err", "compute_trajectory", "bad_prompt", turns=1
      )
      batch = await session.poll_completed(timeout_s=1.0)
      self.assertLen(batch, 1)
      self.assertIsNone(batch[0][0])
      self.assertIsInstance(batch[0][1], RuntimeError)
      self.assertIn("application_level_error", str(batch[0][1]))

      failed = session.pop_failed_tasks()
      self.assertLen(failed, 1)
      self.assertEqual(failed[0][0], "req_app_err")
      self.assertEqual(failed[0][1][0], "compute_trajectory")
      self.assertEqual(failed[0][1][1], ("bad_prompt",))
      self.assertIsInstance(failed[0][2], RuntimeError)

      # Worker must remain active and not be evicted on application-level error.
      self.assertEqual(pool.actors, [handle])
      self.assertEmpty(evicted)

      # Subsequent valid task on the same worker succeeds.
      await session.submit(
          "req_app_ok", "compute_trajectory", "good_prompt", turns=1
      )
      batch2 = await session.poll_completed(timeout_s=1.0)
      self.assertLen(batch2, 1)
      self.assertEqual(batch2[0][0], "[ok] good_prompt")
      self.assertIsNone(batch2[0][1])
      await session.close()

    asyncio.run(_run())

  def test_zero_worker_submit_and_fail_pending_queue_cleanup(self):
    async def _run():
      pool = remote_lib.RoutingActorPool([])
      has_pending = True
      session = remote_lib.PoolExecutionSession(
          pool,
          config=remote_lib.PoolSessionConfig(
              evict_on_failure=True,
              retry_on_worker_failure=True,
              has_pending_workers_fn=lambda: has_pending,
          ),
      )

      # Submitting with empty pool.actors while has_pending_workers_fn is True
      # queues tasks in _pending_queue instead of raising RuntimeError.
      await session.submit("req_q1", "compute_trajectory", "p1", turns=1)
      await session.submit("req_q2", "compute_trajectory", "p2", turns=1)
      self.assertEqual(session.pending_count, 2)
      self.assertEqual(session.in_flight_count, 2)

      # Flip has_pending to False and trigger drain via poll_completed;
      # _fail_pending_queue must drain all queued tasks into pop_failed_tasks()
      # and decrement in_flight_count to 0.
      has_pending = False
      batch = await session.poll_completed(timeout_s=0.5)
      self.assertLen(batch, 2)
      self.assertTrue(
          all(
              res is None and isinstance(exc, RuntimeError)
              for res, exc in batch
          )
      )
      self.assertEqual(session.pending_count, 0)
      self.assertEqual(session.in_flight_count, 0)

      failed = session.pop_failed_tasks()
      self.assertEqual([req_id for req_id, _, _ in failed], ["req_q1", "req_q2"])
      await session.close()

    asyncio.run(_run())

  def test_has_pending_or_completed_work(self):
    class GatedHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.gate = asyncio.Event()
        self._q: asyncio.Queue[Any] = asyncio.Queue()

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del method_name, kwargs
        prompt = args[0] if args else ""

        async def _work():
          await self.gate.wait()
          await self._q.put(
              remote_lib.ExecutionResponse(
                  request_id=request_id or "",
                  result=f"[gated] {prompt}",
              )
          )

        asyncio.create_task(_work())
        return request_id or ""

      async def poll_responses(self, timeout_s=50.0):
        return await asyncio.wait_for(self._q.get(), timeout=timeout_s)

    async def _run():
      handle = GatedHandle()
      pool = remote_lib.RoutingActorPool([handle])
      session = remote_lib.PoolExecutionSession(
          pool,
          config=remote_lib.PoolSessionConfig(max_in_flight_per_worker=1),
      )
      self.assertFalse(session.has_pending_or_completed_work())

      await session.submit("req_1", "compute_trajectory", "p1", route_key=None)
      await session.submit("req_2", "compute_trajectory", "p2", route_key=None)
      self.assertTrue(session.has_pending_or_completed_work())
      self.assertEqual(session._worker_load(handle), 1)
      self.assertEqual(session.pending_count, 1)

      handle.gate.set()
      results = []
      while len(results) < 2:
        batch = await session.poll_completed(timeout_s=1.0)
        results.extend(r for r, _ in batch)
      self.assertCountEqual(results, ["[gated] p1", "[gated] p2"])
      self.assertFalse(session.has_pending_or_completed_work())
      await session.close()

    asyncio.run(_run())

  def test_pool_execution_session_remove_actor_cancels_active_poll_and_requeues(
      self,
  ):
    class TrackingSlowHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.poll_started = asyncio.Event()
        self.poll_cancelled = False

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
        self.poll_started.set()
        try:
          await asyncio.sleep(timeout_s)
        except asyncio.CancelledError:
          self.poll_cancelled = True
          raise
        return None

    async def _run():
      slow = TrackingSlowHandle()
      fast = create_in_process_handle(StubWorkerEngine("fast", latency=0.01))
      pool = remote_lib.RoutingActorPool([slow])
      evicted = []
      session = remote_lib.PoolExecutionSession(
          pool,
          config=remote_lib.PoolSessionConfig(
              evict_on_failure=True,
              retry_on_worker_failure=True,
              on_worker_evicted=lambda actor, exc: evicted.append((actor, exc)),
          ),
      )

      await session.submit("req_1", "compute_trajectory", "p1", turns=1)
      await asyncio.wait_for(slow.poll_started.wait(), timeout=2.0)
      self.assertIn(slow, session._worker_poll_tasks)

      # Add healthy worker and call synchronous remove_actor(slow)
      session.add_actor(fast)
      session.remove_actor(slow, RuntimeError("evicted_by_registry"))

      batch = await session.poll_completed(timeout_s=2.0)
      self.assertLen(batch, 1)
      self.assertEqual(batch[0][0], "[fast] Trajectory for prompt p1 (1 turns)")
      self.assertTrue(slow.poll_cancelled)
      self.assertNotIn(slow, session._worker_poll_tasks)
      self.assertLen(evicted, 1)
      self.assertIs(evicted[0][0], slow)
      await session.close()

    asyncio.run(_run())

  def test_pool_execution_session_ignores_straggler_response_without_popping_dispatched(
      self,
  ):
    class StragglerFirstHandle(remote_lib.ActorHandle):

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
        del method_name, kwargs
        prompt = args[0] if args else ""
        # Enqueue an unrecognized straggler response first, followed by the real
        # response.
        await self._queue.put(
            remote_lib.ExecutionResponse(
                request_id="stale_req_999",
                result="[stale] should_be_ignored",
            )
        )
        await self._queue.put(
            remote_lib.ExecutionResponse(
                request_id=request_id or "",
                result=f"[valid] {prompt}",
            )
        )
        return request_id or ""

      async def poll_responses(self, timeout_s=50.0):
        return await asyncio.wait_for(self._queue.get(), timeout=timeout_s)

    async def _run():
      handle = StragglerFirstHandle()
      pool = remote_lib.RoutingActorPool([handle])
      session = remote_lib.PoolExecutionSession(pool)

      await session.submit("active_req_1", "compute_trajectory", "p_real")
      batch = await session.poll_completed(timeout_s=2.0)

      self.assertLen(batch, 1)
      self.assertEqual(batch[0][0], "[valid] p_real")
      self.assertIsNone(batch[0][1])
      self.assertEqual(session.in_flight_count, 0)
      self.assertEqual(session._worker_load(handle), 0)
      await session.close()

    asyncio.run(_run())

  def test_pool_execution_session_cross_thread_add_and_evict_actor(self):
    class HangingHandle(remote_lib.ActorHandle):

      def __init__(self):
        self.dispatched = threading.Event()

      def submit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def asubmit(self, method_name=None, *args, **kwargs):
        raise NotImplementedError()

      async def dispatch_task(
          self, request_id=None, method_name=None, *args, **kwargs
      ) -> str:
        del method_name, args, kwargs
        self.dispatched.set()
        return request_id or ""

      async def poll_responses(self, timeout_s=50.0):
        await asyncio.sleep(timeout_s)
        return None

    async def _run():
      wedged = HangingHandle()
      healthy = create_in_process_handle(
          StubWorkerEngine("bg_thread_worker", latency=0.01)
      )
      pool = remote_lib.RoutingActorPool([wedged])
      session = remote_lib.PoolExecutionSession(
          pool,
          config=remote_lib.PoolSessionConfig(
              evict_on_failure=True,
              retry_on_worker_failure=True,
          ),
      )

      await session.submit(
          "req_cross", "compute_trajectory", "p_cross", turns=1
      )
      self.assertTrue(wedged.dispatched.wait(timeout=2.0))

      def _bg_thread_mutation():
        session.add_actor(healthy, max_in_flight=2)
        session.remove_actor(wedged, RuntimeError("cross_thread_evict"))

      t = threading.Thread(target=_bg_thread_mutation)
      t.start()

      batch = await session.poll_completed(timeout_s=2.0)
      t.join(timeout=2.0)

      self.assertLen(batch, 1)
      self.assertEqual(
          batch[0][0],
          "[bg_thread_worker] Trajectory for prompt p_cross (1 turns)",
      )
      self.assertEqual(pool.actors, [healthy])
      await session.close()

    asyncio.run(_run())

  def test_pool_execution_session_task_timeout_evicts_wedged_worker_and_retries(
      self,
  ):
    class HungPollHandle(remote_lib.ActorHandle):

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
        # Simulate a SIGSTOP / wedged worker whose RPC never returns on its own.
        await asyncio.sleep(3600.0)
        return None

    async def _run():
      wedged = HungPollHandle()
      healthy = create_in_process_handle(
          StubWorkerEngine("healthy_survivor", latency=0.02)
      )
      pool = remote_lib.RoutingActorPool([wedged, healthy])
      pool.router = lambda actors, *a, **kw: actors[0]
      evicted = []
      async with pool.execution_session(
          config=remote_lib.PoolSessionConfig(
              evict_on_failure=True,
              retry_on_worker_failure=True,
              on_worker_evicted=lambda actor, exc: evicted.append((actor, exc)),
              task_timeout_s=0.15,
          ),
      ) as session:
        self.assertEqual(session.task_timeout_s, 0.15)

        await session.submit(
            "req_to", "compute_trajectory", "p_timeout", turns=1
        )
        batch = await session.poll_completed(timeout_s=2.0)

        self.assertLen(batch, 1)
        self.assertEqual(
            batch[0][0],
            "[healthy_survivor] Trajectory for prompt p_timeout (1 turns)",
        )
        self.assertIsNone(batch[0][1])
        self.assertEqual(pool.actors, [healthy])
        self.assertLen(evicted, 1)
        self.assertIs(evicted[0][0], wedged)
        self.assertIsInstance(evicted[0][1], TimeoutError)

    asyncio.run(_run())

  def test_pool_execution_session_queued_time_does_not_count_toward_task_timeout(
      self,
  ):
    async def _run():
      # Single worker with max_in_flight_per_worker=1 and latency=0.12s.
      # Two tasks take ~0.24s total wall time, but each task spends only ~0.12s
      # in-flight after dispatch (< task_timeout_s=0.20s).
      worker = create_in_process_handle(
          StubWorkerEngine("serial_worker", latency=0.12)
      )
      pool = remote_lib.RoutingActorPool([worker])
      async with pool.execution_session(
          config=remote_lib.PoolSessionConfig(
              evict_on_failure=True,
              retry_on_worker_failure=True,
              max_in_flight_per_worker=1,
              task_timeout_s=0.20,
          ),
      ) as session:
        await session.submit("req_1", "compute_trajectory", "p1", turns=1)
        await session.submit("req_2", "compute_trajectory", "p2", turns=1)
        self.assertEqual(session._worker_load(worker), 1)
        self.assertEqual(session.pending_count, 1)

        results = []
        async for res, exc in session.as_completed():
          self.assertIsNone(exc)
          results.append(res)

        self.assertLen(results, 2)
        self.assertEqual(pool.actors, [worker])

    asyncio.run(_run())

  def test_pool_execution_session_rebinds_across_event_loops(self):
    worker = create_in_process_handle(
        StubWorkerEngine("multi_loop_worker", latency=0.01)
    )
    pool = remote_lib.RoutingActorPool([worker])
    session = remote_lib.PoolExecutionSession(pool)

    async def _step(req_id: str, prompt: str):
      await session.submit(req_id, "compute_trajectory", prompt, turns=1)
      batch = await session.poll_completed(timeout_s=1.0)
      self.assertLen(batch, 1)
      self.assertIsNone(batch[0][1])
      return batch[0][0]

    res1 = asyncio.run(_step("req_loop_1", "p_loop_1"))
    res2 = asyncio.run(_step("req_loop_2", "p_loop_2"))
    asyncio.run(session.close())

    self.assertIn("p_loop_1", res1)
    self.assertIn("p_loop_2", res2)

  def test_pool_execution_session_retain_pending_on_zero_workers(
      self,
  ):
    class CrashingPollHandle(remote_lib.ActorHandle):

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
        raise ConnectionError("last_worker_crashed")

    async def _run():
      crashing = CrashingPollHandle()
      pool = remote_lib.RoutingActorPool([crashing])
      session = remote_lib.PoolExecutionSession(
          pool,
          config=remote_lib.PoolSessionConfig(
              evict_on_failure=True,
              retry_on_worker_failure=True,
              max_task_retries=3,
              retain_pending_on_zero_workers=True,
          ),
      )

      await session.submit(
          "req_zero_1", "compute_trajectory", "p_zero", turns=1
      )
      # Poll briefly so crashing worker fails and is evicted.
      batch = await session.poll_completed(timeout_s=0.05)
      self.assertEmpty(batch)
      self.assertEmpty(pool.actors)
      self.assertEqual(session.pending_count, 1)
      self.assertEqual(session.in_flight_count, 1)
      self.assertEmpty(session.pop_failed_tasks())

      # Dynamically add a replacement worker; the retained pending task drains
      # onto the replacement and completes cleanly.
      replacement = create_in_process_handle(
          StubWorkerEngine("replacement_worker", latency=0.01)
      )
      session.add_actor(replacement)
      batch2 = await session.poll_completed(timeout_s=2.0)
      self.assertLen(batch2, 1)
      self.assertEqual(
          batch2[0][0],
          "[replacement_worker] Trajectory for prompt p_zero (1 turns)",
      )
      self.assertIsNone(batch2[0][1])
      self.assertEqual(session.in_flight_count, 0)
      self.assertEqual(session.pending_count, 0)
      await session.close()

    asyncio.run(_run())

  def test_pool_session_config_validates_arguments(self):
    dummy_handle = create_in_process_handle(StubWorkerEngine("w", latency=0.01))
    with self.assertRaisesRegex(ValueError, "max_task_retries"):
      remote_lib.PoolSessionConfig(max_task_retries=-1)
    with self.assertRaisesRegex(ValueError, "max_in_flight_per_worker"):
      remote_lib.PoolSessionConfig(max_in_flight_per_worker=0)
    with self.assertRaisesRegex(ValueError, "worker_max_in_flight"):
      remote_lib.PoolSessionConfig(worker_max_in_flight={dummy_handle: 0})
    with self.assertRaisesRegex(ValueError, "task_timeout_s"):
      remote_lib.PoolSessionConfig(task_timeout_s=0.0)

if __name__ == "__main__":
  absltest.main()
