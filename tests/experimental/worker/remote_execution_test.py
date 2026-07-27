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
from absl.testing import absltest
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
    engine: Any, default_timeout_s: Optional[float] = remote_lib.DEFAULT_TIMEOUT_S
):
  """Async context manager managing lifecycle of a GrpcRemoteExecutionServer and handle."""
  port = portpicker.pick_unused_port()
  server = remote_lib.GrpcRemoteExecutionServer(engine)
  await server.start_serving_async(port=port)
  handle = remote_lib.GrpcRemoteActorHandle(
      target_address=f"grpc://localhost:{port}",
      default_timeout_s=default_timeout_s,
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
        "compute_trajectory", args=("prompt_1",), kwargs={"turns": 5}
    )
    payload = req.serialize()
    restored = remote_lib.ExecutionRequest.deserialize(payload)
    self.assertEqual(restored.method_name, "compute_trajectory")
    self.assertEqual(restored.args, ("prompt_1",))
    self.assertEqual(restored.kwargs, {"turns": 5})

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
          ("compute_trajectory", ("req_slow_on_A",), {"turns": 1}),
          ("compute_trajectory", ("req_fast_on_B",), {"turns": 1}),
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
          "compute_trajectory", "prompt_async", turns=2
      )
      self.assertTrue(task_id.startswith("task_"))

      # Poll response queue asynchronously
      resp = await handle.poll_responses(timeout_s=1.0)
      self.assertIsNotNone(resp)
      self.assertEqual(
          resp.unwrap(),
          "[async_worker] Trajectory for prompt prompt_async (2 turns)",
      )

    asyncio.run(_run())

  def test_grpc_dispatch_task_and_long_polling(self):
    """Verifies dispatch_task and poll_responses over real gRPC TCP channels."""

    async def _run_test():
      engine = StubWorkerEngine("grpc_async_worker", latency=0.05)
      async with running_grpc_server(engine) as (_, handle):
        task_id = await handle.dispatch_task(
            "compute_trajectory", "prompt_grpc_async", turns=3
        )
        self.assertTrue(task_id.startswith("task_"))

        # Long-poll response queue over gRPC
        resp = await handle.poll_responses(timeout_s=2.0)
        self.assertIsNotNone(resp)
        self.assertEqual(
            resp.unwrap(),
            "[grpc_async_worker] Trajectory for prompt prompt_grpc_async (3"
            " turns)",
        )

    asyncio.run(_run_test())

  def test_as_completed_stream_with_none_results(self):
    """Verifies as_completed_stream handles tasks returning None without deadlocking."""

    class NoneWorker:

      def void_task(self) -> None:
        return None

    async def _run():
      handle = create_in_process_handle(NoneWorker())
      pool = remote_lib.RoutingActorPool([handle])

      tasks = [("void_task", (), {}), ("void_task", (), {})]
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
      await handle_inproc.dispatch_task("compute_trajectory", "p1")
      self.assertIsNone(await handle_inproc.poll_responses(timeout_s=0.0))
      self.assertIsNone(await handle_inproc.poll_responses(timeout_s=0.01))

      # Test over real gRPC TCP channels
      async with running_grpc_server(engine) as (_, handle_grpc):
        await handle_grpc.dispatch_task("compute_trajectory", "p2")
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

      tasks = [("compute_trajectory", ("failing_prompt",), {})]
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

    restored = remote_lib.ExecutionResponse.deserialize(resp.serialize())

    self.assertEqual(restored.error_type, "ValueError")
    self.assertEqual(restored.error_message, "worker 1 is paused!")
    self.assertEqual(restored.traceback, "Traceback: ...")
    self.assertTrue(restored.retryable)
    with self.assertRaises(RuntimeError):
      restored.unwrap()

  def test_execution_response_serialization_success(self):
    res_success = remote_lib.ExecutionResponse(result=42)
    payload = res_success.serialize()
    restored = remote_lib.ExecutionResponse.deserialize(payload)
    self.assertEqual(restored.unwrap(), 42)

  def test_execute_request_captures_traceback(self):
    async def _run():
      engine = StubWorkerEngine("worker_1")
      engine.pause()
      server = remote_lib.InProcessRemoteExecutionServer(engine)
      resp = await server.execute_request(
          remote_lib.ExecutionRequest("compute_trajectory")
      )
      self.assertEqual(resp.error_type, "RuntimeError")
      self.assertIn("currently paused", resp.error_message)
      self.assertIsNotNone(resp.traceback)
      self.assertIn("compute_trajectory", resp.traceback)

    asyncio.run(_run())

  def test_server_error_no_instance_bound_sync(self):
    server = remote_lib.InProcessRemoteExecutionServer()
    req = remote_lib.ExecutionRequest("any_method")
    resp1 = server.execute_sync_request(req)
    self.assertEqual(resp1.error_type, "InstanceNotBoundError")
    self.assertEqual(
        resp1.error_message, "RemoteExecutionServer has no registered instance."
    )

  def test_server_error_no_instance_bound_async(self):
    server = remote_lib.InProcessRemoteExecutionServer()
    req = remote_lib.ExecutionRequest("any_method")

    async def _run_async_err():
      resp6 = await server.execute_request(req)
      self.assertEqual(resp6.error_type, "InstanceNotBoundError")
      self.assertEqual(
          resp6.error_message,
          "RemoteExecutionServer has no registered instance.",
      )

    asyncio.run(_run_async_err())

  def test_server_error_method_not_found_sync(self):
    server = remote_lib.InProcessRemoteExecutionServer(
        StubWorkerEngine("worker_01")
    )
    resp2 = server.execute_sync_request(
        remote_lib.ExecutionRequest("missing_method")
    )
    self.assertEqual(resp2.error_type, "AttributeError")
    self.assertEqual(
        resp2.error_message,
        "Method 'missing_method' not found on bound instance.",
    )

  def test_server_error_method_not_found_async(self):
    server = remote_lib.InProcessRemoteExecutionServer(
        StubWorkerEngine("worker_01")
    )

    async def _run_async_err():
      resp7 = await server.execute_request(
          remote_lib.ExecutionRequest("missing")
      )
      self.assertEqual(resp7.error_type, "AttributeError")
      self.assertEqual(
          resp7.error_message, "Method 'missing' not found on bound instance."
      )

    asyncio.run(_run_async_err())

  def test_server_error_exception_during_sync_execution(self):
    class ThrowingWorker:

      def fail(self):
        raise ValueError("Intentional failure")

    server = remote_lib.InProcessRemoteExecutionServer(ThrowingWorker())
    resp4 = server.execute_sync_request(remote_lib.ExecutionRequest("fail"))
    self.assertEqual(resp4.error_type, "ValueError")
    self.assertEqual(resp4.error_message, "Intentional failure")

  def test_handle_execute_returns_error_for_unserializable_result(self):
    async def _run():
      server = remote_lib.GrpcRemoteExecutionServer(_LockReturner())
      request_bytes = remote_lib.ExecutionRequest("get").serialize()

      response_bytes = await server._handle_execute(request_bytes, context=None)

      resp = remote_lib.ExecutionResponse.deserialize(response_bytes)
      self.assertEqual(resp.error_type, "ExecutionResponseSerializationError")
      self.assertIsNotNone(resp.error_message)

    asyncio.run(_run())

  def test_serialize_returns_error_for_unserializable_result(self):
    resp = remote_lib.ExecutionResponse(result=_LockReturner().get())
    payload = resp.serialize()
    restored = remote_lib.ExecutionResponse.deserialize(payload)
    self.assertEqual(restored.error_type, "ExecutionResponseSerializationError")
    self.assertIn("failed to serialize result", restored.error_message)

  def test_handle_poll_responses_returns_error_for_unserializable_result(self):
    async def _run():
      server = remote_lib.GrpcRemoteExecutionServer(_LockReturner())
      request = remote_lib.ExecutionRequest("get")
      await server.dispatch_task(request)

      response_bytes = await server._handle_poll_responses(b"", context=None)

      resp = remote_lib.ExecutionResponse.deserialize(response_bytes)
      self.assertEqual(resp.error_type, "ExecutionResponseSerializationError")
      self.assertIsNotNone(resp.error_message)

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
      async with running_grpc_server(engine, default_timeout_s=0.3) as (
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


if __name__ == "__main__":
  absltest.main()
