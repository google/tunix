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
from absl.testing import parameterized
import numpy as np
import portpicker
from tunix.experimental.worker import network as network_lib
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

  def echo(self, payload: Any) -> Any:
    return payload


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


class RemoteExecutionTest(parameterized.TestCase):
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
        remote_lib.ExecutionRequest(
            request_id="req_err_5", method_name="fail"
        )
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
    options = dict(network_lib._grpc_options())
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
    self.assertEqual(
        restored_payload["metadata"], {"step": 7, "packed": True}
    )
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
        ack_id = await handle.dispatch_task(
            "req_stream_1", "echo_batch", batch
        )
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

  def test_iter_async_from_chunk_specs_releases_views_on_early_exit(self):
    arr = np.arange(512 * 1024, dtype=np.int32)
    _, manifest, views, raw_buffers, spec_iter = (
        network_lib._prepare_serialized_chunk_specs(
            {"tensor": arr}, chunk_size=256 * 1024
        )
    )

    async def _run():
      ait = network_lib._iter_async_from_chunk_specs(
          manifest, views, raw_buffers, spec_iter, offload=True
      )
      first = await ait.__anext__()
      self.assertEqual(first, manifest)
      header_chunk = await ait.__anext__()
      self.assertNotEmpty(header_chunk)
      data_chunk = await ait.__anext__()
      self.assertLen(data_chunk, 256 * 1024)
      await ait.aclose()
      await asyncio.get_running_loop().run_in_executor(
          network_lib._SERDE_EXECUTOR, lambda: None
      )
      deadline = time.monotonic() + 2.0
      while time.monotonic() < deadline:
        try:
          _ = views[0].nbytes
        except ValueError:
          break
        await asyncio.sleep(0.005)
      for mv in views:
        with self.assertRaises(ValueError):
          _ = mv.nbytes

    asyncio.run(_run())

  def test_iter_serialized_chunks_with_size_preserves_by_value_and_size(
      self,
  ):
    arr = np.arange(128, dtype=np.int32)
    total_bytes, chunks_it = network_lib._iter_serialized_chunks_with_size(
        {"tokens": arr, "step": 3}
    )
    chunks = list(chunks_it)
    self.assertGreater(total_bytes, arr.nbytes)
    restored = network_lib._deserialize_from_chunks(chunks)
    np.testing.assert_array_equal(restored["tokens"], arr)
    self.assertEqual(restored["step"], 3)

    # Verify both local closures and functions simulated in `__main__` are
    # serialized by value via cloudpickle so remote workers without the caller's
    # `__main__` attribute can deserialize and execute them.
    multiplier = 7

    def _main_fn(x: int) -> int:
      return x * multiplier

    _main_fn.__module__ = "__main__"
    closure_payload = {
        "arr": np.ones(16, dtype=np.float32),
        "fn": _main_fn,
    }
    closure_bytes, closure_it = network_lib._iter_serialized_chunks_with_size(
        closure_payload
    )
    self.assertGreater(closure_bytes, 64)
    restored_closure = network_lib._deserialize_from_chunks(list(closure_it))
    self.assertEqual(restored_closure["fn"](6), 42)
    np.testing.assert_array_equal(
        restored_closure["arr"], np.ones(16, dtype=np.float32)
    )

  def test_small_payload_serialize_async_chunks_runs_inline_and_closes(self):
    async def _run():
      req = remote_lib.ExecutionRequest(
          request_id="req_inline",
          method_name="compute_trajectory",
          args=("prompt_inline", np.arange(32, dtype=np.int32)),
          kwargs={"turns": 2},
      )
      with mock.patch.object(
          network_lib._SERDE_EXECUTOR,
          "submit",
          wraps=network_lib._SERDE_EXECUTOR.submit,
      ) as mock_submit:
        ait = req.serialize_async_chunks(chunk_size=64)
        first = await ait.__anext__()
        self.assertNotEmpty(first)
        await ait.aclose()
        with self.assertRaises(StopAsyncIteration):
          await ait.__anext__()
        restored_req = (
            await remote_lib.ExecutionRequest.deserialize_async_chunks(
                req.serialize_async_chunks()
            )
        )
        # Payloads < _ASYNC_OFFLOAD_THRESHOLD_BYTES run inline without executor.
        mock_submit.assert_not_called()
      self.assertEqual(restored_req.request_id, "req_inline")
      self.assertEqual(restored_req.args[0], "prompt_inline")
      np.testing.assert_array_equal(
          restored_req.args[1], np.arange(32, dtype=np.int32)
      )

    asyncio.run(_run())

  def test_copy_view_slice_nogil_matches_memoryview_tobytes(self):
    raw = np.arange(512 * 1024, dtype=np.int16).tobytes()
    mv = memoryview(raw)
    try:
      # Small slice (< _NOGIL_COPY_THRESHOLD_BYTES)
      small = network_lib._copy_view_slice_nogil(mv, 128, 4096)
      self.assertEqual(small, raw[128 : 128 + 4096])
      # Large slice (>= _NOGIL_COPY_THRESHOLD_BYTES) with non-zero offset
      large_len = network_lib._NOGIL_COPY_THRESHOLD_BYTES + 8192
      large = network_lib._copy_view_slice_nogil(mv, 1024, large_len)
      self.assertEqual(large, raw[1024 : 1024 + large_len])
    finally:
      mv.release()

  def test_parallel_chunk_reassembly_and_concurrent_adaptive_slicing(self):
    arr_a = np.arange(1500 * 1024, dtype=np.int16)  # ~3 MiB
    arr_b = np.arange(1500 * 1024, dtype=np.int16) + 7
    req_a = remote_lib.ExecutionRequest(
        request_id="conc_a",
        method_name="step",
        args=({"routed": arr_a},),
        kwargs={},
    )
    req_b = remote_lib.ExecutionRequest(
        request_id="conc_b",
        method_name="step",
        args=({"routed": arr_b},),
        kwargs={},
    )

    async def _roundtrip(req: remote_lib.ExecutionRequest):
      chunks = []
      async for c in req.serialize_async_chunks(chunk_size=4 * 1024 * 1024):
        chunks.append(c)

      async def _gen():
        for c in chunks:
          yield c

      restored = await remote_lib.ExecutionRequest.deserialize_async_chunks(
          _gen()
      )
      return chunks, restored

    async def _run():
      (_, res_a), (chunks_b, res_b) = await asyncio.gather(
          _roundtrip(req_a), _roundtrip(req_b)
      )
      # Concurrent streams sub-slice chunks at <= _CONCURRENT_STREAM_CHUNK_BYTES
      max_data_chunk_b = max(len(c) for c in chunks_b[1:])
      self.assertLessEqual(
          max_data_chunk_b, network_lib._CONCURRENT_STREAM_CHUNK_BYTES
      )
      np.testing.assert_array_equal(res_a.args[0]["routed"], arr_a)
      np.testing.assert_array_equal(res_b.args[0]["routed"], arr_b)
      # Verify restored arrays remain writable
      res_a.args[0]["routed"][0] = 999
      self.assertEqual(res_a.args[0]["routed"][0], 999)

    asyncio.run(_run())
    self.assertEqual(network_lib._ACTIVE_ASYNC_SERDE_STREAMS, 0)

  def test_bulk_transfer_does_not_overwrite_held_result_arrays(self):
    bulk_server = network_lib._BulkTransferServer()
    bulk_server.start()
    try:
      arr1 = np.arange(512 * 1024, dtype=np.int32)  # 2 MiB >= 512 KiB threshold
      arr2 = arr1 + 100
      req1 = remote_lib.ExecutionRequest(
          request_id="bulk_push_1",
          method_name="echo",
          args=({"tensor": arr1},),
          kwargs={},
      )
      req2 = remote_lib.ExecutionRequest(
          request_id="bulk_push_2",
          method_name="echo",
          args=({"tensor": arr2},),
          kwargs={},
      )
      resp = remote_lib.ExecutionResponse(result={"tensor": arr1 + 5})

      async def _push_req_chunks(
          r: remote_lib.ExecutionRequest, host: str, port: int
      ) -> list[bytes]:
        _, _, views, raw_buffers, _ = (
            network_lib._prepare_serialized_chunk_specs(r._as_tuple())
        )
        return [
            c
            async for c in network_lib._stream_bulk_push_chunks(
                views, raw_buffers, host, port
            )
        ]

      async def _run():
        # 1. First _OP_PUSH (Client -> Server) yields a tiny manifest chunk
        push_chunks_1 = await _push_req_chunks(
            req1, "localhost", bulk_server.port
        )
        self.assertLen(push_chunks_1, 1)
        self.assertLess(len(push_chunks_1[0]), 4096)

        async def _push_gen_1():
          for c in push_chunks_1:
            yield c

        restored_req_1 = (
            await remote_lib.ExecutionRequest.deserialize_async_chunks(
                _push_gen_1(), bulk_server=bulk_server
            )
        )
        held_tensor_1 = restored_req_1.args[0]["tensor"]
        np.testing.assert_array_equal(held_tensor_1, arr1)

        # 2. Second _OP_PUSH of identical size while `held_tensor_1` is still
        # referenced must NOT overwrite `held_tensor_1` in-place.
        push_chunks_2 = await _push_req_chunks(
            req2, "localhost", bulk_server.port
        )

        async def _push_gen_2():
          for c in push_chunks_2:
            yield c

        restored_req_2 = (
            await remote_lib.ExecutionRequest.deserialize_async_chunks(
                _push_gen_2(), bulk_server=bulk_server
            )
        )
        np.testing.assert_array_equal(restored_req_2.args[0]["tensor"], arr2)
        np.testing.assert_array_equal(held_tensor_1, arr1)

        # 3. Verify _OP_PULL (Server -> Client)
        async def _pull_gen():
          async for c in resp.serialize_async_chunks(bulk_server=bulk_server):
            yield c

        restored_resp = (
            await remote_lib.ExecutionResponse.deserialize_async_chunks(
                _pull_gen(), bulk_host="localhost"
            )
        )
        assert restored_resp is not None
        np.testing.assert_array_equal(
            restored_resp.unwrap()["tensor"], arr1 + 5
        )

      asyncio.run(_run())
    finally:
      bulk_server.stop()

  def test_bulk_transfer_non_localhost_socket_pool_reuse_without_timeout_lag(
      self,
  ):
    bulk_server = network_lib._BulkTransferServer()
    bulk_server.start()
    try:
      arr = np.arange(256 * 1024, dtype=np.int32)  # 1 MiB >= 512 KiB
      req = remote_lib.ExecutionRequest(
          request_id="non_localhost_req",
          method_name="echo",
          args=({"tensor": arr},),
          kwargs={},
      )
      resp = remote_lib.ExecutionResponse(result={"tensor": arr + 3})

      async def _run():
        # Using 127.0.0.2 exercises the non-localhost socket.create_connection
        # path in _BulkSocketPool._connect. Two rounds of push + pull must
        # reuse the pooled socket in microseconds without stalling on
        # _is_socket_alive.
        for _ in range(2):
          t0 = time.monotonic()
          _, _, views, raw_buffers, _ = (
              network_lib._prepare_serialized_chunk_specs(req._as_tuple())
          )
          push_chunks = [
              c
              async for c in network_lib._stream_bulk_push_chunks(
                  views, raw_buffers, "127.0.0.2", bulk_server.port
              )
          ]

          async def _push_gen(items=push_chunks):
            for c in items:
              yield c

          restored_req = (
              await remote_lib.ExecutionRequest.deserialize_async_chunks(
                  _push_gen(), bulk_server=bulk_server
              )
          )
          np.testing.assert_array_equal(restored_req.args[0]["tensor"], arr)

          async def _pull_gen():
            async for c in resp.serialize_async_chunks(bulk_server=bulk_server):
              yield c

          restored_resp = (
              await remote_lib.ExecutionResponse.deserialize_async_chunks(
                  _pull_gen(), bulk_host="127.0.0.2"
              )
          )
          assert restored_resp is not None
          np.testing.assert_array_equal(
              restored_resp.unwrap()["tensor"], arr + 3
          )
          elapsed = time.monotonic() - t0
          self.assertLess(elapsed, 2.0)

      asyncio.run(_run())
    finally:
      network_lib._DEFAULT_BULK_SOCKET_POOL.close_target(
          "127.0.0.2", bulk_server.port
      )
      bulk_server.stop()

  def test_multi_stripe_bulk_push_and_pull(self):
    bulk_server = network_lib._BulkTransferServer()
    bulk_server.start()
    try:
      with mock.patch.object(network_lib, "_MIN_BULK_STRIPE_BYTES", 512 * 1024):
        arr = np.arange(512 * 1024, dtype=np.int32)  # 2 MiB -> 4 stripes
        stripes = network_lib._plan_bulk_stripes([arr.nbytes])
        self.assertLen(stripes, 4)

        req = remote_lib.ExecutionRequest(
            request_id="multi_stripe_req",
            method_name="echo",
            args=({"tensor": arr},),
            kwargs={},
        )
        resp = remote_lib.ExecutionResponse(result={"tensor": arr + 9})

        async def _run():
          _, _, views, raw_buffers, _ = (
              network_lib._prepare_serialized_chunk_specs(req._as_tuple())
          )
          push_chunks = [
              c
              async for c in network_lib._stream_bulk_push_chunks(
                  views, raw_buffers, "localhost", bulk_server.port
              )
          ]

          async def _push_gen():
            for c in push_chunks:
              yield c

          restored_req = (
              await remote_lib.ExecutionRequest.deserialize_async_chunks(
                  _push_gen(), bulk_server=bulk_server
              )
          )
          np.testing.assert_array_equal(restored_req.args[0]["tensor"], arr)

          async def _pull_gen():
            async for c in resp.serialize_async_chunks(bulk_server=bulk_server):
              yield c

          restored_resp = (
              await remote_lib.ExecutionResponse.deserialize_async_chunks(
                  _pull_gen(), bulk_host="localhost"
              )
          )
          assert restored_resp is not None
          np.testing.assert_array_equal(
              restored_resp.unwrap()["tensor"], arr + 9
          )

        asyncio.run(_run())
    finally:
      bulk_server.stop()

  def test_bulk_transfer_token_isolation_and_bounds_validation(self):
    tid1 = network_lib._allocate_transfer_id()
    tid2 = network_lib._allocate_transfer_id()
    self.assertIsInstance(tid1, bytes)
    self.assertLen(tid1, 16)
    self.assertNotEqual(tid1, tid2)

    bulk_server = network_lib._BulkTransferServer()
    bulk_server.start()
    try:
      entry = bulk_server._get_or_create_incoming(tid1, 1024)
      assert entry is not None
      with self.assertRaisesRegex(ValueError, "Conflicting total_bytes"):
        bulk_server._get_or_create_incoming(tid1, 2048)

      mv = entry.get_or_alloc_view(0, 1024)
      self.assertLen(mv, 1024)
      with self.assertRaisesRegex(ValueError, "Mismatched buf_total_len"):
        entry.get_or_alloc_view(0, 512)
      with self.assertRaisesRegex(
          ValueError, "Invalid bulk buffer allocation exceeds declared"
      ):
        entry.get_or_alloc_view(1, 512)
      with self.assertRaisesRegex(ValueError, "Invalid bulk buffer"):
        entry.get_or_alloc_view(2, 4096)
      entry.release_views()
    finally:
      bulk_server.stop()

  def test_unreachable_bulk_port_surfaces_connection_error_and_rolls_back_session(
      self,
  ):
    async def _run():
      engine = StubWorkerEngine("worker_unreachable_bulk")
      async with running_grpc_server(engine) as (_, handle):
        dead_port = portpicker.pick_unused_port()
        # Force a stale/unreachable bulk port on the handle to simulate a
        # firewall block or server restart on a new ephemeral bulk port.
        handle._bulk_port = dead_port
        arr = np.ones(256 * 1024, dtype=np.int32)  # 1 MiB >= 512 KiB

        with self.assertRaises(OSError):
          await handle.asubmit("echo", {"tensor": arr})
        # Cached bulk port must be invalidated after push failure.
        self.assertIsNone(handle._bulk_port)

        # Verify PoolExecutionSession rolls back _in_flight when dispatch_task
        # fails due to an unreachable bulk port.
        pool = remote_lib.RoutingActorPool([handle])
        async with pool.execution_session() as session:
          handle._bulk_port = dead_port
          with self.assertRaises(OSError):
            await session.submit("req_fail_bulk", "echo", {"tensor": arr})
          self.assertEqual(session._in_flight, 0)
          self.assertEmpty(session._dispatched_tasks.get(handle, set()))

    asyncio.run(_run())

  def test_stale_server_bulk_port_mismatch_invalidates_and_recovers_on_retry(
      self,
  ):
    # V2 + W5#3 regression test: When client has a cached _bulk_port pointing to
    # an auxiliary bulk server (simulating a worker restart onto a new bulk
    # port), both SubmitTask and DispatchTask reject the push manifest with
    # FAILED_PRECONDITION, the client invalidates _bulk_port, and the next
    # asubmit() / dispatch_task() re-discovers the real bulk port and succeeds.
    aux_bulk_server = network_lib._BulkTransferServer()
    aux_port = aux_bulk_server.start()
    try:
      async def _run():
        engine = StubWorkerEngine("worker_port_mismatch")
        port = portpicker.pick_unused_port()
        server = remote_lib.GrpcRemoteExecutionServer(engine)
        await server.start_serving_async(port=port)
        handle = remote_lib.GrpcRemoteActorHandle(
            target_address=f"grpc://127.0.0.2:{port}"
        )
        try:
          arr = np.arange(256 * 1024, dtype=np.int32)  # 1 MiB
          # 1. SubmitTask with stale _bulk_port
          handle._bulk_port = aux_port
          handle._seen_bulk_ports.add(aux_port)
          with self.assertRaisesRegex(
              ConnectionError, "Bulk push manifest port mismatch"
          ):
            await handle.asubmit("echo", {"tensor": arr})
          self.assertIsNone(handle._bulk_port)

          recovered = await handle.asubmit("echo", {"tensor": arr + 7})
          np.testing.assert_array_equal(recovered["tensor"], arr + 7)
          assert server._bulk_server is not None
          self.assertEqual(handle._bulk_port, server._bulk_server.port)

          # 2. DispatchTask with stale _bulk_port
          handle._bulk_port = aux_port
          handle._seen_bulk_ports.add(aux_port)
          with self.assertRaisesRegex(
              ConnectionError, "Bulk push manifest port mismatch"
          ):
            await handle.dispatch_task(
                "req_mismatch_dt", "echo", {"tensor": arr}
            )
          self.assertIsNone(handle._bulk_port)

          ack = await handle.dispatch_task(
              "req_recovered_dt", "echo", {"tensor": arr + 9}
          )
          self.assertEqual(ack, "req_recovered_dt")
          polled = await handle.poll_responses(timeout_s=5.0)
          self.assertIsNotNone(polled)
          np.testing.assert_array_equal(polled.unwrap()["tensor"], arr + 9)
          self.assertEqual(handle._bulk_port, server._bulk_server.port)
        finally:
          await handle.close()
          await server.stop_serving(grace=0.0)

      asyncio.run(_run())
    finally:
      aux_bulk_server.stop()

  def test_cancelling_unrelated_rpc_preserves_cached_bulk_port_and_pool(self):
    # V4 + W5#1 regression test: Cancelling an in-flight RPC must not invalidate
    # _bulk_port or tear down the shared bulk socket pool for the target.
    async def _run():
      started = asyncio.Event()
      release = asyncio.Event()

      class _SlowEngine(StubWorkerEngine):

        async def slow_op(self, x: int) -> int:
          started.set()
          await release.wait()
          return x + 1

      engine = _SlowEngine("slow_cancel_worker")
      port = portpicker.pick_unused_port()
      server = remote_lib.GrpcRemoteExecutionServer(engine)
      await server.start_serving_async(port=port)
      handle = remote_lib.GrpcRemoteActorHandle(
          target_address=f"grpc://127.0.0.2:{port}"
      )
      try:
        arr = np.arange(256 * 1024, dtype=np.int32)
        out = await handle.asubmit("echo", {"tensor": arr})
        np.testing.assert_array_equal(out["tensor"], arr)
        assert server._bulk_server is not None
        expected_bulk_port = server._bulk_server.port
        self.assertEqual(handle._bulk_port, expected_bulk_port)
        pool_key = ("127.0.0.2", expected_bulk_port)
        self.assertNotEmpty(
            network_lib._DEFAULT_BULK_SOCKET_POOL._pools.get(pool_key, ())
        )

        slow_task = asyncio.create_task(handle.asubmit("slow_op", 41))
        await asyncio.wait_for(started.wait(), timeout=5.0)
        slow_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
          await slow_task
        release.set()

        # _bulk_port, _seen_bulk_ports, and pooled sockets must remain intact.
        self.assertEqual(handle._bulk_port, expected_bulk_port)
        self.assertIn(expected_bulk_port, handle._seen_bulk_ports)
        self.assertNotEmpty(
            network_lib._DEFAULT_BULK_SOCKET_POOL._pools.get(pool_key, ())
        )
      finally:
        release.set()
        await handle.close()
        await server.stop_serving(grace=0.0)

    asyncio.run(_run())

  def test_bulk_server_stop_wakes_threads_and_evicts_stale_incoming(self):
    # W7 + L5 regression test: Background _cleanup_loop evicts expired unclaimed
    # entries automatically without requiring a subsequent push.
    bulk_server = network_lib._BulkTransferServer()
    port = bulk_server.start()
    client_sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    try:
      stale_tid = network_lib._allocate_transfer_id()
      stale_entry = bulk_server._get_or_create_incoming(stale_tid, 1024)
      assert stale_entry is not None
      _ = stale_entry.get_or_alloc_view(0, 1024)
      self.assertEqual(bulk_server._unclaimed_bytes, 1024)
      # Backdate created_at past _INCOMING_TTL_S after asserting
      # _unclaimed_bytes so there is no race window between allocation and the
      # assertion.
      stale_entry.created_at = (
          time.monotonic() - network_lib._INCOMING_TTL_S - 10.0
      )

      deadline = time.monotonic() + 2.0
      while stale_tid in bulk_server._incoming and time.monotonic() < deadline:
        time.sleep(0.02)
      self.assertNotIn(stale_tid, bulk_server._incoming)
      self.assertEqual(bulk_server._unclaimed_bytes, 0)

      t0 = time.monotonic()
      bulk_server.stop()
      stop_elapsed = time.monotonic() - t0
      self.assertLess(stop_elapsed, 1.0)
      if bulk_server._accept_thread is not None:
        self.assertFalse(bulk_server._accept_thread.is_alive())
      if bulk_server._cleanup_thread is not None:
        self.assertFalse(bulk_server._cleanup_thread.is_alive())
      for t in list(bulk_server._conn_threads):
        self.assertFalse(t.is_alive())
    finally:
      client_sock.close()

  def test_recv_into_exact_handles_max_rw_count_and_short_reads(self):
    payload = np.arange(1024, dtype=np.uint8).tobytes()
    with mock.patch.object(network_lib, "_MAX_SOCKET_RW_CHUNK_BYTES", 256):
      s_left, s_right = socket.socketpair()
      try:
        s_left.settimeout(2.0)
        s_right.settimeout(2.0)
        s_left.sendall(payload)
        s_left.shutdown(socket.SHUT_WR)

        dst = bytearray(len(payload))
        mv = memoryview(dst)
        try:
          self.assertEqual(network_lib._recv_into_exact(s_right, mv, 0), 0)
          got = network_lib._recv_into_exact(s_right, mv, len(payload))
        finally:
          mv.release()
        self.assertLen(payload, got)
        self.assertEqual(bytes(dst), payload)

        # Verify EOF branch returns partial read count when peer closes early.
        eof_dst = bytearray(64)
        eof_mv = memoryview(eof_dst)
        try:
          eof_got = network_lib._recv_into_exact(s_right, eof_mv, 64)
        finally:
          eof_mv.release()
        self.assertEqual(eof_got, 0)
      finally:
        s_left.close()
        s_right.close()

    # Deterministic short-read fake socket (7 bytes per recv_into call) +
    # kernel SO_RCVTIMEO BlockingIOError -> TimeoutError conversion.
    class _ShortReadFakeSocket:

      def __init__(self, data: bytes, step: int = 7) -> None:
        self._data = data
        self._pos = 0
        self._step = step

      def recv_into(
          self,
          buf: memoryview,  # pylint: disable=g-bare-generic
          nbytes: int,
          flags: int = 0,
      ) -> int:
        del flags
        if self._pos >= len(self._data):
          return 0
        take = min(nbytes, self._step, len(self._data) - self._pos)
        buf[:take] = self._data[self._pos : self._pos + take]
        self._pos += take
        return take

    fake_sock: Any = _ShortReadFakeSocket(payload, step=7)
    short_dst = bytearray(len(payload))
    short_mv = memoryview(short_dst)
    try:
      got_short = network_lib._recv_into_exact(
          fake_sock, short_mv, len(payload)
      )
    finally:
      short_mv.release()
    self.assertLen(payload, got_short)
    self.assertEqual(bytes(short_dst), payload)

    # W5#2 + Simplification K: Partial sendmsg fake socket (11 bytes per call)
    # across a multi-segment _SegmentedBulkView with > _MAX_SENDMSG_IOVS IOVs.
    class _PartialSendmsgSocket:

      def __init__(self, step: int = 11) -> None:
        self.sent = bytearray()
        self._step = step

      def sendmsg(self, buffers: list[memoryview]) -> int:  # pylint: disable=g-bare-generic
        rem = self._step
        total = 0
        for b in buffers:
          take = min(len(b), rem)
          self.sent.extend(b[:take])
          total += take
          rem -= take
          if rem == 0:
            break
        return total

    seg_arrays = [bytearray([i % 251, (i * 7) % 251, 9]) for i in range(600)]
    seg_mvs = [memoryview(b) for b in seg_arrays]
    try:
      multi_seg = network_lib._SegmentedBulkView(seg_mvs)
      partial_sock: Any = _PartialSendmsgSocket(step=11)
      network_lib._send_bulk_view_slice(
          partial_sock, multi_seg, 5, multi_seg.total_len - 10
      )
      expected_bytes = b"".join(bytes(b) for b in seg_arrays)[
          5 : multi_seg.total_len - 5
      ]
      self.assertEqual(bytes(partial_sock.sent), expected_bytes)
      with self.assertRaisesRegex(ValueError, "slice_iov out of bounds"):
        multi_seg.slice_iov(-1, 4)
      with self.assertRaisesRegex(ValueError, "slice_iov out of bounds"):
        multi_seg.slice_iov(0, multi_seg.total_len + 1)
    finally:
      for m in seg_mvs:
        m.release()

    class _TimeoutFakeSocket:

      def recv_into(
          self,
          buf: memoryview,  # pylint: disable=g-bare-generic
          nbytes: int,
          flags: int = 0,
      ) -> int:
        del buf, nbytes, flags
        raise BlockingIOError("Resource temporarily unavailable")

      def sendall(self, data: bytes) -> None:
        del data
        raise BlockingIOError("Resource temporarily unavailable")

      def sendmsg(self, buffers: list[memoryview]) -> int:  # pylint: disable=g-bare-generic
        del buffers
        raise BlockingIOError("Resource temporarily unavailable")

    timeout_sock: Any = _TimeoutFakeSocket()
    err_dst = bytearray(16)
    err_mv = memoryview(err_dst)
    try:
      with self.assertRaises(TimeoutError):
        network_lib._recv_into_exact(timeout_sock, err_mv, 16)
      with self.assertRaises(TimeoutError):
        network_lib._sendall_with_timeout(timeout_sock, b"ping")
      seg_view = network_lib._SegmentedBulkView([err_mv])
      with self.assertRaises(TimeoutError):
        network_lib._send_bulk_view_slice(timeout_sock, seg_view, 0, 16)
    finally:
      err_mv.release()

  def test_non_loopback_handle_discovers_probes_and_streams_bulk_e2e(self):
    port = portpicker.pick_unused_port()
    engine = StubWorkerEngine("non_loopback_worker")
    with background_server(engine, port) as (srv, _):
      handle = remote_lib.GrpcRemoteActorHandle(
          target_address=f"grpc://127.0.0.2:{port}"
      )
      try:
        with mock.patch.object(
            network_lib, "_MIN_BULK_STRIPE_BYTES", 512 * 1024
        ):
          arr = np.arange(512 * 1024, dtype=np.int32)  # 2 MiB -> 4 stripes
          small_meta = [
              np.full(128, idx + 1, dtype=np.float32) for idx in range(8)
          ]
          payload = {"tensor": arr, "small_meta": small_meta}

          # 1. Blocking submit() over 127.0.0.2 with multi-stripe bulk transfer
          sync_out = handle.submit("echo", payload)
          np.testing.assert_array_equal(sync_out["tensor"], arr)
          for idx, m in enumerate(sync_out["small_meta"]):
            np.testing.assert_array_equal(m, small_meta[idx])
          assert srv._bulk_server is not None
          self.assertEqual(handle._bulk_port, srv._bulk_server.port)
          self.assertIn(srv._bulk_server.port, handle._seen_bulk_ports)
          # Verify multiple sockets were pooled from parallel stripes.
          pool_key = ("127.0.0.2", srv._bulk_server.port)
          pooled_socks = network_lib._DEFAULT_BULK_SOCKET_POOL._pools.get(
              pool_key, ()
          )
          self.assertGreaterEqual(len(pooled_socks), 2)

          # 2. Async asubmit() + dispatch_task/poll_responses over 127.0.0.2
          async def _run_async():
            out = await handle.asubmit("echo", payload)
            np.testing.assert_array_equal(out["tensor"], arr)
            ack = await handle.dispatch_task("req_nl_1", "echo", payload)
            self.assertEqual(ack, "req_nl_1")
            polled = await handle.poll_responses(timeout_s=5.0)
            self.assertIsNotNone(polled)
            np.testing.assert_array_equal(polled.unwrap()["tensor"], arr)

          asyncio.run(_run_async())
      finally:
        asyncio.run(handle.close())

  def test_bulk_probe_failure_negative_caches_and_recovers_for_small_req_large_resp(
      self,
  ):
    # W1 + N1 + R1 + R2' + T2(c) regression test:
    # - Covers GetBulkPort UNIMPLEMENTED (permanently disables bulk with inf
    #   cooldown), short-response handling, negative-caching, and non-blocking
    #   background re-probe recovery on a small-request / large-response
    #   workload after _BULK_PROBE_RETRY_COOLDOWN_S expires.
    # - Covers R2' (Option X): cancelling an in-flight background re-probe when
    #   a short-lived asyncio.run() loop exits keeps the 60s cooldown active so
    #   repeated short-lived asyncio.run() calls do not spam GetBulkPort on
    #   every call.
    class _LargeRespEngine(StubWorkerEngine):

      def make_large(self, seed: int) -> dict[str, Any]:
        return {"big": np.full(256 * 1024, seed, dtype=np.int32)}

    async def _run():
      engine = _LargeRespEngine("probe_fallback_worker")
      port = portpicker.pick_unused_port()
      pinned_bulk_port = portpicker.pick_unused_port()
      dead_bulk_port = portpicker.pick_unused_port()
      server = remote_lib.GrpcRemoteExecutionServer(
          engine, bulk_port=pinned_bulk_port
      )
      real_get_bulk_port = server._handle_get_bulk_port
      get_port_calls = 0
      mode = "unimplemented"
      reprobe_hold = asyncio.Event()

      async def _fake_get_bulk_port(
          request_bytes: bytes, context: Any
      ) -> bytes:
        nonlocal get_port_calls
        get_port_calls += 1
        if mode == "unimplemented":
          await context.abort(
              remote_lib._grpc_lib.StatusCode.UNIMPLEMENTED,
              "GetBulkPort not implemented",
          )
        if mode == "short":
          return b"\x01\x02"
        if mode == "dead":
          return remote_lib.struct.pack("<Q", dead_bulk_port)
        if mode == "real_hold":
          await reprobe_hold.wait()
        return await real_get_bulk_port(request_bytes, context)

      server._handle_get_bulk_port = _fake_get_bulk_port
      await server.start_serving_async(port=port)
      assert server._bulk_server is not None
      self.assertEqual(server._bulk_server.port, pinned_bulk_port)
      handle = remote_lib.GrpcRemoteActorHandle(
          target_address=f"grpc://127.0.0.2:{port}"
      )
      try:
        arr = np.arange(256 * 1024, dtype=np.int32)  # 1 MiB
        # 1. UNIMPLEMENTED sets _bulk_port = 0 with infinite cooldown so legacy
        # servers are never re-probed.
        res_unimp = await handle.asubmit("echo", {"tensor": arr})
        np.testing.assert_array_equal(res_unimp["tensor"], arr)
        self.assertEqual(handle._bulk_port, 0)
        self.assertEqual(handle._bulk_probe_failed_until, float("inf"))
        self.assertEqual(get_port_calls, 1)
        _ = await handle.asubmit("echo", {"tensor": arr + 1})
        self.assertEqual(get_port_calls, 1)

        # 2. Short (<8B) response negative-caches _bulk_port = 0.
        handle._invalidate_bulk_port()
        mode = "short"
        res0 = await handle.asubmit("echo", {"tensor": arr})
        np.testing.assert_array_equal(res0["tensor"], arr)
        self.assertEqual(handle._bulk_port, 0)
        self.assertEqual(get_port_calls, 2)

        # 3. Dead bulk port fails probe and stays negative-cached for 3 calls.
        handle._invalidate_bulk_port()
        mode = "dead"
        for i in range(3):
          res = await handle.asubmit("echo", {"tensor": arr + i})
          np.testing.assert_array_equal(res["tensor"], arr + i)
        self.assertEqual(handle._bulk_port, 0)
        self.assertEqual(get_port_calls, 3)
        self.assertEmpty(handle._seen_bulk_ports)

        # 4. W1 + N1 + T2(c): Hold the background re-probe task on
        # `reprobe_hold` so `big_first` deterministically completes via gRPC
        # chunks (`mock_pull.call_count == 0`); then release `reprobe_hold`,
        # await `_resolve_task`, and verify `big_out` uses `_OP_PULL`.
        reprobe_hold.clear()
        mode = "real_hold"
        handle._bulk_probe_failed_until = time.monotonic() - 1.0
        with mock.patch.object(
            network_lib,
            "_pull_stripe_sync",
            wraps=network_lib._pull_stripe_sync,
        ) as mock_pull:
          big_first = await handle.asubmit("make_large", 18)
          np.testing.assert_array_equal(
              big_first["big"], np.full(256 * 1024, 18, dtype=np.int32)
          )
          self.assertEqual(mock_pull.call_count, 0)
          reprobe_hold.set()
          if handle._resolve_task is not None:
            await handle._resolve_task
          self.assertEqual(handle._bulk_port, pinned_bulk_port)
          self.assertEqual(get_port_calls, 4)

          mode = "real"
          mock_pull.reset_mock()
          big_out = await handle.asubmit("make_large", 19)
          np.testing.assert_array_equal(
              big_out["big"], np.full(256 * 1024, 19, dtype=np.int32)
          )
          self.assertGreaterEqual(mock_pull.call_count, 1)
      finally:
        await handle.close()
        await server.stop_serving(grace=0.0)

    asyncio.run(_run())

    # 5. R2' (Option X) regression test: When a background re-probe task spawned
    # inside a short-lived `asyncio.run()` loop is cancelled upon loop close,
    # `_bulk_probe_failed_until` stays at the 60s cooldown so repeated short
    # `asyncio.run()` calls do not fire `GetBulkPort` on every call; once the
    # cooldown expires, the next loop re-probes and recovers `_bulk_port`.
    r2_port = portpicker.pick_unused_port()
    with background_server(StubWorkerEngine("r2_loop_worker"), r2_port) as (
        r2_srv,
        _,
    ):
      r2_handle = remote_lib.GrpcRemoteActorHandle(
          target_address=f"grpc://127.0.0.2:{r2_port}"
      )
      try:
        r2_handle._bulk_port = 0
        r2_handle._bulk_probe_failed_until = time.monotonic() - 1.0
        probe_calls = 0

        async def _slow_probe(host: str, port: int) -> None:
          nonlocal probe_calls
          del host, port
          probe_calls += 1
          await asyncio.sleep(10.0)

        with mock.patch.object(
            network_lib._DEFAULT_BULK_SOCKET_POOL,
            "probe_async",
            side_effect=_slow_probe,
        ):
          first_resolve_task = None
          for idx in range(3):
            out = asyncio.run(
                r2_handle.asubmit("compute_trajectory", f"r2_p{idx}", turns=1)
            )
            self.assertIn(f"r2_p{idx}", out)
            if idx == 0:
              first_resolve_task = r2_handle._resolve_task
              self.assertIsNotNone(first_resolve_task)
            else:
              self.assertIs(r2_handle._resolve_task, first_resolve_task)
        # Only the first short-lived asyncio.run() spawned a re-probe; the next
        # two respected the 60s cooldown instead of spamming GetBulkPort.
        self.assertEqual(probe_calls, 1)
        self.assertEqual(r2_handle._bulk_port, 0)
        self.assertGreater(r2_handle._bulk_probe_failed_until, time.monotonic())

        # Expire cooldown and verify recovery on the next loop.
        r2_handle._bulk_probe_failed_until = time.monotonic() - 1.0

        async def _recover_loop() -> None:
          out2 = await r2_handle.asubmit(
              "compute_trajectory", "r2_rec", turns=1
          )
          self.assertIn("r2_rec", out2)
          if r2_handle._resolve_task is not None:
            await r2_handle._resolve_task

        asyncio.run(_recover_loop())
        assert r2_srv._bulk_server is not None
        self.assertEqual(r2_handle._bulk_port, r2_srv._bulk_server.port)
      finally:
        asyncio.run(r2_handle.close())

  def test_bulk_probe_single_flight_shielding_and_invalidate_mid_probe(self):
    # R2 + R4 regression test:
    # 1. R2: Cancelling one concurrent cold-start caller while the single-flight
    #    `_resolve_task` is in flight must not cancel the probe for remaining
    #    callers.
    # 2. R4: Calling `_invalidate_bulk_port()` (or externally cancelling the
    #    inner `_resolve_task`) while concurrent callers are awaiting
    #    `_ensure_bulk_port()` must NOT raise `CancelledError` in those callers;
    #    they must cleanly receive 0 and fall back to gRPC chunks.
    class _LargeRespEngine(StubWorkerEngine):

      def make_large(self, seed: int) -> dict[str, Any]:
        return {"big": np.full(256 * 1024, seed, dtype=np.int32)}

    async def _run():
      engine = _LargeRespEngine("r2_r4_worker")
      port = portpicker.pick_unused_port()
      pinned_bulk_port = portpicker.pick_unused_port()
      server = remote_lib.GrpcRemoteExecutionServer(
          engine, bulk_port=pinned_bulk_port
      )
      real_get_bulk_port = server._handle_get_bulk_port
      get_port_calls = 0
      probe_entered = asyncio.Event()
      probe_hold = asyncio.Event()

      async def _fake_get_bulk_port(
          request_bytes: bytes, context: Any
      ) -> bytes:
        nonlocal get_port_calls
        get_port_calls += 1
        probe_entered.set()
        await probe_hold.wait()
        return await real_get_bulk_port(request_bytes, context)

      server._handle_get_bulk_port = _fake_get_bulk_port
      await server.start_serving_async(port=port)
      handle = remote_lib.GrpcRemoteActorHandle(
          target_address=f"grpc://127.0.0.2:{port}"
      )
      try:
        # 1. R2: Cancel task_a mid-probe; task_b must still complete and cache
        # pinned_bulk_port with a single GetBulkPort call.
        task_a = asyncio.create_task(handle.asubmit("make_large", 30))
        task_b = asyncio.create_task(handle.asubmit("make_large", 31))
        await asyncio.wait_for(probe_entered.wait(), timeout=5.0)
        task_a.cancel()
        with self.assertRaises(asyncio.CancelledError):
          await task_a
        probe_hold.set()
        out_b = await task_b
        np.testing.assert_array_equal(
            out_b["big"], np.full(256 * 1024, 31, dtype=np.int32)
        )
        self.assertEqual(handle._bulk_port, pinned_bulk_port)
        self.assertEqual(get_port_calls, 1)

        # 2. R4: Call _invalidate_bulk_port() while both asubmit() and
        # poll_responses() are awaiting _ensure_bulk_port(). Neither waiter may
        # raise CancelledError; both must fall back to gRPC chunks and succeed.
        ack = await handle.dispatch_task("req_r4_poll", "make_large", 42)
        self.assertEqual(ack, "req_r4_poll")
        handle._invalidate_bulk_port()
        probe_entered.clear()
        probe_hold.clear()

        submit_waiter = asyncio.create_task(handle.asubmit("make_large", 41))
        poll_waiter = asyncio.create_task(handle.poll_responses(timeout_s=5.0))
        await asyncio.wait_for(probe_entered.wait(), timeout=5.0)
        # Invalidate mid-probe while both waiters are blocked on _resolve_task.
        handle._invalidate_bulk_port()
        probe_hold.set()

        out_submit, out_poll = await asyncio.gather(submit_waiter, poll_waiter)
        np.testing.assert_array_equal(
            out_submit["big"], np.full(256 * 1024, 41, dtype=np.int32)
        )
        self.assertIsNotNone(out_poll)
        np.testing.assert_array_equal(
            out_poll.unwrap()["big"], np.full(256 * 1024, 42, dtype=np.int32)
        )
        # Stale probe generation must not have written back _bulk_port.
        self.assertIsNone(handle._bulk_port)

        # 3. Hardening check: When `handle.close()` cancels an active
        # `_resolve_task` on the running loop while a non-cancelled caller
        # awaits `_ensure_bulk_port()`, `_ensure_bulk_port()` returns 0 instead
        # of raising `CancelledError`, and `_resolve_task` is cancelled cleanly.
        probe_entered.clear()
        probe_hold.clear()
        ensure_waiter = asyncio.create_task(
            handle._ensure_bulk_port(handle._get_bulk_port_rpc)
        )
        await asyncio.wait_for(probe_entered.wait(), timeout=5.0)
        in_flight_resolve = handle._resolve_task
        self.assertIsNotNone(in_flight_resolve)
        await handle.close()
        self.assertEqual(await ensure_waiter, 0)
        assert in_flight_resolve is not None
        self.assertTrue(in_flight_resolve.cancelled())
      finally:
        probe_hold.set()
        await handle.close()
        await server.stop_serving(grace=0.0)

    asyncio.run(_run())

  def test_bulk_pull_failure_invalidates_port_while_user_setstate_error_preserves_it(
      self,
  ):
    class _BadUnpicklePayload:

      def __reduce__(self):
        return (_BadUnpicklePayload, (), {"x": 1})

      def __setstate__(self, state: Any) -> None:
        del state
        raise OSError("simulated user __setstate__ error")

    class _LargeRespEngine(StubWorkerEngine):

      def make_large(self, seed: int) -> dict[str, Any]:
        if seed == -999:
          return {
              "big": np.full(256 * 1024, 1, dtype=np.int32),
              "bad": _BadUnpicklePayload(),
          }
        return {"big": np.full(256 * 1024, seed, dtype=np.int32)}

    async def _run():
      engine = _LargeRespEngine("pull_invalidation_worker")
      port = portpicker.pick_unused_port()
      pinned_bulk_port = portpicker.pick_unused_port()
      server = remote_lib.GrpcRemoteExecutionServer(
          engine, bulk_port=pinned_bulk_port
      )
      await server.start_serving_async(port=port)
      handle = remote_lib.GrpcRemoteActorHandle(
          target_address=f"grpc://127.0.0.2:{port}"
      )
      try:
        out = await handle.asubmit("make_large", 1)
        np.testing.assert_array_equal(
            out["big"], np.full(256 * 1024, 1, dtype=np.int32)
        )
        self.assertEqual(handle._bulk_port, pinned_bulk_port)

        # 1. OSError raised inside user `__setstate__` during response
        # unpickling must NOT invalidate `_bulk_port`.
        with self.assertRaisesRegex(OSError, "simulated user __setstate__"):
          await handle.asubmit("make_large", -999)
        self.assertEqual(handle._bulk_port, pinned_bulk_port)

        # 2. T1(b): PULL failure inside asubmit() and poll_responses() must
        # automatically call handle._invalidate_bulk_port().
        def _fail_pull(*args, **kwargs):
          del args, kwargs
          raise ConnectionError("Simulated mid-pull socket drop")

        with mock.patch.object(
            network_lib, "_pull_stripe_sync", side_effect=_fail_pull
        ):
          with self.assertRaises(ConnectionError):
            await handle.asubmit("make_large", 20)
          self.assertIsNone(handle._bulk_port)

        ack = await handle.dispatch_task("req_pull_fail", "make_large", 21)
        self.assertEqual(ack, "req_pull_fail")
        with mock.patch.object(
            network_lib, "_pull_stripe_sync", side_effect=_fail_pull
        ):
          with self.assertRaises(ConnectionError):
            await handle.poll_responses(timeout_s=5.0)
          self.assertIsNone(handle._bulk_port)
      finally:
        await handle.close()
        await server.stop_serving(grace=0.0)

    asyncio.run(_run())

  def test_enable_bulk_transport_false_negotiation_client_and_server(self):
    arr = np.arange(256 * 1024, dtype=np.int32)  # 1 MiB

    # Case 1: Client disables bulk transport while server has it enabled.
    port1 = portpicker.pick_unused_port()
    with background_server(StubWorkerEngine("srv_bulk_on"), port1):
      hdl_client_off = remote_lib.GrpcRemoteActorHandle(
          target_address=f"grpc://127.0.0.2:{port1}",
          enable_bulk_transport=False,
      )
      try:
        out_sync = hdl_client_off.submit("echo", payload := {"tensor": arr})
        np.testing.assert_array_equal(out_sync["tensor"], payload["tensor"])

        async def _client_off_async():
          out_a = await hdl_client_off.asubmit("echo", {"tensor": arr + 1})
          np.testing.assert_array_equal(out_a["tensor"], arr + 1)
          await hdl_client_off.dispatch_task(
              "req_off_1", "echo", {"tensor": arr + 2}
          )
          polled = await hdl_client_off.poll_responses(timeout_s=5.0)
          self.assertIsNotNone(polled)
          np.testing.assert_array_equal(polled.unwrap()["tensor"], arr + 2)

        asyncio.run(_client_off_async())
        self.assertIsNone(hdl_client_off._bulk_port)
        self.assertEmpty(hdl_client_off._seen_bulk_ports)
      finally:
        asyncio.run(hdl_client_off.close())

    # Case 2: Server disables bulk transport while client has it enabled.
    async def _server_off_async():
      port2 = portpicker.pick_unused_port()
      srv_off = remote_lib.GrpcRemoteExecutionServer(
          StubWorkerEngine("srv_bulk_off"),
          enable_bulk_transport=False,
      )
      await srv_off.start_serving_async(port=port2)
      self.assertIsNone(srv_off._bulk_server)
      hdl_client_on = remote_lib.GrpcRemoteActorHandle(
          target_address=f"grpc://127.0.0.2:{port2}",
          enable_bulk_transport=True,
      )
      try:
        out_a = await hdl_client_on.asubmit("echo", {"tensor": arr + 3})
        np.testing.assert_array_equal(out_a["tensor"], arr + 3)
        await hdl_client_on.dispatch_task(
            "req_srv_off", "echo", {"tensor": arr + 4}
        )
        polled = await hdl_client_on.poll_responses(timeout_s=5.0)
        self.assertIsNotNone(polled)
        np.testing.assert_array_equal(polled.unwrap()["tensor"], arr + 4)
        self.assertEqual(hdl_client_on._bulk_port, 0)
        self.assertEmpty(hdl_client_on._seen_bulk_ports)
      finally:
        await hdl_client_on.close()
        await srv_off.stop_serving(grace=0.0)

    asyncio.run(_server_off_async())

  @parameterized.named_parameters(
      dict(
          testcase_name="boundary_4mib_minus_1_vs_4mib",
          header_len=128,
          buffer_lengths=(
              network_lib._BULK_PACK_MAX_BYTES - 1,
              network_lib._BULK_PACK_MAX_BYTES,
              30 * 1024 * 1024,
              0,
              4096,
          ),
          expected_packed_indices={0, 4},
          expected_nonzero_bulk_count=3,
      ),
      dict(
          testcase_name="w4_1024_x_1mib_coalesced_into_single_packed_tail",
          header_len=128,
          buffer_lengths=tuple([1024 * 1024] * 1024),
          expected_packed_indices=set(range(1024)),
          expected_nonzero_bulk_count=1,
      ),
  )
  def test_plan_bulk_buffer_layout_boundaries(
      self,
      header_len: int,
      buffer_lengths: tuple[int, ...],
      expected_packed_indices: set[int],
      expected_nonzero_bulk_count: int,
  ):
    layout = network_lib._plan_bulk_buffer_layout(header_len, buffer_lengths)
    self.assertEqual(set(layout.packed_offsets.keys()), expected_packed_indices)
    for idx, off in layout.packed_offsets.items():
      self.assertEqual(off % network_lib._BULK_PACK_ALIGN, 0)
      self.assertEqual(layout.bulk_lengths[idx], 0)
    nonzero_bulk = [n for n in layout.bulk_lengths if n > 0]
    self.assertLen(nonzero_bulk, expected_nonzero_bulk_count)
    if expected_nonzero_bulk_count == 1:
      stripes = network_lib._plan_bulk_stripes(layout.bulk_lengths)
      self.assertLessEqual(len(stripes), network_lib._MAX_BULK_STRIPES)

  def test_bulk_transfer_mixed_large_and_many_small_buffers_segmented_sendmsg(
      self,
  ):
    # V1 + V3 + W3 + W4 + N2 + T1(f) regression test:
    # 4,096 small arrays (including odd-sized uint8 and float64 arrays to test
    # 64-byte alignment padding) + pickle header > 64 KiB + zero-length array
    # + standalone 4 MiB array (>= _BULK_PACK_MAX_BYTES) coalesced via
    # _SegmentedBulkView and vectored sendmsg.
    bulk_server = network_lib._BulkTransferServer()
    bulk_server.start()
    try:
      large_arr = np.arange(1024 * 1024, dtype=np.int32)  # 4 MiB standalone
      small_arrays = [
          np.arange((idx % 7) + 1, dtype=np.uint8)
          if idx % 2 == 0
          else np.full(32, idx + 0.5, dtype=np.float64)
          for idx in range(4096)
      ]
      empty_arr = np.array([], dtype=np.float32)
      payload = {
          "large": large_arr,
          "small_list": small_arrays,
          "empty": empty_arr,
      }
      req = remote_lib.ExecutionRequest(
          request_id="req_coalesce",
          method_name="echo",
          args=(payload,),
          kwargs={},
      )
      resp = remote_lib.ExecutionResponse(result=payload)

      async def _run():
        with mock.patch.object(
            network_lib, "_MIN_BULK_STRIPE_BYTES", 256 * 1024
        ):
          _, _, req_views, req_raw_buffers, _ = (
              network_lib._prepare_serialized_chunk_specs(req._as_tuple())
          )
          push_chunks = [
              c
              async for c in network_lib._stream_bulk_push_chunks(
                  req_views, req_raw_buffers, "127.0.0.1", bulk_server.port
              )
          ]
          # Bulk transfer must NOT fall back to multi-chunk gRPC even when the
          # pickle header exceeds 64 KiB and there are 4,096 small buffers.
          self.assertLen(push_chunks, 1)
          meta = remote_lib.pickle.loads(push_chunks[0])
          self.assertIsInstance(meta, network_lib._BulkManifest)
          self.assertEqual(meta.op, network_lib._OP_PUSH)
          layout = network_lib._validate_bulk_manifest(meta)
          self.assertGreater(meta.header_len, 64 * 1024)
          self.assertLen(meta.buffer_lengths, 4098)
          self.assertGreater(meta.total_bulk_bytes, 4 * 1024 * 1024)
          # 2 non-zero bulk views: 1 for large_arr (4 MiB) + 1 trailing
          # _SegmentedBulkView split across multiple stripes.
          self.assertLen([n for n in layout.bulk_lengths if n > 0], 2)
          stripes = network_lib._plan_bulk_stripes(layout.bulk_lengths)
          self.assertGreater(len(stripes), 2)

          async def _push_gen():
            for c in push_chunks:
              yield c

          restored_req = (
              await remote_lib.ExecutionRequest.deserialize_async_chunks(
                  _push_gen(), bulk_server=bulk_server
              )
          )
          restored_push = restored_req.args[0]
          np.testing.assert_array_equal(restored_push["large"], large_arr)
          self.assertEqual(restored_push["large"].ctypes.data % 64, 0)
          self.assertEqual(restored_push["empty"].size, 0)
          self.assertLen(restored_push["small_list"], 4096)
          for idx in (0, 1, 2, 3, 511, 512, 2048, 4095):
            arr_out = restored_push["small_list"][idx]
            np.testing.assert_array_equal(arr_out, small_arrays[idx])
            self.assertEqual(arr_out.ctypes.data % 64, 0)
          # Zero-copy slice views must share the underlying packed buffer, while
          # standalone >= 4 MiB arrays do not share the packed tail buffer.
          self.assertIsNotNone(restored_push["small_list"][1].base)
          self.assertIsNot(
              restored_push["large"].base,
              restored_push["small_list"][1].base,
          )

          # Verify _OP_PULL with the same >64 KiB header and 4,096 small arrays.
          async def _pull_gen():
            async for c in resp.serialize_async_chunks(bulk_server=bulk_server):
              yield c

          restored_resp = (
              await remote_lib.ExecutionResponse.deserialize_async_chunks(
                  _pull_gen(), bulk_host="127.0.0.1"
              )
          )
          assert restored_resp is not None
          restored_pull = restored_resp.unwrap()
          np.testing.assert_array_equal(restored_pull["large"], large_arr)
          self.assertEqual(restored_pull["large"].ctypes.data % 64, 0)
          self.assertEqual(restored_pull["empty"].size, 0)
          for idx in (0, 1, 2, 3, 511, 512, 2048, 4095):
            arr_out = restored_pull["small_list"][idx]
            np.testing.assert_array_equal(arr_out, small_arrays[idx])
            self.assertEqual(arr_out.ctypes.data % 64, 0)

      asyncio.run(_run())
    finally:
      bulk_server.stop()

  @parameterized.named_parameters(
      dict(
          testcase_name="invalid_op_code",
          op=99,
          port=5000,
          total_bulk_bytes=1088,
          header_len=64,
          buffer_lengths=(1024,),
          err_regex="Invalid bulk manifest header fields",
      ),
      dict(
          testcase_name="bool_op_rejected",
          op=True,
          port=5000,
          total_bulk_bytes=1088,
          header_len=64,
          buffer_lengths=(1024,),
          err_regex="Invalid bulk manifest header fields",
      ),
      dict(
          testcase_name="bool_port_rejected",
          op=network_lib._OP_PULL,
          port=True,
          total_bulk_bytes=1088,
          header_len=64,
          buffer_lengths=(1024,),
          err_regex="Invalid bulk manifest header fields",
      ),
      dict(
          testcase_name="port_out_of_range",
          op=network_lib._OP_PULL,
          port=70000,
          total_bulk_bytes=1088,
          header_len=64,
          buffer_lengths=(1024,),
          err_regex="Invalid bulk manifest header fields",
      ),
      dict(
          testcase_name="zero_header_len",
          op=network_lib._OP_PULL,
          port=5000,
          total_bulk_bytes=1024,
          header_len=0,
          buffer_lengths=(1024,),
          err_regex="Invalid bulk manifest header_len",
      ),
      dict(
          testcase_name="bool_header_len_rejected",
          op=network_lib._OP_PULL,
          port=5000,
          total_bulk_bytes=1088,
          header_len=True,
          buffer_lengths=(1024,),
          err_regex="Invalid bulk manifest header_len",
      ),
      dict(
          testcase_name="negative_buffer_length",
          op=network_lib._OP_PULL,
          port=5000,
          total_bulk_bytes=1088,
          header_len=64,
          buffer_lengths=(-4,),
          err_regex="Invalid bulk manifest buffer_lengths",
      ),
      dict(
          testcase_name="bool_buffer_length_rejected",
          op=network_lib._OP_PULL,
          port=5000,
          total_bulk_bytes=1088,
          header_len=64,
          buffer_lengths=(True,),
          err_regex="Invalid bulk manifest buffer_lengths",
      ),
      dict(
          testcase_name="mismatched_total_bulk_bytes",
          op=network_lib._OP_PUSH,
          port=5000,
          total_bulk_bytes=999999,
          header_len=16,
          buffer_lengths=(4 * 1024 * 1024,),
          err_regex="Mismatched total_bulk_bytes in bulk manifest",
      ),
  )
  def test_validate_bulk_manifest_rejects_invalid_fields(
      self,
      op: Any,
      port: Any,
      total_bulk_bytes: Any,
      header_len: Any,
      buffer_lengths: Any,
      err_regex: str,
  ):
    tid = network_lib._allocate_transfer_id()
    manifest = network_lib._BulkManifest(
        op, tid, port, total_bulk_bytes, header_len, buffer_lengths
    )
    with self.assertRaisesRegex(ValueError, err_regex):
      network_lib._validate_bulk_manifest(manifest)

  def test_bulk_manifest_deserialize_rejects_corrupt_and_port_mismatch(self):
    bulk_server = network_lib._BulkTransferServer()
    bulk_server.start()
    try:
      tid = network_lib._allocate_transfer_id()

      async def _single_chunk(data: bytes):
        yield data

      async def _run():
        with self.assertRaisesRegex(
            ValueError, "Invalid stream manifest header"
        ):
          await network_lib._deserialize_from_async_chunks(
              _single_chunk(b"\xff\xfe\xfd")
          )

        wrong_port_push = remote_lib.pickle.dumps(
            network_lib._BulkManifest(
                network_lib._OP_PUSH,
                tid,
                bulk_server.port + 1,
                4 * 1024 * 1024 + 16,
                16,
                (4 * 1024 * 1024,),
            )
        )
        with self.assertRaisesRegex(
            network_lib._BulkTransportError, "Bulk push manifest port mismatch"
        ):
          await network_lib._deserialize_from_async_chunks(
              _single_chunk(wrong_port_push), bulk_server=bulk_server
          )
        with self.assertRaisesRegex(
            network_lib._BulkTransportError,
            "server bulk transport is disabled",
        ):
          await network_lib._deserialize_from_async_chunks(
              _single_chunk(wrong_port_push), bulk_server=None
          )

      asyncio.run(_run())
    finally:
      bulk_server.stop()

  def test_bulk_wire_protocol_fault_injection_and_ttl_eviction(self):
    bulk_server = network_lib._BulkTransferServer()
    port = bulk_server.start()
    try:
      async def _run():
        loop = asyncio.get_running_loop()

        # 1. Invalid OP_PULL bounds fails outgoing transfer future immediately.
        raw_data = bytearray(1024)
        mv = memoryview(raw_data)
        try:
          out_tid, out_fut = bulk_server.register_outgoing(
              [network_lib._SegmentedBulkView((mv,))], 1024, loop
          )
          with socket.create_connection(("127.0.0.1", port), timeout=2.0) as s:
            # Out-of-bounds offset + length > 1024
            s.sendall(
                network_lib._BULK_CMD_STRUCT.pack(
                    network_lib._OP_PULL, out_tid, 0, 0, 1024, 512, 1024
                )
            )
          with self.assertRaisesRegex(ValueError, "Invalid OP_PULL bounds"):
            await asyncio.wait_for(out_fut, timeout=2.0)
          bulk_server.unregister_outgoing(out_tid)
        finally:
          mv.release()

        # 2. Truncated OP_PUSH fails wait_incoming immediately.
        trunc_tid = network_lib._allocate_transfer_id()
        with socket.create_connection(("127.0.0.1", port), timeout=2.0) as s:
          s.sendall(
              network_lib._BULK_CMD_STRUCT.pack(
                  network_lib._OP_PUSH, trunc_tid, 1024, 0, 1024, 0, 1024
              )
          )
          s.sendall(b"x" * 128)
        # Wait briefly for _serve_conn to observe EOF and mark_failed.
        await asyncio.sleep(0.05)
        with self.assertRaisesRegex(ConnectionError, "Bulk push truncated"):
          await bulk_server.wait_incoming(trunc_tid, 1024, [1024])

        # 3. Calling wait_incoming again on an already-completed tid raises
        # RuntimeError immediately instead of hanging.
        with self.assertRaisesRegex(
            RuntimeError, "already completed or evicted"
        ):
          await bulk_server.wait_incoming(trunc_tid, 1024, [1024])

        # 4. V5 + Simplification J regression test: Cumulative buffer allocation
        # across multiple buf_idx values exceeding total_bytes closes the
        # connection (EOF b"") and fails wait_incoming.
        over_tid = network_lib._allocate_transfer_id()
        with socket.create_connection(("127.0.0.1", port), timeout=2.0) as s:
          s.sendall(
              network_lib._BULK_CMD_STRUCT.pack(
                  network_lib._OP_PUSH, over_tid, 1024, 0, 1024, 0, 512
              )
          )
          s.sendall(b"a" * 512)
          ack1 = s.recv(1)
          self.assertEqual(ack1, b"\x01")
          # Second command on same tid requests buf_idx=1 with
          # buf_total_len=1024 (cumulative 2048 > total_bytes 1024) -> EOF.
          s.sendall(
              network_lib._BULK_CMD_STRUCT.pack(
                  network_lib._OP_PUSH, over_tid, 1024, 1, 1024, 0, 512
              )
          )
          ack2 = s.recv(1)
          self.assertEqual(ack2, b"")
        with self.assertRaisesRegex(
            ValueError, "Invalid bulk buffer allocation exceeds declared"
        ):
          await bulk_server.wait_incoming(over_tid, 1024, [512, 512])

      asyncio.run(_run())
    finally:
      bulk_server.stop()

  def test_bulk_unclaimed_budget_wait_and_wakeup(self):
    # H1 + R3 + R3' + T2(d) deterministic thread regression test:
    # 1. Case 1 (R3): While unclaimed orphan `tid_a` occupies budget, a stripe
    #    thread calling `_get_or_create_incoming(tid_b, ..., claim=False)`
    #    blocks inside `_cond.wait_for`. When `tid_a` is claimed, `tid_b`'s
    #    stripe thread wakes up immediately (<< 5s) and allocates `tid_b`.
    # 2. Case 2 (R3'): While unclaimed orphan `tid_orphan` continues occupying
    #    budget, a stripe thread calling `_get_or_create_incoming(tid_c, ...,
    #    claim=False)` blocks inside `_cond.wait_for`. When `wait_incoming`
    #    calls `_get_or_create_incoming(tid_c, ..., claim=True)`, inserting the
    #    claimed entry notifies `_cond` and wakes `tid_c`'s stripe thread
    #    immediately (<< 5s) even though `tid_orphan` is still unclaimed.
    # 3. Case 3: When an orphan is never claimed and `tid` is never claimed,
    #    `_get_or_create_incoming` raises `MemoryError` after timeout.
    bulk_server = network_lib._BulkTransferServer()
    bulk_server.start()
    try:
      with (
          mock.patch.object(network_lib, "_MAX_UNCLAIMED_INCOMING_BYTES", 2048),
          mock.patch.object(network_lib, "_UNCLAIMED_BUDGET_WAIT_S", 5.0),
      ):
        # --- Case 1: Claiming orphan A wakes waiting stripe thread for B ---
        tid_a = network_lib._allocate_transfer_id()
        entry_a = bulk_server._get_or_create_incoming(tid_a, 1500, claim=False)
        self.assertIsNotNone(entry_a)
        self.assertEqual(bulk_server._unclaimed_bytes, 1500)

        tid_b = network_lib._allocate_transfer_id()
        res_b: list[Optional[network_lib._IncomingBulkTransfer]] = []
        entered_b = threading.Event()
        real_wait_for = bulk_server._cond.wait_for

        def _hooked_wait_for(predicate, timeout=None):
          entered_b.set()
          return real_wait_for(predicate, timeout=timeout)

        with mock.patch.object(
            bulk_server._cond, "wait_for", side_effect=_hooked_wait_for
        ):
          t_b = threading.Thread(
              target=lambda: res_b.append(
                  bulk_server._get_or_create_incoming(tid_b, 1000, claim=False)
              ),
              daemon=True,
          )
          t_b.start()
          self.assertTrue(entered_b.wait(timeout=2.0))
          t0 = time.monotonic()
          # Claiming tid_a releases its 1500 unclaimed bytes and notifies _cond.
          claimed_a = bulk_server._get_or_create_incoming(
              tid_a, 1500, claim=True
          )
          self.assertIs(claimed_a, entry_a)
          t_b.join(timeout=2.0)
          self.assertFalse(t_b.is_alive())
          self.assertLess(time.monotonic() - t0, 0.5)
          self.assertLen(res_b, 1)
          self.assertIsNotNone(res_b[0])

        # Clean up tid_b (1000 unclaimed bytes) by claiming it.
        bulk_server._get_or_create_incoming(tid_b, 1000, claim=True)
        self.assertEqual(bulk_server._unclaimed_bytes, 0)

        # --- Case 2 (R3'): Claiming C itself wakes C's stripe thread while
        # orphan remains unclaimed ---
        tid_orphan = network_lib._allocate_transfer_id()
        entry_orphan = bulk_server._get_or_create_incoming(
            tid_orphan, 1500, claim=False
        )
        self.assertIsNotNone(entry_orphan)
        self.assertEqual(bulk_server._unclaimed_bytes, 1500)

        tid_c = network_lib._allocate_transfer_id()
        res_c: list[Optional[network_lib._IncomingBulkTransfer]] = []
        entered_c = threading.Event()

        def _hooked_wait_for_c(predicate, timeout=None):
          entered_c.set()
          return real_wait_for(predicate, timeout=timeout)

        with mock.patch.object(
            bulk_server._cond, "wait_for", side_effect=_hooked_wait_for_c
        ):
          t_c = threading.Thread(
              target=lambda: res_c.append(
                  bulk_server._get_or_create_incoming(tid_c, 1000, claim=False)
              ),
              daemon=True,
          )
          t_c.start()
          self.assertTrue(entered_c.wait(timeout=2.0))
          t1 = time.monotonic()
          # Simulate wait_incoming(tid_c) arriving while tid_orphan still holds
          # the unclaimed budget.
          claimed_c = bulk_server._get_or_create_incoming(
              tid_c, 1000, claim=True
          )
          self.assertIsNotNone(claimed_c)
          t_c.join(timeout=2.0)
          self.assertFalse(t_c.is_alive())
          self.assertLess(time.monotonic() - t1, 0.5)
          self.assertLen(res_c, 1)
          self.assertIs(res_c[0], claimed_c)

        # --- Case 3: Unclaimed budget timeout raises MemoryError ---
        with mock.patch.object(network_lib, "_UNCLAIMED_BUDGET_WAIT_S", 0.03):
          tid_d = network_lib._allocate_transfer_id()
          with self.assertRaisesRegex(MemoryError, "byte budget exceeded"):
            bulk_server._get_or_create_incoming(tid_d, 1000, claim=False)
    finally:
      bulk_server.stop()

  def test_bulk_transfer_cancellation_aborts_sockets_without_blocking_event_loop(
      self,
  ):
    # Create a raw listening TCP socket that accepts connections and holds them
    # open without sending/acking, using an Event to synchronize before cancel.
    stall_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    stall_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    stall_listener.bind(("127.0.0.1", 0))
    stall_listener.listen(8)
    stall_port = int(stall_listener.getsockname()[1])
    accepted_conns: list[socket.socket] = []
    accepted_event = threading.Event()

    def _accept_loop() -> None:
      while True:
        try:
          conn, _ = stall_listener.accept()
          accepted_conns.append(conn)
          accepted_event.set()
        except OSError:
          break

    accept_thread = threading.Thread(target=_accept_loop, daemon=True)
    accept_thread.start()

    try:
      async def _run():
        # 1. Cancel _OP_PULL blocked on recv_into; use a real threading.Event
        # set in the finally block of _pull_stripe_sync (V6) to synchronize on
        # worker thread exit before checking the socket pool.
        tid = network_lib._allocate_transfer_id()
        header_bytes = remote_lib.cloudpickle.dumps(
            ("ok", None, None, None, False, "r1")
        )
        pull_manifest = remote_lib.pickle.dumps(
            network_lib._BulkManifest(
                network_lib._OP_PULL,
                tid,
                stall_port,
                4 * 1024 * 1024 + len(header_bytes),
                len(header_bytes),
                (4 * 1024 * 1024,),
            )
        )

        async def _chunk_gen():
          yield pull_manifest

        pull_layout = network_lib._validate_bulk_manifest(
            remote_lib.pickle.loads(pull_manifest)
        )
        expected_pull_stripes = len(
            network_lib._plan_bulk_stripes(pull_layout.bulk_lengths)
        )
        pull_remaining = expected_pull_stripes
        pull_lock = threading.Lock()
        pull_stripe_exited = threading.Event()
        real_pull_stripe = network_lib._pull_stripe_sync

        def _wrapped_pull_stripe(*args, **kwargs) -> None:
          nonlocal pull_remaining
          try:
            real_pull_stripe(*args, **kwargs)
          finally:
            with pull_lock:
              pull_remaining -= 1
              if pull_remaining == 0:
                pull_stripe_exited.set()

        accepted_event.clear()
        with mock.patch.object(
            network_lib, "_pull_stripe_sync", side_effect=_wrapped_pull_stripe
        ):
          pull_task = asyncio.create_task(
              remote_lib.ExecutionResponse.deserialize_async_chunks(
                  _chunk_gen(), bulk_host="127.0.0.1"
              )
          )
          await asyncio.to_thread(accepted_event.wait, 5.0)
          t0 = time.monotonic()
          pull_task.cancel()
          with self.assertRaises(asyncio.CancelledError):
            await pull_task
          self.assertLess(time.monotonic() - t0, 1.0)
          self.assertTrue(await asyncio.to_thread(pull_stripe_exited.wait, 5.0))
        self.assertEmpty(
            network_lib._DEFAULT_BULK_SOCKET_POOL._pools.get(
                ("127.0.0.1", stall_port), ()
            )
        )

        # 2. Cancel _OP_PUSH blocked waiting for the 1-byte server ACK; use a
        # real threading.Event set in the finally block of _push_stripe_sync.
        accepted_event.clear()
        arr = np.ones(256 * 1024, dtype=np.int32)
        req = remote_lib.ExecutionRequest(
            request_id="req_cancel_push",
            method_name="echo",
            args=({"tensor": arr},),
            kwargs={},
        )
        push_tracker = network_lib._BulkPushTracker()
        push_stripe_exited = threading.Event()
        real_push_stripe = network_lib._push_stripe_sync

        def _wrapped_push_stripe(*args, **kwargs) -> None:
          try:
            real_push_stripe(*args, **kwargs)
          finally:
            push_stripe_exited.set()

        async def _drain_push():
          _, _, views, raw_buffers, _ = (
              network_lib._prepare_serialized_chunk_specs(req._as_tuple())
          )
          async for _ in network_lib._stream_bulk_push_chunks(
              views,
              raw_buffers,
              "127.0.0.1",
              stall_port,
              push_tracker=push_tracker,
          ):
            pass

        with mock.patch.object(
            network_lib, "_push_stripe_sync", side_effect=_wrapped_push_stripe
        ):
          push_task = asyncio.create_task(_drain_push())
          await asyncio.to_thread(accepted_event.wait, 5.0)
          t1 = time.monotonic()
          push_task.cancel()
          with self.assertRaises(asyncio.CancelledError):
            await push_task
          self.assertLess(time.monotonic() - t1, 1.0)
          self.assertTrue(await asyncio.to_thread(push_stripe_exited.wait, 5.0))
        self.assertEmpty(
            network_lib._DEFAULT_BULK_SOCKET_POOL._pools.get(
                ("127.0.0.1", stall_port), ()
            )
        )

        # 3. W6 + T1(a) + T2(a) regression test: Cancel _stream_bulk_pull_chunks
        # on server side using deterministic Event synchronization while a
        # client connection is actively inside vectored sendmsg() on a 256 MiB
        # packed tail, ensuring _release_views does not raise BufferError and
        # propagates CancelledError cleanly.
        bulk_server = network_lib._BulkTransferServer()
        bulk_server.start()
        try:
          obj = {
              "small": [
                  np.full(64 * 1024, i, dtype=np.float32) for i in range(1024)
              ]
          }
          _, _, views, raw_buffers, _ = (
              network_lib._prepare_serialized_chunk_specs(obj)
          )
          pull_gen = network_lib._stream_bulk_pull_chunks(
              views, raw_buffers, bulk_server
          )
          first_chunk = await pull_gen.__anext__()
          _, pull_tid, pull_port, total_bulk, _, buf_lens = (
              remote_lib.pickle.loads(first_chunk)
          )
          packed_idx = len(buf_lens)
          entered_sendmsg = threading.Event()
          real_send_slice = network_lib._send_bulk_view_slice

          def _notify_send_slice(
              sock: socket.socket,
              view: network_lib._SegmentedBulkView,
              offset: int,
              length: int,
          ) -> None:
            entered_sendmsg.set()
            real_send_slice(sock, view, offset, length)

          with mock.patch.object(
              network_lib,
              "_send_bulk_view_slice",
              side_effect=_notify_send_slice,
          ):
            with socket.create_connection(
                ("127.0.0.1", pull_port), timeout=2.0
            ) as raw_s:
              raw_s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
              raw_s.sendall(
                  network_lib._BULK_CMD_STRUCT.pack(
                      network_lib._OP_PULL,
                      pull_tid,
                      0,
                      packed_idx,
                      total_bulk,
                      0,
                      total_bulk,
                  )
              )
              await asyncio.to_thread(entered_sendmsg.wait, 5.0)
              waiter = asyncio.create_task(pull_gen.__anext__())
              await asyncio.sleep(0)
              waiter.cancel()
              with self.assertRaises(asyncio.CancelledError):
                await waiter
              for mv in views:
                with self.assertRaises(ValueError):
                  _ = mv.nbytes
        finally:
          bulk_server.stop()

      asyncio.run(_run())
    finally:
      stall_listener.close()
      for c in accepted_conns:
        c.close()

  def test_bulk_socket_pool_evicts_expired_idle_sockets_and_handles_fork(self):
    bulk_server = network_lib._BulkTransferServer()
    port = bulk_server.start()
    pool = network_lib._BulkSocketPool()
    try:
      s1 = pool.acquire("127.0.0.1", port)
      pool.release("127.0.0.1", port, s1)
      # Backdate s1's release timestamp past _MAX_IDLE_SOCKET_AGE_S.
      key = ("127.0.0.1", port)
      sock_obj, _ = pool._pools[key].pop()
      pool._pools[key].append((
          sock_obj,
          time.monotonic() - network_lib._MAX_IDLE_SOCKET_AGE_S - 5.0,
      ))
      s2 = pool.acquire("127.0.0.1", port)
      self.assertIsNot(s2, s1)
      self.assertEqual(s1.fileno(), -1)
      pool.release("127.0.0.1", port, s2)

      # Simulate os.fork() PID change: inherited sockets must be closed and
      # cleared on next pool operation.
      pool._pid = -1
      s3 = pool.acquire("127.0.0.1", port)
      self.assertIsNot(s3, s2)
      self.assertEqual(s2.fileno(), -1)
      pool.release("127.0.0.1", port, s3)

      # Verify GrpcRemoteExecutionServer(bulk_port=port).start_serving_async()
      # rolls back cleanly when `port` is already in use by `bulk_server`.
      async def _check_port_conflict() -> None:
        grpc_port = portpicker.pick_unused_port()
        conflict_srv = remote_lib.GrpcRemoteExecutionServer(
            StubWorkerEngine("conflict_worker"), bulk_port=port
        )
        with self.assertRaises(OSError):
          await conflict_srv.start_serving_async(port=grpc_port)
        self.assertIsNone(conflict_srv._bulk_server)
        self.assertIsNone(conflict_srv._server)

      asyncio.run(_check_port_conflict())
    finally:
      pool.close_target("127.0.0.1", port)
      bulk_server.stop()

  @parameterized.named_parameters(
      dict(
          testcase_name="grpc_ipv6_with_port",
          target="grpc://[::1]:50051",
          expected=("::1", 50051),
      ),
      dict(
          testcase_name="bare_ipv6_with_port",
          target="[2001:db8::1]:8080",
          expected=("2001:db8::1", 8080),
      ),
      dict(
          testcase_name="bare_ipv6_without_port",
          target="[::1]",
          expected=("::1", None),
      ),
      dict(
          testcase_name="bare_ipv4_with_port",
          target="10.0.0.5:9000",
          expected=("10.0.0.5", 9000),
      ),
      dict(
          testcase_name="bare_hostname_without_port",
          target="localhost",
          expected=("localhost", None),
      ),
  )
  def test_extract_host_and_port_valid(
      self, target: str, expected: tuple[str, Optional[int]]
  ):
    self.assertEqual(network_lib._extract_host_and_port(target), expected)

  @parameterized.named_parameters(
      dict(testcase_name="http_scheme", target="http://10.0.0.5:9000"),
      dict(testcase_name="dns_scheme", target="dns:///worker.ns.svc:50051"),
      dict(testcase_name="ipv4_scheme", target="ipv4:10.0.0.5:9000"),
      dict(testcase_name="unbracketed_ipv6", target="::1:50051"),
      dict(testcase_name="empty_brackets", target="[]:50051"),
      dict(testcase_name="port_zero", target="localhost:0"),
      dict(testcase_name="port_above_65535", target="localhost:70000"),
      dict(testcase_name="fullwidth_unicode_port", target="localhost:５０"),
  )
  def test_extract_host_and_port_invalid(self, target: str):
    with self.assertRaises(ValueError):
      network_lib._extract_host_and_port(target)

  def test_non_host_port_grpc_scheme_allowed_when_bulk_disabled(self):
    # A1 regression test: non-host[:port] gRPC target schemes (e.g. dns:/// or
    # unix:) are rejected with a clear message when enable_bulk_transport=True,
    # and accepted when enable_bulk_transport=False.
    with self.assertRaisesRegex(
        ValueError, "pass enable_bulk_transport=False"
    ):
      remote_lib.GrpcRemoteActorHandle(
          "grpc://dns:///worker.ns.svc:50051", enable_bulk_transport=True
      )
    hdl = remote_lib.GrpcRemoteActorHandle(
        "grpc://dns:///worker.ns.svc:50051", enable_bulk_transport=False
    )
    self.assertEqual(hdl._host_port, "dns:///worker.ns.svc:50051")
    self.assertEqual(hdl._bulk_host, "")
    self.assertFalse(hdl._enable_bulk_transport)


if __name__ == "__main__":
  absltest.main()
