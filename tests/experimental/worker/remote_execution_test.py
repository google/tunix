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
import socket
import threading
import time
from absl.testing import absltest
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
    return f"[{self.worker_id}] Trajectory for prompt {prompt_id} ({turns} turns)"

  async def __call__(
      self, prompt_id: str = "default_prompt", turns: int = 3
  ) -> str:
    return await self.compute_trajectory(prompt_id, turns)


  def pause(self) -> None:
    self.is_paused = True

  def resume(self) -> None:
    self.is_paused = False

  def get_status(self) -> str:
    return f"worker_id={self.worker_id}, count={self.call_count}, paused={self.is_paused}"

  def kv_cache_aware(self, prompt: str = "test") -> str:
    return f"[{self.worker_id}] KV-cache aware routing for {prompt}"





class RemoteExecutionTest(absltest.TestCase):
  """Tests verifying ActorHandle and ActorPool dynamic routing."""

  def test_execution_request_serialization(self):
    req = remote_lib.ExecutionRequest("compute_trajectory", args=("prompt_1",), kwargs={"turns": 5})
    payload = req.serialize()
    restored = remote_lib.ExecutionRequest.deserialize(payload)
    self.assertEqual(restored.method_name, "compute_trajectory")
    self.assertEqual(restored.args, ("prompt_1",))
    self.assertEqual(restored.kwargs, {"turns": 5})

  def test_actor_handle_sync_and_async_invocation(self):
    async def _run_test():
      engine = StubWorkerEngine("actor_01", latency=0.01)
      server = remote_lib.InProcessRemoteExecutionServer(engine)
      handle = remote_lib.InProcessActorHandle(server)

      # Test async execution via asubmit
      res = await handle.asubmit("compute_trajectory", "prompt_a", turns=2)
      self.assertEqual(res, "[actor_01] Trajectory for prompt prompt_a (2 turns)")

      # Test sync execution via submit
      status = await handle.asubmit("get_status")
      self.assertIn("count=1", status)

      # Test exception propagation when paused
      await handle.asubmit("pause")
      with self.assertRaises(RuntimeError) as cm:
        await handle.asubmit("compute_trajectory", "prompt_b")
      self.assertIn("currently paused", str(cm.exception))

    asyncio.run(_run_test())

  def test_actor_pool_load_balancing_and_streaming(self):
    async def _run_test():
      engine_a = StubWorkerEngine("worker_A", latency=0.04)
      engine_b = StubWorkerEngine("worker_B", latency=0.01)

      server_a = remote_lib.InProcessRemoteExecutionServer(engine_a)
      server_b = remote_lib.InProcessRemoteExecutionServer(engine_b)

      handle_a = remote_lib.InProcessActorHandle(server_a)
      handle_b = remote_lib.InProcessActorHandle(server_b)

      pool = remote_lib.RoutingActorPool([handle_a, handle_b])

      # Submit batch of tasks across pool and verify out-of-order completion stream
      tasks = [
          ("compute_trajectory", ("req_slow_on_A",), {"turns": 1}),
          ("compute_trajectory", ("req_fast_on_B",), {"turns": 1}),
      ]

      results = []
      async for res in pool.as_completed_stream(tasks):
        results.append(res)

      # worker_B (latency 0.01) should finish before worker_A (latency 0.04)
      self.assertLen(results, 2)
      self.assertIn("[worker_B] Trajectory for prompt req_fast_on_B", results[0])
      self.assertIn("[worker_A] Trajectory for prompt req_slow_on_A", results[1])

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

      # Verify sticky routing: identical route_keys consistently route to the same worker without method_name
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
      def custom_router(actors, method_name, _args, kwargs):
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
      # KV-cache aware call routes directly to actors[1]
      res_kv = await smart_pool.asubmit("kv_cache_aware", "prompt_v2")
      self.assertIn("[worker_1] KV-cache aware routing for prompt_v2", res_kv)


      # Verify router object providing a method matching `method_name` (`self.router."method_name"()`)
      class MethodSpecificRouter:

        def kv_cache_aware(self, actors, _args, _kwargs):
          return actors[1]

      method_pool = remote_lib.RoutingActorPool(
          [handle_0, handle_1], router=MethodSpecificRouter()
      )
      res_method = await method_pool.asubmit("kv_cache_aware", "any_prompt")
      self.assertIn("[worker_1]", res_method)


    asyncio.run(_run_test())




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
    self.assertEqual(actor_handle.submit("greet", "World"), "Hello World from WorkerX")

    # Verify standalone function task (inprocess)
    self.assertEqual(standalone_task.remote(5), 50)
    self.assertEqual(async_standalone_task.remote(5), 100)

    # Verify coroutine function task when called inside an active async event loop
    async def _verify_in_loop():
      res = await async_standalone_task.remote(7)
      self.assertEqual(res, 140)

    asyncio.run(_verify_in_loop())

    # Verify grpc class actor factory (remote gRPC target via string address)
    grpc_handle = GrpcWorker.remote()

    self.assertIsInstance(grpc_handle, remote_lib.GrpcRemoteActorHandle)
    self.assertEqual(grpc_handle.target_address, "grpc://fake-pod:50051")

    # Verify grpc class actor factory (remote gRPC target via explicit address kwarg)
    explicit_handle = ExplicitGrpcWorker.remote()
    self.assertIsInstance(explicit_handle, remote_lib.GrpcRemoteActorHandle)
    self.assertEqual(explicit_handle.target_address, "grpc://fake-pod:50051")

    # Verify late address binding (dynamic pod allocation at runtime)
    late_handle = DynamicWorker.remote(address="grpc://allocated-pod:50051")
    self.assertIsInstance(late_handle, remote_lib.GrpcRemoteActorHandle)
    self.assertEqual(
        late_handle.target_address, "grpc://allocated-pod:50051"
    )

    # Verify standalone function with non-inprocess transport raises NotImplementedError
    with self.assertRaises(NotImplementedError):
      @remote_lib.remote("grpc://fake-pod:50051")
      def remote_grpc_func(x: int) -> int:
        return x * 2


  def test_real_grpc_tcp_execution(self):
    """Verifies GrpcRemoteExecutionServer and GrpcRemoteActorHandle over physical TCP sockets."""
    if not remote_lib._GRPC_AVAILABLE:
      self.skipTest("grpc library not available in test environment.")

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
        await handle.close()
      finally:
        await server.stop_serving()

    asyncio.run(_run_test())


def _free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("localhost", 0))
    return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
  deadline = time.time() + timeout
  while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
      s.settimeout(0.2)
      if s.connect_ex((host, port)) == 0:
        return True
    time.sleep(0.05)
  return False


def _run_server_in_background(engine, port):
  """Starts a gRPC server on its own background event loop.

  Returns (server, loop, thread) so the caller can drive blocking client calls
  from the main thread while the server runs elsewhere.
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
  return server, loop, thread


def _shutdown_background_server(server, loop, thread) -> None:
  asyncio.run_coroutine_threadsafe(server.stop_serving(), loop).result(timeout=5)
  loop.call_soon_threadsafe(loop.stop)
  thread.join(timeout=5)
  loop.close()


class _EchoEngine:

  def echo(self, blob):
    return blob


class _BoomEngine:

  def explode(self):
    raise ValueError("kaboom")


class _LockReturner:

  def get(self):
    return threading.Lock()  # not serializable by (cloud)pickle


class SubstrateHardeningTest(absltest.TestCase):
  """Regression tests for the RPC substrate correctness/robustness fixes."""

  def test_execution_response_error_round_trips(self):
    resp = remote_lib.ExecutionResponse(
        error_message="boom",
        error_type="ValueError",
        traceback="Traceback: ...",
        retryable=True,
    )

    restored = remote_lib.ExecutionResponse.deserialize(resp.serialize())

    self.assertEqual(restored.error_type, "ValueError")
    self.assertEqual(restored.error_message, "boom")
    self.assertEqual(restored.traceback, "Traceback: ...")
    self.assertTrue(restored.retryable)
    with self.assertRaises(RuntimeError):
      restored.unwrap()

  def test_execute_request_captures_traceback(self):
    async def _run():
      server = remote_lib.InProcessRemoteExecutionServer(_BoomEngine())
      resp = await server.execute_request(
          remote_lib.ExecutionRequest("explode")
      )
      self.assertEqual(resp.error_type, "ValueError")
      self.assertIn("kaboom", resp.error_message)
      self.assertIsNotNone(resp.traceback)
      self.assertIn("explode", resp.traceback)

    asyncio.run(_run())

  def test_handle_execute_returns_error_for_unserializable_result(self):
    async def _run():
      server = remote_lib.GrpcRemoteExecutionServer(_LockReturner())
      request_bytes = remote_lib.ExecutionRequest("get").serialize()

      response_bytes = await server._handle_execute(request_bytes, context=None)

      resp = remote_lib.ExecutionResponse.deserialize(response_bytes)
      self.assertEqual(resp.error_type, "ResultSerializationError")
      self.assertIsNotNone(resp.error_message)

    asyncio.run(_run())

  def test_grpc_large_payload_round_trips(self):
    if not remote_lib._GRPC_AVAILABLE:
      self.skipTest("grpc library not available in test environment.")

    async def _run():
      port = _free_port()
      server = remote_lib.GrpcRemoteExecutionServer(_EchoEngine())
      await server.start_serving_async(port=port)
      try:
        handle = remote_lib.GrpcRemoteActorHandle(
            target_address=f"grpc://localhost:{port}"
        )
        # 8 MiB exceeds gRPC's default ~4 MiB message cap.
        blob = b"x" * (8 * 1024 * 1024)
        echoed = await handle.asubmit("echo", blob)
        self.assertEqual(len(echoed), len(blob))
        await handle.close()
      finally:
        await server.stop_serving()

    asyncio.run(_run())

  def test_grpc_asubmit_times_out_on_slow_worker(self):
    if not remote_lib._GRPC_AVAILABLE:
      self.skipTest("grpc library not available in test environment.")

    async def _run():
      port = _free_port()
      server = remote_lib.GrpcRemoteExecutionServer(
          StubWorkerEngine("slow_worker", latency=2.0)
      )
      await server.start_serving_async(port=port)
      try:
        handle = remote_lib.GrpcRemoteActorHandle(
            target_address=f"grpc://localhost:{port}", default_timeout_s=0.3
        )
        with self.assertRaises(Exception) as cm:
          await handle.asubmit("compute_trajectory", "p", turns=1)
        self.assertIn("deadline", str(cm.exception).lower())
        await handle.close()
      finally:
        await server.stop_serving()

    asyncio.run(_run())

  def test_grpc_sync_submit_survives_repeated_calls(self):
    if not remote_lib._GRPC_AVAILABLE:
      self.skipTest("grpc library not available in test environment.")

    port = _free_port()
    server, loop, thread = _run_server_in_background(
        StubWorkerEngine("sync_worker"), port
    )
    try:
      handle = remote_lib.GrpcRemoteActorHandle(
          target_address=f"grpc://localhost:{port}"
      )
      first = handle.submit("compute_trajectory", "p1", turns=1)
      self.assertIn("sync_worker", first)
      # The 2nd blocking call previously failed by reusing a channel bound to
      # the 1st call's now-closed event loop.
      second = handle.submit("compute_trajectory", "p2", turns=1)
      self.assertIn("sync_worker", second)
    finally:
      _shutdown_background_server(server, loop, thread)

  def test_grpc_sync_start_serving_actually_serves(self):
    if not remote_lib._GRPC_AVAILABLE:
      self.skipTest("grpc library not available in test environment.")

    port = _free_port()
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
        # Fire-and-forget: triggering wait_for_termination unblocks
        # start_serving, which then closes its own loop. The thread exiting is
        # the real signal that the blocking serve returned, so we don't await
        # the stop future (whose result races with that loop closing).
        asyncio.run_coroutine_threadsafe(server.stop_serving(), serve_loop)
      thread.join(timeout=10)
      self.assertFalse(
          thread.is_alive(), "blocking start_serving did not shut down"
      )


if __name__ == "__main__":
  absltest.main()
