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

  def test_actor_handle_sync_and_async_invocation(self):
    async def _run_test():
      engine = StubWorkerEngine("actor_01", latency=0.01)
      server = remote_lib.InProcessRemoteExecutionServer(engine)
      handle = remote_lib.InProcessActorHandle(server)

      # Test async execution via asubmit
      res = await handle.asubmit("compute_trajectory", "prompt_a", turns=2)
      self.assertEqual(
          res, "[actor_01] Trajectory for prompt prompt_a (2 turns)"
      )

      # Test sync execution via submit
      status = handle.submit("get_status")
      self.assertIn("count=1", status)

      # Test sync execution of a coroutine from inside an event loop raises RuntimeError
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

      # Verify sticky routing: identical route_keys consistently route to the same worker without
      # providing method_name argument
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

      # Test synchronous submit across the pool
      sync_res = smart_pool.submit("get_status")
      self.assertIn("count=", sync_res)

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
    self.assertEqual(late_handle.target_address, "grpc://allocated-pod:50051")

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

        # Test GrpcRemoteActorHandle.submit cannot be called inside an event loop
        with self.assertRaisesRegex(
            RuntimeError,
            "submit\\(\\) cannot be called from a running async event loop",
        ):
          handle.submit("get_status")

        # Test GrpcRemoteExecutionServer.start_serving cannot be called inside an event loop
        with self.assertRaisesRegex(
            RuntimeError,
            "start_serving\\(\\) cannot be called from a running async event"
            " loop",
        ):
          server.start_serving(port)

        await handle.close()
      finally:
        await server.stop_serving()

    asyncio.run(_run_test())

  def test_remote_async_func(self):
    @remote_lib.remote
    async def standalone_async_task(x: int) -> int:
      await asyncio.sleep(0.01)
      return x * 10

    async def _run_test():
      res = await standalone_async_task.remote(5)
      self.assertEqual(res, 50)

    asyncio.run(_run_test())

  def test_execution_request_default_call(self):
    req = remote_lib.ExecutionRequest()
    self.assertEqual(req.method_name, "__call__")

    engine = StubWorkerEngine("actor_call")
    server = remote_lib.InProcessRemoteExecutionServer(engine)
    handle = remote_lib.InProcessActorHandle(server)

    res = handle.submit()
    self.assertEqual(
        res, "[actor_call] Trajectory for prompt default_prompt (3 turns)"
    )

  def test_server_register_instance(self):
    server = remote_lib.InProcessRemoteExecutionServer()
    self.assertIsNone(server.bound_instance)

    engine = StubWorkerEngine("dynamic_worker")
    server.register_instance(engine)
    self.assertIsNotNone(server.bound_instance)

    handle = remote_lib.InProcessActorHandle(server)
    self.assertIn("dynamic_worker", handle.submit("get_status"))

  def test_execution_response_serialization_success(self):
    res_success = remote_lib.ExecutionResponse(result=42)
    payload = res_success.serialize()
    restored = remote_lib.ExecutionResponse.deserialize(payload)
    self.assertEqual(restored.unwrap(), 42)

  def test_execution_response_serialization_error_unwrap(self):
    res_error = remote_lib.ExecutionResponse(
        error_message="Oops", error_type="ValueError"
    )
    payload_err = res_error.serialize()
    restored_err = remote_lib.ExecutionResponse.deserialize(payload_err)
    with self.assertRaisesRegex(
        RuntimeError, r"RemoteExecutionError \[ValueError\]: Oops"
    ):
      restored_err.unwrap()

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

  def test_server_error_sync_request_to_coroutine_method(self):
    server = remote_lib.InProcessRemoteExecutionServer(
        StubWorkerEngine("worker_01")
    )
    resp3 = server.execute_sync_request(
        remote_lib.ExecutionRequest("compute_trajectory", args=("p1",))
    )
    self.assertEqual(resp3.error_type, "RuntimeError")
    self.assertEqual(
        resp3.error_message,
        "Method 'compute_trajectory' is a coroutine function; use asubmit().",
    )

  def test_server_error_exception_during_sync_execution(self):
    class ThrowingWorker:

      def fail(self):
        raise ValueError("Intentional failure")

    server = remote_lib.InProcessRemoteExecutionServer(ThrowingWorker())
    resp4 = server.execute_sync_request(remote_lib.ExecutionRequest("fail"))
    self.assertEqual(resp4.error_type, "ValueError")
    self.assertEqual(resp4.error_message, "Intentional failure")

  def test_server_error_exception_during_async_execution(self):
    class ThrowingWorker:

      async def async_fail(self):
        raise ValueError("Async intentional failure")

    server = remote_lib.InProcessRemoteExecutionServer(ThrowingWorker())

    async def _run_async_err():
      resp5 = await server.execute_request(
          remote_lib.ExecutionRequest("async_fail")
      )
      self.assertEqual(resp5.error_type, "ValueError")

    asyncio.run(_run_async_err())

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


if __name__ == "__main__":
  absltest.main()
