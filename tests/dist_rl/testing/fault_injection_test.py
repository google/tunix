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

"""Tests for FaultGate, FaultInjector, FaultInjectingServer, and RunningWorkerServer."""

import asyncio
import threading
from absl.testing import absltest
import grpc
from tunix.dist_rl.testing import fault_injection
from tunix.dist_rl.worker import remote_execution


class _DummyWorker:

  def __init__(self) -> None:
    self.counter = 0

  def ping(self, msg: str = "pong") -> str:
    self.counter += 1
    return msg

  async def ping_async(self, msg: str = "pong_async") -> str:
    self.counter += 1
    return msg


class FaultInjectionTest(absltest.TestCase):

  def test_fault_gate_sync_and_async(self) -> None:
    gate = fault_injection.FaultGate()
    loop = asyncio.new_event_loop()
    async_entered = threading.Event()
    async_released = threading.Event()

    async def _async_waiter():
      async_entered.set()
      await gate.wait_async()
      async_released.set()

    t = threading.Thread(
        target=lambda: loop.run_until_complete(_async_waiter()), daemon=True
    )
    t.start()
    self.assertTrue(async_entered.wait(timeout=5.0))
    self.assertTrue(gate.entered.wait(timeout=5.0))
    self.assertFalse(async_released.is_set())

    gate.release()
    self.assertTrue(async_released.wait(timeout=5.0))
    t.join(timeout=5.0)
    loop.close()

    # Re-release and sync wait should succeed immediately
    gate.wait_sync(timeout=1.0)

  def test_fault_injector_rules(self) -> None:
    injector = fault_injection.FaultInjector()
    gate = fault_injection.FaultGate()

    rule_1 = injector.on(
        worker_id="w0",
        method_name="test_method",
        action=fault_injection.Block(gate),
        nth=2,
    )

    req = remote_execution.ExecutionRequest(method_name="test_method")
    # 1st call matches but does not fire (nth=2)
    self.assertIsNone(injector.match("w0", req))
    self.assertEqual(rule_1.matched_count, 1)
    self.assertEqual(rule_1.fired_count, 0)

    # 2nd call matches and fires
    action = injector.match("w0", req)
    self.assertIsInstance(action, fault_injection.Block)
    self.assertEqual(rule_1.matched_count, 2)
    self.assertEqual(rule_1.fired_count, 1)

    # 3rd call matches but does not fire
    self.assertIsNone(injector.match("w0", req))
    self.assertEqual(rule_1.matched_count, 3)
    self.assertEqual(rule_1.fired_count, 1)

  def test_fault_injector_release_all_gates(self) -> None:
    injector = fault_injection.FaultInjector()
    gate1 = fault_injection.FaultGate()
    gate2 = fault_injection.FaultGate()

    injector.on("w0", "m1", fault_injection.Block(gate1))
    injector.on("w0", "m2", fault_injection.Block(gate2))
    injector.on("w0", "m3", fault_injection.Return("ok"))

    done1 = threading.Event()
    done2 = threading.Event()

    def _waiter(gate: fault_injection.FaultGate, ev: threading.Event) -> None:
      gate.wait_sync(timeout=5.0)
      ev.set()

    t1 = threading.Thread(target=_waiter, args=(gate1, done1), daemon=True)
    t2 = threading.Thread(target=_waiter, args=(gate2, done2), daemon=True)
    t1.start()
    t2.start()

    self.assertTrue(gate1.entered.wait(timeout=5.0))
    self.assertTrue(gate2.entered.wait(timeout=5.0))
    self.assertFalse(done1.is_set())
    self.assertFalse(done2.is_set())

    injector.release_all_gates()

    self.assertTrue(done1.wait(timeout=5.0))
    self.assertTrue(done2.wait(timeout=5.0))
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

  def test_running_worker_server_and_fault_actions(self) -> None:
    injector = fault_injection.FaultInjector()
    worker = _DummyWorker()
    server = fault_injection.FaultInjectingServer(
        worker,
        worker_id="worker_test_0",
        injector=injector,
        enable_bulk_transport=False,
    )
    running = fault_injection.RunningWorkerServer(server)
    port = running.start()
    self.assertGreater(port, 0)

    try:
      handle = remote_execution.ActorHandle.from_address(
          f"grpc://127.0.0.1:{port}",
          enable_bulk_transport=False,
          rpc_timeout_s=5.0,
      )

      # 1. Normal call
      res = handle.submit("ping", "hello")
      self.assertEqual(res, "hello")
      self.assertEqual(worker.counter, 1)

      # 2. Return fault
      injector.on(
          worker_id="worker_test_0",
          method_name="ping",
          action=fault_injection.Return("intercepted"),
          times=1,
      )
      res = handle.submit("ping", "world")
      self.assertEqual(res, "intercepted")
      self.assertEqual(worker.counter, 1)  # worker method not called

      # 3. Raise fault
      injector.on(
          worker_id="worker_test_0",
          method_name="ping",
          action=fault_injection.Raise(ValueError("forced failure")),
          times=1,
      )
      with self.assertRaises(RuntimeError) as ctx:
        handle.submit("ping", "should_fail")
      self.assertIn("forced failure", str(ctx.exception))

      # 4. LoseReply fault (worker executes, then gRPC stream aborts with UNAVAILABLE)
      injector.on(
          worker_id="worker_test_0",
          method_name="ping",
          action=fault_injection.LoseReply(
              ConnectionResetError("lost connection")
          ),
          times=1,
      )
      with self.assertRaises(grpc.RpcError) as ctx:
        handle.submit("ping", "lose_reply")
      self.assertEqual(ctx.exception.code(), grpc.StatusCode.UNAVAILABLE)
      self.assertIn("lost connection", str(ctx.exception))
      self.assertEqual(worker.counter, 2)  # worker method DID execute!

      # 5. Block fault with FaultGate
      gate = fault_injection.FaultGate()
      injector.on(
          worker_id="worker_test_0",
          method_name="ping",
          action=fault_injection.Block(gate),
          times=1,
      )

      block_done = threading.Event()
      block_result = []

      def _blocked_caller():
        r = handle.submit("ping", "unblock_me")
        block_result.append(r)
        block_done.set()

      t = threading.Thread(target=_blocked_caller, daemon=True)
      t.start()
      self.assertTrue(gate.entered.wait(timeout=5.0))
      self.assertFalse(block_done.is_set())

      gate.release()
      self.assertTrue(block_done.wait(timeout=5.0))
      t.join(timeout=5.0)
      self.assertEqual(block_result, ["unblock_me"])
      self.assertEqual(worker.counter, 3)

    finally:
      running.stop()


if __name__ == "__main__":
  absltest.main()
