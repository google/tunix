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

"""Tests for the LifecycleDriver."""

from unittest import mock
from absl.testing import absltest
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import lifecycle
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.worker import mock_worker
from tunix.dist_rl.worker import remote_execution

WorkerState = datatypes.WorkerState


_DUMMY_DATA = {"max_prompt_length": 4, "max_response_tokens": 8}


class LifecycleDriverTest(absltest.TestCase):

  def test_bring_up_transitions_mocks_to_ready(self):
    registry = worker_registry.WorkerRegistry()
    rollout = mock_worker.MockWorker(worker_id="r0", roles={"rollout"})
    trainer = mock_worker.MockWorker(worker_id="t0", roles={"trainer"})
    registry.register(rollout)
    registry.register(trainer)

    lifecycle.LifecycleDriver(registry).bring_up(_DUMMY_DATA)

    self.assertEqual(rollout.state, WorkerState.READY)
    self.assertEqual(trainer.state, WorkerState.READY)

  def test_shutdown_transitions_mocks_to_stopped(self):
    registry = worker_registry.WorkerRegistry()
    trainer = mock_worker.MockWorker(worker_id="t0", roles={"trainer"})
    registry.register(trainer)
    driver = lifecycle.LifecycleDriver(registry)
    driver.bring_up(_DUMMY_DATA)

    driver.shutdown()

    self.assertEqual(trainer.state, WorkerState.STOPPED)

  def test_bring_up_runs_phase_by_phase(self):
    registry = worker_registry.WorkerRegistry()
    log: list[str] = []
    registry.register(mock_worker.MockWorker("a", roles={"trainer"}, log=log))
    registry.register(mock_worker.MockWorker("b", roles={"trainer"}, log=log))

    lifecycle.LifecycleDriver(registry).bring_up(_DUMMY_DATA)

    # All initializes precede all compiles, which precede all starts.
    phases = [entry.split(":")[1] for entry in log]
    self.assertEqual(
        phases,
        ["initialize", "initialize", "compile", "compile", "start", "start"],
    )

  def test_shutdown_skips_unregistered_worker(self):
    registry = worker_registry.WorkerRegistry()
    driver = lifecycle.LifecycleDriver(registry)
    with mock.patch.object(registry, "worker_ids", return_value=["missing"]):
      with self.assertLogs(level="WARNING") as logs:
        driver.shutdown()  # Should not raise LifecycleError
      self.assertIn("unregistered concurrently", logs.output[0])

  def test_shutdown_is_best_effort_and_aggregates_failures(self):
    registry = worker_registry.WorkerRegistry()
    log: list[str] = []
    registry.register(mock_worker.MockWorker("a", roles={"trainer"}, log=log))
    registry.register(
        mock_worker.MockWorker("b", roles={"trainer"}, log=log, fail_stop=True)
    )
    registry.register(mock_worker.MockWorker("c", roles={"trainer"}, log=log))
    driver = lifecycle.LifecycleDriver(registry)

    with self.assertRaises(lifecycle.LifecycleError) as ctx:
      driver.shutdown()

    # Every worker was asked to stop even though "b" raised.
    self.assertEqual(sorted(log), ["a:stop", "b:stop", "c:stop"])
    self.assertEqual([wid for wid, _ in ctx.exception.failures], ["b"])

  def test_bring_up_awaits_async_start_coroutine(self):
    class _AsyncStartWorker(mock_worker.MockWorker):

      def __init__(self, worker_id: str):
        super().__init__(worker_id=worker_id, roles={"rollout"})
        self.async_started = False

      async def start(self) -> datatypes.Response:
        self.async_started = True
        return super().start()

    registry = worker_registry.WorkerRegistry()
    worker = _AsyncStartWorker("r_async")
    registry.register(worker)

    lifecycle.LifecycleDriver(registry).bring_up(_DUMMY_DATA)

    self.assertTrue(worker.async_started)
    self.assertEqual(worker.state, WorkerState.READY)

  def test_bring_up_runs_verify_fn_after_init_and_before_compile(self):
    registry = worker_registry.WorkerRegistry()
    log: list[str] = []

    class _MetadataWorker(mock_worker.MockWorker):

      def initialize(self) -> datatypes.Response:
        resp = super().initialize()
        return datatypes.Response(
            metadata={
                **resp.metadata,
                "drain_timeout_s": 180.0,
                "custom_key": "custom_val",
                "state": "INITIALIZING",
            }
        )

    worker = _MetadataWorker("r0", roles={"rollout"}, log=log)
    registry.register(worker)

    def _verify_fail(infos):
      self.assertLen(infos, 1)
      self.assertEqual(infos[0].resources.get("drain_timeout_s"), 180.0)
      self.assertNotIn("custom_key", infos[0].resources)
      self.assertNotIn("state", infos[0].resources)
      log.append("r0:verify")
      raise ValueError("verification failed before compile")

    driver = lifecycle.LifecycleDriver(registry, verify_fn=_verify_fail)
    with self.assertRaisesRegex(ValueError, "verification failed"):
      driver.bring_up(_DUMMY_DATA)

    self.assertEqual(log, ["r0:initialize", "r0:verify"])

    # Success path ordering: initialize -> verify -> compile -> start.
    log.clear()
    driver_ok = lifecycle.LifecycleDriver(
        registry, verify_fn=lambda infos: log.append("r0:verify")
    )
    driver_ok.bring_up(_DUMMY_DATA)
    self.assertEqual(
        log, ["r0:initialize", "r0:verify", "r0:compile", "r0:start"]
    )

  def test_bring_up_skips_verify_compile_start_when_initialize_fails(self):
    registry = worker_registry.WorkerRegistry()
    bad_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    bad_handle.submit.side_effect = RuntimeError("init failed")
    registry.register_handle("r0", [datatypes.Role.ROLLOUT], bad_handle)
    verify_mock = mock.MagicMock()

    driver = lifecycle.LifecycleDriver(registry, verify_fn=verify_mock)
    with self.assertRaisesRegex(RuntimeError, "init failed"):
      driver.bring_up(_DUMMY_DATA)

    verify_mock.assert_not_called()
    self.assertEqual(
        bad_handle.submit.call_args_list, [mock.call("initialize")]
    )

  def test_bring_up_does_not_touch_reregistered_new_handle(self):
    for old_metadata in ({"drain_timeout_s": 999.0}, {}):
      registry = worker_registry.WorkerRegistry()
      old_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
      new_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
      verify_mock = mock.MagicMock()

      registry.register_handle("r0", [datatypes.Role.ROLLOUT], old_handle)

      def _init_then_reregister(
          method,
          *unused_args,
          reg=registry,
          nh=new_handle,
          meta=old_metadata,
      ):
        if method == "initialize":
          reg.register_handle(
              "r0", [datatypes.Role.ROLLOUT], nh, override=True
          )
        return datatypes.Response(metadata=meta)

      old_handle.submit.side_effect = _init_then_reregister

      driver = lifecycle.LifecycleDriver(registry, verify_fn=verify_mock)
      with self.assertRaisesRegex(RuntimeError, "re-registered"):
        driver.bring_up(_DUMMY_DATA)

      self.assertEqual(
          old_handle.submit.call_args_list, [mock.call("initialize")]
      )
      verify_mock.assert_not_called()
      new_handle.submit.assert_not_called()
      self.assertNotIn("drain_timeout_s", registry.info("r0").resources)


if __name__ == "__main__":
  absltest.main()
