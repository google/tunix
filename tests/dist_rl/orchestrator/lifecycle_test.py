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

import pickle
import threading
import time
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
    self.assertEqual(
        registry.status("r0"), worker_registry.AdmissionStatus.ACTIVE
    )
    self.assertEqual(
        registry.status("t0"), worker_registry.AdmissionStatus.ACTIVE
    )

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
    missing_target = (
        datatypes.WorkerInfo(
            worker_id="missing", roles=frozenset({datatypes.Role.ROLLOUT})
        ),
        mock.MagicMock(spec=remote_execution.ActorHandle),
    )
    with mock.patch.object(
        registry,
        "targets",
        side_effect=lambda *args, **kwargs: (
            [missing_target] if kwargs.get("statuses") is None else []
        ),
    ):
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
    driver.bring_up(_DUMMY_DATA)
    log.clear()

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
    self.assertEqual(
        registry.status("r0"), worker_registry.AdmissionStatus.REGISTERED
    )

    # Success path ordering: initialize -> verify -> compile -> start.
    log.clear()
    driver_ok = lifecycle.LifecycleDriver(
        registry, verify_fn=lambda infos: log.append("r0:verify")
    )
    driver_ok.bring_up(_DUMMY_DATA)
    self.assertEqual(
        log, ["r0:initialize", "r0:verify", "r0:compile", "r0:start"]
    )
    self.assertEqual(
        registry.status("r0"), worker_registry.AdmissionStatus.ACTIVE
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
    self.assertEqual(
        registry.status("r0"), worker_registry.AdmissionStatus.REGISTERED
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
      self.assertEqual(
          registry.status("r0"), worker_registry.AdmissionStatus.REGISTERED
      )

  def test_late_rollout_worker_bring_up_transitions_to_pending_weight_sync(
      self,
  ):
    registry = worker_registry.WorkerRegistry()
    driver = lifecycle.LifecycleDriver(registry)
    driver.enable_late_registration()

    h_late = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_late.submit.side_effect = [
        datatypes.Response(
            metadata={"drain_timeout_s": 120.0, "untracked_key": 1}
        ),
        datatypes.Response(),
        datatypes.Response(),
        datatypes.Response(),
    ]
    registry.register_handle("rollout-late", [datatypes.Role.ROLLOUT], h_late)
    registry.wait_for_workers(
        {datatypes.Role.ROLLOUT: 1},
        statuses=(worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,),
        timeout=5.0,
    )

    self.assertEqual(
        registry.status("rollout-late"),
        worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
    )
    self.assertEqual(
        registry.info("rollout-late").resources.get("drain_timeout_s"), 120.0
    )
    self.assertNotIn("untracked_key", registry.info("rollout-late").resources)
    self.assertEqual(
        [c.args for c in h_late.submit.call_args_list],
        [("initialize",), ("compile", None), ("start",)],
    )
    driver.shutdown()
    with self.assertRaisesRegex(RuntimeError, "registration is closed"):
      registry.register_handle(
          "rollout-after-shutdown", [datatypes.Role.ROLLOUT], h_late
      )

  @mock.patch.object(remote_execution.ActorHandle, "from_address")
  def test_enable_late_registration_unregisters_non_rollout_roles(
      self, mock_from_address
  ):
    remote_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_from_address.return_value = remote_handle
    registry = worker_registry.WorkerRegistry()
    driver = lifecycle.LifecycleDriver(registry)
    driver.enable_late_registration()

    h_actor = mock.MagicMock(spec=remote_execution.ActorHandle)
    registry.register_handle("actor-late", [datatypes.Role.ACTOR], h_actor)
    registry.register(
        mock_worker.MockWorker(worker_id="trainer-late", roles={"trainer"})
    )
    meta = pickle.dumps({
        "service_type": "trainer",
        "service_port": 5000,
        "worker_id": "trainer-remote-late",
    })
    registry.register_from_hostname("host", 0, meta)
    self.assertTrue(
        registry.wait_for_change(
            lambda: not registry.targets(
                statuses=(worker_registry.AdmissionStatus.REGISTERED,)
            ),
            timeout=5.0,
        )
    )

    self.assertNotIn("actor-late", registry)
    self.assertNotIn("trainer-late", registry)
    self.assertNotIn("trainer-remote-late", registry)
    h_actor.sync_close.assert_called_once()
    remote_handle.sync_close.assert_called_once()
    driver.shutdown()

  def test_late_rollout_worker_reregister_before_bring_up_brings_up_new_handle_once(
      self,
  ):
    registry = worker_registry.WorkerRegistry()
    driver = lifecycle.LifecycleDriver(registry)
    driver.enable_late_registration()

    old_init_started = threading.Event()
    old_init_release = threading.Event()
    old_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    new_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    new_handle.submit.side_effect = [
        datatypes.Response(metadata={"drain_timeout_s": 240.0}),
        datatypes.Response(),
        datatypes.Response(),
        datatypes.Response(),
    ]

    def _old_submit(method_name, *args, **kwargs):
      del args, kwargs
      if method_name == "initialize":
        old_init_started.set()
        old_init_release.wait(timeout=5.0)
        return datatypes.Response(metadata={"drain_timeout_s": 120.0})
      return datatypes.Response()

    old_handle.submit.side_effect = _old_submit

    registry.register_handle("r-rejoin", [datatypes.Role.ROLLOUT], old_handle)
    self.assertTrue(old_init_started.wait(timeout=5.0))

    # Same worker_id re-registers while old_handle is still in bring-up.
    registry.register_handle(
        "r-rejoin", [datatypes.Role.ROLLOUT], new_handle, override=True
    )
    old_init_release.set()
    registry.wait_for_workers(
        {datatypes.Role.ROLLOUT: 1},
        statuses=(worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,),
        timeout=5.0,
    )

    self.assertIs(registry.handle("r-rejoin"), new_handle)
    self.assertEqual(
        registry.status("r-rejoin"),
        worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
    )
    self.assertEqual(
        registry.info("r-rejoin").resources.get("drain_timeout_s"), 240.0
    )
    # new_handle was brought up exactly once (never touched by old_handle's
    # bring-up task).
    self.assertEqual(
        [c.args for c in new_handle.submit.call_args_list],
        [("initialize",), ("compile", None), ("start",)],
    )
    driver.shutdown()

  def test_late_rollout_worker_bring_up_failure_unregisters_worker(self):
    registry = worker_registry.WorkerRegistry()
    driver = lifecycle.LifecycleDriver(registry)
    driver.enable_late_registration()

    h_bad = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_bad.submit.side_effect = [
        datatypes.Response(),
        RuntimeError("compile exploded"),
    ]
    registry.register_handle("rollout-bad", [datatypes.Role.ROLLOUT], h_bad)
    self.assertTrue(
        registry.wait_for_change(
            lambda: "rollout-bad" not in registry,
            timeout=5.0,
        )
    )

    self.assertNotIn("rollout-bad", registry)
    h_bad.sync_close.assert_called_once()
    driver.shutdown()

  def test_shutdown_unregisters_registered_workers_and_is_idempotent(self):
    registry = worker_registry.WorkerRegistry()
    h_init = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_active = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_active.submit.return_value = datatypes.Response()
    registry.register_handle("r-active", [datatypes.Role.ROLLOUT], h_active)
    registry.set_status("r-active", worker_registry.AdmissionStatus.ACTIVE)

    driver = lifecycle.LifecycleDriver(registry)
    driver.enable_late_registration()

    init_started = threading.Event()
    init_closed = threading.Event()
    h_init.sync_close.side_effect = init_closed.set

    def _blocking_init(method_name, *args, **kwargs):
      del method_name, args, kwargs
      init_started.set()
      init_closed.wait(timeout=5.0)
      raise RuntimeError("channel closed during initialize")

    h_init.submit.side_effect = _blocking_init
    registry.register_handle("r-init", [datatypes.Role.ROLLOUT], h_init)
    self.assertTrue(init_started.wait(timeout=5.0))
    self.assertEqual(
        registry.status("r-init"), worker_registry.AdmissionStatus.REGISTERED
    )

    driver.shutdown()

    self.assertNotIn("r-init", registry)
    h_init.sync_close.assert_called_once()
    self.assertIn("r-active", registry)
    h_active.submit.assert_called_once_with("stop")

    # Repeated shutdown is safe.
    driver.shutdown()

  def test_late_rollout_worker_direct_active_promotion_when_weight_sync_not_required(
      self,
  ):
    registry = worker_registry.WorkerRegistry()
    driver = lifecycle.LifecycleDriver(registry)
    driver.enable_late_registration(require_weight_sync=False)

    h_late = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_late.submit.return_value = datatypes.Response()
    registry.register_handle("rollout-late", [datatypes.Role.ROLLOUT], h_late)
    registry.wait_for_workers(
        {datatypes.Role.ROLLOUT: 1},
        statuses=(worker_registry.AdmissionStatus.ACTIVE,),
        timeout=5.0,
    )

    self.assertEqual(
        registry.status("rollout-late"),
        worker_registry.AdmissionStatus.ACTIVE,
    )
    self.assertEqual(registry.handles(datatypes.Role.ROLLOUT), [h_late])
    driver.shutdown()

  def test_late_rollout_worker_bring_up_does_not_head_of_line_block_on_slow_worker(
      self,
  ):
    registry = worker_registry.WorkerRegistry()
    driver = lifecycle.LifecycleDriver(registry, max_workers=4)
    driver.enable_late_registration()

    slow_init_started = threading.Event()
    slow_init_release = threading.Event()
    h_slow = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_fast = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_fast.submit.return_value = datatypes.Response()

    def _slow_submit(method_name, *args, **kwargs):
      del args, kwargs
      if method_name == "initialize":
        slow_init_started.set()
        slow_init_release.wait(timeout=5.0)
      return datatypes.Response()

    h_slow.submit.side_effect = _slow_submit
    registry.register_handle("w-slow", [datatypes.Role.ROLLOUT], h_slow)
    self.assertTrue(slow_init_started.wait(timeout=5.0))

    # Register w-fast while w-slow is blocked in initialize.
    registry.register_handle("w-fast", [datatypes.Role.ROLLOUT], h_fast)
    registry.wait_for_workers(
        {datatypes.Role.ROLLOUT: 1},
        statuses=(worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,),
        timeout=5.0,
    )
    self.assertEqual(
        registry.status("w-fast"),
        worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
    )
    self.assertEqual(
        registry.status("w-slow"),
        worker_registry.AdmissionStatus.REGISTERED,
    )

    slow_init_release.set()
    registry.wait_for_workers(
        {datatypes.Role.ROLLOUT: 2},
        statuses=(worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,),
        timeout=5.0,
    )
    self.assertEqual(
        registry.status("w-slow"),
        worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
    )
    driver.shutdown()

  def test_shutdown_closes_blocked_and_queued_late_handles_with_max_workers_one(
      self,
  ):
    registry = worker_registry.WorkerRegistry()
    h_active = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_active.submit.return_value = datatypes.Response()
    registry.register_handle("r-active", [datatypes.Role.ROLLOUT], h_active)
    registry.set_status("r-active", worker_registry.AdmissionStatus.ACTIVE)

    driver = lifecycle.LifecycleDriver(registry, max_workers=1)
    driver.enable_late_registration()
    submit_spy = self.enter_context(
        mock.patch.object(
            driver._late_executor, "submit", wraps=driver._late_executor.submit
        )
    )

    init_started = threading.Event()
    init_closed = threading.Event()
    h_blocked = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_blocked.sync_close.side_effect = init_closed.set

    def _blocking_init(method_name, *args, **kwargs):
      del method_name, args, kwargs
      init_started.set()
      init_closed.wait(timeout=5.0)
      raise RuntimeError("channel closed during initialize")

    h_blocked.submit.side_effect = _blocking_init
    h_queued_1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_queued_2 = mock.MagicMock(spec=remote_execution.ActorHandle)

    registry.register_handle("r-init-0", [datatypes.Role.ROLLOUT], h_blocked)
    self.assertTrue(init_started.wait(timeout=5.0))

    # Queue 2 additional late rollout workers behind the blocked worker.
    registry.register_handle("r-queued-1", [datatypes.Role.ROLLOUT], h_queued_1)
    registry.register_handle("r-queued-2", [datatypes.Role.ROLLOUT], h_queued_2)
    deadline = time.monotonic() + 5.0
    while submit_spy.call_count < 3 and time.monotonic() < deadline:
      time.sleep(0.01)
    self.assertEqual(submit_spy.call_count, 3)

    driver.shutdown()

    self.assertNotIn("r-init-0", registry)
    self.assertNotIn("r-queued-1", registry)
    self.assertNotIn("r-queued-2", registry)
    h_blocked.sync_close.assert_called_once()
    h_queued_1.sync_close.assert_called_once()
    h_queued_2.sync_close.assert_called_once()
    h_queued_1.submit.assert_not_called()
    h_queued_2.submit.assert_not_called()
    self.assertIn("r-active", registry)
    h_active.submit.assert_called_once_with("stop")


if __name__ == "__main__":
  absltest.main()
