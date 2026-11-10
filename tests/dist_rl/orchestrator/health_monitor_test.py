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

"""Unit tests for concurrent heartbeat polling and executor lifecycle."""

import concurrent.futures
from unittest import mock
from absl.testing import absltest
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import health_monitor
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.testing import worker_registry_testing
from tunix.dist_rl.worker import mock_worker
from tunix.dist_rl.worker import remote_execution

WorkerState = datatypes.WorkerState


class HealthMonitorTest(absltest.TestCase):

  def _registry(self, worker) -> worker_registry.WorkerRegistry:
    registry = worker_registry.WorkerRegistry()
    registry.register(worker)
    return registry

  def test_poll_returns_current_reports(self):
    worker = mock_worker.MockWorker("w0", roles={"trainer"})
    worker._state = WorkerState.READY
    monitor = health_monitor.HealthMonitor(self._registry(worker))
    self.addCleanup(monitor.close)
    reports = monitor.poll()
    self.assertEqual(reports["w0"].state, WorkerState.READY)

  def test_poll_skips_unregistered_worker(self):
    registry = worker_registry.WorkerRegistry()
    monitor = health_monitor.HealthMonitor(registry)
    self.addCleanup(monitor.close)
    missing_target = (
        datatypes.WorkerInfo(
            worker_id="missing", roles=frozenset({datatypes.Role.ROLLOUT})
        ),
        mock.MagicMock(spec=remote_execution.ActorHandle),
    )
    with mock.patch.object(
        registry, "targets", autospec=True, return_value=[missing_target]
    ):
      with self.assertLogs(level="WARNING") as logs:
        reports = monitor.poll()
      self.assertEmpty(reports)
      self.assertIn(
          "unregistered or re-registered concurrently", logs.output[0]
      )
    missing_target[1].submit.assert_not_called()

  def test_poll_skips_displaced_handle_on_concurrent_re_registration(self):
    registry = worker_registry.WorkerRegistry()
    monitor = health_monitor.HealthMonitor(registry)
    self.addCleanup(monitor.close)
    old_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    old_handle.submit.side_effect = RuntimeError("closed displaced handle")
    new_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    info = registry.register_handle(
        "w0", [datatypes.Role.ROLLOUT], new_handle, override=True
    )
    with mock.patch.object(
        registry, "targets", autospec=True, return_value=[(info, old_handle)]
    ):
      with self.assertLogs(level="WARNING") as logs:
        reports = monitor.poll()
    self.assertEmpty(reports)
    self.assertIn("unregistered or re-registered concurrently", logs.output[0])
    old_handle.submit.assert_not_called()

  def test_poll_skips_handle_displaced_or_unregistered_during_heartbeat(self):
    registry = worker_registry.WorkerRegistry()
    monitor = health_monitor.HealthMonitor(registry)
    self.addCleanup(monitor.close)

    old_displaced = mock.MagicMock(spec=remote_execution.ActorHandle)
    replacement = mock.MagicMock(spec=remote_execution.ActorHandle)

    def _displace_during_submit(_):
      registry.register_handle(
          "w-rejoined", [datatypes.Role.ROLLOUT], replacement, override=True
      )
      raise RuntimeError("displaced handle closed mid-RPC")

    old_displaced.submit.side_effect = _displace_during_submit
    registry.register_handle(
        "w-rejoined", [datatypes.Role.ROLLOUT], old_displaced
    )

    unregistered = mock.MagicMock(spec=remote_execution.ActorHandle)

    def _unregister_during_submit(_):
      registry.unregister_handle(unregistered)
      raise RuntimeError("unregistered handle closed mid-RPC")

    unregistered.submit.side_effect = _unregister_during_submit
    registry.register_handle(
        "w-evicted", [datatypes.Role.ROLLOUT], unregistered
    )

    healthy = mock.MagicMock(spec=remote_execution.ActorHandle)
    healthy.submit.return_value = datatypes.HealthReport(
        state=WorkerState.READY
    )
    registry.register_handle("w-healthy", [datatypes.Role.ACTOR], healthy)

    with self.assertLogs(level="WARNING") as logs:
      reports = monitor.poll()

    self.assertEqual(
        {wid: r.state for wid, r in reports.items()},
        {"w-healthy": WorkerState.READY},
    )
    self.assertLen(logs.output, 2)
    old_displaced.submit.assert_called_once_with("heartbeat")
    unregistered.submit.assert_called_once_with("heartbeat")
    replacement.submit.assert_not_called()

  def test_poll_filters_by_admission_status(self):
    registry = worker_registry.WorkerRegistry()
    h_active = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_active.submit.return_value = datatypes.HealthReport(
        state=WorkerState.READY
    )
    h_pending = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_pending.submit.return_value = datatypes.HealthReport(
        state=WorkerState.READY
    )
    h_registered = mock.MagicMock(spec=remote_execution.ActorHandle)
    h_registered.submit.side_effect = RuntimeError("still initializing")
    registry.register_handle("w-active", [datatypes.Role.ACTOR], h_active)
    registry.register_handle("w-pending", [datatypes.Role.ROLLOUT], h_pending)
    registry.register_handle(
        "w-registered", [datatypes.Role.ROLLOUT], h_registered
    )
    worker_registry_testing.set_status(
        registry, "w-active", worker_registry.AdmissionStatus.ACTIVE
    )
    worker_registry_testing.set_status(
        registry,
        "w-pending",
        worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
    )

    with health_monitor.HealthMonitor(registry) as monitor:
      reports = monitor.poll(
          statuses=(
              worker_registry.AdmissionStatus.ACTIVE,
              worker_registry.AdmissionStatus.PENDING_WEIGHT_SYNC,
          )
      )

    self.assertEqual(
        {wid: r.state for wid, r in reports.items()},
        {"w-active": WorkerState.READY, "w-pending": WorkerState.READY},
    )
    h_active.submit.assert_called_once_with("heartbeat")
    h_pending.submit.assert_called_once_with("heartbeat")
    h_registered.submit.assert_not_called()

  def test_close_shuts_down_owned_executor(self):
    registry = worker_registry.WorkerRegistry()
    monitor = health_monitor.HealthMonitor(registry)
    monitor.close()
    with self.assertRaisesRegex(
        RuntimeError, "cannot schedule new futures after shutdown"
    ):
      monitor._executor.submit(lambda: None)

  def test_injected_executor_not_shut_down_by_close(self):
    registry = worker_registry.WorkerRegistry()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      monitor = health_monitor.HealthMonitor(registry, executor=executor)
      monitor.close()
      future = executor.submit(lambda: 42)
      self.assertEqual(future.result(), 42)

  def test_context_manager_closes_monitor(self):
    registry = worker_registry.WorkerRegistry()
    with health_monitor.HealthMonitor(registry) as monitor:
      pass
    with self.assertRaisesRegex(
        RuntimeError, "cannot schedule new futures after shutdown"
    ):
      monitor._executor.submit(lambda: None)

  def test_poll_cancels_remaining_futures_on_exception(self):
    worker0 = mock_worker.MockWorker("w0", roles={"trainer"})
    worker1 = mock_worker.MockWorker("w1", roles={"trainer"})
    registry = worker_registry.WorkerRegistry()
    registry.register(worker0)
    registry.register(worker1)
    monitor = health_monitor.HealthMonitor(registry, max_workers=1)
    self.addCleanup(monitor.close)
    with mock.patch.object(
        worker0, "heartbeat", side_effect=RuntimeError("heartbeat failed")
    ):
      with mock.patch.object(
          worker1,
          "heartbeat",
          return_value=datatypes.HealthReport(state=WorkerState.READY),
      ) as mock_hb1:
        with self.assertRaisesRegex(RuntimeError, "heartbeat failed"):
          monitor.poll()
        self.assertEqual(mock_hb1.call_count, 0)
    self.assertEqual(
        monitor._executor.submit(lambda: "clean").result(), "clean"
    )


if __name__ == "__main__":
  absltest.main()
