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

"""Tests for the HealthMonitor per-state deadline policy."""

import concurrent.futures
from unittest import mock
from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import health_monitor
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.worker import mock_worker

WorkerState = datatypes.WorkerState


class _FakeClock:

  def __init__(self):
    self.t = 0.0

  def __call__(self) -> float:
    return self.t


class HealthMonitorTest(absltest.TestCase):

  def _registry(self, worker) -> worker_registry.WorkerRegistry:
    registry = worker_registry.WorkerRegistry()
    registry.register(worker)
    return registry

  def test_poll_returns_current_reports(self):
    worker = mock_worker.MockWorker("w0", roles={"trainer"})
    worker._state = WorkerState.READY
    monitor = health_monitor.HealthMonitor(self._registry(worker))
    reports = monitor.poll()
    self.assertEqual(reports["w0"].state, WorkerState.READY)
    self.assertEmpty(monitor.overdue())

  def test_worker_overdue_past_state_deadline(self):
    worker = mock_worker.MockWorker("w0", roles={"trainer"})
    worker._state = WorkerState.COMPILING
    clock = _FakeClock()
    monitor = health_monitor.HealthMonitor(
        self._registry(worker),
        state_deadlines_s={WorkerState.COMPILING: 100.0},
        clock=clock,
    )
    monitor.poll()  # Enters COMPILING at t=0.
    self.assertEmpty(monitor.overdue())

    clock.t = 100.5  # Past the 100s deadline.
    overdue = monitor.overdue()
    self.assertLen(overdue, 1)
    self.assertEqual(overdue[0].worker_id, "w0")
    self.assertEqual(overdue[0].state, WorkerState.COMPILING)

  def test_steady_state_is_never_overdue(self):
    worker = mock_worker.MockWorker("w0", roles={"trainer"})
    worker._state = WorkerState.READY
    clock = _FakeClock()
    monitor = health_monitor.HealthMonitor(self._registry(worker), clock=clock)
    monitor.poll()
    clock.t = 10_000_000.0
    self.assertEmpty(monitor.overdue())

  def test_poll_skips_unregistered_worker(self):
    registry = worker_registry.WorkerRegistry()
    monitor = health_monitor.HealthMonitor(registry)
    with mock.patch.object(registry, "worker_ids", return_value=["missing"]):
      with self.assertLogs(level="WARNING") as logs:
        reports = monitor.poll()
      self.assertEmpty(reports)
      self.assertIn("unregistered concurrently", logs.output[0])

  def test_state_change_resets_the_deadline_timer(self):
    worker = mock_worker.MockWorker("w0", roles={"trainer"})
    worker._state = WorkerState.COMPILING
    clock = _FakeClock()
    monitor = health_monitor.HealthMonitor(
        self._registry(worker),
        state_deadlines_s={WorkerState.COMPILING: 100.0},
        clock=clock,
    )
    monitor.poll()  # COMPILING since t=0.
    clock.t = 90.0
    worker._state = WorkerState.READY
    monitor.poll()  # Transitions to READY at t=90.
    clock.t = (
        200.0  # Well past the COMPILING deadline, but no longer COMPILING.
    )
    self.assertEmpty(monitor.overdue())

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


  def test_latest_reports_and_get_report(self):
    worker0 = mock_worker.MockWorker("w0", roles={"rollout"})
    worker1 = mock_worker.MockWorker("w1", roles={"rollout"})
    registry = worker_registry.WorkerRegistry()
    registry.register(worker0)
    registry.register(worker1)
    monitor = health_monitor.HealthMonitor(registry)

    # Before poll, latest_reports is empty
    self.assertEmpty(monitor.latest_reports)
    self.assertIsNone(monitor.get_report("w0"))

    report_w0 = datatypes.HealthReport(
        state=WorkerState.READY,
        load_info=datatypes.LoadInfo(
            num_requests_waiting=2,
            num_requests_running=1,
            kv_cache_usage_perc=0.5,
        ),
    )
    report_w1 = datatypes.HealthReport(
        state=WorkerState.READY,
        load_info=datatypes.LoadInfo(
            num_requests_waiting=0,
            num_requests_running=3,
            kv_cache_usage_perc=0.8,
        ),
    )
    with mock.patch.object(worker0, "heartbeat", return_value=report_w0):
      with mock.patch.object(worker1, "heartbeat", return_value=report_w1):
        reports = monitor.poll()
        self.assertEqual(reports["w0"], report_w0)
        self.assertEqual(reports["w1"], report_w1)

    self.assertEqual(monitor.latest_reports, {"w0": report_w0, "w1": report_w1})
    self.assertEqual(monitor.get_report("w0"), report_w0)
    self.assertEqual(monitor.get_report("w1"), report_w1)

    # When a worker is unregistered, next poll removes it from latest_reports
    registry.unregister("w1")
    with mock.patch.object(worker0, "heartbeat", return_value=report_w0):
      monitor.poll()
    self.assertEqual(monitor.latest_reports, {"w0": report_w0})
    self.assertIsNone(monitor.get_report("w1"))
    monitor.close()


if __name__ == "__main__":
  absltest.main()
