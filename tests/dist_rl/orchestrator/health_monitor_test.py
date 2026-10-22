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
from tunix.dist_rl.worker import mock_worker

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
    reports = monitor.poll()
    self.assertEqual(reports["w0"].state, WorkerState.READY)

  def test_poll_skips_unregistered_worker(self):
    registry = worker_registry.WorkerRegistry()
    monitor = health_monitor.HealthMonitor(registry)
    with mock.patch.object(registry, "worker_ids", return_value=["missing"]):
      with self.assertLogs(level="WARNING") as logs:
        reports = monitor.poll()
      self.assertEmpty(reports)
      self.assertIn("unregistered concurrently", logs.output[0])

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


if __name__ == "__main__":
  absltest.main()
