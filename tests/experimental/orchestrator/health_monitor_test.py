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

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import health_monitor
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.worker import abstract_worker


class _StatefulWorker(abstract_worker.Worker):
  """A worker whose reported coarse state can be driven by the test."""

  def __init__(self, worker_id: str, state: str = "READY"):
    self._worker_id = worker_id
    self.state = state

  def initialize(self) -> None:
    pass

  def compile(self, shape_config) -> None:
    del shape_config

  def start(self) -> None:
    pass

  def stop(self) -> None:
    pass

  def health(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(state=self.state)

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id, roles=frozenset({"trainer"})
    )


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
    worker = _StatefulWorker("w0", state="READY")
    monitor = health_monitor.HealthMonitor(self._registry(worker))
    reports = monitor.poll()
    self.assertEqual(reports["w0"].state, "READY")
    self.assertEmpty(monitor.overdue())

  def test_worker_overdue_past_state_deadline(self):
    worker = _StatefulWorker("w0", state="COMPILING")
    clock = _FakeClock()
    monitor = health_monitor.HealthMonitor(
        self._registry(worker),
        state_deadlines_s={"COMPILING": 100.0},
        clock=clock,
    )
    monitor.poll()  # Enters COMPILING at t=0.
    self.assertEmpty(monitor.overdue())

    clock.t = 100.5  # Past the 100s deadline.
    overdue = monitor.overdue()
    self.assertLen(overdue, 1)
    self.assertEqual(overdue[0].worker_id, "w0")
    self.assertEqual(overdue[0].state, "COMPILING")

  def test_steady_state_is_never_overdue(self):
    worker = _StatefulWorker("w0", state="READY")
    clock = _FakeClock()
    monitor = health_monitor.HealthMonitor(
        self._registry(worker), clock=clock
    )
    monitor.poll()
    clock.t = 10_000_000.0
    self.assertEmpty(monitor.overdue())

  def test_state_change_resets_the_deadline_timer(self):
    worker = _StatefulWorker("w0", state="COMPILING")
    clock = _FakeClock()
    monitor = health_monitor.HealthMonitor(
        self._registry(worker),
        state_deadlines_s={"COMPILING": 100.0},
        clock=clock,
    )
    monitor.poll()  # COMPILING since t=0.
    clock.t = 90.0
    worker.state = "READY"
    monitor.poll()  # Transitions to READY at t=90.
    clock.t = 200.0  # Well past the COMPILING deadline, but no longer COMPILING.
    self.assertEmpty(monitor.overdue())


if __name__ == "__main__":
  absltest.main()
