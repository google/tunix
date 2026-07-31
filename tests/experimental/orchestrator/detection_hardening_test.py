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

"""Detection has to survive the failures it is supposed to detect.

Everything that reacts to worker health reads it from here, so a worker that
hangs, or one that never finishes starting, must be visible rather than able
to stop the question being asked.
"""

import threading
from typing import Any

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import health_monitor as health_lib
from tunix.experimental.orchestrator import lifecycle as lifecycle_lib
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.worker import abstract_worker

WorkerState = datatypes.WorkerState


class _Worker(abstract_worker.Worker):
  """A worker whose every phase can be made to hang or fail."""

  def __init__(
      self,
      worker_id: str,
      *,
      state: WorkerState = WorkerState.READY,
      hang_heartbeat: bool = False,
      fail_phase: str = "",
      fail_times: int = 10**6,
  ):
    self._worker_id = worker_id
    self._state = state
    self._hang = hang_heartbeat
    self._fail_phase = fail_phase
    self._fail_times = fail_times
    self.release = threading.Event()
    self.phases: list[str] = []

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id, roles=frozenset({"rollout"})
    )

  def heartbeat(self) -> datatypes.HealthReport:
    if self._hang:
      self.release.wait(timeout=30)
    return datatypes.HealthReport(state=self._state)

  def _phase(self, name: str) -> datatypes.Response:
    self.phases.append(name)
    if name == self._fail_phase and self._fail_times > 0:
      self._fail_times -= 1
      raise RuntimeError(f"{self._worker_id} cannot {name}")
    return datatypes.Response()

  def initialize(self) -> datatypes.Response:
    return self._phase("initialize")

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    del dummy_data
    return self._phase("compile")

  def start(self) -> datatypes.Response:
    return self._phase("start")

  def stop(self) -> datatypes.Response:
    return self._phase("stop")


def _registry(*workers) -> worker_registry.WorkerRegistry:
  registry = worker_registry.WorkerRegistry()
  for worker in workers:
    registry.register(worker)
  return registry


class HeartbeatTimeoutTest(absltest.TestCase):

  def test_a_hung_worker_cannot_wedge_the_poll(self):
    hung = _Worker("hung", hang_heartbeat=True)
    healthy = _Worker("healthy")
    monitor = health_lib.HealthMonitor(
        _registry(hung, healthy), heartbeat_timeout_s=0.2
    )

    try:
      reports = monitor.poll()

      # The healthy worker was still answered for, promptly.
      self.assertEqual(reports["healthy"].state, WorkerState.READY)
      # And the hung one is visible as a problem rather than a hang.
      self.assertEqual(reports["hung"].state, WorkerState.ERROR)
      self.assertIn("heartbeat", reports["hung"].last_error)
    finally:
      hung.release.set()
      monitor.close()

  def test_a_worker_that_answers_is_not_flagged(self):
    monitor = health_lib.HealthMonitor(
        _registry(_Worker("r0")), heartbeat_timeout_s=5.0
    )
    try:
      self.assertEqual(monitor.poll()["r0"].state, WorkerState.READY)
    finally:
      monitor.close()


class StartupDeadlineTest(absltest.TestCase):

  def _monitor(self, worker, now):
    return health_lib.HealthMonitor(_registry(worker), clock=lambda: now[0])

  def test_a_worker_stuck_initializing_is_flagged(self):
    """It answers heartbeats honestly and never becomes useful."""
    now = [0.0]
    monitor = self._monitor(
        _Worker("slow", state=WorkerState.INITIALIZING), now
    )
    try:
      monitor.poll()
      self.assertEmpty(monitor.overdue())

      now[0] += health_lib.DEFAULT_STATE_DEADLINES_S[
          WorkerState.INITIALIZING
      ] + 1
      overdue = monitor.overdue()

      self.assertLen(overdue, 1)
      self.assertEqual(overdue[0].worker_id, "slow")
      self.assertEqual(overdue[0].state, WorkerState.INITIALIZING)
    finally:
      monitor.close()

  def test_a_worker_stuck_pending_is_flagged(self):
    now = [0.0]
    monitor = self._monitor(_Worker("never", state=WorkerState.PENDING), now)
    try:
      monitor.poll()
      now[0] += health_lib.DEFAULT_STATE_DEADLINES_S[WorkerState.PENDING] + 1

      self.assertLen(monitor.overdue(), 1)
    finally:
      monitor.close()

  def test_a_ready_worker_is_never_overdue(self):
    now = [0.0]
    monitor = self._monitor(_Worker("r0"), now)
    try:
      monitor.poll()
      now[0] += 10**6

      self.assertEmpty(monitor.overdue())
    finally:
      monitor.close()


class BringUpTest(absltest.TestCase):

  def test_one_bad_worker_no_longer_stops_the_others(self):
    good, bad = _Worker("good"), _Worker("bad", fail_phase="initialize")
    driver = lifecycle_lib.LifecycleDriver(_registry(good, bad))

    with self.assertRaises(lifecycle_lib.LifecycleError) as caught:
      driver.bring_up(None)

    # The failure is attributed, not just reported as a phase failing.
    self.assertEqual([wid for wid, _ in caught.exception.failures], ["bad"])
    # And the healthy worker still went all the way up.
    self.assertEqual(good.phases, ["initialize", "compile", "start"])

  def test_a_failed_worker_is_dropped_from_later_phases(self):
    bad = _Worker("bad", fail_phase="initialize")
    driver = lifecycle_lib.LifecycleDriver(_registry(bad))

    with self.assertRaises(lifecycle_lib.LifecycleError):
      driver.bring_up(None)

    # Never asked to compile or start after failing to initialize.
    self.assertEqual(bad.phases, ["initialize"])

  def test_a_degraded_fleet_can_be_allowed(self):
    good, bad = _Worker("good"), _Worker("bad", fail_phase="compile")
    driver = lifecycle_lib.LifecycleDriver(_registry(good, bad))

    survivors = driver.bring_up(None, require_all=False)

    self.assertEqual(survivors, ["good"])

  def test_a_worker_that_fails_once_can_be_retried(self):
    flaky = _Worker("flaky", fail_phase="initialize", fail_times=1)
    driver = lifecycle_lib.LifecycleDriver(_registry(flaky))

    survivors = driver.bring_up(None, max_attempts=2)

    self.assertEqual(survivors, ["flaky"])
    self.assertEqual(
        flaky.phases, ["initialize", "initialize", "compile", "start"]
    )

  def test_rejects_a_nonsense_attempt_budget(self):
    driver = lifecycle_lib.LifecycleDriver(_registry(_Worker("r0")))
    with self.assertRaises(ValueError):
      driver.bring_up(None, max_attempts=0)


if __name__ == "__main__":
  absltest.main()
