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

"""Health monitor with per-state deadlines for the control plane.

Polls each worker's `heartbeat()` and tracks how long it has been in its current
coarse state. Transient states (e.g. COMPILING, SYNCING, DRAINING) carry a
deadline; a worker that dwells past its state's deadline is reported as overdue
so the orchestrator can fence or fail it. Steady states (e.g. READY) have no
deadline. Time comes from an injectable monotonic clock so the policy is
testable without sleeping.
"""

from collections.abc import Callable
import concurrent.futures
import dataclasses
import logging
import time

from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import worker_registry

WorkerState = datatypes.WorkerState

# Default max seconds a worker may dwell in a transient state before it is
# considered overdue. States absent here (e.g. "READY", "STOPPED") are untimed.
DEFAULT_STATE_DEADLINES_S: dict[WorkerState, float] = {
    WorkerState.COMPILING: 30 * 60.0,
    WorkerState.SYNCING: 10 * 60.0,
    WorkerState.DRAINING: 5 * 60.0,
}


@dataclasses.dataclass(kw_only=True)
class OverdueWorker:
  """A worker that has exceeded the deadline for its current state.

  Attributes:
    worker_id: The worker's id.
    state: The state it is stuck in.
    elapsed_s: How long it has been in that state.
    deadline_s: The deadline for that state.
  """

  worker_id: str
  state: str
  elapsed_s: float
  deadline_s: float


class HealthMonitor:
  """Polls worker health and flags workers overdue in a transient state."""

  def __init__(
      self,
      registry: worker_registry.WorkerRegistry,
      *,
      state_deadlines_s: dict[WorkerState, float] | None = None,
      clock: Callable[[], float] = time.monotonic,
      max_workers: int = 32,
      executor: concurrent.futures.ThreadPoolExecutor | None = None,
  ):
    self._registry = registry
    self._max_workers = max_workers
    self._deadlines = (
        dict(DEFAULT_STATE_DEADLINES_S)
        if state_deadlines_s is None
        else dict(state_deadlines_s)
    )
    self._clock = clock
    # worker_id -> (state, timestamp it entered that state).
    self._state_since: dict[str, tuple[str, float]] = {}
    self._latest_reports: dict[str, datatypes.HealthReport] = {}
    if executor is not None:
      self._executor = executor
      self._owns_executor = False
    else:
      self._executor = concurrent.futures.ThreadPoolExecutor(
          max_workers=self._max_workers
      )
      self._owns_executor = True

  def close(self) -> None:
    """Shuts down the thread pool executor if owned by this monitor."""
    if self._owns_executor:
      self._executor.shutdown(wait=True)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
    return False

  @property
  def latest_reports(self) -> dict[str, datatypes.HealthReport]:
    """Returns a snapshot of the latest health reports from the last poll."""
    return dict(self._latest_reports)

  def get_report(self, worker_id: str) -> datatypes.HealthReport | None:
    """Returns the latest health report for a specific worker if available."""
    return self._latest_reports.get(worker_id)

  def poll(self) -> dict[str, datatypes.HealthReport]:
    """Polls every worker once, updating state-entry timestamps.

    Returns:
      A mapping of worker_id -> the HealthReport captured this poll.
    """
    reports: dict[str, datatypes.HealthReport] = {}
    worker_ids = self._registry.worker_ids()
    live_ids = set(worker_ids)

    def _poll_worker(wid: str) -> tuple[str, datatypes.HealthReport | None]:
      try:
        worker = self._registry.get(wid)
      except KeyError:
        logging.warning(
            "Worker %r unregistered concurrently, skipping poll.", wid
        )
        return wid, None
      return wid, worker.heartbeat()

    futures = [
        self._executor.submit(_poll_worker, wid) for wid in worker_ids
    ]
    try:
      for future in concurrent.futures.as_completed(futures):
        wid, report = future.result()
        if report is None:
          continue
        reports[wid] = report
        previous = self._state_since.get(wid)
        if previous is None or previous[0] != report.state:
          self._state_since[wid] = (report.state, self._clock())
    finally:
      for future in futures:
        future.cancel()

    # Update latest cached reports
    self._latest_reports.update(reports)

    # Forget workers that have left the registry.
    for worker_id in list(self._latest_reports.keys()):
      if worker_id not in live_ids:
        del self._latest_reports[worker_id]
    for worker_id in self._state_since.keys() - live_ids:
      del self._state_since[worker_id]
    return reports

  def overdue(self) -> list[OverdueWorker]:
    """Returns workers past the deadline for their current state.

    Based on state-entry times recorded by the most recent `poll()` calls.
    """
    now = self._clock()
    result: list[OverdueWorker] = []
    for worker_id, (state, since) in sorted(self._state_since.items()):
      deadline = self._deadlines.get(state)  # pyrefly: ignore[bad-argument-type]
      if deadline is None:
        continue
      elapsed = now - since
      if elapsed > deadline:
        result.append(
            OverdueWorker(
                worker_id=worker_id,
                state=state,
                elapsed_s=elapsed,
                deadline_s=deadline,
            )
        )
    return result
