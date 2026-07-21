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

Polls each worker's `health()` and tracks how long it has been in its current
coarse state. Transient states (e.g. COMPILING, SYNCING, DRAINING) carry a
deadline; a worker that dwells past its state's deadline is reported as overdue
so the orchestrator can fence or fail it. Steady states (e.g. READY) have no
deadline. Time comes from an injectable monotonic clock so the policy is
testable without sleeping.
"""

import dataclasses
import time
from collections.abc import Callable

from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import worker_registry

# Default max seconds a worker may dwell in a transient state before it is
# considered overdue. States absent here (e.g. "READY", "STOPPED") are untimed.
DEFAULT_STATE_DEADLINES_S: dict[str, float] = {
    "COMPILING": 30 * 60.0,
    "SYNCING": 10 * 60.0,
    "DRAINING": 5 * 60.0,
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
      state_deadlines_s: dict[str, float] | None = None,
      clock: Callable[[], float] = time.monotonic,
  ):
    self._registry = registry
    self._deadlines = (
        dict(DEFAULT_STATE_DEADLINES_S)
        if state_deadlines_s is None
        else dict(state_deadlines_s)
    )
    self._clock = clock
    # worker_id -> (state, timestamp it entered that state).
    self._state_since: dict[str, tuple[str, float]] = {}

  def poll(self) -> dict[str, datatypes.HealthReport]:
    """Polls every worker once, updating state-entry timestamps.

    Returns:
      A mapping of worker_id -> the HealthReport captured this poll.
    """
    now = self._clock()
    reports: dict[str, datatypes.HealthReport] = {}
    live_ids = set(self._registry.worker_ids())
    for worker_id in self._registry.worker_ids():
      report = self._registry.get(worker_id).health()
      reports[worker_id] = report
      previous = self._state_since.get(worker_id)
      if previous is None or previous[0] != report.state:
        self._state_since[worker_id] = (report.state, now)
    # Forget workers that have left the registry.
    for worker_id in list(self._state_since):
      if worker_id not in live_ids:
        del self._state_since[worker_id]
    return reports

  def overdue(self) -> list[OverdueWorker]:
    """Returns workers past the deadline for their current state.

    Based on state-entry times recorded by the most recent `poll()` calls.
    """
    now = self._clock()
    result: list[OverdueWorker] = []
    for worker_id, (state, since) in sorted(self._state_since.items()):
      deadline = self._deadlines.get(state)
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
