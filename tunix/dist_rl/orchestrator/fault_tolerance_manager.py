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

"""Control-plane fault-tolerance policy for distributed RL workers."""

import collections
from collections.abc import Callable
import threading
import time

from absl import logging
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import lifecycle
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.worker import remote_execution
from tunix.dist_rl.worker import task_session


_monotonic: Callable[[], float] = time.monotonic


class RolloutCapacityError(RuntimeError):
  """Raised when active rollout capacity stays below the configured floor."""


class FaultToleranceManager:
  """Applies `FaultToleranceConfig` to worker eviction and task retries.

  Derives the `TaskSession` fault-tolerance config, evicts unreachable worker
  handles from `WorkerRegistry`, enforces the rollout capacity floor, and
  aggregates `ft/*` fleet gauges and cumulative fault-tolerance counters. Holds
  no task ledger; worker bring-up, admission, and shutdown belong to
  `LifecycleDriver`.
  """

  def __init__(
      self,
      registry: worker_registry.WorkerRegistry,
      config: datatypes.FaultToleranceConfig | None = None,
      *,
      lifecycle_driver: lifecycle.LifecycleDriver | None = None,
  ):
    """Initializes the manager.

    Args:
      registry: Registry that unreachable worker handles are evicted from.
      config: Fault-tolerance config; `None` uses the defaults (disabled).
      lifecycle_driver: Optional lifecycle driver exposing late rollout worker
        bring-up failure counters.
    """
    self._registry = registry
    self._config = (
        config if config is not None else datatypes.FaultToleranceConfig()
    )
    self._lifecycle_driver = lifecycle_driver
    self._lock = threading.Lock()
    self._task_session_evictions_total = 0
    self._promoted_total = 0
    self._peak_active_rollout_workers = 0
    self._below_min_active_since: float | None = None

  @property
  def config(self) -> datatypes.FaultToleranceConfig:
    """Returns the active `FaultToleranceConfig`."""
    return self._config

  def record_rejoin_promoted(self) -> None:
    """Counts a worker promoted from PENDING_WEIGHT_SYNC to ACTIVE."""
    with self._lock:
      self._promoted_total += 1

  def metrics(
      self,
      *,
      rollout_session: task_session.TaskSession | None = None,
  ) -> dict[str, float]:
    """Returns rollout fleet gauges and cumulative fault-tolerance counters.

    `ft/fleet/*` counts the rollout workers in each `AdmissionStatus` now, and
    `ft/fleet/rollout_active_ratio` reports `active / peak_active` (or `1.0`
    before any rollout worker has reached `ACTIVE`, indicating no capacity lost
    yet). Every other metric is a running total since the manager was created;
    take the per-step difference to see what happened during one step.

    `ft/rollout/task_unrecovered_total` counts tasks lost to worker/transport
    faults (retries exhausted, or no surviving worker), not application errors
    returned by a worker (covered by `rollout/status/*_frac` or run abort). A
    non-zero value means the run is aborting because `poll_rollouts()` re-raises
    the terminal error, so in live per-step W&B logs it normally reads 0 and
    serves mainly as a post-mortem / log signal.

    Args:
      rollout_session: Optional rollout `TaskSession` supplying cumulative task
        retry and unrecovered task counters.
    """
    statuses = collections.Counter(
        status
        for _, _, status in self._registry.targets_with_status(
            datatypes.Role.ROLLOUT, statuses=None
        )
    )
    active = statuses[worker_registry.AdmissionStatus.ACTIVE]
    with self._lock:
      task_session_evictions = self._task_session_evictions_total
      promoted = self._promoted_total
      peak_active = max(self._peak_active_rollout_workers, active)
    active_ratio = (
        float(active) / float(peak_active) if peak_active > 0 else 1.0
    )
    return {
        **{
            f"ft/fleet/rollout_{status.value}": float(statuses[status])
            for status in worker_registry.AdmissionStatus
        },
        "ft/fleet/rollout_active_ratio": active_ratio,
        "ft/evictions_total/task_session": float(task_session_evictions),
        "ft/rollout/task_retries_total": (
            float(rollout_session.task_retries_total)
            if rollout_session is not None
            else 0.0
        ),
        "ft/rollout/task_unrecovered_total": (
            float(rollout_session.task_unrecovered_total)
            if rollout_session is not None
            else 0.0
        ),
        "ft/rejoin/promoted_total": float(promoted),
        "ft/rejoin/bring_up_failures_total": (
            float(self._lifecycle_driver.bring_up_failures_total)
            if self._lifecycle_driver is not None
            else 0.0
        ),
    }

  def _update_rollout_capacity_state(self) -> RolloutCapacityError | None:
    """Updates rollout capacity watermarks/timers and returns any floor error."""
    threshold = (
        self._config.min_active_rollout_ratio if self._config.enabled else None
    )
    log_recovered = False
    log_dropped = False
    error: RolloutCapacityError | None = None
    with self._lock:
      active = len(self._registry.handles(datatypes.Role.ROLLOUT))
      if active > self._peak_active_rollout_workers:
        self._peak_active_rollout_workers = active
      peak = self._peak_active_rollout_workers
      if threshold is None or peak == 0:
        return None
      ratio = active / peak
      if ratio >= threshold:
        log_recovered = self._below_min_active_since is not None
        self._below_min_active_since = None
      else:
        now = _monotonic()
        if self._below_min_active_since is None:
          self._below_min_active_since = now
          log_dropped = True
        elapsed_s = now - self._below_min_active_since
        if elapsed_s >= self._config.min_active_rollout_grace_s:
          error = RolloutCapacityError(
              "Active rollout workers remained below minimum ratio for"
              f" {elapsed_s:.1f}s (grace"
              f" {self._config.min_active_rollout_grace_s:.1f}s):"
              f" {active}/{peak} active ({ratio:.3f} < {threshold:.3f})."
          )
    if log_recovered:
      logging.info(
          "Rollout active capacity recovered to %d/%d (%.1f%% >= %.1f%%).",
          active,
          peak,
          ratio * 100.0,
          threshold * 100.0,
      )
    elif log_dropped:
      logging.warning(
          "Rollout active capacity dropped below minimum: %d/%d (%.1f%% <"
          " %.1f%%); will abort if not recovered within %.1fs.",
          active,
          peak,
          ratio * 100.0,
          threshold * 100.0,
          self._config.min_active_rollout_grace_s,
      )
    return error

  def check_rollout_capacity(self) -> None:
    """Updates the peak active rollout count and enforces the capacity floor.

    The guardrail is armed once at least one rollout worker has reached
    `AdmissionStatus.ACTIVE`. When `config.enabled` is True and
    `config.min_active_rollout_ratio` is set, starts a monotonic timer the first
    time `active / peak_active < min_active_rollout_ratio` (including on worker
    eviction via `on_worker_unreachable()`), resets the timer whenever the ratio
    recovers to `>= min_active_rollout_ratio`, and raises `RolloutCapacityError`
    once the elapsed time below threshold reaches
    `config.min_active_rollout_grace_s`.

    Raises:
      RolloutCapacityError: If the active rollout ratio remains strictly below
        `config.min_active_rollout_ratio` for at least
        `config.min_active_rollout_grace_s` seconds.
    """
    error = self._update_rollout_capacity_state()
    if error is not None:
      raise error

  def task_session_fault_tolerance_config(
      self,
  ) -> task_session.TaskSessionFaultToleranceConfig:
    """Derives the `TaskSessionFaultToleranceConfig` for `TaskSession`."""
    return task_session.TaskSessionFaultToleranceConfig(
        max_task_retries=self._config.effective_max_task_retries,
        on_worker_unreachable=(
            self.on_worker_unreachable if self._config.enabled else None
        ),
    )

  def on_worker_unreachable(
      self,
      handle: remote_execution.ActorHandle,
      exc: Exception,
  ) -> None:
    """Evicts a worker handle that `TaskSession` found unreachable."""
    worker_id = self._registry.unregister_handle(handle)
    if worker_id is None:
      return
    logging.warning(
        "Evicted failed worker %r from WorkerRegistry (task_session): %s",
        worker_id,
        exc,
    )
    # Arm the capacity grace timer at eviction time without raising inside
    # TaskSession's callback; check_rollout_capacity() raises on the next poll
    # or weight-sync boundary once the grace window elapses.
    self._update_rollout_capacity_state()
    with self._lock:
      self._task_session_evictions_total += 1
