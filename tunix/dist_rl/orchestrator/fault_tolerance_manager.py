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
from collections.abc import Callable, Sequence
import threading
import time
from typing import Literal

from absl import logging
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import lifecycle
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.worker import remote_execution
from tunix.dist_rl.worker import task_session


_monotonic: Callable[[], float] = time.monotonic
_GraceTransition = Literal["recovered", "dropped"]

StagingSlotsFn = Callable[
    [Sequence[tuple[datatypes.WorkerInfo, remote_execution.ActorHandle]]],
    int,
]


def _advance_grace_timer(
    is_below: bool,
    since: float | None,
    now_fn: Callable[[], float],
) -> tuple[float | None, _GraceTransition | None, float | None]:
  """Advances one grace timer, returning `(since, transition, elapsed_s)`."""
  if not is_below:
    return None, ("recovered" if since is not None else None), None
  now = now_fn()
  if since is None:
    return now, "dropped", 0.0
  return since, None, now - since


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
    self._staging_slots_fn: StagingSlotsFn | None = None
    self._required_staging_slots: int = 0
    self._below_staging_slots_since: float | None = None

  @property
  def config(self) -> datatypes.FaultToleranceConfig:
    """Returns the active `FaultToleranceConfig`."""
    return self._config

  @property
  def peak_active_rollout_workers(self) -> int:
    """Returns the highest `ACTIVE` rollout worker count observed so far."""
    with self._lock:
      return self._peak_active_rollout_workers

  def record_rejoin_promoted(self) -> None:
    """Counts a late-arriving or rejoined rollout worker promoted to ACTIVE."""
    with self._lock:
      self._promoted_total += 1

  def set_staging_slot_floor(
      self,
      slots_fn: StagingSlotsFn | None,
      required_slots: int = 0,
  ) -> None:
    """Installs or clears the cluster-wide Relay staging-slot capacity floor.

    Args:
      slots_fn: Callback receiving the current `ACTIVE` rollout `(WorkerInfo,
        ActorHandle)` sequence (invoked outside `FaultToleranceManager`'s lock)
        and returning total cluster-wide staging slots, or `None` to clear the
        floor. Any `ValueError` or `RuntimeError` raised by `slots_fn` (such as
        missing worker metadata or 0 physical slots) is surfaced immediately by
        `check_rollout_capacity()` without a grace window.
      required_slots: Minimum cluster-wide staging slots required (`<= 0`
        disables the floor).
    """
    with self._lock:
      if slots_fn is None or required_slots <= 0:
        self._staging_slots_fn = None
        self._required_staging_slots = 0
      else:
        self._staging_slots_fn = slots_fn
        self._required_staging_slots = required_slots
      self._below_staging_slots_since = None

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

  def _evaluate_ratio_floor_locked(
      self,
      active: int,
      peak: int,
      now_fn: Callable[[], float],
  ) -> tuple[
      float,
      float | None,
      _GraceTransition | None,
      RolloutCapacityError | None,
  ]:
    """Evaluates the active-worker ratio floor under `self._lock`."""
    threshold = (
        self._config.min_active_rollout_ratio if self._config.enabled else None
    )
    if threshold is None or peak == 0:
      return 1.0, threshold, None, None
    ratio = active / peak
    grace_s = self._config.min_active_rollout_grace_s
    self._below_min_active_since, event, elapsed_s = _advance_grace_timer(
        ratio < threshold, self._below_min_active_since, now_fn
    )
    error = None
    if elapsed_s is not None and elapsed_s >= grace_s:
      error = RolloutCapacityError(
          "Active rollout workers remained below minimum ratio for"
          f" {elapsed_s:.1f}s (grace {grace_s:.1f}s):"
          f" {active}/{peak} active ({ratio:.3f} < {threshold:.3f})."
      )
    return ratio, threshold, event, error

  def _evaluate_staging_floor_locked(
      self,
      active: int,
      total_slots: int,
      required_slots: int,
      slots_exc: Exception | None,
      now_fn: Callable[[], float],
  ) -> tuple[_GraceTransition | None, Exception | None]:
    """Evaluates a pre-sampled Relay staging-slot count under `self._lock`."""
    if slots_exc is not None:
      return None, slots_exc
    if not self._config.enabled:
      if total_slots < required_slots:
        return (
            None,
            RolloutCapacityError(
                f"Active rollout pool has only {total_slots} staging slots"
                f" across {active} worker(s), below required deadlock-free"
                f" lower bound {required_slots}. Increase"
                " RolloutConfig.staging_max_bytes, rollout_replicas, or"
                " staging_overcommit_factor."
            ),
        )
      return None, None
    grace_s = self._config.min_active_rollout_grace_s
    self._below_staging_slots_since, event, elapsed_s = _advance_grace_timer(
        total_slots < required_slots,
        self._below_staging_slots_since,
        now_fn,
    )
    error = None
    if elapsed_s is not None and elapsed_s >= grace_s:
      error = RolloutCapacityError(
          "Active rollout staging slots remained below required bound for"
          f" {elapsed_s:.1f}s (grace {grace_s:.1f}s): {total_slots} <"
          f" {required_slots} across {active} worker(s)."
      )
    return event, error

  def _update_rollout_capacity_state(self) -> Exception | None:
    """Updates rollout capacity watermarks/timers and returns any floor error."""
    # Invoked exclusively on the orchestrator's single event-loop thread.
    grace_s = self._config.min_active_rollout_grace_s
    now: float | None = None

    def _now() -> float:
      nonlocal now
      if now is None:
        now = _monotonic()
      return now

    targets = self._registry.targets(
        datatypes.Role.ROLLOUT,
        statuses=(worker_registry.AdmissionStatus.ACTIVE,),
    )
    with self._lock:
      slots_fn = self._staging_slots_fn
      required_slots = self._required_staging_slots

    total_slots = 0
    slots_exc: Exception | None = None
    if slots_fn is not None and required_slots > 0:
      try:
        total_slots = slots_fn(targets)
      except (ValueError, RuntimeError) as exc:
        slots_exc = exc

    active = len(targets)
    slots_event: _GraceTransition | None = None
    slots_err: Exception | None = None
    with self._lock:
      if active > self._peak_active_rollout_workers:
        self._peak_active_rollout_workers = active
      peak = self._peak_active_rollout_workers
      ratio, threshold, ratio_event, ratio_err = (
          self._evaluate_ratio_floor_locked(active, peak, _now)
      )
      if (
          slots_fn is not None
          and required_slots > 0
          and self._staging_slots_fn is slots_fn
          and self._required_staging_slots == required_slots
      ):
        slots_event, slots_err = self._evaluate_staging_floor_locked(
            active, total_slots, required_slots, slots_exc, _now
        )
      error: Exception | None = ratio_err or slots_err

    if ratio_event == "recovered":
      assert threshold is not None
      logging.info(
          "Rollout active capacity recovered to %d/%d (%.1f%% >= %.1f%%).",
          active,
          peak,
          ratio * 100.0,
          threshold * 100.0,
      )
    elif ratio_event == "dropped":
      assert threshold is not None
      logging.warning(
          "Rollout active capacity dropped below minimum: %d/%d (%.1f%% <"
          " %.1f%%); will abort if not recovered within %.1fs.",
          active,
          peak,
          ratio * 100.0,
          threshold * 100.0,
          grace_s,
      )
    if slots_event == "recovered":
      logging.info(
          "Active rollout staging slots recovered to %d >= %d across %d"
          " worker(s).",
          total_slots,
          required_slots,
          active,
      )
    elif slots_event == "dropped":
      logging.warning(
          "Active rollout staging slots dropped below required bound: %d < %d"
          " across %d worker(s); will abort if not recovered within %.1fs.",
          total_slots,
          required_slots,
          active,
          grace_s,
      )
    return error

  def check_rollout_capacity(self) -> None:
    """Updates the peak active rollout count and enforces capacity floors.

    Enforces both the active-worker ratio floor (armed once at least one rollout
    worker has reached `AdmissionStatus.ACTIVE`) and any Relay staging-slot
    floor installed via `set_staging_slot_floor`. Under `config.enabled=True`,
    starts monotonic grace timers on shortfall (including at worker eviction via
    `on_worker_unreachable()`), resets them on recovery, and raises
    `RolloutCapacityError` once `config.min_active_rollout_grace_s` elapses.

    Raises:
      RolloutCapacityError: If the active rollout ratio or staging-slot count
        remains below its floor for at least `config.min_active_rollout_grace_s`
        seconds (or immediately when `config.enabled=False`).
      ValueError: If an active worker is missing required staging metadata.
      RuntimeError: If an active worker has 0 physical staging slots.
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
