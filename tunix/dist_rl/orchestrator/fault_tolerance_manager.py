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
import threading

from absl import logging
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import lifecycle
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.worker import remote_execution
from tunix.dist_rl.worker import task_session


class FaultToleranceManager:
  """Applies `FaultToleranceConfig` to worker eviction and task retries.

  Derives the `TaskSession` fault-tolerance config, evicts unreachable worker
  handles from `WorkerRegistry`, and aggregates `ft/*` fleet gauges and
  cumulative fault-tolerance counters. Holds no task ledger; worker bring-up,
  admission, and shutdown belong to `LifecycleDriver`.
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

    `ft/fleet/*` counts the rollout workers in each `AdmissionStatus` now. Every
    other metric is a running total since the manager was created; take the
    per-step difference to see what happened during one step.

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
    with self._lock:
      task_session_evictions = self._task_session_evictions_total
      promoted = self._promoted_total
    return {
        **{
            f"ft/fleet/rollout_{status.value}": float(statuses[status])
            for status in worker_registry.AdmissionStatus
        },
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

  # TODO(b/577214994): Extend the eviction path:
  # - Suspect-then-confirm (F2): a failed `dispatch_task` only marks the handle
  #   suspect (no new tasks) via a future `on_worker_suspect(handle, exc)`;
  #   evict only once the worker's poll loop confirms the failure, and let
  #   unconfirmed suspicion expire. Prerequisite for defaulting
  #   `FaultToleranceConfig.enabled` to `True`.
  # - Notify evicted-but-alive workers (F3): before eviction closes the handle,
  #   best-effort `handle.submit("shutdown")` (ignoring failures) so the
  #   platform (k8s) restarts the worker and it rejoins via late registration.
  #   Today a falsely evicted live worker never re-registers and permanently
  #   shrinks the rollout pool.
  # - Failure accounting: log total failure events and distinct failed workers,
  #   report the failed-worker percentage, and add
  #   `terminate_on_worker_failure_percent` to fail the job past a threshold.
  def on_worker_unreachable(
      self,
      handle: remote_execution.ActorHandle,
      exc: Exception,
  ) -> None:
    """Evicts a worker handle that `TaskSession` found unreachable."""
    worker_id = self._registry.unregister_handle(handle)
    if worker_id is None:
      return
    with self._lock:
      self._task_session_evictions_total += 1
    logging.warning(
        "Evicted failed worker %r from WorkerRegistry (task_session): %s",
        worker_id,
        exc,
    )
