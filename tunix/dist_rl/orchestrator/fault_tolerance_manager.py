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

from absl import logging
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.worker import remote_execution
from tunix.dist_rl.worker import task_session


class FaultToleranceManager:
  """Applies `FaultToleranceConfig` to worker eviction and task retries.

  Derives the `TaskSession` fault-tolerance config and evicts unreachable worker
  handles from `WorkerRegistry`. Holds no task ledger; worker bring-up,
  admission, and shutdown belong to `LifecycleDriver`.
  """

  def __init__(
      self,
      registry: worker_registry.WorkerRegistry,
      config: datatypes.FaultToleranceConfig | None = None,
  ):
    self._registry = registry
    self._config = config or datatypes.FaultToleranceConfig()

  @property
  def config(self) -> datatypes.FaultToleranceConfig:
    """Returns the active `FaultToleranceConfig`."""
    return self._config

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
    """Evicts an unreachable worker handle from the control-plane registry."""
    worker_id = self._registry.unregister_handle(handle)
    if worker_id is not None:
      logging.warning(
          "Evicted unreachable worker %r from WorkerRegistry: %s",
          worker_id,
          exc,
      )
