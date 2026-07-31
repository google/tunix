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

"""Wires the control plane to the worker handles the orchestrator calls.

The control plane ships as three composable pieces that all hang off a shared
registry:

    WorkerRegistry   -- who exists, indexed by role
    LifecycleDriver  -- brings the fleet up phase by phase, and tears it down
    HealthMonitor    -- polls heartbeats and flags workers stuck in a state

and, separately, the orchestrator needs *callable handles* for its compute
primitives. `WorkerFleet` is the missing seam between the two: the same objects
are registered as workers (so they can be brought up and monitored) and handed
to `OrchestratorRLCluster` as handles (so they can be called).

    fleet = WorkerFleet.in_process(rl_cluster)
    fleet.bring_up()
    cluster = fleet.build_cluster(rl_cluster)   # routed to the handles
    ...
    fleet.poll_health()
    fleet.shutdown()

Nothing here is in-process-specific: swapping the handles for RPC-backed ones
(same two contracts) distributes the run without changing the orchestrator or
the loop.
"""

from typing import Any, Mapping, Optional, Sequence, Tuple

from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import health_monitor as health_monitor_lib
from tunix.experimental.orchestrator import inprocess_workers
from tunix.experimental.orchestrator import lifecycle as lifecycle_lib
from tunix.experimental.orchestrator import orchestrator_rl_cluster
from tunix.experimental.orchestrator import rollout_pool as rollout_pool_lib
from tunix.experimental.orchestrator import worker_registry as worker_registry_lib
from tunix.experimental.worker import abstract_worker


def _as_sequence(handles: Any) -> Tuple[Any, ...]:
  """Normalizes a single handle, a sequence of them, or None into a tuple."""
  if handles is None:
    return ()
  if isinstance(handles, (list, tuple)):
    return tuple(handles)
  return (handles,)


class WorkerFleet:
  """Registry + lifecycle + health for the workers the orchestrator drives."""

  def __init__(
      self,
      *,
      trainer: Any = None,
      rollout: Any = None,
      inference: Any = None,
      weight_sync: Any = None,
      rollout_max_concurrency: int = 1,
      state_deadlines_s: Optional[Mapping[str, float]] = None,
      clock: Any = None,
  ):
    """Builds a fleet from the handles the orchestrator will call.

    Args:
      trainer: Trainer handle (`train`, optionally `per_token_logps`).
      rollout: Rollout handle (`generate`), or a sequence of them to run as a
        pool. Every member is registered, so the control plane brings up and
        monitors each one individually.
      inference: Inference handle (`per_token_logps`) for reference scoring.
      weight_sync: Weight-sync handle (`sync`). Not a Worker, so it is not
        registered or lifecycle-managed.
      rollout_max_concurrency: Trajectories each rollout worker generates at
        once when work is dispatched through `rollout_pool`.
      state_deadlines_s: Optional per-state deadlines for the health monitor.
      clock: Optional monotonic clock for the health monitor (tests).
    """
    self._trainer = trainer
    self._rollout_workers = _as_sequence(rollout)
    self._inference = inference
    self._weight_sync = weight_sync

    self._registry = worker_registry_lib.WorkerRegistry()
    # Only handles that are real Workers can be lifecycle-managed and polled.
    for handle in (trainer, *self._rollout_workers, inference):
      if isinstance(handle, abstract_worker.Worker):
        self._registry.register(handle)

    self._rollout_max_concurrency = rollout_max_concurrency
    self._rollout_pool: Optional[rollout_pool_lib.PooledRolloutWorker] = None

    self._lifecycle = lifecycle_lib.LifecycleDriver(self._registry)
    monitor_kwargs: dict[str, Any] = {}
    if state_deadlines_s is not None:
      monitor_kwargs["state_deadlines_s"] = state_deadlines_s
    if clock is not None:
      monitor_kwargs["clock"] = clock
    self._health = health_monitor_lib.HealthMonitor(
        self._registry, **monitor_kwargs
    )

  @classmethod
  def in_process(cls, rl_cluster: Any, **kwargs) -> "WorkerFleet":
    """Builds the full handle set backed by one in-process `RLCluster`."""
    return cls(
        trainer=inprocess_workers.InProcessTrainerWorker(rl_cluster),
        rollout=inprocess_workers.InProcessRolloutWorker(rl_cluster),
        inference=inprocess_workers.InProcessInferenceWorker(rl_cluster),
        weight_sync=inprocess_workers.InProcessWeightSync(rl_cluster),
        **kwargs,
    )

  # --- Control plane --------------------------------------------------------

  def bring_up(self, dummy_data: Any = None) -> None:
    """Runs initialize -> compile -> start across the fleet, phase by phase."""
    self._lifecycle.bring_up(dummy_data)

  def shutdown(self) -> None:
    """Stops every worker (best effort; failures are aggregated and raised)."""
    self._lifecycle.shutdown()

  def poll_health(self) -> dict[str, datatypes.HealthReport]:
    """Heartbeats every registered worker."""
    return self._health.poll()

  def overdue(self) -> list[health_monitor_lib.OverdueWorker]:
    """Workers stuck in a transient state past its deadline."""
    return self._health.overdue()

  # --- Data plane -----------------------------------------------------------

  def build_cluster(self, base: Any) -> Any:
    """Returns an `OrchestratorRLCluster` routed to this fleet's handles.

    The cluster's `generate` takes a whole prompt batch and blocks on it, so it
    is routed to a single rollout worker. Trajectory-level work that should be
    spread over the pool goes through `rollout_pool` instead.

    Args:
      base: The in-process cluster supplying the surface that is not routed
        (config, tokenizer, metrics, step counter).
    """
    return orchestrator_rl_cluster.OrchestratorRLCluster(
        base,
        trainer_worker=self._trainer,
        rollout_worker=self.rollout,
        inference_worker=self._inference,
        weight_sync=self._weight_sync,
    )

  # --- Accessors ------------------------------------------------------------

  @property
  def registry(self) -> worker_registry_lib.WorkerRegistry:
    return self._registry

  @property
  def lifecycle(self) -> lifecycle_lib.LifecycleDriver:
    return self._lifecycle

  @property
  def health(self) -> health_monitor_lib.HealthMonitor:
    return self._health

  @property
  def trainer(self) -> Any:
    return self._trainer

  @property
  def rollout(self) -> Any:
    """The rollout worker used for whole-batch, blocking generation."""
    return self._rollout_workers[0] if self._rollout_workers else None

  @property
  def rollout_workers(self) -> Sequence[Any]:
    """Every rollout worker in the fleet, each managed by the control plane."""
    return self._rollout_workers

  @property
  def rollout_pool(self) -> Optional[rollout_pool_lib.PooledRolloutWorker]:
    """Load-balanced view of the rollout workers, for per-trajectory requests.

    Dispatches one `RolloutRequest` per trajectory across the fleet's rollout
    workers, so it requires workers speaking the `RolloutWorker` contract
    (`generate(request) -> RolloutResponse`). The `InProcessRolloutWorker`
    adapter speaks the cluster's whole-batch `generate(prompts, ...)` instead
    and belongs on `build_cluster`, not here.
    """
    if self._rollout_pool is None and self._rollout_workers:
      self._rollout_pool = rollout_pool_lib.PooledRolloutWorker.from_workers(
          self._rollout_workers,
          max_concurrency=self._rollout_max_concurrency,
      )
    return self._rollout_pool

  @property
  def inference(self) -> Any:
    return self._inference

  @property
  def weight_sync(self) -> Any:
    return self._weight_sync
