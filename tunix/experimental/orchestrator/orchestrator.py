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

"""Cluster Infrastructure Coordinator (orchestrator.py) following Orchestrator V2.

Supervises WorkerRegistry, LifecycleDriver, HealthMonitor, and StartupValidator.
Provides Tier 1 Zero-Boilerplate Managed Program Submission (`run`) and Tier 3
Custom Program Execution (`run_program`).
"""

from collections.abc import Callable, Sequence
from concurrent import futures
from typing import Any

from absl import logging
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.orchestrator import health_monitor
from tunix.experimental.orchestrator import lifecycle
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.orchestrator import startup_validation
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.weight_sync import weight_sync_coordinator
from tunix.experimental.worker import abstract_worker
from tunix.experimental.worker import remote_execution

_STOP_TIMEOUT_S = 60.0  # Timeout for stopping remote workers. 60 should not be touched for any healthy stop.


class ClusterOrchestrator:
  """Supervises cluster hardware, health monitoring, and program execution."""

  def __init__(
      self,
      config: Any = None,
      registry: worker_registry.WorkerRegistry | None = None,
      lifecycle_driver: lifecycle.LifecycleDriver | None = None,
      monitor: health_monitor.HealthMonitor | None = None,
      weight_sync_mode: str | None = None,
  ):
    """Initializes ClusterOrchestrator."""
    self.config = config
    self.registry = registry or worker_registry.WorkerRegistry()
    self.lifecycle_driver = lifecycle_driver or lifecycle.LifecycleDriver(
        self.registry
    )
    self.monitor = monitor or health_monitor.HealthMonitor(self.registry)
    self.engine: distributed_rl_engine.DistributedRLEngine | None = None
    mode = getattr(weight_sync_mode, "value", weight_sync_mode)
    self._weight_sync_mode = str(mode).lower() if mode is not None else None

  def __enter__(self) -> "ClusterOrchestrator":
    """Interactive context manager bring-up."""
    self.bring_up_workers()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.shutdown()

  def register_worker_from_hostname(
      self,
      hostname: str,
      port: int,
      metadata: bytes,
      rpc_timeout_s: float = 1800.0,
  ) -> datatypes.WorkerInfo:
    """Registers a remote worker handle from a hostname and metadata."""
    return self.registry.register_from_hostname(
        hostname=hostname,
        port=port,
        metadata=metadata,
        rpc_timeout_s=rpc_timeout_s,
    )

  def register_worker(
      self, worker: abstract_worker.Worker
  ) -> datatypes.WorkerInfo:
    """Registers a worker in the WorkerRegistry."""
    return self.registry.register(worker)

  def register_worker_handle(
      self,
      worker_id: str,
      roles: Sequence[datatypes.Role | str],
      handle: remote_execution.ActorHandle,
      resources: dict[str, Any] | None = None,
  ) -> datatypes.WorkerInfo:
    """Registers a remote worker handle used directly by DistributedRLEngine."""
    return self.registry.register_handle(
        worker_id=worker_id,
        roles=roles,
        handle=handle,
        resources={"remote": True, **dict(resources or {})},
    )

  def unregister_worker(self, worker_id: str) -> None:
    """Unregisters a worker by its id."""
    self.registry.unregister(worker_id)

  def wait_for_workers(
      self,
      min_workers: dict[datatypes.Role | str, int],
      timeout: float | None = None,
      poll_interval_s: float = 0.5,
  ) -> None:
    """Waits for registered workers to meet the minimum required counts.

    Args:
      min_workers: A dictionary mapping Role or role name to the minimum number
        of workers required.
      timeout: Maximum duration to wait in seconds before raising TimeoutError.
        If None, waits indefinitely until requirements are met.
      poll_interval_s: Time in seconds between polling attempts.

    Raises:
      TimeoutError: If the required worker counts are not met within timeout.
    """
    self.registry.wait_for_workers(
        min_workers=min_workers,
        timeout=timeout,
        poll_interval_s=poll_interval_s,
    )

  def worker_infos(self) -> list[datatypes.WorkerInfo]:
    """Returns worker metadata registered with the orchestrator."""
    return self.registry.infos()

  def worker_handles(
      self, role: datatypes.Role | str
  ) -> list[remote_execution.ActorHandle]:
    """Returns handles for all workers registered under the given role."""
    return self.registry.handles(role)

  @property
  def _remote_worker_handles(
      self,
  ) -> dict[str, list[remote_execution.ActorHandle]]:
    return {role: self.registry.handles(role) for role in self.registry.roles()}

  @property
  def _remote_worker_handles_by_id(
      self,
  ) -> dict[str, remote_execution.ActorHandle]:
    return {
        wid: self.registry.get_handle(wid) for wid in self.registry.worker_ids()
    }

  @property
  def _remote_worker_infos(self) -> dict[str, datatypes.WorkerInfo]:
    return {wid: self.registry.info(wid) for wid in self.registry.worker_ids()}

  def bring_up_workers(self, dummy_data: Any = None) -> None:
    """Brings up all registered workers through lifecycle initialization."""
    logging.info(
        "Bringing up %d registered worker(s)...",
        len(self.worker_infos()),
    )
    self.lifecycle_driver.bring_up(dummy_data)
    self._bring_up_remote_workers(dummy_data)
    self.engine = self._create_engine()
    logging.info("All workers brought up successfully.")

  def shutdown(self) -> None:
    """Shuts down all workers and closes health monitoring resources."""
    logging.info("Shutting down all workers...")
    self.monitor.close()
    self._shutdown_remote_workers()
    self.lifecycle_driver.shutdown()
    logging.info("Shutdown complete.")

  def validate_startup(self, alg_config: Any, training_config: Any) -> None:
    """Validates cluster geometry against configurations."""
    startup_validation.validate_startup(
        self.registry, alg_config, training_config
    )

  def _get_role_members(self, role: datatypes.Role | str) -> list[Any]:
    return self.registry.group(role).members()

  def _get_actor_handles(
      self, role: datatypes.Role | str
  ) -> list[remote_execution.ActorHandle]:
    return self.registry.handles(role)

  def _bring_up_remote_workers(self, dummy_data: Any = None) -> None:
    """Runs lifecycle hooks on registered remote worker handles."""
    worker_ids = [
        wid
        for wid in self.registry.worker_ids()
        if self.registry.info(wid).resources.get("remote", False)
    ]
    for worker_id in worker_ids:
      logging.info("Initializing worker %s.", worker_id)
      self.registry.get_handle(worker_id).submit("initialize")
    for worker_id in worker_ids:
      logging.info("Compiling worker %s.", worker_id)
      self.registry.get_handle(worker_id).submit("compile", dummy_data)
    for worker_id in worker_ids:
      logging.info("Starting worker %s.", worker_id)
      self.registry.get_handle(worker_id).submit("start")

  def _shutdown_remote_workers(self) -> None:
    """Stops remote worker handles best-effort, with a hard timeout."""
    worker_ids = [
        wid
        for wid in self.registry.worker_ids()
        if self.registry.info(wid).resources.get("remote", False)
    ]
    pool = futures.ThreadPoolExecutor(max_workers=4)
    stops = {
        worker_id: pool.submit(
            self.registry.get_handle(worker_id).submit, "stop"
        )
        for worker_id in worker_ids
    }
    for worker_id, fut in stops.items():
      try:
        fut.result(timeout=_STOP_TIMEOUT_S)
      except Exception as err:  # pylint: disable=broad-except
        logging.warning("Failed to stop worker %s: %r", worker_id, err)
    pool.shutdown(wait=False)

  def _create_engine(self) -> distributed_rl_engine.DistributedRLEngine:
    """Constructs a DistributedRLEngine from the registered role groups."""
    rollout_workers = self.registry.handles(datatypes.Role.ROLLOUT)
    actor_workers = self.registry.handles(datatypes.Role.ACTOR)
    critic_workers = self.registry.handles(datatypes.Role.CRITIC)
    reference_workers = self.registry.handles(datatypes.Role.REFERENCE)

    trainer_workers = {}
    if actor_workers:
      trainer_workers[datatypes.Role.ACTOR] = actor_workers[0]
    if critic_workers:
      trainer_workers[datatypes.Role.CRITIC] = critic_workers[0]

    inference_workers = {}
    if reference_workers:
      inference_workers[datatypes.Role.REFERENCE] = reference_workers[0]

    coordinator = None
    if self._weight_sync_mode not in (None, "none"):
      handler = weight_sync_coordinator.create_default_handler(
          mode=self._weight_sync_mode
      )

      handle_to_id = {
          self.registry.get_handle(wid): wid
          for wid in self.registry.worker_ids()
      }
      for role, handles in [
          (datatypes.Role.ACTOR, actor_workers),
          (datatypes.Role.ROLLOUT, rollout_workers),
      ]:
        for h in handles:
          w_id = handle_to_id.get(h, f"local-{role.value}-{id(h)}")
          info = (
              self.registry.info(w_id)
              if w_id in self.registry
              else datatypes.WorkerInfo(
                  worker_id=w_id, roles=frozenset({role.value})
              )
          )
          self.registry.register(
              weight_sync_coordinator.RemoteWorkerShim(h, info),  # pyrefly: ignore[bad-argument-type]
              override=True,
          )

      coordinator = weight_sync_coordinator.WeightSyncCoordinator(
          registry=self.registry,
          handler=handler,
          controller_id="auto-coordinator",
      )

    return distributed_rl_engine.DistributedRLEngine(
        rollout_workers=rollout_workers,
        trainer_workers=trainer_workers,
        inference_workers=inference_workers,
        weight_sync_coordinator=coordinator,
    )

  def run_program(
      self,
      program: rl_program.RLProgram,
      bring_up: bool = True,
      dummy_data: Any = None,
      **kwargs: Any,
  ) -> None:
    """Runs an RL program to completion under supervision."""
    if bring_up:
      self.bring_up_workers(dummy_data=dummy_data)

    self.monitor.poll()
    logging.info("Executing program %s...", type(program).__name__)
    engine = self.engine or self._create_engine()

    program.run(
        engine=engine,
        **kwargs,
    )
    logging.info("Program %s finished.", type(program).__name__)

  def run(
      self,
      algo: algorithm_adapter.AlgorithmAdapter,
      dataset: Any,
      reward_fns: Sequence[Callable[..., Any]] | None = None,
      assembler: batch_assembly.BatchAssembler | None = None,
      generation_args: datatypes.GenerationArgs | None = None,
      program: rl_program.RLProgram | None = None,
      max_steps: int = 1000,
  ) -> None:
    """Managed Program Submission: auto-wires Engine, Assembler, Queues & StandardRLProgram."""
    logging.info("Starting managed RL program run (max_steps=%d)...", max_steps)
    if self.engine is None:
      self.bring_up_workers()
    metrics_logging_options = getattr(
        self.config, "metrics_logging_options", None
    )
    metrics_prefix = getattr(self.config, "metrics_prefix", "")
    active_program = program or rl_program.StandardRLProgram(
        dataset=dataset,
        max_steps=max_steps,
        algo=algo,
        reward_fns=reward_fns,
        assembler=assembler,
        generation_args=generation_args,
        metrics_logging_options=metrics_logging_options,
        metrics_prefix=metrics_prefix,
    )
    try:
      self.run_program(
          program=active_program,
          bring_up=False,
      )
    finally:
      if program is None:
        bg_task = getattr(active_program, "_bg_task", None)
        if bg_task is not None and not bg_task.done():
          bg_task.add_done_callback(lambda _: active_program.close())
        else:
          active_program.close()
