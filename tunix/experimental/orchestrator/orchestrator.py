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

import collections
from collections.abc import Callable, Iterable, Sequence
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
      rollout_router: Any = None,
  ):
    """Initializes ClusterOrchestrator.

    Args:
      config: Optional opaque orchestrator config.
      registry: Worker registry; created if not provided.
      lifecycle_driver: Lifecycle driver; created if not provided.
      monitor: Health monitor; created if not provided.
      rollout_router: Optional rollout worker picker forwarded to the engine's
        RoutingActorPool (e.g. remote_scheduler_router.RemoteSchedulerRouter).
    """
    self.config = config
    self.registry = registry or worker_registry.WorkerRegistry()
    self.lifecycle_driver = lifecycle_driver or lifecycle.LifecycleDriver(
        self.registry
    )
    self.monitor = monitor or health_monitor.HealthMonitor(self.registry)
    self.rollout_router = rollout_router
    self._remote_worker_handles: dict[
        str, list[remote_execution.ActorHandle]
    ] = collections.defaultdict(list)
    self._remote_worker_handles_by_id: dict[
        str, remote_execution.ActorHandle
    ] = {}
    self._remote_worker_infos: dict[str, datatypes.WorkerInfo] = {}
    self.engine: distributed_rl_engine.DistributedRLEngine | None = None
    mode = getattr(weight_sync_mode, "value", weight_sync_mode)
    self._weight_sync_mode = str(mode).lower() if mode is not None else None

  def __enter__(self) -> "ClusterOrchestrator":
    """Interactive context manager bring-up."""
    self.bring_up_workers()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.shutdown()

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
    if not roles:
      raise ValueError(f"worker {worker_id!r} declares no roles")
    if not isinstance(handle, remote_execution.ActorHandle):
      raise TypeError(
          "register_worker_handle expects a remote_execution.ActorHandle, got "
          f"{type(handle)}"
      )
    if (
        worker_id in self._remote_worker_infos
        or worker_id in self.registry.worker_ids()
    ):
      raise ValueError(f"duplicate worker_id: {worker_id!r}")
    role_names = frozenset(
        role.value if isinstance(role, datatypes.Role) else role
        for role in roles
    )
    info = datatypes.WorkerInfo(
        worker_id=worker_id,
        roles=role_names,
        resources={"remote": True, **dict(resources or {})},
    )
    for role in role_names:
      self._remote_worker_handles[role].append(handle)
    self._remote_worker_handles_by_id[worker_id] = handle
    self._remote_worker_infos[worker_id] = info
    logging.info(
        "Registered remote worker %r with roles %s.",
        worker_id,
        sorted(role_names),
    )
    return info

  def unregister_worker(self, worker_id: str) -> None:
    """Unregisters a worker by its id."""
    if worker_id in self._remote_worker_infos:
      info = self._remote_worker_infos.pop(worker_id)
      handle = self._remote_worker_handles_by_id.pop(worker_id)
      for role in info.roles:
        handles = self._remote_worker_handles.get(role)
        if handles is not None:
          self._remote_worker_handles[role] = [
              h for h in handles if h is not handle
          ]
          if not self._remote_worker_handles[role]:
            del self._remote_worker_handles[role]
      return
    self.registry.unregister(worker_id)

  def worker_infos(self) -> list[datatypes.WorkerInfo]:
    """Returns local and remote worker metadata registered with the orchestrator."""
    registry_ids = self.registry.worker_ids()
    return self.registry.infos() + [
        self._remote_worker_infos[worker_id]
        for worker_id in sorted(self._remote_worker_infos)
        if worker_id not in registry_ids
    ]

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
    role_key = role.value if isinstance(role, datatypes.Role) else role
    members = self.registry.group(role_key).members()

    # Fallback in case workers were registered with the enum object directly
    if not members and isinstance(role, datatypes.Role):
      members = self.registry.group(role).members()
    return members

  def _get_actor_handles(
      self, role: datatypes.Role | str
  ) -> list[remote_execution.ActorHandle]:
    role_key = role.value if isinstance(role, datatypes.Role) else role
    handles = list(self._remote_worker_handles.get(role_key, ()))
    handles.extend(
        remote_execution.InProcessActorHandle(
            remote_execution.InProcessRemoteExecutionServer(worker)
        )
        for worker in self._get_role_members(role)
    )
    return handles

  def _bring_up_remote_workers(self, dummy_data: Any = None) -> None:
    """Runs lifecycle hooks on remote worker handles registered directly."""
    worker_ids = sorted(self._remote_worker_infos)
    for worker_id in worker_ids:
      logging.info("Initializing remote worker %s.", worker_id)
      self._remote_worker_handles_by_id[worker_id].submit("initialize")
    for worker_id in worker_ids:
      logging.info("Compiling remote worker %s.", worker_id)
      self._remote_worker_handles_by_id[worker_id].submit("compile", dummy_data)
    for worker_id in worker_ids:
      logging.info("Starting remote worker %s.", worker_id)
      self._remote_worker_handles_by_id[worker_id].submit("start")

  def _shutdown_remote_workers(self) -> None:
    """Stops remote worker handles best-effort, with a hard timeout."""
    pool = futures.ThreadPoolExecutor(max_workers=4)
    stops = {
        worker_id: pool.submit(
            self._remote_worker_handles_by_id[worker_id].submit, "stop"
        )
        for worker_id in sorted(self._remote_worker_infos)
    }
    for worker_id, fut in stops.items():
      try:
        fut.result(timeout=_STOP_TIMEOUT_S)
      except Exception as err:  # pylint: disable=broad-except
        logging.warning("Failed to stop remote worker %s: %r", worker_id, err)
    pool.shutdown(wait=False)

  def _create_engine(self) -> distributed_rl_engine.DistributedRLEngine:
    """Constructs a DistributedRLEngine from the registered role groups."""
    rollout_workers = self._get_actor_handles(datatypes.Role.ROLLOUT)
    actor_workers = self._get_actor_handles(datatypes.Role.ACTOR)
    critic_workers = self._get_actor_handles(datatypes.Role.CRITIC)
    reference_workers = self._get_actor_handles(datatypes.Role.REFERENCE)

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
      from tunix.experimental.weight_sync import weight_sync_coordinator

      handler = weight_sync_coordinator.create_default_handler(
          mode=self._weight_sync_mode
      )

      handle_to_id = {
          v: k for k, v in self._remote_worker_handles_by_id.items()
      }
      for role, handles in [
          (datatypes.Role.ACTOR, actor_workers),
          (datatypes.Role.ROLLOUT, rollout_workers),
      ]:
        for h in handles:
          w_id = handle_to_id.get(h, f"local-{role.value}-{id(h)}")
          info = self._remote_worker_infos.get(w_id) or datatypes.WorkerInfo(
              worker_id=w_id, roles=frozenset({role.value})
          )
          self.registry.register(weight_sync_coordinator.RemoteWorkerShim(h, info), override=True)  # pyrefly: ignore[bad-argument-type]

      coordinator = weight_sync_coordinator.WeightSyncCoordinator(
          registry=self.registry,
          handler=handler,
          controller_id="auto-coordinator",
      )

    return distributed_rl_engine.DistributedRLEngine(
        rollout_workers=rollout_workers,
        trainer_workers=trainer_workers,
        inference_workers=inference_workers,
        router=self.rollout_router,
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
      program: rl_program.RLProgram | None = None,
      max_steps: int = 1000,
  ) -> None:
    """Managed Program Submission: auto-wires Engine, Assembler, Queues & StandardRLProgram."""
    logging.info("Starting managed RL program run (max_steps=%d)...", max_steps)
    if self.engine is None:
      self.bring_up_workers()

    active_assembler = assembler or batch_assembly.SequencePackedBatchAssembler(
        group_size=getattr(algo, "group_size", 8),
        max_packed_len=getattr(algo, "max_packed_len", 8192),
    )
    metrics_logging_options = getattr(
        self.config, "metrics_logging_options", None
    )
    metrics_prefix = getattr(self.config, "metrics_prefix", "")
    active_program = program or rl_program.StandardRLProgram(
        dataset=dataset,
        max_steps=max_steps,
        algo=algo,
        reward_fns=reward_fns,
        assembler=active_assembler,
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
