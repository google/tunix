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

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from absl import logging
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import async_rl_program
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.orchestrator import health_monitor
from tunix.experimental.orchestrator import lifecycle
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.orchestrator import startup_validation
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.worker import abstract_worker
from tunix.experimental.worker import remote_execution


class ClusterOrchestrator:
  """Supervises cluster hardware, health monitoring, and program execution."""

  def __init__(
      self,
      config: Any = None,
      registry: worker_registry.WorkerRegistry | None = None,
      lifecycle_driver: lifecycle.LifecycleDriver | None = None,
      monitor: health_monitor.HealthMonitor | None = None,
  ):
    """Initializes ClusterOrchestrator."""
    self.config = config
    self.registry = registry or worker_registry.WorkerRegistry()
    self.lifecycle_driver = lifecycle_driver or lifecycle.LifecycleDriver(
        self.registry
    )
    self.monitor = monitor or health_monitor.HealthMonitor(self.registry)
    self.engine: distributed_rl_engine.DistributedRLEngine | None = None

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

  def unregister_worker(self, worker_id: str) -> None:
    """Unregisters a worker by its id."""
    self.registry.unregister(worker_id)

  def bring_up_workers(self, dummy_data: Any = None) -> None:
    """Brings up all registered workers through lifecycle initialization."""
    logging.info("Bringing up workers across cluster...")
    self.lifecycle_driver.bring_up(dummy_data)
    self.engine = self._create_engine()

  def shutdown(self) -> None:
    """Shuts down all workers and closes health monitoring resources."""
    logging.info("Shutting down ClusterOrchestrator...")
    self.monitor.close()
    self.lifecycle_driver.shutdown()

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

  def _as_actor_handle(self, worker: Any) -> remote_execution.ActorHandle:
    """Returns an ActorHandle for a registry member.

    WorkerRegistry stores lifecycle workers, while DistributedRLEngine routes
    compute through ActorHandle. Remote worker refs expose their underlying
    handle; local workers are wrapped in an in-process handle for examples/tests.
    """
    if isinstance(worker, remote_execution.ActorHandle):
      return worker
    actor_handle = getattr(worker, "actor_handle", None)
    if isinstance(actor_handle, remote_execution.ActorHandle):
      return actor_handle
    if callable(actor_handle):
      actor_handle = actor_handle()
    if isinstance(actor_handle, remote_execution.ActorHandle):
      return actor_handle
    return remote_execution.InProcessActorHandle(
        remote_execution.InProcessRemoteExecutionServer(worker)
    )

  def _create_engine(self) -> distributed_rl_engine.DistributedRLEngine:
    """Constructs a DistributedRLEngine from the registered role groups."""
    rollout_workers = [
        self._as_actor_handle(worker)
        for worker in self._get_role_members(datatypes.Role.ROLLOUT)
    ]
    actor_workers = [
        self._as_actor_handle(worker)
        for worker in self._get_role_members(datatypes.Role.ACTOR)
    ]
    critic_workers = [
        self._as_actor_handle(worker)
        for worker in self._get_role_members(datatypes.Role.CRITIC)
    ]
    reference_workers = [
        self._as_actor_handle(worker)
        for worker in self._get_role_members(datatypes.Role.REFERENCE)
    ]

    trainer_workers = {}
    if actor_workers:
      trainer_workers[datatypes.Role.ACTOR] = actor_workers[0]
    if critic_workers:
      trainer_workers[datatypes.Role.CRITIC] = critic_workers[0]

    inference_workers = {}
    if reference_workers:
      inference_workers[datatypes.Role.REFERENCE] = reference_workers[0]

    return distributed_rl_engine.DistributedRLEngine(
        rollout_workers=rollout_workers,
        trainer_workers=trainer_workers,
        inference_workers=inference_workers,
    )

  def run_program(
      self,
      program: rl_program.RLProgram,
      train_dataset: Iterable[Any] | None = None,
      num_steps: int | None = None,
      bring_up: bool = True,
      dummy_data: Any = None,
      **kwargs: Any,
  ) -> None:
    """Runs an RL program to completion under supervision."""
    if bring_up:
      self.bring_up_workers(dummy_data=dummy_data)

    self.monitor.poll()
    logging.info("ClusterOrchestrator executing program...")
    engine = self.engine or self._create_engine()

    program.run(engine=engine, train_dataset=train_dataset, num_steps=num_steps, **kwargs)

  def run(
      self,
      algo: algorithm_adapter.AlgorithmAdapter,
      dataset: Any,
      reward_fns: Sequence[Callable[..., Any]] | None = None,
      assembler: batch_assembly.BatchAssembler | None = None,
      program: async_rl_program.AsyncRLProgram | None = None,
      num_steps: int = 1000,
  ) -> None:
    """Managed Program Submission: auto-wires Engine, Assembler, Queues & StandardRLProgram."""
    if self.engine is None:
      self.bring_up_workers()

    active_assembler = assembler or batch_assembly.SequencePackedBatchAssembler(
        max_packed_len=getattr(algo, "max_packed_len", 8192)
    )
    active_program = program or async_rl_program.StandardRLProgram(
        dataset=dataset,
        algo=algo,
        reward_fns=reward_fns,
        assembler=active_assembler,
    )
    self.run_program(
        program=active_program,  # pyrefly: ignore[bad-argument-type]
        train_dataset=dataset,
        num_steps=num_steps,
        bring_up=False,
    )
