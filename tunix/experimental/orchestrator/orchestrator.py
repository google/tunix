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

"""Cluster Orchestrator (Layer 5) for managing distributed RL workflows.

Supervises sub-components (WorkerRegistry, HealthMonitor, LifecycleDriver, and
StartupValidator) and executes top-level RL programs without fault tolerance.
"""

from collections.abc import Iterable
from typing import Any

from absl import logging
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.orchestrator import health_monitor
from tunix.experimental.orchestrator import lifecycle
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.orchestrator import startup_validation
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.worker import abstract_worker


class ClusterOrchestrator:
  """Top-level Cluster Orchestrator (Layer 5) for distributed RL.

  Glues together:
  - WorkerRegistry: Indexes worker handles by role and id.
  - LifecycleDriver: Coordinates initialization, compilation, and shutdown.
  - HealthMonitor: Checks heartbeat reports across registered workers.
  - StartupValidator: Validates runtime geometries prior to training.
  """

  def __init__(
      self,
      registry: worker_registry.WorkerRegistry | None = None,
      lifecycle_driver: lifecycle.LifecycleDriver | None = None,
      monitor: health_monitor.HealthMonitor | None = None,
  ):
    """Initializes ClusterOrchestrator.

    Args:
      registry: Optional WorkerRegistry instance.
      lifecycle_driver: Optional LifecycleDriver instance.
      monitor: Optional HealthMonitor instance.
    """
    self.registry = registry or worker_registry.WorkerRegistry()
    self.lifecycle_driver = lifecycle_driver or lifecycle.LifecycleDriver(
        self.registry
    )
    self.monitor = monitor or health_monitor.HealthMonitor(self.registry)

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

  def _get_role_members(self, role: Any) -> list[Any]:
    members = self.registry.group(role).members()
    if not members and isinstance(role, datatypes.Role):
      members = self.registry.group(role.value).members()
    return members

  def create_engine(
      self,
  ) -> distributed_rl_engine.DistributedRLEngine:
    """Constructs a DistributedRLEngine from the currently registered workers.

    Returns:
      A DistributedRLEngine routing compute to registered role groups.
    """
    rollout_workers = self._get_role_members(datatypes.Role.ROLLOUT)
    actor_workers = self._get_role_members(datatypes.Role.ACTOR)
    critic_workers = self._get_role_members(datatypes.Role.CRITIC)
    reference_workers = self._get_role_members(datatypes.Role.REFERENCE)

    trainer_workers = {}
    if actor_workers:
      trainer_workers[datatypes.Role.ACTOR] = actor_workers[0]
    if critic_workers:
      trainer_workers[datatypes.Role.CRITIC] = critic_workers[0]

    inference_worker = reference_workers[0] if reference_workers else None

    return distributed_rl_engine.DistributedRLEngine(
        rollout_workers=rollout_workers,
        trainer_workers=trainer_workers,
        inference_worker=inference_worker,
    )

  def run_program(
      self,
      program: rl_program.RLProgram,
      train_dataset: Iterable[list[str] | list[list[dict[str, str]]]],
      num_steps: int | None = None,
      bring_up: bool = True,
      dummy_data: Any = None,
      **kwargs: Any,
  ) -> None:
    """Runs an RL program to completion without fault-tolerance monitoring.

    Args:
      program: The RLProgram (Layer 4) instance to execute.
      train_dataset: An iterable yielding batches of prompts for training.
      num_steps: Optional maximum number of training steps to execute.
      bring_up: Whether to invoke bring_up_workers prior to starting the loop.
      dummy_data: Optional dummy data passed to bring_up_workers for
        compilation.
      **kwargs: Additional keyword arguments forwarded to program.run.
    """
    if bring_up:
      self.bring_up_workers(dummy_data=dummy_data)

    # Perform an initial health check across all registered workers.
    self.monitor.poll()

    logging.info("ClusterOrchestrator executing program...")
    program.run(train_dataset=train_dataset, num_steps=num_steps, **kwargs)
