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
Provides supervised RL program execution (`run`).
"""

import contextlib
from typing import Any, Mapping

from absl import logging
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import timeouts
from tunix.dist_rl.orchestrator import distributed_rl_engine
from tunix.dist_rl.orchestrator import health_monitor
from tunix.dist_rl.orchestrator import lifecycle
from tunix.dist_rl.orchestrator import rl_program
from tunix.dist_rl.orchestrator import startup_validation
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.trajectory import store as trajectory_store_lib
from tunix.dist_rl.weight_sync import weight_sync_coordinator


class ClusterOrchestrator:
  """Supervises cluster hardware, health monitoring, and program execution."""

  def __init__(
      self,
      config: Any = None,
      registry: worker_registry.WorkerRegistry | None = None,
      lifecycle_driver: lifecycle.LifecycleDriver | None = None,
      monitor: health_monitor.HealthMonitor | None = None,
      weight_sync_mode: str | None = None,
      trajectory_store_config: Mapping[str, Any] | None = None,
      weight_sync_timeouts: weight_sync_coordinator.PhaseTimeouts | None = None,
  ):
    """Initializes ClusterOrchestrator.

    Args:
      config: Orchestrator configuration.
      registry: Worker registry to use; one is created if omitted.
      lifecycle_driver: Lifecycle driver to use; one is created if omitted.
      monitor: Health monitor to use; one is created if omitted.
      weight_sync_mode: Weight sync mode, if any.
      trajectory_store_config: Trajectory Store configuration for this process,
        or None to run without a store. See `store.TrajectoryStore.from_config`.
        Pass the same config to every process in the run: for the file backend
        it is the shared root_dir and run_id that will make the workers' writes
        visible to this process's reads once read/write wiring is connected.
      weight_sync_timeouts: Per-phase deadlines forwarded to
        `WeightSyncCoordinator`, or None to use `PhaseTimeouts()` defaults.
    """
    self.config = config
    self.registry = (
        registry if registry is not None else worker_registry.WorkerRegistry()
    )
    self.lifecycle_driver = lifecycle_driver or lifecycle.LifecycleDriver(
        self.registry
    )
    self.monitor = monitor or health_monitor.HealthMonitor(self.registry)
    self.engine: distributed_rl_engine.DistributedRLEngine | None = None
    mode = getattr(weight_sync_mode, "value", weight_sync_mode)
    self._weight_sync_mode = str(mode).lower() if mode is not None else None
    self._weight_sync_timeouts = weight_sync_timeouts
    # The sole construction site for this process's Trajectory Store: one
    # ClusterOrchestrator exists per orchestrator process, so building it
    # here — once, in __init__ — is the whole guard. Its lifetime is meant
    # to span the process, not any one run() call, so it is public
    # (`self.trajectory_store`, not `_trajectory_store`) for a caller to
    # thread into whatever RLProgram it constructs; see StandardRLProgram's
    # `trajectory_store` argument.
    # TODO(sizhi): Wire active trajectory reads/writes between
    # orchestrator/program and rollout workers in follow-up CLs.
    self.trajectory_store = (
        trajectory_store_lib.TunixTrajectoryStore.from_config(
            trajectory_store_config
        )
    )
    if self.trajectory_store is not None:
      # Logged so a config mismatch between this process and its workers is one
      # grep away.
      logging.info(
          "[trajectory-store] orchestrator built %s",
          self.trajectory_store.to_redacted_config(),
      )

  def __enter__(self) -> "ClusterOrchestrator":
    """Interactive context manager bring-up."""
    self.bring_up_workers()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.shutdown()

  def bring_up_workers(self, dummy_data: Any = None) -> None:
    """Brings up all registered workers through lifecycle initialization."""
    logging.info(
        "Bringing up %d registered worker(s)...",
        len(self.registry),
    )
    self.lifecycle_driver.bring_up(dummy_data)
    self._verify_rollout_trajectory_store_configs()
    self._verify_rollout_weight_sync_timeouts()
    self.engine = self._create_engine()
    logging.info("All workers brought up successfully.")

  def _verify_rollout_trajectory_store_configs(self) -> None:
    """Verifies that all registered rollout workers match the orchestrator's TrajectoryStore config."""
    expected_config = (
        self.trajectory_store.to_config()
        if self.trajectory_store is not None
        else None
    )
    for info in self.registry.infos(datatypes.Role.ROLLOUT):
      worker_config = info.resources.get("trajectory_store_config")
      if worker_config != expected_config:
        raise ValueError(
            f"Rollout worker {info.worker_id!r} trajectory_store_config "
            f"{worker_config!r} does not match orchestrator "
            f"trajectory_store_config {expected_config!r}."
        )

  def _verify_rollout_weight_sync_timeouts(self) -> None:
    """Verifies that Coordinator `PhaseTimeouts.pre` covers each rollout worker's soft drain window."""
    if self._weight_sync_mode in (None, "none"):
      return
    timeouts_cfg = (
        self._weight_sync_timeouts or weight_sync_coordinator.PhaseTimeouts()
    )
    for info in self.registry.infos(datatypes.Role.ROLLOUT):
      for key in ("partial_rollout", "drain_timeout_s"):
        if key not in info.resources:
          raise ValueError(
              f"Rollout worker {info.worker_id!r} is missing required resource"
              f" key {key!r} in info.resources (present keys:"
              f" {sorted(info.resources)!r})."
          )
      if info.resources["partial_rollout"]:
        continue
      drain_timeout_s = info.resources["drain_timeout_s"]
      required_pre_s = timeouts.pre_weight_sync_timeout_for(drain_timeout_s)
      if required_pre_s > timeouts_cfg.pre:
        raise ValueError(
            f"Rollout worker {info.worker_id!r} has"
            f" drain_timeout_s={drain_timeout_s} (partial_rollout=False),"
            f" which requires PhaseTimeouts.pre >= {required_pre_s}s, but"
            f" orchestrator weight_sync_timeouts.pre is {timeouts_cfg.pre}s."
        )

  def shutdown(self) -> None:
    """Shuts down all workers, closes handles, and closes health monitoring resources."""
    logging.info("Shutting down all workers...")
    with contextlib.ExitStack() as stack:
      # Registered in reverse order of execution (LIFO) so that every stage
      # runs even if a preceding stage raises an exception.
      if self.trajectory_store is not None:
        stack.callback(self.trajectory_store.close)
      stack.callback(self.registry.close)
      stack.callback(self.lifecycle_driver.shutdown)
      stack.callback(self.monitor.close)
    logging.info("Shutdown complete.")

  def validate_startup(self, alg_config: Any, training_config: Any) -> None:
    """Validates cluster geometry against configurations."""
    startup_validation.validate_startup(
        self.registry, alg_config, training_config
    )

  def _create_engine(self) -> distributed_rl_engine.DistributedRLEngine:
    """Constructs a DistributedRLEngine from the registered role groups."""
    coordinator = None
    if self._weight_sync_mode not in (None, "none"):
      handler = weight_sync_coordinator.create_default_handler(
          mode=self._weight_sync_mode
      )
      coordinator = weight_sync_coordinator.WeightSyncCoordinator(
          registry=self.registry,
          handler=handler,
          controller_id="auto-coordinator",
          timeouts=self._weight_sync_timeouts,
      )

    return distributed_rl_engine.DistributedRLEngine(
        registry=self.registry,
        weight_sync_coordinator=coordinator,
    )

  def run(
      self,
      program: rl_program.RLProgram,
      bring_up: bool = True,
      dummy_data: Any = None,
      **kwargs: Any,
  ) -> None:
    """Runs an RL program to completion under supervision.

    Args:
      program: The RL program instance to execute.
      bring_up: Whether to bring up registered workers before execution.
      dummy_data: Optional initialization data passed to worker compilation.
      **kwargs: Additional keyword arguments forwarded to program.run.
    """
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
