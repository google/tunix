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

from collections.abc import Sequence
from concurrent import futures
import contextlib
import os
import pickle
import threading
import time
from typing import Any, Mapping
import uuid

from absl import logging
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.orchestrator import health_monitor
from tunix.experimental.orchestrator import lifecycle
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.orchestrator import startup_validation
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.trajectory import store as trajectory_store_lib
from tunix.experimental.weight_sync import weight_sync_coordinator
from tunix.experimental.worker import abstract_worker
from tunix.experimental.worker import remote_execution


_STOP_TIMEOUT_S = 60.0  # Timeout for stopping remote workers. 60 should not be touched for any healthy stop.
_JAX_CACHE_UPLOAD_TIMEOUT_S = 180.0  # Per-worker bound on a JAX cache upload RPC.


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
      jax_cache_config: Mapping[str, Any] | None = None,
      run_id: str | None = None,
      disable_weight_sync_timeouts: bool | None = None,
      max_concurrent_rollouts_per_worker: int | None = None,
      fault_tolerance_config: datatypes.RolloutFaultToleranceConfig | None = (
          None
      ),
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
      jax_cache_config: Optional JAX compilation cache synchronization config.
      run_id: Optional unique identifier for this orchestrator run. Generated
        automatically if omitted.
      disable_weight_sync_timeouts: When True, sets all weight-sync phase
        deadlines to infinity.
      max_concurrent_rollouts_per_worker: Optional cap on concurrent in-flight
        rollouts dispatched to any single rollout worker.
      fault_tolerance_config: Optional consolidated rollout fault-tolerance
        configuration.
    """
    self.config = config
    self._fault_tolerance_config = (
        fault_tolerance_config or datatypes.RolloutFaultToleranceConfig()
    ).with_overrides(
        max_in_flight_per_worker=max_concurrent_rollouts_per_worker,
    )
    self._max_concurrent_rollouts_per_worker = (
        self._fault_tolerance_config.max_in_flight_per_worker
    )
    self.registry = registry or worker_registry.WorkerRegistry()
    self.lifecycle_driver = lifecycle_driver or lifecycle.LifecycleDriver(
        self.registry
    )
    self.monitor = monitor or health_monitor.HealthMonitor(self.registry)
    self._lock = threading.RLock()
    self._brought_up = False
    self._warmup_data: Any = None
    self._bring_up_executor = futures.ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="worker_bringup"
    )
    self._pending_bring_up_futures: set[futures.Future[Any]] = set()
    self.engine: distributed_rl_engine.DistributedRLEngine | None = None
    self.registry.add_state_listener(self._on_registry_state_changed)
    mode = getattr(weight_sync_mode, "value", weight_sync_mode)
    self._weight_sync_mode = str(mode).lower() if mode is not None else None
    self._disable_weight_sync_timeouts = disable_weight_sync_timeouts
    self.jax_cache_config = dict(jax_cache_config or {})
    self._save_jax_cache = self.jax_cache_config.get(
        "save_jax_cache",
        os.getenv("SAVE_JAX_CACHE", "true").strip().lower() in ("1", "true", "yes"),
    )
    cfg_run_id = (
        trajectory_store_config.get("run_id")
        if trajectory_store_config is not None
        else None
    )
    self.run_id: str = (
        run_id
        or (str(cfg_run_id).strip() if cfg_run_id else "")
        or f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    )
    # The sole construction site for this process's Trajectory Store: one
    # ClusterOrchestrator exists per orchestrator process, so building it
    # here — once, in __init__ — is the whole guard. Its lifetime is meant
    # to span the process, not any one run() call, so it is public
    # (`self.trajectory_store`, not `_trajectory_store`) for a caller to
    # thread into whatever RLProgram it constructs; see StandardRLProgram's
    # `trajectory_store` argument.
    # TODO(sizhi): Wire active trajectory reads/writes between
    # orchestrator/program and rollout workers in follow-up CLs.
    self.trajectory_store_config: dict[str, Any] | None = None
    if trajectory_store_config is not None:
      self.trajectory_store_config = dict(trajectory_store_config)
      if (
          self.trajectory_store_config.get("enabled", False)
          and self.trajectory_store_config.get("backend") == "file"
          and not self.trajectory_store_config.get("run_id")
      ):
        self.trajectory_store_config["run_id"] = self.run_id
    self.trajectory_store = trajectory_store_lib.TrajectoryStore.from_config(
        self.trajectory_store_config
    )
    if self.trajectory_store is not None:
      self.trajectory_store_config = self.trajectory_store.to_config()
      # Logged so a config mismatch between this process and its workers is one
      # grep away.
      logging.info(
          "[trajectory-store] orchestrator built %s",
          self.trajectory_store_config,
      )

  @property
  def fault_tolerance_config(self) -> datatypes.RolloutFaultToleranceConfig:
    return self._fault_tolerance_config

  def _effective_worker_capacity(
      self, info: datatypes.WorkerInfo | None
  ) -> int | None:
    """Computes effective max concurrent rollouts for a worker."""
    worker_cap = None
    if info is not None and info.resources:
      raw_cap = info.resources.get("max_concurrency")
      if raw_cap is not None and int(raw_cap) > 0:
        worker_cap = int(raw_cap)
    default_cap = self._fault_tolerance_config.max_in_flight_per_worker
    if default_cap is not None and worker_cap is not None:
      return min(default_cap, worker_cap)
    if worker_cap is not None:
      return worker_cap
    return default_cap

  def _remote_shims(
      self, *, include_evicted: bool = False
  ) -> dict[str, weight_sync_coordinator.RemoteWorkerShim]:
    """Returns registered RemoteWorkerShims keyed by worker_id."""
    shims: dict[str, weight_sync_coordinator.RemoteWorkerShim] = {}
    for worker_id in self.registry.worker_ids(include_evicted=include_evicted):
      try:
        member = self.registry.get(worker_id)
      except KeyError:
        continue  # Unregistered concurrently.
      if isinstance(member, weight_sync_coordinator.RemoteWorkerShim):
        shims[worker_id] = member
    return shims

  def __enter__(self) -> "ClusterOrchestrator":
    """Interactive context manager bring-up."""
    self.bring_up_workers()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    self.shutdown()

  def register_worker_from_hostname(
      self,
      hostname: str,
      _: int,
      metadata: bytes,
      rpc_timeout_s: float = 1800.0,
  ) -> None:
    """Registers a remote worker handle from a hostname and metadata."""
    md = pickle.loads(metadata)

    # NB: this should align with workers
    service_type = md["service_type"]
    service_address = f"{hostname}:{md['service_port']}"
    worker_id = md["worker_id"]

    logging.info(
        "Discovered %s service (%s) at %s.",
        service_type,
        worker_id,
        service_address,
    )

    match service_type:
      case "trainer":
        role = datatypes.Role.ACTOR
      case "rollout":
        role = datatypes.Role.ROLLOUT
      case "inference":
        role = datatypes.Role.REFERENCE
      case _:
        raise RuntimeError(f"unknown service type {service_type}")

    resources: dict[str, Any] = {"address": service_address}
    if md.get("max_concurrency") is not None:
      resources["max_concurrency"] = int(md["max_concurrency"])
    self.register_worker_handle(
        worker_id=worker_id,
        roles=[role],
        handle=remote_execution.ActorHandle.from_address(
            f"grpc://{service_address}",
            rpc_timeout_s=rpc_timeout_s,
        ),
        resources=resources,
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
      *,
      override: bool = True,
  ) -> datatypes.WorkerInfo:
    """Registers a remote worker handle in the WorkerRegistry."""
    if not roles:
      raise ValueError(f"worker {worker_id!r} declares no roles")
    if not isinstance(handle, remote_execution.ActorHandle):
      raise TypeError(
          "register_worker_handle expects a remote_execution.ActorHandle, got "
          f"{type(handle)}"
      )
    role_names = frozenset(
        role.value if isinstance(role, datatypes.Role) else role
        for role in roles
    )
    info = datatypes.WorkerInfo(
        worker_id=worker_id,
        roles=role_names,
        resources={"remote": True, **dict(resources or {})},
    )
    shim = weight_sync_coordinator.RemoteWorkerShim(handle, info)
    with self._lock:
      if not override and worker_id in self.registry:
        raise ValueError(f"duplicate worker_id: {worker_id!r}")

      if worker_id in self.registry:
        old_member = self.registry.get(worker_id)
        if (
            isinstance(old_member, weight_sync_coordinator.RemoteWorkerShim)
            and self.engine is not None
            and datatypes.Role.ROLLOUT.value in old_member.info().roles
        ):
          self.engine.remove_rollout_worker(old_member.handle)

      initial_state = (
          worker_registry.MembershipState.INITIALIZING
          if self._brought_up
          else worker_registry.MembershipState.ACTIVE
      )
      self.registry.register(
          shim,  # pyrefly: ignore[bad-argument-type]
          override=override,
          state=initial_state,
      )
      logging.info(
          "Registered remote worker %r with roles %s.",
          worker_id,
          sorted(role_names),
      )

      if self._brought_up:
        incarnation = self.registry.incarnation(worker_id)
        fut = self._bring_up_executor.submit(
            self._bring_up_single_remote_worker,
            worker_id,
            info,
            handle,
            incarnation,
        )

        def _on_bring_up_done(done_fut: futures.Future[Any]) -> None:
          with self._lock:
            self._pending_bring_up_futures.discard(done_fut)

        self._pending_bring_up_futures.add(fut)
        fut.add_done_callback(_on_bring_up_done)
    return info

  def _bring_up_single_remote_worker(
      self,
      worker_id: str,
      info: datatypes.WorkerInfo,
      handle: remote_execution.ActorHandle,
      incarnation: int | None,
  ) -> None:
    """Brings up a single dynamically registered remote worker."""
    try:
      if (
          self.trajectory_store_config is not None
          and datatypes.Role.ROLLOUT.value in info.roles
      ):
        logging.info(
            "Configuring TrajectoryStore on dynamic remote rollout worker %s.",
            worker_id,
        )
        handle.submit(
            "with_trajectory_store_config", self.trajectory_store_config
        )
      logging.info("Initializing dynamic remote worker %s.", worker_id)
      handle.submit("initialize")
      logging.info("Compiling dynamic remote worker %s.", worker_id)
      handle.submit("compile", self._warmup_data)
      logging.info("Starting dynamic remote worker %s.", worker_id)
      handle.submit("start")

      with self._lock:
        if (
            worker_id not in self.registry
            or self.registry.incarnation(worker_id) != incarnation
        ):
          return
        target_state = worker_registry.MembershipState.ACTIVE
        if (
            datatypes.Role.ROLLOUT.value in info.roles
            and self.engine is not None
        ):
          coordinator = self.engine.weight_sync_coordinator
          require_sync = coordinator is not None and (
              self.engine.policy_version > 0
              or coordinator.last_committed_version is not None
          )
          if require_sync:
            target_state = worker_registry.MembershipState.PENDING_WEIGHT_SYNC
        self.registry.set_state(
            worker_id,
            target_state,
            expected_incarnation=incarnation,
        )
    except Exception as err:  # pylint: disable=broad-exception-caught
      logging.error(
          "Failed to bring up dynamic remote worker %s: %r", worker_id, err
      )
      with self._lock:
        if incarnation is not None and worker_id in self.registry:
          self.registry.evict(worker_id, expected_incarnation=incarnation)

  def wait_for_pending_bring_ups(self, timeout: float | None = None) -> None:
    """Waits for all in-flight dynamic worker bring-up tasks to complete."""
    with self._lock:
      futs = list(self._pending_bring_up_futures)
    if not futs:
      return
    _, not_done = futures.wait(futs, timeout=timeout)
    if not_done:
      raise TimeoutError(
          f"Timed out after {timeout}s waiting for {len(not_done)} worker"
          " bring-up(s)."
      )

  def unregister_worker(self, worker_id: str) -> None:
    """Unregisters a worker by its id."""
    with self._lock:
      if self.engine is not None and worker_id in self.registry:
        member = self.registry.get(worker_id)
        if (
            isinstance(member, weight_sync_coordinator.RemoteWorkerShim)
            and datatypes.Role.ROLLOUT.value in member.info().roles
        ):
          self.engine.remove_rollout_worker(member.handle)
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
    start_time = time.monotonic()
    while True:
      current_counts = {
          role: len(self.worker_handles(role)) for role in min_workers
      }
      if all(
          current_counts[role] >= target_count
          for role, target_count in min_workers.items()
      ):
        logging.info(
            "All required workers are ready. Current counts: %s",
            current_counts,
        )
        return

      if timeout is not None and (time.monotonic() - start_time) >= timeout:
        raise TimeoutError(
            f"Timed out after {timeout}s waiting for workers. "
            f"Required: {min_workers}, Current: {current_counts}"
        )

      sleep_duration = poll_interval_s
      if timeout is not None:
        remaining = timeout - (time.monotonic() - start_time)
        sleep_duration = min(poll_interval_s, max(0.0, remaining))

      time.sleep(sleep_duration)

  def worker_infos(self) -> list[datatypes.WorkerInfo]:
    """Returns local and remote worker metadata registered with the orchestrator."""
    return self.registry.infos()

  def worker_handles(
      self, role: datatypes.Role | str
  ) -> list[remote_execution.ActorHandle]:
    """Returns handles for all active workers (remote and local) under the given role."""
    return self._get_actor_handles(role)

  def sync_jax_cache(self) -> None:
    """Synchronizes JAX compilation cache across all workers to GCS."""
    if not self._save_jax_cache:
      return

    rollout_gcs_uri = (
        self.jax_cache_config.get("rollout_jax_cache_gcs_dir")
        or os.getenv("ROLLOUT_JAX_CACHE_GCS_DIR")
        or os.getenv("JAX_CACHE_GCS_DIR")
    )
    if not rollout_gcs_uri:
      return

    logging.info("Triggering JAX compilation cache synchronization to GCS...")
    shims = self._remote_shims()
    worker_ids = sorted(shims)
    local_workers = [
        w
        for w in self.registry.workers()
        if not isinstance(w, weight_sync_coordinator.RemoteWorkerShim)
    ]
    if not worker_ids:
      for worker in local_workers:
        try:
          worker.upload_jax_cache(gcs_uri=rollout_gcs_uri)
        except Exception as err:  # pylint: disable=broad-except
          logging.warning("Failed to sync JAX cache on local worker: %r", err)
      return

    rollout_worker_ids = [
        w_id
        for w_id in worker_ids
        if datatypes.Role.ROLLOUT.value in shims[w_id].info().roles
    ]
    if not rollout_worker_ids:
      return

    primary_worker_id = rollout_worker_ids[0]
    logging.info(
        "Triggering JAX compilation cache synchronization to GCS (%s) from rollout worker %s...",
        rollout_gcs_uri,
        primary_worker_id,
    )

    handle = shims[primary_worker_id].handle
    outcome: futures.Future[bool] = futures.Future()

    def _run() -> None:
      try:
        outcome.set_result(
            handle.submit("upload_jax_cache", gcs_uri=rollout_gcs_uri)
        )
      except Exception as err:  # pylint: disable=broad-except
        outcome.set_exception(err)

    sync_timeout_s = float(
        self.jax_cache_config.get("sync_timeout_s", _JAX_CACHE_UPLOAD_TIMEOUT_S)
    )
    # Daemon thread: unlike ThreadPoolExecutor workers it is not joined at
    # interpreter exit, so an abandoned hung RPC (bounded only by the RPC
    # deadline, which can be hours) cannot block process exit.
    threading.Thread(
        target=_run,
        name=f"jax-cache-upload-{primary_worker_id}",
        daemon=True,
    ).start()
    try:
      uploaded = outcome.result(timeout=sync_timeout_s)
    except futures.TimeoutError:
      logging.warning(
          "JAX cache upload on worker %s timed out after %.0fs; abandoning it.",
          primary_worker_id,
          sync_timeout_s,
      )
      return
    except Exception as err:  # pylint: disable=broad-except
      logging.warning(
          "Failed to sync JAX cache on worker %s: %r", primary_worker_id, err
      )
      return
    if uploaded is False:
      logging.warning(
          "Worker %s reported a failed JAX cache upload.", primary_worker_id
      )
      return
    logging.info("Worker %s JAX cache upload finished.", primary_worker_id)

  def bring_up_workers(self, dummy_data: Any = None) -> None:
    """Brings up all registered workers through lifecycle initialization."""
    logging.info(
        "Bringing up %d registered worker(s)...",
        len(self.worker_infos()),
    )
    self._warmup_data = dummy_data
    if self.trajectory_store_config is not None:
      for worker in self._get_role_members(datatypes.Role.ROLLOUT):
        if not isinstance(
            worker, weight_sync_coordinator.RemoteWorkerShim
        ) and hasattr(worker, "with_trajectory_store_config"):
          worker.with_trajectory_store_config(self.trajectory_store_config)
    self.lifecycle_driver.bring_up(dummy_data)
    self._bring_up_remote_workers(dummy_data)
    self.sync_jax_cache()
    with self._lock:
      self.engine = self._create_engine()
      self._brought_up = True
    logging.info("All workers brought up successfully.")

  def shutdown(self) -> None:
    """Shuts down all workers and closes health monitoring resources."""
    logging.info("Shutting down all workers...")
    with contextlib.ExitStack() as stack:
      # Registered in reverse order of execution (LIFO) so that every stage
      # runs even if a preceding stage raises an exception.
      stack.callback(self._bring_up_executor.shutdown, wait=False)
      if self.trajectory_store is not None:
        stack.callback(self.trajectory_store.close)
      stack.callback(self.lifecycle_driver.shutdown)
      stack.callback(self._shutdown_remote_workers)
      stack.callback(self.monitor.close)
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
    return list(members)

  def _get_actor_handles(
      self, role: datatypes.Role | str
  ) -> list[remote_execution.ActorHandle]:
    """Returns ActorHandles for all active workers serving `role`."""
    remote_handles: list[remote_execution.ActorHandle] = []
    local_handles: list[remote_execution.ActorHandle] = []
    for member in self._get_role_members(role):
      if isinstance(member, weight_sync_coordinator.RemoteWorkerShim):
        remote_handles.append(member.handle)
      else:
        local_handles.append(
            remote_execution.InProcessActorHandle(
                remote_execution.InProcessRemoteExecutionServer(member)
            )
        )
    return remote_handles + local_handles

  def _bring_up_remote_workers(self, dummy_data: Any = None) -> None:
    """Runs lifecycle hooks on remote worker handles registered directly."""
    with self._lock:
      shims = self._remote_shims()
      self._brought_up = True
    worker_ids = sorted(shims)
    if not worker_ids:
      return
    try:
      with futures.ThreadPoolExecutor(max_workers=len(worker_ids)) as pool:
        if self.trajectory_store_config is not None:
          def _cfg_store(wid: str) -> None:
            if datatypes.Role.ROLLOUT.value in shims[wid].info().roles:
              logging.info(
                  "Configuring TrajectoryStore on remote rollout worker %s.",
                  wid,
              )
              shims[wid].handle.submit(
                  "with_trajectory_store_config", self.trajectory_store_config
              )

          list(pool.map(_cfg_store, worker_ids))

        def _init_worker(wid: str) -> None:
          logging.info("Initializing remote worker %s.", wid)
          shims[wid].handle.submit("initialize")

        list(pool.map(_init_worker, worker_ids))

        def _compile_worker(wid: str) -> None:
          logging.info("Compiling remote worker %s.", wid)
          shims[wid].handle.submit("compile", dummy_data)

        list(pool.map(_compile_worker, worker_ids))

        def _start_worker(wid: str) -> None:
          logging.info("Starting remote worker %s.", wid)
          shims[wid].handle.submit("start")

        list(pool.map(_start_worker, worker_ids))
    except BaseException:
      with self._lock:
        self._brought_up = False
      raise

  def _shutdown_remote_workers(self) -> None:
    """Stops remote worker handles best-effort, with a hard timeout."""
    shims = self._remote_shims(include_evicted=True)
    if not shims:
      return
    pool = futures.ThreadPoolExecutor(max_workers=4)
    stops = {
        worker_id: pool.submit(shims[worker_id].handle.submit, "stop")
        for worker_id in sorted(shims)
    }
    for worker_id, fut in stops.items():
      try:
        fut.result(timeout=_STOP_TIMEOUT_S)
      except Exception as err:  # pylint: disable=broad-except
        logging.warning("Failed to stop remote worker %s: %r", worker_id, err)
    pool.shutdown(wait=False)

  def _on_registry_state_changed(
      self,
      worker_id: str,
      state: worker_registry.MembershipState,
  ) -> None:
    """Syncs WorkerRegistry state transitions to DistributedRLEngine."""
    with self._lock:
      if (
          not self._brought_up
          or self.engine is None
          or worker_id not in self.registry
          or self.registry.state(worker_id) != state
      ):
        return
      member = self.registry.get(worker_id)
      if not isinstance(member, weight_sync_coordinator.RemoteWorkerShim):
        return
      info = member.info()
      if datatypes.Role.ROLLOUT.value not in info.roles:
        return
      if state == worker_registry.MembershipState.ACTIVE:
        self.engine.add_rollout_worker(
            member.handle,
            max_in_flight=self._effective_worker_capacity(info),
        )
      elif state == worker_registry.MembershipState.EVICTED:
        self.engine.remove_rollout_worker(member.handle)

  def _on_engine_worker_evicted(
      self,
      handle: remote_execution.ActorHandle,
      exc: BaseException | None = None,
  ) -> None:
    """Callback invoked when DistributedRLEngine evicts a failed worker."""
    del exc
    with self._lock:
      wid = getattr(handle, "worker_id", None)
      if wid is not None and wid in self.registry:
        member = self.registry.get(wid)
        if (
            isinstance(member, weight_sync_coordinator.RemoteWorkerShim)
            and member.handle is handle
        ):
          if (
              self.registry.state(wid)
              != worker_registry.MembershipState.EVICTED
          ):
            self.registry.evict(wid)
          return
      for wid, shim in self._remote_shims().items():
        if shim.handle is handle:
          self.registry.evict(wid)

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
      handler = weight_sync_coordinator.create_default_handler(
          mode=self._weight_sync_mode
      )

      existing_handles = {
          shim.handle
          for shim in self._remote_shims(include_evicted=True).values()
      }
      for role, handles in [
          (datatypes.Role.ACTOR, actor_workers),
          (datatypes.Role.ROLLOUT, rollout_workers),
      ]:
        for h in handles:
          if h not in existing_handles:
            w_id = f"local-{role.value}-{id(h)}"
            info = datatypes.WorkerInfo(
                worker_id=w_id, roles=frozenset({role.value})
            )
            self.registry.register(
                weight_sync_coordinator.RemoteWorkerShim(h, info),  # pyrefly: ignore[bad-argument-type]
                override=True,
            )

      coordinator = weight_sync_coordinator.WeightSyncCoordinator(
          registry=self.registry,
          handler=handler,
          controller_id="auto-coordinator",
          disable_timeouts=self._disable_weight_sync_timeouts,
      )

    rollout_worker_capacities: dict[remote_execution.ActorHandle, int] = {}
    for shim in self._remote_shims().values():
      info = shim.info()
      if datatypes.Role.ROLLOUT.value in info.roles:
        cap = self._effective_worker_capacity(info)
        if cap is not None:
          rollout_worker_capacities[shim.handle] = cap

    return distributed_rl_engine.DistributedRLEngine(
        rollout_workers=rollout_workers,
        trainer_workers=trainer_workers,
        inference_workers=inference_workers,
        weight_sync_coordinator=coordinator,
        on_worker_evicted=self._on_engine_worker_evicted,
        rollout_worker_capacities=rollout_worker_capacities or None,
        fault_tolerance_config=self._fault_tolerance_config,
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
