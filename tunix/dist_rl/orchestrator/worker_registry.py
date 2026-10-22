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

"""Worker registry for the orchestrator control plane.

The orchestrator addresses workers by role, not by identity: a role like
"trainer" or "inference" is the unit it schedules against. The `WorkerRegistry`
owns all cluster `ActorHandle`s and `WorkerInfo`s indexed by id and by declared
role, so a fused worker (e.g. one serving both "trainer" and "inference") joins
every matching role query. Role membership is snapshotted at registration time.
"""

from collections.abc import Collection, Mapping, Sequence
import dataclasses
import pickle
import threading
from typing import Any

from absl import logging
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.worker import abstract_worker
from tunix.dist_rl.worker import remote_execution

_LIVENESS_PROBE_TIMEOUT_S = 5.0

AdmissionStatus = datatypes.AdmissionStatus


@dataclasses.dataclass
class _WorkerEntry:
  """Internal registry record for a single registered worker."""

  info: datatypes.WorkerInfo
  handle: remote_execution.ActorHandle
  status: AdmissionStatus = AdmissionStatus.REGISTERED


class WorkerRegistry:
  """Single source of truth owning all cluster ActorHandles and WorkerInfos."""

  def __init__(self):
    self._lock = threading.Lock()
    self._condition = threading.Condition(self._lock)
    self._entries: dict[str, _WorkerEntry] = {}
    self._closed: bool = False

  @property
  def is_closed(self) -> bool:
    """Returns whether worker registration has been closed."""
    with self._lock:
      return self._closed

  def close_registration(self) -> None:
    """Closes the registry to any further worker registrations."""
    with self._condition:
      self._closed = True
      self._condition.notify_all()

  def wait_for_registered(
      self, exclude: Collection[remote_execution.ActorHandle]
  ) -> list[tuple[datatypes.WorkerInfo, remote_execution.ActorHandle]]:
    """Blocks until some `REGISTERED` worker's handle is not in `exclude`.

    Args:
      exclude: Handles the caller is already handling.

    Returns:
      Every `REGISTERED` worker, including those in `exclude`, sorted by
      worker_id; or an empty list once registration is closed.
    """
    with self._condition:
      while not self._closed:
        registered = [
            (entry.info, entry.handle)
            for entry in self._matching_entries_locked(
                statuses=(AdmissionStatus.REGISTERED,)
            )
        ]
        if any(handle not in exclude for _, handle in registered):
          return registered
        self._condition.wait()
      return []

  def _insert_locked(
      self,
      info: datatypes.WorkerInfo,
      handle: remote_execution.ActorHandle,
      override: bool,
  ) -> remote_execution.ActorHandle | None:
    worker_id = info.worker_id
    if not info.roles:
      raise ValueError(f"worker {worker_id!r} declares no roles")
    if self._closed:
      raise RuntimeError(
          f"Worker registration is closed; cannot register {worker_id!r}."
      )
    if worker_id in self._entries and not override:
      raise ValueError(f"duplicate worker_id: {worker_id!r}")

    old_entry = self._entries.get(worker_id)
    displaced_handle = (
        old_entry.handle
        if old_entry is not None and old_entry.handle is not handle
        else None
    )
    self._entries[worker_id] = _WorkerEntry(info=info, handle=handle)
    self._condition.notify_all()
    return displaced_handle

  def register(
      self, worker: abstract_worker.Worker, override: bool = False
  ) -> datatypes.WorkerInfo:
    """Wraps a local Worker in an InProcessActorHandle and registers it.

    Args:
      worker: The worker to register; its `info()` supplies id and roles.
      override: If true, silently overwrites an existing registration with the
        same id.

    Returns:
      The snapshotted `WorkerInfo`.

    Raises:
      ValueError: If the worker declares no roles, or its id is already
        registered (and `override` is False).
      RuntimeError: If worker registration has been closed.
    """
    info = worker.info()
    handle = remote_execution.InProcessActorHandle(
        remote_execution.InProcessRemoteExecutionServer(worker)
    )
    try:
      with self._condition:
        displaced_handle = self._insert_locked(info, handle, override=override)
    except BaseException:
      handle.sync_close()
      raise
    if displaced_handle is not None:
      displaced_handle.sync_close()
    return info

  def register_handle(
      self,
      worker_id: str,
      roles: Sequence[datatypes.Role | str],
      handle: remote_execution.ActorHandle,
      resources: Mapping[str, Any] | None = None,
      override: bool = False,
  ) -> datatypes.WorkerInfo:
    """Registers an ActorHandle under its declared id and roles."""
    if not isinstance(handle, remote_execution.ActorHandle):
      raise TypeError(
          "register_handle expects a remote_execution.ActorHandle, got "
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
    with self._condition:
      displaced_handle = self._insert_locked(info, handle, override=override)
    if displaced_handle is not None:
      displaced_handle.sync_close()
    logging.info(
        "Registered remote worker %r with roles %s.",
        worker_id,
        sorted(info.roles),
    )
    return info

  def _is_remote_worker_alive(self, address: str) -> bool:
    """Probes `address` via a short-timeout `heartbeat` RPC."""
    probe = remote_execution.ActorHandle.from_address(
        f"grpc://{address}",
        rpc_timeout_s=_LIVENESS_PROBE_TIMEOUT_S,
    )
    try:
      probe.submit("heartbeat")
      return True
    except Exception:  # pylint: disable=broad-exception-caught
      return False
    finally:
      probe.sync_close()

  def register_from_hostname(
      self,
      hostname: str,
      _: int,
      metadata: bytes,
      rpc_timeout_s: float = 1800.0,
  ) -> datatypes.WorkerInfo:
    """Registers a remote worker handle from a hostname and metadata.

    A rollout worker re-registering under an existing `worker_id` replaces the
    existing entry if it reuses the same address, or if the existing worker at a
    different address fails a `heartbeat` liveness probe.

    Args:
      hostname: Discovered peer hostname or IP address.
      _: Unused discovery server port supplied by the discovery callback.
      metadata: Pickled worker registration metadata dictionary.
      rpc_timeout_s: Per-RPC timeout in seconds for the created `ActorHandle`.

    Returns:
      The registered `WorkerInfo`.

    Raises:
      RuntimeError: If `service_type` is unknown or registration is closed.
      ValueError: If `worker_id` collides with a non-rollout worker or a live
        rollout worker at a different address.
    """
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

    resources: dict[str, Any] = {"remote": True, "address": service_address}
    if "trajectory_store_config" in md:
      resources["trajectory_store_config"] = md["trajectory_store_config"]
    info = datatypes.WorkerInfo(
        worker_id=worker_id,
        roles=frozenset({role.value}),
        resources=resources,
    )
    handle = remote_execution.ActorHandle.from_address(
        f"grpc://{service_address}",
        rpc_timeout_s=rpc_timeout_s,
    )
    displaced_handle: remote_execution.ActorHandle | None = None
    try:
      with self._condition:
        existing = self._entries.get(worker_id)
        # TODO(linchai): Support re-registration for non-rollout roles once
        # trainer/reference worker fault tolerance is implemented.
        if existing is not None and (
            role != datatypes.Role.ROLLOUT or existing.info.roles != info.roles
        ):
          raise ValueError(
              f"duplicate worker_id {worker_id!r} for role {role.value!r}"
              f" (existing roles {sorted(existing.info.roles)}); only rollout"
              " workers support re-registration"
          )
        # A remote worker at a different address may still be alive: probe it
        # outside the lock before replacing it.
        to_probe: tuple[remote_execution.ActorHandle, str] | None = None
        if existing is not None:
          existing_addr = existing.info.resources.get("address")
          if existing_addr and existing_addr != service_address:
            to_probe = (existing.handle, existing_addr)
        if to_probe is None:
          displaced_handle = self._insert_locked(info, handle, override=True)
      if to_probe is not None:
        existing_handle, existing_addr = to_probe
        if self._is_remote_worker_alive(existing_addr):
          raise ValueError(
              f"duplicate worker_id {worker_id!r}: existing rollout worker at"
              f" {existing_addr} is still alive when {service_address}"
              " attempted to register"
          )
        with self._condition:
          current = self._entries.get(worker_id)
          if current is not None and current.handle is not existing_handle:
            raise ValueError(
                f"duplicate worker_id {worker_id!r}: re-registered"
                f" concurrently while {service_address} was registering"
            )
          displaced_handle = self._insert_locked(info, handle, override=True)
    except BaseException:
      handle.sync_close()
      raise
    if displaced_handle is not None:
      displaced_handle.sync_close()
    logging.info(
        "Registered remote worker %r with roles %s.",
        worker_id,
        sorted(info.roles),
    )
    return info

  def wait_for_workers(
      self,
      min_workers: Mapping[datatypes.Role | str, int],
      timeout: float | None = None,
      *,
      statuses: Collection[AdmissionStatus] | None = None,
  ) -> None:
    """Waits for registered workers to meet the minimum required counts.

    Args:
      min_workers: Mapping from Role or role name to the minimum number of
        workers required.
      timeout: Maximum duration to wait in seconds before raising TimeoutError.
        If None, waits indefinitely until requirements are met.
      statuses: If set, only counts workers in one of these statuses.

    Raises:
      TimeoutError: If the required worker counts are not met within timeout.
    """
    with self._condition:

      def _counts() -> dict[datatypes.Role | str, int]:
        return {
            role: len(self._matching_entries_locked(role, statuses))
            for role in min_workers
        }

      def _satisfied() -> bool:
        current = _counts()
        return all(
            current[role] >= target_count
            for role, target_count in min_workers.items()
        )

      if not self._condition.wait_for(_satisfied, timeout=timeout):
        current_counts = _counts()
        raise TimeoutError(
            f"Timed out after {timeout}s waiting for workers. "
            f"Required: {min_workers}, Current: {current_counts}"
        )
      logging.info(
          "All required workers are ready. Current counts: %s",
          _counts(),
      )

  def update_resources(
      self,
      worker_id: str,
      resources: Mapping[str, Any],
      *,
      expected_handle: remote_execution.ActorHandle,
  ) -> bool:
    """Updates `info.resources` for `worker_id` under the registry lock if `expected_handle` matches."""
    with self._lock:
      entry = self._entries.get(worker_id)
      if entry is None or entry.handle is not expected_handle:
        return False
      entry.info.resources.update(resources)
      return True

  def status(self, worker_id: str) -> AdmissionStatus:
    """Returns the current `AdmissionStatus` for `worker_id`."""
    with self._lock:
      return self._entries[worker_id].status

  def set_status(
      self,
      worker_id: str,
      status: AdmissionStatus,
      *,
      expected_handle: remote_execution.ActorHandle,
      expected_status: AdmissionStatus,
  ) -> bool:
    """Moves `worker_id` from `expected_status` to `status`.

    Args:
      worker_id: The worker to update.
      status: The new status.
      expected_handle: The handle `worker_id` must still be registered with.
      expected_status: The status `worker_id` must currently be in.

    Returns:
      Whether the status changed: False if `worker_id` is no longer registered
      with `expected_handle` or is not in `expected_status`.
    """
    with self._condition:
      entry = self._entries.get(worker_id)
      if (
          entry is None
          or entry.handle is not expected_handle
          or entry.status is not expected_status
      ):
        return False
      entry.status = status
      self._condition.notify_all()
      return True

  def unregister(self, worker_id: str) -> None:
    """Removes a worker from the registry and closes its ActorHandle outside the lock."""
    with self._condition:
      handle = self._entries.pop(worker_id).handle
      self._condition.notify_all()
    handle.sync_close()

  def unregister_handle(
      self, handle: remote_execution.ActorHandle
  ) -> str | None:
    """Evicts the worker owning `handle` (if still registered) and closes it.

    Args:
      handle: The `ActorHandle` of the worker to evict.

    Returns:
      The evicted `worker_id`, or None if `handle` is not registered.
    """
    with self._condition:
      worker_id = next(
          (wid for wid, e in self._entries.items() if e.handle is handle), None
      )
      if worker_id is None:
        return None
      self._entries.pop(worker_id)
      self._condition.notify_all()
    handle.sync_close()
    return worker_id

  def close(self) -> None:
    """Closes all registered ActorHandles outside the lock and clears registry state."""
    with self._condition:
      handles = [self._entries[wid].handle for wid in sorted(self._entries)]
      self._entries.clear()
      self._closed = True
      self._condition.notify_all()

    errors: list[BaseException] = []
    for handle in handles:
      try:
        handle.sync_close()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        errors.append(exc)
    if errors:
      raise RuntimeError(
          f"Failed to close {len(errors)} ActorHandle(s): {errors!r}"
      ) from errors[0]

  def _matching_entries_locked(
      self,
      role: datatypes.Role | str | None = None,
      statuses: Collection[AdmissionStatus] | None = None,
  ) -> list[_WorkerEntry]:
    """Returns entries sorted by `worker_id` matching `role` and `statuses`."""
    allowed = None if statuses is None else frozenset(statuses)
    return [
        entry
        for _, entry in sorted(self._entries.items())
        if (role is None or role in entry.info.roles)
        and (allowed is None or entry.status in allowed)
    ]

  def _matching_entries(
      self,
      role: datatypes.Role | str | None = None,
      statuses: Collection[AdmissionStatus] | None = None,
  ) -> list[_WorkerEntry]:
    with self._lock:
      return self._matching_entries_locked(role, statuses)

  def handle(self, worker_id: str) -> remote_execution.ActorHandle:
    with self._lock:
      return self._entries[worker_id].handle

  def handles(
      self,
      role: datatypes.Role | str,
      *,
      statuses: Collection[AdmissionStatus] | None = (AdmissionStatus.ACTIVE,),
  ) -> list[remote_execution.ActorHandle]:
    """Returns the (possibly empty) list of ActorHandles serving `role`.

    Args:
      role: Role to look up.
      statuses: Statuses to include. Defaults to `ACTIVE` only, so data-plane
        callers never route to workers that are not yet admitted; pass `None` to
        include every registered worker (e.g. before bring-up).
    """
    return [entry.handle for entry in self._matching_entries(role, statuses)]

  def info(self, worker_id: str) -> datatypes.WorkerInfo:
    with self._lock:
      return self._entries[worker_id].info

  def infos(
      self, role: datatypes.Role | str | None = None
  ) -> list[datatypes.WorkerInfo]:
    """Returns WorkerInfos sorted by worker_id (all workers when role is None, or filtered by role)."""
    return [entry.info for entry in self._matching_entries(role)]

  def targets(
      self,
      role: datatypes.Role | str | None = None,
      *,
      statuses: Collection[AdmissionStatus] | None,
  ) -> list[tuple[datatypes.WorkerInfo, remote_execution.ActorHandle]]:
    """Returns `(WorkerInfo, ActorHandle)` pairs sorted by worker_id.

    Args:
      role: Role to filter by, or None for all roles.
      statuses: Statuses to include, or None for every registered worker.
    """
    return [
        (entry.info, entry.handle)
        for entry in self._matching_entries(role, statuses)
    ]

  def roles(self) -> set[str]:
    with self._lock:
      return set().union(*(e.info.roles for e in self._entries.values()))

  def worker_ids(self) -> list[str]:
    with self._lock:
      return sorted(self._entries)

  def __len__(self) -> int:
    with self._lock:
      return len(self._entries)

  def __contains__(self, worker_id: str) -> bool:
    with self._lock:
      return worker_id in self._entries
