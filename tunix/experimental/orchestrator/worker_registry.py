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

"""Worker registry and role groups for the orchestrator control plane.

The orchestrator addresses workers by role, not by identity: a group like
"trainer" or "inference" is the unit it schedules against. The `WorkerRegistry`
indexes registered `ActorHandle`s (both `GrpcRemoteActorHandle` and
`InProcessActorHandle`) by worker id and by the roles each declares.
Role membership is snapshotted from `WorkerInfo` at registration time, so
grouping stays stable even if a worker's live state changes.
"""

import collections
from collections.abc import Iterator, Sequence
import enum
import pickle
import threading
import time
from typing import Any

from absl import logging
from tunix.experimental.common import datatypes
from tunix.experimental.worker import abstract_worker
from tunix.experimental.worker import remote_execution

# Allow InProcessActorHandle to transparently delegate attribute lookups
# (e.g. initialize, compile, start, stop, heartbeat) to its bound instance
# for backwards compatibility with unmigrated components.
if not hasattr(remote_execution.InProcessActorHandle, "__getattr__"):

  def _in_process_getattr(self, name: str) -> Any:
    if name.startswith("_"):
      raise AttributeError(name)
    server = getattr(self, "server", None)
    if server is not None:
      instance = getattr(server, "bound_instance", None)
      if instance is not None and hasattr(instance, name):
        return getattr(instance, name)
    raise AttributeError(
        f"{type(self).__name__!r} object has no attribute {name!r}"
    )

  remote_execution.InProcessActorHandle.__getattr__ = _in_process_getattr

if not hasattr(remote_execution.ActorHandle, "stop"):
  remote_execution.ActorHandle.stop = lambda self: self.submit("stop")
if not hasattr(remote_execution.ActorHandle, "heartbeat"):
  remote_execution.ActorHandle.heartbeat = lambda self: self.submit("heartbeat")


def _normalize_role(role: datatypes.Role | str) -> str:
  """Normalizes a Role enum or string representation to a plain string."""
  if isinstance(role, enum.Enum):
    return str(role.value)
  return str(role)


class WorkerGroup:
  """An ordered, immutable view of the workers serving a single role."""

  def __init__(
      self,
      role: str,
      handles: Sequence[remote_execution.ActorHandle] | None = None,
      *,
      members: Sequence[Any] | None = None,
      infos: Sequence[datatypes.WorkerInfo] | None = None,
      worker_ids: Sequence[str] | None = None,
  ):
    self._role = role
    self._handles = list(handles) if handles is not None else []
    if members is not None:
      self._members = list(members)
    else:
      self._members = list(self._handles)
    self._infos = list(infos) if infos is not None else []
    self._worker_ids = list(worker_ids) if worker_ids is not None else []

  @property
  def role(self) -> str:
    return self._role

  def handles(self) -> list[remote_execution.ActorHandle]:
    """Returns the list of ActorHandles in this group."""
    return list(self._handles)

  def members(self) -> list[Any]:
    """Returns the registered worker members (or handles if no shims)."""
    return list(self._members)

  def infos(self) -> list[datatypes.WorkerInfo]:
    """Returns the snapshotted WorkerInfos in this group."""
    return list(self._infos)

  def worker_ids(self) -> list[str]:
    """Returns the list of worker ids in this group."""
    return list(self._worker_ids)

  def is_empty(self) -> bool:
    return not self._handles

  def __len__(self) -> int:
    return len(self._handles)

  def __iter__(self) -> Iterator[Any]:
    return iter(self._members)

  def __getitem__(self, index: int) -> Any:
    return self._members[index]


class WorkerRegistry:
  """Indexes worker handles by id and by declared role.

  Registration snapshots each worker's `WorkerInfo`; lookups return live
  `ActorHandle`s. Worker ids must be unique.
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._handles: dict[str, remote_execution.ActorHandle] = {}
    self._infos: dict[str, datatypes.WorkerInfo] = {}
    self._role_to_ids: dict[str, set[str]] = collections.defaultdict(set)
    self._shims: dict[str, Any] = {}

  def register_handle(
      self,
      worker_id: str,
      roles: Sequence[datatypes.Role | str] | set[str] | frozenset[str],
      handle: remote_execution.ActorHandle,
      resources: dict[str, Any] | None = None,
      override: bool = False,
  ) -> datatypes.WorkerInfo:
    """Registers an ActorHandle under the specified worker_id and roles.

    Args:
      worker_id: Unique worker identifier.
      roles: Sequence or set of roles served by this worker.
      handle: The ActorHandle (remote or in-process) for this worker.
      resources: Optional resources dictionary (e.g. host/port metadata).
      override: If true, silently overwrites an existing registration with the
        same id.

    Returns:
      The snapshotted WorkerInfo.

    Raises:
      TypeError: If handle is not an ActorHandle.
      ValueError: If roles is empty, or worker_id is duplicate (and override is
        False).
    """
    if not isinstance(handle, remote_execution.ActorHandle):
      raise TypeError(
          "register_handle expects a remote_execution.ActorHandle, got "
          f"{type(handle)}"
      )
    if not roles:
      raise ValueError(f"worker {worker_id!r} declares no roles")

    role_names = frozenset(_normalize_role(r) for r in roles)

    merged_resources = dict(resources or {})
    if "remote" not in merged_resources and not isinstance(
        handle, remote_execution.InProcessActorHandle
    ):
      merged_resources["remote"] = True

    with self._lock:
      if worker_id in self._handles and not override:
        raise ValueError(f"duplicate worker_id: {worker_id!r}")

      # If overriding, clean up old role indexing for this worker
      if worker_id in self._infos:
        old_info = self._infos[worker_id]
        for role in old_info.roles:
          if role in self._role_to_ids:
            self._role_to_ids[role].discard(worker_id)
            if not self._role_to_ids[role]:
              del self._role_to_ids[role]

      info = datatypes.WorkerInfo(
          worker_id=worker_id,
          roles=role_names,
          resources=merged_resources,
      )
      self._handles[worker_id] = handle
      self._infos[worker_id] = info
      for role in role_names:
        self._role_to_ids[role].add(worker_id)

    logging.info(
        "Registered worker %r with roles %s.",
        worker_id,
        sorted(role_names),
    )
    return info

  def register_from_hostname(
      self,
      hostname: str,
      port: int,
      metadata: bytes,
      rpc_timeout_s: float = 1800.0,
      override: bool = False,
  ) -> datatypes.WorkerInfo:
    """Discovers and registers a remote worker from hostname and pickled metadata.

    Args:
      hostname: Hostname or IP address of the worker.
      port: Default port (used if not specified in metadata).
      metadata: Pickled metadata bytes containing service_type, worker_id, and
        optionally service_port.
      rpc_timeout_s: Timeout in seconds for RPC invocations on the handle.
      override: If true, silently overwrites an existing registration with the
        same id.

    Returns:
      The snapshotted WorkerInfo.

    Raises:
      RuntimeError: If service_type in metadata is unknown.
    """
    md = pickle.loads(metadata)  # pylint: disable=g-unsafe-pickle-load

    service_type = md["service_type"]
    service_port = md.get("service_port", port)
    service_address = f"{hostname}:{service_port}"
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

    handle = remote_execution.ActorHandle.from_address(
        f"grpc://{service_address}",
        rpc_timeout_s=rpc_timeout_s,
    )
    return self.register_handle(
        worker_id=worker_id,
        roles=[role],
        handle=handle,
        resources={"address": service_address},
        override=override,
    )

  def register_worker(
      self,
      worker: Any,
      override: bool = False,
  ) -> datatypes.WorkerInfo:
    """Registers a local Worker by wrapping it in an InProcessActorHandle.

    Args:
      worker: The local worker to register; its info() supplies id and roles.
      override: If true, silently overwrites an existing registration with the
        same id.

    Returns:
      The snapshotted WorkerInfo.

    Raises:
      ValueError: If the worker declares no roles, or its id is already
        registered (and override is False).
    """
    info = worker.info()
    worker_id = info.worker_id
    if not info.roles:
      raise ValueError(f"worker {worker_id!r} declares no roles")
    existing_handle = getattr(worker, "_handle", None) or getattr(
        worker, "handle", None
    )
    if isinstance(existing_handle, remote_execution.ActorHandle):
      handle = existing_handle
    else:
      handle = remote_execution.InProcessActorHandle(
          remote_execution.InProcessRemoteExecutionServer(worker)
      )
    reg_info = self.register_handle(
        worker_id=worker_id,
        roles=info.roles,
        handle=handle,
        resources=info.resources,
        override=override,
    )
    if type(worker).__name__ == "RemoteWorkerShim":
      with self._lock:
        self._shims[worker_id] = worker
    return reg_info

  def register(
      self,
      worker: Any,
      override: bool = False,
  ) -> datatypes.WorkerInfo:
    """Backwards-compatible alias for register_worker."""
    return self.register_worker(worker, override=override)

  def unregister(self, worker_id: str) -> None:
    """Removes a worker (and its role memberships) from the registry."""
    with self._lock:
      if worker_id not in self._handles:
        raise KeyError(worker_id)
      info = self._infos.pop(worker_id)
      del self._handles[worker_id]
      self._shims.pop(worker_id, None)
      for role in info.roles:
        members = self._role_to_ids.get(role)
        if members is not None:
          members.discard(worker_id)
          if not members:
            del self._role_to_ids[role]

  def get(self, worker_id: str) -> Any:
    """Returns the ActorHandle (or shim if legacy registered) for worker_id."""
    with self._lock:
      if worker_id not in self._handles:
        raise KeyError(worker_id)
      return self._shims.get(worker_id, self._handles[worker_id])

  def get_handle(self, worker_id: str) -> remote_execution.ActorHandle:
    """Returns the underlying ActorHandle for worker_id."""
    with self._lock:
      if worker_id not in self._handles:
        raise KeyError(worker_id)
      return self._handles[worker_id]

  def info(self, worker_id: str) -> datatypes.WorkerInfo:
    """Returns the snapshotted WorkerInfo for the given worker_id."""
    with self._lock:
      if worker_id not in self._infos:
        raise KeyError(worker_id)
      return self._infos[worker_id]

  def handles(
      self, role: datatypes.Role | str | None = None
  ) -> list[remote_execution.ActorHandle]:
    """Returns handles for all workers, optionally filtered by role."""
    with self._lock:
      if role is None:
        return [self._handles[i] for i in self._handles]
      role_key = _normalize_role(role)
      ids = self._role_to_ids.get(role_key, set())
      return [self._handles[i] for i in self._handles if i in ids]

  def workers(self, role: datatypes.Role | str | None = None) -> list[Any]:
    """Returns local workers supporting direct lifecycle methods."""
    with self._lock:
      if role is None:
        return [
            self._handles[i]
            for i in self._handles
            if not self._infos[i].resources.get("remote", False)
        ]
      role_key = _normalize_role(role)
      ids = self._role_to_ids.get(role_key, set())
      return [
          self._handles[i]
          for i in self._handles
          if i in ids and not self._infos[i].resources.get("remote", False)
      ]

  def group(self, role: datatypes.Role | str) -> WorkerGroup:
    """Returns the (possibly empty) group of workers serving role."""
    role_key = _normalize_role(role)
    with self._lock:
      ids = self._role_to_ids.get(role_key, set())
      handles = [self._handles[i] for i in self._handles if i in ids]
      members = [
          self._shims.get(i, self._handles[i])
          for i in self._handles
          if i in ids
      ]
      infos = [self._infos[i] for i in self._handles if i in ids]
      worker_ids = [i for i in self._handles if i in ids]
      return WorkerGroup(
          role=role_key,
          handles=handles,
          members=members,
          infos=infos,
          worker_ids=worker_ids,
      )

  def roles(self) -> set[str]:
    """Returns the set of currently active role names."""
    with self._lock:
      return set(self._role_to_ids)

  def worker_ids(self) -> list[str]:
    """Returns the list of registered worker ids."""
    with self._lock:
      return list(self._handles)

  def infos(self) -> list[datatypes.WorkerInfo]:
    """Returns the snapshotted WorkerInfos sorted by worker id."""
    with self._lock:
      return [self._infos[i] for i in self._handles]

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
    normalized_min_workers = {
        _normalize_role(role): count for role, count in min_workers.items()
    }
    while True:
      current_counts = {
          role: len(self.handles(role)) for role in normalized_min_workers
      }
      if all(
          current_counts[role] >= target_count
          for role, target_count in normalized_min_workers.items()
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

  def __len__(self) -> int:
    with self._lock:
      return len(self._handles)

  def __contains__(self, worker_id: str) -> bool:
    with self._lock:
      return worker_id in self._handles
