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
owns all cluster `ActorHandle`s and `WorkerInfo`s indexed by id and by declared
role, so a fused worker (e.g. one serving both "trainer" and "inference") joins
every matching role query. Role membership is snapshotted at registration time.
"""

import asyncio
import collections
from collections.abc import Iterator, Mapping, Sequence
import pickle
import threading
from typing import Any, cast

from absl import logging
from tunix.experimental.common import datatypes
from tunix.experimental.worker import abstract_worker
from tunix.experimental.worker import remote_execution


# TODO(tunix-dev): remove this after raiden weight sync redesign.
class RemoteWorkerShim:
  """Presents a remote ActorHandle as a weight sync participant."""

  def __init__(
      self, handle: remote_execution.ActorHandle, info: datatypes.WorkerInfo
  ):
    self._handle = handle
    self._info = info

  def info(self) -> datatypes.WorkerInfo:
    return self._info

  async def prepare_weight_sync(self, *args, **kwargs) -> Any:
    return await self._handle.asubmit("prepare_weight_sync", *args, **kwargs)

  async def release_weight_sync(self, *args, **kwargs) -> Any:
    return await self._handle.asubmit("release_weight_sync", *args, **kwargs)

  async def bind_weight_sync(self, *args, **kwargs) -> Any:
    return await self._handle.asubmit("bind_weight_sync", *args, **kwargs)

  async def get_weight_sync_metadata(self, *args, **kwargs) -> Any:
    return await self._handle.asubmit(
        "get_weight_sync_metadata", *args, **kwargs
    )

  async def pre_weight_sync(self, *args, **kwargs) -> Any:
    return await self._handle.asubmit("pre_weight_sync", *args, **kwargs)

  async def weight_sync(self, *args, **kwargs) -> Any:
    return await self._handle.asubmit("weight_sync", *args, **kwargs)

  async def post_weight_sync(self, *args, **kwargs) -> Any:
    return await self._handle.asubmit("post_weight_sync", *args, **kwargs)

  async def abort_weight_sync(self, *args, **kwargs) -> Any:
    return await self._handle.asubmit("abort_weight_sync", *args, **kwargs)

  async def get_weight_sync_status(self, *args, **kwargs) -> Any:
    return await self._handle.asubmit("get_weight_sync_status", *args, **kwargs)


class WorkerGroup:
  """An ordered, immutable view of the weight-sync participants serving a role."""

  def __init__(
      self,
      role: str,
      members: Sequence[abstract_worker.Worker | RemoteWorkerShim],
  ):
    self._role = role
    self._members = list(members)

  @property
  def role(self) -> str:
    return self._role

  def members(self) -> list[abstract_worker.Worker | RemoteWorkerShim]:
    return list(self._members)

  def is_empty(self) -> bool:
    return not self._members

  def __len__(self) -> int:
    return len(self._members)

  def __iter__(self) -> Iterator[abstract_worker.Worker | RemoteWorkerShim]:
    return iter(self._members)


class WorkerRegistry:
  """Single source of truth owning all cluster ActorHandles and WorkerInfos."""

  def __init__(self):
    self._lock = threading.Lock()
    self._condition = threading.Condition(self._lock)
    self._handles: dict[str, remote_execution.ActorHandle] = {}
    self._infos: dict[str, datatypes.WorkerInfo] = {}
    self._role_to_ids: dict[str, set[str]] = collections.defaultdict(set)

  def _insert_locked(
      self,
      info: datatypes.WorkerInfo,
      handle: remote_execution.ActorHandle,
      override: bool,
  ) -> None:
    worker_id = info.worker_id
    if worker_id in self._handles and not override:
      raise ValueError(f"duplicate worker_id: {worker_id!r}")
    if not info.roles:
      raise ValueError(f"worker {worker_id!r} declares no roles")

    if worker_id in self._infos:
      old_info = self._infos[worker_id]
      for role in old_info.roles:
        if role in self._role_to_ids:
          self._role_to_ids[role].discard(worker_id)
          if not self._role_to_ids[role]:
            del self._role_to_ids[role]

    self._handles[worker_id] = handle
    self._infos[worker_id] = info
    for role in info.roles:
      self._role_to_ids[role].add(worker_id)
    self._condition.notify_all()

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
        registered (and override is False).
    """
    info = worker.info()
    handle = remote_execution.InProcessActorHandle(
        remote_execution.InProcessRemoteExecutionServer(worker)
    )
    with self._condition:
      self._insert_locked(info, handle, override=override)
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
      self._insert_locked(info, handle, override=override)
    logging.info(
        "Registered remote worker %r with roles %s.",
        worker_id,
        sorted(info.roles),
    )
    return info

  def register_from_hostname(
      self,
      hostname: str,
      _: int,
      metadata: bytes,
      rpc_timeout_s: float = 1800.0,
  ) -> datatypes.WorkerInfo:
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
    if "trajectory_store_config" in md:
      resources["trajectory_store_config"] = md["trajectory_store_config"]
    return self.register_handle(
        worker_id=worker_id,
        roles=[role],
        handle=remote_execution.ActorHandle.from_address(
            f"grpc://{service_address}",
            rpc_timeout_s=rpc_timeout_s,
        ),
        resources=resources,
    )

  def wait_for_workers(
      self,
      min_workers: Mapping[datatypes.Role | str, int],
      timeout: float | None = None,
  ) -> None:
    """Waits for registered workers to meet the minimum required counts.

    Args:
      min_workers: Mapping from Role or role name to the minimum number of
        workers required.
      timeout: Maximum duration to wait in seconds before raising TimeoutError.
        If None, waits indefinitely until requirements are met.

    Raises:
      TimeoutError: If the required worker counts are not met within timeout.
    """
    with self._condition:

      def _counts() -> dict[datatypes.Role | str, int]:
        return {
            role: len(self._role_to_ids.get(role, set()))
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

  def unregister(self, worker_id: str) -> None:
    """Removes a worker from the registry and closes its ActorHandle outside the lock."""
    with self._condition:
      info = self._infos.pop(worker_id)
      handle = self._handles.pop(worker_id)
      for role in info.roles:
        members = self._role_to_ids.get(role)
        if members is not None:
          members.discard(worker_id)
          if not members:
            del self._role_to_ids[role]
      self._condition.notify_all()
    asyncio.run(handle.close())

  def close(self) -> None:
    """Closes all registered ActorHandles outside the lock and clears registry state."""
    with self._condition:
      handles = [self._handles[wid] for wid in sorted(self._handles)]
      self._handles.clear()
      self._infos.clear()
      self._role_to_ids.clear()
      self._condition.notify_all()

    errors: list[BaseException] = []
    for handle in handles:
      try:
        asyncio.run(handle.close())
      except Exception as exc:  # pylint: disable=broad-exception-caught
        errors.append(exc)
    if errors:
      raise RuntimeError(
          f"Failed to close {len(errors)} ActorHandle(s): {errors!r}"
      ) from errors[0]

  def handle(self, worker_id: str) -> remote_execution.ActorHandle:
    with self._lock:
      return self._handles[worker_id]

  def handles(
      self, role: datatypes.Role | str
  ) -> list[remote_execution.ActorHandle]:
    """Returns the (possibly empty) list of ActorHandles serving `role`."""
    with self._lock:
      ids = sorted(self._role_to_ids.get(role, set()))
      return [self._handles[i] for i in ids]

  def info(self, worker_id: str) -> datatypes.WorkerInfo:
    with self._lock:
      return self._infos[worker_id]

  def infos(
      self, role: datatypes.Role | str | None = None
  ) -> list[datatypes.WorkerInfo]:
    """Returns WorkerInfos sorted by worker_id (all workers when role is None, or filtered by role)."""
    with self._lock:
      if role is None:
        ids = sorted(self._infos)
      else:
        ids = sorted(self._role_to_ids.get(role, set()))
      return [self._infos[i] for i in ids]

  def group(self, role: datatypes.Role | str) -> WorkerGroup:
    """Returns the (possibly empty) group of weight-sync participants serving `role`."""
    with self._lock:
      ids = sorted(self._role_to_ids.get(role, set()))
      members = [
          cast(abstract_worker.Worker, handle.server.bound_instance)
          if isinstance(
              handle := self._handles[i], remote_execution.InProcessActorHandle
          )
          else RemoteWorkerShim(handle, self._infos[i])
          for i in ids
      ]
      return WorkerGroup(role, members)

  def roles(self) -> set[str]:
    with self._lock:
      return set(self._role_to_ids)

  def worker_ids(self) -> list[str]:
    with self._lock:
      return sorted(self._handles)

  def __len__(self) -> int:
    with self._lock:
      return len(self._handles)

  def __contains__(self, worker_id: str) -> bool:
    with self._lock:
      return worker_id in self._handles
