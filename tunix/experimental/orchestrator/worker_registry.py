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
indexes registered `Worker`s by id and by the roles each declares through
`info()`, so a fused worker (e.g. one serving both "trainer" and "inference")
joins every matching `WorkerGroup`. Role membership is snapshotted from `info()`
at registration time, so grouping stays stable even if a worker's live state
changes.
"""

import collections
from collections.abc import Iterator
import threading

from tunix.experimental.common import datatypes
from tunix.experimental.worker import abstract_worker


class WorkerGroup:
  """An ordered, immutable view of the workers serving a single role."""

  def __init__(self, role: str, members: list[abstract_worker.Worker]):
    self._role = role
    self._members = list(members)

  @property
  def role(self) -> str:
    return self._role

  def members(self) -> list[abstract_worker.Worker]:
    return list(self._members)

  def is_empty(self) -> bool:
    return not self._members

  def __len__(self) -> int:
    return len(self._members)

  def __iter__(self) -> Iterator[abstract_worker.Worker]:
    return iter(self._members)


class WorkerRegistry:
  """Indexes workers by id and by declared role.

  Registration snapshots each worker's `WorkerInfo`; lookups return live worker
  handles. Worker ids must be unique.
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._workers: dict[str, abstract_worker.Worker] = {}
    self._infos: dict[str, datatypes.WorkerInfo] = {}
    self._role_to_ids: dict[str, set[str]] = collections.defaultdict(set)

  def register(
      self, worker: abstract_worker.Worker, override: bool = False
  ) -> datatypes.WorkerInfo:
    """Registers a worker under its declared id and roles.

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
    worker_id = info.worker_id
    with self._lock:
      if worker_id in self._workers and not override:
        raise ValueError(f"duplicate worker_id: {worker_id!r}")
      if not info.roles:
        raise ValueError(f"worker {worker_id!r} declares no roles")

      # If overriding, clean up old role indexing for this worker
      if worker_id in self._infos:
        old_info = self._infos[worker_id]
        for role in old_info.roles:
          if role in self._role_to_ids:
            self._role_to_ids[role].discard(worker_id)
            if not self._role_to_ids[role]:
              del self._role_to_ids[role]

      self._workers[worker_id] = worker
      self._infos[worker_id] = info
      for role in info.roles:
        self._role_to_ids[role].add(worker_id)
    return info

  def unregister(self, worker_id: str) -> None:
    """Removes a worker (and its role memberships) from the registry."""
    with self._lock:
      info = self._infos.pop(worker_id)
      del self._workers[worker_id]
      for role in info.roles:
        members = self._role_to_ids.get(role)
        if members is not None:
          members.discard(worker_id)
          if not members:
            del self._role_to_ids[role]

  def get(self, worker_id: str) -> abstract_worker.Worker:
    with self._lock:
      return self._workers[worker_id]

  def info(self, worker_id: str) -> datatypes.WorkerInfo:
    with self._lock:
      return self._infos[worker_id]

  def group(self, role: str) -> WorkerGroup:
    """Returns the (possibly empty) group of workers serving `role`."""
    with self._lock:
      ids = sorted(self._role_to_ids.get(role, set()))
      return WorkerGroup(role, [self._workers[i] for i in ids])

  def roles(self) -> set[str]:
    with self._lock:
      return set(self._role_to_ids)

  def worker_ids(self) -> list[str]:
    with self._lock:
      return sorted(self._workers)

  def workers(self) -> list[abstract_worker.Worker]:
    with self._lock:
      return [self._workers[i] for i in sorted(self._workers)]

  def infos(self) -> list[datatypes.WorkerInfo]:
    with self._lock:
      return [self._infos[i] for i in sorted(self._workers)]

  def __len__(self) -> int:
    with self._lock:
      return len(self._workers)

  def __contains__(self, worker_id: str) -> bool:
    with self._lock:
      return worker_id in self._workers
