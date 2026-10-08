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
from collections.abc import Collection, Iterator
import enum
import threading

from tunix.experimental.common import datatypes
from tunix.experimental.worker import abstract_worker


class MembershipState(enum.Enum):
  """Lifecycle states for a worker registered in the orchestrator."""

  INITIALIZING = "initializing"
  PENDING_WEIGHT_SYNC = "pending_weight_sync"
  ACTIVE = "active"
  EVICTED = "evicted"


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
    self._states: dict[str, MembershipState] = {}
    self._incarnations: dict[str, int] = {}
    self._state_listeners: list[
        collections.abc.Callable[[str, MembershipState], None]
    ] = []

  def add_state_listener(
      self,
      listener: collections.abc.Callable[[str, MembershipState], None],
  ) -> None:
    """Registers a callback invoked whenever a worker's MembershipState changes."""
    with self._lock:
      self._state_listeners.append(listener)

  def _notify_state_listeners(
      self, worker_id: str, state: MembershipState
  ) -> None:
    with self._lock:
      listeners = list(self._state_listeners)
    for listener in listeners:
      listener(worker_id, state)

  def register(
      self,
      worker: abstract_worker.Worker,
      override: bool = False,
      *,
      state: MembershipState = MembershipState.ACTIVE,
  ) -> datatypes.WorkerInfo:
    """Registers a worker under its declared id and roles.

    Args:
      worker: The worker to register; its `info()` supplies id and roles.
      override: If true, silently overwrites an existing registration with the
        same id.
      state: Initial membership state for the registered worker.

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
      self._states[worker_id] = state
      self._incarnations[worker_id] = self._incarnations.get(worker_id, 0) + 1
      for role in info.roles:
        self._role_to_ids[role].add(worker_id)
    self._notify_state_listeners(worker_id, state)
    return info

  def unregister(self, worker_id: str) -> None:
    """Removes a worker (and its role memberships) from the registry."""
    with self._lock:
      info = self._infos.pop(worker_id)
      del self._workers[worker_id]
      self._states.pop(worker_id, None)
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

  def state(self, worker_id: str) -> MembershipState:
    with self._lock:
      return self._states[worker_id]

  def incarnation(self, worker_id: str) -> int:
    with self._lock:
      if worker_id not in self._workers:
        raise KeyError(worker_id)
      return self._incarnations[worker_id]

  def set_state(
      self,
      worker_id: str,
      state: MembershipState,
      *,
      expected_incarnation: int | None = None,
      expected_state: MembershipState | None = None,
  ) -> bool:
    """Updates the membership state of `worker_id` if incarnation/state match.

    Listeners are only notified on an actual transition; re-applying the
    current state is a no-op so re-entrant eviction paths (session -> registry
    -> engine -> session) do not fire duplicate callbacks.
    """
    with self._lock:
      if worker_id not in self._workers:
        raise KeyError(worker_id)
      if (
          expected_incarnation is not None
          and self._incarnations.get(worker_id) != expected_incarnation
      ):
        return False
      if (
          expected_state is not None
          and self._states.get(worker_id) != expected_state
      ):
        return False
      changed = self._states.get(worker_id) != state
      self._states[worker_id] = state
    if changed:
      self._notify_state_listeners(worker_id, state)
    return True

  def evict(
      self,
      worker_id: str,
      *,
      expected_incarnation: int | None = None,
  ) -> bool:
    """Marks `worker_id` as EVICTED while preserving its incarnation history."""
    with self._lock:
      if worker_id not in self._workers:
        return False
      if (
          expected_incarnation is not None
          and self._incarnations.get(worker_id) != expected_incarnation
      ):
        return False
      changed = self._states.get(worker_id) != MembershipState.EVICTED
      self._states[worker_id] = MembershipState.EVICTED
    if changed:
      self._notify_state_listeners(worker_id, MembershipState.EVICTED)
    return True

  def group(
      self,
      role: str,
      *,
      states: Collection[MembershipState] | None = None,
  ) -> WorkerGroup:
    """Returns the (possibly empty) group of workers serving `role`."""
    allowed_states = (
        (MembershipState.ACTIVE,) if states is None else tuple(states)
    )
    with self._lock:
      ids = sorted(self._role_to_ids.get(role, set()))
      return WorkerGroup(
          role,
          [
              self._workers[i]
              for i in ids
              if self._states.get(i) in allowed_states
          ],
      )

  def roles(self) -> set[str]:
    with self._lock:
      return set(self._role_to_ids)

  def _live_ids(self, include_evicted: bool) -> list[str]:
    """Returns sorted worker ids, excluding EVICTED unless requested."""
    return sorted(
        wid
        for wid in self._workers
        if include_evicted or self._states.get(wid) != MembershipState.EVICTED
    )

  def worker_ids(self, *, include_evicted: bool = False) -> list[str]:
    """Returns sorted worker ids; EVICTED workers are excluded by default."""
    with self._lock:
      return self._live_ids(include_evicted)

  def workers(
      self, *, include_evicted: bool = False
  ) -> list[abstract_worker.Worker]:
    """Returns workers ordered by id; EVICTED workers are excluded by default."""
    with self._lock:
      return [self._workers[i] for i in self._live_ids(include_evicted)]

  def infos(
      self, *, include_evicted: bool = False
  ) -> list[datatypes.WorkerInfo]:
    """Returns worker infos ordered by id; EVICTED workers are excluded by default."""
    with self._lock:
      return [self._infos[i] for i in self._live_ids(include_evicted)]

  def __len__(self) -> int:
    with self._lock:
      return len(self._workers)

  def __contains__(self, worker_id: str) -> bool:
    with self._lock:
      return worker_id in self._workers
