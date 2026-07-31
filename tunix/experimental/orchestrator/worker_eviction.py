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

"""Acting on a worker that has stopped being useful.

Health has been observable for a while and consumed by nothing: a worker could
report itself in error, or sit in a startup state forever, and keep receiving
work regardless. Detection without a reaction is close to worthless -- it puts
the failure in a log and leaves the behavior unchanged.

Evicting means two things happening together, because either alone leaves the
fleet inconsistent: the worker leaves the registry, so the control plane stops
counting it as part of the fleet, and it leaves the dispatch rotation, so no
further work is routed to it.

Eviction is a decision about the future, not a cleanup of the past. Work the
worker was already holding is settled by whoever is running that batch, which
is why this runs between batches rather than during one.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Iterable, Mapping, Optional, Sequence

from absl import logging

from tunix.experimental.common import datatypes

WorkerState = datatypes.WorkerState


class EvictionReason(enum.Enum):
  """Why a worker was taken out of service."""

  REPORTED_ERROR = "reported_error"
  OVERDUE_IN_STATE = "overdue_in_state"
  UNRESPONSIVE = "unresponsive"
  REQUESTED = "requested"


@dataclasses.dataclass(frozen=True)
class Eviction:
  """A worker taken out of service, and why.

  Attributes:
    worker_id: Which worker.
    reason: What disqualified it.
    detail: Human-readable specifics.
    removed_from_dispatch: Whether it also left the dispatch rotation. False
      when there was no pool, or when it was the last worker in one.
  """

  worker_id: str
  reason: EvictionReason
  detail: str
  removed_from_dispatch: bool = False


class WorkerEvictor:
  """Turns health reports into workers leaving the fleet."""

  def __init__(
      self,
      registry: Any,
      *,
      pool: Any = None,
      on_evicted: Optional[Any] = None,
  ):
    """Initializes the evictor.

    Args:
      registry: The control plane's record of who exists.
      pool: The dispatch rotation, when there is one. Without it a worker can
        still leave the registry, but something else must stop sending it
        work.
      on_evicted: Optional notification, called with each `Eviction`.
    """
    self._registry = registry
    self._pool = pool
    self._on_evicted = on_evicted
    self.evicted: list[Eviction] = []

  def evict_unhealthy(
      self,
      reports: Mapping[str, datatypes.HealthReport],
      overdue: Iterable[Any] = (),
  ) -> list[Eviction]:
    """Evicts every worker the latest health says is not usable.

    Args:
      reports: What each worker last said about itself.
      overdue: Workers past the deadline for the state they are in. Separate
        from the reports because a worker stuck in startup is answering
        normally -- what is wrong is that the answer never changes.

    Returns:
      The evictions performed.
    """
    performed: list[Eviction] = []
    seen: set[str] = set()

    for worker_id, report in sorted(reports.items()):
      if report.state != WorkerState.ERROR:
        continue
      seen.add(worker_id)
      performed.append(
          self.evict(
              worker_id,
              EvictionReason.REPORTED_ERROR,
              report.last_error or "reported an error state.",
          )
      )

    for entry in overdue:
      worker_id = entry.worker_id
      if worker_id in seen:
        continue
      seen.add(worker_id)
      performed.append(
          self.evict(
              worker_id,
              EvictionReason.OVERDUE_IN_STATE,
              f"has been {entry.state} for {entry.elapsed_s:.0f}s, past the"
              f" {entry.deadline_s:.0f}s allowed.",
          )
      )

    return [eviction for eviction in performed if eviction is not None]

  def evict(
      self,
      worker_id: str,
      reason: EvictionReason = EvictionReason.REQUESTED,
      detail: str = "",
  ) -> Optional[Eviction]:
    """Takes one worker out of the registry and the dispatch rotation.

    Args:
      worker_id: Which worker.
      reason: Why.
      detail: Specifics for the log.

    Returns:
      What was done, or None if the worker was already gone.
    """
    removed_from_dispatch = False
    if self._pool is not None:
      try:
        removed_from_dispatch = bool(self._pool.remove_worker(worker_id))
      except ValueError as e:
        # Refusing to empty the pool is the pool's call, and the right one:
        # an empty rotation turns every later request into a wait for nothing.
        # Say so rather than pretending the eviction was complete.
        logging.error(
            "Worker %r is unhealthy but is the only one left serving; keeping"
            " it in rotation: %s",
            worker_id,
            e,
        )

    try:
      self._registry.unregister(worker_id)
    except KeyError:
      logging.info("Worker %r had already left the registry.", worker_id)
      if not removed_from_dispatch:
        return None

    eviction = Eviction(
        worker_id=worker_id,
        reason=reason,
        detail=detail,
        removed_from_dispatch=removed_from_dispatch,
    )
    self.evicted.append(eviction)
    logging.warning(
        "Evicted worker %r: %s (%s)", worker_id, detail, reason.value
    )
    if self._on_evicted is not None:
      self._on_evicted(eviction)
    return eviction

  @property
  def evicted_ids(self) -> Sequence[str]:
    return [eviction.worker_id for eviction in self.evicted]
