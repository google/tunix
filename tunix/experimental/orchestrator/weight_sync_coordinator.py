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

"""Installing one set of weights across several rollout workers.

With a single rollout worker, syncing weights is one call that either happened
or raised. With several, the interesting states are in between: three workers
took the new weights and one did not, and the one that did not will happily
keep generating from the old ones. Nothing about that is visible unless the
round is run as a protocol.

So a round is explicit. A version is minted, the trainer stages it, and each
replica is then visited individually -- fenced, told to install, and required
to say which version it ended up on. A replica that cannot reach the version
is quarantined and named, and the round reports exactly who is on the new
weights and who is not. Callers decide what to do with a partial result; they
cannot fail to notice one.

Two things are deliberately not swallowed. A replica that reports a different
version than it was asked for is not treated as merely unlucky -- it is
recorded with that reason, because "installed something else" and "could not
be reached" are different problems. And exceptions that indicate a programming
mistake propagate instead of being logged as a quarantine, so a broken call
signature cannot masquerade as a fleet-wide outage.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, Optional, Sequence

from absl import logging

from tunix.experimental.common import datatypes

# Failures that mean the code is wrong, not the worker. Quarantining on these
# would turn one bug into an apparently unreachable fleet.
_PROGRAMMING_ERRORS = (
    TypeError,
    AttributeError,
    NotImplementedError,
    ImportError,
    IndentationError,
    SyntaxError,
)


class QuarantineReason(enum.Enum):
  """Why a replica is not on the new weights."""

  UNREACHABLE = "unreachable"
  WRONG_VERSION = "wrong_version"


@dataclasses.dataclass(frozen=True)
class QuarantinedReplica:
  """A replica that did not reach the requested version.

  Attributes:
    worker_id: Which replica.
    reason: Whether it could not be reached or installed something else.
    detail: Human-readable specifics for logs and metrics.
  """

  worker_id: str
  reason: QuarantineReason
  detail: str


@dataclasses.dataclass(frozen=True)
class SyncOutcome:
  """What one sync round achieved.

  Attributes:
    version: The version that was staged and requested.
    synced: Ids of replicas confirmed to be running `version`.
    quarantined: Replicas that are not, and why.
  """

  version: int
  synced: Sequence[str]
  quarantined: Sequence[QuarantinedReplica]

  @property
  def all_synced(self) -> bool:
    return not self.quarantined

  @property
  def quarantined_ids(self) -> list[str]:
    return [replica.worker_id for replica in self.quarantined]


class WeightSyncCoordinator:
  """Runs versioned weight-sync rounds across a set of rollout replicas."""

  def __init__(
      self,
      *,
      trainer: Any = None,
      replicas: Sequence[Any] = (),
      max_retries: int = 1,
      controller_id: str = "",
  ):
    """Initializes the coordinator.

    Args:
      trainer: Staged before any replica installs; may be None when weights
        are published by other means.
      replicas: The rollout workers to keep on one version.
      max_retries: Extra attempts per replica before quarantining it. The
        worker contract releases its fence on failure, so a retry re-fetches.
      controller_id: Optional transport controller name carried on requests.

    Raises:
      ValueError: If `max_retries` is negative.
    """
    if max_retries < 0:
      raise ValueError(f"max_retries must be >= 0, got {max_retries}.")
    self._trainer = trainer
    self._replicas = list(replicas)
    self._max_retries = max_retries
    self._controller_id = controller_id

  @property
  def replicas(self) -> Sequence[Any]:
    return tuple(self._replicas)

  def sync(self, version: int) -> SyncOutcome:
    """Installs `version` on every replica, reporting who got there.

    Args:
      version: The version to install, already minted by the caller.

    Returns:
      Which replicas are on `version` and which are not.
    """
    request = datatypes.WeightSyncRequest(
        controller_id=self._controller_id,
        policy_version=version,
        source_metadata=self._stage(version),
    )

    synced: list[str] = []
    quarantined: list[QuarantinedReplica] = []
    # Visited one at a time on purpose: a broadcast cannot tell you that only
    # some of them arrived.
    for replica in self._replicas:
      failure = self._sync_one(replica, request, version)
      if failure is None:
        synced.append(_worker_id(replica))
      else:
        quarantined.append(failure)

    outcome = SyncOutcome(
        version=version, synced=synced, quarantined=quarantined
    )
    _log(outcome)
    return outcome

  def _stage(self, version: int) -> Any:
    """Asks the trainer to publish the weights, returning any coordinates."""
    if self._trainer is None:
      return None
    return self._trainer.prepare_weight_sync(
        datatypes.WeightSyncRequest(
            controller_id=self._controller_id, policy_version=version
        )
    )

  def _sync_one(
      self,
      replica: Any,
      request: datatypes.WeightSyncRequest,
      version: int,
  ) -> Optional[QuarantinedReplica]:
    """Fences and installs on one replica; returns why it failed, or None.

    Raises:
      Exception: Re-raises errors that indicate a programming mistake rather
        than a worker failure.
    """
    worker_id = _worker_id(replica)
    last_error = ""
    for attempt in range(self._max_retries + 1):
      try:
        replica.prepare_weight_sync(request)
        acked = replica.sync_weights(request)
      except _PROGRAMMING_ERRORS:
        # Not a fleet problem. Surfacing it beats reporting every replica as
        # unreachable and moving on.
        raise
      except Exception as e:  # pylint: disable=broad-exception-caught
        last_error = f"{type(e).__name__}: {e}"
        logging.warning(
            "Weight sync attempt %d for replica %r failed: %s",
            attempt + 1,
            worker_id,
            last_error,
        )
        continue

      if acked != version:
        # It answered, so it is reachable; it is just not running what we
        # asked for. Retrying a worker that installs the wrong thing does not
        # help, and calling it unreachable would misdirect the diagnosis.
        return QuarantinedReplica(
            worker_id,
            QuarantineReason.WRONG_VERSION,
            f"reported version {acked}, expected {version}.",
        )
      return None

    return QuarantinedReplica(
        worker_id,
        QuarantineReason.UNREACHABLE,
        last_error or "no response.",
    )


async def run_sync_round(
    pool: Any,
    coordinator: WeightSyncCoordinator,
    version: int,
) -> SyncOutcome:
  """Installs `version` across a pool's workers and stops using the stragglers.

  The round runs while the pool is drained, so no request spans the weight
  change, and any replica that did not reach the version is taken out of
  rotation immediately. Leaving it in would keep feeding it work whose
  trajectories are stamped with weights the trainer has moved past, and the
  version stamp would be the only trace.

  Args:
    pool: The rollout pool to quiet and then prune.
    coordinator: Runs the round.
    version: The version to install.

  Returns:
    Which replicas reached it and which were quarantined.
  """
  async with pool.drained():
    outcome = coordinator.sync(version)

  for replica in outcome.quarantined:
    if pool.remove_worker(replica.worker_id):
      logging.warning(
          "Removed replica %r from rollout dispatch: it is not on weight"
          " version %d.",
          replica.worker_id,
          version,
      )
  return outcome


def _worker_id(replica: Any) -> str:
  """The replica's id, however it exposes one."""
  info = getattr(replica, "info", None)
  if callable(info):
    try:
      return info().worker_id
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  return str(getattr(replica, "worker_id", replica))


def _log(outcome: SyncOutcome) -> None:
  if outcome.all_synced:
    logging.info(
        "Weight sync to version %d reached all %d replicas.",
        outcome.version,
        len(outcome.synced),
    )
    return
  for replica in outcome.quarantined:
    logging.error(
        "Replica %r is not on weight version %d: %s (%s)",
        replica.worker_id,
        outcome.version,
        replica.detail,
        replica.reason.value,
    )
