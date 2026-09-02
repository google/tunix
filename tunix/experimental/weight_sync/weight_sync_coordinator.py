# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Drives weight sync rounds across the orchestrator's worker registry.

`WeightSyncHandler` knows how to talk to a transport; it does not know who the
participants are. This module supplies that half: it looks workers up by role
in the `WorkerRegistry`, collects transport-neutral work-unit metadata from
each side exactly once per round, registers every work unit, and drives the
round. The default deployment uses `RaidenHandler`, but no coordinator,
worker protocol, or public result below contains a Raiden type or planner
option.

One round, in order:

    source                          destination
    ------                          -----------
                                    bind_weight_sync()       idempotent; builds
                                                             the synchronizer if
                                                             absent, keeps ports
                                    get_weight_sync_metadata()
    prepare_weight_sync()                                    (both sides still
      rebind arrays, FFI D2H,                                serving; metadata
      return this round's metadata                           and registration
    [register all units]                                     happen before any
                                    pre_weight_sync()        downtime)
                                      stop admitting, drain,
                                      free prefix + KV cache
    [handler.transfer]  --------->  (transport delivers the prepared weights)
                                    weight_sync()
                                      materialize into the STAGING copy only;
                                      record pending policy version
                                    post_weight_sync()
                                      atomically publish staging as serving,
                                      rebuild KV cache, resume admitting
    release_weight_sync()

On failure before any `weight_sync()` started, every destination that was asked
to prepare is told to `abort_weight_sync()`: discard staging, rebuild the KV
cache, resume serving the old weights.

The staging contract, and what it is NOT. Rollback is only possible because
`weight_sync()` MUST NOT touch the serving copy: publishing is
`post_weight_sync()`'s job and must be atomic per worker. This is a REQUIREMENT
on destination implementations, not something the transport does for them:
`h2d()` writes whatever arrays the synchronizer is currently bound to. Two
known ways a real destination can satisfy it: bind the synchronizer to a
dedicated staging array set and publish with an on-device copy, or use
`WeightSynchronizer.bind_weights()` (rebinds arrays in place, ports stable) to
ping-pong between two buffer sets with a pointer-swap publish. Until a real
McJAX destination passes two rounds under this contract on TPU, treat
multi-round ping-pong as unproven; the fakes here prove the coordinator,
nothing more.

Failure states are not collapsed, and neither are the two axes they live on.
Version consistency (who serves which policy version) and worker health (who is
serving at all) are separate dimensions; `WeightSyncResult.workers` carries the
per-worker record and `state` the round-level summary:

  ABORTED               rollback ran and every destination confirmed it.
  PARTIALLY_COMMITTED   some destinations published, others did not. Mixed
                        versions; no coordinator call can fix it.
  FAILED_NEEDS_RESTART  at least one destination could not be rolled back or
                        is in an unknown state; it may be stuck unserving.
  UNKNOWN_TRANSFER_STATE the transfer call timed out. The executor thread may
                        STILL BE WRITING into destination staging, so nothing
                        was aborted and source staging was NOT released;
                        anything else would destroy memory in use.

After any of the last three, the coordinator is poisoned: further `sync()`
calls raise until `reset_after_recovery()` is called, because starting a new
round against workers in unknown states can corrupt them further (or regress
policy versions).

Rounds are STRICTLY SYNCHRONOUS by design: rollout resumes only once the
serving fleet is on one policy version, so a round either publishes
everywhere or rolls back, and a split fleet is a failure rather than a mode.
That is what the current path needs, and it is the only setting in which the
per-worker records above mean what they say.

TODO(haochengd): support mixed-version serving as an opt-in alternative.
Letting fast destinations serve version N+1 while slow ones still serve N
removes the global barrier and the wait on the slowest worker, at the price
of off-policy rollouts. It is not a relaxation of the checks here but a
second consistency policy, and it needs all of: per-trajectory policy-version
tagging so the trainer can tell which policy produced a sample, an explicit
staleness bound (how many versions behind a destination may serve before it
is drained), a trainer-side contract for what to do with stale trajectories,
and a round state that reports a version spread instead of treating one as
PARTIALLY_COMMITTED. Until those exist, a split fleet stays a failure.

Worker-side contract: workers keep their current round keyed by the
(req_id, uuid) carried in `WeightSyncRequest.extra_config`, make every phase
idempotent for that key, reject phase calls carrying a stale key OR an
unknown key that cannot open a round (only a pre or an abort may), and report
their round through `get_weight_sync_status()`. `WorkerRoundTracker` in this
module implements exactly that and is what the fakes and the on-cluster mock
embed. The report is what the coordinator consults whenever a phase call
fails: an RPC error means "the reply is lost", not "the work did not happen".
One more obligation the tracker cannot carry for the worker: the coordinator
never issues overlapping phase calls for a live round, but a TIMED-OUT phase
RPC may still be executing when the rollback's abort arrives, so destinations
must serialize their phase bodies themselves (the on-cluster samplers hold an
asyncio.Lock across each phase).

This module sits after the worker lifecycle in `lifecycle.py`, it does not
replace it. `LifecycleDriver.bring_up` runs initialize -> compile -> start;
weight sync begins once workers are up and serving.
"""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import functools
import os
import threading
import time
from typing import Any, Optional, Sequence

from absl import logging
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.weight_sync import weight_sync


class NullHandler(weight_sync.WeightSyncHandler):
  """Runs every phase without moving bytes."""

  def register_work_unit(self, metadata: Any) -> None:
    del metadata

  def transfer(
      self,
      src_units: Any,
      dst_units: Any,
      req_id: str | None = None,
      generation: Any = None,
  ) -> weight_sync.TransferResult:
    del src_units, dst_units, generation
    return weight_sync.TransferResult(req_id=req_id or "", success=True)


def create_default_handler(
    mode: str | weight_sync.WeightSyncMode | None = None,
) -> weight_sync.WeightSyncHandler:
  """Creates the default weight sync handler based on options or env vars."""
  mode_name = mode or os.getenv("WEIGHT_SYNC_MODE", "raiden")
  if isinstance(mode_name, weight_sync.WeightSyncMode):
    mode_name = mode_name.value
  mode_name = str(mode_name).lower()
  if mode_name == weight_sync.WeightSyncMode.RAIDEN.value:
    from tunix.experimental.weight_sync import raiden_handler

    handler = raiden_handler.RaidenHandler(
        transfer_options=raiden_handler.make_host_staged_transfer_options()
    )
    logging.info("Built RaidenHandler natively; port %d", handler.port)
    return handler
  elif mode_name in (weight_sync.WeightSyncMode.FALLBACK.value, "noop", "no-op"):
    logging.info(
        "Built fallback NullHandler; weight sync running protocol-only."
    )
    return NullHandler()
  elif mode_name == weight_sync.WeightSyncMode.NONE.value:
    raise ValueError(
        "WEIGHT_SYNC_MODE=none disables weight sync; no handler should be"
        " constructed."
    )
  else:
    raise ValueError(f"Unknown weight sync mode: {mode_name!r}")


class RemoteWorkerShim:
  """Presents a generic remote RPC handle as a weight sync participant."""

  def __init__(self, handle: Any, info: datatypes.WorkerInfo):
    self._handle = handle
    self._info = info

  def info(self) -> datatypes.WorkerInfo:
    return self._info

  def heartbeat(self, *args, **kwargs) -> Any:
    return self._handle.submit("heartbeat", *args, **kwargs)

  def stop(self, *args, **kwargs) -> Any:
    return self._handle.submit("stop", *args, **kwargs)

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


weight_sync.WeightSyncSource.register(RemoteWorkerShim)  # pyrefly: ignore[missing-attribute]
weight_sync.WeightSyncDestination.register(RemoteWorkerShim)  # pyrefly: ignore[missing-attribute]


# The worker-report phases a failed RPC may be reconciled against, per
# coordinator call: exactly the states proving THIS call's work happened (or
# was passed by a later phase of the same round). Explicit sets rather than a
# rank comparison, because "committed" and "aborted" are both terminal but
# prove opposite things: a failed post reconciled against an "aborted" report
# is a real failure, and a failed abort reconciled against "committed" means
# the publish stands and rollback did NOT happen.
_PHASE_ACCEPTS = {
    "pre_weight_sync": frozenset({"prepared", "h2d_done", "committed"}),
    "weight_sync": frozenset({"h2d_done", "committed"}),
    "post_weight_sync": frozenset({"committed"}),
    "abort_weight_sync": frozenset({"aborted"}),
}

# Forward progression of a round on the worker side; `WorkerRoundTracker` uses
# it for duplicate detection. The two terminals share a rank and are NOT
# interchangeable -- the tracker special-cases the conflict (commit after abort
# is refused, abort after commit is a no-op) before this ordering is consulted.
_PHASE_ORDER = {
    "idle": 0,
    "prepared": 1,
    "h2d_done": 2,
    "committed": 3,
    "aborted": 3,
}

# The only phases a round the worker has never seen may arrive as. "prepared"
# is the phase that opens a round; "aborted" is admitted too because a
# cancellation can reach the worker before pre does, and refusing it would
# leave a round nobody can close. A mid-round phase for an unknown round is a
# stray -- a duplicated weight_sync, a late post from an abandoned round -- and
# admitting it would raise the worker's high-water mark past rounds that are
# still legitimately coming.
_ROUND_OPENING_PHASES = frozenset({"prepared", "aborted"})


def _manifest_entries(
    metadata_list: Sequence[weight_sync.WorkUnitMetadata],
    side: str,
    problems: list[str],
) -> dict[str, tuple[Optional[tuple[int, ...]], Optional[int]]]:
  """variable name -> (global shape, item_size), across a side's units.

  Multi-tensor units contribute their variables manifest; single-tensor
  units contribute under their WorkUnitId.data_name. Units of one side share
  names (one per host); the shapes are GLOBAL, so every host must agree --
  a disagreement is recorded as a problem instead of being collapsed away.
  """
  entries: dict[str, tuple[Optional[tuple[int, ...]], Optional[int]]] = {}

  def record(name, shape, item_size):
    value = (shape, item_size)
    if name in entries and entries[name] != value:
      problems.append(
          f"preflight: {side} units disagree on {name!r}:"
          f" {entries[name]} vs {value}"
      )
    entries[name] = value

  for metadata in metadata_list:
    if metadata.variables:
      for variable in metadata.variables:
        record(variable.name, tuple(variable.shape), variable.item_size)
    else:
      record(
          metadata.unit.data_name,
          tuple(metadata.global_shape) if metadata.global_shape else None,
          metadata.item_size,
      )
  return entries


def _manifest_mismatches(
    src_metadata: Sequence[weight_sync.WorkUnitMetadata],
    dst_metadata: Sequence[weight_sync.WorkUnitMetadata],
) -> list[str]:
  """Name/shape/item_size preflight across the two sides.

  The controller pairs variables by EXACT name and silently skips
  mismatches -- a typo'd wire name loses a tensor while the round still
  reports success. This check runs before any destination is quiesced, so
  a mismatch costs nothing but the round attempt.
  """
  problems: list[str] = []
  src = _manifest_entries(src_metadata, "source", problems)
  dst = _manifest_entries(dst_metadata, "destination", problems)
  for name in sorted(set(src) - set(dst)):
    problems.append(
        f"preflight: source variable {name!r} has no destination counterpart"
    )
  for name in sorted(set(dst) - set(src)):
    problems.append(
        f"preflight: destination variable {name!r} has no source counterpart"
    )
  for name in sorted(set(src) & set(dst)):
    src_shape, src_item = src[name]
    dst_shape, dst_item = dst[name]
    if src_shape != dst_shape:
      problems.append(
          f"preflight: {name!r} global shape differs: source {src_shape},"
          f" destination {dst_shape}"
      )
    if src_item != dst_item:
      problems.append(
          f"preflight: {name!r} item_size differs: source {src_item},"
          f" destination {dst_item}"
      )
  return problems


def _worker_id(member: object) -> str:
  """Best-effort worker id, for error messages only.

  `WorkerRegistry.register` calls `worker.info()`, so anything in the registry
  has one. A member that failed a protocol check may still be an arbitrary
  object, and this must not be the thing that raises.
  """
  info = getattr(member, "info", None)
  if callable(info):
    try:
      return getattr(info(), "worker_id", repr(member))
    except Exception:  # pylint: disable=broad-except
      pass
  return repr(member)


class StaleRoundError(RuntimeError):
  """A phase call carried a round key the worker cannot honor.

  Raised when the key is older than the round the worker is on, when it
  reuses the current uuid under a different req_id, when the phase cannot
  open the unknown round it names, or when the requested phase conflicts
  with a terminal phase this round already reached (committing an aborted
  round).
  """


class WorkerRoundTracker:
  """Worker-side round state: (req_id, uuid) keyed idempotency and staleness.

  Destinations embed one and bracket every phase with it:

      if self._tracker.admit(request, "committed"):
        ...do the work (which must itself be re-runnable)...
        self._tracker.complete(request, "committed")

  `admit` returns False when this round already reached (or passed) the phase,
  making duplicate deliveries of the same RPC a no-op; it raises
  `StaleRoundError` when the key is older than the round the worker is on, so
  a late retry from a previous round cannot touch current state. A NEWER key
  resets the tracker only when it arrives as "prepared", the phase that opens
  a round, or as "aborted", which may outrun pre. A newer key arriving
  mid-round -- "h2d_done" or "committed" for a round this worker never opened --
  is refused without touching any state: taking it would move the high-water
  mark past the smaller round still legitimately on its way, and that round
  would then be rejected as stale forever on this worker.

  `complete` is separate from `admit` on purpose: a worker that crashes
  between doing the work and recording it will re-report the previous phase,
  the coordinator will retry, and the work must tolerate re-running. That is
  the crash-consistency window; hiding it inside one call would not close it.

  Thread safety: the tracker serializes its own state, because a remote
  execution server is not guaranteed to deliver phase RPCs on one thread. The
  phase WORK around it is the worker's to serialize -- a timed-out phase RPC
  can still be executing when the rollback's abort arrives, so destinations
  must hold their own lock across each phase body.
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._req_id: Optional[str] = None
    self._uuid: int = -1
    self._phase: str = "idle"
    self._policy_version: int = -1

  @staticmethod
  def _key_of(sync_request: Any) -> tuple[str, int]:
    extra = getattr(sync_request, "extra_config", None) or {}
    return (extra.get("req_id", ""), int(extra.get("uuid", 0)))

  def admit(self, sync_request: Any, phase: str) -> bool:
    req_id, uuid = self._key_of(sync_request)
    with self._lock:
      if self._req_id is not None and uuid < self._uuid:
        raise StaleRoundError(
            f"phase {phase!r} for round (req_id={req_id!r}, uuid={uuid})"
            f" rejected: this worker is on uuid {self._uuid}"
        )
      if (
          self._req_id is not None
          and uuid == self._uuid
          and req_id != self._req_id
      ):
        raise StaleRoundError(
            f"phase {phase!r} carries uuid {uuid} under req_id {req_id!r},"
            f" but this worker's round {uuid} is req_id {self._req_id!r};"
            " a round key must never be reused"
        )
      # `self._req_id is None` covers the sentinel: a tracker that has never
      # seen a round sits at uuid -1, and the two guards above are both
      # conditioned on a round existing, so without this clause an incoming
      # uuid of -1 would slip past every check and open a round as any phase.
      if self._req_id is None or uuid > self._uuid:
        if phase not in _ROUND_OPENING_PHASES:
          raise StaleRoundError(
              f"phase {phase!r} cannot open a round: (req_id={req_id!r},"
              f" uuid={uuid}) is unknown to this worker, which is on uuid"
              f" {self._uuid}; only a pre or an abort may start a round"
          )
        # A new round supersedes whatever came before it.
        self._req_id, self._uuid, self._phase = req_id, uuid, "idle"
        self._policy_version = getattr(sync_request, "policy_version", -1)
      # The two terminals are not ordered against each other. Committing an
      # aborted round is refused loudly -- the staging this round would publish
      # was already discarded. Aborting a committed round is a no-op -- the
      # publish stands, and rollback of a publish needs a new round, not a
      # late abort.
      if phase == "committed" and self._phase == "aborted":
        raise StaleRoundError(
            f"commit refused: round (req_id={req_id!r}, uuid={uuid}) was"
            " already aborted on this worker"
        )
      if phase == "aborted" and self._phase == "committed":
        return False
      if _PHASE_ORDER[phase] <= _PHASE_ORDER.get(self._phase, 0) and (
          self._phase != "idle"
      ):
        return False  # already there: duplicate delivery, no-op
      return True

  def complete(self, sync_request: Any, phase: str) -> None:
    req_id, uuid = self._key_of(sync_request)
    with self._lock:
      if uuid != self._uuid or (
          self._req_id is not None and req_id != self._req_id
      ):
        raise StaleRoundError(
            f"complete({phase!r}) for round (req_id={req_id!r}, uuid={uuid})"
            f" but the worker is on (req_id={self._req_id!r},"
            f" uuid={self._uuid})"
        )
      self._phase = phase
      self._policy_version = getattr(
          sync_request, "policy_version", self._policy_version
      )

  def report(self) -> dict[str, Any]:
    with self._lock:
      return {
          "req_id": self._req_id,
          "uuid": self._uuid,
          "phase": self._phase,
          "policy_version": self._policy_version,
      }


# The participant protocols live in the transport-neutral contracts module
# (weight_sync.py) so every layer shares one definition. Aliases keep the
# existing import sites and test doubles working unchanged.
WeightSyncSource = weight_sync.WeightSyncSource
WeightSyncDestination = weight_sync.WeightSyncDestination


class RoundState(enum.Enum):
  """Round-level summary.

  Per-worker truth lives in `WeightSyncResult.workers`.
  """

  IDLE = "idle"
  PREPARING = "preparing"
  PREPARED = "prepared"
  TRANSFERRING = "transferring"
  H2D_IN_PROGRESS = "h2d_in_progress"
  PENDING_COMMIT = "pending_commit"
  COMMITTED = "committed"
  ABORTED = "aborted"
  PARTIALLY_COMMITTED = "partially_committed"
  FAILED_NEEDS_RESTART = "failed_needs_restart"
  # The transfer call timed out with its executor thread possibly still
  # writing. Nothing was aborted and source staging was not released; doing
  # either could destroy memory a live transfer is using.
  UNKNOWN_TRANSFER_STATE = "unknown_transfer_state"


_POISONING_STATES = frozenset({
    RoundState.PARTIALLY_COMMITTED,
    RoundState.FAILED_NEEDS_RESTART,
    RoundState.UNKNOWN_TRANSFER_STATE,
})


@dataclasses.dataclass(frozen=True)
class PhaseTimeouts:
  """Per-phase deadlines, in seconds.

  Weight sync moves gigabytes; generic RPC defaults (tens of seconds) are not
  enough for D2H, transfer, or H2D, and a too-short deadline turns a slow
  round into a spurious rollback. These deadlines govern the coordinator's
  waiting only; a remote worker adapter must configure its own RPC deadline
  at least this large.
  """

  bind: float = 60.0
  metadata: float = 60.0
  source_prepare: float = 900.0
  pre: float = 180.0
  transfer: float = 1800.0
  h2d: float = 900.0
  post: float = 300.0
  abort: float = 180.0
  status: float = 30.0
  release: float = 120.0


@dataclasses.dataclass(frozen=True)
class WorkerRoundReport:
  """Per-destination outcome of one round.

  `phase` is the worker's last known self-reported phase ("unknown" when the
  worker could not be reached at all). Version consistency and health are
  separate axes: a worker can hold the old weights and still be down.
  """

  worker_id: str
  phase: str
  error: str = ""
  needs_restart: bool = False


@dataclasses.dataclass(frozen=True)
class WeightSyncResult:
  """Outcome of one coordinated round."""

  policy_version: int
  round_index: int
  req_id: str
  uuid: int
  state: RoundState
  transfer: Optional[weight_sync.TransferResult]
  source_units: tuple[weight_sync.WorkUnitId, ...]
  destination_units: tuple[weight_sync.WorkUnitId, ...]
  workers: tuple[WorkerRoundReport, ...] = ()
  failures: tuple[str, ...] = ()

  @property
  def success(self) -> bool:
    return self.state is RoundState.COMMITTED


class WeightSyncError(RuntimeError):
  """A round did not commit. Carries the round's result.

  Raised rather than returned: an unchecked `success=False` silently leaves
  the RL loop training against stale samplers. The caller decides between
  retrying, restarting workers, and stopping the loop; `result.state` and
  `result.workers` say which of those are on the table.
  """

  def __init__(self, message: str, result: WeightSyncResult):
    super().__init__(message)
    self.result = result


class WeightSyncCoordinator:
  """Runs weight sync rounds against a `WorkerRegistry`.

  Async throughout: destination phases run concurrently (serializing them
  multiplies the fleet's downtime by the worker count), and the blocking
  `handler.transfer` runs in an executor so it does not stall the loop.

  Single-flight: a second `sync()` while one is in flight raises immediately.
  Overlapping rounds would have destinations freeing their KV cache while a
  transfer into them is still running.

  Poisoning: a round ending in PARTIALLY_COMMITTED, FAILED_NEEDS_RESTART, or
  UNKNOWN_TRANSFER_STATE poisons the coordinator; `sync()` raises until
  `reset_after_recovery()` is called. Driving new rounds into workers in
  unknown states compounds the damage and can regress policy versions.

  Round uuids are monotonic PER COORDINATOR INSTANCE. After a coordinator
  restart, construct with `first_uuid` greater than any uuid the fleet has
  seen (or restart the workers too); otherwise every worker's tracker
  rejects the new coordinator's rounds as stale.

  Transport-specific planner knobs are handler configuration rather than
  worker request data. The coordinator passes only stable work-unit ids, the
  round request id, and its monotonic generation.

  `AbstractRLEngine.sync_weights()` remains the outward-facing entry point;
  an engine implements it by calling `sync()` here and deciding what a raised
  `WeightSyncError` means for the training loop.
  """

  def __init__(
      self,
      registry: worker_registry.WorkerRegistry,
      handler: weight_sync.WeightSyncHandler,
      source_role: str = datatypes.Role.ACTOR.value,
      destination_role: str = datatypes.Role.ROLLOUT.value,
      controller_id: str = "",
      req_id_prefix: str = "wsync",
      first_uuid: int = 1,
      timeouts: Optional[PhaseTimeouts] = None,
  ):
    self._registry = registry
    self._handler = handler
    self._source_role = source_role
    self._destination_role = destination_role
    self._controller_id = controller_id
    self._req_id_prefix = req_id_prefix
    self._timeouts = timeouts or PhaseTimeouts()

    self._round_index = 0
    self._next_uuid = first_uuid
    self._in_flight = False
    self._poisoned: Optional[str] = None
    self._last_committed_version: Optional[int] = None

  @property
  def round_index(self) -> int:
    """Number of rounds started so far."""
    return self._round_index

  @property
  def last_committed_version(self) -> Optional[int]:
    return self._last_committed_version

  @property
  def poisoned(self) -> Optional[str]:
    """Why the coordinator refuses new rounds, or None."""
    return self._poisoned

  def reset_after_recovery(self) -> None:
    """Clears the poison after the operator restored the fleet.

    Deliberately explicit: the coordinator cannot verify the recovery, it can
    only insist a human (or supervisor) claims it happened.
    """
    logging.warning("coordinator poison cleared: %s", self._poisoned)
    self._poisoned = None

  # ---------------------------------------------------------------- lookup

  def _members(self, role: str) -> list[Any]:
    group = self._registry.group(role)
    if group.is_empty():
      raise ValueError(f"no workers registered for role {role!r}")
    return list(group)

  def _sources(self) -> list[WeightSyncSource]:
    members = self._members(self._source_role)
    for member in members:
      if not isinstance(member, WeightSyncSource):
        raise TypeError(
            f"worker {_worker_id(member)!r} serves role"
            f" {self._source_role!r} but does not implement"
            " prepare_weight_sync / release_weight_sync"
        )
    return members

  def _destinations(self) -> list[WeightSyncDestination]:
    members = self._members(self._destination_role)
    for member in members:
      if not isinstance(member, WeightSyncDestination):
        raise TypeError(
            f"worker {_worker_id(member)!r} serves role"
            f" {self._destination_role!r} but does not implement the"
            " bind / metadata / pre_weight_sync / weight_sync /"
            " post_weight_sync / abort_weight_sync / get_weight_sync_status"
            " destination protocol"
        )
    return members

  # ---------------------------------------------------------------- request

  def build_request(
      self,
      policy_version: int,
      req_id: str = "",
      uuid: int = 0,
      round_index: int = 0,
      source_metadata: Any = None,
      **extra_config,
  ) -> datatypes.WeightSyncRequest:
    """Builds the request carried through every phase of one round.

    The round's identity (req_id, uuid, round_index) rides in `extra_config`:
    workers key their `WorkerRoundTracker` on it.
    """
    return datatypes.WeightSyncRequest(
        controller_id=self._controller_id,
        policy_version=policy_version,
        source_metadata=source_metadata,
        extra_config=dict(
            req_id=req_id, uuid=uuid, round_index=round_index, **extra_config
        ),
    )

  # ------------------------------------------------------------- phase plumbing

  async def _worker_phase(
      self,
      destination: WeightSyncDestination,
      request: datatypes.WeightSyncRequest,
  ) -> Optional[str]:
    """The worker's self-reported phase for THIS round, or None if unknown."""
    try:
      report = await asyncio.wait_for(
          destination.get_weight_sync_status(), self._timeouts.status
      )
    except Exception:  # pylint: disable=broad-except
      return None
    if report.get("req_id") == request.extra_config.get(
        "req_id"
    ) and report.get("uuid") == request.extra_config.get("uuid"):
      return report.get("phase")
    return None

  async def _call_phase(
      self,
      destination: WeightSyncDestination,
      method_name: str,
      request: datatypes.WeightSyncRequest,
      timeout: float,
  ) -> Optional[BaseException]:
    """Runs one phase method on one destination. Returns its failure, if any.

    ANY failure -- timeout, connection reset, transport error -- is reconciled
    against the worker's own round report before being declared real: an RPC
    error means the reply is lost, not that the work did not happen, and for
    `post_weight_sync` in particular the work may be a completed publish.
    """
    method = getattr(destination, method_name)
    accepts = _PHASE_ACCEPTS.get(method_name)
    try:
      await asyncio.wait_for(method(request), timeout)
      return None
    except asyncio.CancelledError:
      raise
    except (KeyboardInterrupt, SystemExit):
      raise  # interpreter shutdown is not a phase failure to reconcile
    except BaseException as e:  # pylint: disable=broad-except
      if accepts is not None:
        phase = await self._worker_phase(destination, request)
        if phase is not None and phase in accepts:
          logging.info(
              "%s on %s failed (%r) but the worker reports phase %s for this"
              " round; treating as success",
              method_name,
              _worker_id(destination),
              e,
              phase,
          )
          return None
      return e

  async def _phase_results(
      self,
      destinations: Sequence[WeightSyncDestination],
      method_name: str,
      request: datatypes.WeightSyncRequest,
      timeout: float,
  ) -> list[tuple[WeightSyncDestination, Optional[BaseException]]]:
    errors = await asyncio.gather(*[
        self._call_phase(d, method_name, request, timeout) for d in destinations
    ])
    return list(zip(destinations, errors))

  async def _phase_on_all(
      self,
      destinations: Sequence[WeightSyncDestination],
      method_name: str,
      request: datatypes.WeightSyncRequest,
      timeout: float,
  ) -> list[str]:
    return [
        f"{_worker_id(destination)}: {method_name}: {error!r}"
        for destination, error in await self._phase_results(
            destinations, method_name, request, timeout
        )
        if error is not None
    ]

  async def _abort_all(
      self,
      destinations: Sequence[WeightSyncDestination],
      request: datatypes.WeightSyncRequest,
  ) -> list[str]:
    """Rolls back every destination given, returning rollback failures.

    Every destination the round touched is aborted, not only the ones whose
    phase call returned cleanly: a worker that raised halfway through pre may
    already have gated admission or freed its cache, and abort is idempotent
    by contract, so over-calling is safe where under-calling strands a worker.
    """
    return await self._phase_on_all(
        destinations, "abort_weight_sync", request, self._timeouts.abort
    )

  # ---------------------------------------------------------------- the round

  async def sync(
      self,
      policy_version: int = 0,
      **extra_config,
  ) -> WeightSyncResult:
    """Runs one full round; returns only if it committed.

    Anything short of COMMITTED raises `WeightSyncError` carrying the result;
    by then rollback (where safe) has already run and the result's state and
    per-worker reports say what the fleet looks like.

    Args:
      policy_version: Version of the weights being pushed. Must not regress
        below the last committed version; pushing the same version again is
        allowed (it is how a failed round is retried).
      **extra_config: Carried to the workers in the request.
    """
    if self._poisoned:
      raise RuntimeError(
          f"coordinator is poisoned ({self._poisoned}); recover the fleet and"
          " call reset_after_recovery() before starting another round"
      )
    if self._in_flight:
      raise RuntimeError(
          "a weight sync round is already in flight; rounds must not overlap"
      )
    if (
        self._last_committed_version is not None
        and policy_version < self._last_committed_version
    ):
      raise ValueError(
          f"policy_version {policy_version} regresses below the last"
          f" committed version {self._last_committed_version}"
      )

    self._in_flight = True
    start_time = time.monotonic()
    try:
      return await self._run_round(policy_version, extra_config)
    finally:
      elapsed_time = time.monotonic() - start_time
      logging.info("Weight sync finished in %.2f seconds.", elapsed_time)
      self._in_flight = False

  async def _run_round(
      self,
      policy_version: int,
      extra_config: dict[str, Any],
  ) -> WeightSyncResult:
    round_index = self._round_index
    self._round_index += 1
    uuid = self._next_uuid
    self._next_uuid += 1
    req_id = f"{self._req_id_prefix}-v{policy_version}-r{round_index}"

    sources = self._sources()
    destinations = self._destinations()

    state = RoundState.PREPARING
    transfer: Optional[weight_sync.TransferResult] = None
    source_units: tuple[weight_sync.WorkUnitId, ...] = ()
    destination_units: tuple[weight_sync.WorkUnitId, ...] = ()
    failures: list[str] = []
    worker_reports: dict[str, WorkerRoundReport] = {}
    quiesce_attempted = False
    release_source = True
    # True exactly while the blocking transfer may be running in its executor
    # thread. Cancellation cannot reach that thread, so a cancel landing in
    # this window gets the timeout treatment, not the rollback treatment.
    transfer_in_flight = False

    def result() -> WeightSyncResult:
      return WeightSyncResult(
          policy_version=policy_version,
          round_index=round_index,
          req_id=req_id,
          uuid=uuid,
          state=state,
          transfer=transfer,
          source_units=source_units,
          destination_units=destination_units,
          workers=tuple(worker_reports.values()),
          failures=tuple(failures),
      )

    def fail(message: str) -> WeightSyncError:
      return WeightSyncError(
          f"round {round_index} (req_id {req_id}, uuid {uuid}): {message};"
          f" final state {state.value}",
          result(),
      )

    def poison_if_needed() -> None:
      if state in _POISONING_STATES and not self._poisoned:
        self._poisoned = (
            f"round {round_index} (req_id {req_id}) ended {state.value}"
        )

    async def record_workers(final_error: str = "") -> None:
      for destination in destinations:
        wid = _worker_id(destination)
        if wid in worker_reports:
          continue
        phase = await self._worker_phase(
            destination, prepared_request or request
        )
        worker_reports[wid] = WorkerRoundReport(
            worker_id=wid,
            phase=phase or "unknown",
            error=final_error if phase is None else "",
            needs_restart=phase is None and quiesce_attempted,
        )

    request = self.build_request(
        policy_version,
        req_id=req_id,
        uuid=uuid,
        round_index=round_index,
        **extra_config,
    )
    prepared_request: Optional[datatypes.WeightSyncRequest] = None

    try:
      # Everything up to `pre` runs while the destinations are still serving:
      # bind (a no-op on an already-bound worker), metadata collection, and
      # registration cost no downtime. Failures here need no rollback either.
      try:
        await asyncio.gather(*[
            asyncio.wait_for(d.bind_weight_sync(), self._timeouts.bind)
            for d in destinations
        ])
        dst_meta_lists = await asyncio.gather(*[
            asyncio.wait_for(
                d.get_weight_sync_metadata(), self._timeouts.metadata
            )
            for d in destinations
        ])
        # Metadata is collected exactly once and the same objects flow to both
        # registration and the request. Collecting twice would hand the
        # controller endpoints from a different rebind than the one staged.
        src_meta_lists = await asyncio.gather(*[
            asyncio.wait_for(
                s.prepare_weight_sync(request), self._timeouts.source_prepare
            )
            for s in sources
        ])
      except asyncio.CancelledError:
        raise
      except Exception as e:  # pylint: disable=broad-except
        failures.append(f"pre-quiesce setup: {e!r}")
        raise fail(
            "bind/metadata/source-prepare failed before any destination was"
            " quiesced; no rollback needed"
        ) from e

      src_metadata = [
          weight_sync.dict_to_metadata(m)
          for per_source in src_meta_lists
          for m in per_source
      ]
      dst_metadata = [
          weight_sync.dict_to_metadata(m)
          for per_dest in dst_meta_lists
          for m in per_dest
      ]
      if not src_metadata or not dst_metadata:
        failures.append(
            f"metadata: {len(src_metadata)} source, {len(dst_metadata)}"
            " destination unit(s)"
        )
        raise fail("metadata collection returned an empty side")
      source_units = tuple(m.unit for m in src_metadata)
      destination_units = tuple(m.unit for m in dst_metadata)

      # Manifest preflight, before registration and before any downtime:
      # the controller pairs variables by exact name and silently skips
      # mismatches, so a bad wire name or shape must stop the round HERE --
      # afterwards it degrades into a lost tensor under a green round.
      if not isinstance(self.handler, NullHandler):
        preflight_problems = _manifest_mismatches(src_metadata, dst_metadata)
        if preflight_problems:
          failures.extend(preflight_problems)
          raise fail(
              "manifest preflight failed before any destination was quiesced;"
              " no rollback needed"
          )
      else:
        logging.debug(
            "NullHandler active; skipping manifest preflight mismatch checks."
        )

      loop = asyncio.get_running_loop()
      # Registrations are independent replacement writes into the handler's
      # owned controller, so they run concurrently. `return_exceptions=True` is
      # load-bearing rather than stylistic: an executor future cannot be
      # cancelled once it is running, so returning on the first exception
      # would leave the other registrations still writing into the
      # controller while this round unwinds and the next one starts. Every
      # future settles before any error is acted on. The shield exists for the
      # same reason cancellation-side: a cancel cannot stop the executor
      # threads, only orphan them -- and an orphaned registration landing
      # AFTER the next round re-registers the same unit would overwrite
      # fresh ports/layout with this round's stale ones. On cancel, every
      # registration is awaited to completion before the cancellation
      # propagates.
      registration_metadata = (*src_metadata, *dst_metadata)
      registration_gather = asyncio.gather(
          *[
              loop.run_in_executor(
                  None, self._handler.register_work_unit, metadata
              )
              for metadata in registration_metadata
          ],
          return_exceptions=True,
      )
      try:
        registrations = await asyncio.shield(registration_gather)
      except asyncio.CancelledError:
        await registration_gather
        raise
      registration_errors: list[BaseException] = []
      for metadata, result_or_error in zip(
          registration_metadata, registrations
      ):
        if isinstance(result_or_error, asyncio.CancelledError):
          raise result_or_error  # cancellation is never a phase failure
        if isinstance(result_or_error, BaseException):
          registration_errors.append(result_or_error)
          # `WeightSyncError` carries these failures structurally, but an
          # uncaught exception renderer normally prints only its outer
          # message. Log the unit and original traceback here so a production
          # registration failure remains diagnosable.
          logging.error(
              "work-unit registration failed: unit=%s shards=%r control=%s"
              " mesh_shape=%s layout=%s global_shape=%s item_size=%s"
              " variables=%d error=%r",
              metadata.unit,
              metadata.shards,
              metadata.control_plane_rpc_address,
              metadata.mesh_shape,
              metadata.layout,
              metadata.global_shape,
              metadata.item_size,
              len(metadata.variables),
              result_or_error,
              exc_info=(
                  type(result_or_error),
                  result_or_error,
                  result_or_error.__traceback__,
              ),
          )
      if registration_errors:
        failures.extend(f"registration: {e!r}" for e in registration_errors)
        raise fail(
            "work-unit registration failed before any destination was"
            " quiesced; no rollback needed"
        )

      prepared_request = self.build_request(
          policy_version,
          req_id=req_id,
          uuid=uuid,
          round_index=round_index,
          source_metadata=tuple(src_metadata),
          **extra_config,
      )

      # --- downtime starts here ---
      quiesce_attempted = True
      pre_failures = await self._phase_on_all(
          destinations, "pre_weight_sync", prepared_request, self._timeouts.pre
      )
      if pre_failures:
        failures += pre_failures
        state = await self._rollback(destinations, prepared_request, failures)
        poison_if_needed()
        await record_workers("pre_weight_sync failed")
        raise fail("pre_weight_sync failed")
      state = RoundState.PREPARED

      state = RoundState.TRANSFERRING
      transfer_in_flight = True
      try:
        transfer = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                functools.partial(
                    self._handler.transfer,
                    src_units=list(source_units),
                    dst_units=list(destination_units),
                    req_id=req_id,
                    generation=uuid,
                ),
            ),
            self._timeouts.transfer,
        )
        transfer_in_flight = False
      except (
          asyncio.TimeoutError,
          weight_sync.TransferOutcomeUnknownError,
      ) as e:
        # Two triggers, one meaning: the coordinator's own deadline elapsed
        # (the executor thread is still running), or the transport reported
        # its RPC outcome unknown (reply lost, server possibly still
        # executing). Either way the transfer may still be WRITING into
        # destination staging. Aborting destinations would discard buffers a
        # live transfer is filling; releasing source staging would pull
        # memory out from under it. Neither is safe, so neither happens: the
        # round parks in UNKNOWN_TRANSFER_STATE, the coordinator poisons
        # itself, and recovery is worker restarts plus reset_after_recovery().
        # (On Python 3.11+ asyncio.TimeoutError is the builtin TimeoutError,
        # so an unwrapped transport timeout also lands here rather than in the
        # rollback branch below.)
        failures.append(
            f"transfer: outcome unknown ({e!r}); the transfer may still be"
            " running"
        )
        state = RoundState.UNKNOWN_TRANSFER_STATE
        release_source = False
        poison_if_needed()
        await record_workers("transfer timed out")
        raise fail(
            "transfer timed out; destinations NOT aborted and source staging"
            " NOT released because the transfer may still be running"
        ) from e
      except asyncio.CancelledError:
        raise
      except Exception as e:  # pylint: disable=broad-except
        # The call returned (by raising): the thread is done, rollback is safe.
        transfer_in_flight = False
        failures.append(f"transfer: {e!r}")
        state = await self._rollback(destinations, prepared_request, failures)
        poison_if_needed()
        await record_workers("transfer raised")
        raise fail("transfer raised") from e

      if not transfer.success:
        failures.append(f"transfer: {transfer.message}")
        state = await self._rollback(destinations, prepared_request, failures)
        poison_if_needed()
        await record_workers("transfer failed")
        raise fail("transfer failed")

      state = RoundState.H2D_IN_PROGRESS
      h2d_failures = await self._phase_on_all(
          destinations, "weight_sync", prepared_request, self._timeouts.h2d
      )
      if h2d_failures:
        # Staging-only H2D means the serving copy is untouched everywhere,
        # so rolling all destinations back is safe even though some finished.
        failures += h2d_failures
        state = await self._rollback(destinations, prepared_request, failures)
        poison_if_needed()
        await record_workers("weight_sync failed")
        raise fail("weight_sync (H2D) failed")

      state = RoundState.PENDING_COMMIT
      post_results = await self._phase_results(
          destinations,
          "post_weight_sync",
          prepared_request,
          self._timeouts.post,
      )
      failed_posts = [(d, e) for d, e in post_results if e is not None]
      if failed_posts:
        state = await self._resolve_post_failures(
            destinations,
            failed_posts,
            prepared_request,
            failures,
            worker_reports,
        )
        if state is not RoundState.COMMITTED:
          poison_if_needed()
          await record_workers("post_weight_sync failed")
          raise fail(f"post_weight_sync did not commit everywhere; see workers")

      state = RoundState.COMMITTED
      self._last_committed_version = policy_version
      for destination in destinations:
        wid = _worker_id(destination)
        worker_reports.setdefault(
            wid, WorkerRoundReport(worker_id=wid, phase="committed")
        )
      return result()

    except asyncio.CancelledError:
      # The caller's task got cancelled mid-round.
      if transfer_in_flight:
        # The cancel landed while the blocking transfer may still be running
        # in its executor thread, which cancellation cannot reach. This is
        # the timeout case with a different trigger: aborting destinations or
        # releasing source staging could destroy memory the live transfer is
        # using, so neither happens and the coordinator poisons itself.
        failures.append(
            "cancelled during transfer; the executor thread may still be"
            " writing"
        )
        state = RoundState.UNKNOWN_TRANSFER_STATE
        release_source = False
        poison_if_needed()
        logging.error(
            "round %d cancelled during transfer: destinations NOT aborted,"
            " source staging NOT released; coordinator poisoned",
            round_index,
        )
        raise
      # Quiesced destinations must not be stranded unserving; the recovery is
      # shielded from the cancellation so it actually runs.
      if (
          quiesce_attempted
          and prepared_request is not None
          and state not in (RoundState.UNKNOWN_TRANSFER_STATE,)
      ):
        if state is RoundState.PENDING_COMMIT:
          # Post RPCs were in flight: some workers may already have
          # published. A blanket rollback would no-op on those (abort after
          # commit protects the publish) while unwinding the rest -- a
          # mixed-version fleet with nobody told. Sort by each worker's own
          # report, abort only the unpublished, and poison if the fleet
          # ended up split.
          async def recover_pending_commit() -> None:
            nonlocal state
            # The SAME classification as _resolve_post_failures, for the
            # same reason: `_worker_phase` returns None both for
            # "unreachable" and "on a different round", and nothing can be
            # inferred from None -- including that the worker did NOT
            # publish. Aborting it could roll back a publish that happened.
            # Only a positively confirmed quiesced-and-unpublished worker
            # is an abort candidate, on this path like every other.
            committed_ws: list[WeightSyncDestination] = []
            recovered_ws: list[WeightSyncDestination] = []
            abortable: list[WeightSyncDestination] = []
            unknown_ws: list[WeightSyncDestination] = []
            for destination in destinations:
              phase = await self._worker_phase(destination, prepared_request)
              if phase == "committed":
                committed_ws.append(destination)
              elif phase == "aborted":
                recovered_ws.append(destination)  # already rolled back
              elif phase in ("prepared", "h2d_done"):
                abortable.append(destination)
              else:
                unknown_ws.append(destination)
            for destination in unknown_ws:
              failures.append(
                  f"{_worker_id(destination)}: state unknown at"
                  " cancellation; not aborted (an abort could roll back a"
                  " publish that happened)"
              )
            abort_failures: list[str] = []
            if abortable and not committed_ws and not unknown_ws:
              # Uniform rollback is allowed only when the round PROVABLY
              # published nowhere. With a commit present, aborting an
              # h2d_done worker would reopen it serving the old policy next
              # to workers serving the new one -- the normal path refuses
              # exactly that (it leaves the worker quiesced with
              # needs_restart), and cancellation is no license for it. With
              # an unknown present, the unknown may BE a commit.
              abort_failures = await self._abort_all(
                  abortable, prepared_request
              )
              failures.extend(abort_failures)
              # A clean abort RPC includes the no-op-on-committed case (a
              # post that finished between classification and the abort), so
              # the reports are re-read: without this, an actually-committed
              # fleet would end unrecorded and unpoisoned.
              recheck = await asyncio.gather(
                  *[self._worker_phase(d, prepared_request) for d in abortable]
              )
              rechecked = list(zip(abortable, recheck))
              abortable = []
              for destination, phase in rechecked:
                if phase == "committed":
                  committed_ws.append(destination)
                  failures.append(
                      f"{_worker_id(destination)}: abort returned cleanly"
                      " but the worker reports committed; the publish"
                      " stands and was NOT rolled back"
                  )
                elif phase == "aborted":
                  recovered_ws.append(destination)
                else:
                  unknown_ws.append(destination)
                  failures.append(
                      f"{_worker_id(destination)}: rollback unconfirmed"
                      " after a clean abort; the worker reports"
                      f" {phase or 'unknown'!r}"
                  )
            elif abortable:
              for destination in abortable:
                failures.append(
                    f"{_worker_id(destination)}: left quiesced, not aborted:"
                    " the round has a commit or an unknown elsewhere, and"
                    " reopening this worker on the old policy would serve"
                    " mixed versions"
                )
            not_committed = abortable + recovered_ws + unknown_ws
            # Same precedence as the normal path: an unknown worker outranks
            # a version split, because "unknown" already contains the
            # possibility of one and nothing about the fleet can be stated
            # until that worker is recovered.
            if unknown_ws or abort_failures:
              state = RoundState.FAILED_NEEDS_RESTART
            elif committed_ws and not_committed:
              state = RoundState.PARTIALLY_COMMITTED
              logging.error(
                  "round %d cancelled during post: %d worker(s) committed,"
                  " %d did not; fleet holds mixed versions",
                  round_index,
                  len(committed_ws),
                  len(not_committed),
              )
            elif committed_ws:
              # Everyone published: the cancel lost the result, not the
              # round. Record the version so the next round's regression
              # guard stays truthful.
              self._last_committed_version = policy_version
            else:
              state = RoundState.ABORTED
            poison_if_needed()

          await asyncio.shield(recover_pending_commit())
        else:
          logging.warning(
              "round %d cancelled mid-flight; rolling destinations back",
              round_index,
          )
          abort_failures = await asyncio.shield(
              self._abort_all(destinations, prepared_request)
          )
          if abort_failures:
            # A dropped rollback failure would leave poisoned=None over a
            # worker that may be stuck unserving.
            failures.extend(abort_failures)
            state = RoundState.FAILED_NEEDS_RESTART
            poison_if_needed()
            logging.error(
                "cancelled-round rollback incomplete: %s", abort_failures
            )
          else:
            state = RoundState.ABORTED
      raise
    finally:
      if release_source:
        release_request = prepared_request or request
        # Shielded AND awaited to completion on cancellation. The shield
        # alone only stops the cancel from killing the release; the awaiting
        # coroutine still leaves immediately, so the release would run on
        # behind a round the caller already considers over -- overlapping the
        # next round's prepare, or a close() that frees what it is reading.
        release_gather = asyncio.gather(
            *[
                asyncio.wait_for(
                    s.release_weight_sync(release_request),
                    self._timeouts.release,
                )
                for s in sources
            ],
            return_exceptions=True,
        )
        try:
          release_errors = await asyncio.shield(release_gather)
        except asyncio.CancelledError:
          release_errors = await release_gather
          for source, error in zip(sources, release_errors):
            if isinstance(error, BaseException):
              logging.warning(
                  "source %s failed to release round %d staging during a"
                  " cancellation: %r",
                  _worker_id(source),
                  round_index,
                  error,
              )
          raise
        for source, error in zip(sources, release_errors):
          if isinstance(error, BaseException):
            logging.warning(
                "source %s failed to release round %d staging: %r",
                _worker_id(source),
                round_index,
                error,
            )
      else:
        logging.error(
            "round %d: source staging deliberately NOT released; a timed-out"
            " transfer may still be reading it",
            round_index,
        )

  async def _resolve_post_failures(
      self,
      destinations: Sequence[WeightSyncDestination],
      failed_posts: Sequence[tuple[WeightSyncDestination, BaseException]],
      request: datatypes.WeightSyncRequest,
      failures: list[str],
      worker_reports: dict[str, WorkerRoundReport],
  ) -> RoundState:
    """Classifies post failures by the worker's own report, then acts.

    `_call_phase` already treated workers reporting "committed" as successes,
    so what arrives here failed AND did not confirm the publish. Each is
    classified by its explicit report:

      "h2d_done"   the publish definitively did not happen; retry once
                   (post is idempotent for the round key).
      "aborted"    the worker rolled this round back (an out-of-band abort,
                   or its tracker refusing to commit an aborted round). It is
                   alive and consistent on the OLD weights -- a version
                   outcome, not a health problem. Never retried: the round
                   key is dead on that worker.
      unknown      the worker is unreachable or on another round; nothing can
                   be inferred, including that it did NOT publish. Marked
                   needs_restart, never retried, never aborted (an abort
                   could roll back a publish that actually happened).

    Outcome: everyone committed after retry -> COMMITTED. Any unknown ->
    FAILED_NEEDS_RESTART. Nobody committed and everyone alive -> rollback ->
    ABORTED. Some committed and some not -> PARTIALLY_COMMITTED.
    """
    retryable: list[WeightSyncDestination] = []
    rolled_back: list[WeightSyncDestination] = []
    unknown: list[WeightSyncDestination] = []
    for destination, error in failed_posts:
      phase = await self._worker_phase(destination, request)
      wid = _worker_id(destination)
      if phase == "committed":
        # The RPC failed but the worker finished publishing between
        # _call_phase's reconciliation poll and this one -- a timed-out post
        # can still be executing at the first poll. Same verdict as
        # _PHASE_ACCEPTS["post_weight_sync"]: success.
        logging.info(
            "%s: post_weight_sync failed (%r) but the worker now reports"
            " committed; treating as success",
            wid,
            error,
        )
        continue
      if phase == "h2d_done":
        retryable.append(destination)
      elif phase == "aborted":
        rolled_back.append(destination)
        failures.append(
            f"{wid}: post_weight_sync: {error!r} (worker aborted the round)"
        )
        worker_reports[wid] = WorkerRoundReport(
            worker_id=wid, phase="aborted", error=repr(error)
        )
      else:
        unknown.append(destination)
        failures.append(f"{wid}: post_weight_sync: {error!r} (state unknown)")
        worker_reports[wid] = WorkerRoundReport(
            worker_id=wid,
            phase=phase or "unknown",
            error=repr(error),
            needs_restart=True,
        )

    still_failed: list[WeightSyncDestination] = []
    if retryable:
      logging.warning(
          "retrying post_weight_sync on %d destination(s) that report"
          " h2d_done: %s",
          len(retryable),
          [_worker_id(d) for d in retryable],
      )
      retry_results = await self._phase_results(
          destinations=retryable,
          method_name="post_weight_sync",
          request=request,
          timeout=self._timeouts.post,
      )
      for destination, error in retry_results:
        if error is not None:
          wid = _worker_id(destination)
          # Classify by the worker's CURRENT report, not the pre-retry one:
          # the same late-completion and out-of-band-abort races apply here,
          # and so does the unknown rule -- a worker that was h2d_done before
          # the retry but cannot confirm it now might have published in
          # between. Only a POSITIVELY confirmed h2d_done is an abort
          # candidate; unknown is never aborted, here or anywhere.
          phase = await self._worker_phase(destination, request)
          if phase == "committed":
            continue
          if phase == "aborted":
            rolled_back.append(destination)
            failures.append(
                f"{wid}: post_weight_sync retry: {error!r} (worker aborted)"
            )
            worker_reports[wid] = WorkerRoundReport(
                worker_id=wid, phase="aborted", error=repr(error)
            )
            continue
          if phase == "h2d_done":
            still_failed.append(destination)
            failures.append(f"{wid}: post_weight_sync retry: {error!r}")
            worker_reports[wid] = WorkerRoundReport(
                worker_id=wid,
                phase="h2d_done",
                error=repr(error),
                needs_restart=True,
            )
            continue
          unknown.append(destination)
          failures.append(
              f"{wid}: post_weight_sync retry: {error!r} (state unknown"
              " after retry)"
          )
          worker_reports[wid] = WorkerRoundReport(
              worker_id=wid,
              phase=phase or "unknown",
              error=repr(error),
              needs_restart=True,
          )

    if not unknown and not still_failed and not rolled_back:
      return RoundState.COMMITTED

    committed_count = (
        len(destinations) - len(unknown) - len(still_failed) - len(rolled_back)
    )
    if unknown:
      return RoundState.FAILED_NEEDS_RESTART
    if committed_count == 0:
      # Nobody published and everyone is alive; a uniform rollback is safe
      # (already-aborted workers treat the extra abort as an idempotent
      # no-op).
      to_abort = list(still_failed) + list(rolled_back)
      rollback_state = await self._rollback(
          to_abort, request, failures, commit_possible=True
      )
      if rollback_state in (RoundState.ABORTED, RoundState.COMMITTED):
        # Every worker confirmed one terminal: the interim per-worker
        # records (h2d_done, needs_restart) are now stale and would direct
        # restarts of healthy workers. The result must agree with its own
        # state.
        phase_name = (
            "aborted" if rollback_state is RoundState.ABORTED else "committed"
        )
        for destination in to_abort:
          wid = _worker_id(destination)
          previous = worker_reports.get(wid)
          worker_reports[wid] = WorkerRoundReport(
              worker_id=wid,
              phase=phase_name,
              error=previous.error if previous else "",
              needs_restart=False,
          )
      return rollback_state
    return RoundState.PARTIALLY_COMMITTED

  async def _rollback(
      self,
      destinations: Sequence[WeightSyncDestination],
      request: datatypes.WeightSyncRequest,
      failures: list[str],
      *,
      commit_possible: bool = False,
  ) -> RoundState:
    """Aborts every destination; reports honestly if any refused.

    A failed abort is already reconciled against the worker's own report by
    `_call_phase` (`_PHASE_ACCEPTS`), so a lost reply over a real rollback
    does not reach `abort_failures` -- anything that does means the worker
    did not report "aborted".

    `commit_possible` says whether any destination can have PUBLISHED this
    round (post ran somewhere). It decides how much a clean abort RPC
    proves:

      False (pre/H2D failures): no post ever ran, so the tracker's only
        clean-RPC blind spot -- the no-op on a committed round -- cannot
        exist. A clean abort IS the rollback; a later status flake is not
        escalated, because poisoning the coordinator over a worker that
        provably held no publish would turn rollbacks into outages
        (`test_stale_round_report_does_not_count` pins this).
      True (post failures): a clean abort might be that no-op over a post
        that completed between classification and the abort. The reports
        are re-read and only their word counts: confirmed aborted
        everywhere -> ABORTED; any commit -> the publish stands
        (PARTIALLY_COMMITTED, or COMMITTED if every single one turns out
        published); anything unconfirmable -> FAILED_NEEDS_RESTART, because
        "aborted" and "possibly serving the new weights" must never be the
        same verdict.
    """
    abort_failures = await self._abort_all(destinations, request)
    if abort_failures:
      failures += abort_failures
      logging.error(
          "rollback incomplete: %d destination(s) failed to abort and may be"
          " stuck unserving: %s",
          len(abort_failures),
          abort_failures,
      )
      return RoundState.FAILED_NEEDS_RESTART
    if not commit_possible:
      return RoundState.ABORTED
    phases = await asyncio.gather(
        *[self._worker_phase(d, request) for d in destinations]
    )
    published = [
        _worker_id(d)
        for d, phase in zip(destinations, phases)
        if phase == "committed"
    ]
    unconfirmed = [
        _worker_id(d)
        for d, phase in zip(destinations, phases)
        if phase not in ("committed", "aborted")
    ]
    if published:
      failures.extend(
          f"{wid}: abort returned cleanly but the worker reports committed;"
          " the publish stands and was NOT rolled back"
          for wid in published
      )
    if unconfirmed:
      failures.extend(
          f"{wid}: rollback unconfirmed after a clean abort; the worker's"
          " report does not say aborted"
          for wid in unconfirmed
      )
    if published and len(published) == len(destinations):
      # Every "rollback target" turns out to have published: the round
      # committed everywhere and the aborts were no-ops. Report the truth.
      return RoundState.COMMITTED
    if published:
      logging.error(
          "rollback found %d destination(s) already published: the fleet"
          " holds mixed versions",
          len(published),
      )
      return RoundState.PARTIALLY_COMMITTED
    if unconfirmed:
      logging.error(
          "rollback unconfirmed on %d destination(s); a clean abort here"
          " can be a no-op over a completed publish, so this is not a"
          " certified rollback",
          len(unconfirmed),
      )
      return RoundState.FAILED_NEEDS_RESTART
    return RoundState.ABORTED


class NoOpWeightSyncCoordinator:
  """Protocol-only no-op coordinator that synchronizes policy versions without tensor transfer."""

  def __init__(self, rollout_handles: Sequence[Any] | None = None):
    self._policy_version = 0
    self._rollout_handles = list(rollout_handles or [])

  async def sync(
      self, policy_version: int | None = None, **kwargs: Any
  ) -> WeightSyncResult:
    """Updates the policy version and broadcasts it to rollout workers."""
    del kwargs
    version = (
        policy_version
        if policy_version is not None
        else self._policy_version + 1
    )
    self._policy_version = version
    logging.info(
        "[NoopWeightSyncCoordinator] Syncing policy_version -> %d (noop)...",
        version,
    )
    for handle in self._rollout_handles:
      try:
        if hasattr(handle, "asubmit"):
          await handle.asubmit("set_policy_version", version=version)
        elif hasattr(handle, "submit"):
          handle.submit("set_policy_version", version=version)
      except Exception as exc:  # pylint: disable=broad-except
        logging.warning(
            "[NoopWeightSyncCoordinator] Failed to set policy_version on worker: %s",
            exc,
        )
    return WeightSyncResult(
        policy_version=version,
        round_index=version,
        req_id=f"noop-r{version}",
        uuid=version,
        state=RoundState.COMMITTED,
        transfer=None,
        source_units=(),
        destination_units=(),
    )

