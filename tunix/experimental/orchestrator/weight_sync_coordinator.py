"""Drives weight sync rounds across the orchestrator's worker registry.

`RaidenHandler` knows how to talk to the transport; it does not know who the
participants are. This module supplies that half: it looks workers up by role
in the `WorkerRegistry`, collects transport metadata from each side exactly
once per round, registers every work unit, and drives the round.

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
    [handler.transfer]  --------->  (passive; bytes land in host staging)
                                    weight_sync()
                                      H2D into the STAGING copy only;
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

Worker-side contract: workers keep their current round keyed by the
(req_id, uuid) carried in `WeightSyncRequest.extra_config`, make every phase
idempotent for that key, reject phase calls carrying a stale key, and report
their round through `get_weight_sync_round()`. `WorkerRoundTracker` in this
module implements exactly that and is what the fakes and the on-Borg mock
embed. The report is what the coordinator consults whenever a phase call
fails: an RPC error means "the reply is lost", not "the work did not happen".
One more obligation the tracker cannot carry for the worker: the coordinator
never issues overlapping phase calls for a live round, but a TIMED-OUT phase
RPC may still be executing when the rollback's abort arrives, so destinations
must serialize their phase bodies themselves (the on-Borg samplers hold an
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
import logging
import threading
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.worker import abstract_worker

from tunix.experimental.orchestrator import weight_sync


TRAINER_ROLE = "trainer"
ROLLOUT_ROLE = "rollout"

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
# interchangeable — the tracker special-cases the conflict (commit after abort
# is refused, abort after commit is a no-op) before this ordering is consulted.
_PHASE_ORDER = {
    "idle": 0,
    "prepared": 1,
    "h2d_done": 2,
    "committed": 3,
    "aborted": 3,
}


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
  reuses the current uuid under a different req_id, or when the requested
  phase conflicts with a terminal phase this round already reached
  (committing an aborted round).
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
  resets the tracker: the coordinator's uuids are monotonic, so newer is
  always the round to follow.

  `complete` is separate from `admit` on purpose: a worker that crashes
  between doing the work and recording it will re-report the previous phase,
  the coordinator will retry, and the work must tolerate re-running. That is
  the crash-consistency window; hiding it inside one call would not close it.

  Thread safety: the tracker serializes its own state, because a remote
  execution server is not guaranteed to deliver phase RPCs on one thread. The
  phase WORK around it is the worker's to serialize — a timed-out phase RPC
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
      if uuid > self._uuid:
        # A new round supersedes whatever came before it.
        self._req_id, self._uuid, self._phase = req_id, uuid, "idle"
        self._policy_version = getattr(sync_request, "policy_version", -1)
      # The two terminals are not ordered against each other. Committing an
      # aborted round is refused loudly — the staging this round would publish
      # was already discarded. Aborting a committed round is a no-op — the
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


@runtime_checkable
class WeightSyncSource(Protocol):
  """The trainer side of a round.

  The source's synchronizer state is per-round: JAX hands back new arrays after
  each optimizer step, so endpoints from a previous round are stale by
  construction. That is why metadata is the return value of the per-round
  prepare call rather than a separately polled property; the same objects are
  registered and carried in the round's request, so what the controller holds
  is exactly what was staged.
  """

  async def prepare_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> Sequence[weight_sync.RaidenWorkUnitMetadata]:
    """Stages this round's weights and returns their transport metadata.

    Rebinds the synchronizer to the current arrays and runs the D2H copy into
    host staging. Returns one metadata entry per physical host (per listener):
    Raiden wants one work unit per host, so a multi-host source returns
    several. Wire-safe values only; no device arrays.
    """
    ...

  async def release_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> Any:
    """Releases this round's staging. Called on every exit path except an
    UNKNOWN_TRANSFER_STATE round (a possibly-live transfer may still be
    reading the staging). Idempotent, and must be safe to call while a
    timed-out prepare for the same round is still running remotely."""
    ...


@runtime_checkable
class WeightSyncDestination(Protocol):
  """The sampler side of a round.

  `pre_weight_sync` / `weight_sync` / `post_weight_sync` are `RolloutWorker`'s
  existing method names used with their documented meanings. What this
  protocol adds on top of the names is the contract that makes rollback
  possible (see module docstring): real admission gating in pre, staging-only
  H2D in weight_sync, atomic idempotent publish in post, an abort path, and
  `WorkerRoundTracker` semantics behind every phase.
  """

  async def bind_weight_sync(self) -> None:
    """Builds the synchronizer and binds its ports if not already bound.

    Idempotent, and called every round: on an already-bound worker it must be
    a no-op that keeps existing ports, because rebinding would invalidate the
    addresses the controller holds. A restarted worker binds fresh ports here
    and the round's registration picks them up.

    Deliberately not named `initialize`: `Worker.initialize` is an abstract
    method on the base class, driven by `LifecycleDriver.bring_up`, and this
    is a distinct later step that needs the model arrays to exist.
    """
    ...

  async def get_weight_sync_metadata(
      self,
  ) -> Sequence[weight_sync.RaidenWorkUnitMetadata]:
    """Transport metadata for this worker, one entry per physical host.

    Called while the worker is still serving; collection and registration cost
    no downtime. Wire-safe values only; no device arrays.
    """
    ...

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs: Any) -> Any:
    """Quiesces the worker so the arriving weights have somewhere to land.

    Must actually gate admission: stop accepting new requests, drain or cancel
    in-flight ones, drop the prefix cache, free the KV cache. The worker is
    not serving from the moment this returns until post or abort. Merely
    setting a pause flag does not satisfy this.
    """
    ...

  async def weight_sync(self, sync_request: Any = None, **kwargs: Any) -> Any:
    """Applies the received bytes: H2D from host staging to the staging copy.

    Called only after the transfer reported success. Must not touch the
    serving copy; records the pending policy version for post to publish.
    """
    ...

  async def post_weight_sync(self, sync_request: Any = None, **kwargs: Any) -> Any:
    """Publishes the pending weights atomically, rebuilds caches, resumes.

    Must be idempotent for the round key: a retry after a lost reply, or
    after a crash between publishing and recording, must converge to the
    same committed state rather than fail or double-apply.
    """
    ...

  async def abort_weight_sync(self, sync_request: Any = None, **kwargs: Any) -> Any:
    """Rolls back to serving the previous weights.

    Invalidates this round's staging — physically or logically: a
    destination whose synchronizer stays bound to the staging buffers cannot
    free them, and instead must guarantee nothing publishes them (the
    tracker's refusal to commit an aborted round is that guarantee).
    Rebuilds the KV cache, resumes admission on the old weights. Never
    touches the serving copy. Idempotent, and safe to call at any phase of a
    round including before pre completed.
    """
    ...

  async def get_weight_sync_round(self) -> Mapping[str, Any]:
    """The worker's view of its current round: `WorkerRoundTracker.report()`.

    Consulted by the coordinator whenever a phase RPC fails, to distinguish a
    lost reply from unfinished work.
    """
    ...


class RemoteParticipantProxy(abstract_worker.Worker):
  """Adapts an actor handle (`submit`/`asubmit`) to the participant protocols.

  `WorkerRegistry.register` takes an `abstract_worker.Worker` and calls
  `.info()` at registration; `GrpcRemoteActorHandle` exposes only method-name
  dispatch. This bridges the two — it IS a `Worker` (so the registry's typed
  signature holds under strict checking), and every method turns into
  `submit`/`asubmit(method_name, ...)` against the remote worker, lifecycle
  surface included, so a proxy can live in the same registry
  `LifecycleDriver` drives.

  Deadlines: the handle's own RPC deadline must be at least the largest
  `PhaseTimeouts` value in use — a `GrpcRemoteActorHandle` built with its
  default (~60s) timeout will cut off a 900s H2D long before the
  coordinator's deadline does. Configure `rpc_timeout_s` when creating the
  handle; the coordinator cannot reach into it.

  Everything crossing the handle goes through cloudpickle, so the wire-safety
  rules on the metadata types apply.
  """

  def __init__(self, handle: Any, info: datatypes.WorkerInfo):
    self._handle = handle
    self._info = info

  def info(self) -> datatypes.WorkerInfo:
    return self._info

  # Worker lifecycle passthroughs (LifecycleDriver compatibility).
  def initialize(self):
    return self._handle.submit("initialize")

  def compile(self, dummy_data: Any):
    return self._handle.submit("compile", dummy_data)

  def start(self):
    return self._handle.submit("start")

  def stop(self):
    return self._handle.submit("stop")

  def heartbeat(self):
    return self._handle.submit("heartbeat")

  # Source protocol.
  async def prepare_weight_sync(self, sync_request: Any = None, **kwargs: Any):
    return await self._handle.asubmit("prepare_weight_sync", sync_request, **kwargs)

  async def release_weight_sync(self, sync_request: Any = None, **kwargs: Any):
    return await self._handle.asubmit("release_weight_sync", sync_request, **kwargs)

  # Destination protocol.
  async def bind_weight_sync(self):
    return await self._handle.asubmit("bind_weight_sync")

  async def get_weight_sync_metadata(self):
    return await self._handle.asubmit("get_weight_sync_metadata")

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs: Any):
    return await self._handle.asubmit("pre_weight_sync", sync_request, **kwargs)

  async def weight_sync(self, sync_request: Any = None, **kwargs: Any):
    return await self._handle.asubmit("weight_sync", sync_request, **kwargs)

  async def post_weight_sync(self, sync_request: Any = None, **kwargs: Any):
    return await self._handle.asubmit("post_weight_sync", sync_request, **kwargs)

  async def abort_weight_sync(self, sync_request: Any = None, **kwargs: Any):
    return await self._handle.asubmit("abort_weight_sync", sync_request, **kwargs)

  async def get_weight_sync_round(self):
    return await self._handle.asubmit("get_weight_sync_round")


class RoundState(enum.Enum):
  """Round-level summary. Per-worker truth lives in `WeightSyncResult.workers`."""

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
  waiting only — a remote handle's own RPC deadline is separate and must be
  configured at least this large (see `RemoteParticipantProxy`).
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
  source_units: tuple[weight_sync.RaidenId, ...]
  destination_units: tuple[weight_sync.RaidenId, ...]
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

  `AbstractRLEngine.sync_weights()` remains the outward-facing entry point;
  an engine implements it by calling `sync()` here and deciding what a raised
  `WeightSyncError` means for the training loop.
  """

  def __init__(
      self,
      registry: worker_registry.WorkerRegistry,
      handler: weight_sync.WeightSyncHandler,
      source_role: str = TRAINER_ROLE,
      destination_role: str = ROLLOUT_ROLE,
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
            " post_weight_sync / abort_weight_sync / get_weight_sync_round"
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
          destination.get_weight_sync_round(), self._timeouts.status
      )
    except Exception:  # pylint: disable=broad-except
      return None
    if (
        report.get("req_id") == request.extra_config.get("req_id")
        and report.get("uuid") == request.extra_config.get("uuid")
    ):
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

    ANY failure — timeout, connection reset, transport error — is reconciled
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
        self._call_phase(d, method_name, request, timeout)
        for d in destinations
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
      *,
      expected_block_count: Optional[int] = None,
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
      expected_block_count: The number of physical pushes each destination
        listener expects under one transfer uuid (per destination unit, not
        a global total and not per device shard). None/0 — the default — is
        AUTO: Raiden's sender-side direct-schedule planning derives the
        count from its own schedule, the only authoritative source (one
        schedule entry can carry several pushes, so caller-side guesses
        from shapes are unreliable). The delegation is scoped to that path;
        other controller paths treat 0 differently, which is why the
        handler, not this coordinator, owns the wire encoding. A positive
        value overrides the transport, for experiments only.
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
    if expected_block_count is not None and expected_block_count < 0:
      raise ValueError(
          "expected_block_count must be None/0 (auto: the transport derives"
          " the count from its own schedule) or a positive override, got"
          f" {expected_block_count}"
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
    try:
      return await self._run_round(
          policy_version, expected_block_count, extra_config
      )
    finally:
      self._in_flight = False

  async def _run_round(
      self,
      policy_version: int,
      expected_block_count: int,
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
    source_units: tuple[weight_sync.RaidenId, ...] = ()
    destination_units: tuple[weight_sync.RaidenId, ...] = ()
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
        phase = await self._worker_phase(destination, prepared_request or request)
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

      src_metadata = [m for per_source in src_meta_lists for m in per_source]
      dst_metadata = [m for per_dest in dst_meta_lists for m in per_dest]
      if not src_metadata or not dst_metadata:
        failures.append(
            f"metadata: {len(src_metadata)} source, {len(dst_metadata)}"
            " destination unit(s)"
        )
        raise fail("metadata collection returned an empty side")
      source_units = tuple(m.unit for m in src_metadata)
      destination_units = tuple(m.unit for m in dst_metadata)

      loop = asyncio.get_running_loop()
      for metadata in (*src_metadata, *dst_metadata):
        await loop.run_in_executor(
            None, self._handler.register_work_unit, metadata
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
                    uuid=uuid,
                    expected_block_count=expected_block_count,
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
        # so an unwrapped socket timeout escaping the transport also lands
        # here rather than in the rollback branch below.)
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
          destinations, "post_weight_sync", prepared_request, self._timeouts.post
      )
      failed_posts = [
          (d, e) for d, e in post_results if e is not None
      ]
      if failed_posts:
        state = await self._resolve_post_failures(
            destinations, failed_posts, prepared_request, failures,
            worker_reports,
        )
        if state is not RoundState.COMMITTED:
          poison_if_needed()
          await record_workers("post_weight_sync failed")
          raise fail(
              f"post_weight_sync did not commit everywhere; see workers"
          )

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
      if quiesce_attempted and prepared_request is not None and state not in (
          RoundState.UNKNOWN_TRANSFER_STATE,
      ):
        if state is RoundState.PENDING_COMMIT:
          # Post RPCs were in flight: some workers may already have
          # published. A blanket rollback would no-op on those (abort after
          # commit protects the publish) while unwinding the rest — a
          # mixed-version fleet with nobody told. Sort by each worker's own
          # report, abort only the unpublished, and poison if the fleet
          # ended up split.
          async def recover_pending_commit() -> None:
            nonlocal state
            committed_ws: list[WeightSyncDestination] = []
            others: list[WeightSyncDestination] = []
            for destination in destinations:
              phase = await self._worker_phase(destination, prepared_request)
              if phase == "committed":
                committed_ws.append(destination)
              else:
                others.append(destination)
            abort_failures: list[str] = []
            if others:
              abort_failures = await self._abort_all(others, prepared_request)
              failures.extend(abort_failures)
            if committed_ws and (others or abort_failures):
              state = RoundState.PARTIALLY_COMMITTED
              logging.error(
                  "round %d cancelled during post: %d worker(s) committed,"
                  " %d rolled back; fleet holds mixed versions",
                  round_index,
                  len(committed_ws),
                  len(others),
              )
            elif abort_failures:
              state = RoundState.FAILED_NEEDS_RESTART
            elif committed_ws:
              # Everyone published: the cancel lost the result, not the
              # round. Record the version so the next round's regression
              # guard stays truthful.
              self._last_committed_version = policy_version
            poison_if_needed()

          await asyncio.shield(recover_pending_commit())
        else:
          logging.warning(
              "round %d cancelled mid-flight; rolling destinations back",
              round_index,
          )
          await asyncio.shield(
              self._abort_all(destinations, prepared_request)
          )
      raise
    finally:
      if release_source:
        release_request = prepared_request or request
        release_errors = await asyncio.shield(
            asyncio.gather(
                *[
                    asyncio.wait_for(
                        s.release_weight_sync(release_request),
                        self._timeouts.release,
                    )
                    for s in sources
                ],
                return_exceptions=True,
            )
        )
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
                   alive and consistent on the OLD weights — a version
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
        # _call_phase's reconciliation poll and this one — a timed-out post
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
          # the same late-completion and out-of-band-abort races apply here.
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
          still_failed.append(destination)
          failures.append(f"{wid}: post_weight_sync retry: {error!r}")
          worker_reports[wid] = WorkerRoundReport(
              worker_id=wid,
              phase=phase or "h2d_done",
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
      rollback_state = await self._rollback(to_abort, request, failures)
      if rollback_state is RoundState.ABORTED:
        # Every abort confirmed: the interim per-worker records (h2d_done,
        # needs_restart) are now stale and would direct restarts of healthy
        # workers. The result must agree with its own state.
        for destination in to_abort:
          wid = _worker_id(destination)
          previous = worker_reports.get(wid)
          worker_reports[wid] = WorkerRoundReport(
              worker_id=wid,
              phase="aborted",
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
  ) -> RoundState:
    """Aborts every destination; reports honestly if any refused."""
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
    return RoundState.ABORTED