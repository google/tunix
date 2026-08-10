"""Weight synchronization transport layer for the RL orchestrator.

The orchestrator owns a single transport controller that serves both the source
(trainer) and the destination (sampler) side of a weight transfer. Workers
register their endpoints and layout with the controller; the controller then
plans the reshard and instructs the source workers to push directly to the
destination workers. No weight bytes pass through the orchestrator.

Layering: this module is the Raiden integration. `RaidenWorkUnitMetadata`
mirrors Raiden's `RegisterWorkUnitRequest` and is deliberately named for it;
it does not pretend to be transport-neutral. The seam the coordinator relies
on is `WeightSyncHandler`: the coordinator sees only that interface, so a
checkpoint- or file-based handler can be substituted without touching it.

Address model. Four addresses are involved and must not be conflated:

  1. The controller's self-dial address: what the handler's own facade dials.
     The controller binds `("::", port)` and its IPV6_V6ONLY setsockopt is
     wrapped in a swallowing try/except, so the listener can come up
     IPv6-only; dialing 127.0.0.1 then times out. Self-dial uses `[::1]`.
  2. The controller's advertised address: what remote workers dial, typically
     a BNS name on Borg. Deployment-specific; passed in, never derived.
  3. Each worker's control-plane RPC address: where the controller sends
     commands (shutdown, prepare). Registered per work unit.
  4. Each worker's data-plane shard addresses: where weight bytes flow.
     Registered per work unit, one entry per shard.

The controller also makes outbound calls to worker control-plane addresses.
Those go through its `WeightSyncWorkerRpcClient`, which needs the name
resolver when worker addresses are BNS names; giving the resolver only to the
facade is not enough.
"""

from __future__ import annotations

import abc
import dataclasses
import logging
from typing import Any, Optional, Sequence

try:
  from GOOGLE_INTERNAL_PACKAGE_PATH.third_party.tpu_raiden.tpu_raiden.rpc import controller_service_pb2
  from GOOGLE_INTERNAL_PACKAGE_PATH.third_party.tpu_raiden.tpu_raiden.rpc import raiden_controller
  from GOOGLE_INTERNAL_PACKAGE_PATH.third_party.tpu_raiden.tpu_raiden.rpc import raiden_service_pb2
except ImportError:
  # OSS / 3p: the public tpu_raiden package root has no GOOGLE_INTERNAL_PACKAGE_PATH prefix.
  from tpu_raiden.rpc import controller_service_pb2  # type: ignore
  from tpu_raiden.rpc import raiden_controller  # type: ignore
  from tpu_raiden.rpc import raiden_service_pb2  # type: ignore


from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from GOOGLE_INTERNAL_PACKAGE_PATH.third_party.tpu_raiden.tpu_raiden.rpc.raiden_controller import RaidenId
else:
  RaidenId = raiden_controller.RaidenId


@dataclasses.dataclass(frozen=True)
class VariableMetadata:
  """Describes a single weight tensor inside a work unit.

  Mirrors `VariableMetadataProto` in raiden_service.proto. A work unit may carry
  several variables; `layer_idx` is the batching dimension Raiden uses to move
  more than one tensor per transfer.
  """

  name: str
  shape: tuple[int, ...]
  mesh_shape: tuple[int, ...]
  layout: tuple[int, ...]
  item_size: int
  layer_idx: int = 0

  def to_proto(self) -> Any:
    return raiden_service_pb2.VariableMetadataProto(
        name=self.name,
        shape=list(self.shape),
        mesh_shape=list(self.mesh_shape),
        layout=list(self.layout),
        item_size=self.item_size,
        layer_idx=self.layer_idx,
    )


@dataclasses.dataclass(frozen=True)
class RaidenWorkUnitMetadata:
  """Everything the controller needs to register one work unit.

  Mirrors `RegisterWorkUnitRequest`. The fields are deliberately kept close to
  the proto rather than simplified, so that the trainer and the sampler can
  implement against the same shape.

  One instance describes one work unit, and Raiden wants one work unit per
  physical host (per listener). A multi-host participant therefore returns a
  sequence of these, not one.

  Wire safety: instances cross process boundaries via cloudpickle when workers
  are remote. Every field must stay plain Python data. No `jax.Array`, no
  device buffers.

  Attributes:
    unit: Identifies the work unit, e.g. RaidenId(job_name="trainer").
    shards: Data-plane addresses, one "ip:port" per participating shard. Repeats
      are expected: a process serving several local devices shares one transfer
      port, so the list carries the shard count while the addresses coincide.
    control_plane_rpc_address: Listener address the controller sends commands
      to. Empty for units that need no listener.
    global_shape: Global shape of the tensor, when the unit carries exactly one.
    mesh_shape: Logical mesh shape the tensor is sharded over.
    layout: minor_to_major layout mapping.
    item_size: Bytes per element.
    variables: Multi-variable manifest. When a unit carries several tensors this
      is populated instead of the single-tensor fields above. Registering
      through `variables` also switches the controller planner off its legacy
      block-relative offset path.
    transfer_rank: Stable rank of this unit inside its producer group.
    transfer_parallelism: Number of parallel transport streams. The controller
      requires this together with pool_manifest/layout_fingerprint/page_tokens
      (`has_reshard_metadata` in raiden_controller.py); setting it alone will
      raise at registration.
  """

  unit: RaidenId
  shards: tuple[str, ...]
  control_plane_rpc_address: str = ""
  global_shape: Optional[tuple[int, ...]] = None
  mesh_shape: Optional[tuple[int, ...]] = None
  layout: Optional[tuple[int, ...]] = None
  item_size: Optional[int] = None
  variables: tuple[VariableMetadata, ...] = ()
  transfer_rank: Optional[int] = None
  transfer_parallelism: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class TransferResult:
  """Outcome of one weight transfer."""

  req_id: str
  success: bool
  message: str = ""


class TransferOutcomeUnknownError(RuntimeError):
  """The transfer RPC's outcome is unknown, NOT known-failed.

  Raised when the client-side call to the controller timed out: the reply is
  lost but the controller may still be executing the transfer, with workers
  still writing into destination staging. Callers must treat this like a
  transfer deadline — no rollback, no source release — never like a failed
  transfer.
  """


class WeightSyncHandler(abc.ABC):
  """The transport interface the coordinator drives.

  The coordinator sees only this. It does not know how bytes move, only that
  units are registered and transfers run.
  """

  @abc.abstractmethod
  def register_work_unit(self, metadata: RaidenWorkUnitMetadata) -> None:
    """Registers one source or destination work unit with the transport.

    Registration is keyed by the unit id, so re-registering the same unit
    replaces its entry rather than adding one. Both sides re-register every
    round: the source because JAX rebinds its arrays after each optimizer step,
    the destination because re-registering is how the controller learns of a
    restarted worker's new ports.

    There is deliberately no `unregister`: the Raiden controller client exposes
    no such RPC, and because the controller keys its registries by unit id,
    repeated registration does not accumulate stale entries.
    """

  @abc.abstractmethod
  def transfer(
      self,
      src_units: Sequence[RaidenId],
      dst_units: Sequence[RaidenId],
      req_id: Optional[str] = None,
      **kwargs: Any,
  ) -> TransferResult:
    """Moves weights from the source units to the destination units.

    This is a blocking call: it returns once the transfer has completed or
    failed. It is deliberately not split into a start/wait pair, because the
    underlying Raiden RPC already blocks until the transfer future resolves.
    Callers running an event loop wrap it in an executor.
    """

  def close(self) -> None:
    """Releases any transport resources. Optional for implementations."""


class RaidenHandler(WeightSyncHandler):
  """TPU Raiden implementation of `WeightSyncHandler`.

  Owns the single `RaidenController` that serves both sides of the transfer.
  Instantiate one of these in the RL orchestrator process.
  """

  def __init__(
      self,
      port: int = 0,
      advertised_address: Optional[str] = None,
      dial_address: Optional[str] = None,
      name_resolver: Optional[Any] = None,
      transfer_parallelism: Optional[int] = None,
      transfer_uuid: int = 1,
  ):
    """Starts the controller and a client facing it.

    Args:
      port: TCP port for the controller. 0 lets the kernel pick one, which is
        what you want outside of a fixed deployment.
      advertised_address: Address remote workers dial to reach this controller,
        e.g. a BNS name on Borg. Defaults to the self-dial address, which is
        only correct when every worker is on this host.
      dial_address: Address this handler's own facade dials. Defaults to
        `[::1]:{port}`. IPv6 loopback is load-bearing: the controller's listener
        can come up IPv6-only (see module docstring), and dialing 127.0.0.1 then
        times out rather than failing fast.
      name_resolver: Optional resolver, e.g. a BNS resolver on Borg. Wired into
        both the facade (this handler's outbound calls) and the controller's own
        `WeightSyncWorkerRpcClient` (the controller's outbound calls to worker
        control-plane addresses). The second wiring matters: workers registered
        under BNS control addresses are unreachable without it.
      transfer_parallelism: Default number of transport streams per unit.
      transfer_uuid: Generation id carried on every transfer this handler starts
        when the caller does not supply one. Status lookups are keyed by
        (req_id, uuid).
    """
    worker_rpc_client = None
    if name_resolver is not None:
      worker_rpc_client = raiden_controller.WeightSyncWorkerRpcClient(
          name_resolver=name_resolver
      )
    self._controller = raiden_controller.RaidenController(
        port=port, worker_rpc_client=worker_rpc_client
    )
    self._server = raiden_controller.RaidenControllerServer(self._controller)
    self._port = self._server.start()
    self._dial_address = dial_address or f"[::1]:{self._port}"
    self._advertised_address = advertised_address or self._dial_address
    self._client = raiden_controller.RaidenControllerClientFacade(
        self._dial_address, name_resolver=name_resolver
    )
    self._transfer_parallelism = transfer_parallelism
    self._transfer_uuid = transfer_uuid
    self._registered: set[RaidenId] = set()
    self._req_counter = 0

  @property
  def port(self) -> int:
    """Port the controller listens on."""
    return self._port

  @property
  def dial_address(self) -> str:
    """Address this handler's own facade dials (loopback)."""
    return self._dial_address

  @property
  def advertised_address(self) -> str:
    """Address remote workers should dial to reach the controller."""
    return self._advertised_address

  @property
  def registered_units(self) -> frozenset[RaidenId]:
    """Units currently registered with the controller."""
    return frozenset(self._registered)

  def register_work_unit(self, metadata: RaidenWorkUnitMetadata) -> None:
    if not metadata.shards:
      raise ValueError(
          f"work unit {metadata.unit} registered without any data-plane"
          " address; the synchronizer must be constructed before registration"
          " so its assigned ports are known"
      )
    self._client.register_work_unit(
        unit=metadata.unit,
        shards=list(metadata.shards),
        control_plane_rpc_address=metadata.control_plane_rpc_address,
        mesh_shape=metadata.mesh_shape,
        layout=metadata.layout,
        global_shape=metadata.global_shape,
        itemsize=metadata.item_size,
        transfer_rank=metadata.transfer_rank,
        transfer_parallelism=(
            metadata.transfer_parallelism
            if metadata.transfer_parallelism is not None
            else self._transfer_parallelism
        ),
        variables=(
            [v.to_proto() for v in metadata.variables]
            if metadata.variables
            else None
        ),
    )
    self._registered.add(metadata.unit)

  def registered_metadata(self) -> list[Any]:
    """Returns every work unit the controller currently knows about.

    Useful when debugging a transfer that plans incorrectly: it shows exactly
    what layout the controller believes each side has.
    """
    return self._client.get_metadata()

  def transfer(
      self,
      src_units: Sequence[RaidenId],
      dst_units: Sequence[RaidenId],
      req_id: Optional[str] = None,
      expected_block_count: Optional[int] = None,
      **kwargs: Any,
  ) -> TransferResult:
    missing = [u for u in (*src_units, *dst_units) if u not in self._registered]
    if missing:
      raise ValueError(f"transfer requested for unregistered units: {missing}")

    if req_id is None:
      self._req_counter += 1
      req_id = f"wsync-{self._req_counter}"

    # Chunked transport, host-memory destination, and a transfer uuid are what
    # every working weight-sync path passes. `use_block_chunks` defaults to
    # False on the client, which selects a different transport in the native
    # layer, so it is set explicitly rather than left to the default.
    kwargs.setdefault("use_block_chunks", True)
    kwargs.setdefault("dst_mem_type", raiden_controller.RaidenMemoryType.DRAM)
    kwargs.setdefault("is_sender", True)
    kwargs.setdefault("uuid", self._transfer_uuid)

    # Controller peer addresses are NEVER defaulted. The controller treats a
    # non-empty destination controller address as a REMOTE peer and
    # synchronously RPCs it to register the schedule; pointed at itself, that
    # RPC re-enters the same req_id, gets the parent's own in-flight future
    # back, and waits on it while the parent waits on the RPC — a
    # deterministic self-deadlock surfacing as the facade's 300s recv
    # timeout. Only a genuine dual-controller topology passes these, and
    # then only the OTHER controller's address.
    for key in ("src_controller_address", "dst_controller_address"):
      address = kwargs.get(key)
      if address and address in (self._advertised_address, self._dial_address):
        raise ValueError(
            f"transfer {req_id}: {key}={address!r} is this handler's own"
            " controller; a self-addressed peer deadlocks"
            " coordinate_transfer. Omit it for single-controller"
            " deployments."
        )

    if expected_block_count is not None and expected_block_count < 0:
      raise ValueError(
          f"transfer {req_id}: expected_block_count must be None (auto) or"
          f" >= 0, got {expected_block_count}"
      )
    resolved_block_count = expected_block_count or 0
    if kwargs["use_block_chunks"] and resolved_block_count == 0:
      # AUTO: the controller's sender-side direct-schedule planning derives
      # the per-destination push count from its OWN schedule (look for
      # "Auto-calculated expected_block_count" in its log) — the only
      # authoritative source, since one schedule entry can carry several
      # pushes. This delegation holds ONLY for that path: a 0 reaching a
      # symmetric is_sender=False registration skips receiver preparation
      # outright (fail-closed there is a tracked controller-side fix). A
      # positive value overrides the controller and is for experiments only.
      logging.info(
          "transfer %s: expected_block_count auto — deferring to the"
          " controller's schedule-derived count",
          req_id,
      )

    try:
      self._client.coordinate_transfer(
          src_units=list(src_units),
          dst_units=list(dst_units),
          req_id=req_id,
          expected_block_count=resolved_block_count,
          **kwargs,
      )
    except TimeoutError as e:
      # The facade's socket timed out waiting for the controller's reply
      # (socket.timeout is TimeoutError). The reply is lost; the transfer
      # may still be running server-side. This must not be reported as a
      # failed transfer — a failure invites rollback, and rollback would
      # discard buffers a live transfer may be writing.
      raise TransferOutcomeUnknownError(
          f"transfer {req_id}: coordinate_transfer timed out client-side;"
          " the controller may still be executing the transfer"
      ) from e
    except RuntimeError as e:
      return TransferResult(req_id=req_id, success=False, message=str(e))

    status = self.transfer_status(req_id, uuid=kwargs["uuid"])
    completed = (
        controller_service_pb2.GetTransferStatusResponse.STATUS_COMPLETED
    )
    if status != completed:
      return TransferResult(
          req_id=req_id,
          success=False,
          message=f"transfer reported status {status}, expected {completed}",
      )

    return TransferResult(req_id=req_id, success=True)

  def transfer_status(self, req_id: str, uuid: int = 0) -> int:
    """Raw transfer status, for callers that poll."""
    return self._client.get_transfer_status(req_id, uuid=uuid)

  def close(self) -> None:
    self._server.stop()
