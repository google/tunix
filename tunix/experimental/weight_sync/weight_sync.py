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

"""Transport-neutral contracts for orchestrated weight synchronization.

Workers, the coordinator, and public results exchange the concrete
`WorkUnitId` and `WorkUnitMetadata` types below. The coordinator depends only
on `WeightSyncHandler`, so a checkpoint, file, or network implementation does
not need to expose another transport's identifiers, protos, or planner options.

The TPU Raiden implementation lives in `raiden_handler.py`. Keeping it in a
separate module makes this contract importable without importing Raiden.
"""

from __future__ import annotations

import abc
import dataclasses
import enum
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable


class WeightSyncMode(str, enum.Enum):
  """Modes for weight synchronization across workers."""

  NONE = "none"
  FALLBACK = "fallback"
  RAIDEN = "raiden"


@dataclasses.dataclass(frozen=True)
class WorkUnitId:
  """Transport-neutral identity for one participant's data work unit.

  The four fields are intentionally sufficient to map losslessly onto
  Raiden's identifier, but none of them requires Raiden.  In particular,
  `data_replica_idx` must not be dropped: two data-parallel replicas may own
  the same named tensor under the same job replica.
  """

  job_name: str
  job_replica_id: str = ""
  data_name: str = ""
  data_replica_idx: int = 0

  def __post_init__(self) -> None:
    if not self.job_name:
      raise ValueError("work-unit job_name must not be empty")
    if self.data_replica_idx < 0:
      raise ValueError("work-unit data_replica_idx must be non-negative")


@dataclasses.dataclass(frozen=True)
class TensorMetadata:
  """Describes a single weight tensor inside a work unit.

  A work unit may carry several variables. `layer_idx` is a stable batching
  ordinal that transports may use when planning groups; transports that do
  not need it may ignore it.

  Attributes:
    name: Variable name.
    shape: Shape of the tensor.
    mesh_shape: Mesh shape for this tensor.
    layout: Layout mapping.
    item_size: Bytes per element.
    layer_idx: Stable batching ordinal.
    sharding_spec: The mesh axis name sharding each TENSOR dimension, empty
      string where that dimension is replicated. This is the subset of JAX
      `PartitionSpec` used by the Tunix/JAX adapters: `P(None, "y")` is
      `("", "y")`. A dimension sharded over the product of several axes -- JAX
      `P(("x", "y"))`, as MoE weights get when tensor and attention-data
      parallelism are combined -- is the axes joined by commas, major first:
      `("x,y",)`. Together
      with the work unit's physical `mesh_axes`, it maps device coordinates onto
      the variable's logical mesh. A concrete transport must reject forms its
      wire representation cannot encode.
  """

  name: str
  shape: tuple[int, ...]
  mesh_shape: tuple[int, ...]
  layout: tuple[int, ...]
  item_size: int
  layer_idx: int = 0
  sharding_spec: tuple[str, ...] = ()

  def __post_init__(self) -> None:
    rank = len(self.shape)
    if not self.name:
      raise ValueError("variable name must not be empty")
    if not self.shape or any(dim <= 0 for dim in self.shape):
      raise ValueError(f"variable {self.name!r} has invalid shape {self.shape}")
    if len(self.mesh_shape) != rank or any(
        dim <= 0 for dim in self.mesh_shape
    ):
      raise ValueError(
          f"variable {self.name!r}: mesh_shape {self.mesh_shape} must have"
          f" rank {rank} and positive dimensions"
      )
    # Tunix/JAX adapters use both full minor-to-major permutations such as
    # (1, 0) and partial layouts such as (-1, 0), where -1 denotes a
    # replicated tensor dimension. The neutral contract therefore validates
    # only the common rank requirement; each concrete handler validates the
    # layout forms its transport can encode.
    if len(self.layout) != rank:
      raise ValueError(
          f"variable {self.name!r}: layout {self.layout} must have rank"
          f" {rank}"
      )
    if self.item_size <= 0:
      raise ValueError(
          f"variable {self.name!r}: item_size must be positive, got"
          f" {self.item_size}"
      )
    if self.layer_idx < 0:
      raise ValueError(
          f"variable {self.name!r}: layer_idx must be non-negative"
      )
    if self.sharding_spec and len(self.sharding_spec) != rank:
      raise ValueError(
          f"variable {self.name!r}: sharding_spec {self.sharding_spec} must"
          f" have rank {rank}"
      )
    named_axes = [a for axis in self.sharding_spec for a in axis.split(",")
                  if a]
    if len(named_axes) != len(set(named_axes)):
      raise ValueError(
          f"variable {self.name!r}: a mesh axis may not shard two tensor"
          f" dimensions: {self.sharding_spec}"
      )


@dataclasses.dataclass(frozen=True)
class WorkUnitMetadata:
  """Wire-safe metadata describing one source or destination work unit.

  This is the concrete, transport-neutral contract between workers and the
  coordinator.  Network transports use the endpoint fields; checkpoint/file
  handlers may leave them empty and consume only the identity and tensor
  placement metadata.  A concrete handler is responsible for validating the
  subset it needs and translating it into its own wire representation.

  One instance describes one independently addressable work unit. A
  multi-host participant therefore normally returns one per physical host.

  Wire safety: instances cross process boundaries via cloudpickle when workers
  are remote. Every field must stay plain Python data. No `jax.Array`, no
  device buffers.

  Attributes:
    unit: Stable identity, e.g. WorkUnitId(job_name="trainer").
    shards: Data-plane addresses, one "ip:port" per participating shard.
      Repeats are expected: a process serving several local devices shares one
      transfer port, so the list may carry repeated addresses while preserving
      the shard count.
    control_plane_rpc_address: Optional listener address used by transports
      that send commands back to workers.
    global_shape: Global shape of the tensor, when the unit carries exactly one.
    mesh_shape: Physical JAX mesh shape for this work unit. For a variable's
      logical per-tensor mesh, see `TensorMetadata.mesh_shape`.
    layout: minor_to_major layout mapping.
    item_size: Bytes per element.
    variables: Multi-variable manifest. When a unit carries several tensors this
      is populated instead of the single-tensor fields above.
    mesh_axes: Names of the mesh axes, in mesh order, e.g. ("fsdp", "tp") or
      the ("x", "y") a jax.sharding.Mesh was built with. The counterpart to a
      variable's `sharding_spec`: the spec names axes, this says which
      physical mesh dimension each name is. Both sides are needed before the
      a transport can map device coordinates without guessing axes from equal
      dimension sizes.
  """

  unit: WorkUnitId
  shards: tuple[str, ...] = ()
  control_plane_rpc_address: str = ""
  global_shape: Optional[tuple[int, ...]] = None
  mesh_shape: Optional[tuple[int, ...]] = None
  layout: Optional[tuple[int, ...]] = None
  item_size: Optional[int] = None
  variables: tuple[TensorMetadata, ...] = ()
  mesh_axes: Optional[tuple[str, ...]] = None

  @classmethod
  def from_dict(cls, d: Any) -> WorkUnitMetadata:
    """Reconstructs WorkUnitMetadata from a dictionary or returns metadata directly."""
    if isinstance(d, cls):
      return d
    if not isinstance(d, dict):
      raise TypeError(f"Expected WorkUnitMetadata or dict, got {type(d)}")

    unit_raw = d.get("unit")
    if isinstance(unit_raw, dict):
      unit = WorkUnitId(**unit_raw)
    elif isinstance(unit_raw, WorkUnitId):
      unit = unit_raw
    else:
      unit = WorkUnitId(job_name=str(unit_raw or "destination"))

    variables_raw = d.get("variables", ())
    variables = []
    for v in variables_raw:
      if isinstance(v, TensorMetadata):
        variables.append(v)
      elif isinstance(v, dict):
        variables.append(
            TensorMetadata(
                name=v["name"],
                shape=tuple(v["shape"]),
                mesh_shape=tuple(v["mesh_shape"]),
                layout=tuple(v["layout"]),
                item_size=int(v["item_size"]),
                layer_idx=int(v.get("layer_idx", 0)),
                sharding_spec=tuple(v.get("sharding_spec", ())),
            )
        )
      elif hasattr(v, "name"):
        variables.append(
            TensorMetadata(
                name=v.name,
                shape=tuple(v.shape),
                mesh_shape=tuple(v.mesh_shape),
                layout=tuple(v.layout),
                item_size=int(v.item_size),
                layer_idx=int(getattr(v, "layer_idx", 0)),
                sharding_spec=tuple(getattr(v, "sharding_spec", ())),
            )
        )

    return cls(
        unit=unit,
        shards=tuple(d.get("shards", ())),
        control_plane_rpc_address=str(d.get("control_plane_rpc_address", "")),
        global_shape=(
            tuple(d["global_shape"])
            if d.get("global_shape") is not None
            else None
        ),
        mesh_shape=(
            tuple(d["mesh_shape"]) if d.get("mesh_shape") is not None else None
        ),
        layout=tuple(d["layout"]) if d.get("layout") is not None else None,
        item_size=(
            int(d["item_size"]) if d.get("item_size") is not None else None
        ),
        variables=tuple(variables),
        mesh_axes=(
            tuple(d["mesh_axes"]) if d.get("mesh_axes") is not None else None
        ),
    )


def dict_to_metadata(d: Any) -> WorkUnitMetadata:
  """Reconstructs WorkUnitMetadata from a dictionary (delegates to WorkUnitMetadata.from_dict)."""
  return WorkUnitMetadata.from_dict(d)


@dataclasses.dataclass(frozen=True)
class TransferResult:
  """Outcome of one weight transfer."""

  req_id: str
  success: bool
  message: str = ""


class TransferOutcomeUnknownError(RuntimeError):
  """The transfer RPC's outcome is unknown, NOT known-failed.

  Raised after a controller future has been created when driving that future,
  waiting for another driver, or reading its final status fails. At that point
  the controller may still be executing the transfer, with workers still
  writing into destination staging. Callers must treat this like a transfer
  deadline -- no rollback, no source release -- never like a failed transfer.
  """


class WeightSyncHandler(abc.ABC):
  """The transport interface the coordinator drives.

  The coordinator sees only this. It does not know how bytes move, only that
  units are registered and transfers run.
  """

  @abc.abstractmethod
  def register_work_unit(self, metadata: WorkUnitMetadata) -> None:
    """Registers one source or destination work unit with the transport.

    Registration is keyed by the unit id, so re-registering the same unit
    replaces its entry rather than adding one. Both sides re-register every
    round: the source because each policy version may prepare a new artifact
    or placement, the destination because replacement registration is how a
    transport learns of a restarted worker's new resources.

    There is deliberately no per-round `unregister`: registration is a
    replace-by-id operation, and handler lifetime cleanup belongs in `close`.
    """

  @abc.abstractmethod
  def transfer(
      self,
      src_units: Sequence[WorkUnitId],
      dst_units: Sequence[WorkUnitId],
      req_id: Optional[str] = None,
      generation: Optional[int] = None,
  ) -> TransferResult:
    """Moves weights from the source units to the destination units.

    This is a blocking call and returns once a terminal outcome is known.
    Callers running an event loop wrap it in an executor.
    """

  def close(self) -> None:
    """Releases any transport resources. Optional for implementations."""


@runtime_checkable
class WeightSyncSource(Protocol):
  """The trainer side of a round.

  This is a structural protocol: a trainer worker satisfies it by implementing
  the methods below and does not inherit from the orchestrator layer. Keeping
  that direction avoids coupling the worker abstraction to its coordinator.

  Metadata is the return value of the per-round prepare call rather than a
  separately polled property. The same objects are registered and carried in
  the round's request, so a handler always sees the identity and placement
  information for exactly the version that was prepared. A JAX/Raiden source
  commonly rebinds new arrays and D2Hs here; a file handler may instead write
  a checkpoint and return metadata naming that prepared artifact.
  """

  async def prepare_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> Sequence[WorkUnitMetadata]:
    """Stages this round's weights and returns their transport metadata.

    Returns one entry per independently addressable work unit. A network
    transport commonly uses one unit per physical host/listener, so a
    multi-host source returns several. Wire-safe values only; no device
    arrays.

    Args:
      sync_request: Optional request context for the sync round.
      **kwargs: Additional transport-specific options.
    """
    ...

  async def release_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> Any:
    """Releases this round's staging.

    Called on every exit path except an UNKNOWN_TRANSFER_STATE round (a
    possibly-live transfer may still be reading the staging). Idempotent, and
    must be safe to call while a timed-out prepare for the same round is still
    running remotely.

    Args:
      sync_request: Optional request context for the sync round.
      **kwargs: Additional transport-specific options.
    """
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
    """Binds this worker's destination-side transport resources.

    Idempotent and called every round. A network transport normally keeps its
    listener endpoints stable across calls; a restarted worker binds fresh
    resources here and the round's replacement registration picks them up.

    Deliberately a separate step from `pre_weight_sync`: the metadata is
    essentially the endpoints, which exist only after bind, and bind,
    metadata collection and registration all run while the worker is STILL
    SERVING. Folding bind into pre would move all three inside the downtime
    window. The failure classes also differ: a bind failure needs no
    rollback because nothing is quiesced yet, while a pre failure already
    requires the abort path.

    Deliberately not named `initialize`: `Worker.initialize` is an abstract
    method on the base class, driven by `LifecycleDriver.bring_up`, and this
    is a distinct later step that needs the model arrays to exist.
    """
    ...

  async def get_weight_sync_metadata(
      self,
  ) -> Sequence[WorkUnitMetadata]:
    """Transport metadata for this worker, one entry per physical host.

    Called while the worker is still serving; collection and registration cost
    no downtime. Wire-safe values only; no device arrays.
    """
    ...

  async def pre_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> Any:
    """Quiesces the worker so the arriving weights have somewhere to land.

    Must actually gate admission: stop accepting new requests, drain or cancel
    in-flight ones, drop the prefix cache, free the KV cache. The worker is
    not serving from the moment this returns until post or abort. Merely
    setting a pause flag does not satisfy this.

    Args:
      sync_request: Optional request context for the sync round.
      **kwargs: Additional transport-specific options.
    """
    ...

  async def weight_sync(self, sync_request: Any = None, **kwargs: Any) -> Any:
    """Materializes the received weights into the staging copy.

    Called only after the transport reported success. A Raiden/JAX worker
    performs H2D from host staging here; a file-backed worker may load its
    prepared checkpoint. It must not touch the serving copy and records the
    pending policy version for post to publish.

    Args:
      sync_request: Optional request context for the sync round.
      **kwargs: Additional transport-specific options.
    """
    ...

  async def post_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> Any:
    """Publishes the pending weights atomically, rebuilds caches, resumes.

    Must be idempotent for the round key: a retry after a lost reply, or
    after a crash between publishing and recording, must converge to the
    same committed state rather than fail or double-apply.

    Args:
      sync_request: Optional request context for the sync round.
      **kwargs: Additional transport-specific options.
    """
    ...

  async def abort_weight_sync(
      self, sync_request: Any = None, **kwargs: Any
  ) -> Any:
    """Rolls back to serving the previous weights.

    Invalidates this round's staging -- physically or logically: a
    destination whose synchronizer stays bound to the staging buffers cannot
    free them, and instead must guarantee nothing publishes them (the
    tracker's refusal to commit an aborted round is that guarantee).
    Rebuilds the KV cache, resumes admission on the old weights. Never
    touches the serving copy. Idempotent, and safe to call at any phase of a
    round including before pre completed.

    There is deliberately no weight copy-back: `weight_sync` writes the
    staging copy only, so serving still holds the previous weights. Making
    the round unpublishable plus restoring admission and KV therefore IS the
    entire rollback. This is the coordinator's internal failure path
    (partial pre, failed transfer, failed H2D, cancellation), not a
    user-facing API; serving the previous version is the normal state
    between any two syncs, and the alternative on every such failure would
    be a fleet restart that lands on the same old version far more
    expensively.

    Args:
      sync_request: Optional request context for the sync round.
      **kwargs: Additional transport-specific options.
    """
    ...

  async def get_weight_sync_status(self) -> Mapping[str, Any]:
    """The worker's view of its current round: `WorkerRoundTracker.report()`.

    Consulted by the coordinator whenever a phase RPC fails, to distinguish a
    lost reply from unfinished work.
    """
    ...
