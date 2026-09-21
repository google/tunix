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
import logging
import os
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable


class WeightSyncMode(str, enum.Enum):
  """Modes for weight synchronization across workers."""

  NONE = "none"
  FALLBACK = "fallback"
  RAIDEN = "raiden"
  GCS = "gcs"

  @classmethod
  def _missing_(cls, value: object) -> Optional[WeightSyncMode]:
    if isinstance(value, str):
      norm = value.strip().lower()
      if norm in ("file", "filesystem", "gcs"):
        return cls.GCS
    return None


DEFAULT_WEIGHT_SYNC_MODE = WeightSyncMode.FALLBACK


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
      `PartitionSpec` used by the Tunix/JAX adapters: `P(None, "y")` is `("",
      "y")`. A dimension sharded over the product of several axes -- JAX
      `P(("x", "y"))`, as MoE weights get when tensor and attention-data
      parallelism are combined -- is the axes joined by commas, major first:
      `("x,y",)`. Together with the work unit's physical `mesh_axes`, it maps
      device coordinates onto the variable's logical mesh. A concrete transport
      must reject forms its wire representation cannot encode.  TODO(tunix-dev):
      replace the comma-joined string with a structured per-dimension tuple,
      e.g. `((), ("tp",), ("attention_dp", "tp"))`. The string form makes every
      consumer re-parse it and reserves the comma. Needs the Raiden handler and
      the MaxText adapter migrated together.
    global_shard_indices: Explicit global shard indices owned by the local
      shards of this variable.
  """

  name: str
  shape: tuple[int, ...]
  mesh_shape: tuple[int, ...]
  layout: tuple[int, ...]
  item_size: int
  layer_idx: int = 0
  sharding_spec: tuple[str, ...] = ()
  global_shard_indices: tuple[int, ...] = ()

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
    named_axes = [
        a for axis in self.sharding_spec for a in axis.split(",") if a
    ]
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
    shards: Data-plane addresses, one "ip:port" per participating shard. Repeats
      are expected: a process serving several local devices shares one transfer
      port, so the list may carry repeated addresses while preserving the shard
      count.
    control_plane_rpc_address: Optional listener address used by transports that
      send commands back to workers.
    global_shape: Global shape of the tensor, when the unit carries exactly one.
    mesh_shape: Physical JAX mesh shape for this work unit. For a variable's
      logical per-tensor mesh, see `TensorMetadata.mesh_shape`.
    layout: minor_to_major layout mapping.
    item_size: Bytes per element.
    variables: Multi-variable manifest. When a unit carries several tensors this
      is populated instead of the single-tensor fields above.
    mesh_axes: Names of the mesh axes, in mesh order, e.g. ("fsdp", "tp") or the
      ("x", "y") a jax.sharding.Mesh was built with. The counterpart to a
      variable's `sharding_spec`: the spec names axes, this says which physical
      mesh dimension each name is. Both sides are needed before the a transport
      can map device coordinates without guessing axes from equal dimension
      sizes.
    transport_mode: Transport this unit is bound on, "ffi" or "tcp". None means
      unreported, not a default. Source and destination need not match.
    use_ffi: `transport_mode == "ffi"` as a bool, for consumers that would
      otherwise compare strings. None when unreported.
    host_subgrid: Optional local host subgrid shape (e.g. from
      `mesh.local_mesh.devices.shape`) for decomposing physical mesh slices.
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
  transport_mode: Optional[str] = None
  use_ffi: Optional[bool] = None
  host_subgrid: Optional[tuple[int, ...]] = None
  artifact_uri: Optional[str] = None
  checksums: Optional[dict[str, float]] = None

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
                global_shard_indices=tuple(v.get("global_shard_indices", ())),
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
                global_shard_indices=tuple(
                    getattr(v, "global_shard_indices", ()) or ()
                ),
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
        transport_mode=d.get("transport_mode"),
        use_ffi=d.get("use_ffi"),
        host_subgrid=(
            tuple(d["host_subgrid"])
            if d.get("host_subgrid") is not None
            else None
        ),
        artifact_uri=(
            str(d["artifact_uri"]) if d.get("artifact_uri") is not None else None
        ),
        checksums=(
            {str(k): float(v) for k, v in d["checksums"].items()}
            if isinstance(d.get("checksums"), Mapping)
            else None
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

  def build_extra_config(
      self, src_metadata: Sequence[WorkUnitMetadata]
  ) -> dict[str, Any]:
    """Builds transport-specific extra_config entries from source metadata."""
    del src_metadata
    return {}

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


class WeightSynchronizer(abc.ABC):
  """Unified worker-side data-plane interface for weight synchronization.

  Implementations (`RaidenWeightSync`, `GCSWeightSync`) share the same
  trainer (`bind`, `d2h`, `work_unit_metadata`, `release`) and rollout
  (`bind`, `work_unit_metadata`, `h2d`, `apply_to_runner`, `checksums`)
  lifecycle contracts.
  """

  job_name: str
  worker_index: int
  names: list[str]
  arrays: list[Any]

  @property
  def bound(self) -> bool:
    return bool(self.names)

  @property
  def active(self) -> bool:
    return self.bound

  @abc.abstractmethod
  def bind(self, state: Any) -> None:
    """Binds or rebinds a model state PyTree for weight synchronization."""

  @abc.abstractmethod
  def d2h(self, sync_request: Any = None) -> None:
    """Exports bound device arrays to the transport staging layer."""

  @abc.abstractmethod
  def h2d(self, sync_request: Any = None, **kwargs: Any) -> None:
    """Imports weights from the transport staging layer into device arrays."""

  @abc.abstractmethod
  def work_unit_metadata(self) -> WorkUnitMetadata:
    """Returns wire-safe WorkUnitMetadata for coordinator registration."""

  def work_unit_metadata_all(self) -> list[WorkUnitMetadata]:
    """Returns work unit metadata list for registration."""
    return [self.work_unit_metadata()]

  @abc.abstractmethod
  def apply_to_runner(self, runner: Any) -> None:
    """Applies updated arrays after H2D to the inference runner's state."""

  @abc.abstractmethod
  def checksums(self, sample: Optional[int] = 3) -> dict[str, Any]:
    """Computes per-tensor float32 L1 abs-sum checksums and grand total."""

  def metrics(self) -> dict[str, Any]:
    """Returns transport metrics dictionary."""
    return {}

  def release(self, sync_request: Any = None) -> None:
    """Releases or cleans up per-round transport staging resources."""
    del sync_request

  def close(self) -> None:
    """Closes the synchronizer and releases persistent resources."""


def is_verify_weights_enabled() -> bool:
  """Returns True if weight checksum verification is enabled via VERIFY_WEIGHTS."""
  return os.environ.get("VERIFY_WEIGHTS", "").strip().lower() in (
      "1",
      "true",
      "yes",
      "y",
      "t",
  )


def verify_weight_checksums(
    src_checksums: Mapping[str, Any],
    dst_checksums: Mapping[str, Any],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> dict[str, Any]:
  """Verifies that destination weight checksums match source weight checksums.

  Guarded by the `VERIFY_WEIGHTS` environment variable. When enabled, checks
  tensor count, element count, grand total L1 norm, and every shared per-tensor
  L1 norm within relative tolerance `rtol` and absolute tolerance `atol`.
  Raises `RuntimeError` if any mismatch is found.

  Args:
    src_checksums: Checksum mapping from the trainer source.
    dst_checksums: Checksum mapping from the rollout destination.
    rtol: Relative tolerance for float32 L1 reduction comparison.
    atol: Absolute tolerance for float32 L1 reduction comparison.

  Returns:
    A summary dictionary of verification metrics.
  """
  if not is_verify_weights_enabled():
    return {"verified": False, "skipped": True}

  if not src_checksums or not dst_checksums:
    raise RuntimeError(
        "Cannot verify weight checksums: empty checksum mapping "
        f"(src keys={len(src_checksums or {})}, dst keys={len(dst_checksums or {})})."
    )

  for count_key in ("__tensor_count__", "__element_count__"):
    if count_key in src_checksums and count_key in dst_checksums:
      src_c = int(src_checksums[count_key])
      dst_c = int(dst_checksums[count_key])
      if src_c != dst_c:
        raise RuntimeError(
            f"Weight checksum verification failed on {count_key}: "
            f"source={src_c} != destination={dst_c}"
        )

  mismatches: list[str] = []
  max_rel_err = 0.0
  matched_tensors = 0

  from tunix.experimental.weight_sync import raiden_synchronizer  # pylint: disable=g-import-not-at-top

  def _norm(mapping: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in mapping.items():
      if k.startswith("__"):
        out[k] = float(v)
      else:
        canon = raiden_synchronizer._param_key(k) or k  # pylint: disable=protected-access
        out[canon] = float(v)
    return out

  norm_src = _norm(src_checksums)
  norm_dst = _norm(dst_checksums)

  # Compare __grand_total__ and all shared canonical tensor keys
  src_keys = {k for k in norm_src if not k.startswith("__")}
  dst_keys = {k for k in norm_dst if not k.startswith("__")}
  shared_keys = sorted(src_keys & dst_keys)

  keys_to_check = ["__grand_total__"] + shared_keys
  for key in keys_to_check:
    if key not in norm_src or key not in norm_dst:
      continue
    s_val = float(norm_src[key])
    d_val = float(norm_dst[key])
    abs_err = abs(s_val - d_val)
    denom = max(abs(s_val), abs(d_val), 1e-8)
    rel_err = abs_err / denom
    if key != "__grand_total__":
      matched_tensors += 1
      max_rel_err = max(max_rel_err, rel_err)
    if abs_err > atol and rel_err > rtol:
      mismatches.append(
          f"{key}: src={s_val:.6f}, dst={d_val:.6f}, "
          f"abs_err={abs_err:.3e}, rel_err={rel_err:.3e}"
      )

  if mismatches:
    raise RuntimeError(
        f"WEIGHT VERIFICATION FAILED: {len(mismatches)} checksum mismatch(es) "
        f"(rtol={rtol}, atol={atol}): {mismatches[:10]}"
    )

  summary = {
      "verified": True,
      "tensor_count": int(dst_checksums.get("__tensor_count__", matched_tensors)),
      "element_count": int(dst_checksums.get("__element_count__", 0)),
      "matched_tensors": matched_tensors,
      "src_grand_total": float(src_checksums.get("__grand_total__", 0.0)),
      "dst_grand_total": float(dst_checksums.get("__grand_total__", 0.0)),
      "max_rel_err": max_rel_err,
  }
  logging.info(
      "WEIGHT VERIFICATION PASSED: tensor_count=%d, matched_tensors=%d, "
      "element_count=%d, grand_total=(src=%.6f, dst=%.6f), max_rel_err=%.3e",
      summary["tensor_count"],
      summary["matched_tensors"],
      summary["element_count"],
      summary["src_grand_total"],
      summary["dst_grand_total"],
      summary["max_rel_err"],
  )
  return summary


def create_weight_synchronizer(
    mode: WeightSyncMode | str,
    job_name: str,
    state: Any = None,
    *,
    worker_index: int = 0,
    staging_dir: Optional[str] = None,
    **kwargs: Any,
) -> WeightSynchronizer:
  """Factory creating a WeightSynchronizer (`RaidenWeightSync` or `GCSWeightSync`)."""
  resolved_mode = (
      mode if isinstance(mode, WeightSyncMode) else WeightSyncMode(str(mode))
  )
  if resolved_mode == WeightSyncMode.RAIDEN:
    from tunix.experimental.weight_sync import raiden_synchronizer  # pylint: disable=g-import-not-at-top

    WeightSynchronizer.register(raiden_synchronizer.RaidenSynchronizer)
    if state is not None:
      kwargs["state"] = state
    return raiden_synchronizer.RaidenSynchronizer(
        job_name=job_name,
        worker_index=worker_index,
        **kwargs,
    )
  if resolved_mode == WeightSyncMode.GCS:
    from tunix.experimental.weight_sync import gcs_weight_sync  # pylint: disable=g-import-not-at-top

    return gcs_weight_sync.GCSWeightSync(
        job_name=job_name,
        state=state,
        worker_index=worker_index,
        staging_dir=staging_dir,
        **kwargs,
    )
  raise ValueError(
      f"Unsupported WeightSyncMode {resolved_mode!r} for WeightSynchronizer."
  )

