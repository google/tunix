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

"""TPU Raiden implementation of the weight synchronization transport.

`RaidenHandler` is the orchestrator-facing facade. It accepts only the neutral
types in `weight_sync` and hides the Raiden controller, protos, future-driving
rules, and planner options behind `_RaidenTransport`.

The transport owns one controller for both sides of a transfer. Workers
register their endpoints and layouts; the controller plans the reshard and
instructs source workers to push directly to destination workers. No weight
bytes pass through the orchestrator.

The owned controller is called directly. `RaidenControllerClientFacade` is a
remote stub that serializes a proto and opens a network connection; using it
for an in-process controller would also hide planner arguments supported only
by `RaidenController.start_transfer`. The server remains necessary for remote
controller clients and a genuine peer controller.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
from typing import Any, Mapping, Optional, Sequence

try:
  from tpu_raiden.rpc import controller_service_pb2
  from tpu_raiden.rpc import raiden_controller
  from tpu_raiden.rpc import raiden_service_pb2
except ImportError:
  from tpu_sync.rpc import controller_service_pb2
  from tpu_sync.rpc import raiden_controller
  from tpu_sync.rpc import raiden_service_pb2

from tunix.experimental.orchestrator import weight_sync


@dataclasses.dataclass(frozen=True)
class RaidenTransferOptions:
  """Raiden planner knobs that are constant for one deployment."""

  parallelism: Optional[int] = None
  expected_block_count: Optional[int] = None
  skip_d2h: bool = False
  skip_tiling: Optional[Mapping[int, bool]] = None
  group_size: int = 1

  def __post_init__(self) -> None:
    if self.parallelism is not None and self.parallelism <= 0:
      raise ValueError("parallelism must be positive when specified")
    if self.group_size <= 0:
      raise ValueError("group_size must be positive")
    if self.expected_block_count is not None and self.expected_block_count < 0:
      raise ValueError(
          "expected_block_count must be None (auto) or non-negative"
      )
    if self.skip_tiling is not None and any(
        layer_idx < 0 for layer_idx in self.skip_tiling
    ):
      raise ValueError("skip_tiling layer indices must be non-negative")
    if self.skip_d2h and self.skip_tiling is None:
      raise ValueError(
          "skip_d2h=True requires an explicit skip_tiling map describing"
          " the source staging format"
      )


def make_host_staged_transfer_options(
      parallelism: int = 16,
      group_size: int = 128,
      max_layers: int = 512,
  ) -> RaidenTransferOptions:
    """Transfer options for host-staged (CPU) weight sources.

    The controller derives skip_tiling from slice geometry when none is
    given, marking aligned tensors as pre-tiled — but a host-staged source
    can only stage logical bytes, so the derived plan installs permuted
    data. An explicit all-False map keeps every tensor on the logical path.

    Args:
     parallelism: Number of concurrent transfer streams.
     group_size: Number of tensors the controller moves per group.
     max_layers: Upper bound for the skip_tiling map. The receiver drops
      indices past its own layer count, so a generous value is safe.

    Returns:
     Options matching the validated e2e run and the pathways benchmark
     client.
    """
    return RaidenTransferOptions(
        parallelism=parallelism,
        group_size=group_size,
        skip_tiling={i: False for i in range(max_layers)},
    )


class _RaidenTransport:
  """Private Raiden controller, registration, and transfer implementation."""

  def __init__(
      self,
      port: int = 0,
      advertised_address: Optional[str] = None,
      loopback_address: Optional[str] = None,
      name_resolver: Optional[raiden_controller.NameResolver] = None,
      transfer_parallelism: Optional[int] = None,
      transfer_uuid: int = 1,
      transfer_options: Optional[RaidenTransferOptions] = None,
  ):
    """Starts the controller and its listener.

    Args:
      port: TCP port for the controller. Zero lets the kernel choose one.
      advertised_address: Address remote clients or a peer controller dial.
      loopback_address: This controller's same-host spelling, used to reject a
        self-addressed peer. Defaults to IPv6 loopback because a listener may
        be IPv6-only.
      name_resolver: Resolver for outbound worker control-plane calls.
      transfer_parallelism: Convenience default for transport parallelism.
      transfer_uuid: Default transfer generation.
      transfer_options: Raiden planner configuration. Mutually exclusive with
        `transfer_parallelism`.
    """
    if transfer_parallelism is not None and transfer_parallelism <= 0:
      raise ValueError("transfer_parallelism must be positive when specified")
    if transfer_uuid <= 0:
      raise ValueError("transfer_uuid must be positive")
    if transfer_options is not None and transfer_parallelism is not None:
      raise ValueError(
          "set transfer parallelism either through transfer_options or"
          " transfer_parallelism, not both"
      )
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
    self._loopback_address = (
        loopback_address or f"[::1]:{self._port}"
    )
    self._advertised_address = advertised_address or self._loopback_address
    self._transfer_options = transfer_options or RaidenTransferOptions(
        parallelism=transfer_parallelism
    )
    self._transfer_uuid = transfer_uuid
    self._registered: set[weight_sync.WorkUnitId] = set()
    self._registered_lock = threading.Lock()
    self._request_lock = threading.Lock()
    self._request_uuids: dict[str, int] = {}
    self._req_counter = 0

  @property
  def port(self) -> int:
    """Port the controller listens on."""
    return self._port

  @property
  def loopback_address(self) -> str:
    """This controller's same-host address. Nothing dials it by default."""
    return self._loopback_address

  @property
  def advertised_address(self) -> str:
    """Address remote controller clients or a peer controller should dial."""
    return self._advertised_address

  @property
  def registered_units(self) -> frozenset[weight_sync.WorkUnitId]:
    """Neutral work-unit ids currently registered with the controller."""
    with self._registered_lock:
      return frozenset(self._registered)

  @staticmethod
  def _to_raiden_id(unit: weight_sync.WorkUnitId) -> Any:
    return raiden_controller.RaidenId(
        job_name=unit.job_name,
        job_replica_id=unit.job_replica_id,
        data_name=unit.data_name,
        data_replica_idx=unit.data_replica_idx,
    )

  @staticmethod
  def _to_variable_proto(tensor: weight_sync.TensorMetadata) -> Any:
    return raiden_service_pb2.VariableMetadataProto(
        name=tensor.name,
        shape=list(tensor.shape),
        mesh_shape=list(tensor.mesh_shape),
        layout=list(tensor.layout),
        item_size=tensor.item_size,
        layer_idx=tensor.layer_idx,
        sharding_spec=list(tensor.sharding_spec),
    )

  def register_work_unit(self, metadata: weight_sync.WorkUnitMetadata) -> None:
    if not metadata.shards:
      raise ValueError(
          f"work unit {metadata.unit} registered without any data-plane"
          " address; the synchronizer must be constructed before registration"
          " so its assigned ports are known"
      )
    self._validate_metadata(metadata)
    self._controller.register_work_unit(
        unit=self._to_raiden_id(metadata.unit),
        shards=list(metadata.shards),
        control_plane_rpc_address=metadata.control_plane_rpc_address,
        mesh_shape=metadata.mesh_shape,
        layout=metadata.layout,
        global_shape=metadata.global_shape,
        itemsize=metadata.item_size,
        mesh_axes=list(metadata.mesh_axes) if metadata.mesh_axes else None,
        variables=(
            [self._to_variable_proto(tensor) for tensor in metadata.variables]
            if metadata.variables
            else None
        ),
    )
    with self._registered_lock:
      self._registered.add(metadata.unit)

  @staticmethod
  def _validate_metadata(metadata: weight_sync.WorkUnitMetadata) -> None:
    """Rejects manifests current Raiden cannot interpret unambiguously."""
    if not metadata.variables:
      return
    if not metadata.mesh_shape or not metadata.mesh_axes:
      raise ValueError(
          f"work unit {metadata.unit}: variables require the physical"
          " mesh_shape and mesh_axes"
      )
    if len(metadata.mesh_shape) != len(metadata.mesh_axes):
      raise ValueError(
          f"work unit {metadata.unit}: physical mesh_shape"
          f" {metadata.mesh_shape} and mesh_axes {metadata.mesh_axes} must"
          " have the same rank"
      )
    if len(set(metadata.mesh_axes)) != len(metadata.mesh_axes):
      raise ValueError(f"work unit {metadata.unit}: mesh_axes must be unique")
    physical_axes = dict(zip(metadata.mesh_axes, metadata.mesh_shape))
    keys: set[tuple[str, int]] = set()
    for tensor in metadata.variables:
      if not tensor.sharding_spec:
        raise ValueError(
            f"work unit {metadata.unit}: variable {tensor.name!r} must"
            " provide sharding_spec"
        )
      key = (tensor.name, tensor.layer_idx)
      if key in keys:
        raise ValueError(
            f"work unit {metadata.unit}: duplicate variable/layer {key}"
        )
      keys.add(key)
      for dim, axis in enumerate(tensor.sharding_spec):
        logical_size = tensor.mesh_shape[dim]
        if not axis:
          if logical_size != 1:
            raise ValueError(
                f"work unit {metadata.unit}: replicated dimension {dim}"
                f" of {tensor.name!r} must have logical mesh size 1, got"
                f" {logical_size}"
            )
        elif axis not in physical_axes:
          raise ValueError(
              f"work unit {metadata.unit}: variable {tensor.name!r} names"
              f" unknown mesh axis {axis!r}"
          )
        elif logical_size != physical_axes[axis]:
          raise ValueError(
              f"work unit {metadata.unit}: variable {tensor.name!r} maps axis"
              f" {axis!r} to logical size {logical_size}, but the physical"
              f" mesh has size {physical_axes[axis]}"
          )

  def transfer(
      self,
      src_units: Sequence[weight_sync.WorkUnitId],
      dst_units: Sequence[weight_sync.WorkUnitId],
      req_id: Optional[str] = None,
      generation: Optional[int] = None,
      expected_block_count: Optional[int] = None,
      parallelism: Optional[int] = None,
      skip_d2h: Optional[bool] = None,
      skip_tiling: Optional[dict[int, bool]] = None,
      group_size: Optional[int] = None,
      **kwargs: Any,
  ) -> weight_sync.TransferResult:
    """Moves weights between registered neutral work-unit ids."""
    with self._registered_lock:
      missing = [
          unit
          for unit in (*src_units, *dst_units)
          if unit not in self._registered
      ]
    if missing:
      raise ValueError(f"transfer requested for unregistered units: {missing}")

    options = self._transfer_options
    if expected_block_count is None:
      expected_block_count = options.expected_block_count
    if expected_block_count is not None and expected_block_count < 0:
      raise ValueError(
          "expected_block_count must be None (auto) or >= 0, got"
          f" {expected_block_count}"
      )
    resolved_uuid = self._transfer_uuid if generation is None else generation
    if resolved_uuid <= 0:
      raise ValueError(f"uuid must be positive, got {resolved_uuid}")
    resolved_parallelism = (
        options.parallelism if parallelism is None else parallelism
    )
    resolved_skip_d2h = options.skip_d2h if skip_d2h is None else skip_d2h
    resolved_skip_tiling = (
        dict(options.skip_tiling)
        if skip_tiling is None and options.skip_tiling is not None
        else skip_tiling
    )
    resolved_group_size = (
        options.group_size if group_size is None else group_size
    )
    if resolved_parallelism is not None and resolved_parallelism <= 0:
      raise ValueError(
          f"parallelism must be positive when specified, got"
          f" {resolved_parallelism}"
      )
    if resolved_group_size <= 0:
      raise ValueError(
          f"group_size must be positive, got {resolved_group_size}"
      )
    if resolved_skip_tiling is not None and any(
        layer < 0 for layer in resolved_skip_tiling
    ):
      raise ValueError("skip_tiling layer indices must be non-negative")
    if resolved_skip_d2h and resolved_skip_tiling is None:
      raise ValueError(
          "skip_d2h=True requires an explicit skip_tiling map describing"
          " the source staging format"
      )

    try:
      asyncio.get_running_loop()
    except RuntimeError:
      pass
    else:
      raise RuntimeError(
          "RaidenHandler.transfer is blocking; call it from an executor"
      )

    kwargs.setdefault("use_block_chunks", True)
    kwargs.setdefault("dst_mem_type", raiden_controller.RaidenMemoryType.DRAM)
    kwargs.setdefault("is_sender", True)
    for key in ("src_controller_address", "dst_controller_address"):
      address = kwargs.get(key)
      if address and address in (
          self._advertised_address,
          self._loopback_address,
      ):
        raise ValueError(
            f"transfer {req_id}: {key}={address!r} is this handler's own"
            " controller; a self-addressed peer deadlocks the transfer."
            " Omit it for single-controller deployments."
        )

    resolved_block_count = expected_block_count or 0
    if kwargs["use_block_chunks"] and resolved_block_count == 0:
      logging.info(
          "transfer %s: expected_block_count auto; deferring to the"
          " controller's schedule-derived count",
          req_id,
      )

    loop = asyncio.new_event_loop()
    try:
      with self._request_lock:
        if req_id is None:
          while True:
            self._req_counter += 1
            candidate = f"wsync-{self._req_counter}"
            if candidate not in self._request_uuids:
              req_id = candidate
              break
        assert req_id is not None
        previous_uuid = self._request_uuids.get(req_id)
        if previous_uuid is not None and previous_uuid != resolved_uuid:
          raise ValueError(
              f"transfer {req_id!r} is already bound to uuid"
              f" {previous_uuid}, not {resolved_uuid}"
          )
        self._request_uuids[req_id] = resolved_uuid
        try:
          future = self._controller.start_transfer(
              src_units=[self._to_raiden_id(unit) for unit in src_units],
              dst_units=[self._to_raiden_id(unit) for unit in dst_units],
              req_id=req_id,
              expected_block_count=resolved_block_count,
              parallelism=resolved_parallelism,
              skip_d2h=resolved_skip_d2h,
              skip_tiling=resolved_skip_tiling,
              group_size=resolved_group_size,
              uuid=resolved_uuid,
              **kwargs,
          )
        except (RuntimeError, TimeoutError) as error:
          return weight_sync.TransferResult(
              req_id=req_id, success=False, message=str(error)
          )

      try:
        if future.try_start():
          loop.run_until_complete(future.wait())
        else:
          future.wait_threadsafe()
        status = self._controller.get_transfer_status(req_id)
      except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
          raise
        raise weight_sync.TransferOutcomeUnknownError(
            f"transfer {req_id}: the controller future or final status could"
            " not be observed; the transfer may still be running"
        ) from error

      completed = (
          controller_service_pb2.GetTransferStatusResponse.STATUS_COMPLETED
      )
      failed = controller_service_pb2.GetTransferStatusResponse.STATUS_FAILED
      if status == completed:
        return weight_sync.TransferResult(req_id=req_id, success=True)
      if status == failed:
        return weight_sync.TransferResult(
            req_id=req_id,
            success=False,
            message="transfer reported STATUS_FAILED",
        )
      raise weight_sync.TransferOutcomeUnknownError(
          f"transfer {req_id}: controller reported non-terminal status"
          f" {status} after its future resolved"
      )
    finally:
      loop.close()

  def close(self) -> None:
    self._server.stop()


class RaidenHandler(weight_sync.WeightSyncHandler):
  """Orchestrator-facing facade for the private TPU Raiden transport.

  The public surface accepts and returns the transport-neutral values from
  `weight_sync`. Controller objects, Raiden protos, request futures, and raw
  transfer statuses remain inside `_RaidenTransport`.
  """

  def __init__(
      self,
      port: int = 0,
      advertised_address: Optional[str] = None,
      loopback_address: Optional[str] = None,
      name_resolver: Optional[raiden_controller.NameResolver] = None,
      transfer_parallelism: Optional[int] = None,
      transfer_uuid: int = 1,
      transfer_options: Optional[RaidenTransferOptions] = None,
  ):
    """Starts the private transport and its controller listener.

    Args:
      port: TCP port for the controller. Zero lets the kernel choose one.
      advertised_address: Address remote clients or a peer controller dial.
      loopback_address: This controller's same-host spelling, used to reject a
        self-addressed peer. Defaults to IPv6 loopback because a listener may
        be IPv6-only.
      name_resolver: Resolver for outbound worker control-plane calls.
      transfer_parallelism: Convenience default for transport parallelism.
      transfer_uuid: Default transfer generation.
      transfer_options: Raiden planner configuration. Mutually exclusive with
        `transfer_parallelism`.
    """
    self._transport = _RaidenTransport(
        port=port,
        advertised_address=advertised_address,
        loopback_address=loopback_address,
        name_resolver=name_resolver,
        transfer_parallelism=transfer_parallelism,
        transfer_uuid=transfer_uuid,
        transfer_options=transfer_options,
    )

  @property
  def port(self) -> int:
    """Port the controller listens on."""
    return self._transport.port

  @property
  def loopback_address(self) -> str:
    """This controller's same-host address. Nothing dials it by default."""
    return self._transport.loopback_address

  @property
  def advertised_address(self) -> str:
    """Address remote controller clients or a peer controller should dial."""
    return self._transport.advertised_address

  @property
  def registered_units(self) -> frozenset[weight_sync.WorkUnitId]:
    """Neutral work-unit ids currently registered with the controller."""
    return self._transport.registered_units

  def register_work_unit(self, metadata: weight_sync.WorkUnitMetadata) -> None:
    self._transport.register_work_unit(metadata)

  def transfer(
      self,
      src_units: Sequence[weight_sync.WorkUnitId],
      dst_units: Sequence[weight_sync.WorkUnitId],
      req_id: Optional[str] = None,
      generation: Optional[int] = None,
      expected_block_count: Optional[int] = None,
      parallelism: Optional[int] = None,
      skip_d2h: Optional[bool] = None,
      skip_tiling: Optional[dict[int, bool]] = None,
      group_size: Optional[int] = None,
      **kwargs: Any,
  ) -> weight_sync.TransferResult:
    """Moves weights between registered neutral work-unit ids."""
    return self._transport.transfer(
        src_units=src_units,
        dst_units=dst_units,
        req_id=req_id,
        generation=generation,
        expected_block_count=expected_block_count,
        parallelism=parallelism,
        skip_d2h=skip_d2h,
        skip_tiling=skip_tiling,
        group_size=group_size,
        **kwargs,
    )

  def close(self) -> None:
    self._transport.close()