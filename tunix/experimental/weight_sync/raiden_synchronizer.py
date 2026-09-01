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

"""Binds one host's weights to the raiden transport and exports its metadata."""

from __future__ import annotations

import collections
import ipaddress
import os
import socket
from typing import Any, List, Optional, Tuple

from absl import logging
import jax
import jax.numpy as jnp
from tunix.experimental.weight_sync import weight_sync

_ws_lib: Any = None
try:
  from tpu_sync.api.jax import weight_synchronizer as _ws_lib  # pytype: disable=import-error  pylint: disable=g-import-not-at-top
except ImportError:
  _ws_lib = None

_raiden_ffi: Any = None
try:
  from tpu_sync.frameworks.jax import weight_synchronizer_ffi as _raiden_ffi  # pytype: disable=import-error  pylint: disable=g-import-not-at-top
except ImportError:
  _raiden_ffi = None


def local_ip() -> str:
  for family, probe in (
      (socket.AF_INET, ("8.8.8.8", 80)),
      (socket.AF_INET6, ("2001:4860:4860::8888", 80)),
  ):
    try:
      s = socket.socket(family, socket.SOCK_DGRAM)
      try:
        s.connect(probe)
        ip = s.getsockname()[0]
      finally:
        s.close()
      return f"[{ip}]" if ":" in ip else ip
    except OSError:
      continue
  return "localhost"


def unpack_ip(row: Any) -> str:
  """Unpacks IP address from uint32 array row."""
  raw_bytes = b"".join(
      int(x).to_bytes(4, byteorder="little", signed=True) for x in row[:4]
  )
  if raw_bytes[:10] == b"\x00" * 10 and raw_bytes[10:12] == b"\xff\xff":
    return str(ipaddress.IPv4Address(raw_bytes[12:16]))
  addr_str = str(ipaddress.IPv6Address(raw_bytes))
  return f"[{addr_str}]" if ":" in addr_str else addr_str


def flatten_weights(state: Any) -> Tuple[List[str], List[Any]]:
  """Returns (names, arrays) for every array leaf, in stable tree order."""
  names, arrays = [], []
  for path, leaf in jax.tree_util.tree_leaves_with_path(state):
    arr = getattr(leaf, "value", leaf)
    if hasattr(arr, "shape") and hasattr(arr, "dtype"):
      names.append(jax.tree_util.keystr(path))
      arrays.append(arr)
  return names, arrays


def _bindable(arr: Any) -> bool:
  """True if the native layer can bind this leaf."""
  try:
    devices = arr.devices()
  except AttributeError:
    return False
  on_local_hw = all(
      getattr(d, "platform", "?") in ("cpu", "tpu") for d in devices
  )
  return on_local_hw and jnp.issubdtype(arr.dtype, jnp.number)


def _filter_bindable(
    names: List[str], arrays: List[Any]
) -> Tuple[List[str], List[Any]]:
  """Drops leaves the native layer cannot bind; binding them is undefined
  behavior (observed: random RuntimeError or SIGSEGV on RNG-key arrays)."""
  logging.vlog(
      1,
      "raiden bind census: %s",
      collections.Counter(
          (type(a).__name__, str(getattr(a, "dtype", "?"))) for a in arrays
      ),
  )
  keep_names: List[str] = []
  keep_arrays: List[Any] = []
  dropped = []
  for name, arr in zip(names, arrays):
    if _bindable(arr):
      keep_names.append(name)
      keep_arrays.append(arr)
    else:
      dropped.append(name)
  if dropped:
    logging.warning(
        "raiden bind dropped %d unbindable leaves: %s", len(dropped), dropped[:5]
    )
  return keep_names, keep_arrays


def _axis_name(axis: Any) -> str:
  if axis is None:
    return ""
  if isinstance(axis, str):
    return axis
  return ",".join(axis)


def _tensor_metadata(name: str, arr: Any, layer_idx: int):
  sharding: Any = getattr(arr, "sharding", None)
  spec = tuple(getattr(sharding, "spec", ()) or ())
  spec = (spec + (None,) * arr.ndim)[: arr.ndim]
  try:
    local = sharding.shard_shape(tuple(arr.shape))
    mesh_shape = tuple(g // l for g, l in zip(arr.shape, local))
  except Exception:  # pylint: disable=broad-exception-caught
    mesh_shape = (1,) * arr.ndim
  return weight_sync.TensorMetadata(
      name=name,
      shape=tuple(arr.shape),
      mesh_shape=mesh_shape,
      layout=tuple(reversed(range(arr.ndim))),
      item_size=arr.dtype.itemsize,
      layer_idx=layer_idx,
      sharding_spec=tuple(_axis_name(a) for a in spec),
  )


class RaidenSynchronizer:
  """One host's weights on the raiden transport, plus its registration metadata.

  Used by both the trainer and the sampler. Construct with a state to bind
  right away, or leave it out and call `bind` when the weights exist; every
  later `bind` rebinds the same transport. Known limits: one mesh axis per
  tensor dim, and without the tpu_sync wheel the metadata carries no shard
  addresses, so the handler refuses registration.
  """

  def __init__(
      self,
      job_name: str,
      state: Any = None,
      *,
      worker_index: int = 0,
      auto_h2d: bool = False,
      use_ffi: Optional[bool] = None,
      parallelism: int = 4,
      bind_ip: Optional[str] = None,
  ):
    is_proxy = "proxy" in os.environ.get("JAX_PLATFORMS", "")
    if use_ffi is None:
      use_ffi = is_proxy
    self.job_name = job_name
    self.worker_index = worker_index
    self.names: List[str] = []
    self.arrays: List[Any] = []
    self.ip = bind_ip or local_ip()
    self._auto_h2d = auto_h2d
    self._use_ffi = use_ffi
    self._parallelism = parallelism
    self._sync: Any = None
    self._ips: List[str] = []
    self._unique_listeners: List[str] = []
    if state is not None:
      self.bind(state)

  @property
  def bound(self) -> bool:
    return bool(self.names)

  @property
  def active(self) -> bool:
    return self._sync is not None or bool(self._ips)

  @property
  def use_ffi(self) -> bool:
    return self._use_ffi

  def bind(self, state: Any) -> None:
    """Binds this host's weights, or rebinds them after a training step."""
    self.names, self.arrays = _filter_bindable(*flatten_weights(state))
    if self._use_ffi:
      return
    if _ws_lib is None:
      return
    if self._sync is None:
      self._sync = _ws_lib.WeightSynchronizer(
          self.arrays,
          local_port=0,
          parallelism=self._parallelism,
          # Rebinding deadlocks on the retained usage holds otherwise; the
          # caller keeps Python references to the bound arrays regardless.
          unsafe_skip_buffer_lock=True,
          listener_port=0,
          bind_ip=None,
          auto_h2d=self._auto_h2d,
      )
    else:
      self._sync.bind_weights(self.arrays)

  def _require_sync(self, op: str) -> Any:
    if self._sync is None and not self._use_ffi:
      raise RuntimeError(f"{self.job_name}: bind() must run before {op}")
    return self._sync

  def d2h(self) -> None:
    if self._use_ffi:
      if not self.arrays:
        raise RuntimeError(f"{self.job_name}: bind() must run before d2h()")
      if _raiden_ffi is None:
        raise RuntimeError("weight_synchronizer_ffi is not available for FFI weight sync.")

      import numpy as np  # pylint: disable=g-import-not-at-top
      from jax.experimental import multihost_utils  # pylint: disable=g-import-not-at-top

      mesh = getattr(getattr(self.arrays[0], "sharding", None), "mesh", None)
      if mesh is None:
        raise ValueError("Arrays must be sharded on a Mesh for FFI weight sync.")

      slice_byte_sizes = [
          int(np.prod(arr.sharding.shard_shape(arr.shape)) * arr.dtype.itemsize)
          for arr in self.arrays
      ]
      sizes_sharding = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(None)
      )
      slice_byte_sizes_sharded = jax.device_put(
          jnp.array(slice_byte_sizes, dtype=jnp.int32), sizes_sharding
      )

      task_mesh_shape = tuple(mesh.shape[a] for a in mesh.axis_names)
      global_ids = jnp.array(
          [d.id for d in mesh.devices.flatten()], dtype=jnp.int32
      ).reshape(task_mesh_shape)
      shard_idx = jax.device_put(
          global_ids,
          jax.sharding.NamedSharding(
              mesh, jax.sharding.PartitionSpec(*mesh.axis_names)
          ),
      )

      src_devices = mesh.devices.flatten()
      num_processes = len(set(getattr(d, "process_index", 0) for d in src_devices))
      devices_per_host = len(src_devices) // max(1, num_processes)

      logging.info(
          "Initializing Pathways weight synchronizer and executing D2H via FFI (%d layers, %d devices/host)",
          len(self.arrays),
          devices_per_host,
      )
      src_ws_info = _raiden_ffi.init_weight_synchronizer_and_d2h(
          device_arrays=self.arrays,
          shard_idx=shard_idx,
          mesh=mesh,
          slice_byte_sizes=slice_byte_sizes_sharded,
          parallelism=self._parallelism,
          num_layers=len(self.arrays),
          listener_port=0,
          num_shards=devices_per_host,
      )

      local_ws_info = multihost_utils.global_array_to_host_local_array(
          src_ws_info,
          mesh,
          jax.sharding.PartitionSpec(*mesh.axis_names, None),
      )
      gathered_ws_info = multihost_utils.process_allgather(local_ws_info).reshape(
          -1, 6
      )

      self._ips, listeners = [], []
      for row in gathered_ws_info:
        ip = unpack_ip(row)
        self._ips.append(f"{ip}:{int(row[4])}")
        listeners.append(f"{ip}:{int(row[5])}")

      self._unique_listeners = []
      for listener in listeners:
        if listener not in self._unique_listeners:
          self._unique_listeners.append(listener)
      logging.info(
          "FFI D2H complete. Shards: %s, Control plane: %s",
          self._ips,
          self._unique_listeners,
      )
      return

    self._require_sync("d2h()").d2h()

  def h2d(self) -> None:
    if not self.bound:
      raise RuntimeError(f"{self.job_name}: bind() must run before h2d()")
    if self._sync is not None:
      self._sync.h2d()

  def metrics(self) -> dict:
    return self._sync.get_metrics() if self._sync else {}

  def checksums(self, sample: int = 3) -> dict:
    """Per-tensor float32 abs-sums for cross-process verification."""

    def total(arr):
      return float(jnp.sum(jnp.abs(arr).astype(jnp.float32)))

    head = {
        name: total(arr)
        for name, arr in list(zip(self.names, self.arrays))[:sample]
    }
    head["__grand_total__"] = float(sum(total(a) for a in self.arrays))
    return head

  def work_unit_metadata(self) -> weight_sync.WorkUnitMetadata:
    variables = tuple(
        _tensor_metadata(name, arr, idx)
        for idx, (name, arr) in enumerate(zip(self.names, self.arrays))
    )
    mesh_axes: tuple = ()
    mesh_shape = None
    for arr in self.arrays:
      mesh = getattr(getattr(arr, "sharding", None), "mesh", None)
      if mesh is not None:
        mesh_axes = tuple(mesh.axis_names)
        mesh_shape = tuple(mesh.shape[a] for a in mesh.axis_names)
        break
    if mesh_shape is None:
      mesh_axes = ("fsdp",)
      mesh_shape = (1,)
    if self._use_ffi:
      shards = tuple(self._ips)
      control_addr = self._unique_listeners[0] if self._unique_listeners else ""
    else:
      data_addr = (
          f"{self.ip}:{self._sync.local_port}" if self._sync else ""
      )
      control_addr = (
          f"{self.ip}:{self._sync.listener_port}"
          if self._sync and self._sync.listener_port
          else ""
      )
      num_shards = self._sync.num_shards if self._sync else 1
      shards = (data_addr,) * num_shards if data_addr else ()
    # Index 0 keeps the default replica id "": transfer callers construct
    # WorkUnitId(job_name=...) without a replica, and registration lookups
    # must match it for single-replica units.
    unit = weight_sync.WorkUnitId(
        job_name=self.job_name,
        job_replica_id=str(self.worker_index) if self.worker_index else "",
    )
    return weight_sync.WorkUnitMetadata(
        unit=unit,
        shards=shards,
        control_plane_rpc_address=control_addr,
        mesh_shape=mesh_shape,
        variables=variables,
        mesh_axes=mesh_axes or None,
    )