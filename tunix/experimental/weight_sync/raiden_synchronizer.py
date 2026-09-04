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
import inspect
import ipaddress
import os
import socket
from typing import Any, List, Optional, Tuple

from absl import logging
import jax
from jax.experimental import compute_on
import jax.numpy as jnp
from tunix.experimental.weight_sync import weight_sync


def _log_rss(tag: str) -> None:
  """Logs process peak RSS (GB) -- pinpoints which bind() stage spikes host

  memory, since ru_maxrss is a high-water mark that only grows.

  Off by default; enable with --v=1. `resource` is imported lazily because it
  is Unix-only.
  """
  if not logging.vlog_is_on(1):
    return
  import resource  # pylint: disable=g-import-not-at-top

  rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
  logging.vlog(
      1, "raiden bind rss checkpoint [%s]: %.1f GB (peak)", tag, rss_gb
  )


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


def _ensure_ffi_compute_on_compat() -> None:
  """Bridges TPU-sync wheels that call the newer compute_on decorator API."""
  try:
    from jax.experimental import compute_on  # pytype: disable=import-error  pylint: disable=g-import-not-at-top,unused-import
  except ImportError:
    pass
  compute_on_mod = getattr(jax, "_src", None)
  if compute_on_mod is None:
    return
  compute_on_mod = getattr(compute_on_mod, "compute_on", None)
  if compute_on_mod is None:
    return

  try:
    params = inspect.signature(compute_on_mod.compute_on).parameters
  except (TypeError, ValueError):
    params = {}
  if "out_memory_spaces" in params:
    return

  compute_on2 = getattr(compute_on_mod, "compute_on2", None)
  if compute_on2 is None:
    raise RuntimeError(
        "Installed JAX lacks compute_on compatibility required by the TPU-sync"
        " FFI wheel."
    )

  compute_on_mod.compute_on = compute_on2
  logging.warning(
      "Patched jax._src.compute_on.compute_on to compute_on2 for TPU-sync FFI"
      " compatibility."
  )


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
  """Unpacks an IP address from the FFI synchronizer metadata row."""
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


def _bindable(arr: Any, *, allow_proxy: bool = False) -> bool:
  """True if the native layer can bind this leaf.

  Binding an unsupported leaf (e.g. RNG keys) can SIGSEGV, so only
  floating-point, rank>=1 arrays on supported local platforms qualify.
  """
  try:
    if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
      return False
    if arr.ndim < 1:
      return False
    if not jnp.issubdtype(arr.dtype, jnp.floating):
      return False
    devices = arr.devices()
    if not devices:
      return False
    supported_platforms = {"tpu", "cpu"}
    if allow_proxy:
      supported_platforms.add("proxy")
    return all(
        getattr(d, "platform", "?") in supported_platforms for d in devices
    )
  except Exception:
    return False


def _filter_bindable(
    names: List[str], arrays: List[Any], *, allow_proxy: bool = False
) -> Tuple[List[str], List[Any]]:
  """Drops leaves _bindable rejects."""
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
    if _bindable(arr, allow_proxy=allow_proxy):
      arr.block_until_ready()
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
      parallelism: int = 4,
      bind_ip: Optional[str] = None,
  ):
    is_proxy = "proxy" in os.environ.get("JAX_PLATFORMS", "")
    self.job_name = job_name
    self.worker_index = worker_index
    self.names: List[str] = []
    self.arrays: List[Any] = []
    self.ip = bind_ip or local_ip()
    self._auto_h2d = auto_h2d
    self._is_proxy = is_proxy
    self._parallelism = parallelism
    self._sync: Any = None
    self._ips: List[str] = []
    self._unique_listeners: List[str] = []
    self._ffi_mesh: Any = None
    self._ffi_shard_idx: Any = None
    if state is not None:
      self.bind(state)

  @property
  def bound(self) -> bool:
    return bool(self.names)

  @property
  def active(self) -> bool:
    return self._sync is not None or bool(self._ips)

  def _init_ffi_transport(self, *, is_d2h: bool) -> None:
    if not self.arrays:
      raise RuntimeError(
          f"{self.job_name}: bind() must stage arrays before FFI init"
      )
    if _raiden_ffi is None:
      raise RuntimeError(
          "weight_synchronizer_ffi is not available for FFI weight sync."
      )

    import numpy as np  # pylint: disable=g-import-not-at-top
    from jax.experimental import multihost_utils  # pylint: disable=g-import-not-at-top

    _ensure_ffi_compute_on_compat()
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
    num_processes = len(
        set(getattr(d, "process_index", 0) for d in src_devices)
    )
    devices_per_host = len(src_devices) // max(1, num_processes)

    if is_d2h:
      logging.info(
          "Initializing Pathways weight synchronizer and executing D2H via FFI"
          " (%d layers, %d devices/host)",
          len(self.arrays),
          devices_per_host,
      )
      ws_info = _raiden_ffi.init_weight_synchronizer_and_d2h(
          device_arrays=self.arrays,
          shard_idx=shard_idx,
          mesh=mesh,
          slice_byte_sizes=slice_byte_sizes_sharded,
          parallelism=self._parallelism,
          num_layers=len(self.arrays),
          listener_port=0,
          num_shards=devices_per_host,
      )
    else:
      logging.info(
          "Initializing Pathways weight synchronizer for H2D via FFI (%d"
          " layers, %d devices/host)",
          len(self.arrays),
          devices_per_host,
      )

      # TODO(b/557061810): Re-enable this once the bug is fixed and the FFI
      # call is verified to work.
      # ws_info = _raiden_ffi.init_weight_synchronizer(
      #     device_array=self.arrays[0],
      #     shard_idx=shard_idx,
      #     mesh=mesh,
      #     slice_byte_sizes=slice_byte_sizes_sharded,
      #     parallelism=self._parallelism,
      #     num_layers=len(self.arrays),
      #     listener_port=0,
      #     num_shards=devices_per_host,
      # )

      @compute_on.compute_on(
          compute_type="device_host",
          out_memory_spaces=jax.memory.Space.Device,
      )
      def _local_init(anchor, s_idx, sizes):
        axis_names = mesh.axis_names
        out_shape = tuple([1] * len(axis_names)) + (6,)
        return jax.ffi.ffi_call(
            "init_weight_synchronizer",
            jax.ShapeDtypeStruct(out_shape, jnp.int32),
            has_side_effect=True,
        )(
            anchor,
            s_idx,
            sizes,
            local_port=np.int32(0),
            parallelism=np.int32(self._parallelism),
            num_layers=np.int32(len(self.arrays)),
            listener_port=np.int32(0),
            num_shards=np.int32(devices_per_host),
        )

      ws_info = jax.shard_map(
          _local_init,
          mesh=mesh,
          in_specs=(
              self.arrays[0].sharding.spec,
              jax.sharding.PartitionSpec(*mesh.axis_names),
              jax.sharding.PartitionSpec(None),
          ),
          out_specs=jax.sharding.PartitionSpec(*mesh.axis_names, None),
      )(self.arrays[0], shard_idx, slice_byte_sizes_sharded)

    local_ws_info = multihost_utils.global_array_to_host_local_array(
        ws_info,
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
    self._ffi_mesh = mesh
    self._ffi_shard_idx = shard_idx

  def _ffi_h2d(self) -> None:
    if _raiden_ffi is None:
      raise RuntimeError(
          "weight_synchronizer_ffi is not available for FFI weight sync."
      )
    if self._ffi_mesh is None or self._ffi_shard_idx is None:
      raise RuntimeError(f"{self.job_name}: bind() must run before h2d()")
    self.arrays = list(
        _raiden_ffi.multi_h2d(self.arrays, self._ffi_shard_idx, self._ffi_mesh)
    )
    for arr in self.arrays:
      arr.block_until_ready()

  def bind(self, state: Any) -> None:
    """Binds this host's weights, or rebinds them after a training step."""
    _log_rss("bind:start")
    # Clear previous buffers before staging to avoid holding duplicate weight
    # copies in host memory during rebinds.
    self.names = []
    self.arrays = []
    self.names, self.arrays = _filter_bindable(
        *flatten_weights(state), allow_proxy=self._is_proxy
    )
    del state
    _log_rss("bind:after_flatten")
    logging.info(
        "%s bind prepared %d arrays (proxy_runtime=%s)",
        self.job_name,
        len(self.arrays),
        self._is_proxy,
    )
    if self._is_proxy:
      self._ips = []
      self._unique_listeners = []
      if self._auto_h2d:
        self._init_ffi_transport(is_d2h=False)
        logging.info(
            "%s FFI destination transport ready: shards=%s control=%s",
            self.job_name,
            self._ips,
            self._unique_listeners,
        )
      return
    if _ws_lib is None:
      return
    if self._sync is None:
      logging.info(
          "%s creating native WeightSynchronizer for %d arrays",
          self.job_name,
          len(self.arrays),
      )
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
      logging.info(
          "%s native WeightSynchronizer ready: data_port=%s listener_port=%s"
          " num_shards=%s",
          self.job_name,
          getattr(self._sync, "local_port", None),
          getattr(self._sync, "listener_port", None),
          getattr(self._sync, "num_shards", None),
      )
      _log_rss("bind:after_native_construct")
    else:
      logging.info("%s rebinding %d arrays", self.job_name, len(self.arrays))
      self._sync.bind_weights(self.arrays)
      _log_rss("bind:after_native_rebind")

  def _require_sync(self, op: str) -> Any:
    if self._sync is None and not self._is_proxy:
      raise RuntimeError(f"{self.job_name}: bind() must run before {op}")
    return self._sync

  def d2h(self) -> None:
    if self._is_proxy:
      try:
        self._init_ffi_transport(is_d2h=True)
        logging.info(
            "FFI D2H complete. Shards: %s, Control plane: %s",
            self._ips,
            self._unique_listeners,
        )
        return
      except Exception:
        logging.exception(
            "FFI D2H failed for %s with %d staged arrays",
            self.job_name,
            len(self.arrays),
        )
        raise

    self._require_sync("d2h()").d2h()

  def h2d(self) -> None:
    if not self.bound:
      raise RuntimeError(f"{self.job_name}: bind() must run before h2d()")
    if self._is_proxy:
      self._ffi_h2d()
      return
    if self._sync is not None:
      self._sync.h2d()
      jax.block_until_ready(self.arrays)

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
    # Metadata for cross-checking the weight-sync result.
    head["__tensor_count__"] = len(self.arrays)
    head["__element_count__"] = int(sum(a.size for a in self.arrays))
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
    if self._is_proxy:
      shards = tuple(self._ips)
      control_addr = self._unique_listeners[0] if self._unique_listeners else ""
    else:
      data_addr = f"{self.ip}:{self._sync.local_port}" if self._sync else ""
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
