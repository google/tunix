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
import contextlib
import inspect
import os
import re
import socket
from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.weight_sync import weight_sync
from tunix.utils import mesh


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


_lazy_modules: dict[str, Any] = {}


def _lazy_import_module(module_path: str) -> Any:
  """Imports a module lazily by path, caching the result."""
  if module_path not in _lazy_modules:
    try:
      import importlib  # pylint: disable=g-import-not-at-top

      _lazy_modules[module_path] = importlib.import_module(module_path)
    except ImportError:
      _lazy_modules[module_path] = None
  return _lazy_modules[module_path]


def _get_ws_lib() -> Any:
  """Imports tpu_sync weight_synchronizer lazily to prevent early C++ library symbol collisions."""
  if "_ws_lib" in globals():
    return globals()["_ws_lib"]
  mod = _lazy_import_module(
      "tpu_sync.api.jax.weight_synchronizer"
  )
  return mod


def _get_raiden_ffi() -> Any:
  """Imports tpu_sync weight_synchronizer_ffi lazily to avoid loading XLA runtime early."""
  if "_raiden_ffi" in globals():
    return globals()["_raiden_ffi"]
  mod = _lazy_import_module(
      "tpu_sync.frameworks.jax.weight_synchronizer_ffi"
  )
  return mod


def __getattr__(name: str) -> Any:
  if name == "_raiden_ffi":
    return _get_raiden_ffi()
  if name == "_ws_lib":
    return _get_ws_lib()
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _ensure_ffi_compute_on_compat() -> None:
  """Bridges TPU-sync wheels that call the newer compute_on decorator API."""
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
  # The wheel does `from jax.experimental import compute_on` and then calls
  # `compute_on.compute_on(...)`. jax.experimental.compute_on binds the name at
  # import time, so patching jax._src alone leaves the caller on the old
  # two-arg version and the decorator dies with
  #   TypeError: compute_on() got an unexpected keyword argument 'out_memory_spaces'
  try:
    from jax.experimental import compute_on as _public_compute_on

    _public_compute_on.compute_on = compute_on2
  except ImportError:
    pass
  logging.warning(
      "Patched jax._src.compute_on.compute_on (and jax.experimental."
      "compute_on) to compute_on2 for TPU-sync FFI compatibility."
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


def flatten_weights(state: Any) -> Tuple[List[str], List[Any]]:
  """Returns (names, arrays) for every array leaf, in stable tree order."""
  names, arrays = [], []
  for path, leaf in jax.tree_util.tree_leaves_with_path(state):
    arr = getattr(leaf, "value", leaf)
    if hasattr(arr, "shape") and hasattr(arr, "dtype"):
      names.append(jax.tree_util.keystr(path))
      arrays.append(arr)
  return names, arrays


# One bracketed component (`['w']`, `[0]`) or one dotted component (`w`).
_PATH_COMPONENT_RE = re.compile(r"\[([^\]]*)\]|([^.\[\]]+)")
# A trailing ordinal folded into the name, as in `layers_0`.
_FOLDED_INDEX_RE = re.compile(r"^(.+?)_(\d+)$")


def _param_key(name: str) -> str:
  """Canonical dotted key for pairing a bound name with a runner tree path.

  The two sides spell the same parameter differently: `flatten_weights` reports
  JAX keystr (`['layers'][0]['w']`), callers pass bare names (`w1`) or
  attribute paths (`layers.0.w`), and unscanned MaxText layers arrive with the
  ordinal folded in (`layers_0`). Reducing every form to `layers.0.w` lets the
  match ignore which side produced the name.
  """
  segments: List[str] = []
  for bracketed, dotted in _PATH_COMPONENT_RE.findall(name):
    component = (bracketed or dotted).strip("'\"")
    if not component:
      continue
    folded = _FOLDED_INDEX_RE.match(component)
    segments.extend(folded.groups() if folded else (component,))
  # Wrapper roots and the nnx `.value` leaf are not part of the identity.
  while segments and segments[0] in ("base", "model"):
    segments.pop(0)
  if segments and segments[-1] == "value":
    segments.pop()
  return ".".join(segments)


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


_KV_CACHE_SEGMENTS = frozenset(
    {"cache", "cached_prefill_key", "cached_prefill_value"}
)


def _is_kv_cache(name: str) -> bool:
  """True for inference KV-cache buffers, which are not weights.

  The rollout's MaxText model carries per-layer KV cache arrays
  (`...['attention']['cache']['cached_prefill_key']`). They are floating point
  and rank>=1, so `_bindable` accepts them, but the trainer has no counterpart
  and manifest preflight then rejects them as destination-only variables.
  """
  return any(seg in _KV_CACHE_SEGMENTS for seg in _param_key(name).split("."))


def _filter_bindable(
    names: List[str], arrays: List[Any], *, allow_proxy: bool = False
) -> Tuple[List[str], List[Any]]:
  """Drops leaves _bindable rejects, plus inference-only KV cache buffers."""
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
  cache_dropped = []
  for name, arr in zip(names, arrays):
    if _is_kv_cache(name):
      cache_dropped.append(name)
    elif _bindable(arr, allow_proxy=allow_proxy):
      arr.block_until_ready()
      keep_names.append(name)
      keep_arrays.append(arr)
    else:
      dropped.append(name)
  if cache_dropped:
    logging.debug(
        "raiden bind skipped %d KV-cache leaves: %s",
        len(cache_dropped),
        cache_dropped[:5],
    )
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


def _devices_per_host(devices: List[Any]) -> int:
  """Devices sharing one physical host, i.e. Raiden's `num_shards`.

  The native layer derives `submanager_idx = shard_idx / num_shards` and
  `slot = shard_idx % num_shards`, so this must be the real per-host device
  count. Overstate it and every host allocates staging for the whole slice but
  fills only its own share, leaving the rest of its SetGlobalShardIndices at
  -1 -- the transfer then completes green while delivering only the shards one
  host happened to own.

  Groups by host key (resolving Pathways `logical_task` or `process_index` via
  mesh topology).
  """
  host_keys = [mesh.device_host_key(d) for d in devices]
  per_host = collections.Counter(k for k in host_keys if k is not None)
  per_task = collections.Counter(getattr(d, "task_id", None) for d in devices)
  del per_task[None]
  if len(per_task) > len(per_host):
    per_host = per_task

  if not per_host:
    logging.warning(
        "no host or task metadata on any of %d device(s); assuming a single"
        " host",
        len(devices),
    )
    return len(devices)
  counts = set(per_host.values())
  if len(counts) > 1:
    # No right answer for a ragged slice; understating only wastes staging,
    # overstating drops another host's shards.
    logging.warning(
        "uneven devices per host %s; using the smallest (%d)",
        dict(per_host),
        min(counts),
    )
    return min(counts)
  return counts.pop()


def _derive_ffi_host_slots(
    devices: np.ndarray,
    num_shards: int,
) -> Tuple[np.ndarray, np.ndarray]:
  """Derives per-device (host_idx, local_slot) for FFI, grouping by physical host.

  Under single-controller Pathways (`proxy` platform), every device reports
  `process_index == 0`, and split-physical-axis meshes (e.g. a 4x4x4 v5p slice
  reshaped to `(fsdp=32, context=2)`) interleave devices from different
  physical hosts in `mesh.devices.flat`. Grouping by `mesh.device_host_key`
  ensures all devices on a physical host join that host's
  `WeightSynchronizerBase` instance (`submanager_idx = host_idx`,
  `local_slot in 0..num_shards-1`) regardless of mesh axis ordering.

  Args:
    devices: NumPy object array of JAX devices from `mesh.devices`.
    num_shards: Number of TPU devices per physical host.

  Returns:
    A tuple `(host_indices, local_slots)` of `int32` NumPy arrays with the same
    shape as `devices`.
  """
  if num_shards <= 0:
    raise ValueError(f"num_shards must be positive, got {num_shards}")
  flat_devices = list(devices.flat)
  num_devices = len(flat_devices)
  if num_devices % num_shards != 0:
    raise ValueError(
        f"Mesh has {num_devices} devices, which is not a multiple of"
        f" num_shards={num_shards}."
    )
  raw_keys = [mesh.device_host_key(d) for d in flat_devices]
  if num_devices <= num_shards:
    flat_indices = np.arange(num_devices, dtype=np.int32)
    return (
        (flat_indices // num_shards).reshape(devices.shape),
        (flat_indices % num_shards).reshape(devices.shape),
    )
  if any(k is None for k in raw_keys):
    raise ValueError(
        "mesh.device_host_key returned None for a multi-host mesh of"
        f" {num_devices} devices (num_shards={num_shards})."
    )
  if len(set(raw_keys)) == 1:
    flat_indices = np.arange(num_devices, dtype=np.int32)
    return (
        (flat_indices // num_shards).reshape(devices.shape),
        (flat_indices % num_shards).reshape(devices.shape),
    )
  host_id_map: dict[tuple[int, int], int] = {}
  next_slot: dict[int, int] = {}
  host_indices = np.empty(num_devices, dtype=np.int32)
  local_slots = np.empty(num_devices, dtype=np.int32)
  for flat_idx, key in enumerate(raw_keys):
    assert key is not None
    if key not in host_id_map:
      host_id = len(host_id_map)
      host_id_map[key] = host_id
      next_slot[host_id] = 0
    host_id = host_id_map[key]
    slot = next_slot[host_id]
    if slot >= num_shards:
      raise ValueError(
          f"Host {key!r} (host_idx={host_id}) owns more than"
          f" num_shards={num_shards} mesh devices."
      )
    host_indices[flat_idx] = host_id
    local_slots[flat_idx] = slot
    next_slot[host_id] = slot + 1
  for host_id, count in next_slot.items():
    if count != num_shards:
      raise ValueError(
          f"Host index {host_id} owns {count} mesh devices, expected"
          f" num_shards={num_shards}."
      )
  return (
      host_indices.reshape(devices.shape),
      local_slots.reshape(devices.shape),
  )


class _RaidenFfiModule(Protocol):
  """Structural protocol for the `raiden_jax.ffi` module."""

  _derive_host_slots: Callable[
      [np.ndarray, int], Tuple[np.ndarray, np.ndarray]
  ]
  _prepare_shard_info: Callable[
      [jax.Array, jax.sharding.Mesh, int], jax.Array
  ]
  _derive_slot_numa_nodes: Callable[
      [np.ndarray, np.ndarray, np.ndarray, int], np.ndarray
  ]
  _resolve_device_numa_nodes: Callable[[jax.sharding.Mesh], np.ndarray]


# TODO(b/557061810): Remove `_patch_ffi_host_slots` once the pinned
# `tpu_sync_jax` wheel in `oss/scripts/install_raiden.sh` accepts explicit
# `host_slots` in `start_weight_synchronizer`.
@contextlib.contextmanager
def _patch_ffi_host_slots(
    raiden_ffi: _RaidenFfiModule,
    host_indices: np.ndarray,
    local_slots: np.ndarray,
):
  """Patches `raiden_ffi` so non-contiguous Pathways meshes group by host."""
  orig_derive = raiden_ffi._derive_host_slots  # pylint: disable=protected-access
  orig_prepare = raiden_ffi._prepare_shard_info  # pylint: disable=protected-access

  def _derive_fixed(
      devices: np.ndarray, num_shards: int
  ) -> Tuple[np.ndarray, np.ndarray]:
    del devices, num_shards
    return host_indices, local_slots

  def _prepare_host_major_shard_info(
      shard_idx: jax.Array,
      array_mesh: jax.sharding.Mesh,
      num_shards: int,
  ) -> jax.Array:
    if shard_idx.ndim > len(array_mesh.axis_names) and shard_idx.shape[-1] >= 3:
      return shard_idx
    slot_numa_nodes_np = raiden_ffi._derive_slot_numa_nodes(  # pylint: disable=protected-access
        raiden_ffi._resolve_device_numa_nodes(array_mesh),  # pylint: disable=protected-access
        host_indices,
        local_slots,
        num_shards,
    )
    spec = jax.sharding.PartitionSpec(*array_mesh.axis_names)
    sharding = jax.sharding.NamedSharding(array_mesh, spec)
    local_slots_arr = jax.device_put(
        jnp.array(local_slots, dtype=jnp.int32), sharding
    )
    host_indices_arr = jax.device_put(
        jnp.array(host_indices, dtype=jnp.int32), sharding
    )
    # Assign host-major global_mesh_idx = host_idx * num_shards + local_slot in
    # column 3 while keeping column 0 (`shard_idx`) as the flat mesh index.
    global_mesh_indices_np = (
        host_indices * np.int32(num_shards) + local_slots
    ).astype(np.int32)
    global_mesh_indices = jax.device_put(
        jnp.array(global_mesh_indices_np, dtype=jnp.int32), sharding
    )
    slot_numa_nodes = jax.device_put(
        jnp.array(slot_numa_nodes_np, dtype=jnp.int32),
        jax.sharding.NamedSharding(
            array_mesh, jax.sharding.PartitionSpec(*array_mesh.axis_names, None)
        ),
    )
    if shard_idx.shape != array_mesh.devices.shape:
      shard_idx = shard_idx.reshape(array_mesh.devices.shape)
    return jnp.concatenate(
        [
            jnp.stack(
                [
                    shard_idx,
                    local_slots_arr,
                    host_indices_arr,
                    global_mesh_indices,
                ],
                axis=-1,
            ),
            slot_numa_nodes,
        ],
        axis=-1,
    )

  raiden_ffi._derive_host_slots = _derive_fixed  # pylint: disable=protected-access
  raiden_ffi._prepare_shard_info = _prepare_host_major_shard_info  # pylint: disable=protected-access
  try:
    yield
  finally:
    raiden_ffi._derive_host_slots = orig_derive  # pylint: disable=protected-access
    raiden_ffi._prepare_shard_info = orig_prepare  # pylint: disable=protected-access


def _compute_global_shard_index(
    indices: Tuple[slice, ...], shape: Tuple[int, ...]
) -> int:
  """Flattens a shard's per-dimension slices into its row-major grid index."""
  shard_idx = 0
  stride = 1
  for s, dim in zip(reversed(indices), reversed(shape)):
    start = s.start or 0
    step = s.stop - start if s.stop is not None else dim
    num_shards = dim // step if step > 0 else 1
    idx = start // step if step > 0 else 0
    shard_idx += idx * stride
    stride *= num_shards
  return shard_idx


def _global_shard_indices(
    arr: Any,
    sharding: Any,
    devices: Optional[Sequence[jax.Device]] = None,
) -> Tuple[int, ...]:
  """Grid index of each shard `arr` is published from, in shard order.

  A proxy (Pathways) array holds no local data, so the shards of every mesh
  device are listed instead of the addressable ones. When `devices` is supplied
  (e.g. host-grouped FFI shard order), shard indices follow that exact device
  sequence so `WorkUnitMetadata.variables[i].global_shard_indices[p]` aligns
  with `WorkUnitMetadata.shards[p]`.

  Args:
    arr: The array.
    sharding: `arr.sharding`.
    devices: Optional explicit device sequence overriding `mesh.devices.flat`.

  Returns:
    One flattened grid index per shard.
  """
  shape = tuple(arr.shape)
  index_map = sharding.devices_indices_map(shape)
  if devices is not None:
    return tuple(
        _compute_global_shard_index(index_map[d], shape)
        for d in devices
        if d in index_map
    )
  shards = getattr(arr, "addressable_shards", None)
  if shards and not any(s.device.platform == "proxy" for s in shards):
    return tuple(_compute_global_shard_index(s.index, shape) for s in shards)
  mesh_obj = getattr(sharding, "mesh", None)
  mesh_devices = (
      mesh_obj.devices.flat if hasattr(mesh_obj, "devices") else index_map
  )
  return tuple(
      _compute_global_shard_index(index_map[d], shape)
      for d in mesh_devices
      if d in index_map
  )


def _compute_mesh_global_shard_indices(
    arrays: List[Any], array_mesh: Optional[Any]
) -> Optional[List[int]]:
  """Global index of each local device, in addressable-shard order.

  Mirrors the default the native `WeightSynchronizer` derives. Passing it
  explicitly makes `get_local_endpoints()` name shards by these indices.

  Args:
    arrays: The bound arrays.
    array_mesh: The mesh the arrays are sharded over, if any.

  Returns:
    The indices, or None when the leaves carry no shards.
  """
  if not arrays or not hasattr(arrays[0], "addressable_shards"):
    return None
  shards = arrays[0].addressable_shards
  if hasattr(array_mesh, "devices"):
    position = {d: i for i, d in enumerate(array_mesh.devices.flat)}
    if shards and all(s.device in position for s in shards):
      return [position[s.device] for s in shards]
  offset = (
      jax.process_index() * len(jax.local_devices())
      if jax.process_count() > 1
      else 0
  )
  return [offset + i for i in range(len(shards))]


def _routable_address(endpoint: str, ip: str) -> str:
  """Replaces a wildcard or loopback host in `endpoint` with `ip`."""
  host, sep, port = endpoint.partition(":")
  if sep and host in ("", "0.0.0.0", "127.0.0.1"):
    return f"{ip}:{port}"
  return endpoint


def _tensor_metadata(
    name: str,
    arr: Any,
    layer_idx: int,
    *,
    devices: Optional[Sequence[jax.Device]] = None,
):
  """Constructs `TensorMetadata` for `arr`."""
  sharding: Any = getattr(arr, "sharding", None)
  spec = tuple(getattr(sharding, "spec", ()) or ())
  spec = (spec + (None,) * arr.ndim)[: arr.ndim]
  if sharding is not None and hasattr(sharding, "shard_shape"):
    try:
      local = sharding.shard_shape(tuple(arr.shape))
      mesh_shape = tuple(g // l for g, l in zip(arr.shape, local))
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Could not compute mesh_shape for %s from sharding: %s, falling back"
          " to 1D",
          name,
          e,
      )
      mesh_shape = (1,) * arr.ndim
  else:
    mesh_shape = (1,) * arr.ndim
  return weight_sync.TensorMetadata(
      name=name,
      shape=tuple(arr.shape),
      mesh_shape=mesh_shape,
      layout=tuple(reversed(range(arr.ndim))),
      item_size=arr.dtype.itemsize,
      layer_idx=layer_idx,
      sharding_spec=tuple(_axis_name(a) for a in spec),
      global_shard_indices=(
          _global_shard_indices(arr, sharding, devices=devices)
          if hasattr(sharding, "devices_indices_map")
          else ()
      ),
  )


class RaidenSynchronizer:
  """One host's weights on the raiden transport, plus its registration metadata.

  Used by both the trainer and the sampler. Construct with a state to bind
  right away, or leave it out and call `bind` when the weights exist; every
  later `bind` rebinds the same transport. Without the tpu_sync wheel the
  metadata carries no shard addresses, so the handler refuses registration.
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
    self._control_addr: str = ""
    self._ffi_mesh: Any = None
    self._ffi_devices: Optional[Tuple[jax.Device, ...]] = None
    self._global_shard_indices: Optional[List[int]] = None
    if state is not None:
      self.bind(state)

  @property
  def bound(self) -> bool:
    return bool(self.names)

  @property
  def active(self) -> bool:
    # Under FFI there is no native `_sync` and `_ips` fill only in d2h(), so
    # bound arrays are the only pre-d2h signal. MaxText gates d2h() on this.
    if self._is_proxy:
      return self.bound
    return self._sync is not None or bool(self._ips)

  def _init_ffi_transport(self, *, is_d2h: bool) -> None:
    if not self.arrays:
      raise RuntimeError(
          f"{self.job_name}: bind() must stage arrays before FFI init"
      )
    raiden_ffi = _get_raiden_ffi()
    if raiden_ffi is None:
      raise RuntimeError(
          "weight_synchronizer_ffi is not available for FFI weight sync."
      )
    api_version = getattr(raiden_ffi, "ENDPOINTS_API_VERSION", None)
    if api_version != 1:
      raise RuntimeError(
          "Incompatible tpu_sync weight_synchronizer_ffi"
          f" (ENDPOINTS_API_VERSION={api_version!r}, expected 1); reinstall"
          " via scripts/install_raiden.sh."
      )

    _ensure_ffi_compute_on_compat()
    sharding = self.arrays[0].sharding
    if not isinstance(sharding, jax.sharding.NamedSharding):
      raise ValueError(
          "Arrays must use NamedSharding for FFI weight sync, got"
          f" {type(sharding).__name__}."
      )
    array_mesh = sharding.mesh

    src_devices = list(array_mesh.devices.flat)
    devices_per_host = _devices_per_host(src_devices)
    # Loud on purpose: a wrong value here is silent, and costs exactly the
    # shards of every host but one.
    logging.warning(
        "raiden ffi: %d device(s), devices_per_host=%d (task_id=%s)",
        len(src_devices),
        devices_per_host,
        sorted({getattr(d, "task_id", None) for d in src_devices}, key=str),
    )

    logging.info(
        "Initializing Pathways weight synchronizer via FFI (d2h=%s, %d layers,"
        " %d devices/host)",
        is_d2h,
        len(self.arrays),
        devices_per_host,
    )
    host_indices, local_slots = _derive_ffi_host_slots(
        array_mesh.devices, devices_per_host
    )
    contiguous_hosts = [
        i // devices_per_host for i in range(len(src_devices))
    ]
    if host_indices.reshape(-1).tolist() == contiguous_hosts:
      eps = raiden_ffi.start_weight_synchronizer(
          self.arrays,
          array_mesh,
          d2h=is_d2h,
          parallelism=self._parallelism,
          listener_port=0,
          num_shards=devices_per_host,
      )
      self._ips, self._control_addr = eps.single_unit_registration()
      self._ffi_devices = None
    else:
      with _patch_ffi_host_slots(raiden_ffi, host_indices, local_slots):
        eps = raiden_ffi.start_weight_synchronizer(
            self.arrays,
            array_mesh,
            d2h=is_d2h,
            parallelism=self._parallelism,
            listener_port=0,
            num_shards=devices_per_host,
        )
      ips: list[str] = []
      ffi_devices: list[jax.Device] = []
      for host in eps.hosts:
        if len(host.shards) != devices_per_host:
          raise RuntimeError(
              f"Host {host.host_index} reported {len(host.shards)} shards,"
              f" expected {devices_per_host}."
          )
        for s in host.shards:
          ips.append(s.data_address)
          ffi_devices.append(s.device)
      self._ips = ips
      self._control_addr = ",".join(h.control_address for h in eps.hosts)
      self._ffi_devices = tuple(ffi_devices)
    logging.warning(
        "raiden ffi endpoints: %d shard(s) across %d host(s) for %d mesh"
        " device(s); mesh ids=%s; endpoints=%s; control=%s",
        len(self._ips),
        len(eps.hosts),
        len(src_devices),
        [d.id for d in src_devices],
        self._ips,
        self._control_addr,
    )
    self._ffi_mesh = array_mesh

  def _ffi_h2d(self) -> None:
    raiden_ffi = _get_raiden_ffi()
    if raiden_ffi is None:
      raise RuntimeError(
          "weight_synchronizer_ffi is not available for FFI weight sync."
      )
    if self._ffi_mesh is None:
      raise RuntimeError(f"{self.job_name}: bind() must run before h2d()")
    self.arrays = list(
        raiden_ffi.multi_h2d(self.arrays, mesh=self._ffi_mesh)
    )
    jax.block_until_ready(self.arrays)

  def bind(self, state: Any) -> None:
    """Binds this host's weights, or rebinds them after a training step."""
    _log_rss("bind:start")
    # Clear previous buffers before staging to avoid holding duplicate weight
    # copies in host memory during rebinds.
    self.names = []
    self.arrays = []
    self.names, self.arrays = _filter_bindable(
        # Proxy arrays are bindable only under FFI, which binds them in place;
        # a non-Pathways process should not be seeing them at all.
        *flatten_weights(state),
        allow_proxy=self._is_proxy,
    )
    del state
    _log_rss("bind:after_flatten")
    array_mesh = None
    for arr in self.arrays:
      array_mesh = getattr(getattr(arr, "sharding", None), "mesh", None)
      if array_mesh is not None:
        break
    logging.info(
        "%s bind prepared %d arrays (proxy_runtime=%s)",
        self.job_name,
        len(self.arrays),
        self._is_proxy,
    )
    if self._is_proxy:
      self._ips = []
      self._control_addr = ""
      self._ffi_devices = None
      if self._auto_h2d:
        self._init_ffi_transport(is_d2h=False)
        logging.info(
            "%s FFI destination transport ready: shards=%s control=%s",
            self.job_name,
            self._ips,
            self._control_addr,
        )
      return
    ws_lib = _get_ws_lib()
    if ws_lib is None:
      return
    if self._sync is None:
      logging.info(
          "%s creating native WeightSynchronizer for %d arrays",
          self.job_name,
          len(self.arrays),
      )
      # Fixed for the synchronizer's lifetime; rebinds keep the same devices.
      self._global_shard_indices = _compute_mesh_global_shard_indices(
          self.arrays, array_mesh
      )
      self._sync = ws_lib.WeightSynchronizer(
          self.arrays,
          local_port=0,
          parallelism=self._parallelism,
          # Rebinding deadlocks on the retained usage holds otherwise; the
          # caller keeps Python references to the bound arrays regardless.
          unsafe_skip_buffer_lock=True,
          listener_port=0,
          bind_ip=None,
          auto_h2d=self._auto_h2d,
          global_shard_indices=self._global_shard_indices,
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
      if not self.bound:
        raise RuntimeError(f"{self.job_name}: bind() must run before {op}")
      raise RuntimeError(
          f"{self.job_name}: tpu_sync native weight_synchronizer library "
          "(_ws_lib) could not be loaded."
      )
    return self._sync

  def d2h(self) -> None:
    if self._is_proxy:
      try:
        self._init_ffi_transport(is_d2h=True)
        logging.info(
            "FFI D2H complete. Shards: %s, Control plane: %s",
            self._ips,
            self._control_addr,
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

  # TODO(tunix-dev): drop this once bind() records the runner's leaf identity so
  # h2d() writes back in place, making the name matching below unnecessary.
  def apply_to_runner(self, runner: Any) -> None:
    """Applies updated arrays after H2D to the runner's state_leaves and state."""
    if runner is None or not self.arrays:
      return
    if not hasattr(runner, "state_leaves") or runner.state_leaves is None:
      raise ValueError(
          f"{self.job_name}: runner does not have a valid 'state_leaves'"
          " attribute."
      )
    if not hasattr(runner, "state") or runner.state is None:
      raise ValueError(
          f"{self.job_name}: runner does not have a valid 'state' attribute."
      )

    new_leaves = list(runner.state_leaves)
    runner_leaves_with_path = list(
        jax.tree_util.tree_leaves_with_path(runner.state)
    )
    if len(new_leaves) != len(runner_leaves_with_path):
      raise RuntimeError(
          f"{self.job_name}: runner.state_leaves length ({len(new_leaves)})"
          " does not match runner.state leaves count"
          f" ({len(runner_leaves_with_path)})."
      )

    key_to_entries: dict[str, List[Any]] = collections.defaultdict(list)
    for idx, (name, arr) in enumerate(zip(self.names, self.arrays)):
      key_to_entries[_param_key(name)].append((idx, name, arr))

    matched_indices = set()

    def _claim(key: str):
      """First unapplied array whose key equals, or is a suffix of, `key`."""
      for entry in key_to_entries.get(key, ()):
        if entry[0] not in matched_indices:
          return entry
      # A runner path may carry a wrapper prefix the bound name lacks; match on
      # a whole-segment suffix so `a.b.w` still finds `b.w`, never `ab.w`.
      for candidate, entries in key_to_entries.items():
        if not key.endswith("." + candidate):
          continue
        for entry in entries:
          if entry[0] not in matched_indices:
            return entry
      return None

    for i, (path, leaf) in enumerate(runner_leaves_with_path):
      p_str = jax.tree_util.keystr(path)
      entry = _claim(_param_key(p_str))

      if entry is not None:
        idx, orig_name, arr = entry
        leaf_arr = getattr(leaf, "value", leaf)
        if hasattr(leaf_arr, "shape") and leaf_arr.shape != arr.shape:
          raise ValueError(
              f"Shape mismatch for parameter '{orig_name}' (runner path"
              f" '{p_str}'): runner shape {leaf_arr.shape} vs synchronizer"
              f" shape {arr.shape}"
          )
        new_leaves[i] = arr
        matched_indices.add(idx)

    if len(matched_indices) != len(self.arrays):
      unmatched = [
          self.names[j]
          for j in range(len(self.arrays))
          if j not in matched_indices
      ]
      raise RuntimeError(
          f"{self.job_name}: Not all synchronizer arrays were matched in"
          f" runner.state! Matched {len(matched_indices)} of {len(self.arrays)}"
          f" arrays. Unmatched {len(unmatched)} parameters, e.g.:"
          f" {unmatched[:10]}"
      )

    runner.state_leaves = tuple(new_leaves)
    runner.state = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(runner.state), new_leaves
    )
    logging.info(
        "%s apply_to_runner: successfully applied %d arrays to runner state and"
        " state_leaves (total runner leaves: %d).",
        self.job_name,
        len(matched_indices),
        len(new_leaves),
    )

  def work_unit_metadata_all(self) -> List[weight_sync.WorkUnitMetadata]:
    """Returns work unit metadata for registration.

    In proxy/FFI mode, control_plane_rpc_address contains comma-separated
    per-host control addresses. The coordinator registers a single work unit and
    the Raiden controller broadcasts to all hosts in parallel.
    """
    return [self.work_unit_metadata()]

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
    # Registration pairs tensors by position, so the totals only compare when
    # both sides bound the same set. Check these before trusting a mismatch.
    head["__tensor_count__"] = len(self.arrays)
    head["__element_count__"] = int(sum(a.size for a in self.arrays))
    return head

  def _shard_addresses(self) -> Tuple[str, ...]:
    """Data-plane address of each local shard.

    Raiden's weight synchronizer reports one endpoint, its bound data NIC, for
    every shard; a multi-NUMA host may report one per NUMA node. Shards no
    endpoint covers use the default port.

    Returns:
      One `host:port` per local shard, in shard order; empty before the native
      synchronizer exists.
    """
    if self._sync is None:
      return ()
    num_shards = self._sync.num_shards
    # Endpoints name shards by the global indices passed at construction, or
    # by local index when Raiden derived them itself.
    local_index = {g: i for i, g in enumerate(self._global_shard_indices or ())}
    shards = [""] * num_shards
    for endpoint in self._sync.get_local_endpoints():
      address = _routable_address(endpoint["endpoint"], self.ip)
      for shard in endpoint["shards"]:
        i = local_index.get(int(shard), int(shard))
        if 0 <= i < num_shards:
          shards[i] = address
    default = f"{self.ip}:{self._sync.local_port}"
    return tuple(s or default for s in shards)

  def work_unit_metadata(self) -> weight_sync.WorkUnitMetadata:
    mesh = None
    for arr in self.arrays:
      mesh = getattr(getattr(arr, "sharding", None), "mesh", None)
      if mesh is not None:
        break
    if mesh is None:
      mesh_axes, mesh_shape = ("fsdp",), (1,)
    else:
      # Advertise the same mesh the shards were built on.
      mesh_axes = tuple(mesh.axis_names)
      mesh_shape = tuple(int(mesh.shape[a]) for a in mesh.axis_names)
    # Publish the canonical key, not the raw keystr: the controller pairs
    # variables by EXACT name, and the two sides root the same tree
    # differently (trainer `['base'][...]` vs rollout `['model'][...]`, plus
    # the nnx `.value` leaf). `_param_key` already normalises both away, so
    # canonicalising here is what makes the manifests line up.
    variables = tuple(
        _tensor_metadata(
            _param_key(name), arr, idx, devices=self._ffi_devices
        )
        for idx, (name, arr) in enumerate(zip(self.names, self.arrays))
    )
    if self._is_proxy:
      shards = tuple(self._ips)
      control_addr = self._control_addr
    else:
      control_addr = (
          f"{self.ip}:{self._sync.listener_port}"
          if self._sync and self._sync.listener_port
          else ""
      )
      shards = self._shard_addresses()
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
        transport_mode="ffi" if self._is_proxy else "tcp",
        use_ffi=self._is_proxy,
    )


def patch_raiden_worker_sync() -> None:
  """Monkey-patches tpu_inference.rl.raiden_worker_sync.RaidenWorkerSync to delegate apply_to_runner."""
  if os.environ.get("JAX_PLATFORMS") == "cpu":
    return
  # Resolved through importlib, like the other optional deps in this module, so
  # static dependency analysis does not try to follow tpu-inference -- it is not
  # a declared dependency and is absent in many environments.
  rws = _lazy_import_module("tpu_inference.rl.raiden_worker_sync")
  if rws is None:
    logging.debug("tpu_inference not available to patch")
    return
  try:
    if getattr(rws.RaidenWorkerSync, "_patched_by_tunix", False):
      return
    orig_apply = getattr(rws.RaidenWorkerSync, "apply_to_runner", None)

    def _patched_apply_to_runner(self, runner: Any) -> None:
      if self._sync is not None and hasattr(self._sync, "apply_to_runner"):
        self._sync.apply_to_runner(runner)
        return
      if orig_apply is not None:
        orig_apply(self, runner)

    rws.RaidenWorkerSync.apply_to_runner = _patched_apply_to_runner
    rws.RaidenWorkerSync._patched_by_tunix = True
    logging.info(
        "Successfully patched RaidenWorkerSync.apply_to_runner with Tunix"
        " delegation."
    )
  except AttributeError as e:
    logging.debug("tpu_inference not available to patch: %s", e)
