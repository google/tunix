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

"""Per-host synchronizer wrapper: binds the synchronizer, emits WorkUnitMetadata."""

from __future__ import annotations

import socket
from typing import Any, List, Optional, Tuple

from absl import logging
import jax
from tunix.experimental.orchestrator import weight_sync

_ws_lib: Any = None
try:
  from GOOGLE_INTERNAL_PACKAGE_PATH.third_party.tpu_raiden.tpu_sync.api.jax import weight_synchronizer as _ws_lib
except ImportError:
  try:
    from tpu_sync.api.jax import weight_synchronizer as _ws_lib  # pytype: disable=import-error
  except ImportError:
    _ws_lib = None


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


def to_host_cpu_state(state: Any) -> Any:
  """Pulls arrays to client host memory; proxy arrays cannot bind directly."""
  cpu = jax.local_devices(backend="cpu")[0]

  def pull(leaf):
    arr = getattr(leaf, "value", leaf)
    if hasattr(arr, "shape") and hasattr(arr, "dtype"):
      return jax.device_put(jax.device_get(arr), cpu)
    return leaf

  return jax.tree_util.tree_map(pull, state)


def flatten_weights(state: Any) -> Tuple[List[str], List[Any]]:
  """Returns (names, arrays) for every array leaf, in stable tree order."""
  names, arrays = [], []
  for path, leaf in jax.tree_util.tree_leaves_with_path(state):
    arr = getattr(leaf, "value", leaf)
    if hasattr(arr, "shape") and hasattr(arr, "dtype"):
      names.append(jax.tree_util.keystr(path))
      arrays.append(arr)
  return names, arrays


def _filter_bindable(
    names: List[str], arrays: List[Any]
) -> Tuple[List[str], List[Any]]:
  """Drops leaves the native layer cannot bind; binding them is undefined
  behavior (observed: random RuntimeError or SIGSEGV on RNG-key arrays)."""
  census: dict = {}
  keep_names: List[str] = []
  keep_arrays: List[Any] = []
  dropped = []
  for name, arr in zip(names, arrays):
    if hasattr(arr, "devices"):
      plats = tuple(sorted({getattr(d, "platform", "?") for d in arr.devices()}))
    else:
      plats = ("no-devices",)
    key = (type(arr).__name__, str(getattr(arr, "dtype", "?")), plats)
    census[key] = census.get(key, 0) + 1
    ok = hasattr(arr, "devices") and all(
        p.lower() in ("cpu", "tpu") for p in plats
    )
    try:
      ok = ok and not jax.dtypes.issubdtype(arr.dtype, jax.dtypes.extended)
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    if ok:
      keep_names.append(name)
      keep_arrays.append(arr)
    else:
      dropped.append((name, key))
  logging.info("raiden bind census: %s", census)
  if dropped:
    logging.warning(
        "raiden bind dropped %d incompatible leaves: %s",
        len(dropped),
        dropped[:5],
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
  spec = spec + (None,) * (arr.ndim - len(spec))
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
  """One host's weight synchronizer plus its registration metadata.

  Shared by the trainer (host-staged CPU arrays) and the rollout worker
  (device arrays). Known limits: one mesh axis per tensor dim (multi-axis
  dim shardings are rejected by the handler's manifest validation), and
  without the tpu_sync wheel the metadata carries no shard addresses, so
  registration will be refused by the handler.
  """

  def __init__(
      self,
      job_name: str,
      state: Any,
      *,
      auto_h2d: bool = False,
      parallelism: int = 4,
      bind_ip: Optional[str] = None,
  ):
    self.job_name = job_name
    self.names, self.arrays = _filter_bindable(*flatten_weights(state))
    self.ip = bind_ip or local_ip()
    self._sync: Any = None
    if _ws_lib is not None:
      self._sync = _ws_lib.WeightSynchronizer(
          self.arrays,
          local_port=0,
          parallelism=parallelism,
          # Rebinding deadlocks on the retained usage holds otherwise; the
          # caller keeps Python references to the bound arrays regardless.
          unsafe_skip_buffer_lock=True,
          listener_port=0,
          bind_ip=None,
          auto_h2d=auto_h2d,
      )

  @property
  def active(self) -> bool:
    return self._sync is not None

  def d2h(self) -> None:
    if self._sync:
      self._sync.d2h()

  def h2d(self) -> None:
    if self._sync:
      self._sync.h2d()

  def rebind(self, state: Any) -> None:
    """Rebinds after the training step replaced the weight arrays."""
    self.names, self.arrays = _filter_bindable(*flatten_weights(state))
    if self._sync:
      self._sync.bind_weights(self.arrays)

  def metrics(self) -> dict:
    return self._sync.get_metrics() if self._sync else {}

  def checksums(self, sample: int = 3) -> dict:
    """Per-tensor float32 abs-sums for cross-process verification."""
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

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
      mesh_axes = ("fsdp", "tp")
      mesh_shape = (1, 1)
    data_addr = (
        f"{self.ip}:{self._sync.local_port}" if self._sync else ""
    )
    control_addr = (
        f"{self.ip}:{self._sync.listener_port}"
        if self._sync and self._sync.listener_port
        else ""
    )
    num_shards = self._sync.num_shards if self._sync else 1
    return weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId(job_name=self.job_name),
        shards=(data_addr,) * num_shards if data_addr else (),
        control_plane_rpc_address=control_addr,
        mesh_shape=mesh_shape,
        variables=variables,
        mesh_axes=mesh_axes or None,
    )