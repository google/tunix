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
import gc
import resource
import socket
from typing import Any, List, Optional, Tuple

from absl import logging
import jax
import jax.numpy as jnp
from tunix.experimental.orchestrator import weight_sync


def _log_rss(tag: str) -> None:
  """Logs process peak RSS (GB) -- pinpoints which bind() stage spikes host
  memory, since ru_maxrss is a high-water mark that only grows.
  """
  rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
  logging.info("raiden bind rss checkpoint [%s]: %.1f GB (peak)", tag, rss_gb)

_ws_lib: Any = None
try:
  from tpu_sync.api.jax import weight_synchronizer as _ws_lib  # pylint: disable=g-import-not-at-top
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
  """Pulls arrays to client host memory; proxy arrays cannot bind directly.

  Drains the input tree's leaves as they're processed (rather than
  `jax.tree_util.tree_map`, which holds the entire input tree alive while
  building the entire output tree) so peak host memory stays close to one
  copy of the state, not two -- at 30B-A3B scale, with padded MoE weights,
  keeping both alive at once is the difference between fitting in host RAM
  and an OOMKill during Raiden's D2H staging.
  """
  cpu = jax.local_devices(backend="cpu")[0]
  leaves, treedef = jax.tree_util.tree_flatten(state)
  del state
  new_leaves = []
  for i in range(len(leaves)):
    leaf = leaves[i]
    leaves[i] = None
    arr = getattr(leaf, "value", leaf)
    if hasattr(arr, "shape") and hasattr(arr, "dtype"):
      new_leaves.append(jax.device_put(jax.device_get(arr), cpu))
    else:
      new_leaves.append(leaf)
    del leaf, arr
    # A dropped proxy-backed source array only releases its Pathways-proxy
    # transit buffer once Python actually collects it. At 30B-A3B scale
    # (100+ tensors, tens of GB each) relying on incidental refcount-zero
    # collection let the proxy's staged-but-uncollected buffers pile up
    # alongside this container's own host copy, and their combined size
    # exceeded node-level free memory (a kubelet eviction, not a cgroup
    # OOM) even though no single container had crossed its own limit.
    # Forcing a collection every few leaves keeps that pileup bounded.
    if i % 4 == 3:
      gc.collect()
  del leaves
  gc.collect()
  return jax.tree_util.tree_unflatten(treedef, new_leaves)


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
  """True if the native layer can bind this leaf.

  Binding an unsupported leaf (e.g. RNG keys) can SIGSEGV, so only
  floating-point, rank>=1, fully TPU- or CPU-resident arrays qualify. CPU is
  allowed because host_stage deliberately copies proxy-backed (Pathways)
  arrays to host CPU memory before bind() gets here -- rejecting "not TPU"
  would drop every leaf it just staged.
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
    return all(getattr(d, "platform", "?") in ("tpu", "cpu") for d in devices)
  except Exception:
    return False


def _filter_bindable(
    names: List[str], arrays: List[Any]
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
    if _bindable(arr):
      if hasattr(arr, "block_until_ready"):
        try:
          # binding an in-flight buffer is part of what SIGSEGVs
          arr.block_until_ready()
        except Exception:
          pass
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
      host_stage: bool = False,
      parallelism: int = 4,
      bind_ip: Optional[str] = None,
  ):
    self.job_name = job_name
    self.worker_index = worker_index
    self.names: List[str] = []
    self.arrays: List[Any] = []
    self.ip = bind_ip or local_ip()
    self._auto_h2d = auto_h2d
    self._host_stage = host_stage
    self._parallelism = parallelism
    self._sync: Any = None
    if state is not None:
      self.bind(state)

  @property
  def bound(self) -> bool:
    return bool(self.names)

  @property
  def active(self) -> bool:
    return self._sync is not None

  def bind(self, state: Any) -> None:
    """Binds this host's weights, or rebinds them after a training step.

    With host_stage the arrays are copied to local CPU memory first; arrays
    backed by the pathways proxy cannot bind in place.
    """
    _log_rss("bind:start")
    if self._host_stage:
      state = to_host_cpu_state(state)
    _log_rss("bind:after_host_stage")
    self.names, self.arrays = _filter_bindable(*flatten_weights(state))
    del state
    _log_rss("bind:after_flatten")
    if _ws_lib is None:
      return
    if self._sync is None:
      self._sync = _ws_lib.WeightSynchronizer(
          self.arrays,
          local_port=0,
          parallelism=self._parallelism,
          # rebinding deadlocks on retained usage holds otherwise
          unsafe_skip_buffer_lock=True,
          listener_port=0,
          bind_ip=None,
          auto_h2d=self._auto_h2d,
      )
      _log_rss("bind:after_native_construct")
    else:
      self._sync.bind_weights(self.arrays)
      _log_rss("bind:after_native_rebind")

  def _require_sync(self, op: str) -> Any:
    if self._sync is None:
      raise RuntimeError(f"{self.job_name}: bind() must run before {op}")
    return self._sync

  def d2h(self) -> None:
    self._require_sync("d2h()").d2h()

  def h2d(self) -> None:
    # native h2d() dispatches an async device copy; block so a
    # checksum/read immediately after doesn't observe pre-transfer buffer
    # contents.
    self._require_sync("h2d()").h2d()
    jax.block_until_ready(self.arrays)

  def metrics(self) -> dict:
    if self._sync is None:
      return {}
    if hasattr(self._sync, "get_metrics"):
      return self._sync.get_metrics()
    if hasattr(self._sync, "metrics"):
      return self._sync.metrics()
    return {}

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
    # layer_idx must be a *unique* slot per tensor: the native transport
    # indexes host buffers directly by it (layer_names_[l],
    # GetHostPointer(layer_idx_to_use, i)). A name-derived index that
    # defaults non-per-layer tensors (e.g. decoder_norm.scale, embeddings)
    # to 0 collides them with layer 0's own tensors in that same slot,
    # corrupting its buffer bookkeeping -- this was confirmed as the root
    # cause of a native "Push range out of bounds" weight-sync error.
    # Position in the (sorted-by-key) flatten order is what's unique here.
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
    data_addr = (
        f"{self.ip}:{self._sync.local_port}" if self._sync else ""
    )
    control_addr = (
        f"{self.ip}:{self._sync.listener_port}"
        if self._sync and self._sync.listener_port
        else ""
    )
    num_shards = self._sync.num_shards if self._sync else 1
    # Index 0 keeps the default replica id "": transfer callers construct
    # WorkUnitId(job_name=...) without a replica, and registration lookups
    # must match it for single-replica units.
    unit = weight_sync.WorkUnitId(
        job_name=self.job_name,
        job_replica_id=str(self.worker_index) if self.worker_index else "",
    )
    return weight_sync.WorkUnitMetadata(
        unit=unit,
        shards=(data_addr,) * num_shards if data_addr else (),
        control_plane_rpc_address=control_addr,
        mesh_shape=mesh_shape,
        variables=variables,
        mesh_axes=mesh_axes or None,
    )