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

"""Sharding utilities."""

from collections.abc import Mapping, Sequence
import json
from typing import Any, Tuple

import jax
from jax import numpy as jnp
from jax.experimental import multihost_utils
from jax.interpreters import pxla
import jax.sharding as shd
import numpy as np


def place_array_on_target(
    arr: jax.Array | np.ndarray,
    target: shd.Sharding | jax.Device,
) -> jax.Array:
  """Places a host or device array onto a local device or single-/multi-host sharding.

  Args:
    arr: Input `jax.Array` or `np.ndarray` to place.
    target: Target `jax.sharding.Sharding` or `jax.Device`.

  Returns:
    A `jax.Array` placed on `target`.

  Raises:
    ValueError: If `arr` is a non-fully-addressable `jax.Array` whose existing
      sharding does not match `target`.
  """
  if isinstance(arr, jax.Array):
    if arr.sharding == target:
      return arr
    if not arr.is_fully_addressable:
      raise ValueError(
          'Cannot re-place a non-fully-addressable multi-host jax.Array '
          f'(current sharding={arr.sharding}, target={target}).'
      )
  if isinstance(target, shd.Sharding) and not target.is_fully_addressable:
    host_arr = np.asarray(jax.device_get(arr))
    return jax.make_array_from_callback(
        host_arr.shape, target, lambda idx: host_arr[idx]
    )
  with jax.transfer_guard('allow'):
    return jax.device_put(arr, target)


def place_arrays_on_targets(
    arrays: Sequence[jax.Array | np.ndarray],
    targets: Sequence[shd.Sharding | jax.Device],
) -> list[jax.Array]:
  """Places a sequence of host or device arrays onto local or multi-host targets.

  Batches placement into a single `jax.device_put` call when all targets are
  fully addressable on this process, and falls back to per-array placement via
  `place_array_on_target` when any target sharding spans multiple hosts.

  Args:
    arrays: Input `jax.Array` or `np.ndarray` values to place.
    targets: Target `jax.sharding.Sharding` or `jax.Device` per input array.

  Returns:
    A list of `jax.Array` instances placed on `targets`.

  Raises:
    ValueError: If `arrays` and `targets` have different lengths.
  """
  if len(arrays) != len(targets):
    raise ValueError(
        f'Length mismatch: got {len(arrays)} arrays and {len(targets)} targets.'
    )
  if not arrays:
    return []
  if any(
      isinstance(t, shd.Sharding) and not t.is_fully_addressable
      for t in targets
  ):
    return [
        place_array_on_target(arr, target)
        for arr, target in zip(arrays, targets, strict=True)
    ]
  with jax.transfer_guard('allow'):
    return list(jax.device_put(list(arrays), list(targets)))


def spec_on_mesh(
    spec: shd.PartitionSpec | Sequence[str | tuple[str, ...] | None] | None,
    mesh: shd.Mesh,
) -> shd.PartitionSpec:
  """Returns `spec` with the axes that `mesh` does not have replicated.

  For example, `('fsdp', 'tp')` is `P(None, 'tp')` on a `('dp', 'tp')` mesh,
  and `(('dp', 'fsdp'), None)` is `P('dp', None)`.

  Args:
    spec: The partition spec, or None for a replicated array.
    mesh: The mesh to shard on.
  """

  def _axis_on_mesh(axis):
    axes = tuple(
        a
        for a in (axis if isinstance(axis, tuple) else (axis,))
        if a in mesh.axis_names
    )
    return axes if len(axes) > 1 else (axes[0] if axes else None)

  return shd.PartitionSpec(*(_axis_on_mesh(a) for a in spec or ()))


# TODO(abheesht17): Use this function for all models and unify with the fn in
# sft/sharding_utils.py.
def shard(
    x: jnp.ndarray | np.ndarray,
    s: Tuple[str | None, ...] | shd.PartitionSpec,
    eager: bool = False,
):
  """Shards a JAX array.

  Args:
    x: The JAX array to shard.
    s: The sharding spec.
    eager: If True, sharding is done eagerly via place_array_on_target.
      Otherwise, sharding is done lazily via jax.lax.with_sharding_constraint.

  Returns:
    The sharded JAX array.
  """
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return jnp.asarray(x)
  sharding = shd.NamedSharding(mesh, spec_on_mesh(s, mesh))
  if eager:
    return place_array_on_target(x, sharding)
  return jax.lax.with_sharding_constraint(x, sharding)


def broadcast_host0_json(payload: Any) -> Any:
  """Returns process 0's JSON-serializable `payload` on every process."""
  is_host0 = jax.process_index() == 0
  buf = (
      np.frombuffer(json.dumps(payload).encode('utf-8'), dtype=np.uint8)
      if is_host0
      else np.zeros(0, dtype=np.uint8)
  )
  if jax.process_count() > 1:
    size = multihost_utils.broadcast_one_to_all(
        np.int32(buf.size), is_source=is_host0
    )
    src = buf if is_host0 else np.zeros(int(size), dtype=np.uint8)
    buf = np.asarray(
        multihost_utils.broadcast_one_to_all(src, is_source=is_host0)
    )
  return json.loads(buf.tobytes())


def _needs_multihost_gather(arr: Any) -> bool:
  """Returns whether `arr` needs a cross-host allgather to materialize."""
  return (
      jax.process_count() > 1
      and isinstance(arr, jax.Array)
      and not arr.is_fully_addressable
  )


def gather_tensor_to_host(arr: Any) -> np.ndarray:
  """Materializes a single global or host-local tensor as a host NumPy array.

  Gating `multihost_utils.process_allgather(..., tiled=True)` on
  `_needs_multihost_gather` handles both single-controller and multi-process
  JAX runtimes cleanly:

  - On single-controller runtimes such as Pathways on Cloud
    (`jax.process_count() == 1`, `is_fully_addressable == True`),
    `process_allgather(arr, tiled=True)` also returns `np.asarray(arr)`
    (whereas `tiled=False` expands `()` to `(1,)`), so routing directly to
    `jax.device_get(arr)` avoids unnecessary `multihost_utils` indirection.
  - On multi-process JAX (McJAX, `jax.process_count() > 1`), calling
    `process_allgather(arr, tiled=True)` unconditionally on a host-local
    `np.ndarray`, fully-addressable `jax.Array`, or 0-D scalar stacks or
    concatenates across all `N = jax.process_count()` hosts, inflating
    `(D, ...)` to `(N * D, ...)` and turning a 0-D scalar `()` into a 1-D
    `(N,)` array. Restricting `process_allgather(..., tiled=True)` to
    non-fully-addressable `jax.Array` values preserves exact shapes (including
    0-D scalars) via `jax.device_get(arr)`.

  Args:
    arr: A `jax.Array` (sharded or replicated) or host array-like object.

  Returns:
    Host `np.ndarray` containing the full tensor value.
  """
  if _needs_multihost_gather(arr):
    return np.asarray(multihost_utils.process_allgather(arr, tiled=True))
  return np.asarray(jax.device_get(arr))


def gather_tensors_to_host0(
    tensors: Mapping[str, Any],
    *,
    is_host0: bool,
) -> dict[str, np.ndarray]:
  """Gathers `tensors` in sorted key order, materializing only on host 0.

  Non-host-0 processes still enter `process_allgather(..., tiled=True)` for
  non-fully-addressable `jax.Array` values (required by SPMD collectives to
  avoid deadlocking host 0), while skipping `jax.device_get` on host-local
  arrays.

  Args:
    tensors: Mapping of source tensor names to `jax.Array` or host array-like
      values.
    is_host0: Whether the current process is process 0.

  Returns:
    Mapping of source tensor names to host `np.ndarray` values on host 0, or an
    empty dict on non-host-0 processes.
  """
  host_tensors: dict[str, np.ndarray] = {}
  for key in sorted(tensors):
    arr = tensors[key]
    if is_host0:
      host_tensors[key] = gather_tensor_to_host(arr)
    elif _needs_multihost_gather(arr):
      multihost_utils.process_allgather(arr, tiled=True)
  return host_tensors
