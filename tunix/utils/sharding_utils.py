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

from collections.abc import Sequence
from typing import Tuple

import jax
from jax import numpy as jnp
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
