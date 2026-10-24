# Copyright 2025 Google LLC
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

"""Utilities for sharding tensors."""

import os
from typing import Tuple

import jax
from jax.interpreters import pxla
import jax.sharding as shd
import numpy as np


def _env_flag(name: str) -> bool:
  """True iff the environment variable `name` is set to 1/true/True."""
  return (os.environ.get(name, "0") or "0").strip() in ("1", "true", "True")


def per_host_safe_logps_enabled() -> bool:
  """Whether per-host-safe logprobs (TUNIX_PER_HOST_SAFE_LOGPS) are on.

  TUNIX_PER_HOST_SAFE_LOGPS=1 (default OFF) only acts together with the
  per-host learner (TUNIX_PER_HOST_LEARNER=1) on more than one process. There
  every process passes its own, different rows to `shard_input`, which builds
  the global array with `jax.make_array_from_process_local_data` (local rows x
  process count). With the flag on, `shard_input` judges the leading axis by
  that global row count before falling back to replication, and
  `RLCluster.get_ref_per_token_logps` / `get_actor_per_token_logps` chunk the
  process-local batch before globalising it and return host-local rows.

  Returns:
    True iff TUNIX_PER_HOST_SAFE_LOGPS=1, TUNIX_PER_HOST_LEARNER=1 and the job
    has more than one process.
  """
  return (
      _env_flag("TUNIX_PER_HOST_SAFE_LOGPS")
      and _env_flag("TUNIX_PER_HOST_LEARNER")
      and jax.process_count() > 1
  )


def shard_input(
    input_data: jax.Array, data_sharding_axis: Tuple[str, ...]
) -> jax.Array:
  """Shards the input data across the available devices.

  Args:
    input_data: The input data to be sharded, expected to be a TrainingInput
      dataclass.
    data_sharding_axis: The sharding axis for the input data, e.g. ("fsdp",).

  Returns:
    The sharded TrainingInput.
  """
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty:
    return input_data

  pspec = shd.PartitionSpec(*data_sharding_axis)
  # Check if the input is already sharded with the target mesh to avoid
  # re-sharding.
  is_sharded = jax.tree.map(
      lambda x: isinstance(x, jax.Array)
      and hasattr(x, "sharding")
      and hasattr(x.sharding, "mesh")
      and x.sharding.mesh == mesh
      and hasattr(x.sharding, "spec")
      and x.sharding.spec == pspec,
      input_data,
  )
  if all(jax.tree.leaves(is_sharded)):
    return input_data

  # Skip sharding if the input contains globally distributed arrays that are
  # not fully addressable by the local process. Attempting to recreate them via
  # `jax.make_array_from_process_local_data` would fail with a ValueError.
  is_global = jax.tree.map(
      lambda x: isinstance(x, jax.Array) and not x.is_fully_addressable,
      input_data,
  )
  if any(jax.tree.leaves(is_global)):
    return input_data

  # TUNIX_PER_HOST_SAFE_LOGPS: process-local rows of the per-host learner are
  # concatenated across processes, so axis 0 is judged by the global size.
  leading_dim_factor = (
      jax.process_count() if per_host_safe_logps_enabled() else 1
  )
  with jax.transfer_guard("allow"):
    return jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(
            get_sharding(
                x,
                mesh=mesh,
                pspec=pspec,
                leading_dim_factor=leading_dim_factor,
            ),
            x,
        ),
        input_data,
    )


def get_sharding(
    x: jax.Array,
    mesh: shd.Mesh,
    pspec: shd.PartitionSpec,
    leading_dim_factor: int = 1,
):
  """Get a sharding for an tensor given a mesh and partition spec.

  Args:
    x: The array to shard.
    mesh: The mesh.
    pspec: The partition spec to use if every sharded axis divides evenly.
    leading_dim_factor: The global size of axis 0 is `x.shape[0] *
      leading_dim_factor`: the process count when `x` holds process-local rows
      that `jax.make_array_from_process_local_data` concatenates across
      processes (TUNIX_PER_HOST_SAFE_LOGPS), 1 otherwise.

  Returns:
    `NamedSharding(mesh, pspec)`, or a replicated sharding.
  """
  # Only shard arrays with rank > 0.
  if not isinstance(x, (np.ndarray, jax.Array)) or x.ndim == 0:
    return shd.NamedSharding(mesh, shd.PartitionSpec())  # Replicated

  # Don't shard if rank is not sufficient.
  if x.ndim < len(pspec):
    return shd.NamedSharding(mesh, shd.PartitionSpec())  # Replicated

  # Check for divisibility for all sharded axes.
  for i, axis_name in enumerate(pspec):
    if axis_name is not None:
      axis_names = axis_name if isinstance(axis_name, tuple) else (axis_name,)
      for name in axis_names:
        if name is not None and name not in mesh.shape:
          return shd.NamedSharding(mesh, shd.PartitionSpec())
        axis_size = mesh.shape[name]
        dim = x.shape[i] * (leading_dim_factor if i == 0 else 1)
        if dim % axis_size != 0:
          # Replicate if not evenly divisible.
          return shd.NamedSharding(mesh, shd.PartitionSpec())
  return shd.NamedSharding(mesh, pspec)


def process_row_ranges(x: jax.Array) -> list[tuple[int, int]]:
  """This process's rows of a global array as merged [start, stop) ranges."""
  num_rows = x.shape[0]
  spans = sorted({
      (
          shard.index[0].start or 0,
          num_rows if shard.index[0].stop is None else shard.index[0].stop,
      )
      for shard in x.addressable_shards
  })
  merged = []
  for start, stop in spans:
    if merged and start <= merged[-1][1]:
      merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
    else:
      merged.append((start, stop))
  return merged


def process_local_rows(x: jax.Array) -> np.ndarray:
  """Returns this process's rows of a row-sharded global array (host copy).

  The inverse of `jax.make_array_from_process_local_data`, which is what
  `shard_input` applies to process-local data: this process's rows are its
  addressable row blocks in ascending global order. Blocks split along later
  axes are put back together; every element of this process's rows must be
  addressable. Fully addressable inputs are returned whole.
  (`multihost_utils.global_array_to_host_local_array` needs the process's
  devices to form a contiguous sub-mesh, which a trainer FSDP mesh need not.)

  Args:
    x: The array.

  Returns:
    This process's rows.

  Raises:
    ValueError: Part of this process's rows is not addressable here.
  """
  if not isinstance(x, jax.Array) or x.is_fully_addressable:
    return np.asarray(x)
  if x.ndim == 0:
    return np.asarray(x.addressable_data(0))
  ranges = process_row_ranges(x)
  offsets, total = {}, 0
  for start, stop in ranges:
    offsets[start] = total
    total += stop - start
  out = np.empty((total,) + tuple(x.shape[1:]), dtype=x.dtype)
  covered = np.zeros(out.shape, dtype=bool)
  for shard in x.addressable_shards:
    index = [
        (s.start or 0, x.shape[d] if s.stop is None else s.stop)
        for d, s in enumerate(shard.index)
    ]
    row_start = max(r for r in offsets if r <= index[0][0])
    first = offsets[row_start] + index[0][0] - row_start
    dest = (slice(first, first + index[0][1] - index[0][0]),) + tuple(
        slice(a, b) for a, b in index[1:]
    )
    out[dest] = np.asarray(shard.data)
    covered[dest] = True
  if not covered.all():
    raise ValueError(
        f"process_local_rows: rows {ranges} of an array of shape {x.shape} with"
        f" sharding {x.sharding} are only partly addressable on process"
        f" {jax.process_index()}."
    )
  return out
