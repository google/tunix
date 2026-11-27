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

"""Orbax checkpointing on a subset of the processes of a shared JAX world.

When trainer and rollout processes share one `jax.distributed` runtime (e.g.
for TPU ICI weight sync), only the trainer processes checkpoint. Orbax barriers
and primary-host decisions default to every process of the world, which
deadlocks on rollout processes that never enter Orbax. Scope them to the
trainer processes by passing `multiprocessing_options(...)` to Orbax v0 objects
and `configure_v1_context(...)` to the Orbax v1 `Context`, after calling
`enable_distributed_process_ids()` on every process of the world.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
import tempfile
import threading
from typing import Any

from absl import flags
from absl import logging
from etils import epath
import jax
import numpy as np
import orbax.checkpoint as ocp_v0
from orbax.checkpoint import checkpoint_manager as ocp_cm
from orbax.checkpoint import multihost as ocp_multihost
from orbax.checkpoint import v1 as ocp_v1


def mesh_process_indices(mesh: jax.sharding.Mesh | None) -> frozenset[int]:
  """Returns the JAX process indices owning at least one device in `mesh`.

  Args:
    mesh: Target JAX mesh, or None when no mesh is available.

  Returns:
    Frozenset of integer process indices (empty if `mesh` is None).
  """
  if mesh is None:
    return frozenset()
  return frozenset(int(d.process_index) for d in mesh.devices.flat)


def state_process_indices(state: Any) -> frozenset[int]:
  """Returns the JAX process indices owning the `jax.Array` leaves of `state`.

  Args:
    state: PyTree such as `nnx.state(model)`; non-array leaves are ignored.

  Returns:
    Frozenset of integer process indices owning the arrays in `state`.
  """
  return frozenset(
      int(d.process_index)
      for leaf in jax.tree_util.tree_leaves(state)
      if isinstance(leaf, jax.Array)
      for d in leaf.sharding.device_set
  )


def is_subworld(active_processes: Collection[int] | None) -> bool:
  """Returns whether `active_processes` is a strict subset of this JAX world.

  Args:
    active_processes: Participating process indices, or None for the full world.

  Returns:
    True iff `jax.process_count() > 1` and `active_processes` is a non-empty
    strict subset of the JAX world.
  """
  if not active_processes:
    return False
  world_size = jax.process_count()
  if world_size <= 1:
    return False
  return len(frozenset(int(p) for p in active_processes)) < world_size


_ORBAX_DISTRIBUTED_ID_FLAG = "experimental_orbax_use_distributed_process_id"


def _orbax_uses_distributed_ids() -> bool:
  """Returns whether Orbax v0 uses distributed task ids as process indices."""
  return flags.FLAGS.is_parsed() and bool(
      flags.FLAGS[_ORBAX_DISTRIBUTED_ID_FLAG].value
  )


def enable_distributed_process_ids() -> None:
  """Makes Orbax v0 and v1 agree on distributed task ids. Collective.

  On GKE TPU slices `jax.process_index()` follows the torus order while the
  coordination service, Orbax v1 and subset barriers use the distributed task
  id. This turns on Orbax's distributed-process-id mode, so Orbax v0 uses the
  task id too, and builds Orbax's runtime -> distributed id map, which v0 reads
  for device ownership. Every process of the JAX world must call it once right
  after `jax.distributed.initialize`, including those that never checkpoint.

  Raises:
    RuntimeError: If the Orbax flag does not take effect.
  """
  if flags.FLAGS.is_parsed():
    flags.FLAGS[_ORBAX_DISTRIBUTED_ID_FLAG].value = True
  else:
    # Orbax ignores flags until absl parses them; parse only this one.
    flags.FLAGS(
        ["tunix", f"--{_ORBAX_DISTRIBUTED_ID_FLAG}=true"], known_only=True
    )
  if not _orbax_uses_distributed_ids():
    raise RuntimeError(f"Could not enable --{_ORBAX_DISTRIBUTED_ID_FLAG}.")
  if not ocp_multihost.is_runtime_to_distributed_ids_initialized():
    ocp_multihost.initialize_runtime_to_distributed_ids()
  logging.info(
      "Orbax uses distributed process id %d (jax.process_index()=%d).",
      ocp_multihost.process_index(),
      jax.process_index(),
  )


_DISTRIBUTED_IDS_LOCK = threading.Lock()
_DISTRIBUTED_IDS: dict[frozenset[int], dict[int, int]] = {}


def _gather_distributed_ids(procs: frozenset[int]) -> dict[int, int]:
  """Maps each of `procs` to its distributed task id. Collective over `procs`.

  Gathers every process's `orbax.checkpoint.v1.multihost.process_index()` (the
  distributed task id) over a mesh of only the devices of `procs`, so the
  other processes of the world are not involved.
  """
  devices = sorted(
      (d for d in jax.devices() if d.process_index in procs),
      key=lambda d: (d.process_index, d.id),
  )
  mesh = jax.sharding.Mesh(np.asarray(devices), ("p",))
  own_id = ocp_v1.multihost.process_index()
  local = jax.make_array_from_callback(
      (len(devices),),
      jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("p")),
      lambda idx: np.full(np.empty(len(devices))[idx].shape, own_id, np.int32),
  )
  replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  ids = np.asarray(
      jax.jit(lambda x: x, out_shardings=replicated)(local).addressable_data(0)
  )
  mapping: dict[int, int] = {}
  for device, dist_id in zip(devices, ids, strict=True):
    if mapping.setdefault(device.process_index, int(dist_id)) != dist_id:
      raise RuntimeError(f"Inconsistent distributed ids gathered: {ids}.")
  if len(set(mapping.values())) != len(mapping):
    raise RuntimeError(f"Non-unique distributed ids: {mapping}.")
  return mapping


def _orbax_processes(active_processes: Collection[int]) -> set[int]:
  """Converts `jax.process_index()` values to the ids Orbax barriers expect.

  Orbax compares `active_processes` and `primary_host` against its own
  process index and passes them to the coordination service, which identifies
  processes by distributed task id. Callers pass `jax.process_index()` values
  (e.g. from mesh devices), so they are translated to task ids. Collective over
  `active_processes`; the result is cached per set.

  Args:
    active_processes: Participating `jax.process_index()` values, including
      this process.

  Returns:
    The distributed task ids of `active_processes`.

  Raises:
    ValueError: If `active_processes` is empty or misses this process.
    RuntimeError: If the ids differ while Orbax v0 still uses
      `jax.process_index()` (see `enable_distributed_process_ids`).
  """
  procs = frozenset(int(p) for p in active_processes)
  if not procs:
    raise ValueError("active_processes must be non-empty.")
  if jax.process_index() not in procs:
    raise ValueError(
        f"Current JAX process {jax.process_index()} is not in"
        f" active_processes={sorted(procs)}."
    )
  with _DISTRIBUTED_IDS_LOCK:
    mapping = _DISTRIBUTED_IDS.get(procs)
    if mapping is None:
      mapping = _gather_distributed_ids(procs)
      _DISTRIBUTED_IDS[procs] = mapping
  if not _orbax_uses_distributed_ids() and any(
      runtime_id != dist_id for runtime_id, dist_id in mapping.items()
  ):
    raise RuntimeError(
        f"JAX runtime -> distributed process ids {mapping} differ, but Orbax v0"
        " uses jax.process_index(); call enable_distributed_process_ids() on"
        " every process after jax.distributed.initialize()."
    )
  return set(mapping.values())


def multiprocessing_options(
    active_processes: Collection[int],
    *,
    barrier_sync_key_prefix: str | None = None,
) -> ocp_v1.options.MultiprocessingOptions:
  """Returns Orbax v1 `MultiprocessingOptions` scoped to `active_processes`.

  Collective over `active_processes`. Use `.v0()` for the Orbax v0 API.

  Args:
    active_processes: Participating `jax.process_index()` values, including
      this process.
    barrier_sync_key_prefix: Optional prefix for Orbax barrier keys.

  Returns:
    Options whose `active_processes` are the distributed task ids of
    `active_processes`, with the smallest as `primary_host`.
  """
  procs = _orbax_processes(active_processes)
  return ocp_v1.options.MultiprocessingOptions(
      primary_host=min(procs),
      active_processes=procs,
      barrier_sync_key_prefix=barrier_sync_key_prefix,
  )


def configure_v1_context(
    context: ocp_v1.Context,
    active_processes: Collection[int] | None,
) -> ocp_v1.Context:
  """Returns a v1 `Context` scoped to `active_processes` when in a sub-world.

  Args:
    context: Base Orbax v1 `Context`.
    active_processes: Participating process indices, or None for the full world.

  Returns:
    `context` unchanged when not in a sub-world, otherwise a child `Context`
    with `primary_host` and `active_processes` scoped to `active_processes`
    (as distributed task ids, see `multiprocessing_options`). Collective over
    `active_processes` in a sub-world.
  """
  if active_processes is None or not is_subworld(active_processes):
    return context
  procs = _orbax_processes(active_processes)
  # The child copies every option of `context`, the barrier prefix included.
  scoped = ocp_v1.Context(context)
  scoped.multiprocessing.primary_host = min(procs)
  scoped.multiprocessing.active_processes = procs
  return scoped


_ROOT_DIRECTORY_LOCK = threading.Lock()
_ROOT_DIRECTORY_CHECKED = False
_ORIG_CREATE_ROOT_DIRECTORY: Callable[..., None] | None = None


def _create_root_directory(
    directory: epath.PathLike,
    mp_options: ocp_v0.options.MultiprocessingOptions,
    file_options: ocp_v0.options.FileOptions | None = None,
) -> None:
  """Orbax's `_create_root_directory`, also for `active_processes` subsets."""
  if mp_options.active_processes is None:
    assert _ORIG_CREATE_ROOT_DIRECTORY is not None
    _ORIG_CREATE_ROOT_DIRECTORY(directory, mp_options, file_options)
    return
  path = epath.Path(directory)
  if ocp_multihost.is_primary_host(mp_options.primary_host):
    path.mkdir(
        parents=True,
        exist_ok=True,
        mode=file_options.path_permission_mode if file_options else None,
    )
    logging.info("Created checkpoint root directory=%s", path)
  key = "CheckpointManager:create_directory"
  if mp_options.barrier_sync_key_prefix:
    key = f"{mp_options.barrier_sync_key_prefix}_{key}"
  ocp_multihost.sync_global_processes(
      key, processes=mp_options.active_processes
  )


def _orbax_creates_subworld_root_directory() -> bool:
  """Returns whether Orbax creates root directories with `active_processes`.

  Probes with a manager whose only active process is this one, so no other
  process takes part.
  """
  own_id = ocp_multihost.process_index()
  options = ocp_v0.CheckpointManagerOptions(
      multiprocessing_options=ocp_v0.options.MultiprocessingOptions(
          primary_host=own_id, active_processes={own_id}
      )
  )
  with tempfile.TemporaryDirectory() as tmp_dir:
    try:
      ocp_v0.CheckpointManager(
          epath.Path(tmp_dir) / "root", options=options
      ).close()
    except NotImplementedError:
      return False
  return True


def allow_subworld_root_directory_creation() -> None:
  """Lets Orbax create checkpoint root directories with `active_processes`.

  Orbax up to 0.12.12 raises `NotImplementedError` from
  `CheckpointManager(create=True)` when `active_processes` is set, and
  `orbax.checkpoint.v1.training.Checkpointer` always passes `create=True`. If
  the installed Orbax still does, this installs, once per process, a
  replacement that creates the directory on the primary host and syncs only the
  active processes; it delegates to Orbax when `active_processes` is None.
  Otherwise it changes nothing.
  """
  # TODO(tunix-dev): Remove once the minimum Orbax includes
  # https://github.com/google/orbax/pull/3828 (issue
  # https://github.com/google/orbax/issues/3827).
  global _ROOT_DIRECTORY_CHECKED, _ORIG_CREATE_ROOT_DIRECTORY
  with _ROOT_DIRECTORY_LOCK:
    if _ROOT_DIRECTORY_CHECKED:
      return
    _ROOT_DIRECTORY_CHECKED = True
    if _orbax_creates_subworld_root_directory():
      return
    # pylint: disable=protected-access
    _ORIG_CREATE_ROOT_DIRECTORY = ocp_cm._create_root_directory
    ocp_cm._create_root_directory = _create_root_directory
    # pylint: enable=protected-access
    if not _orbax_creates_subworld_root_directory():
      raise RuntimeError("Could not patch Orbax root directory creation.")
