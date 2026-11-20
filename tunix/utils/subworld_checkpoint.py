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

"""Sub-world Orbax checkpointing helpers for shared multi-process JAX worlds.

When trainer and rollout processes share one `jax.distributed` runtime (e.g.
for TPU ICI weight sync), only the trainer processes participate in checkpoint
save and restore. By default, Orbax sets `active_processes=None`, which causes
every barrier (`multihost.sync_global_processes` / `sync_global_devices`) to
wait on ALL processes in the JAX world, deadlocking on rollout processes that
never enter checkpointing.

This module scopes Orbax v0 (`Checkpointer`, `PyTreeCheckpointHandler`) and
Orbax v1 (`Context`, `training.Checkpointer`) barriers and primary-host checks
strictly to the trainer's `active_processes`.
"""

from __future__ import annotations

from collections.abc import Collection
import contextlib
import dataclasses
import threading
from typing import Any, Iterator

from absl import logging
from etils import epath
import jax
from jax.experimental import multihost_utils
import orbax.checkpoint as ocp_v0
from orbax.checkpoint import checkpoint_manager as ocp_cm
from orbax.checkpoint import utils as ocp_utils
from orbax.checkpoint import v1 as ocp_v1
from orbax.checkpoint._src.multihost import multihost as ocp_multihost
from orbax.checkpoint._src.serialization import type_handler_registry as ocp_thr
from orbax.checkpoint._src.serialization import type_handlers as ocp_th
from orbax.checkpoint.experimental.v1._src.training import checkpointer as ocp_v1_ckpt

_PATCH_LOCK = threading.Lock()
_PATCH_DEPTH = 0
_ACTIVE_PROCESSES: frozenset[int] | None = None
_ORIG_FUNCS: dict[str, Any] = {}


def mesh_process_indices(mesh: jax.sharding.Mesh | None) -> frozenset[int]:
  """Returns the JAX process indices that own at least one device in `mesh`."""
  if mesh is None:
    return frozenset()
  devices = getattr(mesh, "devices", None)
  flat = getattr(devices, "flat", None)
  if flat is None:
    return frozenset()
  return frozenset(
      int(p)
      for d in flat
      if isinstance((p := getattr(d, "process_index", None)), int)
  )


def state_process_indices(state: Any) -> frozenset[int]:
  """Returns the JAX process indices owning arrays in `state`."""
  procs: set[int] = set()
  for leaf in jax.tree_util.tree_leaves(state):
    arr = getattr(leaf, "value", leaf)
    sharding = getattr(arr, "sharding", None)
    device_set = getattr(sharding, "device_set", None)
    if device_set:
      procs.update(int(d.process_index) for d in device_set)
    elif hasattr(arr, "devices") and callable(arr.devices):
      try:
        procs.update(int(d.process_index) for d in arr.devices())
      except Exception:  # pylint: disable=broad-exception-caught
        pass
  return frozenset(procs)


def is_subworld(active_processes: Collection[int] | None) -> bool:
  """Returns True when `active_processes` is a strict subset of this JAX world."""
  if not active_processes:
    return False
  world_size = jax.process_count()
  if world_size <= 1:
    return False
  procs = frozenset(int(p) for p in active_processes)
  return 0 < len(procs) < world_size


def v0_multiprocessing_options(
    active_processes: Collection[int],
) -> ocp_v0.options.MultiprocessingOptions:
  """Builds Orbax v0 `MultiprocessingOptions` scoped to `active_processes`."""
  procs = set(int(p) for p in active_processes)
  if not procs:
    raise ValueError("active_processes must be non-empty.")
  return ocp_v0.options.MultiprocessingOptions(
      primary_host=min(procs),
      active_processes=procs,
  )


def v1_multiprocessing_options(
    active_processes: Collection[int],
) -> ocp_v1.options.MultiprocessingOptions:
  """Builds Orbax v1 `MultiprocessingOptions` scoped to `active_processes`."""
  procs = set(int(p) for p in active_processes)
  if not procs:
    raise ValueError("active_processes must be non-empty.")
  return ocp_v1.options.MultiprocessingOptions(
      primary_host=min(procs),
      active_processes=procs,
  )


def configure_v1_context(
    context: ocp_v1.Context,
    active_processes: Collection[int] | None,
) -> ocp_v1.Context:
  """Returns a v1 `Context` scoped to `active_processes` when in a sub-world."""
  if not is_subworld(active_processes):
    return context
  assert active_processes is not None
  procs = set(int(p) for p in active_processes)
  primary_host = min(procs)
  # Clone via `ocp_v1.Context(context, ...)` so frozen contexts work too.
  return ocp_v1.Context(
      context,
      multiprocessing_options=ocp_v1.options.MultiprocessingOptions(
          primary_host=primary_host,
          active_processes=procs,
          barrier_sync_key_prefix=(
              context.multiprocessing_options.barrier_sync_key_prefix
          ),
      ),
  )


def _create_root_directory_subworld(
    directory: epath.PathLike,
    multiprocessing_options: ocp_v0.options.MultiprocessingOptions,
    file_options: ocp_v0.options.FileOptions | None = None,
) -> None:
  """Creates `directory` on `primary_host` and syncs only `active_processes`.

  Orbax's stock `_create_root_directory` raises `NotImplementedError` when
  `create=True` and `active_processes` is set, even though `ocp.v1.training.Checkpointer`
  hardcodes `create=True`. This helper performs the exact same mkdir + barrier
  scoped to `active_processes`.
  """
  if multiprocessing_options.active_processes is None:
    _ORIG_FUNCS["create_root_directory"](
        directory, multiprocessing_options, file_options
    )
    return
  path = epath.Path(directory)
  if not path.exists() and ocp_utils.is_primary_host(
      multiprocessing_options.primary_host
  ):
    path.mkdir(
        parents=True,
        exist_ok=True,
        mode=file_options.path_permission_mode if file_options else None,
    )
    logging.info("Created sub-world checkpoint directory=%s", path)
  ocp_multihost.sync_global_processes(
      ocp_multihost.unique_barrier_key(
          "CheckpointManager:create_directory",
          prefix=multiprocessing_options.barrier_sync_key_prefix,
      ),
      timeout=ocp_multihost.coordination_timeout(),
      processes=multiprocessing_options.active_processes,
  )


def _set_global_array_handler_primary_host(primary_host: int) -> None:
  reg = getattr(ocp_thr, "GLOBAL_TYPE_HANDLER_REGISTRY", None)
  entries = getattr(reg, "_type_registry", ())
  for _, handler in entries:
    if isinstance(handler, ocp_th.ArrayHandler):
      handler._primary_host = primary_host  # pylint: disable=protected-access


def _install_patches() -> None:
  """Installs process-scoped wrappers on Orbax v0/v1 and JAX global barriers."""
  _ORIG_FUNCS["create_root_directory"] = ocp_cm._create_root_directory  # pylint: disable=protected-access
  _ORIG_FUNCS["v0_checkpointer_init"] = ocp_v0.Checkpointer.__init__
  _ORIG_FUNCS["v0_checkpoint_manager_init"] = ocp_v0.CheckpointManager.__init__
  _ORIG_FUNCS["v0_pytree_handler_init"] = (
      ocp_v0.PyTreeCheckpointHandler.__init__
  )
  _ORIG_FUNCS["v0_standard_handler_init"] = (
      ocp_v0.StandardCheckpointHandler.__init__
  )
  _ORIG_FUNCS["v0_array_handler_init"] = ocp_th.ArrayHandler.__init__
  _ORIG_FUNCS["v1_checkpointer_init"] = ocp_v1_ckpt.Checkpointer.__init__
  _ORIG_FUNCS["sync_global_devices"] = multihost_utils.sync_global_devices
  reg = getattr(ocp_thr, "GLOBAL_TYPE_HANDLER_REGISTRY", None)
  _ORIG_FUNCS["global_array_primary_hosts"] = [
      (handler, getattr(handler, "_primary_host", 0))
      for _, handler in getattr(reg, "_type_registry", ())
      if isinstance(handler, ocp_th.ArrayHandler)
  ]

  def _wrapped_v0_checkpointer_init(self, *args, **kwargs):
    procs = _ACTIVE_PROCESSES
    if procs is not None:
      mp = kwargs.get("multiprocessing_options")
      if mp is None or getattr(mp, "active_processes", None) is None:
        kwargs["multiprocessing_options"] = v0_multiprocessing_options(procs)
    return _ORIG_FUNCS["v0_checkpointer_init"](self, *args, **kwargs)

  def _wrapped_v0_checkpoint_manager_init(self, *args, **kwargs):
    procs = _ACTIVE_PROCESSES
    if procs is not None:
      opts = kwargs.get("options")
      if opts is None and len(args) >= 3:
        opts = args[2]
        args = list(args)
        if opts is None:
          opts = ocp_v0.CheckpointManagerOptions(
              multiprocessing_options=v0_multiprocessing_options(procs)
          )
        elif getattr(
            getattr(opts, "multiprocessing_options", None),
            "active_processes",
            None,
        ) is None:
          opts = dataclasses.replace(
              opts, multiprocessing_options=v0_multiprocessing_options(procs)
          )
        args[2] = opts
      else:
        if opts is None:
          kwargs["options"] = ocp_v0.CheckpointManagerOptions(
              multiprocessing_options=v0_multiprocessing_options(procs)
          )
        elif getattr(
            getattr(opts, "multiprocessing_options", None),
            "active_processes",
            None,
        ) is None:
          kwargs["options"] = dataclasses.replace(
              opts, multiprocessing_options=v0_multiprocessing_options(procs)
          )
    return _ORIG_FUNCS["v0_checkpoint_manager_init"](self, *args, **kwargs)

  def _wrapped_v0_pytree_handler_init(self, *args, **kwargs):
    procs = _ACTIVE_PROCESSES
    if procs is not None:
      mp = kwargs.get("multiprocessing_options")
      if mp is None or getattr(mp, "active_processes", None) is None:
        kwargs["multiprocessing_options"] = v0_multiprocessing_options(procs)
    return _ORIG_FUNCS["v0_pytree_handler_init"](self, *args, **kwargs)

  def _wrapped_v0_standard_handler_init(self, *args, **kwargs):
    procs = _ACTIVE_PROCESSES
    if procs is not None:
      mp = kwargs.get("multiprocessing_options")
      if mp is None or getattr(mp, "active_processes", None) is None:
        kwargs["multiprocessing_options"] = v0_multiprocessing_options(procs)
    return _ORIG_FUNCS["v0_standard_handler_init"](self, *args, **kwargs)

  def _wrapped_v0_array_handler_init(self, *args, **kwargs):
    procs = _ACTIVE_PROCESSES
    if procs is not None and "primary_host" not in kwargs and len(args) < 2:
      kwargs["primary_host"] = min(procs)
    return _ORIG_FUNCS["v0_array_handler_init"](self, *args, **kwargs)

  def _wrapped_v1_checkpointer_init(self, *args, **kwargs):
    procs = _ACTIVE_PROCESSES
    if procs is not None:
      ctx = kwargs.get("context")
      if ctx is None:
        ctx = ocp_v1.Context()
      if ctx.multiprocessing_options.active_processes is None:
        kwargs["context"] = configure_v1_context(ctx, procs)
    return _ORIG_FUNCS["v1_checkpointer_init"](self, *args, **kwargs)

  def _wrapped_sync_global_devices(name: str):
    procs = _ACTIVE_PROCESSES
    if procs is not None:
      ocp_multihost.sync_global_processes(name, processes=set(procs))
      return
    _ORIG_FUNCS["sync_global_devices"](name)

  ocp_cm._create_root_directory = _create_root_directory_subworld  # pylint: disable=protected-access
  ocp_v0.Checkpointer.__init__ = _wrapped_v0_checkpointer_init
  ocp_v0.CheckpointManager.__init__ = _wrapped_v0_checkpoint_manager_init
  ocp_v0.PyTreeCheckpointHandler.__init__ = _wrapped_v0_pytree_handler_init
  ocp_v0.StandardCheckpointHandler.__init__ = _wrapped_v0_standard_handler_init
  ocp_th.ArrayHandler.__init__ = _wrapped_v0_array_handler_init
  ocp_v1_ckpt.Checkpointer.__init__ = _wrapped_v1_checkpointer_init
  multihost_utils.sync_global_devices = _wrapped_sync_global_devices


def _restore_patches() -> None:
  ocp_cm._create_root_directory = _ORIG_FUNCS["create_root_directory"]  # pylint: disable=protected-access
  ocp_v0.Checkpointer.__init__ = _ORIG_FUNCS["v0_checkpointer_init"]
  ocp_v0.CheckpointManager.__init__ = _ORIG_FUNCS["v0_checkpoint_manager_init"]
  ocp_v0.PyTreeCheckpointHandler.__init__ = _ORIG_FUNCS[
      "v0_pytree_handler_init"
  ]
  ocp_v0.StandardCheckpointHandler.__init__ = _ORIG_FUNCS[
      "v0_standard_handler_init"
  ]
  ocp_th.ArrayHandler.__init__ = _ORIG_FUNCS["v0_array_handler_init"]
  ocp_v1_ckpt.Checkpointer.__init__ = _ORIG_FUNCS["v1_checkpointer_init"]
  multihost_utils.sync_global_devices = _ORIG_FUNCS["sync_global_devices"]
  for handler, orig_primary in _ORIG_FUNCS.get(
      "global_array_primary_hosts", ()
  ):
    handler._primary_host = orig_primary  # pylint: disable=protected-access
  _ORIG_FUNCS.clear()


@contextlib.contextmanager
def patch_orbax_for_subworld(
    active_processes: Collection[int] | None,
) -> Iterator[None]:
  """Scopes Orbax v0/v1 checkpointers and barriers to `active_processes`.

  No-op when `active_processes` covers the entire JAX world (or single-process).
  Safe to nest on the same process set.

  Args:
    active_processes: JAX process indices participating in checkpointing (e.g.
      the trainer hosts in a shared ICI slice).
  """
  if not is_subworld(active_processes):
    yield
    return

  assert active_processes is not None
  procs = frozenset(int(p) for p in active_processes)
  if jax.process_index() not in procs:
    raise ValueError(
        f"Current JAX process {jax.process_index()} is not in"
        f" active_processes={sorted(procs)}."
    )

  global _PATCH_DEPTH, _ACTIVE_PROCESSES
  with _PATCH_LOCK:
    prev_procs = _ACTIVE_PROCESSES
    if _PATCH_DEPTH == 0:
      _install_patches()
    _PATCH_DEPTH += 1
    _ACTIVE_PROCESSES = procs
    _set_global_array_handler_primary_host(min(procs))
  try:
    yield
  finally:
    with _PATCH_LOCK:
      _PATCH_DEPTH -= 1
      _ACTIVE_PROCESSES = prev_procs
      if _PATCH_DEPTH == 0:
        _restore_patches()
      elif prev_procs is not None:
        _set_global_array_handler_primary_host(min(prev_procs))
