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

"""GCS-based weight synchronization transport for Tunix and vLLM."""

from __future__ import annotations

import collections
import importlib
import logging
import os
import re
import time
from typing import Any, List, Optional, Sequence

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from tunix.experimental.weight_sync import weight_sync

logger = logging.getLogger(__name__)

_PATH_COMPONENT_RE = re.compile(r"\[([^\]]+)\]|\.?([^\.\[\]]+)")
_FOLDED_INDEX_RE = re.compile(r"^([a-zA-Z_]+)_(\d+)$")


def _param_key(name: str) -> str:
  """Canonical dotted key for pairing a parameter name with a runner tree path.

  Reduces both `layers_0.attention.w` and `['layers'][0]['attention']['w']`
  to `layers.0.attention.w`. Drops wrapper roots ('base', 'model') and the
  trailing '.value' attribute.
  """
  segments: List[str] = []
  for bracketed, dotted in _PATH_COMPONENT_RE.findall(name):
    component = (bracketed or dotted).strip("'\"")
    if not component:
      continue
    folded = _FOLDED_INDEX_RE.match(component)
    segments.extend(folded.groups() if folded else (component,))
  while segments and segments[0] in ("base", "model"):
    segments.pop(0)
  if segments and segments[-1] == "value":
    segments.pop()
  return ".".join(segments)


def apply_checkpoint_to_runner(runner: Any, checkpoint_path: str) -> int:
  """Restores weights from an Orbax checkpoint and updates runner state in place.

  Args:
    runner: The TPUModelRunner instance holding `state` and `state_leaves`.
    checkpoint_path: Path to the Orbax checkpoint directory (local or gs://).

  Returns:
    The number of parameters successfully matched and updated.
  """
  if not hasattr(runner, "state") or runner.state is None:
    raise ValueError("runner does not have a valid 'state' attribute.")

  cpu_devices = jax.devices("cpu")
  if cpu_devices:
    fallback_sharding = jax.sharding.SingleDeviceSharding(cpu_devices[0])
  else:
    fallback_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

  logger.info(
      "Restoring GCS weights from checkpoint: %s (fallback_sharding=%s)",
      checkpoint_path,
      fallback_sharding,
  )
  checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
  restore_args = ocp.args.StandardRestore(fallback_sharding=fallback_sharding)
  restored = None
  for attempt in range(60):
    try:
      restored = checkpointer.restore(checkpoint_path, args=restore_args)
      break
    except Exception as e:
      if "incomplete checkpoint" in str(e).lower() or "not found" in str(e).lower() or "does not exist" in str(e).lower():
        logger.info(
            "Waiting for checkpoint to be fully written (attempt %d/60): %s",
            attempt + 1,
            e,
        )
        time.sleep(2)
      else:
        raise
  if restored is None:
    restored = checkpointer.restore(checkpoint_path, args=restore_args)

  names: List[str] = []
  arrays: List[Any] = []
  for path, leaf in jax.tree_util.tree_leaves_with_path(restored):
    arr = getattr(leaf, "value", leaf)
    if hasattr(arr, "shape") and hasattr(arr, "dtype"):
      names.append(jax.tree_util.keystr(path))
      arrays.append(arr)

  new_leaves = list(runner.state_leaves)
  runner_leaves_with_path = list(
      jax.tree_util.tree_leaves_with_path(runner.state)
  )
  if len(new_leaves) != len(runner_leaves_with_path):
    raise RuntimeError(
        f"runner.state_leaves length ({len(new_leaves)}) does not match "
        f"runner.state leaves count ({len(runner_leaves_with_path)})."
    )

  key_to_entries: dict[str, List[Any]] = collections.defaultdict(list)
  for idx, (name, arr) in enumerate(zip(names, arrays)):
    key_to_entries[_param_key(name)].append((idx, name, arr))

  matched_indices = set()

  def _claim(key: str):
    for entry in key_to_entries.get(key, ()):
      if entry[0] not in matched_indices:
        return entry
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
            f"Shape mismatch for parameter '{orig_name}' (runner path '{p_str}'): "
            f"runner shape {leaf_arr.shape} vs checkpoint shape {arr.shape}"
        )
      if hasattr(leaf_arr, "sharding") and leaf_arr.sharding is not None:
        if not hasattr(arr, "sharding") or arr.sharding != leaf_arr.sharding:
          arr = jax.device_put(arr, leaf_arr.sharding)
      if hasattr(leaf_arr, "delete") and callable(leaf_arr.delete):
        try:
          leaf_arr.delete()
        except Exception:
          pass
      new_leaves[i] = arr
      matched_indices.add(idx)

  runner.state_leaves = tuple(new_leaves)
  runner.state = jax.tree_util.tree_unflatten(
      jax.tree_util.tree_structure(runner.state), new_leaves
  )
  if hasattr(runner, "model") and runner.model is not None:
    try:
      from flax import nnx
      nnx.update(runner.model, runner.state)
    except Exception:
      pass
  logger.info(
      "Successfully applied %d/%d checkpoint arrays to runner state "
      "(total runner leaves: %d).",
      len(matched_indices),
      len(arrays),
      len(new_leaves),
  )
  del restored, names, arrays, key_to_entries, new_leaves, runner_leaves_with_path
  import gc
  gc.collect()
  return len(matched_indices)


class GcsSyncHandler(weight_sync.WeightSyncHandler):
  """Orchestrator-facing handler for GCS-based weight sync.

  Unlike Raiden, the GCS transport does not move bytes over TCP/P2P
  during `transfer()`. Instead, the trainer writes an unscanned checkpoint to
  shared storage (GCS) during `prepare_weight_sync()`,
  and rollout workers restore it during `weight_sync()`. Manifest preflight is
  not required.
  """

  needs_manifest_preflight = False

  def __init__(self):
    self._registered: set[weight_sync.WorkUnitId] = set()

  @property
  def registered_units(self) -> frozenset[weight_sync.WorkUnitId]:
    return frozenset(self._registered)

  def register_work_unit(self, metadata: weight_sync.WorkUnitMetadata) -> None:
    self._registered.add(metadata.unit)

  def transfer(
      self,
      src_units: Sequence[weight_sync.WorkUnitId],
      dst_units: Sequence[weight_sync.WorkUnitId],
      req_id: Optional[str] = None,
      generation: Optional[int] = None,
      **kwargs: Any,
  ) -> weight_sync.TransferResult:
    del src_units, dst_units, generation, kwargs
    return weight_sync.TransferResult(req_id=req_id or "", success=True)

  def close(self) -> None:
    self._registered.clear()


# Alias for backward compatibility
FilesystemSyncHandler = GcsSyncHandler


def patch_tpu_worker_gcs_sync() -> None:
  """Monkey-patches TPUWorker with `load_gcs_weights` method."""
  worker_modules = [
      "tpu_inference.worker.tpu_worker",
      "vllm_torchtpu.worker.tpu_worker",
  ]
  patched = False
  for mod_name in worker_modules:
    try:
      mod = importlib.import_module(mod_name)
      worker_cls = getattr(mod, "TPUWorker", None)
      if worker_cls is not None:
        if getattr(worker_cls, "_patched_gcs_sync", False):
          patched = True
          continue

        def _load_gcs_weights(self, checkpoint_path: str) -> None:
          logger.info(
              "TPUWorker executing load_gcs_weights(%s)...",
              checkpoint_path,
          )
          runner = getattr(self, "model_runner", None)
          if runner is None:
            raise RuntimeError("TPUWorker has no model_runner attribute.")
          apply_checkpoint_to_runner(runner, checkpoint_path)

        worker_cls.load_gcs_weights = _load_gcs_weights
        worker_cls.load_filesystem_weights = _load_gcs_weights
        worker_cls._patched_gcs_sync = True
        logger.info(
            "Successfully patched %s.TPUWorker with load_gcs_weights.",
            mod_name,
        )
        patched = True
    except (ImportError, AttributeError):
      pass
  if not patched:
    logger.debug(
        "No TPUWorker module found to patch for GCS sync (expected outside rollout worker)."
    )


# Alias for backward compatibility
patch_tpu_worker_filesystem_sync = patch_tpu_worker_gcs_sync
