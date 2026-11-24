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

"""File-based Safetensors/Orbax worker weight synchronizer for distributed RL.

Stages model parameters from a Trainer slice to GCS (`gs://`), CNS (`/GOOGLE_INTERNAL_STOAGE_PATH/`), or
local storage and restores them asynchronously onto an asymmetric Sampler or
Rollout slice (for example, Trainer `8H` FSDP/TP -> Sampler `2H` TP/EP).

Key capabilities in `"safetensors"` mode (the default):
  * **Decoupled HBM waves vs. shard files**: `max_bucket_bytes` (default
    `128 MiB`) caps temporary per-rank HBM/D2H buffers, while `max_shard_bytes`
    (default `2.5 GiB`, hard ceiling `4 GiB`) groups up to 32 waves into one
    standard `.safetensors` shard via streaming GCS Parallel Composite Upload.
  * **Manifest SSOT & pure Identity mode**: Pass declarative `to_hf_manifest` /
    `from_hf_manifest` (e.g. `qwen3.params.build_to_hf_manifest(cfg)`) to emit
    standard HuggingFace weights for vLLM/SGLang or cross-framework sync, or
    omit both for pure Identity NNX-to-NNX sync with zero transpose heuristics.
  * **Index-based host shard filtering**: Destinations parse
    `model.safetensors.index.json` up front and skip every shard that contains
    zero tensors owned by the local host.
  * **Default-fast multi-process `/dev/shm` transport**: Automatically manages
    a persistent :class:`safetensors_saver.MultiprocStagePool` on writers and
    multi-process `ShmRangeReader` workers on readers—no manual `TUNIX_GCS_*`
    environment flags required.
  * **AOT XLA compilation & upfront validation**: Pre-compiles `SavePlan`
    kernels during :meth:`GCSSynchronizer.bind` (`aot_compile=True`) and runs
    :func:`validate_weight_sync_runtime` automatically.

Recognized environment variables (all optional):
  * `TUNIX_GCS_WRITE_PROCS`: Override auto writer process count on `gs://`.
  * `TUNIX_CNS_PROCS`: Override reader process count on 1P `/GOOGLE_INTERNAL_STOAGE_PATH/`.
  * `VERIFY_WEIGHTS`: Set `"true"` to enable per-round L1 checksum verification.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence, Set as AbstractSet
from concurrent import futures
import contextlib
import dataclasses
import functools
import hashlib
import itertools
import json
import math
import os
import tempfile
import time
from typing import Any, Protocol, TypeVar, runtime_checkable
import weakref

from absl import logging
from etils import epath
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from orbax import checkpoint as ocp
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.weight_sync import weight_sync
from tunix.dist_rl.weight_sync import weight_utils
from tunix.models import hf_mapping
from tunix.models import safetensors_io
from tunix.models import safetensors_loader
from tunix.models import safetensors_saver
from tunix.utils import sharding_utils

_T = TypeVar("_T")
_VALID_CHECKPOINT_FORMATS = ("safetensors", "orbax")
_ORBAX_MARKERS = (
    "_CHECKPOINT_METADATA",
    "_METADATA",
    "manifest.ocdbt",
    "default",
)
_HOST0_ERROR_TYPES: Mapping[str, type[Exception]] = {
    "FileNotFoundError": FileNotFoundError,
    "KeyError": KeyError,
    "ValueError": ValueError,
    "OSError": OSError,
    "RuntimeError": RuntimeError,
}
# Enable multi-process `/GOOGLE_INTERNAL_STOAGE_PATH/` readers starting at 256 MiB (`gs://` auto-scales
# via `safetensors_io.auto_gcs_read_procs`).
_LARGE_READ_THRESHOLD_BYTES = 256 << 20
# Bounded persistent `/dev/shm` circular ring ceiling (192 GiB): pre-faulted
# once on Sync 1 and recycled per committed file/layer group across syncs so
# large models (e.g. 397B@2H) run in 1 continuous barrier-free pass while
# avoiding cold 4 KiB tmpfs page-fault stalls (`shmem_enabled=[never]`).
_MAX_PERSISTENT_READ_ARENA_BYTES = safetensors_io.MAX_READ_ARENA_BYTES
_STAGE_POOL_COUNTER = itertools.count(1)
ValidationReport = safetensors_saver.ValidationReport


def _manifest_digest(manifest: hf_mapping.RuleSpec) -> str:
  """Returns deterministic SHA-256 hex digest of a normalized manifest."""
  encoded = json.dumps(
      hf_mapping.parse_manifest(manifest), sort_keys=True, default=str
  ).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def run_blocking(fn: Callable[[], _T]) -> _T:
  """Runs blocking storage I/O off any active thread-local asyncio event loop.

  Orbax checkpointers spawn internal asyncio event loops that raise
  `RuntimeError` if invoked inside an already-running loop. When an event loop
  is active on the current thread, this helper dispatches `fn` on a 1-worker
  thread pool; otherwise it invokes `fn()` inline.

  Args:
    fn: Zero-argument callable performing blocking checkpoint I/O.

  Returns:
    The return value of `fn()`.

  Example:
    ```python
    from tunix.experimental.weight_sync import gcs_synchronizer

    result = gcs_synchronizer.run_blocking(lambda: checkpointer.restore(path))
    ```
  """
  try:
    asyncio.get_running_loop()
    has_running_loop = True
  except RuntimeError:
    has_running_loop = False
  if not has_running_loop:
    return fn()
  with futures.ThreadPoolExecutor(max_workers=1) as executor:
    return executor.submit(fn).result()


def _resolve_step_dir_name(
    sync_request: datatypes.WeightSyncRequest | None,
) -> str:
  """Resolves deterministic `step_<policy_version>_<req_id>` directory name."""
  if sync_request is None:
    raise ValueError("sync_request is required for GCSSynchronizer.d2h().")
  version = sync_request.policy_version
  if version is None or isinstance(version, bool):
    raise ValueError(
        "sync_request.policy_version must be an integer for"
        f" GCSSynchronizer.d2h(), got {version!r}."
    )
  extra = sync_request.extra_config
  if "req_id" not in extra:
    raise ValueError(
        "sync_request.extra_config['req_id'] is required for"
        " GCSSynchronizer.d2h()."
    )
  req_id = str(extra["req_id"]).strip()
  if not req_id or "/" in req_id or ".." in req_id:
    raise ValueError(
        "Invalid sync_request.extra_config['req_id'] for checkpoint directory:"
        f" {req_id!r}."
    )
  return f"step_{version}_{req_id}"


def extract_source_artifact(
    sync_request: datatypes.WeightSyncRequest | None,
) -> tuple[str, dict[str, float | int] | None]:
  """Extracts the single canonical source `artifact_uri` and `checksums`.

  Validates that every source work unit in `sync_request.source_metadata`
  agrees on one non-empty `artifact_uri` and identical optional `checksums`.

  Args:
    sync_request: Destination `WeightSyncRequest` carrying `source_metadata`.

  Returns:
    2-tuple `(artifact_uri, checksums)` where `checksums` may be `None`.

  Raises:
    ValueError: If `sync_request` or `source_metadata` is missing, any source
      omits `artifact_uri`, or source units report conflicting URIs/checksums.

  Example:
    ```python
    from tunix.experimental.weight_sync import gcs_synchronizer

    uri, sums = gcs_synchronizer.extract_source_artifact(sync_request)
    ```
  """
  if sync_request is None or not sync_request.source_metadata:
    raise ValueError(
        "sync_request with non-empty source_metadata is required to resolve"
        " source checkpoint artifact_uri."
    )
  raw = sync_request.source_metadata
  entries = (
      [raw]
      if isinstance(raw, (weight_sync.WorkUnitMetadata, dict))
      else list(raw)
  )
  artifact_uris: set[str] = set()
  checksums: dict[str, float | int] | None = None
  for entry in entries:
    meta = weight_sync.WorkUnitMetadata.from_dict(entry)
    if meta.artifact_uri is None or not meta.artifact_uri.strip():
      raise ValueError(
          f"Source unit {meta.unit} did not publish a non-empty artifact_uri."
      )
    artifact_uris.add(meta.artifact_uri)
    if meta.checksums is None:
      continue
    unit_sums = dict(meta.checksums)
    if checksums is not None and unit_sums != checksums:
      raise ValueError(
          "Source work units published conflicting checksums dictionaries."
      )
    checksums = unit_sums

  if len(artifact_uris) != 1:
    raise ValueError(
        "Source work units published conflicting artifact_uri values:"
        f" {sorted(artifact_uris)!r}"
    )
  return next(iter(artifact_uris)), checksums


def _build_restore_specs(
    names: Sequence[str],
    arrays: Sequence[Any],
) -> tuple[dict[str, jax.ShapeDtypeStruct], dict[str, ocp.RestoreArgs]]:
  """Builds target `ShapeDtypeStruct` and `RestoreArgs` trees by `param_key`."""
  item: dict[str, jax.ShapeDtypeStruct] = {}
  restore_args: dict[str, ocp.RestoreArgs] = {}
  for name, arr in zip(names, arrays, strict=True):
    key = weight_utils.param_key(name)
    shape = tuple(arr.shape)
    if isinstance(arr, jax.Array):
      item[key] = jax.ShapeDtypeStruct(
          shape=shape, dtype=arr.dtype, sharding=arr.sharding
      )
      restore_args[key] = ocp.ArrayRestoreArgs(
          restore_type=jax.Array,
          dtype=arr.dtype,
          sharding=arr.sharding,
          global_shape=shape,
          strict=True,
      )
    else:
      item[key] = jax.ShapeDtypeStruct(shape=shape, dtype=arr.dtype)
      restore_args[key] = ocp.RestoreArgs(
          restore_type=np.ndarray,
          dtype=arr.dtype,
      )
  return item, restore_args


def _broadcast_host0_status(
    fn: Callable[[], tuple[Any, Any]],
) -> tuple[Any, Any]:
  """Runs `fn()` on host 0, broadcasting status/errors on multi-host JAX."""
  if jax.process_count() <= 1:
    return fn()
  local_val: Any = None
  status: dict[str, Any] = {"ok": True}
  if jax.process_index() == 0:
    try:
      local_val, payload = fn()
      status["payload"] = payload
    except Exception as exc:  # pylint: disable=broad-exception-caught
      msg = (
          str(exc.args[0])
          if isinstance(exc, KeyError) and exc.args
          else str(exc)
      )
      err_type = type(exc).__name__
      if err_type not in _HOST0_ERROR_TYPES and isinstance(exc, OSError):
        err_type = "OSError"
      elif err_type not in _HOST0_ERROR_TYPES and isinstance(exc, ValueError):
        err_type = "ValueError"
      status = {"ok": False, "error_type": err_type, "error": msg}
  synced = sharding_utils.broadcast_host0_json(status)
  if not synced["ok"]:
    err_cls = _HOST0_ERROR_TYPES.get(synced.get("error_type", ""), RuntimeError)
    raise err_cls(synced["error"])
  return local_val, synced.get("payload")


def _sync_multihost_error(local_exc: Exception | None, context: str) -> None:
  """Coordinates per-host exceptions across McJAX ranks to avoid deadlocks."""
  if jax.process_count() <= 1:
    if local_exc is not None:
      raise local_exc
    return
  flags = multihost_utils.process_allgather(
      np.asarray(1 if local_exc is not None else 0, dtype=np.int32),
      tiled=False,
  )
  if local_exc is not None:
    raise local_exc
  failed = [
      int(idx)
      for idx, flag in enumerate(np.asarray(flags).reshape(-1))
      if int(flag) != 0
  ]
  if failed:
    raise RuntimeError(f"{context} failed on peer process(es) {failed}.")


def _coordinate_multihost_read_setup(
    all_files: Sequence[Any],
    files: list[Any],
    setup_exc: Exception | None,
    resolved_uri: str,
) -> list[Any]:
  """Syncs setup errors and guards asymmetric empty `files` on multi-host."""
  if jax.process_count() <= 1:
    if setup_exc is not None:
      raise setup_exc
    return files
  gathered = multihost_utils.process_allgather(
      np.asarray(
          [1 if setup_exc is not None else 0, 1 if files else 0], dtype=np.int32
      ),
      tiled=False,
  )
  if setup_exc is not None:
    raise setup_exc
  rows = np.asarray(gathered).reshape(-1, 2)
  failed = [int(idx) for idx, row in enumerate(rows) if int(row[0]) != 0]
  if failed:
    raise RuntimeError(
        f"Safetensors restore setup for {resolved_uri} failed on peer"
        f" process(es) {failed}."
    )
  if not files and any(int(row[1]) != 0 for row in rows):
    return list(all_files)
  return files


def _validate_positive_int(
    value: int | None, field_name: str, *, allow_none: bool = False
) -> None:
  """Raises `ValueError` if `value` is a boolean or not a positive integer."""
  if value is None and allow_none:
    return
  if value is None or isinstance(value, bool) or value <= 0:
    raise ValueError(
        f"GCSSynchronizerConfig.{field_name} must be a positive int, got"
        f" {value!r}."
    )


def _mesh_devices_for_arrays(arrays: Sequence[Any]) -> list[jax.Device] | None:
  """Returns flattened `Mesh` devices from the first `NamedSharding` array."""
  for arr in arrays:
    if (
        isinstance(arr, jax.Array)
        and isinstance(arr.sharding, jax.sharding.NamedSharding)
        and isinstance(arr.sharding.mesh, jax.sharding.Mesh)
    ):
      return list(arr.sharding.mesh.devices.flat)
  return None


def _resolve_write_procs(target_uri: str | None, configured: int | None) -> int:
  """Resolves `MultiprocStagePool` worker process count for `target_uri`."""
  if not target_uri or not (
      target_uri.startswith("gs://") or target_uri.startswith("/GOOGLE_INTERNAL_STOAGE_PATH/")
  ):
    return 1
  if configured is not None:
    return configured
  if target_uri.startswith("gs://"):
    raw = os.environ.get("TUNIX_GCS_WRITE_PROCS", "").strip()
    if raw:
      val = int(raw)
      _validate_positive_int(val, "TUNIX_GCS_WRITE_PROCS")
      return val
    cpus = os.cpu_count()
    return min(48, max(16, 32 if cpus is None else cpus))
  return 16


def _resolve_read_procs(
    resolved_uri: str, configured: int | None, host_bytes: int
) -> int:
  """Resolves multi-process `/dev/shm` reader count for `resolved_uri`."""
  if resolved_uri.startswith("gs://"):
    return safetensors_io.auto_gcs_read_procs(host_bytes, configured)
  if resolved_uri.startswith("/GOOGLE_INTERNAL_STOAGE_PATH/"):
    if configured is not None:
      return configured
    raw = os.environ.get("TUNIX_CNS_PROCS", "").strip()
    if raw:
      val = int(raw)
      if val < 0:
        raise ValueError(f"TUNIX_CNS_PROCS must be non-negative, got {val!r}.")
      return val
    return 16 if host_bytes >= _LARGE_READ_THRESHOLD_BYTES else 0
  return 0


def _build_safetensors_export_inputs(
    names: Sequence[str],
    arrays: Sequence[Any],
    max_bucket_bytes: int,
    to_hf_manifest: hf_mapping.RuleSpec | None = None,
) -> tuple[
    dict[str, jax.Array],
    list[safetensors_saver.TensorExportSpec],
    dict[str, int],
    list[jax.Device] | None,
]:
  """Builds flat state, `TensorExportSpec`s, unstack counts, and mesh devices."""
  flat_state = {
      weight_utils.param_key(name): jnp.asarray(arr)
      for name, arr in zip(names, arrays, strict=True)
  }
  devices = _mesh_devices_for_arrays(arrays)
  export_state = flat_state
  if to_hf_manifest is not None:
    parsed = hf_mapping.parse_manifest(to_hf_manifest)
    dropped = set(parsed["dropped_keys"]) | {
        dst for _, dst in parsed["tied_keys"]
    }
    if parsed["rules"]:
      effective = {**parsed, "tied_keys": [], "dropped_keys": list(dropped)}
      specs, unstack_counts = (
          safetensors_saver.build_export_specs_from_manifest(
              flat_state, effective
          )
      )
      return flat_state, specs, unstack_counts, devices
    export_state = {k: v for k, v in flat_state.items() if k not in dropped}
  specs: list[safetensors_saver.TensorExportSpec] = []
  unstack_counts: dict[str, int] = {}
  for key, j_arr in export_state.items():
    nbytes = int(j_arr.size) * j_arr.dtype.itemsize
    is_rep = safetensors_saver.infer_is_replicated(j_arr)
    if j_arr.ndim == 3 and j_arr.shape[0] > 1 and nbytes > max_bucket_bytes:
      unstack_counts[key] = int(j_arr.shape[0])
      specs.append(
          safetensors_saver.TensorExportSpec(
              source_key=key,
              hf_key=f"{key}.experts.{{expert}}",
              transpose_axes=None,
              unstack_axis=0,
              nbytes=nbytes,
              is_replicated=is_rep,
          )
      )
    else:
      shape_2d = (
          (int(j_arr.shape[0]), int(math.prod(j_arr.shape[1:])))
          if j_arr.ndim >= 3
          else None
      )
      specs.append(
          safetensors_saver.TensorExportSpec(
              source_key=key,
              hf_key=key,
              shape_2d=shape_2d,
              transpose_axes=None,
              nbytes=nbytes,
              is_replicated=is_rep,
          )
      )
  return flat_state, specs, unstack_counts, devices


# pylint: disable=protected-access
def _build_safetensors_source_map(
    keys: Sequence[str],
    arrays: Sequence[Any],
) -> dict[str, safetensors_loader._SourcePlan]:
  """Builds pure-Identity `safetensors_loader` source plans (zero transpose)."""
  source_map: dict[str, safetensors_loader._SourcePlan] = {}
  for key, arr in zip(keys, arrays, strict=True):
    shape = tuple(int(d) for d in arr.shape)
    tf = (None, shape) if len(shape) >= 3 else None
    source_map[key] = safetensors_loader._SourcePlan(
        target_key=key, transform=tf
    )
    if len(shape) == 3 and shape[0] > 1:
      num_exp = shape[0]
      for idx in range(num_exp):
        source_map[f"{key}.experts.{idx}"] = safetensors_loader._SourcePlan(
            target_key=key,
            transform=None,
            expert_info=(idx, num_exp),
        )
  return source_map


def _build_safetensors_restore_plans(
    keys: Sequence[str],
    arrays: Sequence[Any],
    from_hf_manifest: hf_mapping.RuleSpec | None = None,
) -> tuple[
    dict[str, safetensors_loader._SourcePlan],
    AbstractSet[str],
    Any,
]:
  """Builds `(source_map, composite_sources, composite_executor)` for load."""
  if from_hf_manifest is None:
    return _build_safetensors_source_map(keys, arrays), frozenset(), None
  parsed = hf_mapping.parse_manifest(from_hf_manifest)
  if parsed["rules"]:
    return safetensors_loader.SpmdHostStager.compile_manifest(parsed)
  dropped = frozenset(parsed["dropped_keys"])
  active_tied = [(s, d) for s, d in parsed["tied_keys"] if d not in dropped]
  excluded = dropped | {d for _, d in active_tied}
  id_pairs = [
      (k, a) for k, a in zip(keys, arrays, strict=True) if k not in excluded
  ]
  source_map = _build_safetensors_source_map(
      [k for k, _ in id_pairs], [a for _, a in id_pairs]
  )
  if not active_tied:
    return source_map, frozenset(), None
  tied_srcs = frozenset(s for s, _ in active_tied)
  for src_key in tied_srcs:
    source_map[src_key] = safetensors_loader._SourcePlan(
        src_key, is_raw_source=True
    )
  return source_map, tied_srcs, hf_mapping.make_streaming_executor(parsed)


def _load_safetensors_weight_map(resolved_uri: str) -> dict[str, str]:
  """Loads and validates `weight_map` from `index.json`, or `{}` if absent."""
  # Observed on GCS: probing `.exists()` before `.read_text()` doubles metadata
  # RTT on every restore. Read directly and treat missing index files as `{}`
  # across POSIX (`FileNotFoundError`), 1P gfile (`NOT_FOUND`), and 3P GCS.
  index_path = epath.Path(resolved_uri) / "model.safetensors.index.json"
  try:
    raw_text = index_path.read_text()
  except Exception as exc:  # pylint: disable=broad-exception-caught
    if safetensors_io.is_not_found_error(exc):
      return {}
    raise
  raw_index = json.loads(raw_text)
  if (
      not isinstance(raw_index, Mapping)
      or "weight_map" not in raw_index
      or not isinstance(raw_index["weight_map"], Mapping)
  ):
    raise ValueError(
        f"Safetensors index {index_path} must contain a 'weight_map' mapping."
    )
  raw_wm = raw_index["weight_map"]
  if not raw_wm:
    raise FileNotFoundError(f"No safetensors shards listed in {index_path}.")
  for name in dict.fromkeys(raw_wm.values()):
    if (
        not isinstance(name, str)
        or os.path.basename(name) != name
        or "\\" in name
        or name in ("", ".", "..")
    ):
      raise ValueError(
          f"Invalid shard filename {name!r} in {index_path}; must be a"
          " single relative filename within the checkpoint directory."
      )
  return {str(k): str(v) for k, v in raw_wm.items()}


def _resolve_safetensors_layout(
    ckpt_path: epath.Path,
) -> tuple[list[str] | None, dict[str, str]]:
  """Returns `(validated_shard_names, weight_map)`, or `(None, {})` for Orbax."""
  if any(tag in ckpt_path.name for tag in (".tmp", ".orbax-checkpoint-tmp")):
    raise FileNotFoundError(
        f"Checkpoint directory is not finalized: {ckpt_path}."
    )
  weight_map = _load_safetensors_weight_map(str(ckpt_path))
  if weight_map:
    return sorted(dict.fromkeys(weight_map.values())), weight_map
  if (ckpt_path / "model.safetensors").exists():
    return ["model.safetensors"], {}
  if any((ckpt_path / marker).exists() for marker in _ORBAX_MARKERS):
    return None, {}
  raise FileNotFoundError(
      f"No finalized safetensors or Orbax checkpoint found in {ckpt_path}."
  )


def resolve_safetensors_files(ckpt_path: epath.Path) -> list[epath.Path] | None:
  """Returns validated safetensors shards, `None` for Orbax, or raises."""
  shard_names, _ = _resolve_safetensors_layout(ckpt_path)
  if shard_names is None:
    return None
  return [ckpt_path / name for name in shard_names]


def _shard_needed_by_host(
    hf_key: str,
    source_map: Mapping[str, safetensors_loader._SourcePlan],
    composite_sources: AbstractSet[str],
    target_specs: Mapping[str, safetensors_loader._TargetSpec],
) -> bool:
  """Returns True when `hf_key` contributes >=1 tensor slice to this host."""
  if hf_key in composite_sources:
    return True
  if hf_key not in source_map:
    return False
  plan = source_map[hf_key]
  if plan.target_key not in target_specs:
    return False
  if plan.expert_info is None:
    return True
  expert_idx, num_experts = plan.expert_info
  spec = target_specs[plan.target_key]
  if spec.global_shape[0] != num_experts:
    return True
  exp_slice = spec.host_slices[0]
  return exp_slice.start <= expert_idx < exp_slice.stop


def _filter_host_shards(
    files: Sequence[Any],
    weight_map: Mapping[str, str],
    source_map: Mapping[str, safetensors_loader._SourcePlan],
    composite_sources: AbstractSet[str],
    target_specs: Mapping[str, safetensors_loader._TargetSpec],
) -> list[Any]:
  """Skips shards whose `weight_map` keys own zero slices on this host."""
  if not weight_map:
    return list(files)
  needed_shards = {
      shard_name
      for hf_key, shard_name in weight_map.items()
      if _shard_needed_by_host(
          hf_key, source_map, composite_sources, target_specs
      )
  }
  return [f for f in files if f.name in needed_shards]


def _compute_read_arena_budget(host_bytes: int) -> int:
  """Returns bounded `/dev/shm` circular-ring arena capacity for `host_bytes`."""
  if host_bytes <= 0:
    return 0
  # `available_shm_bytes()` already reserves 20% headroom on `/dev/shm`.
  max_safe = max(4096, safetensors_io.available_shm_bytes())
  desired = min(_MAX_PERSISTENT_READ_ARENA_BYTES, max(64 << 20, host_bytes * 4))
  return min(max_safe, desired)


def _open_read_arena(
    host_bytes: int,
    arena_provider: (
        Callable[[int], safetensors_loader.ShmRangeReader | None] | None
    ),
) -> contextlib.AbstractContextManager[safetensors_loader.ShmRangeReader]:
  """Returns a context manager yielding a persistent or ephemeral read arena."""
  shared_arena = arena_provider(host_bytes) if arena_provider else None
  if shared_arena is not None:
    return contextlib.nullcontext(shared_arena)
  return safetensors_loader.ShmRangeReader(
      shm_budget_bytes=_compute_read_arena_budget(host_bytes)
  )


def _free_partial_params(params: list[Any]) -> None:
  """Deletes partially-loaded device arrays and clears `params` for tracebacks."""
  loaded = [p for p in params if isinstance(p, jax.Array)]
  params.clear()
  if loaded:
    weight_utils.delete_array_buffers(loaded)


def _finalize_restored_params(
    keys: Sequence[str],
    params: list[Any],
    arrays: Sequence[Any],
    resolved_uri: str,
) -> dict[str, Any]:
  """Validates completeness and casts mismatched leaves to bound dtypes."""
  missing = [
      key
      for key, loaded in zip(keys, params, strict=True)
      if isinstance(loaded, jax.ShapeDtypeStruct)
  ]
  if missing:
    raise KeyError(
        f"Missing {len(missing)} weight tensor(s) in safetensors checkpoint"
        f" {resolved_uri}: {missing[:5]!r}"
    )
  restored: dict[str, Any] = {}
  for idx, (key, loaded, target_arr) in enumerate(
      zip(keys, params, arrays, strict=True)
  ):
    expected_dtype = jnp.dtype(target_arr.dtype)
    if isinstance(loaded, jax.Array) and loaded.dtype != expected_dtype:
      casted = loaded.astype(expected_dtype)
      weight_utils.delete_array_buffers([loaded])
      params[idx] = casted
      loaded = casted
    restored[key] = loaded
  return restored


def _resolve_restore_shard_list(
    resolved_uri: str,
    shard_names: Sequence[str] | None,
    weight_map: Mapping[str, str] | None,
) -> tuple[list[str], Mapping[str, str]]:
  """Resolves validated committed `(shard_names, weight_map)` without glob."""
  if shard_names is not None:
    return list(shard_names), ({} if weight_map is None else weight_map)
  if weight_map:
    return sorted(dict.fromkeys(weight_map.values())), weight_map
  probed_shards, probed_wm = _resolve_safetensors_layout(
      epath.Path(resolved_uri)
  )
  if probed_shards is None:
    raise FileNotFoundError(
        f"Expected safetensors checkpoint, found Orbax at {resolved_uri}."
    )
  return probed_shards, probed_wm


def _is_rapid_zonal_consensus(target_uri: str) -> bool:
  """Returns cluster-agreed Rapid Zonal status via process-0 broadcast."""
  if not target_uri.startswith("gs://"):
    return False
  if jax.process_count() <= 1:
    return safetensors_io.is_rapid_zonal_bucket(target_uri)
  is_rapid = (
      jax.process_index() == 0
      and safetensors_io.is_rapid_zonal_bucket(target_uri)
  )
  with jax.transfer_guard("allow"):
    agreed = multihost_utils.broadcast_one_to_all(
        np.int32(int(is_rapid)), is_source=jax.process_index() == 0
    )
  return bool(int(agreed))


def _resolve_save_shard_policy(
    job_name: str,
    target_uri: str,
    config: GCSSynchronizerConfig,
    backend: safetensors_saver.StorageBackend,
    *,
    is_rapid: bool | None = None,
) -> tuple[int | None, bool]:
  """Resolves `(max_shard_bytes, per_wave_shards)` for `target_uri`."""
  rapid = (
      _is_rapid_zonal_consensus(target_uri) if is_rapid is None else is_rapid
  )
  if rapid:
    if config.max_shard_bytes is not None or (
        config.per_wave_shards is not None and not config.per_wave_shards
    ):
      logging.info(
          "%s detected Rapid Zonal bucket for %s; using per-wave DirectPath"
          " uploads (overriding max_shard_bytes=%s, per_wave_shards=%s).",
          job_name,
          target_uri,
          config.max_shard_bytes,
          config.per_wave_shards,
      )
    return None, True
  use_wave = (
      backend.default_per_wave_shards
      if config.per_wave_shards is None
      else config.per_wave_shards
  )
  return config.max_shard_bytes, use_wave


def _resolve_restore_sources_and_headers(
    resolved_uri: str,
    shard_names: Sequence[str] | None,
    weight_map: Mapping[str, str] | None,
    shard_headers: Mapping[str, tuple[int, Mapping[str, Any]]] | None,
) -> tuple[
    list[Any],
    Mapping[str, str],
    Mapping[str, tuple[int, Mapping[str, Any]]] | None,
]:
  """Resolves validated shard sources, `weight_map`, and optional headers."""
  sidecar_fut = None
  if shard_headers is None and shard_names is None and weight_map is None:
    sidecar_fut = safetensors_loader._SHARED_POOL.submit(
        safetensors_io.load_sidecar_headers, resolved_uri
    )
  try:
    resolved_shards, resolved_wm = _resolve_restore_shard_list(
        resolved_uri, shard_names, weight_map
    )
  except BaseException:
    if sidecar_fut is not None:
      sidecar_fut.cancel()
    raise
  resolved_headers = (
      sidecar_fut.result() if sidecar_fut is not None else shard_headers
  )
  all_files: list[Any] = []
  if resolved_shards:
    all_files = safetensors_io.build_range_sources_from_names(
        resolved_uri, resolved_shards
    )
  if not resolved_headers or set(resolved_headers) != set(resolved_shards):
    return all_files, resolved_wm, None
  local_sizes = None
  if not resolved_wm and "://" not in resolved_uri:
    local_dir = os.path.expanduser(resolved_uri)
    local_sizes = {
        name: os.path.getsize(p)
        for name in resolved_shards
        if os.path.isfile(p := os.path.join(local_dir, name))
    }
  usable_headers = (
      resolved_headers
      if safetensors_io.sidecar_matches_weight_map(
          resolved_headers, resolved_wm, file_sizes=local_sizes
      )
      else None
  )
  return all_files, resolved_wm, usable_headers


def _restore_safetensors(
    resolved_uri: str,
    names: Sequence[str],
    arrays: Sequence[Any],
    *,
    from_hf_manifest: hf_mapping.RuleSpec | None = None,
    read_procs: int | None = None,
    shard_names: Sequence[str] | None = None,
    weight_map: Mapping[str, str] | None = None,
    shard_headers: Mapping[str, tuple[int, Mapping[str, Any]]] | None = None,
    arena_provider: (
        Callable[[int], safetensors_loader.ShmRangeReader | None] | None
    ) = None,
) -> tuple[dict[str, Any], int, int]:
  """Restores safetensors checkpoint in 1 continuous circular-ring pass."""
  keys = [weight_utils.param_key(name) for name in names]
  local_dev0 = jax.local_devices()[0]
  params: list[jax.ShapeDtypeStruct | jax.Array] = [
      jax.ShapeDtypeStruct(shape=tuple(arr.shape), dtype=arr.dtype)
      for arr in arrays
  ]
  shardings: list[jax.sharding.Sharding | jax.Device] = [
      arr.sharding if isinstance(arr, jax.Array) else local_dev0
      for arr in arrays
  ]
  target_specs = safetensors_loader._build_target_specs(keys, params, shardings)
  source_map, composite_sources, composite_executor = (
      _build_safetensors_restore_plans(
          keys, arrays, from_hf_manifest=from_hf_manifest
      )
  )
  host_bytes = sum(
      math.prod(target_specs[k].host_shape) * p.dtype.itemsize
      for k, p in zip(keys, params, strict=True)
      if k in target_specs
  )
  all_files: list[Any] = []
  files: list[Any] = []
  resolved_headers: Mapping[str, tuple[int, Mapping[str, Any]]] | None = None
  target_procs = 0
  setup_exc: Exception | None = None
  try:
    all_files, resolved_wm, resolved_headers = (
        _resolve_restore_sources_and_headers(
            resolved_uri, shard_names, weight_map, shard_headers
        )
    )
    files = _filter_host_shards(
        all_files, resolved_wm, source_map, composite_sources, target_specs
    )
    target_procs = _resolve_read_procs(resolved_uri, read_procs, host_bytes)
    if resolved_uri.startswith("/GOOGLE_INTERNAL_STOAGE_PATH/") and target_procs > 0:
      safetensors_io.CnsPath.spawn_procs(target_procs)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    setup_exc = exc
  files = _coordinate_multihost_read_setup(
      all_files, files, setup_exc, resolved_uri
  )
  bound_dtypes = {np.dtype(arr.dtype) for arr in arrays}
  stream_dtype = (
      next(iter(bound_dtypes))
      if len(bound_dtypes) == 1 and composite_executor is None
      else None
  )
  load_exc: Exception | None = None
  restored: dict[str, Any] = {}
  try:
    if files:
      with _open_read_arena(host_bytes, arena_provider) as arena:
        logging.info(
            "Safetensors restore %s: reading %d/%d shards on host %d in 1 pass"
            " (read_procs=%d, host_mib=%d, arena_mib=%d).",
            resolved_uri,
            len(files),
            len(all_files),
            jax.process_index(),
            target_procs,
            host_bytes >> 20,
            arena.capacity >> 20,
        )
        safetensors_loader.stream_load_files(
            files,
            source_map,
            composite_sources,
            composite_executor,
            stream_dtype,
            target_specs,
            keys,
            params,
            arena=arena,
            read_procs=target_procs,
            shard_headers=resolved_headers,
        )
    restored = _finalize_restored_params(keys, params, arrays, resolved_uri)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    load_exc = exc
  if load_exc is not None:
    _free_partial_params(params)
    restored.clear()
  try:
    _sync_multihost_error(load_exc, f"Safetensors restore {resolved_uri}")
  except BaseException:
    _free_partial_params(params)
    restored.clear()
    raise
  return restored, len(all_files), len(files)


# pylint: enable=protected-access


def _validate_to_hf_manifest(
    key_set: AbstractSet[str],
    keys: Sequence[str],
    arrays: Sequence[Any],
    to_hf_manifest: hf_mapping.RuleSpec,
    errors: list[str],
) -> None:
  """Checks that `to_hf_manifest` is valid and covers all bound keys."""
  try:
    parsed = hf_mapping.parse_manifest(to_hf_manifest)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    errors.append(f"Invalid to_hf_manifest: {exc}")
    return
  if not key_set or not parsed["rules"]:
    return
  dropped = set(parsed["dropped_keys"])
  tied_dst = {dst for _, dst in parsed["tied_keys"]}
  effective = {
      **parsed,
      "tied_keys": [],
      "dropped_keys": [*parsed["dropped_keys"], *tied_dst],
  }
  flat_sds = {
      k: jax.ShapeDtypeStruct(tuple(a.shape), a.dtype)
      for k, a in zip(keys, arrays, strict=True)
  }
  try:
    specs, _ = safetensors_saver.build_export_specs_from_manifest(
        flat_sds, effective
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    errors.append(f"Invalid to_hf_manifest: {exc}")
    return
  missing = sorted(
      key_set - ({s.source_key for s in specs} | dropped | tied_dst)
  )
  if missing:
    errors.append(
        f"to_hf_manifest missing {len(missing)} bound key(s): {missing[:5]!r}"
    )


def _validate_from_hf_manifest(
    key_set: AbstractSet[str],
    from_hf_manifest: hf_mapping.RuleSpec,
    errors: list[str],
) -> None:
  """Checks that `from_hf_manifest` is valid and covers all bound keys."""
  try:
    parsed = hf_mapping.parse_manifest(from_hf_manifest)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    errors.append(f"Invalid from_hf_manifest: {exc}")
    return
  if not key_set:
    return
  dropped = frozenset(parsed["dropped_keys"])
  if not parsed["rules"]:
    missing = sorted(key_set & dropped)
    if missing:
      errors.append(
          f"from_hf_manifest missing {len(missing)} bound key(s):"
          f" {missing[:5]!r}"
      )
    return
  try:
    src_map, _, _ = safetensors_loader.SpmdHostStager.compile_manifest(parsed)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    errors.append(f"Invalid from_hf_manifest: {exc}")
    return
  targets = {p.target_key for p in src_map.values() if not p.is_raw_source}
  for rule in parsed["rules"]:
    if dropped.isdisjoint((*rule["source_keys"], *rule["target_keys"])):
      targets.update(rule["target_keys"])
  targets.update(dst for _, dst in parsed["tied_keys"] if dst not in dropped)
  missing = sorted(key_set - targets)
  if missing:
    errors.append(
        f"from_hf_manifest missing {len(missing)} bound key(s): {missing[:5]!r}"
    )


def _check_aot_hbm_warning(
    config: GCSSynchronizerConfig,
    compiled_plan: safetensors_saver.CompiledSavePlan | None,
    warnings: list[str],
) -> None:
  """Appends a warning when AOT temporary HBM exceeds 4x `max_bucket_bytes`."""
  if (
      compiled_plan is not None
      and compiled_plan.peak_temp_hbm_bytes > 4 * config.max_bucket_bytes
  ):
    warnings.append(
        "AOT peak_temp_hbm_bytes"
        f" ({compiled_plan.peak_temp_hbm_bytes >> 20} MiB) exceeds 4x"
        f" max_bucket_bytes ({(4 * config.max_bucket_bytes) >> 20} MiB);"
        " consider lowering max_bucket_bytes."
    )


def _collect_sync_warnings(
    config: GCSSynchronizerConfig,
    compiled_plan: safetensors_saver.CompiledSavePlan | None,
    warnings: list[str],
) -> None:
  """Appends reader parallelism and AOT HBM overhead warnings."""
  is_GOOGLE_INTERNAL_STOAGE_PATH = str(config.staging_dir).startswith("/GOOGLE_INTERNAL_STOAGE_PATH/")
  GOOGLE_INTERNAL_STOAGE_PATH_procs = (
      os.environ["TUNIX_CNS_PROCS"].strip()
      if is_GOOGLE_INTERNAL_STOAGE_PATH and "TUNIX_CNS_PROCS" in os.environ
      else ""
  )
  if config.read_procs == 1 or GOOGLE_INTERNAL_STOAGE_PATH_procs in ("0", "1"):
    warnings.append(
        "Safetensors reader parallelism is pinned <= 1; leave read_procs=None"
        " (auto) or set >= 16 to avoid Python GIL bottlenecks on remote reads."
    )
  _check_aot_hbm_warning(config, compiled_plan, warnings)


def _validate_writer_runtime_config(
    config: GCSSynchronizerConfig,
    arrays: Sequence[Any],
    shm_dir: str,
    existing_shm_bytes: int,
    errors: list[str],
    warnings: list[str],
    *,
    max_shard_bytes: int | None = None,
) -> None:
  """Delegates writer-only bucket/shard/pool validation to `safetensors_saver`."""
  total_bytes = sum(int(a.size) * a.dtype.itemsize for a in arrays)
  devices = _mesh_devices_for_arrays(arrays)
  num_ranks = len(devices) if devices else 1
  write_procs = _resolve_write_procs(config.staging_dir, config.write_procs)
  staging = config.staging_dir or ""
  is_remote = (
      staging.startswith("gs://")
      or staging == "/GOOGLE_INTERNAL_STOAGE_PATH"
      or staging.startswith("/GOOGLE_INTERNAL_STOAGE_PATH/")
  )
  has_explicit_write_procs = config.write_procs is not None or (
      staging.startswith("gs://") and "TUNIX_GCS_WRITE_PROCS" in os.environ
  )
  base = safetensors_saver.validate_safetensors_config(
      output_dir=config.staging_dir,
      max_bucket_bytes=config.max_bucket_bytes,
      max_shard_bytes=max_shard_bytes,
      total_bytes=total_bytes,
      num_ranks=num_ranks,
      write_procs=(write_procs if has_explicit_write_procs else None),
      shm_dir=("" if existing_shm_bytes > 0 else shm_dir),
      strict=False,
  )
  errors.extend(base.errors)
  warnings.extend(base.warnings)
  if existing_shm_bytes <= 0 or not is_remote or not os.path.isdir(shm_dir):
    return
  st = os.statvfs(shm_dir)
  effective_free = st.f_bavail * st.f_frsize + existing_shm_bytes
  bucket = config.max_bucket_bytes
  if effective_free < 2 * bucket:
    errors.append(
        f"{shm_dir} free space ({effective_free >> 20} MiB) is below 2x"
        f" max_bucket_bytes ({(2 * bucket) >> 20} MiB); mount an"
        " emptyDir(medium=Memory) volume in GKE/Docker."
    )
  elif effective_free < 8 * bucket:
    warnings.append(
        f"{shm_dir} free space ({effective_free >> 20} MiB) is below 8x"
        f" max_bucket_bytes ({(8 * bucket) >> 20} MiB); pipeline"
        " concurrency may throttle."
    )


def validate_weight_sync_runtime(
    config: GCSSynchronizerConfig,
    *,
    names: Sequence[str] = (),
    arrays: Sequence[Any] = (),
    to_hf_manifest: hf_mapping.RuleSpec | None = None,
    from_hf_manifest: hf_mapping.RuleSpec | None = None,
    compiled_plan: safetensors_saver.CompiledSavePlan | None = None,
    shm_dir: str = "/dev/shm",
    existing_shm_bytes: int = 0,
    strict: bool | None = None,
    is_rapid: bool = False,
) -> ValidationReport:
  """Validates weight-sync configuration, manifests, AOT HBM, and host runtime.

  Invoked automatically by :class:`GCSSynchronizer` in ``__init__`` and
  :meth:`GCSSynchronizer.bind`, and also callable directly as a preflight check
  before launching a multi-host RL job.

  Checks performed:
    1. Delegates bucket/shard sizing (`max_bucket_bytes <= max_shard_bytes <=
       4 GiB`), `/dev/shm` capacity, `/GOOGLE_INTERNAL_STOAGE_PATH/` syntax, GCS OAuth2 token, and
       multi-VIP DNS checks to `safetensors_saver.validate_safetensors_config`.
    2. Warns if reader parallelism (`read_procs`, or `TUNIX_CNS_PROCS` on
       `/GOOGLE_INTERNAL_STOAGE_PATH/` staging paths) is pinned `<= 1` (single-process GIL bottleneck).
    3. When `names` and `arrays` are provided, validates that every bound key
       is covered by `to_hf_manifest` and/or `from_hf_manifest`.
    4. When `compiled_plan` is provided, warns if AOT `peak_temp_hbm_bytes`
       exceeds `4 * config.max_bucket_bytes`.

  Args:
    config: Active :class:`GCSSynchronizerConfig`.
    names: Optional sequence of bound parameter names.
    arrays: Optional sequence of bound parameter arrays or `ShapeDtypeStruct`s.
    to_hf_manifest: Optional declarative `Tunix -> HF` export manifest.
    from_hf_manifest: Optional declarative `HF -> Tunix` restore manifest.
    compiled_plan: Optional :class:`safetensors_saver.CompiledSavePlan`.
    shm_dir: Shared-memory mount path to inspect (defaults to `"/dev/shm"`).
    existing_shm_bytes: Bytes already allocated in `shm_dir` by a live
      `MultiprocStagePool` on this synchronizer, credited back on re-bind.
    strict: If True, raises `ValueError` on any error OR warning. Defaults to
      `config.strict_validation`.
    is_rapid: Whether `config.staging_dir` is already known to be a Rapid Zonal
      bucket (skips per-shard cap checks overridden by Rapid per-wave uploads).

  Returns:
    A :class:`ValidationReport` listing all detected `errors` and `warnings`.

  Raises:
    ValueError: When `strict` is True and any error or warning is detected.

  Example:
    ```python
    cfg = gcs_synchronizer.GCSSynchronizerConfig(
        staging_dir="gs://my-bucket/rl_sync",
        max_bucket_bytes=256 << 20,
        max_shard_bytes=2560 << 20,
    )
    report = gcs_synchronizer.validate_weight_sync_runtime(
        cfg,
        to_hf_manifest=qwen3_params.build_to_hf_manifest(model_cfg),
    )
    report.raise_if_invalid(strict=False, context="preflight")
    ```
  """
  errors: list[str] = []
  warnings: list[str] = []
  is_readonly_reader = from_hf_manifest is not None and to_hf_manifest is None
  if config.checkpoint_format == "safetensors" and not is_readonly_reader:
    _validate_writer_runtime_config(
        config,
        arrays,
        shm_dir,
        existing_shm_bytes,
        errors,
        warnings,
        max_shard_bytes=(None if is_rapid else config.max_shard_bytes),
    )
  _collect_sync_warnings(config, compiled_plan, warnings)
  keys = [weight_utils.param_key(n) for n in names] if names and arrays else []
  key_set = set(keys)
  if to_hf_manifest is not None:
    _validate_to_hf_manifest(key_set, keys, arrays, to_hf_manifest, errors)
  if from_hf_manifest is not None:
    _validate_from_hf_manifest(key_set, from_hf_manifest, errors)
  report = ValidationReport(errors=tuple(errors), warnings=tuple(warnings))
  use_strict = config.strict_validation if strict is None else strict
  if use_strict:
    report.raise_if_invalid(strict=True, context="GCSSynchronizer")
  return report


@dataclasses.dataclass(frozen=True)
class GCSSynchronizerConfig:
  """Configuration for file-based Safetensors/Orbax weight synchronization.

  Explicitly separates **HBM/D2H wave memory** (`max_bucket_bytes`) from
  **output `.safetensors` shard file size** (`max_shard_bytes`):

    * `max_bucket_bytes` (default `128 MiB`) bounds temporary per-rank HBM and
      D2H staging memory during each export wave so large MoE models never OOM.
    * `max_shard_bytes` (default `2.5 GiB`, hard ceiling `4 GiB`) packs up to
      32 consecutive waves per rank into ONE `.safetensors` file. On `gs://`,
      waves stream immediately via :class:`safetensors_saver.MultiprocStagePool`
      and finalize server-side via GCS Parallel Composite Upload.

  Environment variables (used when the corresponding field is `None`):
    * `TUNIX_GCS_WRITE_PROCS`: Override auto writer subprocess count on `gs://`.
    * `TUNIX_CNS_PROCS`: Override auto reader subprocesses on 1P `/GOOGLE_INTERNAL_STOAGE_PATH/`.

  Attributes:
    staging_dir: Base directory or URI (`gs://...`, `/GOOGLE_INTERNAL_STOAGE_PATH/...`, or local path)
      where the source worker writes per-round `step_<policy_version>_<req_id>`
      checkpoints. Required on the source worker; optional on read-only
      destination workers (`h2d`-only).
    max_to_keep: Maximum number of staged source checkpoints retained after
      :meth:`GCSSynchronizer.release`. Must be `>= 0`.
    checkpoint_format: Serialization format: `"safetensors"` (default, fast
      multi-process `/dev/shm` + multi-VIP HTTP) or `"orbax"` (legacy baseline).
    max_bucket_bytes: Per-rank HBM/D2H wave bucket limit in bytes (default `128
      MiB`). Must be `>= 1` and `<= max_shard_bytes`.
    max_shard_bytes: Maximum output `.safetensors` shard file size in bytes
      (default `2.5 GiB`; hard ceiling `MAX_SAFETENSORS_SHARD_BYTES` = `4 GiB`).
      Set `None` to fall back to legacy `per_wave_shards` / per-rank shards.
    per_wave_shards: Legacy override for per-wave shard files (only active when
      `max_shard_bytes is None`).
    async_write: Whether `"safetensors"` export overlaps D2H and storage writes.
    write_procs: Explicit :class:`safetensors_saver.MultiprocStagePool` worker
      process count for remote writes (`gs://` / `/GOOGLE_INTERNAL_STOAGE_PATH/`), or `None` for auto
      (`min(48, max(16, cpu_count))` on `gs://`, `16` on `/GOOGLE_INTERNAL_STOAGE_PATH/`). Set `1` to
      force in-process writes.
    read_procs: Explicit `/dev/shm` reader worker process count for remote reads
      (`gs://` / `/GOOGLE_INTERNAL_STOAGE_PATH/`), or `None` for auto (`8..96` scaled by model bytes on
      `gs://` when bound model `>= 64 MiB`).
    aot_compile: When True (default), pre-compiles `SavePlan` XLA kernels during
      :meth:`GCSSynchronizer.bind` to eliminate first-step JIT stalls and audit
      `peak_temp_hbm_bytes`.
    strict_validation: When True, promotes runtime validation warnings (e.g.
      single-process bottleneck or low `/dev/shm`) to hard `ValueError`s.
    use_ocdbt: Whether legacy `"orbax"` writes OCDBT tables.
    use_zarr3: Whether legacy `"orbax"` writes Zarr v3 chunks.
    use_compression: Whether legacy `"orbax"` enables chunk compression.
    save_concurrent_gb: Max concurrent GB written per host (legacy `"orbax"`).
    restore_concurrent_gb: Max concurrent GB read per host (legacy `"orbax"`).
    cleanup_on_start: When True on a source worker with `staging_dir` set,
      removes pre-existing `step_*` directories during `__init__`.

  Example:
    ```python
    from tunix.experimental.weight_sync import gcs_synchronizer

    # Zero manual env vars required: auto-sizes write_procs, read_procs,
    # /dev/shm ring arena, GCS parallel composite upload, and AOT compile.
    trainer_cfg = gcs_synchronizer.GCSSynchronizerConfig(
        staging_dir="gs://my-bucket/rl_weight_sync/run1",
        max_to_keep=2,
        checkpoint_format="safetensors",
        max_bucket_bytes=256 * 1024 * 1024,
        max_shard_bytes=2560 * 1024 * 1024,
        aot_compile=True,
    )
    # Read-only Sampler / Rollout worker does not need staging_dir.
    sampler_cfg = gcs_synchronizer.GCSSynchronizerConfig(
        checkpoint_format="safetensors",
    )
    ```
  """

  staging_dir: str | None = None
  max_to_keep: int = 1
  checkpoint_format: str = "safetensors"
  max_bucket_bytes: int = safetensors_saver.DEFAULT_MAX_BUCKET_BYTES
  max_shard_bytes: int | None = safetensors_saver.DEFAULT_MAX_SHARD_BYTES
  per_wave_shards: bool | None = None
  async_write: bool = True
  write_procs: int | None = None
  read_procs: int | None = None
  aot_compile: bool = True
  strict_validation: bool = False
  use_ocdbt: bool = True
  use_zarr3: bool = False
  use_compression: bool = True
  save_concurrent_gb: int = 96
  restore_concurrent_gb: int = 96
  cleanup_on_start: bool = False

  def __post_init__(self) -> None:
    if self.staging_dir is not None and not self.staging_dir.strip():
      raise ValueError(
          "GCSSynchronizerConfig.staging_dir must not be empty when set."
      )
    if isinstance(self.max_to_keep, bool) or self.max_to_keep < 0:
      raise ValueError(
          "GCSSynchronizerConfig.max_to_keep must be a non-negative int, got"
          f" {self.max_to_keep!r}."
      )
    if self.checkpoint_format not in _VALID_CHECKPOINT_FORMATS:
      raise ValueError(
          "GCSSynchronizerConfig.checkpoint_format must be 'safetensors' or"
          f" 'orbax', got {self.checkpoint_format!r}."
      )
    _validate_positive_int(self.max_bucket_bytes, "max_bucket_bytes")
    _validate_positive_int(
        self.max_shard_bytes, "max_shard_bytes", allow_none=True
    )
    _validate_positive_int(self.write_procs, "write_procs", allow_none=True)
    _validate_positive_int(self.read_procs, "read_procs", allow_none=True)
    _validate_positive_int(self.save_concurrent_gb, "save_concurrent_gb")
    _validate_positive_int(self.restore_concurrent_gb, "restore_concurrent_gb")
    if (
        self.max_shard_bytes is not None
        and self.max_shard_bytes > safetensors_saver.MAX_SAFETENSORS_SHARD_BYTES
    ):
      raise ValueError(
          "GCSSynchronizerConfig.max_shard_bytes exceeds 4 GiB ceiling"
          f" ({safetensors_saver.MAX_SAFETENSORS_SHARD_BYTES}), got"
          f" {self.max_shard_bytes}."
      )
    if (
        self.max_shard_bytes is not None
        and self.max_bucket_bytes > self.max_shard_bytes
    ):
      raise ValueError(
          f"GCSSynchronizerConfig.max_bucket_bytes ({self.max_bucket_bytes})"
          f" exceeds max_shard_bytes ({self.max_shard_bytes})."
      )
    if self.cleanup_on_start and self.staging_dir is None:
      raise ValueError(
          "GCSSynchronizerConfig.cleanup_on_start=True requires staging_dir."
      )


class GCSSynchronizer:
  """Worker-side weight synchronizer staging weights via Safetensors or Orbax.

  `verify_weights` compares checksums in the wire namespace: with a manifest,
  both peers project their bound arrays onto the safetensors wire tensors on
  device and checksum those. The destination re-derives its checksums from the
  staged arrays, which verifies transport plus the manifest round-trip; this is
  exact for rename, transpose, reshape, (un)stack, and tied rules, while lossy
  rules such as dtype casts mismatch loudly.

  Supports both **manifest-driven HuggingFace export/import** (interoperable
  with vLLM, SGLang, HuggingFace, and Tunix models) and **pure Identity
  NNX-to-NNX sync** across arbitrary asymmetric Trainer/Sampler TPU meshes.

  Automatically manages a persistent
  :class:`safetensors_saver.MultiprocStagePool`
  for remote writes (`gs://` / `/GOOGLE_INTERNAL_STOAGE_PATH/`), multi-process `/dev/shm` range readers,
  index-based host shard filtering, AOT XLA kernel compilation during
  :meth:`bind`, and upfront runtime validation. Always call :meth:`close` (or
  use the synchronizer for the lifetime of the worker) to release subprocesses
  and `/dev/shm` arenas cleanly.

  Attributes:
    job_name: Logical worker group name (e.g. `"trainer"` or `"rollout"`).
    worker_index: Host/worker ordinal advertised in `WorkUnitMetadata`.
    data_replica_idx: Data-parallel replica ordinal.
    config: Active :class:`GCSSynchronizerConfig`.
    names: Flattened canonical parameter path names currently bound.
    arrays: Live device arrays currently bound.

  Example (a) — Manifest-based Trainer -> Sampler / vLLM (HuggingFace format):
    ```python
    from flax import nnx
    from tunix.dist_rl.common import datatypes
    from tunix.experimental.weight_sync import gcs_handler
    from tunix.experimental.weight_sync import gcs_synchronizer
    from tunix.models.qwen3 import params as qwen3_params

    to_hf = qwen3_params.build_to_hf_manifest(model_cfg)
    from_hf = qwen3_params.build_from_hf_manifest(model_cfg)

    trainer_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir="gs://my-bucket/ws_qwen3",
            max_bucket_bytes=256 << 20,
            max_shard_bytes=2560 << 20,
        ),
        to_hf_manifest=to_hf,
    )
    sampler_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="rollout",
        config=gcs_synchronizer.GCSSynchronizerConfig(),
        from_hf_manifest=from_hf,
    )
    trainer_sync.bind(nnx.state(trainer_model))  # Runs validation + AOT compile
    sampler_sync.bind(nnx.state(sampler_model))

    req = datatypes.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "step_1", "uuid": 1}
    )
    trainer_sync.d2h(req)  # Writes standard HF .safetensors shards to GCS
    src_meta = trainer_sync.work_unit_metadata()

    # Coordinator verifies finalization before Sampler reads:
    handler = gcs_handler.GCSHandler()
    handler.register_work_unit(sampler_sync.work_unit_metadata())
    handler.register_work_unit(src_meta)
    assert handler.transfer(
        [src_meta.unit], [sampler_sync.work_unit_metadata().unit]
    ).success

    # Sampler skips shards it does not own and restores directly onto its mesh:
    sampler_sync.h2d(
        datatypes.WeightSyncRequest(
            policy_version=1, source_metadata=(src_meta,)
        )
    )
    sampler_sync.apply_to_runner(sampler_runner)
    trainer_sync.close()
    sampler_sync.close()
    ```

  Example (b) — Homogeneous Tunix NNX -> NNX Identity sync (no manifest):
    ```python
    # Omit to_hf_manifest / from_hf_manifest to preserve exact Tunix keys and
    # tensor layouts with zero transpose heuristics across asymmetric meshes.
    trainer_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir="gs://my-bucket/ws_nnx_identity",
        ),
    )
    sampler_sync = gcs_synchronizer.GCSSynchronizer(job_name="rollout")
    trainer_sync.bind(nnx.state(trainer_model))
    sampler_sync.bind(nnx.state(sampler_model))

    req = datatypes.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "s1", "uuid": 1}
    )
    trainer_sync.d2h(req)
    sampler_sync.h2d(
        datatypes.WeightSyncRequest(
            policy_version=1,
            source_metadata=(trainer_sync.work_unit_metadata(),),
        )
    )
    sampler_sync.apply_to_sampler(sampler)
    trainer_sync.close()
    sampler_sync.close()
    ```
  """

  def __init__(
      self,
      job_name: str,
      config: GCSSynchronizerConfig | None = None,
      *,
      worker_index: int | None = None,
      data_replica_idx: int = 0,
      verify_weights: bool | None = None,
      to_hf_manifest: hf_mapping.RuleSpec | None = None,
      from_hf_manifest: hf_mapping.RuleSpec | None = None,
  ) -> None:
    """Initializes `GCSSynchronizer`, validates config, and optionally cleans up.

    Args:
      job_name: Non-empty logical worker role (e.g. `"trainer"`, `"rollout"`).
      config: Optional :class:`GCSSynchronizerConfig` (defaults to standard
        safetensors settings).
      worker_index: Optional explicit worker index (defaults to
        `jax.process_index()`).
      data_replica_idx: Data-parallel replica ordinal for metadata routing.
      verify_weights: Enables per-round L1 checksum verification when True;
        defaults to `VERIFY_WEIGHTS=true` env var.
      to_hf_manifest: Optional declarative `Tunix -> HF` manifest (`RuleSpec`)
        used during `d2h()` export.
      from_hf_manifest: Optional declarative `HF -> Tunix` manifest (`RuleSpec`)
        used during `h2d()` restore.

    Raises:
      ValueError: If `job_name` is empty, `worker_index < 0`, or upfront
        runtime validation fails.
    """
    if not job_name or not job_name.strip():
      raise ValueError("GCSSynchronizer job_name must not be empty.")
    resolved_worker_index = (
        jax.process_index() if worker_index is None else worker_index
    )
    if resolved_worker_index < 0:
      raise ValueError(
          "GCSSynchronizer worker_index must be non-negative, got"
          f" {resolved_worker_index}."
      )
    self.job_name: str = job_name
    self.worker_index: int = resolved_worker_index
    self.data_replica_idx: int = data_replica_idx
    self.config: GCSSynchronizerConfig = (
        config if config is not None else GCSSynchronizerConfig()
    )
    env_verify = os.environ.get("VERIFY_WEIGHTS", "").strip().lower() == "true"
    self._verify_weights: bool = (
        env_verify if verify_weights is None else bool(verify_weights)
    )
    self._to_hf_manifest: dict[str, Any] | None = (
        hf_mapping.parse_manifest(to_hf_manifest)
        if to_hf_manifest is not None
        else None
    )
    self._from_hf_manifest: dict[str, Any] | None = (
        hf_mapping.parse_manifest(from_hf_manifest)
        if from_hf_manifest is not None
        else None
    )
    self.names: list[str] = []
    self.arrays: list[Any] = []
    self._bound: bool = False
    self._pending_h2d_arrays: list[Any] | None = None
    self._staged_checkpoints: list[str] = []
    self._current_artifact_uri: str | None = None
    self._current_checksums: dict[str, float | int] | None = None
    self._last_d2h_secs: float = 0.0
    self._last_h2d_secs: float = 0.0
    self._d2h_count: int = 0
    self._h2d_count: int = 0
    self._prune_executor: futures.ThreadPoolExecutor | None = None
    self._stage_pool: safetensors_saver.MultiprocStagePool | None = None
    self._stage_pool_sig: tuple[int, int, int, bool, bool] | None = None
    self._stage_pool_finalizer: weakref.finalize | None = None
    self._read_arena: safetensors_loader.ShmRangeReader | None = None
    self._read_arena_budget: int = 0
    self._read_arena_host_bytes: int = 0
    self._read_arena_finalizer: weakref.finalize | None = None
    self._save_plan: safetensors_saver.SavePlan | None = None
    self._compiled_save_plan: safetensors_saver.CompiledSavePlan | None = None
    self._plan_sig: tuple[Any, ...] | None = None
    self._last_export_metrics: safetensors_saver.SaveMetrics | None = None
    self._last_validation_report: ValidationReport | None = None
    self._wire_l1_sums: (
        Callable[[dict[str, Any]], dict[str, jax.Array]] | None
    ) = None
    self._wire_l1_sums_digest: str | None = None
    self._last_total_shards: int = 0
    self._last_filtered_shards: int = 0

    self._last_validation_report = validate_weight_sync_runtime(
        self.config,
        to_hf_manifest=self._to_hf_manifest,
        from_hf_manifest=self._from_hf_manifest,
        strict=False,
    )
    self._last_validation_report.raise_if_invalid(
        strict=self.config.strict_validation, context=self.job_name
    )
    self._maybe_cleanup_on_start()

  def _maybe_cleanup_on_start(self) -> None:
    """Removes pre-existing `step_*` staging directories when enabled."""
    if not self.config.cleanup_on_start or self.config.staging_dir is None:
      return
    assert self.config.staging_dir is not None
    staging_root = epath.Path(self.config.staging_dir)

    def _cleanup() -> tuple[None, None]:
      if not staging_root.exists():
        return None, None
      for child in staging_root.glob("step_*"):
        safetensors_saver.remove_tree(child)
      return None, None

    _broadcast_host0_status(_cleanup)

  @property
  def bound(self) -> bool:
    """Whether live model parameters are currently bound."""
    return self._bound

  @property
  def active(self) -> bool:
    """Whether this synchronizer is actively bound and ready for transfers."""
    return self._bound

  @property
  def verify_weights(self) -> bool:
    """Whether per-round L1 checksum verification is enabled."""
    return self._verify_weights

  @property
  def last_validation_report(self) -> ValidationReport | None:
    """Returns the most recent :class:`ValidationReport` from init or bind."""
    return self._last_validation_report

  def _checkpointer(self) -> ocp.Checkpointer:
    """Builds an Orbax checkpointer with barriers isolated to this synchronizer."""
    mp_options = ocp.options.MultiprocessingOptions(
        barrier_sync_key_prefix=f"gcs_ws_{self.job_name}"
    )
    handler = ocp.PyTreeCheckpointHandler(
        save_concurrent_gb=self.config.save_concurrent_gb,
        restore_concurrent_gb=self.config.restore_concurrent_gb,
        use_ocdbt=self.config.use_ocdbt,
        use_zarr3=self.config.use_zarr3,
        use_compression=self.config.use_compression,
        multiprocessing_options=mp_options,
    )
    return ocp.Checkpointer(handler, multiprocessing_options=mp_options)

  def _close_stage_pool(self, *, abort: bool = True) -> None:
    """Detaches finalizer, clears references, and closes the stage pool."""
    fin = self._stage_pool_finalizer
    self._stage_pool_finalizer = None
    if fin is not None:
      fin.detach()
    pool = self._stage_pool
    self._stage_pool = None
    self._stage_pool_sig = None
    if pool is not None:
      pool.close(abort=abort)

  def _ensure_stage_pool(
      self, target_uri: str, *, is_rapid: bool | None = None
  ) -> safetensors_saver.MultiprocStagePool | None:
    """Lazily creates and reuses a persistent remote `MultiprocStagePool`."""
    procs = _resolve_write_procs(target_uri, self.config.write_procs)
    if procs <= 1:
      self._close_stage_pool(abort=True)
      return None
    # Size the `/dev/shm` staging arena proportionally for small models (e.g.
    # 0.6B -> 4 GiB, <0.1s pre-fault) while letting large models (e.g. 397B@8H,
    # ~91.5 GiB/host) auto-size up to 128 GiB so acquire_slot() never stalls.
    host_bytes = sum(
        int(a.size) * a.dtype.itemsize for a in self.arrays
    ) // max(1, jax.process_count())
    shm_budget = int(
        max(4 << 30, min(128 << 30, host_bytes * 2))
        if 0 < host_bytes < (64 << 30)
        else 0
    )
    use_gcs_http = target_uri.startswith("gs://")
    rapid = (
        _is_rapid_zonal_consensus(target_uri)
        if is_rapid is None and use_gcs_http
        else bool(is_rapid)
    )
    use_gcs_rapid = use_gcs_http and rapid
    pool_sig = (
        procs,
        self.config.max_bucket_bytes,
        shm_budget,
        use_gcs_http,
        use_gcs_rapid,
    )
    if (
        self._stage_pool is not None
        and self._stage_pool_sig == pool_sig
        and self._stage_pool.is_healthy()
    ):
      return self._stage_pool
    self._close_stage_pool(abort=True)
    shm_root = (
        "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    )
    stage_dir = os.path.join(
        shm_root,
        f"tunix_ws_stage_{os.getpid()}_{next(_STAGE_POOL_COUNTER)}",
    )
    pool = safetensors_saver.MultiprocStagePool(
        stage_dir=stage_dir,
        num_procs=procs,
        shm_budget_bytes=shm_budget,
        max_bucket_bytes=self.config.max_bucket_bytes,
        use_gcs_http_write=use_gcs_http,
        use_gcs_rapid_grpc=use_gcs_rapid,
    )
    self._stage_pool = pool
    self._stage_pool_sig = pool_sig
    self._stage_pool_finalizer = weakref.finalize(self, pool.close, abort=True)
    return self._stage_pool

  def _ensure_read_arena(
      self, resolved_uri: str, host_bytes: int
  ) -> safetensors_loader.ShmRangeReader | None:
    """Lazily creates and reuses a persistent bounded remote read arena.

    Observed on 397B@2H (us-central1): tearing down and recreating
    `ShmRangeReader` every sync triggered Linux 4 KiB tmpfs cold page-fault
    stalls (~3.2 GB/s) and `mm.close()+unlink()` overhead on every sync.
    Reusing one bounded circular-ring arena (`<= 192 GiB`, pre-faulted once and
    recycled per committed file/layer group) across syncs keeps pages warm at
    8-10 GB/s/host with 0 GiB `/dev/shm` growth and full reclamation in
    `close()`.

    Args:
      resolved_uri: Remote or local checkpoint URI being restored.
      host_bytes: Total host-local parameter bytes to restore.

    Returns:
      Persistent :class:`safetensors_loader.ShmRangeReader` for `gs://` or
      `/GOOGLE_INTERNAL_STOAGE_PATH/` URIs, or `None` for local paths (ephemeral arena).
    """
    if host_bytes <= 0 or not (
        resolved_uri.startswith("gs://") or resolved_uri.startswith("/GOOGLE_INTERNAL_STOAGE_PATH/")
    ):
      return None
    desired = min(
        _MAX_PERSISTENT_READ_ARENA_BYTES, max(64 << 20, host_bytes * 4)
    )
    # Check reuse BEFORE querying `statvfs`: once the persistent arena faults
    # pages in `/dev/shm`, free space drops by the arena's footprint, which
    # would otherwise shrink `max_safe` on Sync 2 and spuriously recreate it.
    if self._read_arena is not None and self._read_arena.mm is not None:
      if (
          host_bytes <= self._read_arena_host_bytes
          or desired <= self._read_arena_budget
      ):
        self._read_arena_host_bytes = max(
            self._read_arena_host_bytes, host_bytes
        )
        return self._read_arena
      self._close_read_arena()
    budget = _compute_read_arena_budget(host_bytes)
    arena = safetensors_loader.ShmRangeReader(shm_budget_bytes=budget)
    self._read_arena = arena
    self._read_arena_budget = budget
    self._read_arena_host_bytes = host_bytes
    self._read_arena_finalizer = weakref.finalize(
        self, arena.__exit__, None, None, None
    )
    return self._read_arena

  def _close_read_arena(self) -> None:
    """Unmaps worker `/dev/shm` mmaps and unlinks the persistent read arena."""
    fin = self._read_arena_finalizer
    self._read_arena_finalizer = None
    if fin is not None:
      fin.detach()
    arena = self._read_arena
    self._read_arena = None
    self._read_arena_budget = 0
    self._read_arena_host_bytes = 0
    if arena is not None:
      arena.__exit__(None, None, None)

  def _prepare_save_plan(
      self, target_uri: str, *, is_rapid: bool | None = None
  ) -> tuple[
      dict[str, jax.Array],
      safetensors_saver.SavePlan,
      safetensors_saver.CompiledSavePlan | None,
      list[jax.Device] | None,
  ]:
    """Builds or reuses cached `SavePlan` and AOT `CompiledSavePlan`."""
    flat_state, specs, unstack_counts, devices = (
        _build_safetensors_export_inputs(
            self.names,
            self.arrays,
            self.config.max_bucket_bytes,
            to_hf_manifest=self._to_hf_manifest,
        )
    )
    num_ranks = len(devices) if devices else 1
    backend = safetensors_saver.resolve_storage_backend(target_uri)
    effective_max_shard, use_wave_shards = _resolve_save_shard_policy(
        self.job_name, target_uri, self.config, backend, is_rapid=is_rapid
    )
    effective_wave_shards = (
        use_wave_shards if effective_max_shard is None else False
    )
    manifest_digest = (
        _manifest_digest(self._to_hf_manifest)
        if self._to_hf_manifest is not None
        else None
    )
    layout_sig = tuple(
        (
            k,
            tuple(int(d) for d in a.shape),
            a.dtype.name,
            a.sharding if isinstance(a, jax.Array) else None,
        )
        for k, a in flat_state.items()
    )
    sig = (
        self.config.max_bucket_bytes,
        effective_max_shard,
        effective_wave_shards,
        num_ranks,
        self.config.aot_compile,
        manifest_digest,
        layout_sig,
    )
    if self._save_plan is None or self._plan_sig != sig:
      plan = safetensors_saver.build_save_plan(
          specs,
          num_ranks=num_ranks,
          max_bucket_bytes=self.config.max_bucket_bytes,
          unstack_counts=unstack_counts,
          per_wave_shards=use_wave_shards,
          max_shard_bytes=effective_max_shard,
      )
      compiled = (
          safetensors_saver.compile_save_plan(flat_state, plan, devices=devices)
          if self.config.aot_compile
          else None
      )
      self._save_plan = plan
      self._compiled_save_plan = compiled
      self._plan_sig = sig
    return flat_state, self._save_plan, self._compiled_save_plan, devices

  def _save_checkpoint(self, ckpt_path: epath.Path, ckpt_uri: str) -> None:
    """Writes bound weights to `ckpt_path` in `config.checkpoint_format`."""
    if self.config.checkpoint_format == "orbax":
      flat_tree = {
          weight_utils.param_key(name): arr
          for name, arr in zip(self.names, self.arrays, strict=True)
      }
      with self._checkpointer() as checkpointer:
        checkpointer.save(
            ckpt_path,
            args=ocp.args.PyTreeSave(item=flat_tree),
            force=True,
        )
      return
    is_rapid = _is_rapid_zonal_consensus(ckpt_uri)
    flat_state, plan, compiled_plan, devices = self._prepare_save_plan(
        ckpt_uri, is_rapid=is_rapid
    )
    pool: safetensors_saver.MultiprocStagePool | None = None
    pool_exc: Exception | None = None
    try:
      pool = self._ensure_stage_pool(ckpt_uri, is_rapid=is_rapid)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      pool_exc = exc
    _sync_multihost_error(pool_exc, f"Stage pool setup for {ckpt_uri}")
    pool_ctx = pool.activate() if pool is not None else contextlib.nullcontext()
    try:
      with pool_ctx:
        self._last_export_metrics = (
            safetensors_saver.export_safetensors_with_plan(
                flat_state,
                plan,
                ckpt_uri,
                devices=devices,
                async_write=self.config.async_write,
                defer_contiguous=True,
                compiled_plan=compiled_plan,
            )
        )
    except BaseException:
      if pool is not None:
        self._close_stage_pool(abort=True)
      raise

  def _restore_checkpoint(self, resolved_uri: str) -> dict[str, Any]:
    """Restores checkpoint weights from `resolved_uri` via Safetensors/Orbax."""
    ckpt_path = epath.Path(resolved_uri)
    # Overlap optional `.tunix_headers.json` sidecar read with `index.json`
    # only for safetensors (avoid wasted remote GET on Orbax restores).
    sidecar_fut = None
    if self.config.checkpoint_format != "orbax":
      # pylint: disable-next=protected-access
      sidecar_fut = safetensors_loader._SHARED_POOL.submit(
          safetensors_io.load_sidecar_headers, resolved_uri
      )

    def _probe_host0() -> tuple[None, dict[str, Any]]:
      shard_names, weight_map = _resolve_safetensors_layout(ckpt_path)
      return None, {
          "is_safetensors": shard_names is not None,
          "shard_names": shard_names,
          "weight_map": weight_map,
      }

    try:
      _, meta = _broadcast_host0_status(_probe_host0)
    except BaseException:
      if sidecar_fut is not None:
        sidecar_fut.cancel()
      raise
    if meta["is_safetensors"]:
      sidecar_headers = (
          sidecar_fut.result() if sidecar_fut is not None else None
      )
      restored, total_shards, filtered_shards = _restore_safetensors(
          resolved_uri,
          self.names,
          self.arrays,
          from_hf_manifest=self._from_hf_manifest,
          read_procs=self.config.read_procs,
          shard_names=meta["shard_names"],
          weight_map=meta["weight_map"],
          shard_headers=sidecar_headers,
          arena_provider=lambda hb: self._ensure_read_arena(resolved_uri, hb),
      )
      self._last_total_shards = total_shards
      self._last_filtered_shards = filtered_shards
      return restored
    if sidecar_fut is not None:
      sidecar_fut.cancel()
    item, restore_args = _build_restore_specs(self.names, self.arrays)
    with self._checkpointer() as checkpointer:
      return checkpointer.restore(
          ckpt_path,
          args=ocp.args.PyTreeRestore(item=item, restore_args=restore_args),
      )

  def _validate_and_aot_on_bind(self) -> None:
    """Validates bound weights/manifests and runs AOT compilation if enabled."""
    if self._stage_pool is not None and not self._stage_pool.is_healthy():
      self._close_stage_pool(abort=True)
    existing_shm = (
        self._stage_pool.total_arena_bytes
        if self._stage_pool is not None
        else 0
    )
    # Resolve Rapid Zonal consensus once BEFORE strict validation so hosts
    # cannot diverge on a transient bucket probe and deadlock collectives.
    will_eager_writer = (
        self.config.checkpoint_format == "safetensors"
        and self.config.staging_dir is not None
        and self._to_hf_manifest is not None
    )
    is_rapid = (
        _is_rapid_zonal_consensus(self.config.staging_dir)
        if will_eager_writer and self.config.staging_dir is not None
        else False
    )
    report = validate_weight_sync_runtime(
        self.config,
        names=self.names,
        arrays=self.arrays,
        to_hf_manifest=self._to_hf_manifest,
        from_hf_manifest=self._from_hf_manifest,
        existing_shm_bytes=existing_shm,
        strict=False,
        is_rapid=is_rapid,
    )
    self._last_validation_report = report
    report.raise_if_invalid(
        strict=self.config.strict_validation, context=self.job_name
    )
    # Only eagerly AOT-compile and warm the writer stage pool during bind()
    # when an explicit export manifest identifies this worker as a writer;
    # rollout readers (including Identity readers) may carry a shared
    # `staging_dir` and would otherwise waste XLA compile time and `/dev/shm`.
    if not will_eager_writer or self.config.staging_dir is None:
      return
    if self.config.aot_compile:
      _, _, compiled, _ = self._prepare_save_plan(
          self.config.staging_dir, is_rapid=is_rapid
      )
      hbm_warnings: list[str] = []
      _check_aot_hbm_warning(self.config, compiled, hbm_warnings)
      if hbm_warnings:
        self._last_validation_report = ValidationReport(
            errors=report.errors,
            warnings=report.warnings + tuple(hbm_warnings),
        )
        ValidationReport(warnings=tuple(hbm_warnings)).raise_if_invalid(
            strict=self.config.strict_validation, context=self.job_name
        )
    self._ensure_stage_pool(self.config.staging_dir, is_rapid=is_rapid)

  def bind(self, state: Any) -> None:
    """Binds live model parameters, validates manifests, and AOT-compiles.

    Flattens `state`, filters out non-parameter leaves and KV caches, checks
    key uniqueness and manifest coverage, and—when `checkpoint_format ==
    "safetensors"`, `staging_dir` and `to_hf_manifest` are set, and
    `config.aot_compile` is True—pre-compiles all XLA export kernels once so
    the first `d2h()` incurs zero
    JIT compilation stall.

    Args:
      state: Model parameter PyTree (e.g. `nnx.state(model)` or a flat
        `dict[str, jax.Array]`).

    Raises:
      ValueError: If `state` has no bindable arrays, colliding canonical keys,
        missing manifest mappings, or invalid runtime configuration.

    Example:
      ```python
      trainer_sync.bind(nnx.state(trainer_model))
      print("AOT kernels:", trainer_sync.metrics()["aot_num_kernels"])
      ```
    """
    raw_names, raw_arrays = weight_utils.flatten_weights(state)
    keep_names, keep_arrays = weight_utils.filter_bindable(
        raw_names, raw_arrays, allow_proxy=True, label="gcs"
    )
    if not keep_arrays:
      raise ValueError(
          f"{self.job_name}: no bindable weight arrays found in state."
      )
    canonical_keys = [weight_utils.param_key(name) for name in keep_names]
    has_invalid_keys = any(not key for key in canonical_keys) or (
        len(set(canonical_keys)) != len(canonical_keys)
    )
    if has_invalid_keys:
      raise ValueError(
          f"{self.job_name}: bound parameter names produced empty or colliding"
          " canonical keys."
      )
    prev_state = (
        self.names,
        self.arrays,
        self._bound,
        self._save_plan,
        self._compiled_save_plan,
        self._plan_sig,
        self._last_validation_report,
    )
    self.names = keep_names
    self.arrays = keep_arrays
    try:
      self._validate_and_aot_on_bind()
    except BaseException:
      (
          self.names,
          self.arrays,
          self._bound,
          self._save_plan,
          self._compiled_save_plan,
          self._plan_sig,
          self._last_validation_report,
      ) = prev_state
      raise
    self.abort()
    self._bound = True

  def d2h(
      self, sync_request: datatypes.WeightSyncRequest | None = None
  ) -> None:
    """Exports bound weights to `<staging_dir>/step_<policy_version>_<req_id>`.

    In `"safetensors"` mode on `gs://` or `/GOOGLE_INTERNAL_STOAGE_PATH/`, automatically activates the
    persistent :class:`safetensors_saver.MultiprocStagePool` (with streaming
    GCS Parallel Composite Upload for `max_shard_bytes` shard groups) and
    reuses the AOT-compiled `SavePlan`. Updates `work_unit_metadata()` with the
    finalized `artifact_uri` and optional L1 checksums.

    Args:
      sync_request: `WeightSyncRequest` carrying integer `policy_version` and
        non-empty `extra_config["req_id"]`.

    Raises:
      RuntimeError: If called before :meth:`bind`.
      ValueError: If `config.staging_dir` is unset or `sync_request` is invalid.

    Example:
      ```python
      req = datatypes.WeightSyncRequest(
          policy_version=2, extra_config={"req_id": "step_2", "uuid": 2}
      )
      trainer_sync.d2h(req)
      src_meta = trainer_sync.work_unit_metadata()
      ```
    """
    if not self._bound:
      raise RuntimeError(f"{self.job_name}: bind() must run before d2h().")
    if self.config.staging_dir is None:
      raise ValueError(
          f"{self.job_name}: GCSSynchronizerConfig.staging_dir is required for"
          " source d2h()."
      )
    step_name = _resolve_step_dir_name(sync_request)
    ckpt_path = epath.Path(self.config.staging_dir) / step_name
    ckpt_uri = str(ckpt_path)

    t0 = time.perf_counter()
    run_blocking(lambda: self._save_checkpoint(ckpt_path, ckpt_uri))
    self._last_d2h_secs = time.perf_counter() - t0
    self._d2h_count += 1
    self._current_artifact_uri = ckpt_uri
    if ckpt_uri in self._staged_checkpoints:
      self._staged_checkpoints.remove(ckpt_uri)
    self._staged_checkpoints.append(ckpt_uri)
    self._current_checksums = (
        self._staged_checksums(self.arrays) if self._verify_weights else None
    )
    logging.info(
        "%s staged %d weight tensors to %s in %.2fs.",
        self.job_name,
        len(self.arrays),
        ckpt_uri,
        self._last_d2h_secs,
    )

  def h2d(
      self,
      sync_request: datatypes.WeightSyncRequest | None = None,
      *,
      checkpoint_path: str | None = None,
      source_checksums: Mapping[str, float | int] | None = None,
  ) -> None:
    """Restores a staged checkpoint into pending device buffers pre-commit.

    Parses `model.safetensors.index.json` first to filter out shards that hold
    zero tensors owned by this host, streams only needed byte ranges via
    multi-process `/dev/shm` `ShmRangeReader`, places them directly onto the
    bound destination shardings, and verifies L1 checksums if enabled. Call
    :meth:`apply_to_runner` or :meth:`apply_to_sampler` to commit atomically,
    or :meth:`abort` to discard staged buffers without modifying live weights.

    Args:
      sync_request: `WeightSyncRequest` whose `source_metadata` carries the
        source `artifact_uri` (used when `checkpoint_path` is `None`).
      checkpoint_path: Optional explicit checkpoint URI overriding
        `sync_request`.
      source_checksums: Optional expected checksum mapping overriding
        `sync_request.source_metadata`.

    Raises:
      RuntimeError: If called before :meth:`bind` or if checksum verification
        detects a mismatch (staging buffers are freed automatically).
      ValueError: If neither `checkpoint_path` nor valid `sync_request` is
        provided.
      KeyError: If a bound parameter is missing from the checkpoint.

    Example:
      ```python
      sampler_sync.h2d(
          datatypes.WeightSyncRequest(
              policy_version=2, source_metadata=(src_meta,)
          )
      )
      sampler_sync.apply_to_runner(sampler_runner)
      ```
    """
    if not self._bound:
      raise RuntimeError(f"{self.job_name}: bind() must run before h2d().")
    if checkpoint_path is not None and not checkpoint_path.strip():
      raise ValueError("checkpoint_path must not be empty when provided.")
    resolved_uri, extracted_checksums = (
        (checkpoint_path, None)
        if checkpoint_path is not None
        else extract_source_artifact(sync_request)
    )
    resolved_checksums = (
        source_checksums
        if source_checksums is not None
        else extracted_checksums
    )
    self.abort()
    verify = self._verify_weights and resolved_checksums is not None
    t0 = time.perf_counter()
    restored = run_blocking(lambda: self._restore_checkpoint(resolved_uri))
    pending_arrays = [
        restored[weight_utils.param_key(name)] for name in self.names
    ]
    jax.block_until_ready(pending_arrays)
    self._last_h2d_secs = time.perf_counter() - t0
    self._h2d_count += 1

    if verify:
      dst_checksums = self._staged_checksums(pending_arrays)
      try:
        weight_utils.verify_weight_checksums(resolved_checksums, dst_checksums)
      except Exception:
        weight_utils.delete_array_buffers(
            pending_arrays, keep_ids={id(a) for a in self.arrays}
        )
        raise

    self._pending_h2d_arrays = pending_arrays
    logging.info(
        "%s restored %d weight tensors from %s into staging in %.2fs.",
        self.job_name,
        len(pending_arrays),
        resolved_uri,
        self._last_h2d_secs,
    )

  def _staged_checksums(self, staged: Sequence[Any]) -> dict[str, float | int]:
    """Returns full L1 checksums of `staged` in the safetensors wire namespace.

    With a manifest, the projection onto wire tensors and the float32 abs-sum
    run in one jitted program over the (sharded) arrays, so nothing is gathered
    to a host and XLA fuses each layout op into its reduction. Every host
    computes identical values. In identity mode the bound names are the wire
    names.

    Args:
      staged: Arrays parallel to `self.names` (bound or freshly restored).
    """
    wire_manifest, inputs = self._wire_inputs(staged)
    if wire_manifest is None:
      return weight_utils.compute_checksums(
          self.names, staged, sample=None, canonical_keys=True
      )
    project = functools.partial(hf_mapping.to_hf_impl, wire_manifest)
    wire_digest = _manifest_digest(wire_manifest)
    if self._wire_l1_sums is None or self._wire_l1_sums_digest != wire_digest:
      # Built once per manifest so later syncs reuse the compiled program.
      self._wire_l1_sums = jax.jit(
          lambda tree: {
              k: jnp.sum(jnp.abs(v), dtype=jnp.float32)
              for k, v in project(tree).items()
          }
      )
      self._wire_l1_sums_digest = wire_digest
    sums = jax.device_get(self._wire_l1_sums(inputs))
    sizes = {k: v.size for k, v in jax.eval_shape(project, inputs).items()}
    out: dict[str, float | int] = {
        weight_utils.param_key(k): float(v) for k, v in sums.items()
    }
    out["__grand_total__"] = float(math.fsum(out.values()))
    out["__tensor_count__"] = len(sums)
    out["__element_count__"] = int(sum(sizes.values()))
    return out

  def _commit_pending_arrays(
      self, install_fn: Callable[[Sequence[Any]], None]
  ) -> None:
    """Installs staged arrays into target state and frees superseded buffers."""
    staged = self._pending_h2d_arrays
    if staged is None:
      install_fn(self.arrays)
      return
    old_bound = self.arrays
    install_fn(staged)
    weight_utils.delete_array_buffers(
        old_bound, keep_ids={id(a) for a in staged}
    )
    self.arrays = staged
    self._pending_h2d_arrays = None

  def apply_to_runner(self, runner: Any) -> None:
    """Atomically commits staged H2D weights into `runner` and frees old HBM.

    Replaces `runner.state` leaves with the pending arrays staged by
    :meth:`h2d`, deletes superseded device buffers to avoid HBM leaks, and
    updates this synchronizer's bound references.

    Args:
      runner: Model runner object (e.g. vLLM `TPUModelRunner` or NNX runner)
        exposing `.state` and `.state_leaves`, or `None` (no-op).

    Example:
      ```python
      sampler_sync.h2d(sync_req)
      sampler_sync.apply_to_runner(worker.model_runner)
      ```
    """
    if runner is None or (not self.arrays and self._pending_h2d_arrays is None):
      return
    self._commit_pending_arrays(
        lambda target_arrays: weight_utils.apply_to_runner(
            self.job_name,
            self.names,
            target_arrays,
            runner,
            delete_replaced_buffers=True,
        )
    )

  def apply_to_sampler(self, sampler: weight_utils.SyncedSamplerTarget) -> None:
    """Atomically commits staged H2D weights into `sampler` and frees old HBM.

    Updates `sampler.transformer_state` (and `sampler.model_runner` when set)
    with arrays staged by :meth:`h2d` and deletes superseded device buffers.

    Args:
      sampler: Sampler target satisfying `weight_utils.SyncedSamplerTarget`.

    Example:
      ```python
      sampler_sync.h2d(sync_req)
      sampler_sync.apply_to_sampler(rollout_sampler)
      ```
    """
    if not self.arrays and self._pending_h2d_arrays is None:
      return
    self._commit_pending_arrays(
        lambda target_arrays: weight_utils.apply_to_sampler(
            self.job_name,
            self.names,
            target_arrays,
            sampler,
            delete_replaced_buffers=True,
        )
    )

  def _wire_inputs(
      self, arrays: Sequence[Any]
  ) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    """Selects the bound->wire manifest and the `arrays` it projects, by key.

    Peers are matched 1:1 on what crosses the wire, not on each engine's
    private state names. A writer projects through `to_hf_manifest`; a reader
    through the inverse of the first `from_hf_manifest` rule producing each wire
    tensor. Both sides go through `to_hf_impl`, so `tied_keys` targets are
    dropped exactly as `d2h()` drops them on write. Later fan-out rules
    re-derive targets from already-claimed wire tensors (e.g. a tied `lm_head`
    materialized from the embedding), so their targets are excluded.

    Args:
      arrays: Arrays (or shape structs) parallel to `self.names`.

    Returns:
      `(wire_manifest, inputs)`; `wire_manifest` is `None` in identity mode,
      where the bound names are the wire names.
    """
    bound = {
        weight_utils.param_key(name): arr
        for name, arr in zip(self.names, arrays, strict=True)
    }
    if self._to_hf_manifest is not None or self._from_hf_manifest is None:
      return self._to_hf_manifest, bound
    claimed: set[str] = set()
    producers: list[Mapping[str, Any]] = []
    for rule in self._from_hf_manifest["rules"]:
      if claimed.issuperset(rule["source_keys"]):
        for target in rule["target_keys"]:
          bound.pop(target, None)
      else:
        producers.append(rule)
        claimed.update(rule["source_keys"])
    return (
        hf_mapping.invert_manifest(
            {**self._from_hf_manifest, "rules": producers}
        ),
        bound,
    )

  def _wire_tensors(self) -> list[tuple[str, Any]]:
    """Returns `(wire_name, array_or_shape_dtype_struct)` per wire tensor."""
    wire_manifest, bound = self._wire_inputs(self.arrays)
    if wire_manifest is None:
      return list(bound.items())
    abstract = {
        k: jax.ShapeDtypeStruct(arr.shape, arr.dtype)
        for k, arr in bound.items()
    }
    wire = jax.eval_shape(
        functools.partial(hf_mapping.to_hf_impl, wire_manifest), abstract
    )
    return sorted(wire.items())

  def work_unit_metadata(self) -> weight_sync.WorkUnitMetadata:
    """Returns wire-safe `WorkUnitMetadata` with canonical tensor manifest.

    Returns:
      `WorkUnitMetadata` containing unit identity, mesh topology, parameter
      manifest, latest staged `artifact_uri`, and optional `checksums`.

    Raises:
      RuntimeError: If called before :meth:`bind`.

    Example:
      ```python
      meta = trainer_sync.work_unit_metadata()
      handler.register_work_unit(meta)
      ```
    """
    if not self._bound:
      raise RuntimeError(
          f"{self.job_name}: bind() must run before work_unit_metadata()."
      )
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None
    for arr in self.arrays:
      if isinstance(arr, jax.Array) and isinstance(
          arr.sharding, jax.sharding.NamedSharding
      ):
        mesh = arr.sharding.mesh
        break
    if mesh is None:
      mesh_axes, mesh_shape = ("fsdp",), (1,)
    else:
      mesh_axes = tuple(mesh.axis_names)
      mesh_shape = tuple(int(mesh.shape[a]) for a in mesh.axis_names)
    variables = tuple(
        weight_utils.tensor_metadata(name, arr, layer_idx)
        for layer_idx, (name, arr) in enumerate(self._wire_tensors())
    )
    unit = weight_sync.WorkUnitId(
        job_name=self.job_name,
        job_replica_id=str(self.worker_index) if self.worker_index else "",
        data_replica_idx=self.data_replica_idx,
    )
    return weight_sync.WorkUnitMetadata(
        unit=unit,
        mesh_shape=mesh_shape,
        variables=variables,
        mesh_axes=mesh_axes,
        transport_mode=weight_sync.WeightSyncMode.GCS.value,
        use_ffi=False,
        artifact_uri=self._current_artifact_uri,
        checksums=(
            dict(self._current_checksums)
            if self._current_checksums is not None
            else None
        ),
    )

  def work_unit_metadata_all(self) -> list[weight_sync.WorkUnitMetadata]:
    """Returns single-element metadata list for this host.

    Returns:
      One-element list `[self.work_unit_metadata()]`.

    Example:
      ```python
      for meta in trainer_sync.work_unit_metadata_all():
        handler.register_work_unit(meta)
      ```
    """
    return [self.work_unit_metadata()]

  def checksums(self, sample: int | None = 3) -> dict[str, float | int]:
    """Computes L1 checksums over staged (if pending) or active bound arrays.

    Args:
      sample: Number of tensors to sample for lightweight checksum validation,
        or `None` to checksum all bound tensors.

    Returns:
      Dictionary mapping canonical parameter keys to scalar checksum values.

    Example:
      ```python
      sums = trainer_sync.checksums(sample=3)
      ```
    """
    target = (
        self._pending_h2d_arrays
        if self._pending_h2d_arrays is not None
        else self.arrays
    )
    if not target:
      return weight_utils.compute_checksums([], [])
    return weight_utils.compute_checksums(
        self.names, target, sample=sample, canonical_keys=True
    )

  def metrics(self) -> dict[str, Any]:
    """Returns timing, AOT compilation, shard-filter, and export telemetry.

    Returns:
      Dictionary with transport mode, `d2h`/`h2d` counts and latencies, AOT
      kernel count and peak temp HBM, and host shard-filtering counters.

    Example:
      ```python
      stats = sampler_sync.metrics()
      print(stats["last_filtered_shards"], stats["last_total_shards"])
      ```
    """
    compiled = self._compiled_save_plan
    exp = self._last_export_metrics
    return {
        "transport_mode": weight_sync.WeightSyncMode.GCS.value,
        "d2h_count": self._d2h_count,
        "h2d_count": self._h2d_count,
        "last_d2h_secs": self._last_d2h_secs,
        "last_h2d_secs": self._last_h2d_secs,
        "staged_checkpoints": len(self._staged_checkpoints),
        "aot_compile_secs": compiled.compile_time_sec if compiled else 0.0,
        "aot_num_kernels": compiled.num_unique_kernels if compiled else 0,
        "aot_peak_temp_hbm_bytes": (
            compiled.peak_temp_hbm_bytes if compiled else 0
        ),
        "last_total_shards": self._last_total_shards,
        "last_filtered_shards": self._last_filtered_shards,
        "last_export_hbm_secs": exp.hbm_transform_and_route_sec if exp else 0.0,
        "last_export_d2h_secs": exp.d2h_and_contiguous_sec if exp else 0.0,
        "last_export_disk_secs": exp.disk_write_sec if exp else 0.0,
        "last_export_shard_count": len(exp.shard_paths) if exp else 0,
    }

  def release(
      self, sync_request: datatypes.WeightSyncRequest | None = None
  ) -> None:
    """Prunes source-staged checkpoints exceeding `config.max_to_keep`.

    Deletes expired step directories asynchronously on `gs://` (from process 0)
    or synchronously on local filesystems.

    Args:
      sync_request: Optional unused request kept for synchronizer parity.

    Example:
      ```python
      trainer_sync.release()
      ```
    """
    del sync_request
    while len(self._staged_checkpoints) > self.config.max_to_keep:
      oldest = self._staged_checkpoints.pop(0)
      if oldest == self._current_artifact_uri:
        # Stop advertising a checkpoint that is about to be deleted.
        self._current_artifact_uri = None
        self._current_checksums = None
      if jax.process_index() != 0:
        continue
      if oldest.startswith("gs://"):
        if self._prune_executor is None:
          self._prune_executor = futures.ThreadPoolExecutor(max_workers=1)
        self._prune_executor.submit(_prune_checkpoint, oldest)
      else:
        _prune_checkpoint(oldest)

  def abort(
      self, sync_request: datatypes.WeightSyncRequest | None = None
  ) -> None:
    """Discards uncommitted destination H2D buffers without touching storage.

    Args:
      sync_request: Optional unused request kept for synchronizer parity.

    Example:
      ```python
      sampler_sync.abort()
      ```
    """
    del sync_request
    if self._pending_h2d_arrays is not None:
      weight_utils.delete_array_buffers(
          self._pending_h2d_arrays, keep_ids={id(a) for a in self.arrays}
      )
      self._pending_h2d_arrays = None

  def close(self) -> None:
    """Releases pending buffers, read arena, stage pool, AOT caches, and bind.

    Idempotent: unmaps and unlinks the persistent remote `/dev/shm` read arena
    (`ShmRangeReader`), shuts down the persistent :class:`MultiprocStagePool`
    (closing subprocesses and unlinking its `/dev/shm` arena), waits for
    background GCS pruning tasks, drops cached `SavePlan` / AOT executables,
    and unbinds live weight references.

    Example:
      ```python
      try:
        trainer_sync.d2h(req)
      finally:
        trainer_sync.close()
      ```
    """
    try:
      with contextlib.ExitStack() as stack:
        prune_exec = self._prune_executor
        self._prune_executor = None
        if prune_exec is not None:
          stack.callback(prune_exec.shutdown, wait=True)
        stack.callback(self._close_stage_pool, abort=True)
        stack.callback(self._close_read_arena)
        self.abort()
    finally:
      self._save_plan = None
      self._compiled_save_plan = None
      self._plan_sig = None
      self._wire_l1_sums = None
      self._wire_l1_sums_digest = None
      self.names = []
      self.arrays = []
      self._bound = False
      self._current_artifact_uri = None
      self._current_checksums = None

  def __enter__(self) -> GCSSynchronizer:
    return self

  def __exit__(self, *_: object) -> None:
    self.close()


def _prune_checkpoint(path_str: str) -> None:
  """Removes an expired staged checkpoint directory, logging storage errors."""
  try:
    safetensors_saver.remove_tree(epath.Path(path_str))
  except Exception as exc:  # pylint: disable=broad-exception-caught
    logging.warning("Failed to prune old GCS checkpoint %s: %s", path_str, exc)


@runtime_checkable
class GCSVllmWorkerLike(Protocol):
  """Structural contract for a vLLM `TPUWorker` hosting `GCSSynchronizer`.

  Attributes:
    rank: Integer TPU worker rank index.
    model_runner: Runner object exposing live model weights for in-place commit.

  Example:
    ```python
    assert isinstance(tpu_worker, gcs_synchronizer.GCSVllmWorkerLike)
    ```
  """

  rank: int
  model_runner: Any

  def get_weights_state(self) -> Any:
    """Returns the worker's live weight PyTree/NNX state."""
    ...

  def refresh_model_state_leaves(self) -> None:
    """Refreshes cached state-leaf references after weight commit."""
    ...


_WORKER_SYNCHRONIZERS: weakref.WeakKeyDictionary[
    GCSVllmWorkerLike, GCSSynchronizer
] = weakref.WeakKeyDictionary()


def require_gcs_worker_sync(
    worker: GCSVllmWorkerLike, op: str
) -> GCSSynchronizer:
  """Returns the bound `GCSSynchronizer` on `worker`, or fails fast.

  Args:
    worker: vLLM TPU worker previously bound via `tpu_worker_bind_gcs_sync`.
    op: Caller operation name included in error messages.

  Returns:
    Bound `GCSSynchronizer` associated with `worker`.

  Raises:
    RuntimeError: If `worker` has not been bound yet.

  Example:
    ```python
    sync = gcs_synchronizer.require_gcs_worker_sync(worker, "gcs_h2d")
    ```
  """
  if worker not in _WORKER_SYNCHRONIZERS:
    raise RuntimeError(f"bind_gcs_sync must run before {op}.")
  return _WORKER_SYNCHRONIZERS[worker]


def tpu_worker_bind_gcs_sync(
    worker: GCSVllmWorkerLike,
    worker_index: int = 0,
    parallelism: int = 4,
    job_name: str = "rollout",
    *,
    config: GCSSynchronizerConfig | None = None,
    to_hf_manifest: hf_mapping.RuleSpec | None = None,
    from_hf_manifest: hf_mapping.RuleSpec | None = None,
) -> dict[str, Any]:
  """Binds `worker`'s live model weights to a `GCSSynchronizer` instance.

  Creates (or updates on re-bind) the `GCSSynchronizer` attached to `worker`
  and binds `worker.get_weights_state()`.

  Args:
    worker: TPU worker satisfying :class:`GCSVllmWorkerLike`.
    worker_index: Optional explicit worker index (`0` falls back to
      `worker.rank`).
    parallelism: Unused concurrency hint kept for hook signature parity.
    job_name: Logical role name (defaults to `"rollout"`).
    config: Optional :class:`GCSSynchronizerConfig`.
    to_hf_manifest: Optional declarative `Tunix -> HF` export manifest.
    from_hf_manifest: Optional declarative `HF -> Tunix` restore manifest.

  Returns:
    Wire-safe `dict` representation of `WorkUnitMetadata`.

  Raises:
    ValueError: If `worker.get_weights_state()` returns `None` or validation
      fails.

  Example:
    ```python
    meta_dict = gcs_synchronizer.tpu_worker_bind_gcs_sync(
        worker, job_name="rollout", from_hf_manifest=from_hf
    )
    ```
  """
  del parallelism
  state = worker.get_weights_state()
  if state is None:
    raise ValueError(
        "worker.get_weights_state() returned None; cannot bind GCSSynchronizer."
    )
  resolved_idx = worker_index if worker_index != 0 else worker.rank
  existing = _WORKER_SYNCHRONIZERS.get(worker)
  if existing is None:
    sync = GCSSynchronizer(
        job_name=job_name,
        config=config,
        worker_index=resolved_idx,
        to_hf_manifest=to_hf_manifest,
        from_hf_manifest=from_hf_manifest,
    )
    try:
      sync.bind(state)
    except BaseException:
      sync.close()
      raise
    _WORKER_SYNCHRONIZERS[worker] = sync
    weakref.finalize(worker, sync.close)
    return dataclasses.asdict(sync.work_unit_metadata())
  # pylint: disable=protected-access
  prev = (
      existing.job_name,
      existing.worker_index,
      existing.config,
      existing._to_hf_manifest,
      existing._from_hf_manifest,
      existing._wire_l1_sums,
      existing._wire_l1_sums_digest,
  )
  try:
    parsed_to = (
        hf_mapping.parse_manifest(to_hf_manifest)
        if to_hf_manifest is not None
        else existing._to_hf_manifest
    )
    parsed_from = (
        hf_mapping.parse_manifest(from_hf_manifest)
        if from_hf_manifest is not None
        else existing._from_hf_manifest
    )
    existing.job_name = job_name
    existing.worker_index = resolved_idx
    if config is not None:
      existing.config = config
    if (
        parsed_to != existing._to_hf_manifest
        or parsed_from != existing._from_hf_manifest
    ):
      existing._wire_l1_sums = None
      existing._wire_l1_sums_digest = None
    existing._to_hf_manifest = parsed_to
    existing._from_hf_manifest = parsed_from
    existing.bind(state)
  except BaseException:
    (
        existing.job_name,
        existing.worker_index,
        existing.config,
        existing._to_hf_manifest,
        existing._from_hf_manifest,
        existing._wire_l1_sums,
        existing._wire_l1_sums_digest,
    ) = prev
    raise
  # pylint: enable=protected-access
  return dataclasses.asdict(existing.work_unit_metadata())


def tpu_worker_get_gcs_metadata(worker: GCSVllmWorkerLike) -> dict[str, Any]:
  """Returns wire-safe registration metadata for `worker`'s bound weights.

  Args:
    worker: Bound vLLM TPU worker.

  Returns:
    Serialized `WorkUnitMetadata` dictionary.

  Example:
    ```python
    meta_dict = gcs_synchronizer.tpu_worker_get_gcs_metadata(worker)
    ```
  """
  return dataclasses.asdict(
      require_gcs_worker_sync(worker, "get_gcs_metadata").work_unit_metadata()
  )


def tpu_worker_gcs_h2d(
    worker: GCSVllmWorkerLike,
    checkpoint_path: str,
    source_checksums: Mapping[str, float | int] | None = None,
) -> dict[str, float | int]:
  """Stages restored weights on `worker` and verifies checksums pre-commit.

  Args:
    worker: Bound vLLM TPU worker.
    checkpoint_path: Staged `gs://` or local checkpoint directory URI.
    source_checksums: Optional expected L1 checksums from the Trainer.

  Returns:
    Computed checksum dictionary if `verify_weights` is enabled, else `{}`.

  Example:
    ```python
    sums = gcs_synchronizer.tpu_worker_gcs_h2d(
        worker, "gs://my-bucket/ws/step_1_s1", source_checksums=src_sums
    )
    ```
  """
  sync = require_gcs_worker_sync(worker, "gcs_h2d")
  sync.h2d(checkpoint_path=checkpoint_path, source_checksums=source_checksums)
  return sync.checksums() if sync.verify_weights else {}


def tpu_worker_gcs_commit(worker: GCSVllmWorkerLike) -> None:
  """Atomically commits staged weights into `worker.model_runner`.

  Args:
    worker: Bound vLLM TPU worker with pending staged weights from `gcs_h2d`.

  Example:
    ```python
    gcs_synchronizer.tpu_worker_gcs_commit(worker)
    ```
  """
  sync = require_gcs_worker_sync(worker, "gcs_commit")
  sync.apply_to_runner(worker.model_runner)
  worker.refresh_model_state_leaves()


def tpu_worker_gcs_abort(worker: GCSVllmWorkerLike) -> None:
  """Discards uncommitted staged weights on `worker` before KV reallocation.

  Args:
    worker: vLLM TPU worker (no-op if unbound).

  Example:
    ```python
    gcs_synchronizer.tpu_worker_gcs_abort(worker)
    ```
  """
  if worker in _WORKER_SYNCHRONIZERS:
    _WORKER_SYNCHRONIZERS[worker].abort()


def tpu_worker_gcs_metrics(worker: GCSVllmWorkerLike) -> dict[str, Any]:
  """Returns `GCSSynchronizer` transfer metrics for `worker`.

  Args:
    worker: Bound vLLM TPU worker.

  Returns:
    Telemetry dictionary from `GCSSynchronizer.metrics()`.

  Example:
    ```python
    stats = gcs_synchronizer.tpu_worker_gcs_metrics(worker)
    ```
  """
  return require_gcs_worker_sync(worker, "gcs_metrics").metrics()
