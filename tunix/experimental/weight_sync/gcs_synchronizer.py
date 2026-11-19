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

"""Safetensors/Orbax checkpoint-staged worker weight synchronizer."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from concurrent import futures
import dataclasses
import functools
import io
import json
import os
import struct
import time
from typing import Any, Protocol, TypeVar, cast, runtime_checkable
import weakref

from absl import logging
from etils import epath
import jax
from jax.experimental import multihost_utils
import numpy as np
from orbax import checkpoint as ocp
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.weight_sync import weight_sync
from tunix.dist_rl.weight_sync import weight_utils
from tunix.models import hf_mapping
from tunix.models import safetensors_loader
from tunix.utils import sharding_utils

_T = TypeVar("_T")
_VALID_CHECKPOINT_FORMATS = ("safetensors", "orbax")
_DEFAULT_MAX_SHARD_BYTES = 2560 * 1024 * 1024
_MAX_SAFETENSORS_SHARD_BYTES = 4 * 1024 * 1024 * 1024
_ORBAX_MARKERS = (
    "_CHECKPOINT_METADATA",
    "_METADATA",
    "manifest.ocdbt",
    "default",
)
# pylint: disable=protected-access
_NP_TO_SAFETENSORS_DTYPE: Mapping[np.dtype[Any], str] = {
    dt: code for code, dt in safetensors_loader._SAFETENSORS_DTYPES.items()
}
# pylint: enable=protected-access


def run_blocking(fn: Callable[[], _T]) -> _T:
  """Runs blocking checkpoint I/O off any active thread-local event loop."""
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
  """Extracts the single canonical source `artifact_uri` and `checksums`."""
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


def _resolve_restore_source(
    sync_request: datatypes.WeightSyncRequest | None,
    checkpoint_path: str | None,
    source_checksums: Mapping[str, float | int] | None,
) -> tuple[str, Mapping[str, float | int] | None]:
  """Resolves the checkpoint URI and the source checksums to verify against.

  An explicit `checkpoint_path` bypasses `sync_request`; explicit
  `source_checksums` override those published in `sync_request`.

  Args:
    sync_request: Request carrying the source work units' artifact metadata.
    checkpoint_path: Explicit checkpoint URI that bypasses `sync_request`.
    source_checksums: Explicit source checksums overriding published ones.

  Returns:
    `(checkpoint_uri, checksums_or_none)`.
  """
  if checkpoint_path is not None:
    if not checkpoint_path.strip():
      raise ValueError("checkpoint_path must not be empty when provided.")
    return checkpoint_path, source_checksums
  uri, extracted_checksums = extract_source_artifact(sync_request)
  return uri, (
      source_checksums if source_checksums is not None else extracted_checksums
  )


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


def _validate_positive_int(
    value: int, field_name: str, *, max_value: int | None = None
) -> None:
  """Raises `ValueError` if `value` is a bool, `<= 0`, or `> max_value`."""
  if isinstance(value, bool) or value <= 0:
    raise ValueError(
        f"GCSSynchronizerConfig.{field_name} must be a positive int, got"
        f" {value!r}."
    )
  if max_value is not None and value > max_value:
    raise ValueError(
        f"GCSSynchronizerConfig.{field_name} must be <= {max_value}, got"
        f" {value!r}."
    )


def _parse_optional_manifest(
    manifest: hf_mapping.RuleSpec | None,
) -> dict[str, Any] | None:
  """Normalizes an optional `RuleSpec` into a validated manifest dictionary."""
  return hf_mapping.parse_manifest(manifest) if manifest is not None else None


def _apply_host_manifest(
    manifest: Mapping[str, Any],
    tensors: Mapping[str, np.ndarray],
    *,
    drop_tied_targets: bool,
) -> dict[str, np.ndarray]:
  """Executes a normalized `hf_mapping` manifest on host NumPy buffers."""
  effective: Mapping[str, Any] = manifest
  if drop_tied_targets and manifest["tied_keys"]:
    tied_dst = [dst for _, dst in manifest["tied_keys"]]
    effective = {
        **manifest,
        "tied_keys": [],
        "dropped_keys": [*manifest["dropped_keys"], *tied_dst],
    }
  pending: dict[str, Any] = {}
  out = hf_mapping.make_streaming_executor(effective, pending=pending)(tensors)
  if pending:
    raise ValueError(
        "Incomplete manifest rules during weight sync; unconsumed source keys:"
        f" {sorted(pending)!r}"
    )
  return out


def _wire_checksums(
    tensors: Mapping[str, np.ndarray],
) -> dict[str, float | int]:
  """Computes full L1 checksums over host tensors keyed by wire name."""
  return weight_utils.compute_checksums(
      list(tensors), list(tensors.values()), sample=None, canonical_keys=True
  )


def _broadcast_host0_checksums(
    sums: Mapping[str, float | int] | None,
) -> dict[str, float | int]:
  """Shares host 0's wire checksums so every host publishes or verifies them."""
  is_host0 = jax.process_index() == 0
  payload = (
      np.frombuffer(json.dumps(sums).encode("utf-8"), dtype=np.uint8)
      if is_host0
      else np.zeros(0, dtype=np.uint8)
  )
  if jax.process_count() > 1:
    size = multihost_utils.broadcast_one_to_all(
        np.int32(payload.size), is_source=is_host0
    )
    buf = payload if is_host0 else np.zeros(int(size), dtype=np.uint8)
    payload = np.asarray(
        multihost_utils.broadcast_one_to_all(buf, is_source=is_host0)
    )
  return json.loads(payload.tobytes())


def _write_safetensors_shard(
    shard_path: epath.Path,
    shard_items: Sequence[tuple[str, np.ndarray]],
) -> int:
  """Atomically writes one `.safetensors` file via `.tmp` + `replace()`."""
  header: dict[str, Any] = {}
  payloads: list[np.ndarray] = []
  offset = 0
  for key, arr in shard_items:
    np_dt = np.dtype(arr.dtype)
    if np_dt not in _NP_TO_SAFETENSORS_DTYPE:
      raise ValueError(f"Unsupported safetensors dtype {np_dt!r} for {key!r}.")
    c_arr = np.ascontiguousarray(arr)
    if c_arr.dtype.byteorder == ">":
      c_arr = c_arr.byteswap().view(c_arr.dtype.newbyteorder("<"))
    end = offset + int(c_arr.nbytes)
    header[key] = {
        "dtype": _NP_TO_SAFETENSORS_DTYPE[np_dt],
        "shape": [int(d) for d in c_arr.shape],
        "data_offsets": [offset, end],
    }
    payloads.append(c_arr.view(np.uint8))
    offset = end

  raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
  padded_header = raw_header + (b" " * ((-len(raw_header)) % 8))
  tmp_path = shard_path.parent / f"{shard_path.name}.tmp"
  with cast(io.BufferedIOBase, tmp_path.open("wb")) as fh:
    fh.write(struct.pack("<Q", len(padded_header)))
    fh.write(padded_header)
    for raw_bytes in payloads:
      fh.write(raw_bytes.tobytes())
  tmp_path.replace(shard_path)
  return offset


def _partition_safetensors_shards(
    host_tensors: Mapping[str, np.ndarray],
    max_shard_bytes: int,
) -> list[list[tuple[str, np.ndarray]]]:
  """Groups `(key, array)` pairs greedily within `max_shard_bytes`."""
  shards: list[list[tuple[str, np.ndarray]]] = []
  cur_shard: list[tuple[str, np.ndarray]] = []
  cur_bytes = 0
  for key, arr in host_tensors.items():
    nbytes = int(arr.nbytes)
    if nbytes > _MAX_SAFETENSORS_SHARD_BYTES:
      raise ValueError(
          f"Tensor {key!r} ({nbytes} B) exceeds 4 GiB safetensors shard"
          f" ceiling ({_MAX_SAFETENSORS_SHARD_BYTES} B)."
      )
    if cur_shard and cur_bytes + nbytes > max_shard_bytes:
      shards.append(cur_shard)
      cur_shard = []
      cur_bytes = 0
    cur_shard.append((key, arr))
    cur_bytes += nbytes
  if cur_shard or not shards:
    shards.append(cur_shard)
  return shards


def _publish_safetensors_shards(
    ckpt_path: epath.Path,
    shards: Sequence[Sequence[tuple[str, np.ndarray]]],
) -> None:
  """Writes safetensors shard(s) and publishes finalization marker last."""
  if ckpt_path.exists():
    ckpt_path.rmtree(missing_ok=True)
  ckpt_path.mkdir(parents=True, exist_ok=True)
  if len(shards) == 1:
    _write_safetensors_shard(ckpt_path / "model.safetensors", shards[0])
    return
  total_bytes = 0
  weight_map: dict[str, str] = {}
  num_shards = len(shards)
  for idx, items in enumerate(shards, start=1):
    fname = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
    total_bytes += _write_safetensors_shard(ckpt_path / fname, items)
    weight_map.update((key, fname) for key, _ in items)
  index_payload = json.dumps(
      {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
      indent=2,
      sort_keys=True,
  )
  tmp_index = ckpt_path / "model.safetensors.index.json.tmp"
  tmp_index.write_text(index_payload)
  tmp_index.replace(ckpt_path / "model.safetensors.index.json")


def _save_safetensors_checkpoint(
    ckpt_path: epath.Path,
    names: Sequence[str],
    arrays: Sequence[Any],
    *,
    max_shard_bytes: int,
    to_hf_manifest: Mapping[str, Any] | None,
    checksum: bool,
) -> dict[str, float | int] | None:
  """Gathers weights, writes atomic safetensors shard(s), and syncs hosts.

  Args:
    ckpt_path: Checkpoint directory to write.
    names: Bound tensor names, parallel to `arrays`.
    arrays: Bound device arrays to gather and write.
    max_shard_bytes: Upper bound on bytes per safetensors shard.
    to_hf_manifest: Optional manifest converting bound tensors to wire names.
    checksum: Whether to checksum the written wire tensors.

  Returns:
    With `checksum`, L1 checksums of exactly the written wire tensors (from
    host 0, shared with all hosts); else `None`.
  """
  multihost = jax.process_count() > 1
  host_tensors: dict[str, np.ndarray] = {}
  wire_sums: dict[str, float | int] | None = None
  for name, arr in zip(names, arrays, strict=True):
    gathered = (
        multihost_utils.process_allgather(arr, tiled=True)
        if multihost
        else jax.device_get(arr)
    )
    if jax.process_index() == 0:
      host_tensors[weight_utils.param_key(name)] = np.asarray(gathered)

  if jax.process_index() == 0:
    if to_hf_manifest is not None:
      host_tensors = _apply_host_manifest(
          to_hf_manifest, host_tensors, drop_tied_targets=True
      )
    if checksum:
      wire_sums = _wire_checksums(host_tensors)
    shards = _partition_safetensors_shards(host_tensors, max_shard_bytes)
    _publish_safetensors_shards(ckpt_path, shards)

  if multihost:
    multihost_utils.sync_global_devices("gcs_weight_sync:safetensors_d2h")
  return _broadcast_host0_checksums(wire_sums) if checksum else None


def _is_safetensors_checkpoint(
    ckpt_path: epath.Path, default_format: str
) -> bool:
  """Detects safetensors vs. Orbax checkpoint from finalized storage markers."""
  if (ckpt_path / "model.safetensors.index.json").exists() or (
      ckpt_path / "model.safetensors"
  ).exists():
    return True
  if any((ckpt_path / marker).exists() for marker in _ORBAX_MARKERS):
    return False
  return default_format == "safetensors"


def _load_safetensors_shards(
    ckpt_path: epath.Path,
    needed_keys: frozenset[str],
) -> dict[str, np.ndarray]:
  """Reads `needed_keys` across single- or multi-shard safetensors files."""
  index_path = ckpt_path / "model.safetensors.index.json"
  single_path = ckpt_path / "model.safetensors"
  if index_path.exists():
    raw_index = json.loads(index_path.read_text())
    weight_map = raw_index["weight_map"] if "weight_map" in raw_index else {}
    files = [ckpt_path / name for name in dict.fromkeys(weight_map.values())]
  elif single_path.exists():
    files = [single_path]
  else:
    files = sorted(ckpt_path.glob("*.safetensors"))
  if not files:
    raise FileNotFoundError(f"No safetensors shards found in {ckpt_path}.")

  loaded: dict[str, np.ndarray] = {}
  seen_keys: set[str] = set()
  for shard_file in files:
    skipped: list[str] = []
    # pylint: disable=protected-access
    loaded.update(
        safetensors_loader._load_file_tensors(
            shard_file, needed_keys, None, skipped, seen_keys
        )
    )
    # pylint: enable=protected-access
  return loaded


def _read_host0_safetensors(
    ckpt_path: epath.Path,
    keys: Sequence[str],
    from_hf_manifest: Mapping[str, Any] | None,
    *,
    checksum: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float | int] | None]:
  """Loads, checksums (wire names), and optionally converts weights on host 0."""
  needed_keys = frozenset(keys)
  if from_hf_manifest is not None:
    tied_src = frozenset(src for src, _ in from_hf_manifest["tied_keys"])
    needed_keys = (
        needed_keys | hf_mapping.active_source_keys(from_hf_manifest) | tied_src
    )
  host_tensors = _load_safetensors_shards(ckpt_path, needed_keys)
  wire_sums = _wire_checksums(host_tensors) if checksum else None
  if from_hf_manifest is not None:
    host_tensors = _apply_host_manifest(
        from_hf_manifest, host_tensors, drop_tied_targets=False
    )
  missing = [k for k in keys if k not in host_tensors]
  if missing:
    raise KeyError(
        f"Missing {len(missing)} weight tensor(s) in {ckpt_path}:"
        f" {missing[:5]!r}"
    )
  return host_tensors, wire_sums


def _restore_safetensors_checkpoint(
    ckpt_path: epath.Path,
    names: Sequence[str],
    arrays: Sequence[Any],
    *,
    from_hf_manifest: Mapping[str, Any] | None,
    checksum: bool,
) -> tuple[dict[str, jax.Array], dict[str, float | int] | None]:
  """Reads safetensors on host 0, broadcasts over ICI, and shards on targets.

  Args:
    ckpt_path: Checkpoint directory to read.
    names: Bound tensor names, parallel to `arrays`.
    arrays: Bound arrays whose shapes, dtypes, and shardings are restored.
    from_hf_manifest: Optional manifest converting wire tensors to bound names.
    checksum: Whether to checksum the wire tensors as read.

  Returns:
    Restored arrays by `param_key`, plus (with `checksum`) L1 checksums of the
    wire tensors as read, before `from_hf_manifest` is applied.
  """
  keys = [weight_utils.param_key(name) for name in names]
  is_host0 = jax.process_index() == 0
  multihost = jax.process_count() > 1
  host_tensors: dict[str, np.ndarray] = {}
  wire_sums: dict[str, float | int] | None = None
  if is_host0:
    host_tensors, wire_sums = _read_host0_safetensors(
        ckpt_path, keys, from_hf_manifest, checksum=checksum
    )
  local_dev0 = jax.local_devices()[0]
  restored: dict[str, jax.Array] = {}
  for key, arr in zip(keys, arrays, strict=True):
    exp_shape = tuple(int(d) for d in arr.shape)
    exp_dtype = np.dtype(arr.dtype)
    if is_host0:
      src_arr = host_tensors.pop(key)
      if src_arr.shape != exp_shape:
        raise ValueError(
            f"Shape mismatch for {key!r}: checkpoint {src_arr.shape} vs bound"
            f" {exp_shape}."
        )
      host_buf = np.asarray(src_arr, dtype=exp_dtype)
    else:
      host_buf = np.empty(exp_shape, dtype=exp_dtype)
    if multihost:
      host_buf = multihost_utils.broadcast_one_to_all(
          host_buf, is_source=is_host0
      )
    target = arr.sharding if isinstance(arr, jax.Array) else local_dev0
    restored[key] = sharding_utils.place_array_on_target(host_buf, target)
  return restored, (_broadcast_host0_checksums(wire_sums) if checksum else None)


@dataclasses.dataclass(frozen=True)
class GCSSynchronizerConfig:
  """Configuration for Safetensors/Orbax checkpoint-staged weight sync."""

  staging_dir: str | None = None
  max_to_keep: int = 1
  checkpoint_format: str = "safetensors"
  max_shard_bytes: int = _DEFAULT_MAX_SHARD_BYTES
  use_ocdbt: bool = True
  use_zarr3: bool = False
  use_compression: bool = True
  save_concurrent_gb: int = 96
  restore_concurrent_gb: int = 96
  cleanup_on_start: bool = False

  def __post_init__(self) -> None:
    if self.staging_dir is None and "WEIGHT_SYNC_GCS_DIR" in os.environ:
      env_dir = os.environ["WEIGHT_SYNC_GCS_DIR"].strip()
      if env_dir:
        object.__setattr__(self, "staging_dir", env_dir)
    elif self.staging_dir is not None and not self.staging_dir.strip():
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
    _validate_positive_int(
        self.max_shard_bytes,
        "max_shard_bytes",
        max_value=_MAX_SAFETENSORS_SHARD_BYTES,
    )
    _validate_positive_int(self.save_concurrent_gb, "save_concurrent_gb")
    _validate_positive_int(self.restore_concurrent_gb, "restore_concurrent_gb")
    if self.cleanup_on_start and self.staging_dir is None:
      raise ValueError(
          "GCSSynchronizerConfig.cleanup_on_start=True requires staging_dir."
      )


class GCSSynchronizer:
  """Worker-side weight synchronizer staging weights via Safetensors or Orbax.

  `verify_weights` checks transport integrity in the wire namespace: with a
  manifest, both peers checksum exactly the safetensors tensors written/read;
  manifest transform correctness is covered by the wire preflight and tests.
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
    env_verify = (
        os.environ["VERIFY_WEIGHTS"].strip().lower() == "true"
        if "VERIFY_WEIGHTS" in os.environ
        else False
    )
    self._verify_weights: bool = (
        env_verify if verify_weights is None else bool(verify_weights)
    )
    self._to_hf_manifest = _parse_optional_manifest(to_hf_manifest)
    self._from_hf_manifest = _parse_optional_manifest(from_hf_manifest)
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
    self._maybe_cleanup_on_start()

  def _maybe_cleanup_on_start(self) -> None:
    """Removes pre-existing `step_*` staging directories when enabled."""
    if not self.config.cleanup_on_start or self.config.staging_dir is None:
      return
    if jax.process_index() == 0:
      staging_root = epath.Path(self.config.staging_dir)
      step_dirs = (
          [c for c in staging_root.iterdir() if c.name.startswith("step_")]
          if staging_root.exists()
          else ()
      )
      for child in step_dirs:
        child.rmtree(missing_ok=True)
    if jax.process_count() > 1:
      multihost_utils.sync_global_devices("gcs_weight_sync:cleanup_on_start")

  @property
  def bound(self) -> bool:
    return self._bound

  @property
  def active(self) -> bool:
    return self._bound

  @property
  def verify_weights(self) -> bool:
    return self._verify_weights

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

  def bind(self, state: Any) -> None:
    """Binds live model parameters for source saving or destination restore."""
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
    self.abort()
    self.names = keep_names
    self.arrays = keep_arrays
    self._bound = True

  def d2h(
      self, sync_request: datatypes.WeightSyncRequest | None = None
  ) -> None:
    """Saves bound weights to `<staging_dir>/step_<policy_version>_<req_id>`."""
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
    wire_checksums: dict[str, float | int] | None = None
    if self.config.checkpoint_format == "orbax":
      flat_tree = {
          weight_utils.param_key(name): arr
          for name, arr in zip(self.names, self.arrays, strict=True)
      }

      def _save_orbax() -> None:
        with self._checkpointer() as checkpointer:
          checkpointer.save(
              ckpt_path,
              args=ocp.args.PyTreeSave(item=flat_tree),
              force=True,
          )

      run_blocking(_save_orbax)
    else:
      wire_checksums = run_blocking(
          functools.partial(
              _save_safetensors_checkpoint,
              ckpt_path,
              self.names,
              self.arrays,
              max_shard_bytes=self.config.max_shard_bytes,
              to_hf_manifest=self._to_hf_manifest,
              checksum=self._verify_weights
              and self._to_hf_manifest is not None,
          )
      )

    self._last_d2h_secs = time.perf_counter() - t0
    self._d2h_count += 1
    self._current_artifact_uri = ckpt_uri
    if ckpt_uri in self._staged_checkpoints:
      self._staged_checkpoints.remove(ckpt_uri)
    self._staged_checkpoints.append(ckpt_uri)
    if wire_checksums is not None:
      self._current_checksums = wire_checksums
    elif self._verify_weights:
      # Identity mode: the bound names are the wire names.
      self._current_checksums = weight_utils.compute_checksums(
          self.names, self.arrays, sample=None, canonical_keys=True
      )
    else:
      self._current_checksums = None
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
    """Restores checkpoint into staged device buffers and verifies checksums."""
    if not self._bound:
      raise RuntimeError(f"{self.job_name}: bind() must run before h2d().")
    resolved_uri, resolved_checksums = _resolve_restore_source(
        sync_request, checkpoint_path, source_checksums
    )

    self.abort()
    ckpt_path = epath.Path(resolved_uri)
    verify = self._verify_weights and resolved_checksums is not None
    t0 = time.perf_counter()

    def _restore() -> tuple[dict[str, Any], dict[str, float | int] | None]:
      if _is_safetensors_checkpoint(ckpt_path, self.config.checkpoint_format):
        return _restore_safetensors_checkpoint(
            ckpt_path,
            self.names,
            self.arrays,
            from_hf_manifest=self._from_hf_manifest,
            checksum=verify and self._from_hf_manifest is not None,
        )
      item, restore_args = _build_restore_specs(self.names, self.arrays)
      with self._checkpointer() as checkpointer:
        return (
            checkpointer.restore(
                ckpt_path,
                args=ocp.args.PyTreeRestore(
                    item=item, restore_args=restore_args
                ),
            ),
            None,
        )

    restored, wire_checksums = run_blocking(_restore)
    pending_arrays = [
        restored[weight_utils.param_key(name)] for name in self.names
    ]
    jax.block_until_ready(pending_arrays)
    self._last_h2d_secs = time.perf_counter() - t0
    self._h2d_count += 1

    if verify:
      dst_checksums = self._staged_checksums(wire_checksums, pending_arrays)
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

  def _staged_checksums(
      self,
      wire_checksums: dict[str, float | int] | None,
      staged: Sequence[Any],
  ) -> dict[str, float | int]:
    """Returns restore-time wire checksums, else full ones keyed by bound name.

    With a manifest, `h2d()` checksums the wire tensors as read; in identity
    mode the bound names are the wire names.

    Args:
      wire_checksums: Wire-namespace checksums from the restore, if computed.
      staged: Restored arrays parallel to `self.names`.
    """
    if wire_checksums is not None:
      return wire_checksums
    return weight_utils.compute_checksums(
        self.names, staged, sample=None, canonical_keys=True
    )

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
    """Atomically installs staged H2D arrays into `runner` and frees old HBM."""
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
    """Atomically installs staged H2D arrays into `sampler` and frees old HBM."""
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

  def _wire_tensors(self) -> list[tuple[str, Any]]:
    """Projects the bound tensors onto the safetensors wire namespace.

    Peers are matched 1:1 on what crosses the wire, not on each engine's
    private state names. A writer projects through `to_hf_manifest`; a reader
    through the inverse of the first `from_hf_manifest` rule producing each wire
    tensor. Both sides go through `to_hf_impl`, so `tied_keys` targets are
    dropped exactly as `d2h()` drops them on write. Later fan-out rules
    re-derive targets from already-claimed wire tensors (e.g. a tied `lm_head`
    materialized from the embedding), so their targets are not on the wire
    either. Identity mode writes the bound names.

    Returns:
      `(wire_name, array_or_shape_dtype_struct)` pairs, one per wire tensor.
    """
    bound = {
        weight_utils.param_key(name): arr
        for name, arr in zip(self.names, self.arrays, strict=True)
    }
    fan_out_targets: set[str] = set()
    if self._to_hf_manifest is not None:
      wire_manifest = self._to_hf_manifest
    elif self._from_hf_manifest is not None:
      claimed: set[str] = set()
      producers: list[Mapping[str, Any]] = []
      for rule in self._from_hf_manifest["rules"]:
        if claimed.issuperset(rule["source_keys"]):
          fan_out_targets.update(rule["target_keys"])
        else:
          producers.append(rule)
          claimed.update(rule["source_keys"])
      wire_manifest = hf_mapping.invert_manifest(
          {**self._from_hf_manifest, "rules": producers}
      )
    else:
      return list(bound.items())
    abstract = {
        k: jax.ShapeDtypeStruct(arr.shape, arr.dtype)
        for k, arr in bound.items()
        if k not in fan_out_targets
    }
    wire = jax.eval_shape(
        functools.partial(hf_mapping.to_hf_impl, wire_manifest), abstract
    )
    return sorted(wire.items())

  def work_unit_metadata(self) -> weight_sync.WorkUnitMetadata:
    """Returns wire-safe `WorkUnitMetadata` with canonical tensor manifest."""
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
    """Returns single-element metadata list for this host."""
    return [self.work_unit_metadata()]

  def checksums(self, sample: int | None = 3) -> dict[str, float | int]:
    """Computes L1 checksums over staged (if pending) or active bound arrays."""
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
    """Returns timing and staging counters for observability."""
    return {
        "transport_mode": weight_sync.WeightSyncMode.GCS.value,
        "d2h_count": self._d2h_count,
        "h2d_count": self._h2d_count,
        "last_d2h_secs": self._last_d2h_secs,
        "last_h2d_secs": self._last_h2d_secs,
        "staged_checkpoints": len(self._staged_checkpoints),
    }

  def release(
      self, sync_request: datatypes.WeightSyncRequest | None = None
  ) -> None:
    """Prunes source-staged checkpoints exceeding `config.max_to_keep`."""
    del sync_request
    while len(self._staged_checkpoints) > self.config.max_to_keep:
      oldest = self._staged_checkpoints.pop(0)
      if oldest == self._current_artifact_uri:
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
    """Discards uncommitted destination H2D buffers without touching storage."""
    del sync_request
    if self._pending_h2d_arrays is not None:
      weight_utils.delete_array_buffers(
          self._pending_h2d_arrays, keep_ids={id(a) for a in self.arrays}
      )
      self._pending_h2d_arrays = None

  def close(self) -> None:
    """Releases pending buffers, waits on background pruning, and unbinds."""
    self.abort()
    if self._prune_executor is not None:
      self._prune_executor.shutdown(wait=True)
      self._prune_executor = None
    self.names = []
    self.arrays = []
    self._bound = False
    self._current_artifact_uri = None
    self._current_checksums = None


def _prune_checkpoint(path_str: str) -> None:
  """Removes an expired staged checkpoint directory, logging storage errors."""
  try:
    epath.Path(path_str).rmtree(missing_ok=True)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    logging.warning("Failed to prune old GCS checkpoint %s: %s", path_str, exc)


@runtime_checkable
class GCSVllmWorkerLike(Protocol):
  """Structural contract for a vLLM `TPUWorker` hosting `GCSSynchronizer`."""

  rank: int
  model_runner: Any

  def get_weights_state(self) -> Any:
    ...

  def refresh_model_state_leaves(self) -> None:
    ...


_WORKER_SYNCHRONIZERS: weakref.WeakKeyDictionary[
    GCSVllmWorkerLike, GCSSynchronizer
] = weakref.WeakKeyDictionary()


def require_gcs_worker_sync(
    worker: GCSVllmWorkerLike, op: str
) -> GCSSynchronizer:
  """Returns the bound `GCSSynchronizer` on `worker`, or fails fast."""
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
  """Binds `worker`'s live model weights to `GCSSynchronizer`."""
  del parallelism
  state = worker.get_weights_state()
  if state is None:
    raise ValueError(
        "worker.get_weights_state() returned None; cannot bind GCSSynchronizer."
    )
  resolved_idx = worker_index if worker_index != 0 else worker.rank
  if worker not in _WORKER_SYNCHRONIZERS:
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

  sync = _WORKER_SYNCHRONIZERS[worker]
  prev = (
      sync.job_name,
      sync.worker_index,
      sync.config,
      sync._to_hf_manifest,  # pylint: disable=protected-access
      sync._from_hf_manifest,  # pylint: disable=protected-access
  )
  try:
    parsed_to = (
        hf_mapping.parse_manifest(to_hf_manifest)
        if to_hf_manifest is not None
        else sync._to_hf_manifest  # pylint: disable=protected-access
    )
    parsed_from = (
        hf_mapping.parse_manifest(from_hf_manifest)
        if from_hf_manifest is not None
        else sync._from_hf_manifest  # pylint: disable=protected-access
    )
    sync.job_name = job_name
    sync.worker_index = resolved_idx
    if config is not None:
      sync.config = config
    sync._to_hf_manifest = parsed_to  # pylint: disable=protected-access
    sync._from_hf_manifest = parsed_from  # pylint: disable=protected-access
    sync.bind(state)
  except BaseException:
    (
        sync.job_name,
        sync.worker_index,
        sync.config,
        sync._to_hf_manifest,  # pylint: disable=protected-access
        sync._from_hf_manifest,  # pylint: disable=protected-access
    ) = prev
    raise
  return dataclasses.asdict(sync.work_unit_metadata())


def tpu_worker_get_gcs_metadata(worker: GCSVllmWorkerLike) -> dict[str, Any]:
  """Returns wire-safe registration metadata for `worker`'s bound weights."""
  return dataclasses.asdict(
      require_gcs_worker_sync(worker, "get_gcs_metadata").work_unit_metadata()
  )


def tpu_worker_gcs_h2d(
    worker: GCSVllmWorkerLike,
    checkpoint_path: str,
    source_checksums: Mapping[str, float | int] | None = None,
) -> dict[str, float | int]:
  """Stages restored weights on `worker` and verifies checksums pre-commit."""
  sync = require_gcs_worker_sync(worker, "gcs_h2d")
  sync.h2d(checkpoint_path=checkpoint_path, source_checksums=source_checksums)
  return sync.checksums() if sync.verify_weights else {}


def tpu_worker_gcs_commit(worker: GCSVllmWorkerLike) -> None:
  """Atomically commits staged weights into `worker.model_runner`."""
  sync = require_gcs_worker_sync(worker, "gcs_commit")
  sync.apply_to_runner(worker.model_runner)
  worker.refresh_model_state_leaves()


def tpu_worker_gcs_abort(worker: GCSVllmWorkerLike) -> None:
  """Discards uncommitted staged weights on `worker` before KV reallocation."""
  if worker in _WORKER_SYNCHRONIZERS:
    _WORKER_SYNCHRONIZERS[worker].abort()


def tpu_worker_gcs_metrics(worker: GCSVllmWorkerLike) -> dict[str, Any]:
  """Returns `GCSSynchronizer` transfer metrics for `worker`."""
  return require_gcs_worker_sync(worker, "gcs_metrics").metrics()
