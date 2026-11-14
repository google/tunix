# Copyright 2025 Google LLC
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

"""Safetensors writers for sharded, Subtree all-to-all, and LoRA exports.

Covers sharded state export, Subtree all-to-all export, and LoRA-merged
export.
"""

# pylint: disable=g-bare-generic

import collections
from collections.abc import Iterator, Mapping, Sequence
import concurrent.futures
import contextlib
import contextvars
import dataclasses
import functools
import hashlib
import itertools
import json
import math
import mmap
import os
import queue
import re
import shutil
import struct
import subprocess
import threading
import time
from typing import Any, BinaryIO, Callable, Protocol, cast, runtime_checkable

from absl import logging
from etils import epath
from flax import nnx
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import safetensors.flax as safe_flax
from tunix.models import hf_mapping
from tunix.models import safetensors_io
from tunix.models import safetensors_loader
from tunix.models import safetensors_stage_worker
from tunix.utils import sharding_utils

try:
  from pyarrow import fs as _pa_fs  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]
except ImportError:
  _pa_fs = None

gfile: Any = None

# Default 128 MiB HBM wave bucket: keeps 2.5 GiB shard groups at <=20 waves
# (<=32 GCS compose limit), avoids ~1.9 GiB temp-HBM warnings, and halves HBM
# wave latency vs 256 MiB.
DEFAULT_MAX_BUCKET_BYTES = 128 * 1024 * 1024
# Soft cap on bytes per shard file (2.5 GiB), in line with typical multi-GiB
# Hugging Face checkpoint shards; a larger single tensor gets its own shard.
DEFAULT_MAX_SHARD_BYTES = 2560 * 1024 * 1024
# Hard 4 GiB per-shard ceiling matching HF / GCS / /dev/shm large-slot limits.
MAX_SAFETENSORS_SHARD_BYTES = 4 * 1024 * 1024 * 1024
# GCS JSON API compose endpoint caps sourceObjects at 32 components per request.
_MAX_GCS_COMPOSE_PARTS = 32
# Slots larger than 160 MiB (1.25x default bucket; e.g. 311 MB 0.6B or 2.03 GB
# 397B vocab shards) are striped across <=32 parallel >=64 MiB GCS uploads.
_GCS_STRIPE_THRESHOLD_BYTES = 160 * 1024 * 1024
# Minimum 64 MiB GCS stripe size amortizes per-object PUT/compose/DELETE cost.
_GCS_MIN_STRIPE_BYTES = 64 * 1024 * 1024
# 16-byte safetensors header framing (8-byte len prefix + '{}' + <=7-byte pad).
_EST_SAFETENSORS_HEADER_OVERHEAD_BYTES = 16
# Per-key JSON metadata upper bound (dtype, shape, data_offsets, punctuation).
_EST_SAFETENSORS_KEY_HEADER_BYTES = 84

_SAVE_PLAN_CACHE: dict[tuple[Any, ...], 'SavePlan'] = {}
_SHARED_WAVE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=64)
_SHARED_WRITE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=128)


@dataclasses.dataclass(frozen=True)
class ValidationReport:
  """Structured result returned by safetensors config and runtime validators.

  Attributes:
    errors: Fatal configuration or environment violations that will fail export
      or corrupt output (for example, `max_shard_bytes > 4 GiB` or `/dev/shm`
      smaller than `2 * max_bucket_bytes`).
    warnings: Non-fatal misconfigurations that degrade throughput or portability
      (for example, `write_procs <= 1`, single-VIP GCS resolution, or estimated
      shard counts above 1,000).

  Example:
    ```python
    report = safetensors_saver.validate_safetensors_config(
        output_dir='gs://my-bucket/ckpts/step_1',
        max_bucket_bytes=256 << 20,
        max_shard_bytes=2560 << 20,
    )
    if not report.ok:
      report.raise_if_invalid(strict=False, context='export')
    ```
  """

  errors: tuple[str, ...] = ()
  warnings: tuple[str, ...] = ()

  @property
  def ok(self) -> bool:
    """Returns True when zero hard errors were detected."""
    return not self.errors

  def raise_if_invalid(
      self, *, strict: bool = False, context: str = ''
  ) -> None:
    """Logs warnings and raises `ValueError` on errors (or warnings if strict).

    Args:
      strict: When True, promotes non-fatal warnings into `ValueError`.
      context: Optional label prepended to logged warnings and raised errors.

    Raises:
      ValueError: If `errors` is non-empty, or if `strict=True` and `warnings`
        is non-empty.

    Example:
      ```python
      report.raise_if_invalid(strict=True, context='trainer_bind')
      ```
    """
    prefix = f'{context}: ' if context else ''
    for msg in self.warnings:
      logging.warning('%s%s', prefix, msg)
    failures = (*self.errors, *self.warnings) if strict else self.errors
    if failures:
      raise ValueError(f'{prefix}' + '; '.join(failures))


@dataclasses.dataclass(frozen=True)
class TensorExportSpec:
  """Specification for exporting a single model parameter tensor.

  Attributes:
    source_key: Dotted Tunix parameter key inside `flat_state`.
    hf_key: Target HuggingFace safetensors key (or `{expert}` template when
      `unstack_axis` is set).
    shape_2d: Optional target reshape applied after LoRA merge on HBM.
    transpose_axes: Optional permutation applied on HBM before routing.
    unstack_axis: Optional expert axis to split into per-expert HF keys.
    unstack_range: Optional `(start, end)` expert slice for auto-chunked MoE.
    nbytes: Exported tensor payload size in bytes.
    lora_a_key: Optional LoRA-A key in `flat_state` merged on HBM.
    lora_b_key: Optional LoRA-B key in `flat_state` merged on HBM.
    lora_scaling: Scalar LoRA merge multiplier (`alpha / rank`).
    is_replicated: True when the source tensor is replicated across ranks.

  Example:
    ```python
    spec = safetensors_saver.TensorExportSpec(
        source_key='layers.0.mlp.down_proj.w',
        hf_key='model.layers.0.mlp.down_proj.weight',
        transpose_axes=(1, 0),
        nbytes=16 << 20,
    )
    ```
  """

  source_key: str
  hf_key: str
  shape_2d: tuple[int, ...] | None = None
  transpose_axes: tuple[int, ...] | None = None
  unstack_axis: int | None = None
  unstack_range: tuple[int, int] | None = None
  nbytes: int = 0
  lora_a_key: str | None = None
  lora_b_key: str | None = None
  lora_scaling: float = 1.0
  is_replicated: bool = False


@dataclasses.dataclass(frozen=True)
class WaveBucket:
  """One communication and D2H transfer wave across ranks.

  Attributes:
    wave_index: Zero-based wave index within a `SavePlan`.
    rank_assignments: Mapping from owner rank to assigned `TensorExportSpec`s.
    rank_bytes: Mapping from owner rank to total wave payload bytes.
    max_bucket_bytes: Optional per-rank HBM/D2H byte ceiling for this wave.

  Example:
    ```python
    wave = plan.waves[0]
    print(wave.wave_index, len(wave.rank_assignments), wave.rank_bytes)
    ```
  """

  wave_index: int
  rank_assignments: dict[int, tuple[TensorExportSpec, ...]]
  rank_bytes: dict[int, int]
  max_bucket_bytes: int | None = None


@dataclasses.dataclass(frozen=True)
class SavePlan:
  """LPT bin-packed subtree all-to-all export plan.

  Attributes:
    num_ranks: Total device ranks participating in export.
    total_bytes: Total exported payload bytes across all waves and ranks.
    max_bucket_bytes: Per-rank per-wave memory budget in bytes.
    waves: Ordered sequence of `WaveBucket` transfer waves.
    rank_total_bytes: Mapping from rank index to total assigned payload bytes.
    weight_map: HuggingFace `model.safetensors.index.json` key-to-shard map.
    per_wave_shards: Whether shards are emitted per wave/group rather than one
      monolithic file per rank.
    wave_shards: `(rank, wave_index)` shard descriptors when `shard_groups` is
      empty.
    max_shard_bytes: Target physical shard ceiling in bytes (`<= 4 GiB`).
    shard_groups: `(rank, (wave_idx, ...))` multi-wave shard grouping table.

  Example:
    ```python
    plan = safetensors_saver.build_save_plan(specs, num_ranks=len(devices))
    print(len(plan.waves), len(plan.shard_groups), plan.total_bytes)
    ```
  """

  num_ranks: int
  total_bytes: int
  max_bucket_bytes: int | None
  waves: tuple[WaveBucket, ...]
  rank_total_bytes: dict[int, int]
  weight_map: dict[str, str]
  per_wave_shards: bool = False
  wave_shards: tuple[tuple[int, int], ...] = ()
  max_shard_bytes: int | None = None
  shard_groups: tuple[tuple[int, tuple[int, ...]], ...] = ()


@dataclasses.dataclass(frozen=True)
class CompiledSavePlan:
  """AOT-compiled JAX executables and peak HBM stats for a `SavePlan`.

  Built via :func:`compile_save_plan` using abstract `jax.ShapeDtypeStruct`
  lowering (`.lower(...).compile()`), eliminating XLA compilation latency during
  the first checkpoint export or weight-sync step.

  Attributes:
    executables: Mapping from kernel signature keys to compiled
      `jax.stages.Compiled` callables covering unique HBM transform, local-rank
      concat, fused wave, and all-to-all routing shapes.
    num_unique_kernels: Total number of distinct XLA executables compiled
      (typically 3-6 for large MoE models because identical layers deduplicate).
    compile_time_sec: Wall-clock AOT lowering and compilation time in seconds.
    peak_temp_hbm_bytes: Maximum temporary HBM allocation reported by
      `compiled.memory_analysis().temp_size_in_bytes` across compiled kernels (0
      when the backend does not report memory stats).
    rank_mesh: 1D `Mesh(devices, ('rank',))` shared across wave executions.
    rank_shd: `NamedSharding(rank_mesh, PartitionSpec('rank'))` shared across
      wave executions.

  Example:
    ```python
    compiled = safetensors_saver.compile_save_plan(
        flat_state, plan, output_dtype=jnp.bfloat16, devices=devices
    )
    print(compiled.num_unique_kernels, compiled.peak_temp_hbm_bytes)
    ```
  """

  executables: Mapping[tuple[Any, ...], Any]
  num_unique_kernels: int
  compile_time_sec: float
  peak_temp_hbm_bytes: int
  rank_mesh: jax.sharding.Mesh | None = None
  rank_shd: jax.sharding.NamedSharding | None = None


@dataclasses.dataclass(frozen=True)
class SaveMetrics:
  """Execution timing and memory breakdown for a safetensors export.

  Attributes:
    total_bytes: Total tensor bytes written across all shards.
    num_waves: Number of waves executed.
    num_ranks: Number of device ranks participating.
    plan_time_sec: Plan/header setup wall-clock seconds.
    hbm_transform_and_route_sec: On-HBM transform and ICI routing seconds.
    d2h_and_contiguous_sec: Host D2H transfer and slice materialization seconds.
    disk_write_sec: Storage write and flush wall-clock seconds.
    total_time_sec: End-to-end export wall-clock seconds.
    peak_wave_rank_bytes: Peak single-wave per-rank payload size in bytes.
    shard_paths: Tuple of finalized shard paths emitted by this host.

  Example:
    ```python
    metrics = safetensors_saver.export_safetensors_with_plan(
        flat_state, plan, output_dir, devices=devices
    )
    print(metrics.total_time_sec, len(metrics.shard_paths))
    ```
  """

  total_bytes: int
  num_waves: int
  num_ranks: int
  plan_time_sec: float
  hbm_transform_and_route_sec: float
  d2h_and_contiguous_sec: float
  disk_write_sec: float
  total_time_sec: float
  peak_wave_rank_bytes: int
  shard_paths: tuple[str, ...]


join_path = safetensors_loader.path_to_key


def shard_filename(shard_idx: int, total_shards: int) -> str:
  """Returns the canonical HuggingFace safetensors shard filename.

  Args:
    shard_idx: Zero-based shard index (`0 <= shard_idx < total_shards`).
    total_shards: Total number of shards in the checkpoint.

  Returns:
    `'model.safetensors'` for single-shard checkpoints, or
    `'model-XXXXX-of-YYYYY.safetensors'` for sharded checkpoints.

  Example:
    ```python
    name = safetensors_saver.shard_filename(0, 4)
    assert name == 'model-00001-of-00004.safetensors'
    ```
  """
  return (
      'model.safetensors'
      if total_shards <= 1
      else f'model-{shard_idx + 1:05d}-of-{total_shards:05d}.safetensors'
  )


def expand_spec_keys(
    spec: TensorExportSpec, counts: Mapping[str, int]
) -> tuple[str, ...]:
  """Expands HF keys for a `TensorExportSpec` (including stacked MoE experts).

  Args:
    spec: Export specification, optionally containing `unstack_axis` or an
      auto-chunked `unstack_range`.
    counts: Mapping from `source_key` to total expert count along
      `unstack_axis`.

  Returns:
    Ordered tuple of concrete HuggingFace tensor keys produced by `spec`.

  Example:
    ```python
    keys = safetensors_saver.expand_spec_keys(
        spec, {'layers.0.mlp.w': 8}
    )
    ```
  """
  if spec.unstack_range is not None:
    start, end = spec.unstack_range
    return tuple(spec.hf_key.format(expert=i) for i in range(start, end))
  if spec.unstack_axis is None or spec.source_key not in counts:
    return (spec.hf_key,)
  return tuple(
      spec.hf_key.format(expert=i) for i in range(counts[spec.source_key])
  )


def _wave_tensor_key(spec: TensorExportSpec) -> str:
  """Returns a unique per-rank wave dictionary key for `spec`."""
  return (
      spec.hf_key.format(expert=spec.unstack_range[0])
      if spec.unstack_range is not None
      else spec.hf_key
  )


# ==============================================================================
# Layer 3: Pluggable Storage Transport Layer (1P CNS & 3P GCS)
# ==============================================================================

# 48 MiB CNS stripe matches the rs=6.3:sl=48M parallel-open stripe length.
_CNS_PARALLEL_STRIPE_BYTES = 48 << 20
to_gfile_path = safetensors_stage_worker.to_gfile_path
configure_stage_write_flags = (
    safetensors_stage_worker.configure_stage_write_flags
)
upload_gcs_slot_http = safetensors_stage_worker.upload_gcs_slot_http
compose_gcs_objects_http = safetensors_stage_worker.compose_gcs_objects_http
delete_gcs_object_http = safetensors_stage_worker.delete_gcs_object_http
run_stage_worker_stdio = safetensors_stage_worker.run_stage_worker_stdio


@dataclasses.dataclass(frozen=True)
class StorageBackend:
  """Target-aware storage policy and stream opener for Local, GCS, and CNS.

  Attributes:
    kind: Storage class (`'local'`, `'gcs'`, or `'GOOGLE_INTERNAL_STOAGE_PATH'`).
    default_per_wave_shards: Default sharding mode for this backend.
    default_write_threads: Default host thread concurrency for wave flushes.
    default_write_buffer_bytes: Stream chunk size in bytes (`0` for direct).
    requires_bytes_chunks: Whether streams require `bytes` chunks over raw
      `memoryview` writes.
    stage_writer_fn: Optional `/dev/shm` `MultiprocStagePool.open_writer` hook.
    flush_fn: Optional pool flush callback invoked during finalization.
    register_composite_fn: Optional GCS composite registration hook.

  Example:
    ```python
    backend = safetensors_saver.resolve_storage_backend('gs://my-bucket/out')
    backend.makedirs('gs://my-bucket/out')
    with backend.open_writer('gs://my-bucket/out/model.safetensors') as f:
      backend.write_view(f, memoryview(b'payload'))
    backend.flush()
    ```
  """

  kind: str = 'local'
  default_per_wave_shards: bool = True
  default_write_threads: int = 8
  default_write_buffer_bytes: int = 0
  requires_bytes_chunks: bool = False
  stage_writer_fn: Callable[[str], Any] | None = None
  flush_fn: Callable[[], None] | None = None
  register_composite_fn: Callable[[str, Sequence[str]], None] | None = None

  def makedirs(self, path: str) -> None:
    """Creates output directory if required by the storage backend."""
    if self.kind == 'local':
      os.makedirs(path, exist_ok=True)
    elif self.kind == 'GOOGLE_INTERNAL_STOAGE_PATH' and jax.process_index() == 0:
      configure_stage_write_flags()
      epath.Path(path).mkdir(parents=True, exist_ok=True)

  def open_writer(self, path: str) -> Any:
    """Opens a sequential binary output stream for path."""
    if self.stage_writer_fn is not None:
      return self.stage_writer_fn(path)
    if self.kind == 'gcs' and _pa_fs is not None:
      return _pa_fs.GcsFileSystem().open_output_stream(
          path.removeprefix('gs://'),
          buffer_size=self.default_write_buffer_bytes,
      )
    if self.kind in ('gcs', 'GOOGLE_INTERNAL_STOAGE_PATH'):
      if self.kind == 'GOOGLE_INTERNAL_STOAGE_PATH':
        configure_stage_write_flags()
      return epath.Path(path).open('wb')
    return open(path, 'wb', buffering=16 << 20)

  def flush(self) -> None:
    """Flushes any asynchronous background staging pool attached to backend."""
    if self.flush_fn is not None:
      self.flush_fn()

  def register_composite(
      self, dst_path: str, part_paths: Sequence[str]
  ) -> None:
    """Registers GCS parallel composite parts to assemble on `flush()`."""
    if self.register_composite_fn is not None:
      self.register_composite_fn(dst_path, part_paths)

  def write_view(self, f: Any, view: memoryview, chunk_bytes: int = 0) -> None:
    """Writes `view` to `f` as `bytes` chunks (CNS/GCS) or zero-copy `memoryview`."""
    if not self.requires_bytes_chunks:
      f.write(view)
      return
    step = chunk_bytes if chunk_bytes > 0 else self.default_write_buffer_bytes
    if step <= 0:
      step = max(1, len(view))
    for off in range(0, len(view), step):
      f.write(bytes(view[off : off + step]))


_LOCAL_BACKEND = StorageBackend('local', True, 8, 0, False)
_CNS_BACKEND = StorageBackend('GOOGLE_INTERNAL_STOAGE_PATH', True, 64, 8 << 20, True)
_GCS_BACKEND = StorageBackend('gcs', True, 64, 8 << 20, True)
_ACTIVE_CNS_BACKEND: contextvars.ContextVar[StorageBackend | None] = (
    contextvars.ContextVar('_ACTIVE_CNS_BACKEND', default=None)
)
_ACTIVE_GCS_BACKEND: contextvars.ContextVar[StorageBackend | None] = (
    contextvars.ContextVar('_ACTIVE_GCS_BACKEND', default=None)
)
_CNS_CELL_RE = re.compile(r'^[A-Za-z0-9_]+-d$')
_CNS_PARAM_TOK = (
    r'(?:ttl=\d+[dhms]|gc=[01]|rs=\d+\.\d+|r=\d+(?:\.\d+)?|sl=\d+[KMG]?)'
)
_CNS_PARAM_SEG_RE = re.compile(rf'^{_CNS_PARAM_TOK}(?::{_CNS_PARAM_TOK})*$')


def validate_GOOGLE_INTERNAL_STOAGE_PATH_output_dir(output_dir: str) -> None:
  """Validates a `/GOOGLE_INTERNAL_STOAGE_PATH/<cell>-d/<subpath>` directory path before export.

  Rejects malformed CNS cell names, unsafe parameter suffixes, or `%` tokens
  before allocating devices or opening streams.

  Args:
    output_dir: Candidate `/GOOGLE_INTERNAL_STOAGE_PATH/<cell>-d/<subpath>` output path.

  Raises:
    ValueError: If `output_dir` is not a valid `/GOOGLE_INTERNAL_STOAGE_PATH/<cell>-d/<subpath>` path.

  Example:
    ```python
    safetensors_saver.validate_GOOGLE_INTERNAL_STOAGE_PATH_output_dir('/GOOGLE_INTERNAL_STOAGE_PATH/cell-d/home/user/ckpt')
    ```
  """
  if '%' in output_dir:
    raise ValueError(
        f'Invalid CNS output_dir {output_dir!r}: "%" per-file parameter'
        ' suffixes break .tmp atomic rename; use colon-delimited parameter'
        ' directory components (e.g. ttl=1d:gc=0:rs=6.3/<subdir>).'
    )
  parts = output_dir.rstrip('/').split('/')[1:]
  if (
      not output_dir.startswith('/GOOGLE_INTERNAL_STOAGE_PATH/')
      or len(parts) < 3
      or '' in parts
      or not _CNS_CELL_RE.match(parts[1])
  ):
    raise ValueError(
        f'Invalid CNS output_dir {output_dir!r}: must match'
        ' /GOOGLE_INTERNAL_STOAGE_PATH/<cell>-d/<subpath> (e.g. /GOOGLE_INTERNAL_STOAGE_PATH/cell-d/out).'
    )
  for seg in parts[2:]:
    if (':' in seg or '=' in seg) and not _CNS_PARAM_SEG_RE.match(seg):
      raise ValueError(
          f'Invalid CNS parameter directory {seg!r} in {output_dir!r}:'
          ' expected colon-delimited k=v tokens (ttl, gc, rs, r, sl).'
      )
  if ':' in parts[-1] or '=' in parts[-1]:
    raise ValueError(
        f'Invalid CNS output_dir {output_dir!r}: leaf segment {parts[-1]!r}'
        ' is a parameter directory; append a concrete subdirectory name.'
    )


def _is_valid_pos_int(value: Any) -> bool:
  """Returns True when `value` is a positive non-bool integer."""
  return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _check_gcs_auth_and_vips(warnings: list[str]) -> None:
  """Records warnings when GCS OAuth2 credentials or multi-VIP DNS degrade."""
  if not safetensors_io.GsHttpPath.resolve_token():
    warnings.append(
        'GCS OAuth2 token not found via TUNIX_GCS_TOKEN or GCE metadata;'
        ' multi-VIP HTTP transport may fall back or fail.'
    )
  try:
    gcs_vips = safetensors_io.resolve_gcs_vips()
  except OSError:
    gcs_vips = ()
  if len(gcs_vips) <= 1:
    warnings.append(
        'GCS DNS resolved <= 1 VIP; multi-VIP TCP striping is degraded.'
    )


def _check_remote_runtime_config(
    output_dir: str,
    bucket_bytes: int,
    write_procs: int | None,
    shm_dir: str,
    errors: list[str],
    warnings: list[str],
) -> None:
  """Checks `/dev/shm`, worker process count, CNS syntax, and GCS auth/VIPs."""
  is_gcs = output_dir.startswith('gs://')
  is_GOOGLE_INTERNAL_STOAGE_PATH = output_dir == '/GOOGLE_INTERNAL_STOAGE_PATH' or output_dir.startswith('/GOOGLE_INTERNAL_STOAGE_PATH/')
  if is_GOOGLE_INTERNAL_STOAGE_PATH:
    try:
      validate_GOOGLE_INTERNAL_STOAGE_PATH_output_dir(output_dir)
    except ValueError as exc:
      errors.append(str(exc))
  if not (is_gcs or is_GOOGLE_INTERNAL_STOAGE_PATH):
    return
  free_bytes = -1
  if os.path.isdir(shm_dir):
    st = os.statvfs(shm_dir)
    free_bytes = st.f_bavail * st.f_frsize
  if 0 <= free_bytes < 2 * bucket_bytes:
    errors.append(
        f'{shm_dir} free space ({free_bytes >> 20} MiB) is below 2x'
        f' max_bucket_bytes ({(2 * bucket_bytes) >> 20} MiB); mount an'
        ' emptyDir(medium=Memory) volume in GKE/Docker.'
    )
  elif 0 <= free_bytes < 8 * bucket_bytes:
    warnings.append(
        f'{shm_dir} free space ({free_bytes >> 20} MiB) is below 8x'
        f' max_bucket_bytes ({(8 * bucket_bytes) >> 20} MiB); pipeline'
        ' concurrency may throttle.'
    )
  if write_procs is not None and write_procs <= 1:
    warnings.append(
        f'write_procs={write_procs} disables multi-process staging and'
        ' bottlenecks remote uploads on a single Python GIL.'
    )
  if is_gcs:
    _check_gcs_auth_and_vips(warnings)


def _validate_bucket_and_shard_limits(
    max_bucket_bytes: int | None,
    max_shard_bytes: int | None,
    total_bytes: int,
    num_ranks: int,
    errors: list[str],
    warnings: list[str],
) -> int:
  """Validates bucket/shard size bounds and returns an effective bucket size."""
  if not _is_valid_pos_int(num_ranks):
    errors.append(f'num_ranks must be a positive int, got {num_ranks!r}.')
  valid_bucket = max_bucket_bytes is None or _is_valid_pos_int(max_bucket_bytes)
  if not valid_bucket:
    errors.append(
        f'max_bucket_bytes must be a positive int, got {max_bucket_bytes!r}.'
    )
  bucket = (
      max_bucket_bytes
      if (max_bucket_bytes is not None and valid_bucket)
      else DEFAULT_MAX_BUCKET_BYTES
  )
  if max_shard_bytes is None:
    return bucket
  if (
      not _is_valid_pos_int(max_shard_bytes)
      or max_shard_bytes > MAX_SAFETENSORS_SHARD_BYTES
  ):
    errors.append(
        'max_shard_bytes must be in [1,'
        f' {MAX_SAFETENSORS_SHARD_BYTES}] (4 GiB ceiling), got'
        f' {max_shard_bytes!r}.'
    )
    return bucket
  if max_bucket_bytes is not None and valid_bucket and bucket > max_shard_bytes:
    errors.append(
        f'max_bucket_bytes ({bucket}) exceeds max_shard_bytes'
        f' ({max_shard_bytes}).'
    )
  elif total_bytes > 0:
    ranks = num_ranks if _is_valid_pos_int(num_ranks) else 1
    est_shards = max(ranks, math.ceil(total_bytes / max_shard_bytes))
    if est_shards > 1000:
      warnings.append(
          f'Estimated shard count ({est_shards}) exceeds 1,000; increase'
          ' max_shard_bytes (e.g. 2.5 GiB) to reduce metadata/HTTP overhead.'
      )
  return bucket


def validate_safetensors_config(
    output_dir: str | None = None,
    *,
    max_bucket_bytes: int | None = DEFAULT_MAX_BUCKET_BYTES,
    max_shard_bytes: int | None = DEFAULT_MAX_SHARD_BYTES,
    total_bytes: int = 0,
    num_ranks: int = 1,
    write_procs: int | None = None,
    shm_dir: str = '/dev/shm',
    strict: bool = False,
) -> ValidationReport:
  """Validates safetensors export sizing, `/dev/shm` capacity, and remote I/O.

  Performs up-front checks before launching multi-host exports so size limits,
  small Docker `/dev/shm` mounts, single-process GIL bottlenecks, and missing
  GCS auth/VIPs are caught immediately.

  Checks performed:
    1. `max_bucket_bytes >= 1`, `1 <= max_shard_bytes <= 4 GiB`, and
       `max_bucket_bytes <= max_shard_bytes` (hard error if violated).
    2. Estimated shard count `> 1,000` when `total_bytes > 0` (warning).
    3. `/dev/shm` free capacity for `gs://` and `/GOOGLE_INTERNAL_STOAGE_PATH/` outputs: `< 2x`
       `max_bucket_bytes` is a hard error; `< 8x` is a warning.
    4. Explicit `write_procs <= 1` on remote targets (warning).
    5. Valid `/GOOGLE_INTERNAL_STOAGE_PATH/<cell>-d/...` syntax on CNS; OAuth2 token and multi-VIP DNS
       resolution on `gs://`.

  Args:
    output_dir: Destination directory or URI (`gs://...`, `/GOOGLE_INTERNAL_STOAGE_PATH/...`, or local).
      When `None`, only size invariants are checked.
    max_bucket_bytes: Per-rank HBM/D2H wave bucket cap in bytes.
    max_shard_bytes: Maximum `.safetensors` output file size in bytes (hard
      ceiling `MAX_SAFETENSORS_SHARD_BYTES` = 4 GiB).
    total_bytes: Optional total model size in bytes for shard-count estimation.
    num_ranks: Number of participating TPU/GPU ranks (for shard estimation).
    write_procs: Configured `MultiprocStagePool` worker count, if explicitly set
      by the caller.
    shm_dir: Shared-memory mount path to inspect (defaults to `'/dev/shm'`).
    strict: If True, raises `ValueError` whenever any error OR warning occurs.

  Returns:
    A :class:`ValidationReport` containing `errors` and `warnings`.

  Raises:
    ValueError: When `strict=True` and any error or warning is detected.

  Example:
    ```python
    report = safetensors_saver.validate_safetensors_config(
        output_dir='gs://my-bucket/ckpts/step_1',
        max_bucket_bytes=256 << 20,
        max_shard_bytes=2560 << 20,
        total_bytes=786 * (1 << 30),
        num_ranks=64,
        write_procs=48,
    )
    report.raise_if_invalid(strict=False, context='export')
    ```
  """
  errors: list[str] = []
  warnings: list[str] = []
  bucket = _validate_bucket_and_shard_limits(
      max_bucket_bytes,
      max_shard_bytes,
      total_bytes,
      num_ranks,
      errors,
      warnings,
  )
  if output_dir:
    _check_remote_runtime_config(
        output_dir, bucket, write_procs, shm_dir, errors, warnings
    )
  report = ValidationReport(errors=tuple(errors), warnings=tuple(warnings))
  if strict:
    report.raise_if_invalid(strict=True)
  return report


def resolve_storage_backend(output_dir: str) -> StorageBackend:
  """Resolves `StorageBackend` (local POSIX, `gs://` GCS, or `/GOOGLE_INTERNAL_STOAGE_PATH/` CNS).

  Args:
    output_dir: Destination directory path or URI.

  Returns:
    Active `StorageBackend` instance configured for `output_dir`.

  Example:
    ```python
    backend = safetensors_saver.resolve_storage_backend('/tmp/ckpt')
    assert backend.kind == 'local'
    ```
  """
  if output_dir.startswith('gs://'):
    return _ACTIVE_GCS_BACKEND.get() or _GCS_BACKEND
  if output_dir == '/GOOGLE_INTERNAL_STOAGE_PATH' or output_dir.startswith('/GOOGLE_INTERNAL_STOAGE_PATH/'):
    validate_GOOGLE_INTERNAL_STOAGE_PATH_output_dir(output_dir)
    return _ACTIVE_CNS_BACKEND.get() or _CNS_BACKEND
  return _LOCAL_BACKEND


def copy_non_safetensors_files(local_model_path: str, output_dir: str) -> None:
  """Copies non-safetensors metadata and tokenizer files to `output_dir`.

  Args:
    local_model_path: Source directory with config/tokenizer files.
    output_dir: Destination directory (local, `gs://`, or remote).

  Example:
    ```python
    safetensors_saver.copy_non_safetensors_files('/tmp/hf_src', '/tmp/ckpt')
    ```
  """
  backend = resolve_storage_backend(output_dir)
  if (
      backend.kind == 'local'
      and resolve_storage_backend(local_model_path).kind == 'local'
      and os.path.abspath(local_model_path) == os.path.abspath(output_dir)
  ):
    return
  if local_model_path.rstrip('/') == output_dir.rstrip('/'):
    return
  src_root = epath.Path(local_model_path)
  if not src_root.is_dir():
    return
  out_prefix = output_dir.rstrip('/')
  for entry in src_root.iterdir():
    if (
        entry.name.endswith(('.safetensors', '.safetensors.index.json', '.tmp'))
        or not entry.is_file()
    ):
      continue
    dst = f'{out_prefix}/{entry.name}'
    with entry.open('rb') as fsrc, backend.open_writer(dst) as fdst:
      while chunk := fsrc.read(16 << 20):
        backend.write_view(fdst, memoryview(chunk))


# ==============================================================================
# Layer 1: High-Level Streaming Write Pipeline & /dev/shm Ring Arena
# ==============================================================================

# Decouple `/dev/shm` ring capacity (up to 128 GiB so GCS uploads never stall
# `acquire_slot()`) from Python's in-flight host flush queue.
_SHM_RING_MAX_BYTES = 128 * 1024 * 1024 * 1024
# Cap host flush queue at 4 waves so glibc/tcmalloc reuses a hot working set.
_MAX_INFLIGHT_FLUSH_WAVES = 4
# Cap in-flight host flush RAM at 4 GiB to prevent COS 4 KiB page-fault storms.
_MAX_INFLIGHT_FLUSH_BYTES = 4 * 1024 * 1024 * 1024
# Cap abort-path FLUSH drain at 5s so a wedged GCS stream fails fast instead of
# hanging pool teardown or activate() error unwinding.
_ABORT_FLUSH_TIMEOUT_SEC = 5.0
_WORKER_DESYNC_PREFIXES = (
    'worker exited',
    'worker flush timed out',
    'unexpected worker response',
    'worker stdout EOF',
)


def compute_shm_ring_geometry(
    shm_budget_bytes: int = 0,
    max_bucket_bytes: int = DEFAULT_MAX_BUCKET_BYTES,
    num_large_slots: int = 2,
) -> tuple[int, int, int, int]:
  """Derives `/dev/shm` ring geometry for `MultiprocStagePool`.

  Args:
    shm_budget_bytes: Explicit arena byte budget, or `0` to auto-size from free
      `/dev/shm` up to `min(128 GiB, max(32 GiB, free_cap // 3))`.
    max_bucket_bytes: Wave bucket ceiling used to size regular ring slots.
    num_large_slots: Count of oversized slots reserved for unchunked tensors.

  Returns:
    4-tuple `(slot_bytes, num_reg_slots, large_slot_bytes, num_large_slots)`.

  Example:
    ```python
    slot_b, n_reg, lg_b, n_lg = safetensors_saver.compute_shm_ring_geometry(
        shm_budget_bytes=0, max_bucket_bytes=128 << 20
    )
    ```
  """
  # `available_shm_bytes()` already applies the 80% `/dev/shm` safety cap.
  avail = safetensors_io.available_shm_bytes()
  free_cap = max(64 << 20, avail) if avail > 0 else (32 << 30)
  if shm_budget_bytes > 0:
    budget = min(shm_budget_bytes, free_cap) if avail > 0 else shm_budget_bytes
  else:
    large_host_cap = max(32 << 30, free_cap // 3)
    budget = min(free_cap, _SHM_RING_MAX_BYTES, large_host_cap)
  raw_slot = max_bucket_bytes + (max_bucket_bytes >> 4)
  slot_bytes = max(4096, (min(raw_slot, budget // 4) // 4096) * 4096)
  # 32x bucket (4.0 GiB at 128 MiB default, capped at
  # MAX_SAFETENSORS_SHARD_BYTES) fits any valid single shard up to 4 GiB.
  raw_large = min(
      MAX_SAFETENSORS_SHARD_BYTES, max_bucket_bytes * _MAX_GCS_COMPOSE_PARTS
  )
  max_large = budget // max(4, num_large_slots * 2)
  large_slot_bytes = max(slot_bytes, (min(raw_large, max_large) // 4096) * 4096)
  actual_large = (
      num_large_slots
      if num_large_slots > 0 and budget >= 4 * large_slot_bytes
      else 0
  )
  reg_budget = max(slot_bytes, budget - actual_large * large_slot_bytes)
  num_reg_slots = max(1, reg_budget // slot_bytes)
  return slot_bytes, num_reg_slots, large_slot_bytes, actual_large


def slot_offset(
    slot_id: int,
    slot_bytes: int,
    num_reg_slots: int,
    large_slot_bytes: int,
) -> int:
  """Returns byte offset in the `/dev/shm` ring arena for `slot_id`.

  Args:
    slot_id: Slot index (`< num_reg_slots` for regular, else large slot).
    slot_bytes: Regular slot stride in bytes.
    num_reg_slots: Number of regular slots in the arena.
    large_slot_bytes: Oversized slot stride in bytes.

  Returns:
    Starting byte offset of `slot_id` inside `arena.bin`.

  Example:
    ```python
    off = safetensors_saver.slot_offset(
        slot_id=4, slot_bytes=320 << 20, num_reg_slots=4,
        large_slot_bytes=1 << 30,
    )
    ```
  """
  if slot_id < num_reg_slots:
    return slot_id * slot_bytes
  return (
      num_reg_slots * slot_bytes + (slot_id - num_reg_slots) * large_slot_bytes
  )


def _close_parallel_handles(p_handles: Sequence[Any]) -> list[str]:
  """Closes parallel `p`-mode CNS stripe handles with IMMUTABLE mutability."""
  if not p_handles or gfile is None:
    return []
  close_opts = gfile.options_pb2.Options()
  close_opts.close.mutability = gfile.types_pb2.CloseOptions.IMMUTABLE
  errs: list[str] = []
  for pf in p_handles:
    try:
      pf.Close(close_opts)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      errs.append(repr(exc))
  return errs


@runtime_checkable
class _AbortableWriter(Protocol):
  """Structural protocol for stream handles supporting `abort()` and `close()`."""

  def abort(self) -> None:
    ...

  def close(self) -> None:
    ...


class StagedShardWriter:
  """Stages a safetensors shard in a shared pre-faulted `/dev/shm` mmap slot.

  Acquires a regular slot on construction, transparently upgrades to a large
  slot if writes exceed `pool.slot_bytes`, and enqueues the populated slot to
  `MultiprocStagePool` on `close()`.

  Example:
    ```python
    with pool.open_writer('gs://my-bucket/ckpts/model.safetensors') as wf:
      wf.write(header_bytes)
      wf.write(tensor_bytes)
    ```
  """

  def __init__(self, dst_path: str, pool: 'MultiprocStagePool') -> None:
    """Initializes a staged writer targeting `dst_path` on `pool`.

    Args:
      dst_path: Final remote or local destination path for this shard/part.
      pool: Active `MultiprocStagePool` owning the `/dev/shm` arena.
    """
    self._dst_path = dst_path
    self._pool = pool
    self._slot_id = pool.acquire_slot(large=False)
    self._mm = pool.arena_mmap
    self._base = pool.slot_offset(self._slot_id)
    self._cap = pool.slot_bytes
    self._pos = 0
    self._closed = False
    self._aborted = False

  def _upgrade_to_large_slot(self, need: int) -> None:
    """Upgrades from a regular wave slot to a large single-file mmap slot."""
    if need > self._pool.large_slot_bytes or self._pool.num_large_slots <= 0:
      raise ValueError(
          f'Shard {self._dst_path} size {need} exceeds large slot capacity'
          f' {self._pool.large_slot_bytes}'
      )
    large_id = self._pool.acquire_slot(large=True)
    new_base = self._pool.slot_offset(large_id)
    if self._pos > 0:
      self._mm[new_base : new_base + self._pos] = self._mm[
          self._base : self._base + self._pos
      ]
    self._pool.release_slot(self._slot_id)
    self._slot_id = large_id
    self._base = new_base
    self._cap = self._pool.large_slot_bytes

  def write(self, data: bytes | bytearray | memoryview) -> int:
    """Appends `data` into the `/dev/shm` slot, upgrading to a large slot if needed."""
    with memoryview(data) as raw_mv, raw_mv.cast('B') as mv:
      n = len(mv)
      if self._pos + n > self._cap:
        self._upgrade_to_large_slot(self._pos + n)
      off = self._base + self._pos
      if n <= 65536:
        self._mm[off : off + n] = bytes(mv)
      else:
        dst = np.frombuffer(self._mm, dtype=np.uint8, count=n, offset=off)
        dst[:] = np.frombuffer(mv, dtype=np.uint8)
      self._pos += n
      return n

  def abort(self) -> None:
    """Releases the `/dev/shm` slot without enqueueing a partial shard upload."""
    if not self._closed:
      self._closed = True
      self._aborted = True
      self._pool.release_slot(self._slot_id)

  def close(self) -> None:
    """Enqueues the populated `/dev/shm` slot for background remote streaming."""
    if not self._closed:
      self._closed = True
      self._pool.enqueue_slot(self._slot_id, self._pos, self._dst_path)

  def __enter__(self) -> 'StagedShardWriter':
    return self

  def __exit__(self, exc_type: object = None, *_: object) -> None:
    if exc_type is not None or self._aborted:
      self.abort()
    else:
      self.close()


def _hashed_part_uri(dst_uri: str, suffix: str) -> str:
  """Builds a 4-hex-char hash-prefixed ephemeral GCS part URI in the same dir.

  Scatters ~3,000 concurrent temporary `.part.`/`.spart.` uploads across 65,536
  lexicographic prefixes inside `output_dir` to defeat GCS sequential-prefix
  hotspotting while keeping `compose` in the same bucket and cleanup scoped.

  Args:
    dst_uri: Final composed GCS shard URI.
    suffix: Part suffix (for example `'part.0'` or `'spart.0'`).

  Returns:
    Hash-prefixed ephemeral GCS URI inside the same directory as `dst_uri`.
  """
  h4 = hashlib.md5(f'{dst_uri}.{suffix}'.encode('utf-8')).hexdigest()[:4]
  parent_dir, name = dst_uri.rsplit('/', 1)
  return f'{parent_dir}/{h4}.{name}.{suffix}'


def _delete_gcs_parts_direct(part_uris: Sequence[str]) -> None:
  """Best-effort parent-process cleanup when worker subprocesses are down."""
  if not part_uris:
    return
  futs = [
      _SHARED_WRITE_POOL.submit(delete_gcs_object_http, u) for u in part_uris
  ]
  for fut in futs:
    try:
      fut.result()
    except OSError as exc:
      logging.warning('Best-effort GCS part cleanup failed: %s', exc)


class MultiprocStagePool:
  """Multi-process `/dev/shm` ring pool streaming shards to 1P CNS or 3P GCS.

  Eliminates Python GIL and single-socket SSL bottlenecks on remote storage by
  spawning `num_procs` persistent worker subprocesses that share one pre-faulted
  `/dev/shm` ring arena (`mmap`).

  When used as a context manager (`with MultiprocStagePool(...)`) or via
  `activate()`, it installs a context-scoped override for the matching remote
  backend (GCS when `use_gcs_http_write=True`, CNS when
  `use_gcs_http_write=False`) so :func:`export_safetensors_with_plan`
  automatically routes writes through `/dev/shm` slots, and enables streaming
  GCS Parallel Composite Upload (`.part.<k>` -> server-side `compose` -> delete)
  for multi-wave shard groups (`max_shard_bytes`).

  In distributed weight sync (`GCSSynchronizer`), a persistent pool is managed
  automatically across steps—no manual context manager is required. Use
  `with MultiprocStagePool(...)` directly when calling
  :func:`export_safetensors_with_plan` standalone against `gs://` or `/GOOGLE_INTERNAL_STOAGE_PATH/`.

  Runtime prerequisites:
    * `/dev/shm` should have at least `8 * max_bucket_bytes` free (default arena
      auto-sizes up to `min(128 GiB, max(32 GiB, 80% free /dev/shm // 3))`).
    * For 3P `gs://`, workers automatically set `TUNIX_STAGE_BACKEND=gcs_http`
      and stripe keep-alive HTTPS uploads across all discovered GCS VIPs using
      GCE metadata or `TUNIX_GCS_TOKEN`.

  Attributes:
    slot_bytes: Regular `/dev/shm` slot size in bytes.
    num_reg_slots: Count of regular ring slots in the arena.
    large_slot_bytes: Oversized `/dev/shm` slot size in bytes.
    num_large_slots: Count of oversized ring slots in the arena.
    total_arena_bytes: Total allocated `/dev/shm` ring arena size in bytes.
    arena_mmap: Pre-faulted shared `mmap` backing all ring slots.

  Example:
    ```python
    plan = safetensors_saver.build_save_plan(
        specs,
        num_ranks=len(devices),
        max_bucket_bytes=128 << 20,
        max_shard_bytes=2560 << 20,
    )
    with safetensors_saver.MultiprocStagePool(
        stage_dir='/dev/shm/tunix_export_stage',
        num_procs=48,
        max_bucket_bytes=128 << 20,
    ):
      metrics = safetensors_saver.export_safetensors_with_plan(
          flat_state,
          plan,
          'gs://my-bucket/checkpoints/step_100',
          devices=devices,
          async_write=True,
      )
    ```
  """

  def __init__(
      self,
      stage_dir: str,
      num_procs: int = 16,
      shm_budget_bytes: int = 0,
      max_bucket_bytes: int = DEFAULT_MAX_BUCKET_BYTES,
      num_large_slots: int = 2,
      use_gcs_http_write: bool = True,
  ) -> None:
    """Initializes the shared `/dev/shm` arena and spawns worker subprocesses.

    Args:
      stage_dir: Directory (typically under `/dev/shm`) holding `arena.bin`.
      num_procs: Number of persistent worker subprocesses (recommended: 32-48 on
        GCP TPU hosts for `gs://`, 16 on 1P `/GOOGLE_INTERNAL_STOAGE_PATH/`).
      shm_budget_bytes: Explicit ring-arena byte budget, or `0` to auto-size
        from free `/dev/shm` up to 128 GiB.
      max_bucket_bytes: Expected wave payload size used to size regular slots.
      num_large_slots: Number of oversized slots (up to `min(4 GiB, 32x bucket)`
        ) reserved for single unchunked tensors such as vocab embeddings.
      use_gcs_http_write: Sets `TUNIX_STAGE_BACKEND=gcs_http` in worker
        processes so `gs://` slots stream over multi-VIP HTTP/1.1 Keep-Alive.
    """
    self._stage_dir = stage_dir
    self._use_gcs_http_write = use_gcs_http_write
    os.makedirs(stage_dir, exist_ok=True)
    (
        self.slot_bytes,
        self.num_reg_slots,
        self.large_slot_bytes,
        self.num_large_slots,
    ) = compute_shm_ring_geometry(
        shm_budget_bytes, max_bucket_bytes, num_large_slots
    )
    self.total_arena_bytes = (
        self.num_reg_slots * self.slot_bytes
        + self.num_large_slots * self.large_slot_bytes
    )
    self._lock = threading.Lock()
    self._worker_bytes = [0] * num_procs
    self._stdin_locks = [threading.Lock() for _ in range(num_procs)]
    self._slot_refs: dict[int, int] = {}
    self._parallel_handles: list[Any] = []
    self._stripe_composites: list[tuple[str, tuple[str, ...]]] = []
    self._pending_composites: list[tuple[str, tuple[str, ...]]] = []
    self._composite_part_uris: set[str] = set()
    self._dirty_workers: set[int] = set()
    self._worker_err: str | None = None
    self._bg_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    self._bg_futs: list[concurrent.futures.Future[None]] = []
    self._free_slots: queue.SimpleQueue[int] = queue.SimpleQueue()
    for s in range(self.num_reg_slots):
      self._free_slots.put(s)
    self._free_large_slots: queue.SimpleQueue[int] = queue.SimpleQueue()
    for s in range(
        self.num_reg_slots, self.num_reg_slots + self.num_large_slots
    ):
      self._free_large_slots.put(s)
    self._flush_replies: list[queue.SimpleQueue[str]] = [
        queue.SimpleQueue() for _ in range(num_procs)
    ]
    self._backend_tokens: (
        tuple[
            contextvars.Token[StorageBackend | None] | None,
            contextvars.Token[StorageBackend | None] | None,
        ]
        | None
    ) = None
    self._closed = False
    self._closing = False
    self.arena_mmap, self._procs = self._spawn_workers(num_procs)

  def _cleanup_failed_spawn(
      self,
      procs: Sequence[subprocess.Popen[str]],
      arena_mm: mmap.mmap | None,
  ) -> None:
    """Terminates partially spawned workers and releases `/dev/shm` on error."""
    for p in procs:
      safetensors_io.terminate_proc(p)
    self._bg_pool.shutdown(wait=False)
    if arena_mm is not None:
      arena_mm.close()
    shutil.rmtree(self._stage_dir, ignore_errors=True)

  def _await_workers_ready(
      self, procs: Sequence[subprocess.Popen[str]], timeout_sec: float = 30.0
  ) -> None:
    """Waits for worker `READY` lines via `DefaultSelector` and starts watchers."""
    deadline = time.monotonic() + timeout_sec
    for w, p in enumerate(procs):
      safetensors_io.await_worker_ready(
          p,
          deadline,
          error_cls=RuntimeError,
          err_prefix='Stage worker failed startup handshake',
      )
      threading.Thread(
          target=self._watch_stdout,
          args=(p.stdout, self._flush_replies[w]),
          daemon=True,
      ).start()

  def _spawn_workers(
      self, num_procs: int
  ) -> tuple[mmap.mmap, list[subprocess.Popen[str]]]:
    """Spawns persistent worker subprocesses and initializes `/dev/shm` arena."""
    stage_backend = 'gcs_http' if self._use_gcs_http_write else 'gfile'
    flag_args = (
        configure_stage_write_flags()
        if (stage_backend == 'gfile' and gfile is not None)
        else ()
    )
    arena_path = os.path.join(self._stage_dir, 'arena.bin')
    arena_mm: mmap.mmap | None = None
    procs: list[subprocess.Popen[str]] = []
    try:
      arena_mm = safetensors_io.create_shm_mmap(
          arena_path, self.total_arena_bytes, prefault=True
      )
      cmd = safetensors_io.build_worker_cmd(
          safetensors_stage_worker.__file__,
          (
              'from tunix.models import safetensors_stage_worker as w;'
              ' w.run_stage_worker_stdio()'
          ),
          flag_args,
      )
      try:
        proc_idx = int(jax.process_index())
      except (RuntimeError, ValueError):
        proc_idx = 0
      base_env = {
          **os.environ,
          'TUNIX_STAGE_ARENA_PATH': arena_path,
          'TUNIX_STAGE_NUM_WORKERS': str(num_procs),
          'TUNIX_STAGE_PROCESS_IDX': str(proc_idx),
          'TUNIX_STAGE_BACKEND': stage_backend,
          'CUDA_VISIBLE_DEVICES': '',
          'TPU_VISIBLE_CHIPS': '',
          'TPU_CHIPS_PER_HOST_BOUNDS': '',
      }
      if self._use_gcs_http_write:
        base_env['TUNIX_GCS_VIPS'] = ','.join(
            safetensors_io.resolve_gcs_vips()
        )
      else:
        base_env.pop('TUNIX_GCS_VIPS', None)
      procs = [
          subprocess.Popen(
              cmd,
              stdin=subprocess.PIPE,
              stdout=subprocess.PIPE,
              text=True,
              bufsize=1,
              env={**base_env, 'TUNIX_STAGE_WORKER_IDX': str(w)},
          )
          for w in range(num_procs)
      ]
      self._await_workers_ready(procs)
      # Unlink the arena path once all workers have mapped it so SIGKILL never
      # orphans multi-GiB `/dev/shm` files.
      with contextlib.suppress(OSError):
        os.unlink(arena_path)
      return arena_mm, procs
    except BaseException:
      self._cleanup_failed_spawn(procs, arena_mm)
      raise

  def is_healthy(self) -> bool:
    """Returns True iff the pool is open with all worker processes alive."""
    return (
        not self._closed
        and self._worker_err is None
        and bool(self._procs)
        and all(p.poll() is None for p in self._procs)
    )

  def _watch_stdout(
      self,
      stdout: Any,
      flush_q: 'queue.SimpleQueue[str]',
  ) -> None:
    """Drains worker stdout messages and detects unexpected EOF crashes."""
    for raw in stdout:
      line = raw.rstrip('\n')
      if line.startswith('FREE\t'):
        self.release_slot(int(line.split('\t', 1)[1]))
      elif line:
        flush_q.put(line)
    if not self._closed:
      with self._lock:
        if self._worker_err is None:
          self._worker_err = 'worker stdout EOF'
      self._free_slots.put(-1)
      self._free_large_slots.put(-1)
    flush_q.put('ERR\tworker stdout EOF')

  def slot_offset(self, slot_id: int) -> int:
    """Returns the byte offset of `slot_id` inside `arena_mmap`."""
    return slot_offset(
        slot_id, self.slot_bytes, self.num_reg_slots, self.large_slot_bytes
    )

  def acquire_slot(self, large: bool = False) -> int:
    """Acquires a free regular or large slot index, blocking on backpressure."""
    q = self._free_large_slots if large else self._free_slots
    slot_id = q.get()
    if slot_id < 0 or self._worker_err is not None:
      if slot_id < 0:
        q.put(-1)
      raise RuntimeError(
          f'Stage worker exited unexpectedly: {self._worker_err}'
      )
    return slot_id

  def release_slot(self, slot_id: int, count: int = 1) -> None:
    """Decrements `slot_id` reference count and returns it to the free queue."""
    if count <= 0:
      return
    with self._lock:
      rem = (
          self._slot_refs[slot_id] if slot_id in self._slot_refs else 1
      ) - count
      if rem > 0:
        self._slot_refs[slot_id] = rem
        return
      self._slot_refs.pop(slot_id, None)
    q = (
        self._free_large_slots
        if slot_id >= self.num_reg_slots
        else self._free_slots
    )
    q.put(slot_id)

  def _pick_worker(self, nbytes: int) -> int:
    with self._lock:
      w_idx = min(range(len(self._procs)), key=self._worker_bytes.__getitem__)
      self._worker_bytes[w_idx] += nbytes
      return w_idx

  def open_writer(self, dst_path: str) -> StagedShardWriter:
    """Opens a `StagedShardWriter` backed by this `/dev/shm` ring pool."""
    return StagedShardWriter(dst_path, self)

  def _send_slot_cmd(
      self, base: int, nbytes: int, slot_id: int, dst_path: str
  ) -> None:
    w_idx = self._pick_worker(nbytes)
    with self._lock:
      self._dirty_workers.add(w_idx)
    p = self._procs[w_idx]
    with self._stdin_locks[w_idx]:
      assert p.stdin is not None
      p.stdin.write(f'SLOT\t{base}\t{nbytes}\t{slot_id}\t{dst_path}\n')
      p.stdin.flush()

  def _enqueue_gcs_stripes(
      self, slot_id: int, base: int, nbytes: int, dst_path: str
  ) -> None:
    """Stripes a large GCS slot (>160 MiB) across <=32 parallel workers/VIPs."""
    stripe = max(
        _GCS_MIN_STRIPE_BYTES,
        (nbytes + _MAX_GCS_COMPOSE_PARTS - 1) // _MAX_GCS_COMPOSE_PARTS,
    )
    num_stripes = (nbytes + stripe - 1) // stripe
    part_uris = tuple(
        _hashed_part_uri(dst_path, f'spart.{k}') for k in range(num_stripes)
    )
    with self._lock:
      self._slot_refs[slot_id] = num_stripes
      self._stripe_composites.append((dst_path, part_uris))
    dispatched = 0
    try:
      for s_idx, part_uri in enumerate(part_uris):
        s_off = base + s_idx * stripe
        s_len = min(stripe, nbytes - s_idx * stripe)
        self._send_slot_cmd(s_off, s_len, slot_id, part_uri)
        dispatched += 1
    except BaseException:
      self.release_slot(slot_id, num_stripes - dispatched)
      raise

  def _dispatch_parallel_stripes(
      self, slot_id: int, base: int, nbytes: int, GOOGLE_INTERNAL_STOAGE_PATH_path: str
  ) -> None:
    """Opens a parallel CNS handle (`'p'`) and dispatches 48 MiB stripes."""
    del slot_id, base, nbytes, GOOGLE_INTERNAL_STOAGE_PATH_path
    raise NotImplementedError('Parallel stripe writes require 1P gfile.')

  def enqueue_slot(self, slot_id: int, nbytes: int, dst_path: str) -> None:
    """Dispatches a populated `/dev/shm` slot across worker subprocesses."""
    base = self.slot_offset(slot_id)
    if (
        nbytes > self.slot_bytes
        and ':rs=6.3/' in dst_path
        and gfile is not None
    ):
      stripe = _CNS_PARALLEL_STRIPE_BYTES
      num_stripes = (nbytes + stripe - 1) // stripe
      with self._lock:
        self._slot_refs[slot_id] = num_stripes
        self._bg_futs.append(
            self._bg_pool.submit(
                self._dispatch_parallel_stripes,
                slot_id,
                base,
                nbytes,
                dst_path,
            )
        )
      return
    # Observed in 0.6B (311 MB vocab) and 397B (2.034 GB vocab): monolithic
    # single-slot uploads >160 MiB pinned one worker/VIP for ~15s. Split into
    # <=32 parallel >=64 MiB hash-prefixed parts and server-side compose.
    # Skip inner striping for parts belonging to a multi-wave composite so
    # GCS's cumulative 32-component compose limit is never exceeded.
    with self._lock:
      is_composite_part = dst_path in self._composite_part_uris
    if (
        dst_path.startswith('gs://')
        and self._use_gcs_http_write
        and nbytes > _GCS_STRIPE_THRESHOLD_BYTES
        and not is_composite_part
    ):
      self._enqueue_gcs_stripes(slot_id, base, nbytes, dst_path)
      return
    with self._lock:
      self._slot_refs[slot_id] = 1
    try:
      self._send_slot_cmd(base, nbytes, slot_id, dst_path)
    except BaseException:
      self.release_slot(slot_id)
      raise

  def prewarm(self, GOOGLE_INTERNAL_STOAGE_PATH_dir: str, proc_idx: int | None = None) -> None:
    """Pre-warms worker RPC/storage channels with small probe files."""
    proc = jax.process_index() if proc_idx is None else proc_idx
    payload = b'w' * min(65536, self.slot_bytes)
    for i in range(len(self._procs) * 2):
      with self.open_writer(
          os.path.join(GOOGLE_INTERNAL_STOAGE_PATH_dir, f'warm_p{proc}_{i:03d}.bin')
      ) as wf:
        wf.write(payload)
    self.flush()

  def register_gcs_composite(
      self, dst_gs_uri: str, part_gs_uris: Sequence[str]
  ) -> None:
    """Queues a multi-wave GCS shard for compose + part cleanup on `flush()`."""
    parts = tuple(part_gs_uris)
    with self._lock:
      self._pending_composites.append((dst_gs_uri, parts))
      self._composite_part_uris.update(parts)

  def _write_worker_cmd(self, w: int, line: str) -> str | None:
    """Writes one command line to worker `w` stdin, returning error on failure."""
    p = self._procs[w]
    try:
      with self._stdin_locks[w]:
        assert p.stdin is not None
        p.stdin.write(line)
        p.stdin.flush()
      return None
    except (OSError, ValueError) as exc:
      return repr(exc)

  def _await_worker_flush_reply(
      self,
      w: int,
      *,
      abort: bool = False,
      abort_timeout_sec: float = _ABORT_FLUSH_TIMEOUT_SEC,
  ) -> str | None:
    """Reads one worker FLUSH reply without hanging on dead or wedged procs."""
    deadline = (time.monotonic() + abort_timeout_sec) if abort else math.inf
    while True:
      wait_s = min(0.5, max(0.0, deadline - time.monotonic()))
      try:
        resp = self._flush_replies[w].get(block=wait_s > 0.0, timeout=wait_s)
      except queue.Empty:
        if self._procs[w].poll() is not None:
          return 'worker exited during flush'
        if time.monotonic() >= deadline:
          return 'worker flush timed out during abort'
        continue
      if resp.startswith('ERR\t'):
        return resp.split('\t', 1)[1]
      return None if resp == 'DONE' else f'unexpected worker response: {resp!r}'

  def _flush_dirty_workers(self, *, abort: bool = False) -> list[str]:
    """Flushes only workers that received commands since the last flush."""
    with self._lock:
      dirty = sorted(self._dirty_workers)
      self._dirty_workers.clear()
    errs: list[str] = []
    sent: list[int] = []
    fatal_errs: list[str] = []
    for w in dirty:
      write_err = self._write_worker_cmd(w, 'FLUSH\n')
      if write_err is None:
        sent.append(w)
        continue
      errs.append(write_err)
      fatal_errs.append(write_err)
    abort_deadline = time.monotonic() + _ABORT_FLUSH_TIMEOUT_SEC
    for w in sent:
      rem = max(0.0, abort_deadline - time.monotonic())
      reply_err = self._await_worker_flush_reply(
          w, abort=abort, abort_timeout_sec=rem
      )
      if reply_err is None:
        continue
      errs.append(reply_err)
      if reply_err.startswith(_WORKER_DESYNC_PREFIXES):
        fatal_errs.append(reply_err)
    if fatal_errs:
      # Poison the pool and reap dirty workers so subsequent cleanup or
      # persistent `activate()` steps never consume stale `DONE` replies.
      with self._lock:
        if self._worker_err is None:
          self._worker_err = fatal_errs[0]
      self._free_slots.put(-1)
      self._free_large_slots.put(-1)
      for w in dirty:
        safetensors_io.terminate_proc(self._procs[w], timeout=0.1)
    return errs

  def _dispatch_worker_composites(
      self,
      composites: Sequence[tuple[str, tuple[str, ...]]],
      *,
      abort: bool = False,
  ) -> list[str]:
    """Runs GCS compose + pipelined part deletes across warm worker pools."""
    if not composites:
      return []
    n_procs = len(self._procs)
    errs: list[str] = []
    for idx, (dst, parts) in enumerate(composites):
      w = idx % n_procs
      payload = '\t'.join(('COMPOSE', dst, *parts))
      write_err = self._write_worker_cmd(w, f'{payload}\n')
      if write_err is not None:
        errs.append(write_err)
        with self._lock:
          if self._worker_err is None:
            self._worker_err = write_err
      else:
        with self._lock:
          self._dirty_workers.add(w)
    errs.extend(self._flush_dirty_workers(abort=abort))
    return errs

  def _finalize_composite_stage(
      self,
      composites: Sequence[tuple[str, tuple[str, ...]]],
      errs: list[str],
      *,
      abort: bool = False,
  ) -> None:
    """Executes one compose stage, or deletes orphan parts if `errs` is set."""
    if not composites:
      return
    if not errs:
      errs.extend(self._dispatch_worker_composites(composites))
      if not errs:
        return
    if self._worker_err is None:
      if not self._dispatch_worker_composites(
          [('', parts) for _, parts in composites], abort=abort
      ):
        return
    part_uris: list[str] = []
    for _, parts in composites:
      part_uris.extend(parts)
    _delete_gcs_parts_direct(tuple(part_uris))

  def flush(self, *, abort: bool = False) -> None:
    """Waits for all worker slot uploads, GCS compose/cleanup, and CNS handles."""
    with self._lock:
      bg_futs = list(self._bg_futs)
      self._bg_futs.clear()
    errs: list[str] = ['aborted'] if abort else []
    if self._worker_err is not None:
      errs.append(self._worker_err)
    for fut in bg_futs:
      try:
        fut.result()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        errs.append(repr(exc))
    with self._lock:
      for i in range(len(self._worker_bytes)):
        self._worker_bytes[i] = 0
      p_handles = list(self._parallel_handles)
      self._parallel_handles.clear()
      stripe_composites = list(self._stripe_composites)
      self._stripe_composites.clear()
      composites = list(self._pending_composites)
      self._pending_composites.clear()
      self._composite_part_uris.clear()
    errs.extend(self._flush_dirty_workers(abort=abort))
    errs.extend(_close_parallel_handles(p_handles))
    # Finalize large-slot stripes first so any striped wave part exists before
    # outer multi-wave shard groups compose across waves.
    self._finalize_composite_stage(stripe_composites, errs, abort=abort)
    self._finalize_composite_stage(composites, errs, abort=abort)
    if errs and not abort:
      raise RuntimeError(f'Stage worker flush failed: {errs[0]}')

  def _shutdown_workers(self) -> None:
    """Sends QUIT, closes pipes, and reaps all stage workers safely."""
    for w in range(len(self._procs)):
      self._write_worker_cmd(w, 'QUIT\n')
    for p in self._procs:
      safetensors_io.terminate_proc(p)

  def close(self, *, abort: bool = False) -> None:
    """Flushes pending slots, terminates workers, and removes `/dev/shm` arena."""
    if self._closed or self._closing:
      return
    self._closing = True
    try:
      self.flush(abort=abort)
    finally:
      self._closed = True
      try:
        self._bg_pool.shutdown(wait=True)
        self._shutdown_workers()
      finally:
        with contextlib.suppress(BufferError):
          self.arena_mmap.close()
        shutil.rmtree(self._stage_dir, ignore_errors=True)

  def _install_backends(
      self,
  ) -> tuple[
      contextvars.Token[StorageBackend | None] | None,
      contextvars.Token[StorageBackend | None] | None,
  ]:
    """Attaches this pool to the matching context-scoped remote backend."""
    if self._use_gcs_http_write:
      gcs_base = _ACTIVE_GCS_BACKEND.get() or _GCS_BACKEND
      gcs_tok = _ACTIVE_GCS_BACKEND.set(
          dataclasses.replace(
              gcs_base,
              default_write_buffer_bytes=0,
              requires_bytes_chunks=False,
              stage_writer_fn=self.open_writer,
              flush_fn=self.flush,
              register_composite_fn=self.register_gcs_composite,
          )
      )
      return None, gcs_tok
    GOOGLE_INTERNAL_STOAGE_PATH_base = _ACTIVE_CNS_BACKEND.get() or _CNS_BACKEND
    GOOGLE_INTERNAL_STOAGE_PATH_tok = _ACTIVE_CNS_BACKEND.set(
        dataclasses.replace(
            GOOGLE_INTERNAL_STOAGE_PATH_base,
            default_write_buffer_bytes=0,
            requires_bytes_chunks=False,
            stage_writer_fn=self.open_writer,
            flush_fn=self.flush,
        )
    )
    return GOOGLE_INTERNAL_STOAGE_PATH_tok, None

  @contextlib.contextmanager
  def activate(self) -> Iterator['MultiprocStagePool']:
    """Temporarily attaches this persistent pool to its matching remote backend.

    Unlike `with MultiprocStagePool(...)`, exiting `activate()` restores the
    previous backend without terminating worker subprocesses, allowing a
    single warm pool to serve multiple consecutive export steps.
    """
    GOOGLE_INTERNAL_STOAGE_PATH_tok, gcs_tok = self._install_backends()
    try:
      yield self
    except BaseException:
      # Drain dirty workers and delete orphan GCS parts so an aborted export
      # never poisons this persistent pool or composes partial shards later.
      self.flush(abort=True)
      raise
    finally:
      if GOOGLE_INTERNAL_STOAGE_PATH_tok is not None:
        _ACTIVE_CNS_BACKEND.reset(GOOGLE_INTERNAL_STOAGE_PATH_tok)
      if gcs_tok is not None:
        _ACTIVE_GCS_BACKEND.reset(gcs_tok)

  def __enter__(self) -> 'MultiprocStagePool':
    self._backend_tokens = self._install_backends()
    return self

  def __exit__(self, exc_type: object = None, *_: object) -> None:
    try:
      self.close(abort=exc_type is not None)
    finally:
      if self._backend_tokens is not None:
        GOOGLE_INTERNAL_STOAGE_PATH_tok, gcs_tok = self._backend_tokens
        if GOOGLE_INTERNAL_STOAGE_PATH_tok is not None:
          _ACTIVE_CNS_BACKEND.reset(GOOGLE_INTERNAL_STOAGE_PATH_tok)
        if gcs_tok is not None:
          _ACTIVE_GCS_BACKEND.reset(gcs_tok)
        self._backend_tokens = None


@dataclasses.dataclass(frozen=True)
class _DeferredHostSlice:
  """Zero-copy 1D contiguous host slice materialized from packed HBM buffer."""

  buf_1d: np.ndarray
  start: int
  end: int
  shape: tuple[int, ...]

  def __array__(
      self, dtype: np.dtype | None = None, copy: bool | None = None
  ) -> np.ndarray:
    del copy
    arr = self.buf_1d[self.start : self.end].reshape(self.shape)
    return arr.astype(dtype) if dtype is not None else arr


HostTensor = np.ndarray | _DeferredHostSlice


def _write_buffered_views(
    backend: StorageBackend,
    fh: Any,
    buf: bytearray,
    tensors: dict[str, HostTensor],
    limit: int,
) -> None:
  """Streams tensor views into `fh`, coalescing small writes up to `limit`."""
  for raw in _tensor_byte_views(tensors):
    if buf and len(buf) + len(raw) > limit:
      backend.write_view(fh, memoryview(buf), limit)
      buf.clear()
    if len(raw) <= limit:
      buf.extend(raw)
    else:
      backend.write_view(fh, raw, limit)
  if buf:
    backend.write_view(fh, memoryview(buf), limit)


class _SequentialShardGroup:
  """Multi-wave shard stream opened on wave 0 and closed after its final wave."""

  def __init__(
      self,
      backend: StorageBackend,
      path: str,
      header: bytes,
      remaining_waves: int | None,
  ) -> None:
    self.backend = backend
    self.path = path
    self._header = header
    self._remaining = remaining_waves
    self._fh: Any = None

  def write_wave(
      self, tensors: dict[str, HostTensor], write_buffer_bytes: int | None
  ) -> None:
    """Writes one wave's tensors and closes the stream on the final wave."""
    buf = bytearray()
    if self._fh is None:
      self._fh = self.backend.open_writer(self.path)
      buf.extend(self._header)
      self._header = b''
    limit = (
        self.backend.default_write_buffer_bytes
        if write_buffer_bytes is None
        else write_buffer_bytes
    )
    try:
      _write_buffered_views(self.backend, self._fh, buf, tensors, limit)
    except BaseException:
      self.abort()
      raise
    if self._remaining is not None:
      self._remaining -= 1
      if self._remaining <= 0:
        self.close()

  def abort(self) -> None:
    """Aborts or closes the underlying stream on write failure."""
    if self._fh is None:
      return
    fh, self._fh = self._fh, None
    if isinstance(fh, _AbortableWriter):
      fh.abort()
    else:
      fh.close()

  def close(self) -> None:
    """Closes the underlying stream if still open."""
    if self._fh is None:
      return
    fh, self._fh = self._fh, None
    fh.close()


def build_rank_header_bytes(
    rank: int,
    plan: SavePlan,
    flat_state: Mapping[str, jax.Array],
    output_dtype: jnp.dtype | None,
    wave_index: int | None = None,
    wave_indices: Sequence[int] | None = None,
) -> bytes:
  """Builds the 8-byte length prefix and wave-ordered JSON safetensors header.

  Args:
    rank: Device rank owning the target shard.
    plan: Precomputed `SavePlan`.
    flat_state: Flat parameter dictionary supplying tensor shapes and dtypes.
    output_dtype: Target export dtype, or `None` to keep source dtypes.
    wave_index: Optional single wave index for per-wave shards.
    wave_indices: Optional ordered wave indices for multi-wave grouped shards.

  Returns:
    Packed `<Q` header length prefix followed by UTF-8 safetensors JSON header.

  Example:
    ```python
    hdr = safetensors_saver.build_rank_header_bytes(
        rank=0, plan=plan, flat_state=flat_state, output_dtype=jnp.bfloat16
    )
    ```
  """
  header: dict[str, Any] = {}
  offset = 0
  if wave_indices is not None:
    waves = tuple(plan.waves[i] for i in wave_indices)
  elif wave_index is not None:
    waves = (plan.waves[wave_index],)
  else:
    waves = plan.waves
  # Order `data_offsets` in exact wave execution order so local and remote
  # streams (`gs://`, `/GOOGLE_INTERNAL_STOAGE_PATH/`) write strictly sequentially with zero `seek()`.
  specs = list(
      itertools.chain.from_iterable(w.rank_assignments[rank] for w in waves)
  )
  for spec in specs:
    base = flat_state[spec.source_key]
    dtype = jnp.dtype(output_dtype or base.dtype)
    dtype_str = safetensors_io.from_np_dtype(dtype)
    raw_s = spec.shape_2d or base.shape
    ax = spec.transpose_axes or tuple(range(len(raw_s)))
    shape = tuple(map(raw_s.__getitem__, ax))
    unstack = spec.unstack_axis is not None
    counts = {spec.source_key: shape[0]} if unstack else {}
    t_shape = list(shape[1:] if unstack else shape)
    step = math.prod(t_shape) * dtype.itemsize
    for hf_key in expand_spec_keys(spec, counts):
      header[hf_key] = {
          'dtype': dtype_str,
          'shape': t_shape,
          'data_offsets': [offset, offset + step],
      }
      offset += step
  return safetensors_io.format_safetensors_header(header)


def _direct_stream_backend(backend: StorageBackend) -> StorageBackend:
  """Bypasses `/dev/shm` slot staging for streams held open across waves."""
  if backend.kind in ('GOOGLE_INTERNAL_STOAGE_PATH', 'gcs') and backend.stage_writer_fn:
    return dataclasses.replace(
        backend,
        default_write_buffer_bytes=8 << 20,
        requires_bytes_chunks=True,
        stage_writer_fn=None,
    )
  return backend


def _open_shard_group_targets(
    backend: StorageBackend,
    flat_state: Mapping[str, jax.Array],
    plan: SavePlan,
    output_dir: str,
    output_dtype: jnp.dtype | None,
    devs: Sequence[jax.Device],
    multi_host: bool,
    proc: int,
) -> tuple[dict[Any, Any], list[str], int]:
  """Prepares per-wave targets for `plan.shard_groups` (GCS compose or stream)."""
  open_files: dict[Any, Any] = {}
  shard_paths: list[str] = []
  total_shards = len(plan.shard_groups)
  use_gcs_compose = (
      backend.kind == 'gcs' and backend.register_composite_fn is not None
  )
  for shard_idx, (r, w_idxs) in enumerate(plan.shard_groups):
    if multi_host and devs[r % len(devs)].process_index != proc:
      continue
    path = os.path.join(output_dir, shard_filename(shard_idx, total_shards))
    hdr = build_rank_header_bytes(
        r, plan, flat_state, output_dtype, wave_indices=w_idxs
    )
    shard_paths.append(path)
    if len(w_idxs) == 1:
      open_files[(w_idxs[0], r)] = (backend, path, hdr)
    elif use_gcs_compose:
      part_paths = tuple(
          _hashed_part_uri(path, f'part.{k}') for k in range(len(w_idxs))
      )
      backend.register_composite(path, part_paths)
      hdrs = (hdr, *(b'' for _ in range(len(w_idxs) - 1)))
      open_files.update(
          zip(
              ((w_idx, r) for w_idx in w_idxs),
              ((backend, p, h) for p, h in zip(part_paths, hdrs)),
          )
      )
    else:
      # Multi-wave sequential shard groups hold streams open across waves;
      # bypass `/dev/shm` slot staging so remote `/GOOGLE_INTERNAL_STOAGE_PATH/` runs never exhaust
      # `num_large_slots` and deadlock waiting on slots held by open groups.
      group_writer = _SequentialShardGroup(
          _direct_stream_backend(backend), path, hdr, len(w_idxs)
      )
      open_files.update(
          dict.fromkeys(((w_idx, r) for w_idx in w_idxs), group_writer)
      )
  return open_files, shard_paths, total_shards


def open_local_shard_files(
    flat_state: Mapping[str, jax.Array],
    plan: SavePlan,
    output_dir: str,
    output_dtype: jnp.dtype | None,
    devices: Sequence[jax.Device] | None,
) -> tuple[dict[Any, Any], list[str], int]:
  """Opens local rank safetensors files or prepares shard-group / wave targets.

  Args:
    flat_state: Flat model parameter mapping.
    plan: Precomputed `SavePlan` specifying shards and wave groupings.
    output_dir: Destination local, `gs://`, or `/GOOGLE_INTERNAL_STOAGE_PATH/` directory.
    output_dtype: Optional cast dtype for exported tensors.
    devices: Ordered TPU/GPU devices, or `None` for single-host CPU mode.

  Returns:
    3-tuple `(open_files, shard_paths, num_active_ranks)` where `open_files`
    maps rank or `(wave_idx, rank)` keys to open writers or composite targets.

  Example:
    ```python
    open_files, paths, active = safetensors_saver.open_local_shard_files(
        flat_state, plan, '/tmp/ckpt', jnp.bfloat16, devices
    )
    ```
  """
  backend = resolve_storage_backend(output_dir)
  proc = jax.process_index()
  devs = tuple(jax.devices() if devices is None else devices)
  multi_host = bool(devs) and jax.process_count() > 1
  if plan.shard_groups:
    return _open_shard_group_targets(
        backend,
        flat_state,
        plan,
        output_dir,
        output_dtype,
        devs,
        multi_host,
        proc,
    )
  active = sorted({r for r, _ in plan.wave_shards})
  items = (
      plan.wave_shards
      if plan.per_wave_shards
      else tuple((r, None) for r in active)
  )
  open_files: dict[Any, Any] = {}
  shard_paths: list[str] = []
  try:
    for shard_idx, (r, w_idx) in enumerate(items):
      if multi_host and devs[r % len(devs)].process_index != proc:
        continue
      path = os.path.join(output_dir, shard_filename(shard_idx, len(items)))
      hdr = build_rank_header_bytes(r, plan, flat_state, output_dtype, w_idx)
      shard_paths.append(path)
      if w_idx is not None:
        open_files[(w_idx, r)] = (backend, path, hdr)
        continue
      open_files[r] = _SequentialShardGroup(
          _direct_stream_backend(backend), path, hdr, remaining_waves=None
      )
  except BaseException:
    _close_unique_handles(open_files, abort=True)
    raise
  return open_files, shard_paths, len(items)


def _tensor_byte_views(tensors: dict[str, HostTensor]) -> list[memoryview]:
  """Extracts 1D uint8 memoryviews, coalescing contiguous `_DeferredHostSlice`s."""
  views: list[memoryview] = []
  prev: _DeferredHostSlice | None = None
  for key in list(tensors.keys()):
    val = tensors.pop(key)
    if isinstance(val, _DeferredHostSlice):
      if (
          prev is not None
          and prev.buf_1d is val.buf_1d
          and prev.end == val.start
      ):
        prev = _DeferredHostSlice(prev.buf_1d, prev.start, val.end, ())
        views[-1] = memoryview(
            prev.buf_1d[prev.start : prev.end].view(np.uint8)
        )
      else:
        prev = val
        views.append(memoryview(val.buf_1d[val.start : val.end].view(np.uint8)))
    else:
      prev = None
      views.append(
          memoryview(np.ascontiguousarray(val).view(np.uint8).reshape(-1))
      )
  return views


def _write_rank_tensors(
    target: Any,
    tensors: dict[str, HostTensor],
    write_buffer_bytes: int | None = None,
) -> None:
  """Writes one rank's contiguous host tensors using backend stream policy."""
  if isinstance(target, _SequentialShardGroup):
    target.write_wave(tensors, write_buffer_bytes)
    return
  backend, path, hdr = target
  limit = (
      backend.default_write_buffer_bytes
      if write_buffer_bytes is None
      else write_buffer_bytes
  )
  # Use context manager so StagedShardWriter.__exit__ calls abort() on write
  # failure instead of close() enqueueing a truncated shard.
  with backend.open_writer(path) as f:
    _write_buffered_views(backend, f, bytearray(hdr), tensors, limit)


def flush_wave_to_files(
    open_files: Mapping[Any, Any],
    host_wave: dict[int, dict[str, HostTensor]],
    wave_index: int = 0,
    write_buffer_bytes: int = 0,
    max_workers: int = 8,
) -> float:
  """Writes one wave's host tensors via persistent `_SHARED_WRITE_POOL`.

  Args:
    open_files: Mapping returned by `open_local_shard_files`.
    host_wave: Per-rank dictionary of materialized host tensors (consumed).
    wave_index: Zero-based wave index being flushed.
    write_buffer_bytes: Stream write chunk size override (`0` uses default).
    max_workers: Maximum concurrent write threads for this wave.

  Returns:
    Elapsed wall-clock write time in seconds for this wave.

  Example:
    ```python
    write_sec = safetensors_saver.flush_wave_to_files(
        open_files, host_wave, wave_index=0, max_workers=8
    )
    ```
  """
  t0 = time.perf_counter()
  tasks: list[tuple[Any, dict[str, HostTensor]]] = []
  for rank in list(host_wave.keys()):
    tensors = host_wave.pop(rank)
    key = (wave_index, rank) if (wave_index, rank) in open_files else rank
    if key in open_files and tensors:
      tasks.append((open_files[key], tensors))
  if len(tasks) == 1 or max_workers <= 1:
    for tgt, t in tasks:
      _write_rank_tensors(tgt, t, write_buffer_bytes)
  elif len(tasks) > 1:
    futs = [
        _SHARED_WRITE_POOL.submit(
            _write_rank_tensors, tgt, t, write_buffer_bytes
        )
        for tgt, t in tasks
    ]
    for fut in futs:
      fut.result()
  return time.perf_counter() - t0


def _write_index_and_metadata(
    backend: StorageBackend,
    plan: SavePlan,
    output_dir: str,
    local_model_path: str | None,
    num_active_ranks: int,
    total_bytes: int | None = None,
) -> None:
  """Copies non-safetensors metadata and writes `model.safetensors.index.json`."""
  if local_model_path is not None:
    copy_non_safetensors_files(local_model_path, output_dir)
  if num_active_ranks > 1:
    index_payload = {
        'metadata': {
            'total_size': (
                plan.total_bytes if total_bytes is None else total_bytes
            )
        },
        'weight_map': plan.weight_map,
    }
    index_path = os.path.join(output_dir, 'model.safetensors.index.json')
    # Compact JSON cuts 92k-key 397B index serialization (~3x) and upload size
    # while remaining sorted, valid HuggingFace safetensors index JSON.
    index_bytes = json.dumps(
        index_payload, separators=(',', ':'), sort_keys=True
    ).encode('utf-8')
    if backend.kind == 'local' and backend.stage_writer_fn is None:
      tmp_path = f'{index_path}.tmp'
      try:
        with open(tmp_path, 'wb') as f:
          f.write(index_bytes)
        os.replace(tmp_path, index_path)
      except BaseException:
        with contextlib.suppress(OSError):
          os.unlink(tmp_path)
        raise
    else:
      # Context manager ensures StagedShardWriter aborts rather than uploading
      # a partial index JSON on write failure.
      with backend.open_writer(index_path) as f:
        backend.write_view(f, memoryview(index_bytes))
  backend.flush()


def finalize_safetensors_metadata(
    plan: SavePlan,
    output_dir: str,
    local_model_path: str | None,
    num_active_ranks: int,
    *,
    total_bytes: int | None = None,
) -> None:
  """Flushes shard uploads/composites, writes index JSON, and copies metadata.

  Must be called after all waves are flushed. On multi-host runs, propagates
  per-host `backend.flush()` failures across all ranks before `pre_index`,
  writes `model.safetensors.index.json` on process 0, and executes a
  `post_index` barrier with host-0 error broadcast so every process observes a
  committed checkpoint or fails cleanly without deadlocking.

  Args:
    plan: Export `SavePlan` containing `weight_map` and `total_bytes`.
    output_dir: Destination local, `gs://`, or `/GOOGLE_INTERNAL_STOAGE_PATH/` directory.
    local_model_path: Optional local HF directory for non-weight metadata files.
    num_active_ranks: Count of active shards/ranks emitted.
    total_bytes: Optional actual exported payload byte count (overrides
      `plan.total_bytes` when `output_dtype` differs from plan dtype).

  Raises:
    RuntimeError: On multi-host runs when another host fails `backend.flush()`
      or process 0 fails writing `model.safetensors.index.json`.
    Exception: Re-raises the local `backend.flush()` or host-0 write exception
      on the failing process after synchronizing failure status across ranks.

  Example:
    ```python
    safetensors_saver.finalize_safetensors_metadata(
        plan, output_dir, local_model_path=None, num_active_ranks=len(devices)
    )
    ```
  """
  backend = resolve_storage_backend(output_dir)
  if jax.process_count() <= 1:
    backend.flush()
    _write_index_and_metadata(
        backend,
        plan,
        output_dir,
        local_model_path,
        num_active_ranks,
        total_bytes=total_bytes,
    )
    return
  flush_exc: Exception | None = None
  try:
    backend.flush()
  except Exception as exc:  # pylint: disable=broad-exception-caught
    flush_exc = exc
  failed = np.asarray(
      multihost_utils.process_allgather(
          np.int32(1 if flush_exc is not None else 0), tiled=False
      )
  ).reshape(-1)
  if flush_exc is not None:
    raise flush_exc
  if np.any(failed):
    bad_ranks = np.flatnonzero(failed).tolist()
    raise RuntimeError(
        f'safetensors_saver backend.flush() failed on process(es) {bad_ranks}.'
    )
  multihost_utils.sync_global_devices('safetensors_saver:pre_index')
  status: dict[str, str] = {}
  host0_exc: Exception | None = None
  if jax.process_index() == 0:
    try:
      _write_index_and_metadata(
          backend,
          plan,
          output_dir,
          local_model_path,
          num_active_ranks,
          total_bytes=total_bytes,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      host0_exc = exc
      status = {
          'error': f'{type(exc).__name__}: {exc}'[:_MAX_BROADCAST_ERROR_CHARS]
      }
  multihost_utils.sync_global_devices('safetensors_saver:post_index')
  shared = sharding_utils.broadcast_host0_json(status)
  if host0_exc is not None:
    raise host0_exc
  if 'error' in shared:
    raise RuntimeError(
        f'finalize_safetensors_metadata failed on process 0: {shared["error"]}'
    )


def submit_and_drain_wave_flush(
    pending: list[tuple[concurrent.futures.Future[float], int]],
    limit: int,
    open_files: Mapping[Any, Any],
    host_wave: dict[int, dict[str, HostTensor]],
    wave: WaveBucket,
    buf_bytes: int,
    threads: int,
) -> float:
  """Drains older flushes down to in-flight bounds, then submits `wave`.

  Args:
    pending: Mutable queue of `(future, wave_bytes)` pairs currently writing.
    limit: Maximum concurrent in-flight wave flushes allowed (`>= 1`).
    open_files: Target handles from `open_local_shard_files`.
    host_wave: Materialized host tensors for `wave`.
    wave: `WaveBucket` being flushed.
    buf_bytes: Stream write buffer size in bytes.
    threads: Per-wave write thread concurrency.

  Returns:
    Cumulative write seconds drained from completed older waves.

  Example:
    ```python
    pending = []
    drained_sec = safetensors_saver.submit_and_drain_wave_flush(
        pending, 4, open_files, host_wave, plan.waves[0], 0, 8
    )
    ```
  """
  if not host_wave:
    return 0.0
  if limit <= 0:
    return flush_wave_to_files(
        open_files, host_wave, wave.wave_index, buf_bytes, threads
    )
  host_bytes = sum(wave.rank_bytes[r] for r in host_wave)
  t_wait = time.perf_counter()
  while pending and (
      len(pending) >= limit
      or host_bytes + sum(b for _, b in pending) > _MAX_INFLIGHT_FLUSH_BYTES
  ):
    pending.pop(0)[0].result()
  waited = time.perf_counter() - t_wait
  pending.append((
      _SHARED_WAVE_POOL.submit(
          flush_wave_to_files,
          open_files,
          host_wave,
          wave.wave_index,
          buf_bytes,
          threads,
      ),
      host_bytes,
  ))
  return waited


# ==============================================================================
# Layer 2: Accelerator & SPMD Wave Execution Layer (TPU / PJRT + RuleSpec)
# ==============================================================================


def infer_is_replicated(arr: jax.Array | jax.ShapeDtypeStruct) -> bool:
  """Infers whether `arr` is replicated across its sharding or a 1D vector.

  Args:
    arr: JAX array or abstract `ShapeDtypeStruct`.

  Returns:
    True if `arr` is 1D small/unsharded or fully replicated across its mesh.

  Example:
    ```python
    is_rep = safetensors_saver.infer_is_replicated(flat_state['norm.scale'])
    ```
  """
  shd = arr.sharding
  if shd is None:
    return True
  if isinstance(shd, jax.sharding.NamedSharding):
    return not any(shd.spec)
  if isinstance(shd, jax.sharding.Sharding) and shd.is_fully_replicated:
    return True
  num_devs = max(len(shd.device_set), 1)
  if isinstance(arr, jax.Array):
    shards = arr.addressable_shards
    return num_devs <= 1 or (bool(shards) and shards[0].data.shape == arr.shape)
  return num_devs <= 1


def extract_flat_state_dict(model: nnx.Module) -> dict[str, jax.Array]:
  """Extracts a flat path-to-array dictionary from an NNX model.

  Excludes non-parameter leaves (such as RNG/batch states) and joins tuple
  paths using `join_path`.

  Args:
    model: Flax NNX module instance whose parameters will be exported.

  Returns:
    Dictionary mapping dotted parameter keys to `jax.Array` values.

  Example:
    ```python
    flat_state = safetensors_saver.extract_flat_state_dict(model)
    ```
  """
  return {
      join_path(path): jnp.asarray(variable.value)
      for path, variable in nnx.to_flat_state(
          nnx.state(model, (nnx.LoRAParam, nnx.Param))
      )
  }


def make_tensor_export_spec(
    flat_state: Mapping[str, jax.Array | jax.ShapeDtypeStruct],
    source_key: str,
    hf_key: str,
    *,
    shape_2d: Sequence[int] | None = None,
    transpose_axes: Sequence[int] | None = None,
    unstack_axis: int | None = None,
    lora_scaling: float = 1.0,
    itemsize: int | None = None,
) -> TensorExportSpec:
  """Constructs a `TensorExportSpec` from `flat_state[source_key]`.

  Automatically discovers matching `<source_key>_lora_a` / `_lora_b` leaves
  and populates byte size and replication metadata.

  Args:
    flat_state: Flat parameter dictionary (arrays or `ShapeDtypeStruct`s).
    source_key: Dotted source key inside `flat_state`.
    hf_key: Target HuggingFace key (or `{expert}` template when unstacking).
    shape_2d: Optional target reshape dimensions on HBM.
    transpose_axes: Optional axis permutation on HBM.
    unstack_axis: Optional expert axis to unstack during export.
    lora_scaling: LoRA merge factor (`alpha / rank`).
    itemsize: Override element byte width (e.g. `2` for bfloat16 output).

  Returns:
    Populated `TensorExportSpec` ready for `build_save_plan`.

  Example:
    ```python
    spec = safetensors_saver.make_tensor_export_spec(
        flat_state,
        source_key='layers.0.mlp.down_proj.w',
        hf_key='model.layers.0.mlp.down_proj.weight',
        transpose_axes=(1, 0),
        itemsize=2,
    )
    ```
  """
  arr = flat_state[source_key]
  lora_a_key, lora_b_key = f'{source_key}_lora_a', f'{source_key}_lora_b'
  has_lora = lora_a_key in flat_state and lora_b_key in flat_state
  return TensorExportSpec(
      source_key=source_key,
      hf_key=hf_key,
      shape_2d=tuple(shape_2d) if shape_2d is not None else None,
      transpose_axes=(
          tuple(transpose_axes) if transpose_axes is not None else None
      ),
      unstack_axis=unstack_axis,
      nbytes=int(math.prod(arr.shape)) * (itemsize or arr.dtype.itemsize),
      lora_a_key=lora_a_key if has_lora else None,
      lora_b_key=lora_b_key if has_lora else None,
      lora_scaling=lora_scaling,
      is_replicated=infer_is_replicated(arr),
  )


def _extract_expert_key_template(target_keys: Sequence[str]) -> str:
  """Infers a `{expert}` format template from indexed unstacked target keys."""
  if len(target_keys) <= 1:
    return target_keys[0]
  # Compare indices 0 and 1 (never 0 and -1: for 11 experts, '0' and '10' both
  # end in '0', which would falsely eat '0' into the suffix).
  first, second = target_keys[0], target_keys[1]
  pre_len = 0
  while (
      pre_len < len(first)
      and pre_len < len(second)
      and first[pre_len] == second[pre_len]
  ):
    pre_len += 1
  suf_len = 0
  max_suf = min(len(first), len(second)) - pre_len
  while (
      suf_len < max_suf
      and first[len(first) - 1 - suf_len] == second[len(second) - 1 - suf_len]
  ):
    suf_len += 1
  prefix = first[:pre_len]
  suffix = first[len(first) - suf_len :] if suf_len else ''
  if not all(k == f'{prefix}{i}{suffix}' for i, k in enumerate(target_keys)):
    raise ValueError(
        'Cannot infer {expert} template for non-contiguous unstack targets:'
        f' {target_keys[:3]!r}...'
    )
  return f'{prefix}{{expert}}{suffix}'


_SUPPORTED_MANIFEST_EXPORT_OPS = frozenset({'reshape', 'transpose', 'unstack'})


def _parse_export_rule_ops(
    src_key: str, rule_ops: Sequence[Mapping[str, Any]]
) -> tuple[Sequence[int] | None, Sequence[int] | None, int | None]:
  """Validates and extracts `(reshape, transpose, unstack)` ops for HBM export."""
  ops: dict[str, Any] = {
      'reshape': {'dst_shape': None},
      'unstack': {'axis': None},
      'transpose': {'axes': None},
  }
  seen_ops: set[str] = set()
  post_unstack_t = False
  flat_ops = itertools.chain.from_iterable(step.items() for step in rule_ops)
  for op_name, op_cfg in flat_ops:
    if op_name not in _SUPPORTED_MANIFEST_EXPORT_OPS - seen_ops:
      raise ValueError(
          f'Unsupported or duplicate manifest export op {op_name!r} for'
          f' {src_key!r}; each of reshape/transpose/unstack may appear once.'
      )
    if op_name == 'reshape' and seen_ops & {'transpose', 'unstack'}:
      raise ValueError(
          f'Manifest rule for {src_key!r} places reshape after transpose or'
          ' unstack, which HBM export cannot execute in order.'
      )
    if op_name == 'unstack' and int(op_cfg['axis']) != 0:
      raise ValueError(
          f'Manifest rule for {src_key!r} unstacks axis {op_cfg["axis"]!r};'
          ' only axis=0 is supported.'
      )
    post_unstack_t |= op_name == 'transpose' and 'unstack' in seen_ops
    seen_ops.add(op_name)
    ops[op_name] = op_cfg
  axes = ops['transpose']['axes']
  unstack_axis = ops['unstack']['axis']
  # Lift 2D post-unstack transpose axes into 3D HBM coordinates (unstack is
  # validated to axis=0 above; pre-unstack transposes are already 3D).
  if post_unstack_t and axes is not None:
    axes = (0, *(int(a) + (1 if int(a) >= 0 else 0) for a in axes))
  return ops['reshape']['dst_shape'], axes, unstack_axis


def build_export_specs_from_manifest(
    flat_state: Mapping[str, jax.Array | jax.ShapeDtypeStruct],
    to_hf_manifest: Mapping[str, Any],
    *,
    lora_scaling: float = 1.0,
    output_dtype: jnp.dtype | None = None,
) -> tuple[list[TensorExportSpec], dict[str, int]]:
  """Builds `TensorExportSpec`s strictly from a declarative `Tunix -> HF` manifest.

  Treats `to_hf_manifest` (built via `hf_mapping.invert_manifest(...)` or
  model-specific helpers such as `qwen3.params.build_to_hf_manifest(cfg)`) as
  the single source of truth for tensor renaming, reshapes, transposes, and MoE
  expert unstacking—applying zero shape heuristics.

  Compare with Identity export (when no manifest is supplied): Identity mode
  preserves Tunix parameter names and layouts verbatim (`transpose_axes=None`),
  whereas `build_export_specs_from_manifest` converts weights into canonical
  HuggingFace layout consumable by vLLM, SGLang, HuggingFace Transformers, or
  any Tunix sampler loading via the matching `from_hf_manifest`.

  Args:
    flat_state: Mapping from dot-delimited Tunix parameter keys (e.g.
      `'layers.0.mlp.gate_proj'`) to `jax.Array` or `jax.ShapeDtypeStruct`.
    to_hf_manifest: Declarative `Tunix -> HF` PrimOp manifest dict (`RuleSpec`)
      containing `'rules'`.
    lora_scaling: Multiplier `alpha / rank` applied when `<key>_lora_a` and
      `<key>_lora_b` exist in `flat_state`.
    output_dtype: Optional target cast dtype (e.g. `jnp.bfloat16`) used to size
      each exported tensor accurately.

  Returns:
    Tuple `(specs, unstack_counts)` ready for :func:`build_save_plan`:
      * `specs`: List of :class:`TensorExportSpec` entries matching every rule
        whose `source_keys[0]` is present in `flat_state`.
      * `unstack_counts`: Mapping `{source_key: num_experts}` for every stacked
        3D MoE tensor unstacked along axis 0.

  Raises:
    ValueError: If an unstacked rule's target keys do not follow a contiguous
      `0..N-1` indexed naming pattern, or if a rule uses unsupported ops.

  Example:
    ```python
    from tunix.models import safetensors_saver
    from tunix.models.qwen3 import params as qwen3_params

    flat_state = safetensors_saver.extract_flat_state_dict(model)
    to_hf = qwen3_params.build_to_hf_manifest(model.config)
    specs, unstack_counts = safetensors_saver.build_export_specs_from_manifest(
        flat_state, to_hf, output_dtype=jnp.bfloat16
    )
    plan = safetensors_saver.build_save_plan(
        specs,
        num_ranks=len(devices),
        max_bucket_bytes=128 << 20,
        max_shard_bytes=2560 << 20,
        unstack_counts=unstack_counts,
    )
    ```
  """
  parsed = hf_mapping.parse_manifest(to_hf_manifest)
  dropped = set(parsed['dropped_keys']) if 'dropped_keys' in parsed else set()
  itemsize = jnp.dtype(output_dtype).itemsize if output_dtype else None
  specs: list[TensorExportSpec] = []
  unstack_counts: dict[str, int] = {}
  for rule in parsed['rules']:
    src_keys, tgt_keys = rule['source_keys'], rule['target_keys']
    if any(k in dropped for k in src_keys):
      continue
    if len(src_keys) != 1:
      if any(k in flat_state for k in src_keys):
        raise ValueError(
            'build_export_specs_from_manifest does not support multi-source'
            f' rules: {src_keys!r}.'
        )
      continue
    if src_keys[0] not in flat_state:
      continue
    src_key = src_keys[0]
    shape_2d, axes, unstack_axis = _parse_export_rule_ops(src_key, rule['ops'])
    hf_key = tgt_keys[0]
    if unstack_axis is not None:
      unstack_axis = int(unstack_axis)
      hf_key = _extract_expert_key_template(tgt_keys)
      unstack_counts[src_key] = len(tgt_keys)
    specs.append(
        make_tensor_export_spec(
            flat_state,
            src_key,
            hf_key,
            shape_2d=shape_2d,
            transpose_axes=axes,
            unstack_axis=unstack_axis,
            lora_scaling=lora_scaling,
            itemsize=itemsize,
        )
    )
  return specs, unstack_counts


def _choose_moe_chunk_count(
    n_exp: int, nbytes: int, limit: int, num_ranks: int, min_chunk_bytes: int
) -> int:
  """Selects an exact expert divisor `>= min_chunks`, or the largest valid one."""
  can_split_by_ranks = (
      n_exp % num_ranks == 0
      and nbytes % num_ranks == 0
      and nbytes // num_ranks >= min_chunk_bytes
  )
  min_chunks = max(
      math.ceil(nbytes / limit),
      num_ranks if can_split_by_ranks else 1,
  )
  valid_divisors = [
      d
      for d in range(2, n_exp + 1)
      if n_exp % d == 0 and nbytes % d == 0 and nbytes // d >= min_chunk_bytes
  ]
  return next(
      (d for d in valid_divisors if d >= min_chunks),
      valid_divisors[-1] if valid_divisors else 1,
  )


def _expand_moe_chunk_specs(
    specs: Sequence[TensorExportSpec],
    counts: Mapping[str, int],
    limit: int,
    num_ranks: int,
    rank_align_bytes: int = 4096,
) -> list[TensorExportSpec]:
  """Splits `> limit` stacked 3D MoE specs along `unstack_axis=0` into chunks."""
  expanded: list[TensorExportSpec] = []
  min_chunk_bytes = num_ranks * rank_align_bytes
  for spec in specs:
    n_exp = counts[spec.source_key] if spec.source_key in counts else 0
    if (
        spec.unstack_axis != 0
        or spec.unstack_range is not None
        or spec.nbytes <= limit
        or n_exp <= 1
    ):
      expanded.append(spec)
      continue
    n_chunks = _choose_moe_chunk_count(
        n_exp, spec.nbytes, limit, num_ranks, min_chunk_bytes
    )
    if n_chunks <= 1:
      expanded.append(spec)
      continue
    step = n_exp // n_chunks
    c_bytes = spec.nbytes // n_chunks
    expanded.extend(
        dataclasses.replace(
            spec, nbytes=c_bytes, unstack_range=(i * step, (i + 1) * step)
        )
        for i in range(n_chunks)
    )
  return expanded


def _build_shard_groups(
    waves: Sequence[WaveBucket],
    num_ranks: int,
    max_shard_bytes: int | None,
    counts: Mapping[str, int] | None = None,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
  """Groups consecutive waves synchronously across ranks within `max_shard_bytes`.

  Uses synchronized wave cuts across ranks and emits `(rank, wave_indices)` in
  wave-group-major, ascending-rank order so MoE experts distributed across ranks
  stay inside one self-contained wave-group and never straddle shard boundaries.
  Reserves estimated safetensors header bytes so multi-wave shards stay within
  `max_shard_bytes`.

  Args:
    waves: Ordered sequence of planned `WaveBucket`s.
    num_ranks: Total number of export ranks.
    max_shard_bytes: Per-shard file byte limit, or `None` to disable grouping.
    counts: Optional `{source_key: num_experts}` map for header key estimation.

  Returns:
    Tuple of `(rank, wave_indices)` pairs defining each `.safetensors` shard.
  """
  if max_shard_bytes is None:
    return ()
  key_counts = counts or {}
  expand_spec = functools.partial(expand_spec_keys, counts=key_counts)
  wave_groups: list[list[int]] = []
  cur_waves: list[int] = []
  cur_rank_bytes = [0] * num_ranks
  for w in waves:
    w_bytes = [
        w.rank_bytes[r]
        + sum(
            len(k) + _EST_SAFETENSORS_KEY_HEADER_BYTES
            for k in itertools.chain.from_iterable(
                map(expand_spec, w.rank_assignments[r])
            )
        )
        for r in range(num_ranks)
    ]
    max_after = max(
        (
            cur_rank_bytes[r]
            + w_bytes[r]
            + min(cur_rank_bytes[r] + w_bytes[r], 1)
            * _EST_SAFETENSORS_HEADER_OVERHEAD_BYTES
            for r in range(num_ranks)
        ),
        default=0,
    )
    if cur_waves and (
        max_after > max_shard_bytes or len(cur_waves) >= _MAX_GCS_COMPOSE_PARTS
    ):
      wave_groups.append(cur_waves)
      cur_waves = []
      cur_rank_bytes = [0] * num_ranks
    cur_waves.append(w.wave_index)
    cur_rank_bytes = [
        cur_rank_bytes[r] + w_bytes[r] for r in range(num_ranks)
    ]
  if cur_waves:
    wave_groups.append(cur_waves)
  groups = [
      (r, tuple(wi for wi in gw if waves[wi].rank_bytes[r] > 0))
      for gw, r in itertools.product(wave_groups, range(num_ranks))
  ]
  return tuple(item for item in groups if item[1])


def _build_weight_map(
    grid: Mapping[tuple[int, int], Sequence[TensorExportSpec]],
    counts: Mapping[str, int],
    rank_loads: Mapping[int, int],
    num_ranks: int,
    per_wave_shards: bool,
    wave_shards: Sequence[tuple[int, int]],
    shard_groups: Sequence[tuple[int, tuple[int, ...]]],
) -> dict[str, str]:
  """Maps every expanded HF tensor key to its output `.safetensors` filename."""
  entries: list[tuple[str, int, int]] = []
  if shard_groups:
    total = len(shard_groups)
    for idx, (r, w_idxs) in enumerate(shard_groups):
      fname = shard_filename(idx, total)
      entries.extend((fname, w, r) for w in w_idxs)
  elif per_wave_shards:
    total = len(wave_shards)
    entries = [
        (shard_filename(idx, total), w, r)
        for idx, (r, w) in enumerate(wave_shards)
    ]
  else:
    active = [r for r in range(num_ranks) if rank_loads[r] > 0]
    rank_fname = {
        r: shard_filename(i, len(active)) for i, r in enumerate(active)
    }
    entries = [(rank_fname[r], w, r) for r, w in wave_shards]
  weight_map: dict[str, str] = {}
  for fname, w, r in entries:
    for spec in grid[(w, r)]:
      for hf_k in expand_spec_keys(spec, counts):
        weight_map[hf_k] = fname
  return weight_map


def _pack_specs_into_wave_grid(
    expanded_specs: Sequence[TensorExportSpec],
    num_ranks: int,
    limit: int,
    ranks_per_host: int,
) -> tuple[dict[tuple[int, int], list[TensorExportSpec]], dict[int, int], int]:
  """LPT bin-packs expanded specs across `(wave, rank)` buckets."""
  rank_loads = dict.fromkeys(range(num_ranks), 0)
  wave_idx = dict.fromkeys(range(num_ranks), 0)
  wave_loads = dict.fromkeys(range(num_ranks), 0)
  grid: dict[tuple[int, int], list[TensorExportSpec]] = collections.defaultdict(
      list
  )
  ordered = sorted(
      expanded_specs,
      key=lambda s: (
          s.unstack_range is None,
          -s.nbytes,
          s.source_key,
          s.unstack_range or (),
      ),
  )
  for spec in ordered:
    r = min(
        range(num_ranks),
        key=lambda k, s=spec: (
            wave_idx[k] + (1 if limit - s.nbytes < wave_loads[k] > 0 else 0),
            rank_loads[k],
            k % ranks_per_host if s.unstack_range is None else 0,
            k,
        ),
    )
    rank_loads[r] += spec.nbytes
    if limit - spec.nbytes < wave_loads[r] > 0:
      wave_idx[r] += 1
      wave_loads[r] = 0
    wave_loads[r] += spec.nbytes
    grid[(wave_idx[r], r)].append(spec)

  num_waves = max(wave_idx.values(), default=0) + 1
  if (
      num_waves > 1
      and len({s.unstack_range is None for s in expanded_specs}) == 2
  ):
    grid = collections.defaultdict(
        list, {((w + 1) % num_waves, r): v for (w, r), v in grid.items()}
    )
  return grid, rank_loads, num_waves


def build_save_plan(
    specs: Sequence[TensorExportSpec],
    num_ranks: int = 1,
    max_bucket_bytes: int | None = None,
    unstack_counts: Mapping[str, int] | None = None,
    per_wave_shards: bool = False,
    rank_align_bytes: int = 4096,
    ranks_per_host: int = 8,
    max_shard_bytes: int | None = None,
) -> SavePlan:
  """Builds an LPT bin-packed Subtree all-to-all `SavePlan` across ranks and waves.

  Explicitly decouples **HBM/D2H wave memory** (`max_bucket_bytes`) from
  **output `.safetensors` file size** (`max_shard_bytes`):

    * `max_bucket_bytes` (e.g. `128 MiB`) bounds temporary HBM/ICI/D2H memory
      per rank per wave so giant MoE models never OOM during export.
    * `max_shard_bytes` (e.g. `2.5 GiB`, hard ceiling `4 GiB`) groups up to 32
      consecutive waves per rank into ONE standard-sized `.safetensors` shard
      file (`model-XXXXX-of-YYYYY.safetensors`). On `gs://` with
      :class:`MultiprocStagePool`, each wave still streams immediately as a
      `~128 MiB` `.part.<k>` upload and is server-side composed into the final
      `~2.5 GiB` shard on `flush()`.
    * When `max_shard_bytes is None`, falls back to legacy `per_wave_shards`
      mode (one file per wave-rank when `True`, or one file per rank when
      `False`).

  Args:
    specs: Sequence of :class:`TensorExportSpec` descriptors to export.
    num_ranks: Total participating device ranks across all hosts (must be >= 1).
    max_bucket_bytes: Per-rank HBM/D2H wave byte ceiling (recommended:
      `DEFAULT_MAX_BUCKET_BYTES` = 128 MiB). `None` puts all tensors in wave 0.
    unstack_counts: Optional `{source_key: num_experts}` map enabling automatic
      chunking of oversized 3D MoE tensors across waves.
    per_wave_shards: Legacy per-wave file mode used only when `max_shard_bytes
      is None`.
    rank_align_bytes: Minimum byte alignment per rank when splitting MoE chunks.
    ranks_per_host: Local accelerator devices per host (used to balance dense
      tensors across hosts).
    max_shard_bytes: Optional maximum `.safetensors` shard file size in bytes.
      Must lie in `[1, MAX_SAFETENSORS_SHARD_BYTES]` (4 GiB). Recommended:
      `DEFAULT_MAX_SHARD_BYTES` (2.5 GiB).

  Returns:
    A cached, immutable :class:`SavePlan` with wave assignments, `shard_groups`,
    and complete `weight_map` for `model.safetensors.index.json`.

  Raises:
    ValueError: If `num_ranks`, `max_bucket_bytes`, `rank_align_bytes`, or
      `ranks_per_host` is `< 1`, if `max_shard_bytes` is outside
      `[1, MAX_SAFETENSORS_SHARD_BYTES]`, or if
      `max_bucket_bytes > max_shard_bytes`.

  Example:
    ```python
    plan = safetensors_saver.build_save_plan(
        specs,
        num_ranks=64,
        max_bucket_bytes=safetensors_saver.DEFAULT_MAX_BUCKET_BYTES,
        max_shard_bytes=safetensors_saver.DEFAULT_MAX_SHARD_BYTES,
        unstack_counts=unstack_counts,
    )
    print(f'waves={len(plan.waves)}, shards={len(plan.shard_groups)}')
    ```
  """
  limit = max_bucket_bytes if max_bucket_bytes is not None else (1 << 62)
  if min(num_ranks, limit, rank_align_bytes, ranks_per_host) < 1:
    raise ValueError(f'Invalid plan args: {num_ranks=}, {max_bucket_bytes=}')
  valid_shard = max_shard_bytes is None or (
      _is_valid_pos_int(max_shard_bytes)
      and max_shard_bytes <= MAX_SAFETENSORS_SHARD_BYTES
  )
  if not valid_shard:
    raise ValueError(
        f'max_shard_bytes must be in [1, {MAX_SAFETENSORS_SHARD_BYTES}], got'
        f' {max_shard_bytes!r}'
    )
  if (
      max_bucket_bytes is not None
      and max_shard_bytes is not None
      and max_bucket_bytes > max_shard_bytes
  ):
    raise ValueError(
        f'max_bucket_bytes ({max_bucket_bytes}) exceeds max_shard_bytes'
        f' ({max_shard_bytes}).'
    )
  counts = unstack_counts or {}
  cache_key = (
      tuple(specs),
      num_ranks,
      max_bucket_bytes,
      tuple(sorted(counts.items())),
      per_wave_shards,
      rank_align_bytes,
      ranks_per_host,
      max_shard_bytes,
  )
  if cache_key in _SAVE_PLAN_CACHE:
    return _SAVE_PLAN_CACHE[cache_key]
  expanded_specs = _expand_moe_chunk_specs(
      specs, counts, limit, num_ranks, rank_align_bytes
  )
  grid, rank_loads, num_waves = _pack_specs_into_wave_grid(
      expanded_specs, num_ranks, limit, ranks_per_host
  )
  wave_shards = tuple((r, w) for w, r in sorted(grid))
  waves = tuple(
      WaveBucket(
          w,
          {r: tuple(grid[(w, r)]) for r in range(num_ranks)},
          {r: sum(s.nbytes for s in grid[(w, r)]) for r in range(num_ranks)},
          max_bucket_bytes=max_bucket_bytes,
      )
      for w in range(num_waves)
  )
  shard_groups = _build_shard_groups(
      waves, num_ranks, max_shard_bytes, counts=counts
  )
  weight_map = _build_weight_map(
      grid,
      counts,
      rank_loads,
      num_ranks,
      per_wave_shards,
      wave_shards,
      shard_groups,
  )
  plan = SavePlan(
      num_ranks=num_ranks,
      total_bytes=sum(rank_loads.values()),
      max_bucket_bytes=max_bucket_bytes,
      waves=waves,
      rank_total_bytes=rank_loads,
      weight_map=weight_map,
      per_wave_shards=per_wave_shards,
      wave_shards=wave_shards,
      max_shard_bytes=max_shard_bytes,
      shard_groups=shard_groups,
  )
  _SAVE_PLAN_CACHE[cache_key] = plan
  return plan


_as_axis_tuple = safetensors_loader.as_axis_tuple


def _compatible_mesh_axes(
    in_shd: jax.sharding.Sharding | None,
    rank_shd: jax.sharding.Sharding | None,
) -> (
    tuple[
        jax.sharding.Mesh,
        jax.sharding.Mesh,
        tuple[Any, ...],
        jax.sharding.PartitionSpec,
    ]
    | None
):
  """Validates compatible device order and returns active non-singleton axes."""
  if not (
      isinstance(in_shd, jax.sharding.NamedSharding)
      and isinstance(rank_shd, jax.sharding.NamedSharding)
      and isinstance(in_shd.mesh, jax.sharding.Mesh)
      and isinstance(rank_shd.mesh, jax.sharding.Mesh)
  ):
    return None
  in_m, rk_m = in_shd.mesh, rank_shd.mesh
  active = tuple(a for a in in_m.axis_names if in_m.shape[a] > 1)
  if (
      not active
      or len(rk_m.axis_names) != 1
      or len(rank_shd.spec) != 1
      or rank_shd.spec[0] is None
      or not all(isinstance(a, str) for a in active)
      or tuple(in_m.devices.flat) != tuple(rk_m.devices.flat)
  ):
    return None
  return in_m, rk_m, active, in_shd.spec


_merge_reshape_spec = safetensors_loader.merge_reshape_spec


def _plan_axis0_reshard_ops(
    sp: tuple[Any, ...],
    axes: tuple[int, ...],
    active_axes: tuple[Any, ...],
) -> tuple[
    bool,
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[Any, ...],
    tuple[tuple[Any, int], ...],
]:
  """Chooses pre/post-transpose order, replicated slices, and inner all_to_all."""
  active_set = set(active_axes)
  norm = [tuple(a for a in _as_axis_tuple(e) if a in active_set) for e in sp]
  a2a_before_tr = (
      bool(axes)
      and axes[0] == 0
      and axes != tuple(range(len(sp)))
      and len(norm) > 1
      and bool(norm[axes[-1]])
      and not bool(norm[-1])
  )
  work_spec = sp if a2a_before_tr else tuple(sp[i] for i in axes)
  work_norm = norm if a2a_before_tr else [norm[i] for i in axes]
  used = set(itertools.chain.from_iterable(work_norm))
  per_dim = [
      [(a, d) for a in reversed(dim_ax)]
      for d, dim_ax in enumerate(work_norm)
      if d > 0 and dim_ax
  ]
  per_dim.sort(key=lambda grp: active_axes.index(grp[0][0]))
  rep_axes = tuple(ax for ax in active_axes if ax not in used)
  return (
      a2a_before_tr,
      work_spec,
      work_norm[0] if work_norm else (),
      rep_axes,
      tuple(itertools.chain.from_iterable(per_dim)),
  )


def _apply_axis0_reshard(
    x: jax.Array,
    in_mesh: jax.sharding.Mesh,
    work_spec: tuple[Any, ...],
    out_order: tuple[Any, ...],
    rep_axes: tuple[Any, ...],
    inner_ops: tuple[tuple[Any, int], ...],
) -> jax.Array:
  """Runs `shard_map` dynamic-slice + all_to_all moving shard axes to axis 0."""
  def _body(local_x: jax.Array) -> jax.Array:
    for ax in rep_axes:
      chunk = local_x.shape[0] // in_mesh.shape[ax]
      local_x = jax.lax.dynamic_slice_in_dim(
          local_x, jax.lax.axis_index(ax) * chunk, chunk, axis=0
      )
    for ax, d in inner_ops:
      local_x = jax.lax.all_to_all(
          local_x, axis_name=ax, split_axis=0, concat_axis=d, tiled=True
      )
    return local_x

  return jax.shard_map(
      _body,
      mesh=in_mesh,
      in_specs=jax.sharding.PartitionSpec(*work_spec),
      out_specs=jax.sharding.PartitionSpec(
          out_order, *([None] * (x.ndim - 1))
      ),
      check_vma=False,
  )(x)


def _reshard_to_rank_generic(
    arr: jax.Array,
    in_sharding: jax.sharding.Sharding | None,
    shape_2d: tuple[int, ...] | None,
    transpose_axes: tuple[int, ...] | None,
    output_dtype: jnp.dtype | None,
    rank_shd: jax.sharding.Sharding | None,
) -> jax.Array | None:
  """Reshards any N-D NamedSharding tensor to 1D `rank_shd` without all-gather."""
  compat = _compatible_mesh_axes(in_sharding, rank_shd)
  if compat is None:
    return None
  in_mesh, rank_mesh, active_axes, in_spec = compat
  target_shape = arr.shape if shape_2d is None else shape_2d
  if not target_shape:
    return None
  sp = _merge_reshape_spec(arr.shape, in_spec, in_mesh, target_shape)
  if sp is None:
    return None
  x = jnp.reshape(arr, target_shape).astype(
      arr.dtype if output_dtype is None else output_dtype
  )
  axes = tuple(range(x.ndim)) if transpose_axes is None else transpose_axes
  a2a_before_tr, work_spec, existing_d0, rep_axes, inner_ops = (
      _plan_axis0_reshard_ops(sp, axes, active_axes)
  )
  if not a2a_before_tr:
    x = jnp.transpose(x, axes)
  out_order = existing_d0 + rep_axes + tuple(ax for ax, _ in inner_ops)
  if x.shape[0] % math.prod(in_mesh.shape[a] for a in out_order) != 0:
    return None
  if rep_axes or inner_ops:
    x = _apply_axis0_reshard(
        x, in_mesh, work_spec, out_order, rep_axes, inner_ops
    )
  if a2a_before_tr:
    x = jnp.transpose(x, axes)
  if out_order != active_axes:
    x = jax.lax.with_sharding_constraint(
        x,
        jax.sharding.NamedSharding(
            in_mesh,
            jax.sharding.PartitionSpec(active_axes, *([None] * (x.ndim - 1))),
        ),
    )
  n_r = rank_mesh.size
  flat = (n_r, -1, 128) if x.size % (n_r * 1024) == 0 else (n_r, -1)
  return jax.lax.with_sharding_constraint(jnp.reshape(x, flat), rank_shd)


def _split_hbm_chunks(
    out: jax.Array,
    num_chunks: int,
    flatten_ranks: int,
    out_sharding: jax.sharding.Sharding | None,
    from_rank_shd: bool = False,
) -> jax.Array | tuple[jax.Array, ...]:
  """Splits transformed tensor into `num_chunks` along axis 0 on HBM."""
  if num_chunks <= 1:
    return out
  if flatten_ranks <= 0:
    if out_sharding is not None:
      out = jax.lax.with_sharding_constraint(out, out_sharding)
    return tuple(jnp.split(out, num_chunks, axis=0))
  if (
      from_rank_shd
      and isinstance(out_sharding, jax.sharding.NamedSharding)
      and num_chunks % flatten_ranks == 0
  ):
    ax = out_sharding.spec[0]
    out = jax.shard_map(
        lambda loc: jax.lax.all_to_all(
            jnp.reshape(loc, (num_chunks // flatten_ranks, flatten_ranks, -1)),
            axis_name=ax,
            split_axis=1,
            concat_axis=0,
            tiled=True,
        ),
        mesh=out_sharding.mesh,
        in_specs=out_sharding.spec,
        out_specs=jax.sharding.PartitionSpec(None, ax, None),
        check_vma=False,
    )(out)
  else:
    out = jnp.reshape(out, (num_chunks, flatten_ranks, -1))
    if isinstance(out_sharding, jax.sharding.NamedSharding):
      out = jax.lax.with_sharding_constraint(
          out,
          jax.sharding.NamedSharding(
              out_sharding.mesh,
              jax.sharding.PartitionSpec(None, *out_sharding.spec),
          ),
      )
  return tuple(
      jnp.squeeze(c, axis=0) for c in jnp.split(out, num_chunks, axis=0)
  )


@functools.partial(
    jax.jit,
    static_argnames=(
        'shape_2d',
        'transpose_axes',
        'output_dtype',
        'out_sharding',
        'flatten_ranks',
        'num_chunks',
        'in_sharding',
    ),
)
def _jit_transform_on_hbm(
    base_arr: jax.Array,
    lora_a: jax.Array | None,
    lora_b: jax.Array | None,
    lora_scaling: float,
    shape_2d: tuple[int, ...] | None,
    transpose_axes: tuple[int, ...] | None,
    output_dtype: jnp.dtype | None,
    out_sharding: jax.sharding.Sharding | None = None,
    flatten_ranks: int = 0,
    num_chunks: int = 1,
    in_sharding: jax.sharding.Sharding | None = None,
) -> jax.Array | tuple[jax.Array, ...]:
  """Jitted on-HBM LoRA merge, reshape, and transpose kernel."""
  resharded = None
  if lora_a is None and flatten_ranks > 0:
    resharded = _reshard_to_rank_generic(
        base_arr,
        in_sharding,
        shape_2d,
        transpose_axes,
        output_dtype,
        out_sharding,
    )
  if resharded is not None:
    return _split_hbm_chunks(
        resharded, num_chunks, flatten_ranks, out_sharding, from_rank_shd=True
    )
  arr = jnp.reshape(base_arr, shape_2d or base_arr.shape)
  if lora_a is not None and lora_b is not None:
    a_2d = lora_a.reshape(-1, lora_a.shape[-1])
    b_2d = lora_b.reshape(lora_b.shape[0], -1)
    delta = (a_2d @ b_2d) * lora_scaling
    arr = arr + jnp.reshape(delta, arr.shape).astype(arr.dtype)
  axes = transpose_axes or tuple(range(arr.ndim))
  out = jnp.transpose(arr, axes).astype(output_dtype or arr.dtype)
  if num_chunks > 1:
    return _split_hbm_chunks(out, num_chunks, flatten_ranks, out_sharding)
  if flatten_ranks > 0:
    shape = (
        (flatten_ranks, -1, 128)
        if out.size % (flatten_ranks * 1024) == 0
        else (flatten_ranks, -1)
    )
    out = jnp.reshape(out, shape)
  return (
      jax.lax.with_sharding_constraint(out, out_sharding)
      if out_sharding is not None
      else out
  )


def _chunk_count_and_index(
    base_arr: jax.Array | jax.ShapeDtypeStruct, spec: TensorExportSpec
) -> tuple[int, int]:
  """Returns `(num_chunks, chunk_index)` for `spec.unstack_range`."""
  if spec.unstack_range is None:
    return 1, 0
  raw_s = spec.shape_2d or base_arr.shape
  ax = spec.transpose_axes or tuple(range(len(raw_s)))
  span = spec.unstack_range[1] - spec.unstack_range[0]
  return raw_s[ax[0]] // span, spec.unstack_range[0] // span


def _resolve_transform_sharding(
    base_arr: jax.Array | jax.ShapeDtypeStruct,
    spec: TensorExportSpec,
    out_sharding: jax.sharding.NamedSharding | None,
    num_chunks: int = 1,
) -> tuple[jax.sharding.Sharding | None, int]:
  """Resolves target sharding and `flatten_ranks` for `_jit_transform_on_hbm`."""
  num_devs = len(out_sharding.device_set) if out_sharding is not None else 0
  chunk_elems = int(math.prod(base_arr.shape)) // max(1, num_chunks)
  unaligned = num_devs > 0 and chunk_elems % (num_devs * 1024) != 0
  is_rep = spec.is_replicated or infer_is_replicated(base_arr) or unaligned
  if not is_rep:
    return out_sharding, num_devs
  b_shd = base_arr.sharding
  if isinstance(b_shd, jax.sharding.NamedSharding):
    if unaligned or any(b_shd.spec):
      return (
          jax.sharding.NamedSharding(b_shd.mesh, jax.sharding.PartitionSpec()),
          0,
      )
    return None, 0
  if unaligned and out_sharding is not None:
    return (
        jax.sharding.NamedSharding(
            out_sharding.mesh, jax.sharding.PartitionSpec()
        ),
        0,
    )
  return None, 0


def _uses_multi_rank_routing(
    spec: TensorExportSpec,
    arr: jax.Array | jax.ShapeDtypeStruct,
    num_ranks: int,
) -> bool:
  """Returns True when `arr` should route through the multi-rank fast path."""
  return num_ranks > 1 and not (spec.is_replicated or infer_is_replicated(arr))


def _array_nbytes(arr: jax.Array | jax.ShapeDtypeStruct) -> int:
  """Returns total byte size for a `jax.Array` or `jax.ShapeDtypeStruct`."""
  return int(math.prod(arr.shape)) * arr.dtype.itemsize


def _abstract_sds(
    arr: jax.Array | jax.ShapeDtypeStruct,
) -> jax.ShapeDtypeStruct:
  """Converts `arr` to an abstract `jax.ShapeDtypeStruct` with its sharding."""
  return jax.ShapeDtypeStruct(arr.shape, arr.dtype, sharding=arr.sharding)


def _sds_key(x: Any) -> Any:
  """Returns a hashable `(shape, dtype, sharding)` key for AOT executable lookup."""
  if x is None:
    return None
  return (tuple(int(d) for d in x.shape), jnp.dtype(x.dtype), x.sharding)


def _transform_hbm_key(
    base_arr: Any,
    lora_a: Any,
    lora_b: Any,
    shape_2d: tuple[int, ...] | None,
    transpose_axes: tuple[int, ...] | None,
    output_dtype: jnp.dtype | None,
    out_sharding: jax.sharding.Sharding | None,
    flatten_ranks: int,
    num_chunks: int,
) -> tuple[Any, ...]:
  return (
      'transform',
      _sds_key(base_arr),
      _sds_key(lora_a),
      _sds_key(lora_b),
      shape_2d,
      transpose_axes,
      output_dtype,
      out_sharding,
      flatten_ranks,
      num_chunks,
  )


def _fused_wave_key(
    base_arrs: Sequence[Any],
    shapes_2d: tuple[tuple[int, ...] | None, ...],
    transpose_axes: tuple[tuple[int, ...] | None, ...],
    output_dtype: jnp.dtype | None,
    out_sharding: jax.sharding.NamedSharding,
) -> tuple[Any, ...]:
  return (
      'fused_wave',
      tuple(_sds_key(a) for a in base_arrs),
      shapes_2d,
      transpose_axes,
      output_dtype,
      out_sharding,
  )


def _concat_local_key(
    local_shards: Sequence[Any],
    mesh: jax.sharding.Mesh,
    axis_name: str,
) -> tuple[Any, ...]:
  return (
      'concat_local',
      tuple(_sds_key(a) for a in local_shards),
      mesh,
      axis_name,
  )


def _exchange_key(
    ordered_shards: Sequence[Any],
    mesh: jax.sharding.Mesh,
    axis_name: str,
    order: tuple[int, ...],
    in_offsets: tuple[int, ...],
    send_sizes: tuple[int, ...],
    max_recv: int,
    use_ragged: bool,
    perm: Any,
    ragged_meta: Any,
) -> tuple[Any, ...]:
  """Builds an AOT compilation cache key for `_jit_exchange_shards_all_to_all`."""
  return (
      'exchange',
      tuple(_sds_key(a) for a in ordered_shards),
      mesh,
      axis_name,
      order,
      in_offsets,
      send_sizes,
      max_recv,
      use_ragged,
      _sds_key(perm),
      _sds_key(ragged_meta),
  )


def _compiled_temp_hbm_bytes(compiled: Any) -> int:
  """Returns temporary HBM bytes from `compiled.memory_analysis()`, or 0."""
  try:
    stats = compiled.memory_analysis()
  except (NotImplementedError, AttributeError, RuntimeError):
    return 0
  if stats is None:
    return 0
  return max(0, int(stats.temp_size_in_bytes))


def _compiled_output_sds(compiled: Any) -> Any:
  """Returns a PyTree of `jax.ShapeDtypeStruct` with exact output shardings."""
  return jax.tree.map(
      lambda info, shd: jax.ShapeDtypeStruct(
          info.shape,
          info.dtype,
          sharding=shd if isinstance(shd, jax.sharding.Sharding) else None,
      ),
      compiled.out_info,
      compiled.output_shardings,
  )


def transform_tensor_on_hbm(
    base_arr: jax.Array,
    spec: TensorExportSpec,
    lora_a: jax.Array | None = None,
    lora_b: jax.Array | None = None,
    output_dtype: jnp.dtype | None = None,
    target_device: jax.Device | None = None,
    out_sharding: jax.sharding.NamedSharding | None = None,
    compiled_plan: CompiledSavePlan | None = None,
) -> jax.Array:
  """Applies on-HBM LoRA merge, reshape, transpose, cast, and optional routing.

  Args:
    base_arr: Source device array from `flat_state`.
    spec: `TensorExportSpec` describing transforms and optional expert slice.
    lora_a: Optional LoRA-A factor array to merge on HBM.
    lora_b: Optional LoRA-B factor array to merge on HBM.
    output_dtype: Optional cast dtype (`None` keeps `base_arr.dtype`).
    target_device: Optional single destination device placement.
    out_sharding: Optional `NamedSharding` target (e.g. 1D rank sharding).
    compiled_plan: Optional `CompiledSavePlan` for zero-JIT AOT dispatch.

  Returns:
    Transformed `jax.Array` on HBM ready for all-to-all or direct D2H.

  Example:
    ```python
    arr = safetensors_saver.transform_tensor_on_hbm(
        flat_state[spec.source_key], spec, output_dtype=jnp.bfloat16
    )
    ```
  """
  num_chunks, chunk_idx = _chunk_count_and_index(base_arr, spec)
  shd, flatten_ranks = _resolve_transform_sharding(
      base_arr, spec, out_sharding, num_chunks
  )
  key = _transform_hbm_key(
      base_arr,
      lora_a,
      lora_b,
      spec.shape_2d,
      spec.transpose_axes,
      output_dtype,
      shd,
      flatten_ranks,
      num_chunks,
  )
  if compiled_plan is not None and key in compiled_plan.executables:
    res = compiled_plan.executables[key](
        base_arr, lora_a, lora_b, float(spec.lora_scaling)
    )
  else:
    res = _jit_transform_on_hbm(
        base_arr,
        lora_a,
        lora_b,
        float(spec.lora_scaling),
        spec.shape_2d,
        spec.transpose_axes,
        output_dtype,
        shd,
        flatten_ranks,
        num_chunks,
        base_arr.sharding,
    )
  out = res[chunk_idx] if isinstance(res, tuple) else res
  return (
      jax.device_put(out, target_device) if target_device is not None else out
  )


def build_tile_perm_row(
    tile_cols: Sequence[int],
    num_ranks: int,
    m_recv: int,
    use_ragged: bool,
) -> np.ndarray:
  """Builds 1D `(8, 128)`-tile permutation packing `(num_ranks, cols_i)` slices.

  Args:
    tile_cols: Per-tensor column counts in 128-element tile units.
    num_ranks: Number of ranks participating in the exchange.
    m_recv: Total output tile columns per rank.
    use_ragged: Whether ragged all-to-all packing is active.

  Returns:
    1D `int32` index array mapping concatenated per-rank tiles to contiguous
    tensor layouts.

  Example:
    ```python
    perm = safetensors_saver.build_tile_perm_row(
        tile_cols=[4, 8], num_ranks=4, m_recv=12, use_ragged=True
    )
    ```
  """
  total_tiles = sum(tile_cols)
  row_tiles = total_tiles if (use_ragged and total_tiles > 0) else m_recv
  n_tiles = num_ranks * m_recv
  if total_tiles <= 0 or (len(tile_cols) <= 1 and row_tiles == total_tiles):
    return np.arange(n_tiles, dtype=np.int32)
  active_tiles = num_ranks * row_tiles
  grid = np.arange(active_tiles, dtype=np.int32).reshape(num_ranks, row_tiles)
  edges = tuple(itertools.accumulate((0, *tile_cols)))
  parts = [
      grid[:, edges[i] : edges[i + 1]].reshape(-1)
      for i in range(len(tile_cols))
  ]
  if total_tiles < row_tiles:
    parts.append(grid[:, total_tiles:row_tiles].reshape(-1))
  if active_tiles < n_tiles:
    parts.append(np.arange(active_tiles, n_tiles, dtype=np.int32))
  return np.concatenate(parts, axis=0)


def build_grouped_ragged_meta(
    weights: Sequence[int],
    owner_groups: Sequence[tuple[int, ...]],
    num_ranks: int,
) -> tuple[np.ndarray, int]:
  """Builds per-rank `(4, num_ranks)` ragged_all_to_all metadata for shards.

  Args:
    weights: Per-tensor payload sizes (in tile/chunk units).
    owner_groups: Per-tensor tuple of owner ranks receiving each slice.
    num_ranks: Total number of participating ranks.

  Returns:
    2-tuple `(meta, max_recv)` where `meta` has shape
    `(num_ranks, 4, num_ranks)` holding `(in_off, send_sz, out_off, recv_sz)`
    and `max_recv` is the maximum per-rank receive length.

  Example:
    ```python
    meta, max_recv = safetensors_saver.build_grouped_ragged_meta(
        weights=[4, 4], owner_groups=[(0,), (1,)], num_ranks=2
    )
    ```
  """
  in_off = np.zeros((num_ranks, num_ranks), dtype=np.int32)
  send_sz = np.zeros((num_ranks, num_ranks), dtype=np.int32)
  next_expected = np.zeros((num_ranks, num_ranks), dtype=np.int32)
  in_cursor = 0
  for w, grp in zip(weights, owner_groups):
    tiles = w // 1024
    k = len(grp)
    for s in range(num_ranks):
      dest = grp[(s * k) // num_ranks]
      if send_sz[s, dest] == 0:
        in_off[s, dest] = in_cursor
      elif next_expected[s, dest] != in_cursor:
        raise ValueError(
            'build_grouped_ragged_meta requires contiguous per-destination runs'
            f' (src={s}, dst={dest}, expected={next_expected[s, dest]},'
            f' got={in_cursor}).'
        )
      send_sz[s, dest] += tiles
      next_expected[s, dest] = in_cursor + tiles
    in_cursor += tiles
  out_off = np.zeros((num_ranks, num_ranks), dtype=np.int32)
  recv_cursors = np.zeros(num_ranks, dtype=np.int32)
  for s in range(num_ranks):
    for d in range(num_ranks):
      if send_sz[s, d] > 0:
        out_off[s, d] = recv_cursors[d]
        recv_cursors[d] += send_sz[s, d]
  meta = np.stack([in_off, send_sz, out_off, send_sz.T], axis=1)
  m_recv = max(1, math.ceil(int(np.max(recv_cursors, initial=0)) / num_ranks))
  return meta, m_recv * 1024


def _pack_bf16_u16_to_u32(
    arr: jax.Array, leading_shape: tuple[int, ...]
) -> jax.Array:
  """Packs bf16 -> u32 via same-width u16 shift (avoids 64x Reduce(dim=2))."""
  u16 = jax.lax.bitcast_convert_type(
      arr.reshape(*leading_shape, 256), jnp.uint16
  )
  return u16[..., 0::2].astype(jnp.uint32) | (
      u16[..., 1::2].astype(jnp.uint32) << jnp.uint32(16)
  )


def _pad_u32_d2h(arr_u32: jax.Array, num_ranks: int = 1) -> jax.Array:
  """Pads packed u32 rows to a multiple of 8 per rank for TPU `(M, 128)` D2H."""
  arr_3d = arr_u32.reshape(num_ranks, -1, 128)
  rem = arr_3d.shape[1] % 8
  if rem:
    arr_3d = jnp.pad(arr_3d, ((0, 0), (0, 8 - rem), (0, 0)))
  return arr_3d.reshape(-1, 128)


def _pack_bf16_to_u32_d2h(arr: jax.Array, num_ranks: int = 1) -> jax.Array:
  """Packs bfloat16 into a 2D `(M, 128)` uint32 array on TPU for fast D2H."""
  if (
      jax.default_backend() == 'tpu'
      and arr.dtype == jnp.bfloat16
      and arr.size % (num_ranks * 256) == 0
  ):
    return _pad_u32_d2h(_pack_bf16_u16_to_u32(arr, (num_ranks, -1)), num_ranks)
  return arr.reshape(-1)


def _ragged_exchange_tiles(
    flat_inputs: Sequence[jax.Array],
    order: tuple[int, ...],
    my_rank: jax.Array,
    local_meta: jax.Array | None,
    in_offsets: tuple[int, ...],
    send_sizes: tuple[int, ...],
    num_ranks: int,
    m_recv: int,
    axis_name: str,
    pack_pre: bool,
) -> jax.Array:
  """Packs tiles and executes `ragged_all_to_all` on `axis_name`."""
  tile_rows = 4 if pack_pre else 8
  if pack_pre:
    # Concatenate even/odd u16 lanes separately (user_count=1 each) so XLA
    # fuses both concats into the single bit_or loop with 0 extra 128 MiB temp.
    u16s = tuple(
        jax.lax.bitcast_convert_type(
            flat_inputs[i].reshape(-1, 4, 256), jnp.uint16
        )
        for i in order
    )
    lo = (
        u16s[0][..., 0::2]
        if len(u16s) == 1
        else jnp.concatenate(tuple(p[..., 0::2] for p in u16s), axis=0)
    )
    hi = (
        u16s[0][..., 1::2]
        if len(u16s) == 1
        else jnp.concatenate(tuple(p[..., 1::2] for p in u16s), axis=0)
    )
    send_tiles = lo.astype(jnp.uint32) | (
        hi.astype(jnp.uint32) << jnp.uint32(16)
    )
  else:
    send_tiles = jnp.concatenate(
        tuple(flat_inputs[i].reshape(-1, 8, 128) for i in order), axis=0
    )
  if local_meta is not None:
    meta_2d = jnp.squeeze(local_meta, axis=0)
    in_off, s_sz, out_off, r_sz = meta_2d[0], meta_2d[1], meta_2d[2], meta_2d[3]
  else:
    in_off = jnp.asarray(tuple(x // 1024 for x in in_offsets), dtype=jnp.int32)
    s_sz = jnp.asarray(tuple(s // 1024 for s in send_sizes), dtype=jnp.int32)
    out_off = my_rank * s_sz
    r_sz = jnp.repeat(s_sz[my_rank], num_ranks)
  return jax.lax.ragged_all_to_all(
      send_tiles,
      jnp.zeros((num_ranks * m_recv, tile_rows, 128), dtype=send_tiles.dtype),
      in_off,
      s_sz,
      out_off,
      r_sz,
      axis_name=axis_name,
  )


@functools.partial(
    jax.jit,
    static_argnames=(
        'mesh',
        'axis_name',
        'order',
        'in_offsets',
        'send_sizes',
        'max_recv',
        'use_ragged',
    ),
)
def _jit_exchange_shards_all_to_all(
    local_shards: tuple[jax.Array, ...],
    mesh: jax.sharding.Mesh,
    axis_name: str,
    order: tuple[int, ...],
    in_offsets: tuple[int, ...],
    send_sizes: tuple[int, ...],
    max_recv: int,
    use_ragged: bool,
    perm: jax.Array | None = None,
    ragged_meta: jax.Array | None = None,
) -> jax.Array:
  """Jitted shard_map collective for 1D ragged or padded all-to-all."""
  num_ranks = mesh.shape[axis_name]
  m_recv = max_recv // 1024

  def _collective_body(
      flat_inputs: Sequence[jax.Array],
      local_perm: jax.Array | None,
      local_meta: jax.Array | None,
  ) -> jax.Array:
    my_rank = jax.lax.axis_index(axis_name)
    pack_pre = (
        use_ragged
        and jax.default_backend() == 'tpu'
        and all(a.dtype == jnp.bfloat16 for a in flat_inputs)
        and sum(flat_inputs[i].size for i in order) <= num_ranks * max_recv
    )
    if use_ragged:
      raw_tiles = _ragged_exchange_tiles(
          flat_inputs,
          order,
          my_rank,
          local_meta,
          in_offsets,
          send_sizes,
          num_ranks,
          m_recv,
          axis_name,
          pack_pre,
      )
    elif local_meta is not None:
      send_tiles = jnp.concatenate(
          tuple(flat_inputs[i].reshape(-1, 8, 128) for i in order), axis=0
      )
      padded = jnp.pad(send_tiles, ((0, m_recv), (0, 0), (0, 0)))
      in_off = jnp.squeeze(local_meta, axis=0)[0]
      rows = [
          jax.lax.dynamic_slice_in_dim(padded, in_off[r], m_recv, axis=0)
          for r in range(num_ranks)
      ]
      raw_tiles = jax.lax.all_to_all(
          jnp.stack(rows, axis=0),
          axis_name=axis_name,
          split_axis=0,
          concat_axis=0,
      ).reshape(num_ranks * m_recv, 8, 128)
    else:
      send_buf = jnp.concatenate(
          tuple(map(jnp.ravel, (flat_inputs[i] for i in order))), axis=0
      )
      rows = [
          jnp.pad(
              send_buf[in_offsets[r] : in_offsets[r] + send_sizes[r]],
              (0, max_recv - send_sizes[r]),
          )
          for r in range(num_ranks)
      ]
      raw_tiles = jax.lax.all_to_all(
          jnp.stack(rows, axis=0),
          axis_name=axis_name,
          split_axis=0,
          concat_axis=0,
      ).reshape(num_ranks * m_recv, 8, 128)
    if local_perm is not None:
      raw_tiles = raw_tiles[local_perm.reshape(-1)]
    if pack_pre:
      return _pad_u32_d2h(raw_tiles)
    return _pack_bf16_to_u32_d2h(raw_tiles)

  in_specs = (jax.sharding.PartitionSpec(axis_name),) * len(local_shards)
  perm_spec = None if perm is None else jax.sharding.PartitionSpec(axis_name)
  meta_spec = (
      None if ragged_meta is None else jax.sharding.PartitionSpec(axis_name)
  )
  return jax.shard_map(
      _collective_body,
      mesh=mesh,
      in_specs=(in_specs, perm_spec, meta_spec),
      out_specs=jax.sharding.PartitionSpec(axis_name),
      check_vma=False,
  )(local_shards, perm, ragged_meta)


@functools.partial(jax.jit, static_argnames=('mesh', 'axis_name'))
def _jit_concat_local_rank_shards(
    local_shards: tuple[jax.Array, ...],
    mesh: jax.sharding.Mesh,
    axis_name: str,
) -> jax.Array:
  """Concatenates identity-routed rank shards into 1D linear HBM layout."""
  num_ranks = mesh.shape[axis_name]
  shd = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(axis_name))
  parts = tuple(s.reshape(num_ranks, -1, 128) for s in local_shards)
  out = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=1)
  return jax.lax.with_sharding_constraint(
      _pack_bf16_to_u32_d2h(out, num_ranks), shd
  )


@functools.partial(
    jax.jit,
    static_argnames=(
        'shapes_2d',
        'transpose_axes',
        'output_dtype',
        'out_sharding',
        'in_shardings',
    ),
)
def _jit_transform_and_concat_fusable_wave(
    base_arrs: tuple[jax.Array, ...],
    shapes_2d: tuple[tuple[int, ...] | None, ...],
    transpose_axes: tuple[tuple[int, ...] | None, ...],
    output_dtype: jnp.dtype | None,
    out_sharding: jax.sharding.NamedSharding,
    in_shardings: tuple[jax.sharding.Sharding | None, ...] | None = None,
) -> jax.Array:
  """Fuses transpose, dtype cast, and 1D linear concat in one HBM pass."""
  num_ranks = out_sharding.mesh.size
  src_shds = in_shardings or (None,) * len(base_arrs)
  parts = []
  for arr, s2d, axes, in_shd in zip(
      base_arrs, shapes_2d, transpose_axes, src_shds
  ):
    resharded = _reshard_to_rank_generic(
        arr, in_shd, s2d, axes, output_dtype, out_sharding
    )
    if resharded is not None:
      parts.append(resharded)
      continue
    x = jnp.reshape(arr, s2d or arr.shape)
    ax = axes or tuple(range(x.ndim))
    x = jnp.transpose(x, ax).astype(output_dtype or x.dtype)
    parts.append(jnp.reshape(x, (num_ranks, -1, 128)))
  out = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=1)
  return jax.lax.with_sharding_constraint(
      _pack_bf16_to_u32_d2h(out, num_ranks), out_sharding
  )


def _validate_owner_groups(
    owner_groups: Sequence[tuple[int, ...]], num_shards: int, num_ranks: int
) -> None:
  """Validates multi-owner `owner_groups` invariants for direct public calls."""
  if len(owner_groups) != num_shards or any(not g for g in owner_groups):
    raise ValueError(
        'owner_groups must match local_shards length with non-empty groups.'
    )
  rank_counts = collections.Counter()
  for grp in owner_groups:
    rank_counts.update(grp)
  for grp in owner_groups:
    if len(grp) > 1 and (
        num_ranks % len(grp) != 0 or any(rank_counts[r] != 1 for r in grp)
    ):
      raise ValueError(
          'Multi-owner group must divide num_ranks and own exclusive ranks:'
          f' group={grp}, num_ranks={num_ranks}.'
      )


def _compute_exchange_plan(
    local_shards: Sequence[Any],
    owner_ranks: Sequence[int],
    owner_groups: Sequence[tuple[int, ...]] | None,
    num_ranks: int,
) -> tuple[
    tuple[Any, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    int,
    bool,
    bool,
    list[list[int]],
    np.ndarray | None,
]:
  """Computes pure-NumPy all-to-all offsets, metadata, and alignment checks."""
  if len(local_shards) != len(owner_ranks):
    raise ValueError('local_shards and owner_ranks must have identical length')
  order = np.argsort(owner_ranks, kind='stable')
  ordered_shards = tuple(local_shards[int(i)] for i in order)
  weights = [math.prod(a.shape) // num_ranks for a in local_shards]
  send_sizes_arr = np.bincount(
      owner_ranks, weights=weights, minlength=num_ranks
  ).astype(np.int32)
  in_offsets = np.cumsum([0] + send_sizes_arr[:-1].tolist()).astype(np.int32)
  if owner_groups is None:
    meta_np, max_recv = None, int(np.max(send_sizes_arr, initial=0))
  else:
    _validate_owner_groups(owner_groups, len(local_shards), num_ranks)
    meta_np, max_recv = build_grouped_ragged_meta(
        [weights[int(i)] for i in order],
        [owner_groups[int(i)] for i in order],
        num_ranks,
    )
  rank_cols: list[list[int]] = [[] for _ in range(num_ranks)]
  for idx in order:
    if not owner_groups or len(owner_groups[int(idx)]) == 1:
      rank_cols[owner_ranks[int(idx)]].append(weights[int(idx)])
  bad_weights = any(
      w <= 0 or math.prod(a.shape) % (num_ranks * 1024) != 0
      for a, w in zip(local_shards, weights)
  )
  if (
      max_recv <= 0
      or max_recv % 1024 != 0
      or bad_weights
      or bool(np.any(send_sizes_arr % 1024))
  ):
    raise ValueError(
        'exchange_shards_ragged_all_to_all requires 1024-aligned per-rank'
        f' shards: weights={weights}, send_sizes={send_sizes_arr.tolist()},'
        f' max_recv={max_recv}'
    )
  use_ragged = jax.default_backend() == 'tpu' or meta_np is not None
  need_perm = num_ranks > 1 and any(
      len(c) > 1 or (not use_ragged and 0 < sum(c) < max_recv)
      for c in rank_cols
  )
  return (
      ordered_shards,
      tuple(range(len(ordered_shards))),
      tuple(in_offsets.tolist()),
      tuple(send_sizes_arr.tolist()),
      max_recv,
      use_ragged,
      need_perm,
      rank_cols,
      meta_np,
  )


def exchange_shards_ragged_all_to_all(
    local_shards: Sequence[jax.Array],
    owner_ranks: Sequence[int],
    mesh: jax.sharding.Mesh,
    axis_name: str,
    owner_groups: Sequence[tuple[int, ...]] | None = None,
    compiled_plan: CompiledSavePlan | None = None,
) -> jax.Array:
  """Exchanges 1D sharded tensors to their owner ranks via `all_to_all`.

  Args:
    local_shards: Sequence of 1D rank-sharded device arrays to exchange.
    owner_ranks: Primary owner rank for each tensor in `local_shards`.
    mesh: 1D `jax.sharding.Mesh` spanning all participating ranks.
    axis_name: Mesh axis name (typically `'rank'`).
    owner_groups: Optional per-tensor multi-rank owner groups.
    compiled_plan: Optional `CompiledSavePlan` for AOT-compiled routing.

  Returns:
    Rank-sharded 1D `jax.Array` holding each rank's received tensor bytes.

  Example:
    ```python
    recv = safetensors_saver.exchange_shards_ragged_all_to_all(
        local_shards, owner_ranks, rank_mesh, 'rank'
    )
    ```
  """
  if local_shards and len({jnp.dtype(a.dtype) for a in local_shards}) > 1:
    dtypes = sorted({str(jnp.dtype(a.dtype)) for a in local_shards})
    raise ValueError(
        'exchange_shards_ragged_all_to_all requires homogeneous dtypes,'
        f' got {dtypes}.'
    )
  num_ranks = mesh.shape[axis_name]
  (
      ordered_shards,
      order_tup,
      in_off_tup,
      send_sz_tup,
      max_recv,
      use_ragged,
      need_perm,
      rank_cols,
      meta_np,
  ) = _compute_exchange_plan(local_shards, owner_ranks, owner_groups, num_ranks)
  rank_shd = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec(axis_name)
  )
  ragged_meta = None
  if meta_np is not None:
    ragged_meta = jax.make_array_from_callback(
        meta_np.shape, rank_shd, lambda idx: meta_np[idx]
    )
  perm = None
  if need_perm:
    m_recv = max_recv // 1024
    perm_np = np.stack([
        build_tile_perm_row(
            [w // 1024 for w in rank_cols[r]], num_ranks, m_recv, use_ragged
        )
        for r in range(num_ranks)
    ])
    perm = jax.make_array_from_callback(
        (num_ranks, num_ranks * m_recv), rank_shd, lambda idx: perm_np[idx]
    )
  key = _exchange_key(
      ordered_shards,
      mesh,
      axis_name,
      order_tup,
      in_off_tup,
      send_sz_tup,
      max_recv,
      use_ragged,
      perm,
      ragged_meta,
  )
  if compiled_plan is not None and key in compiled_plan.executables:
    return compiled_plan.executables[key](ordered_shards, perm, ragged_meta)
  return _jit_exchange_shards_all_to_all(
      ordered_shards,
      mesh,
      axis_name,
      order_tup,
      in_off_tup,
      send_sz_tup,
      max_recv,
      use_ragged,
      perm,
      ragged_meta,
  )


def unpack_host_tensor(
    spec: TensorExportSpec, host_arr: np.ndarray
) -> dict[str, np.ndarray]:
  """Unpacks a host array into C-contiguous safetensors entries.

  When `spec.unstack_axis` is set, splits the expert axis into individual
  `spec.hf_key.format(expert=i)` entries respecting `spec.unstack_range`.

  Args:
    spec: `TensorExportSpec` describing naming and optional expert unstacking.
    host_arr: Host NumPy array materialized from device.

  Returns:
    Dictionary mapping concrete HuggingFace tensor keys to C-contiguous arrays.

  Example:
    ```python
    entries = safetensors_saver.unpack_host_tensor(spec, host_arr)
    ```
  """
  if spec.unstack_axis is None:
    return {spec.hf_key: np.ascontiguousarray(host_arr)}
  start = spec.unstack_range[0] if spec.unstack_range is not None else 0
  count = host_arr.shape[spec.unstack_axis]
  return {
      spec.hf_key.format(expert=start + idx): np.ascontiguousarray(
          host_arr[idx]
      )
      for idx in range(count)
  }


def _pop_or_transform_chunk(
    flat_state: Mapping[str, jax.Array],
    spec: TensorExportSpec,
    output_dtype: jnp.dtype | None,
    out_shd: jax.sharding.NamedSharding | None,
    chunk_cache: dict[str, list[jax.Array | None]],
    compiled_plan: CompiledSavePlan | None = None,
) -> jax.Array:
  """Pops a cached HBM chunk or transforms and splits `spec.source_key` once."""
  base_arr = flat_state[spec.source_key]
  lora_a = flat_state[spec.lora_a_key] if spec.lora_a_key else None
  lora_b = flat_state[spec.lora_b_key] if spec.lora_b_key else None
  num_chunks, chunk_idx = _chunk_count_and_index(base_arr, spec)
  if spec.source_key not in chunk_cache:
    shd, flatten_ranks = _resolve_transform_sharding(
        base_arr, spec, out_shd, num_chunks
    )
    key = _transform_hbm_key(
        base_arr,
        lora_a,
        lora_b,
        spec.shape_2d,
        spec.transpose_axes,
        output_dtype,
        shd,
        flatten_ranks,
        num_chunks,
    )
    if compiled_plan is not None and key in compiled_plan.executables:
      chunks = compiled_plan.executables[key](
          base_arr, lora_a, lora_b, float(spec.lora_scaling)
      )
    else:
      chunks = _jit_transform_on_hbm(
          base_arr,
          lora_a,
          lora_b,
          float(spec.lora_scaling),
          spec.shape_2d,
          spec.transpose_axes,
          output_dtype,
          shd,
          flatten_ranks,
          num_chunks,
          base_arr.sharding,
      )
    chunk_cache[spec.source_key] = list(
        chunks if isinstance(chunks, tuple) else (chunks,)
    )
  entries = chunk_cache[spec.source_key]
  cached = entries[chunk_idx]
  entries[chunk_idx] = None
  if not any(x is not None for x in entries) and spec.source_key in chunk_cache:
    del chunk_cache[spec.source_key]
  if cached is not None:
    return cached
  return transform_tensor_on_hbm(
      base_arr,
      spec,
      lora_a=lora_a,
      lora_b=lora_b,
      output_dtype=output_dtype,
      out_sharding=out_shd,
      compiled_plan=compiled_plan,
  )


def _is_noop_transform(
    base: jax.Array | jax.ShapeDtypeStruct,
    spec: TensorExportSpec,
    output_dtype: jnp.dtype | None,
    out_shd: jax.sharding.NamedSharding | None,
) -> bool:
  """Returns True when `base` already matches target shape, dtype, and sharding."""
  target_dt = jnp.dtype(output_dtype or base.dtype)
  if base.dtype != target_dt or {
      spec.lora_a_key,
      spec.shape_2d,
      spec.transpose_axes,
      spec.unstack_range,
  } != {None}:
    return False
  is_sharded = isinstance(base.sharding, jax.sharding.NamedSharding) and any(
      base.sharding.spec
  )
  if out_shd is None:
    return not is_sharded
  return (
      base.sharding == out_shd
      and base.ndim == 2
      and math.prod(base.shape) % (len(out_shd.device_set) * 1024) == 0
  )


def transform_wave_spec(
    flat_state: Mapping[str, jax.Array],
    rank: int,
    spec: TensorExportSpec,
    output_dtype: jnp.dtype | None,
    devices: Sequence[jax.Device] | None,
    rank_shd: jax.sharding.NamedSharding | None = None,
    chunk_cache: dict[str, list[jax.Array | None]] | None = None,
    compiled_plan: CompiledSavePlan | None = None,
) -> jax.Array:
  """Transforms a single spec on HBM for its assigned wave rank.

  Args:
    flat_state: Flat parameter dictionary supplying source device arrays.
    rank: Target owner rank index for this spec.
    spec: `TensorExportSpec` to transform.
    output_dtype: Optional cast dtype (`None` keeps source dtype).
    devices: Ordered mesh devices, or `None` for single-host execution.
    rank_shd: Optional 1D `NamedSharding` over `('rank',)` for routing.
    chunk_cache: Optional cache sharing pre-sliced MoE chunks across waves.
    compiled_plan: Optional `CompiledSavePlan` for AOT execution.

  Returns:
    Transformed `jax.Array` on HBM.

  Example:
    ```python
    arr = safetensors_saver.transform_wave_spec(
        flat_state, rank=0, spec=spec, output_dtype=jnp.bfloat16,
        devices=devices,
    )
    ```
  """
  del rank, devices
  base_arr = flat_state[spec.source_key]
  out_shd = None
  if rank_shd is not None and _uses_multi_rank_routing(
      spec, base_arr, len(rank_shd.device_set)
  ):
    out_shd = rank_shd
  if spec.unstack_range is not None and chunk_cache is not None:
    return _pop_or_transform_chunk(
        flat_state, spec, output_dtype, out_shd, chunk_cache, compiled_plan
    )
  if _is_noop_transform(base_arr, spec, output_dtype, out_shd):
    return base_arr
  return transform_tensor_on_hbm(
      base_arr,
      spec,
      lora_a=flat_state[spec.lora_a_key] if spec.lora_a_key else None,
      lora_b=flat_state[spec.lora_b_key] if spec.lora_b_key else None,
      output_dtype=output_dtype,
      out_sharding=out_shd,
      compiled_plan=compiled_plan,
  )


_OversizedEntry = tuple[int, TensorExportSpec, Any, int]


@dataclasses.dataclass(frozen=True)
class _WaveD2HHandle:
  """Pending asynchronous D2H transfer handle for 2-wave overlap."""

  rep_specs: tuple[tuple[int, TensorExportSpec], ...]
  rep_bufs: tuple[jax.Array, ...]
  defer_contiguous: bool
  hbm_sec: float
  num_ranks: int = 1
  shd_batches: tuple[
      tuple[
          tuple[jax.Array, ...],
          tuple[int, ...],
          Mapping[
              int, Sequence[tuple[int, TensorExportSpec, tuple[int, ...]]]
          ],
          bool,
      ],
      ...,
  ] = ()
  pair_order: tuple[tuple[int, str], ...] = ()
  oversized_batches: tuple[tuple[tuple[_OversizedEntry, ...], int], ...] = ()
  devices: tuple[jax.Device, ...] = ()
  compiled_plan: CompiledSavePlan | None = None


def _resolve_rank_mesh_shd(
    devices: Sequence[jax.Device], compiled_plan: CompiledSavePlan | None
) -> tuple[jax.sharding.Mesh, jax.sharding.NamedSharding]:
  """Returns the 1D rank mesh and sharding, reusing `compiled_plan` if set."""
  if (
      compiled_plan is not None
      and compiled_plan.rank_mesh is not None
      and compiled_plan.rank_shd is not None
  ):
    return compiled_plan.rank_mesh, compiled_plan.rank_shd
  rank_mesh = jax.sharding.Mesh(np.array(devices), ('rank',))
  rank_shd = jax.sharding.NamedSharding(
      rank_mesh, jax.sharding.PartitionSpec('rank')
  )
  return rank_mesh, rank_shd


def _split_oversized_batch(
    pairs: Sequence[tuple[int, TensorExportSpec]],
    arrays: Sequence[Any],
    num_ranks: int,
    max_bucket_bytes: int | None,
) -> tuple[
    list[tuple[int, TensorExportSpec]], list[Any], list[_OversizedEntry], int
]:
  """Splits oversized non-unstackable sharded specs into tile-aligned steps."""
  if (
      not pairs
      or num_ranks <= 1
      or not max_bucket_bytes
      or max_bucket_bytes < 0
  ):
    return list(pairs), list(arrays), [], 0
  max_tiles = max_bucket_bytes // (num_ranks * 1024 * arrays[0].dtype.itemsize)
  if max_tiles < 1:
    return list(pairs), list(arrays), [], 0
  norm_pairs, norm_arrays, oversized = [], [], []
  for (r, spec), arr in zip(pairs, arrays):
    total_elems = math.prod(arr.shape)
    if (
        spec.unstack_axis is None
        and spec.unstack_range is None
        and _array_nbytes(arr) > max_bucket_bytes
        and total_elems % (num_ranks * 1024) == 0
    ):
      oversized.append((r, spec, arr, total_elems // (num_ranks * 1024)))
    else:
      norm_pairs.append((r, spec))
      norm_arrays.append(arr)
  oversized.sort(key=lambda e: e[0] % num_ranks)
  return norm_pairs, norm_arrays, oversized, (max_tiles if oversized else 0)


def _ensure_rank_sharding(
    arr: Any, rank_shd: jax.sharding.NamedSharding
) -> Any:
  if arr.sharding == rank_shd:
    return arr
  if isinstance(arr, jax.ShapeDtypeStruct):
    return jax.ShapeDtypeStruct(arr.shape, arr.dtype, sharding=rank_shd)
  return jax.device_put(arr, rank_shd)


def _oversized_exchange_args(
    oversized: Sequence[_OversizedEntry],
    step_tiles: int,
    rank_mesh: jax.sharding.Mesh,
    rank_shd: jax.sharding.NamedSharding,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
  """Returns `(ex_key, lower_args)` for 1-kernel dynamic ragged A2A."""
  num_ranks = rank_mesh.size
  shards = tuple(_ensure_rank_sharding(a, rank_shd) for _, _, a, _ in oversized)
  order, zeros = tuple(range(len(shards))), (0,) * num_ranks
  meta_sds = jax.ShapeDtypeStruct(
      (num_ranks, 4, num_ranks), jnp.int32, sharding=rank_shd
  )
  args = (
      shards, rank_mesh, 'rank', order, zeros, zeros, step_tiles * 1024,
      jax.default_backend() == 'tpu', None, meta_sds,
  )
  return _exchange_key(*args), args


def _build_oversized_step_meta(
    owner_spans: Mapping[int, tuple[int, int]],
    num_ranks: int,
    t0: int,
    step_tiles: int,
) -> tuple[np.ndarray, dict[int, int]]:
  """Builds per-step `(num_ranks, 4, num_ranks)` ragged_all_to_all metadata."""
  in_off = np.zeros((num_ranks, num_ranks), dtype=np.int32)
  send_sz = np.zeros((num_ranks, num_ranks), dtype=np.int32)
  out_off = np.zeros((num_ranks, num_ranks), dtype=np.int32)
  step_cnts: dict[int, int] = {}
  ranks_col = np.arange(num_ranks, dtype=np.int32)
  for ow, (ow_base, tot_ow) in owner_spans.items():
    cnt = min(step_tiles, tot_ow - t0)
    if cnt > 0:
      in_off[:, ow], send_sz[:, ow] = ow_base + t0, cnt
      out_off[:, ow] = ranks_col * cnt
      step_cnts[ow] = cnt
  return np.stack([in_off, send_sz, out_off, send_sz.T], axis=1), step_cnts


def _materialize_d2h_targets(
    targets: Sequence[jax.Array],
) -> list[np.ndarray]:
  """Materializes local device D2H targets in parallel across devices."""
  if len(targets) > 1:
    return list(_SHARED_WRITE_POOL.map(np.asarray, targets))
  return [np.asarray(x) for x in targets]


def _d2h_buf_to_1d(
    arr: np.ndarray, *, is_packed_bf16: bool = False
) -> np.ndarray:
  """Flattens a host D2H buffer, zero-copy decoding `(M, 128)` uint32 bf16."""
  if (
      is_packed_bf16
      and arr.dtype == np.uint32
      and arr.ndim == 2
      and arr.shape[1] == 128
  ):
    return arr.reshape(-1).view(jnp.bfloat16.dtype)
  return arr.reshape(-1)


def _scatter_oversized_step(
    active: Sequence[tuple[int, int, jax.Array]],
    targets: Sequence[jax.Array],
    owner_items: Mapping[int, Sequence[tuple[str, int, int]]],
    assembled: Mapping[tuple[int, str], np.ndarray],
    num_ranks: int,
    t0: int,
    step_tiles: int,
    is_tpu: bool,
    *,
    is_packed_bf16: bool = False,
) -> None:
  """Copies one streamed step's received tiles into pre-allocated host arrays."""
  local_bufs = _materialize_d2h_targets(targets)
  for (ow, cnt, _), raw_buf in zip(active, local_bufs):
    stride = cnt if is_tpu else step_tiles
    buf = _d2h_buf_to_1d(raw_buf, is_packed_bf16=is_packed_bf16)[
        : num_ranks * stride * 1024
    ].reshape(num_ranks, stride, 1024)[:, :cnt, :]
    for key, nt, rel_s in owner_items[ow]:
      lo, hi = max(t0, rel_s), min(t0 + cnt, rel_s + nt)
      if lo < hi:
        assembled[(ow, key)][:, lo - rel_s : hi - rel_s, :] = buf[
            :, lo - t0 : hi - t0, :
        ]


def _dispatch_oversized_step(
    step: int,
    step_tiles: int,
    owner_spans: Mapping[int, tuple[int, int]],
    num_ranks: int,
    rank_shd: jax.sharding.NamedSharding,
    exec_fn: Any,
    shards: tuple[jax.Array, ...],
    lower_args: tuple[Any, ...],
    dev_to_rank: Mapping[jax.Device, int],
    is_tpu: bool,
) -> tuple[
    int,
    list[tuple[int, int, jax.Array]],
    tuple[jax.Array, ...],
    jax.Array,
    bool,
    float,
]:
  """Dispatches one oversized ragged A2A step and starts async D2H."""
  t_h, t0 = time.perf_counter(), step * step_tiles
  meta_np, step_cnts = _build_oversized_step_meta(
      owner_spans, num_ranks, t0, step_tiles
  )
  ragged_meta = jax.make_array_from_callback(
      meta_np.shape, rank_shd, lambda idx, m=meta_np: m[idx]
  )
  exchanged = (
      exec_fn(shards, None, ragged_meta)
      if exec_fn is not None
      else _jit_exchange_shards_all_to_all(*lower_args[:-1], ragged_meta)
  )
  is_packed_bf16 = (
      shards[0].dtype == jnp.bfloat16 and exchanged.dtype == jnp.uint32
  )
  active = [
      (dev_to_rank[s.device], step_cnts[dev_to_rank[s.device]], s.data)
      for s in exchanged.addressable_shards
      if s.device in dev_to_rank and dev_to_rank[s.device] in step_cnts
  ]
  targets = tuple(
      _slice_d2h_target(
          data,
          num_ranks * (cnt if is_tpu else step_tiles) * 1024,
          is_tpu,
          is_packed_bf16=is_packed_bf16,
      )
      for _, cnt, data in active
  )
  for t in targets:
    t.copy_to_host_async()
  return (
      t0,
      active,
      targets,
      exchanged,
      is_packed_bf16,
      time.perf_counter() - t_h,
  )


def _stream_oversized_batch_to_host(
    oversized: Sequence[_OversizedEntry],
    step_tiles: int,
    devices: Sequence[jax.Device],
    compiled_plan: CompiledSavePlan | None,
) -> tuple[dict[int, dict[str, HostTensor]], float, float]:
  """Streams oversized rank-sharded specs via 1-kernel dynamic ragged A2A."""
  num_ranks = len(devices)
  rank_mesh, rank_shd = _resolve_rank_mesh_shd(devices, compiled_plan)
  ex_key, lower_args = _oversized_exchange_args(
      oversized, step_tiles, rank_mesh, rank_shd
  )
  shards = lower_args[0]
  owner_items: dict[int, list[tuple[str, int, int]]] = collections.defaultdict(
      list
  )
  owner_spans: dict[int, tuple[int, int]] = {}
  orig_shapes: dict[tuple[int, str], tuple[int, ...]] = {}
  proc, dt = jax.process_index(), np.dtype(shards[0].dtype)
  assembled: dict[tuple[int, str], np.ndarray] = {}
  cursor = 0
  for r, spec, arr, nt in oversized:
    ow, key = r % num_ranks, _wave_tensor_key(spec)
    ow_base, rel_s = owner_spans.get(ow, (cursor, 0))
    owner_items[ow].append((key, nt, rel_s))
    owner_spans[ow], orig_shapes[(ow, key)] = (ow_base, rel_s + nt), arr.shape
    if devices[ow].process_index == proc:
      assembled[(ow, key)] = np.empty((num_ranks, nt, 1024), dtype=dt)
    cursor += nt
  dev_to_rank = {devices[ow]: ow for ow in owner_items}
  is_tpu = jax.default_backend() == 'tpu'
  exec_fn = compiled_plan.executables.get(ex_key) if compiled_plan else None
  num_steps = max(
      math.ceil(tot / step_tiles) for _, tot in owner_spans.values()
  )
  hbm_sec = d2h_sec = 0.0
  pending = None
  for step in range(num_steps + 1):
    nxt = None
    if step < num_steps:
      t0, active, targets, exchanged, is_packed, step_hbm = (
          _dispatch_oversized_step(
              step,
              step_tiles,
              owner_spans,
              num_ranks,
              rank_shd,
              exec_fn,
              shards,
              lower_args,
              dev_to_rank,
              is_tpu,
          )
      )
      hbm_sec += step_hbm
      nxt = (t0, active, targets, exchanged, is_packed)
    if pending is not None:
      prev_t0, prev_active, prev_targets, _, prev_packed = pending
      t_d = time.perf_counter()
      _scatter_oversized_step(
          prev_active,
          prev_targets,
          owner_items,
          assembled,
          num_ranks,
          prev_t0,
          step_tiles,
          is_tpu,
          is_packed_bf16=prev_packed,
      )
      d2h_sec += time.perf_counter() - t_d
    pending = nxt
  host_out: dict[int, dict[str, HostTensor]] = collections.defaultdict(dict)
  for (ow, key), full in assembled.items():
    host_out[ow][key] = full.reshape(orig_shapes[(ow, key)])
  return dict(host_out), hbm_sec, d2h_sec


def _partition_by_dtype(
    pairs: Sequence[tuple[int, TensorExportSpec]],
    arrays: Sequence[Any],
) -> list[tuple[list[tuple[int, TensorExportSpec]], list[Any]]]:
  """Groups sharded `(pairs, arrays)` by `dtype` so `jnp.concatenate` matches."""
  if len(arrays) <= 1 or len({a.dtype for a in arrays}) == 1:
    return [(list(pairs), list(arrays))] if pairs else []
  by_dt: dict[Any, tuple[list[tuple[int, TensorExportSpec]], list[Any]]] = {}
  for pair, arr in zip(pairs, arrays):
    dt = arr.dtype
    if dt not in by_dt:
      by_dt[dt] = ([], [])
    by_dt[dt][0].append(pair)
    by_dt[dt][1].append(arr)
  return list(by_dt.values())


def _spec_export_elems(
    spec: TensorExportSpec,
    flat_state: Mapping[str, Any] | None,
    fallback_elem_bytes: int = 1,
) -> int:
  """Returns exported element count for `spec` from `flat_state` shape."""
  if flat_state is None or spec.source_key not in flat_state:
    return spec.nbytes // fallback_elem_bytes
  base = flat_state[spec.source_key]
  raw_s = spec.shape_2d or base.shape
  total_elems = int(math.prod(raw_s))
  if spec.unstack_range is None:
    return total_elems
  ax = spec.transpose_axes or tuple(range(len(raw_s)))
  span = spec.unstack_range[1] - spec.unstack_range[0]
  return (total_elems // raw_s[ax[0]]) * span


def _group_batch_shards(
    pairs: Sequence[tuple[int, TensorExportSpec]],
    dev_arrays: Sequence[Any],
    num_ranks: int,
    rank_shd: jax.sharding.NamedSharding,
    *,
    is_packed_bf16: bool = False,
    flat_state: Mapping[str, Any] | None = None,
) -> tuple[
    tuple[Any, ...],
    list[int],
    tuple[tuple[int, ...], ...] | None,
    dict[int, list[tuple[int, TensorExportSpec, tuple[int, ...]]]],
]:
  """Deduplicates fused multi-chunk arrays and builds owner rank groups."""
  packed_mult = 2 if is_packed_bf16 else 1
  grouped: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
  chunk_elems_list: list[int] = []
  for (rk, spec), a in zip(pairs, dev_arrays):
    elem_bytes = 2 if is_packed_bf16 else a.dtype.itemsize
    a_elems = int(math.prod(a.shape)) * packed_mult
    c_elems = _spec_export_elems(spec, flat_state, elem_bytes)
    chunk_elems_list.append(c_elems)
    if spec.unstack_range and a_elems > c_elems > 0:
      grouped[spec.source_key].append((spec.unstack_range[0], rk % num_ranks))
  raw_sharded: list[Any] = []
  owners: list[int] = []
  owner_groups: list[tuple[int, ...]] = []
  seen_src: set[str] = set()
  seen_buf_ids: set[int] = set()
  rank_items: dict[
      int, list[tuple[int, TensorExportSpec, tuple[int, ...]]]
  ] = collections.defaultdict(list)
  for ((rk, spec), a), c_elems in zip(zip(pairs, dev_arrays), chunk_elems_list):
    dest = rk % num_ranks
    if spec.source_key not in grouped:
      raw_sharded.append(a)
      owners.append(dest)
      owner_groups.append((dest,))
      rank_items[dest].append((rk, spec, a.shape))
      continue
    rank_items[dest].append((rk, spec, (c_elems,)))
    # Multi-source fused identity waves share one packed buffer across K
    # source keys (`[fused] * len(pairs)`); deduplicate by source and buffer
    # identity so the single packed buffer is not concatenated K times.
    if spec.source_key in seen_src or id(a) in seen_buf_ids:
      continue
    seen_src.add(spec.source_key)
    seen_buf_ids.add(id(a))
    grp = tuple(r for _, r in sorted(grouped[spec.source_key]))
    raw_sharded.append(a)
    owners.append(grp[0])
    owner_groups.append(grp)
  sharded = tuple(_ensure_rank_sharding(a, rank_shd) for a in raw_sharded)
  return (
      sharded,
      owners,
      tuple(owner_groups) if grouped else None,
      rank_items,
  )


def _slice_d2h_target(
    data: jax.Array,
    cut_elems: int,
    is_tpu: bool,
    *,
    is_packed_bf16: bool = False,
) -> jax.Array:
  """Slices under-filled TPU D2H shard while preserving 2D `(M, 128)` layout."""
  is_u32_bf16 = (
      is_packed_bf16
      and data.ndim == 2
      and data.shape[1] == 128
      and data.dtype == jnp.uint32
  )
  eff_cut = (cut_elems + 1) // 2 if is_u32_bf16 else cut_elems
  if not (is_tpu and 0 < eff_cut < data.size):
    return data
  if is_u32_bf16 and data.size - eff_cut <= 1024:
    return data
  if data.ndim == 2 and data.shape[1] == 128:
    rows = ((eff_cut + 127) // 128 + 7) // 8 * 8
    return data[: min(data.shape[0], rows)]
  return data.reshape(-1)[:eff_cut]


def _is_packed_identity_shard(arr: Any) -> bool:
  """Returns True when a single identity-routed shard is already D2H-packed."""
  return arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 128)


def _start_batch_d2h_with_layout(
    pairs: Sequence[tuple[int, TensorExportSpec]],
    dev_arrays: Sequence[jax.Array],
    devices: Sequence[jax.Device],
    compiled_plan: CompiledSavePlan | None = None,
    *,
    is_packed_bf16: bool = False,
    flat_state: Mapping[str, Any] | None = None,
) -> tuple[
    tuple[jax.Array, ...],
    tuple[int, ...],
    dict[int, list[tuple[int, TensorExportSpec, tuple[int, ...]]]],
    float,
]:
  """Executes HBM+ICI all-to-all with on-HBM assembly and initiates D2H."""
  t0 = time.perf_counter()
  num_ranks = len(devices)
  rank_mesh, rank_shd = _resolve_rank_mesh_shd(devices, compiled_plan)
  sharded, owners, owner_groups, rank_items = _group_batch_shards(
      pairs,
      dev_arrays,
      num_ranks,
      rank_shd,
      is_packed_bf16=is_packed_bf16,
      flat_state=flat_state,
  )
  is_identity = bool(owner_groups) and set(owner_groups) == {
      tuple(range(num_ranks))
  }
  compiled_exec = compiled_plan.executables if compiled_plan is not None else {}
  if (
      is_identity
      and len(sharded) == 1
      and _is_packed_identity_shard(sharded[0])
  ):
    exchanged = sharded[0]
  elif is_identity:
    cat_key = _concat_local_key(sharded, rank_mesh, 'rank')
    exchanged = (
        compiled_exec[cat_key](sharded)
        if cat_key in compiled_exec
        else _jit_concat_local_rank_shards(sharded, rank_mesh, 'rank')
    )
  else:
    exchanged = exchange_shards_ragged_all_to_all(
        sharded,
        owners,
        rank_mesh,
        'rank',
        owner_groups=owner_groups,
        compiled_plan=compiled_plan,
    )
  out_packed_bf16 = is_packed_bf16 or (
      bool(sharded)
      and sharded[0].dtype == jnp.bfloat16
      and exchanged.dtype == jnp.uint32
  )
  dev_to_rank = {devices[r]: r for r in rank_items}
  is_tpu = jax.default_backend() == 'tpu'
  active_shards = [
      (dev_to_rank[s.device], s.data)
      for s in exchanged.addressable_shards
      if s.device in dev_to_rank
  ]
  cut = {
      r: sum(math.prod(sh) for _, _, sh in items)
      for r, items in rank_items.items()
  }
  d2h_targets = tuple(
      _slice_d2h_target(
          data, cut[r], is_tpu, is_packed_bf16=out_packed_bf16
      )
      for r, data in active_shards
  )
  for arr in d2h_targets:
    arr.copy_to_host_async()
  return (
      d2h_targets,
      tuple(r for r, _ in active_shards),
      rank_items,
      time.perf_counter() - t0,
  )


def _finish_batch_d2h(
    d2h_targets: Sequence[jax.Array],
    active_ranks: Sequence[int],
    rank_items: Mapping[
        int, Sequence[tuple[int, TensorExportSpec, tuple[int, ...]]]
    ],
    defer_contiguous: bool = False,
    *,
    is_packed_bf16: bool = False,
) -> tuple[dict[int, dict[str, HostTensor]], float]:
  """Completes parallel D2H transfer and slices contiguous per-rank tensors."""
  t1 = time.perf_counter()
  local_bufs = _materialize_d2h_targets(d2h_targets)
  host_batch: dict[int, dict[str, HostTensor]] = collections.defaultdict(dict)
  for r, buf in zip(active_ranks, local_bufs):
    buf_1d = _d2h_buf_to_1d(buf, is_packed_bf16=is_packed_bf16)
    offset = 0
    for plan_rank, spec, shape in rank_items[r]:
      step = math.prod(shape)
      lazy = _DeferredHostSlice(buf_1d, offset, offset + step, shape)
      host_batch[plan_rank][_wave_tensor_key(spec)] = (
          lazy if defer_contiguous else np.asarray(lazy)
      )
      offset += step
  return dict(host_batch), time.perf_counter() - t1


def _is_identity_rank_wave(wave: WaveBucket, num_ranks: int) -> bool:
  """Returns True iff every source in `wave` maps strictly 0..num_ranks-1."""
  ra = wave.rank_assignments
  if num_ranks <= 1 or len(ra) != num_ranks or not ra.get(0):
    return False
  keys0 = tuple(s.source_key for s in ra[0])
  if len(set(keys0)) != len(keys0):
    return False
  prev_starts = [-1] * len(keys0)
  for r in range(num_ranks):
    specs = ra.get(r, ())
    if tuple(s.source_key for s in specs) != keys0:
      return False
    starts = [
        s.unstack_range[0]
        for s in specs
        if s.unstack_range is not None and not s.is_replicated
    ]
    if len(starts) != len(keys0) or any(
        cur <= prv for prv, cur in zip(prev_starts, starts)
    ):
      return False
    prev_starts = starts
  return True


def _is_fusable_source_group(
    base: jax.Array | jax.ShapeDtypeStruct,
    items: Sequence[tuple[int, TensorExportSpec]],
    wave: WaveBucket,
    num_ranks: int,
) -> bool:
  """Checks if a multi-chunk MoE source group can route without HBM split."""
  spec0 = items[0][1]
  raw_s = spec0.shape_2d or base.shape
  ax = spec0.transpose_axes or tuple(range(len(raw_s)))
  span = sum(
      s.unstack_range[1] - s.unstack_range[0]
      for _, s in items
      if s.unstack_range is not None
  )
  # Multi-item identity-wave fusion requires every source in `wave` to map
  # chunks strictly in ascending `unstack_range[0]` order across 0..num_ranks-1;
  # otherwise non-identity grouped ragged A2A would mismatch on-HBM layout.
  identity_group = (
      [r for r, _ in items] == list(range(num_ranks))
      and _is_identity_rank_wave(wave, num_ranks)
  )
  single_rank_specs = identity_group or all(
      len(wave.rank_assignments[r]) == 1 for r, _ in items
  )
  return (
      len(items) > 1
      and num_ranks % len(items) == 0
      and span == raw_s[ax[0]]
      and _uses_multi_rank_routing(spec0, base, num_ranks)
      and math.prod(base.shape) % (num_ranks * 1024) == 0
      and single_rank_specs
  )


def _fusable_wave_sources(
    flat_state: Mapping[str, jax.Array | jax.ShapeDtypeStruct],
    wave: WaveBucket,
    num_ranks: int,
) -> set[str]:
  """Returns source_keys in `wave` eligible for zero-split grouped routing."""
  if jax.default_backend() != 'tpu' or num_ranks <= 1:
    return set()
  by_src: dict[str, list[tuple[int, TensorExportSpec]]] = (
      collections.defaultdict(list)
  )
  for r, specs in wave.rank_assignments.items():
    for s in specs:
      if s.unstack_range is not None:
        by_src[s.source_key].append((r, s))
  return {
      src
      for src, items in by_src.items()
      if _is_fusable_source_group(flat_state[src], items, wave, num_ranks)
  }


def _can_fuse_identity_wave(
    wave: WaveBucket,
    fusable: set[str],
    num_ranks: int,
    flat_state: Mapping[str, Any] | None = None,
    output_dtype: jnp.dtype | None = None,
) -> bool:
  """Checks if all wave specs can fuse into one single-pass HBM kernel."""
  if not _is_identity_rank_wave(wave, num_ranks):
    return False
  r0 = wave.rank_assignments[0]
  if not all(s.source_key in fusable and s.lora_a_key is None for s in r0):
    return False
  if output_dtype is None and flat_state is not None:
    return len({flat_state[s.source_key].dtype for s in r0}) == 1
  return True


def _transform_wave_pairs(
    flat_state: Mapping[str, jax.Array],
    wave: WaveBucket,
    pairs: Sequence[tuple[int, TensorExportSpec]],
    output_dtype: jnp.dtype | None,
    devices: Sequence[jax.Device] | None,
    rank_shd: jax.sharding.NamedSharding | None,
    chunk_cache: dict[str, list[jax.Array | None]] | None = None,
    compiled_plan: CompiledSavePlan | None = None,
) -> list[jax.Array]:
  """Transforms all `(rank, spec)` pairs in `wave` with fused or cached chunks."""
  num_ranks = len(devices) if devices else 0
  fusable = _fusable_wave_sources(flat_state, wave, num_ranks)
  if rank_shd is not None and _can_fuse_identity_wave(
      wave, fusable, num_ranks, flat_state, output_dtype
  ):
    r0 = wave.rank_assignments[0]
    base_arrs = tuple(flat_state[s.source_key] for s in r0)
    shapes_2d = tuple(s.shape_2d for s in r0)
    axes = tuple(s.transpose_axes for s in r0)
    key = _fused_wave_key(base_arrs, shapes_2d, axes, output_dtype, rank_shd)
    if compiled_plan is not None and key in compiled_plan.executables:
      fused = compiled_plan.executables[key](base_arrs)
    else:
      in_shds = tuple(a.sharding for a in base_arrs)
      fused = _jit_transform_and_concat_fusable_wave(
          base_arrs, shapes_2d, axes, output_dtype, rank_shd, in_shds
      )
    return [fused] * len(pairs)
  fused_bufs: dict[str, jax.Array] = {}
  dev_arrays: list[jax.Array] = []
  for r, s in pairs:
    if s.source_key in fusable:
      if s.source_key not in fused_bufs:
        full_spec = dataclasses.replace(s, unstack_range=None)
        fused_bufs[s.source_key] = transform_wave_spec(
            flat_state,
            r,
            full_spec,
            output_dtype,
            devices,
            rank_shd,
            compiled_plan=compiled_plan,
        )
      dev_arrays.append(fused_bufs[s.source_key])
    else:
      dev_arrays.append(
          transform_wave_spec(
              flat_state,
              r,
              s,
              output_dtype,
              devices,
              rank_shd,
              chunk_cache,
              compiled_plan=compiled_plan,
          )
      )
  return dev_arrays


def _compile_and_record(
    executables: dict[tuple[Any, ...], Any],
    key: tuple[Any, ...],
    jit_fn: Any,
    lower_args: tuple[Any, ...],
    peak_box: list[int],
) -> Any:
  """Lowers and compiles `jit_fn(*lower_args)` once per unique `key`."""
  if key not in executables:
    compiled = jit_fn.lower(*lower_args).compile()
    executables[key] = compiled
    peak_box[0] = max(peak_box[0], _compiled_temp_hbm_bytes(compiled))
  return executables[key]


def _compile_chunk_sds(
    flat_sds: Mapping[str, jax.ShapeDtypeStruct],
    spec: TensorExportSpec,
    output_dtype: jnp.dtype | None,
    out_shd: jax.sharding.NamedSharding | None,
    chunk_cache: dict[str, list[jax.ShapeDtypeStruct | None]],
    executables: dict[tuple[Any, ...], Any],
    peak_box: list[int],
) -> jax.ShapeDtypeStruct | None:
  """AOT-compiles chunked `_jit_transform_on_hbm` and pops one chunk SDS."""
  base = flat_sds[spec.source_key]
  lora_a = flat_sds[spec.lora_a_key] if spec.lora_a_key else None
  lora_b = flat_sds[spec.lora_b_key] if spec.lora_b_key else None
  num_chunks, chunk_idx = _chunk_count_and_index(base, spec)
  if spec.source_key not in chunk_cache:
    shd, flatten_ranks = _resolve_transform_sharding(
        base, spec, out_shd, num_chunks
    )
    key = _transform_hbm_key(
        base,
        lora_a,
        lora_b,
        spec.shape_2d,
        spec.transpose_axes,
        output_dtype,
        shd,
        flatten_ranks,
        num_chunks,
    )
    compiled = _compile_and_record(
        executables,
        key,
        _jit_transform_on_hbm,
        (
            base,
            lora_a,
            lora_b,
            float(spec.lora_scaling),
            spec.shape_2d,
            spec.transpose_axes,
            output_dtype,
            shd,
            flatten_ranks,
            num_chunks,
            base.sharding,
        ),
        peak_box,
    )
    out = _compiled_output_sds(compiled)
    chunk_cache[spec.source_key] = list(
        out if isinstance(out, tuple) else (out,)
    )
  entries = chunk_cache[spec.source_key]
  cached = entries[chunk_idx]
  entries[chunk_idx] = None
  if not any(x is not None for x in entries):
    del chunk_cache[spec.source_key]
  return cached


def _compile_transform_spec(
    flat_sds: Mapping[str, jax.ShapeDtypeStruct],
    spec: TensorExportSpec,
    output_dtype: jnp.dtype | None,
    rank_shd: jax.sharding.NamedSharding | None,
    chunk_cache: dict[str, list[jax.ShapeDtypeStruct | None]] | None,
    executables: dict[tuple[Any, ...], Any],
    peak_box: list[int],
) -> jax.ShapeDtypeStruct:
  """AOT-compiles one spec transform and returns its abstract output SDS."""
  base = flat_sds[spec.source_key]
  out_shd = None
  if rank_shd is not None and _uses_multi_rank_routing(
      spec, base, len(rank_shd.device_set)
  ):
    out_shd = rank_shd
  if spec.unstack_range is not None and chunk_cache is not None:
    cached = _compile_chunk_sds(
        flat_sds,
        spec,
        output_dtype,
        out_shd,
        chunk_cache,
        executables,
        peak_box,
    )
    if cached is not None:
      return cached
  if _is_noop_transform(base, spec, output_dtype, out_shd):
    return base
  lora_a = flat_sds[spec.lora_a_key] if spec.lora_a_key else None
  lora_b = flat_sds[spec.lora_b_key] if spec.lora_b_key else None
  num_chunks, chunk_idx = _chunk_count_and_index(base, spec)
  shd, flatten_ranks = _resolve_transform_sharding(
      base, spec, out_shd, num_chunks
  )
  key = _transform_hbm_key(
      base,
      lora_a,
      lora_b,
      spec.shape_2d,
      spec.transpose_axes,
      output_dtype,
      shd,
      flatten_ranks,
      num_chunks,
  )
  compiled = _compile_and_record(
      executables,
      key,
      _jit_transform_on_hbm,
      (
          base,
          lora_a,
          lora_b,
          float(spec.lora_scaling),
          spec.shape_2d,
          spec.transpose_axes,
          output_dtype,
          shd,
          flatten_ranks,
          num_chunks,
          base.sharding,
      ),
      peak_box,
  )
  out = _compiled_output_sds(compiled)
  return out[chunk_idx] if isinstance(out, tuple) else out


def _compile_wave_transforms(
    flat_sds: Mapping[str, jax.ShapeDtypeStruct],
    wave: WaveBucket,
    pairs: Sequence[tuple[int, TensorExportSpec]],
    output_dtype: jnp.dtype | None,
    num_ranks: int,
    rank_shd: jax.sharding.NamedSharding | None,
    chunk_cache: dict[str, list[jax.ShapeDtypeStruct | None]],
    executables: dict[tuple[Any, ...], Any],
    peak_box: list[int],
) -> list[jax.ShapeDtypeStruct]:
  """AOT-compiles all transforms for `wave` and returns abstract output SDS."""
  fusable = _fusable_wave_sources(flat_sds, wave, num_ranks)
  if rank_shd is not None and _can_fuse_identity_wave(
      wave, fusable, num_ranks, flat_sds, output_dtype
  ):
    r0 = wave.rank_assignments[0]
    base_arrs = tuple(flat_sds[s.source_key] for s in r0)
    shapes_2d = tuple(s.shape_2d for s in r0)
    axes = tuple(s.transpose_axes for s in r0)
    key = _fused_wave_key(base_arrs, shapes_2d, axes, output_dtype, rank_shd)
    in_shds = tuple(a.sharding for a in base_arrs)
    compiled = _compile_and_record(
        executables,
        key,
        _jit_transform_and_concat_fusable_wave,
        (base_arrs, shapes_2d, axes, output_dtype, rank_shd, in_shds),
        peak_box,
    )
    return [_compiled_output_sds(compiled)] * len(pairs)
  fused_sds: dict[str, jax.ShapeDtypeStruct] = {}
  out_sds: list[jax.ShapeDtypeStruct] = []
  for _, spec in pairs:
    if spec.source_key in fusable:
      if spec.source_key not in fused_sds:
        full_spec = dataclasses.replace(spec, unstack_range=None)
        fused_sds[spec.source_key] = _compile_transform_spec(
            flat_sds,
            full_spec,
            output_dtype,
            rank_shd,
            None,
            executables,
            peak_box,
        )
      out_sds.append(fused_sds[spec.source_key])
    else:
      out_sds.append(
          _compile_transform_spec(
              flat_sds,
              spec,
              output_dtype,
              rank_shd,
              chunk_cache,
              executables,
              peak_box,
          )
      )
  return out_sds


def _compile_dtype_routing(
    shd_pairs: Sequence[tuple[int, TensorExportSpec]],
    shd_sds: Sequence[jax.ShapeDtypeStruct],
    rank_mesh: jax.sharding.Mesh,
    rank_shd: jax.sharding.NamedSharding,
    executables: dict[tuple[Any, ...], Any],
    peak_box: list[int],
    max_bucket_bytes: int | None = None,
    *,
    is_packed_bf16: bool = False,
    flat_state: Mapping[str, Any] | None = None,
) -> None:
  """AOT-compiles routing collectives for one homogeneous-dtype sharded batch."""
  num_ranks = rank_mesh.size
  shd_pairs, shd_sds, oversized, step_tiles = _split_oversized_batch(
      shd_pairs, shd_sds, num_ranks, max_bucket_bytes
  )
  # Stream each oversized tensor independently so multi-oversized waves (e.g.
  # untied embed_tokens + lm_head) never concatenate >1 full tensor into HBM.
  for entry in oversized:
    ex_key, lower_args = _oversized_exchange_args(
        (entry,), step_tiles, rank_mesh, rank_shd
    )
    _compile_and_record(
        executables,
        ex_key,
        _jit_exchange_shards_all_to_all,
        lower_args,
        peak_box,
    )
  if not shd_pairs:
    return
  sharded, owners, owner_groups, _ = _group_batch_shards(
      shd_pairs,
      shd_sds,
      num_ranks,
      rank_shd,
      is_packed_bf16=is_packed_bf16,
      flat_state=flat_state,
  )
  if owner_groups and set(owner_groups) == {tuple(range(num_ranks))}:
    if not (len(sharded) == 1 and _is_packed_identity_shard(sharded[0])):
      cat_key = _concat_local_key(sharded, rank_mesh, 'rank')
      _compile_and_record(
          executables,
          cat_key,
          _jit_concat_local_rank_shards,
          (sharded, rank_mesh, 'rank'),
          peak_box,
      )
    return
  (
      ordered_shards,
      order_tup,
      in_off_tup,
      send_sz_tup,
      max_recv,
      use_ragged,
      need_perm,
      _,
      meta_np,
  ) = _compute_exchange_plan(sharded, owners, owner_groups, num_ranks)
  meta_sds = (
      None
      if meta_np is None
      else jax.ShapeDtypeStruct(meta_np.shape, jnp.int32, sharding=rank_shd)
  )
  perm_sds = None
  if need_perm:
    perm_sds = jax.ShapeDtypeStruct(
        (num_ranks, num_ranks * (max_recv // 1024)),
        jnp.int32,
        sharding=rank_shd,
    )
  ex_key = _exchange_key(
      ordered_shards,
      rank_mesh,
      'rank',
      order_tup,
      in_off_tup,
      send_sz_tup,
      max_recv,
      use_ragged,
      perm_sds,
      meta_sds,
  )
  _compile_and_record(
      executables,
      ex_key,
      _jit_exchange_shards_all_to_all,
      (
          ordered_shards,
          rank_mesh,
          'rank',
          order_tup,
          in_off_tup,
          send_sz_tup,
          max_recv,
          use_ragged,
          perm_sds,
          meta_sds,
      ),
      peak_box,
  )


def _compile_wave_routing(
    pairs: Sequence[tuple[int, TensorExportSpec]],
    dev_sds: Sequence[jax.ShapeDtypeStruct],
    rank_mesh: jax.sharding.Mesh,
    rank_shd: jax.sharding.NamedSharding,
    executables: dict[tuple[Any, ...], Any],
    peak_box: list[int],
    max_bucket_bytes: int | None = None,
    *,
    flat_state: Mapping[str, Any] | None = None,
    output_dtype: jnp.dtype | None = None,
) -> None:
  """AOT-compiles wave concat or all-to-all routing collectives if required."""
  num_ranks = rank_mesh.size
  routed = [
      (p, a)
      for p, a in zip(pairs, dev_sds)
      if _uses_multi_rank_routing(p[1], a, num_ranks)
  ]
  if not routed:
    return
  shd_pairs, shd_sds = zip(*routed)
  for sub_pairs, sub_sds in _partition_by_dtype(shd_pairs, shd_sds):
    in_packed_bf16 = False
    if flat_state is not None and sub_pairs[0][1].source_key in flat_state:
      target_dt = jnp.dtype(
          output_dtype or flat_state[sub_pairs[0][1].source_key].dtype
      )
      in_packed_bf16 = (
          target_dt == jnp.bfloat16 and sub_sds[0].dtype == jnp.uint32
      )
    _compile_dtype_routing(
        sub_pairs,
        sub_sds,
        rank_mesh,
        rank_shd,
        executables,
        peak_box,
        max_bucket_bytes=max_bucket_bytes,
        is_packed_bf16=in_packed_bf16,
        flat_state=flat_state,
    )


def compile_save_plan(
    flat_state: Mapping[str, jax.Array | jax.ShapeDtypeStruct],
    plan: SavePlan,
    *,
    output_dtype: jnp.dtype | None = None,
    devices: Sequence[jax.Device] | None = None,
) -> CompiledSavePlan:
  """Ahead-of-time compiles all XLA kernels required by `plan`.

  Walks every wave in `plan` using abstract `jax.ShapeDtypeStruct` descriptors
  (allocating zero live device buffers), lowers and compiles every unique
  transform, fused-wave, concat, and all-to-all kernel signature once via
  `.lower(...).compile()`, and records peak temporary HBM overhead from
  `compiled.memory_analysis()`.

  Pass the returned :class:`CompiledSavePlan` to
  :func:`export_safetensors_with_plan` to eliminate first-step XLA JIT
  compilation stalls and catch HBM OOM risks before runtime.

  Args:
    flat_state: Mapping from flattened parameter keys to live `jax.Array`s or
      abstract `jax.ShapeDtypeStruct`s with shardings attached.
    plan: :class:`SavePlan` produced by :func:`build_save_plan`.
    output_dtype: Optional cast dtype matching the export call (for example,
      `jnp.bfloat16`), or `None` to preserve each parameter's dtype.
    devices: Device sequence matching the export mesh, or `None` for host-local
      / single-device exports.

  Returns:
    A :class:`CompiledSavePlan` holding deduplicated compiled XLA executables,
    `peak_temp_hbm_bytes`, `compile_time_sec`, and `num_unique_kernels`.

  Example:
    ```python
    plan = safetensors_saver.build_save_plan(
        specs,
        num_ranks=len(devices),
        max_bucket_bytes=256 << 20,
        max_shard_bytes=2560 << 20,
    )
    compiled = safetensors_saver.compile_save_plan(
        flat_state, plan, output_dtype=jnp.bfloat16, devices=devices
    )
    print(
        f'Compiled {compiled.num_unique_kernels} kernels in'
        f' {compiled.compile_time_sec:.2f}s'
        f' (peak temp HBM={compiled.peak_temp_hbm_bytes >> 20} MiB)'
    )
    metrics = safetensors_saver.export_safetensors_with_plan(
        flat_state,
        plan,
        'gs://my-bucket/checkpoints/step_1',
        output_dtype=jnp.bfloat16,
        devices=devices,
        compiled_plan=compiled,
    )
    ```
  """
  t0 = time.perf_counter()
  flat_sds = {k: _abstract_sds(v) for k, v in flat_state.items()}
  rank_mesh, rank_shd = (
      _resolve_rank_mesh_shd(devices, None) if devices else (None, None)
  )
  num_ranks = len(devices) if devices else 0
  executables: dict[tuple[Any, ...], Any] = {}
  chunk_cache: dict[str, list[jax.ShapeDtypeStruct | None]] = {}
  peak_box = [0]
  for wave in plan.waves:
    pairs = list(
        itertools.chain.from_iterable(
            zip(itertools.repeat(r), specs)
            for r, specs in wave.rank_assignments.items()
        )
    )
    dev_sds = _compile_wave_transforms(
        flat_sds,
        wave,
        pairs,
        output_dtype,
        num_ranks,
        rank_shd,
        chunk_cache,
        executables,
        peak_box,
    )
    if rank_mesh is not None and rank_shd is not None:
      _compile_wave_routing(
          pairs,
          dev_sds,
          rank_mesh,
          rank_shd,
          executables,
          peak_box,
          max_bucket_bytes=wave.max_bucket_bytes,
          flat_state=flat_sds,
          output_dtype=output_dtype,
      )
  return CompiledSavePlan(
      executables=executables,
      peak_temp_hbm_bytes=peak_box[0],
      compile_time_sec=time.perf_counter() - t0,
      num_unique_kernels=len(executables),
      rank_mesh=rank_mesh,
      rank_shd=rank_shd,
  )


def dispatch_wave_on_hbm(
    flat_state: Mapping[str, jax.Array],
    wave: WaveBucket,
    output_dtype: jnp.dtype | None,
    devices: Sequence[jax.Device],
    defer_contiguous: bool = False,
    chunk_cache: dict[str, list[jax.Array | None]] | None = None,
    compiled_plan: CompiledSavePlan | None = None,
) -> _WaveD2HHandle:
  """Executes wave HBM+ICI transforms and starts async D2H transfer.

  Args:
    flat_state: Flat model parameter dictionary.
    wave: `WaveBucket` to execute on HBM.
    output_dtype: Optional cast dtype (`None` keeps source dtype).
    devices: Ordered mesh devices.
    defer_contiguous: When True, defers contiguous host copies.
    chunk_cache: Optional shared MoE chunk cache across waves.
    compiled_plan: Optional `CompiledSavePlan` for AOT dispatch.

  Returns:
    In-flight `_WaveD2HHandle` finalized via `complete_wave_d2h`.

  Example:
    ```python
    handle = safetensors_saver.dispatch_wave_on_hbm(
        flat_state, plan.waves[0], jnp.bfloat16, devices
    )
    host_wave, hbm_s, d2h_s = safetensors_saver.complete_wave_d2h(handle)
    ```
  """
  t0 = time.perf_counter()
  pairs = list(
      itertools.chain.from_iterable(
          zip(itertools.repeat(rank), specs)
          for rank, specs in wave.rank_assignments.items()
      )
  )
  _, rank_shd = _resolve_rank_mesh_shd(devices, compiled_plan)
  dev_arrays = _transform_wave_pairs(
      flat_state,
      wave,
      pairs,
      output_dtype,
      devices,
      rank_shd,
      chunk_cache,
      compiled_plan=compiled_plan,
  )
  num_ranks, proc = len(devices), jax.process_index()
  rep_specs: list[tuple[int, TensorExportSpec]] = []
  rep_bufs: list[jax.Array] = []
  shd_pairs: list[tuple[int, TensorExportSpec]] = []
  shd_arrays: list[jax.Array] = []
  for (r, s), a in zip(pairs, dev_arrays):
    if _uses_multi_rank_routing(s, a, num_ranks):
      shd_pairs.append((r, s))
      shd_arrays.append(a)
    elif devices[r % num_ranks].process_index == proc:
      rep_specs.append((r, s))
      rep_bufs.append(a.addressable_shards[0].data)
  shd_batches: list[tuple[Any, ...]] = []
  oversized_batches: list[tuple[tuple[_OversizedEntry, ...], int]] = []
  for sub_pairs, sub_arrays in _partition_by_dtype(shd_pairs, shd_arrays):
    target_dt = jnp.dtype(
        output_dtype or flat_state[sub_pairs[0][1].source_key].dtype
    )
    in_packed_bf16 = (
        target_dt == jnp.bfloat16 and sub_arrays[0].dtype == jnp.uint32
    )
    norm_pairs, norm_arrays, oversized, step_tiles = _split_oversized_batch(
        sub_pairs, sub_arrays, num_ranks, wave.max_bucket_bytes
    )
    if norm_pairs:
      targs, ranks, items, _ = _start_batch_d2h_with_layout(
          norm_pairs,
          norm_arrays,
          devices,
          compiled_plan=compiled_plan,
          is_packed_bf16=in_packed_bf16,
          flat_state=flat_state,
      )
      out_packed_bf16 = (
          target_dt == jnp.bfloat16
          and bool(targs)
          and targs[0].dtype == jnp.uint32
      )
      shd_batches.append((targs, ranks, items, out_packed_bf16))
    oversized_batches.extend(((entry,), step_tiles) for entry in oversized)
  if not shd_batches and not oversized_batches:
    jax.block_until_ready(dev_arrays)
  for buf in rep_bufs:
    buf.copy_to_host_async()
  return _WaveD2HHandle(
      rep_specs=tuple(rep_specs),
      rep_bufs=tuple(rep_bufs),
      defer_contiguous=defer_contiguous,
      hbm_sec=time.perf_counter() - t0,
      num_ranks=num_ranks,
      shd_batches=tuple(shd_batches),
      pair_order=tuple((r, _wave_tensor_key(s)) for r, s in pairs),
      oversized_batches=tuple(oversized_batches),
      devices=tuple(devices),
      compiled_plan=compiled_plan,
  )


def complete_wave_d2h(
    handle: _WaveD2HHandle,
) -> tuple[dict[int, dict[str, HostTensor]], float, float]:
  """Completes a dispatched `_WaveD2HHandle` into host rank dictionaries.

  Args:
    handle: Asynchronous wave handle returned by `dispatch_wave_on_hbm`.

  Returns:
    3-tuple `(host_wave, hbm_sec, d2h_sec)`.

  Example:
    ```python
    host_wave, hbm_s, d2h_s = safetensors_saver.complete_wave_d2h(handle)
    ```
  """
  host_wave: dict[int, dict[str, HostTensor]] = collections.defaultdict(dict)
  hbm_sec, b_d2h = handle.hbm_sec, 0.0
  for d2h_targets, active_ranks, rank_items, is_packed_bf16 in (
      handle.shd_batches
  ):
    b_wave, sub_d2h = _finish_batch_d2h(
        d2h_targets,
        active_ranks,
        rank_items,
        defer_contiguous=handle.defer_contiguous,
        is_packed_bf16=is_packed_bf16,
    )
    for r, r_dict in b_wave.items():
      host_wave[r].update(r_dict)
    b_d2h += sub_d2h
  for entries, step_tiles in handle.oversized_batches:
    ov_wave, ov_hbm, ov_d2h = _stream_oversized_batch_to_host(
        entries, step_tiles, handle.devices, handle.compiled_plan
    )
    for r, r_dict in ov_wave.items():
      host_wave[r].update(r_dict)
    hbm_sec += ov_hbm
    b_d2h += ov_d2h
  t_d2h = time.perf_counter()
  order = None if handle.defer_contiguous else 'C'
  for (r, spec), arr in zip(
      handle.rep_specs, jax.device_get(list(handle.rep_bufs))
  ):
    host_wave[r][_wave_tensor_key(spec)] = np.asarray(arr, order=order)
  for r, key in handle.pair_order:
    if r in host_wave and key in host_wave[r]:
      host_wave[r][key] = host_wave[r].pop(key)
  return dict(host_wave), hbm_sec, b_d2h + (time.perf_counter() - t_d2h)


def materialize_wave_to_host(
    flat_state: Mapping[str, jax.Array],
    wave: WaveBucket,
    output_dtype: jnp.dtype | None,
    devices: Sequence[jax.Device] | None,
    defer_contiguous: bool = False,
    chunk_cache: dict[str, list[jax.Array | None]] | None = None,
    compiled_plan: CompiledSavePlan | None = None,
) -> tuple[dict[int, dict[str, HostTensor]], float, float]:
  """Executes one wave's on-HBM transforms and D2H host materialization.

  Args:
    flat_state: Flat model parameter dictionary.
    wave: `WaveBucket` to materialize.
    output_dtype: Optional cast dtype.
    devices: Ordered mesh devices, or `None` for single-host CPU mode.
    defer_contiguous: When True, avoids redundant host copies via slice views.
    chunk_cache: Optional shared MoE chunk cache across waves.
    compiled_plan: Optional `CompiledSavePlan` for AOT dispatch.

  Returns:
    3-tuple `(host_wave, hbm_sec, d2h_sec)`.

  Example:
    ```python
    host_wave, hbm_s, d2h_s = safetensors_saver.materialize_wave_to_host(
        flat_state, plan.waves[0], jnp.bfloat16, devices
    )
    ```
  """
  if devices:
    return complete_wave_d2h(
        dispatch_wave_on_hbm(
            flat_state,
            wave,
            output_dtype,
            devices,
            defer_contiguous=defer_contiguous,
            chunk_cache=chunk_cache,
            compiled_plan=compiled_plan,
        )
    )
  t0 = time.perf_counter()
  pairs = list(
      itertools.chain.from_iterable(
          [(rank, spec) for spec in specs]
          for rank, specs in wave.rank_assignments.items()
      )
  )
  dev_arrays = _transform_wave_pairs(
      flat_state,
      wave,
      pairs,
      output_dtype,
      None,
      None,
      chunk_cache,
      compiled_plan=compiled_plan,
  )
  jax.block_until_ready(dev_arrays)
  hbm_sec = time.perf_counter() - t0
  t1 = time.perf_counter()
  host_arrays = jax.device_get(dev_arrays)
  del dev_arrays
  host_wave: dict[int, dict[str, HostTensor]] = collections.defaultdict(dict)
  for (rank, spec), host_arr in zip(pairs, host_arrays):
    host_wave[rank][_wave_tensor_key(spec)] = (
        np.asarray(host_arr)
        if defer_contiguous
        else np.ascontiguousarray(host_arr)
    )
  return dict(host_wave), hbm_sec, time.perf_counter() - t1


def step_wave_to_host(
    flat_state: Mapping[str, jax.Array],
    waves: Sequence[WaveBucket],
    idx: int,
    d2h_handle: _WaveD2HHandle | None,
    output_dtype: jnp.dtype | None,
    devices: Sequence[jax.Device] | None,
    defer_contiguous: bool = True,
    chunk_cache: dict[str, list[jax.Array | None]] | None = None,
    compiled_plan: CompiledSavePlan | None = None,
) -> tuple[
    dict[int, dict[str, HostTensor]], float, float, _WaveD2HHandle | None
]:
  """Advances one wave in the 2-wave HBM+ICI / D2H overlap pipeline.

  Dispatches wave `idx + 1` on HBM while completing the async D2H transfer for
  wave `idx`, hiding PCIe/ICI latency across consecutive waves. When wave `idx`
  contains oversized streaming batches, drains wave `idx` first so wave
  `idx + 1` does not overlap HBM with full oversized tensors.

  Args:
    flat_state: Flat model parameter dictionary.
    waves: Ordered sequence of all `WaveBucket`s in the `SavePlan`.
    idx: Index of the wave being completed on this iteration.
    d2h_handle: Pre-dispatched handle for `waves[idx]`, or `None`.
    output_dtype: Optional cast dtype.
    devices: Ordered mesh devices, or `None` for non-overlapped fallback.
    defer_contiguous: When True, preserves zero-copy host views.
    chunk_cache: Optional shared MoE chunk cache across waves.
    compiled_plan: Optional `CompiledSavePlan` for AOT dispatch.

  Returns:
    4-tuple `(host_wave, hbm_sec, d2h_sec, next_d2h_handle)`.

  Example:
    ```python
    host_wave, hbm_s, d2h_s, next_h = safetensors_saver.step_wave_to_host(
        flat_state, plan.waves, 0, handle, jnp.bfloat16, devices
    )
    ```
  """
  if d2h_handle is None or devices is None:
    host_wave, hbm_sec, d2h_sec = materialize_wave_to_host(
        flat_state,
        waves[idx],
        output_dtype,
        devices,
        defer_contiguous=defer_contiguous,
        chunk_cache=chunk_cache,
        compiled_plan=compiled_plan,
    )
    return host_wave, hbm_sec, d2h_sec, None

  def _dispatch_next() -> _WaveD2HHandle | None:
    if idx + 1 >= len(waves):
      return None
    return dispatch_wave_on_hbm(
        flat_state,
        waves[idx + 1],
        output_dtype,
        devices,
        defer_contiguous,
        chunk_cache=chunk_cache,
        compiled_plan=compiled_plan,
    )

  if d2h_handle.oversized_batches:
    host_wave, hbm_sec, d2h_sec = complete_wave_d2h(d2h_handle)
    next_handle = _dispatch_next()
  else:
    next_handle = _dispatch_next()
    host_wave, hbm_sec, d2h_sec = complete_wave_d2h(d2h_handle)
  return host_wave, hbm_sec, d2h_sec, next_handle


def _close_unique_handles(
    open_files: Mapping[Any, Any], *, abort: bool = False
) -> None:
  """Closes or aborts each unique shared file or shard-group handle."""
  for fh in {
      id(v): v for v in open_files.values() if not isinstance(v, tuple)
  }.values():
    if abort and isinstance(fh, _AbortableWriter):
      fh.abort()
    else:
      fh.close()


def export_safetensors_with_plan(
    flat_state: Mapping[str, jax.Array],
    plan: SavePlan,
    output_dir: str,
    *,
    local_model_path: str | None = None,
    output_dtype: jnp.dtype | None = None,
    devices: Sequence[jax.Device] | None = None,
    plan_time_sec: float = 0.0,
    async_write: bool = False,
    write_buffer_bytes: int | None = None,
    max_write_threads: int | None = None,
    defer_contiguous: bool = True,
    compiled_plan: CompiledSavePlan | None = None,
) -> SaveMetrics:
  """Executes a `SavePlan` with 2-wave HBM+ICI / D2H overlap and writes shards.

  Streams each wave from accelerator HBM to host RAM via subtree all-to-all and
  writes Hugging Face `.safetensors` shards plus `model.safetensors.index.json`
  to local disk, `/GOOGLE_INTERNAL_STOAGE_PATH/`, or `gs://`.

  When wrapped in a :class:`MultiprocStagePool` context (or `pool.activate()`):
    * Single-wave shards stream directly through `/dev/shm` slots to CNS or
      multi-VIP GCS HTTP/1.1 Keep-Alive workers.
    * Multi-wave GCS shard groups (`plan.shard_groups` created via
      `max_shard_bytes`) stream `.part.<k>` objects concurrently per wave,
      assemble each final `2.5–4.0 GiB` `.safetensors` shard server-side via
      parallel GCS `compose`, and delete all `.part.<k>` objects before writing
      `model.safetensors.index.json`.

  Args:
    flat_state: Mapping from flattened parameter keys to live `jax.Array`s.
    plan: :class:`SavePlan` built by :func:`build_save_plan`.
    output_dir: Destination directory path (local path, `/GOOGLE_INTERNAL_STOAGE_PATH/<cell>-d/...`, or
      `gs://<bucket>/<prefix>`).
    local_model_path: Optional source model directory from which non-safetensors
      files (such as `config.json` and tokenizer files) are copied by rank 0.
    output_dtype: Optional dtype cast applied on HBM (for example,
      `jnp.bfloat16`), or `None` to preserve each tensor's dtype.
    devices: Ordered `jax.Device` sequence matching `plan.num_ranks`, or `None`
      for single-host / unsharded export.
    plan_time_sec: Optional planning duration added to
      `SaveMetrics.total_time_sec`.
    async_write: If True, overlaps background shard writes with subsequent HBM
      and D2H waves. Automatically caps in-flight waves to `1` when writing
      sequentially to shared stream handles.
    write_buffer_bytes: Override for per-stream write chunk buffer size, or
      `None` to use the storage backend default.
    max_write_threads: Override for per-wave write concurrency, or `None` to use
      the storage backend default.
    defer_contiguous: If True, avoids redundant host copies by writing zero-copy
      `_DeferredHostSlice` byte views directly from D2H buffers.
    compiled_plan: Optional :class:`CompiledSavePlan` from
      :func:`compile_save_plan` to dispatch pre-compiled XLA executables with
      zero JIT stalls.

  Returns:
    :class:`SaveMetrics` detailing total bytes, stage timings (`hbm`, `d2h`,
    `disk`), peak per-rank wave bytes, and written shard paths.

  Example:
    ```python
    specs, unstack_counts = safetensors_saver.build_export_specs_from_manifest(
        flat_state, to_hf_manifest
    )
    plan = safetensors_saver.build_save_plan(
        specs,
        num_ranks=len(devices),
        max_bucket_bytes=256 << 20,
        max_shard_bytes=2560 << 20,
        unstack_counts=unstack_counts,
    )
    compiled = safetensors_saver.compile_save_plan(
        flat_state, plan, output_dtype=jnp.bfloat16, devices=devices
    )
    with safetensors_saver.MultiprocStagePool(
        stage_dir='/dev/shm/export_stage',
        num_procs=48,
        max_bucket_bytes=256 << 20,
    ):
      metrics = safetensors_saver.export_safetensors_with_plan(
          flat_state,
          plan,
          'gs://my-bucket/checkpoints/step_1',
          output_dtype=jnp.bfloat16,
          devices=devices,
          async_write=True,
          compiled_plan=compiled,
      )
    ```
  """
  start_total = time.perf_counter()
  devs = tuple(jax.devices() if devices is None else devices)
  backend = resolve_storage_backend(output_dir)
  buf_bytes = (
      backend.default_write_buffer_bytes
      if write_buffer_bytes is None
      else write_buffer_bytes
  )
  threads = (
      backend.default_write_threads
      if max_write_threads is None
      else max_write_threads
  )
  backend.makedirs(output_dir)
  if jax.process_count() > 1:
    multihost_utils.sync_global_devices('safetensors_saver:post_makedirs')
  t_open = time.perf_counter()
  open_files, shard_paths, num_active = open_local_shard_files(
      flat_state, plan, output_dir, output_dtype, devs
  )
  disk_sec = time.perf_counter() - t_open
  total_hbm_sec = total_d2h_sec = 0.0
  peak_wave_rank_bytes = 0
  has_shared_stream = any(not isinstance(v, tuple) for v in open_files.values())
  limit = 0
  if async_write:
    limit = 1 if has_shared_stream else _MAX_INFLIGHT_FLUSH_WAVES
  pending: list[tuple[concurrent.futures.Future[float], int]] = []
  chunk_cache: dict[str, list[jax.Array | None]] = {}
  d2h_handle = None
  succeeded = False
  try:
    if devs and len(plan.waves) > 1:
      d2h_handle = dispatch_wave_on_hbm(
          flat_state,
          plan.waves[0],
          output_dtype,
          devs,
          defer_contiguous,
          chunk_cache=chunk_cache,
          compiled_plan=compiled_plan,
      )

    for idx, wave in enumerate(plan.waves):
      peak_wave_rank_bytes = max(
          peak_wave_rank_bytes, *wave.rank_bytes.values(), 0
      )
      host_wave, hbm_sec, d2h_sec, d2h_handle = step_wave_to_host(
          flat_state,
          plan.waves,
          idx,
          d2h_handle,
          output_dtype,
          devs,
          defer_contiguous,
          chunk_cache=chunk_cache,
          compiled_plan=compiled_plan,
      )
      total_hbm_sec += hbm_sec
      total_d2h_sec += d2h_sec
      disk_sec += submit_and_drain_wave_flush(
          pending,
          limit,
          open_files,
          host_wave,
          wave,
          buf_bytes,
          threads,
      )
    for fut, _ in pending:
      t_wait = time.perf_counter()
      fut.result()
      disk_sec += time.perf_counter() - t_wait

    t_close = time.perf_counter()
    _close_unique_handles(open_files)
    succeeded = True
    wave_groups = itertools.chain.from_iterable(
        w.rank_assignments.values() for w in plan.waves
    )
    all_specs = itertools.chain.from_iterable(wave_groups)
    actual_total_bytes = sum(
        _spec_export_elems(s, flat_state)
        * jnp.dtype(output_dtype or flat_state[s.source_key].dtype).itemsize
        for s in all_specs
    )
    finalize_safetensors_metadata(
        plan,
        output_dir,
        local_model_path,
        num_active,
        total_bytes=actual_total_bytes,
    )
    disk_sec += time.perf_counter() - t_close
  finally:
    if not succeeded:
      # Drain in-flight background wave flushes before aborting shared handles
      # so worker threads never race on a closed/aborted stream.
      concurrent.futures.wait([fut for fut, _ in pending])
      _close_unique_handles(open_files, abort=True)
  total_sec = time.perf_counter() - start_total + plan_time_sec
  return SaveMetrics(
      total_bytes=actual_total_bytes,
      num_waves=len(plan.waves),
      num_ranks=plan.num_ranks,
      plan_time_sec=plan_time_sec,
      hbm_transform_and_route_sec=total_hbm_sec,
      d2h_and_contiguous_sec=total_d2h_sec,
      disk_write_sec=disk_sec,
      total_time_sec=total_sec,
      peak_wave_rank_bytes=peak_wave_rank_bytes,
      shard_paths=tuple(shard_paths),
  )


def _merge_lora_delta(
    lora_a: Any,
    lora_b: Any,
    rank: int,
    alpha: float,
    state_key: str,
    transpose_rules: dict[str, tuple[int, ...]] | None,
) -> jnp.ndarray:
  """Computes and shapes merged LoRA delta array for base state update."""
  a_val = jnp.asarray(
      lora_a.value if isinstance(lora_a, nnx.Variable) else lora_a
  )
  b_val = jnp.asarray(
      lora_b.value if isinstance(lora_b, nnx.Variable) else lora_b
  )
  if a_val.ndim == 3:
    a_val = a_val.reshape(a_val.shape[0] * a_val.shape[1], a_val.shape[2])
  if b_val.ndim == 3:
    b_val = b_val.reshape(b_val.shape[0], b_val.shape[1] * b_val.shape[2])
  delta = (a_val @ b_val) * (alpha / rank)
  if transpose_rules:
    for t_key, rule in transpose_rules.items():
      if t_key in state_key:
        return delta.transpose(rule)
  return delta


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: Any,
    rank: int,
    alpha: float,
    state_key_transform_fn: Callable[[str], str],
    custom_layer_extractor_fn: (
        Callable[[dict[str, list[Any]]], dict[str, list[Any]]] | None
    ) = None,
    transpose_rules: dict[str, tuple[int, ...]] | None = None,
):
  """Saves a model with LoRA weights merged in single-file safetensors format.

  Legacy single-host helper that copies non-safetensors metadata from
  `local_model_path`, merges LoRA adapters into base projections on host, and
  writes `model.safetensors` to `output_dir`. For distributed multi-host TPU
  exports, prefer :func:`export_safetensors_with_plan`.

  Args:
    local_model_path: Local directory containing tokenizer/config files.
    output_dir: Local destination directory (recreated if already present).
    lora_model: Flax NNX LoRA-wrapped model instance.
    rank: LoRA rank `r`.
    alpha: LoRA scaling factor `alpha`.
    state_key_transform_fn: Maps dotted Tunix keys to HuggingFace weight names.
    custom_layer_extractor_fn: Optional hook to reshape/merge grouped layers.
    transpose_rules: Optional regex-to-axes transpose overrides.

  Example:
    ```python
    safetensors_saver.save_lora_merged_model_as_safetensors(
        local_model_path='/tmp/hf_base',
        output_dir='/tmp/hf_merged',
        lora_model=model,
        rank=16,
        alpha=32.0,
        state_key_transform_fn=lambda k: k,
    )
    ```
  """
  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
  os.makedirs(output_dir)

  lora_layers: dict[str, list[Any]] = {}

  for path, value in nnx.iter_graph(lora_model):
    if isinstance(value, nnx.LoRAParam):
      path_str = join_path(path[:-1])
      if path_str in lora_layers:
        assert (
            'lora_b' in path[-1]  # pyrefly: ignore[not-iterable]
        ), f'Expect second LoRAParam to be lora_b, got {path[-1]}'
        lora_layers[path_str].append(value)
      else:
        assert (
            'lora_a' in path[-1]  # pyrefly: ignore[not-iterable]
        ), f'Expect first LoRAParam to be lora_a, got {path[-1]}'
        lora_layers[path_str] = [value]

  if custom_layer_extractor_fn:
    lora_layers = custom_layer_extractor_fn(lora_layers)

  base_state = safe_flax.load_file(local_model_path + '/model.safetensors')

  for path, (lora_a, lora_b) in lora_layers.items():
    state_key = state_key_transform_fn(path)
    assert state_key in base_state, (
        f'LoRA layer {path} not found in base model state dict'
        f' {base_state.keys()}'
    )
    combined_lora = _merge_lora_delta(
        lora_a, lora_b, rank, alpha, state_key, transpose_rules
    )
    base_state[state_key] += combined_lora.astype(base_state[state_key].dtype)

  safetensors_path = os.path.join(output_dir, 'model.safetensors')
  safe_flax.save_file(base_state, safetensors_path)

  copy_non_safetensors_files(local_model_path, output_dir)


# Truncation limit for the host-0 error message that is serialized into the
# JSON payload broadcast to the other processes.
_MAX_BROADCAST_ERROR_CHARS = 4096
# Tensor payloads are written as memoryview slices of this size so backends
# that copy bytes-like inputs never copy a whole tensor; 16 MiB amortizes
# per-write overhead.
_WRITE_CHUNK_BYTES = 16 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class SafetensorsSaveResult:
  """Outcome of a committed `save_safetensors` call, identical on all processes.

  Attributes:
    paths: Committed files; the last entry is the commit marker
      (`model.safetensors.index.json` when sharded, else `model.safetensors`).
    checksums: `checksum_fn` output over the written tensors, else `None`.
  """

  paths: tuple[str, ...]
  checksums: dict[str, float | int] | None


def _safetensors_dtype(key: str, arr: np.ndarray) -> str:
  """Returns the safetensors dtype tag for `arr`; raises if unsupported."""
  np_dt = np.dtype(arr.dtype)
  if np_dt not in safetensors_io.NP_TO_SAFETENSORS_DTYPE:
    raise ValueError(f'Unsupported safetensors dtype {np_dt!r} for {key!r}.')
  return safetensors_io.NP_TO_SAFETENSORS_DTYPE[np_dt]


def encode_safetensors_header(
    shard_items: Sequence[tuple[str, np.ndarray]],
) -> tuple[bytes, list[np.ndarray], int]:
  """Encodes the safetensors wire header and contiguous byte views for a shard.

  Args:
    shard_items: Ordered `(tensor_name, array)` pairs for one shard.

  Returns:
    A `(header_bytes, payloads, total_bytes)` tuple where `header_bytes` is the
    8-byte little-endian header length followed by the 8-byte-aligned UTF-8
    JSON header, `payloads` are contiguous `uint8` views of each tensor in
    header order, and `total_bytes` is the sum of tensor payload byte sizes.
  """
  header: dict[str, Any] = {}
  payloads: list[np.ndarray] = []
  offset = 0
  for key, arr in shard_items:
    dtype = _safetensors_dtype(key, arr)
    c_arr = np.ascontiguousarray(arr) if arr.ndim > 0 else np.asarray(arr)
    end = offset + int(c_arr.nbytes)
    header[key] = {
        'dtype': dtype,
        'shape': [int(d) for d in c_arr.shape],
        'data_offsets': [offset, end],
    }
    payloads.append(c_arr.reshape(-1).view(np.uint8))
    offset = end

  raw_header = json.dumps(header, separators=(',', ':')).encode('utf-8')
  padded_header = raw_header + (b' ' * ((-len(raw_header)) % 8))
  header_bytes = struct.pack('<Q', len(padded_header)) + padded_header
  return header_bytes, payloads, offset


def write_safetensors_shard(
    shard_path: epath.Path,
    shard_items: Sequence[tuple[str, np.ndarray]],
) -> int:
  """Writes one `.safetensors` file via `.tmp` + `replace()`; returns bytes."""
  header_bytes, payloads, total_bytes = encode_safetensors_header(shard_items)
  tmp_path = shard_path.parent / f'{shard_path.name}.tmp'
  try:
    with cast(BinaryIO, tmp_path.open('wb')) as fh:
      fh.write(header_bytes)
      for raw_bytes in payloads:
        # Chunk memoryview writes to keep local POSIX zero-copy while bounding
        # internal C++/remote-stream buffer copies to <= _WRITE_CHUNK_BYTES.
        mv = memoryview(raw_bytes).cast('B')
        for off in range(0, len(mv), _WRITE_CHUNK_BYTES):
          fh.write(mv[off : off + _WRITE_CHUNK_BYTES])
    tmp_path.replace(shard_path)
  except BaseException:
    with contextlib.suppress(Exception):
      tmp_path.unlink(missing_ok=True)
    raise
  return total_bytes


def partition_safetensors_shards(
    host_tensors: Mapping[str, np.ndarray],
    max_shard_bytes: int,
) -> list[list[tuple[str, np.ndarray]]]:
  """Groups `(key, array)` pairs greedily within `max_shard_bytes`.

  A single tensor whose byte size exceeds `max_shard_bytes` is placed in its
  own 1-tensor shard.

  Args:
    host_tensors: Mapping of tensor names to host NumPy arrays.
    max_shard_bytes: Positive soft upper bound on bytes per shard.

  Returns:
    Non-empty list of shards, each a list of `(key, array)` pairs.
  """
  if (
      isinstance(max_shard_bytes, bool)
      or not isinstance(max_shard_bytes, int)
      or max_shard_bytes <= 0
  ):
    raise ValueError(
        f'max_shard_bytes must be a positive int, got {max_shard_bytes!r}.'
    )
  shards: list[list[tuple[str, np.ndarray]]] = []
  cur_shard: list[tuple[str, np.ndarray]] = []
  cur_bytes = 0
  for key, arr in host_tensors.items():
    nbytes = int(arr.nbytes)
    if cur_shard and cur_bytes + nbytes > max_shard_bytes:
      shards.append(cur_shard)
      cur_shard = []
      cur_bytes = 0
    cur_shard.append((key, arr))
    cur_bytes += nbytes
  if cur_shard or not shards:
    shards.append(cur_shard)
  return shards


def remove_tree(path: epath.Path) -> None:
  """Deletes `path` and everything under it on local and remote stores.

  Some object-store backends (e.g., `gs://`) create a `dir/` placeholder object
  on `mkdir`, and `rmtree` may remove only that placeholder while leaving child
  objects behind. Deleting children bottom-up before calling `rmtree` ensures
  complete cleanup across local filesystems and remote object stores.

  Args:
    path: File or directory to delete; a missing path is a no-op.
  """
  if not path.is_dir():
    path.unlink(missing_ok=True)
    return
  for child in path.iterdir():
    remove_tree(child)
  path.rmtree(missing_ok=True)


def write_safetensors_index(
    output_dir: epath.Path,
    weight_map: Mapping[str, str],
    total_bytes: int,
) -> epath.Path:
  """Atomically writes `model.safetensors.index.json` as the commit marker."""
  index_path = output_dir / 'model.safetensors.index.json'
  tmp_index = output_dir / f'{index_path.name}.tmp'
  try:
    tmp_index.write_text(
        json.dumps(
            {'metadata': {'total_size': total_bytes}, 'weight_map': weight_map},
            indent=2,
            sort_keys=True,
        )
    )
    tmp_index.replace(index_path)
  except BaseException:
    with contextlib.suppress(Exception):
      tmp_index.unlink(missing_ok=True)
    raise
  return index_path


def publish_safetensors_shards(
    output_dir: epath.Path,
    shards: Sequence[Sequence[tuple[str, np.ndarray]]],
) -> list[str]:
  """Replaces `output_dir` with shard files, writing the commit marker last."""
  # Validate before deleting so a bad dtype cannot destroy a prior export.
  for items in shards:
    for key, arr in items:
      _safetensors_dtype(key, arr)
  (output_dir / 'model.safetensors.index.json').unlink(missing_ok=True)
  remove_tree(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  if len(shards) == 1:
    single = output_dir / 'model.safetensors'
    write_safetensors_shard(single, shards[0])
    return [str(single)]
  paths: list[str] = []
  total_bytes = 0
  weight_map: dict[str, str] = {}
  for idx, items in enumerate(shards, start=1):
    fname = f'model-{idx:05d}-of-{len(shards):05d}.safetensors'
    total_bytes += write_safetensors_shard(output_dir / fname, items)
    weight_map.update((key, fname) for key, _ in items)
    paths.append(str(output_dir / fname))
  index_path = write_safetensors_index(output_dir, weight_map, total_bytes)
  return [*paths, str(index_path)]


def _commit_host_tensors(
    host_tensors: Mapping[str, np.ndarray],
    output_dir: epath.Path,
    *,
    parsed_manifest: Mapping[str, Any] | None,
    max_shard_bytes: int,
    checksum_fn: (
        Callable[[Mapping[str, np.ndarray]], dict[str, float | int]] | None
    ),
) -> dict[str, Any]:
  """Transforms, partitions, and publishes gathered host tensors on host 0."""
  if parsed_manifest is not None:
    host_tensors = hf_mapping.apply_host_manifest(
        parsed_manifest, host_tensors, drop_tied_targets=True
    )
  checksums = checksum_fn(host_tensors) if checksum_fn is not None else None
  if checksums is not None:
    json.dumps(checksums)
  shards = partition_safetensors_shards(host_tensors, max_shard_bytes)
  paths = publish_safetensors_shards(output_dir, shards)
  return {'paths': paths, 'checksums': checksums}


def save_safetensors(
    tensors: Mapping[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    to_hf_manifest: hf_mapping.RuleSpec | None = None,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    checksum_fn: (
        Callable[[Mapping[str, np.ndarray]], dict[str, float | int]] | None
    ) = None,
) -> SafetensorsSaveResult:
  """Gathers `tensors` to process 0 and commits them as safetensors shards.

  Collective: every process must call it with the same arguments. Process 0
  replaces `output_dir` with shards of at most `max_shard_bytes` each and
  writes the commit marker last (`model.safetensors.index.json` when sharded,
  else the single `model.safetensors`); readers must treat a directory without
  its marker as uncommitted. A cross-process barrier then keeps every process
  from returning before the commit, and any host-0 exception is broadcast to
  non-zero processes so multi-host runs fail cleanly instead of deadlocking.

  Args:
    tensors: Global `jax.Array` or host arrays keyed by source tensor name.
    output_dir: Destination directory (local path or `gs://` URI).
    to_hf_manifest: Optional `hf_mapping` manifest converting source names and
      layouts to the written ones; tied targets are dropped.
    max_shard_bytes: Soft upper bound on bytes per shard; a single tensor larger
      than this limit is placed in its own 1-tensor shard.
    checksum_fn: Optional checksum over the written (post-manifest) tensors.

  Returns:
    Committed paths and checksums, shared from process 0 to every process.

  Raises:
    ValueError: If `output_dir`, `max_shard_bytes`, `to_hf_manifest`, or any
      tensor dtype is invalid.
    RuntimeError: On non-zero processes if process 0 fails during commit.
    Exception: Re-raises the original exception on process 0 if commit fails.
  """
  if not os.fspath(output_dir):
    raise ValueError('save_safetensors output_dir must not be empty.')
  if (
      isinstance(max_shard_bytes, bool)
      or not isinstance(max_shard_bytes, int)
      or max_shard_bytes <= 0
  ):
    raise ValueError(
        f'max_shard_bytes must be a positive int, got {max_shard_bytes!r}.'
    )
  parsed_manifest = (
      hf_mapping.parse_manifest(to_hf_manifest)
      if to_hf_manifest is not None
      else None
  )
  multihost = jax.process_count() > 1
  is_host0 = jax.process_index() == 0
  host_tensors = sharding_utils.gather_tensors_to_host0(
      tensors, is_host0=is_host0
  )

  out_path = epath.Path(output_dir)
  if not multihost:
    committed = _commit_host_tensors(
        host_tensors,
        out_path,
        parsed_manifest=parsed_manifest,
        max_shard_bytes=max_shard_bytes,
        checksum_fn=checksum_fn,
    )
    shared = sharding_utils.broadcast_host0_json(committed)
    return SafetensorsSaveResult(
        paths=tuple(shared['paths']), checksums=shared['checksums']
    )

  committed: dict[str, Any] = {}
  host0_exc: Exception | None = None
  if is_host0:
    try:
      committed = _commit_host_tensors(
          host_tensors,
          out_path,
          parsed_manifest=parsed_manifest,
          max_shard_bytes=max_shard_bytes,
          checksum_fn=checksum_fn,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      host0_exc = exc
      committed = {
          'error': f'{type(exc).__name__}: {exc}'[:_MAX_BROADCAST_ERROR_CHARS]
      }
  multihost_utils.sync_global_devices(
      f'tunix:save_safetensors:{os.fspath(output_dir)}'
  )
  shared = sharding_utils.broadcast_host0_json(committed)
  if host0_exc is not None:
    raise host0_exc
  if 'error' in shared:
    raise RuntimeError(
        f'save_safetensors failed on process 0: {shared["error"]}'
    )
  return SafetensorsSaveResult(
      paths=tuple(shared['paths']), checksums=shared['checksums']
  )
