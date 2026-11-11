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

import collections
from collections.abc import Mapping, Sequence
import concurrent.futures
import contextlib
import dataclasses
import functools
import itertools
import json
import math
import os
import shutil
import struct
import threading
import time
from typing import Any, BinaryIO, Callable, cast

from etils import epath
from flax import nnx
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import safetensors.flax as safe_flax
from tunix.models import hf_mapping
from tunix.models import safetensors_io
from tunix.utils import sharding_utils

# Soft cap on bytes per shard file (2.5 GiB), in line with typical multi-GiB
# Hugging Face checkpoint shards; a larger single tensor gets its own shard.
DEFAULT_MAX_SHARD_BYTES = 2560 * 1024 * 1024
# Hard cap on concurrent in-flight host wave flushes to bound host RAM usage.
_MAX_INFLIGHT_FLUSH_WAVES = 4

try:
  from pyarrow import fs as _pa_fs  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]
except ImportError:
  _pa_fs = None

_GCS_FS_CACHE: dict[Any, Any] = {}
_SAVE_PLAN_CACHE: dict[tuple[Any, ...], 'SavePlan'] = {}
_SHARED_WAVE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=64)
_SHARED_WRITE_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=128)


def _get_gcs_fs(pa_fs: Any) -> Any:
  """Returns a cached PyArrow `GcsFileSystem` instance keyed by class."""
  cls = pa_fs.GcsFileSystem
  fs = _GCS_FS_CACHE.get(cls)
  if fs is None:
    fs = cls()
    _GCS_FS_CACHE[cls] = fs
  return fs


@dataclasses.dataclass(frozen=True)
class TensorExportSpec:
  """Specification for exporting a single model parameter tensor."""

  source_key: str
  hf_key: str
  shape_2d: tuple[int, ...] | None = None
  transpose_axes: tuple[int, ...] | None = None
  unstack_axis: int | None = None
  nbytes: int = 0
  lora_a_key: str | None = None
  lora_b_key: str | None = None
  lora_scaling: float = 1.0
  is_replicated: bool = False


def infer_is_replicated(arr: jax.Array) -> bool:
  """Infers whether `arr` is fully replicated across its sharding."""
  shd = arr.sharding
  num_devs = max(len(shd.device_set), 1)
  if isinstance(shd, jax.sharding.NamedSharding):
    return not any(shd.spec)
  shards = arr.addressable_shards
  return num_devs <= 1 or (bool(shards) and shards[0].data.shape == arr.shape)


@dataclasses.dataclass(frozen=True)
class WaveBucket:
  """One communication and D2H transfer wave across ranks."""

  wave_index: int
  rank_assignments: dict[int, tuple[TensorExportSpec, ...]]
  rank_bytes: dict[int, int]


@dataclasses.dataclass(frozen=True)
class StorageBackend:
  """Target-aware storage policy and stream opener for local and remote paths."""

  kind: str = 'local'
  default_per_wave_shards: bool = True
  default_write_threads: int = 8
  default_write_buffer_bytes: int = 0
  requires_bytes_chunks: bool = False

  def makedirs(self, path: str) -> None:
    """Creates output directory if required by the storage backend."""
    if self.kind == 'local':
      os.makedirs(path, exist_ok=True)

  def cleanup_stale_shards(self, output_dir: str) -> None:
    """Removes stale shard and index files on process 0 before export."""
    if jax.process_index() != 0:
      return
    if self.kind == 'gcs' and _pa_fs is not None:
      fs = _get_gcs_fs(_pa_fs)
      prefix = output_dir.removeprefix('gs://').rstrip('/')
      selector = _pa_fs.FileSelector(
          prefix, allow_not_found=True, recursive=False
      )
      stale = [
          info.path
          for info in fs.get_file_info(selector)
          if info.base_name == 'model.safetensors.index.json'
          or info.base_name.endswith('.safetensors')
      ]
      stale.sort(key=lambda p: not p.endswith('model.safetensors.index.json'))
      for obj_path in stale:
        fs.delete_file(obj_path)
      return
    out_path = epath.Path(output_dir)
    if not out_path.is_dir():
      return
    (out_path / 'model.safetensors.index.json').unlink(missing_ok=True)
    for child in out_path.iterdir():
      if child.name.endswith('.safetensors'):
        child.unlink(missing_ok=True)

  def open_writer(self, path: str) -> Any:
    """Opens a sequential binary output stream for path."""
    if self.kind == 'gcs' and _pa_fs is not None:
      return _get_gcs_fs(_pa_fs).open_output_stream(
          path.removeprefix('gs://'),
          buffer_size=self.default_write_buffer_bytes,
      )
    if self.kind == 'gcs':
      return epath.Path(path).open('wb')
    return open(path, 'wb', buffering=16 << 20)

  def write_view(
      self, f: Any, view: 'memoryview[Any]', chunk_bytes: int | None = None
  ) -> None:
    """Writes `view` to `f` as `bytes` chunks or zero-copy `memoryview`."""
    if not self.requires_bytes_chunks:
      f.write(view)
      return
    raw_step = (
        self.default_write_buffer_bytes if chunk_bytes is None else chunk_bytes
    )
    step = raw_step if raw_step > 0 else max(len(view), 1)
    for off in range(0, len(view), step):
      f.write(bytes(view[off : off + step]))


_LOCAL_BACKEND = StorageBackend('local', True, 8, 0, False)
_GCS_BACKEND = StorageBackend('gcs', True, 64, 64 << 20, False)


def resolve_storage_backend(output_dir: str) -> StorageBackend:
  """Resolves `StorageBackend` for local POSIX or remote URIs."""
  if output_dir.startswith('gs://'):
    return _GCS_BACKEND
  return _LOCAL_BACKEND


@dataclasses.dataclass(frozen=True)
class _OpenShardFile:
  """Persistent shard file handle paired with its resolved `StorageBackend`."""

  backend: StorageBackend
  file: Any

  def close(self) -> None:
    self.file.close()


@dataclasses.dataclass(frozen=True)
class SavePlan:
  """LPT bin-packed subtree all-to-all export plan."""

  num_ranks: int
  total_bytes: int
  max_bucket_bytes: int | None
  waves: tuple[WaveBucket, ...]
  rank_total_bytes: dict[int, int]
  weight_map: dict[str, str]
  per_wave_shards: bool = False
  wave_shards: tuple[tuple[int, int], ...] = ()


@dataclasses.dataclass(frozen=True)
class SaveMetrics:
  """Execution timing and memory breakdown for a safetensors export."""

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


def join_path(path: Sequence[object]) -> str:
  return '.'.join([str(field) for field in path])


def copy_non_safetensors_files(local_model_path: str, output_dir: str) -> None:
  """Copies non-safetensors metadata/tokenizer files from `local_model_path`."""
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


def extract_flat_state_dict(model: nnx.Module) -> dict[str, jax.Array]:
  """Extracts a flat path-to-array dictionary of params/LoRA from an NNX model."""
  return {
      join_path(path): jnp.asarray(variable.value)
      for path, variable in nnx.to_flat_state(
          nnx.state(model, (nnx.LoRAParam, nnx.Param))
      )
  }


def shard_filename(shard_idx: int, total_shards: int) -> str:
  """Returns the canonical safetensors shard filename."""
  return (
      'model.safetensors'
      if total_shards <= 1
      else f'model-{shard_idx + 1:05d}-of-{total_shards:05d}.safetensors'
  )


def expand_spec_keys(
    spec: TensorExportSpec, counts: Mapping[str, int]
) -> tuple[str, ...]:
  """Expands HF keys for a TensorExportSpec (including stacked MoE experts)."""
  if spec.unstack_axis is None or spec.source_key not in counts:
    return (spec.hf_key,)
  return tuple(
      spec.hf_key.format(expert=i) for i in range(counts[spec.source_key])
  )


def build_save_plan(
    specs: Sequence[TensorExportSpec],
    num_ranks: int = 1,
    max_bucket_bytes: int | None = None,
    unstack_counts: Mapping[str, int] | None = None,
    per_wave_shards: bool = False,
) -> SavePlan:
  """Builds an LPT bin-packed SavePlan across ranks and waves."""
  limit = max_bucket_bytes if max_bucket_bytes is not None else (1 << 62)
  if min(num_ranks, limit) < 1:
    raise ValueError(f'Invalid plan args: {num_ranks=}, {max_bucket_bytes=}')
  counts = unstack_counts or {}
  cache_key = (
      tuple(specs),
      num_ranks,
      max_bucket_bytes,
      tuple(sorted(counts.items())),
      per_wave_shards,
  )
  if cache_key in _SAVE_PLAN_CACHE:
    return _SAVE_PLAN_CACHE[cache_key]
  rank_loads = dict.fromkeys(range(num_ranks), 0)
  wave_idx = dict.fromkeys(range(num_ranks), 0)
  wave_loads = dict.fromkeys(range(num_ranks), 0)
  grid: dict[tuple[int, int], list[TensorExportSpec]] = collections.defaultdict(
      list
  )

  for spec in sorted(specs, key=lambda s: (-s.nbytes, s.source_key)):
    r = min(
        range(num_ranks),
        key=lambda k, s=spec: (
            wave_idx[k] + int(limit - s.nbytes < wave_loads[k] > 0),
            rank_loads[k],
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
  wave_shards = tuple((r, w) for w, r in sorted(grid))
  waves = tuple(
      WaveBucket(
          w,
          {r: tuple(grid[(w, r)]) for r in range(num_ranks)},
          {r: sum(s.nbytes for s in grid[(w, r)]) for r in range(num_ranks)},
      )
      for w in range(num_waves)
  )
  active = sorted({r for r, _ in wave_shards})
  rank_fname = {r: shard_filename(i, len(active)) for i, r in enumerate(active)}
  total_shd = len(wave_shards)
  shd_fnames = (
      tuple(shard_filename(i, total_shd) for i in range(total_shd))
      if per_wave_shards
      else tuple(rank_fname[r] for r, _ in wave_shards)
  )
  shard_specs = itertools.chain.from_iterable(
      [(fname, s) for s in grid[(w, r)]]
      for fname, (r, w) in zip(shd_fnames, wave_shards)
  )
  weight_map = dict(
      itertools.chain.from_iterable(
          [(hf_k, fname) for hf_k in expand_spec_keys(spec, counts)]
          for fname, spec in shard_specs
      )
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
  )
  _SAVE_PLAN_CACHE[cache_key] = plan
  return plan


@functools.partial(
    jax.jit,
    static_argnames=(
        'shape_2d',
        'transpose_axes',
        'output_dtype',
        'out_sharding',
        'flatten_ranks',
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
) -> jax.Array:
  """Jitted on-HBM LoRA merge, reshape, and transpose kernel."""
  arr = jnp.reshape(base_arr, shape_2d or base_arr.shape)
  if lora_a is not None and lora_b is not None:
    a_2d = lora_a.reshape(-1, lora_a.shape[-1])
    b_2d = lora_b.reshape(lora_b.shape[0], -1)
    delta = (a_2d @ b_2d) * lora_scaling
    arr = arr + jnp.reshape(delta, arr.shape).astype(arr.dtype)
  axes = transpose_axes or tuple(range(arr.ndim))
  out = jnp.transpose(arr, axes).astype(output_dtype or arr.dtype)
  # After on-HBM LoRA merge and axis transpose, a tensor's sharding may lie on a
  # non-leading axis or multi-axis training mesh `(fsdp, tp)`. Fusing
  # `reshape(flatten_ranks, -1)` (equivalent to `reshape(-1)` with `num_ranks *
  # 1024` alignment) and `jax.lax.with_sharding_constraint(..., out_sharding)`
  # inside `_jit_transform_on_hbm` reshards the tensor into a uniform 1D
  # `PartitionSpec('rank')` (`PartitionSpec('r')`) layout inside the same XLA
  # kernel, eliminating per-tensor eager `jax.device_put` dispatch overhead and
  # guaranteeing 1024-aligned per-rank shards for `ragged_all_to_all`. We
  # reshape to 2D `(num_ranks, -1)` rather than sharding multidimensional
  # tensors along dim 0 because: (a) a single wave batches heterogeneous
  # 1D/2D/3D tensors with different trailing shapes into one `jnp.concatenate`
  # buffer for a single `ragged_all_to_all`, (b) dim 0 may not be divisible by
  # `num_ranks` (e.g. 384 experts on 512 cores) or may yield `< 1024`
  # elements/rank, and (c) when dim 0 is divisible by `num_ranks`,
  # `reshape(num_ranks, -1)` is a zero-copy C-contiguous view identical to
  # dim-0 sharding.
  if flatten_ranks > 0:
    out = jnp.reshape(out, (flatten_ranks, -1))
  return (
      jax.lax.with_sharding_constraint(out, out_sharding)
      if out_sharding is not None
      else out
  )


def _resolve_transform_sharding(
    base_arr: jax.Array,
    spec: TensorExportSpec,
    out_sharding: jax.sharding.NamedSharding | None,
) -> tuple[jax.sharding.Sharding | None, int]:
  """Resolves target sharding and `flatten_ranks` for `_jit_transform_on_hbm`."""
  num_devs = len(out_sharding.device_set) if out_sharding is not None else 0
  unaligned = num_devs > 0 and base_arr.size % (num_devs * 1024) != 0
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
  if (
      unaligned or not infer_is_replicated(base_arr)
  ) and out_sharding is not None:
    return (
        jax.sharding.NamedSharding(
            out_sharding.mesh, jax.sharding.PartitionSpec()
        ),
        0,
    )
  return None, 0


def transform_tensor_on_hbm(
    base_arr: jax.Array,
    spec: TensorExportSpec,
    lora_a: jax.Array | None = None,
    lora_b: jax.Array | None = None,
    output_dtype: jnp.dtype | None = None,
    target_device: jax.Device | None = None,
    out_sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Applies on-HBM LoRA merge, reshape, transpose, and optional routing."""
  shd, flatten_ranks = _resolve_transform_sharding(base_arr, spec, out_sharding)
  res = _jit_transform_on_hbm(
      base_arr,
      lora_a,
      lora_b,
      spec.lora_scaling,
      spec.shape_2d,
      spec.transpose_axes,
      output_dtype,
      shd,
      flatten_ranks,
  )
  return (
      jax.device_put(res, target_device) if target_device is not None else res
  )


def build_tile_perm_row(
    tile_cols: Sequence[int],
    num_ranks: int,
    m_recv: int,
    use_ragged: bool,
) -> np.ndarray:
  """Builds 1D `(8, 128)`-tile permutation packing `(num_ranks, cols_i)` slices."""
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
) -> jax.Array:
  """Jitted shard_map collective for 1D ragged or padded all-to-all."""
  num_ranks = mesh.shape[axis_name]
  m_recv = max_recv // 1024

  def _collective_body(
      flat_inputs: Sequence[jax.Array], local_perm: jax.Array | None
  ) -> jax.Array:
    reshaped = tuple(map(jnp.ravel, (flat_inputs[i] for i in order)))
    send_buf = jnp.concatenate(reshaped, axis=0)
    my_rank = jax.lax.axis_index(axis_name)
    if use_ragged:
      s_tiles = tuple(s // 1024 for s in send_sizes)
      in_off = jnp.asarray(
          tuple(x // 1024 for x in in_offsets), dtype=jnp.int32
      )
      s_sz = jnp.asarray(s_tiles, dtype=jnp.int32)
      raw_tiles = jax.lax.ragged_all_to_all(
          send_buf.reshape(-1, 8, 128),
          jnp.zeros((num_ranks * m_recv, 8, 128), dtype=send_buf.dtype),
          in_off,
          s_sz,
          my_rank * s_sz,
          jnp.repeat(s_sz[my_rank], num_ranks),
          axis_name=axis_name,
      )
    else:
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
    return raw_tiles.reshape(-1)

  in_specs = (jax.sharding.PartitionSpec(axis_name),) * len(local_shards)
  perm_spec = None if perm is None else jax.sharding.PartitionSpec(axis_name)
  return jax.shard_map(
      _collective_body,
      mesh=mesh,
      in_specs=(in_specs, perm_spec),
      out_specs=jax.sharding.PartitionSpec(axis_name),
      check_vma=False,
  )(local_shards, perm)


def exchange_shards_ragged_all_to_all(
    local_shards: Sequence[jax.Array],
    owner_ranks: Sequence[int],
    mesh: jax.sharding.Mesh,
    axis_name: str,
) -> jax.Array:
  """Exchanges 1D sharded tensors to their owner ranks via all_to_all."""
  if len(local_shards) != len(owner_ranks):
    raise ValueError('local_shards and owner_ranks must have identical length')
  if local_shards and len({jnp.dtype(a.dtype) for a in local_shards}) > 1:
    dtypes = sorted({str(jnp.dtype(a.dtype)) for a in local_shards})
    raise ValueError(
        'exchange_shards_ragged_all_to_all requires homogeneous dtypes,'
        f' got {dtypes}.'
    )
  num_ranks = mesh.shape[axis_name]
  order = np.argsort(owner_ranks, kind='stable')
  ordered_shards = tuple(local_shards[int(i)] for i in order)
  weights = [math.prod(a.shape) // num_ranks for a in local_shards]
  rank_cols: list[list[int]] = [[] for _ in range(num_ranks)]
  for idx in order:
    rank_cols[owner_ranks[int(idx)]].append(weights[int(idx)])
  send_sizes_arr = np.bincount(
      owner_ranks, weights=weights, minlength=num_ranks
  ).astype(np.int32)
  in_offsets = np.cumsum([0] + send_sizes_arr[:-1].tolist()).astype(np.int32)
  max_recv = int(np.max(send_sizes_arr, initial=0))
  unaligned = [
      w
      for a, w in zip(local_shards, weights)
      if w <= 0 or w % 1024 != 0 or math.prod(a.shape) % (num_ranks * 1024) != 0
  ]
  if (
      max_recv <= 0
      or max_recv % 1024 != 0
      or unaligned
      or bool(np.any(send_sizes_arr % 1024))
  ):
    raise ValueError(
        'exchange_shards_ragged_all_to_all requires 1024-aligned per-rank'
        f' shards: weights={weights}, send_sizes={send_sizes_arr.tolist()},'
        f' max_recv={max_recv}'
    )
  use_ragged = jax.default_backend() == 'tpu'
  m_recv = max_recv // 1024
  need_perm = any(len(c) > 1 for c in rank_cols) or (
      not use_ragged and any(0 < sum(c) < max_recv for c in rank_cols)
  )
  perm = None
  if need_perm:
    perm_np = np.stack([
        build_tile_perm_row(
            [w // 1024 for w in rank_cols[r]], num_ranks, m_recv, use_ragged
        )
        for r in range(num_ranks)
    ])
    perm = jax.make_array_from_callback(
        (num_ranks, num_ranks * m_recv),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(axis_name)),
        lambda idx: perm_np[idx],
    )
  return _jit_exchange_shards_all_to_all(
      ordered_shards,
      mesh,
      axis_name,
      tuple(range(len(ordered_shards))),
      tuple(in_offsets.tolist()),
      tuple(send_sizes_arr.tolist()),
      max_recv,
      use_ragged,
      perm,
  )


def unpack_host_tensor(
    spec: TensorExportSpec, host_arr: np.ndarray
) -> dict[str, np.ndarray]:
  """Unpacks a host array into C_CONTIGUOUS safetensors entries."""
  if spec.unstack_axis is None:
    return {spec.hf_key: np.ascontiguousarray(host_arr)}
  count = host_arr.shape[spec.unstack_axis]
  return {
      spec.hf_key.format(expert=idx): np.ascontiguousarray(host_arr[idx])
      for idx in range(count)
  }


def transform_wave_spec(
    flat_state: Mapping[str, jax.Array],
    rank: int,
    spec: TensorExportSpec,
    output_dtype: jnp.dtype | None,
    devices: Sequence[jax.Device] | None,
    rank_shd: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
  """Transforms a single spec on HBM for its assigned wave rank."""
  del rank
  base_arr = flat_state[spec.source_key]
  num_devs = len(devices) if devices else 0
  is_rep = spec.is_replicated or infer_is_replicated(base_arr)
  out_shd = rank_shd if (num_devs > 1 and not is_rep) else None
  target_dt = jnp.dtype(output_dtype or base_arr.dtype)
  is_sharded = isinstance(
      base_arr.sharding, jax.sharding.NamedSharding
  ) and any(base_arr.sharding.spec)
  if out_shd is None:
    same_shd = not is_sharded
  else:
    same_shd = (
        base_arr.sharding == out_shd
        and base_arr.ndim == 2
        and base_arr.size % (len(out_shd.device_set) * 1024) == 0
    )
  if (
      {spec.lora_a_key, spec.shape_2d, spec.transpose_axes} == {None}
      and same_shd
      and base_arr.dtype == target_dt
  ):
    return base_arr
  return transform_tensor_on_hbm(
      base_arr,
      spec,
      lora_a=flat_state[spec.lora_a_key] if spec.lora_a_key else None,
      lora_b=flat_state[spec.lora_b_key] if spec.lora_b_key else None,
      output_dtype=output_dtype,
      out_sharding=out_shd,
  )


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

_RankItem = tuple[int, TensorExportSpec, tuple[int, ...]]
_D2HBatch = tuple[
    tuple[jax.Array, ...],
    tuple[int, ...],
    Mapping[int, Sequence[_RankItem]],
]


@dataclasses.dataclass(frozen=True)
class _WaveD2HHandle:
  """Pending asynchronous D2H transfer handle for 2-wave overlap."""

  d2h_batches: tuple[_D2HBatch, ...]
  rep_specs: tuple[tuple[int, TensorExportSpec], ...]
  rep_bufs: tuple[jax.Array, ...]
  pair_order: tuple[tuple[int, str], ...]
  defer_contiguous: bool
  hbm_sec: float


def _partition_by_dtype(
    pairs: Sequence[tuple[int, TensorExportSpec]],
    arrays: Sequence[jax.Array],
) -> list[tuple[list[tuple[int, TensorExportSpec]], list[jax.Array]]]:
  """Groups sharded `(pairs, arrays)` by `dtype` so `jnp.concatenate` matches."""
  if len(arrays) <= 1 or len({a.dtype for a in arrays}) == 1:
    return [(list(pairs), list(arrays))] if pairs else []
  by_dt: dict[
      Any, tuple[list[tuple[int, TensorExportSpec]], list[jax.Array]]
  ] = {}
  for pair, arr in zip(pairs, arrays):
    dt = arr.dtype
    if dt not in by_dt:
      by_dt[dt] = ([], [])
    by_dt[dt][0].append(pair)
    by_dt[dt][1].append(arr)
  return list(by_dt.values())


def _read_replicated_shard(arr: jax.Array, hf_key: str) -> jax.Array:
  """Returns addressable shard 0 after verifying it holds the full shape."""
  shards = arr.addressable_shards
  if not shards or shards[0].data.shape != arr.shape:
    got = shards[0].data.shape if shards else None
    raise ValueError(
        f'Expected replicated array for {hf_key!r} with shape {arr.shape},'
        f' got local shard shape {got}.'
    )
  return shards[0].data


def _reorder_host_wave(
    raw_wave: Mapping[int, Mapping[str, HostTensor]],
    pair_order: Sequence[tuple[int, str]],
) -> dict[int, dict[str, HostTensor]]:
  """Orders each rank's host tensors to match `build_rank_header_bytes`."""
  ordered: dict[int, dict[str, HostTensor]] = collections.defaultdict(dict)
  for r, key in pair_order:
    if r in raw_wave:
      ordered[r][key] = raw_wave[r][key]
  return dict(ordered)


def start_batch_d2h(
    pairs: Sequence[tuple[int, TensorExportSpec]],
    dev_arrays: Sequence[jax.Array],
    devices: Sequence[jax.Device],
) -> tuple[
    tuple[jax.Array, ...],
    tuple[int, ...],
    dict[int, list[_RankItem]],
    float,
]:
  """Executes HBM+ICI all-to-all and initiates `copy_to_host_async`."""
  t0 = time.perf_counter()
  num_ranks = len(devices)
  rank_mesh = jax.sharding.Mesh(np.array(devices), ('rank',))
  rank_shd = jax.sharding.NamedSharding(
      rank_mesh, jax.sharding.PartitionSpec('rank')
  )
  sharded_list: list[jax.Array] = []
  for a in dev_arrays:
    if a.sharding == rank_shd:
      sharded_list.append(a)
    else:
      flat_a = a.reshape(num_ranks, -1) if a.shape[0] % num_ranks != 0 else a
      sharded_list.append(jax.device_put(flat_a, rank_shd))
  sharded = tuple(sharded_list)
  owners = [r % num_ranks for r, _ in pairs]
  exchanged = exchange_shards_ragged_all_to_all(
      sharded, owners, rank_mesh, 'rank'
  )
  jax.block_until_ready(exchanged)
  hbm_sec = time.perf_counter() - t0
  rank_items: dict[int, list[_RankItem]] = collections.defaultdict(list)
  for (rk, spec), a in zip(pairs, sharded):
    rank_items[rk % num_ranks].append((rk, spec, a.shape))
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
  max_cut = max((cut[r] for r, _ in active_shards), default=0)
  d2h_targets = tuple(
      data[:max_cut] if (is_tpu and 0 < max_cut <= data.size // 2) else data
      for _, data in active_shards
  )
  for arr in d2h_targets:
    arr.copy_to_host_async()
  return (
      d2h_targets,
      tuple(r for r, _ in active_shards),
      rank_items,
      hbm_sec,
  )


def finish_batch_d2h(
    d2h_targets: Sequence[jax.Array],
    active_ranks: Sequence[int],
    rank_items: Mapping[int, Sequence[_RankItem]],
    defer_contiguous: bool = False,
) -> tuple[dict[int, dict[str, HostTensor]], float]:
  """Completes D2H transfer and constructs zero-copy 1D host slices."""
  t1 = time.perf_counter()
  local_bufs = jax.device_get(list(d2h_targets))
  host_batch: dict[int, dict[str, HostTensor]] = collections.defaultdict(dict)
  for dev_rank, buf in zip(active_ranks, local_bufs):
    buf_1d = np.asarray(buf).reshape(-1)
    offset = 0
    for plan_rank, spec, shape in rank_items[dev_rank]:
      step = math.prod(shape)
      lazy = _DeferredHostSlice(buf_1d, offset, offset + step, shape)
      host_batch[plan_rank][spec.hf_key] = (
          lazy if defer_contiguous else np.asarray(lazy)
      )
      offset += step
  return dict(host_batch), time.perf_counter() - t1


def exchange_wave_batch_to_host(
    pairs: Sequence[tuple[int, TensorExportSpec]],
    dev_arrays: Sequence[jax.Array],
    devices: Sequence[jax.Device],
    defer_contiguous: bool = False,
) -> tuple[dict[int, dict[str, HostTensor]], float, float]:
  """Routes one batch of sharded wave tensors via all-to-all to local host."""
  d2h_targets, active_ranks, rank_items, hbm_sec = start_batch_d2h(
      pairs, dev_arrays, devices
  )
  host_batch, d2h_sec = finish_batch_d2h(
      d2h_targets, active_ranks, rank_items, defer_contiguous=defer_contiguous
  )
  return host_batch, hbm_sec, d2h_sec


def route_wave_shards_to_host(
    pairs: Sequence[tuple[int, TensorExportSpec]],
    dev_arrays: Sequence[jax.Array],
    devices: Sequence[jax.Device],
    t0: float = 0.0,
    defer_contiguous: bool = False,
) -> tuple[dict[int, dict[str, HostTensor]], float, float]:
  """Routes wave tensors (replicated + sharded batches) to local host ranks."""
  start_t = time.perf_counter() if t0 <= 0.0 else t0
  num_ranks, proc = len(devices), jax.process_index()
  rep_specs: list[tuple[int, TensorExportSpec]] = []
  rep_bufs: list[jax.Array] = []
  shd_pairs: list[tuple[int, TensorExportSpec]] = []
  shd_arrays: list[jax.Array] = []
  for (r, s), a in zip(pairs, dev_arrays):
    if not (s.is_replicated or infer_is_replicated(a)):
      shd_pairs.append((r, s))
      shd_arrays.append(a)
    elif devices[r % num_ranks].process_index == proc:
      rep_specs.append((r, s))
      rep_bufs.append(_read_replicated_shard(a, s.hf_key))
  host_wave: dict[int, dict[str, HostTensor]] = collections.defaultdict(dict)
  b_hbm = b_d2h = 0.0
  batches = _partition_by_dtype(shd_pairs, shd_arrays)
  if not batches:
    jax.block_until_ready(dev_arrays)
  for dt_pairs, dt_arrays in batches:
    b_wave, batch_hbm, batch_d2h = exchange_wave_batch_to_host(
        dt_pairs, dt_arrays, devices, defer_contiguous=defer_contiguous
    )
    b_hbm += batch_hbm
    b_d2h += batch_d2h
    for r, tensors in b_wave.items():
      host_wave[r].update(tensors)
  t_d2h = time.perf_counter()
  order = None if defer_contiguous else 'C'
  for (r, spec), arr in zip(rep_specs, jax.device_get(rep_bufs)):
    host_wave[r][spec.hf_key] = np.asarray(arr, order=order)
  d2h_total = b_d2h + (time.perf_counter() - t_d2h)
  hbm_sec = (t_d2h - start_t - b_d2h) if (t0 > 0.0 or not shd_pairs) else b_hbm
  ordered = _reorder_host_wave(
      host_wave, tuple((r, s.hf_key) for r, s in pairs)
  )
  return ordered, hbm_sec, d2h_total


def dispatch_wave_on_hbm(
    flat_state: Mapping[str, jax.Array],
    wave: WaveBucket,
    output_dtype: jnp.dtype | None,
    devices: Sequence[jax.Device],
    defer_contiguous: bool = False,
) -> _WaveD2HHandle:
  """Executes wave HBM+ICI transforms and starts async D2H (`copy_to_host_async`)."""
  t0 = time.perf_counter()
  pairs = list(
      itertools.chain.from_iterable(
          zip(itertools.repeat(rank), specs)
          for rank, specs in wave.rank_assignments.items()
      )
  )
  rank_shd = jax.sharding.NamedSharding(
      jax.sharding.Mesh(np.array(devices), ('rank',)),
      jax.sharding.PartitionSpec('rank'),
  )
  dev_arrays = [
      transform_wave_spec(flat_state, r, s, output_dtype, devices, rank_shd)
      for r, s in pairs
  ]
  num_ranks, proc = len(devices), jax.process_index()
  rep_specs: list[tuple[int, TensorExportSpec]] = []
  rep_bufs: list[jax.Array] = []
  shd_pairs: list[tuple[int, TensorExportSpec]] = []
  shd_arrays: list[jax.Array] = []
  need_a2a = num_ranks > 1
  for (r, s), a in zip(pairs, dev_arrays):
    if need_a2a and not (s.is_replicated or infer_is_replicated(a)):
      shd_pairs.append((r, s))
      shd_arrays.append(a)
    elif not need_a2a:
      rep_specs.append((r, s))
      rep_bufs.append(a)
    elif devices[r % num_ranks].process_index == proc:
      rep_specs.append((r, s))
      rep_bufs.append(_read_replicated_shard(a, s.hf_key))
  d2h_batches: list[_D2HBatch] = []
  if shd_pairs:
    for dt_pairs, dt_arrays in _partition_by_dtype(shd_pairs, shd_arrays):
      d2h_targets, active_ranks, rank_items, _ = start_batch_d2h(
          dt_pairs, dt_arrays, devices
      )
      d2h_batches.append((d2h_targets, active_ranks, rank_items))
  else:
    jax.block_until_ready(dev_arrays)
  for buf in rep_bufs:
    buf.copy_to_host_async()
  return _WaveD2HHandle(
      d2h_batches=tuple(d2h_batches),
      rep_specs=tuple(rep_specs),
      rep_bufs=tuple(rep_bufs),
      pair_order=tuple((r, s.hf_key) for r, s in pairs),
      defer_contiguous=defer_contiguous,
      hbm_sec=time.perf_counter() - t0,
  )


def complete_wave_d2h(
    handle: _WaveD2HHandle,
) -> tuple[dict[int, dict[str, HostTensor]], float, float]:
  """Completes a dispatched `_WaveD2HHandle` into host rank dictionaries."""
  host_wave: dict[int, dict[str, HostTensor]] = collections.defaultdict(dict)
  b_d2h = 0.0
  for d2h_targets, active_ranks, rank_items in handle.d2h_batches:
    if d2h_targets:
      b_wave, batch_d2h = finish_batch_d2h(
          d2h_targets,
          active_ranks,
          rank_items,
          defer_contiguous=handle.defer_contiguous,
      )
      b_d2h += batch_d2h
      for r, tensors in b_wave.items():
        host_wave[r].update(tensors)
  t_d2h = time.perf_counter()
  order = None if handle.defer_contiguous else 'C'
  for (r, spec), arr in zip(
      handle.rep_specs, jax.device_get(list(handle.rep_bufs))
  ):
    host_wave[r][spec.hf_key] = np.asarray(arr, order=order)
  ordered = _reorder_host_wave(host_wave, handle.pair_order)
  return ordered, handle.hbm_sec, b_d2h + (time.perf_counter() - t_d2h)


def materialize_wave_to_host(
    flat_state: Mapping[str, jax.Array],
    wave: WaveBucket,
    output_dtype: jnp.dtype | None,
    devices: Sequence[jax.Device] | None,
    defer_contiguous: bool = False,
) -> tuple[dict[int, dict[str, HostTensor]], float, float]:
  """Executes one wave's on-HBM transforms and D2H contiguous materialization."""
  t0 = time.perf_counter()
  devs = tuple(jax.devices() if devices is None else devices)
  pairs = list(
      itertools.chain.from_iterable(
          [(rank, spec) for spec in specs]
          for rank, specs in wave.rank_assignments.items()
      )
  )
  rank_shd = None
  if devs:
    rank_shd = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.array(devs), ('rank',)),
        jax.sharding.PartitionSpec('rank'),
    )
  dev_arrays = [
      transform_wave_spec(flat_state, r, s, output_dtype, devs, rank_shd)
      for r, s in pairs
  ]
  if len(devs) > 1:
    return route_wave_shards_to_host(
        pairs, dev_arrays, devs, t0, defer_contiguous=defer_contiguous
    )
  jax.block_until_ready(dev_arrays)
  hbm_sec = time.perf_counter() - t0

  t1 = time.perf_counter()
  host_arrays = jax.device_get(dev_arrays)
  del dev_arrays
  host_wave: dict[int, dict[str, HostTensor]] = collections.defaultdict(dict)
  for (rank, spec), host_arr in zip(pairs, host_arrays):
    host_wave[rank][spec.hf_key] = (
        np.asarray(host_arr)
        if defer_contiguous
        else np.ascontiguousarray(host_arr)
    )
  d2h_sec = time.perf_counter() - t1
  return dict(host_wave), hbm_sec, d2h_sec


def build_rank_header_bytes(
    rank: int,
    plan: SavePlan,
    flat_state: Mapping[str, jax.Array],
    output_dtype: jnp.dtype | None,
    wave_index: int | None = None,
) -> bytes:
  """Builds the 8-byte length prefix and wave-ordered JSON safetensors header."""
  header: dict[str, Any] = {}
  offset = 0
  waves = (plan.waves[wave_index],) if wave_index is not None else plan.waves
  # Order `data_offsets` in exact wave execution order so local and remote
  # streams (`gs://`, remote filesystems) write strictly sequentially with
  # zero `seek()`.
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


def open_local_shard_files(
    flat_state: Mapping[str, jax.Array],
    plan: SavePlan,
    output_dir: str,
    output_dtype: jnp.dtype | None,
    devices: Sequence[jax.Device] | None,
) -> tuple[dict[Any, Any], list[str], int]:
  """Opens local rank safetensors files or prepares per-wave shard specs."""
  backend = resolve_storage_backend(output_dir)
  active = sorted({r for r, _ in plan.wave_shards})
  items = (
      plan.wave_shards
      if plan.per_wave_shards
      else tuple((r, None) for r in active)
  )
  proc = jax.process_index()
  devs = tuple(jax.devices() if devices is None else devices)
  multi_host = bool(devs) and jax.process_count() > 1
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
      entry = _OpenShardFile(backend, backend.open_writer(path))
      open_files[r] = entry
      backend.write_view(entry.file, memoryview(hdr))
  except BaseException:
    for entry in open_files.values():
      if isinstance(entry, _OpenShardFile):
        with contextlib.suppress(Exception):
          entry.close()
    raise
  return open_files, shard_paths, len(items)


def _await_all_futures(
    futs: Sequence[concurrent.futures.Future[Any]],
) -> tuple[float, BaseException | None]:
  """Awaits all futures even on error, returning `(wait_sec, first_exc)`."""
  wait_sec = 0.0
  first_exc: BaseException | None = None
  for fut in futs:
    t_wait = time.perf_counter()
    try:
      fut.result()
    except BaseException as exc:  # pylint: disable=broad-exception-caught
      first_exc = exc if first_exc is None else first_exc
    wait_sec += time.perf_counter() - t_wait
  return wait_sec, first_exc


def _write_rank_tensors(
    target: Any,
    tensors: dict[str, HostTensor],
    write_buffer_bytes: int | None = None,
    sem: threading.BoundedSemaphore | None = None,
) -> None:
  """Writes one rank's contiguous host tensors using backend stream policy."""
  with sem if sem is not None else contextlib.nullcontext():
    is_spec = isinstance(target, tuple)
    if is_spec:
      backend, f, buf = (
          target[0],
          target[0].open_writer(target[1]),
          bytearray(target[2]),
      )
    elif isinstance(target, _OpenShardFile):
      backend, f, buf = target.backend, target.file, bytearray()
    else:
      backend, f, buf = _LOCAL_BACKEND, target, bytearray()
    limit = (
        backend.default_write_buffer_bytes
        if write_buffer_bytes is None
        else write_buffer_bytes
    )
    try:
      for key in list(tensors.keys()):
        raw = memoryview(
            np.ascontiguousarray(tensors.pop(key)).view(np.uint8).reshape(-1)
        )
        if buf and len(buf) + len(raw) > limit:
          backend.write_view(f, memoryview(buf), limit)
          buf.clear()
        if len(raw) <= limit:
          buf.extend(raw)
        else:
          backend.write_view(f, raw, limit)
      if buf:
        backend.write_view(f, memoryview(buf), limit)
    finally:
      if is_spec:
        f.close()


def flush_wave_to_files(
    open_files: Mapping[Any, Any],
    host_wave: dict[int, dict[str, HostTensor]],
    wave_index: int = 0,
    write_buffer_bytes: int | None = None,
    max_workers: int = 8,
    write_sem: threading.BoundedSemaphore | None = None,
) -> float:
  """Writes one wave's contiguous host tensors using persistent `_SHARED_WRITE_POOL`."""
  t0 = time.perf_counter()
  tasks: list[tuple[Any, dict[str, HostTensor]]] = []
  for rank in list(host_wave.keys()):
    tensors = host_wave.pop(rank)
    if not tensors:
      continue
    key = (wave_index, rank) if (wave_index, rank) in open_files else rank
    if key not in open_files:
      raise KeyError(
          f'No open shard target for wave={wave_index}, rank={rank};'
          f' available keys={sorted(open_files, key=str)}.'
      )
    tasks.append((open_files[key], tensors))
  if len(tasks) <= 1 or max_workers <= 1:
    for tgt, t in tasks:
      _write_rank_tensors(tgt, t, write_buffer_bytes, write_sem)
    return time.perf_counter() - t0
  sem = (
      write_sem
      if write_sem is not None
      else threading.BoundedSemaphore(max(1, max_workers))
  )
  futs = [
      _SHARED_WRITE_POOL.submit(
          _write_rank_tensors, tgt, t, write_buffer_bytes, sem
      )
      for tgt, t in tasks
  ]
  _, first_exc = _await_all_futures(futs)
  if first_exc is not None:
    raise first_exc
  return time.perf_counter() - t0


def finalize_safetensors_metadata(
    plan: SavePlan,
    output_dir: str,
    local_model_path: str | None,
    num_active_ranks: int,
) -> None:
  """Writes model.safetensors.index.json and copies non-safetensors metadata."""
  if jax.process_index() != 0:
    return
  if num_active_ranks > 1:
    index_payload = {
        'metadata': {'total_size': plan.total_bytes},
        'weight_map': plan.weight_map,
    }
    index_path = os.path.join(output_dir, 'model.safetensors.index.json')
    backend = resolve_storage_backend(output_dir)
    index_bytes = json.dumps(index_payload, indent=2, sort_keys=True).encode(
        'utf-8'
    )
    with backend.open_writer(index_path) as f:
      backend.write_view(f, memoryview(index_bytes))
  if local_model_path is not None:
    copy_non_safetensors_files(local_model_path, output_dir)


def step_wave_to_host(
    flat_state: Mapping[str, jax.Array],
    waves: Sequence[WaveBucket],
    idx: int,
    d2h_handle: _WaveD2HHandle | None,
    output_dtype: jnp.dtype | None,
    devices: Sequence[jax.Device] | None,
    defer_contiguous: bool = True,
) -> tuple[
    dict[int, dict[str, HostTensor]], float, float, _WaveD2HHandle | None
]:
  """Advances one wave in the 2-wave HBM+ICI / D2H overlap pipeline."""
  if d2h_handle is None or devices is None:
    host_wave, hbm_sec, d2h_sec = materialize_wave_to_host(
        flat_state,
        waves[idx],
        output_dtype,
        devices,
        defer_contiguous=defer_contiguous,
    )
    return host_wave, hbm_sec, d2h_sec, None
  next_handle = None
  if idx + 1 < len(waves):
    next_handle = dispatch_wave_on_hbm(
        flat_state,
        waves[idx + 1],
        output_dtype,
        devices,
        defer_contiguous,
    )
  host_wave, hbm_sec, d2h_sec = complete_wave_d2h(d2h_handle)
  return host_wave, hbm_sec, d2h_sec, next_handle


def submit_and_drain_wave_flush(
    pending: list[concurrent.futures.Future[float]],
    limit: int,
    open_files: Mapping[Any, Any],
    host_wave: dict[int, dict[str, HostTensor]],
    wave_index: int,
    buf_bytes: int | None,
    threads: int,
    write_sem: threading.BoundedSemaphore | None = None,
) -> float:
  """Drains older flushes below `limit`, then submits a wave flush.

  Draining happens before the submit so that `limit=1` never has two flushes
  appending to one shard file at once.

  Args:
    pending: In-flight flushes, oldest first.
    limit: Maximum number of waves in flight; 0 writes synchronously.
    open_files: Open shard files or per-wave shard specs from the plan.
    host_wave: This host's `{rank: {hf_key: tensor}}` buffers for the wave.
    wave_index: Index of the wave being flushed.
    buf_bytes: Write buffer size in bytes.
    threads: Maximum writer threads for this wave.
    write_sem: Optional semaphore bounding concurrent writer threads.

  Returns:
    Seconds spent blocked on writes.
  """
  if not host_wave:
    return 0.0
  if limit <= 0:
    return flush_wave_to_files(
        open_files, host_wave, wave_index, buf_bytes, threads, write_sem
    )
  t_wait = time.perf_counter()
  while len(pending) >= limit:
    pending.pop(0).result()
  waited = time.perf_counter() - t_wait
  pending.append(
      _SHARED_WAVE_POOL.submit(
          flush_wave_to_files,
          open_files,
          host_wave,
          wave_index,
          buf_bytes,
          threads,
          write_sem,
      )
  )
  return waited


def _drain_and_close(
    pending: list[concurrent.futures.Future[float]],
    open_files: Mapping[Any, Any],
    per_wave_shards: bool,
) -> float:
  """Waits for all pending flush futures and closes persistent shard handles."""
  futs = list(pending)
  pending.clear()
  wait_sec, first_exc = _await_all_futures(futs)
  handles = () if per_wave_shards else tuple(open_files.values())
  for fh in handles:
    try:
      fh.close()
    except BaseException as exc:  # pylint: disable=broad-exception-caught
      first_exc = exc if first_exc is None else first_exc
  if first_exc is not None:
    raise first_exc
  return wait_sec


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
) -> SaveMetrics:
  """Executes a SavePlan with 2-wave HBM+ICI / D2H overlap and writes shards."""
  start_total = time.perf_counter()
  devs = tuple(jax.devices() if devices is None else devices)
  multihost = jax.process_count() > 1
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
  backend.cleanup_stale_shards(output_dir)
  if multihost:
    multihost_utils.sync_global_devices(
        f'tunix:export_safetensors:prepare:{output_dir}'
    )
  t_open = time.perf_counter()
  open_files, shard_paths, num_active = open_local_shard_files(
      flat_state, plan, output_dir, output_dtype, devs
  )
  disk_sec = time.perf_counter() - t_open
  total_hbm_sec = total_d2h_sec = 0.0
  peak_wave_rank_bytes = 0
  if not async_write:
    limit = 0
  elif plan.per_wave_shards:
    limit = min(_MAX_INFLIGHT_FLUSH_WAVES, max(1, threads))
  else:
    limit = 1
  write_sem = threading.BoundedSemaphore(max(1, threads))
  pending: list[concurrent.futures.Future[float]] = []
  d2h_handle = None
  try:
    if len(devs) > 1 < len(plan.waves):
      d2h_handle = dispatch_wave_on_hbm(
          flat_state, plan.waves[0], output_dtype, devs, defer_contiguous
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
      )
      total_hbm_sec += hbm_sec
      total_d2h_sec += d2h_sec
      disk_sec += submit_and_drain_wave_flush(
          pending,
          limit,
          open_files,
          host_wave,
          wave.wave_index,
          buf_bytes,
          threads,
          write_sem,
      )
  except BaseException:
    with contextlib.suppress(Exception):
      _drain_and_close(pending, open_files, plan.per_wave_shards)
    raise
  t_close = time.perf_counter()
  disk_sec += _drain_and_close(pending, open_files, plan.per_wave_shards)
  if multihost:
    multihost_utils.sync_global_devices(
        f'tunix:export_safetensors:pre_index:{output_dir}'
    )
  finalize_safetensors_metadata(plan, output_dir, local_model_path, num_active)
  if multihost:
    multihost_utils.sync_global_devices(
        f'tunix:export_safetensors:post_index:{output_dir}'
    )
  disk_sec += time.perf_counter() - t_close
  total_sec = time.perf_counter() - start_total + plan_time_sec
  return SaveMetrics(
      total_bytes=plan.total_bytes,
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
  a_val = jnp.asarray(getattr(lora_a, 'value', lora_a))
  b_val = jnp.asarray(getattr(lora_b, 'value', lora_b))
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
  """Saves a model with LoRA weights merged in safetensors format.

  This is a generic function that can be used for any model architecture.
  Model-specific logic is provided via callback functions.

  Args:
    local_model_path: Path to the base model safetensors checkpoint directory.
    output_dir: Directory where the merged model will be saved.
    lora_model: Model instance with LoRA weights.
    rank: LoRA rank used during training.
    alpha: LoRA alpha used during training.
    state_key_transform_fn: Function that transforms model layer paths to
      safetensors state dict keys.
    custom_layer_extractor_fn: Optional function that updates the extracted LoRA
      layer dictionary; it should accept the current layer dict and return a
      dict of the new/updated LoRA layers' names as strings to a tuple of the
      corresponding lora pair.
    transpose_rules: Optional mapping of state key substrings to transpose
      rules.
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

  # Load base model state
  base_state = safe_flax.load_file(local_model_path + '/model.safetensors')

  # Apply LoRA deltas
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

  # Save merged model
  safetensors_path = os.path.join(output_dir, 'model.safetensors')
  safe_flax.save_file(base_state, safetensors_path)

  # Copy non-safetensors files (config, tokenizer, etc.)
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
