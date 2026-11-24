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

"""Common utilities for loading model weights from safetensors files."""

import bisect
import collections
from collections.abc import Callable, Mapping, Sequence, Set as AbstractSet
import concurrent.futures
import contextlib
import functools
import io
import itertools
import json
import math
import mmap
import os
import queue
import threading
import time
from typing import Any, NamedTuple, cast

from absl import logging
from etils import epath
from flax import nnx
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from tunix.models import hf_mapping
from tunix.models import safetensors_io
from tunix.oss import utils
from tunix.utils import compat
from tunix.utils import env_utils
from tunix.utils import sharding_utils

load_file_from_gcs = utils.load_file_from_gcs
to_np_dtype = safetensors_io.to_np_dtype
_read_safetensors_header = safetensors_io.read_safetensors_header
resolve_range_sources = safetensors_io.resolve_range_sources

_CachedRangeFile = safetensors_io.CachedRangeFile
_RangeSource = safetensors_io.RangeSource
_GsHttpPath = safetensors_io.GsHttpPath
_GsHttpFile = safetensors_io.GsHttpFile
_ShmArena = safetensors_io.ShmArena

_READ_WORKERS = safetensors_io.READ_WORKERS
_MAX_STREAMS = safetensors_io.MAX_STREAMS
_READ_CHUNK_BYTES = safetensors_io.READ_CHUNK_BYTES
_DIRECT_SLICE_BYTES = safetensors_io.DIRECT_SLICE_BYTES
_MAX_COALESCE_GAP = safetensors_io.MAX_COALESCE_GAP
_MAX_COALESCE_SPAN = safetensors_io.MAX_COALESCE_SPAN
_SHARED_POOL = safetensors_io.SHARED_POOL
_PACK_POOL = safetensors_io.PACK_POOL
_H2D_POOL = concurrent.futures.ThreadPoolExecutor(8)
_MOE_LOCK = threading.Lock()
_PARALLEL_PACK_MIN_BYTES = 8 * 1024 * 1024
_H2D_QUEUE_MAXSIZE = 16
_PACK_FLUSH_BYTES = 512 * 1024 * 1024
_ICI_MIN_BYTES = 1 * 1024 * 1024
_MOE_MIN_ROW_BYTES = 256 * 1024
_RELEASE_GROUP = object()


@functools.partial(jax.jit, static_argnums=(1,), donate_argnums=(0,))
def _jit_donate_transpose(arr: jax.Array, perm: tuple[int, ...]) -> jax.Array:
  """Transposes a single-device `jax.Array` in-place via buffer donation."""
  return jax.lax.transpose(arr, perm)

_TensorTransform = tuple[Sequence[int] | None, Sequence[int] | None] | None
# (Absolute file offset, row stride in bytes, 2D uint8 view (rows, row_bytes).)
_Chunk = tuple[int, int, np.ndarray]
# (File index, file byte offset, byte length, /dev/shm byte offset.)
_RangeTask = tuple[int, int, int, int]
# (File idx, group idx, file byte offset, byte length, /dev/shm byte offset.)
_RingTask = tuple[int, int, int, int, int]
# (/dev/shm byte offset, row stride, rows, row_bytes, destination 2D view.)
_DeferredUnpack = tuple[int, int, int, int, np.ndarray]
_StreamExecutor = Callable[[Mapping[str, Any]], dict[str, Any]]


def stoi(s):
  """Convert string to int if possible, otherwise return as is."""
  try:
    return int(s)
  except ValueError:
    return s


def path_to_key(path):
  """Convert path to string key."""
  return '.'.join(
      str(stoi(key.key if hasattr(key, 'key') else key)) for key in path
  )


def flatten_model_state(model: Any) -> dict[str, jax.Array]:
  """Flattens an NNX model's parameter state into a dotted-key dictionary."""
  _, lora_state, state, _ = nnx.split(model, nnx.LoRAParam, nnx.Param, ...)
  if jax.tree_util.tree_leaves(lora_state.to_pure_dict()):
    raise ValueError(
        'Cannot flatten base model state with unmerged LoRAParam leaves.'
    )
  leaves, _ = jax.tree_util.tree_flatten_with_path(state.to_pure_dict())
  return {path_to_key(path): jnp.asarray(val) for path, val in leaves}


_validate_tensor_meta = safetensors_io.validate_tensor_meta


def resolve_checkpoint_dir(file_dir: str) -> str:
  """Resolves a `gs://` checkpoint URI to a local directory path if needed."""
  if file_dir.startswith('gs://'):
    return load_file_from_gcs(file_dir)
  return file_dir


def read_safetensors_headers(file_dir: str) -> dict[str, dict[str, Any]]:
  """Reads and merges tensor header metadata across all shards in `file_dir`."""
  file_dir = resolve_checkpoint_dir(file_dir)
  headers: dict[str, dict[str, Any]] = {}
  for f in sorted(epath.Path(file_dir).expanduser().glob('*.safetensors')):
    _, header = _read_safetensors_header(f)
    headers.update((k, v) for k, v in header.items() if k != '__metadata__')
  return headers


def read_checkpoint_keys(file_dir: str) -> frozenset[str]:
  """Returns tensor keys from `model.safetensors.index.json` or headers."""
  local_dir = resolve_checkpoint_dir(file_dir)
  index_path = (
      epath.Path(local_dir).expanduser() / 'model.safetensors.index.json'
  )
  if index_path.is_file():
    data = json.loads(index_path.read_text())
    if (
        isinstance(data, Mapping)
        and isinstance(data.get('weight_map'), Mapping)
        and data['weight_map']
    ):
      return frozenset(str(k) for k in data['weight_map'])
  return frozenset(read_safetensors_headers(local_dir))


def load_safetensors_with_offsets(filepath):
  """Loads safetensors file and returns tensor metadata with offsets.

  Args:
    filepath: The path to the safetensors file.

  Returns:
    A tuple containing:
      - contiguous_array: A numpy array containing the concatenated tensor data.
      - tensor_metadata: A list of dictionaries, each containing metadata
        (name, start_byte, end_byte, shape, dtype) for a tensor.
      - mm: The mmap object used to read the file.
      - f: The file handle.

  Raises:
    OSError: If the file or tensor byte range is truncated or out of bounds.
    ValueError: If the safetensors header or tensor metadata is invalid.
  """
  header_size, header = _read_safetensors_header(epath.Path(filepath))
  data_block_start_offset_bytes = 8 + header_size
  file_size = os.path.getsize(filepath)
  data_size_bytes = file_size - data_block_start_offset_bytes
  if data_size_bytes < 0:
    raise OSError(f'Truncated safetensors data section in {filepath}.')

  tensor_metadata = []
  for tensor_name, metadata in header.items():
    if tensor_name == '__metadata__':
      continue
    start_byte, end_byte, _, shape = _validate_tensor_meta(
        filepath, tensor_name, metadata
    )
    if end_byte > data_size_bytes:
      raise OSError(
          f'Tensor {tensor_name!r} exceeds file bounds in {filepath}.'
      )
    tensor_metadata.append({
        'name': tensor_name,
        'start_byte': start_byte,
        'end_byte': end_byte,
        'shape': shape,
        'dtype': metadata['dtype'],
    })

  f = open(filepath, 'rb')
  try:
    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    contiguous_array = np.frombuffer(
        mm,
        dtype=np.uint8,
        count=data_size_bytes,
        offset=data_block_start_offset_bytes,
    )
  except BaseException:
    f.close()
    raise

  return contiguous_array, tensor_metadata, mm, f


# ==============================================================================
# Layer 1: High-Level Streaming Read Pipeline & Byte-Range Coalescer
# ==============================================================================


class ShmRangeReader(safetensors_io.ShmArena):
  """Hardware- and cloud-agnostic `/dev/shm` ring, span coalescer, and scheduler."""

  def __init__(self, size_bytes: int = 0, shm_budget_bytes: int = 0):
    super().__init__(size_bytes=size_bytes, shm_budget_bytes=shm_budget_bytes)
    self.unit_file: list[int] = []
    self.unit_rem: list[int] = []
    self.unit_unpacks: list[list[_DeferredUnpack]] = []
    self.unit_active_readers: list[int] = []
    self.unit_consumed: list[bool] = []
    self.task_unit: list[int] = []
    self.ready_units: set[int] = set()
    self.file_last_unit: set[int] = set()
    self.oldest_unreleased_unit = 0

  def __enter__(self) -> 'ShmRangeReader':
    return self

  def reset(self) -> None:
    """Resets per-batch cursor and unit state while reusing warm `/dev/shm`."""
    self.cursor = 0
    self.ring_base = 0
    self.ring_capacity = self.capacity
    self.in_flight_bytes = 0
    self.next_admit_unit = 0
    self.oldest_unreleased_unit = 0
    self.aborted = False
    self.unit_file.clear()
    self.unit_shm_bytes.clear()
    self.unit_rem.clear()
    self.unit_unpacks.clear()
    self.unit_active_readers.clear()
    self.unit_consumed.clear()
    self.task_unit.clear()
    self.ready_units.clear()
    self.file_last_unit.clear()
    self.file_shm_bytes.clear()

  def _drain_fifo_releases(self) -> list[int]:
    """Collects contiguous consumed units with zero active readers in FIFO order."""
    ready: list[int] = []
    n = len(self.unit_consumed)
    while (
        self.oldest_unreleased_unit < n
        and self.unit_consumed[self.oldest_unreleased_unit]
        and self.unit_active_readers[self.oldest_unreleased_unit] == 0
    ):
      ready.append(self.oldest_unreleased_unit)
      self.oldest_unreleased_unit += 1
    return ready

  def release_unit(self, unit_idx: int) -> None:
    """Recycles units strictly in FIFO order after consumption and hedged reads."""
    with self.cond:
      if unit_idx >= len(self.unit_consumed):
        to_release = [unit_idx]
      else:
        self.unit_consumed[unit_idx] = True
        to_release = self._drain_fifo_releases()
    for idx in to_release:
      super().release_unit(idx)

  def window_limits(self) -> tuple[int, int, int, int]:
    """Returns `(sub_chunk, min_direct, max_gap, max_span)` for the ring."""
    sub_chunk = _READ_CHUNK_BYTES
    min_direct = min(_DIRECT_SLICE_BYTES, sub_chunk // 2)
    max_gap = min(_MAX_COALESCE_GAP, sub_chunk // 8)
    max_span = min(
        _MAX_COALESCE_SPAN, sub_chunk * 4, max(64, self.ring_capacity // 2)
    )
    return sub_chunk, min_direct, max_gap, max_span

  def record_unit(
      self,
      file_idx: int,
      shm_bytes: int,
      num_tasks: int,
      unpacks: list[_DeferredUnpack],
  ) -> None:
    """Registers a sub-file `/dev/shm` admission and unpacking unit."""
    unit_idx = len(self.unit_file)
    self.unit_file.append(file_idx)
    self.unit_shm_bytes.append(shm_bytes)
    self.unit_rem.append(num_tasks)
    self.unit_unpacks.append(unpacks)
    self.unit_active_readers.append(0)
    self.unit_consumed.append(False)
    self.task_unit.extend([unit_idx] * num_tasks)

  def stage_strided_chunk(
      self,
      offset: int,
      stride: int,
      buf_2d: np.ndarray,
      min_direct: int,
      max_gap: int,
      max_span: int,
      staged: list[tuple[int, int, int, int, int, np.ndarray]],
  ) -> None:
    """Decomposes one non-direct or strided chunk into bounded staged spans."""
    rows, row_bytes = buf_2d.shape
    if row_bytes < min_direct and stride - row_bytes <= max_gap:
      row_step = max(1, (max_span - row_bytes) // max(1, stride) + 1)
    else:
      row_step = 1
    for row_start in range(0, rows, row_step):
      row_block = buf_2d[row_start : row_start + row_step]
      block_rows = row_block.shape[0]
      block_offset = offset + row_start * stride
      col_step = max_span if block_rows == 1 else row_bytes
      block_stride = stride if block_rows > 1 else 0
      for col_off in range(0, row_bytes, col_step):
        piece_len = min(col_step, row_bytes - col_off)
        span_start = block_offset + col_off
        span_end = span_start + (block_rows - 1) * stride + piece_len
        staged.append((
            span_start,
            span_end,
            block_stride,
            block_rows,
            piece_len,
            row_block[:, col_off : col_off + piece_len],
        ))

  def split_chunk_spans(
      self,
      file_idx: int,
      chunks: Sequence[_Chunk],
      raw_tasks: list[_RangeTask],
  ) -> list[tuple[int, int, int, int, int, np.ndarray]]:
    """Splits direct chunks into tasks and normalizes staged spans `<= max_span`."""
    sub_chunk, min_direct, max_gap, max_span = self.window_limits()
    staged: list[tuple[int, int, int, int, int, np.ndarray]] = []
    direct_spans: list[tuple[int, int, int]] = []
    for offset, stride, buf_2d in sorted(chunks, key=lambda c: c[0]):
      rows, row_bytes = buf_2d.shape
      if rows * row_bytes == 0:
        continue
      base_off = buf_2d.ctypes.data - self.base_ptr
      is_in_shm = 0 <= base_off <= self.capacity - row_bytes
      if rows == 1 and row_bytes >= min_direct and is_in_shm:
        # Observed in 397B@2H weight sync: consecutive 8 MiB MoE experts sit
        # contiguously in both the shard file and the 3D /dev/shm MoE buffer,
        # issuing ~46k separate 8 MiB GETs. Coalesce adjacent direct spans
        # before 16 MiB splitting to halve GCS GET RPCs with zero extra copy.
        if (
            direct_spans
            and direct_spans[-1][0] + direct_spans[-1][1] == offset
            and direct_spans[-1][2] + direct_spans[-1][1] == base_off
        ):
          prev_off, prev_len, prev_shm = direct_spans[-1]
          direct_spans[-1] = (prev_off, prev_len + row_bytes, prev_shm)
        else:
          direct_spans.append((offset, row_bytes, base_off))
        continue
      self.stage_strided_chunk(
          offset, stride, buf_2d, min_direct, max_gap, max_span, staged
      )
    for d_off, d_len, d_shm in direct_spans:
      raw_tasks.extend(
          (file_idx, d_off + s, min(sub_chunk, d_len - s), d_shm + s)
          for s in range(0, d_len, sub_chunk)
      )
    return staged

  def find_group_end(
      self,
      staged_spans: Sequence[tuple[int, int, int, int, int, np.ndarray]],
      start_idx: int,
      min_direct: int,
      max_gap: int,
      max_span: int,
  ) -> tuple[int, int]:
    """Finds the exclusive end index and byte offset of a coalesced span group."""
    span_start = staged_spans[start_idx][0]
    span_end = staged_spans[start_idx][1]
    next_idx = start_idx + 1
    while next_idx < len(staged_spans):
      cand_start = staged_spans[next_idx][0]
      cand_end = staged_spans[next_idx][1]
      max_piece = max(span_end - span_start, cand_end - cand_start)
      allowed_gap = max_gap if max_piece < min_direct else 0
      merged_end = max(span_end, cand_end)
      if (
          cand_start - span_end > allowed_gap
          or merged_end - span_start > max_span
      ):
        break
      span_end = merged_end
      next_idx += 1
    return next_idx, span_end

  def emit_coalesced_group(
      self,
      file_idx: int,
      span_start: int,
      span_end: int,
      stage_cursor: int,
      sub_chunk: int,
      group_items: Sequence[tuple[int, int, int, int, int, np.ndarray]],
      raw_tasks: list[_RangeTask],
      deferred_unpacks: list[_DeferredUnpack],
  ) -> tuple[int, int]:
    """Allocates `/dev/shm` ring space for one coalesced group and records tasks."""
    span_len = span_end - span_start
    aligned_len = (span_len + 63) & ~63
    if stage_cursor + aligned_len > self.capacity:
      unit_shm = max(0, self.capacity - stage_cursor) + aligned_len
      span_shm_offset = self.ring_base
    else:
      unit_shm = aligned_len
      span_shm_offset = stage_cursor
    next_cursor = span_shm_offset + aligned_len
    task_start = len(raw_tasks)
    raw_tasks.extend(
        (
            file_idx,
            span_start + s,
            min(sub_chunk, span_len - s),
            span_shm_offset + s,
        )
        for s in range(0, span_len, sub_chunk)
    )
    group_unpacks = [
        (
            span_shm_offset + (item_start - span_start),
            stride,
            rows,
            row_bytes,
            buf_2d,
        )
        for item_start, _, stride, rows, row_bytes, buf_2d in group_items
    ]
    self.record_unit(
        file_idx, unit_shm, len(raw_tasks) - task_start, group_unpacks
    )
    deferred_unpacks.extend(group_unpacks)
    return next_cursor, unit_shm

  def coalesce_file_chunks(
      self,
      file_idx: int,
      chunks: Sequence[_Chunk],
      stage_cursor: int,
      raw_tasks: list[_RangeTask],
      deferred_unpacks: list[_DeferredUnpack],
  ) -> int:
    """Coalesces small/strided spans with adaptive gap and splits `16 MiB` tasks."""
    sub_chunk, min_direct, max_gap, max_span = self.window_limits()
    file_tasks_start = len(raw_tasks)
    staged_spans = self.split_chunk_spans(file_idx, chunks, raw_tasks)
    if len(raw_tasks) > file_tasks_start:
      raw_tasks[file_tasks_start:] = sorted(
          raw_tasks[file_tasks_start:], key=lambda t: -t[2]
      )
      self.record_unit(file_idx, 0, len(raw_tasks) - file_tasks_start, [])
    shm_used = 0
    idx = 0
    while idx < len(staged_spans):
      span_start = staged_spans[idx][0]
      next_idx, span_end = self.find_group_end(
          staged_spans, idx, min_direct, max_gap, max_span
      )
      stage_cursor, unit_shm = self.emit_coalesced_group(
          file_idx,
          span_start,
          span_end,
          stage_cursor,
          sub_chunk,
          staged_spans[idx:next_idx],
          raw_tasks,
          deferred_unpacks,
      )
      shm_used += unit_shm
      idx = next_idx
    if len(raw_tasks) == file_tasks_start:
      self.record_unit(file_idx, 0, 0, [])
    self.file_last_unit.add(len(self.unit_file) - 1)
    self.file_shm_bytes[file_idx] = shm_used
    return stage_cursor

  def plan_coalesced_tasks(
      self,
      planned_all: Sequence[tuple[Sequence[Any], Sequence[_Chunk]]],
  ) -> tuple[list[_RangeTask], list[_DeferredUnpack]]:
    """Builds `(file_idx, start, length, shm_off)` tasks and unpack items."""
    # Never move `ring_base` below `self.cursor`: direct `/dev/shm` tensors
    # live in `[0, self.cursor)` while staged ring spans own `[ring_base, cap)`.
    self.ring_base = (self.cursor + 63) & ~63
    self.ring_capacity = max(64, self.capacity - self.ring_base)
    stage_cursor = self.ring_base
    raw_tasks: list[_RangeTask] = []
    deferred_unpacks: list[_DeferredUnpack] = []
    for file_idx, (_, chunks) in enumerate(planned_all):
      stage_cursor = self.coalesce_file_chunks(
          file_idx, chunks, stage_cursor, raw_tasks, deferred_unpacks
      )
    self.cursor = stage_cursor
    self.ready_units = {
        unit_idx for unit_idx, rem in enumerate(self.unit_rem) if rem == 0
    }
    self.in_flight_bytes = 0
    self.next_admit_unit = 0
    self.admit_units()
    return raw_tasks, deferred_unpacks

  def unpack_one(self, item: _DeferredUnpack) -> None:
    """Unpacks one coalesced contiguous or strided span from `/dev/shm`."""
    rel_shm_off, stride, rows, row_bytes, buf_2d = item
    shm = self.shm_uint8
    if shm is None:
      return
    span = (rows - 1) * stride + row_bytes
    stage_view = shm[rel_shm_off : rel_shm_off + span]
    if rows == 1:
      buf_2d[:] = stage_view.reshape(1, row_bytes)
    else:
      buf_2d[:] = np.lib.stride_tricks.as_strided(
          stage_view, shape=(rows, row_bytes), strides=(stride, 1)
      )

  def complete_task(self, task_idx: int, done_flags: bytearray) -> None:
    """Marks `task_idx` complete and notifies the consumer when its unit is ready."""
    with self.cond:
      if done_flags[task_idx]:
        return
      done_flags[task_idx] = 1
      if task_idx < len(self.task_unit):
        unit_idx = self.task_unit[task_idx]
        self.unit_rem[unit_idx] -= 1
        if self.unit_rem[unit_idx] == 0:
          self.ready_units.add(unit_idx)
          self.cond.notify_all()

  def _acquire_unit_reader(
      self, task_idx: int, unit_idx: int, done_flags: bytearray
  ) -> tuple[bool, bool]:
    """Waits for unit admission and pins `unit_idx` against ring reuse."""
    with self.cond:
      if unit_idx >= self.next_admit_unit:
        self.cond.wait_for(
            lambda: unit_idx < self.next_admit_unit or self.aborted
        )
      if self.aborted or done_flags[task_idx]:
        return False, False
      tracked = unit_idx < len(self.unit_active_readers)
      if tracked:
        self.unit_active_readers[unit_idx] += 1
      return True, tracked

  def _release_unit_reader(self, unit_idx: int) -> None:
    """Unpins `unit_idx` and drains newly releasable FIFO ring units."""
    with self.cond:
      self.unit_active_readers[unit_idx] -= 1
      to_release = self._drain_fifo_releases()
    for idx in to_release:
      super().release_unit(idx)

  def read_one_task(
      self,
      task_idx: int,
      files: Sequence[_RangeSource],
      tasks: Sequence[_RangeTask],
      handles: dict[tuple[int, int], _CachedRangeFile],
      done_flags: bytearray,
      stream_id: int,
  ) -> None:
    """Reads a single range task `(file_idx, start, length, shm_off)` into `/dev/shm`."""
    if done_flags[task_idx]:
      return
    file_idx, start, length, shm_off = tasks[task_idx]
    unit_idx = self.task_unit[task_idx] if task_idx < len(self.task_unit) else 0
    ok, tracked = self._acquire_unit_reader(task_idx, unit_idx, done_flags)
    if not ok:
      return
    try:
      handle_key = (file_idx, stream_id)
      fh = handles.get(handle_key)
      if fh is None:
        fh = files[file_idx].open('rb', stream_id=stream_id)
        handles[handle_key] = fh
      fh.done_flags = done_flags
      fh.task_idx = task_idx
      if fh.proc_entry is not None:
        self._used_procs = True
      assert self.shm_uint8 is not None
      n = fh.read_shm_range(
          start,
          length,
          self.shm_uint8,
          shm_off,
          self.shm_path,
          self.capacity,
      )
      if n == length:
        self.complete_task(task_idx, done_flags)
      elif not done_flags[task_idx]:
        raise OSError(
            f'Read {n} of {length} bytes at offset {start} of'
            f' {files[file_idx]}.'
        )
    finally:
      if tracked:
        self._release_unit_reader(unit_idx)

  @staticmethod
  def _stagger_window_tasks(
      win_tasks: Sequence[int],
      tasks: Sequence[_RangeTask],
      host_idx: int,
  ) -> list[int]:
    """Rotates and round-robins one ring-safe window of files across hosts."""
    by_file: dict[int, list[int]] = collections.defaultdict(list)
    for t_idx in win_tasks:
      by_file[tasks[t_idx][0]].append(t_idx)
    file_lists = list(by_file.values())
    if len(file_lists) <= 1:
      return list(win_tasks)
    shift = (host_idx * 17) % len(file_lists)
    rotated = file_lists[shift:] + file_lists[:shift]
    staggered: list[int] = []
    for grp in itertools.zip_longest(*rotated):
      for t_idx in grp:
        if t_idx is not None:
          staggered.append(t_idx)
    return staggered

  def _build_staggered_task_order(
      self, tasks: Sequence[_RangeTask]
  ) -> list[int]:
    """Builds a deadlock-free, per-host staggered fetch order over `tasks`.

    Observed in 397B@2H weight sync: sequential task dispatch made both Sampler
    hosts hammer `model-00001..00064` in lock-step (throttling dense batch 0 to
    ~1.05 GB/s/host). Grouping into ring-safe windows (<=32 files and <=50% of
    ring_capacity), rotating by `jax.process_index()`, and round-robining keeps
    all 288 streams spread across distinct GCS objects while preserving
    `consume_ready_files` MoE expert-commit order and zero ring deadlock.

    Args:
      tasks: Coalesced `/dev/shm` byte-range tasks in file/offset order.

    Returns:
      Task index permutation in ring-safe staggered fetch order.
    """
    if len(tasks) <= 1 or not self.task_unit:
      return list(range(len(tasks)))
    host_idx = jax.process_index()
    ring_limit = max(1, self.ring_capacity // 2)
    order: list[int] = []
    win_tasks: list[int] = []
    win_files: set[int] = set()
    win_budget = 0
    cursor = 0
    n_tasks = len(tasks)
    for unit_idx, f_idx in enumerate(self.unit_file):
      u_start = cursor
      while cursor < n_tasks and self.task_unit[cursor] == unit_idx:
        cursor += 1
      if cursor == u_start:
        continue
      u_bytes = self.unit_shm_bytes[unit_idx]
      if win_tasks and (
          win_budget + u_bytes > ring_limit
          or (f_idx not in win_files and len(win_files) >= 32)
      ):
        order.extend(self._stagger_window_tasks(win_tasks, tasks, host_idx))
        win_tasks = []
        win_files.clear()
        win_budget = 0
      win_tasks.extend(range(u_start, cursor))
      win_files.add(f_idx)
      win_budget += u_bytes
    if win_tasks:
      order.extend(self._stagger_window_tasks(win_tasks, tasks, host_idx))
    return order

  def _hedge_remaining_tasks(
      self,
      stream_id: int,
      files: Sequence[_RangeSource],
      tasks: Sequence[_RangeTask],
      handles: dict[tuple[int, int], _CachedRangeFile],
      done_flags: bytearray,
      hedge_iter: itertools.count,
      task_order: Sequence[int] | None,
  ) -> None:
    """Speculatively re-issues already-admitted tail tasks on an alternate VIP."""
    n_tasks = len(tasks)
    while (hedge_raw := next(hedge_iter)) < n_tasks:
      pos = n_tasks - 1 - hedge_raw
      steal_idx = task_order[pos] if task_order is not None else pos
      steal_unit = (
          self.task_unit[steal_idx] if steal_idx < len(self.task_unit) else 0
      )
      # Only hedge already-admitted units so hedge workers never stall on
      # ring credit barriers for unadmitted tail shards.
      if not done_flags[steal_idx] and steal_unit < self.next_admit_unit:
        self.read_one_task(
            steal_idx, files, tasks, handles, done_flags, stream_id + 1
        )

  def worker_stream_loop(
      self,
      stream_id: int,
      files: Sequence[_RangeSource],
      tasks: Sequence[_RangeTask],
      primary_iter: itertools.count,
      hedge_iter: itertools.count,
      done_flags: bytearray,
      enable_hedge: bool,
      task_order: Sequence[int] | None = None,
  ) -> None:
    """Runs primary work-stealing followed by spread-out tail hedging."""
    n_tasks = len(tasks)
    handles: dict[tuple[int, int], _CachedRangeFile] = {}
    try:
      while (pos := next(primary_iter)) < n_tasks:
        task_idx = task_order[pos] if task_order is not None else pos
        self.read_one_task(
            task_idx, files, tasks, handles, done_flags, stream_id
        )
      if enable_hedge:
        self._hedge_remaining_tasks(
            stream_id, files, tasks, handles, done_flags, hedge_iter, task_order
        )
    except BaseException:
      with self.cond:
        self.aborted = True
        self.cond.notify_all()
      for _ in range(n_tasks):
        next(primary_iter)
        next(hedge_iter)
      raise
    finally:
      for fh in handles.values():
        fh.close()
      with self.cond:
        self.cond.notify_all()

  def consume_ready_files(
      self,
      total_files: int,
      futures: Sequence[concurrent.futures.Future[None]],
      on_file_ready: Callable[[int], None] | None,
  ) -> None:
    """Streams unit unpacks, recycles `/dev/shm` slots, and invokes consumer."""
    if on_file_ready is None:
      return
    for unit_idx in range(len(self.unit_file)):
      with self.cond:
        self.cond.wait_for(
            lambda u=unit_idx: (
                u in self.ready_units
                or self.aborted
                or all(fut.done() for fut in futures)
            )
        )
        if unit_idx not in self.ready_units:
          return
      unpacks = self.unit_unpacks[unit_idx]
      if unpacks:
        list(_PACK_POOL.map(self.unpack_one, unpacks))
        self.unit_unpacks[unit_idx] = []
      self.release_unit(unit_idx)
      if unit_idx in self.file_last_unit:
        file_idx = self.unit_file[unit_idx]
        on_file_ready(file_idx)
        self.release_file(file_idx, total_files)

  def execute_tasks(
      self,
      files: Sequence[_RangeSource],
      tasks: Sequence[_RangeTask],
      enable_hedge: bool = False,
      on_file_ready: Callable[[int], None] | None = None,
  ) -> None:
    """Executes range tasks while streaming ready files to `on_file_ready`."""
    if not tasks:
      self.consume_ready_files(len(files), (), on_file_ready)
      return
    if self.next_admit_unit == 0 and not self.unit_file:
      self.next_admit_unit = len(tasks) + 1
    total_bytes = sum(t[2] for t in tasks)
    active_procs = files[0].ensure_procs(total_bytes) if files else 0
    # Cap worker threads at _MAX_STREAMS (288 = 96 GCS procs x 3 threads/proc,
    # while keeping 16 read procs x 16 threads = 256) instead of launching
    # 16 * 96 = 1,536 queued tasks into the 288-thread SHARED_POOL.
    max_workers = min(_MAX_STREAMS, _READ_WORKERS * max(1, active_procs))
    num_workers = min(
        max_workers,
        len(tasks),
        max(1, (total_bytes + _READ_CHUNK_BYTES - 1) // _READ_CHUNK_BYTES),
    )
    if enable_hedge:
      num_workers = min(max_workers, max(num_workers, len(tasks) * 2))
    task_order = self._build_staggered_task_order(tasks)
    primary_iter = itertools.count()
    hedge_iter = itertools.count()
    done_flags = bytearray(len(tasks))
    start_w = 1 if on_file_ready is None else 0
    futures = [
        _SHARED_POOL.submit(
            self.worker_stream_loop,
            w_id,
            files,
            tasks,
            primary_iter,
            hedge_iter,
            done_flags,
            enable_hedge,
            task_order,
        )
        for w_id in range(start_w, num_workers)
    ]
    try:
      if on_file_ready is None:
        self.worker_stream_loop(
            0,
            files,
            tasks,
            primary_iter,
            hedge_iter,
            done_flags,
            enable_hedge,
            task_order,
        )
      else:
        self.consume_ready_files(len(files), futures, on_file_ready)
    finally:
      with self.cond:
        self.aborted = True
        self.cond.notify_all()
      for _ in range(len(tasks)):
        next(primary_iter)
        next(hedge_iter)
      concurrent.futures.wait(futures)
    for future in futures:
      future.result()

  @staticmethod
  def admissible_max_group(
      oldest: int,
      file_virt_start: Sequence[int],
      file_virt_end: Sequence[int],
      capacity: int,
  ) -> int:
    """Returns the highest group index admissible while `oldest` is unrecycled."""
    num_groups = len(file_virt_start)
    if oldest >= num_groups:
      return num_groups - 1
    base_virt = file_virt_start[oldest]
    max_g = oldest
    while (
        max_g + 1 < num_groups
        and file_virt_end[max_g + 1] - base_virt <= capacity
    ):
      max_g += 1
    return max_g

  @staticmethod
  def _build_diagonal_ring_tasks(
      group_raw_tasks: Sequence[Sequence[_RingTask]],
      file_virt_start: Sequence[int],
      file_virt_end: Sequence[int],
      pinned_commits: Sequence[frozenset[int]],
      capacity: int,
      ring_wraps: bool,
  ) -> tuple[list[_RingTask], list[int], list[int], list[int]]:
    """Orders ring tasks with deadlock-free 5-file diagonal staggering."""
    num_groups = len(group_raw_tasks)
    if not ring_wraps:
      diag_width = min(5, max(1, num_groups))
    else:
      min_slack = 5
      for s in range(num_groups):
        adm = ShmRangeReader.admissible_max_group(
            s, file_virt_start, file_virt_end, capacity
        )
        if adm < num_groups - 1:
          pin_max = max(pinned_commits[s])
          min_slack = min(min_slack, adm - pin_max + 1)
      diag_width = min(5, max(1, min_slack))
    host_idx = jax.process_index()
    keyed: list[tuple[float, int, int, _RingTask]] = []
    group_counts = [len(gt) for gt in group_raw_tasks]
    for g, g_tasks in enumerate(group_raw_tasks):
      denom = max(1, len(g_tasks))
      tie = (g + host_idx) % max(1, num_groups)
      for r_idx, task in enumerate(g_tasks):
        keyed.append((g + (r_idx / denom) * diag_width, tie, r_idx, task))
    keyed.sort(key=lambda item: (item[0], item[1], item[2]))
    tasks = [item[3] for item in keyed]
    group_end_idx = [0] * num_groups
    group_min_start = [len(tasks)] * num_groups
    for pos, task in enumerate(tasks):
      g = task[1]
      group_end_idx[g] = pos + 1
      if pos < group_min_start[g]:
        group_min_start[g] = pos
    suffix_min_start = [len(tasks)] * (num_groups + 1)
    for g in range(num_groups - 1, -1, -1):
      suffix_min_start[g] = min(group_min_start[g], suffix_min_start[g + 1])
    return tasks, group_counts, group_end_idx, suffix_min_start

  def _bind_ring_group(
      self,
      file_idx: int,
      items: Sequence['_PlannedTensor'],
      moe_info: Mapping[str, '_MoeRingInfo'],
      moe_bufs: dict[str, np.ndarray],
      ring_cur: '_RingCursor',
      limits: tuple[int, int, int, int],
  ) -> tuple[
      list['_HostTensor'],
      list[_Chunk],
      list[_RingTask],
      list[_DeferredUnpack],
      int,
      int,
  ]:
    """Binds direct (dense + 3D MoE) and strided buffers for one ring group."""
    sub_chunk, min_direct, max_gap, max_span = limits
    v_start = ring_cur.virt_pos
    tensors: list[_HostTensor] = []
    chunks: list[_Chunk] = []
    direct_budget = self.capacity // 2
    direct_used = 0
    for item in items:
      buf_2d = _alloc_item_buffer(
          item,
          moe_info,
          moe_bufs,
          ring_cur,
          min_direct,
          direct_budget - direct_used,
          tensors,
      )
      if item.moe_key is None and buf_2d.ctypes.data >= self.base_ptr:
        if buf_2d.ctypes.data < self.base_ptr + self.capacity:
          direct_used += (buf_2d.nbytes + 63) & ~63
      chunks.extend(_chunk_item_buffer(item, buf_2d))
    raw_direct: list[_RangeTask] = []
    staged = self.split_chunk_spans(file_idx, chunks, raw_direct)
    raw_direct.sort(key=lambda t: -t[2])
    group_tasks: list[_RingTask] = [
        (f_i, file_idx, st, ln, off) for f_i, st, ln, off in raw_direct
    ]
    group_unpacks: list[_DeferredUnpack] = []
    idx = 0
    while idx < len(staged):
      span_start = staged[idx][0]
      next_idx, span_end = self.find_group_end(
          staged, idx, min_direct, max_gap, max_span
      )
      span_len = span_end - span_start
      _, _, shm_off = ring_cur.alloc(span_len)
      for s in range(0, span_len, sub_chunk):
        group_tasks.append((
            file_idx,
            file_idx,
            span_start + s,
            min(sub_chunk, span_len - s),
            shm_off + s,
        ))
      for item_start, _, stride, rows, row_bytes, buf_2d in staged[
          idx:next_idx
      ]:
        group_unpacks.append((
            shm_off + (item_start - span_start),
            stride,
            rows,
            row_bytes,
            buf_2d,
        ))
      idx = next_idx
    return (
        tensors,
        chunks,
        group_tasks,
        group_unpacks,
        v_start,
        ring_cur.virt_pos,
    )

  def plan_circular_ring(
      self, items_per_file: Sequence[Sequence['_PlannedTensor']]
  ) -> '_CircularRingPlan | None':
    """Builds a 1-pass circular `/dev/shm` ring plan across all checkpoint files."""
    if self.shm_uint8 is None or not items_per_file:
      return None
    self.ring_base = 0
    self.ring_capacity = self.capacity
    limits = self.window_limits()
    moe_info, group_deps = _analyze_moe_ring_layout(
        items_per_file, self.capacity, limits[1]
    )
    ring_cur = _RingCursor(self.capacity, self.shm_uint8)
    num_groups = len(items_per_file)
    planned_all: list[tuple[list[_HostTensor], list[_Chunk]]] = []
    moe_bufs: dict[str, np.ndarray] = {}
    group_unpacks: list[list[_DeferredUnpack]] = []
    group_raw_tasks: list[list[_RingTask]] = []
    file_virt_start: list[int] = []
    file_virt_end: list[int] = []
    pinned_sets = [{g} for g in range(num_groups)]
    for info in moe_info.values():
      if info.ring_eligible:
        # Pin all writer groups (not just last_file) so intermediate MoE shards
        # finish unpacking into the ring-allocated buffer before it recycles.
        pinned_sets[info.first_file].update(info.writers)
    for file_idx, items in enumerate(items_per_file):
      tensors, chunks, g_tasks, g_unpacks, v_start, v_end = (
          self._bind_ring_group(
              file_idx, items, moe_info, moe_bufs, ring_cur, limits
          )
      )
      planned_all.append((tensors, chunks))
      group_raw_tasks.append(g_tasks)
      group_unpacks.append(g_unpacks)
      file_virt_start.append(v_start)
      file_virt_end.append(v_end)
      self.file_shm_bytes[file_idx] = v_end - v_start
    for key, info in moe_info.items():
      if key not in moe_bufs:
        continue
      expected_count = info.moe_shape[0]
      if info.seen_experts != frozenset(range(expected_count)):
        missing = sorted(set(range(expected_count)) - info.seen_experts)
        raise ValueError(
            f'Missing local MoE experts for {key!r}: arrived'
            f' {len(info.seen_experts)}/{expected_count}, missing local'
            f' indices {missing}.'
        )
      planned_all[info.last_file][0].append(
          _HostTensor(
              key,
              info.stack_tf,
              moe_bufs[key].ravel(),
              info.dtype,
              (expected_count, *info.shard_shape),
              info.moe_post,
          )
      )
    ring_wraps = ring_cur.virt_pos > self.capacity
    pinned_commits = [frozenset(s) for s in pinned_sets]
    if ring_wraps and _ring_exceeds_window(
        file_virt_start, file_virt_end, pinned_commits, self.capacity
    ):
      return None
    tasks, group_counts, group_end_idx, suffix_min_start = (
        self._build_diagonal_ring_tasks(
            group_raw_tasks,
            file_virt_start,
            file_virt_end,
            pinned_commits,
            self.capacity,
            ring_wraps,
        )
    )
    return _CircularRingPlan(
        planned_all=planned_all,
        moe_bufs=moe_bufs,
        group_unpacks=group_unpacks,
        tasks=tasks,
        group_counts=group_counts,
        group_end_idx=group_end_idx,
        suffix_min_start=suffix_min_start,
        file_virt_start=file_virt_start,
        file_virt_end=file_virt_end,
        pinned_commits=pinned_commits,
        group_deps=[frozenset(group_deps[g]) for g in range(num_groups)],
        total_virt_bytes=ring_cur.virt_pos,
        ring_wraps=ring_wraps,
    )

  @staticmethod
  def _poll_gcs_groups(
      pool: safetensors_io.GcsShmReadPool,
      plan: '_CircularRingPlan',
      download_done: list[bool],
      tail_since: list[float],
  ) -> None:
    """Marks completed GCS groups and triggers `0.60 s` per-group urgent hedges."""
    now = time.monotonic()
    t_cnt = int(pool.ctrl.hdr.task_counter)
    for g, done in enumerate(download_done):
      if done:
        continue
      rem = int(pool.ctrl.group_rem[g])
      if rem <= 0:
        download_done[g] = True
      elif rem <= 8 and t_cnt >= plan.group_end_idx[g]:
        if tail_since[g] == 0.0:
          tail_since[g] = now
        elif now - tail_since[g] >= safetensors_io.GCS_URGENT_HEDGE_DELAY_S:
          pool.trigger_urgent_hedge(plan.group_end_idx[g])

  def execute_gcs_ring(
      self,
      pool: safetensors_io.GcsShmReadPool,
      files: Sequence[_RangeSource],
      plan: '_CircularRingPlan',
      on_group_ready: Callable[[int, Callable[..., None]], None],
      *,
      ici_config: '_ReadIciConfig | None' = None,
  ) -> None:
    """Executes `plan` on `GcsShmReadPool` with urgent hedging & async H2D."""
    self.ensure_prefaulted(
        min(self.capacity, plan.total_virt_bytes), touch_ptes=False
    )
    self._used_gcs_pool = True
    driver = _RingH2DDriver(
        self, plan, on_group_ready, ici_config=ici_config
    )
    num_groups = len(plan.group_counts)
    batch_started = False
    failed = False
    try:
      if not plan.tasks:
        driver.run_all_empty()
        return
      file_paths = [cast(_GsHttpPath, f).http_path for f in files]
      token = cast(_GsHttpPath, files[0]).token
      pool.start_batch(
          self.shm_path,
          self.capacity,
          file_paths,
          token,
          plan.tasks,
          plan.group_counts,
          driver.current_max_allowed(),
      )
      batch_started = True
      tail_since: list[float] = [0.0] * num_groups
      download_done = [c == 0 for c in plan.group_counts]
      # When the circular ring wraps, hold credit reclamation for group `g`
      # until zero primary or hedged GCS workers read into `g`'s shm slice.
      group_active = pool.ctrl.group_active if plan.ring_wraps else None
      while not driver.all_submitted():
        pool.check_health()
        self._poll_gcs_groups(pool, plan, download_done, tail_since)
        driver.submit_ready_groups(download_done)
        pool.update_max_allowed(
            driver.reclaim_credits(group_active=group_active)
        )
        if not driver.all_submitted():
          time.sleep(0.001)
      driver.finish()
      pool.update_max_allowed(len(plan.tasks))
    except BaseException:
      failed = True
      if batch_started:
        pool.abort_batch()
      raise
    finally:
      try:
        if batch_started:
          pool.wait_batch_drained(raise_on_timeout=not failed)
      finally:
        driver.close()

  def _run_threaded_ring_worker(
      self,
      stream_id: int,
      files: Sequence[_RangeSource],
      plan: '_CircularRingPlan',
      state: '_ThreadedRingState',
  ) -> None:
    """Reads admitted circular-ring tasks over `SHARED_POOL`."""
    handles: dict[tuple[int, int], _CachedRangeFile] = {}
    num_tasks = len(plan.tasks)
    try:
      while (task_idx := state.claim_task(num_tasks)) >= 0:
        f_idx, g_idx, start, length, shm_off = plan.tasks[task_idx]
        key = (f_idx, stream_id)
        fh = handles.get(key)
        if fh is None:
          fh = files[f_idx].open('rb', stream_id=stream_id)
          handles[key] = fh
        if fh.proc_entry is not None:
          self._used_procs = True
        assert self.shm_uint8 is not None
        got = fh.read_shm_range(
            start, length, self.shm_uint8, shm_off, self.shm_path, self.capacity
        )
        if got != length:
          raise OSError(
              f'Read {got} of {length} bytes at offset {start} of'
              f' {files[f_idx]}.'
          )
        state.complete_group_task(g_idx)
    except BaseException as exc:
      state.abort(exc)
      raise
    finally:
      for fh in handles.values():
        fh.close()

  def execute_threaded_ring(
      self,
      files: Sequence[_RangeSource],
      plan: '_CircularRingPlan',
      on_group_ready: Callable[[int, Callable[..., None]], None],
      *,
      ici_config: '_ReadIciConfig | None' = None,
  ) -> None:
    """Executes `plan` via `_SHARED_POOL` with sliding-window ring & async H2D."""
    self.ensure_prefaulted(min(self.capacity, plan.total_virt_bytes))
    driver = _RingH2DDriver(
        self, plan, on_group_ready, ici_config=ici_config
    )
    state: _ThreadedRingState | None = None
    futures: list[concurrent.futures.Future[None]] = []
    try:
      if not plan.tasks:
        driver.run_all_empty()
        return
      total_bytes = sum(t[3] for t in plan.tasks)
      active_procs = files[0].ensure_procs(total_bytes) if files else 0
      max_workers = min(_MAX_STREAMS, _READ_WORKERS * max(1, active_procs))
      num_workers = min(
          max_workers,
          len(plan.tasks),
          max(1, (total_bytes + _READ_CHUNK_BYTES - 1) // _READ_CHUNK_BYTES),
      )
      state = _ThreadedRingState(
          list(plan.group_counts), driver.current_max_allowed()
      )
      futures = [
          _SHARED_POOL.submit(
              self._run_threaded_ring_worker, w_id, files, plan, state
          )
          for w_id in range(num_workers)
      ]
      download_done = [c == 0 for c in plan.group_counts]
      while not driver.all_submitted():
        state.sync_done_groups(download_done)
        driver.submit_ready_groups(download_done)
        state.advance_and_wait(driver.reclaim_credits(), driver.all_submitted())
      driver.finish()
    finally:
      if state is not None:
        state.abort(None)
      if futures:
        concurrent.futures.wait(futures)
      driver.close()
    for fut in futures:
      fut.result()


class _PlannedTensor(NamedTuple):
  """Zero-allocation tensor slice metadata parsed from one shard header."""

  file_idx: int
  base_off: int
  first_off: int
  outer_count: int
  row_stride: int
  row_bytes: int
  shard_shape: tuple[int, ...]
  dtype: np.dtype
  plan: '_SourcePlan'
  sliced_tf: _TensorTransform
  post_slices: tuple[slice, ...] | None
  moe_key: str | None
  local_expert_idx: int
  num_local_experts: int


class _MoeRingInfo(NamedTuple):
  """Cross-file lifetime and ring eligibility metadata for a 3D MoE buffer."""

  first_file: int
  last_file: int
  moe_shape: tuple[int, int, int]
  dtype: np.dtype
  shard_shape: tuple[int, ...]
  stack_tf: _TensorTransform
  moe_post: tuple[slice, ...] | None
  ring_eligible: bool
  writers: frozenset[int]
  seen_experts: frozenset[int]


class _CircularRingPlan(NamedTuple):
  """Precomputed circular `/dev/shm` layout, MoE deps, and diagonal tasks."""

  planned_all: list[tuple[list['_HostTensor'], list[_Chunk]]]
  moe_bufs: dict[str, np.ndarray]
  group_unpacks: list[list[_DeferredUnpack]]
  tasks: list[_RingTask]
  group_counts: list[int]
  group_end_idx: list[int]
  suffix_min_start: list[int]
  file_virt_start: list[int]
  file_virt_end: list[int]
  pinned_commits: list[frozenset[int]]
  group_deps: list[frozenset[int]]
  total_virt_bytes: int
  ring_wraps: bool


class _RingCursor:
  """Monotonic virtual-address allocator wrapping physically on `[0, capacity)`."""

  def __init__(self, capacity: int, shm_uint8: np.ndarray) -> None:
    self.capacity = capacity
    self.shm_uint8 = shm_uint8
    self.virt_pos = 0

  def alloc(self, nbytes: int) -> tuple[int, int, int]:
    aligned = (nbytes + 63) & ~63
    off = self.virt_pos % self.capacity
    if off + aligned > self.capacity:
      self.virt_pos += self.capacity - off
      off = 0
    v_start = self.virt_pos
    self.virt_pos += aligned
    return v_start, self.virt_pos, off


class _ThreadedRingState:
  """Synchronization state between coordinator and `_SHARED_POOL` ring workers."""

  def __init__(self, group_rem: list[int], max_allowed: int) -> None:
    self.lock = threading.Lock()
    self.cond = threading.Condition(self.lock)
    self.group_rem = group_rem
    self.max_allowed = max_allowed
    self.next_task = 0
    self.aborted = False
    self.error: BaseException | None = None

  def claim_task(self, num_tasks: int) -> int:
    """Claims the next admitted ring task index, or returns -1 when done."""
    with self.cond:
      while (
          not self.aborted
          and self.next_task < num_tasks
          and self.next_task >= self.max_allowed
      ):
        self.cond.wait(timeout=0.05)
      if self.aborted or self.next_task >= num_tasks:
        return -1
      idx = self.next_task
      self.next_task += 1
      return idx

  def complete_group_task(self, group_idx: int) -> None:
    with self.cond:
      self.group_rem[group_idx] -= 1
      if self.group_rem[group_idx] == 0:
        self.cond.notify_all()

  def sync_done_groups(self, download_done: list[bool]) -> None:
    with self.cond:
      if self.error is not None:
        raise self.error
      for g, done in enumerate(download_done):
        if not done and self.group_rem[g] <= 0:
          download_done[g] = True

  def advance_and_wait(self, new_max: int, all_submitted: bool) -> None:
    with self.cond:
      if new_max > self.max_allowed:
        self.max_allowed = new_max
        self.cond.notify_all()
      if not all_submitted and self.error is None:
        self.cond.wait(timeout=0.005)

  def abort(self, exc: BaseException | None) -> None:
    with self.cond:
      self.aborted = True
      if exc is not None and self.error is None:
        self.error = exc
      self.cond.notify_all()


class _ReadIciConfig(NamedTuple):
  """Typed ICI resharding config passed from `stream_load_files` to `_RingH2DDriver`."""

  ici_plans: Mapping[int, '_ReadIciPlan']
  ordered_ici_indices: Sequence[int]
  params: list[jax.ShapeDtypeStruct | jax.Array]
  target_dtype: np.dtype | None


class _RingH2DDriver:
  """Overlaps out-of-order CPU pack, async TPU H2D, ring recycling, and ICI."""

  def __init__(
      self,
      arena: ShmRangeReader,
      plan: _CircularRingPlan,
      on_group_ready: Callable[[int, Callable[..., None]], None],
      *,
      ici_config: _ReadIciConfig | None = None,
  ) -> None:
    self._arena = arena
    self._plan = plan
    self._on_group_ready = on_group_ready
    self._num_groups = len(plan.group_counts)
    self._ici_plans: Mapping[int, _ReadIciPlan] = (
        ici_config.ici_plans if ici_config is not None else {}
    )
    self._ordered_ici = (
        tuple(ici_config.ordered_ici_indices) if ici_config is not None else ()
    )
    self._params = ici_config.params if ici_config is not None else None
    self._target_dtype = (
        None
        if ici_config is None or ici_config.target_dtype is None
        else jnp.dtype(ici_config.target_dtype)
    )
    self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    self._h2d_q: queue.Queue[tuple[Any, ...] | None] = queue.Queue(
        maxsize=_H2D_QUEUE_MAXSIZE
    )
    self._h2d_err: BaseException | None = None
    self._ici_err: BaseException | None = None
    self._closed = False
    self._pending_group: (
        tuple[int, list[jax.Array], list[tuple[int, jax.Array]]] | None
    ) = None
    self._submitted = [False] * self._num_groups
    self._done = [False] * self._num_groups
    self._done_lock = threading.Lock()
    self._futures: list[concurrent.futures.Future[None]] = []
    self._oldest_unrecycled = 0
    self._submitted_count = 0
    self._group_enqueued = False
    self._ici_cond = threading.Condition()
    self._ici_ready: dict[int, jax.Array] = {}
    self._ici_aborted = False
    self._ici_eof = False
    self._ici_thread: threading.Thread | None = None
    if self._ordered_ici:
      self._ici_thread = threading.Thread(target=self._ici_loop, daemon=True)
      self._ici_thread.start()
    self._h2d_thread = threading.Thread(target=self._h2d_loop, daemon=True)
    self._h2d_thread.start()

  def _flush_pending_group(self) -> None:
    """Waits for DMA if ring wraps, releases ring credits, then wakes ICI."""
    pending = self._pending_group
    if pending is None:
      return
    self._pending_group = None
    group_idx, placed, ici_items = pending
    if self._plan.ring_wraps and placed:
      jax.block_until_ready(placed)
    self._arena.release_file(group_idx, self._num_groups)
    with self._done_lock:
      self._done[group_idx] = True
    if ici_items:
      with self._ici_cond:
        for idx, arr in ici_items:
          self._ici_ready[idx] = arr
        self._ici_cond.notify_all()

  def _finish_h2d_loop(self, cur_placed: list[jax.Array]) -> None:
    """Flushes or blocks live DMA buffers and signals `_ici_thread` on exit."""
    try:
      if self._h2d_err is None and not self._closed:
        self._flush_pending_group()
      else:
        if self._pending_group is not None:
          cur_placed.extend(self._pending_group[1])
          self._pending_group = None
        if cur_placed:
          jax.block_until_ready(cur_placed)
    except BaseException as exc:  # pylint: disable=broad-exception-caught
      self._h2d_err = self._h2d_err or exc
    finally:
      if self._ici_thread is not None:
        with self._ici_cond:
          if self._h2d_err is not None or self._closed:
            self._ici_aborted = True
          else:
            self._ici_eof = True
          self._ici_cond.notify_all()

  def _h2d_loop(self) -> None:
    """Commits packed shards to HBM and releases wrapped ring groups safely."""
    cur_placed: list[jax.Array] = []
    cur_ici: list[tuple[int, jax.Array]] = []
    try:
      while True:
        timeout = 0.005 if self._pending_group is not None else None
        try:
          item = self._h2d_q.get(timeout=timeout)
        except queue.Empty:
          self._flush_pending_group()
          continue
        if item is None:
          break
        if item[0] is _RELEASE_GROUP:
          self._flush_pending_group()
          self._pending_group = (item[1], cur_placed, cur_ici)
          if not self._plan.ring_wraps:
            self._flush_pending_group()
          cur_placed, cur_ici = [], []
          continue
        placed, ici_items = _commit_packed_shards(
            *item, ici_param_indices=self._ici_plans or None
        )
        cur_placed.extend(placed)
        cur_ici.extend(ici_items)
    except BaseException as exc:  # pylint: disable=broad-exception-caught
      self._h2d_err = exc
    finally:
      self._finish_h2d_loop(cur_placed)

  def _ici_loop(self) -> None:
    """Dispatches inverse `shard_map` kernels in lockstep canonical order."""
    assert self._params is not None
    staged_arr: jax.Array | None = None
    try:
      for param_idx in self._ordered_ici:
        with self._ici_cond:
          while (
              param_idx not in self._ici_ready
              and not self._ici_aborted
              and not self._ici_eof
          ):
            self._ici_cond.wait(timeout=0.05)
          if self._ici_aborted:
            return
          if param_idx not in self._ici_ready:
            raise RuntimeError(
                f'Missing ICI staged parameter at index {param_idx}.'
            )
          staged_arr = self._ici_ready.pop(param_idx)
        self._params[param_idx] = _run_read_ici_kernel(
            self._ici_plans[param_idx], staged_arr, self._target_dtype
        )
        staged_arr = None
    except BaseException as exc:  # pylint: disable=broad-exception-caught
      with self._ici_cond:
        self._ici_err = exc
        self._ici_aborted = True
        self._ici_cond.notify_all()
    finally:
      if not self._plan.ring_wraps:
        with self._ici_cond:
          pending = list(self._ici_ready.values())
          self._ici_ready.clear()
        if staged_arr is not None:
          pending.append(staged_arr)
        live = [a for a in pending if not a.is_deleted()]
        if live:
          jax.block_until_ready(live)

  def enqueue_h2d(self, *payload: Any) -> None:
    """Enqueues one packed shard batch onto the bounded async H2D stage."""
    if payload and payload[0] is not _RELEASE_GROUP:
      self._group_enqueued = True
    while True:
      if (err := self._h2d_err or self._ici_err) is not None:
        raise err
      if self._closed:
        return
      try:
        self._h2d_q.put(payload, timeout=0.05)
        return
      except queue.Full:
        continue

  def all_submitted(self) -> bool:
    return self._submitted_count >= self._num_groups

  def current_max_allowed(self) -> int:
    adm = ShmRangeReader.admissible_max_group(
        self._oldest_unrecycled,
        self._plan.file_virt_start,
        self._plan.file_virt_end,
        self._arena.capacity,
    )
    return self._plan.suffix_min_start[adm + 1]

  def _consume_group(self, group_idx: int) -> None:
    """Unpacks a downloaded group and enqueues its H2D + release work."""
    unpacks = self._plan.group_unpacks[group_idx]
    if unpacks:
      list(_PACK_POOL.map(self._arena.unpack_one, unpacks))
      self._plan.group_unpacks[group_idx] = []
    self._group_enqueued = False
    self._on_group_ready(group_idx, self.enqueue_h2d)
    if self._group_enqueued:
      self.enqueue_h2d(_RELEASE_GROUP, group_idx)
    else:
      self._arena.release_file(group_idx, self._num_groups)
      with self._done_lock:
        self._done[group_idx] = True

  def submit_ready_groups(self, download_done: Sequence[bool]) -> None:
    for g in range(self._num_groups):
      if self._submitted[g] or not download_done[g]:
        continue
      if all(self._submitted[d] for d in self._plan.group_deps[g]):
        self._submitted[g] = True
        self._submitted_count += 1
        self._futures.append(self._executor.submit(self._consume_group, g))

  def reclaim_credits(
      self, group_active: Sequence[int] | None = None
  ) -> int:
    """Propagates consumer errors and advances the admissible ring window."""
    if (err := self._h2d_err or self._ici_err) is not None:
      raise err
    for fut in self._futures:
      if fut.done():
        fut.result()
    with self._done_lock:
      while self._oldest_unrecycled < self._num_groups and all(
          self._done[c] and (group_active is None or int(group_active[c]) == 0)
          for c in self._plan.pinned_commits[self._oldest_unrecycled]
      ):
        self._oldest_unrecycled += 1
    return self.current_max_allowed()

  def run_all_empty(self) -> None:
    try:
      self.submit_ready_groups([True] * self._num_groups)
      self.finish()
    finally:
      self.close()

  def finish(self) -> None:
    """Drains unpack, H2D, and ICI stages and propagates any worker error."""
    for fut in self._futures:
      fut.result()
    while self._h2d_thread.is_alive():
      try:
        self._h2d_q.put(None, timeout=0.05)
        break
      except queue.Full:
        continue
    self._h2d_thread.join()
    if self._ici_thread is not None:
      self._ici_thread.join()
    if (err := self._h2d_err or self._ici_err) is not None:
      raise err

  def close(self) -> None:
    """Shuts down all worker stages and fences any pending staged arrays."""
    self._closed = True
    self._executor.shutdown(wait=True)
    if self._h2d_thread.is_alive():
      while True:
        try:
          self._h2d_q.get_nowait()
        except queue.Empty:
          break
      self._h2d_q.put(None)
      self._h2d_thread.join()
    if self._ici_thread is not None:
      with self._ici_cond:
        self._ici_aborted = True
        self._ici_cond.notify_all()
      self._ici_thread.join()
    if self._ici_ready:
      live = [a for a in self._ici_ready.values() if not a.is_deleted()]
      self._ici_ready.clear()
      if live:
        jax.block_until_ready(live)


def _ring_exceeds_window(
    file_virt_start: Sequence[int],
    file_virt_end: Sequence[int],
    pinned_commits: Sequence[frozenset[int]],
    capacity: int,
) -> bool:
  """Returns True if any group or pinned MoE span cannot fit in `capacity`."""
  half = capacity // 2
  for s, (vs, ve) in enumerate(zip(file_virt_start, file_virt_end)):
    if ve - vs > half:
      return True
    if file_virt_end[max(pinned_commits[s])] - vs > capacity:
      return True
  return False


_MOE_STACK_TF: Mapping[_TensorTransform, _TensorTransform] = {
    ((1, 0), None): ((0, 2, 1), None),
}


def _analyze_moe_ring_layout(
    items_per_file: Sequence[Sequence[_PlannedTensor]],
    capacity: int,
    min_direct: int,
) -> tuple[dict[str, _MoeRingInfo], list[set[int]]]:
  """Computes cross-file MoE spans, ring eligibility, and H2D dependencies."""
  num_groups = len(items_per_file)
  file_est = [0] * num_groups
  moe_writers: dict[str, set[int]] = {}
  moe_seen: dict[str, set[int]] = {}
  moe_sigs: dict[str, set[tuple[int, int, np.dtype, tuple[int, ...], int]]] = {}
  moe_sample: dict[str, _PlannedTensor] = {}
  for g, items in enumerate(items_per_file):
    for item in items:
      if item.moe_key is None:
        file_est[g] += (item.outer_count * item.row_bytes + 63) & ~63
        continue
      moe_writers.setdefault(item.moe_key, set()).add(g)
      moe_seen.setdefault(item.moe_key, set()).add(item.local_expert_idx)
      moe_sigs.setdefault(item.moe_key, set()).add((
          item.outer_count,
          item.row_bytes,
          item.dtype,
          item.shard_shape,
          item.num_local_experts,
      ))
      moe_sample.setdefault(item.moe_key, item)
  for key, writers in moe_writers.items():
    if len(moe_sigs[key]) > 1:
      raise ValueError(
          f'Inconsistent MoE expert geometry or dtype for {key!r}:'
          f' {sorted(str(s) for s in moe_sigs[key])}.'
      )
    it = moe_sample[key]
    file_est[min(writers)] += (
        it.num_local_experts * it.outer_count * it.row_bytes + 63
    ) & ~63
  moe_info: dict[str, _MoeRingInfo] = {}
  group_deps: list[set[int]] = [set() for _ in range(num_groups)]
  for key, writers in moe_writers.items():
    it = moe_sample[key]
    first_f, last_f = min(writers), max(writers)
    moe_shape = (it.num_local_experts, it.outer_count, it.row_bytes)
    moe_nbytes = math.prod(moe_shape)
    span_est = sum(file_est[first_f : last_f + 1])
    ring_ok = (
        moe_nbytes >= min_direct
        and it.outer_count == 1
        and it.row_bytes >= min(min_direct, _MOE_MIN_ROW_BYTES)
        and moe_nbytes <= capacity // 4
        and span_est <= capacity // 2
    )
    stack_tf = _MOE_STACK_TF.get(it.sliced_tf)
    moe_post = (
        (slice(None), *it.post_slices) if it.post_slices is not None else None
    )
    moe_info[key] = _MoeRingInfo(
        first_f, last_f, moe_shape, it.dtype, it.shard_shape,
        stack_tf, moe_post, ring_ok, frozenset(writers),
        frozenset(moe_seen[key]),
    )
    group_deps[last_f].update(writers - {last_f})
  return moe_info, group_deps


def _alloc_item_buffer(
    item: _PlannedTensor,
    moe_info: Mapping[str, _MoeRingInfo],
    moe_bufs: dict[str, np.ndarray],
    ring_cur: _RingCursor,
    min_direct: int,
    direct_rem: int,
    tensors: list['_HostTensor'],
) -> np.ndarray:
  """Allocates a circular-ring `/dev/shm` or heap buffer for one tensor item."""
  if item.moe_key is not None:
    if item.moe_key not in moe_bufs:
      info = moe_info[item.moe_key]
      nbytes = math.prod(info.moe_shape)
      if info.ring_eligible:
        _, _, off = ring_cur.alloc(nbytes)
        moe_bufs[item.moe_key] = ring_cur.shm_uint8[
            off : off + nbytes
        ].reshape(info.moe_shape)
      else:
        moe_bufs[item.moe_key] = np.empty(info.moe_shape, dtype=np.uint8)
    return moe_bufs[item.moe_key][item.local_expert_idx]
  nbytes = item.outer_count * item.row_bytes
  can_direct = (
      not item.plan.is_raw_source
      and item.outer_count == 1
      and item.row_bytes >= min_direct
      and nbytes <= ring_cur.capacity // 4
      and nbytes <= direct_rem
  )
  if can_direct:
    _, _, off = ring_cur.alloc(nbytes)
    buf_2d = ring_cur.shm_uint8[off : off + nbytes].reshape((1, item.row_bytes))
  else:
    buf_2d = np.empty((item.outer_count, item.row_bytes), dtype=np.uint8)
  tensors.append(
      _HostTensor(
          item.plan.target_key,
          item.sliced_tf,
          buf_2d.ravel(),
          item.dtype,
          item.shard_shape,
          item.post_slices,
      )
  )
  return buf_2d


def _chunk_item_buffer(
    item: _PlannedTensor, buf_2d: np.ndarray
) -> list[_Chunk]:
  """Splits one tensor item buffer into bounded `_READ_CHUNK_BYTES` chunks."""
  if item.outer_count == 1:
    return [
        (
            item.base_off + item.first_off + off,
            0,
            buf_2d[:, off : off + _READ_CHUNK_BYTES],
        )
        for off in range(0, item.row_bytes, _READ_CHUNK_BYTES)
    ]
  step = max(1, _READ_CHUNK_BYTES // max(1, min(item.row_stride, 65536)))
  return [
      (
          item.base_off + item.first_off + r * item.row_stride,
          item.row_stride,
          buf_2d[r : r + step],
      )
      for r in range(0, item.outer_count, step)
  ]


# ==============================================================================
# Layer 2: Accelerator & SPMD Host-Buffer Layer (TPU / PJRT + `RuleSpec`)
# ==============================================================================


class _TargetSpec(NamedTuple):
  """Target parameter metadata and local-host shard geometry."""

  param_index: int
  global_shape: tuple[int, ...]
  host_shape: tuple[int, ...]
  sharding: jax.sharding.Sharding | jax.Device
  host_slices: tuple[slice, ...]
  device_slices: tuple[tuple[jax.Device, tuple[slice, ...]], ...]


class _ReadIciPlan(NamedTuple):
  """Geometric inverse-`shard_map` ICI plan for one target parameter."""

  mesh: jax.sharding.Mesh
  src_shape: tuple[int, ...]
  target_shape: tuple[int, ...]
  axes: tuple[int, ...]
  stage_axis: int
  stage_spec: jax.sharding.PartitionSpec
  host_stage_spec: jax.sharding.PartitionSpec
  host_stage_sharding: jax.sharding.NamedSharding
  post_tr_spec: jax.sharding.PartitionSpec
  target_sharding: jax.sharding.NamedSharding
  rep_axes: tuple[Any, ...]
  inner_ops: tuple[tuple[Any, int], ...]
  rep_spans_hosts: bool


def as_axis_tuple(entry: Any) -> tuple[Any, ...]:
  """Normalizes a `PartitionSpec` dimension entry to a tuple of axis names."""
  if entry is None:
    return ()
  return entry if isinstance(entry, tuple) else (entry,)


def _consume_merged_dim(
    in_shape: tuple[int, ...],
    sp_in: tuple[tuple[Any, ...], ...],
    in_mesh: jax.sharding.Mesh,
    idx: int,
    target_d: int,
) -> tuple[int, Any] | None:
  """Consumes contiguous source dims whose product equals `target_d`."""
  n_in = len(in_shape)
  if target_d == 1:
    step = int(idx < n_in and in_shape[idx] == 1 and not sp_in[idx])
    return idx + step, None
  prod_d, axes_g, prev_one = 1, [], True
  while prod_d < target_d and idx < n_in:
    dim_sz, dim_axes = in_shape[idx], sp_in[idx]
    shard_prod = math.prod(in_mesh.shape[a] for a in dim_axes)
    if dim_sz % shard_prod != 0 or (dim_axes and not prev_one):
      return None
    axes_g.extend(dim_axes)
    prev_one = prev_one and (dim_sz // shard_prod == 1)
    prod_d *= dim_sz
    idx += 1
  if prod_d != target_d:
    return None
  return idx, (axes_g[0] if len(axes_g) == 1 else (tuple(axes_g) or None))


def merge_reshape_spec(
    in_shape: tuple[int, ...],
    in_spec: jax.sharding.PartitionSpec,
    in_mesh: jax.sharding.Mesh,
    target_shape: tuple[int, ...],
) -> tuple[Any, ...] | None:
  """Maps `in_spec` across a contiguous reshape `in_shape -> target_shape`."""
  n_in = len(in_shape)
  raw_axes = [
      as_axis_tuple(in_spec[i] if i < len(in_spec) else None)
      for i in range(n_in)
  ]
  if any(
      a not in in_mesh.shape for a in itertools.chain.from_iterable(raw_axes)
  ):
    return None
  sp_in = tuple(
      tuple(a for a in axes if in_mesh.shape[a] > 1) for axes in raw_axes
  )
  merged, idx = [], 0
  for target_d in target_shape:
    step = _consume_merged_dim(in_shape, sp_in, in_mesh, idx, target_d)
    if step is None:
      return None
    idx, ax = step
    merged.append(ax)
  rem_ok = all(in_shape[i] == 1 and not sp_in[i] for i in range(idx, n_in))
  return tuple(merged) if rem_ok else None


def _host_dim_spans(
    sharding: jax.sharding.NamedSharding, shape: tuple[int, ...], axis: int
) -> dict[int, tuple[int, int, int]]:
  """Returns `{process_index: (min_start, max_stop, covered_len)}` on `axis`."""
  by_proc: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
  dim = shape[axis]
  for dev, idx in sharding.devices_indices_map(shape).items():
    lo, hi = (idx[axis] or slice(None)).indices(dim)[:2]
    by_proc[dev.process_index].append((lo, hi))
  return {
      p: (
          min(a for a, _ in sp),
          max(b for _, b in sp),
          sum(b - a for a, b in sp),
      )
      for p, sp in by_proc.items()
  }


def _axis_pspec(
    ndim: int, axis: int, axes: tuple[Any, ...]
) -> jax.sharding.PartitionSpec:
  """Builds an `ndim`-D `PartitionSpec` sharding only `axis` over `axes`."""
  elem = axes[0] if len(axes) == 1 else axes
  return jax.sharding.PartitionSpec(
      *([None] * axis + [elem] + [None] * (ndim - 1 - axis))
  )


def _resolve_host_stage_spec(
    mesh: jax.sharding.Mesh,
    stage_spec: jax.sharding.PartitionSpec,
    active_axes: tuple[Any, ...],
    src_shape: tuple[int, ...],
    stage_axis: int,
    target_sharding: jax.sharding.NamedSharding,
    target_shape: tuple[int, ...],
    is_moe: bool,
) -> jax.sharding.PartitionSpec | None:
  """Selects a host-contiguous staging spec (`stage_spec` or `active_axes`)."""
  active_spec = _axis_pspec(len(src_shape), stage_axis, active_axes)
  candidates = (
      (stage_spec,) if active_spec == stage_spec else (stage_spec, active_spec)
  )
  tgt_spans = (
      _host_dim_spans(target_sharding, target_shape, 0) if is_moe else None
  )
  for cand in candidates:
    spans = _host_dim_spans(
        jax.sharding.NamedSharding(mesh, cand), src_shape, stage_axis
    )
    if len(spans) <= 1:
      return cand
    if not all(cov == hi - lo for lo, hi, cov in spans.values()):
      continue
    if tgt_spans is None or all(
        tgt_spans[p][0] <= lo and hi <= tgt_spans[p][1]
        for p, (lo, hi, _) in spans.items()
    ):
      return cand
  return None


def _rep_spans_hosts(
    mesh: jax.sharding.Mesh,
    active_axes: tuple[Any, ...],
    rep_axes: tuple[Any, ...],
) -> bool:
  """Returns True iff replication along `rep_axes` spans multiple hosts."""
  if not rep_axes:
    return False
  used_pos = tuple(
      i
      for i, ax in enumerate(mesh.axis_names)
      if ax in active_axes and ax not in rep_axes
  )
  seen: dict[tuple[int, ...], int] = {}
  for coord in np.ndindex(mesh.devices.shape):
    key = tuple(coord[i] for i in used_pos)
    proc = mesh.devices[coord].process_index
    if seen.setdefault(key, proc) != proc:
      return True
  return False


def _select_read_stage_ops(
    mesh_size: int,
    active_axes: tuple[Any, ...],
    pre_tr_list: Sequence[tuple[Any, ...]],
    src_shape: tuple[int, ...],
    is_moe: bool,
) -> (
    tuple[
        int,
        tuple[Any, ...],
        tuple[tuple[Any, int], ...],
        jax.sharding.PartitionSpec,
    ]
    | None
):
  """Selects `stage_axis`, `rep_axes`, `inner_ops`, and `stage_spec`."""
  stage_axis = None
  if src_shape[0] % mesh_size == 0:
    stage_axis = 0
  elif (
      not is_moe
      and len(src_shape) >= 2
      and src_shape[0] <= 8
      and not pre_tr_list[0]
      and src_shape[1] % mesh_size == 0
  ):
    stage_axis = 1
  if stage_axis is None:
    return None
  used = set(itertools.chain.from_iterable(pre_tr_list))
  rep_axes = tuple(ax for ax in active_axes if ax not in used)
  inner_groups = [
      [(a, d) for a in reversed(dim_ax)]
      for d, dim_ax in enumerate(pre_tr_list)
      if d != stage_axis and dim_ax
  ]
  inner_groups.sort(key=lambda grp: active_axes.index(grp[0][0]))
  inner_ops = tuple(itertools.chain.from_iterable(inner_groups))
  stage_axes = (
      pre_tr_list[stage_axis] + rep_axes + tuple(ax for ax, _ in inner_ops)
  )
  stage_spec = _axis_pspec(len(src_shape), stage_axis, stage_axes)
  return stage_axis, rep_axes, inner_ops, stage_spec


def plan_read_ici_reshard(
    src_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    target_sharding: jax.sharding.Sharding | jax.Device,
    permute: Sequence[int] | None = None,
    *,
    is_moe: bool = False,
) -> _ReadIciPlan | None:
  """Plans zero-hardcoding inverse `shard_map` ICI resharding for one tensor."""
  if not isinstance(target_sharding, jax.sharding.NamedSharding):
    return None
  mesh = target_sharding.mesh
  if (
      not isinstance(mesh, jax.sharding.Mesh)
      or not src_shape
      or 0 in src_shape
      or mesh.size <= 1
  ):
    return None
  axes = tuple(range(len(src_shape))) if permute is None else tuple(permute)
  if len(axes) != len(src_shape):
    return None
  post_tr_shape = tuple(src_shape[p] for p in axes)
  if math.prod(post_tr_shape) != math.prod(target_shape):
    return None
  post_tr_sp = merge_reshape_spec(
      target_shape, target_sharding.spec, mesh, post_tr_shape
  )
  if post_tr_sp is None:
    return None
  active_axes = tuple(ax for ax in mesh.axis_names if mesh.shape[ax] > 1)
  active_set = set(active_axes)
  pre_tr_list: list[tuple[Any, ...]] = [() for _ in range(len(src_shape))]
  for j, src_d in enumerate(axes):
    pre_tr_list[src_d] = tuple(
        a for a in as_axis_tuple(post_tr_sp[j]) if a in active_set
    )
  selected = _select_read_stage_ops(
      mesh.size, active_axes, pre_tr_list, src_shape, is_moe
  )
  if selected is None:
    return None
  stage_axis, rep_axes, inner_ops, stage_spec = selected
  host_stage_spec = _resolve_host_stage_spec(
      mesh, stage_spec, active_axes, src_shape, stage_axis,
      target_sharding, target_shape, is_moe,
  )
  if host_stage_spec is None:
    return None
  return _ReadIciPlan(
      mesh=mesh,
      src_shape=src_shape,
      target_shape=target_shape,
      axes=axes,
      stage_axis=stage_axis,
      stage_spec=stage_spec,
      host_stage_spec=host_stage_spec,
      host_stage_sharding=jax.sharding.NamedSharding(mesh, host_stage_spec),
      post_tr_spec=jax.sharding.PartitionSpec(*post_tr_sp),
      target_sharding=target_sharding,
      rep_axes=rep_axes,
      inner_ops=inner_ops,
      rep_spans_hosts=_rep_spans_hosts(mesh, active_axes, rep_axes),
  )


def make_read_ici_kernel(
    plan: _ReadIciPlan, out_dtype: jnp.dtype | None = None
) -> Any:
  """Builds a donated JIT kernel resharding contiguous host input to target."""
  identity_tr = plan.axes == tuple(range(len(plan.axes)))
  stage_shd = (
      jax.sharding.NamedSharding(plan.mesh, plan.stage_spec)
      if plan.host_stage_spec != plan.stage_spec
      else None
  )
  cast_dtype = None if out_dtype is None else jnp.dtype(out_dtype)

  def _body(loc: jax.Array) -> jax.Array:
    for ax, d in reversed(plan.inner_ops):
      loc = jax.lax.all_to_all(
          loc, axis_name=ax, split_axis=d, concat_axis=plan.stage_axis,
          tiled=True,
      )
    for ax in reversed(plan.rep_axes):
      loc = jax.lax.all_gather(
          loc, axis_name=ax, axis=plan.stage_axis, tiled=True
      )
    return loc if identity_tr else jnp.transpose(loc, plan.axes)

  @functools.partial(jax.jit, donate_argnums=(0,))
  def _kernel(staged_x: jax.Array) -> jax.Array:
    narrow = (
        cast_dtype is not None
        and cast_dtype.itemsize < staged_x.dtype.itemsize
    )
    x = staged_x.astype(cast_dtype) if narrow else staged_x
    if stage_shd is not None:
      x = jax.lax.with_sharding_constraint(x, stage_shd)
    x = jax.shard_map(
        _body, mesh=plan.mesh, in_specs=plan.stage_spec,
        out_specs=plan.post_tr_spec, check_vma=False,
    )(x)
    x = jnp.reshape(x, plan.target_shape)
    if cast_dtype is not None and x.dtype != cast_dtype:
      x = x.astype(cast_dtype)
    return jax.lax.with_sharding_constraint(x, plan.target_sharding)

  return _kernel


@functools.lru_cache(maxsize=256)
def _compile_read_ici_kernel(
    plan: _ReadIciPlan,
    src_dtype: jnp.dtype,
    out_dtype: jnp.dtype | None,
) -> Any:
  """Compiles and caches the inverse `shard_map` ICI kernel for `plan`."""
  sds = jax.ShapeDtypeStruct(
      plan.src_shape, src_dtype, sharding=plan.host_stage_sharding
  )
  return make_read_ici_kernel(plan, out_dtype=out_dtype).lower(sds).compile()


def _run_read_ici_kernel(
    plan: _ReadIciPlan,
    staged_x: jax.Array,
    out_dtype: jnp.dtype | None,
) -> jax.Array:
  """Executes the AOT-cached inverse `shard_map` kernel for `plan`."""
  return _compile_read_ici_kernel(
      plan, jnp.dtype(staged_x.dtype), out_dtype
  )(staged_x)


def _compute_ordered_ici_indices(
    ici_plans: Mapping[int, _ReadIciPlan],
    planned_all: Sequence[tuple[Sequence['_HostTensor'], Sequence[_Chunk]]],
    stage_specs: Mapping[str, _TargetSpec],
) -> list[int]:
  """Computes a host-identical canonical ICI execution order across all hosts."""
  sorted_ids = sorted(ici_plans)
  if not sorted_ids:
    return []
  local_last: dict[int, int] = {}
  for file_idx, (file_tensors, _) in enumerate(planned_all):
    for t in file_tensors:
      if t.jax_key in stage_specs:
        local_last[stage_specs[t.jax_key].param_index] = file_idx
  ranks = [local_last.get(idx, 0) for idx in sorted_ids]
  if jax.process_count() > 1:
    with jax.transfer_guard('allow'):
      gathered = np.asarray(
          multihost_utils.process_allgather(
              np.asarray(ranks, dtype=np.int32), tiled=False
          )
      )
    ranks = np.max(gathered.reshape(-1, len(sorted_ids)), axis=0).tolist()
  return [idx for _, idx in sorted(zip(ranks, sorted_ids))]


class _SourcePlan(NamedTuple):
  """Compiled `RuleSpec` plan for a single checkpoint source key."""

  target_key: str
  transform: _TensorTransform = None
  expert_info: tuple[int, int] | None = None
  is_raw_source: bool = False


class _HostTensor(NamedTuple):
  """A mapped checkpoint tensor and the host shard bytes it is read into."""

  jax_key: str
  transform: _TensorTransform
  raw: np.ndarray
  dtype: np.dtype
  shape: tuple[int, ...]
  post_slices: tuple[slice, ...] | None = None


_DIRECT_RULE_OPS: frozenset[tuple[str, ...]] = frozenset({
    (),
    ('transpose',),
    ('reshape',),
    ('transpose', 'reshape'),
})


class SpmdHostStager:
  """Layer 2 TPU/PJRT `RuleSpec` planner, 8-thread H2D, and HBM transposer."""

  @staticmethod
  def classify_rule(
      rule: Mapping[str, Any],
      disallowed_keys: AbstractSet[str],
  ) -> list[tuple[str, _SourcePlan]] | None:
    """Classifies a rule into direct 1:1 or MoE `stack` plans if sliceable."""
    src_keys = rule['source_keys']
    tgt_keys = rule['target_keys']
    if len(tgt_keys) != 1 or not disallowed_keys.isdisjoint(
        (*src_keys, *tgt_keys)
    ):
      return None
    target_key = tgt_keys[0]
    ops = rule['ops']
    op_sig = tuple(next(iter(step)) for step in ops)
    op_dict: dict[str, Any] = {
        'transpose': {'axes': None},
        'reshape': {'dst_shape': None},
        'stack': {'axis': -1},
    }
    for step in ops:
      op_dict.update(step)
    axes_raw = op_dict['transpose']['axes']
    dst_raw = op_dict['reshape']['dst_shape']
    axes = tuple(axes_raw) if axes_raw is not None else None
    dst = tuple(dst_raw) if dst_raw is not None else None
    if len(src_keys) == 1 and op_sig in _DIRECT_RULE_OPS:
      tf = (axes, dst) if (axes or dst) else None
      return [(src_keys[0], _SourcePlan(target_key, tf))]
    if (
        op_sig in (('stack',), ('transpose', 'stack'))
        and int(op_dict['stack']['axis']) == 0
        and axes in (None, (1, 0))
    ):
      n_exp = len(src_keys)
      return [
          (k, _SourcePlan(target_key, (axes, None), (idx, n_exp)))
          for idx, k in enumerate(src_keys)
      ]
    return None

  @staticmethod
  def compile_manifest(
      manifest: hf_mapping.RuleSpec,
      pending: dict[str, Any] | None = None,
  ) -> tuple[dict[str, _SourcePlan], frozenset[str], _StreamExecutor | None]:
    """Compiles a `RuleSpec` into `source_map` and optional composite executor."""
    spec = hf_mapping.parse_manifest(manifest)
    dropped = frozenset(spec['dropped_keys'])
    active_tied = [(s, d) for s, d in spec['tied_keys'] if d not in dropped]
    active_rules = [
        r
        for r in spec['rules']
        if dropped.isdisjoint((*r['source_keys'], *r['target_keys']))
    ]
    src_counts = collections.Counter(
        itertools.chain.from_iterable(r['source_keys'] for r in active_rules)
    )
    shared_keys = {k for k, c in src_counts.items() if c > 1} | {
        s for s, _ in active_tied
    }
    source_map: dict[str, _SourcePlan] = {}
    composite_rules: list[Mapping[str, Any]] = []
    for rule in active_rules:
      plans = SpmdHostStager.classify_rule(rule, shared_keys)
      if plans is not None:
        source_map.update(plans)
        continue
      composite_rules.append(rule)
      source_map.update(
          (k, _SourcePlan(k, is_raw_source=True)) for k in rule['source_keys']
      )
    composite_sources = frozenset(
        itertools.chain.from_iterable(r['source_keys'] for r in composite_rules)
    )
    executor: _StreamExecutor | None = None
    if composite_rules or active_tied:
      executor = hf_mapping.make_streaming_executor(
          {**spec, 'rules': composite_rules}, pending=pending
      )
    return source_map, composite_sources, executor

  @staticmethod
  def read_mapped_header(
      filepath: Any,
      source_map: Mapping[str, _SourcePlan],
      skipped_keys: list[str],
      seen_sources: set[str] | None = None,
      preloaded_header: tuple[int, Mapping[str, Any]] | None = None,
  ) -> tuple[int, list[tuple[int, int, _SourcePlan, Mapping[str, Any]]]]:
    """Reads `filepath`'s header and returns `(base_off, sorted_entries)`."""
    if preloaded_header is not None:
      header_size, header = preloaded_header
    else:
      header_size, header = _read_safetensors_header(filepath)
    entries: list[tuple[int, int, _SourcePlan, Mapping[str, Any]]] = []
    for name, meta in header.items():
      if name == '__metadata__':
        continue
      start_byte, end_byte, _, _ = _validate_tensor_meta(filepath, name, meta)
      plan = source_map.get(name)
      if plan is None:
        skipped_keys.append(name)
        continue
      if seen_sources is not None:
        with _MOE_LOCK:
          if name in seen_sources:
            raise ValueError(
                f'Duplicate tensor keys across safetensors shards: {[name]}'
            )
          seen_sources.add(name)
      entries.append((start_byte, end_byte, plan, meta))
    entries.sort(key=lambda entry: entry[0])
    return 8 + header_size, entries

  @staticmethod
  def _maybe_transpose(arr: jax.Array, perm: tuple[int, ...]) -> jax.Array:
    """Applies `perm` on-device only when non-identity to avoid cold XLA JIT."""
    if perm == tuple(range(arr.ndim)):
      return arr
    if isinstance(arr, jax.Array):
      return _jit_donate_transpose(arr, perm)
    return arr.transpose(perm)

  @staticmethod
  def parallel_device_put(
      flat_shards: Sequence[np.ndarray],
      flat_devices: Sequence[jax.Device],
      flat_permutations: Sequence[tuple[int, ...]],
  ) -> list[jax.Array]:
    """Transfers and transposes host shards in parallel across `_H2D_POOL`."""
    by_device: dict[jax.Device, list[int]] = collections.defaultdict(list)
    for idx, dev in enumerate(flat_devices):
      by_device[dev].append(idx)
    if len(by_device) <= 1:
      with jax.transfer_guard('allow'):
        raw_placed = jax.device_put(list(flat_shards), list(flat_devices))
      return [
          SpmdHostStager._maybe_transpose(a, p)
          for a, p in zip(raw_placed, flat_permutations)
      ]

    def put_device(idxs: list[int]) -> tuple[list[int], list[jax.Array]]:
      with jax.transfer_guard('allow'):
        dev_arrays = list(
            jax.device_put(
                [flat_shards[i] for i in idxs], [flat_devices[i] for i in idxs]
            )
        )
      return idxs, [
          SpmdHostStager._maybe_transpose(a, flat_permutations[i])
          for i, a in zip(idxs, dev_arrays)
      ]

    placed: list[jax.Array] = [cast(Any, None)] * len(flat_shards)
    for idxs, dev_arrays in _H2D_POOL.map(put_device, by_device.values()):
      for i, arr in zip(idxs, dev_arrays):
        placed[i] = arr
    return placed

  @staticmethod
  def stage_device_shards(
      arr: np.ndarray,
      spec: _TargetSpec,
      flat_shards: list[np.ndarray],
      flat_devices: list[jax.Device],
      flat_permutations: list[tuple[int, ...]],
  ) -> None:
    """Slices a host shard for each local device and records its layout order."""
    for dev, rel_slices in spec.device_slices:
      sub_shard = arr[rel_slices]
      order = tuple(
          sorted(
              range(sub_shard.ndim),
              key=lambda a, s=sub_shard: (-s.strides[a], -s.shape[a]),
          )
      )
      flat_shards.append(sub_shard.transpose(order))
      flat_devices.append(dev)
      flat_permutations.append(
          tuple(sorted(range(sub_shard.ndim), key=lambda a, o=order: o[a]))
      )

  @staticmethod
  def invert_reshape_slices(
      perm_shape: tuple[int, ...],
      post_shape: tuple[int, ...],
      target_slices: tuple[slice, ...],
  ) -> tuple[list[slice], tuple[int, ...] | None] | None:
    """Inverts a 2D->3D or equal-rank reshape for `target_slices` if aligned."""
    if perm_shape == post_shape:
      sliced_reshape = tuple(s.stop - s.start for s in target_slices)
      return list(target_slices), sliced_reshape
    if len(perm_shape) != 2 or len(post_shape) != 3:
      return None
    first_slice, mid_slice, last_slice = target_slices
    if perm_shape[0] == post_shape[0] and last_slice == slice(0, post_shape[2]):
      head_dim = post_shape[2]
      perm_slices = [
          first_slice,
          slice(mid_slice.start * head_dim, mid_slice.stop * head_dim),
      ]
      sliced_reshape = (
          first_slice.stop - first_slice.start,
          mid_slice.stop - mid_slice.start,
          head_dim,
      )
      return perm_slices, sliced_reshape
    if perm_shape[1] == post_shape[2] and mid_slice == slice(0, post_shape[1]):
      head_dim = post_shape[1]
      perm_slices = [
          slice(first_slice.start * head_dim, first_slice.stop * head_dim),
          last_slice,
      ]
      sliced_reshape = (
          first_slice.stop - first_slice.start,
          head_dim,
          last_slice.stop - last_slice.start,
      )
      return perm_slices, sliced_reshape
    return None

  @staticmethod
  def compute_slice_geometry(
      start: int,
      src_shape: tuple[int, ...],
      dtype: np.dtype,
      src_slices: tuple[slice, ...],
  ) -> tuple[tuple[int, ...], int, int, int, int]:
    """Computes `(shard_shape, first_off, outer_count, row_stride, row_bytes)`."""
    itemsize = dtype.itemsize
    shard_shape = tuple(s.stop - s.start for s in src_slices)
    first_dim_start = src_slices[0].start if src_slices else 0
    nbytes = math.prod(shard_shape) * itemsize
    first_off = start + (
        first_dim_start * math.prod(src_shape[1:]) * itemsize if nbytes else 0
    )
    outer_count = 1
    row_stride = nbytes
    row_bytes = nbytes
    if nbytes and src_slices[1:] != tuple(slice(0, d) for d in src_shape[1:]):
      tail_bytes = math.prod(src_shape[2:]) * itemsize
      row_stride = src_shape[1] * tail_bytes
      row_bytes = shard_shape[1] * tail_bytes
      first_off = (
          start
          + first_dim_start * row_stride
          + src_slices[1].start * tail_bytes
      )
      outer_count = shard_shape[0]
    return shard_shape, first_off, outer_count, row_stride, row_bytes

  @staticmethod
  def alloc_moe_expert_slice(
      plan: _SourcePlan,
      arena: ShmRangeReader | None,
      shard_shape: tuple[int, ...],
      dtype: np.dtype,
      outer_count: int,
      row_bytes: int,
      post_slices: tuple[slice, ...] | None,
      target_specs: Mapping[str, _TargetSpec] | None,
      moe_bufs: dict[str, np.ndarray],
      tensors: list[_HostTensor],
      moe_seen: dict[str, set[int]],
      moe_dtypes: dict[str, np.dtype],
  ) -> np.ndarray | None:
    """Allocates a 3D MoE expert stack in `/dev/shm` and returns this expert's view."""
    if plan.expert_info is None:
      return None
    expert_idx, num_experts = plan.expert_info
    stacked_key = plan.target_key
    spec = target_specs.get(stacked_key) if target_specs is not None else None
    expert_slice = (
        spec.host_slices[0] if spec is not None else slice(0, num_experts)
    )
    local_idx = expert_idx - expert_slice.start
    num_local_experts = expert_slice.stop - expert_slice.start
    if not (0 <= local_idx < num_local_experts):
      raise ValueError(
          f'MoE expert index {expert_idx} out of host slice {expert_slice} for'
          f' {stacked_key!r}.'
      )
    moe_shape = (num_local_experts, outer_count, row_bytes)
    with _MOE_LOCK:
      buf_3d = moe_bufs.get(stacked_key)
      if buf_3d is None:
        buf_3d = (
            arena.alloc(moe_shape)
            if arena is not None
            else np.empty(moe_shape, dtype=np.uint8)
        )
        moe_bufs[stacked_key] = buf_3d
        moe_dtypes[stacked_key] = dtype
      expected_dt = moe_dtypes.get(stacked_key, dtype)
      if buf_3d.shape != moe_shape or expected_dt != dtype:
        raise ValueError(
            f'Inconsistent MoE expert geometry or dtype for {stacked_key!r}:'
            f' got ({moe_shape}, {dtype}), expected'
            f' ({buf_3d.shape}, {expected_dt}).'
        )
      seen = moe_seen.setdefault(stacked_key, set())
      before = len(seen)
      seen.add(local_idx)
      all_arrived = (
          len(seen) == num_local_experts and before < num_local_experts
      )
    if all_arrived:
      stack_tf: _TensorTransform = (
          ((0, 2, 1), None) if plan.transform == ((1, 0), None) else None
      )
      # `post_slices` comes from `spec.host_slices[1:]` (omitting axis 0);
      # prepend `slice(None)` so slicing applies to axes 1..N of the 3D stack.
      moe_post = (
          (slice(None), *post_slices) if post_slices is not None else None
      )
      tensors.append(
          _HostTensor(
              stacked_key,
              stack_tf,
              buf_3d.ravel(),
              dtype,
              (num_local_experts, *shard_shape),
              moe_post,
          )
      )
    return buf_3d[local_idx]

  @staticmethod
  def _find_last_moe_file(
      planned_all: Sequence[tuple[list[_HostTensor], list[_Chunk]]],
      intervals: Sequence[tuple[int, int, str]],
  ) -> dict[str, int]:
    """Maps each MoE buffer key to the highest `file_idx` writing into it."""
    starts = [lo for lo, _, _ in intervals]
    last_file: dict[str, int] = {}
    for file_idx, (_, chunks) in enumerate(planned_all):
      for _, _, buf_2d in chunks:
        ptr = buf_2d.ctypes.data
        pos = bisect.bisect_right(starts, ptr) - 1
        if pos >= 0 and ptr < intervals[pos][1]:
          last_file[intervals[pos][2]] = file_idx
    return last_file

  @staticmethod
  def validate_moe_completeness(
      moe_bufs: Mapping[str, np.ndarray],
      moe_seen: Mapping[str, AbstractSet[int]],
  ) -> None:
    """Raises `ValueError` if any MoE buffer missed local expert indices."""
    for key, buf in moe_bufs.items():
      expected_count = buf.shape[0]
      seen = moe_seen.get(key, frozenset())
      if seen != set(range(expected_count)):
        missing = sorted(set(range(expected_count)) - seen)
        raise ValueError(
            f'Missing local MoE experts for {key!r}: arrived'
            f' {len(seen)}/{expected_count}, missing local indices {missing}.'
        )

  @staticmethod
  def align_moe_commit_files(
      planned_all: Sequence[tuple[list[_HostTensor], list[_Chunk]]],
      moe_bufs: Mapping[str, np.ndarray],
      moe_seen: Mapping[str, AbstractSet[int]] | None = None,
  ) -> None:
    """Moves each stacked MoE `_HostTensor` onto the last shard writing it.

    HuggingFace lexicographic sharding (e.g. `experts.127` before `experts.2`)
    and wave-rotated checkpoints can place `expert_slice.stop - 1` in an earlier
    shard than other local experts. Moving commit ownership to the highest
    `file_idx` targeting each `moe_bufs[key]` guarantees all experts are filled
    before `consume_one_file` uploads the 3D tensor to TPU.
    """
    if moe_seen is not None and moe_bufs:
      SpmdHostStager.validate_moe_completeness(moe_bufs, moe_seen)
    if not moe_bufs or len(planned_all) <= 1:
      return
    intervals = sorted(
        (buf.ctypes.data, buf.ctypes.data + buf.nbytes, key)
        for key, buf in moe_bufs.items()
        if buf.nbytes > 0
    )
    if not intervals:
      return
    last_file = SpmdHostStager._find_last_moe_file(planned_all, intervals)
    for file_idx, (file_tensors, _) in enumerate(planned_all):
      keep: list[_HostTensor] = []
      for t in file_tensors:
        dst_idx = last_file.get(t.jax_key, file_idx)
        if dst_idx != file_idx:
          planned_all[dst_idx][0].append(t)
        else:
          keep.append(t)
      file_tensors[:] = keep

  @staticmethod
  def locate_slice_spans(
      start: int,
      arena: ShmRangeReader | None,
      src_shape: tuple[int, ...],
      dtype: np.dtype,
      src_slices: tuple[slice, ...],
      plan: _SourcePlan,
      sliced_tf: _TensorTransform,
      post_slices: tuple[slice, ...] | None,
      target_specs: Mapping[str, _TargetSpec] | None,
      moe_bufs: dict[str, np.ndarray],
      tensors: list[_HostTensor],
      moe_seen: dict[str, set[int]],
      moe_dtypes: dict[str, np.dtype],
  ) -> tuple[int, int, int, int, np.ndarray]:
    """Locates byte spans and allocates `/dev/shm` host or 3D MoE buffers."""
    shard_shape, first_off, outer_count, row_stride, row_bytes = (
        SpmdHostStager.compute_slice_geometry(
            start, src_shape, dtype, src_slices
        )
    )
    expert_buf = SpmdHostStager.alloc_moe_expert_slice(
        plan,
        arena,
        shard_shape,
        dtype,
        outer_count,
        row_bytes,
        post_slices,
        target_specs,
        moe_bufs,
        tensors,
        moe_seen,
        moe_dtypes,
    )
    if expert_buf is not None:
      return first_off, outer_count, row_stride, row_bytes, expert_buf
    min_direct = min(_DIRECT_SLICE_BYTES, _READ_CHUNK_BYTES // 2)
    if arena is not None and outer_count == 1 and row_bytes >= min_direct:
      buf_2d = arena.alloc((1, row_bytes))
    else:
      buf_2d = np.empty((outer_count, row_bytes), dtype=np.uint8)
    tensors.append(
        _HostTensor(
            plan.target_key,
            sliced_tf,
            buf_2d.ravel(),
            dtype,
            shard_shape,
            post_slices,
        )
    )
    return first_off, outer_count, row_stride, row_bytes, buf_2d

  @staticmethod
  def read_one_chunk(
      fh: io.BufferedIOBase,
      offset: int,
      stride: int,
      buf_2d: np.ndarray,
      filepath: Any,
  ) -> None:
    """Reads a single contiguous or strided chunk into `buf_2d`."""
    rows, row_bytes = buf_2d.shape
    if rows > 1 and stride <= 65536:
      targets = [np.empty((rows - 1) * stride + row_bytes, dtype=np.uint8)]
    else:
      targets = buf_2d
    for target in targets:
      fh.seek(offset)
      if (n := fh.readinto(target)) != target.nbytes:
        raise OSError(
            f'Read {n} of {target.nbytes} bytes at offset {offset} of'
            f' {filepath}.'
        )
      offset += stride
    if targets is not buf_2d:
      buf_2d[:] = np.lib.stride_tricks.as_strided(
          targets[0], shape=(rows, row_bytes), strides=(stride, 1)
      )

  @staticmethod
  def inspect_file_items(
      filepath: Any,
      file_idx: int,
      source_map: Mapping[str, _SourcePlan],
      skipped_keys: list[str],
      target_specs: Mapping[str, _TargetSpec] | None = None,
      ici_plans: dict[int, _ReadIciPlan] | None = None,
      stage_specs: dict[str, _TargetSpec] | None = None,
      seen_sources: set[str] | None = None,
      preloaded_header: tuple[int, Mapping[str, Any]] | None = None,
  ) -> list[_PlannedTensor]:
    """Parses shard header and computes local-host slice geometry without allocs."""
    base_off, raw_entries = SpmdHostStager.read_mapped_header(
        filepath,
        source_map,
        skipped_keys,
        seen_sources=seen_sources,
        preloaded_header=preloaded_header,
    )
    items: list[_PlannedTensor] = []
    for start, _, plan, meta in raw_entries:
      src_shape = tuple(meta['shape'])
      dtype = safetensors_io.SAFETENSORS_DTYPES[meta['dtype']]
      should_read, src_slices, sliced_tf, post_slices, eff_spec = (
          _resolve_file_entry(
              plan, src_shape, dtype, target_specs, ici_plans, stage_specs
          )
      )
      if not should_read:
        continue
      shard_shape, first_off, outer_count, row_stride, row_bytes = (
          SpmdHostStager.compute_slice_geometry(
              start, src_shape, dtype, src_slices
          )
      )
      moe_key: str | None = None
      local_exp = 0
      num_local_exp = 0
      if plan.expert_info is not None:
        expert_idx, num_experts = plan.expert_info
        moe_key = plan.target_key
        exp_slice = (
            eff_spec.host_slices[0]
            if eff_spec is not None
            else slice(0, num_experts)
        )
        local_exp = expert_idx - exp_slice.start
        num_local_exp = exp_slice.stop - exp_slice.start
        if not (0 <= local_exp < num_local_exp):
          raise ValueError(
              f'MoE expert index {expert_idx} out of host slice {exp_slice} for'
              f' {moe_key!r}.'
          )
      items.append(
          _PlannedTensor(
              file_idx, base_off, first_off, outer_count, row_stride,
              row_bytes, shard_shape, dtype, plan, sliced_tf, post_slices,
              moe_key, local_exp, num_local_exp,
          )
      )
    return items

  @staticmethod
  def consume_one_file(
      file_idx: int,
      files: Sequence[_RangeSource],
      planned_all: list[tuple[list[_HostTensor], list[_Chunk]]],
      skipped_per_file: Sequence[list[str]],
      source_map: Mapping[str, _SourcePlan],
      composite_sources: AbstractSet[str],
      composite_executor: _StreamExecutor | None,
      target_dtype: np.dtype | None,
      target_specs: Mapping[str, _TargetSpec],
      keys: Sequence[str],
      params: list[jax.ShapeDtypeStruct | jax.Array],
      moe_bufs: dict[str, np.ndarray],
      block_h2d: bool = False,
      h2d_sink: Callable[..., None] | None = None,
      stage_specs: Mapping[str, _TargetSpec] | None = None,
  ) -> None:
    """Transforms, executes composite rules, and places ready tensors onto TPU."""
    start_time = time.monotonic()
    file_source = files[file_idx]
    skipped_keys = skipped_per_file[file_idx]
    file_tensors, _ = planned_all[file_idx]
    planned_all[file_idx] = ([], [])
    tensors = _load_file_tensors(
        file_source,
        source_map,
        target_dtype,
        skipped_keys,
        target_specs,
        preplanned=(file_tensors, []),
        stage_specs=stage_specs,
    )
    if moe_bufs:
      for k in tensors:
        moe_bufs.pop(k, None)
    if composite_executor is not None:
      raw_step = {
          k: tensors.pop(k) for k in list(tensors) if k in composite_sources
      }
      if raw_step:
        tensors.update(composite_executor(raw_step))
    if skipped_keys:
      logging.warning(
          'Skipped loading %d keys because they could not be mapped to '
          'model weights. This may be expected, for example when loading '
          'only text weights from a multimodal checkpoint. Missing keys: '
          '[%s]',
          len(skipped_keys),
          ', '.join(skipped_keys),
      )
    num_matched = _match_params(
        keys,
        target_specs,
        tensors,
        params,
        block_h2d=block_h2d,
        h2d_sink=h2d_sink,
        stage_specs=stage_specs,
    )
    logging.info(
        'Loaded %s: %d tensors in %.2fs.',
        file_source.name,
        num_matched,
        time.monotonic() - start_time,
    )
    del tensors


def _ensure_source_map(
    manifest_or_map: hf_mapping.RuleSpec | Mapping[str, _SourcePlan],
) -> Mapping[str, _SourcePlan]:
  """Returns `manifest_or_map` if already compiled, else compiles `RuleSpec`."""
  if isinstance(manifest_or_map, Mapping) and (
      not manifest_or_map
      or isinstance(next(iter(manifest_or_map.values())), _SourcePlan)
  ):
    return cast(Mapping[str, _SourcePlan], manifest_or_map)
  return SpmdHostStager.compile_manifest(
      cast(hf_mapping.RuleSpec, manifest_or_map)
  )[0]


def _build_spec_geometry(
    param_index: int,
    shape: tuple[int, ...],
    sharding: jax.sharding.Sharding | jax.Device,
) -> _TargetSpec:
  """Builds `_TargetSpec` host bounding slices and per-device slices."""
  full = tuple(slice(0, d) for d in shape)
  if isinstance(sharding, jax.Device):
    addr_map = {sharding: full}
  else:
    addr_map = sharding.addressable_devices_indices_map(shape)
  bounds = [
      (
          dev,
          tuple(
              (s or slice(None)).indices(d)[:2]
              for s, d in zip(idx or full, shape)
          ),
      )
      for dev, idx in addr_map.items()
  ]
  if not bounds:
    empty_slices = tuple(slice(0, 0) for _ in shape)
    return _TargetSpec(
        param_index, shape, (0,) * len(shape), sharding, empty_slices, ()
    )
  host_slices = tuple(
      slice(
          min(b[a][0] for _, b in bounds),
          max(b[a][1] for _, b in bounds),
      )
      for a in range(len(shape))
  )
  host_shape = tuple(s.stop - s.start for s in host_slices)
  dev_slices = tuple(
      (
          dev,
          tuple(
              slice(
                  b[a][0] - host_slices[a].start,
                  b[a][1] - host_slices[a].start,
              )
              for a in range(len(shape))
          ),
      )
      for dev, b in bounds
  )
  return _TargetSpec(
      param_index, shape, host_shape, sharding, host_slices, dev_slices
  )


def _ici_source_geometry(
    plan: _SourcePlan, src_shape: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...] | None, tuple[int, ...]] | None:
  """Returns `(full_src, perm, expected_shape)` for ICI eligibility checks."""
  if (exp_info := plan.expert_info) is not None:
    full_src = (exp_info[1], *src_shape)
    perm = (0, 2, 1) if plan.transform == ((1, 0), None) else None
    dst_shape = None
  else:
    full_src = src_shape
    raw_perm, dst_shape = plan.transform or (None, None)
    perm = tuple(raw_perm) if raw_perm is not None else None
  if perm is not None and len(perm) != len(full_src):
    return None
  post_tr = tuple(full_src[p] for p in perm) if perm is not None else full_src
  return full_src, perm, tuple(dst_shape) if dst_shape else post_tr


def _maybe_plan_ici_spec(
    plan: _SourcePlan,
    src_shape: tuple[int, ...],
    dtype: np.dtype,
    spec: _TargetSpec,
    ici_plans: dict[int, _ReadIciPlan],
    stage_specs: dict[str, _TargetSpec],
) -> _TargetSpec | None:
  """Plans and caches an ICI stage `_TargetSpec` when activation gates pass."""
  if (cached := stage_specs.get(plan.target_key)) is not None:
    if (
        plan.expert_info is not None
        and (plan.expert_info[1], *src_shape)
        != ici_plans[spec.param_index].src_shape
    ):
      raise ValueError(
          f'Shape mismatch for {plan.target_key}: got'
          f' {(plan.expert_info[1], *src_shape)}, expected'
          f' {ici_plans[spec.param_index].src_shape}.'
      )
    return cached
  if (geom := _ici_source_geometry(plan, src_shape)) is None:
    return None
  full_src, perm, expected_shape = geom
  if (
      expected_shape != spec.global_shape
      or math.prod(full_src) * dtype.itemsize < _ICI_MIN_BYTES
  ):
    return None
  ici_plan = plan_read_ici_reshard(
      full_src, spec.global_shape, spec.sharding, perm,
      is_moe=plan.expert_info is not None,
  )
  if ici_plan is None or not (
      ici_plan.inner_ops
      or ici_plan.rep_spans_hosts
      or ici_plan.host_stage_spec != ici_plan.stage_spec
  ):
    return None
  cand = _build_spec_geometry(
      spec.param_index, ici_plan.src_shape, ici_plan.host_stage_sharding
  )
  with _MOE_LOCK:
    ici_plans.setdefault(spec.param_index, ici_plan)
    base_key = plan.target_key.removesuffix('.value')
    stage_specs.setdefault(base_key, cand)
    stage_specs.setdefault(f'{base_key}.value', cand)
    return stage_specs[plan.target_key]


def _resolve_file_entry(
    plan: _SourcePlan,
    src_shape: tuple[int, ...],
    dtype: np.dtype,
    target_specs: Mapping[str, _TargetSpec] | None,
    ici_plans: dict[int, _ReadIciPlan] | None,
    stage_specs: dict[str, _TargetSpec] | None,
) -> tuple[
    bool,
    tuple[slice, ...],
    _TensorTransform,
    tuple[slice, ...] | None,
    _TargetSpec | None,
]:
  """Resolves host read slices via ICI staging when active, else direct host."""
  spec = (
      None
      if target_specs is None or plan.is_raw_source
      else target_specs.get(plan.target_key)
  )
  if spec is not None and ici_plans is not None and stage_specs is not None:
    if (
        st := _maybe_plan_ici_spec(
            plan, src_shape, dtype, spec, ici_plans, stage_specs
        )
    ) is not None:
      if (exp := plan.expert_info) is None:
        return True, st.host_slices, None, None, st
      return (
          st.host_slices[0].start <= exp[0] < st.host_slices[0].stop,
          st.host_slices[1:],
          None,
          None,
          st,
      )
  return (*_invert_transform_slices(plan, src_shape, target_specs), spec)


def _build_target_specs(
    keys: Sequence[str],
    params: Sequence[jax.ShapeDtypeStruct | jax.Array],
    shardings: Sequence[jax.sharding.Sharding | jax.Device],
) -> dict[str, _TargetSpec]:
  """Builds a lookup table from parameter keys to `_TargetSpec`."""
  specs: dict[str, _TargetSpec] = {}
  for i, (key, param, sharding) in enumerate(
      zip(keys, params, shardings, strict=True)
  ):
    spec = _build_spec_geometry(i, tuple(param.shape), sharding)
    base_key = key.removesuffix('.value')
    specs[base_key] = spec
    specs[f'{base_key}.value'] = spec
  return specs


def _resolve_target_slices(
    plan: _SourcePlan,
    post_shape: tuple[int, ...],
    target_specs: Mapping[str, _TargetSpec] | None,
) -> tuple[bool, tuple[slice, ...] | None]:
  """Returns `(should_read, post_transform_slices)` for `plan` on this host."""
  if (
      plan.is_raw_source
      or target_specs is None
      or plan.target_key not in target_specs
  ):
    return True, None
  spec = target_specs[plan.target_key]
  if plan.expert_info is None:
    if 0 in post_shape:
      return True, None
    is_partial = (
        post_shape == spec.global_shape and spec.host_shape != spec.global_shape
    )
    return True, spec.host_slices if is_partial else None
  expert_idx, num_experts = plan.expert_info
  if (num_experts, *post_shape) != spec.global_shape:
    raise ValueError(
        f'Shape mismatch for {plan.target_key}: got'
        f' {(num_experts, *post_shape)}, expected {spec.global_shape}.'
    )
  in_host = spec.host_slices[0].start <= expert_idx < spec.host_slices[0].stop
  if 0 in post_shape:
    return in_host, None
  is_partial = in_host and spec.host_shape[1:] != spec.global_shape[1:]
  return in_host, spec.host_slices[1:] if is_partial else None


def _invert_transform_slices(
    plan: _SourcePlan,
    src_shape: tuple[int, ...],
    target_specs: Mapping[str, _TargetSpec] | None,
) -> tuple[bool, tuple[slice, ...], _TensorTransform, tuple[slice, ...] | None]:
  """Maps host `target_slices` in post-transform space back to `src_shape`."""
  full_src = tuple(slice(0, d) for d in src_shape)
  permute, reshape = plan.transform or (None, None)
  perm_axes = tuple(permute or range(len(src_shape)))
  perm_shape = tuple(src_shape[p] for p in perm_axes)
  post_shape = tuple(reshape or perm_shape)
  should_read, target_slices = _resolve_target_slices(
      plan, post_shape, target_specs
  )
  if target_slices is None:
    return should_read, full_src, plan.transform, None

  perm_slices = list(target_slices)
  sliced_reshape: tuple[int, ...] | None = None
  if reshape is not None:
    inverted = SpmdHostStager.invert_reshape_slices(
        perm_shape, post_shape, target_slices
    )
    if inverted is None:
      return True, full_src, plan.transform, target_slices
    perm_slices, sliced_reshape = inverted

  src_slices = [slice(0, 0)] * len(src_shape)
  for idx, src_ax in enumerate(perm_axes):
    src_slices[src_ax] = perm_slices[idx]
  if (len(src_slices) >= 3 and tuple(src_slices[2:]) != full_src[2:]) or (
      len(src_slices) >= 2
      and src_slices[0] == full_src[0]
      and src_slices[1] != full_src[1]
  ):
    return True, full_src, plan.transform, target_slices
  return True, tuple(src_slices), (permute, sliced_reshape), None


def _plan_file_reads(
    filepath: Any,
    manifest_or_map: hf_mapping.RuleSpec | Mapping[str, _SourcePlan],
    skipped_keys: list[str],
    target_specs: Mapping[str, _TargetSpec] | None = None,
    moe_bufs: dict[str, np.ndarray] | None = None,
    arena: ShmRangeReader | None = None,
    moe_seen: dict[str, set[int]] | None = None,
    moe_dtypes: dict[str, np.dtype] | None = None,
    preloaded_header: tuple[int, Mapping[str, Any]] | None = None,
) -> tuple[list[_HostTensor], list[_Chunk]]:
  """Allocates host buffers for the mapped local-host shards of one file."""
  source_map = _ensure_source_map(manifest_or_map)
  base_off, raw_entries = SpmdHostStager.read_mapped_header(
      filepath, source_map, skipped_keys, preloaded_header=preloaded_header
  )
  tensors: list[_HostTensor] = []
  chunks: list[_Chunk] = []
  active_moe = moe_bufs if moe_bufs is not None else {}
  active_seen = moe_seen if moe_seen is not None else {}
  active_dtypes = moe_dtypes if moe_dtypes is not None else {}
  for start, _, plan, meta in raw_entries:
    src_shape = tuple(meta['shape'])
    dtype = safetensors_io.SAFETENSORS_DTYPES[meta['dtype']]
    should_read, src_slices, sliced_tf, post_slices = _invert_transform_slices(
        plan, src_shape, target_specs
    )
    if not should_read:
      continue
    first_off, outer_count, row_stride, row_bytes, buf_2d = (
        SpmdHostStager.locate_slice_spans(
            start,
            arena,
            src_shape,
            dtype,
            src_slices,
            plan,
            sliced_tf,
            post_slices,
            target_specs,
            active_moe,
            tensors,
            active_seen,
            active_dtypes,
        )
    )
    if outer_count == 1:
      chunks += [
          (
              base_off + first_off + off,
              0,
              buf_2d[:, off : off + _READ_CHUNK_BYTES],
          )
          for off in range(0, row_bytes, _READ_CHUNK_BYTES)
      ]
      continue
    step = max(1, _READ_CHUNK_BYTES // max(1, min(row_stride, 65536)))
    chunks += [
        (
            base_off + first_off + r * row_stride,
            row_stride,
            buf_2d[r : r + step],
        )
        for r in range(0, outer_count, step)
    ]
  if moe_seen is None and active_moe:
    SpmdHostStager.validate_moe_completeness(active_moe, active_seen)
  return tensors, chunks


def _read_chunk_queue(
    filepath: Any, pending: collections.deque[_Chunk]
) -> None:
  """Reads queued chunks into their buffers over one private file handle."""
  with cast(io.BufferedIOBase, filepath.open('rb')) as fh:
    while True:
      try:
        offset, stride, buf_2d = pending.popleft()
      except IndexError:
        return
      SpmdHostStager.read_one_chunk(fh, offset, stride, buf_2d, filepath)


def _apply_transform(
    arr: np.ndarray, transform: _TensorTransform
) -> np.ndarray:
  """Applies an optional `(permute, reshape)` transform as a zero-copy view."""
  if transform is None:
    return arr
  permute, reshape = transform
  if permute == (1, 0) and reshape is not None and len(reshape) == 3:
    if arr.shape == (reshape[1] * reshape[2], reshape[0]):
      return arr.reshape(reshape[1], reshape[2], reshape[0]).transpose(
          (2, 0, 1)
      )
    if arr.shape == (reshape[2], reshape[0] * reshape[1]):
      return arr.reshape(reshape[2], reshape[0], reshape[1]).transpose(
          (1, 2, 0)
      )
  if permute:
    arr = arr.transpose(permute)
  if reshape:
    arr = arr.reshape(reshape)
  return arr


def _load_file_tensors(
    filepath: Any,
    manifest_or_map: hf_mapping.RuleSpec | Mapping[str, _SourcePlan],
    target_dtype: np.dtype | None,
    skipped_keys: list[str],
    target_specs: Mapping[str, _TargetSpec] | None = None,
    preplanned: tuple[list[_HostTensor], list[_Chunk]] | None = None,
    stage_specs: Mapping[str, _TargetSpec] | None = None,
) -> dict[str, np.ndarray]:
  """Reads the mapped local-host shards of one safetensors file into host RAM."""
  if preplanned is not None:
    tensors, chunks = preplanned
  else:
    tensors, chunks = _plan_file_reads(
        filepath, manifest_or_map, skipped_keys, target_specs
    )
  if chunks:
    total_bytes = sum(b.nbytes for _, _, b in chunks)
    num_workers = min(
        _READ_WORKERS,
        len(chunks),
        max(1, (total_bytes + _READ_CHUNK_BYTES - 1) // _READ_CHUNK_BYTES),
    )
    pending_chunks = collections.deque(chunks)
    futures = [
        _SHARED_POOL.submit(_read_chunk_queue, filepath, pending_chunks)
        for _ in range(num_workers - 1)
    ]
    try:
      _read_chunk_queue(filepath, pending_chunks)
    finally:
      pending_chunks.clear()
      concurrent.futures.wait(futures)
    for future in futures:
      future.result()
  del chunks

  loaded: dict[str, np.ndarray] = {}
  for t in tensors:
    if t.jax_key in loaded:
      raise RuntimeError(
          f'Duplicate key {t.jax_key} found within file {filepath.name}.'
      )
    eff_dtype = (
        None
        if (stage_specs is not None and t.jax_key in stage_specs)
        else target_dtype
    )
    raw_arr = np.require(
        t.raw.view(t.dtype).reshape(t.shape),
        dtype=eff_dtype,
        requirements=('C', 'A', 'W'),
    )
    loaded[t.jax_key] = _apply_transform(raw_arr, t.transform)[
        t.post_slices or ()
    ]
  return loaded


def _pack_device_shards(
    flat_shards: Sequence[np.ndarray],
    flat_devices: Sequence[jax.Device],
) -> list[np.ndarray]:
  """Packs strided or CPU-aliased shards in parallel via `_PACK_POOL`."""
  out = list(flat_shards)
  dsts: list[np.ndarray] = []
  srcs: list[np.ndarray] = []
  seen_packed: dict[
      tuple[int, tuple[int, ...], tuple[int, ...], np.dtype, bool], np.ndarray
  ] = {}
  for idx, (arr, dev) in enumerate(zip(flat_shards, flat_devices)):
    is_cpu = dev.platform == 'cpu'
    if arr.flags['C_CONTIGUOUS'] and (not is_cpu or arr.flags['OWNDATA']):
      continue
    # `flat_shards` keeps every `arr` alive for this call, so `ctypes.data`
    # uniquely identifies the underlying buffer without address reuse.
    view_key = (arr.ctypes.data, arr.shape, arr.strides, arr.dtype, is_cpu)
    if (cached := seen_packed.get(view_key)) is not None:
      out[idx] = cached
      continue
    dst = np.empty(arr.shape, dtype=arr.dtype)
    seen_packed[view_key] = dst
    out[idx] = dst
    if arr.nbytes == 0:
      continue
    if (
        arr.nbytes < _PARALLEL_PACK_MIN_BYTES
        or arr.ndim < 2
        or arr.shape[0] < 4
    ):
      dsts.append(dst)
      srcs.append(arr)
      continue
    step = (arr.shape[0] + 7) // 8
    dsts.extend(dst[s : s + step] for s in range(0, arr.shape[0], step))
    srcs.extend(arr[s : s + step] for s in range(0, arr.shape[0], step))
  if dsts:
    list(_PACK_POOL.map(np.copyto, dsts, srcs))
  return out


def _commit_packed_shards(
    matched_specs: Sequence[_TargetSpec],
    flat_shards: Sequence[np.ndarray],
    flat_devices: Sequence[jax.Device],
    flat_permutations: Sequence[tuple[int, ...]],
    params: list[jax.ShapeDtypeStruct | jax.Array],
    block_h2d: bool = False,
    ici_param_indices: AbstractSet[int] | Mapping[int, Any] | None = None,
) -> tuple[list[jax.Array], list[tuple[int, jax.Array]]]:
  """Transfers packed host shards to devices and populates `params` or ICI."""
  placed = SpmdHostStager.parallel_device_put(
      flat_shards, flat_devices, flat_permutations
  )
  try:
    if block_h2d:
      jax.block_until_ready(placed)
    ici_items: list[tuple[int, jax.Array]] = []
    cursor = 0
    for spec in matched_specs:
      n = len(spec.device_slices)
      dev_arrays = placed[cursor : cursor + n]
      cursor += n
      if isinstance(spec.sharding, jax.Device):
        arr = dev_arrays[0]
      else:
        arr = jax.make_array_from_single_device_arrays(
            spec.global_shape, spec.sharding, dev_arrays
        )
      if (
          ici_param_indices is not None
          and spec.param_index in ici_param_indices
      ):
        ici_items.append((spec.param_index, arr))
      else:
        params[spec.param_index] = arr
    return placed, ici_items
  except BaseException:
    jax.block_until_ready(placed)
    raise


def _match_params(
    keys: Sequence[str],
    target_specs: Mapping[str, _TargetSpec],
    tensors: Mapping[str, np.ndarray | jax.Array],
    params: list[jax.ShapeDtypeStruct | jax.Array],
    block_h2d: bool = False,
    h2d_sink: Callable[..., None] | None = None,
    stage_specs: Mapping[str, _TargetSpec] | None = None,
) -> int:
  """Slices matched host shards to local cards and updates `params` in-place."""
  matched_specs: list[_TargetSpec] = []
  flat_shards: list[np.ndarray] = []
  flat_devices: list[jax.Device] = []
  flat_permutations: list[tuple[int, ...]] = []
  total_matched = 0
  staged_bytes = 0
  for key in keys:
    if (raw_tensor := tensors.get(key, tensors.get(f'{key}.value'))) is None:
      continue
    spec = (
        stage_specs[key]
        if stage_specs is not None and key in stage_specs
        else target_specs[key]
    )
    arr = np.asarray(jax.device_get(raw_tensor))
    if arr.shape == spec.global_shape:
      arr = arr[spec.host_slices]
    if arr.shape != spec.host_shape:
      raise ValueError(
          f'Shape mismatch for {key}: got {arr.shape}, expected'
          f' {spec.host_shape} (global {spec.global_shape})'
      )
    next_bytes = sum(arr[sl].nbytes for _, sl in spec.device_slices)
    if (
        h2d_sink is not None
        and matched_specs
        and staged_bytes + next_bytes > _PACK_FLUSH_BYTES
    ):
      packed = _pack_device_shards(flat_shards, flat_devices)
      h2d_sink(matched_specs, packed, flat_devices, flat_permutations, params)
      total_matched += len(matched_specs)
      matched_specs, flat_shards, flat_devices, flat_permutations = (
          [],
          [],
          [],
          [],
      )
      staged_bytes = 0
    matched_specs.append(spec)
    SpmdHostStager.stage_device_shards(
        arr, spec, flat_shards, flat_devices, flat_permutations
    )
    staged_bytes += next_bytes
  if not matched_specs:
    return total_matched
  # On CPU backends, copying views into owned buffers keeps host arrays safe
  # after `/dev/shm` closes; TPU uses zero-copy C-contiguous views + DMA fence.
  flat_shards = _pack_device_shards(flat_shards, flat_devices)
  if h2d_sink is not None:
    h2d_sink(
        matched_specs, flat_shards, flat_devices, flat_permutations, params
    )
  else:
    _commit_packed_shards(
        matched_specs,
        flat_shards,
        flat_devices,
        flat_permutations,
        params,
        block_h2d=block_h2d,
    )
  return total_matched + len(matched_specs)


def _estimate_host_arena_budget(
    params: Sequence[jax.ShapeDtypeStruct | jax.Array],
    target_specs: Mapping[str, _TargetSpec],
) -> int:
  """Estimates a bounded `/dev/shm` budget `<= MAX_READ_ARENA_BYTES`."""
  if 'TUNIX_SHM_BUDGET_BYTES' in os.environ:
    return 0
  seen: set[int] = set()
  est_bytes = 0
  for spec in target_specs.values():
    if spec.param_index in seen:
      continue
    seen.add(spec.param_index)
    est_bytes += (
        math.prod(spec.host_shape)
        * np.dtype(params[spec.param_index].dtype).itemsize
    )
  avail = safetensors_io.available_shm_bytes()
  target = max(64 * 1024 * 1024, est_bytes * 2)
  return min(avail, target, safetensors_io.MAX_READ_ARENA_BYTES)


def _preloaded_headers_for_files(
    files: Sequence[_RangeSource],
    shard_headers: Mapping[str, tuple[int, Mapping[str, Any]]] | None,
) -> list[tuple[int, Mapping[str, Any]] | None]:
  """Aligns optional sidecar headers with `files` order for parallel planning."""
  if not shard_headers:
    return [None] * len(files)
  return [shard_headers.get(f.name) for f in files]


def _maybe_acquire_gcs_read_pool(
    files: Sequence[_RangeSource],
    plan: _CircularRingPlan,
    read_procs: int | None,
) -> safetensors_io.GcsShmReadPool | None:
  """Returns the persistent GCS `/dev/shm` read pool when eligible for `plan`."""
  if (
      not isinstance(files[0], _GsHttpPath)
      or not plan.tasks
      or not safetensors_io.GcsShmReadPool.can_handle(
          len(plan.tasks),
          len(plan.group_counts),
          sum(len(cast(_GsHttpPath, f).http_path) + 1 for f in files),
      )
  ):
    return None
  total_bytes = sum(t[3] for t in plan.tasks)
  target_procs = safetensors_io.auto_gcs_read_procs(total_bytes, read_procs)
  return safetensors_io.ensure_gcs_read_pool(target_procs, files[0].vips)


def _stream_subfile_fallback(
    arena: ShmRangeReader,
    files: Sequence[_RangeSource],
    source_map: Mapping[str, _SourcePlan],
    composite_sources: AbstractSet[str],
    composite_executor: _StreamExecutor | None,
    target_dtype: np.dtype | None,
    target_specs: Mapping[str, _TargetSpec],
    keys: Sequence[str],
    params: list[jax.ShapeDtypeStruct | jax.Array],
    *,
    shard_headers: Mapping[str, tuple[int, Mapping[str, Any]]] | None = None,
) -> None:
  """Runs sub-file unit-recycling streaming when a file exceeds half the arena."""
  arena.reset()
  arena.ensure_prefaulted(arena.capacity)
  skipped_per_file: list[list[str]] = [[] for _ in files]
  moe_bufs: dict[str, np.ndarray] = {}
  moe_seen: dict[str, set[int]] = {}
  moe_dtypes: dict[str, np.dtype] = {}
  preloaded_headers = _preloaded_headers_for_files(files, shard_headers)
  planned_all = list(
      _SHARED_POOL.map(
          _plan_file_reads,
          files,
          itertools.repeat(source_map),
          skipped_per_file,
          itertools.repeat(target_specs),
          itertools.repeat(moe_bufs),
          itertools.repeat(arena),
          itertools.repeat(moe_seen),
          itertools.repeat(moe_dtypes),
          preloaded_headers,
      )
  )
  SpmdHostStager.align_moe_commit_files(planned_all, moe_bufs, moe_seen)
  tasks, _ = arena.plan_coalesced_tasks(planned_all)

  def _consume(file_idx: int) -> None:
    SpmdHostStager.consume_one_file(
        file_idx,
        files,
        planned_all,
        skipped_per_file,
        source_map,
        composite_sources,
        composite_executor,
        target_dtype,
        target_specs,
        keys,
        params,
        moe_bufs,
        block_h2d=True,
    )

  try:
    arena.execute_tasks(
        files,
        tasks,
        enable_hedge=files[0].supports_hedge if files else False,
        on_file_ready=_consume,
    )
  finally:
    jax.block_until_ready([p for p in params if isinstance(p, jax.Array)])


def _all_hosts_agree(flag: bool) -> bool:
  """Returns True iff every host reports `flag=True` across the cluster."""
  if jax.process_count() <= 1:
    return flag
  with jax.transfer_guard('allow'):
    return bool(
        np.all(
            multihost_utils.process_allgather(
                np.asarray(flag, dtype=np.bool_), tiled=False
            )
        )
    )


def stream_load_files(
    files: Sequence[_RangeSource],
    source_map: Mapping[str, _SourcePlan],
    composite_sources: AbstractSet[str],
    composite_executor: _StreamExecutor | None,
    target_dtype: np.dtype | None,
    target_specs: Mapping[str, _TargetSpec],
    keys: Sequence[str],
    params: list[jax.ShapeDtypeStruct | jax.Array],
    *,
    arena: ShmRangeReader | None = None,
    read_procs: int | None = None,
    shard_headers: Mapping[str, tuple[int, Mapping[str, Any]]] | None = None,
) -> None:
  """Loads `files` in 1 continuous circular-ring pass with async H2D overlap."""
  if not files:
    _all_hosts_agree(True)
    return
  own_arena = arena is None
  if own_arena:
    active_arena = ShmRangeReader(
        shm_budget_bytes=_estimate_host_arena_budget(params, target_specs)
    )
    ctx = active_arena
  else:
    active_arena = arena
    ctx = contextlib.nullcontext(active_arena)
  with ctx:
    active_arena.reset()
    skipped_per_file: list[list[str]] = [[] for _ in files]
    ici_plans: dict[int, _ReadIciPlan] = {}
    stage_specs: dict[str, _TargetSpec] = {}
    seen_sources: set[str] = set()
    preloaded_headers = _preloaded_headers_for_files(files, shard_headers)
    items_per_file = list(
        _SHARED_POOL.map(
            SpmdHostStager.inspect_file_items,
            files,
            range(len(files)),
            itertools.repeat(source_map),
            skipped_per_file,
            itertools.repeat(target_specs),
            itertools.repeat(ici_plans),
            itertools.repeat(stage_specs),
            itertools.repeat(seen_sources),
            preloaded_headers,
        )
    )
    plan = active_arena.plan_circular_ring(items_per_file)
    if not _all_hosts_agree(plan is not None) or plan is None:
      _stream_subfile_fallback(
          active_arena,
          files,
          source_map,
          composite_sources,
          composite_executor,
          target_dtype,
          target_specs,
          keys,
          params,
          shard_headers=shard_headers,
      )
      return

    ordered_ici = _compute_ordered_ici_indices(
        ici_plans, plan.planned_all, stage_specs
    )
    ici_config = (
        _ReadIciConfig(ici_plans, ordered_ici, params, target_dtype)
        if ordered_ici
        else None
    )

    def _consume_ring_group(
        group_idx: int, h2d_sink: Callable[..., None] | None = None
    ) -> None:
      SpmdHostStager.consume_one_file(
          group_idx,
          files,
          plan.planned_all,
          skipped_per_file,
          source_map,
          composite_sources,
          composite_executor,
          target_dtype,
          target_specs,
          keys,
          params,
          plan.moe_bufs,
          block_h2d=plan.ring_wraps and h2d_sink is None,
          h2d_sink=h2d_sink,
          stage_specs=stage_specs or None,
      )

    pool = _maybe_acquire_gcs_read_pool(files, plan, read_procs)
    try:
      if pool is not None:
        active_arena.execute_gcs_ring(
            pool, files, plan, _consume_ring_group, ici_config=ici_config
        )
      else:
        active_arena.execute_threaded_ring(
            files, plan, _consume_ring_group, ici_config=ici_config
        )
    finally:
      if not plan.ring_wraps:
        # Non-wrapping rings skip per-group `block_until_ready`, so always
        # finish async `device_put` DMA (including on error or external arenas)
        # before `/dev/shm` can be unmapped or reset.
        jax.block_until_ready([p for p in params if isinstance(p, jax.Array)])


def _verify_load_completeness(
    pending: Mapping[str, Any] | None,
    missing_keys: Sequence[str],
    hf_layout: str,
) -> None:
  """Raises `ValueError` if multi-source rules or model params are incomplete."""
  if pending:
    raise ValueError(
        f'Incomplete multi-source weight rules for declared HF layout'
        f' {hf_layout!r}; unconsumed checkpoint keys in pending buffer:'
        f' {sorted(pending)}'
    )
  if missing_keys:
    raise ValueError(
        'Checkpoint did not populate all model parameters for declared HF'
        f' layout {hf_layout!r}; missing: {list(missing_keys)}'
    )


def _match_opt_params(
    keys: Sequence[str],
    params: Sequence[jax.ShapeDtypeStruct | jax.Array],
    tensors: Mapping[str, np.ndarray | jax.Array],
) -> list[tuple[int, np.ndarray | jax.Array]]:
  """Returns `(param_index, tensor)` for each param found in `tensors`."""
  matches = []
  for i, key in enumerate(keys):
    for k in (key, f'{key}.value'):  # nnx.Param may add a `.value` suffix.
      if k not in tensors:
        continue
      if tensors[k].shape != params[i].shape:
        raise ValueError(
            f'Shape mismatch for {k}: got {tensors[k].shape}, expected'
            f' {params[i].shape}'
        )
      matches.append((i, tensors[k]))
      break
  return matches


def _read_opt_host_state(
    files: Sequence[epath.Path],
    source_keys: frozenset[str],
    mmaps: list[mmap.mmap],
    file_handles: list[Any],
) -> dict[str, np.ndarray]:
  """Memory-maps safetensors shards and collects zero-copy host tensor views."""
  host_state: dict[str, np.ndarray] = {}
  for f in files:
    contiguous_array, tensor_metadata, mm, fh = load_safetensors_with_offsets(f)
    mmaps.append(mm)
    file_handles.append(fh)
    skipped_keys: list[str] = []
    for meta in tensor_metadata:
      name = meta['name']
      if name not in source_keys:
        skipped_keys.append(name)
        continue
      if name in host_state:
        raise ValueError(
            f'Duplicate tensor keys across safetensors shards: {[name]}'
        )
      raw = contiguous_array[meta['start_byte'] : meta['end_byte']]
      host_state[name] = raw.view(
          safetensors_io.SAFETENSORS_DTYPES[meta['dtype']]
      ).reshape(meta['shape'])
    if skipped_keys:
      logging.warning(
          'Skipped loading %d keys because they could not be mapped to model '
          'weights. This may be expected, for example when loading only text '
          'weights from a multimodal checkpoint. Missing keys: [%s]',
          len(skipped_keys),
          ', '.join(skipped_keys),
      )
  return host_state


def _load_optional_weight_map(
    file_dir: str,
) -> tuple[dict[str, str], int | None]:
  """Loads `(weight_map, total_size)` from optional `index.json`."""
  root_dir = (
      epath.Path(file_dir)
      if '://' in str(file_dir)
      else epath.Path(file_dir).expanduser()
  )
  try:
    raw = json.loads((root_dir / 'model.safetensors.index.json').read_text())
  except Exception as exc:  # pylint: disable=broad-exception-caught
    if safetensors_io.is_not_found_error(exc):
      return {}, None
    raise
  if not isinstance(raw, dict):
    return {}, None
  wm = raw.get('weight_map')
  meta = raw.get('metadata')
  raw_total = meta.get('total_size') if isinstance(meta, dict) else None
  total_size = raw_total if safetensors_io.is_strict_int(raw_total) else None
  return (dict(wm) if isinstance(wm, dict) else {}), total_size


def load_and_create_model_orig(
    file_dir: str,
    model_class,
    model_class_kwargs,
    config,
    manifest: hf_mapping.RuleSpec,
    mesh=None,
    dtype: jnp.dtype | None = None,
):
  """Loads safetensors files using the 3-layer `/dev/shm` SPMD streaming reader."""
  spec = hf_mapping.parse_manifest(manifest)
  pending: dict[str, Any] = {}
  source_map, composite_sources, composite_executor = (
      SpmdHostStager.compile_manifest(spec, pending=pending)
  )
  sidecar_fut = _SHARED_POOL.submit(
      safetensors_io.load_sidecar_headers, file_dir
  )
  wm_fut = _SHARED_POOL.submit(_load_optional_weight_map, file_dir)
  sidecar_headers = sidecar_fut.result()
  weight_map: dict[str, str] = {}
  total_size: int | None = None
  local_sizes: dict[str, int] | None = None
  if sidecar_headers is not None:
    weight_map, total_size = wm_fut.result()
    if not weight_map and '://' not in str(file_dir):
      local_dir = os.path.expanduser(file_dir)
      local_sizes = {
          name: os.path.getsize(p)
          for name in sidecar_headers
          if os.path.isfile(p := os.path.join(local_dir, name))
      }
  else:
    wm_fut.cancel()
  if sidecar_headers is not None and safetensors_io.sidecar_matches_weight_map(
      sidecar_headers,
      weight_map,
      total_size=total_size,
      file_sizes=local_sizes,
  ):
    files = safetensors_io.build_range_sources_from_names(
        file_dir, sorted(sidecar_headers)
    )
  else:
    sidecar_headers = None
    files = resolve_range_sources(file_dir)

  context_manager = (
      compat.set_mesh(mesh) if mesh is not None else contextlib.nullcontext()
  )
  with context_manager:
    model = nnx.eval_shape(
        lambda: model_class(
            config, rngs=nnx.Rngs(params=0), **model_class_kwargs
        )
    )

  graph_def, abs_state = nnx.split(model)
  leaves, treedef = jax.tree_util.tree_flatten_with_path(
      abs_state.to_pure_dict()
  )
  keys = [path_to_key(path) for path, _ in leaves]
  params = [param for _, param in leaves]
  if mesh is not None:
    shardings = treedef.flatten_up_to(
        nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
    )
  else:
    shardings = [jax.local_devices()[0]] * len(params)
  target_specs = _build_target_specs(keys, params, shardings)
  target_dtype = None if dtype is None else np.dtype(dtype)
  stream_load_files(
      files,
      source_map,
      composite_sources,
      composite_executor,
      target_dtype,
      target_specs,
      keys,
      params,
      shard_headers=sidecar_headers,
  )
  _verify_load_completeness(
      pending,
      [k for k, p in zip(keys, params) if isinstance(p, jax.ShapeDtypeStruct)],
      spec['model_type'] if 'model_type' in spec else '',
  )
  return nnx.merge(graph_def, jax.tree_util.tree_unflatten(treedef, params))


def load_and_create_model_opt(
    file_dir: str,
    model_class,
    model_class_kwargs,
    config,
    manifest: hf_mapping.RuleSpec,
    mesh=None,
    dtype: jnp.dtype | None = None,
):
  """Loads safetensors files using local POSIX `mmap` and `RuleSpec` execution."""
  spec = hf_mapping.parse_manifest(manifest)
  pending: dict[str, Any] = {}
  source_keys = hf_mapping.active_source_keys(spec)
  stream_executor = hf_mapping.make_streaming_executor(spec, pending=pending)
  hf_layout = spec['model_type'] if 'model_type' in spec else ''
  file_dir = resolve_checkpoint_dir(file_dir)
  files = sorted(epath.Path(file_dir).expanduser().glob('*.safetensors'))
  if not files:
    raise ValueError(f'No safetensors found in {file_dir}')

  context_manager = (
      compat.set_mesh(mesh) if mesh is not None else contextlib.nullcontext()
  )
  with context_manager:
    model = nnx.eval_shape(
        lambda: model_class(
            config, rngs=nnx.Rngs(params=0), **model_class_kwargs
        )
    )

  graph_def, abs_state = nnx.split(model)
  leaves, treedef = jax.tree_util.tree_flatten_with_path(
      abs_state.to_pure_dict()
  )
  keys = [path_to_key(path) for path, _ in leaves]
  params = [param for _, param in leaves]
  if mesh is not None:
    shardings = treedef.flatten_up_to(
        nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
    )
  else:
    shardings = [jax.local_devices()[0]] * len(params)

  target_dtype = None if dtype is None else np.dtype(dtype)
  mmaps: list[mmap.mmap] = []
  file_handles: list[Any] = []
  host_state: dict[str, Any] = {}
  matches: list[tuple[int, np.ndarray | jax.Array]] = []
  tensor: Any = None
  try:
    host_state = _read_opt_host_state(files, source_keys, mmaps, file_handles)
    host_state = stream_executor(host_state)
    matches = _match_opt_params(keys, params, host_state)
    matched_idx = {i for i, _ in matches}
    missing = [k for i, k in enumerate(keys) if i not in matched_idx]
    _verify_load_completeness(pending, missing, hf_layout)
    for i, tensor in matches:
      arr = (
          tensor
          if target_dtype is None
          else np.asarray(tensor, dtype=target_dtype)
      )
      params[i] = sharding_utils.place_array_on_target(arr, shardings[i])
      del arr
  finally:
    tensor = None
    matches.clear()
    host_state.clear()
    pending.clear()
    for mm in mmaps:
      # CPU PJRT zero-copy `jax.Array`s pin exported mmap buffers until GC.
      with contextlib.suppress(BufferError):
        mm.close()
    for fh in file_handles:
      fh.close()

  return nnx.merge(graph_def, jax.tree_util.tree_unflatten(treedef, params))


def load_and_create_model(
    file_dir: str,
    model_class,
    config,
    manifest: hf_mapping.RuleSpec,
    mesh=None,
    dtype: jnp.dtype | None = None,
    mode: str = 'auto',
    model_class_kwargs: dict[str, Any] | None = None,
):
  """Loads safetensors files and creates an NNX model using `manifest` rules.

  Args:
    file_dir: Directory containing the safetensors files.
    model_class: The NNX model class to instantiate.
    config: The configuration object for the model.
    manifest: Declarative PrimOp `HF -> Tunix` weight-mapping manifest for the
      checkpoint's declared layout; its `model_type` names the layout in errors.
    mesh: An optional JAX device mesh for sharding.
    dtype: Optional dtype to cast the loaded tensors to.
    mode: The mode to use for loading the model. Options are ('auto',
      'optimized', 'original').
    model_class_kwargs: Optional keyword arguments to pass to the model class.

  Returns:
    An NNX model instance with weights loaded from the safetensors files.
  """
  if model_class_kwargs is None:
    model_class_kwargs = {}
  is_internal = (
      env_utils.is_internal_env() or env_utils.is_pathways_initialized()
  )
  if mode == 'auto':
    mode = 'original'
  if (mode, is_internal) == ('optimized', True):
    raise ValueError(
        'Optimized mode is not supported in internal or Pathways-initialized'
        ' environments.'
    )

  args = (
      file_dir,
      model_class,
      model_class_kwargs,
      config,
      manifest,
      mesh,
      dtype,
  )
  if mode == 'optimized':
    return load_and_create_model_opt(*args)
  return load_and_create_model_orig(*args)
