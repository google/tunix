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

import collections
from collections.abc import Callable, Mapping, Sequence
import concurrent.futures
import contextlib
import io
import json
import mmap
import os
import struct
import time
from typing import Any, NamedTuple, cast

from absl import logging
from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.models import hf_mapping
from tunix.models import safetensors_io
from tunix.oss import utils
from tunix.utils import compat
from tunix.utils import env_utils
from tunix.utils import sharding_utils

load_file_from_gcs = utils.load_file_from_gcs


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


_MAX_HEADER_BYTES = 100_000_000


def _read_safetensors_header(
    filepath: epath.Path,
) -> tuple[int, dict[str, Any]]:
  """Reads and validates the 8-byte length prefix and JSON header of `filepath`."""
  with cast(io.BufferedIOBase, filepath.open('rb')) as fh:
    size_bytes = fh.read(8)
    if len(size_bytes) != 8:
      raise OSError(f'Truncated safetensors header in {filepath}.')
    header_size = struct.unpack('<Q', size_bytes)[0]
    if header_size == 0 or header_size > _MAX_HEADER_BYTES:
      raise ValueError(
          f'Invalid safetensors header size {header_size} in {filepath}.'
      )
    header_bytes = fh.read(header_size)
    if len(header_bytes) != header_size:
      raise OSError(f'Truncated safetensors metadata in {filepath}.')
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
      raise ValueError(
          f'Safetensors header must be a JSON object in {filepath}.'
      )
    return header_size, header


def _validate_tensor_meta(
    filepath: epath.Path | str, name: str, meta: Any
) -> tuple[int, int, np.dtype, tuple[int, ...]]:
  """Validates safetensors dtype, non-negative shape, and exact byte span."""
  if (
      not isinstance(meta, Mapping)
      or meta.get('dtype') not in safetensors_io.SAFETENSORS_DTYPES
      or not isinstance(meta.get('shape'), Sequence)
      or not isinstance(meta.get('data_offsets'), Sequence)
      or len(meta['data_offsets']) != 2
  ):
    raise ValueError(f'Invalid metadata for {name!r} in {filepath}.')
  np_dt = safetensors_io.SAFETENSORS_DTYPES[meta['dtype']]
  raw_shape = meta['shape']
  if any(not isinstance(d, int) or d < 0 for d in raw_shape):
    raise ValueError(f'Invalid shape {raw_shape!r} for {name!r} in {filepath}.')
  shape = tuple(raw_shape)
  start, end = meta['data_offsets']
  expected_bytes = int(np.prod(shape, dtype=np.int64)) * np_dt.itemsize
  if (
      not isinstance(start, int)
      or not isinstance(end, int)
      or start < 0
      or end - start != expected_bytes
  ):
    raise ValueError(
        f'Invalid data_offsets {(start, end)!r} for {name!r} in {filepath}.'
    )
  return start, end, np_dt, shape


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


def to_np_dtype(dtype):
  if isinstance(dtype, str):
    if dtype not in safetensors_io.SAFETENSORS_DTYPES:
      raise ValueError(f'Unsupported safetensors dtype: {dtype!r}')
    return safetensors_io.SAFETENSORS_DTYPES[dtype]
  return np.dtype(dtype)


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


_Chunk = tuple[int, np.ndarray]  # (Absolute file offset, flat uint8 buffer.)
_StreamExecutor = Callable[[Mapping[str, Any]], dict[str, Any]]

# Each read worker owns one file handle and pulls `_Chunk`s from a shared
# queue; `readinto` fills preallocated host buffers with the GIL released.
_READ_WORKERS = 16
_READ_CHUNK_BYTES = 32 * 1024 * 1024


class _HostTensor(NamedTuple):
  """A mapped checkpoint tensor and the host bytes it is read into."""

  jax_key: str
  raw: np.ndarray  # Flat uint8 view into a span buffer.
  dtype: np.dtype
  shape: tuple[int, ...]


def _plan_file_reads(
    filepath: epath.Path,
    source_keys: frozenset[str],
    skipped_keys: list[str],
) -> tuple[list[_HostTensor], list[_Chunk]]:
  """Allocates host buffers for the mapped tensors of one safetensors file.

  Offset-adjacent tensors share one span buffer of at most `_READ_CHUNK_BYTES`,
  so a run of small tensors costs one read; a larger tensor gets its own
  buffer, read as `_READ_CHUNK_BYTES` stripes. Unmapped tensors are not read.

  Args:
    filepath: The safetensors file.
    source_keys: Active checkpoint source keys consumed by the manifest.
    skipped_keys: Receives the checkpoint keys that map to no model weight.

  Returns:
    The mapped tensors in file order (buffers not yet filled), and the
    `(file_offset, buffer)` chunks whose reads fill them.

  Raises:
    OSError: If the safetensors header or metadata section is truncated.
  """
  header_size, header = _read_safetensors_header(filepath)

  mapped: list[tuple[int, int, str, np.dtype, tuple[int, ...]]] = []
  for name, meta in header.items():
    if name == '__metadata__':
      continue
    start, end, np_dt, shape = _validate_tensor_meta(filepath, name, meta)
    if name not in source_keys:
      skipped_keys.append(name)
      continue
    mapped.append((start, end, name, np_dt, shape))
  mapped.sort(key=lambda entry: entry[0])

  tensors: list[_HostTensor] = []
  chunks: list[_Chunk] = []
  i = 0
  while i < len(mapped):
    j = i + 1  # Grow the span over offset-adjacent tensors while it fits.
    while (
        j < len(mapped)
        and mapped[j][0] == mapped[j - 1][1]
        and mapped[j][1] - mapped[i][0] <= _READ_CHUNK_BYTES
    ):
      j += 1
    lo, hi = mapped[i][0], mapped[j - 1][1]
    buf = np.empty(hi - lo, dtype=np.uint8)
    chunks += [
        (8 + header_size + lo + off, buf[off : off + _READ_CHUNK_BYTES])
        for off in range(0, hi - lo, _READ_CHUNK_BYTES)
    ]
    tensors += [
        _HostTensor(
            jax_key=jax_key,
            raw=buf[start - lo : end - lo],
            dtype=np_dt,
            shape=shape,
        )
        for start, end, jax_key, np_dt, shape in mapped[i:j]
    ]
    i = j
  return tensors, chunks


def _read_chunk_queue(
    filepath: epath.Path, pending: collections.deque[_Chunk]
) -> None:
  """Reads queued chunks into their buffers over one private file handle."""
  with cast(io.BufferedIOBase, filepath.open('rb')) as fh:
    while True:
      try:
        offset, buf = pending.popleft()
      except IndexError:  # Drained, or cleared after another worker failed.
        return
      fh.seek(offset)
      # GFile `readinto` (`ClifReadInto` -> `InputBuffer::Read`) releases the
      # GIL and loops until `buf` is full: it returns -1 on a read error and a
      # short count only at EOF. `io.BufferedReader.readinto` (OSS) also loops.
      if (n := fh.readinto(buf)) != buf.nbytes:
        raise OSError(
            f'Read {n} of {buf.nbytes} bytes at offset {offset} of {filepath}.'
        )


def _load_file_tensors(
    filepath: epath.Path,
    source_keys: frozenset[str],
    target_dtype: np.dtype | None,
    skipped_keys: list[str],
    seen_keys: set[str] | None = None,
) -> dict[str, np.ndarray]:
  """Reads the mapped tensors of one safetensors file into host arrays.

  Every destination buffer is preallocated, so parallel reads need no byte
  budget, locks or per-buffer refcounts: threads, open handles and memory are
  bounded by construction.

  Args:
    filepath: The safetensors file.
    source_keys: Active checkpoint source keys consumed by the manifest.
    target_dtype: Optional host dtype to cast the tensors to.
    skipped_keys: Receives the checkpoint keys that map to no model weight.
    seen_keys: Optional cross-shard set tracking already-loaded source keys.

  Returns:
    Host arrays keyed by checkpoint source key.
  """
  tensors, chunks = _plan_file_reads(filepath, source_keys, skipped_keys)
  if chunks:
    pending_chunks = collections.deque(chunks)
    with concurrent.futures.ThreadPoolExecutor(_READ_WORKERS) as pool:
      futures = [
          pool.submit(_read_chunk_queue, filepath, pending_chunks)
          for _ in range(min(_READ_WORKERS, len(chunks)))
      ]
      concurrent.futures.wait(
          futures, return_when=concurrent.futures.FIRST_EXCEPTION
      )
      pending_chunks.clear()  # After a failure, stop the other workers early.
    for future in futures:
      future.result()
  del chunks

  pending = collections.deque(tensors)
  del tensors  # Popping below frees each span once its tensors are converted.
  loaded: dict[str, np.ndarray] = {}
  while pending:
    t = pending.popleft()
    arr = t.raw.view(t.dtype).reshape(t.shape)
    # Single pass casts and enforces C-contiguous, aligned, writable layout;
    # zero-copy no-op when the raw span view already satisfies all flags.
    loaded[t.jax_key] = np.require(
        arr, dtype=target_dtype, requirements=('C', 'A', 'W')
    )
  if seen_keys is not None:
    dup_keys = seen_keys & loaded.keys()
    if dup_keys:
      raise ValueError(
          f'Duplicate tensor keys across safetensors shards: {sorted(dup_keys)}'
      )
    seen_keys.update(loaded)
  return loaded


def _match_params(
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


def load_and_create_model_orig(
    file_dir: str,
    model_class,
    model_class_kwargs,
    config,
    source_keys: frozenset[str],
    stream_executor: _StreamExecutor,
    mesh=None,
    dtype: jnp.dtype | None = None,
    pending: Mapping[str, Any] | None = None,
    hf_layout: str = '',
):
  """Generic function to load model from safetensors files.

  Args:
      file_dir: Directory containing safetensors files
      model_class: Model class to instantiate
      model_class_kwargs: Optional keyword arguments to pass to the model class.
      config: Model configuration
      source_keys: Active checkpoint source keys consumed by the manifest
      stream_executor: Streaming executor running declarative manifest rules
      mesh: Optional JAX mesh for sharding
      dtype: Optional dtype to cast loaded parameters to
      pending: Optional pending multi-source buffer to verify empty post-load
      hf_layout: Declared HF checkpoint layout, named in load errors

  Returns:
      Model instance with loaded weights
  """
  file_dir = resolve_checkpoint_dir(file_dir)

  files = sorted(epath.Path(file_dir).expanduser().glob('*.safetensors'))

  if not files:
    raise ValueError(f'No safetensors found in {file_dir}')

  # Create model structure
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

  # One file at a time bounds host memory (all-file loading OOMs the Pathways
  # IFRT proxy on 32B models) and allows one batched device_put per file.
  seen_keys: set[str] = set()
  for f in files:
    start_time = time.monotonic()
    skipped_keys: list[str] = []
    tensors = stream_executor(
        _load_file_tensors(
            f, source_keys, target_dtype, skipped_keys, seen_keys
        )
    )

    if skipped_keys:
      logging.warning(
          'Skipped loading %d keys because they could not be mapped to model '
          'weights. This may be expected, for example when loading only text '
          'weights from a multimodal checkpoint. Missing keys: [%s]',
          len(skipped_keys),
          ', '.join(skipped_keys),
      )

    matches = _match_params(keys, params, tensors)
    if matches:
      # `device_get` turns any `jax.Array` returned by `stream_executor` into a
      # host array, so placement is a host-to-device copy, not a D2D program.
      placed = sharding_utils.place_arrays_on_targets(
          jax.device_get([tensor for _, tensor in matches]),
          [shardings[i] for i, _ in matches],
      )
      for (i, _), arr in zip(matches, placed, strict=True):
        params[i] = arr
      del placed
    logging.info(
        'Loaded %s: %d tensors in %.2fs.',
        f.name,
        len(matches),
        time.monotonic() - start_time,
    )
    del tensors, matches  # Free host buffers before the next file.

  _verify_load_completeness(
      pending,
      [k for k, p in zip(keys, params) if isinstance(p, jax.ShapeDtypeStruct)],
      hf_layout,
  )
  return nnx.merge(graph_def, jax.tree_util.tree_unflatten(treedef, params))


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
      host_state[name] = raw.view(to_np_dtype(meta['dtype'])).reshape(
          meta['shape']
      )
    if skipped_keys:
      logging.warning(
          'Skipped loading %d keys because they could not be mapped to model '
          'weights. This may be expected, for example when loading only text '
          'weights from a multimodal checkpoint. Missing keys: [%s]',
          len(skipped_keys),
          ', '.join(skipped_keys),
      )
  return host_state


def load_and_create_model_opt(
    file_dir: str,
    model_class,
    model_class_kwargs,
    config,
    source_keys: frozenset[str],
    stream_executor: _StreamExecutor,
    mesh=None,
    dtype: jnp.dtype | None = None,
    pending: Mapping[str, Any] | None = None,
    hf_layout: str = '',
):
  """Loads safetensors files and creates an NNX model.

  This version is optimized for linux file systems.

  Args:
    file_dir: Directory containing the safetensors files.
    model_class: The NNX model class to instantiate.
    model_class_kwargs: Optional keyword arguments to pass to the model class.
    config: The configuration object for the model.
    source_keys: Active checkpoint source keys consumed by the manifest.
    stream_executor: Streaming executor running declarative manifest rules.
    mesh: An optional JAX device mesh for sharding.
    dtype: Optional dtype to cast the loaded tensors to.
    pending: Optional pending multi-source buffer to verify empty post-load.
    hf_layout: Declared HF checkpoint layout, named in load errors.

  Returns:
    An NNX model instance with weights loaded from the safetensors files.

  Raises:
    ValueError: If no safetensors files are found in the specified directory.
  """
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
    matches = _match_params(keys, params, host_state)
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
    if isinstance(pending, dict):
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
  """Loads safetensors files and creates an NNX model.

  Args:
    file_dir: Directory containing the safetensors files.
    model_class: The NNX model class to instantiate.
    config: The configuration object for the model.
    manifest: Declarative PrimOp `HF -> Tunix` weight-mapping manifest for the
      checkpoint's declared layout. Filters file reads by an O(1) `frozenset`
      of active `source_keys` and wires `hf_mapping.make_streaming_executor`;
      its `model_type` names the declared layout in load errors.
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
  file_dir = resolve_checkpoint_dir(file_dir)
  spec = hf_mapping.parse_manifest(manifest)
  pending: dict[str, Any] = {}
  source_keys = hf_mapping.active_source_keys(spec)
  stream_executor = hf_mapping.make_streaming_executor(spec, pending=pending)

  is_internal = (
      env_utils.is_internal_env() or env_utils.is_pathways_initialized()
  )
  if mode == 'auto':
    mode = 'original' if is_internal else 'optimized'
  if (mode, is_internal) == ('optimized', True):
    raise ValueError(
        'Optimized mode is not supported in GOOGLE_INTERNAL_PACKAGE_PATH or pathways initialized.'
    )
  if (mode, is_internal) == ('original', False):
    logging.warning('Optimized mode is faster and recommended if possible.')

  args = (
      file_dir,
      model_class,
      model_class_kwargs,
      config,
      source_keys,
      stream_executor,
      mesh,
      dtype,
      pending,
      spec['model_type'] if 'model_type' in spec else '',
  )
  if mode == 'optimized':
    return load_and_create_model_opt(*args)
  return load_and_create_model_orig(*args)
