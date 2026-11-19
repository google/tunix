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
from collections.abc import Mapping, Sequence
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
import ml_dtypes
import numpy as np
from tunix.oss import utils
from tunix.utils import compat
from tunix.utils import env_utils
from tunix.utils import sharding_utils
from tunix.utils import torch_utils

load_file_from_gcs = utils.load_file_from_gcs

torch_key_to_jax_key = torch_utils.torch_key_to_jax_key


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


def to_np_dtype(dtype):
  if dtype == 'BF16' or dtype == jnp.bfloat16:
    return ml_dtypes.bfloat16
  elif dtype == 'F16' or dtype == jnp.float16:
    return np.float16
  elif dtype == 'F32' or dtype == jnp.float32:
    return np.float32
  elif dtype == 'F64' or dtype == jnp.float64:
    return np.float64


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
  """
  with open(filepath, 'rb') as f:
    header_size_bytes = f.read(8)
    header_size = struct.unpack('<Q', header_size_bytes)[0]
    header_bytes = f.read(header_size)
    header = json.loads(header_bytes.decode('utf-8'))

  data_block_start_offset_bytes = 8 + header_size

  tensor_metadata = []

  for tensor_name, metadata in header.items():
    if tensor_name == '__metadata__':
      continue

    dtype = metadata['dtype']
    start_byte, end_byte = metadata['data_offsets']
    shape = tuple(metadata['shape'])

    tensor_metadata.append({
        'name': tensor_name,
        'start_byte': start_byte,
        'end_byte': end_byte,
        'shape': shape,
        'dtype': dtype,
    })

  file_size = os.path.getsize(filepath)
  data_size_bytes = file_size - data_block_start_offset_bytes

  f = open(filepath, 'rb')

  mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
  contiguous_array = np.frombuffer(
      mm,
      dtype=np.uint8,
      count=data_size_bytes,
      offset=data_block_start_offset_bytes,
  )

  return contiguous_array, tensor_metadata, mm, f


_TensorTransform = tuple[Sequence[int] | None, Sequence[int] | None] | None
_KeyMap = Mapping[str, tuple[str, _TensorTransform]]
_Chunk = tuple[int, np.ndarray]  # (Absolute file offset, flat uint8 buffer.)

_SAFETENSORS_DTYPES: Mapping[str, np.dtype] = {
    'BOOL': np.dtype(np.bool_),
    'U8': np.dtype(np.uint8),
    'I8': np.dtype(np.int8),
    'U16': np.dtype(np.uint16),
    'I16': np.dtype(np.int16),
    'U32': np.dtype(np.uint32),
    'I32': np.dtype(np.int32),
    'U64': np.dtype(np.uint64),
    'I64': np.dtype(np.int64),
    'F8_E4M3': np.dtype(ml_dtypes.float8_e4m3fn),
    'F8_E5M2': np.dtype(ml_dtypes.float8_e5m2),
    'BF16': np.dtype(ml_dtypes.bfloat16),
    'F16': np.dtype(np.float16),
    'F32': np.dtype(np.float32),
    'F64': np.dtype(np.float64),
}
# Each read worker owns one file handle and pulls `_Chunk`s from a shared
# queue; `readinto` fills preallocated host buffers with the GIL released.
_READ_WORKERS = 16
_READ_CHUNK_BYTES = 32 * 1024 * 1024


class _HostTensor(NamedTuple):
  """A mapped checkpoint tensor and the host bytes it is read into."""

  jax_key: str
  transform: _TensorTransform
  raw: np.ndarray  # Flat uint8 view into a span buffer.
  dtype: np.dtype
  shape: tuple[int, ...]


def _plan_file_reads(
    filepath: epath.Path, key_map: _KeyMap, skipped_keys: list[str]
) -> tuple[list[_HostTensor], list[_Chunk]]:
  """Allocates host buffers for the mapped tensors of one safetensors file.

  Offset-adjacent tensors share one span buffer of at most `_READ_CHUNK_BYTES`,
  so a run of small tensors costs one read; a larger tensor gets its own
  buffer, read as `_READ_CHUNK_BYTES` stripes. Unmapped tensors are not read.

  Args:
    filepath: The safetensors file.
    key_map: Regex mapping from checkpoint keys to model keys and transforms.
    skipped_keys: Receives the checkpoint keys that map to no model weight.

  Returns:
    The mapped tensors in file order (buffers not yet filled), and the
    `(file_offset, buffer)` chunks whose reads fill them.

  Raises:
    OSError: If the safetensors header or metadata section is truncated.
  """
  with cast(io.BufferedIOBase, filepath.open('rb')) as fh:
    size_bytes = fh.read(8)
    if len(size_bytes) != 8:
      raise OSError(f'Truncated safetensors header in {filepath}.')
    header_size = struct.unpack('<Q', size_bytes)[0]
    header_bytes = fh.read(header_size)
    if len(header_bytes) != header_size:
      raise OSError(f'Truncated safetensors metadata in {filepath}.')
    header = json.loads(header_bytes)

  mapped = []  # (start, end, jax_key, transform, meta), data-section offsets.
  for name, meta in header.items():
    if name == '__metadata__':
      continue
    try:
      jax_key, transform = torch_key_to_jax_key(key_map, name)
    except ValueError:
      skipped_keys.append(name)
      continue
    mapped.append((*meta['data_offsets'], jax_key, transform, meta))
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
            transform=transform,
            raw=buf[start - lo : end - lo],
            dtype=_SAFETENSORS_DTYPES[meta['dtype']],
            shape=tuple(meta['shape']),
        )
        for start, end, jax_key, transform, meta in mapped[i:j]
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


def _apply_transform(
    arr: np.ndarray, transform: _TensorTransform
) -> np.ndarray:
  """Applies an optional `(permute, reshape)` key-mapping transform."""
  if transform is None:
    return arr
  permute, reshape = transform
  if permute:
    arr = arr.transpose(permute)
  if reshape:
    arr = arr.reshape(reshape)
  return arr


def _load_file_tensors(
    filepath: epath.Path,
    key_map: _KeyMap,
    target_dtype: np.dtype | None,
    skipped_keys: list[str],
) -> dict[str, np.ndarray]:
  """Reads the mapped tensors of one safetensors file into host arrays.

  Every destination buffer is preallocated, so parallel reads need no byte
  budget, locks or per-buffer refcounts: threads, open handles and memory are
  bounded by construction.

  Args:
    filepath: The safetensors file.
    key_map: Regex mapping from checkpoint keys to model keys and transforms.
    target_dtype: Optional host dtype to cast the tensors to.
    skipped_keys: Receives the checkpoint keys that map to no model weight.

  Returns:
    Host arrays keyed by model key.

  Raises:
    RuntimeError: If two checkpoint keys in the file map to the same model key.
  """
  tensors, chunks = _plan_file_reads(filepath, key_map, skipped_keys)
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
    if t.jax_key in loaded:
      raise RuntimeError(
          f'Duplicate key {t.jax_key} found within file {filepath.name}.'
      )
    arr = _apply_transform(t.raw.view(t.dtype).reshape(t.shape), t.transform)
    # Single pass casts and enforces C-contiguous, aligned, writable layout;
    # zero-copy no-op when the raw span view already satisfies all flags.
    loaded[t.jax_key] = np.require(
        arr, dtype=target_dtype, requirements=('C', 'A', 'W')
    )
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


def load_and_create_model_orig(
    file_dir: str,
    model_class,
    model_class_kwargs,
    config,
    key_mapping,
    mesh=None,
    preprocess_fn=None,
    dtype: jnp.dtype | None = None,
):
  """Generic function to load model from safetensors files.

  Args:
      file_dir: Directory containing safetensors files
      model_class: Model class to instantiate
      model_class_kwargs: Optional keyword arguments to pass to the model class.
      config: Model configuration
      key_mapping: Function that returns key mapping dictionary
      mesh: Optional JAX mesh for sharding
      preprocess_fn: Optional function to preprocess loaded parameters
      dtype: Optional dtype to cast loaded parameters to

  Returns:
      Model instance with loaded weights
  """
  if file_dir.startswith('gs://'):
    file_dir = load_file_from_gcs(file_dir)

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
  key_map = key_mapping(config)
  target_dtype = None if dtype is None else np.dtype(dtype)

  # One file at a time bounds host memory (all-file loading OOMs the Pathways
  # IFRT proxy on 32B models) and allows one batched device_put per file.
  for f in files:
    start_time = time.monotonic()
    skipped_keys: list[str] = []
    tensors = _load_file_tensors(f, key_map, target_dtype, skipped_keys)
    # Apply preprocessing if provided (e.g., for MoE expert stacking)
    if preprocess_fn is not None:
      tensors = preprocess_fn(tensors)

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
      # `device_get` turns any `jax.Array` returned by `preprocess_fn` into a
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

  return nnx.merge(graph_def, jax.tree_util.tree_unflatten(treedef, params))


def load_and_create_model_opt(
    file_dir: str,
    model_class,
    model_class_kwargs,
    config,
    key_mapping,
    mesh=None,
    preprocess_fn=None,
    dtype: jnp.dtype | None = None,
):
  """Loads safetensors files and creates an NNX model.

  This version is optimized for linux file systems.

  Args:
    file_dir: Directory containing the safetensors files.
    model_class: The NNX model class to instantiate.
    model_class_kwargs: Optional keyword arguments to pass to the model class.
    config: The configuration object for the model.
    key_mapping: A function that takes the config and returns a mapping from
      torch keys to jax keys and optional transformations.
    mesh: An optional JAX device mesh for sharding.
    preprocess_fn: An optional function to preprocess the loaded state dict
      before sharding.
    dtype: Optional dtype to cast the loaded tensors to.

  Returns:
    An NNX model instance with weights loaded from the safetensors files.

  Raises:
    ValueError: If no safetensors files are found in the specified directory.
  """
  if file_dir.startswith('gs://'):
    file_dir = load_file_from_gcs(file_dir)

  files = list(epath.Path(file_dir).expanduser().glob('*.safetensors'))

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
  nnx_state_dict = abs_state.to_pure_dict()

  if mesh is not None:
    sharding_dict = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
  else:
    device = jax.local_devices()[0]
    sharding_dict = jax.tree.map(lambda _: device, nnx_state_dict)

  key_map = key_mapping(config)

  arrays = []
  mmaps = []
  file_handles = []
  for f in files:
    contiguous_array, tensor_metadata, mm, fh = load_safetensors_with_offsets(f)
    arrays.append((contiguous_array, tensor_metadata))
    mmaps.append(mm)
    file_handles.append(fh)

  state_dict = {}
  skipped_keys = []
  for array, metadata_list in arrays:
    for metadata in metadata_list:
      try:
        jax_key_mapped, transform = torch_key_to_jax_key(
            key_map, metadata['name']
        )
      except ValueError:
        skipped_keys.append(metadata['name'])
        continue
      raw_bytes = array[metadata['start_byte'] : metadata['end_byte']]
      parameter = raw_bytes.view(to_np_dtype(metadata['dtype'])).reshape(
          metadata['shape']
      )
      if transform is not None:
        permute, reshape = transform
        if permute:
          parameter = parameter.transpose(permute)
        if reshape:
          parameter = parameter.reshape(reshape)
      state_dict[jax_key_mapped] = parameter

    if skipped_keys:
      logging.warning(
          'Skipped loading %d keys because they could not be mapped to model '
          'weights. This may be expected, for example when loading only text '
          'weights from a multimodal checkpoint. Missing keys: [%s]',
          len(skipped_keys),
          ', '.join(skipped_keys),
      )

  if preprocess_fn is not None:
    state_dict = preprocess_fn(state_dict)

  def shard_state(state_dict):
    def _shard_state(path, sharding):
      key = path_to_key(path)
      tensor = (
          state_dict[key] if key in state_dict else state_dict[f'{key}.value']
      )
      if dtype is not None:
        tensor = tensor.astype(to_np_dtype(dtype))
      return sharding_utils.place_array_on_target(tensor, sharding)

    return _shard_state

  shard_function = shard_state(state_dict)
  state_dict = jax.tree.map_with_path(shard_function, sharding_dict)

  def _assert_shapes_match(path, x, y):
    assert (
        x.shape == y.shape
    ), f'Shape mismatch for {path}: expected {y.shape}, got {x.shape}'
    return x

  jax.tree.map_with_path(_assert_shapes_match, state_dict, nnx_state_dict)

  for fh in file_handles:
    fh.close()

  return nnx.merge(graph_def, state_dict)


def load_and_create_model(
    file_dir: str,
    model_class,
    config,
    key_mapping,
    mesh=None,
    preprocess_fn=None,
    dtype: jnp.dtype | None = None,
    mode: str = 'auto',
    model_class_kwargs: dict[str, Any] | None = None,
):
  """Loads safetensors files and creates an NNX model.

  Args:
    file_dir: Directory containing the safetensors files.
    model_class: The NNX model class to instantiate.
    config: The configuration object for the model.
    key_mapping: A function that takes the config and returns a mapping from
      torch keys to jax keys and optional transformations.
    mesh: An optional JAX device mesh for sharding.
    preprocess_fn: An optional function to preprocess the loaded state dict
      before sharding.
    dtype: Optional dtype to cast the loaded tensors to.
    mode: The mode to use for loading the model. Options are ('auto',
      'optimized', 'original').
    model_class_kwargs: Optional keyword arguments to pass to the model class.

  Returns:
    An NNX model instance with weights loaded from the safetensors files.
  """
  if model_class_kwargs is None:
    model_class_kwargs = {}

  if mode == 'auto':
    if env_utils.is_internal_env() or env_utils.is_pathways_initialized():
      mode = 'original'
    else:
      mode = 'optimized'

  # TODO(tunix-dev): Fix optimized mode when pathways is initialized.
  if mode == 'optimized':
    if env_utils.is_internal_env() or env_utils.is_pathways_initialized():
      raise ValueError(
          'Optimized mode is not supported in GOOGLE_INTERNAL_PACKAGE_PATH or pathways initialized.'
      )

  if mode == 'original':
    if (
        not env_utils.is_internal_env()
        and not env_utils.is_pathways_initialized()
    ):
      logging.warning('Optimized mode is faster and recommended if possible.')

  args = (
      file_dir,
      model_class,
      model_class_kwargs,
      config,
      key_mapping,
      mesh,
      preprocess_fn,
      dtype,
  )
  if mode == 'optimized':
    return load_and_create_model_opt(*args)
  else:
    return load_and_create_model_orig(*args)
