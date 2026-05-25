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

"""Common utilities for loading OSS model weights from safetensors files.

Not compatible with GOOGLE_INTERNAL_PACKAGE_PATH.
"""

import concurrent.futures
import contextlib
import json
import mmap
import os
import struct
import threading

from absl import logging
from etils import epath
from flax import nnx
import jax
from jax.experimental import colocated_python
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import safetensors.flax as safetensors
from tunix.oss import utils
from tunix.utils import compat
from tunix.utils import env_utils
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


def get_tensor_spec(metadata, transform, target_dtype):
  """Returns the shape and dtype for the given tensor metadata.

  Args:
    metadata: The tensor metadata.
    transform: The tensor transformation.
    target_dtype: The target dtype.

  Returns:
    The shape and dtype spec.
  """
  shape = list(metadata['shape'])
  dtype = to_np_dtype(metadata['dtype'])
  if target_dtype:
    dtype = to_np_dtype(target_dtype)
  if transform:
    permute, reshape = transform
    if permute:
      shape = [shape[i] for i in permute]
    if reshape:
      shape = list(reshape)
  return jax.ShapeDtypeStruct(shape=tuple(shape), dtype=dtype)


def load_safetensors_with_offsets(filepath):
  """Loads safetensors file and returns tensor metadata with offsets.

  Args:
    filepath: The path to the safetensors file.

  Returns:
    A tuple containing:
      - contiguous_array: A numpy array containing the concatenated tensor data.
      - tensor_metadata: A list of dictionaries, each containing metadata
        (name, offset_elements, size_elements, shape, dtype) for a tensor.
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

  itemsize = 2  # Default to bfloat16
  common_dtype = None
  for tensor_name, metadata in header.items():
    if tensor_name == '__metadata__':
      continue

    dtype = metadata['dtype']
    if common_dtype is None:
      common_dtype = dtype
      np_type = to_np_dtype(dtype)
      itemsize = np.dtype(np_type).itemsize

    start_byte, end_byte = metadata['data_offsets']
    shape = tuple(metadata['shape'])

    size_bytes = end_byte - start_byte
    size_elements = size_bytes // itemsize
    offset_elements = start_byte // itemsize

    tensor_metadata.append({
        'name': tensor_name,
        'offset_elements': offset_elements,
        'size_elements': size_elements,
        'shape': shape,
        'dtype': dtype,
    })

  file_size = os.path.getsize(filepath)
  data_size_bytes = file_size - data_block_start_offset_bytes
  total_elements = data_size_bytes // itemsize

  f = open(filepath, 'rb')

  mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
  contiguous_array = np.frombuffer(
      mm,
      dtype=to_np_dtype(common_dtype),
      count=total_elements,
      offset=data_block_start_offset_bytes,
  )

  return contiguous_array, tensor_metadata, mm, f


# TODO(tunix-dev): Deduplicate the logic in this function and the original
# version in load_and_create_model_orig.
def _load_tensors_from_files(
    files, key_map, dtype, preprocess_fn, target_cpu_device
):
  file_lock = threading.Lock()

  file_loaded_tensors = {}
  skipped_keys = []
  for f in files:
    with safetensors.safe_open(f, framework='numpy') as sf:
      keys = sf.keys()

      def process_key(k_name, f, sf_file, file_loaded_tensors):
        try:
          with file_lock:
            v = sf_file.get_tensor(k_name)  # get_tensor is not thread-safe
          try:
            jax_key_mapped, transform = torch_utils.torch_key_to_jax_key(
                key_map, k_name
            )
          except ValueError:
            skipped_keys.append(k_name)
            return

          if transform is not None:
            permute, reshape = transform
            if permute:
              v = v.transpose(permute)
            if reshape:
              v = v.reshape(reshape)

          current_arr = np.array(v)
          if dtype and current_arr.dtype != dtype:
            np_dtype = to_np_dtype(dtype)
            current_arr = current_arr.astype(np_dtype)

          if jax_key_mapped in file_loaded_tensors:
            raise ValueError(
                f'Duplicate key {jax_key_mapped} found within file {f.name}.'
            )
          file_loaded_tensors[jax_key_mapped] = jax.device_put(
              current_arr, target_cpu_device
          )

        except Exception as e:
          raise RuntimeError(
              f'Failed to load tensor {k_name} from file {f.name}: {e}'
          ) from e

      with concurrent.futures.ThreadPoolExecutor(
          max_workers=os.cpu_count()
      ) as executor:
        futures = [
            executor.submit(process_key, key, f, sf, file_loaded_tensors)
            for key in keys
        ]

      for future in concurrent.futures.as_completed(futures):
        if future.exception():
          raise future.exception()

  # Apply preprocessing if provided (e.g., for MoE expert stacking)
  if preprocess_fn is not None:
    file_loaded_tensors = preprocess_fn(file_loaded_tensors)

  return file_loaded_tensors, skipped_keys


@colocated_python.colocated_python_class
class RemoteModelLoader:
  """Loader for model weights on remote workers."""

  def __init__(
      self, files, key_map, dtype, preprocess_fn, target_cpu_device_id
  ):
    self.files = files
    self.key_map = key_map
    self.dtype = dtype
    self.preprocess_fn = preprocess_fn
    self.target_cpu_device_id = target_cpu_device_id

  def load(self):
    """Loads model weights from safetensors files on remote workers."""
    cpu_devices = jax.devices()
    target_cpu_device = None
    for d in cpu_devices:
      if d.id == self.target_cpu_device_id:
        target_cpu_device = d
        break

    if target_cpu_device is None:
      raise RuntimeError(
          f'Worker could not find device with ID {self.target_cpu_device_id}. '
          f'Available devices: {cpu_devices}'
      )

    file_loaded_tensors, skipped_keys = _load_tensors_from_files(
        self.files,
        self.key_map,
        self.dtype,
        self.preprocess_fn,
        target_cpu_device,
    )

    if skipped_keys:
      logging.warning(
          'Skipped loading %d keys because they could not be mapped to model '
          'weights. This may be expected, for example when loading only text '
          'weights from a multimodal checkpoint. Missing keys: [%s]',
          len(skipped_keys),
          ', '.join(skipped_keys),
      )

    return file_loaded_tensors


def load_and_create_model_orig(
    file_dir: str,
    model_class,
    config,
    key_mapping,
    mesh=None,
    preprocess_fn=None,
    dtype: jnp.dtype | None = None,
    remote_loading: bool = False,
):
  """Generic function to load model from safetensors files.

  Args:
      file_dir: Directory containing safetensors files
      model_class: Model class to instantiate
      config: Model configuration
      key_mapping: Function that returns key mapping dictionary
      mesh: Optional JAX mesh for sharding
      preprocess_fn: Optional function to preprocess loaded parameters
      dtype: Optional dtype to cast loaded parameters to
      remote_loading: Whether to load from remote workers.

  Returns:
      Model instance with loaded weights
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
    model = nnx.eval_shape(lambda: model_class(config, rngs=nnx.Rngs(params=0)))

  graph_def, abs_state = nnx.split(model)
  state_dict = abs_state.to_pure_dict()

  if mesh is not None:
    sharding_dict = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
  else:
    sharding_dict = None

  key_map = key_mapping(config)

  if remote_loading:
    if not env_utils.is_pathways_initialized():
      raise ValueError(
          'Remote loading is only supported when Pathways is initialized.'
      )

    if mesh is None:
      raise ValueError(
          'A JAX mesh must be provided when running with Pathways.'
      )

    cpu_devices = colocated_python.colocated_cpu_devices(
        tuple(mesh.devices.flatten())
    )
    if not cpu_devices:
      raise ValueError('No CPU devices found for colocated python.')

    worker_map = {}
    for cpu_device in cpu_devices:
      logical_task_id = cpu_device.logical_task_id
      if logical_task_id not in worker_map:
        worker_map[logical_task_id] = []
      worker_map[logical_task_id].append(cpu_device)
    worker_indices = sorted(worker_map.keys())
    num_workers = len(worker_indices)
    logging.info('Number of unique workers: %s', num_workers)

    key_to_spec = {}
    file_to_keys = {}
    for file_path in files:
      file_to_keys[file_path] = set()
      with open(file_path, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_size).decode('utf-8'))
        for tensor_name, metadata in header.items():
          if tensor_name == '__metadata__':
            continue
          jax_key_mapped, transform = torch_utils.torch_key_to_jax_key(
              key_map, tensor_name
          )
          if jax_key_mapped in key_to_spec:
            raise ValueError(
                "f'Duplicate key {jax_key_mapped} found within file {filepath}."
            )
          key_to_spec[jax_key_mapped] = get_tensor_spec(
              metadata, transform, dtype
          )
          file_to_keys[file_path].add(jax_key_mapped)

    files_per_worker = [[] for _ in range(num_workers)]
    for i, file_path in enumerate(files):
      files_per_worker[i % num_workers].append(file_path)

    def make_worker_out_specs_fn(specs):
      return lambda: specs

    futures = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=os.cpu_count()
    ) as executor:
      for i, worker_index in enumerate(worker_indices):
        worker_files = files_per_worker[i]
        target_cpu_device = worker_map[worker_index][0]
        worker_device_sharding = jax.sharding.SingleDeviceSharding(
            target_cpu_device
        )

        worker_keys = set()
        for file_path in worker_files:
          worker_keys.update(file_to_keys.get(file_path, set()))

        worker_out_specs = {}
        for k in worker_keys:
          if k in key_to_spec:
            s = key_to_spec[k]
            worker_out_specs[k] = jax.ShapeDtypeStruct(
                shape=s.shape, dtype=s.dtype, sharding=worker_device_sharding
            )
          else:
            logging.error(
                'Key %s found in file_to_keys but not in key_to_spec', k
            )

        remote_loader = RemoteModelLoader(
            worker_files, key_map, dtype, preprocess_fn, target_cpu_device.id
        )

        load_specialized = remote_loader.load.specialize(
            devices=[target_cpu_device],
            out_specs_fn=make_worker_out_specs_fn(worker_out_specs),
        )
        future = executor.submit(load_specialized)
        futures.append(future)

    all_cpu_arrays = {}
    for future in concurrent.futures.as_completed(futures):
      worker_results = future.result()
      for key, value in worker_results.items():
        if key in all_cpu_arrays:
          logging.warning('Duplicate key %s detected across workers.', key)
        all_cpu_arrays[key] = value

    def _transfer_to_tpu(path, state, sharding):
      current_path_key = path_to_key(path)
      possible_keys = [current_path_key, f'{current_path_key}.value']
      for k in possible_keys:
        if k in all_cpu_arrays:
          cpu_array = all_cpu_arrays[k]
          if cpu_array.shape != state.shape:
            raise ValueError(
                f'Shape mismatch for {k}: got {cpu_array.shape}, expected'
                f' {state.shape}'
            )
          return jax.device_put(cpu_array, sharding)
      return state

    state_dict = jax.tree_util.tree_map_with_path(
        _transfer_to_tpu, state_dict, sharding_dict
    )
  else:
    file_lock = threading.Lock()

    # Load tensors from all files
    skipped_keys = []
    for f in files:
      file_loaded_tensors = {}
      with safetensors.safe_open(f, framework='numpy') as sf:
        keys = sf.keys()

        def process_key(k_name, f, sf_file, file_loaded_tensors):
          try:
            with file_lock:
              v = sf_file.get_tensor(k_name)  # get_tensor is not thread-safe
            try:
              jax_key_mapped, transform = torch_utils.torch_key_to_jax_key(
                  key_map, k_name
              )
            except ValueError:
              skipped_keys.append(k_name)
              return

            if transform is not None:
              permute, reshape = transform
              if permute:
                v = v.transpose(permute)
              if reshape:
                v = v.reshape(reshape)

            current_arr = jnp.array(v)
            if dtype and current_arr.dtype != dtype:
              current_arr = current_arr.astype(dtype)

            if jax_key_mapped in file_loaded_tensors:
              raise ValueError(
                  f'Duplicate key {jax_key_mapped} found within file {f.name}.'
              )
            file_loaded_tensors[jax_key_mapped] = current_arr

          except Exception as e:
            raise RuntimeError(
                f'Failed to load tensor {k_name} from file {f.name}: {e}'
            ) from e

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
          futures = [
              executor.submit(process_key, key, f, sf, file_loaded_tensors)
              for key in keys
          ]

        for future in concurrent.futures.as_completed(futures):
          if future.exception():
            raise future.exception()

      # Apply preprocessing if provided (e.g., for MoE expert stacking)
      if preprocess_fn is not None:
        file_loaded_tensors = preprocess_fn(file_loaded_tensors)

      if skipped_keys:
        logging.warning(
            'Skipped loading %d keys because they could not be mapped to model '
            'weights. This may be expected, for example when loading only text '
            'weights from a multimodal checkpoint. Missing keys: [%s]',
            len(skipped_keys),
            ', '.join(skipped_keys),
        )

      def make_update_tensor_fn(current_file_tensors):
        def update_tensor(path, param, shard=None):
          current_path_key = path_to_key(path)

          # nnx.Param adds a .value suffix to the key
          possible_keys = [current_path_key, f'{current_path_key}.value']

          for k in possible_keys:
            if k in current_file_tensors:
              loaded_arr = current_file_tensors[k]
              if loaded_arr.shape != param.shape:
                raise ValueError(
                    f'Shape mismatch for {k}: got'
                    f' {loaded_arr.shape}, expected {param.shape}'
                )
              if shard is not None:
                return jax.device_put(loaded_arr, shard)
              else:
                return jax.device_put(loaded_arr, jax.devices()[0])

          return param

        return update_tensor

      current_file_update_tensor = make_update_tensor_fn(file_loaded_tensors)

      if sharding_dict is not None:
        state_dict = jax.tree.map_with_path(
            current_file_update_tensor, state_dict, sharding_dict
        )
      else:
        state_dict = jax.tree.map_with_path(
            current_file_update_tensor, state_dict
        )

  return nnx.merge(graph_def, state_dict)


def load_and_create_model_opt(
    file_dir: str,
    model_class,
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
    model = nnx.eval_shape(lambda: model_class(config, rngs=nnx.Rngs(params=0)))

  graph_def, abs_state = nnx.split(model)
  nnx_state_dict = abs_state.to_pure_dict()

  if mesh is not None:
    sharding_dict = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
  else:
    device = jax.devices()[0]
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
      parameter = array[
          metadata['offset_elements'] : metadata['offset_elements']
          + metadata['size_elements']
      ].reshape(metadata['shape'])
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
      possible_keys = [
          key,
          f'{key}.value',
      ]  # for nnx.Param where a value suffix will be there
      tensor = None
      for k in possible_keys:
        if k in state_dict:
          tensor = state_dict[k]
          break
      if tensor is not None and dtype is not None:
        np_dtype = to_np_dtype(dtype)
        tensor = tensor.astype(np_dtype)
      return jax.device_put(tensor, sharding)

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
    remote_loading: bool = False,
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

  Returns:
    An NNX model instance with weights loaded from the safetensors files.
  """
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

  if mode == 'optimized':
    return load_and_create_model_opt(
        file_dir, model_class, config, key_mapping, mesh, preprocess_fn, dtype
    )
  else:
    return load_and_create_model_orig(
        file_dir,
        model_class,
        config,
        key_mapping,
        mesh,
        preprocess_fn,
        dtype,
        remote_loading,
    )
