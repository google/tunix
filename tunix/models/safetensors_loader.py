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

"""Common utilities for loading OSS model weights from safetensors files. Not compatible with GOOGLE_INTERNAL_PACKAGE_PATH."""

import contextlib
import json
import mmap
import os
import struct

import concurrent.futures
import threading
from absl import logging
import safetensors.flax as safetensors


from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from tunix.oss import utils
from tunix.utils import compat
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


def load_and_create_model_new(
    file_dir: str,
    model_class,
    config,
    key_mapping,
    mesh=None,
    preprocess_fn=None,
    dtype: jnp.dtype | None = None,
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
  state_dict = abs_state.to_pure_dict()

  if mesh is not None:
    sharding_dict = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
  else:
    device = jax.devices()[0]
    sharding_dict = jax.tree.map(lambda _: device, state_dict)

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
  for array, metadata_list in arrays:
    for metadata in metadata_list:
      jax_key_mapped, transform = torch_key_to_jax_key(
          key_map, metadata['name']
      )
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

  if preprocess_fn is not None:
    state_dict = preprocess_fn(state_dict)

  def shard_state(state_dict):
    def _shard_state(path, sharding):
      key = path_to_key(path)
      tensor = state_dict[key]
      if dtype is not None:
        np_dtype = to_np_dtype(dtype)
        tensor = tensor.astype(np_dtype)
      return jax.device_put(tensor, sharding)

    return _shard_state

  shard_function = shard_state(state_dict)
  state_dict = jax.tree.map_with_path(shard_function, sharding_dict)

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

  Returns:
      Model instance with loaded weights
  """

  if file_dir.startswith('gs://'):
    file_dir = load_file_from_gcs(file_dir)

  files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))

  if not files:
    raise ValueError(f"No safetensors found in {file_dir}")

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

  file_lock = threading.Lock()

  # Load tensors from all files
  skipped_keys = []
  for f in files:
    file_loaded_tensors = {}
    with safetensors.safe_open(f, framework="numpy") as sf:
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
                f"Duplicate key {jax_key_mapped} found within file {f.name}."
            )
          file_loaded_tensors[jax_key_mapped] = current_arr

        except Exception as e:
          raise RuntimeError(
              f"Failed to load tensor {k_name} from file {f.name}: {e}"
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
          "Skipped loading %d keys because they could not be mapped to model "
          "weights. This may be expected, for example when loading only text "
          "weights from a multimodal checkpoint. Missing keys: [%s]",
          len(skipped_keys),
          ", ".join(skipped_keys),
      )

    def make_update_tensor_fn(current_file_tensors):
      def update_tensor(path, param, shard=None):
        current_path_key = path_to_key(path)
        if current_path_key in current_file_tensors:
          loaded_arr = current_file_tensors[current_path_key]
          if loaded_arr.shape != param.shape:
            raise ValueError(
                f"Shape mismatch for {current_path_key}: got"
                f" {loaded_arr.shape}, expected {param.shape}"
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