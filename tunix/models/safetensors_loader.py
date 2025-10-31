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

import concurrent.futures
import contextlib
import functools
import operator
import os
import re
import threading
from typing import Any

from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import safetensors.flax as safetensors


def torch_key_to_jax_key(mapping, source_key):
  """Convert torch key to jax key using the provided mapping."""
  subs = [
      (re.sub(pat, repl, source_key), reshape)
      for pat, (repl, reshape) in mapping.items()
      if re.match(pat, source_key)
  ]
  if len(subs) != 1:
    raise ValueError(f"Only one key should be found: {subs} for {source_key}")
  else:
    return subs[0]


def stoi(s):
  """Convert string to int if possible, otherwise return as is."""
  try:
    return int(s)
  except ValueError:
    return s


def path_to_key(path):
  """Convert path to string key."""
  return ".".join(
      str(stoi(key.key if hasattr(key, "key") else key)) for key in path
  )


def flatten_nested_dict(nested_dict: dict[str, Any], parent_key="", sep="."):
  """Flatten a nested dictionary."""
  items = []

  for key, value in nested_dict.items():
    new_key = parent_key + sep + str(key) if parent_key else key

    if isinstance(value, dict) and value:
      items.extend(flatten_nested_dict(value, new_key, sep=sep).items())
    else:
      items.append((new_key, value))

  return dict(items)


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
  files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))

  if not files:
    raise ValueError(f"No safetensors found in {file_dir}")

  # Create model structure
  context_manager = mesh if mesh is not None else contextlib.nullcontext()

  with context_manager:
    model = nnx.eval_shape(lambda: model_class(config, rngs=nnx.Rngs(params=0)))

  graph_def, abs_state = nnx.split(model)
  state_dict = abs_state.to_pure_dict()

  if mesh is not None:
    sharding_dict = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
    sharding_dict = flatten_nested_dict(sharding_dict)
  else:
    sharding_dict = None

  key_map = key_mapping(config)

  file_lock = threading.Lock()

  # Load tensors from all files
  def process_file(f):
    file_loaded_tensors = {}
    with safetensors.safe_open(f, framework="numpy") as sf:
      keys = sf.keys()

      def process_key(k_name, f, sf_file, file_loaded_tensors):
        try:
          with file_lock:
            v = sf_file.get_tensor(k_name)  # get_tensor is not thread-safe
          jax_key_mapped, transform = torch_key_to_jax_key(key_map, k_name)

          if transform is not None:
            permute, reshape = transform
            if permute:
              v = v.transpose(permute)
            if reshape:
              v = v.reshape(reshape)

          current_arr = jnp.array(v)
          if dtype and current_arr.dtype != dtype:
            current_arr = current_arr.astype(dtype)

          if sharding_dict is not None:
            current_arr = jax.device_put(
                current_arr, sharding_dict[jax_key_mapped]
            )
          else:
            current_arr = jax.device_put(current_arr, jax.devices()[0])

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

    return file_loaded_tensors

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=len(files)
  ) as executor:
    all_tensors = list(executor.map(process_file, files))

  merged_tensors = functools.reduce(operator.or_, all_tensors)

  def update_tensor(path, param, merged_tensors):
    current_path_key = path_to_key(path)
    if current_path_key in merged_tensors:
      loaded_arr = merged_tensors[current_path_key]
      if loaded_arr.shape != param.shape:
        raise ValueError(
            f"Shape mismatch for {current_path_key}: got"
            f" {loaded_arr.shape}, expected {param.shape}"
        )
      return loaded_arr
    raise ValueError(f"Tensor {current_path_key} not found in merged tensors.")

  state_dict = jax.tree.map_with_path(
      functools.partial(update_tensor, merged_tensors=merged_tensors),
      state_dict,
  )

  return nnx.merge(graph_def, state_dict)
