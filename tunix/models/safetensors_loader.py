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

"""Pure utilities for loading model weights from safetensors files."""

import re
from typing import Dict, Tuple, Optional, Callable, Any, TypeVar
from etils import epath
from flax import nnx
import jax
import safetensors.flax as safetensors

ModelT = TypeVar('ModelT')
ConfigT = TypeVar('ConfigT')


def stoi(s):
    """Convert string to int if possible, otherwise return as is."""
    try:
        return int(s)
    except ValueError:
        return s


def torch_key_to_jax_key(mapping: Dict[str, Tuple[str, Optional[Tuple]]], source_key: str) -> Tuple[str, Optional[Tuple]]:
    """Convert torch key to jax key using the provided mapping.
    
    Args:
        mapping: Dictionary mapping regex patterns to (jax_key, transform) tuples
        source_key: The original torch parameter key
        
    Returns:
        Tuple of (jax_key, transform) where transform is (permute, reshape) or None
        
    Raises:
        ValueError: If no match or multiple matches found
    """
    subs = [
        (re.sub(pat, repl, source_key), reshape)
        for pat, (repl, reshape) in mapping.items()
        if re.match(pat, source_key)
    ]
    if len(subs) != 1:
        raise ValueError(f"Expected exactly one key match, got: {subs} for {source_key}")
    return subs[0]


def path_to_key(path) -> str:
    """Convert nnx path to string key for parameter lookup."""
    return ".".join(
        str(stoi(key.key if hasattr(key, "key") else key)) for key in path
    )


def apply_tensor_transform(tensor: jax.Array, transform: Optional[Tuple]) -> jax.Array:
    """Apply permutation and reshape transforms to tensor.
    
    Args:
        tensor: Input tensor
        transform: Tuple of (permute, reshape) or None
        
    Returns:
        Transformed tensor
    """
    if transform is None:
        return tensor
    
    permute, reshape = transform
    if permute:
        tensor = tensor.transpose(permute)
    if reshape:
        tensor = tensor.reshape(reshape)
    return tensor


def load_tensors_from_safetensors(
    file_dir: str,
    key_mapping: Dict[str, Tuple[str, Optional[Tuple]]],
) -> Dict[str, jax.Array]:
    """Load and transform tensors from safetensors files.
    
    Args:
        file_dir: Directory containing safetensors files
        key_mapping: Mapping from torch keys to (jax_key, transform) tuples
        
    Returns:
        Dictionary of loaded and transformed tensors
        
    Raises:
        ValueError: If no safetensors files found or duplicate keys detected
        RuntimeError: If tensor loading fails
    """
    files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
    
    if not files:
        raise ValueError(f"No safetensors found in {file_dir}")
    
    all_loaded_tensors = {}
    
    for f in files:
        with safetensors.safe_open(f, framework="numpy") as sf:
            for k_name in sf.keys():
                try:
                    v = sf.get_tensor(k_name)
                    jax_key_mapped, transform = torch_key_to_jax_key(key_mapping, k_name)
                    
                    # Apply transforms
                    v = apply_tensor_transform(jax.numpy.array(v), transform)
                    
                    if jax_key_mapped in all_loaded_tensors:
                        raise ValueError(
                            f"Duplicate key {jax_key_mapped} found in file {f.name}."
                        )
                    all_loaded_tensors[jax_key_mapped] = v
                    
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load tensor {k_name} from file {f.name}: {e}"
                    ) from e
    
    return all_loaded_tensors


def create_model_with_weights(
    model_class: type,
    config: ConfigT,
    loaded_tensors: Dict[str, jax.Array],
    mesh: Optional[jax.sharding.Mesh] = None,
) -> ModelT:
    """Create model instance and populate with loaded weights.
    
    Args:
        model_class: Model class to instantiate
        config: Model configuration object
        loaded_tensors: Dictionary of parameter tensors
        mesh: Optional JAX mesh for sharding
        
    Returns:
        Model instance with loaded weights
        
    Raises:
        ValueError: If tensor shapes don't match model expectations
    """
    # Create model structure
    model = nnx.eval_shape(lambda: model_class(config, rngs=nnx.Rngs(params=0)))
    graph_def, abs_state = nnx.split(model)
    state_dict = abs_state.to_pure_dict()
    
    # Setup sharding if provided
    if mesh is not None:
        sharding_dict = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
    else:
        sharding_dict = None
    
    # Update state dict with loaded tensors
    def update_tensor(path, param, shard=None):
        current_path_key = path_to_key(path)
        if current_path_key in loaded_tensors:
            loaded_arr = loaded_tensors[current_path_key]
            if loaded_arr.shape != param.shape:
                raise ValueError(
                    f"Shape mismatch for {current_path_key}: got"
                    f" {loaded_arr.shape}, expected {param.shape}"
                )
            device = shard if shard is not None else jax.devices()[0]
            return jax.device_put(loaded_arr, device)
        return param
    
    if sharding_dict is not None:
        state_dict = jax.tree.map_with_path(update_tensor, state_dict, sharding_dict)
    else:
        state_dict = jax.tree.map_with_path(update_tensor, state_dict)
    
    return nnx.merge(graph_def, state_dict)


def create_model_from_safetensors(
    file_dir: str,
    model_class: type,
    config: ConfigT,
    key_mapping: Dict[str, Tuple[str, Optional[Tuple]]],
    mesh: Optional[jax.sharding.Mesh] = None,
    preprocess_fn: Optional[Callable[[Dict[str, jax.Array]], Dict[str, jax.Array]]] = None,
) -> ModelT:
    """Complete pipeline to load model from safetensors files.
    
    Args:
        file_dir: Directory containing safetensors files
        model_class: Model class to instantiate
        config: Model configuration object
        key_mapping: Mapping from torch keys to (jax_key, transform) tuples
        mesh: Optional JAX mesh for sharding
        preprocess_fn: Optional function to preprocess loaded parameters
        
    Returns:
        Model instance with loaded weights
    """
    loaded_tensors = load_tensors_from_safetensors(
        file_dir=file_dir,
        key_mapping=key_mapping,
    )
    
    # Apply preprocessing if provided
    if preprocess_fn is not None:
        loaded_tensors = preprocess_fn(loaded_tensors)
    
    return create_model_with_weights(
        model_class=model_class,
        config=config,
        loaded_tensors=loaded_tensors,
        mesh=mesh,
    )