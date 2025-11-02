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

"""Utilities for creating randomly initialized model weights.

This mirrors the shape/sharding handling of `safetensors_loader.load_and_create_model`
but generates random parameters instead of loading them from files.
"""

import contextlib

from flax import nnx
import jax
import jax.numpy as jnp


from line_profiler import profile

@profile
def create_dummy_model(
    model_class,
    config,
    mesh=None,
    dtype: jnp.dtype | None = None,
    random_seed: int = 0,
    scale: float = 0.02,
):
  """Create a model with random-initialized parameters.

  Args:
    model_class: Model class to instantiate.
    config: Model configuration.
    mesh: Optional JAX mesh or mesh-like context for sharding.
    dtype: Optional dtype for parameter initialization.
    random_seed: RNG seed for initialization.
    scale: Scaling factor applied to the random normal values.

  Returns:
    Model instance with randomly initialized weights (sharded if a mesh is provided).
  """
  context_manager = mesh if mesh is not None else contextlib.nullcontext()

  with context_manager:
    # Build abstract model to obtain param shapes without allocating full tensors.
    abs_model = nnx.eval_shape(lambda: model_class(config, rngs=nnx.Rngs(params=0)))

  graph_def, abs_state = nnx.split(abs_model)

  state_dict = abs_state.to_pure_dict()
  if mesh is not None:
    sharding_dict = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
  else:
    sharding_dict = None

  rngs = nnx.Rngs(random_seed)

  @profile
  def make_random_init_fn(rngs, scale, dtype):
    @profile
    def init_fn(path, param, shard=None):
      arr = scale * rngs.params.normal(param.shape, dtype)
      if shard is not None:
        return jax.device_put(arr, shard)
      return arr

    return init_fn

  random_init_fn = make_random_init_fn(rngs, scale, dtype)

  if sharding_dict is not None:
    state_dict = jax.tree.map_with_path(random_init_fn, state_dict, sharding_dict)
  else:
    state_dict = jax.tree.map_with_path(random_init_fn, state_dict)

  return nnx.merge(graph_def, state_dict)
