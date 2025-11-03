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
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx


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

  @partial(nnx.jit, static_argnums=(2, 3,))
  def make_param(rngs, scale, shape, dt):
    return scale * rngs.params.normal(shape, dt)

  def make_random_tensor(path, param, shard=None):
    shape = param.shape
    dt = dtype or getattr(param, "dtype", None) or jnp.float32

    if shard is None:
      return make_param(rngs, scale, shape, dt)
    else:
      shard_shape = shard.shard_shape(shape)

      def _callback(index):
        return make_param(rngs, scale, shard_shape, dt)

      return jax.make_array_from_callback(shape, shard, _callback)

  if sharding_dict is not None:
    state_dict = jax.tree.map_with_path(make_random_tensor, state_dict, sharding_dict)
  else:
    state_dict = jax.tree.map_with_path(make_random_tensor, state_dict)

  return nnx.merge(graph_def, state_dict)
