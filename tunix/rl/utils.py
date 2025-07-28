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

"""Simple utils used by GRPO."""

from flax import nnx
from flax.nnx import statelib
import jax
import jaxtyping
import numpy as np

Mesh = jax.sharding.Mesh
NamedSharding = jax.sharding.NamedSharding


def to_flat_dict(
    tree: jaxtyping.PyTree | statelib.State,
) -> tuple[dict[tuple[str, ...], jaxtyping.Array], jaxtyping.PyTreeDef]:
  if isinstance(tree, statelib.State):
    tree = nnx.to_pure_dict(tree)
  flattened, tree_def = jax.tree.flatten_with_path(tree)
  return {tuple(k.key for k in keys): v for keys, v in flattened}, tree_def


def get_pytree_mesh_info(tree: jaxtyping.PyTree) -> Mesh | None:
  """Returns the mesh info for the pytree."""
  mesh_info = set()

  def _get_mesh_info(leaf: jaxtyping.PyTree):
    if isinstance(leaf, jax.Array):
      if hasattr(leaf, 'sharding') and leaf.sharding:
        sharding = leaf.sharding
        if isinstance(sharding, NamedSharding):
          mesh_info.add(sharding.mesh)
    return leaf

  jax.tree_util.tree_map(_get_mesh_info, tree)
  if len(mesh_info) > 1:
    raise ValueError(
        f'All leaves of the pytree must have the same mesh. Found: {mesh_info}'
    )
  return mesh_info.pop() if mesh_info else None


def is_sharing_weights(
    m1: nnx.Module,
    m2: nnx.Module,
) -> bool:
  """Returns whether two models are sharing same copy of weights."""
  s1 = nnx.state(m1)
  s2 = nnx.state(m2)
  return np.all(
      jax.tree.map(
          lambda x, y: x is y,
          jax.tree_util.tree_leaves(s1),
          jax.tree_util.tree_leaves(s2),
      )
  )
