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

"""Simple utils used by RL algorithms."""

import collections
import gc
import operator
from typing import Any, List, Optional, Tuple

from absl import logging
from flax import nnx
from flax.nnx import filterlib
from flax.nnx import statelib
import jax
import jaxtyping
import numpy as np
from tunix.oss import utils


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
      if hasattr(leaf, "sharding") and leaf.sharding:
        sharding = leaf.sharding
        if isinstance(sharding, NamedSharding):
          mesh_info.add(sharding.mesh)
    return leaf

  jax.tree_util.tree_map(_get_mesh_info, tree)
  if len(mesh_info) > 1:
    raise ValueError(
        f"All leaves of the pytree must have the same mesh. Found: {mesh_info}"
    )
  return mesh_info.pop() if mesh_info else None


def _is_same_state(s1: jaxtyping.PyTree, s2: jaxtyping.PyTree) -> bool:
  """Returns whether two states refer to the same Params."""
  return np.all(
      jax.tree.map(
          lambda x, y: x is y,
          jax.tree_util.tree_leaves(s1),
          jax.tree_util.tree_leaves(s2),
      )
  )


def is_sharing_weights(
    m1: Optional[nnx.Module],
    m2: Optional[nnx.Module],
) -> bool:
  """Returns whether two models are sharing same copy of weights."""
  if m1 is None or m2 is None:
    return False

  s1 = nnx.state(m1)
  s2 = nnx.state(m2)
  return _is_same_state(s1, s2)


def is_sharing_backbone(
    m1: nnx.Module,
    m2: nnx.Module,
) -> bool:
  """Returns whether two models are sharing same copy of backbone."""
  s1 = nnx.state(m1, filterlib.Not(nnx.LoRAParam))
  s2 = nnx.state(m2, filterlib.Not(nnx.LoRAParam))
  return _is_same_state(s1, s2)


def pathways_hbm_usage_gb(devices: Any) -> List[Tuple[float, Optional[float]]]:
  """Returns the HBM usage for each device when using Pathways.

  Args:
    devices: The devices to get the HBM usage for.

  Returns:
    A list of tuples, where each tuple contains the HBM usage and limit for a
    device.
  """
  live_arrays = jax.live_arrays()
  hbm_used = collections.defaultdict(int)
  # TODO(lancewang): Find a way to get the accurate hbm limit on Pathways.
  hbm_limit = None
  for array in live_arrays:
    assert hasattr(array, "sharding") and hasattr(
        array.sharding, "device_set"
    ), (
        "This function must not be called within jax tracer (e.g. jit, vmap,"
        " grad)"
    )
    for device in array.sharding.device_set:
      hbm_used[device] += (
          array.dtype.itemsize * array.size // len(array.sharding.device_set)
      )
  return [(hbm_used[device], hbm_limit) for device in devices]


def jax_hbm_usage_gb(devices: Any) -> List[Tuple[float, float]]:
  hbm_used = []
  for d in devices:
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    hbm_used.append((used, limit))
  return hbm_used


def show_hbm_usage(title=""):
  """Prints the current HBM usage.

  Args:
    title: The title to print before the HBM usage.
  """
  fmt_size = utils.humanize_binary_size
  devices = jax.devices()
  # Force a GC sweep to catch recently deallocated arrays
  gc.collect()

  if utils.pathways_available():
    logging.info("%s - Using Pathways compatible HBM stats collector", title)
    hbm_stats = pathways_hbm_usage_gb(devices)
    for i, (used, _) in enumerate(hbm_stats):
      logging.info("Using %s on %s", fmt_size(used), devices[i])
  else:
    logging.info(
        "%s - Pathways not available. Using defaultHBM stats collector", title
    )
    hbm_stats = jax_hbm_usage_gb(devices)

    for i, (used, limit) in enumerate(hbm_stats):
      logging.info(
          "Using %s / %s (%s) on %s",
          fmt_size(used),
          fmt_size(limit),
          used / limit,
          devices[i],
      )


def put_params_on_memory_kind(
    params: jaxtyping.PyTree,
    memory_kind: str,
) -> jaxtyping.PyTree:
  """Puts params on the given memory kind."""
  assert memory_kind in [
      "device",
      "pinned_host",
      "unpinned_host",
  ], f"Unsupported memory kind: {memory_kind}"
  original_shardings = jax.tree.map(lambda x: x.sharding, params)
  logging.info("original_shardings: %s", original_shardings)
  is_on_device = jax.tree_util.tree_reduce(
      operator.or_,
      jax.tree.map(lambda x: x.memory_kind == "device", original_shardings),
  )
  if (is_on_device and memory_kind == "device") or (
      not is_on_device and memory_kind == "pinned_host"
  ):
    logging.info(
        "Params are already on the requested memory kind: %s", memory_kind
    )
    return params

  def _get_new_sharding(x):
    if isinstance(x, jax.NamedSharding):
      return jax.NamedSharding(x.mesh, x.spec, memory_kind=memory_kind)
    else:
      return x.with_memory_kind(memory_kind)

  new_shardings = jax.tree.map(_get_new_sharding, original_shardings)
  params_on_memory_kind = jax.device_put(
      params,
      new_shardings,
  )
  shardings = jax.tree.map(lambda x: x.sharding, params_on_memory_kind)
  logging.info("params_on_memory_kind shardings: %s", shardings)
  return params_on_memory_kind
