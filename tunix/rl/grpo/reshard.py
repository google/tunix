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

"""Resharding functions."""

from concurrent import futures
import functools
import math
import threading
import time
from typing import Callable
from absl import logging
import jax
import jaxtyping


# TODO(tsbao): move this to util
def callback_on_ready(
    x: jaxtyping.PyTree,
    success: Callable[[], None],
    failure: Callable[[Exception], None],
):
  """Callback to invoke when the Jax array is ready."""
  fut = futures.Future()

  def callback(f):
    e = f.exception()
    if e is None:
      success()
    else:
      failure(e)

  fut.add_done_callback(callback)

  def wait():
    try:
      jax.block_until_ready(x)
    except Exception as e:  # pylint: disable=broad-exception-caught
      fut.set_exception(e)
    else:
      fut.set_result(x)

  threading.Thread(target=wait).start()


# BEGIN GOOGLE-INTERNAL
# TODO(linchai): OSS it once split_by_mesh_axis is available in pathwaysutils.
def _identity(x):
  return x


INTERMEDIATE_SPLIT_SUFFIX = '_intermediate_split'
INTERMEDIATE_REPLICA_SUFFIX = '_intermediate_replica'


def _maybe_find_intermediate_sharding(source_sharding, target_sharding):
  """Maybe finds an intermediate sharding to reshard to before target sharding.

  This function tries to find an intermediate sharding that can be used to
  reshard the source sharding to the target sharding. This is useful when
  resharding from a source sharding to a target sharding that requires an
  all-gather, which can be expensive.

  For example, consider resharding an array from src_sharding (e.g., [fsdp: 8,
  tp: 1]) to target_sharding (e.g., [fsdp: 1, tp: 4]). In this case, the source
  has a larger sharding factor (8) than the target largest sharding factor (4)
  on the tp dimension.
  To avoid an expensive all-gather, we can introduce an intermediate sharding
  (e.g., [fsdp_split: 4, fsdp_replica: 2, tp: 1]). This intermediate sharding
  allows us to reshard the source array by still sharding along the fsdp
  dimension and replicating it on the remaining devices. Then we can just
  reshard any replica of the source to the target as normal.

  Args:
    source_sharding: The source sharding.
    target_sharding: The target sharding.

  Returns:
    An intermediate sharding, or None if no intermediate sharding can be found.
  """
  src_mesh = source_sharding.mesh
  dst_mesh = target_sharding.mesh

  def _get_sharding_dims(sharding, mesh):
    sharding_dims = {}
    for i, axis_name in enumerate(sharding.spec):
      if axis_name is None:
        sharding_dims[(i, None)] = 1
      else:
        sharding_dims[(i, mesh.axis_names.index(axis_name))] = mesh.shape[
            axis_name
        ]

    largest_shards = max(sharding_dims.values()) if len(sharding_dims) else 1
    if len(sharding_dims) < len(mesh.shape):
      for mi, mesh_axis in enumerate(mesh.axis_names):
        matched = any(mesh_axis == keys[1] for keys in sharding_dims)
        if not matched:
          sharding_dims[(None, mi)] = 1
    return sharding_dims, largest_shards

  src_sharding_dims, src_largest_shards = _get_sharding_dims(
      source_sharding, src_mesh
  )
  dst_sharding_dims, dst_largest_shards = _get_sharding_dims(
      target_sharding, dst_mesh
  )
  # Not able to handle resharding with undividable shardings.
  if src_largest_shards % dst_largest_shards != 0:
    logging.warning(
        'Resharding with undividable shardings is not supported.'
        ' source_sharding=%s, target_sharding=%s',
        source_sharding,
        target_sharding,
    )
    return None

  total_source_sharding_dims = math.prod(list(src_sharding_dims.values()))
  total_dst_sharding_dims = math.prod(list(dst_sharding_dims.values()))
  if (
      total_source_sharding_dims <= total_dst_sharding_dims
      or total_source_sharding_dims % total_dst_sharding_dims != 0
  ):
    return None

  new_split_dim_shards = None
  new_split_axis = None
  replicas = src_largest_shards // dst_largest_shards

  # Find gcd(src_dim_shards, dst_dim_shards),
  # If all of them are 1s, an all-gather is needed as the single replica of
  # the source cannot be presented by any sharded form on the target devices.
  gcd_shards = []
  for (sharding_mesh_axis_idx, src_dim_shards), (_, dst_dim_shards) in zip(
      src_sharding_dims.items(), dst_sharding_dims.items()
  ):
    gcd_dim_shards = math.gcd(src_dim_shards, dst_dim_shards)
    if gcd_dim_shards == 1:
      if (
          src_dim_shards > dst_dim_shards
          and src_dim_shards == src_largest_shards
      ):
        new_split_axis = sharding_mesh_axis_idx
        new_split_dim_shards = (src_dim_shards // replicas, replicas)
    gcd_shards.append(gcd_dim_shards)

  if math.prod(gcd_shards) != 1 or new_split_axis is None:
    return None

  # Generate the intermediate sharding.
  new_split_mesh_axis_name = (
      src_mesh.axis_names[new_split_axis[1]] + INTERMEDIATE_SPLIT_SUFFIX
  )
  new_split_mesh_replica_axis_name = (
      src_mesh.axis_names[new_split_axis[1]] + INTERMEDIATE_REPLICA_SUFFIX
  )
  intermediate_mesh = jax.sharding.Mesh(
      src_mesh.devices.reshape(
          tuple(
              list(src_mesh.devices.shape[: new_split_axis[1]])
              + [new_split_dim_shards[0], new_split_dim_shards[1]]
              + list(src_mesh.devices.shape[new_split_axis[1] + 1 :])
          )
      ),
      axis_names=tuple(
          list(src_mesh.axis_names[: new_split_axis[1]])
          + [new_split_mesh_axis_name, new_split_mesh_replica_axis_name]
          + list(src_mesh.axis_names[new_split_axis[1] + 1 :])
      ),
  )

  intermediate_spec = tuple(
      list(source_sharding.spec[: new_split_axis[0]])
      + [new_split_mesh_axis_name]
      + list(source_sharding.spec[new_split_axis[0] + 1 :])
  )
  intermediate_sharding = jax.sharding.NamedSharding(
      intermediate_mesh,
      jax.sharding.PartitionSpec(*intermediate_spec),
      memory_kind=source_sharding.memory_kind,
  )
  return intermediate_sharding


# TODO(linchai): add donate_input support.
def _experimental_pre_reshard(splitfn, src_pytree, target_shardings):
  """Simple heuristic to determine if resharding with replicated all-gather is needed.

  A replicated all-gather often results to heavy HBM occupation which we need to
  avoid. This funciton currently only handles the case like resharding from
  [fsdp: 8, tp: 1] to [fsdp: 1, tp: 4].
  We will improve the coverage on more complex cases along the development.

  Args:
    splitfn: The split function.
    src_pytree: The source jax Array.
    target_shardings: The target sharding.

  Returns:
    Pre-resharded src_pytree.
  """
  src_shardings = jax.tree_util.tree_map(
      lambda x: x.sharding,
      src_pytree,
  )
  intermediate_shardings = jax.tree_util.tree_map(
      _maybe_find_intermediate_sharding,
      src_shardings,
      target_shardings,
  )

  src_leaves_with_path, src_treedef = jax.tree_util.tree_flatten_with_path(
      src_pytree
  )
  intermediate_sharding_leaves_with_path, _ = (
      jax.tree_util.tree_flatten_with_path(intermediate_shardings)
  )
  intermediate_sharding_leaves_with_path = {
      path: intermediate_sharding
      for path, intermediate_sharding in intermediate_sharding_leaves_with_path
  }

  to_split_src_pytree_leaves = []
  to_split_src_pytree_leaves_indexes = []
  to_split_intermediate_sharding_leaves = []

  intermediate_mesh = None
  to_update_src_pytree_leaves = []

  for i, (path, src) in enumerate(src_leaves_with_path):
    to_update_src_pytree_leaves.append(src)
    if intermediate_sharding := intermediate_sharding_leaves_with_path.get(
        path, None
    ):
      # The to_split_axis should always be the same along all the intermediate
      # shardings.
      if intermediate_mesh is None:
        intermediate_mesh = intermediate_sharding.mesh
      to_split_src_pytree_leaves.append(src)
      to_split_src_pytree_leaves_indexes.append(i)
      to_split_intermediate_sharding_leaves.append(intermediate_sharding)

  if intermediate_mesh is None:
    # No pre-resharding is needed.
    return src_pytree

  to_split_axis = None
  for axis_name in intermediate_mesh.axis_names:
    if axis_name.endswith(INTERMEDIATE_REPLICA_SUFFIX):
      to_split_axis = axis_name
      break
  assert (
      to_split_axis is not None
  ), f'No replica axis found in the intermediate mesh {intermediate_mesh}.'

  temp_source = jax.jit(
      _identity,
      out_shardings=to_split_intermediate_sharding_leaves,
  )(to_split_src_pytree_leaves)

  # Update the to_split_src_pytree_leaves with the new splitted array.
  to_split_src_pytree_leaves, *_ = splitfn(temp_source, to_split_axis)

  for i in range(len(to_split_src_pytree_leaves_indexes)):
    to_update_src_pytree_leaves[to_split_src_pytree_leaves_indexes[i]] = (
        to_split_src_pytree_leaves[i]
    )
  updated_src_pytree = jax.tree_util.tree_unflatten(
      src_treedef, to_update_src_pytree_leaves
  )
  return updated_src_pytree


# END GOOGLE-INTERNAL


def reshard_pytree(
    source: jaxtyping.PyTree,
    target: jaxtyping.PyTree,
    cache_plan: bool = True,
    donate_input: bool = False,
    use_experimental_pre_reshard: bool = True,
) -> jaxtyping.PyTree:
  """Reshard input pytree from source sharding and mesh to target sharding and mesh.

  From source to target, both the sharding and mesh can be different.

  Args:
    source: The input source pytree to reshard.
    target: The target pytree to reshard to. Contains target mesh and named
      sharding information.
    cache_plan: Whether to cache the resharding plan. This can largely speed up
      the resharding process. Turn off with cautious.
    donate_input: Whether to donate the input (source) to the reshard.
    use_experimental_pre_reshard: Whether to use the experimental pre-reshard
      API.

  Returns:
    The resharded pytree.
  """

  dst_shardings = jax.tree_util.tree_map(
      lambda x: jax.sharding.NamedSharding(
          x.sharding.mesh,
          x.sharding.spec,
          memory_kind=x.sharding.memory_kind,
      ),
      target,
  )

  start = time.time()

  reshardfn = None

  # BEGIN GOOGLE-INTERNAL
  try:
    from google3.learning.pathways.jax.ifrt import client as ifrt_client  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

    if use_experimental_pre_reshard:
      updated_source = _experimental_pre_reshard(
          ifrt_client.split_by_mesh_axis, source, dst_shardings
      )
      reshardfn = functools.partial(
          ifrt_client.experimental_reshard, x=updated_source
      )
    else:
      reshardfn = functools.partial(ifrt_client.experimental_reshard, x=source)
  except ImportError:
    logging.error(
        'Cannot import Pathways client. Make sure you have'
        ' linked //learning/pathways/jax/ifrt:client to your runtime binary.'
    )
  # END GOOGLE-INTERNAL

  # Do not remove this check. It's used in google internally.
  if reshardfn is None:
    try:
      import pathwaysutils  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
      from pathwaysutils.experimental import reshard as experimental_reshard  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

      reshardfn = functools.partial(experimental_reshard.reshard, x=source)
      # No-op if already initialized.
      pathwaysutils.initialize()
    except ImportError:
      logging.error(
          'Cannot import PathwaysUtils and experimental reshard API. Make sure'
          ' //third_party/py/pathwaysutils/experimental:reshard is linked to'
          ' your binary.'
      )
  if reshardfn is None:
    logging.info('No resharding API is available. Fallback to device_put.')
    resharded_array = jax.device_put(source, dst_shardings)
  else:
    resharded_array = reshardfn(
        sharding=dst_shardings,
        donate_input=donate_input,
        cache_resharding_plans=cache_plan,
    )

  callback_on_ready(
      resharded_array,
      lambda: logging.info('Reshard finished in %.2fs', time.time() - start),
      lambda e: logging.error(
          'Reshard failed in %.2fs: %s', time.time() - start, e
      ),
  )
  return resharded_array
