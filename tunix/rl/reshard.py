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
# Keep this import for google internal usage.
import math  # pylint: disable=unused-import
import os
import threading
import time
from typing import Any, Callable

from absl import logging
import jax
import jaxtyping
from flax import nnx
from tunix.rl import utils

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
  if not isinstance(
      source_sharding, jax.sharding.NamedSharding
  ) or not isinstance(target_sharding, jax.sharding.NamedSharding):
    logging.vlog(
        2,
        'None-NamedSharding does not need intermediate sharding.'
        f' {source_sharding=}, {target_sharding=}',
    )
    return None
  src_mesh = source_sharding.mesh
  dst_mesh = target_sharding.mesh

  def _get_sharding_dims(sharding, mesh):
    sharding_dims = {}
    for i, axis_name in enumerate(sharding.spec):
      if axis_name is None:
        sharding_dims[(i, None)] = 1
      elif isinstance(axis_name, str):
        sharding_dims[(i, mesh.axis_names.index(axis_name))] = mesh.shape[
            axis_name
        ]
      elif isinstance(axis_name, (tuple, list)):
        for _, axis in enumerate(axis_name):
          if axis is None:
            sharding_dims[(i, None)] = 1
          # Only handles two-level logical axis rules for now.
          elif isinstance(axis, str):
            sharding_dims[(i, mesh.axis_names.index(axis))] = (
                mesh.shape[axis]
            )
          else:
            raise ValueError(f'Unsupported axis name: {axis_name}')
      else:
        raise ValueError(
            f'Unsupported axis name: {axis_name} with type {type(axis_name)}'
        )

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
    logging.debug(
        'Resharding with undividable shardings is not optimized with'
        ' experimental pre-reshard.'
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

  to_split_src_pytree_leaves = {}
  to_split_src_pytree_leaves_indexes = {}
  to_split_intermediate_sharding_leaves = {}

  intermediate_mesh = None
  to_update_src_pytree_leaves = []

  for i, (path, src) in enumerate(src_leaves_with_path):
    to_update_src_pytree_leaves.append(src)
    if intermediate_sharding := intermediate_sharding_leaves_with_path.get(
        path, None
    ):
      # The to_split_axis should always be the same along all the intermediate
      # shardings.
      intermediate_mesh = intermediate_sharding.mesh
      to_split_src_pytree_leaves.setdefault(intermediate_mesh, []).append(src)
      to_split_src_pytree_leaves_indexes.setdefault(intermediate_mesh, []).append(i)
      to_split_intermediate_sharding_leaves.setdefault(intermediate_mesh, []).append(intermediate_sharding)

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

  for key in to_split_src_pytree_leaves.keys():
    temp_source = jax.jit(
        _identity,
        out_shardings=to_split_intermediate_sharding_leaves[key],
    )(to_split_src_pytree_leaves[key])

  # Update the to_split_src_pytree_leaves with the new splitted array.
    updated_to_split_src_pytree_leaves, *_ = splitfn(temp_source, to_split_axis)

    for i in range(len(to_split_src_pytree_leaves_indexes[key])):
      to_update_src_pytree_leaves[to_split_src_pytree_leaves_indexes[key][i]] = (
          updated_to_split_src_pytree_leaves[i]
      )

  updated_src_pytree = jax.tree_util.tree_unflatten(
      src_treedef, to_update_src_pytree_leaves
  )
  return updated_src_pytree


#


def _get_reshard_fn_pathwaysutils(
    *,
    cache_resharding_plans: bool,
    donate: bool,
    use_experimental_pre_reshard: bool,
):
  """Returns a reshard function using pathwaysutils.

  Args:
    cache_resharding_plans: Whether to cache resharding plans.
    donate: Whether to donate the input buffer.
    use_experimental_pre_reshard: Ignored.

  Returns:
    A reshard function.
  """
  # This import is expected to fail sometimes internally if pathwaysutils is
  # not linked to the binary.
  try:
    from pathwaysutils.experimental import reshard as experimental_reshard  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
    from pathwaysutils.experimental import split_by_mesh_axis  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
    from pathwaysutils import jax as pw_jax  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
  except ImportError:
    logging.info(
        'Cannot import PathwaysUtils and experimental reshard API.'
    )
    raise
  else:
    if 'proxy' not in os.getenv('JAX_PLATFORMS', ''):
      raise EnvironmentError(
          'Pathways proxy is not available. Make sure you have enabled Pathways'
          ' proxy as jax backend, e.g. os.environ["JAX_PLATFORMS"] = "proxy".'
      )

    def reshard_fn(
        x: Any,
        sharding: jax.sharding.Sharding | Any,
    ):

      if use_experimental_pre_reshard:
        try:
          # This will raise an ImportError if the API is not available.
          pw_jax.jaxlib_pathways._split_by_mesh_axis  # pylint: disable=protected-access
        except ImportError:
          logging.debug(
              'split_by_mesh_axis is not available until JAX 0.8.0. Skipping'
              ' pre-reshard.'
          )
        else:
          x = _experimental_pre_reshard(
              split_by_mesh_axis.split_by_mesh_axis, x, sharding
          )


      return experimental_reshard.reshard(
          x,
          sharding,
          donate=donate,
          cache_resharding_plans=cache_resharding_plans,
      )

  return reshard_fn


def _get_reshard_fn_jax_device_put(
    *,
    donate: bool,
    cache_resharding_plans: bool = False,  # pylint: disable=unused-argument
    use_experimental_pre_reshard: bool = False,  # pylint: disable=unused-argument
):
  return functools.partial(
      jax.device_put,
      donate=donate,
  )


def _get_reshard_fn(
    cache_resharding_plans: bool,
    donate: bool,
    use_experimental_pre_reshard: bool,
    get_reshard_fns: list[Callable[..., Any]],
):
  """Returns a reshard function.

  Args:
    cache_resharding_plans: Whether to cache resharding plans.
    donate: Whether to donate the input buffer.
    use_experimental_pre_reshard: Whether to use experimental pre-reshard.
    get_reshard_fns: A list of reshard functions to try to use.

  Returns:
    A reshard function.
  """
  for get_reshard_fn in get_reshard_fns:
    try:
      reshard_fn = get_reshard_fn(
          cache_resharding_plans=cache_resharding_plans,
          donate=donate,
          use_experimental_pre_reshard=use_experimental_pre_reshard,
      )
    except (ImportError, EnvironmentError):
      logging.debug('Could not support {get_reshard_fn=}.', exc_info=True)
    else:
      return reshard_fn

  raise ValueError('Could not find a reshard function from {get_reshard_fns=}.')


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
      sharding information. This can be a pytree containing jax.Array or
      jax.sharding.NamedSharding.
    cache_plan: Whether to cache the resharding plan. This can largely speed up
      the resharding process. Turn off with caution.
    donate_input: Whether to donate the input (source) to the reshard.
    use_experimental_pre_reshard: Whether to use the experimental pre-reshard
      API.

  Returns:
    The resharded pytree.
  """

  def _get_dst_sharding(x):
    if isinstance(
        x, jax.sharding.NamedSharding | jax.sharding.SingleDeviceSharding
    ):
      return x
    else:
      return jax.sharding.NamedSharding(
          x.sharding.mesh,
          x.sharding.spec,
          memory_kind=x.sharding.memory_kind,
      )

  dst_shardings = jax.tree_util.tree_map(
      _get_dst_sharding,
      target,
  )

  reshard_fn = _get_reshard_fn(
      cache_resharding_plans=cache_plan,
      donate=donate_input,
      use_experimental_pre_reshard=use_experimental_pre_reshard,
      get_reshard_fns=[
          #
          _get_reshard_fn_pathwaysutils,
          _get_reshard_fn_jax_device_put,
      ],
  )

  start = time.time()

  resharded_array = reshard_fn(source, dst_shardings)

  callback_on_ready(
      resharded_array,
      lambda: logging.info('Reshard finished in %.2fs', time.time() - start),
      lambda e: logging.error(
          'Reshard failed in %.2fs: %s', time.time() - start, e
      ),
  )
  return resharded_array


def reshard_model_to_mesh(model: nnx.Module, mesh: jax.sharding.Mesh):
  """Reshard the lora model if the mesh is specified and the lora model mesh is not the same as the input mesh."""
  model_mesh = utils.get_pytree_mesh_info(nnx.state(model))
  if mesh is not None and model_mesh != mesh:
    with mesh:
      graph_def, state = nnx.split(model)
      default_memory_kind = jax.devices()[0].default_memory().kind
      dst_shardings = jax.tree_util.tree_map(
          lambda x: jax.sharding.NamedSharding(
              mesh,
              x,
              memory_kind=default_memory_kind,
          ),
          nnx.get_partition_spec(state),
      )
      model = nnx.merge(graph_def, reshard_pytree(state, dst_shardings))
  return model
