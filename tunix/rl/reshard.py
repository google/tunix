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
    in_data: jaxtyping.PyTree,
    x: jaxtyping.PyTree,
    success: Callable[[float], None],
    failure: Callable[[float, Exception], None],
):
  """Callback to invoke when the Jax array is ready."""
  fut = futures.Future()
  start_time = [-1.0]

  def callback(f):
    e = f.exception()
    if e is None:
      success(start_time[0])
    else:
      failure(start_time[0], e)

  fut.add_done_callback(callback)

  def wait():
    try:
      jax.block_until_ready(in_data)
      start_time[0] = time.perf_counter()
      jax.block_until_ready(x)
    except Exception as e:  # pylint: disable=broad-exception-caught
      fut.set_exception(e)
    else:
      fut.set_result((x))

  threading.Thread(target=wait).start()


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

      # TODO(b/476149699): Migrate to new API once it's verified.
      if use_experimental_pre_reshard:
        in_sharding = jax.tree_util.tree_map(
            lambda x: x.sharding,
            x,
        )
        return (
            experimental_reshard.sidechannel_reshard_with_intermediate_sharding(
                x,
                in_sharding,
                sharding,
                donate=donate,
                cache_resharding_plans=cache_resharding_plans,
            )
        )
      else:
        return experimental_reshard.reshard(
            x,
            sharding,
            donate=donate,
            may_alias=None,
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

  start = time.perf_counter()

  resharded_array = reshard_fn(source, dst_shardings)

  callback_on_ready(
      source,
      resharded_array,
      lambda reshard_start: logging.info(
          'Input ready in %.2fs. Reshard finished in %.2fs',
          reshard_start - start,
          time.perf_counter() - reshard_start,
      ),
      lambda reshard_start, e: logging.error(
          'Input ready in %.2fs. Reshard failed in %.2fs: %s',
          reshard_start - start,
          time.perf_counter() - reshard_start,
          e,
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
