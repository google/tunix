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
import numpy as np
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
    from pathwaysutils.experimental import reshard as experimental_reshard  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]
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


def _env_flag(name: str, default: str = '0') -> bool:
  """True iff the environment variable `name` is set to 1/true/True."""
  return (os.environ.get(name, default) or default).strip() in (
      '1',
      'true',
      'True',
  )


# Multi-controller (jax.distributed, no Pathways) trainer -> in-process engine
# weight publish. Enabled by TUNIX_MULTI_CONTROLLER_RESHARD=1.
_MC_REPLICATE_FNS: dict[Any, Any] = {}
_MC_STATE = {'barrier_done': False, 'logged': False}


def _multicontroller_pre_reshard_barrier() -> None:
  """Synchronizes all processes once before the first multi-controller publish.

  The first trainer -> engine publish is the first collective that every
  process must join after engine start-up. Hosts reach it with a start-up
  spread wider than the runtime settle window, and a late peer could abort
  the collective for everyone ("continuator has halted" on cold boots). A
  device barrier plus a tiny all-gather first makes every host confirm it is
  up and warms the collective path. One-time and best-effort; disable with
  TUNIX_PRE_RESHARD_BARRIER=0.
  """
  if _MC_STATE['barrier_done']:
    return
  _MC_STATE['barrier_done'] = True
  if not _env_flag('TUNIX_PRE_RESHARD_BARRIER', '1'):
    return
  if jax.process_count() <= 1:
    return
  from jax.experimental import multihost_utils  # pylint: disable=g-import-not-at-top

  try:
    multihost_utils.sync_global_devices('tunix_mc_pre_reshard')
    gathered = multihost_utils.process_allgather(
        np.asarray([jax.process_index()], dtype=np.int32)
    )
    np.asarray(gathered)
    multihost_utils.sync_global_devices('tunix_mc_pre_reshard_done')
    logging.info(
        'Multi-controller pre-reshard barrier passed on %d processes.',
        jax.process_count(),
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning('Multi-controller pre-reshard barrier failed: %s', e)


def _multicontroller_reshard_leaf(
    x: Any, sharding: jax.sharding.Sharding, donate: bool = False
) -> Any:
  """Moves one trainer leaf onto an engine sharding of a subset of its devices.

  Colocated multi-controller layout: the trainer array is sharded over every
  device of the job, while each process's in-process engine holds its copy on
  that process's local devices only. The source is therefore not fully
  addressable from any process and `jax.device_put` refuses ("Cannot reshard
  an input that is not fully addressable"). Instead every process runs the
  same jitted identity whose input sharding is the source sharding and whose
  output is replicated over the source mesh; XLA emits the all-gather over
  ICI, after which each process copies its local replica onto the engine
  devices (device to device, no host round-trip). The values are bit-exact;
  only the transport changes.

  Args:
    x: The source leaf.
    sharding: The target sharding.
    donate: Forwarded to `jax.device_put` on the paths that use it.

  Returns:
    The leaf placed on `sharding`.
  """
  if not isinstance(x, jax.Array) or x.is_fully_addressable:
    return jax.device_put(x, sharding, donate=donate)
  src_sharding = x.sharding
  src_devices = src_sharding.device_set
  dst_devices = sharding.device_set
  if not (
      isinstance(src_sharding, jax.sharding.NamedSharding)
      and dst_devices
      and dst_devices < src_devices
  ):
    # Same device set (handled by device_put) or a layout this path does not
    # cover; device_put raises a descriptive error in the latter case.
    return jax.device_put(x, sharding, donate=donate)
  key = (src_sharding, x.shape, str(x.dtype))
  replicate = _MC_REPLICATE_FNS.get(key)
  if replicate is None:
    replicate = jax.jit(
        lambda a: a,
        in_shardings=src_sharding,
        out_shardings=jax.sharding.NamedSharding(
            src_sharding.mesh, jax.sharding.PartitionSpec()
        ),
    )
    _MC_REPLICATE_FNS[key] = replicate
  replicated = replicate(x)
  # Bound the transient replicated copies to one leaf at a time.
  jax.block_until_ready(replicated)
  return jax.device_put(replicated.addressable_data(0), sharding)


def _get_reshard_fn_multicontroller(
    *,
    cache_resharding_plans: bool,
    donate: bool,
    use_experimental_pre_reshard: bool,
):
  """Returns the multi-controller reshard function.

  Enabled by TUNIX_MULTI_CONTROLLER_RESHARD=1.

  Args:
    cache_resharding_plans: Ignored; the jitted transfers are always cached.
    donate: Forwarded to `jax.device_put` on the paths that use it.
    use_experimental_pre_reshard: Ignored.

  Raises:
    EnvironmentError: The flag is off or the job has a single process, so the
      next reshard function in the chain is used.
  """
  del cache_resharding_plans, use_experimental_pre_reshard
  if not _env_flag('TUNIX_MULTI_CONTROLLER_RESHARD'):
    raise EnvironmentError('TUNIX_MULTI_CONTROLLER_RESHARD is not set.')
  if jax.process_count() <= 1:
    raise EnvironmentError('Single-process job; no multi-controller reshard.')

  def reshard_fn(x: Any, sharding: Any):
    _multicontroller_pre_reshard_barrier()
    if not _MC_STATE['logged']:
      _MC_STATE['logged'] = True
      logging.info(
          'TUNIX_MULTI_CONTROLLER_RESHARD: publishing weights with the'
          ' multi-controller transfer on %d processes.',
          jax.process_count(),
      )
    return jax.tree.map(
        functools.partial(_multicontroller_reshard_leaf, donate=donate),
        x,
        sharding,
    )

  return reshard_fn


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
          _get_reshard_fn_multicontroller,
          _get_reshard_fn_jax_device_put,
      ],
  )

  start = time.time()
  def _is_prng_key(x):
    """Check if an array is a JAX PRNG key."""
    return hasattr(x, 'dtype') and jax.dtypes.issubdtype(
        x.dtype, jax.dtypes.prng_key
    )

  impl_tree = jax.tree.map(
      lambda x: jax.random.key_impl(x) if _is_prng_key(x) else None,
      source,
  )
  unwrapped_tree = jax.tree.map(
      lambda x: jax.random.key_data(x) if _is_prng_key(x) else x,
      source,
  )

  resharded_tree = reshard_fn(
      unwrapped_tree,
      dst_shardings,
  )

  resharded_array = jax.tree.map(
      lambda val, impl: (
          jax.random.wrap_key_data(val, impl=impl) if impl is not None else val
      ),
      resharded_tree,
      impl_tree,
  )

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
