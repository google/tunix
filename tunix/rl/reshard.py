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


#


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
      the resharding process. Turn off with cautious.
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

  start = time.time()

  reshardfn = None

  #

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
