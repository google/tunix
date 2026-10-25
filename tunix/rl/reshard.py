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


# One jitted transfer for all leaves of a publish (TUNIX_BATCHED_PUBLISH).
_BP_STATE: dict[str, Any] = {'pub': 0, 'exe': {}, 'off': None, 'nosync': False}


def _batched_publish_mode() -> str:
  """'' (off), or the TUNIX_BATCHED_PUBLISH value ('1'/'true' or '2')."""
  mode = (os.environ.get('TUNIX_BATCHED_PUBLISH', '0') or '0').strip()
  return mode if mode in ('1', '2', 'true', 'True') else ''


def batched_publish_active() -> bool:
  """Whether TUNIX_BATCHED_PUBLISH can batch this job's publishes.

  True only where the batched path can actually run: the flag is on, the
  multi-controller reshard is on (TUNIX_MULTI_CONTROLLER_RESHARD=1), the job
  has more than one process, and no failed check has disabled batching
  (that verdict is agreed on by every process).

  Returns:
    Whether batched publishing is active.
  """
  return bool(
      _batched_publish_mode()
      and _env_flag('TUNIX_MULTI_CONTROLLER_RESHARD')
      and jax.process_count() > 1
      and not _BP_STATE['off']
  )


def _bp_should_log(pub: int) -> bool:
  return pub <= 3 or pub % 50 == 0


def _bp_vote(ok: bool) -> list[int]:
  """Every process's verdict (a collective when there are several)."""
  if jax.process_count() == 1:
    return [int(ok)]
  from jax.experimental import multihost_utils  # pylint: disable=g-import-not-at-top

  votes = multihost_utils.process_allgather(
      np.asarray([int(ok)], np.int32), tiled=False
  )
  return [int(v) for v in np.asarray(votes).reshape(-1)]


def _bp_decline(xs: list[Any], ts: list[Any]) -> str | None:
  """Why a leaf would not take the per-leaf bridge (global metadata only)."""
  if _BP_STATE['off']:
    return 'disabled(%s)' % _BP_STATE['off']
  if not xs:
    return 'empty'
  for x, s in zip(xs, ts):
    if not isinstance(x, jax.Array) or not isinstance(s, jax.sharding.Sharding):
      return 'non_array_or_non_sharding'
    if jax.dtypes.issubdtype(x.dtype, jax.dtypes.extended):
      return 'extended_dtype'
    if x.is_fully_addressable:
      return 'fully_addressable'
    if not isinstance(x.sharding, jax.sharding.NamedSharding):
      return 'src_not_named'
    if not (s.device_set and s.device_set < x.sharding.device_set):
      return 'not_bridge_topology'
  return None


def _bp_groups(xs: list[Any]) -> list[list[int]]:
  """Leaf indices grouped by source mesh and TUNIX_BATCHED_PUBLISH_GROUP_MB."""
  cap = (
      float(os.environ.get('TUNIX_BATCHED_PUBLISH_GROUP_MB', '2048') or '2048')
      * 2**20
  )
  groups, cur, cur_bytes, cur_mesh = [], [], 0, None
  for i, x in enumerate(xs):
    mesh = x.sharding.mesh
    if cur and (mesh != cur_mesh or cur_bytes + x.nbytes > cap):
      groups.append(cur)
      cur, cur_bytes = [], 0
    cur.append(i)
    cur_bytes += x.nbytes
    cur_mesh = mesh
  if cur:
    groups.append(cur)
  return groups


def _bp_place(
    reps: list[Any], ts: list[Any], zero_copy: bool
) -> tuple[list[Any], int]:
  """Places replicated leaves on their targets; local only.

  With `zero_copy`, a fully replicated target is assembled from the
  replicated array's own shards on the target devices (no copy); everything
  else is one batched `device_put` of a local replica.

  Args:
    reps: Leaves replicated over their source mesh.
    ts: Target shardings.
    zero_copy: Whether to alias replicated targets (TUNIX_BATCHED_PUBLISH=2).

  Returns:
    The placed leaves and how many were assembled without a copy.
  """
  outs, put = [None] * len(reps), []
  for i, (r, s) in enumerate(zip(reps, ts)):
    if zero_copy and s.is_fully_replicated:
      by_device = {shard.device: shard.data for shard in r.addressable_shards}
      devices = list(s.addressable_devices)
      if all(d in by_device for d in devices):
        outs[i] = jax.make_array_from_single_device_arrays(
            r.shape, s, [by_device[d] for d in devices]
        )
        continue
    put.append(i)
  if put:
    placed = jax.device_put(
        [reps[i].addressable_data(0) for i in put], [ts[i] for i in put]
    )
    for i, y in zip(put, placed):
      outs[i] = y
  return outs, len(reps) - len(put)


def _bp_check(pub: int, xs: list[Any], ts: list[Any], outs: list[Any]) -> bool:
  """Byte-exact comparison against the per-leaf path; every process votes.

  Checks up to four leaves (first, last, two inner; each at most
  TUNIX_BATCHED_PUBLISH_CHECK_MB, default 64) on every local shard, that no
  output aliases the reference, and that a one-bit flip is detected. The
  per-leaf reference is a collective, so it runs unconditionally on every
  process.

  Args:
    pub: Publish counter (for logging).
    xs: Source leaves.
    ts: Target shardings.
    outs: Batched outputs.

  Returns:
    Whether all processes passed.
  """
  n = len(xs)
  try:
    cap = float(os.environ.get('TUNIX_BATCHED_PUBLISH_CHECK_MB', '64')) * 2**20
  except ValueError:
    cap = 64.0 * 2**20
  sel = [
      i for i in sorted({0, n // 3, 2 * n // 3, n - 1}) if xs[i].nbytes <= cap
  ]
  refs = [_multicontroller_reshard_leaf(xs[i], ts[i]) for i in sel]
  shards = bad = alias = 0
  negctl, note = 'not_run', ''
  try:
    jax.block_until_ready(refs)
    meta = sum(
        o.dtype != x.dtype or o.shape != x.shape or o.sharding != s
        for o, x, s in zip(outs, xs, ts)
    )
    for k, i in enumerate(sel):
      got = {sh.device.id: sh.data for sh in outs[i].addressable_shards}
      want = {sh.device.id: sh.data for sh in refs[k].addressable_shards}
      if set(got) != set(want) or outs[i].sharding != refs[k].sharding:
        bad += 1
        continue
      for d in sorted(got):
        alias += (
            got[d].unsafe_buffer_pointer() == want[d].unsafe_buffer_pointer()
        )
        a, b = np.asarray(got[d]).tobytes(), np.asarray(want[d]).tobytes()
        shards += 1
        bad += a != b
        if negctl == 'not_run' and b:
          flipped = bytearray(b)
          flipped[len(flipped) // 2] ^= 1
          negctl = 'detected' if bytes(flipped) != a else 'blind'
    ok = bool(
        not meta and not bad and not alias and shards and negctl == 'detected'
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    ok, note = False, ' error=%r' % (e,)
  votes = _bp_vote(ok)
  logging.info(
      'TUNIX_BATCHED_PUBLISH check pub=%d leaves=%s shards=%d mismatched=%d'
      ' aliased=%d negctl=%s processes_ok=%d/%d%s',
      pub,
      sel,
      shards,
      bad,
      alias,
      negctl,
      sum(votes),
      len(votes),
      note,
  )
  return all(votes)


def _batched_publish(source: Any, target: Any) -> Any:
  """Moves every leaf of one publish with one jitted program per group.

  Same transfer as `_multicontroller_reshard_leaf`, batched: one identity jit
  per group of leaves (source shardings in, replicated over the source mesh
  out), compiled once per group signature, then one local placement. All
  decisions before the first collective use global metadata only, so every
  process takes the same branch; new compilations are agreed on by a vote.
  The first TUNIX_BATCHED_PUBLISH_CHECK publishes are compared byte-exactly
  against the per-leaf path and a failed check disables batching for the
  run. TUNIX_BATCHED_PUBLISH_NOSYNC=1 skips the timing-only syncs after the
  checked publishes.

  Args:
    source: The source tree.
    target: The tree of target shardings.

  Returns:
    The published tree, or None to fall back to the per-leaf path.
  """
  _BP_STATE['pub'] += 1
  pub, t0, stage = _BP_STATE['pub'], time.perf_counter(), 'pre'
  try:
    checks = int(os.environ.get('TUNIX_BATCHED_PUBLISH_CHECK', '0') or '0')
    xs, treedef = jax.tree_util.tree_flatten(source)
    ts = treedef.flatten_up_to(target)
    why = _bp_decline(xs, ts)
    if why:
      if _bp_should_log(pub):
        logging.info(
            'TUNIX_BATCHED_PUBLISH pub=%d: per-leaf path (%s), %d leaves.',
            pub,
            why,
            len(xs),
        )
      return None
    groups = _bp_groups(xs)
    keys = [
        tuple((xs[i].sharding, xs[i].shape, xs[i].dtype) for i in g)
        for g in groups
    ]
    new, ok, err = {}, True, ''
    for key, g in zip(keys, groups):
      if key in _BP_STATE['exe'] or key in new:
        continue
      try:
        ins = tuple(xs[i].sharding for i in g)
        outs_sh = tuple(
            jax.sharding.NamedSharding(s.mesh, jax.sharding.PartitionSpec())
            for s in ins
        )
        fn = jax.jit(lambda *a: a, in_shardings=ins, out_shardings=outs_sh)
        new[key] = fn.lower(*[xs[i] for i in g]).compile()
      except Exception as e:  # pylint: disable=broad-exception-caught
        ok, err = False, repr(e)[:300]
        break
    if new or not ok:
      stage = 'vote'
      votes = _bp_vote(ok)
      if not all(votes):
        logging.warning(
            'TUNIX_BATCHED_PUBLISH pub=%d: per-leaf path (compile votes %s):'
            ' %s',
            pub,
            votes,
            err,
        )
        return None
      _BP_STATE['exe'].update(new)
    t1 = time.perf_counter()
    stage = 'gather'
    sync = (
        pub <= checks or os.environ.get('TUNIX_BATCHED_PUBLISH_NOSYNC') != '1'
    )
    if not sync and not _BP_STATE['nosync']:
      _BP_STATE['nosync'] = True
      logging.info('TUNIX_BATCHED_PUBLISH_NOSYNC=1 from pub=%d.', pub)
    reps = [None] * len(xs)
    for key, g in zip(keys, groups):
      out = _BP_STATE['exe'][key](*[xs[i] for i in g])
      if sync:
        jax.block_until_ready(out)
      for j, i in enumerate(g):
        reps[i] = out[j]
    t2 = time.perf_counter()
    stage = 'place'
    try:
      outs, zero_copied = _bp_place(
          reps, ts, _batched_publish_mode() == '2'
      )
      if sync:
        jax.block_until_ready(outs)
    except Exception as e:  # pylint: disable=broad-exception-caught
      # Local-only recovery; never a collective.
      logging.warning('TUNIX_BATCHED_PUBLISH pub=%d: place failed: %r', pub, e)
      outs = [
          jax.device_put(r.addressable_data(0), s) for r, s in zip(reps, ts)
      ]
      zero_copied = 0
      jax.block_until_ready(outs)
    del reps
  except Exception as e:  # pylint: disable=broad-exception-caught
    if stage != 'pre':
      # Past the first collective a per-leaf fallback on this process alone
      # would run different collectives than its peers: fail loudly.
      logging.error(
          'TUNIX_BATCHED_PUBLISH pub=%d failed at %s: %r', pub, stage, e
      )
      raise
    logging.warning('TUNIX_BATCHED_PUBLISH pub=%d: per-leaf path: %r', pub, e)
    return None
  t3 = time.perf_counter()
  if _bp_should_log(pub):
    logging.info(
        'TUNIX_BATCHED_PUBLISH pub=%d: %d leaves in %d groups, %d bytes,'
        ' zero-copy=%d, compile=%.1fms gather=%.1fms place=%.1fms',
        pub,
        len(xs),
        len(groups),
        sum(x.nbytes for x in xs),
        zero_copied,
        (t1 - t0) * 1e3,
        (t2 - t1) * 1e3,
        (t3 - t2) * 1e3,
    )
  if pub <= checks and not _bp_check(pub, xs, ts, outs):
    _BP_STATE['off'] = 'check_failed_pub%d' % pub
    logging.error(
        'TUNIX_BATCHED_PUBLISH pub=%d: check failed; batching disabled for the'
        ' run, redoing this publish per leaf.',
        pub,
    )
    return None
  return jax.tree_util.tree_unflatten(treedef, outs)


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
    if _batched_publish_mode():
      out = _batched_publish(x, sharding)
      if out is not None:
        return out
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
