# Copyright 2026 Google LLC
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

"""Tests for the batched multi-controller weight publish.

With TUNIX_BATCHED_PUBLISH, a multi-controller publish
(TUNIX_MULTI_CONTROLLER_RESHARD=1) moves its leaves with one jitted transfer
per group of leaves instead of one per leaf, and must give exactly the
per-leaf result.

The single-process tests run on several CPU devices and mock
`jax.process_count` and `_bp_vote` where they need more processes. The leaves
of a single process are always fully addressable, which `_bp_decline`
rejects, so the tests of the transfer let them through (`_bridge`);
`_bp_decline` itself is tested with stand-ins for leaves that span processes.
`TwoProcessPublishTest` runs the real path on two processes with four CPU
devices each (gloo collectives), where every source leaf spans both
processes.
"""

# pylint: disable=protected-access

import contextlib
import gc
import json
import os
import socket
import subprocess
import sys
import tempfile
import traceback
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import reshard

_CHILD_ENV = 'TUNIX_BATCHED_PUBLISH_TEST_CHILD'
_NUM_PROCESSES = 2
_LOCAL_DEVICES = 4
# A 1 KiB group cap: splits the `_LEAVES` tree into [a, b] and [c, d, e].
_GROUP_MB_1KIB = str(1 / 1024)

# name: (shape, dtype, source spec, target spec). The targets mix replicated
# layouts (assembled without a copy under TUNIX_BATCHED_PUBLISH=2) and
# sharded ones. Every dim divides the mesh axis sizes used below.
_LEAVES = {
    'a_embed': ((16, 8), np.float32, ('data', 'model'), (None, 'tp')),
    'b_norm': ((8,), np.float32, ('model',), ()),
    'c_kernel': (
        (4, 8, 8), jnp.bfloat16, (None, 'data', 'model'), (None, None, 'tp')
    ),
    'd_bias': ((8,), jnp.bfloat16, (), ()),
    'e_step': ((16,), np.int32, (('data', 'model'),), ('tp',)),
}
_REPLICATED_TARGETS = 2  # b_norm and d_bias.

if _CHILD_ENV not in os.environ:
  os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'


def _mesh(devices, shape, names) -> jax.sharding.Mesh:
  return jax.sharding.Mesh(np.asarray(devices).reshape(shape), names)


def _named(mesh, *spec) -> jax.sharding.NamedSharding:
  return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))


def _tree(src_mesh, dst_mesh, names=tuple(_LEAVES)):
  """Source leaves over `src_mesh` and target shardings over `dst_mesh`.

  Args:
    src_mesh: The mesh of the source (trainer) leaves.
    dst_mesh: The mesh of the targets (engine), a subset of its devices.
    names: The `_LEAVES` to build.

  Returns:
    The source tree, the tree of target shardings and the global values.
  """
  rng = np.random.default_rng(0)
  source, target, values = {}, {}, {}
  for name in names:
    shape, dtype, src_spec, dst_spec = _LEAVES[name]
    if np.issubdtype(dtype, np.integer):
      value = rng.integers(-1000, 1000, shape).astype(dtype)
    else:
      value = rng.standard_normal(shape).astype(dtype)
    source[name] = jax.make_array_from_callback(
        shape, _named(src_mesh, *src_spec), lambda index, v=value: v[index]
    )
    target[name] = _named(dst_mesh, *dst_spec)
    values[name] = value
  return source, target, values


def _local_bytes(x: jax.Array) -> dict[int, bytes]:
  return {
      shard.device.id: np.asarray(shard.data).tobytes()
      for shard in x.addressable_shards
  }


def _mismatched(got, want) -> list[str]:
  """Leaves whose sharding, dtype or local bytes differ between two trees."""
  return [
      name
      for name in sorted(want)
      if got[name].sharding != want[name].sharding
      or got[name].dtype != want[name].dtype
      or _local_bytes(got[name]) != _local_bytes(want[name])
  ]


def _mismatched_values(tree, values) -> list[str]:
  """Leaves whose local shards differ from the global values."""
  return [
      name
      for name in sorted(values)
      if any(
          np.asarray(shard.data).tobytes()
          != np.asarray(values[name][shard.index]).tobytes()
          for shard in tree[name].addressable_shards
      )
  ]


def _per_leaf(source, target):
  """The per-leaf multi-controller transfer: the reference."""
  return jax.tree.map(reshard._multicontroller_reshard_leaf, source, target)


def _bridge():
  """Lets the leaves of a single process take the batched path.

  They are always fully addressable, which `_bp_decline` rejects; every other
  reason it gives still applies.

  Returns:
    A patch of `reshard._bp_decline`.
  """
  decline = reshard._bp_decline

  def bridged(xs, ts):
    why = decline(xs, ts)
    return None if why == 'fully_addressable' else why

  return mock.patch.object(reshard, '_bp_decline', bridged)


class _PublishTestCase(parameterized.TestCase):
  """Fresh publish state and environment, and a source and a target mesh."""

  def setUp(self):
    super().setUp()
    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest('needs at least 2 devices')
    half = len(devices) // 2
    # A process's engine holds its copy on a strict subset of the devices
    # that the trainer's source leaves span.
    self.src_mesh = _mesh(devices[: 2 * half], (2, half), ('data', 'model'))
    self.dst_mesh = _mesh(devices[:half], (half,), ('tp',))
    self.enter_context(
        mock.patch.object(reshard, '_BP_STATE', reshard._BatchedPublishState())
    )
    self.enter_context(mock.patch.dict(os.environ))
    for name in list(os.environ):
      if name.startswith('TUNIX_'):
        del os.environ[name]

  def tree(self, names=tuple(_LEAVES)):
    return _tree(self.src_mesh, self.dst_mesh, names)


class GroupsTest(_PublishTestCase):

  def _other_mesh(self):
    """The source devices under other axis names: a different mesh."""
    devices = self.src_mesh.devices
    return _mesh(devices, devices.shape, ('x', 'y'))

  def _leaves(self, sizes, mesh):
    """Float32 leaves of the given sizes in bytes, over `mesh`."""
    return [
        jax.device_put(np.zeros(size // 4, np.float32), _named(mesh))
        for size in sizes
    ]

  def test_groups_split_at_the_byte_cap_and_at_a_mesh_change(self):
    os.environ['TUNIX_BATCHED_PUBLISH_GROUP_MB'] = _GROUP_MB_1KIB
    other = self._other_mesh()
    xs = self._leaves([256] * 5 + [2048, 256], self.src_mesh)
    xs += self._leaves([256, 256], other)
    # Four 256-byte leaves fill a 1 KiB group, a leaf over the cap is a group
    # of its own, and a new source mesh starts a new group.
    self.assertEqual(
        reshard._bp_groups(xs), [[0, 1, 2, 3], [4], [5], [6], [7, 8]]
    )

  def test_default_cap_splits_at_mesh_changes_only(self):
    other = self._other_mesh()
    xs = self._leaves([256] * 3, self.src_mesh)
    xs += self._leaves([256], other) + self._leaves([256], self.src_mesh)
    self.assertEqual(reshard._bp_groups(xs), [[0, 1, 2], [3], [4]])


class DeclineTest(_PublishTestCase):

  def _spanning(self, sharding):
    """A stand-in for a source leaf that spans processes."""
    leaf = mock.MagicMock(spec=jax.Array)
    leaf.dtype = np.dtype(np.float32)
    leaf.is_fully_addressable = False
    leaf.sharding = sharding
    return leaf

  def test_reasons_for_the_per_leaf_path(self):
    dst = _named(self.dst_mesh)
    spanning = self._spanning(_named(self.src_mesh, 'data'))
    single = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    cases = [
        ('empty', [], []),
        ('non_array_or_non_sharding', [np.zeros(8, np.float32)], [dst]),
        ('non_array_or_non_sharding', [spanning], [None]),
        ('extended_dtype', [jax.random.key(0)], [dst]),
        ('fully_addressable', [jnp.zeros(8)], [dst]),
        ('src_not_named', [self._spanning(single)], [dst]),
        # The targets must be on a strict subset of the source devices.
        ('not_bridge_topology', [spanning], [_named(self.src_mesh)]),
        ('not_bridge_topology', [self._spanning(dst)], [_named(self.src_mesh)]),
        # One leaf that cannot take the bridge declines the whole publish.
        ('fully_addressable', [spanning, jnp.zeros(8)], [dst, dst]),
        (None, [spanning, spanning], [dst, _named(self.dst_mesh, 'tp')]),
    ]
    for want, xs, ts in cases:
      with self.subTest(want=want, leaves=len(xs)):
        self.assertEqual(reshard._bp_decline(xs, ts), want)

  def test_a_failed_check_declines_every_later_publish(self):
    reshard._BP_STATE.off = 'check_failed_pub1'
    self.assertEqual(
        reshard._bp_decline(
            [self._spanning(_named(self.src_mesh))], [_named(self.dst_mesh)]
        ),
        'disabled(check_failed_pub1)',
    )


class PlaceTest(_PublishTestCase):

  def setUp(self):
    super().setUp()
    self.value = np.arange(32, dtype=np.float32).reshape(4, 8)
    # Replicated over the source mesh, as a gather leaves it.
    self.rep = jax.device_put(self.value, _named(self.src_mesh))
    self.shardings = [_named(self.dst_mesh), _named(self.dst_mesh, None, 'tp')]

  def _check_values(self, outs):
    for out, sharding in zip(outs, self.shardings):
      self.assertEqual(out.sharding, sharding)
      np.testing.assert_array_equal(np.asarray(out), self.value)

  def test_zero_copy_assembles_replicated_targets_from_the_replica(self):
    outs, zero_copied = reshard._bp_place(
        [self.rep, self.rep], self.shardings, zero_copy=True
    )
    self.assertEqual(zero_copied, 1)
    buffers = {
        shard.device: shard.data.unsafe_buffer_pointer()
        for shard in self.rep.addressable_shards
    }
    for shard in outs[0].addressable_shards:
      self.assertEqual(
          shard.data.unsafe_buffer_pointer(), buffers[shard.device]
      )
    self._check_values(outs)

  def test_without_zero_copy_every_leaf_is_put(self):
    outs, zero_copied = reshard._bp_place(
        [self.rep, self.rep], self.shardings, zero_copy=False
    )
    self.assertEqual(zero_copied, 0)
    self._check_values(outs)


class CheckTest(_PublishTestCase):

  def setUp(self):
    super().setUp()
    source, target, _ = self.tree()
    with _bridge():
      published = reshard._batched_publish(source, target)
    self.xs, treedef = jax.tree.flatten(source)
    self.ts = treedef.flatten_up_to(target)
    self.outs = treedef.flatten_up_to(published)
    self.references = []
    reference = reshard._multicontroller_reshard_leaf

    def recording_reference(x, sharding):
      self.references.append(self._index(x))
      return reference(x, sharding)

    self.enter_context(
        mock.patch.object(
            reshard, '_multicontroller_reshard_leaf', recording_reference
        )
    )

  def _index(self, x):
    return next(i for i, y in enumerate(self.xs) if y is x)

  def test_passes_on_the_batched_outputs(self):
    self.assertTrue(reshard._bp_check(1, self.xs, self.ts, self.outs))
    # Five leaves under the default cap: the first, the last, two inner ones.
    self.assertEqual(self.references, [0, 1, 3, 4])

  def test_compares_the_sampled_leaves_under_the_cap(self):
    # A 64-byte cap drops the sampled a_embed (512 bytes) and keeps b_norm
    # (32), d_bias (16) and e_step (64).
    os.environ['TUNIX_BATCHED_PUBLISH_CHECK_MB'] = str(64 / 2**20)
    self.assertTrue(reshard._bp_check(1, self.xs, self.ts, self.outs))
    self.assertEqual(self.references, [1, 3, 4])

  def test_fails_without_a_sampled_leaf_under_the_cap(self):
    # Nothing is compared, so the check does not pass.
    os.environ['TUNIX_BATCHED_PUBLISH_CHECK_MB'] = str(8 / 2**20)
    with mock.patch.object(reshard, '_bp_vote', return_value=[0]) as vote:
      self.assertFalse(reshard._bp_check(1, self.xs, self.ts, self.outs))
    self.assertEqual(self.references, [])
    vote.assert_called_once_with(False)

  def test_fails_on_different_bytes(self):
    self.outs[0] = jax.device_put(np.asarray(self.outs[0]) + 1, self.ts[0])
    self.assertFalse(reshard._bp_check(1, self.xs, self.ts, self.outs))

  def test_fails_on_a_wrong_sharding_of_a_leaf_it_does_not_compare(self):
    self.outs[2] = jax.device_put(self.outs[2], _named(self.dst_mesh))
    self.assertFalse(reshard._bp_check(1, self.xs, self.ts, self.outs))
    self.assertNotIn(2, self.references)

  def test_fails_on_outputs_that_alias_their_reference(self):
    with mock.patch.object(
        reshard,
        '_multicontroller_reshard_leaf',
        lambda x, sharding: self.outs[self._index(x)],
    ):
      self.assertFalse(reshard._bp_check(1, self.xs, self.ts, self.outs))

  def test_fails_everywhere_if_another_process_fails(self):
    with mock.patch.object(reshard, '_bp_vote', return_value=[1, 0]) as vote:
      self.assertFalse(reshard._bp_check(1, self.xs, self.ts, self.outs))
    vote.assert_called_once_with(True)


class BatchedPublishTest(_PublishTestCase):

  def _record_placements(self) -> list[int]:
    """Records the zero-copy count of every `_bp_place` call."""
    counts = []
    place = reshard._bp_place

    def recording_place(reps, ts, zero_copy):
      outs, zero_copied = place(reps, ts, zero_copy)
      counts.append(zero_copied)
      return outs, zero_copied

    self.enter_context(mock.patch.object(reshard, '_bp_place', recording_place))
    return counts

  @parameterized.product(mode=('1', '2'), group_mb=('2048', _GROUP_MB_1KIB))
  def test_matches_the_per_leaf_path(self, mode, group_mb):
    os.environ.update(
        TUNIX_BATCHED_PUBLISH=mode,
        TUNIX_BATCHED_PUBLISH_GROUP_MB=group_mb,
        TUNIX_BATCHED_PUBLISH_CHECK='1',
    )
    source, target, values = self.tree()
    counts = self._record_placements()
    with _bridge():
      got = reshard._batched_publish(source, target)
    self.assertIsNotNone(got)
    self.assertEqual(_mismatched(got, _per_leaf(source, target)), [])
    self.assertEqual(_mismatched_values(got, values), [])
    self.assertIsNone(reshard._BP_STATE.off)  # The check passed.
    # One placement per group.
    self.assertLen(counts, 1 if group_mb == '2048' else 2)
    self.assertEqual(sum(counts), _REPLICATED_TARGETS if mode == '2' else 0)

  def test_each_group_is_placed_and_released_before_the_next_gather(self):
    if jax.device_count() < 4:
      self.skipTest('needs at least 4 devices')
    os.environ['TUNIX_BATCHED_PUBLISH_GROUP_MB'] = _GROUP_MB_1KIB
    # No replicated source leaf, so every live array replicated over all
    # source devices is a gathered replica.
    source, target, _ = self.tree(('a_embed', 'b_norm', 'c_kernel', 'e_step'))
    src_devices = set(self.src_mesh.devices.flat)

    def replicas():
      return sum(
          1
          for a in jax.live_arrays()
          if not a.is_deleted()
          and a.is_fully_replicated
          and a.sharding.device_set == src_devices
      )

    # Free cyclic garbage now and not during the publish, so that only the
    # publish changes the count.
    gc.collect()
    if gc.isenabled():
      gc.disable()
      self.addCleanup(gc.enable)
    baseline = replicas()
    events = []
    compile_group, place = reshard._bp_compile, reshard._bp_place

    def recording_compile(group):
      transfer = compile_group(group)

      def gather(*leaves):
        before = replicas() - baseline
        out = transfer(*leaves)
        events.append(('gather', len(leaves), before, replicas() - baseline))
        return out

      return gather

    def recording_place(reps, ts, zero_copy):
      events.append(('place', len(reps)))
      return place(reps, ts, zero_copy)

    with (
        _bridge(),
        mock.patch.object(reshard, '_bp_compile', recording_compile),
        mock.patch.object(reshard, '_bp_place', recording_place),
    ):
      got = reshard._batched_publish(source, target)
    self.assertEqual(_mismatched(got, _per_leaf(source, target)), [])
    groups = reshard._bp_groups(jax.tree.leaves(source))
    self.assertLen(groups, 2)
    # No replica is alive when a group is gathered, the gather leaves one per
    # leaf of the group, and the group is placed before the next gather.
    self.assertEqual(
        events,
        [
            event
            for g in groups
            for event in (('gather', len(g), 0, len(g)), ('place', len(g)))
        ],
    )
    self.assertEqual(replicas(), baseline)

  def test_new_compilations_need_every_vote(self):
    source, target, _ = self.tree()
    with (
        _bridge(),
        mock.patch.object(reshard, '_bp_vote', return_value=[1, 0]) as vote,
    ):
      self.assertIsNone(reshard._batched_publish(source, target))
    vote.assert_called_once_with(True)
    # Nothing is kept and batching stays on: the next publish votes again.
    self.assertEmpty(reshard._BP_STATE.exe)
    self.assertIsNone(reshard._BP_STATE.off)
    with (
        _bridge(),
        mock.patch.object(reshard, '_bp_vote', return_value=[1, 1]) as vote,
    ):
      self.assertIsNotNone(reshard._batched_publish(source, target))
    vote.assert_called_once_with(True)
    self.assertNotEmpty(reshard._BP_STATE.exe)

  def test_a_local_compile_failure_votes_against(self):
    source, target, _ = self.tree()
    with (
        _bridge(),
        mock.patch.object(
            reshard, '_bp_compile', side_effect=RuntimeError('no compile')
        ),
        mock.patch.object(reshard, '_bp_vote', return_value=[0, 1]) as vote,
    ):
      self.assertIsNone(reshard._batched_publish(source, target))
    vote.assert_called_once_with(False)

  def test_known_groups_compile_and_vote_once(self):
    source, target, _ = self.tree()
    with (
        _bridge(),
        mock.patch.object(reshard, '_bp_vote', return_value=[1, 1]) as vote,
        mock.patch.object(
            reshard, '_bp_compile', wraps=reshard._bp_compile
        ) as compile_group,
    ):
      for _ in range(3):
        self.assertIsNotNone(reshard._batched_publish(source, target))
    compile_group.assert_called_once()  # One group under the default cap.
    vote.assert_called_once_with(True)

  def test_a_failure_before_the_first_collective_falls_back(self):
    os.environ['TUNIX_BATCHED_PUBLISH_GROUP_MB'] = 'not-a-number'
    source, target, _ = self.tree()
    with _bridge(), mock.patch.object(reshard, '_bp_vote') as vote:
      self.assertIsNone(reshard._batched_publish(source, target))
    vote.assert_not_called()

  def test_a_failure_after_the_first_collective_raises(self):
    source, target, _ = self.tree()

    def broken_compile(group):
      del group

      def gather(*leaves):
        raise RuntimeError(f'gather of {len(leaves)} leaves failed')

      return gather

    with (
        _bridge(),
        mock.patch.object(reshard, '_bp_compile', broken_compile),
        mock.patch.object(reshard, '_bp_vote', return_value=[1, 1]),
        self.assertRaisesRegex(RuntimeError, 'gather of 5 leaves failed'),
    ):
      reshard._batched_publish(source, target)

  def test_a_failed_vote_raises(self):
    source, target, _ = self.tree()
    with (
        _bridge(),
        mock.patch.object(
            reshard, '_bp_vote', side_effect=RuntimeError('vote failed')
        ),
        self.assertRaisesRegex(RuntimeError, 'vote failed'),
    ):
      reshard._batched_publish(source, target)

  def test_a_failed_placement_is_redone_locally(self):
    source, target, values = self.tree()
    with (
        _bridge(),
        mock.patch.object(
            reshard, '_bp_place', side_effect=RuntimeError('place failed')
        ),
        mock.patch.object(reshard, '_bp_vote', return_value=[1, 1]) as vote,
    ):
      got = reshard._batched_publish(source, target)
    self.assertEqual(_mismatched(got, _per_leaf(source, target)), [])
    self.assertEqual(_mismatched_values(got, values), [])
    # The recovery is local: the compile vote is the only collective.
    vote.assert_called_once_with(True)

  def test_a_failed_check_disables_batching_for_the_run(self):
    os.environ['TUNIX_BATCHED_PUBLISH_CHECK'] = '2'
    source, target, _ = self.tree()
    first = jax.tree.leaves(source)[0]
    reference = reshard._multicontroller_reshard_leaf

    def wrong_reference(x, sharding):
      ref = reference(x, sharding)
      if x is first:
        ref = jax.device_put(np.asarray(ref) + 1, sharding)
      return ref

    with (
        _bridge(),
        mock.patch.object(
            reshard, '_multicontroller_reshard_leaf', wrong_reference
        ),
    ):
      self.assertIsNone(reshard._batched_publish(source, target))
    self.assertEqual(reshard._BP_STATE.off, 'check_failed_pub1')
    with _bridge(), mock.patch.object(reshard, '_bp_compile') as compile_group:
      self.assertIsNone(reshard._batched_publish(source, target))
    compile_group.assert_not_called()


class ReshardPytreeTest(_PublishTestCase):
  """`reshard_pytree` on the multi-controller path, two processes mocked."""

  def setUp(self):
    super().setUp()
    os.environ.update(
        TUNIX_MULTI_CONTROLLER_RESHARD='1', TUNIX_PRE_RESHARD_BARRIER='0'
    )
    self.enter_context(mock.patch.dict(reshard._MC_STATE))
    self.enter_context(mock.patch.object(jax, 'process_count', return_value=2))
    # Every process agrees.
    self.enter_context(
        mock.patch.object(
            reshard, '_bp_vote', side_effect=lambda ok: [int(ok)] * 2
        )
    )

  def test_flag_off_keeps_the_per_leaf_path(self):
    source, target, values = self.tree()
    with mock.patch.object(reshard, '_batched_publish') as batched_publish:
      got = reshard.reshard_pytree(source, target)
    batched_publish.assert_not_called()
    self.assertEqual(_mismatched_values(got, values), [])

  @parameterized.parameters('1', '2')
  def test_batched_publish_matches_the_per_leaf_path(self, mode):
    source, target, values = self.tree()
    want = reshard.reshard_pytree(source, target)
    os.environ.update(
        TUNIX_BATCHED_PUBLISH=mode, TUNIX_BATCHED_PUBLISH_CHECK='1'
    )
    batched = []
    batched_publish = reshard._batched_publish

    def recording_batched_publish(x, sharding):
      out = batched_publish(x, sharding)
      batched.append(out is not None)
      return out

    with (
        _bridge(),
        mock.patch.object(
            reshard, '_batched_publish', recording_batched_publish
        ),
    ):
      got = reshard.reshard_pytree(source, target)
    self.assertEqual(batched, [True])
    self.assertEqual(_mismatched(got, want), [])
    self.assertEqual(_mismatched_values(got, values), [])

  def test_a_declined_publish_takes_the_per_leaf_path(self):
    # Without `_bridge`, the leaves of a single process are declined.
    os.environ['TUNIX_BATCHED_PUBLISH'] = '1'
    source, target, values = self.tree()
    with mock.patch.object(reshard, '_bp_compile') as compile_group:
      got = reshard.reshard_pytree(source, target)
    compile_group.assert_not_called()
    self.assertEqual(_mismatched_values(got, values), [])


def _run_child():
  """One process of the 2-process publish job (see `_run_job`)."""
  args = json.loads(os.environ[_CHILD_ENV])
  process_id = args['process_id']
  jax.config.update('jax_cpu_collectives_implementation', 'gloo')
  jax.distributed.initialize(
      f"localhost:{args['port']}",
      num_processes=_NUM_PROCESSES,
      process_id=process_id,
  )
  # Every source leaf spans both processes; each process's engine holds its
  # copy on its own devices.
  src_mesh = _mesh(jax.devices(), (2, -1), ('data', 'model'))
  dst_mesh = _mesh(jax.local_devices(), (-1,), ('tp',))
  source, target, values = _tree(src_mesh, dst_mesh)
  # TUNIX_BATCHED_PUBLISH is unset: the per-leaf path, the reference.
  want = reshard.reshard_pytree(source, target)
  results = {'reference_mismatched': _mismatched_values(want, values)}

  batched, zero_copied = [], []
  batched_publish, place = reshard._batched_publish, reshard._bp_place

  def recording_batched_publish(x, sharding):
    out = batched_publish(x, sharding)
    batched.append(out is not None)
    return out

  def recording_place(reps, ts, zero_copy):
    outs, n = place(reps, ts, zero_copy)
    zero_copied.append(n)
    return outs, n

  def corrupting_place(reps, ts, zero_copy):
    outs, n = place(reps, ts, zero_copy)
    return [out + 1 for out in outs], n

  def publish(fresh_state=False, **env):
    """One publish: what it did and whether it matches the reference."""
    if fresh_state:
      reshard._BP_STATE = reshard._BatchedPublishState()
    os.environ.update(env)
    del batched[:], zero_copied[:]
    got = reshard.reshard_pytree(source, target)
    return {
        'batched': list(batched),
        'groups': len(zero_copied),
        'zero_copied': sum(zero_copied),
        'mismatched': _mismatched(got, want),
        'compiled': len(reshard._BP_STATE.exe),
        'off': reshard._BP_STATE.off,
    }

  def on_process_1(patch):
    return patch if process_id == 1 else contextlib.nullcontext()

  with (
      mock.patch.object(reshard, '_batched_publish', recording_batched_publish),
      mock.patch.object(reshard, '_bp_place', recording_place),
  ):
    for mode in ('1', '2'):
      results[f'mode{mode}'] = publish(
          fresh_state=True,
          TUNIX_BATCHED_PUBLISH=mode,
          TUNIX_BATCHED_PUBLISH_CHECK='1',
          TUNIX_BATCHED_PUBLISH_GROUP_MB=_GROUP_MB_1KIB,
      )
    # A compile that fails on process 1 only: both processes take the
    # per-leaf path, and the next publish compiles and batches on both.
    failing_compile = mock.patch.object(
        reshard, '_bp_compile', side_effect=RuntimeError('injected')
    )
    with on_process_1(failing_compile):
      results['vote_failed'] = publish(
          fresh_state=True,
          TUNIX_BATCHED_PUBLISH='1',
          TUNIX_BATCHED_PUBLISH_CHECK='0',
      )
    results['vote_next'] = publish()
    # A check that fails on process 1 only disables batching on both.
    corrupting = mock.patch.object(reshard, '_bp_place', corrupting_place)
    with on_process_1(corrupting):
      results['check_failed'] = publish(
          fresh_state=True, TUNIX_BATCHED_PUBLISH_CHECK='1'
      )
    results['check_next'] = publish()

  with open(
      os.path.join(args['out_dir'], f'process{process_id}.json'),
      'w',
      encoding='utf-8',
  ) as f:
    json.dump(results, f)


def _free_port() -> int:
  with socket.socket() as s:
    s.bind(('localhost', 0))
    return s.getsockname()[1]


def _run_job(out_dir: str) -> list[dict[str, Any]]:
  """Runs `_run_child` on 2 processes with 4 CPU devices each (gloo)."""
  repo_root = os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  )
  env = {
      k: v
      for k, v in os.environ.items()
      if not k.startswith('TUNIX_')
      and k.lower() not in ('http_proxy', 'https_proxy', 'no_proxy')
  }
  env.update(
      JAX_PLATFORMS='cpu',
      XLA_FLAGS=f'--xla_force_host_platform_device_count={_LOCAL_DEVICES}',
      TUNIX_MULTI_CONTROLLER_RESHARD='1',
      TUNIX_PRE_RESHARD_BARRIER='0',
      PYTHONPATH=os.pathsep.join([repo_root, env.get('PYTHONPATH', '')]),
  )
  port = _free_port()
  procs = []
  try:
    for process_id in range(_NUM_PROCESSES):
      env[_CHILD_ENV] = json.dumps(
          {'process_id': process_id, 'port': port, 'out_dir': out_dir}
      )
      procs.append(
          subprocess.Popen(
              [sys.executable, os.path.abspath(__file__)],
              env=dict(env),
              stdout=subprocess.PIPE,
              stderr=subprocess.STDOUT,
              text=True,
          )
      )
    logs = [p.communicate(timeout=600)[0] for p in procs]
  finally:
    for p in procs:
      if p.poll() is None:
        p.kill()
  if any(p.returncode for p in procs):
    raise AssertionError(
        '2-process publish job failed:\n'
        + '\n'.join(log[-3000:] for log in logs)
    )
  results = []
  for process_id in range(_NUM_PROCESSES):
    with open(
        os.path.join(out_dir, f'process{process_id}.json'), encoding='utf-8'
    ) as f:
      results.append(json.load(f))
  return results


class TwoProcessPublishTest(absltest.TestCase):
  """The real path: source leaves that span two processes."""

  def test_batched_publish_across_processes(self):
    with tempfile.TemporaryDirectory() as out_dir:
      results = _run_job(out_dir)
    for process_id, result in enumerate(results):
      with self.subTest(process_id=process_id):
        # The per-leaf reference holds the global values.
        self.assertEqual(result['reference_mismatched'], [])
        # Both modes batch in two groups, match the per-leaf path bit for bit
        # and pass the byte-exact check.
        for mode, zero_copied in (('1', 0), ('2', _REPLICATED_TARGETS)):
          self.assertEqual(
              result[f'mode{mode}'],
              {
                  'batched': [True],
                  'groups': 2,
                  'zero_copied': zero_copied,
                  'mismatched': [],
                  'compiled': 2,
                  'off': None,
              },
          )
        # A compile failure on process 1 only: no process batches or keeps
        # a compilation; the next publish batches on both.
        self.assertEqual(result['vote_failed']['batched'], [False])
        self.assertEqual(result['vote_failed']['compiled'], 0)
        self.assertEqual(result['vote_failed']['mismatched'], [])
        self.assertEqual(result['vote_next']['batched'], [True])
        self.assertEqual(result['vote_next']['mismatched'], [])
        # A check failure on process 1 only disables batching on every
        # process; the publish is redone per leaf.
        for name in ('check_failed', 'check_next'):
          self.assertEqual(result[name]['batched'], [False])
          self.assertEqual(result[name]['mismatched'], [])
          self.assertEqual(result[name]['off'], 'check_failed_pub1')


if __name__ == '__main__':
  if _CHILD_ENV in os.environ:
    exit_code = 0
    try:
      _run_child()
    except Exception:  # pylint: disable=broad-exception-caught
      traceback.print_exc()
      exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip the distributed shutdown barrier; the peer may already be gone.
    os._exit(exit_code)
  absltest.main()
