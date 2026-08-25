# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for fixed-placement replicated data-parallel contracts."""

import functools

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tunix.rl import dp_training


class DPTrainingTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.contract = dp_training.DPTrainingContract(
        dp_size=16,
        tp_size=4,
        global_prompts=32,
        num_generations=8,
        local_trajectories=16,
    )

  def test_production_contract_uses_fixed_prompt_major_placement(self):
    self.contract.validate()
    self.assertEqual(self.contract.total_devices, 64)
    self.assertEqual(self.contract.global_trajectories, 256)
    self.assertEqual(self.contract.local_prompts, 2)
    ranks = self.contract.trajectory_ranks()
    np.testing.assert_array_equal(ranks[:16], np.zeros(16, np.int32))
    np.testing.assert_array_equal(ranks[-16:], np.full(16, 15, np.int32))
    self.assertEqual(
        tuple(len(indices) for indices in self.contract.rank_indices()),
        (16,) * 16,
    )

  def test_rank_major_groups_cover_all_trajectories_once(self):
    groups = self.contract.rank_major_reverse_groups()
    self.assertLen(groups, 16)
    self.assertEqual(groups[0], tuple(range(0, 256, 16)))
    self.assertEqual(groups[-1], tuple(range(15, 256, 16)))
    self.assertEqual(
        sorted(index for group in groups for index in group), list(range(256))
    )

  def test_contract_rejects_partial_prompt_group(self):
    with self.assertRaisesRegex(ValueError, 'complete prompt groups'):
      dp_training.DPTrainingContract(
          dp_size=16,
          tp_size=4,
          global_prompts=32,
          num_generations=8,
          local_trajectories=15,
      ).validate()

  def test_group_validation_rejects_split_group(self):
    valid = np.repeat(np.arange(32), 8)
    self.contract.validate_prompt_groups(valid)
    split = valid.copy()
    split[15], split[16] = split[16], split[15]
    with self.assertRaisesRegex(ValueError, 'split across DP ranks'):
      self.contract.validate_prompt_groups(split)

  def test_partition_inventory_rejects_dp_parameter_shard(self):
    summary = dp_training.validate_dp_replicated_partition_specs(
        {'left': P(None, 'tp'), 'right': P('tp', None)}, label='params'
    )
    self.assertEqual(summary, {'leaves': 2, 'dp_partitioned_leaves': 0})
    with self.assertRaisesRegex(ValueError, 'not replicated'):
      dp_training.validate_dp_replicated_partition_specs(
          {'bad': P('dp', 'tp')}, label='params'
      )

  def test_initialized_training_state_inventory_is_dp_replicated(self):
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('dp', 'tp'))
    sharding = jax.sharding.NamedSharding(mesh, P('tp'))

    class StateModule(nnx.Module):

      def __init__(self, offset):
        value = jax.device_put(
            jnp.arange(8, dtype=jnp.float32).reshape(2, 4) + offset,
            sharding,
        )
        self.value = nnx.Param(value, sharding=(None, 'tp'))

    states = [nnx.state(StateModule(offset)) for offset in (0.0, 1.0, 2.0)]
    inventory = dp_training.inspect_training_state_inventories(
        model=states[0], optimizer=states[1], accumulator=states[2]
    )
    self.assertEqual(set(inventory), {'model', 'optimizer', 'accumulator'})
    for summary in inventory.values():
      self.assertEqual(summary['arrays'], 1)
      self.assertEqual(summary['dp_partitioned_leaves'], 0)
      self.assertEqual(summary['tp_partitioned_leaves'], 1)
      self.assertEqual(summary['logical_bytes'], 32)

  def test_initialized_state_inventory_rejects_actual_dp_shard(self):
    if len(jax.devices()) < 2:
      self.skipTest('requires at least two forced CPU or accelerator devices')
    mesh = Mesh(np.asarray(jax.devices()[:2]).reshape(2, 1), ('dp', 'tp'))
    sharding = jax.sharding.NamedSharding(mesh, P('dp', None))

    class BadState(nnx.Module):

      def __init__(self):
        value = jax.device_put(jnp.ones((2, 4), jnp.float32), sharding)
        self.value = nnx.Param(value, sharding=('dp', None))

    with self.assertRaisesRegex(ValueError, 'not replicated'):
      dp_training.inspect_dp_replicated_state(
          nnx.state(BadState()), label='bad-state'
      )

  def test_dp16_tree_has_four_reduce_and_four_broadcast_rounds(self):
    reduce_rounds, broadcast_rounds = (
        dp_training.fixed_dp_tree_permutations(16)
    )
    self.assertLen(reduce_rounds, 4)
    self.assertLen(broadcast_rounds, 4)
    self.assertEqual(dp_training.fixed_dp_collective_count(16), 8)
    self.assertEqual(reduce_rounds[0][0], (1, 0))
    self.assertEqual(reduce_rounds[-1], ((8, 0),))
    self.assertEqual(broadcast_rounds[0], ((0, 8),))
    self.assertEqual(broadcast_rounds[-1][0], (0, 1))

  def test_tree_rejects_non_power_of_two_dp(self):
    for dp_size in (1, 3, 6, 15):
      with self.subTest(dp_size=dp_size):
        with self.assertRaisesRegex(ValueError, 'power-of-two'):
          dp_training.fixed_dp_tree_permutations(dp_size)

  def test_fixed_dp2_compatibility_order_is_unchanged(self):
    left = {'g': jnp.asarray([1.0e8, 1.0], jnp.float32)}
    right = {'g': jnp.asarray([-1.0e8, 2.0], jnp.float32)}
    expected = jax.tree.map(
        lambda x, y: (
            jax.lax.optimization_barrier(x)
            + jax.lax.optimization_barrier(y)
        ),
        left,
        right,
    )
    actual = dp_training.fixed_dp2_sum(left, right)
    np.testing.assert_array_equal(actual['g'], expected['g'])

  def test_fixed_dp16_sum_uses_registered_tree_not_left_fold(self):
    values = [1.0e8, 1.0, -1.0e8, 2.0] + [0.0] * 12
    contributions = [jnp.asarray(value, jnp.float32) for value in values]
    tree_result = dp_training.fixed_dp_sum(contributions)
    left_fold = contributions[0]
    for contribution in contributions[1:]:
      left_fold = left_fold + contribution
    self.assertEqual(float(tree_result), 0.0)
    self.assertEqual(float(left_fold), 2.0)

  def test_rank_isolation_reconstructs_dp16_group_cotangent(self):
    cotangent = jnp.arange(48, dtype=jnp.float32).reshape(16, 3)
    isolated = [
        dp_training.isolate_dp_rank_cotangent(
            cotangent, rank=rank, dp_size=16
        )
        for rank in range(16)
    ]
    np.testing.assert_array_equal(sum(isolated), cotangent)

  def test_replica_gate_rejects_rank_dependent_result(self):
    replicas = np.broadcast_to(np.arange(4, dtype=np.float32), (16, 4)).copy()
    self.assertEqual(
        dp_training.assert_dp_replicas_equal(
            replicas, dp_size=16, label='gradient'
        ),
        {'dp_replicas': 16, 'mismatched_replicas': 0},
    )
    replicas[7, 2] += np.float32(1.0)
    with self.assertRaisesRegex(ValueError, 'DP ranks'):
      dp_training.assert_dp_replicas_equal(
          replicas, dp_size=16, label='gradient'
      )

  def test_dp16_collective_is_exact_on_all_replicas(self):
    if len(jax.devices()) != 64:
      self.skipTest('requires exactly 64 forced CPU or accelerator devices')
    mesh = Mesh(np.asarray(jax.devices()).reshape(16, 4), ('dp', 'tp'))

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=P('dp', 'tp'),
        out_specs=P('dp', 'tp'),
        check_vma=False,
    )
    def reduce_rows(value):
      return dp_training.fixed_dp_collective(value, dp_size=16)

    values = np.arange(64, dtype=np.float32).reshape(16, 4)
    values[0, 0] = np.float32(1.0e8)
    values[1, 0] = np.float32(1.0)
    values[2, 0] = np.float32(-1.0e8)
    values[3, 0] = np.float32(2.0)
    compiled = jax.jit(reduce_rows)
    stablehlo = str(
        compiled.lower(jnp.asarray(values)).compiler_ir(dialect='stablehlo')
    )
    self.assertEqual(stablehlo.count('stablehlo.collective_permute'), 8)
    result = compiled(jnp.asarray(values))
    expected = np.stack(
        [
            np.asarray(
                dp_training.fixed_dp_sum(
                    [jnp.asarray(values[rank, column]) for rank in range(16)]
                )
            )
            for column in range(4)
        ]
    )
    host = np.asarray(result)
    np.testing.assert_array_equal(host, np.broadcast_to(expected, (16, 4)))
    dp_training.assert_dp_replicas_equal(
        host, dp_size=16, label='post-reduction gradient'
    )

  def test_rank_gradient_reducer_stages_one_contribution_per_dp_rank(self):
    if len(jax.devices()) != 64:
      self.skipTest('requires exactly 64 forced CPU or accelerator devices')
    mesh = Mesh(np.asarray(jax.devices()).reshape(16, 4), ('dp', 'tp'))
    sharding = jax.sharding.NamedSharding(mesh, P('tp'))
    template = {
        'weight': jax.device_put(jnp.zeros((8,), jnp.float32), sharding)
    }
    reducer = dp_training.FixedDPRankGradientReducer(
        template, dp_size=16
    )
    reducer.begin()
    for rank in range(16):
      contribution = {
          'weight': jax.device_put(
              jnp.full((8,), rank + 1, jnp.float32), sharding
          )
      }
      reducer.add(rank, contribution)
    reduced, report = reducer.finalize()
    np.testing.assert_array_equal(
        np.asarray(reduced['weight']), np.full((8,), 136.0, np.float32)
    )
    self.assertEqual(report['rank_contributions'], 16)
    self.assertLen(set(report['rank_local_fingerprints']), 16)
    self.assertEqual(report['rank_local_fingerprint_unique_count'], 16)
    self.assertEqual(report['rank_local_fingerprint_duplicate_count'], 0)
    self.assertTrue(report['rank_local_fingerprints_distinct'])
    self.assertEqual(report['reduction_transactions'], 1)
    self.assertEqual(report['reduction_rounds'], 8)
    self.assertEqual(report['replica_check_flags'], 16)
    self.assertTrue(report['post_reduction_all_finite'])
    self.assertTrue(report['post_reduction_replicas_exact'])

  def test_rank_gradient_reducer_accepts_duplicate_production_gradients(self):
    if len(jax.devices()) < 4:
      self.skipTest('requires at least four forced CPU or accelerator devices')
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2), ('data', 'model')
    )
    sharding = jax.sharding.NamedSharding(mesh, P('model'))
    template = jax.device_put(jnp.zeros((8,), jnp.float32), sharding)
    reducer = dp_training.FixedDPRankGradientReducer(
        template,
        dp_size=2,
        dp_axis='data',
        require_distinct_fingerprints=False,
    )
    reducer.begin()
    for rank in range(2):
      reducer.add(rank, template)
    reduced, report = reducer.finalize()
    np.testing.assert_array_equal(
        np.asarray(reduced), np.zeros((8,), np.float32)
    )
    self.assertEqual(report['rank_contributions'], 2)
    self.assertEqual(report['rank_local_fingerprint_unique_count'], 1)
    self.assertEqual(report['rank_local_fingerprint_duplicate_count'], 1)
    self.assertFalse(report['rank_local_fingerprints_distinct'])
    self.assertTrue(report['post_reduction_replicas_exact'])

  def test_rank_gradient_reducer_keeps_strict_probe_mode(self):
    if len(jax.devices()) < 4:
      self.skipTest('requires at least four forced CPU or accelerator devices')
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2), ('data', 'model')
    )
    sharding = jax.sharding.NamedSharding(mesh, P('model'))
    template = jax.device_put(jnp.zeros((8,), jnp.float32), sharding)
    reducer = dp_training.FixedDPRankGradientReducer(
        template, dp_size=2, dp_axis='data'
    )
    reducer.begin()
    for rank in range(2):
      reducer.add(rank, template)
    with self.assertRaisesRegex(ValueError, 'fingerprints are not distinct'):
      reducer.finalize()

  def test_rank_gradient_reducer_accepts_explicit_data_axis(self):
    if len(jax.devices()) < 4:
      self.skipTest('requires at least four forced CPU or accelerator devices')
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2), ('data', 'model')
    )
    sharding = jax.sharding.NamedSharding(mesh, P('model'))
    template = jax.device_put(jnp.zeros((8,), jnp.float32), sharding)
    reducer = dp_training.FixedDPRankGradientReducer(
        template, dp_size=2, dp_axis='data'
    )
    reducer.begin()
    for rank in range(2):
      reducer.add(
          rank,
          jax.device_put(
              jnp.full((8,), rank + 1, jnp.float32), sharding
          ),
      )
    reduced, report = reducer.finalize()
    np.testing.assert_array_equal(
        np.asarray(reduced), np.full((8,), 3.0, np.float32)
    )
    self.assertEqual(report['dp_axis'], 'data')
    self.assertEqual(report['rank_contributions'], 2)
    self.assertEqual(report['reduction_rounds'], 2)
    self.assertTrue(report['post_reduction_replicas_exact'])

  def test_rank_gradient_reducer_consumes_parallel_staged_table(self):
    if len(jax.devices()) < 4:
      self.skipTest('requires at least four forced CPU or accelerator devices')
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2), ('data', 'model')
    )
    template_sharding = jax.sharding.NamedSharding(mesh, P('model'))
    staged_sharding = jax.sharding.NamedSharding(
        mesh, P('data', 'model')
    )
    template = jax.device_put(jnp.zeros((8,), jnp.float32), template_sharding)
    staged = jax.device_put(
        jnp.stack(
            (
                jnp.arange(8, dtype=jnp.float32) + 1.0,
                jnp.arange(8, dtype=jnp.float32) + 11.0,
            )
        ),
        staged_sharding,
    )
    reducer = dp_training.FixedDPRankGradientReducer(
        template, dp_size=2, dp_axis='data'
    )
    reduced, report = reducer.finalize_staged(staged)
    np.testing.assert_array_equal(
        np.asarray(reduced), np.arange(8, dtype=np.float32) * 2.0 + 12.0
    )
    self.assertEqual(report['rank_contributions'], 2)
    self.assertEqual(report['rank_gradient_staging_mode'], 'parallel_table')
    self.assertLen(set(report['rank_local_fingerprints']), 2)
    self.assertEqual(report['reduction_rounds'], 2)
    self.assertTrue(report['post_reduction_replicas_exact'])

  def test_dp8_tp8_rank_gradient_reducer_consumes_finite_staged_table(self):
    if len(jax.devices()) != 64:
      self.skipTest('requires exactly 64 forced CPU or accelerator devices')
    mesh = Mesh(np.asarray(jax.devices()).reshape(8, 8), ('data', 'model'))
    template_sharding = jax.sharding.NamedSharding(mesh, P('model'))
    staged_sharding = jax.sharding.NamedSharding(
        mesh, P('data', 'model')
    )
    template = jax.device_put(jnp.zeros((32,), jnp.float32), template_sharding)
    staged_host = np.stack(
        [np.arange(32, dtype=np.float32) + 10.0 * rank for rank in range(8)]
    )
    staged = jax.device_put(staged_host, staged_sharding)
    reducer = dp_training.FixedDPRankGradientReducer(
        template, dp_size=8, dp_axis='data'
    )
    reduced, report = reducer.finalize_staged(staged)
    np.testing.assert_array_equal(np.asarray(reduced), staged_host.sum(axis=0))
    self.assertEqual(report['dp_size'], 8)
    self.assertEqual(report['reduction_rounds'], 6)
    self.assertTrue(report['post_reduction_all_finite'])
    self.assertTrue(report['post_reduction_replicas_exact'])

  def test_parallel_staged_common_nan_is_nonfinite_not_replica_mismatch(self):
    if len(jax.devices()) < 4:
      self.skipTest('requires at least four forced CPU or accelerator devices')
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2), ('data', 'model')
    )
    template_sharding = jax.sharding.NamedSharding(mesh, P('model'))
    staged_sharding = jax.sharding.NamedSharding(
        mesh, P('data', 'model')
    )
    template = jax.device_put(jnp.zeros((8,), jnp.float32), template_sharding)
    staged_host = np.ones((2, 8), np.float32)
    staged_host[:, 3] = np.nan
    staged = jax.device_put(staged_host, staged_sharding)
    reducer = dp_training.FixedDPRankGradientReducer(
        template,
        dp_size=2,
        dp_axis='data',
        require_distinct_fingerprints=False,
    )
    with self.assertRaisesRegex(
        ValueError, 'staged DP gradient contains non-finite values'
    ) as context:
      reducer.finalize_staged(staged)
    self.assertNotIn('unequal replicas', str(context.exception))

  def test_finite_replica_mismatch_remains_fatal(self):
    if len(jax.devices()) < 4:
      self.skipTest('requires at least four forced CPU or accelerator devices')
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2), ('data', 'model')
    )
    template_sharding = jax.sharding.NamedSharding(mesh, P('model'))
    staged_sharding = jax.sharding.NamedSharding(
        mesh, P('data', 'model')
    )
    template = jax.device_put(jnp.zeros((8,), jnp.float32), template_sharding)
    staged = jax.device_put(jnp.ones((2, 8), jnp.float32), staged_sharding)
    reducer = dp_training.FixedDPRankGradientReducer(
        template,
        dp_size=2,
        dp_axis='data',
        require_distinct_fingerprints=False,
    )
    reducer._compare = lambda _: jnp.asarray([True, False])
    with self.assertRaisesRegex(ValueError, 'unequal replicas'):
      reducer.finalize_staged(staged)

  def test_rank_gradient_reducer_rejects_unsharded_parallel_table(self):
    if len(jax.devices()) < 4:
      self.skipTest('requires at least four forced CPU or accelerator devices')
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2), ('data', 'model')
    )
    template_sharding = jax.sharding.NamedSharding(mesh, P('model'))
    replicated_sharding = jax.sharding.NamedSharding(mesh, P())
    template = jax.device_put(jnp.zeros((8,), jnp.float32), template_sharding)
    staged = jax.device_put(
        jnp.zeros((2, 8), jnp.float32), replicated_sharding
    )
    reducer = dp_training.FixedDPRankGradientReducer(
        template,
        dp_size=2,
        dp_axis='data',
        require_distinct_fingerprints=False,
    )
    with self.assertRaisesRegex(ValueError, 'sharding changed'):
      reducer.finalize_staged(staged)

  def test_dp2_tp2_rank_parallel_vjp_matches_serial_rank_isolation(self):
    if len(jax.devices()) < 4:
      self.skipTest('requires at least four forced CPU or accelerator devices')
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2), ('data', 'model')
    )
    weight = jax.device_put(
        jnp.arange(24, dtype=jnp.float32).reshape(4, 6) / 17.0,
        jax.sharding.NamedSharding(mesh, P(None, 'model')),
    )
    values = jax.device_put(
        jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 13.0,
        jax.sharding.NamedSharding(mesh, P('data')),
    )
    cotangent = jax.device_put(
        jnp.arange(36, dtype=jnp.float32).reshape(2, 3, 6) / 19.0,
        jax.sharding.NamedSharding(mesh, P('data', None, 'model')),
    )

    def forward(local_weight, local_values):
      return jnp.einsum('bsk,kh->bsh', local_values, local_weight)

    _, global_pullback = jax.vjp(forward, weight, values)
    serial_rows = jnp.stack([
        global_pullback(
            dp_training.isolate_dp_rank_cotangent(
                cotangent, rank=rank, dp_size=2
            )
        )[0]
        for rank in range(2)
    ])
    full_weight_gradient, full_value_gradient = global_pullback(cotangent)

    def local_pullback(local_weight, local_values, local_cotangent):
      _, pullback = jax.vjp(forward, local_weight, local_values)
      weight_gradient, value_gradient = pullback(local_cotangent)
      return jnp.expand_dims(weight_gradient, 0), value_gradient

    parallel_pullback = jax.shard_map(
        local_pullback,
        mesh=mesh,
        in_specs=(P(), P('data'), P('data')),
        out_specs=(P('data'), P('data')),
        axis_names={'data'},
        check_vma=False,
    )
    staged, value_gradient = jax.jit(parallel_pullback)(
        weight, values, cotangent
    )
    np.testing.assert_array_equal(np.asarray(staged), np.asarray(serial_rows))
    np.testing.assert_array_equal(
        np.asarray(value_gradient), np.asarray(full_value_gradient)
    )

    reducer = dp_training.FixedDPRankGradientReducer(
        full_weight_gradient, dp_size=2, dp_axis='data'
    )
    reduced, report = reducer.finalize_staged(staged)
    np.testing.assert_array_equal(
        np.asarray(reduced), np.asarray(full_weight_gradient)
    )
    self.assertEqual(report['rank_gradient_staging_mode'], 'parallel_table')

    serial_host = np.asarray(serial_rows)
    perturbed = serial_host.copy()
    perturbed[1, 0, 0] = np.nextafter(
        perturbed[1, 0, 0], np.float32(np.inf), dtype=np.float32
    )
    self.assertFalse(np.array_equal(serial_host, perturbed))

  def test_rank_gradient_reducer_rejects_rank_cadence_fault(self):
    if len(jax.devices()) != 64:
      self.skipTest('requires exactly 64 forced CPU or accelerator devices')
    mesh = Mesh(np.asarray(jax.devices()).reshape(16, 4), ('dp', 'tp'))
    sharding = jax.sharding.NamedSharding(mesh, P('tp'))
    template = jax.device_put(jnp.ones((8,), jnp.float32), sharding)
    reducer = dp_training.FixedDPRankGradientReducer(
        template, dp_size=16, require_distinct_fingerprints=False
    )
    reducer.begin()
    reducer.add(0, template)
    with self.assertRaisesRegex(ValueError, 'expected rank 1, got 0'):
      reducer.add(0, template)
    with self.assertRaisesRegex(ValueError, 'missing rank contributions'):
      reducer.finalize()


if __name__ == '__main__':
  absltest.main()
