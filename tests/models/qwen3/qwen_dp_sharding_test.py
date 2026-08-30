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

"""Tests for Qwen3 replicated-DP sharding metadata."""

from absl.testing import absltest
import numpy as np

from tunix.models.qwen3 import model


class QwenDPShardingTest(absltest.TestCase):

  def test_parameters_are_replicated_over_dp(self):
    config = model.ShardingConfig.get_data_parallel_sharding()
    parameter_specs = (
        config.emb_vd,
        config.emb_dv,
        config.q_weight_dnh,
        config.kv_weight_dnh,
        config.o_weight_nhd,
        config.ffw_weight_df,
        config.ffw_weight_fd,
        config.rms_norm_weight,
        config.score_weight_d1,
        config.exp_weight_edf,
        config.exp_weight_efd,
    )
    self.assertNotIn('dp', repr(parameter_specs))
    self.assertEqual(config.act_btd[0], 'dp')
    self.assertEqual(config.act_btf[0], 'dp')
    self.assertEqual(config.act_btnh[0], 'dp')

  def test_rejects_ambiguous_data_axis(self):
    for axis in ('', 'fsdp', 'tp'):
      with self.subTest(axis=axis):
        with self.assertRaises(ValueError):
          model.ShardingConfig.get_data_parallel_sharding(axis)


class EmbedderGatherOutShardingTest(absltest.TestCase):
  """The embedder gather must name its output sharding on a DP x TP mesh.

  With emb_vd=P(tp, None) the vocabulary rows live on different model-axis
  devices than the data-axis-sharded ids that index them, so the gather's
  output sharding is ambiguous and JAX refuses to infer it.  These tests pin
  the recipe's answer and the guards that keep single-device paths untouched.
  """

  def test_helper_returns_none_without_a_mesh_or_on_cpu(self):
    config = model.ShardingConfig.get_data_parallel_sharding(
        data_axis='data', tp_axis='model'
    )
    # No physical mesh is bound in this process, and the backend is CPU:
    # either guard alone must keep the plain gather.
    self.assertIsNone(model._activation_out_sharding(config.act_btd))

  def test_helper_builds_the_activation_spec_under_a_mesh(self):
    import unittest.mock as mock  # pylint: disable=g-import-not-at-top

    import jax  # pylint: disable=g-import-not-at-top
    import jax.sharding as shd  # pylint: disable=g-import-not-at-top

    config = model.ShardingConfig.get_data_parallel_sharding(
        data_axis='data', tp_axis='model'
    )
    devices = jax.devices()
    mesh = shd.Mesh(
        np.asarray(devices[:1]).reshape(1, 1), axis_names=('data', 'model')
    )
    fake_env = mock.MagicMock()
    fake_env.physical_mesh = mesh
    fake_device = mock.MagicMock()
    fake_device.platform = 'tpu'
    with mock.patch.object(
        model.pxla.thread_resources, 'env', fake_env
    ), mock.patch.object(jax, 'devices', return_value=[fake_device]):
      out = model._activation_out_sharding(config.act_btd)
    self.assertIsInstance(out, shd.NamedSharding)
    self.assertEqual(out.spec, shd.PartitionSpec('data', None, 'model'))

  def test_gather_with_out_sharding_matches_the_plain_gather(self):
    """The API the fix uses returns the same values as indexing."""
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    import jax.sharding as shd  # pylint: disable=g-import-not-at-top

    table = jnp.arange(12 * 4, dtype=jnp.float32).reshape(12, 4)
    ids = jnp.asarray([[0, 5, 11], [3, 3, 7]], jnp.int32)
    mesh = shd.Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1),
        axis_names=('data', 'model'),
    )
    spec = shd.NamedSharding(mesh, shd.PartitionSpec(None, None, None))
    np.testing.assert_array_equal(
        table.at[(ids,)].get(out_sharding=spec), table[(ids,)]
    )


class ExplicitAxisShardTest(absltest.TestCase):
  """`shard` must speak the resharding API on an Explicit-axis mesh.

  Sharding-in-types meshes carry the sharding in the avals, so
  `with_sharding_constraint` degenerates to an assert there and rejects a
  spec naming those axes outright; the DP x TP recipe therefore has to state
  the same intent through `reshard`.
  """

  def _explicit_mesh(self):
    import jax  # pylint: disable=g-import-not-at-top
    import jax.sharding as shd  # pylint: disable=g-import-not-at-top

    devices = jax.devices()
    if len(devices) < 2:
      self.skipTest(
          'needs >= 2 devices; run with '
          'XLA_FLAGS=--xla_force_host_platform_device_count=8'
      )
    grid = np.asarray(devices[:2]).reshape(2, 1)
    return shd.Mesh(
        grid, ('data', 'model'), axis_types=(shd.AxisType.Explicit,) * 2
    )

  def test_detects_explicit_axes(self):
    import jax  # pylint: disable=g-import-not-at-top
    import jax.sharding as shd  # pylint: disable=g-import-not-at-top

    explicit = self._explicit_mesh()
    self.assertTrue(model._mesh_has_explicit_axes(explicit))
    auto = shd.Mesh(
        np.asarray(jax.devices()[:2]).reshape(2, 1), ('data', 'model')
    )
    self.assertFalse(model._mesh_has_explicit_axes(auto))

  def test_reshard_replaces_the_rejected_constraint(self):
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    import jax.sharding as shd  # pylint: disable=g-import-not-at-top

    mesh = self._explicit_mesh()
    spec = shd.NamedSharding(mesh, shd.PartitionSpec('data', None, 'model'))
    x = jnp.arange(2 * 3 * 1, dtype=jnp.float32).reshape(2, 3, 1)
    jax.sharding.set_mesh(mesh)
    try:
      # The rejection is an AssertionError when every axis is Explicit and a
      # ValueError when the mesh mixes types (the shape seen in the failing
      # cluster run); both mean the constraint form is unusable here.
      with self.assertRaises((AssertionError, ValueError)):
        jax.lax.with_sharding_constraint(x, spec)
      resharded = jax.sharding.reshard(x, spec)
      self.assertEqual(resharded.sharding.spec, spec.spec)
      np.testing.assert_array_equal(np.asarray(resharded), np.asarray(x))
    finally:
      # Clearing is the documented way back to no bound mesh; passing the
      # abstract mesh that get_abstract_mesh returns is rejected.
      jax.sharding.set_mesh(None)


class ExplicitSplashKernelShardingTest(absltest.TestCase):
  """Splash kernel inputs must already match shard_map on Explicit meshes."""

  def _mesh(self, *, explicit: bool):
    import jax  # pylint: disable=g-import-not-at-top
    import jax.sharding as shd  # pylint: disable=g-import-not-at-top

    devices = jax.devices()
    if len(devices) < 8:
      self.skipTest(
          'needs 8 devices; run with '
          'XLA_FLAGS=--xla_force_host_platform_device_count=8'
      )
    kwargs = {}
    if explicit:
      kwargs['axis_types'] = (shd.AxisType.Explicit,) * 2
    return shd.Mesh(
        np.asarray(devices[:8]).reshape(2, 4),
        ('data', 'model'),
        **kwargs,
    )

  def _kernel_and_specs(self, mesh):
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash,  # pylint: disable=g-import-not-at-top,g-importing-member
    )
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_mask as mask_lib,  # pylint: disable=g-import-not-at-top,g-importing-member
    )
    import jax.sharding as shd  # pylint: disable=g-import-not-at-top

    causal_mask = mask_lib.CausalMask((128, 128))
    multi_head_mask = mask_lib.MultiHeadMask([causal_mask for _ in range(4)])
    block_sizes = splash.BlockSizes(
        block_q=128,
        block_kv=128,
        block_q_dkv=128,
        block_kv_dkv=128,
        block_kv_dkv_compute=128,
        block_q_dq=128,
        block_kv_dq=128,
    )
    kernel = splash.make_splash_mha(
        multi_head_mask,
        block_sizes=block_sizes,
        head_shards=4,
        q_seq_shards=1,
    )
    specs = kernel.manual_sharding_spec(
        shd.NamedSharding(mesh, shd.PartitionSpec('model', None))
    )
    return kernel, specs

  def test_real_kernel_negative_and_explicit_reshard(self):
    import jax  # pylint: disable=g-import-not-at-top
    import jax.sharding as shd  # pylint: disable=g-import-not-at-top
    from jax.experimental.shard_map import shard_map  # pylint: disable=g-import-not-at-top,g-importing-member

    mesh = self._mesh(explicit=True)
    kernel, specs = self._kernel_and_specs(mesh)
    def is_spec(value):
      return isinstance(value, shd.PartitionSpec)

    kernel_leaves = jax.tree_util.tree_leaves(kernel)
    spec_leaves = jax.tree_util.tree_leaves(specs, is_leaf=is_spec)
    self.assertLen(kernel_leaves, len(spec_leaves))
    self.assertGreater(len(kernel_leaves), 0)

    # Reproduce the production admission failure with a real Splash mask leaf:
    # its value is replicated while shard_map expects a model-sharded input.
    first = jax.device_put(
        kernel_leaves[0], shd.NamedSharding(mesh, shd.PartitionSpec())
    )
    first_spec = spec_leaves[0]
    identity = shard_map(
        lambda value: value,
        mesh=mesh,
        in_specs=first_spec,
        out_specs=first_spec,
        check_rep=False,
    )
    jax.sharding.set_mesh(mesh)
    try:
      with self.assertRaisesRegex(ValueError, 'does not match the specs'):
        identity(first)

      placed = model._reshard_splash_kernel_for_explicit_mesh(
          kernel, specs, mesh
      )
      placed_leaves = jax.tree_util.tree_leaves(placed)
      for before, after, spec in zip(
          kernel_leaves, placed_leaves, spec_leaves, strict=True
      ):
        expected_spec = shd.PartitionSpec(
            *tuple(spec), *(None,) * (after.ndim - len(spec))
        )
        self.assertEqual(after.sharding.spec, expected_spec)
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))

      # The formerly rejected real leaf now satisfies the exact same map.
      np.testing.assert_array_equal(
          np.asarray(identity(placed_leaves[0])),
          np.asarray(kernel_leaves[0]),
      )
    finally:
      jax.sharding.set_mesh(None)

  def test_auto_mesh_keeps_the_historical_kernel_object(self):
    mesh = self._mesh(explicit=False)
    kernel, specs = self._kernel_and_specs(mesh)
    self.assertIs(
        model._reshard_splash_kernel_for_explicit_mesh(kernel, specs, mesh),
        kernel,
    )


if __name__ == '__main__':
  absltest.main()
