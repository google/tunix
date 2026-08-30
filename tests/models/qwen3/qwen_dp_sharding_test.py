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


if __name__ == '__main__':
  absltest.main()


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
