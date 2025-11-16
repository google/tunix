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

import os
from absl.testing import absltest
import chex
from flax import nnx
import jax
from jax import sharding
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.rl import utils
from tunix.tests import test_common as tc

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'


class UtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.num_cpus = 4
    chex.set_n_cpu_devices(self.num_cpus)
    self.device_count = jax.device_count()

  def test_get_pytree_mesh_info(self):
    mesh1 = sharding.Mesh(
        np.array(jax.devices()[: self.device_count // 2]).reshape(
            1, self.device_count // 2
        ),
        ('fsdp', 'tp'),
    )
    model1 = tc.get_lora_model(
        tc.ToyTransformer(
            rngs=nnx.Rngs(0), vocab_size=tc.MockVocab().GetPieceSize()
        ),
        mesh=mesh1,
    )
    self.assertEqual(utils.get_pytree_mesh_info(nnx.state(model1)), mesh1)

    mesh2 = sharding.Mesh(
        np.array(jax.devices()[self.device_count // 2 :]).reshape(
            1, self.device_count // 2
        ),
        ('fsdp', 'tp'),
    )
    model2 = tc.get_lora_model(
        tc.ToyTransformer(
            rngs=nnx.Rngs(0), vocab_size=tc.MockVocab().GetPieceSize()
        ),
        mesh=mesh2,
    )
    self.assertEqual(utils.get_pytree_mesh_info(nnx.state(model2)), mesh2)

    self.assertNotEqual(mesh1, mesh2)

    model3 = tc.get_lora_model(
        tc.ToyTransformer(
            rngs=nnx.Rngs(0), vocab_size=tc.MockVocab().GetPieceSize()
        ),
    )
    self.assertIsNone(utils.get_pytree_mesh_info(nnx.state(model3)))

  def test_is_sharing_weights(self):
    m1 = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=tc.MockVocab().GetPieceSize()
    )
    m2 = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=tc.MockVocab().GetPieceSize()
    )
    m3 = nnx.clone(m1)
    self.assertIsNot(nnx.state(m1), nnx.state(m2))
    self.assertIsNot(nnx.state(m1), nnx.state(m3))
    self.assertIsNot(nnx.state(m2), nnx.state(m3))
    self.assertFalse(utils.is_sharing_weights(m1, m2))
    self.assertFalse(utils.is_sharing_weights(m2, m3))
    self.assertTrue(utils.is_sharing_weights(m1, m3))

  def test_chunk_slices_by_size(self):
    x = [0, 1, 2, 3, 4]
    y = [x[s] for s in utils.chunk_slices_by_size(stop=len(x), step=2)]
    self.assertEqual(y, [[0, 1], [2, 3], [4]])

  def test_get_batch_slice(self):
    x = {
        'a': np.array([[1], [2], [3], [4], [5], [6]]),
        'b': {'c': np.array([[7], [8], [9], [10], [11], [12]])},
    }
    y = [
        utils.get_batch_slice(x, s)
        for s in utils.chunk_slices_by_size(stop=6, step=2)
    ]
    expected = [
        {'a': np.array([[1], [2]]), 'b': {'c': np.array([[7], [8]])}},
        {'a': np.array([[3], [4]]), 'b': {'c': np.array([[9], [10]])}},
        {'a': np.array([[5], [6]]), 'b': {'c': np.array([[11], [12]])}},
    ]
    jax.tree_util.tree_map(np.testing.assert_array_equal, expected, y)

  def test_merge_micro_batches(self):
    batches = [
        {
            'a': [1, 2],
            'b': {'c': np.array([3, 4]), 'd': np.array([5])},
            'e': np.array([6, 7]),
        },
        {
            'a': [10, 11],
            'b': {'c': np.array([12, 13]), 'd': np.array([14, 15])},
            'e': np.array([16]),
        },
    ]
    merged = utils.merge_micro_batches(batches)
    self.assertEqual(merged['a'], [1, 2, 10, 11])
    jax.tree_util.tree_map(
        np.testing.assert_array_equal,
        merged['b'],
        {'c': np.array([3, 4, 12, 13]), 'd': np.array([5, 14, 15])},
    )
    jax.tree_util.tree_map(
        np.testing.assert_array_equal, merged['e'], np.array([6, 7, 16])
    )

  def test_create_critic_model(self):
    actor_model = tc.ToyTransformer(
        rngs=nnx.Rngs(0), vocab_size=tc.MockVocab().GetPieceSize()
    )
    critic_model = utils.create_critic_model(actor_model)

    x = jnp.array([[1, 2, 3], [4, 5, 6]])
    positions = jnp.arange(x.shape[1])
    attn_mask = common.make_causal_attn_mask(jnp.ones_like(x))
    out, _ = critic_model(x, positions, None, attn_mask)
    self.assertEqual(out.shape, (2, 3, 1))


if __name__ == '__main__':
  absltest.main()
