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

from absl.testing import absltest
import chex
from flax import nnx
import jax
from jax import sharding
import numpy as np
from tunix.rl import utils
from tunix.tests import test_common as tc


class UtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.num_cpus = 4
    chex.set_n_cpu_devices(self.num_cpus)
    assert len(jax.devices()) == self.num_cpus

  def test_get_pytree_mesh_info(self):
    mesh1 = sharding.Mesh(
        np.array(jax.devices()[: self.num_cpus // 2]).reshape(1, 2),
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
        np.array(jax.devices()[self.num_cpus // 2 :]).reshape(1, 2),
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


if __name__ == '__main__':
  absltest.main()
