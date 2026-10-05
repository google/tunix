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

"""Tests for Qwen3 MoE dispatch path selection."""

import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.models.qwen3 import model as qwen3_model


class _MegabloxCalled(Exception):
  pass


class MoEDispatchTest(parameterized.TestCase):

  def _layer(self):
    config = qwen3_model.ModelConfig(
        num_layers=1,
        vocab_size=32,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        head_dim=8,
        num_kv_heads=2,
        rope_theta=10_000,
        norm_eps=1e-6,
        num_experts=4,
        num_experts_per_tok=2,
    )
    return qwen3_model.MoELayer(config, rngs=nnx.Rngs(0))

  def _run(self, platform):
    layer = self._layer()
    x = jnp.ones((1, 4, 16), dtype=jnp.float32)
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1), ('fsdp', 'tp')
    )
    fake_device = types.SimpleNamespace(platform=platform)
    with mesh, mock.patch.object(
        jax, 'devices', return_value=[fake_device]
    ), mock.patch.object(
        qwen3_model.megablox, 'gmm', side_effect=_MegabloxCalled
    ):
      return layer(x)

  @parameterized.parameters('gpu', 'cpu')
  def test_non_tpu_uses_dense_fallback(self, platform):
    out = self._run(platform)
    self.assertEqual(out.shape, (1, 4, 16))

  def test_tpu_uses_megablox(self):
    with self.assertRaises(_MegabloxCalled):
      self._run('tpu')


if __name__ == '__main__':
  absltest.main()
