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

"""Focused Qwen3 forward tests."""

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from tunix.models.qwen3 import model as qwen3_model


class Qwen3RopeThetaTest(absltest.TestCase):

  def test_attention_uses_configured_rope_theta(self):
    def attention_output(rope_theta):
      config = qwen3_model.ModelConfig(
          num_layers=1,
          vocab_size=32,
          embed_dim=16,
          hidden_dim=32,
          num_heads=2,
          head_dim=8,
          num_kv_heads=1,
          rope_theta=rope_theta,
          norm_eps=1e-6,
      )
      attention = qwen3_model.Attention(config, rngs=nnx.Rngs(42))
      x = jax.random.normal(jax.random.key(7), (1, 4, 16))
      positions = jnp.array([[0, 1024, 2048, 3072]], dtype=jnp.int32)
      mask = jnp.tril(jnp.ones((1, 4, 4), dtype=jnp.bool_))
      _, output = attention.block(x, positions, None, mask)
      return np.asarray(output)

    output_1m = attention_output(1_000_000)
    output_5m = attention_output(5_000_000)
    self.assertGreater(np.max(np.abs(output_1m - output_5m)), 1e-4)


if __name__ == "__main__":
  absltest.main()
