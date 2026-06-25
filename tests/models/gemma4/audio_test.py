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

"""Tests for audio."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from tunix.models.gemma4 import audio


class AudioTest(parameterized.TestCase):

  def test_audio_encoder_shape(self):
    config = audio.AudioEncoderConfig(
        num_layers=2,  # fewer layers for faster test
        model_dims=128,
        lm_model_dims=256,
        atten_num_heads=4,
    )

    rngs = nnx.Rngs(0)
    model = audio.AudioEncoder(config, rngs=rngs)

    batch_size = 2
    num_samples = 16000  # 1 second

    x = jax.random.normal(jax.random.PRNGKey(1), (batch_size, num_samples))
    sequence_lengths = jnp.full((batch_size,), num_samples, dtype=jnp.int32)

    output, mask = model(x, sequence_lengths)

    expected_seq_len = 25
    expected_shape = (batch_size, expected_seq_len, config.lm_model_dims)

    self.assertEqual(output.shape, expected_shape)
    self.assertEqual(mask.shape, (batch_size, expected_seq_len))
    self.assertFalse(jnp.any(jnp.isnan(output)))


if __name__ == "__main__":
  absltest.main()
