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

"""Tests for the Gemma 3 model."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax import nnx

from tunix.models.gemma3 import model as gemma3_model


def _make_tiny_config() -> gemma3_model.ModelConfig:
  return gemma3_model.ModelConfig(
      num_layers=1,
      num_embed=32,
      embed_dim=16,
      hidden_dim=32,
      num_heads=2,
      head_dim=8,
      num_kv_heads=1,
      sliding_window_size=8,
      shd_config=gemma3_model.ShardingConfig.get_default_sharding(),
      param_dtype=jnp.float32,
  )


class Gemma3ModelTest(absltest.TestCase):

  def test_skip_lm_head_records_hidden_states_without_logits(self):
    model = gemma3_model.Gemma3(_make_tiny_config(), rngs=nnx.Rngs(0))
    tokens = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    positions = jnp.arange(tokens.shape[1], dtype=jnp.int32)[None, :]
    attention_mask = model.get_attention_mask(tokens)

    hidden_states, cache = model(
        tokens,
        positions,
        None,
        attention_mask,
        output_hidden_states=True,
        skip_lm_head=True,
    )

    self.assertIsNone(cache)
    self.assertEqual(hidden_states.shape, (1, 3, model.config.embed_dim))
    intermediate = nnx.pop(model, nnx.Intermediate)
    np.testing.assert_allclose(
        np.asarray(intermediate['all_hidden_states'].value[0]),
        np.asarray(hidden_states),
    )

  def test_skip_lm_head_matches_logits_before_projection(self):
    model = gemma3_model.Gemma3(_make_tiny_config(), rngs=nnx.Rngs(0))
    tokens = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    positions = jnp.arange(tokens.shape[1], dtype=jnp.int32)[None, :]
    attention_mask = model.get_attention_mask(tokens)

    hidden_states, _ = model(
        tokens, positions, None, attention_mask, skip_lm_head=True
    )
    logits = model.compute_final_logits(hidden_states)

    expected_logits, _ = model(tokens, positions, None, attention_mask)
    np.testing.assert_allclose(
        np.asarray(logits), np.asarray(expected_logits), rtol=1e-5, atol=1e-5
    )

  def test_headless_jit_does_not_trace_vocabulary_projection(self):
    model = gemma3_model.Gemma3(_make_tiny_config(), rngs=nnx.Rngs(0))
    tokens = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    positions = jnp.arange(tokens.shape[1], dtype=jnp.int32)[None, :]
    attention_mask = model.get_attention_mask(tokens)

    graphdef, state = nnx.split(model)

    def run_headless(state, tokens, positions, attention_mask):
      model_local = nnx.merge(graphdef, state)
      hidden_states, _ = model_local(
          tokens,
          positions,
          None,
          attention_mask,
          output_hidden_states=True,
          skip_lm_head=True,
      )
      return hidden_states

    jaxpr = str(
        jax.make_jaxpr(run_headless)(state, tokens, positions, attention_mask)
    )
    self.assertNotIn('embedder_decode', jaxpr)
    self.assertEqual(
        nnx.jit(run_headless)(state, tokens, positions, attention_mask).shape,
        (1, 3, model.config.embed_dim),
    )


if __name__ == '__main__':
  absltest.main()
