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

"""Equivalence test for Gemma 4 against reference implementation."""

from absl.testing import absltest
import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.models.gemma4 import model as tunix_model
from tunix.models.gemma4 import params as tunix_params


# Attempt to import upstream gemma 4
from gemma.gm.nn.gemma4 import _config as up_config
from gemma.gm.nn.gemma4 import _modules as up_modules
from gemma.gm.nn.gemma4 import _transformer as up_transformer





class EquivalenceTest(absltest.TestCase):

  def test_logits_match(self):


    vocab_size = 1000
    embed_dim = 128
    hidden_dim = 256
    num_heads = 4
    head_dim = 32
    num_kv_heads = 1

    up_cfg = up_config.TransformerConfig(
        num_embed=vocab_size,

        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        final_logit_softcap=None,

        use_post_attn_norm=True,
        use_post_ffw_norm=True,
        attention_types=(
            up_modules.AttentionType.LOCAL_SLIDING,
        ),

        global_rope_proportion=1.0,
        local_rope_proportion=1.0,
        sliding_window_size=512,
    )

    tunix_cfg = tunix_model.ModelConfig(
        num_layers=1,
        num_embed=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        sliding_window_size=512,
    )

    upstream = up_transformer.Transformer(config=up_cfg)

    rng = jax.random.PRNGKey(0)
    tokens = jax.random.randint(rng, (2, 32), 0, vocab_size)
    init_vars = upstream.init(rng, tokens)

    mapped_params = tunix_params.map_from_upstream_checkpoint(
        init_vars["params"]
    )



    rngs = nnx.Rngs(0)
    B, T = tokens.shape
    causal_mask = jnp.tile(jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))[None, ...], (B, 1, 1))


    model = tunix_model.Gemma4(tunix_cfg, rngs=rngs)

    up_logits = upstream.apply(init_vars, tokens, attention_mask=causal_mask)


    # Load mapped weights into Tunix model
    nnx.update(model, mapped_params)

    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    tunix_logits, _ = model(tokens, positions=positions, attention_mask=causal_mask)


    self.assertEqual(up_logits.logits.shape, tunix_logits.shape)
    model_state = nnx.state(model)
    model_keys = set(flax.traverse_util.flatten_dict(nnx.to_pure_dict(model_state)).keys())
    mapped_keys = set(flax.traverse_util.flatten_dict(mapped_params).keys())

    missing_in_mapped = model_keys - mapped_keys
    extra_in_mapped = mapped_keys - model_keys

    if missing_in_mapped or extra_in_mapped:
      self.fail(f"Missing in mapped keys: {missing_in_mapped}\nExtra in mapped keys: {extra_in_mapped}")

    # Sanity check: verify weights were loaded
    loaded_weight = model.layers[0].mlp.gate_proj.kernel.value
    expected_weight = mapped_params['layers'][0]['mlp']['gate_proj']['kernel']
    self.assertTrue(jnp.allclose(loaded_weight, expected_weight), f"Weights not loaded! Max diff: {jnp.max(jnp.abs(loaded_weight - expected_weight))}")

    print(f"Tunix logits mean: {jnp.mean(tunix_logits)}, max: {jnp.max(tunix_logits)}")
    print(f"Upstream logits mean: {jnp.mean(up_logits.logits)}, max: {jnp.max(up_logits.logits)}")

    stats = (
        f"Tunix logits mean: {jnp.mean(tunix_logits)}, max: {jnp.max(tunix_logits)}\n"
        f"Upstream logits mean: {jnp.mean(up_logits.logits)}, max: {jnp.max(up_logits.logits)}"
    )
    try:
        np.testing.assert_allclose(up_logits.logits, tunix_logits, rtol=1e-3, atol=1e-3)
    except AssertionError as e:
        self.fail(f"Logits mismatch! Stats:\n{stats}\nError: {e}")




    print("Logits match!")


if __name__ == "__main__":
  absltest.main()
