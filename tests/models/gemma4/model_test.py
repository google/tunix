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

"""Tests for Gemma 4 model."""

from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from tunix.models.gemma4 import model as model_lib


def _make_config(**overrides):
  """Minimal Gemma4 config for unit tests."""
  defaults = dict(
      num_layers=1,
      num_embed=128,
      embed_dim=256,
      hidden_dim=512,
      num_heads=4,
      head_dim=64,
      num_kv_heads=1,
      sliding_window_size=8,
      use_sliding_window_kv_cache=True,
      use_flash_attention=False,
      frac_shared_layers=0.0,
      per_layer_input_dim=0,
      final_logit_softcap=None,
  )
  defaults.update(overrides)
  return model_lib.ModelConfig(**defaults)


def _make_inputs(batch, seq_len, num_embed):
  """Standard tokens, positions, and causal mask."""
  tokens = jax.random.randint(
      jax.random.PRNGKey(0), (batch, seq_len), 0, num_embed
  )
  positions = jnp.tile(jnp.arange(seq_len)[None, :], (batch, 1))
  mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, ...]
  return tokens, positions, mask


class ModelTest(parameterized.TestCase):

  def test_forward_pass_dense(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 1
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 1
    config.frac_shared_layers = 0.0

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs)

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (2, 32), 0, config.num_embed
    )

    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]

    logits, _ = model(tokens, positions=positions, attention_mask=attn_mask)
    self.assertEqual(logits.shape, (2, 32, config.num_embed))
    print(f"{logits.shape=}")

  def test_forward_pass_moe(self):
    config = model_lib.ModelConfig.gemma4_26b_a4b()
    config.num_layers = 1
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 1
    config.num_experts = 4
    config.num_experts_per_tok = 2
    config.expert_dim = 128

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs)

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (2, 32), 0, config.num_embed
    )
    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]
    logits, _ = model(tokens, positions=positions, attention_mask=attn_mask)

    self.assertEqual(logits.shape, (2, 32, config.num_embed))

  def test_remat_block(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 1
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 1
    config.remat_config = model_lib.RematConfig.BLOCK
    config.frac_shared_layers = 0.0

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs)

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (2, 32), 0, config.num_embed
    )

    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]

    def loss_fn(model, tokens, positions, attn_mask):
      logits, _ = model(tokens, positions=positions, attention_mask=attn_mask)
      return jnp.sum(logits)

    loss, grads = nnx.value_and_grad(loss_fn)(
        model, tokens, positions, attn_mask
    )
    self.assertIsNotNone(loss)
    self.assertIsNotNone(grads)

  def test_remat_decoder(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 1
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 1
    config.remat_config = model_lib.RematConfig.DECODER
    config.frac_shared_layers = 0.0

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs)

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (2, 32), 0, config.num_embed
    )

    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]

    def loss_fn(model, tokens, positions, attn_mask):
      logits, _ = model(tokens, positions=positions, attention_mask=attn_mask)
      return jnp.sum(logits)

    loss, grads = nnx.value_and_grad(loss_fn)(
        model, tokens, positions, attn_mask
    )
    self.assertIsNotNone(loss)
    self.assertIsNotNone(grads)

  def test_remat_while_loop_trace_context(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 1
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 1
    config.remat_config = model_lib.RematConfig.BLOCK
    config.frac_shared_layers = 0.0

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs)

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (2, 32), 0, config.num_embed
    )
    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]

    graphdef, state = nnx.split(model, nnx.Param)

    def decode_fn(params):
      def body_fn(step, _):
        transformer = nnx.merge(graphdef, params)
        logits, _ = transformer(
            tokens, positions=positions, attention_mask=attn_mask
        )
        return step + 1, logits

      return jax.lax.while_loop(
          lambda state: state[0] < 1,
          lambda state: body_fn(state[0], state[1]),
          (jnp.array(0), jnp.zeros((2, 32, config.num_embed))),
      )

    compiled_decode = jax.jit(decode_fn)
    _, logits = compiled_decode(state)
    self.assertEqual(logits.shape, (2, 32, config.num_embed))

  @parameterized.named_parameters(
      dict(
          testcase_name='local_sliding',
          attn_type=model_lib.AttentionType.LOCAL_SLIDING,
          use_sw_cache=True,
          sw_size=8,
      ),
      dict(
          testcase_name='global',
          attn_type=model_lib.AttentionType.GLOBAL,
          use_sw_cache=False,
          sw_size=None,
      ),
  )
  def test_chunked_prefill(self, attn_type, use_sw_cache, sw_size):
    cache_len = sw_size or 32
    config = _make_config(
        attention_pattern=(attn_type,),
        use_sliding_window_kv_cache=use_sw_cache,
        sliding_window_size=sw_size,
    )
    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))
    cache = model.init_cache(1, cache_len, jnp.float32)

    # Chunk 1: fill cache.
    c1_len = 8
    tok1, pos1, mask1 = _make_inputs(1, c1_len, config.num_embed)
    _, cache = model(tok1, positions=pos1, cache=cache, attention_mask=mask1)
    self.assertEqual(int(cache['layer_0']['end_index'][0]), c1_len)

    # Chunk 2: chunked prefill (KV = prefix + suffix, so kv_len > q_len).
    c2_len = 4
    tok2 = jax.random.randint(
        jax.random.PRNGKey(1), (1, c2_len), 0, config.num_embed
    )
    pos2 = jnp.arange(c1_len, c1_len + c2_len)[None, :]
    mask2 = jnp.ones((1, c2_len, c1_len + c2_len), dtype=jnp.bool_)

    logits, cache = model(
        tok2,
        positions=pos2,
        cache=cache,
        attention_mask=mask2,
        is_chunked_prefill=True,
        prefix_length=c1_len,
    )
    self.assertEqual(logits.shape, (1, c2_len, config.num_embed))
    self.assertEqual(int(cache['layer_0']['end_index'][0]), c1_len + c2_len)

  def test_kv_sharing_lifecycle(self):
    """3-tuple consumer + KV sharing in both prefill and decode paths."""
    config = _make_config(
        num_layers=4,
        frac_shared_layers=0.5,
        sliding_window_size=32,
        use_sliding_window_kv_cache=False,
        attention_pattern=(
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.GLOBAL,
        ),
    )
    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))
    cache = model.init_cache(1, 32, jnp.float32)

    # Shared layers should NOT have cache entries.
    self.assertLess(len(cache), config.num_layers)

    # Phase 1: prefill — exercises transient_kvs 3-tuple packing.
    tok1, pos1, mask1 = _make_inputs(1, 8, config.num_embed)
    _, cache = model(tok1, positions=pos1, cache=cache, attention_mask=mask1)

    # Verify origin layer wrote non-zero KV into cache.
    origin_key = next(k for k in cache if cache[k] is not None)
    self.assertTrue(jnp.any(cache[origin_key]['k'] != 0))

    # Phase 2: chunked prefill — exercises 3-tuple unpacking + kv_shared_cache.
    tok2 = jax.random.randint(
        jax.random.PRNGKey(1), (1, 4), 0, config.num_embed
    )
    pos2 = jnp.arange(8, 12)[None, :]
    mask2 = jnp.ones((1, 4, 8 + 4), dtype=jnp.bool_)
    logits, cache = model(
        tok2,
        positions=pos2,
        cache=cache,
        attention_mask=mask2,
        is_chunked_prefill=True,
        prefix_length=8,
    )
    self.assertEqual(logits.shape, (1, 4, config.num_embed))

    # Phase 3: decode — exercises kv_shared_cache = new_cache[shared_layer].
    tok3 = jax.random.randint(
        jax.random.PRNGKey(2), (1, 1), 0, config.num_embed
    )
    pos3 = jnp.array([[12]])
    mask3 = jnp.ones((1, 1, 32), dtype=jnp.bool_)
    logits, cache = model(
        tok3,
        positions=pos3,
        cache=cache,
        attention_mask=mask3,
    )
    self.assertEqual(logits.shape, (1, 1, config.num_embed))

  @parameterized.named_parameters(
      dict(
          testcase_name='standard_cache',
          use_sw_cache=False,
          sw_size=32,
          cache_len=32,
      ),
      dict(
          testcase_name='sliding_window_cache',
          use_sw_cache=True,
          sw_size=8,
          cache_len=8,
      ),
  )
  def test_decode(self, use_sw_cache, sw_size, cache_len):
    config = _make_config(
        attention_pattern=(model_lib.AttentionType.LOCAL_SLIDING,),
        use_sliding_window_kv_cache=use_sw_cache,
        sliding_window_size=sw_size,
    )
    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))
    cache = model.init_cache(1, cache_len, jnp.float32)

    # Prefill.
    prefill_len = 8
    tok_pf, pos_pf, mask_pf = _make_inputs(1, prefill_len, config.num_embed)
    _, cache = model(
        tok_pf, positions=pos_pf, cache=cache, attention_mask=mask_pf
    )

    # Decode 3 tokens one at a time.
    for step in range(3):
      tok = jax.random.randint(
          jax.random.PRNGKey(step + 10), (1, 1), 0, config.num_embed
      )
      pos = jnp.array([[prefill_len + step]])
      mask = jnp.ones((1, 1, cache_len), dtype=jnp.bool_)
      logits, cache = model(
          tok,
          positions=pos,
          cache=cache,
          attention_mask=mask,
      )
      self.assertEqual(logits.shape, (1, 1, config.num_embed))
      self.assertEqual(
          int(cache['layer_0']['end_index'][0]), prefill_len + step + 1
      )

  def test_pad_masking(self):
    """PAD positions: KV zeroed, cache protected, end_index counts non-PAD."""
    config = _make_config(
        attention_pattern=(model_lib.AttentionType.LOCAL_SLIDING,),
    )
    model = model_lib.Gemma4(config, rngs=nnx.Rngs(0))
    cache = model.init_cache(1, 8, jnp.float32)

    # Chunk 1.
    tok1, pos1, mask1 = _make_inputs(1, 4, config.num_embed)
    _, cache = model(tok1, positions=pos1, cache=cache, attention_mask=mask1)

    # Chunk 2: 8 tokens, last 3 are PAD.
    seq_len, pad_len = 8, 3
    real_len = seq_len - pad_len
    tok2 = jax.random.randint(
        jax.random.PRNGKey(2), (1, seq_len), 0, config.num_embed
    )
    pos2 = jnp.arange(4, 4 + seq_len)[None, :]
    input_mask = (
        jnp.ones((1, seq_len), dtype=jnp.bool_).at[:, -pad_len:].set(False)
    )
    mask2 = jnp.ones((1, seq_len, 8 + seq_len), dtype=jnp.bool_)

    _, cache = model(
        tok2,
        positions=pos2,
        cache=cache,
        attention_mask=mask2,
        is_chunked_prefill=True,
        prefix_length=4,
        input_mask=input_mask,
    )
    # end_index advances by real_len (5), not seq_len (8).
    self.assertEqual(int(cache['layer_0']['end_index'][0]), 4 + real_len)


if __name__ == '__main__':
  absltest.main()
