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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import qwix
from tunix.models import cache_utils
from tunix.models.gemma4 import attention as attention_lib
from tunix.models.gemma4 import model as model_lib


class ModelTest(parameterized.TestCase):

  def test_gemma4_12b_config(self):
    config = model_lib.ModelConfig.gemma4_12b()

    self.assertEqual(config.num_layers, 48)
    self.assertEqual(config.num_embed, 262144)
    self.assertEqual(config.embed_dim, 3840)
    self.assertEqual(config.hidden_dim, 15360)
    self.assertEqual(config.num_heads, 16)
    self.assertEqual(config.head_dim, 256)
    self.assertEqual(config.num_kv_heads, 8)
    self.assertEqual(config.num_global_kv_heads, 1)
    self.assertEqual(config.global_key_size, 512)
    self.assertEqual(config.sliding_window_size, 1024)
    self.assertTrue(config.k_eq_v_global)
    self.assertEqual(config.per_layer_input_dim, 0)
    self.assertEqual(
        config.attention_pattern,
        (
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.GLOBAL,
        ),
    )

  def test_gemma4_12b_it_config_matches_base(self):
    config = model_lib.ModelConfig.gemma4_12b()
    it_config = model_lib.ModelConfig.gemma4_12b_it()

    self.assertEqual(it_config.num_layers, config.num_layers)
    self.assertEqual(it_config.embed_dim, config.embed_dim)
    self.assertEqual(it_config.hidden_dim, config.hidden_dim)
    self.assertEqual(it_config.num_heads, config.num_heads)
    self.assertEqual(it_config.num_kv_heads, config.num_kv_heads)
    self.assertEqual(it_config.num_global_kv_heads, config.num_global_kv_heads)
    self.assertEqual(it_config.attention_pattern, config.attention_pattern)

  @parameterized.named_parameters(
      (
          'all_unshared',
          0.0,
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
      ),
      (
          'half_shared',
          0.5,
          [0, 1, 2, 3, 4, 5, 4, 4, 4, 4, 4, 5],
      ),
  )
  def test_kv_cache_sharing_patterns(
      self, frac_shared_layers, expected_patterns
  ):
    patterns = model_lib.create_kv_cache_sharing_patterns(
        num_layers=12,
        frac_shared_layers=frac_shared_layers,
        share_global=True,
        share_local=True,
        attention_types=model_lib.GEMMA4_ATTENTION_PATTERN * 2,
    )
    self.assertEqual(patterns, expected_patterns)

  def test_kv_cache_sharing_patterns_raises_on_missing_lender(self):
    with self.assertRaisesRegex(
        ValueError,
        r'Cannot share KV cache for layer \d+ of type AttentionType\..*: no'
        r' unshared layer',
    ):
      model_lib.create_kv_cache_sharing_patterns(
          num_layers=6,
          frac_shared_layers=0.5,
          share_global=True,
          share_local=True,
          attention_types=model_lib.GEMMA4_ATTENTION_PATTERN,
      )

  def test_forward_pass_kv_cache_sharing_lifecycle(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 4
    config.num_embed = 128
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 2
    config.num_global_kv_heads = 2
    config.global_key_size = 64
    config.sliding_window_size = 8
    config.frac_shared_layers = 0.5
    config.attention_pattern = (
        model_lib.AttentionType.LOCAL_SLIDING,
        model_lib.AttentionType.GLOBAL,
    )

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs)

    self.assertEqual(model.kv_cache_sharing_patterns, [0, 1, 0, 1])

    # 1. Init Cache: unshared layers (0, 1) are allocated; shared (2, 3) are skipped.
    cache = model.init_cache(batch_size=1, max_seq_len=16, dtype=jnp.float32)
    self.assertEqual(set(cache.keys()), {'layer_0', 'layer_1'})
    self.assertNotIn('layer_2', cache)
    self.assertNotIn('layer_3', cache)

    # 2. Prefill Step (T=4)
    prefill_tokens = jax.random.randint(
        jax.random.PRNGKey(0), (1, 4), 0, config.num_embed
    )
    prefill_positions = jnp.arange(4)[None, :]
    prefill_mask = jnp.tril(jnp.ones((4, 4), dtype=jnp.bool_))[None, ...]

    logits, updated_cache = model(
        prefill_tokens,
        positions=prefill_positions,
        cache=cache,
        attention_mask=prefill_mask,
    )

    self.assertEqual(logits.shape, (1, 4, config.num_embed))
    self.assertFalse(jnp.isnan(logits).any())
    self.assertEqual(set(updated_cache.keys()), {'layer_0', 'layer_1'})
    self.assertNotIn('layer_2', updated_cache)
    self.assertNotIn('layer_3', updated_cache)
    self.assertEqual(int(updated_cache['layer_0']['end_index'][0]), 4)
    self.assertEqual(int(updated_cache['layer_1']['end_index'][0]), 4)

    # 3. Decode Step (T=1)
    decode_tokens = jax.random.randint(
        jax.random.PRNGKey(1), (1, 1), 0, config.num_embed
    )
    decode_positions = jnp.array([[4]])
    decode_mask = jnp.ones((1, 1, 16), dtype=jnp.bool_)

    decode_logits, final_cache = model(
        decode_tokens,
        positions=decode_positions,
        cache=updated_cache,
        attention_mask=decode_mask,
    )

    self.assertEqual(decode_logits.shape, (1, 1, config.num_embed))
    self.assertFalse(jnp.isnan(decode_logits).any())
    self.assertEqual(set(final_cache.keys()), {'layer_0', 'layer_1'})
    self.assertNotIn('layer_2', final_cache)
    self.assertNotIn('layer_3', final_cache)
    self.assertEqual(int(final_cache['layer_0']['end_index'][0]), 5)
    self.assertEqual(int(final_cache['layer_1']['end_index'][0]), 5)

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

  def test_forward_pass_gemma4_12b(self):
    config = model_lib.ModelConfig.gemma4_12b()
    config.num_layers = 6
    config.num_embed = 128
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 2
    config.num_global_kv_heads = 1

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs)

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (1, 8), 0, config.num_embed
    )
    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]
    logits, _ = model(tokens, positions=positions, attention_mask=attn_mask)

    self.assertEqual(logits.shape, (1, 8, config.num_embed))

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

  def test_remat_qwix_lora_compatibility(self):
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

    lora_provider = qwix.LoraProvider(
        module_path='.*q_einsum|.*kv_einsum|.*attn_vec_einsum|.*gate_proj|.*up_proj|.*down_proj',
        rank=4,
        alpha=2.0,
    )
    model_input = model.get_model_input()
    lora_model = qwix.apply_lora_to_model(model, lora_provider, **model_input)
    lora_model.set_attributes(qwix_rngs=nnx.Rngs(0))

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (2, 32), 0, config.num_embed
    )
    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]

    @nnx.jit
    def train_step(m, tok, pos, mask):
      def loss_fn(model_in):
        logits, _ = model_in(tok, positions=pos, attention_mask=mask)
        return jnp.sum(logits)

      return nnx.value_and_grad(loss_fn)(m)

    loss, grads = train_step(lora_model, tokens, positions, attn_mask)
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

  def test_forward_pass_vision(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 1
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 1
    config.frac_shared_layers = 0.0
    config.vision_encoder = model_lib.vision.VisionEncoderConfig(
        d_model=64,
        num_layers=1,
        num_heads=2,
        ffw_hidden=128,
        patch_size=4,
        output_length=5,
        use_clipped_linears=True,
    )

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs, text_only=False)

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (1, 32), 0, config.num_embed
    )
    tokens = tokens.at[0, 10:15].set(model_lib.IMAGE_SOFT_TOKEN_PLACEHOLDER)

    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]

    soft_token_counts = (5,)
    max_patches = config.vision_encoder.max_patches
    patch_dim = config.vision_encoder.patch_size**2 * 3
    patches = jnp.zeros((1, max_patches, patch_dim), dtype=jnp.float32)
    positions_xy = jnp.full((1, max_patches, 2), -1, dtype=jnp.int32)

    images = model_lib.PreprocessedVisionInput(
        patches=patches,
        positions_xy=positions_xy,
        soft_token_counts=soft_token_counts,
    )

    logits, _ = model(
        tokens,
        positions=positions,
        attention_mask=attn_mask,
        images=images,
    )
    self.assertEqual(logits.shape, (1, 32, config.num_embed))

  def test_forward_pass_vision_bidirectional(self):
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
    config.vision_encoder = model_lib.vision.VisionEncoderConfig(
        d_model=64,
        num_layers=1,
        num_heads=2,
        ffw_hidden=128,
        patch_size=4,
        output_length=5,
        use_clipped_linears=True,
    )
    config.use_bidirectional_attention = "vision"

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs, text_only=False)

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (1, 32), 0, config.num_embed
    )
    tokens = tokens.at[0, 10:15].set(model_lib.IMAGE_SOFT_TOKEN_PLACEHOLDER)

    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]

    soft_token_counts = (5,)
    max_patches = config.vision_encoder.max_patches
    patch_dim = config.vision_encoder.patch_size**2 * 3
    patches = jnp.zeros((1, max_patches, patch_dim), dtype=jnp.float32)
    positions_xy = jnp.full((1, max_patches, 2), -1, dtype=jnp.int32)

    images = model_lib.PreprocessedVisionInput(
        patches=patches,
        positions_xy=positions_xy,
        soft_token_counts=soft_token_counts,
    )

    logits, _ = model(
        tokens,
        positions=positions,
        attention_mask=attn_mask,
        images=images,
    )
    self.assertEqual(logits.shape, (1, 32, config.num_embed))

  def test_forward_pass_vision_batch(self):
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
    config.vision_encoder = model_lib.vision.VisionEncoderConfig(
        d_model=64,
        num_layers=1,
        num_heads=2,
        ffw_hidden=128,
        patch_size=4,
        output_length=5,
        use_clipped_linears=True,
    )
    config.use_bidirectional_attention = "vision"

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs, text_only=False)

    batch_size = 2
    seq_len = 32
    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (batch_size, seq_len), 0, config.num_embed
    )
    # Image placeholders: token shape represents visual soft tokens within sequences.
    tokens = tokens.at[0, 10:15].set(model_lib.IMAGE_SOFT_TOKEN_PLACEHOLDER)
    tokens = tokens.at[1, 5:8].set(model_lib.IMAGE_SOFT_TOKEN_PLACEHOLDER)
    tokens = tokens.at[1, 20:25].set(model_lib.IMAGE_SOFT_TOKEN_PLACEHOLDER)

    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]
    attn_mask = jnp.broadcast_to(attn_mask, (batch_size, seq_len, seq_len))

    # Test batched vision inputs
    soft_token_counts = ((5,), (3, 5))
    max_n_images = 2
    max_patches = config.vision_encoder.max_patches
    patch_dim = config.vision_encoder.patch_size**2 * 3

    # Dimensions for patches: (batch, max_n_images * max_patches, patch_dim)
    patches = jnp.zeros(
        (batch_size, max_n_images * max_patches, patch_dim), dtype=jnp.float32
    )
    positions_xy = jnp.full(
        (batch_size, max_n_images * max_patches, 2), -1, dtype=jnp.int32
    )

    images = model_lib.PreprocessedVisionInput(
        patches=patches,
        positions_xy=positions_xy,
        soft_token_counts=soft_token_counts,
    )

    logits, _ = model(
        tokens,
        positions=positions,
        attention_mask=attn_mask,
        images=images,
    )
    self.assertEqual(logits.shape, (batch_size, seq_len, config.num_embed))

  def test_forward_pass_audio(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 1
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 1
    config.frac_shared_layers = 0.0
    config.audio_encoder = model_lib.audio.ConformerConfig(
        num_layers=1,
        model_dims=64,
        lm_model_dims=256,
        atten_num_heads=2,
    )

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs, text_only=False)

    key = jax.random.key(0)

    batch_size = 1
    num_clips = 1
    num_samples = 16000
    key, audio_key = jax.random.split(key)
    audio = jax.random.normal(audio_key, (batch_size, num_clips, num_samples))
    audio_seq_len = jnp.array([[num_samples]])
    audios = model_lib.PreprocessedAudioInput(
        audios=audio,
        sequence_lengths=audio_seq_len,
    )

    seq_len = 32  # total num of tokens
    _, token_key = jax.random.split(key)
    tokens = jax.random.randint(
        token_key, (batch_size, seq_len), 0, config.num_embed
    )
    # 16000 audio samples => 25 soft tokens
    tokens = tokens.at[0, 5:30].set(model_lib.AUDIO_SOFT_TOKEN_PLACEHOLDER)

    positions = jnp.tile(jnp.arange(seq_len)[None, :], (batch_size, 1))
    attn_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
    attn_mask = jnp.broadcast_to(attn_mask, (batch_size, seq_len, seq_len))

    logits, _ = model(
        tokens,
        positions=positions,
        attention_mask=attn_mask,
        audios=audios,
    )
    self.assertEqual(logits.shape, (batch_size, 32, config.num_embed))

  def test_forward_pass_audio_heterogeneous(self):
    """Test batch with varying number of clips and audio sequence_lengths."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 1
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 1
    config.frac_shared_layers = 0.0
    config.audio_encoder = model_lib.audio.ConformerConfig(
        num_layers=1,
        model_dims=64,
        lm_model_dims=256,
        atten_num_heads=2,
    )

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs, text_only=False)

    key = jax.random.key(0)

    batch_size = 2
    max_clips = 2
    num_samples = 16000  # Max samples per clip

    # Batch 0: 1 valid clip (16000), 1 padding clip (0)
    # Batch 1: 1 valid clip (16000), 1 valid clip (8000)
    sequence_lengths = jnp.array([[16000, 0], [16000, 8000]])

    key, audio_key = jax.random.split(key)
    audio = jax.random.normal(audio_key, (batch_size, max_clips, num_samples))

    # Total soft tokens:
    # 16000 samples => 25 soft tokens
    # 8000 samples => 12 soft tokens
    # Batch 0: 25 + 0 = 25 valid soft tokens
    # Batch 1: 25 + 12 = 37 valid soft tokens

    seq_len = 64  # Total text sequence length
    _, token_key = jax.random.split(key)
    tokens = jax.random.randint(
        token_key, (batch_size, seq_len), 0, config.num_embed
    )

    # Inject placeholders
    tokens = tokens.at[0, 5:30].set(model_lib.AUDIO_SOFT_TOKEN_PLACEHOLDER)
    tokens = tokens.at[1, 5:42].set(model_lib.AUDIO_SOFT_TOKEN_PLACEHOLDER)

    positions = jnp.tile(jnp.arange(seq_len)[None, :], (batch_size, 1))
    attn_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    attn_mask = jnp.broadcast_to(attn_mask, (batch_size, seq_len, seq_len))

    audios = model_lib.PreprocessedAudioInput(
        audios=audio,
        sequence_lengths=sequence_lengths,
    )

    logits, _ = model(
        tokens,
        positions=positions,
        attention_mask=attn_mask,
        audios=audios,
    )
    self.assertEqual(logits.shape, (batch_size, seq_len, config.num_embed))

  def test_forward_pass_chunked_prefill_with_kv_cache_sharing(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 4
    config.num_embed = 128
    config.embed_dim = 128
    config.hidden_dim = 256
    config.num_heads = 2
    config.head_dim = 64
    config.num_kv_heads = 1
    config.sliding_window_size = 16
    config.use_sliding_window_kv_cache = True
    config.frac_shared_layers = 0.5
    config.prefix_bucket_boundaries = (0, 16, 32)
    config.attention_pattern = (
        model_lib.AttentionType.LOCAL_SLIDING,
        model_lib.AttentionType.GLOBAL,
    )

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs)

    # Initialize cache: layers 0 and 1 are unshared origins, layers 2 and 3 share.
    cache = model.init_cache(batch_size=1, max_seq_len=16, dtype=jnp.float32)
    cache['layer_0']['end_index'] = jnp.array([16])
    cache['layer_1']['end_index'] = jnp.array([16])

    suffix_len = 8
    prefix_len = 16
    total_len = prefix_len + suffix_len
    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (1, suffix_len), 0, config.num_embed
    )
    positions = jnp.arange(prefix_len, total_len, dtype=jnp.int32)[None, :]
    attn_mask = jnp.ones((1, suffix_len, total_len), dtype=jnp.bool_)

    logits, updated_cache = model(
        tokens,
        positions=positions,
        cache=cache,
        attention_mask=attn_mask,
        is_chunked_prefill=True,
        prefix_length=prefix_len,
    )

    self.assertEqual(logits.shape, (1, suffix_len, config.num_embed))
    self.assertFalse(jnp.isnan(logits).any())
    self.assertIsNotNone(updated_cache)
    self.assertEqual(set(updated_cache.keys()), {'layer_0', 'layer_1'})
    self.assertNotIn('layer_2', updated_cache)
    self.assertNotIn('layer_3', updated_cache)
    self.assertEqual(int(updated_cache['layer_0']['end_index'][0]), 24)
    self.assertEqual(int(updated_cache['layer_1']['end_index'][0]), 24)

  def test_forward_pass_chunked_prefill_shared_layer_respects_partial_valid_mask(
      self,
  ):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 4
    config.num_embed = 128
    config.embed_dim = 128
    config.hidden_dim = 256
    config.num_heads = 2
    config.head_dim = 64
    config.num_kv_heads = 1
    config.sliding_window_size = 16
    config.use_sliding_window_kv_cache = True
    config.frac_shared_layers = 0.5
    config.prefix_bucket_boundaries = (0, 16, 32)
    config.attention_pattern = (
        model_lib.AttentionType.LOCAL_SLIDING,
        model_lib.AttentionType.GLOBAL,
    )

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs)

    # Initialize cache: 16 slots, partially filled with prefix_len=8.
    cache_len = 16
    prefix_len = 8
    clean_cache = model.init_cache(
        batch_size=1, max_seq_len=cache_len, dtype=jnp.float32
    )
    clean_cache['layer_0']['end_index'] = jnp.array([prefix_len])
    clean_cache['layer_1']['end_index'] = jnp.array([prefix_len])

    suffix_len = 8
    total_len = cache_len + suffix_len
    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (1, suffix_len), 0, config.num_embed
    )
    positions = jnp.arange(
        prefix_len, prefix_len + suffix_len, dtype=jnp.int32
    )[None, :]
    attn_mask = jnp.ones((1, suffix_len, total_len), dtype=jnp.bool_)

    clean_logits, _ = model(
        tokens,
        positions=positions,
        cache=clean_cache,
        attention_mask=attn_mask,
        is_chunked_prefill=True,
        prefix_length=prefix_len,
    )

    # Inject 999.0 noise into uninitialized slots [8:16] of cache['layer_0'].
    corrupt_cache = model.init_cache(
        batch_size=1, max_seq_len=cache_len, dtype=jnp.float32
    )
    corrupt_cache['layer_0']['end_index'] = jnp.array([prefix_len])
    corrupt_cache['layer_1']['end_index'] = jnp.array([prefix_len])
    corrupt_cache['layer_0']['k'] = (
        corrupt_cache['layer_0']['k'].at[:, prefix_len:, ...].set(999.0)
    )
    corrupt_cache['layer_0']['v'] = (
        corrupt_cache['layer_0']['v'].at[:, prefix_len:, ...].set(999.0)
    )

    corrupt_logits, _ = model(
        tokens,
        positions=positions,
        cache=corrupt_cache,
        attention_mask=attn_mask,
        is_chunked_prefill=True,
        prefix_length=prefix_len,
    )

    self.assertFalse(jnp.isnan(clean_logits).any())
    self.assertFalse(jnp.isnan(corrupt_logits).any())
    np.testing.assert_allclose(corrupt_logits, clean_logits, atol=1e-5)

  def test_shared_cache_prefix_bucketing_when_cache_is_none(self):
    """Verifies Attention and DecoderLayer bucket prefix_length when cache is None and kv_shared_cache is passed."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 2
    config.num_embed = 128
    config.embed_dim = 64
    config.hidden_dim = 128
    config.num_heads = 2
    config.head_dim = 32
    config.num_kv_heads = 2
    config.global_key_size = 32
    config.num_global_kv_heads = 2
    config.k_eq_v_global = False
    config.sliding_window_size = 16
    config.prefix_bucket_boundaries = (0, 8, 16)
    config.use_flash_attention = False

    rngs = nnx.Rngs(0)
    decoder_layer = model_lib.GemmaDecoderLayer(
        config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=rngs,
    )
    attn = decoder_layer.attn

    prefix_len = 5
    expected_bucketed_prefix = 8
    suffix_len = 4
    total_kv_len = expected_bucketed_prefix + suffix_len

    kv_shared_cache = {
        'k': jnp.zeros((1, total_kv_len, attn.num_kv_heads, attn.head_dim)),
        'v': jnp.zeros((1, total_kv_len, attn.num_kv_heads, attn.head_dim)),
        'prior_end_index': jnp.array([prefix_len]),
    }

    x = jnp.zeros((1, suffix_len, config.embed_dim))
    segment_pos = jnp.arange(prefix_len, prefix_len + suffix_len)[None, :]
    attn_mask = jnp.ones((1, suffix_len, total_kv_len + 8), dtype=jnp.bool_)

    # 1. Test Attention.__call__ directly (non-remat)
    with mock.patch.object(attn, 'block', wraps=attn.block) as mock_attn_block:
      _, attn_out, _ = attn(
          x,
          segment_pos,
          cache=None,
          attn_mask=attn_mask,
          kv_shared_cache=kv_shared_cache,
          is_chunked_prefill=True,
          prefix_length=prefix_len,
      )
      self.assertEqual(
          mock_attn_block.call_args.kwargs['prefix_length'],
          expected_bucketed_prefix,
      )
      self.assertEqual(attn_out.shape, x.shape)
      self.assertFalse(jnp.isnan(attn_out).any())

    # 2. Test Attention.__call__ with BLOCK remat
    attn.config.remat_config = attention_lib.RematConfig.BLOCK
    with mock.patch.object(
        attention_lib.cache_utils,
        'maybe_bucket_prefix_length',
        wraps=cache_utils.maybe_bucket_prefix_length,
    ) as mock_bucket:
      _, attn_out, _ = attn(
          x,
          segment_pos,
          cache=None,
          attn_mask=attn_mask,
          kv_shared_cache=kv_shared_cache,
          is_chunked_prefill=True,
          prefix_length=prefix_len,
      )
      self.assertEqual(mock_bucket.call_args.args[0], prefix_len)
      self.assertIs(mock_bucket.call_args.args[1], kv_shared_cache)
      self.assertEqual(attn_out.shape, x.shape)
      self.assertFalse(jnp.isnan(attn_out).any())
    attn.config.remat_config = attention_lib.RematConfig.NONE

    # 3. Test GemmaDecoderLayer.__call__
    with mock.patch.object(
        decoder_layer, 'block', wraps=decoder_layer.block
    ) as mock_layer_block:
      _, layer_out, _ = decoder_layer(
          x,
          segment_pos,
          cache=None,
          attn_mask=attn_mask,
          kv_shared_cache=kv_shared_cache,
          is_chunked_prefill=True,
          prefix_length=prefix_len,
      )
      self.assertEqual(
          mock_layer_block.call_args.kwargs['prefix_length'],
          expected_bucketed_prefix,
      )
      self.assertEqual(layer_out.shape, x.shape)
      self.assertFalse(jnp.isnan(layer_out).any())

    # Sanity check helper behavior with vs without kv_shared_cache
    unbucketed_len = cache_utils.maybe_bucket_prefix_length(
        prefix_len,
        None,
        is_chunked_prefill=True,
        boundaries=config.prefix_bucket_boundaries,
    )
    self.assertEqual(unbucketed_len, prefix_len)

    bucketed_len = cache_utils.maybe_bucket_prefix_length(
        prefix_len,
        kv_shared_cache,
        is_chunked_prefill=True,
        boundaries=config.prefix_bucket_boundaries,
    )
    self.assertEqual(bucketed_len, expected_bucketed_prefix)

  def test_chunked_prefill_bucket_padding_forces_eager(self):
    """Verifies DecoderLayer forces eager attention if and only if bucket padding was introduced."""
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 1
    config.num_embed = 128
    config.embed_dim = 64
    config.hidden_dim = 128
    config.num_heads = 2
    config.head_dim = 32
    config.num_kv_heads = 2
    config.global_key_size = 32
    config.num_global_kv_heads = 2
    config.k_eq_v_global = False
    config.prefix_bucket_boundaries = (0, 8, 16)

    rngs = nnx.Rngs(0)
    decoder_layer = model_lib.GemmaDecoderLayer(
        config,
        attn_type=model_lib.AttentionType.GLOBAL,
        rngs=rngs,
    )
    suffix_len = 4
    x = jnp.zeros((1, suffix_len, config.embed_dim))
    cache = {
        'k': jnp.zeros((1, 16, 2, 32)),
        'v': jnp.zeros((1, 16, 2, 32)),
        'end_index': jnp.array([16]),
    }

    # Case 1: prefix_len = 5, bucketed_prefix = 8 (padding introduced -> force_eager must be True)
    segment_pos = jnp.arange(5, 5 + suffix_len)[None, :]
    attn_mask = jnp.ones((1, suffix_len, 8 + suffix_len), dtype=jnp.bool_)
    with mock.patch.object(
        decoder_layer, 'block', wraps=decoder_layer.block
    ) as mock_block:
      decoder_layer(
          x,
          segment_pos,
          cache=cache,
          attn_mask=attn_mask,
          is_chunked_prefill=True,
          prefix_length=5,
      )
      self.assertTrue(
          mock_block.call_args.kwargs['force_eager'],
          'Expected force_eager=True when bucketed_prefix != prefix_length',
      )

    # Case 2: prefix_len = 8, bucketed_prefix = 8 (no padding -> force_eager must remain False)
    segment_pos = jnp.arange(8, 8 + suffix_len)[None, :]
    attn_mask = jnp.ones((1, suffix_len, 8 + suffix_len), dtype=jnp.bool_)
    with mock.patch.object(
        decoder_layer, 'block', wraps=decoder_layer.block
    ) as mock_block:
      decoder_layer(
          x,
          segment_pos,
          cache=cache,
          attn_mask=attn_mask,
          is_chunked_prefill=True,
          prefix_length=8,
      )
      self.assertFalse(
          mock_block.call_args.kwargs['force_eager'],
          'Expected force_eager=False when bucketed_prefix == prefix_length',
      )

  def test_chunked_prefill_preconditions_and_defaults(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.frac_shared_layers = 0.0
    config.prefix_bucket_boundaries = ()
    config.num_layers = 2
    config.num_embed = 128
    config.embed_dim = 64
    config.hidden_dim = 128
    config.num_heads = 2
    config.head_dim = 32
    config.num_kv_heads = 2
    config.global_key_size = 32
    config.num_global_kv_heads = 2
    config.k_eq_v_global = False
    config.use_flash_attention = False

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs)
    cache = model.init_cache(1, 16, jnp.float32)
    cache['layer_0']['end_index'] = jnp.array([4])
    cache['layer_1']['end_index'] = jnp.array([4])
    tokens = jnp.zeros((1, 4), dtype=jnp.int32)
    attn_mask = jnp.ones((1, 4, 8), dtype=jnp.bool_)

    # 1. Negative prefix_length raises ValueError
    with self.assertRaisesRegex(ValueError, r'prefix_length.*non-negative'):
      model(tokens, cache=cache, is_chunked_prefill=True, prefix_length=-1)

    # 2. Chunked prefill without cache raises ValueError
    with self.assertRaisesRegex(ValueError, r'cache.*required'):
      model(tokens, cache=None, is_chunked_prefill=True, prefix_length=4)

    # 3. prefix_length > 0 when is_chunked_prefill is False raises ValueError
    with self.assertRaisesRegex(
        ValueError, r'prefix_length.*is_chunked_prefill.*False'
    ):
      model(tokens, cache=cache, is_chunked_prefill=False, prefix_length=4)

    # 4. Chunked prefill with omitted input_mask safely defaults to ones
    logits, _ = model(
        tokens,
        attention_mask=attn_mask,
        cache=cache,
        is_chunked_prefill=True,
        prefix_length=4,
        input_mask=None,
    )
    self.assertEqual(logits.shape, (1, 4, config.num_embed))

    # 5. Omitted positions auto-offsets positions
    with mock.patch.object(
        model.layers[0], 'block', wraps=model.layers[0].block
    ) as mock_layer:
      attn_mask_8 = jnp.ones((1, 4, 12), dtype=jnp.bool_)
      cache['layer_0']['end_index'] = jnp.array([8])
      cache['layer_1']['end_index'] = jnp.array([8])
      model(
          tokens,
          positions=None,
          attention_mask=attn_mask_8,
          cache=cache,
          is_chunked_prefill=True,
          prefix_length=8,
      )
      passed_positions = mock_layer.call_args.args[1]
      expected_positions = jnp.arange(8, 12, dtype=jnp.int32)[None, :]
      np.testing.assert_array_equal(passed_positions, expected_positions)


if __name__ == '__main__':
  absltest.main()
