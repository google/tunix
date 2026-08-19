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

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
import jax.numpy as jnp
import numpy as np
import qwix
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


def _on_tpu():
  """Returns True if running on TPU. Must be called after absl.app.run()."""
  return jax.default_backend() == "tpu"


def _run_chunked_prefill(model, total_len, chunk_len, batch=1):
  """Runs a two-chunk prefill and returns logits from the second chunk.

  Chunk 1: tokens[0:chunk_len]  (square attention)
  Chunk 2: tokens[chunk_len:2*chunk_len]  (rectangular: cached prefix + chunk)

  Returns:
    Logits from chunk 2, shape (batch, chunk_len, num_embed).
  """
  num_embed = model.config.num_embed
  tokens = jax.random.randint(
      jax.random.PRNGKey(1), (batch, total_len), 0, num_embed
  )

  # Initialize cache
  cache = model.init_cache(batch, total_len, model.config.dtype)

  # --- Chunk 1: positions [0, chunk_len) ---
  tok1 = tokens[:, :chunk_len]
  pos1 = jnp.tile(jnp.arange(chunk_len)[None, :], (batch, 1))
  mask1 = jnp.tril(jnp.ones((chunk_len, chunk_len), dtype=jnp.bool_))[None, ...]

  _, cache = model(
      tok1,
      positions=pos1,
      attention_mask=mask1,
      cache=cache,
  )

  # --- Chunk 2: positions [chunk_len, 2*chunk_len) ---
  tok2 = tokens[:, chunk_len : 2 * chunk_len]
  pos2 = jnp.tile(jnp.arange(chunk_len, 2 * chunk_len)[None, :], (batch, 1))
  # Causal mask for the full sequence, then slice the last chunk_len rows.
  full_mask = jnp.tril(
      jnp.ones((2 * chunk_len, 2 * chunk_len), dtype=jnp.bool_)
  )
  mask2 = full_mask[chunk_len:, :][None, ...]

  logits, _ = model(
      tok2,
      positions=pos2,
      attention_mask=mask2,
      cache=cache,
      is_chunked_prefill=True,
      prefix_length=chunk_len,
  )
  return logits


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

  def test_kv_cache_sharing_patterns_type_aware(self):
    patterns = model_lib.create_kv_cache_sharing_patterns(
        num_layers=12,
        frac_shared_layers=0.5,
        share_global=True,
        share_local=True,
        attention_types=model_lib.GEMMA4_ATTENTION_PATTERN * 2,
    )
    self.assertEqual(patterns, [0, 1, 2, 3, 4, 5, 4, 4, 4, 4, 4, 5])

  def test_kv_cache_sharing_patterns_raises_on_missing_lender(self):
    with self.assertRaises(ValueError):
      model_lib.create_kv_cache_sharing_patterns(
          num_layers=6,
          frac_shared_layers=0.5,
          share_global=True,
          share_local=True,
          attention_types=model_lib.GEMMA4_ATTENTION_PATTERN,
      )

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


class FlashAttentionMaskTest(parameterized.TestCase):
  """Mask correctness unit tests (pure numpy — no model needed)."""

  def test_local_mask_matches_manual(self):
    """Verify LocalMask with offset produces the correct sliding window mask."""
    chunk_len = 1024
    sw_size = 512
    cache_len = sw_size
    kv_len = cache_len + chunk_len
    prefix_len = cache_len

    # Splash mask with offset
    splash_mask = mask_lib.LocalMask(
        (chunk_len, kv_len),
        window_size=(sw_size - 1, 0),
        offset=prefix_len,
    )
    splash_array = splash_mask[np.s_[:, :]]

    # Manual mask (replicating _build_chunked_prefill_mask logic for LOCAL)
    position_offset = prefix_len
    valid_cache_len = prefix_len
    row_pos = np.arange(chunk_len) + position_offset
    col_pos_cache = np.arange(cache_len) + (position_offset - valid_cache_len)
    col_pos_suffix = np.arange(chunk_len) + position_offset
    col_pos = np.concatenate([col_pos_cache, col_pos_suffix])
    manual_mask = (col_pos[None, :] > (row_pos[:, None] - sw_size)) & (
        col_pos[None, :] <= row_pos[:, None]
    )

    np.testing.assert_array_equal(splash_array, manual_mask)

  def test_causal_mask_matches_manual(self):
    """Verify CausalMask with offset for GLOBAL chunked prefill."""
    chunk_len = 1024
    prefix_len = 2048
    kv_len = prefix_len + chunk_len

    splash_mask = mask_lib.CausalMask(
        (chunk_len, kv_len),
        offset=prefix_len,
    )
    splash_array = splash_mask[np.s_[:, :]]

    # Manual: q[i] can attend to kv[j] where i + offset >= j
    row = np.arange(chunk_len)[:, None] + prefix_len
    col = np.arange(kv_len)[None, :]
    manual_mask = row >= col

    np.testing.assert_array_equal(splash_array, manual_mask)

  @parameterized.parameters(
      # (chunk_len, sw_size) — various sizes to test edge cases
      (256, 128),
      (512, 256),
      (1024, 512),
      (2048, 1024),
  )
  def test_local_mask_offset_parameterized(self, chunk_len, sw_size):
    """LocalMask with offset is correct for various chunk/window sizes."""
    cache_len = sw_size
    kv_len = cache_len + chunk_len

    splash_mask = mask_lib.LocalMask(
        (chunk_len, kv_len),
        window_size=(sw_size - 1, 0),
        offset=cache_len,
    )
    splash_array = splash_mask[np.s_[:, :]]

    # Each Q position q[i] at logical position (i + cache_len) should attend
    # to KV positions in [i + cache_len - (sw_size - 1), i + cache_len].
    for i in range(0, chunk_len, max(1, chunk_len // 8)):
      logical_q = i + cache_len
      expected_start = max(0, logical_q - (sw_size - 1))
      expected_end = logical_q
      # Verify True positions in row i
      true_cols = np.where(splash_array[i])[0]
      if len(true_cols) > 0:
        self.assertEqual(true_cols[0], expected_start)
        self.assertEqual(true_cols[-1], expected_end)
        self.assertEqual(len(true_cols), expected_end - expected_start + 1)

  def test_local_mask_square_no_offset(self):
    """Square LocalMask (chunk 1) should produce standard sliding window."""
    seq_len = 512
    sw_size = 128

    splash_mask = mask_lib.LocalMask(
        (seq_len, seq_len),
        window_size=(sw_size - 1, 0),
        offset=0,
    )
    splash_array = splash_mask[np.s_[:, :]]

    # Manual: standard causal sliding window
    row = np.arange(seq_len)[:, None]
    col = np.arange(seq_len)[None, :]
    manual_mask = (col <= row) & (col > row - sw_size)

    np.testing.assert_array_equal(splash_array, manual_mask)


class SplitAttentionMergeTest(parameterized.TestCase):
  """NaN-safe LSE merge unit tests (pure jax/numpy — no TPU)."""

  def test_fully_masked_prefix_is_finite_and_equals_suffix(self):
    """out_prefix=NaN, lse_prefix=-inf must yield finite suffix-only."""
    b, n, t, h = 1, 2, 3, 4
    rng = np.random.default_rng(0)
    out_prefix = jnp.full((b, n, t, h), jnp.nan, dtype=jnp.float32)
    lse_prefix = jnp.full((b, n, t), -jnp.inf, dtype=jnp.float32)
    out_suffix = jnp.asarray(
        rng.standard_normal((b, n, t, h)), dtype=jnp.float32
    )
    lse_suffix = jnp.asarray(rng.standard_normal((b, n, t)), dtype=jnp.float32)

    encoded = model_lib._merge_split_attention(
        out_prefix, lse_prefix, out_suffix, lse_suffix, jnp.float32
    )

    self.assertTrue(bool(jnp.isfinite(encoded).all()))
    np.testing.assert_allclose(
        np.asarray(encoded), np.asarray(out_suffix), rtol=1e-6, atol=1e-6
    )

  def test_both_finite_matches_analytic_lse_average(self):
    """Both partitions finite → equals analytic LSE-weighted avg."""
    b, n, t, h = 2, 2, 3, 4
    rng = np.random.default_rng(1)
    out_prefix = rng.standard_normal((b, n, t, h)).astype(np.float32)
    out_suffix = rng.standard_normal((b, n, t, h)).astype(np.float32)
    lse_prefix = rng.standard_normal((b, n, t)).astype(np.float32)
    lse_suffix = rng.standard_normal((b, n, t)).astype(np.float32)

    encoded = model_lib._merge_split_attention(
        jnp.asarray(out_prefix),
        jnp.asarray(lse_prefix),
        jnp.asarray(out_suffix),
        jnp.asarray(lse_suffix),
        jnp.float32,
    )

    max_lse = np.maximum(lse_prefix, lse_suffix)
    wp = np.exp(lse_prefix - max_lse)
    ws = np.exp(lse_suffix - max_lse)
    expected = (wp[..., None] * out_prefix + ws[..., None] * out_suffix) / (
        wp + ws
    )[..., None]

    np.testing.assert_allclose(
        np.asarray(encoded), expected, rtol=1e-5, atol=1e-5
    )

  def test_partial_row_masking_all_rows_finite(self):
    """Subset of rows fully-masked → all rows finite, others keep prefix."""
    b, n, t, h = 1, 1, 4, 3
    rng = np.random.default_rng(2)
    out_prefix = rng.standard_normal((b, n, t, h)).astype(np.float32)
    out_suffix = rng.standard_normal((b, n, t, h)).astype(np.float32)
    lse_prefix = rng.standard_normal((b, n, t)).astype(np.float32)
    lse_suffix = rng.standard_normal((b, n, t)).astype(np.float32)

    # Mask rows 1 and 3 of the prefix partition (NaN out / -inf lse).
    masked_rows = [1, 3]
    out_prefix[:, :, masked_rows, :] = np.nan
    lse_prefix[:, :, masked_rows] = -np.inf

    encoded = model_lib._merge_split_attention(
        jnp.asarray(out_prefix),
        jnp.asarray(lse_prefix),
        jnp.asarray(out_suffix),
        jnp.asarray(lse_suffix),
        jnp.float32,
    )
    encoded = np.asarray(encoded)

    self.assertTrue(np.isfinite(encoded).all())
    # Masked rows must equal suffix-only.
    np.testing.assert_allclose(
        encoded[:, :, masked_rows, :],
        out_suffix[:, :, masked_rows, :],
        rtol=1e-6,
        atol=1e-6,
    )
    # Unmasked rows still reflect the (finite) prefix contribution.
    unmasked = [0, 2]
    max_lse = np.maximum(lse_prefix, lse_suffix)
    wp = np.exp(lse_prefix - max_lse)
    ws = np.exp(lse_suffix - max_lse)
    expected = (
        wp[..., None] * np.nan_to_num(out_prefix) + ws[..., None] * out_suffix
    ) / (wp + ws)[..., None]
    np.testing.assert_allclose(
        encoded[:, :, unmasked, :],
        expected[:, :, unmasked, :],
        rtol=1e-5,
        atol=1e-5,
    )


class LogicalSlidingWindowMaskTest(parameterized.TestCase):
  """Logical-position decode sliding window (pure jax — no TPU)."""

  def _mask_from_valid_slots(self, cache_len, valid_slots):
    m = np.zeros((1, 1, cache_len), dtype=np.int32)
    m[0, 0, list(valid_slots)] = 1
    return jnp.asarray(m)

  @parameterized.parameters(
      # (cache_len, sw, left_pad, end)
      (16, 4, 3, 10),
      (16, 8, 0, 15),
      (32, 4, 5, 5),  # single valid slot
      (12, 3, 2, 9),
  )
  def test_normal_decode_mask_unchanged(self, cache_len, sw, left_pad, end):
    """Contiguous left-pad decode: gated result is byte-identical to original."""
    attn_mask = self._mask_from_valid_slots(cache_len, range(left_pad, end + 1))

    # No gap should be detected for a contiguous valid region.
    has_gap = model_lib._has_physical_gap(attn_mask)
    self.assertFalse(bool(jnp.any(has_gap)))

    physical = model_lib.create_sliding_window_mask(
        attn_mask, sliding_window_size=sw
    )
    logical = model_lib.create_logical_sliding_window_mask(
        attn_mask, sliding_window_size=sw
    )

    # Replicate the caller's gated selection.
    gated = jnp.where(has_gap, logical, physical)
    original_final = physical * attn_mask
    gated_final = gated * attn_mask

    # Byte-identical to the original physical-window path.
    np.testing.assert_array_equal(
        np.asarray(gated_final), np.asarray(original_final)
    )
    # And the logical window itself matches the physical one when contiguous.
    np.testing.assert_array_equal(
        np.asarray(logical), np.asarray(physical * attn_mask)
    )

  def test_chunked_gap_keeps_real_tokens_in_window(self):
    """A right-pad gap must not starve real prompt tokens from the window."""
    cache_len, sw = 16, 8
    # Real prompt at physical [0..3] (logical 0..3), gap at [4..13] (zeroed),
    # decode/gen token at physical 14 (logical 4). Slot 15 is future -> invalid.
    real_prompt = [0, 1, 2, 3]
    gen_slot = 14
    attn_mask = self._mask_from_valid_slots(cache_len, real_prompt + [gen_slot])

    has_gap = model_lib._has_physical_gap(attn_mask)
    self.assertTrue(bool(jnp.all(has_gap)))

    physical = np.asarray(
        model_lib.create_sliding_window_mask(attn_mask, sliding_window_size=sw)
        * attn_mask
    )[0, 0]
    logical = np.asarray(
        model_lib.create_logical_sliding_window_mask(
            attn_mask, sliding_window_size=sw
        )
    )[0, 0]

    # The physical window starves the real prompt: only the gen token survives.
    self.assertEqual(list(np.where(physical)[0]), [gen_slot])
    # The logical window keeps every real prompt token plus the gen token.
    self.assertEqual(
        list(np.where(logical)[0]), sorted(real_prompt + [gen_slot])
    )
    # Gap slots and the future slot remain masked.
    for j in list(range(4, 14)) + [15]:
      self.assertEqual(logical[j], False)


class FlashAttentionBlockSizeTest(parameterized.TestCase):
  """Block-size divisibility parameterized test."""

  @parameterized.parameters(
      model_lib.ModelConfig.gemma4_e2b,
      model_lib.ModelConfig.gemma4_e4b,
      model_lib.ModelConfig.gemma4_31b,
      model_lib.ModelConfig.gemma4_26b_a4b,
  )
  def test_block_kv_divisibility(self, config_factory):
    """block_kv must divide kv_len and be a multiple of 128 (NUM_LANES)."""
    config = config_factory()
    sw = config.sliding_window_size
    block_q = config.flash_attention_block_size
    block_kv = min(block_q, sw)
    chunk_len = block_q  # Minimum valid chunk size
    kv_len = sw + chunk_len

    self.assertEqual(chunk_len % block_q, 0)
    self.assertEqual(kv_len % block_kv, 0)
    self.assertEqual(
        block_kv % 128,
        0,
        f"block_kv={block_kv} not a multiple of 128 (NUM_LANES)",
    )

  @parameterized.parameters(
      model_lib.ModelConfig.gemma4_e2b,
      model_lib.ModelConfig.gemma4_e4b,
      model_lib.ModelConfig.gemma4_31b,
      model_lib.ModelConfig.gemma4_26b_a4b,
  )
  def test_block_kv_divides_multiple_chunk_sizes(self, config_factory):
    """block_kv should work for chunk_len = 1x, 2x, 4x block_q."""
    config = config_factory()
    sw = config.sliding_window_size
    block_q = config.flash_attention_block_size
    block_kv = min(block_q, sw)

    for multiplier in [1, 2, 4]:
      chunk_len = block_q * multiplier
      kv_len = sw + chunk_len
      self.assertEqual(
          chunk_len % block_q,
          0,
          f"chunk_len={chunk_len} not divisible by block_q={block_q}",
      )
      self.assertEqual(
          kv_len % block_kv,
          0,
          f"kv_len={kv_len} not divisible by block_kv={block_kv} "
          f"(multiplier={multiplier})",
      )


class FlashAttentionModelTest(parameterized.TestCase):
  """Model-level flash attention tests (TPU-only)."""

  def setUp(self):
    super().setUp()
    if not _on_tpu():
      self.skipTest("Flash attention requires TPU")

  def _make_flash_config(self, **extra):
    """Config suitable for flash + sliding window chunked prefill."""
    defaults = dict(
        num_layers=6,
        sliding_window_size=512,
        use_sliding_window_kv_cache=True,
        flash_attention_block_size=512,
        attention_pattern=(
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.GLOBAL,
        ),
    )
    defaults.update(extra)
    return _make_config(**defaults)

  def _make_model_pair(self, flash_config):
    """Create flash and non-flash models with shared weights.

    Returns:
      (model_flash, model_noflash) with identical parameters.
    """
    rngs = nnx.Rngs(42)
    config_noflash = dataclasses.replace(
        flash_config, use_flash_attention=False
    )
    model_nf = model_lib.Gemma4(config_noflash, rngs=rngs)

    config_flash = dataclasses.replace(flash_config, use_flash_attention=True)
    # Create flash model with same rngs seed for identical init
    model_f = model_lib.Gemma4(config_flash, rngs=nnx.Rngs(42))

    return model_f, model_nf

  def test_flash_vs_nonflash_chunked_prefill(self):
    """Output logits with flash should match non-flash within tolerance."""
    config = self._make_flash_config()
    model_f, model_nf = self._make_model_pair(config)

    chunk_len = 512
    total_len = chunk_len * 2

    logits_f = _run_chunked_prefill(model_f, total_len, chunk_len)
    logits_nf = _run_chunked_prefill(model_nf, total_len, chunk_len)

    # Across 6 transformer layers on TPU, bfloat16/float32 accumulator
    # differences between Splash Flash and XLA dot_general compound to ~0.32.
    np.testing.assert_allclose(
        np.array(logits_f),
        np.array(logits_nf),
        atol=0.5,
        rtol=0.05,
    )

  def test_flash_with_shared_layers(self):
    """Flash attention works correctly with KV-sharing layers."""
    config = self._make_flash_config(
        num_layers=12,
        attention_pattern=model_lib.GEMMA4_ATTENTION_PATTERN * 2,
        frac_shared_layers=6.0 / 12,
    )
    model_f, model_nf = self._make_model_pair(config)

    chunk_len = 512
    total_len = chunk_len * 2

    logits_f = _run_chunked_prefill(model_f, total_len, chunk_len)
    logits_nf = _run_chunked_prefill(model_nf, total_len, chunk_len)

    # Across 12 shared transformer layers on TPU, kernel rounding differences
    # compound to ~0.64.
    np.testing.assert_allclose(
        np.array(logits_f),
        np.array(logits_nf),
        atol=0.75,
        rtol=0.05,
    )

  def test_partial_cache_fill_matches_nonflash(self):
    """Partial-cache LOCAL_SLIDING chunked prefill must match eager exactly.

    When prefix_length < sliding_window_size the sliding-window KV ring is only
    partially filled, and flash's static relative offset (kv_len - q_len) would
    slide the window past the real cached tokens (see cl/933189977). The eager
    fallback makes flash bypass the splash kernel here, so its logits match the
    non-flash reference to machine precision.
    """
    config = self._make_flash_config(
        num_layers=2,
        sliding_window_size=512,
        flash_attention_block_size=256,
        attention_pattern=(
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.LOCAL_SLIDING,
        ),
    )
    model_f, model_nf = self._make_model_pair(config)

    chunk_len = 256  # < sw=512: partial window -> eager fallback engages
    total_len = chunk_len * 2

    logits_f = _run_chunked_prefill(model_f, total_len, chunk_len)
    logits_nf = _run_chunked_prefill(model_nf, total_len, chunk_len)

    # Chunk 1 square prefill produces ~0.007 rounding difference on TPU between
    # Splash Flash and eager, which is inherited by Chunk 2's eager fallback.
    np.testing.assert_allclose(
        np.array(logits_f),
        np.array(logits_nf),
        atol=0.02,
        rtol=0.02,
    )

  def test_full_window_chunked_prefill_stays_flash(self):
    """prefix_length == sliding_window_size keeps the flash path (no fallback).

    The guard only fires for 0 < prefix_length < sliding_window_size. With a
    full window the flash relative offset is correct, so flash stays ON and
    still matches eager. This protects the 8K judge from a perf regression (S2).
    """
    config = self._make_flash_config(
        num_layers=2,
        sliding_window_size=512,
        flash_attention_block_size=512,
        attention_pattern=(
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.LOCAL_SLIDING,
        ),
    )
    model_f, model_nf = self._make_model_pair(config)

    chunk_len = 512  # == sw: window full -> guard skipped -> flash path
    total_len = chunk_len * 2

    logits_f = _run_chunked_prefill(model_f, total_len, chunk_len)
    logits_nf = _run_chunked_prefill(model_nf, total_len, chunk_len)

    # Real splash kernel vs eager: full-window parity holds within chunk-2
    # TPU rounding tolerance (~0.016 across 2 layers).
    np.testing.assert_allclose(
        np.array(logits_f),
        np.array(logits_nf),
        atol=0.02,
        rtol=0.02,
    )

  def test_bucketed_prefix_partial_cache_matches_nonflash(self):
    """Fallback must key on the RAW prefix_length, not the bucketed value.

    With prefix_bucket_boundaries=(512,), a raw prefix of 256 buckets UP to
    512 (== sliding_window_size). A guard placed on the bucketed value would
    see 512 < 512 -> False and leave flash ON in the exact broken
    partial-cache case. The fix keys on the raw prefix in __call__, so the
    eager fallback still engages and logits match non-flash. See cl/933189977.
    """
    config = self._make_flash_config(
        num_layers=2,
        sliding_window_size=512,
        flash_attention_block_size=256,
        prefix_bucket_boundaries=(512,),
        attention_pattern=(
            model_lib.AttentionType.LOCAL_SLIDING,
            model_lib.AttentionType.LOCAL_SLIDING,
        ),
    )
    model_f, model_nf = self._make_model_pair(config)

    chunk_len = 256  # raw < sw=512, but buckets up to 512
    total_len = chunk_len * 2

    logits_f = _run_chunked_prefill(model_f, total_len, chunk_len)
    logits_nf = _run_chunked_prefill(model_nf, total_len, chunk_len)

    # If the guard used the bucketed prefix (512), flash would stay on and
    # diverge; keying on the raw prefix keeps parity within chunk-1 rounding.
    np.testing.assert_allclose(
        np.array(logits_f),
        np.array(logits_nf),
        atol=0.02,
        rtol=0.02,
    )

  def test_global_layers_fallback_to_nonflash(self):
    """GLOBAL layers fall back to non-flash during rectangular chunks."""
    config = self._make_flash_config()
    model_f, model_nf = self._make_model_pair(config)

    chunk_len = 512
    total_len = chunk_len * 2

    logits_f = _run_chunked_prefill(model_f, total_len, chunk_len)
    logits_nf = _run_chunked_prefill(model_nf, total_len, chunk_len)

    # Full model outputs should be close across 6 layers on TPU.
    np.testing.assert_allclose(
        np.array(logits_f),
        np.array(logits_nf),
        atol=0.5,
        rtol=0.05,
    )

  def test_flash_rectangular_with_remat(self):
    """Rectangular flash attention works with remat_config=BLOCK."""
    config = self._make_flash_config(
        remat_config=model_lib.RematConfig.BLOCK,
    )
    model_f, model_nf = self._make_model_pair(config)

    chunk_len = 512
    total_len = chunk_len * 2

    logits_f = _run_chunked_prefill(model_f, total_len, chunk_len)
    logits_nf = _run_chunked_prefill(model_nf, total_len, chunk_len)

    np.testing.assert_allclose(
        np.array(logits_f),
        np.array(logits_nf),
        atol=0.5,
        rtol=0.05,
    )


class ChunkedPrefillRaggedTest(parameterized.TestCase):
  """Covers the warm-prefix fix for ragged (non-uniform) input_mask."""

  def _make_ragged_model(self):
    # Default config (sliding-window KV cache), matching the known-good setup
    # used by the existing chunked-prefill tests. The end_index advance we are
    # testing is returned in a branch shared by both cache layouts, so the
    # sliding-window path exercises it correctly.
    return model_lib.Gemma4(_make_config(), rngs=nnx.Rngs(0))

  def test_ragged_input_mask_advances_by_batch_max(self):
    """end_index == num_real_tokens for a ragged batch; assertion gone."""
    model = self._make_ragged_model()
    num_embed = model.config.num_embed
    batch = 2
    chunk_len = 8  # chunk 2 width == seq_len; larger than any real suffix.
    total_len = 2 * chunk_len
    prefix_len = chunk_len

    # Distinct real-token counts so all three candidate advances differ:
    #   old (input_mask[0] sum) -> prefix_len + r0
    #   buggy seq_len           -> prefix_len + chunk_len
    #   fixed (batch-max)       -> prefix_len + r1
    r0, r1 = 3, 6
    self.assertLess(r0, r1)
    self.assertLess(r1, chunk_len)

    tokens = jax.random.randint(
        jax.random.PRNGKey(1), (batch, total_len), 1, num_embed
    )

    cache = model.init_cache(batch, total_len, model.config.dtype)

    # --- Chunk 1: full prefix, all real ---
    tok1 = tokens[:, :chunk_len]
    pos1 = jnp.tile(jnp.arange(chunk_len)[None, :], (batch, 1))
    mask1 = jnp.tril(jnp.ones((chunk_len, chunk_len), dtype=jnp.bool_))[
        None, ...
    ]
    _, cache = model(tok1, positions=pos1, attention_mask=mask1, cache=cache)
    self.assertEqual(int(cache["layer_0"]["end_index"][0]), prefix_len)

    # --- Chunk 2: ragged suffix, right-padded with PAD (id 0) ---
    tok2 = tokens[:, chunk_len : 2 * chunk_len]
    real_counts = np.array([r0, r1], dtype=np.int32)
    col = np.arange(chunk_len)[None, :]
    input_mask_np = col < real_counts[:, None]  # non-uniform!
    input_mask = jnp.asarray(input_mask_np)
    # Zero out PAD-position token ids for realism.
    tok2 = jnp.where(input_mask, tok2, 0)

    self.assertGreater(
        int(input_mask[1].sum()),
        int(input_mask[0].sum()),
        "input_mask must be non-uniform to exercise the fix",
    )

    pos2 = jnp.tile(jnp.arange(chunk_len, 2 * chunk_len)[None, :], (batch, 1))
    full_mask = jnp.tril(
        jnp.ones((2 * chunk_len, 2 * chunk_len), dtype=jnp.bool_)
    )
    mask2 = full_mask[chunk_len:, :][None, ...]

    # With the deleted assertion this simply runs; the old code raised
    # "Chunked prefill requires uniform padding across the batch."
    logits, new_cache = model(
        tok2,
        positions=pos2,
        attention_mask=mask2,
        cache=cache,
        is_chunked_prefill=True,
        prefix_length=prefix_len,
        input_mask=input_mask,
    )

    end_index = int(new_cache["layer_0"]["end_index"][0])
    num_real_tokens = prefix_len + r1
    self.assertEqual(
        end_index,
        num_real_tokens,
        f"end_index should be num_real_tokens={num_real_tokens} (batch-max), "
        f"not seq_len={prefix_len + chunk_len} or element-0="
        f"{prefix_len + r0}",
    )
    self.assertEqual(logits.shape, (batch, chunk_len, num_embed))


class PrefixBucketTest(parameterized.TestCase):
  """Tests the prefix-length bucket ladders and bucketing helpers."""

  def test_pow2_buckets_default_ladder(self):
    self.assertEqual(
        model_lib.pow2_buckets(),
        (0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072),
    )

  def test_linear_buckets(self):
    self.assertEqual(
        model_lib.linear_buckets(step=512, max_len=2048),
        (0, 512, 1024, 1536, 2048),
    )

  def test_bucket_prefix_length_rounds_up(self):
    boundaries = model_lib.pow2_buckets()
    # Between rungs (128 < 200 <= 256) rounds up to 256.
    self.assertEqual(
        model_lib._bucket_prefix_length(200, 100000, boundaries), 256
    )

  def test_bucket_prefix_length_caps_at_cache_len(self):
    boundaries = model_lib.pow2_buckets()
    # Above the top boundary returns cache_len.
    self.assertEqual(
        model_lib._bucket_prefix_length(200000, 150000, boundaries), 150000
    )
    # Result never exceeds cache_len even when the bucket is larger.
    self.assertEqual(model_lib._bucket_prefix_length(200, 150, boundaries), 150)

  def test_maybe_bucket_passthrough(self):
    boundaries = model_lib.pow2_buckets()
    # Passthrough when not chunked prefill.
    self.assertEqual(
        model_lib._maybe_bucket_prefix_length(500, None, False, boundaries), 500
    )
    # Passthrough when prefix_length <= 0.
    self.assertEqual(
        model_lib._maybe_bucket_prefix_length(0, None, True, boundaries), 0
    )
    # Passthrough when boundaries is empty (bucketing disabled).
    self.assertEqual(
        model_lib._maybe_bucket_prefix_length(500, None, True, ()), 500
    )

  def test_maybe_bucket_pow2_rounding(self):
    cache = {"v": jnp.zeros((1, 1024, 1, 64))}  # cache_len == 1024
    self.assertEqual(
        model_lib._maybe_bucket_prefix_length(
            200, cache, True, model_lib.pow2_buckets()
        ),
        256,
    )

  def test_maybe_bucket_linear_rounding(self):
    boundaries = model_lib.linear_buckets(step=512, max_len=2048)
    cache = {"v": jnp.zeros((1, 2048, 1, 64))}  # cache_len == 2048
    self.assertEqual(
        model_lib._maybe_bucket_prefix_length(600, cache, True, boundaries),
        1024,
    )


class AttentionSimplificationTest(parameterized.TestCase):
  """CPU tests for the attention.py fail-fast guards (Changes 1, 3a, 3b)."""

  def _attn(self, attn_type, **overrides):
    config = _make_config(**overrides)
    return model_lib.Attention(config, attn_type, rngs=nnx.Rngs(0))

  # --- Change 1: split-without-flash config guard ---------------------------

  def test_split_without_flash_raises(self):
    with self.assertRaisesRegex(ValueError, "use_flash_attention=True"):
      _make_config(use_split_attention=True, use_flash_attention=False)

  def test_split_with_flash_ok(self):
    # Positive control: split + flash must NOT raise.
    config = _make_config(use_split_attention=True, use_flash_attention=True)
    self.assertTrue(config.use_split_attention)

  # --- Change 3a: LOCAL chunked-prefill mask builder ------------------------

  def test_shared_local_missing_origin_end_index_raises(self):
    attn = self._attn(model_lib.AttentionType.LOCAL_SLIDING)
    q_len, prefix_kv_len = 4, 4
    attn_mask = jnp.ones((1, q_len, prefix_kv_len + q_len), dtype=jnp.bool_)
    with self.assertRaisesRegex(ValueError, "shared LOCAL layer"):
      attn._build_local_chunked_prefill_mask(
          attn_mask,
          q_len,
          prefix_kv_len,
          prior_end_index=jnp.asarray(prefix_kv_len),
          kv_shared_cache={},  # origin prior_end_index absent -> must raise
          prefix_length=prefix_kv_len,
          kv_valid_mask=None,
          has_own_cache=False,
      )

  def test_local_chunked_prefill_mask_own_cache_ok(self):
    # Real (non-raising) branch: origin layer with its own cache.
    attn = self._attn(model_lib.AttentionType.LOCAL_SLIDING)
    q_len, prefix_kv_len = 4, 4
    attn_mask = jnp.ones((1, q_len, prefix_kv_len + q_len), dtype=jnp.bool_)
    out = attn._build_local_chunked_prefill_mask(
        attn_mask,
        q_len,
        prefix_kv_len,
        prior_end_index=jnp.asarray(prefix_kv_len),
        kv_shared_cache=None,
        prefix_length=prefix_kv_len,
        kv_valid_mask=None,
        has_own_cache=True,
    )
    self.assertEqual(out.shape, (1, q_len, prefix_kv_len + q_len))
    self.assertEqual(out.dtype, jnp.bool_)

  def test_local_chunked_prefill_mask_shared_origin_ok(self):
    # Real (non-raising) branch: shared layer with propagated origin index.
    attn = self._attn(model_lib.AttentionType.LOCAL_SLIDING)
    q_len, prefix_kv_len = 4, 4
    attn_mask = jnp.ones((1, q_len, prefix_kv_len + q_len), dtype=jnp.bool_)
    out = attn._build_local_chunked_prefill_mask(
        attn_mask,
        q_len,
        prefix_kv_len,
        prior_end_index=jnp.asarray(prefix_kv_len),
        kv_shared_cache={"prior_end_index": jnp.asarray(prefix_kv_len)},
        prefix_length=prefix_kv_len,
        kv_valid_mask=None,
        has_own_cache=False,
    )
    self.assertEqual(out.shape, (1, q_len, prefix_kv_len + q_len))

  # --- Change 3b: GLOBAL chunked-prefill mask builder -----------------------

  def test_shared_global_missing_origin_end_index_raises(self):
    attn = self._attn(model_lib.AttentionType.GLOBAL)
    q_len, prefix_length = 4, 4
    kv_len = prefix_length + q_len
    attn_mask = jnp.ones((1, q_len, kv_len), dtype=jnp.bool_)
    with self.assertRaisesRegex(ValueError, "shared GLOBAL layer"):
      attn._build_global_chunked_prefill_mask(
          attn_mask,
          q_len,
          kv_len,
          prior_end_index=jnp.asarray(prefix_length),
          kv_shared_cache={},  # origin prior_end_index absent -> must raise
          prefix_length=prefix_length,
          kv_valid_mask=None,
          has_own_cache=False,
      )

  def test_global_chunked_prefill_mask_own_cache_ok(self):
    attn = self._attn(model_lib.AttentionType.GLOBAL)
    q_len, prefix_length = 4, 4
    kv_len = prefix_length + q_len
    attn_mask = jnp.ones((1, q_len, kv_len), dtype=jnp.bool_)
    out = attn._build_global_chunked_prefill_mask(
        attn_mask,
        q_len,
        kv_len,
        prior_end_index=jnp.asarray(prefix_length),
        kv_shared_cache=None,
        prefix_length=prefix_length,
        kv_valid_mask=None,
        has_own_cache=True,
    )
    self.assertEqual(out.shape, (1, q_len, kv_len))
    self.assertEqual(out.dtype, jnp.bool_)

  def test_global_chunked_prefill_mask_shared_origin_ok(self):
    attn = self._attn(model_lib.AttentionType.GLOBAL)
    q_len, prefix_length = 4, 4
    kv_len = prefix_length + q_len
    attn_mask = jnp.ones((1, q_len, kv_len), dtype=jnp.bool_)
    out = attn._build_global_chunked_prefill_mask(
        attn_mask,
        q_len,
        kv_len,
        prior_end_index=jnp.asarray(prefix_length),
        kv_shared_cache={"prior_end_index": jnp.asarray(prefix_length)},
        prefix_length=prefix_length,
        kv_valid_mask=None,
        has_own_cache=False,
    )
    self.assertEqual(out.shape, (1, q_len, kv_len))


if __name__ == "__main__":
  absltest.main()
