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
from flax import nnx
import jax
import jax.numpy as jnp
from tunix.models.gemma4 import model as model_lib


class ModelTest(absltest.TestCase):

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


if __name__ == "__main__":
  absltest.main()
