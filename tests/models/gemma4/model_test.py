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
    tokens = tokens.at[0, 10:15].set(model_lib.VISION_TOKEN_PLACEHOLDER)

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
    tokens = tokens.at[0, 10:15].set(model_lib.VISION_TOKEN_PLACEHOLDER)

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
    tokens = tokens.at[0, 10:15].set(model_lib.VISION_TOKEN_PLACEHOLDER)
    tokens = tokens.at[1, 5:8].set(model_lib.VISION_TOKEN_PLACEHOLDER)
    tokens = tokens.at[1, 20:25].set(model_lib.VISION_TOKEN_PLACEHOLDER)

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
    config.audio_encoder = model_lib.audio.AudioEncoderConfig(
        num_layers=1,
        model_dims=64,
        lm_model_dims=128,
        atten_num_heads=2,
    )

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs, text_only=False)

    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (1, 32), 0, config.num_embed
    )
    tokens = tokens.at[0, 10:15].set(model_lib.AUDIO_TOKEN_PLACEHOLDER)

    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]

    soft_token_counts = (5,)
    audio_data = jnp.zeros((1, 1, 16000), dtype=jnp.float32)
    audio_lengths = jnp.full((1, 1), 16000, dtype=jnp.int32)

    audios = model_lib.PreprocessedAudioInput(
        audio=audio_data,
        audio_lengths=audio_lengths,
        soft_token_counts=soft_token_counts,
    )

    logits, _ = model(
        tokens,
        positions=positions,
        attention_mask=attn_mask,
        audios=audios,
    )
    self.assertEqual(logits.shape, (1, 32, config.num_embed))

  def test_forward_pass_audio_batch(self):
    config = model_lib.ModelConfig.gemma4_e2b()
    config.num_layers = 1
    config.embed_dim = 256
    config.hidden_dim = 512
    config.num_heads = 4
    config.head_dim = 64
    config.num_kv_heads = 1
    config.frac_shared_layers = 0.0
    config.audio_encoder = model_lib.audio.AudioEncoderConfig(
        num_layers=1,
        model_dims=64,
        lm_model_dims=128,
        atten_num_heads=2,
    )

    rngs = nnx.Rngs(0)
    model = model_lib.Gemma4(config, rngs=rngs, text_only=False)

    batch_size = 2
    seq_len = 32
    tokens = jax.random.randint(
        jax.random.PRNGKey(0), (batch_size, seq_len), 0, config.num_embed
    )
    tokens = tokens.at[0, 10:15].set(model_lib.AUDIO_TOKEN_PLACEHOLDER)
    tokens = tokens.at[1, 5:8].set(model_lib.AUDIO_TOKEN_PLACEHOLDER)
    tokens = tokens.at[1, 20:25].set(model_lib.AUDIO_TOKEN_PLACEHOLDER)

    positions = jnp.tile(
        jnp.arange(tokens.shape[1])[None, :], (tokens.shape[0], 1)
    )
    attn_mask = jnp.tril(
        jnp.ones((tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )[None, ...]
    attn_mask = jnp.broadcast_to(attn_mask, (batch_size, seq_len, seq_len))

    soft_token_counts = ((5,), (3, 5))
    max_n_audios = 2
    max_samples = 16000

    audio_data = jnp.zeros(
        (batch_size, max_n_audios, max_samples), dtype=jnp.float32
    )
    audio_lengths = jnp.full(
        (batch_size, max_n_audios), max_samples, dtype=jnp.int32
    )
    audio_lengths = audio_lengths.at[0, 1].set(0)

    audios = model_lib.PreprocessedAudioInput(
        audio=audio_data,
        audio_lengths=audio_lengths,
        soft_token_counts=soft_token_counts,
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
