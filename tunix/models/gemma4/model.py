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

"""Gemma4 model."""

import dataclasses
from functools import partial
import itertools
from typing import Any, Optional, Tuple
from flax import nnx
import jax
from jax import numpy as jnp
import jaxtyping
from tunix.generate.mappings import BackendMappingMixin
from tunix.utils import compat

# Re-export symbols to preserve public API paths after module decomposition.
# pylint: disable=g-multiple-import,unused-import

from tunix.models.gemma4 import audio
from tunix.models.gemma4 import moe
from tunix.models.gemma4 import vision

from tunix.models.gemma4.config import (
    AUDIO_SOFT_TOKEN_PLACEHOLDER,
    IMAGE_SOFT_TOKEN_PLACEHOLDER,
    AttentionType,
    Cache,
    GEMMA4_ATTENTION_PATTERN,
    K_MASK,
    LayerCache,
    ModelConfig,
    PreprocessedAudioInput,
    PreprocessedVisionInput,
    RematConfig,
    ShardingConfig,
    SplashAttentionImpl,
    create_kv_cache_sharing_patterns,
)
from tunix.models.gemma4.layers import (
    Einsum,
    Embedder,
    RMSNorm,
    _add_bidirectional_mask,
    _make_block_mask_indices,
    _make_dummy_images,
    _merge_flat_embeddings_inner,
    apply_rope,
    merge_flat_embeddings,
)
from tunix.models.gemma4.attention import (
    Attention,
    create_sliding_window_mask,
    find_last_one_index,
)

# pylint: enable=g-multiple-import,unused-import


class FeedForward(nnx.Module):
  """Feed forward module."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      hidden_dim: int | None = None,
      rngs: nnx.Rngs,
  ):
    self.config = config
    h_dim = hidden_dim if hidden_dim is not None else config.hidden_dim
    self.gate_proj = nnx.Linear(
        config.embed_dim,
        h_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            config.shd_config.ffw_weight_df,
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

    self.up_proj = nnx.Linear(
        config.embed_dim,
        h_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            config.shd_config.ffw_weight_df,
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.down_proj = nnx.Linear(
        h_dim,
        config.embed_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(), config.shd_config.ffw_weight_fd
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def block(self, x):
    return self.down_proj(nnx.gelu(self.gate_proj(x)) * self.up_proj(x))

  def __call__(self, x):
    remat_config = getattr(self.config, 'remat_config', RematConfig.NONE)
    if (
        remat_config == RematConfig.BLOCK
        or remat_config == RematConfig.BLOCK.value
    ):
      graphdef, state = nnx.split(self)

      def _checkpointed_block(state, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        return module.block(*args, **kwargs)

      return jax.checkpoint(_checkpointed_block)(state, x)
    else:
      return self.block(x)


class DecoderLayer(nnx.Module):
  """Decoder layer."""

  def __init__(
      self,
      config: ModelConfig,
      attn_type: AttentionType,
      *,
      hidden_dim: int | None = None,
      rngs: nnx.Rngs,
  ):

    self.config = config
    self.pre_attention_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

    self.attn = Attention(
        config=config,
        attn_type=attn_type,
        rngs=rngs,
    )
    self.post_attention_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.pre_ffw_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.mlp = FeedForward(config=config, hidden_dim=hidden_dim, rngs=rngs)

    if config.enable_moe:
      self.moe_pre_ffw_norm = RMSNorm(
          config.embed_dim,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
      self.moe = moe.MoERagged(
          config=config,
          rngs=rngs,
      )
      self.moe_post_ffw_norm = RMSNorm(
          config.embed_dim,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
      self.dense_post_ffw_norm = RMSNorm(
          config.embed_dim,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
    self.post_ffw_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

    if config.per_layer_input_dim > 0:

      self.per_layer_input_gate = Einsum(
          einsum_str='BTD,DP->BTP',
          shape=(config.embed_dim, config.per_layer_input_dim),
          sharding=config.shd_config.per_layer_input_gate,
          rngs=rngs,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )

      self.per_layer_projection = Einsum(
          einsum_str='BTP,PD->BTD',
          shape=(config.per_layer_input_dim, config.embed_dim),
          sharding=config.shd_config.per_layer_projection,
          rngs=rngs,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )

      self.post_per_layer_input_norm = RMSNorm(
          config.embed_dim,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )

    self.skip_scale = nnx.Param(jnp.ones((1,), dtype=config.param_dtype))

  def block(
      self,
      x,
      segment_pos,
      cache,
      attn_mask,
      per_layer_input=None,
      kv_shared_cache=None,
      segment_ids=None,
  ):
    norm = self.pre_attention_norm(x)
    cache, attn, kv = self.attn(
        norm,
        segment_pos,
        cache,
        attn_mask,
        kv_shared_cache=kv_shared_cache,
        segment_ids=segment_ids,
    )
    attn = self.post_attention_norm(attn)
    attn += x

    norm_ffw = self.pre_ffw_norm(attn)
    ffw = self.mlp(norm_ffw)
    if self.config.enable_moe:
      ffw = self.dense_post_ffw_norm(ffw)
      moe_norm_ffw = self.moe_pre_ffw_norm(attn)
      moe_out = self.moe(moe_norm_ffw, router_input=attn)
      moe_out = self.moe_post_ffw_norm(moe_out)
      ffw += moe_out
    ffw = self.post_ffw_norm(ffw)

    ffw += attn

    if self.config.per_layer_input_dim > 0 and per_layer_input is not None:
      gating_input = ffw
      mapped = self.per_layer_input_gate(gating_input)
      mapped = jax.nn.gelu(mapped) * per_layer_input
      mapped = self.per_layer_projection(mapped)
      mapped = self.post_per_layer_input_norm(mapped)
      ffw += mapped

    ffw = ffw * self.skip_scale.value
    return cache, ffw, kv

  def __call__(
      self,
      x,
      segment_pos,
      cache,
      attn_mask,
      per_layer_input=None,
      kv_shared_cache=None,
      segment_ids=None,
  ):
    remat_config = getattr(self.config, 'remat_config', RematConfig.NONE)
    if (
        remat_config == RematConfig.DECODER
        or remat_config == RematConfig.DECODER.value
    ):
      graphdef, state = nnx.split(self)

      def _checkpointed_block(state, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        return module.block(*args, **kwargs)

      return jax.checkpoint(_checkpointed_block)(
          state,
          x,
          segment_pos,
          cache,
          attn_mask,
          per_layer_input,
          kv_shared_cache,
          segment_ids,
      )
    else:
      return self.block(
          x,
          segment_pos,
          cache,
          attn_mask,
          per_layer_input,
          kv_shared_cache,
          segment_ids=segment_ids,
      )

  def init_cache(self, batch_size, max_seq_len, dtype):
    return self.attn.init_cache(batch_size, max_seq_len, dtype)


class Gemma4(BackendMappingMixin, nnx.Module):
  """Gemma4 model."""

  def __init__(
      self, config: ModelConfig, *, rngs: nnx.Rngs, text_only: bool = True
  ):
    self.text_only = text_only
    if text_only:
      config = dataclasses.replace(
          config, vision_encoder=None, audio_encoder=None
      )
    self.config = config
    self.embedder = Embedder(config, rngs=rngs)

    if config.vision_encoder is not None:
      self.vision_encoder = vision.VisionEncoder(
          rngs=rngs,
          config=config.vision_encoder,
          param_dtype=config.param_dtype,
          shd_config=config.shd_config.vision_shd,
      )

    if config.audio_encoder is not None:
      self.audio_encoder = audio.AudioTokenizer(
          rngs=rngs,
          config=config.audio_encoder,
      )

    pattern = (
        config.attention_pattern
        if config.attention_pattern
        else GEMMA4_ATTENTION_PATTERN
    )
    attention_types = [
        attn_type
        for _, attn_type in zip(
            range(config.num_layers), itertools.cycle(pattern)
        )
    ]
    self.kv_cache_sharing_patterns = create_kv_cache_sharing_patterns(
        num_layers=config.num_layers,
        frac_shared_layers=config.frac_shared_layers,
        share_global=True,
        share_local=True,
        attention_types=tuple(attention_types),
    )
    # Layers that shared layers depend on.
    self.shared_layer_origins = {
        j for i, j in enumerate(self.kv_cache_sharing_patterns) if i != j
    }

    self.layers = compat.ModuleList()
    for i in range(config.num_layers):
      attn_type = attention_types[i]
      h_dim = config.hidden_dim
      if (
          self.kv_cache_sharing_patterns[i] != i
          and config.override_kv_shared_ffw_hidden is not None
      ):
        h_dim = config.override_kv_shared_ffw_hidden
      self.layers.append(
          DecoderLayer(
              config=config, attn_type=attn_type, hidden_dim=h_dim, rngs=rngs
          )
      )

    self.final_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def __call__(
      self,
      tokens,
      positions=None,
      cache=None,
      attention_mask=None,
      segment_ids=None,
      decode_only_last_token: bool = False,
      images: PreprocessedVisionInput | None = None,
      audios: PreprocessedAudioInput | None = None,
      skip_lm_head: bool = False,
  ):
    if positions is None:
      B, T = tokens.shape  # pylint: disable=invalid-name
      positions = jnp.tile(jnp.arange(T)[None, :], (B, 1))

    return_cache = cache is not None
    new_cache = {}
    is_prefill = tokens.shape[1] > 1

    x = self.embedder.encode(tokens)
    if self.config.vision_encoder is not None and images is not None:
      soft_embeddings = self._encode_vision(images)
      mask = tokens == IMAGE_SOFT_TOKEN_PLACEHOLDER
      x = merge_flat_embeddings(
          text_embeddings=x,
          multimodal_embeddings=soft_embeddings,
          mask=mask,
      )

    if self.config.audio_encoder is not None and audios is not None:
      soft_embeddings = self._encode_audio(audios)
      mask = tokens == AUDIO_SOFT_TOKEN_PLACEHOLDER
      x = merge_flat_embeddings(
          text_embeddings=x,
          multimodal_embeddings=soft_embeddings,
          mask=mask,
      )

    sliding_attention_mask = None
    if (
        is_prefill
        and self.config.use_bidirectional_attention == 'vision'
        and images is not None
        and attention_mask is not None
    ):
      bidirectional_mask = tokens == IMAGE_SOFT_TOKEN_PLACEHOLDER
      sliding_attention_mask = _add_bidirectional_mask(
          attention_mask, bidirectional_mask
      )

    per_layer_inputs = None
    if self.config.per_layer_input_dim > 0:
      per_layer_inputs = self.embedder.encode_per_layer_input(x, tokens)

    # Stores the raw KV projections for the current forward pass. Used for
    # KV cache sharing during prefill.
    transient_kvs = {}

    for i, layer in enumerate(self.layers):
      layer_name = f'layer_{i}'

      shared_idx = self.kv_cache_sharing_patterns[i]
      is_shared = shared_idx != i
      if is_shared:
        assert shared_idx in self.shared_layer_origins
        layer_cache = None
        shared_layer_name = f'layer_{shared_idx}'
        if is_prefill:
          # During prefill, use full KV projections from the shared layer.
          shared_k, shared_v = transient_kvs[shared_layer_name]
          kv_shared_cache = {'k': shared_k, 'v': shared_v}
        else:
          # During decoding, use the shared layer's cache (which may be
          # an optimized sliding window ring cache).
          kv_shared_cache = new_cache.get(shared_layer_name)
      else:
        layer_cache = cache[layer_name] if cache else None
        kv_shared_cache = None

      layer_attn_mask = attention_mask
      if (
          sliding_attention_mask is not None
          and layer.attn.attn_type == AttentionType.LOCAL_SLIDING
      ):
        layer_attn_mask = sliding_attention_mask

      layer_cache, x, layers_kvs = layer(
          x,
          positions,
          layer_cache,
          layer_attn_mask,
          per_layer_input=per_layer_inputs[:, :, i, :]
          if per_layer_inputs is not None
          else None,
          kv_shared_cache=kv_shared_cache,
          segment_ids=segment_ids,
      )
      if is_prefill and i in self.shared_layer_origins:
        transient_kvs[layer_name] = layers_kvs
      if not is_shared:
        new_cache[layer_name] = layer_cache

    x = self.final_norm(x)
    if skip_lm_head:
      return x, (new_cache if return_cache else None)

    if decode_only_last_token:
      # Only compute logits for the last token. This can significantly reduce
      # memory requirements during prefill (when sampling), since we only need
      # the logits for the last token to sample from.
      x = x[:, -1:, :]

    logits = self.compute_final_logits(x)

    return logits, (new_cache if return_cache else None)  # pytype: disable=container-type-mismatch

  def _encode_vision(self, vision_input: PreprocessedVisionInput):
    """Encode images into the same space as the text embeddings."""
    assert self.vision_encoder is not None

    batch_size = vision_input.patches.shape[0]

    if len(vision_input.soft_token_counts) > 0 and isinstance(
        vision_input.soft_token_counts[0], int
    ):
      soft_token_counts = (vision_input.soft_token_counts,)
    else:
      soft_token_counts = vision_input.soft_token_counts

    max_n_images = max((len(counts) for counts in soft_token_counts), default=0)  # pyrefly: ignore[bad-argument-type]
    if max_n_images == 0:
      return jnp.zeros((batch_size, 0, self.config.embed_dim))

    patches = vision_input.patches
    positions_xy = vision_input.positions_xy
    max_patches = patches.shape[1] // max_n_images

    patches = jnp.reshape(
        patches, (batch_size * max_n_images, max_patches, patches.shape[2])
    )
    positions_xy = jnp.reshape(
        positions_xy,
        (batch_size * max_n_images, max_patches, positions_xy.shape[2]),
    )

    encoder_outputs = self.vision_encoder(patches, positions_xy)

    embeddings, mask = encoder_outputs[0]

    batch_tokens = []
    max_tokens_per_batch = 0
    for b in range(batch_size):
      per_image_tokens = []
      counts = soft_token_counts[b] if b < len(soft_token_counts) else ()
      for i in range(len(counts)):  # pyrefly: ignore[bad-argument-type]
        idx = b * max_n_images + i
        expected_count = counts[i]  # pyrefly: ignore[bad-index]
        if mask is not None:
          valid_indices = jnp.nonzero(mask[idx], size=expected_count)[0]  # pyrefly: ignore[bad-argument-type]
          real_tokens = embeddings[idx][valid_indices]
        else:
          real_tokens = embeddings[idx][:expected_count]
        per_image_tokens.append(real_tokens)

      if per_image_tokens:
        b_tokens = jnp.concatenate(per_image_tokens, axis=0)
      else:
        b_tokens = jnp.zeros((0, embeddings.shape[-1]))
      batch_tokens.append(b_tokens)
      max_tokens_per_batch = max(max_tokens_per_batch, b_tokens.shape[0])

    padded_batch_tokens = []
    for b_tokens in batch_tokens:
      pad_len = max_tokens_per_batch - b_tokens.shape[0]
      if pad_len > 0:
        b_tokens = jnp.pad(b_tokens, ((0, pad_len), (0, 0)))
      padded_batch_tokens.append(b_tokens)

    all_tokens = jnp.stack(padded_batch_tokens, axis=0)
    all_tokens = self.embedder.encode_vision(all_tokens[:, None, :, :])
    all_tokens = all_tokens[:, 0, :, :]
    return all_tokens

  def _encode_audio(self, audio_input: PreprocessedAudioInput):
    """Encode audio.

    Args:
      audio_input: The audio input.

    Returns:
      Padded audio embeddings as a tensor of shape, with padding
      at the end of the sequences. (batch_size, max_tokens)
    """
    batch_size, num_clips = audio_input.audios.shape[:2]

    # Encode audio clips.
    clips = audio_input.audios.reshape(batch_size * num_clips, -1)
    clip_lengths = audio_input.sequence_lengths.reshape(batch_size * num_clips)
    embeddings, pad_mask = self.audio_encoder(clips, clip_lengths)

    flat_embeddings = embeddings.reshape(batch_size, -1, embeddings.shape[-1])
    flat_pad_mask = pad_mask.reshape(batch_size, -1)  # True => Pad.

    # Handle padding in the embeddings.
    # To avoid JIT recompilation, we want to keep the output shape consistent
    # across invocations with differring values of audio_input.sequence_lengths
    # (of course, as long as audio_input.audios is padded to the same shape).
    # Thus, we don't simply truncate each clip's embeddings as that would create
    # variable length output. We keep the length of embeddings the same, but
    # move valid (non-padding) embeddings to the beginning of sequence
    # (i.e. pack valid embeddings into one contiguous sequence).
    max_tokens = flat_pad_mask.shape[-1]
    indices = jnp.arange(max_tokens)
    indices = jnp.where(flat_pad_mask, max_tokens, indices)
    sorted_indices = jnp.argsort(indices, axis=-1)
    packed_embeddings = jnp.take_along_axis(
        flat_embeddings, sorted_indices[..., None], axis=1
    )

    result = self.embedder.encode_audio(packed_embeddings)
    return result

  def compute_final_logits(
      self,
      x: jaxtyping.Array,
  ) -> jaxtyping.Array:
    """Computes the final logits from the model output."""
    logits = self.embedder.decode(x).astype(jnp.float32)
    if self.config.final_logit_softcap is not None:
      logits /= self.config.final_logit_softcap
      logits = jnp.tanh(logits) * self.config.final_logit_softcap
    return logits

  def init_cache(self, batch_size, max_seq_len, dtype):
    cache = {}
    for i, layer in enumerate(self.layers):
      if self.kv_cache_sharing_patterns[i] != i:
        continue  # Skip shared layers.
      cache[f'layer_{i}'] = layer.init_cache(batch_size, max_seq_len, dtype)
    return cache

  def get_model_input(self):
    """Returns a dummy model input for the transformer.

    This dummy input has a batch size compatible with FSDP sharding on a
    2-device axis.
    """
    dummy_batch_size = 2
    dummy_seq_len = 2
    return {
        'tokens': jnp.ones((dummy_batch_size, dummy_seq_len), dtype=jnp.int32),
        'positions': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'cache': None,
        'attention_mask': jnp.ones(
            (dummy_batch_size, 1, dummy_seq_len), dtype=jnp.bool
        ),
    }

  @property
  def num_embed(self) -> int:
    return self.config.num_embed
