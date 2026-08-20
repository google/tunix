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

"""Gemma4 model layers."""

from typing import Any, Tuple
import flax
from flax import nnx
import jax
from jax import numpy as jnp
import jaxtyping
from tunix.models.gemma4.config import ModelConfig
from tunix.models.gemma4.config import ShardingConfig
from tunix.utils.sharding_utils import shard


class Embedder(nnx.Module):
  """Embedder module."""

  def __init__(
      self,
      config: ModelConfig,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.vocab_size = config.num_embed
    self.embed_dim = config.embed_dim
    self.param_dtype = config.param_dtype

    self.input_embedding = nnx.Param(
        nnx.initializers.normal(dtype=self.param_dtype)(
            rngs.params(), (self.vocab_size, self.embed_dim)
        ),
        sharding=config.shd_config.emb_vd,
    )

    if config.per_layer_input_dim > 0:
      self.per_layer_model_projection = Einsum(
          einsum_str='BTD,DX->BTX',
          shape=(
              self.embed_dim,
              config.num_layers * config.per_layer_input_dim,
          ),
          sharding=config.shd_config.per_layer_model_projection,
          w_scale=(float(self.embed_dim) ** -0.5),
          rngs=rngs,
          dtype=self.config.dtype,
          param_dtype=self.param_dtype,
      )

      self.per_layer_projection_norm = RMSNorm(
          config.per_layer_input_dim,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=self.config.dtype,
          param_dtype=self.param_dtype,
      )
      self.per_layer_input_embedding = nnx.Param(
          nnx.initializers.normal(dtype=self.param_dtype)(
              rngs.params(),
              (self.vocab_size, config.num_layers * config.per_layer_input_dim),
          ),
          sharding=config.shd_config.per_layer_input_embedding,
      )

    if config.vision_encoder is not None:
      self.mm_input_projection = Einsum(
          einsum_str='...tm,md->...td',
          shape=(config.vision_encoder.d_model, self.embed_dim),
          sharding=config.shd_config.vision_proj,
          rngs=rngs,
          dtype=self.config.dtype,
          param_dtype=self.param_dtype,
      )
      self.mm_pre_projection_norm = RMSNorm(
          config.vision_encoder.d_model,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=self.config.dtype,
          param_dtype=self.param_dtype,
          with_scale=False,
      )

    if config.audio_encoder is not None:
      self.audio_input_projection = Einsum(
          einsum_str='...tm,md->...td',
          shape=(config.audio_encoder.lm_model_dims, self.embed_dim),
          rngs=rngs,
          sharding=config.shd_config.audio_proj,
          dtype=self.config.dtype,
          param_dtype=self.param_dtype,
      )
      self.audio_soft_embedding_norm = RMSNorm(
          self.embed_dim,
          rngs=rngs,
          sharding=config.shd_config,
          dtype=self.config.dtype,
          param_dtype=self.param_dtype,
          with_scale=False,
      )

  def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = self.input_embedding[(x,)]
    x *= jnp.sqrt(x.shape[-1]).astype(x.dtype)
    x = jnp.astype(x, self.config.dtype)
    x = shard(x, self.config.shd_config.act_btd)  # pyrefly: ignore[bad-argument-type]
    return x

  def encode_vision(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = self.mm_pre_projection_norm(x)  # pyrefly: ignore[bad-argument-type]
    x = self.mm_input_projection(x)
    return x

  def encode_audio(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    # projection and then norm is consistent with upstream gemma4.
    x = self.audio_input_projection(x)
    x = self.audio_soft_embedding_norm(x)
    return x

  def encode_per_layer_input(
      self, x: jaxtyping.ArrayLike, t: jaxtyping.ArrayLike
  ) -> jaxtyping.Array:
    t = jnp.where(
        jnp.logical_and(t >= 0, t < self.vocab_size), t, jnp.zeros_like(t)  # pyrefly: ignore[unsupported-operation]
    )
    x = self.per_layer_model_projection(x)
    x = jnp.reshape(
        x,
        (
            *x.shape[:-1],
            self.config.num_layers,
            self.config.per_layer_input_dim,
        ),
    )
    x = self.per_layer_projection_norm(x)
    y = self.per_layer_input_embedding.value[t]
    y = jnp.reshape(
        y,
        (
            *y.shape[:-1],
            self.config.num_layers,
            self.config.per_layer_input_dim,
        ),
    )
    y *= jnp.sqrt(self.config.per_layer_input_dim).astype(y.dtype)
    return (x + y) * jax.lax.rsqrt(2.0).astype(x.dtype)

  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = jnp.astype(x, self.config.dtype)
    w = jnp.astype(self.input_embedding.value, self.config.dtype)
    return jnp.dot(x, w.T)


def _make_dummy_images(
    vision_encoder: Any,
):
  """Make dummy patches/positions for initializing the vision encoder."""
  max_patches = vision_encoder.max_patches
  patch_dim = vision_encoder.patch_size**2 * 3
  dummy_patches = jnp.zeros((1, max_patches, patch_dim), dtype=jnp.float32)
  dummy_positions = jnp.full((1, max_patches, 2), -1, dtype=jnp.int32)
  return dummy_patches, dummy_positions


def _make_block_mask_indices(
    bidirectional_mask: jaxtyping.ArrayLike,  # (B, L)
) -> jaxtyping.ArrayLike:
  padded_mask = jnp.pad(bidirectional_mask, [(0, 0), (1, 0)], constant_values=0)
  boundary = padded_mask[..., 1:] > padded_mask[..., :-1]
  numbered_boundary = jnp.cumsum(boundary, axis=-1)
  return bidirectional_mask * numbered_boundary


def _add_bidirectional_mask(
    attn_mask: jaxtyping.ArrayLike,  # (B, L, L)/(B, L, KV_L) or (B, H, L, L)/(B, H, L, KV_L)
    bidirectional_mask: jaxtyping.ArrayLike,  # (B, L)
) -> jaxtyping.ArrayLike:
  q_block_indices = _make_block_mask_indices(bidirectional_mask)
  kv_block_indices = q_block_indices

  attn_shape = jnp.shape(attn_mask)
  kv_shape = jnp.shape(kv_block_indices)

  attn_kv_len = attn_shape[-1]
  if attn_kv_len != kv_shape[-1]:
    if attn_kv_len > kv_shape[-1]:
      pad_len = attn_kv_len - kv_shape[-1]
      kv_block_indices = jnp.pad(kv_block_indices, [(0, 0), (0, pad_len)])
    else:
      kv_block_indices = kv_block_indices[..., -attn_kv_len:]  # pyrefly: ignore[bad-index]

  bidir_cond = (kv_block_indices[:, None, :] == q_block_indices[..., None]) & (  # pyrefly: ignore[bad-index]
      q_block_indices[..., None] > 0  # pyrefly: ignore[bad-index]
  )

  if len(attn_shape) == 4:
    bidir_cond = jnp.expand_dims(bidir_cond, axis=1)

  attn_mask = attn_mask | bidir_cond
  return attn_mask


def _merge_flat_embeddings_inner(
    text_embeddings: jaxtyping.Array,  # (L, D)
    multimodal_embeddings: jaxtyping.Array,  # (T, D)
    mask: jaxtyping.Array,  # (L)
) -> jaxtyping.Array:
  target_pos = jnp.nonzero(mask, size=multimodal_embeddings.shape[0])
  first_pos = text_embeddings[0]
  merged = text_embeddings.at[target_pos, :].set(multimodal_embeddings)
  merged = merged.at[0].set(first_pos)
  return merged


def merge_flat_embeddings(
    *,
    text_embeddings: jaxtyping.Array,  # (B, L, D)
    multimodal_embeddings: jaxtyping.Array,  # (B, T, D)
    mask: jaxtyping.Array,  # (B, L)
) -> jaxtyping.Array:
  return jax.vmap(_merge_flat_embeddings_inner, in_axes=(0, 0, 0))(
      text_embeddings, multimodal_embeddings, mask
  )


class Einsum(nnx.Module):
  """Einsum module."""

  def __init__(
      self,
      einsum_str: str,
      shape: flax.typing.Shape,
      *,
      rngs: nnx.Rngs,
      sharding: Tuple[str | None, ...],
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
      w_scale: float | None = None,
  ):
    self.einsum_str = einsum_str
    self.dtype = dtype
    self.w_scale = w_scale

    self.shape = shape
    self.w = nnx.Param(
        nnx.initializers.normal(dtype=param_dtype)(rngs.params(), shape),
        sharding=sharding,
    )

  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    w = self.w.value
    if self.w_scale is not None:
      w = w * self.w_scale
    x = jnp.astype(x, self.dtype)
    w = jnp.astype(w, self.dtype)
    return jnp.einsum(self.einsum_str, x, w)


class RMSNorm(nnx.Module):
  """RMSNorm layer."""

  def __init__(
      self,
      dim: int,
      *,
      rngs: nnx.Rngs,
      sharding: ShardingConfig = ShardingConfig.get_default_sharding(),
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
      with_scale: bool = True,
  ):
    self.with_scale = with_scale
    if with_scale:
      self.scale = nnx.Param(
          nnx.initializers.ones_init()(rngs.params(), dim).astype(param_dtype),  # pyrefly: ignore[bad-argument-type]
          sharding=sharding.rms_norm_weight,
      )
    self.dtype = dtype

  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    x = jnp.astype(x, jnp.float32)
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_inputs = x * jax.lax.rsqrt(var + 1e-06).astype(x.dtype)
    if self.with_scale:
      scale = jnp.expand_dims(self.scale.value, axis=range(len(x.shape) - 1))
      normed_inputs = normed_inputs * scale
    return normed_inputs.astype(self.dtype)


def apply_rope(
    inputs: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: int,
    scale_factor: float = 1.0,
    rope_proportion: float = 1.0,
) -> jax.Array:
  """Applies RoPE.

  Let B denote batch size, L denote sequence length, N denote number of heads,
  and H denote head dimension. Note that H must be divisible by 2.

  Args:
    inputs: Array of shape [B, L, N, H].
    positions:  Array of shape [B, L].
    base_frequency: Base frequency used to compute rotations.
    scale_factor: The scale factor used for positional interpolation, allowing
      an expansion of sequence length beyond the pre-trained context length.
    rope_proportion: The proportion of the head dimension to apply RoPE to.

  Returns:
    Array of shape [B, L, N, H].
  """
  head_dim = inputs.shape[-1]
  rope_angles = int(rope_proportion * head_dim // 2)
  nope_angles = head_dim // 2 - rope_angles
  freq_exponents = (2.0 / head_dim) * jnp.arange(
      0, rope_angles, dtype=jnp.float32
  )
  timescale = jnp.pad(
      base_frequency**freq_exponents,
      (0, nope_angles),
      mode='constant',
      constant_values=(0, jnp.inf),
  )

  sinusoid_inp = (
      positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  if scale_factor < 1.0:
    raise ValueError(f'scale_factor must be >= 1.0, got {scale_factor}')
  sinusoid_inp /= scale_factor

  sin = jnp.sin(sinusoid_inp)
  cos = jnp.cos(sinusoid_inp)

  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)
