# Copyright 2025 Google LLC
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

"""Gemma implementation of the vision encoder."""

from __future__ import annotations

import dataclasses
from typing import Tuple, cast

import chex
import einops
from flax import nnx
from flax.nnx import initializers
import jax
from jax import numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping
import numpy as np
from tunix.utils import compat

BEGIN_IMAGE_TOKEN = 255999
# TODO(abheesht17): Pull this from the preprocessor because it is different for
# HF and GDM Gemma 3 tokenisers.
END_IMAGE_TOKEN = 256000
NEW_LINE_TOKEN = 108
TOKEN_PLACEHOLDER = 262144
NUM_PLACEHOLDER_TOKENS_PER_IMAGE = 256
NUM_TOKENS_PER_MEDIA = NUM_PLACEHOLDER_TOKENS_PER_IMAGE + 4


def shard(x: jnp.ndarray, s: Tuple[str, ...]):
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == "cpu":
    return x
  return jax.lax.with_sharding_constraint(
      x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
  )


@dataclasses.dataclass(slots=True, frozen=True)
class SigLIPShardingConfig:
  """Sharding configuration for SigLIP vision encoder."""

  emb_patch_kernel: Tuple[str | None, ...]
  emb_patch_bias: Tuple[str | None, ...]
  emb_pos_kernel: Tuple[str | None, ...]

  attn_qkv_kernel: Tuple[str | None, ...]
  attn_out_kernel: Tuple[str | None, ...]
  attn_qkv_bias: Tuple[str | None, ...]
  attn_out_bias: Tuple[str | None, ...]

  fc1_kernel: Tuple[str | None, ...]
  fc1_bias: Tuple[str | None, ...]
  fc2_kernel: Tuple[str | None, ...]
  fc2_bias: Tuple[str | None, ...]

  act_btd: Tuple[str | None, ...]
  act_bnts: Tuple[str | None, ...]
  act_bhwd: Tuple[str | None, ...]
  layer_norm: Tuple[str | None, ...]

  @staticmethod
  def get_default_sharding(is_sampling: bool = False):
    fsdp = "fsdp" if not is_sampling else None
    return SigLIPShardingConfig(
        emb_patch_kernel=(None, None, None, "tp"),
        emb_patch_bias=("tp",),
        emb_pos_kernel=(None, None, "tp"),
        attn_qkv_kernel=(fsdp, "tp"),
        attn_out_kernel=("tp", fsdp),
        attn_qkv_bias=("tp",),
        attn_out_bias=(fsdp,),
        # Activation sharding
        act_btd=(fsdp, None, "tp"),
        act_bnts=(fsdp, "tp", None, None),
        act_bhwd=(fsdp, None, None, "tp"),
        layer_norm=("tp",),
        fc1_kernel=(fsdp, "tp"),
        fc1_bias=("tp",),
        fc2_kernel=("tp", fsdp),
        fc2_bias=(fsdp,),
    )


class VisionAttention(nnx.Module):
  """Attention layer."""

  def __init__(
      self,
      hidden_dim: int,
      num_heads: int,
      dropout: float = 0.0,
      *,
      rngs: nnx.Rngs,
      dtype_mm: jnp.dtype = jnp.float32,
      shd_config: SigLIPShardingConfig | None = None,
  ):
    self.hidden_dim = hidden_dim
    self.num_heads = num_heads
    self.head_dim = hidden_dim // num_heads
    self.scale = self.head_dim**-0.5

    # Projections
    self.query_proj = nnx.Linear(
        hidden_dim,
        hidden_dim,
        kernel_init=nnx.with_partitioning(
            initializers.xavier_uniform(),
            shd_config.attn_qkv_kernel if shd_config else (),
        ),
        bias_init=nnx.with_partitioning(
            initializers.normal(stddev=1e-6),
            shd_config.attn_qkv_bias if shd_config else (),
        ),
        param_dtype=dtype_mm,
        rngs=rngs,
    )
    self.key_proj = nnx.Linear(
        hidden_dim,
        hidden_dim,
        kernel_init=nnx.with_partitioning(
            initializers.xavier_uniform(),
            shd_config.attn_qkv_kernel if shd_config else (),
        ),
        bias_init=nnx.with_partitioning(
            initializers.normal(stddev=1e-6),
            shd_config.attn_qkv_bias if shd_config else (),
        ),
        param_dtype=dtype_mm,
        rngs=rngs,
    )
    self.value_proj = nnx.Linear(
        hidden_dim,
        hidden_dim,
        kernel_init=nnx.with_partitioning(
            initializers.xavier_uniform(),
            shd_config.attn_qkv_kernel if shd_config else (),
        ),
        bias_init=nnx.with_partitioning(
            initializers.normal(stddev=1e-6),
            shd_config.attn_qkv_bias if shd_config else (),
        ),
        param_dtype=dtype_mm,
        rngs=rngs,
    )
    self.out_proj = nnx.Linear(
        hidden_dim,
        hidden_dim,
        kernel_init=nnx.with_partitioning(
            initializers.xavier_uniform(),
            shd_config.attn_out_kernel if shd_config else (),
        ),
        bias_init=nnx.with_partitioning(
            initializers.normal(stddev=1e-6),
            shd_config.attn_out_bias if shd_config else (),
        ),
        param_dtype=dtype_mm,
        rngs=rngs,
    )
    self.dropout = nnx.Dropout(rate=dropout)

    self.shd_config = shd_config

  def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
    batch_size, seq_length, _ = x.shape
    desired_shape = (batch_size, seq_length, self.num_heads, self.head_dim)

    q = self.query_proj(x)
    k = self.key_proj(x)
    v = self.value_proj(x)
    if self.shd_config:
      q = shard(q, self.shd_config.act_btd)
      k = shard(k, self.shd_config.act_btd)
      v = shard(v, self.shd_config.act_btd)

    q = q.reshape(desired_shape)
    k = k.reshape(desired_shape)
    v = v.reshape(desired_shape)

    logits = jnp.einsum("BTNH,BSNH->BNTS", q, k)
    if self.shd_config:
      logits = shard(logits, self.shd_config.act_bnts)
    logits *= self.scale

    probs = jax.nn.softmax(logits, axis=-1)
    probs = self.dropout(probs, deterministic=deterministic)

    out = jnp.einsum("BNTS,BSNH->BTNH", probs, v).reshape(
        batch_size, seq_length, self.hidden_dim
    )
    if self.shd_config:
      out = shard(out, self.shd_config.act_btd)

    # 5. Final Output Projection
    out = self.out_proj(out)
    if self.shd_config:
      out = shard(out, self.shd_config.act_btd)
    return out


class MlpBlock(nnx.Module):
  """Transformer MLP / feed-forward block."""

  def __init__(
      self,
      width: int,
      block_id: int,
      *,
      rngs: nnx.Rngs,
      mlp_dim: int | None = None,  # Defaults to 4x input dim
      dropout: float = 0.0,
      dtype_mm: jnp.dtype = jnp.float32,
      shd_config: SigLIPShardingConfig | None = None,
  ):
    self.block_id = block_id
    self.mlp_dim = mlp_dim
    self.dropout_rate = dropout
    self.dtype_mm = dtype_mm
    self.width = width
    mlp_dim = self.mlp_dim or 4 * self.width
    self.fc1 = nnx.Linear(
        self.width,
        mlp_dim,
        kernel_init=nnx.with_partitioning(
            initializers.xavier_uniform(),
            shd_config.fc1_kernel if shd_config else (),
        ),
        bias_init=nnx.with_partitioning(
            initializers.normal(stddev=1e-6),
            shd_config.fc1_bias if shd_config else (),
        ),
        param_dtype=self.dtype_mm,
        rngs=rngs,
    )
    self.dropout = nnx.Dropout(rate=self.dropout_rate)
    self.fc2 = nnx.Linear(
        mlp_dim,
        self.width,
        kernel_init=nnx.with_partitioning(
            initializers.xavier_uniform(),
            shd_config.fc2_kernel if shd_config else (),
        ),
        bias_init=nnx.with_partitioning(
            initializers.normal(stddev=1e-6),
            shd_config.fc2_bias if shd_config else (),
        ),
        param_dtype=self.dtype_mm,
        rngs=rngs,
    )

    self.shd_config = shd_config

  def __call__(
      self, x: jaxtyping.ArrayLike, deterministic: bool = True
  ) -> jaxtyping.ArrayLike:
    """Applies Transformer MlpBlock module.

    Args:
      x: Input tensor.
      deterministic: Whether to run in deterministic mode (e.g., disable
        dropout).

    Returns:
      The output tensor.
    """
    x = self.fc1(x)
    x = nnx.gelu(x, approximate=True)

    if self.shd_config:
      x = shard(x, self.shd_config.act_btd)

    x = self.dropout(x, deterministic=deterministic)
    if self.shd_config:
      x = shard(x, self.shd_config.act_btd)

    x = self.fc2(x)
    if self.shd_config:
      x = shard(x, self.shd_config.act_btd)
    return x


class Encoder1DBlock(nnx.Module):
  """Single transformer encoder block (MHSA + MLP)."""

  def __init__(
      self,
      *,
      width: int,
      block_id: int,
      mlp_dim: int | None = None,  # Defaults to 4x input dim
      num_heads: int = 12,
      dropout: float = 0.0,
      rngs: nnx.Rngs,
      dtype_mm: jnp.dtype = jnp.float32,
      shd_config: SigLIPShardingConfig | None = None,
  ):
    self.ln1 = nnx.LayerNorm(
        num_features=width,
        scale_init=nnx.with_partitioning(
            initializers.ones,
            shd_config.layer_norm if shd_config else (),
        ),
        bias_init=nnx.with_partitioning(
            initializers.zeros,
            shd_config.layer_norm if shd_config else (),
        ),
        rngs=rngs,
    )
    self.attn = VisionAttention(
        hidden_dim=width,
        num_heads=num_heads,
        dropout=dropout,
        dtype_mm=dtype_mm,
        rngs=rngs,
    )
    self.dropout = nnx.Dropout(rate=dropout)
    self.ln2 = nnx.LayerNorm(
        num_features=width,
        scale_init=nnx.with_partitioning(
            initializers.ones,
            shd_config.layer_norm if shd_config else (),
        ),
        bias_init=nnx.with_partitioning(
            initializers.zeros,
            shd_config.layer_norm if shd_config else (),
        ),
        rngs=rngs,
    )
    self.mlp = MlpBlock(
        width=width,
        block_id=block_id,
        mlp_dim=mlp_dim,
        dropout=dropout,
        dtype_mm=dtype_mm,
        rngs=rngs,
        shd_config=shd_config,
    )

    self.block_id = block_id
    self.width = width
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads
    self.dropout_rate = dropout
    self.dtype_mm = dtype_mm

  def __call__(
      self, x: jaxtyping.ArrayLike, deterministic: bool = True
  ) -> jaxtyping.ArrayLike:
    """Applies Encoder1DBlock module.

    Args:
      x: Input tensor.
      deterministic: Whether to run in deterministic mode (e.g., disable
        dropout).

    Returns:
      The output tensor.
    """
    y = self.ln1(x)
    y = self.attn(y, deterministic=deterministic)
    y = self.dropout(y, deterministic=deterministic)
    x = x + y

    y = self.ln2(x)
    y = self.mlp(y, deterministic=deterministic)
    y = self.dropout(y, deterministic=deterministic)
    x = x + y
    return x


class Encoder(nnx.Module):
  """Transformer Model Encoder for sequence to sequence translation."""

  def __init__(
      self,
      *,
      width: int,
      depth: int,
      mlp_dim: int | None = None,  # Defaults to 4x input dim
      num_heads: int = 12,
      dropout: float = 0.0,
      rngs: nnx.Rngs,
      dtype_mm: jnp.dtype = jnp.float32,
      shd_config: SigLIPShardingConfig | None = None,
  ):
    self.blocks = compat.ModuleList([
        Encoder1DBlock(
            width=width,
            block_id=i,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dropout=dropout,
            dtype_mm=dtype_mm,
            rngs=rngs,
            shd_config=shd_config,
        )
        for i in range(depth)
    ])
    self.encoder_norm = nnx.LayerNorm(
        num_features=width,
        scale_init=nnx.with_partitioning(
            initializers.ones,
            shd_config.layer_norm if shd_config else (),
        ),
        bias_init=nnx.with_partitioning(
            initializers.zeros,
            shd_config.layer_norm if shd_config else (),
        ),
        rngs=rngs,
    )

    self.width = width
    self.depth = depth
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads
    self.dropout_rate = dropout
    self.dtype_mm = dtype_mm

  def __call__(
      self, x: jaxtyping.ArrayLike, deterministic: bool = True
  ) -> jaxtyping.ArrayLike:
    """Applies Encoder module.

    Args:
      x: Input tensor.
      deterministic: Whether to run in deterministic mode (e.g., disable
        dropout).

    Returns:
      The output tensor.
    """
    for block in self.blocks:
      x = block(x, deterministic=deterministic)
    x = self.encoder_norm(x)
    return x


class ViTModel(nnx.Module):
  """ViT model.

  Attributes:
    patch_size: The size to patchify images.
    width: The model dimension of the vision encoder.
    depth: The number of the layers.
    mlp_dim: The hidden dimension in the ffw layers.
    num_heads: The number of the heads.
    dropout: The dropout rate.
    dtype_mm: The dtype to convert the input to.
  """

  def __init__(
      self,
      *,
      patch_size: tuple[int, int] = (14, 14),
      width: int = 1152,
      depth: int = 27,
      mlp_dim: int | None = 4304,  # Defaults to 4x input dim
      num_heads: int = 16,
      dropout: float = 0.0,
      rngs: nnx.Rngs,
      dtype_mm: jnp.dtype = jnp.float32,
      shd_config: SigLIPShardingConfig | None = None,
  ):
    self.embedding = nnx.Conv(
        in_features=3,
        out_features=width,
        kernel_size=patch_size,
        strides=patch_size,
        padding="VALID",
        param_dtype=dtype_mm,
        kernel_init=nnx.with_partitioning(
            initializers.lecun_normal(),
            shd_config.emb_patch_kernel if shd_config else (),
        ),
        bias_init=nnx.with_partitioning(
            initializers.zeros,
            shd_config.emb_patch_bias if shd_config else (),
        ),
        rngs=rngs,
    )

    # Values to compute shape are based on default image size 896x896
    # and patch size 14x14 -> 16x16=256 patches.
    pos_emb_shape = (896 // patch_size[0]) * (896 // patch_size[1])
    self.pos_embedding = nnx.Param(
        initializers.normal(stddev=1 / np.sqrt(width))(
            rngs.params(), (1, pos_emb_shape, width)
        ),
        sharding=shd_config.emb_pos_kernel if shd_config else (),
    )

    self.dropout = nnx.Dropout(rate=dropout)

    self.transformer = Encoder(
        width=width,
        depth=depth,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        dropout=dropout,
        dtype_mm=dtype_mm,
        rngs=rngs,
        shd_config=shd_config,
    )

    # Passed attributes.
    self.patch_size = patch_size
    self.width = width
    self.depth = depth
    self.mlp_dim = mlp_dim
    self.num_heads = num_heads
    self.dropout_rate = dropout
    self.dtype_mm = dtype_mm
    self.shd_config = shd_config

  def __call__(
      self,
      image: jaxtyping.ArrayLike,  # B H W C
      *,
      train: bool = False,
  ) -> jaxtyping.ArrayLike:
    """Applies ViTModel module.

    Args:
      image: Input image tensor.
      train: Whether to run in training mode (e.g., enable dropout).

    Returns:
      The output tensor.
    """
    image = jnp.asarray(image, self.dtype_mm)

    # Patch extraction
    x = self.embedding(image)
    if self.shd_config:
      x = shard(x, self.shd_config.act_bhwd)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # Add position embeddings.
    x = x + self.pos_embedding.value

    n, l, c = x.shape  # pylint: disable=unused-variable
    x = self.dropout(x, deterministic=not train)

    x = self.transformer(x, deterministic=not train)

    return x


class VisionExit(nnx.Module):
  """The vision exit layer.

  Possibly downsample the soft tokens to a required output length.

  Attributes:
    output_length: The embed will be spatially avg-pooled to this output length.
  """

  def __init__(self, *, output_length: int = 256, rngs: nnx.Rngs):  # pytype: disable=unused-argument
    self.output_length = output_length

  def __call__(
      self, x: jaxtyping.ArrayLike  # B INPUT_LENGTH D
  ) -> jaxtyping.ArrayLike:  # B OUTPUT_LENGTH D
    """Applies VisionExit module.

    Args:
      x: Input tensor.

    Returns:
      The output tensor.
    """
    cur_length = x.shape[1]
    if cur_length == self.output_length:
      return x

    cur_width = int(cur_length**0.5)
    output_width = int(self.output_length**0.5)
    x = einops.rearrange(x, " b (h w) d -> b h w d", h=cur_width, w=cur_width)

    window = cur_width // output_width
    window_shape = (window, window)
    x = nnx.avg_pool(x, window_shape=window_shape, strides=window_shape)
    return einops.rearrange(x, "b h w d -> b (h w) d")


class SigLiPFromPatches(nnx.Module):
  """SigLIP vision encoder forward pass from PatchifiedMedia."""

  def __init__(
      self,
      *,
      num_mm_tokens_per_image_prepool: int = 4096,
      num_mm_tokens_per_image: int = 256,
      image_height: int = 896,
      image_width: int = 896,
      image_channels: int = 3,
      apply_stop_gradient: bool = True,
      rngs: nnx.Rngs | None = None,
      dtype_mm: jnp.dtype = jnp.float32,
      shd_config: SigLIPShardingConfig | None = None,
  ):
    if rngs is None:
      rngs = nnx.Rngs(0)
    self.siglip_encoder = ViTModel(
        rngs=rngs, dtype_mm=dtype_mm, shd_config=shd_config
    )
    self.siglip_exit = VisionExit(
        output_length=num_mm_tokens_per_image, rngs=rngs
    )

    # Passed attributes.
    self.num_mm_tokens_per_image_prepool = num_mm_tokens_per_image_prepool
    self.num_mm_tokens_per_image = num_mm_tokens_per_image
    self.image_height = image_height
    self.image_width = image_width
    self.image_channels = image_channels
    self.apply_stop_gradient = apply_stop_gradient

  def __call__(
      self,
      patches: jaxtyping.ArrayLike,  # B N P D
  ) -> jaxtyping.ArrayLike:  # B N siglip_embed_dim
    """Applies SigLiPFromPatches module.

    Args:
      patches: Input patches tensor.

    Returns:
      The output tensor.
    """
    chex.assert_rank(patches, 4)
    batch_size, num_frames, num_patches, num_channels = patches.shape
    num_patches_one_side = (
        self.image_height // self.siglip_encoder.patch_size[0]
    )
    chex.assert_equal(num_channels, 3 * self.siglip_encoder.patch_size[0] ** 2)
    chex.assert_equal(num_patches, num_patches_one_side**2)
    flattened_images = einops.rearrange(
        patches,
        "b n (h w) c -> (b n) h w c",
        h=num_patches_one_side,
        w=num_patches_one_side,
        c=num_channels,
    )
    flattened_images = einops.rearrange(
        flattened_images,
        "b h w (p q c) -> b (h p) (w q) c",
        h=num_patches_one_side,
        w=num_patches_one_side,
        p=self.siglip_encoder.patch_size[0],
        q=self.siglip_encoder.patch_size[0],
        c=3,
    )

    soft_tokens = self.siglip_encoder(flattened_images)

    if self.num_mm_tokens_per_image_prepool != self.num_mm_tokens_per_image:
      soft_tokens = self.siglip_exit(soft_tokens)
      assert soft_tokens.shape[-2] == self.siglip_exit.output_length

    soft_tokens = einops.rearrange(
        soft_tokens, "(b n) ... -> b n ...", b=batch_size, n=num_frames
    )
    soft_tokens = cast(jax.Array, soft_tokens)

    if self.apply_stop_gradient:
      soft_tokens = jax.lax.stop_gradient(soft_tokens)
    return soft_tokens

  def patchify_images(
      self, images: jaxtyping.ArrayLike  # *B H W C
  ) -> jaxtyping.ArrayLike:  # *B P D
    """Patchify images.

    Args:
      images: The images to patchify.

    Returns:
      The patches of the images of shape (*batch, num_patches, patch_size *
      patch_size * channels)
    """
    *batch_dims, _, _, _ = images.shape
    images = einops.rearrange(images, "... h w c -> (...) h w c")

    patches = _patchify_images(
        images,
        patch_size=self.siglip_encoder.patch_size,
    )
    patches = patches.reshape((*batch_dims,) + patches.shape[1:])
    return patches


def _patchify_images(
    images: jaxtyping.ArrayLike,  # B H W C
    *,
    patch_size: tuple[int, int],
    padding: str = "VALID",
) -> jaxtyping.ArrayLike:  # B P D
  """Extract patches from images.

  Args:
    images: input batch of images of shape [B, H, W, C].
    patch_size: size of extracted patches.
    padding: padding algorithm to use.

  Returns:
    Tensor of shape [batch, num patches, patch_size * patch_size * C]
  """
  channels = images.shape[-1]
  patches = jax.lax.conv_general_dilated_patches(
      lhs=images,
      filter_shape=patch_size,
      window_strides=patch_size,
      padding=padding,
      rhs_dilation=[1, 1],
      dimension_numbers=("NHWC", "OIHW", "NHWC"),
      precision=jax.lax.Precision.HIGH,
  )
  patches = einops.rearrange(
      patches, "b ph pw (c p) -> b (ph pw) (p c)", c=channels
  )
  return patches
