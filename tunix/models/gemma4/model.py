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
import enum
from functools import partial
import itertools
from typing import Any, Tuple
import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
from jax.experimental.shard_map import shard_map
from jax.interpreters import pxla
import jax.sharding as shd
from jax.sharding import PartitionSpec as P
import jaxtyping
import numpy as np
from tunix.generate.mappings import BackendMappingMixin
from tunix.models.gemma4 import moe
from tunix.models.gemma4 import vision
from tunix.utils import compat
from tunix.utils import env_utils
from tunix.utils.sharding_utils import shard

TOKEN_PLACEHOLDER = -2


@dataclasses.dataclass(frozen=True)
class PreprocessedVisionInput:
  patches: Any
  positions_xy: Any
  soft_token_counts: tuple[int, ...] | tuple[tuple[int, ...], ...]


jax.tree_util.register_dataclass(
    PreprocessedVisionInput,
    data_fields=['patches', 'positions_xy'],
    meta_fields=['soft_token_counts'],
)


env_utils.setup_sharding_environment()


LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


class RematConfig(enum.Enum):
  NONE = enum.auto()
  BLOCK = enum.auto()
  DECODER = enum.auto()


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  """Sharding configuration for gemma transformer."""

  emb_vd: Tuple[str | None, ...]
  q_weight_ndh: Tuple[str | None, ...]
  kv_weight_cndh: Tuple[str | None, ...]
  qkv_weight_cndh: Tuple[str | None, ...]
  o_weight_nhd: Tuple[str | None, ...]
  ffw_weight_df: Tuple[str | None, ...]
  ffw_weight_fd: Tuple[str | None, ...]
  rms_norm_weight: Tuple[str | None, ...]
  act_btd: Tuple[str | None, ...]
  act_btf: Tuple[str | None, ...]
  act_btnh: Tuple[str | None, ...]
  vision_proj: Tuple[str | None, ...]
  vision_soft_emb_norm_weight: Tuple[str | None, ...]
  # MoE sharding
  exp_weight_edf: Tuple[str | None, ...]
  exp_weight_efd: Tuple[str | None, ...]
  # PLE sharding
  per_layer_model_projection: Tuple[str | None, ...]
  per_layer_input_gate: Tuple[str | None, ...]
  per_layer_projection: Tuple[str | None, ...]
  per_layer_input_embedding: Tuple[str | None, ...]
  vision_shd: vision.VisionShardingConfig | None = None

  @staticmethod
  def get_default_sharding(is_sampling: bool = False):
    fsdp = 'fsdp' if not is_sampling else None

    return ShardingConfig(
        emb_vd=('tp', fsdp),
        q_weight_ndh=('tp', fsdp, None),
        kv_weight_cndh=(None, 'tp', fsdp, None),
        qkv_weight_cndh=(None, 'tp', fsdp, None),
        o_weight_nhd=('tp', None, fsdp),
        ffw_weight_df=(fsdp, 'tp'),
        ffw_weight_fd=('tp', fsdp),
        rms_norm_weight=('tp',),
        act_btd=('fsdp', None, None if is_sampling else 'tp'),
        act_btf=('fsdp', None, 'tp'),
        act_btnh=('fsdp', None, 'tp', None),
        vision_proj=(fsdp, 'tp'),
        vision_soft_emb_norm_weight=('tp',),
        exp_weight_edf=(fsdp, None, None, 'tp'),
        exp_weight_efd=(fsdp, 'tp', None),
        per_layer_model_projection=(fsdp, None, 'tp'),
        per_layer_input_gate=(fsdp, 'tp'),
        per_layer_projection=('tp', fsdp),
        per_layer_input_embedding=('tp', None, fsdp),
        vision_shd=vision.VisionShardingConfig.get_default_sharding(is_sampling),
    )


@dataclasses.dataclass(slots=True, kw_only=True)
class ModelConfig:
  """Transformer config."""

  num_layers: int
  num_embed: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  final_logit_softcap: float = 30.0
  sliding_window_size: int | None = None
  per_layer_input_dim: int = 0
  num_global_kv_heads: int | None = None
  global_key_size: int = 512
  attention_pattern: tuple['AttentionType', ...] | None = None
  frac_shared_layers: float = 0.0
  global_rope_proportion: float = 0.25
  local_rope_proportion: float = 1.0
  k_eq_v_global: bool = False
  override_kv_shared_ffw_hidden: int | None = None

  local_base_frequency: int = 10_000
  global_base_frequency: int = 1_000_000
  local_scale_factor: float = 1.0
  global_scale_factor: float = 1.0

  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE
  param_dtype: jnp.dtype = jnp.float32
  dtype: jnp.dtype = jnp.float32
  use_flash_attention: bool = False
  flash_attention_block_size: int = 1024
  use_sliding_window_kv_cache: bool = False

  # MoE config
  enable_moe: bool = False
  num_experts: int | None = None
  num_experts_per_tok: int | None = None
  expert_dim: int | None = None
  moe_dense_hidden_dim: int | None = None

  # Vision config
  vision_encoder: vision.VisionEncoderConfig | None = None
  use_bidirectional_attention: str | None = None

  def __post_init__(self):
    if self.use_sliding_window_kv_cache and self.use_flash_attention:
      raise ValueError(
          'Flash attention and sliding window KV cache are mutually exclusive.'
      )

  @classmethod
  def gemma4_e2b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    return cls(
        num_layers=35,
        num_embed=262144,
        embed_dim=1536,
        hidden_dim=1536 * 4,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        sliding_window_size=512,
        shd_config=sharding_config,
        per_layer_input_dim=256,
        frac_shared_layers=20.0 / 35,
        override_kv_shared_ffw_hidden=int(1536 * 4 * 2),
        attention_pattern=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
        vision_encoder=vision.VisionEncoderConfig(use_clipped_linears=True),
    )

  @classmethod
  def gemma4_e4b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    return cls(
        num_layers=42,
        num_embed=262144,
        embed_dim=2560,
        hidden_dim=2560 * 4,
        num_heads=8,
        head_dim=256,
        num_kv_heads=2,
        sliding_window_size=512,
        shd_config=sharding_config,
        per_layer_input_dim=256,
        frac_shared_layers=18.0 / 42,
        attention_pattern=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
        vision_encoder=vision.VisionEncoderConfig(use_clipped_linears=True),
    )

  @classmethod
  def gemma4_31b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    return cls(
        num_layers=60,
        num_embed=262144,
        embed_dim=5376,
        hidden_dim=5376 * 4,
        num_heads=32,
        head_dim=256,
        num_kv_heads=16,
        num_global_kv_heads=4,
        sliding_window_size=1024,
        shd_config=sharding_config,
        k_eq_v_global=True,
        attention_pattern=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
        vision_encoder=vision.VisionEncoderConfig(
            d_model=1152,
            num_layers=27,
            num_heads=16,
            ffw_hidden=4304,
            use_clipped_linears=False,
            standardize_embeddings=True,
        ),
        use_bidirectional_attention='vision',
    )

  @classmethod
  def gemma4_26b_a4b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    return cls(
        num_layers=30,
        num_embed=262144,
        embed_dim=2816,
        hidden_dim=2112,  # Dense shared MLP branch
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        num_global_kv_heads=2,
        sliding_window_size=1024,
        shd_config=sharding_config,
        enable_moe=True,
        num_experts=128,
        expert_dim=704,
        num_experts_per_tok=8,
        moe_dense_hidden_dim=2112,
        k_eq_v_global=True,
        global_rope_proportion=0.25,
        attention_pattern=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        ),
        vision_encoder=vision.VisionEncoderConfig(
            d_model=1152,
            num_layers=27,
            num_heads=16,
            ffw_hidden=4304,
            output_length=280,
            use_clipped_linears=False,
            standardize_embeddings=True,
        ),
        use_bidirectional_attention='vision',
    )


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
          einsum_str='BTD,DNP->BTNP',
          shape=(self.embed_dim, config.num_layers, config.per_layer_input_dim),
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
              (self.vocab_size, config.num_layers, config.per_layer_input_dim),
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

  def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = self.input_embedding[(x,)]
    x *= jnp.sqrt(x.shape[-1]).astype(x.dtype)
    x = jnp.astype(x, self.config.dtype)
    x = shard(x, self.config.shd_config.act_btd)
    return x

  def encode_vision(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = self.mm_pre_projection_norm(x)
    x = self.mm_input_projection(x)
    return x

  def encode_per_layer_input(
      self, x: jaxtyping.ArrayLike, t: jaxtyping.ArrayLike
  ) -> jaxtyping.Array:
    t = jnp.where(
        jnp.logical_and(t >= 0, t < self.vocab_size), t, jnp.zeros_like(t)
    )
    x = self.per_layer_model_projection(x)
    x = self.per_layer_projection_norm(x)
    y = self.per_layer_input_embedding.value[t]
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
      kv_block_indices = kv_block_indices[..., -attn_kv_len:]

  bidir_cond = (
      (kv_block_indices[:, None, :] == q_block_indices[..., None])
      & (q_block_indices[..., None] > 0)
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


def find_last_one_index(attn_mask: jnp.ndarray) -> jnp.ndarray:
  """Finds the index of the last (rightmost) '1' from attn_mask."""
  cache_len = attn_mask.shape[-1]

  # 1. check if the entire row is all zeros.
  all_zeros_mask = jnp.all(attn_mask == 0, axis=-1)

  # 2. reverse the rows in the attn_mask
  reversed_matrix = attn_mask[:, :, ::-1]

  # 3. find the fist 1 from the right.
  first_one_from_right = jnp.argmax(reversed_matrix, axis=-1)

  # 4. covert back to the original index
  last_one_index_original = cache_len - 1 - first_one_from_right

  # 5. return the final index, 0 for rows are all zeros.
  final_indices = jnp.where(
      all_zeros_mask,
      0,
      last_one_index_original,
  )

  return final_indices.squeeze(axis=-1)


def create_sliding_window_mask(
    attn_mask: jnp.ndarray,  # [B, seq_len, cache_len] seq_len=1 for decoding
    sliding_window_size: int,
) -> jnp.ndarray:
  """Helper function to create sliding window mask for local attention."""
  upper_index = find_last_one_index(attn_mask)

  # 1. compute the window start position
  window_start_pos = upper_index - sliding_window_size + 1

  # 2. create window mask
  abs_pos = jnp.arange(attn_mask.shape[-1])
  window_mask = abs_pos[None, :] >= window_start_pos[:, None]

  # 3. create causal mask
  causal_mask = abs_pos[None, :] <= upper_index[:, None]

  # 4. create final mask
  final_mask = window_mask & causal_mask
  return final_mask[:, None, :]  # [B, 1, cache_len]


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
          nnx.initializers.ones_init()(rngs.params(), dim).astype(param_dtype),
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


K_MASK = -2.3819763e38


class AttentionType(enum.Enum):
  GLOBAL = 1
  LOCAL_SLIDING = 2


GEMMA4_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


def create_kv_cache_sharing_patterns(
    num_layers: int,
    frac_shared_layers: float,
    share_global: bool,
    share_local: bool,
    attention_types: tuple[AttentionType, ...],
) -> list[int]:
  """Creates a list of layer indices for which KV cache is used."""
  kv_cache_sharing_patterns = []
  num_unshared_layers = int(num_layers - frac_shared_layers * num_layers)
  for i in range(num_layers):
    if i < num_unshared_layers:
      kv_cache_sharing_patterns.append(i)
    else:
      if attention_types[i] == AttentionType.GLOBAL and share_global:
        kv_cache_sharing_patterns.append(num_unshared_layers - 1)
      elif attention_types[i] == AttentionType.LOCAL_SLIDING and share_local:
        kv_cache_sharing_patterns.append(num_unshared_layers - 2)
      else:
        kv_cache_sharing_patterns.append(i)
  return kv_cache_sharing_patterns


class Attention(nnx.Module):
  """Attention module."""

  def __init__(
      self,
      config: ModelConfig,
      attn_type: AttentionType,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.rope_proportion = (
        config.global_rope_proportion
        if attn_type == AttentionType.GLOBAL
        else config.local_rope_proportion
    )
    self.attn_type = attn_type
    self.rope_base_frequency = (
        config.local_base_frequency
        if attn_type == AttentionType.LOCAL_SLIDING
        else config.global_base_frequency
    )
    self.rope_scale_factor = (
        config.local_scale_factor
        if attn_type == AttentionType.LOCAL_SLIDING
        else config.global_scale_factor
    )

    self.num_kv_heads = config.num_kv_heads
    self.head_dim = config.head_dim
    if attn_type == AttentionType.GLOBAL:
      if config.num_global_kv_heads is not None:
        self.num_kv_heads = config.num_global_kv_heads
      if config.global_key_size is not None:
        self.head_dim = config.global_key_size

    self.attn_vec_einsum = Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(config.num_heads, self.head_dim, config.embed_dim),
        rngs=rngs,
        sharding=config.shd_config.o_weight_nhd,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.q_einsum = Einsum(
        einsum_str='BTD,NDH->BTNH',
        shape=(config.num_heads, config.embed_dim, self.head_dim),
        rngs=rngs,
        sharding=config.shd_config.q_weight_ndh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

    # Retrieve TP size from the active JAX mesh
    tp_size = 1
    mesh = None
    if hasattr(jax.sharding, 'get_abstract_mesh'):
      try:
        m = jax.sharding.get_abstract_mesh()
        if m is not None and not getattr(m, 'empty', False):
          mesh = m
      except Exception:
        pass
    if mesh is None:
      mesh = pxla.thread_resources.env.physical_mesh
    if mesh is not None and 'tp' in mesh.axis_names:
      tp_size = mesh.shape['tp']

    self.act_btnh_kv = config.shd_config.act_btnh
    if self.num_kv_heads % tp_size != 0:
      import logging as python_logging
      python_logging.info(
          f"num_kv_heads={self.num_kv_heads} is not divisible by TP size {tp_size}, "
          f"sharding k/v projections on head_dim instead of kv-heads."
      )
      fsdp = config.shd_config.act_btnh[0]
      self.act_btnh_kv = (fsdp, None, None, 'tp')

    k_eq_v = (
        config.k_eq_v_global if attn_type == AttentionType.GLOBAL else False
    )
    if k_eq_v:
      k_sharding = config.shd_config.q_weight_ndh
      if self.num_kv_heads % tp_size != 0:
        fsdp = config.shd_config.q_weight_ndh[1]
        k_sharding = (None, fsdp, 'tp')

      self.k_einsum = Einsum(
          einsum_str='BSD,KDH->BSKH',
          shape=(
              self.num_kv_heads,
              config.embed_dim,
              self.head_dim,
          ),
          rngs=rngs,
          sharding=k_sharding,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
    else:
      if self.num_kv_heads == 1:
        kv_sharding = (None, None, 'fsdp', None)
      else:
        kv_sharding = config.shd_config.kv_weight_cndh
<<<<<<< HEAD
=======
        if self.num_kv_heads % tp_size != 0:
          fsdp = config.shd_config.kv_weight_cndh[2]
          kv_sharding = (None, None, fsdp, 'tp')
>>>>>>> e18a1738 (snapshot)

      self.kv_einsum = Einsum(
          einsum_str='BSD,CKDH->CBSKH',
          shape=(
              2,
              self.num_kv_heads,
              config.embed_dim,
              self.head_dim,
          ),
          rngs=rngs,
          sharding=kv_sharding,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
    self._query_norm = RMSNorm(
        self.head_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self._key_norm = RMSNorm(
        self.head_dim,
        rngs=rngs,
        sharding=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
      kv_shared_cache: LayerCache | None = None,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[
      LayerCache | None,
      jaxtyping.Array,
      tuple[jaxtyping.Array, jaxtyping.Array],
  ]:
    x = x.astype(self.config.dtype)
    seq_len = x.shape[1]
    query_proj = self.q_einsum(x)
    query_proj = shard(query_proj, self.config.shd_config.act_btnh)
    query_proj = self._query_norm(query_proj)
    query_proj = apply_rope(
        query_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
        rope_proportion=self.rope_proportion,
    )

    if kv_shared_cache is not None:
      assert cache is None
      key_proj = kv_shared_cache['k']
      value_proj = kv_shared_cache['v']
    else:
      if hasattr(self, 'k_einsum'):  # case where k_eq_v is True
        key_proj = self.k_einsum(x)
        value_proj = key_proj
      else:
        key_proj, value_proj = self.kv_einsum(x)

      key_proj = shard(key_proj, self.config.shd_config.act_btnh)
      value_proj = shard(value_proj, self.config.shd_config.act_btnh)

      # Apply norms to computed KV
      value_var = jnp.mean(jnp.square(value_proj), axis=-1, keepdims=True)
      value_proj = value_proj * jax.lax.rsqrt(value_var + 1e-06)
      key_proj = self._key_norm(key_proj)
      key_proj = apply_rope(
          key_proj,
          segment_pos,
          base_frequency=self.rope_base_frequency,
          scale_factor=self.rope_scale_factor,
          rope_proportion=self.rope_proportion,
      )

    if cache is not None:
      assert kv_shared_cache is None
      # Update cache with new kv projections
      cache_len = cache['v'].shape[1]
      if seq_len > 1:  # prefill
        if self.config.use_sliding_window_kv_cache:
          # Sliding window cache update (prefill).
          # Does not support chunked prefill.
          valid_len = min(seq_len, cache_len)
          latest_indices = jnp.arange(seq_len - valid_len, seq_len) % cache_len
          cache_v = (
              cache['v']
              .at[:, latest_indices, ...]
              .set(value_proj[:, -valid_len:, ...])
          )
          cache_k = (
              cache['k']
              .at[:, latest_indices, ...]
              .set(key_proj[:, -valid_len:, ...])
          )
        else:
          cache_v = cache['v'].at[:, :seq_len, ...].set(value_proj)
          cache_k = cache['k'].at[:, :seq_len, ...].set(key_proj)

        new_cache = {
            'v': cache_v,
            'k': cache_k,
            'end_index': cache['end_index'] + seq_len,
        }
      else:  # decode
        end_index = cache['end_index'][0]
        slice_indices = (0, end_index % cache_len, 0, 0)
        value_proj = jax.lax.dynamic_update_slice(
            cache['v'], value_proj, slice_indices
        )
        key_proj = jax.lax.dynamic_update_slice(
            cache['k'], key_proj, slice_indices
        )
        new_cache = {
            'v': value_proj,
            'k': key_proj,
            'end_index': cache['end_index'] + seq_len,
        }
    else:
      new_cache = {
          'v': value_proj,
          'k': key_proj,
      }

    b, _, qh, _ = query_proj.shape
    _, _, kh, _ = key_proj.shape

    if self.config.use_flash_attention and seq_len > 1:
      query_proj = query_proj.transpose(0, 2, 1, 3)
      key_proj = key_proj.transpose(0, 2, 1, 3)
      value_proj = value_proj.transpose(0, 2, 1, 3)

      mesh = pxla.thread_resources.env.physical_mesh
      if self.attn_type == AttentionType.LOCAL_SLIDING:
        mask = mask_lib.LocalMask(
            (seq_len, seq_len),
            window_size=(self.config.sliding_window_size - 1, 0),
            offset=0,
        )
      else:
        mask = mask_lib.CausalMask((seq_len, seq_len))

      multi_head_mask = mask_lib.MultiHeadMask([mask for _ in range(qh)])

      block_sizes = splash.BlockSizes(
          block_q=self.config.flash_attention_block_size,
          block_kv=self.config.flash_attention_block_size,
          block_q_dkv=self.config.flash_attention_block_size,
          block_kv_dkv=self.config.flash_attention_block_size,
          block_kv_dkv_compute=self.config.flash_attention_block_size,
          block_q_dq=self.config.flash_attention_block_size,
          block_kv_dq=self.config.flash_attention_block_size,
      )

      shd_b, shd_t, shd_n, shd_h = self.config.shd_config.act_btnh
      if (
          mesh is not None
          and shd_b is not None
          and shd_b in mesh.shape
          and b % mesh.shape[shd_b] != 0
      ):
        shd_b = None
      head_shards = (
          mesh.shape[shd_n] if shd_n is not None and shd_n in mesh.shape else 1
      )
      q_seq_shards = (
          mesh.shape[shd_t] if shd_t is not None and shd_t in mesh.shape else 1
      )

      splash_attn_kernel = splash.make_splash_mha(
          multi_head_mask,
          block_sizes=block_sizes,
          head_shards=head_shards,
          q_seq_shards=q_seq_shards,
      )

      shd_spec = P(shd_b, shd_n, shd_t, shd_h)
      shd_n_kv = (
          shd_n
          if mesh is not None
          and shd_n is not None
          and shd_n in mesh.shape
          and kh % mesh.shape[shd_n] == 0
          else None
      )
      unsharded_seq_kv = P(shd_b, shd_n_kv, None, shd_h)
      kernel_spec = splash_attn_kernel.manual_sharding_spec(
          shd.NamedSharding(mesh, P(shd_n, shd_t))
      )

      if segment_ids is not None:
        seg_spec = P(shd_b, shd_t)
        unsharded_seg_spec = P(shd_b, None)

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                kernel_spec,
                shd_spec,
                unsharded_seq_kv,
                unsharded_seq_kv,
                seg_spec,
                unsharded_seg_spec,
            ),
            out_specs=shd_spec,
            check_rep=False,
        )
        def sharded_splash_attn(
            kernel, q_block, k_block, v_block, q_seg_block, kv_seg_block
        ):
          seg_ids = splash.SegmentIds(q=q_seg_block, kv=kv_seg_block)
          return jax.vmap(kernel)(
              q_block, k_block, v_block, segment_ids=seg_ids
          )

        qkv: jaxtyping.Array = sharded_splash_attn(
            splash_attn_kernel,
            query_proj,
            key_proj,
            value_proj,
            segment_ids,
            segment_ids,
        )
      else:

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                kernel_spec,
                shd_spec,
                unsharded_seq_kv,
                unsharded_seq_kv,
            ),
            out_specs=shd_spec,
            check_rep=False,
        )
        def sharded_splash_attn(kernel, q_block, k_block, v_block):
          return jax.vmap(kernel)(q_block, k_block, v_block)

        qkv: jaxtyping.Array = sharded_splash_attn(
            splash_attn_kernel,
            query_proj,
            key_proj,
            value_proj,
        )
      encoded = qkv.transpose(0, 2, 1, 3)
      query_proj = query_proj.transpose(0, 2, 1, 3)
      key_proj = key_proj.transpose(0, 2, 1, 3)
      value_proj = value_proj.transpose(0, 2, 1, 3)

    else:
      if self.use_gqa:
        b, t, kg, h = query_proj.shape
        n_groups = kg // self.num_kv_heads
        query_reshaped = query_proj.reshape(
            (b, t, self.num_kv_heads, n_groups, h)
        )
        logits = jnp.einsum('BTKGH,BSKH->BTKGS', query_reshaped, key_proj)
        b, t, k, g, s = logits.shape
        logits = logits.reshape((b, t, k * g, s))
      else:
        logits = jnp.einsum('BTNH,BSNH->BTNS', query_proj, key_proj)

      if seq_len > 1:
        # Only compute attention scores for the actual sequence length.
        attn_mask = attn_mask[..., :seq_len]

      if self.attn_type == AttentionType.LOCAL_SLIDING:
        if (
            segment_pos.shape[1] == 1
            and self.config.use_sliding_window_kv_cache
        ):
          # for decoding with sliding window cache
          active_cache = cache if cache is not None else kv_shared_cache
          if active_cache is None:
            raise ValueError(
                'Cache or shared cache is required for local sliding attention'
                ' in decoding.'
            )
          cache_len = key_proj.shape[1]
          end_idx = active_cache['end_index']
          if cache is None and kv_shared_cache is not None:
            # In case of shared KV cache, the origin layer already updated the
            # end index. We need to subtract 1 to get the correct end index of
            # the previous token.
            end_idx = end_idx - 1
          end_idx = end_idx[:, None, None]
          p = jnp.arange(cache_len)[None, None, :]

          # map physical index to logical index
          logical_indices = end_idx - ((end_idx - p) % cache_len)

          # identify uninitialized slots (before the cache fills up)
          valid_physical = logical_indices >= 0
          logical_indices = jnp.maximum(0, logical_indices)

          attn_mask = jnp.take_along_axis(attn_mask, logical_indices, axis=-1)
          attn_mask = attn_mask * valid_physical
        elif segment_pos.shape[1] == 1:
          # for decoding without sliding window cache
          sliding_mask = create_sliding_window_mask(
              attn_mask,
              sliding_window_size=self.config.sliding_window_size,
          )
          attn_mask = sliding_mask * attn_mask
        else:  # for prefill
          all_ones = jnp.ones_like(attn_mask)
          sliding_mask = jnp.triu(
              all_ones, -1 * self.config.sliding_window_size + 1
          ) * jnp.tril(all_ones, self.config.sliding_window_size - 1)
          attn_mask = sliding_mask * attn_mask

      attn = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)
      attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
          key_proj.dtype
      )

      if self.use_gqa:
        b, t, kg, s = attn.shape
        n_groups = kg // self.num_kv_heads
        probs_reshaped = attn.reshape((b, t, self.num_kv_heads, n_groups, s))
        encoded = jnp.einsum('BTKGS,BSKH->BTKGH', probs_reshaped, value_proj)
        b, t, k, g, h = encoded.shape
        encoded = encoded.reshape((b, t, k * g, h))
      else:
        encoded = jnp.einsum('BTNS,BSNH->BTNH', attn, value_proj)

    attn_output = self.attn_vec_einsum(encoded)
    attn_output = shard(attn_output, self.config.shd_config.act_btd)
    return new_cache, attn_output, (key_proj, value_proj)

  @property
  def use_gqa(self):
    return self.num_kv_heads != self.config.num_heads and self.num_kv_heads > 1

  def __call__(
      self,
      x,
      segment_pos,
      cache,
      attn_mask,
      kv_shared_cache=None,
      segment_ids=None,
  ):
    remat_config = getattr(self.config, 'remat_config', RematConfig.NONE)
    if (
        remat_config == RematConfig.BLOCK
        or remat_config == RematConfig.BLOCK.value
    ):
      # nnx.remat needs to be applied to the unbound function and take self
      # as the first argument. graph_updates=False prevents TraceContextError
      # when mutating params across jax transformation trace levels.
      return nnx.remat(self.block.__func__)(
          self, x, segment_pos, cache, attn_mask, kv_shared_cache, segment_ids
      )
    else:
      return self.block(
          x,
          segment_pos,
          cache,
          attn_mask,
          kv_shared_cache=kv_shared_cache,
          segment_ids=segment_ids,
      )

  def init_cache(self, batch_size, max_seq_len, dtype):
    cache_len = max_seq_len
    if (
        self.config.use_sliding_window_kv_cache
        and self.attn_type == AttentionType.LOCAL_SLIDING
        and self.config.sliding_window_size is not None
    ):
      cache_len = min(max_seq_len, self.config.sliding_window_size)

    cache_shape = (batch_size, cache_len, self.num_kv_heads, self.head_dim)
    k = shard(
        np.zeros(cache_shape, dtype),
        self.config.shd_config.act_btnh,
        eager=True,
    )
    v = shard(
        np.zeros(cache_shape, dtype),
        self.config.shd_config.act_btnh,
        eager=True,
    )
    end_index = shard(
        np.zeros((batch_size,), np.int32),
        self.config.shd_config.act_btnh[:1],
        eager=True,
    )
    return {'k': k, 'v': v, 'end_index': end_index}


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
      return nnx.remat(self.block.__func__)(self, x)
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
      return nnx.remat(self.block.__func__)(
          self,
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
    if text_only and config.vision_encoder is not None:
      config = dataclasses.replace(config, vision_encoder=None)
    self.config = config
    self.embedder = Embedder(config, rngs=rngs)

    if config.vision_encoder is not None:
      self.vision_encoder = vision.VisionEncoder(
          rngs=rngs,
          config=config.vision_encoder,
          param_dtype=config.param_dtype,
          shd_config=config.shd_config.vision_shd,
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
      mask = tokens == TOKEN_PLACEHOLDER
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
      bidirectional_mask = tokens == TOKEN_PLACEHOLDER
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

    max_n_images = max((len(counts) for counts in soft_token_counts), default=0)
    if max_n_images == 0:
      return jnp.zeros((batch_size, 0, self.config.embed_dim))

    patches = vision_input.patches
    positions_xy = vision_input.positions_xy
    max_patches = patches.shape[1] // max_n_images

    patches = jnp.reshape(
        patches, (batch_size * max_n_images, max_patches, patches.shape[2])
    )
    positions_xy = jnp.reshape(
        positions_xy, (batch_size * max_n_images, max_patches, positions_xy.shape[2])
    )

    encoder_outputs = self.vision_encoder(patches, positions_xy)

    embeddings, mask = encoder_outputs[0]

    batch_tokens = []
    max_tokens_per_batch = 0
    for b in range(batch_size):
      per_image_tokens = []
      counts = soft_token_counts[b] if b < len(soft_token_counts) else ()
      for i in range(len(counts)):
        idx = b * max_n_images + i
        expected_count = counts[i]
        if mask is not None:
          valid_indices = jnp.nonzero(mask[idx], size=expected_count)[0]
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
