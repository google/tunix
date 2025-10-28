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

"""Gemma3 model."""

import dataclasses
import enum
import itertools
from typing import Tuple

import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping

if hasattr(flax.config, 'flax_always_shard_variable'):
  flax.config.update('flax_always_shard_variable', False)


LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


class RematConfig(enum.Enum):
  NONE = enum.auto()  # No remat, all activations will be stored in HBM.
  BLOCK = enum.auto()  # Remat the entire attn block.


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
    )


class QueryPreAttentionNormalisation(enum.Enum):
  """Initialization strategy."""

  # Whether to scale the query by 1/sqrt(head_dim)
  BY_ONE_OVER_SQRT_HEAD_DIM = enum.auto()

  # Whether to scale the query by `1/sqrt(embed_dim // num_heads)`
  BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS = enum.auto()


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
  sliding_window_size: int | None = None
  local_base_frequency: int = 10_000
  global_base_frequency: int = 10_000
  local_scale_factor: float = 1.0
  global_scale_factor: float = 1.0
  query_pre_attn_norm: QueryPreAttentionNormalisation = (
      QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM
  )
  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE
  param_dtype: jnp.dtype = jnp.bfloat16

  @classmethod
  def gemma3_270m(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    """Gemma3-270M text-only config."""
    return cls(
        num_layers=18,
        num_embed=262144,
        embed_dim=640,
        hidden_dim=2048,
        num_heads=4,
        head_dim=256,
        num_kv_heads=1,
        sliding_window_size=512,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        shd_config=sharding_config,
    )

  @classmethod
  def gemma3_1b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    return cls(
        num_layers=26,
        num_embed=262144,
        embed_dim=1152,
        hidden_dim=6 * 1152,
        num_heads=4,
        head_dim=256,
        num_kv_heads=1,
        sliding_window_size=512,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        shd_config=sharding_config,
    )

  @classmethod
  def gemma3_4b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    """Gemma3-4B text-only config."""
    return cls(
        num_layers=34,
        num_embed=262144,
        embed_dim=2560,
        hidden_dim=2560 * 8 // 2,
        num_heads=8,
        head_dim=256,
        num_kv_heads=4,
        sliding_window_size=1024,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        global_scale_factor=8.0,
        shd_config=sharding_config,
    )

  @classmethod
  def gemma3_12b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    """Gemma3-12B text-only config."""
    return cls(
        num_layers=48,
        num_embed=262144,
        embed_dim=30 * 128,
        hidden_dim=8 * 30 * 128 // 2,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
        sliding_window_size=1024,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        global_scale_factor=8.0,
        shd_config=sharding_config,
    )

  @classmethod
  def gemma3_27b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    """Gemma3-27B text-only config."""
    return cls(
        num_layers=62,
        num_embed=262144,
        embed_dim=5376,
        hidden_dim=5376 * 8 // 2,
        num_heads=32,
        head_dim=128,
        num_kv_heads=16,
        query_pre_attn_norm=QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS,
        sliding_window_size=1024,
        local_base_frequency=10_000,
        global_base_frequency=1_000_000,
        global_scale_factor=8.0,
        shd_config=sharding_config,
    )


def shard(x: jnp.ndarray, s: Tuple[str, ...]):
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return x
  return jax.lax.with_sharding_constraint(
      x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
  )


class Embedder(nnx.Module):
  """Embedder module."""

  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.input_embedding = nnx.Param(
        nnx.initializers.normal(dtype=param_dtype)(
            rngs.params(), (vocab_size, embed_dim)
        ),
        sharding=shd_config.emb_vd,
    )
    self.shd_config = shd_config

  @jax.named_scope('embedder_encode')
  def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = self.input_embedding[(x,)]
    x *= jnp.sqrt(x.shape[-1]).astype(x.dtype)
    x = shard(x, self.shd_config.act_btd)
    return x

  @jax.named_scope('embedder_decode')
  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    return jnp.dot(x, self.input_embedding.value.T)

  @property
  def embed_dim(self):
    return self.input_embedding.value.shape[1]

  @property
  def num_embed(self):
    return self.input_embedding.value.shape[0]


class Einsum(nnx.Module):
  """Einsum is a convenience module for parameterized tensor multiplication."""

  def __init__(
      self,
      einsum_str: str,
      shape: flax.typing.Shape,
      *,
      rngs: nnx.Rngs,
      sharding: Tuple[str | None, ...],
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.einsum_str = einsum_str
    self.shape = shape
    self.w = nnx.Param(
        nnx.initializers.normal(dtype=param_dtype)(rngs.params(), shape),
        sharding=sharding,
    )

  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    return jnp.einsum(self.einsum_str, x, self.w.value)


@jax.named_scope('rope')
def apply_rope(
    inputs: jaxtyping.Array,  # [B, L]
    positions: jaxtyping.Array,  # [B, L]
    *,
    head_dim: int,
    base_frequency: int,
    scale_factor: float = 1.0,
) -> jaxtyping.Array:
  """Applies RoPE."""
  fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
  timescale = base_frequency**fraction

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


K_MASK = -2.3819763e38  # Set to a large negative number.


class AttentionType(enum.Enum):
  GLOBAL = 1
  LOCAL_SLIDING = 2


GEMMA3_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


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


class Attention(nnx.Module):
  """Attention module."""

  def __init__(
      self,
      *,
      num_heads: int,
      num_kv_heads: int,
      features: int,
      head_dim: int,
      attn_type: AttentionType,
      rngs: nnx.Rngs,
      sliding_window_size: int | None,
      rope_base_frequency: int,
      rope_scale_factor: float,
      query_pre_attn_norm: QueryPreAttentionNormalisation,
      shd_config: ShardingConfig,
      remat_config: RematConfig,
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    if attn_type == AttentionType.LOCAL_SLIDING and sliding_window_size is None:
      raise ValueError(
          '`sliding_window_size` must be set if `attn_type` is Local Sliding.'
      )

    self.sliding_window_size = sliding_window_size
    self.attn_type = attn_type
    self.rope_base_frequency = rope_base_frequency
    self.rope_scale_factor = rope_scale_factor
    self.query_pre_attn_norm = query_pre_attn_norm
    self.shd_config = shd_config
    self.remat_config = remat_config
    self.attn_vec_einsum = Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(num_heads, head_dim, features),
        rngs=rngs,
        sharding=shd_config.o_weight_nhd,
        param_dtype=param_dtype,
    )
    if num_heads == num_kv_heads:
      self.qkv_einsum = Einsum(
          einsum_str='BTD,SNDH->SBTNH',
          shape=(3, num_heads, features, head_dim),
          rngs=rngs,
          sharding=shd_config.qkv_weight_cndh,
          param_dtype=param_dtype,
      )
    else:
      self.q_einsum = Einsum(
          einsum_str='BTD,NDH->BTNH',
          shape=(num_heads, features, head_dim),
          rngs=rngs,
          sharding=shd_config.q_weight_ndh,
          param_dtype=param_dtype,
      )
      self.kv_einsum = Einsum(
          einsum_str='BSD,CKDH->CBSKH',
          shape=(2, num_kv_heads, features, head_dim),
          rngs=rngs,
          sharding=(None, None, 'fsdp', None)
          if num_kv_heads == 1
          else shd_config.kv_weight_cndh,
          param_dtype=param_dtype,
      )
    # No sharding on head_dim.
    self._query_norm = RMSNorm(head_dim, rngs=rngs, param_dtype=param_dtype)
    self._key_norm = RMSNorm(head_dim, rngs=rngs, param_dtype=param_dtype)

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    seq_len = x.shape[1]

    if self.use_qkv_einsum:
      query_proj, key_proj, value_proj = self.qkv_einsum(x)
    else:
      query_proj = self.q_einsum(x)
      key_proj, value_proj = self.kv_einsum(x)

    query_proj = shard(query_proj, self.shd_config.act_btnh)
    key_proj = shard(key_proj, self.shd_config.act_btnh)
    value_proj = shard(value_proj, self.shd_config.act_btnh)

    query_proj = self._query_norm(query_proj)
    key_proj = self._key_norm(key_proj)

    query_proj = apply_rope(
        query_proj,
        segment_pos,
        head_dim=self.head_dim,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
    )
    query_scaled = query_proj * self.query_pre_attn_scalar
    key_proj = apply_rope(
        key_proj,
        segment_pos,
        head_dim=self.head_dim,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
    )

    # Cache is left aligned.
    if cache is not None:
      end_index = cache['end_index'][0]
      slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
      value_proj = jax.lax.dynamic_update_slice(
          cache['v'],
          value_proj,
          slice_indices,
      )
      key_proj = jax.lax.dynamic_update_slice(
          cache['k'], key_proj, slice_indices
      )

    if self.use_gqa:
      # Reshape matrices to enable einsums over groups.
      b, t, kg, h = query_scaled.shape
      query_scaled = query_scaled.reshape(
          (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
      )
      logits = jnp.einsum('BTKGH,BSKH->BTKGS', query_scaled, key_proj)
      b, t, k, g, s = logits.shape
      logits = logits.reshape((b, t, k * g, s))
    else:
      # [batch_size, seq_len, num_heads, cache_size]
      # If cache is None, then cache_size = seq_len.
      logits = jnp.einsum('BTNH,BSNH->BTNS', query_scaled, key_proj)

    if self.attn_type == AttentionType.LOCAL_SLIDING:
      if segment_pos.shape[1] == 1:  # for decoding
        sliding_mask = create_sliding_window_mask(
            attn_mask,
            sliding_window_size=self.sliding_window_size,
        )
      else:  # for prefill
        all_ones = jnp.ones_like(attn_mask)
        sliding_mask = jnp.triu(
            all_ones, -1 * self.sliding_window_size + 1
        ) * jnp.tril(all_ones, self.sliding_window_size - 1)
      attn_mask = sliding_mask * attn_mask

    padded_logits = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, K_MASK)

    probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)

    if self.use_gqa:
      # Reshape matrices to enable einsums over groups.
      b, t, kg, h = probs.shape
      probs = probs.reshape(
          (b, t, self.num_kv_heads, int(kg / self.num_kv_heads), h)
      )
      encoded = jnp.einsum('BTKGS,BSKH->BTKGH', probs, value_proj)
      b, t, k, g, h = encoded.shape
      encoded = encoded.reshape((b, t, k * g, h))
    else:
      # [batch_size, seq_len, num_heads, head_dim]
      encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)

    attn_output = self.attn_vec_einsum(encoded)
    attn_output = shard(attn_output, self.shd_config.act_btd)

    if cache is not None:
      new_cache = {
          'v': value_proj,
          'k': key_proj,
          'end_index': cache['end_index'] + seq_len,
      }
    else:
      new_cache = None

    return new_cache, attn_output

  @jax.named_scope('attention')
  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    if self.remat_config == RematConfig.BLOCK:
      # nnx.remat needs to be applied to the unbound function and take self
      # as the first argument.
      return nnx.remat(self.block.__func__)(
          self, x, segment_pos, cache, attn_mask
      )
    else:
      return self.block(x, segment_pos, cache, attn_mask)

  @property
  def head_dim(self):
    return self.attn_vec_einsum.shape[1]

  @property
  def features(self):
    return self.attn_vec_einsum.shape[2]

  @property
  def query_pre_attn_scalar(self):
    match self.query_pre_attn_norm:
      case QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM:
        return self.head_dim**-0.5
      case (
          QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS
      ):
        return (self.features // self.num_heads) ** -0.5

  @property
  def num_heads(self):
    return (
        self.qkv_einsum.shape[1]
        if self.use_qkv_einsum
        else self.q_einsum.shape[0]
    )

  @property
  def num_kv_heads(self):
    return (
        self.qkv_einsum.shape[1]
        if self.use_qkv_einsum
        else self.kv_einsum.shape[1]
    )

  @property
  def use_qkv_einsum(self):
    return hasattr(self, 'qkv_einsum') and self.qkv_einsum is not None

  @property
  def use_gqa(self):
    return self.num_kv_heads != self.num_heads and self.num_kv_heads > 1


class FeedForward(nnx.Module):
  """Feed forward module."""

  def __init__(
      self,
      features: int,
      hidden_dim: int,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.shd_config = shd_config
    kernel_init_fn = nnx.initializers.zeros_init()
    self.gate_proj = nnx.Linear(
        in_features=features,
        out_features=hidden_dim,
        use_bias=False,
        rngs=rngs,
        param_dtype=param_dtype,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, shd_config.ffw_weight_df
        ),
    )
    self.up_proj = nnx.Linear(
        in_features=features,
        out_features=hidden_dim,
        use_bias=False,
        rngs=rngs,
        param_dtype=param_dtype,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, shd_config.ffw_weight_df
        ),
    )
    self.down_proj = nnx.Linear(
        in_features=hidden_dim,
        out_features=features,
        use_bias=False,
        rngs=rngs,
        param_dtype=param_dtype,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, shd_config.ffw_weight_fd
        ),
    )

  @jax.named_scope('feed_forward')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    with jax.named_scope('gate_proj'):
      ff_gate = self.gate_proj(x)
    gate_value = nnx.gelu(ff_gate)

    with jax.named_scope('up_proj'):
      ff1 = self.up_proj(x)
    activations = gate_value * ff1
    activations = shard(activations, self.shd_config.act_btf)

    with jax.named_scope('down_proj'):
      outputs = self.down_proj(activations)
    return outputs


class Block(nnx.Module):
  """Transformer block."""

  def __init__(
      self,
      *,
      num_heads: int,
      num_kv_heads: int,
      embed_dim: int,
      head_dim: int,
      hidden_dim: int,
      attn_type: AttentionType,
      rngs: nnx.Rngs,
      sliding_window_size: int | None,
      rope_base_frequency: int,
      rope_scale_factor: float,
      query_pre_attn_norm: QueryPreAttentionNormalisation,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      remat_config: RematConfig = RematConfig.NONE,
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.pre_attention_norm = RMSNorm(
        embed_dim,
        rngs=rngs,
        sharding=shd_config.rms_norm_weight,
        param_dtype=param_dtype,
    )
    self.attn = Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        features=embed_dim,
        head_dim=head_dim,
        attn_type=attn_type,
        sliding_window_size=sliding_window_size,
        rope_base_frequency=rope_base_frequency,
        rope_scale_factor=rope_scale_factor,
        query_pre_attn_norm=query_pre_attn_norm,
        rngs=rngs,
        shd_config=shd_config,
        remat_config=remat_config,
        param_dtype=param_dtype,
    )
    self.post_attention_norm = RMSNorm(
        embed_dim,
        rngs=rngs,
        sharding=shd_config.rms_norm_weight,
        param_dtype=param_dtype,
    )
    self.pre_ffw_norm = RMSNorm(
        embed_dim,
        rngs=rngs,
        sharding=shd_config.rms_norm_weight,
        param_dtype=param_dtype,
    )
    self.mlp = FeedForward(
        features=embed_dim,
        hidden_dim=hidden_dim,
        rngs=rngs,
        shd_config=shd_config,
        param_dtype=param_dtype,
    )
    self.post_ffw_norm = RMSNorm(
        embed_dim,
        rngs=rngs,
        sharding=shd_config.rms_norm_weight,
        param_dtype=param_dtype,
    )

  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    inputs_normalized = self.pre_attention_norm(x)
    cache, attn_output = self.attn(
        inputs_normalized,
        segment_pos,
        cache,
        attn_mask,
    )
    attn_output = self.post_attention_norm(attn_output)

    attn_output += x

    outputs = self.pre_ffw_norm(attn_output)
    outputs = self.mlp(outputs)
    outputs = self.post_ffw_norm(outputs)

    outputs += attn_output
    return cache, outputs


class RMSNorm(nnx.Module):
  """RMSNorm layer."""

  def __init__(
      self,
      dim: int,
      *,
      rngs: nnx.Rngs,
      sharding: tuple[str, ...] = (),
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.scale = nnx.Param(
        nnx.initializers.zeros_init()(rngs.params(), dim).astype(param_dtype),
        sharding=sharding,
    )

  @jax.named_scope('rms_norm')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_inputs = x * jax.lax.rsqrt(var + 1e-06)

    # normed_inputs is a rank-K tensor, K > 1 (K is typically 2 or 3). scale is
    # a rank-1 tensor. To avoid implicit rank-promotion, reshape scale to
    # a (1, ..., 1, D) tensor, so the rank of scale matches normed_inputs.
    scale = jnp.expand_dims(self.scale.value, axis=range(len(x.shape) - 1))
    normed_inputs = normed_inputs * (1 + scale)
    return normed_inputs


class Gemma3(nnx.Module, pytree=False):
  """Gemma transformer."""

  def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.embedder = Embedder(
        vocab_size=config.num_embed,
        embed_dim=config.embed_dim,
        rngs=rngs,
        shd_config=config.shd_config,
        param_dtype=config.param_dtype,
    )
    self.layers = [
        Block(
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            embed_dim=config.embed_dim,
            head_dim=config.head_dim,
            hidden_dim=config.hidden_dim,
            sliding_window_size=config.sliding_window_size,
            attn_type=attn_type,
            rope_base_frequency=config.local_base_frequency
            if attn_type == AttentionType.LOCAL_SLIDING
            else config.global_base_frequency,
            rope_scale_factor=config.local_scale_factor
            if attn_type == AttentionType.LOCAL_SLIDING
            else config.global_scale_factor,
            query_pre_attn_norm=config.query_pre_attn_norm,
            rngs=rngs,
            shd_config=config.shd_config,
            remat_config=config.remat_config,
            param_dtype=config.param_dtype,
        )
        for _, attn_type in zip(
            range(config.num_layers), itertools.cycle(GEMMA3_ATTENTION_PATTERN)
        )
    ]
    self.final_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        sharding=config.shd_config.rms_norm_weight,
        param_dtype=config.param_dtype,
    )

  def __call__(
      self,
      last_tokens: jaxtyping.Array,  # [B, L]
      positions: jaxtyping.Array,  # [B, L]
      cache: Cache | None,  # (sequence length L')
      attention_mask: jaxtyping.Array,  # [B, L, L']
      output_hidden_states: bool = False,
  ) -> tuple[jaxtyping.Array, Cache | None]:
    """Transformer forward pass.

    You can run this forward pass two ways: with or without an attention kv
    cache.

    Args:
      last_tokens: input sequence of tokens.
      positions: input absolute positions.
      cache: Attention KV cache or None.
      attention_mask: transformer input mask.
      output_hidden_states: whether to output the hidden states.

    Returns:
      predicted_logits, new_cache

      predicted_logits: output logits predicted by the model
      new_cache: updated cache if the input cache is not None, None elsewhere.
    """
    new_cache = None if cache is None else {}
    x = self.embedder.encode(last_tokens)
    for i, layer in enumerate(self.layers):
      layer_name = f'layer_{i}'
      layer_cache = cache[layer_name] if cache else None
      with jax.named_scope(layer_name):
        layer_cache, x = layer(
            x,
            positions,
            layer_cache,
            attention_mask,
        )
      if cache is not None:
        new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch

    x = self.final_norm(x)
    if output_hidden_states:
      self.sow(nnx.Intermediate, 'all_hidden_states', x)
    logits = self.embedder.decode(x)

    return logits, new_cache  # pytype: disable=bad-return-type

  def get_model_input(self):
    """Returns a dummy model input for the transformer.

    This dummy input has a batch size compatible with FSDP sharding on a
    2-device axis.
    """
    dummy_batch_size = 2
    dummy_seq_len = 1
    return {
        'last_tokens': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'positions': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'cache': None,
        'attention_mask': jnp.ones(
            (dummy_batch_size, 1, dummy_seq_len), dtype=jnp.bool
        ),
    }

  @property
  def embed_dim(self) -> int:
    return self.embedder.embed_dim

  @property
  def num_embed(self) -> int:
    return self.embedder.num_embed

  @property
  def num_layers(self) -> int:
    return len(self.layers)
