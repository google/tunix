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

"""Gemma transformer."""

from collections.abc import Iterable
import dataclasses
import enum
from typing import Any, Callable, Tuple

import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping
from tunix.models.gemma import params as params_lib

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


if hasattr(flax.config, 'flax_always_shard_variable'):
  flax.config.update('flax_always_shard_variable', False)


class AttentionType(enum.Enum):
  GLOBAL = 1
  LOCAL_SLIDING = 2


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
  score_weight_d1: Tuple[str | None, ...]

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
        score_weight_d1=(fsdp, None),
    )


@dataclasses.dataclass(frozen=True)
class ModelConfig:
  """Configuration for the gemma transformer."""

  num_layers: int
  num_embed: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  final_logit_softcap: float | None
  use_post_attn_norm: bool
  use_pre_ffw_norm: bool
  use_post_ffw_norm: bool
  attention_types: Iterable[AttentionType]
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE

  @classmethod
  def gemma_2b(cls):
    num_layers = 18
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=2048,
        hidden_dim=16384,
        num_heads=8,
        head_dim=256,
        num_kv_heads=1,
        final_logit_softcap=None,
        attention_types=(AttentionType.GLOBAL,) * num_layers,
        use_post_attn_norm=True,
        use_pre_ffw_norm=False,
        use_post_ffw_norm=False,
    )

  @classmethod
  def gemma_7b(cls):
    num_layers = 28
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=3072,
        hidden_dim=24576,
        num_heads=16,
        head_dim=256,
        num_kv_heads=16,
        final_logit_softcap=None,
        attention_types=(AttentionType.GLOBAL,) * num_layers,
        use_post_attn_norm=True,
        use_pre_ffw_norm=False,
        use_post_ffw_norm=False,
    )

  @classmethod
  def gemma2_2b(cls):
    num_layers = 26
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=2304,
        hidden_dim=9216,
        num_heads=8,
        head_dim=256,
        num_kv_heads=4,
        final_logit_softcap=30.0,
        attention_types=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        )
        * int(num_layers / 2),
        use_post_attn_norm=True,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
    )

  @classmethod
  def gemma2_9b(cls):
    num_layers = 42
    return cls(
        num_layers=num_layers,
        num_embed=256128,
        embed_dim=3584,
        hidden_dim=28672,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        final_logit_softcap=30.0,
        attention_types=(
            AttentionType.LOCAL_SLIDING,
            AttentionType.GLOBAL,
        )
        * int(num_layers / 2),
        use_post_attn_norm=True,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        attn_logits_soft_cap=50.0,
        sliding_window_size=4096,
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
  ):
    self.input_embedding = nnx.Param(
        nnx.initializers.normal()(rngs.params(), (vocab_size, embed_dim)),
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
  ):
    self.einsum_str = einsum_str
    self.shape = shape
    self.w = nnx.Param(
        nnx.initializers.normal()(rngs.params(), shape), sharding=sharding
    )

  @jax.named_scope('einsum')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    return jnp.einsum(self.einsum_str, x, self.w.value)


@jax.named_scope('rope')
def apply_rope(
    inputs: jaxtyping.Array,  # [B, L]
    positions: jaxtyping.Array,  # [B, L]
    head_dim: int,
    max_wavelength: int = 10_000,
) -> jaxtyping.Array:
  """Applies RoPE."""
  fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
  timescale = max_wavelength**fraction

  sinusoid_inp = (
      positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  sin = jnp.sin(sinusoid_inp)
  cos = jnp.cos(sinusoid_inp)

  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)


K_MASK = -2.3819763e38  # Set to a large negative number.


def _create_sliding_mask(
    segment_pos: jnp.ndarray,  # [B, seq_len]
    cache_len: int,
    sliding_window_size: int,
):
  """Helper function to create sliding mask for local attention.

  It generates the sliding mask based on current segment position, the window
  is [segment_pos - window, segment_pos + window]

  Args:
    segment_pos: Current segment position of shape [B, seq_len].
    cache_len: The length of the cache.
    sliding_window_size: The size of the sliding window.

  Returns:
    The sliding mask of shape [B, seq_len, cache_len].
  """
  cache_positions = jnp.arange(cache_len)

  cache_positions = cache_positions[None, None, :]  # [1, 1, cache_len]
  segment_pos = segment_pos[:, :, None]  # [B, seq_len, 1]

  # abs_pos - window <= key_abs_pos <= abs_pos + window
  sliding_mask = cache_positions >= segment_pos - sliding_window_size + 1
  sliding_mask *= cache_positions <= segment_pos + sliding_window_size - 1
  return sliding_mask  # [B, seq_len, cache_len]


class Attention(nnx.Module):
  """Attention module."""

  def __init__(
      self,
      num_heads: int,
      num_kv_heads: int,
      features: int,
      head_dim: int,
      attn_type: AttentionType,
      *,
      rngs: nnx.Rngs,
      sliding_window_size: int | None = None,
      attn_logits_soft_cap: float | None = None,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      remat_config: RematConfig = RematConfig.BLOCK,
  ):
    if attn_type == AttentionType.LOCAL_SLIDING and sliding_window_size is None:
      raise ValueError(
          '`sliding_window_size` must be set if `attn_type` is Local Sliding.'
      )

    self.sliding_window_size = sliding_window_size
    self.attn_type = attn_type
    self.attn_logits_soft_cap = attn_logits_soft_cap
    self.shd_config = shd_config
    self.remat_config = remat_config
    self.attn_vec_einsum = Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(num_heads, head_dim, features),
        rngs=rngs,
        sharding=shd_config.o_weight_nhd,
    )
    if num_heads == num_kv_heads:
      self.qkv_einsum = Einsum(
          einsum_str='BTD,SNDH->SBTNH',
          shape=(3, num_heads, features, head_dim),
          rngs=rngs,
          sharding=shd_config.qkv_weight_cndh,
      )
    else:
      self.q_einsum = Einsum(
          einsum_str='BTD,NDH->BTNH',
          shape=(num_heads, features, head_dim),
          rngs=rngs,
          sharding=shd_config.q_weight_ndh,
      )
      self.kv_einsum = Einsum(
          einsum_str='BSD,CKDH->CBSKH',
          shape=(2, num_kv_heads, features, head_dim),
          rngs=rngs,
          sharding=(None, None, 'fsdp', None)
          if num_kv_heads == 1
          else shd_config.kv_weight_cndh,
      )

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

    query_proj = apply_rope(
        query_proj,
        segment_pos,
        head_dim=self.head_dim,
    )
    query_scaled = query_proj * self.head_dim**-0.5
    key_proj = apply_rope(
        key_proj,
        segment_pos,
        head_dim=self.head_dim,
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

    if self.attn_logits_soft_cap is not None:
      logits = jnp.tanh(logits / self.attn_logits_soft_cap)
      logits = logits * self.attn_logits_soft_cap

    if self.attn_type == AttentionType.LOCAL_SLIDING:
      sliding_mask = _create_sliding_mask(
          segment_pos,
          cache_len=attn_mask.shape[-1],
          sliding_window_size=self.sliding_window_size,
      )
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
      return nnx.remat(self.block)(x, segment_pos, cache, attn_mask)
    else:
      return self.block(x, segment_pos, cache, attn_mask)

  @property
  def head_dim(self):
    return self.attn_vec_einsum.shape[1]

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
  ):
    self.shd_config = shd_config
    kernel_init_fn = nnx.initializers.zeros_init()
    self.gate_proj = nnx.Linear(
        in_features=features,
        out_features=hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, shd_config.ffw_weight_df
        ),
    )
    self.up_proj = nnx.Linear(
        in_features=features,
        out_features=hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, shd_config.ffw_weight_df
        ),
    )
    self.down_proj = nnx.Linear(
        in_features=hidden_dim,
        out_features=features,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, shd_config.ffw_weight_fd
        ),
    )

  @jax.named_scope('feed_forward')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    ff_gate = self.gate_proj(x)
    gate_value = nnx.gelu(ff_gate)

    ff1 = self.up_proj(x)
    activations = gate_value * ff1
    activations = shard(activations, self.shd_config.act_btf)

    outputs = self.down_proj(activations)
    return outputs


class Block(nnx.Module):
  """Transformer block."""

  def __init__(
      self,
      num_heads: int,
      num_kv_heads: int,
      embed_dim: int,
      head_dim: int,
      hidden_dim: int,
      use_pre_ffw_norm: bool,
      use_post_attn_norm: bool,
      use_post_ffw_norm: bool,
      attn_type: AttentionType,
      *,
      rngs: nnx.Rngs,
      attn_logits_soft_cap: float | None,
      sliding_window_size: int | None = None,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      remat_config: RematConfig = RematConfig.BLOCK,
  ):
    self.pre_attention_norm = RMSNorm(
        embed_dim, rngs=rngs, shd_config=shd_config
    )
    self.attn = Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        features=embed_dim,
        head_dim=head_dim,
        attn_type=attn_type,
        attn_logits_soft_cap=attn_logits_soft_cap,
        sliding_window_size=sliding_window_size,
        rngs=rngs,
        shd_config=shd_config,
        remat_config=remat_config,
    )
    if use_post_attn_norm:
      self.post_attn_norm = RMSNorm(embed_dim, rngs=rngs, shd_config=shd_config)

    if use_pre_ffw_norm:
      self.pre_ffw_norm = RMSNorm(embed_dim, rngs=rngs, shd_config=shd_config)
    self.mlp = FeedForward(
        features=embed_dim,
        hidden_dim=hidden_dim,
        rngs=rngs,
        shd_config=shd_config,
    )
    if use_post_ffw_norm:
      self.post_ffw_norm = RMSNorm(embed_dim, rngs=rngs, shd_config=shd_config)

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

    if self.use_post_attn_norm:
      attn_output = self.post_attn_norm(attn_output)

    attn_output += x

    if self.use_pre_ffw_norm:
      outputs = self.pre_ffw_norm(attn_output)
    else:
      outputs = attn_output
    outputs = self.mlp(outputs)

    if self.use_post_ffw_norm:
      outputs = self.post_ffw_norm(outputs)

    outputs += attn_output
    return cache, outputs

  @property
  def use_post_attn_norm(self):
    return hasattr(self, 'post_attn_norm') and self.post_attn_norm is not None

  @property
  def use_post_ffw_norm(self):
    return hasattr(self, 'post_ffw_norm') and self.post_ffw_norm is not None

  @property
  def use_pre_ffw_norm(self):
    return hasattr(self, 'pre_ffw_norm') and self.pre_ffw_norm is not None

class RMSNorm(nnx.Module):
  """RMSNorm layer."""

  def __init__(
      self,
      dim: int,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.scale = nnx.Param(
        nnx.initializers.zeros_init()(rngs.params(), dim),
        sharding=shd_config.rms_norm_weight,
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


def _flatten_path(path: tuple[str | int, ...]) -> str:
  def f(item) -> str:
    if isinstance(item, str):
      return f'{item}'
    elif isinstance(item, int):
      return f'[{item}]'
    else:
      raise ValueError(f'Unexpected type {type(item)}')

  return '.'.join([f(item) for item in path]).replace('.[', '[')


def module_from_linen_variables(
    module_factory: Callable[[], nnx.Module],
    variables: flax.typing.VariableDict,
    map_key_fn: None | (
        Callable[[tuple[str, ...]], tuple[str | int, ...]]
    ) = None,
    assign_val_fn: None | (
        Callable[
            [
                dict[tuple[str, ...], Any],
                tuple[str | int, ...],
                flax.typing.VariableDict,
            ],
            dict[tuple[str, ...], Any],
        ]
    ) = None,
) -> nnx.Module:
  """Returns an `nnx.Module` initialized with the `variables` of a linen module.

  Args:
    module_factory: A no-args callable that returns an `nnx.Module`.
    variables: A dictionary of variables.
    map_key_fn: An optional function for mapping keys in the `variables`
      dictionary to keys in the `nnx.Module`'s state. If not provided it is
      assumed that after removing the collection name the keys in the
      `variables` dictionary are the same as the keys in the `nnx.Module`'s
      state.
    assign_val_fn: An optional function for assigning values in the `variables`
      dictionary to keys in the `nnx.Module`'s state. If not provided it is
      assumed that the values in the `variables` dictionary are the same as the
      values in the `nnx.Module`'s state.

  Returns:
    An `nnx.Module` initialized with the `variables` of a linen module.
  """
  if map_key_fn is None:

    def map_key_fn(path: tuple[str, ...]) -> tuple[str | int, ...]:
      return path[1:] if 'params' in variables else path

  if assign_val_fn is None:

    def assign_val_fn(
        state: dict[tuple[str, ...], Any],
        mapped_path: tuple[str | int, ...],
        val: Any,
    ) -> dict[tuple[str, ...], Any]:
      state[mapped_path].value = val
      return state

  mdl: nnx.Module = nnx.eval_shape(module_factory)
  graph_def, state = nnx.split(mdl)
  state = dict(state.flat_state())
  for path, val in flax.traverse_util.flatten_dict(variables).items():
    mapped_path = map_key_fn(path)
    if mapped_path not in state:
      raise ValueError(
          f"'{mdl.__class__.__name__}.{_flatten_path(mapped_path)}' doesn't "
          f' exist (original path={path}).'
      )
    state = assign_val_fn(state, mapped_path, val)
  state = nnx.State.from_flat_path(state)

  return nnx.merge(graph_def, state)


def _map_linen_var_names(key: tuple[str, ...]) -> tuple[str | int, ...]:
  """Maps linen variable names to nnx variable names."""
  new_key = []
  for k in key:
    if k.startswith('layer_'):
      prefix, suffix = k.split('layer_')
      if prefix:
        raise ValueError(
            'Invalid parameter key format. Expected key to start directly '
            f"with 'layer_', but found a prefix: '{prefix}' in key '{k}'."
        )
      new_key.append('layers')
      new_key.append(int(suffix))
    elif k == 'gating_einsum':
      new_key.append('gate_proj')
      new_key.append('kernel')
    elif k == 'linear':
      new_key.append('down_proj')
      new_key.append('kernel')
    elif k == 'post_attention_norm':
      new_key.append('post_attn_norm')
    else:
      new_key.append(k)

  return tuple(new_key)


def _assign_linen_params_to_nnx_state(
    state: dict[tuple[str, ...], Any],
    mapped_path: tuple[str | int, ...],
    val: Any,
) -> dict[tuple[str, ...], Any]:
  if 'gate_proj' in mapped_path:
    state[mapped_path].value = val[0]
    state[mapped_path[:-2] + ('up_proj', 'kernel')].value = val[1]
  else:
    state[mapped_path].value = val
  return state


class Transformer(nnx.Module, pytree=False):
  """Gemma transformer."""

  @classmethod
  def from_params(
      cls, params: params_lib.Params, version: str
  ) -> 'Transformer':

    if version in ['2b', '2b-it', '1.1-2b-it']:
      config = ModelConfig.gemma_2b()
    elif version in ['7b', '7b-it', '1.1-7b-it']:
      config = ModelConfig.gemma_7b()
    elif version in ['2-2b', '2-2b-it']:
      config = ModelConfig.gemma2_2b()
    elif version in ['2-9b', '2-9b-it']:
      config = ModelConfig.gemma2_9b()
    else:
      raise ValueError(f'Unsupported version: {version}')

    return module_from_linen_variables(
        module_factory=lambda: cls(config, rngs=nnx.Rngs(params=0)),
        variables=params['transformer'],
        map_key_fn=_map_linen_var_names,
        assign_val_fn=_assign_linen_params_to_nnx_state,
    )

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.embedder = Embedder(
        vocab_size=config.num_embed,
        embed_dim=config.embed_dim,
        rngs=rngs,
        shd_config=config.shd_config,
    )
    self.layers = [
        Block(
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            embed_dim=config.embed_dim,
            head_dim=config.head_dim,
            hidden_dim=config.hidden_dim,
            sliding_window_size=config.sliding_window_size,
            use_post_attn_norm=config.use_post_attn_norm,
            use_pre_ffw_norm=config.use_pre_ffw_norm,
            use_post_ffw_norm=config.use_post_ffw_norm,
            attn_logits_soft_cap=config.attn_logits_soft_cap,
            attn_type=attn_type,
            rngs=rngs,
            shd_config=config.shd_config,
            remat_config=config.remat_config,
        )
        for _, attn_type in zip(
            range(config.num_layers), config.attention_types
        )
    ]
    self.final_norm = RMSNorm(
        config.embed_dim, rngs=rngs, shd_config=config.shd_config
    )
    self.final_logits_softcap = config.final_logit_softcap

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

    if self.final_logits_softcap is not None:
      logits /= self.final_logits_softcap
      logits = jnp.tanh(logits) * self.final_logits_softcap
    return logits, new_cache  # pytype: disable=bad-return-type

  @property
  def embed_dim(self) -> int:
    return self.embedder.embed_dim

  @property
  def num_embed(self) -> int:
    return self.embedder.num_embed

  @property
  def num_layers(self) -> int:
    return len(self.layers)

  def init_cache(
      self, batch_size: int, cache_size: int, dtype: jnp.dtype
  ) -> Cache:
    """Initializes the cache for the model."""
    config = self.config
    shape = (batch_size, cache_size, config.num_kv_heads, config.head_dim)
    k = jnp.zeros(shape, dtype=dtype)
    v = jnp.zeros(shape, dtype=dtype)
    end_index = jnp.zeros((batch_size,), dtype=jnp.int32)
    # Jax array is immutable, so updates to each layer creates new arrays.
    return {
        f'layer_{i}': {'k': k, 'v': v, 'end_index': end_index}
        for i in range(config.num_layers)
    }

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


class TransformerWithScoreHead(nnx.Module):
  """Gemma transformer with a score head."""

  def __init__(self, transformer: Transformer, rngs: nnx.Rngs):
    """Initializes the transformer with a score head.

    Args:
      transformer: The transformer backbone.
      rngs: The random number generator.
    """

    self.transformer = transformer
    self.score = nnx.Linear(
        in_features=transformer.embed_dim,
        out_features=1,
        use_bias=False,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.normal(),
            transformer.config.shd_config.score_weight_d1,
        ),
        rngs=rngs,
    )

  def __call__(self, *args, **kwargs):
    self.transformer(*args, **kwargs, output_hidden_states=True)
    hidden_states = nnx.pop(self.transformer, nnx.Intermediate)[
        'all_hidden_states'
    ].value[-1]
    score = self.score(hidden_states)
    return score
