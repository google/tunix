
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen2 model."""
import dataclasses
import enum
from functools import partial
from typing import Tuple
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
from tunix.generate.mappings import BackendMappingMixin
from tunix.utils import compat
from tunix.utils import env_utils
env_utils.setup_sharding_environment()
K_MASK = -2.3819763e38
LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]
class RematConfig(enum.Enum):
  NONE = enum.auto()  #  No remat, all activations will be stored in HBM.
  BLOCK = enum.auto()  # Remat the entire attn block.
@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  """Sharding configuration for Qwen3 model."""
  emb_vd: Tuple[str | None, ...]
  emb_dv: Tuple[str | None, ...]
  q_weight_dnh: Tuple[str | None, ...]
  kv_weight_dnh: Tuple[str | None, ...]
  o_weight_nhd: Tuple[str | None, ...]
  ffw_weight_df: Tuple[str | None, ...]
  ffw_weight_fd: Tuple[str | None, ...]
  rms_norm_weight: Tuple[str | None, ...]
  act_btd: Tuple[str | None, ...]
  act_btf: Tuple[str | None, ...]
  act_btnh: Tuple[str | None, ...]
  exp_weight_cdf: Tuple[str | None, ...]
  exp_weight_cfd: Tuple[str | None, ...]
  qkv_bias: Tuple[str | None, ...]
  @staticmethod
  def get_default_sharding(is_sampling: bool = False):
    fsdp = 'fsdp' if not is_sampling else None
    return ShardingConfig(
        emb_vd=('tp', fsdp),
        emb_dv=(fsdp, 'tp'),
        q_weight_dnh=(fsdp, 'tp', None),
        kv_weight_dnh=(fsdp, 'tp', None),
        o_weight_nhd=('tp', None, fsdp),
        ffw_weight_df=(fsdp, 'tp'),
        ffw_weight_fd=('tp', fsdp),
        rms_norm_weight=('tp',),
        act_btd=('fsdp', None, None if is_sampling else 'tp'),
        act_btf=('fsdp', None, 'tp'),
        act_btnh=('fsdp', None, 'tp', None),
        exp_weight_cdf=('fsdp', None, 'tp'),
        exp_weight_cfd=('fsdp', 'tp', None),
        qkv_bias=('tp',),
    )
@dataclasses.dataclass(slots=True)
class ModelConfig:
  """Configuration for the Qwen2 model."""
  num_layers: int
  vocab_size: int
  embed_dim: int
  hidden_dim: int
  num_heads: int
  head_dim: int
  num_kv_heads: int
  rope_theta: int
  norm_eps: float
  use_tied_embedding: bool = False
  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE
  dtype: jnp.dtype = jnp.float32
  critical_dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32
  use_flash_attention: bool = False
  # qwen2.5-0.5B and qwen2.5-coder-0.5B share the same config.
  @classmethod
  def qwen2p5_0p5b(cls):  # qwen2.5-0.5B
    return cls(
        num_layers=24,
        vocab_size=151936,
        embed_dim=896,
        hidden_dim=4864,
        num_heads=14,
        head_dim=64,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=1_000_000,
        use_tied_embedding=True,
    )
  # DeepSeek-R1-Distill-Qwen-1.5B
  @classmethod
  def deepseek_r1_distill_qwen_1p5b(cls):
    return cls(
        num_layers=28,
        vocab_size=151936,
        embed_dim=1536,
        hidden_dim=8960,
        num_heads=12,
        head_dim=128,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=10000,
        use_tied_embedding=False,
    )
  @classmethod
  def qwen2p5_1p5b(cls):  # qwen2.5-1.5B
    return cls(
        num_layers=28,
        vocab_size=151936,
        embed_dim=1536,
        hidden_dim=8960,
        num_heads=12,
        head_dim=128,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=1_000_000,
        use_tied_embedding=True,
    )
  @classmethod
  def qwen2p5_math_1p5b(cls):  # qwen2.5-math-1.5B
    return cls(
        num_layers=28,
        vocab_size=151936,
        embed_dim=1536,
        hidden_dim=8960,
        num_heads=12,
        head_dim=128,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=10000,
        use_tied_embedding=True,
    )
  # qwen2.5-coder-3B and qwen2.5-3B share the same config.
  @classmethod
  def qwen2p5_3b(cls):
    return cls(
        num_layers=36,
        vocab_size=151936,
        embed_dim=2048,
        hidden_dim=11008,
        num_heads=16,
        head_dim=128,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=1_000_000,
        use_tied_embedding=True,
    )
  # qwen2.5-7B and qwen2.5-coder-7B share the same config.
  @classmethod
  def qwen2p5_7b(cls):
    return cls(
        num_layers=28,
        vocab_size=152064,
        embed_dim=3584,
        hidden_dim=18944,
        num_heads=28,
        head_dim=128,
        num_kv_heads=4,
        norm_eps=1e-06,
        rope_theta=1_000_000,
        use_tied_embedding=False,
    )
  # TODO(linchai): add other qwen2.5 model configs.
def shard(x: jnp.ndarray, s: Tuple[str, ...]):
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return x
  return jax.lax.with_sharding_constraint(
      x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
  )
class Einsum(nnx.Module):
  """Einsum is a convenience module for parameterized tensor multiplication."""
  def __init__(
      self,
      einsum_str: str,
      shape: flax.typing.Shape,
      *,
      rngs: nnx.Rngs,
      sharding: Tuple[str | None, ...],
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    self.einsum_str = einsum_str
    self.shape = shape
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.w = nnx.Param(
        rngs.params.normal(shape, dtype=param_dtype),
        sharding=sharding,
    )
  @jax.named_scope('einsum')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = jnp.astype(x, self.dtype)
    w = jnp.astype(self.w.value, self.dtype)
    return jnp.einsum(self.einsum_str, x, w)
class Embedder(nnx.Module):
  """Embedder module."""
  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    self.input_embedding = nnx.Param(
        rngs.params.normal((vocab_size, embed_dim), dtype=param_dtype),
        out_sharding=shd_config.emb_vd,
    )
    self.shd_config = shd_config
    self.dtype = dtype
  @jax.named_scope('embedder_encode')
  def encode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = self.input_embedding[(x,)]
    x = jnp.astype(x, self.dtype)
    x = shard(x, self.shd_config.act_btd)
    return x
  @jax.named_scope('embedder_decode')
  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = jnp.astype(x, self.dtype)
    w = jnp.astype(self.input_embedding.value, self.dtype)
    return jnp.dot(x, w.T)
def _generate_pos_embeddings(
    positions: jax.Array,
    features: int,
    rope_theta: int,
) -> tuple[jax.Array, jax.Array]:
  """Generate Sin/Cos for Rotary Embeddings.
  Generates sinusoids at (features//2) different timescales, where the
  timescales form a geometric series from min_timescale to max_timescale
  (max_timescale is not included, but would be the next element in the series).
  Sinusoids are evaluated at integer positions i in [0, length).
  The outputs are computed as:
  sin[b, t, j] = sin(rope_pos[b, t] / timescale[j])
  cos[b, t, j] = cos(rope_pos[b, t] / timescale[j])
  Args:
      postions: [batch, time]
      features: head_dim.
      rope_theta: the rope_theta parameter.
  Returns:
      sin: a float32 array with shape [length, features // 2]
      cos: a float32 array with shape [length, features // 2]
  """
  # Forked from: flaxformer/components/embedding.py;l=592
  fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
  timescale = rope_theta**fraction
  rotational_frequency = 1.0 / timescale
  # Must use high precision einsum here, since rounding off to a bfloat16 is
  # catastrophic. bfloat16 rounds 257 to 256, but sin(257) is very different
  # from sin(256).
  sinusoid_inp = jnp.einsum(
      'BT,k->BTk',
      positions,
      rotational_frequency,
      precision=jax.lax.Precision.HIGHEST,
  )
  return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)
def apply_rotary_embedding(
    x: jax.Array, sin: jax.Array, cos: jax.Array
) -> jax.Array:
  assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  # [B, T, head_dim] -> [B, h, T, head_dim]
  sin, cos = sin[:, :, None, :], cos[:, :, None, :]
  return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
class RMSNorm(nnx.Module):
  """RMSNorm layer."""
  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float = 1e-06,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    self.w = nnx.Param(
        jnp.ones(dim, param_dtype),
        out_sharding=shd_config.rms_norm_weight,
    )
    self.norm_eps = norm_eps
    self.dtype = dtype
  @jax.named_scope('rms_norm')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    x = jnp.astype(x, jnp.float32)
    rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.norm_eps)
    return jnp.astype(
        jnp.astype(self.w.value, jnp.float32) * (x / rms),
        self.dtype,
    )
class Attention(nnx.Module):
  """Attention module."""
  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.shd_config = config.shd_config
    self.q_proj = Einsum(
        einsum_str='BTD,DNH->BTNH',
        shape=(config.embed_dim, config.num_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.q_weight_dnh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.k_proj = Einsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.v_proj = Einsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.o_proj = Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(config.num_heads, config.head_dim, config.embed_dim),
        rngs=rngs,
        sharding=self.shd_config.o_weight_nhd,
        dtype=config.critical_dtype,
        param_dtype=config.param_dtype,
    )
    self.n_rep = config.num_heads // config.num_kv_heads
    self.scale = self.head_dim**-0.5
    self.q_bias = nnx.Param(
        jnp.zeros(config.num_heads * config.head_dim, dtype=config.param_dtype),
        out_sharding=self.shd_config.qkv_bias,
    )
    self.k_bias = nnx.Param(
        jnp.zeros(
            config.num_kv_heads * config.head_dim, dtype=config.param_dtype
        ),
        out_sharding=self.shd_config.qkv_bias,
    )
    self.v_bias = nnx.Param(
        jnp.zeros(
            config.num_kv_heads * config.head_dim, dtype=config.param_dtype
        ),
        out_sharding=self.shd_config.qkv_bias,
    )
  def block(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      sin: jaxtyping.Array,
      cos: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    """Attention block."""
    seq_len = x.shape[1]
    query_proj = self.q_proj(x)
    b, t, n, h = query_proj.shape
    query_proj = jnp.reshape(query_proj, (b, t, n * h)) + self.q_bias.astype(
        self.config.dtype
    )
    query_proj = jnp.reshape(query_proj, (b, t, n, h))
    key_proj = self.k_proj(x)
    _, s, k, h = key_proj.shape
    key_proj = jnp.reshape(key_proj, (b, s, k * h)) + self.k_bias.astype(
        self.config.dtype
    )
    key_proj = jnp.reshape(key_proj, (b, s, k, h))
    value_proj = self.v_proj(x)
    value_proj = jnp.reshape(value_proj, (b, s, k * h)) + self.v_bias.astype(
        self.config.dtype
    )
    value_proj = jnp.reshape(value_proj, (b, s, k, h))
    query_proj = shard(query_proj, self.shd_config.act_btnh)
    key_proj = shard(key_proj, self.shd_config.act_btnh)
    value_proj = shard(value_proj, self.shd_config.act_btnh)
    query_proj = apply_rotary_embedding(
        query_proj,
        sin,
        cos,
    )
    key_proj = apply_rotary_embedding(
        key_proj,
        sin,
        cos,
    )
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
    b, t, qh, d = query_proj.shape
    _, s, kh, _ = key_proj.shape
    # GQA
    query_proj = query_proj.reshape((b, t, kh, qh // kh, d))
    attn = jnp.einsum('BTHGD,BSHD->BHGTS', query_proj, key_proj) * self.scale
    if attn_mask is not None:
      attn = jnp.where(attn_mask[:, None, None, :, :], attn, K_MASK)
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
        key_proj.dtype
    )
    qkv = jnp.einsum('BHGTS,BSHD->BTHGD', attn, value_proj)
    qkv = qkv.reshape((b, t, qh, d))
    outputs = self.o_proj(qkv)
    outputs = shard(outputs, self.shd_config.act_btd)
    if cache is not None:
      new_cache = {
          'v': value_proj,
          'k': key_proj,
          'end_index': cache['end_index'] + seq_len,
      }
    else:
      new_cache = None
    return new_cache, outputs
  def splash_block(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      sin: jaxtyping.Array,
      cos: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    """Attention block."""
    seq_len = x.shape[1]
    query_proj = self.q_proj(x)
    b, t, n, h = query_proj.shape
    query_proj = jnp.reshape(query_proj, (b, t, n * h)) + self.q_bias.astype(
        self.config.dtype
    )
    query_proj = jnp.reshape(query_proj, (b, t, n, h))
    key_proj = self.k_proj(x)
    _, s, k, h = key_proj.shape
    key_proj = jnp.reshape(key_proj, (b, s, k * h)) + self.k_bias.astype(
        self.config.dtype
    )
    key_proj = jnp.reshape(key_proj, (b, s, k, h))
    value_proj = self.v_proj(x)
    value_proj = jnp.reshape(value_proj, (b, s, k * h)) + self.v_bias.astype(
        self.config.dtype
    )
    value_proj = jnp.reshape(value_proj, (b, s, k, h))
    query_proj = shard(query_proj, self.shd_config.act_btnh)
    key_proj = shard(key_proj, self.shd_config.act_btnh)
    value_proj = shard(value_proj, self.shd_config.act_btnh)
    query_proj = apply_rotary_embedding(
        query_proj,
        sin,
        cos,
    )
    key_proj = apply_rotary_embedding(
        key_proj,
        sin,
        cos,
    )
    if cache is not None:
      end_index = cache['end_index'][0]
      slice_indices = (0, end_index % cache['v'].shape[2], 0, 0)
      value_proj = jax.lax.dynamic_update_slice(
          cache['v'],
          value_proj,
          slice_indices,
      )
      key_proj = jax.lax.dynamic_update_slice(
          cache['k'], key_proj, slice_indices
      )
    query_proj = query_proj.transpose(0, 2, 1, 3)
    key_proj = key_proj.transpose(0, 2, 1, 3)
    value_proj = value_proj.transpose(0, 2, 1, 3)
    query_proj = query_proj * self.scale
    b, qh, t, d = query_proj.shape
    _, kh, s, _ = key_proj.shape
    mesh = pxla.thread_resources.env.physical_mesh
    causal_mask = mask_lib.CausalMask((seq_len, seq_len))
    multi_head_mask = mask_lib.MultiHeadMask([causal_mask for _ in range(qh)])
    # TODO: config block size
    block_sizes = splash.BlockSizes(
        block_q=1024,
        block_kv=1024,
        block_q_dkv=1024,
        block_kv_dkv=1024,
        block_kv_dkv_compute=1024,
        block_q_dq=1024,
        block_kv_dq=1024,
    )
    # For sharding by batch, head_shards and q_seq_shards remain 1
    attn_kernel = splash.make_splash_mha(
        multi_head_mask, block_sizes=block_sizes, head_shards=1, q_seq_shards=1
    )
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P('fsdp', None, None, None),
            P('fsdp', None, None, None),
            P('fsdp', None, None, None),
        ),
        out_specs=P('fsdp', None, None, None),
        check_rep=False,
    )
    def sharded_attn(q_block, k_block, v_block):
      # Inside shard_map, we act on a local shard (batch_size // num_devices)
      # The kernel expects (heads, seq, dim), so we use vmap for the local batch
      return jax.vmap(attn_kernel)(q_block, k_block, v_block)
    qkv = sharded_attn(query_proj, key_proj, value_proj)
    qkv = qkv.transpose(0, 2, 1, 3)
    outputs = self.o_proj(qkv)
    outputs = shard(outputs, self.shd_config.act_btd)
    if cache is not None:
      new_cache = {
          'v': value_proj,
          'k': key_proj,
          'end_index': cache['end_index'] + seq_len,
      }
    else:
      new_cache = None
    return new_cache, outputs
  @jax.named_scope('attention')
  def __call__(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      sin: jaxtyping.Array,
      cos: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    block_fn = (
        self.splash_block if self.config.use_flash_attention else self.block
    )
    if self.config.remat_config == RematConfig.BLOCK:
      # nnx.remat needs to be applied to the unbound function and take self
      # as the first argument.
      return nnx.remat(block_fn.__func__)(self, x, cache, attn_mask, sin, cos)
    else:
      return block_fn(x, cache, attn_mask, sin, cos)
  @property
  def head_dim(self):
    return self.o_proj.shape[1]
  @property
  def num_heads(self):
    return self.q_proj.shape[0]
  @property
  def num_kv_heads(self):
    return self.k_proj.shape[1]
class MLP(nnx.Module):
  """MLP module."""
  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.shd_config = config.shd_config
    kernel_init_fn = nnx.initializers.zeros_init()
    self.gate_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        dtype=config.critical_dtype,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_df
        ),
    )
    self.up_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        dtype=config.critical_dtype,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_df
        ),
    )
    self.down_proj = nnx.Linear(
        in_features=config.hidden_dim,
        out_features=config.embed_dim,
        use_bias=False,
        rngs=rngs,
        dtype=config.critical_dtype,
        param_dtype=config.param_dtype,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_fd
        ),
    )
  def block(
      self,
      x: jaxtyping.Array,
  ) -> jaxtyping.Array:
    activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
    activations = shard(activations, self.shd_config.act_btf)
    outputs = self.down_proj(activations)
    return outputs
  @jax.named_scope('feed_forward')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    if self.config.remat_config == RematConfig.BLOCK:
      return nnx.remat(self.block.__func__)(self, x)
    else:
      return self.block(x)
class DecoderLayer(nnx.Module):
  """DecoderLayer."""
  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.input_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        shd_config=config.shd_config,
        dtype=config.critical_dtype,
        param_dtype=config.param_dtype,
    )
    self.attn = Attention(
        config=config,
        rngs=rngs,
    )
    self.post_attention_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        shd_config=config.shd_config,
        dtype=config.critical_dtype,
        param_dtype=config.param_dtype,
    )
    self.mlp = MLP(
        config=config,
        rngs=rngs,
    )
  def __call__(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
      sin,
      cos,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    inputs_normalized = self.input_layernorm(x)
    cache, attn_output = self.attn(
        inputs_normalized,
        cache,
        attn_mask,
        sin,
        cos,
    )
    attn_output += x
    residual = attn_output
    attn_output = self.post_attention_layernorm(attn_output)
    outputs = self.mlp(attn_output)
    outputs = residual + outputs
    return cache, outputs
class Qwen2(BackendMappingMixin, nnx.Module):
  """Qwen2.5 model."""
  BACKEND_PACKAGE_PATH = __name__
  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.config = config
    self.embedder = Embedder(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        rngs=rngs,
        shd_config=shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.layers = compat.ModuleList([
        DecoderLayer(config=config, rngs=rngs) for _ in range(config.num_layers)
    ])
    self.final_norm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        shd_config=shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    if not self.config.use_tied_embedding:
      self.lm_head = Einsum(
          einsum_str='BTD,DV->BTV',
          shape=(config.embed_dim, config.vocab_size),
          rngs=rngs,
          sharding=shd_config.emb_dv,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )
  def __call__(
      self,
      input_tokens: jaxtyping.Array,  # [B, L]
      positions: jaxtyping.Array,  # [B, L]
      cache: Cache | None,  # (sequence length L')
      attention_mask: jaxtyping.Array,  # [B, L, L']
      output_hidden_states: bool = False,
  ) -> tuple[jaxtyping.Array, Cache | None]:
    """Qwen2 model.
    Args:
      input_tokens: input sequence of tokens.
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
    x = self.embedder.encode(input_tokens)
    sin, cos = _generate_pos_embeddings(
        positions, self.config.head_dim, self.config.rope_theta
    )
    sin, cos = sin.astype(x.dtype), cos.astype(x.dtype)
    for i, layer in enumerate(self.layers):
      layer_name = f'layer_{i}'
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = layer(
          x,
          layer_cache,
          attention_mask,
          sin,
          cos,
      )
      if cache is not None:
        new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch
    x = self.final_norm(x)
    if output_hidden_states:
      self.sow(nnx.Intermediate, 'all_hidden_states', x)
    # Qwen2.5 0.5B-3B uses tied embedding, sharing weights for input and output.
    if self.config.use_tied_embedding:
      logits = self.embedder.decode(x)
    else:
      logits = self.lm_head(x)
    return jnp.astype(logits, jnp.float32), new_cache  # pytype: disable=bad-return-type
  def get_model_input(self):
    """Returns a dummy model input for the transformer.
    This dummy input has a batch size compatible with FSDP sharding on a
    2-device axis.
    """
    dummy_batch_size = 2
    dummy_seq_len = 1
    return {
        'input_tokens': jnp.ones(
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
