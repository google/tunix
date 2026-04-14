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

"""Qwen3 model."""

import dataclasses
import enum
from functools import partial
from typing import Tuple

import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu import megablox
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


def round_up_to_base(x: int, base: int, threshold: int | None = None) -> int:
  if threshold is not None and x < threshold:
    return threshold
  return ((x + base - 1) // base) * base


def get_global_input_output_offsets(global_send_sizes, num_ep):
  """Calculates explicit buffer offsets for ragged_all_to_all."""
  global_input_offsets = jnp.concatenate(
      [jnp.zeros((num_ep, 1), dtype=jnp.int32), global_send_sizes[:, :-1]],
      axis=1,
  )
  global_input_offsets = jnp.cumsum(global_input_offsets, axis=1)

  global_output_offsets = jnp.concatenate(
      [jnp.zeros((1, num_ep), dtype=jnp.int32), global_send_sizes[:-1]], axis=0
  )
  global_output_offsets = jnp.cumsum(global_output_offsets, axis=0)
  return global_input_offsets, global_output_offsets


@jax.custom_vjp
def _custom_permute(x, permute_indices):
  return x[permute_indices]


def _custom_permute_fwd(x, permute_indices):
  return _custom_permute(x, permute_indices), permute_indices


def _custom_permute_bwd(res, g):
  permute_indices = res
  unpermute_indices = jnp.argsort(permute_indices)
  return g[unpermute_indices], None


_custom_permute.defvjp(_custom_permute_fwd, _custom_permute_bwd)


class RematConfig(enum.Enum):
  NONE = enum.auto()  # No remat, all activations will be stored in HBM.
  BLOCK = enum.auto()  # Remat the entire attn block.
  DECODER = enum.auto()  # Remat the entire decoder layer (attn + ffw).


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
  exp_weight_edf: Tuple[str | None, ...]
  exp_weight_efd: Tuple[str | None, ...]

  @staticmethod
  def get_default_sharding(is_sampling: bool = False, enable_sp: bool = False):
    fsdp = 'fsdp' if not is_sampling else None
    sp = 'sp' if (not is_sampling and enable_sp) else None
    fsdp = (fsdp, sp) if fsdp and sp else fsdp

    return ShardingConfig(
        emb_vd=('tp', fsdp),
        emb_dv=(fsdp, 'tp'),
        q_weight_dnh=(fsdp, 'tp', None),
        kv_weight_dnh=(fsdp, 'tp', None),
        o_weight_nhd=('tp', None, fsdp),
        ffw_weight_df=(fsdp, 'tp'),
        ffw_weight_fd=('tp', fsdp),
        rms_norm_weight=('tp',),
        act_btd=('fsdp', sp, None if is_sampling else 'tp'),
        act_btf=('fsdp', sp, 'tp'),
        act_btnh=('fsdp', sp, 'tp', None),
        exp_weight_edf=('fsdp', None, 'tp'),
        exp_weight_efd=('fsdp', 'tp', None),
    )


@dataclasses.dataclass(slots=True)
class ModelConfig:
  """Configuration for the Qwen3 model."""

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
  num_experts: int | None = None
  num_experts_per_tok: int | None = None
  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE
  dtype: jnp.dtype = jnp.float32
  param_dtype: jnp.dtype = jnp.float32
  use_flash_attention: bool = False
  flash_attention_block_size: int = 1024

  @classmethod
  def qwen3_0p6b(cls):  # qwen3-0.6B
    return cls(
        num_layers=28,
        vocab_size=151936,
        embed_dim=1024,
        hidden_dim=3072,
        num_heads=16,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-06,
        rope_theta=1_000_000,
        use_tied_embedding=True,
    )

  qwen3_0p6b_base = qwen3_0p6b  # qwen3-0.6B-base

  @classmethod
  def qwen3_1p7b(cls):  # qwen3-1.7B
    return cls(
        num_layers=28,
        vocab_size=151936,
        embed_dim=2048,
        hidden_dim=6144,
        num_heads=16,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-06,
        rope_theta=1_000_000,
        use_tied_embedding=True,
    )

  qwen3_1p7b_base = qwen3_1p7b  # qwen3-1.7B-base

  @classmethod
  def qwen3_4b(cls):  # qwen3-4B
    return cls(
        num_layers=36,
        vocab_size=151936,
        embed_dim=2560,
        hidden_dim=9728,
        num_heads=32,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-06,
        rope_theta=1_000_000,
        use_tied_embedding=True,
    )

  qwen3_4b_base = qwen3_4b  # qwen3-4B-base

  @classmethod
  def _qwen3_4b_2507(cls):  # Qwen3-4B-Instruct-2507 and Qwen3-4B-Thinking-2507
    return cls(
        num_layers=36,
        vocab_size=151936,
        embed_dim=2560,
        hidden_dim=9728,
        num_heads=32,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-06,
        rope_theta=5_000_000,
        use_tied_embedding=True,
    )

  @classmethod
  def qwen3_4b_instruct_2507(cls):  # Qwen3-4B-Instruct-2507
    return cls._qwen3_4b_2507()

  @classmethod
  def qwen3_4b_thinking_2507(cls):  # Qwen3-4B-Thinking-2507
    return cls._qwen3_4b_2507()

  @classmethod
  def qwen3_8b(cls):  # qwen3-8B
    return cls(
        num_layers=36,
        vocab_size=151936,
        embed_dim=4096,
        hidden_dim=12288,
        num_heads=32,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-06,
        rope_theta=1_000_000,
    )

  qwen3_8b_base = qwen3_8b  # qwen3-8B-base

  @classmethod
  def qwen3_14b(cls):  # qwen3-14B
    return cls(
        num_layers=40,
        vocab_size=151936,
        embed_dim=5120,
        hidden_dim=17408,
        num_heads=40,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-06,
        rope_theta=1_000_000,
    )

  qwen3_14b_base = qwen3_14b  # qwen3-14B-base

  @classmethod
  def qwen3_30b_a3b(cls):  # qwen3-30B-a3b
    return cls(
        num_layers=48,
        vocab_size=151936,
        embed_dim=2048,
        hidden_dim=768,
        num_heads=32,
        head_dim=128,
        num_kv_heads=4,
        norm_eps=1e-06,
        rope_theta=1_000_000,
        num_experts=128,
        num_experts_per_tok=8,
    )

  @classmethod
  def qwen3_32b(cls):  # qwen3-32B
    return cls(
        num_layers=64,
        vocab_size=151936,
        embed_dim=5120,
        hidden_dim=25600,
        num_heads=64,
        head_dim=128,
        num_kv_heads=8,
        norm_eps=1e-06,
        rope_theta=1_000_000,
    )


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
    self.w = nnx.Param(
        nnx.initializers.glorot_uniform()(
            rngs.params(), shape, dtype=param_dtype
        ),
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
        nnx.initializers.normal(dtype=param_dtype)(
            rngs.params(), (vocab_size, embed_dim)
        ),
        sharding=shd_config.emb_vd,
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


def apply_rope(
    inputs: jaxtyping.Array,  # [B, L]
    positions: jaxtyping.Array,  # [B, L]
    head_dim: int,
    rope_theta: int = 1_000_000,
) -> jaxtyping.Array:
  """Applies RoPE."""
  fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
  timescale = rope_theta**fraction

  sinusoid_inp = (
      positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  sin = jnp.sin(sinusoid_inp).astype(inputs.dtype)
  cos = jnp.cos(sinusoid_inp).astype(inputs.dtype)

  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)


class RMSNorm(nnx.Module):
  """RMSNorm layer."""

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float = 1e-06,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      dtype: jnp.dtype,
      param_dtype: jnp.dtype,
  ):
    self.w = nnx.Param(
        nnx.initializers.ones_init()(rngs.params(), dim, param_dtype),
        sharding=shd_config.rms_norm_weight,
    )
    self.norm_eps = norm_eps
    self.dtype = dtype

  @jax.named_scope('rms_norm')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    x = jnp.astype(x, jnp.float32)
    rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.norm_eps)
    return jnp.astype(
        jnp.astype(self.w.value, jnp.float32) * (x / rms), self.dtype
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
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.q_norm = RMSNorm(
        config.head_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=self.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.k_norm = RMSNorm(
        config.head_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=self.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.n_rep = config.num_heads // config.num_kv_heads
    self.scale = self.head_dim**-0.5

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    seq_len = x.shape[1]

    query_proj = self.q_norm(self.q_proj(x))
    key_proj = self.k_norm(self.k_proj(x))
    value_proj = self.v_proj(x)

    query_proj = shard(query_proj, self.shd_config.act_btnh)
    key_proj = shard(key_proj, self.shd_config.act_btnh)
    value_proj = shard(value_proj, self.shd_config.act_btnh)

    query_proj = apply_rope(
        query_proj,
        segment_pos,
        head_dim=self.head_dim,
    )
    key_proj = apply_rope(
        key_proj,
        segment_pos,
        head_dim=self.head_dim,
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
    _, _, kh, _ = key_proj.shape

    # NB: flash attention doesn't work for decoding step
    if self.config.use_flash_attention and seq_len > 1:
      query_proj = query_proj.transpose(0, 2, 1, 3)
      key_proj = key_proj.transpose(0, 2, 1, 3)
      value_proj = value_proj.transpose(0, 2, 1, 3)

      query_proj = query_proj * self.scale

      mesh = pxla.thread_resources.env.physical_mesh
      causal_mask = mask_lib.CausalMask((seq_len, seq_len))
      multi_head_mask = mask_lib.MultiHeadMask([causal_mask for _ in range(qh)])

      block_sizes = splash.BlockSizes(
          block_q=self.config.flash_attention_block_size,
          block_kv=self.config.flash_attention_block_size,
          block_q_dkv=self.config.flash_attention_block_size,
          block_kv_dkv=self.config.flash_attention_block_size,
          block_kv_dkv_compute=self.config.flash_attention_block_size,
          block_q_dq=self.config.flash_attention_block_size,
          block_kv_dq=self.config.flash_attention_block_size,
      )

      shd_b, shd_t, shd_n, shd_h = self.shd_config.act_btnh
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
      unsharded_seq = P(shd_b, shd_n, None, shd_h)
      kernel_spec = splash_attn_kernel.manual_sharding_spec(
          shd.NamedSharding(mesh, P(shd_n, shd_t))
      )

      @partial(
          shard_map,
          mesh=mesh,
          in_specs=(kernel_spec, shd_spec, unsharded_seq, unsharded_seq),
          out_specs=shd_spec,
          check_rep=False,
      )
      def sharded_splash_attn(kernel, q_block, k_block, v_block):
        return jax.vmap(kernel)(q_block, k_block, v_block)

      qkv = sharded_splash_attn(
          splash_attn_kernel, query_proj, key_proj, value_proj
      )
      qkv = qkv.transpose(0, 2, 1, 3)
    else:
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

  @jax.named_scope('attention')
  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    if (
        self.config.remat_config == RematConfig.BLOCK
        or self.config.remat_config == RematConfig.BLOCK.value
    ):
      # nnx.remat needs to be applied to the unbound function and take self
      # as the first argument.
      return nnx.remat(self.block.__func__)(
          self, x, segment_pos, cache, attn_mask
      )
    else:
      return self.block(x, segment_pos, cache, attn_mask)

  @property
  def head_dim(self):
    return self.o_proj.shape[1]

  @property
  def num_heads(self):
    return self.q_proj.shape[0]

  @property
  def num_kv_heads(self):
    return self.k_proj.shape[1]


class MoELayer(nnx.Module):
  """Sharded Megablox MoE layer."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.shd_config = config.shd_config
    self.experts_per_tok = config.num_experts_per_tok
    self.num_experts = config.num_experts
    self.router = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.num_experts,
        use_bias=False,
        rngs=rngs,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.gate_proj = nnx.Param(
        nnx.initializers.normal(dtype=config.param_dtype)(
            rngs.params(),
            (config.num_experts, config.embed_dim, config.hidden_dim),
        ),
        sharding=self.shd_config.exp_weight_edf,
    )
    self.up_proj = nnx.Param(
        nnx.initializers.normal(dtype=config.param_dtype)(
            rngs.params(),
            (config.num_experts, config.embed_dim, config.hidden_dim),
        ),
        sharding=self.shd_config.exp_weight_edf,
    )
    self.down_proj = nnx.Param(
        nnx.initializers.normal(dtype=config.param_dtype)(
            rngs.params(),
            (config.num_experts, config.hidden_dim, config.embed_dim),
        ),
        sharding=self.shd_config.exp_weight_efd,
    )
    self.dtype = config.dtype

  def __call__(self, x, use_megablox=True):
    scores = self.router(x).astype(jnp.float32)  # [B,T,E]
    routing_weights, routing_idx = jax.lax.top_k(
        jax.nn.softmax(scores, axis=-1), self.experts_per_tok
    )
    routing_weights = (
        routing_weights / jnp.sum(routing_weights, axis=-1, keepdims=True)
    ).astype(self.dtype)

    # TODO(tsbao): Incorporate input mask to filter out tokens that shouldn't be dispatched

    mesh = pxla.thread_resources.env.physical_mesh

    # -------------------------------------------------------------
    # Fallback to Vanilla dense routing if CPU or un-meshed environment
    # -------------------------------------------------------------
    if not use_megablox or (mesh.empty or jax.devices()[0].platform == 'cpu'):
      dispatch_mask = jax.nn.one_hot(
          routing_idx, num_classes=self.num_experts, dtype=self.dtype
      )  # [B, T, K, E]
      dispatch_mask = jnp.swapaxes(dispatch_mask, -1, -2)  # [B, T, E, K]
      dispatched_input = jnp.einsum(
          'BTID,BTEK->BTED', x[:, :, None, :], dispatch_mask
      ).astype(self.dtype)

      expert_outputs = []
      for i in range(self.num_experts):
        expert_input = dispatched_input[:, :, i, :]
        gate_proj = jnp.astype(self.gate_proj.value[i], self.dtype)
        up_proj = jnp.astype(self.up_proj.value[i], self.dtype)
        activations = nnx.silu(
            jnp.einsum('BTD,DF->BTF', expert_input, gate_proj)
        ) * jnp.einsum('BTD,DF->BTF', expert_input, up_proj)
        down_proj = jnp.astype(self.down_proj.value[i], self.dtype)
        expert_output = jnp.einsum('BTF,FD->BTD', activations, down_proj)
        expert_outputs.append(expert_output)

      stacked_outputs = jnp.stack(expert_outputs, axis=2)  # [B, T, E, D]
      routing_weights = jnp.tile(
          routing_weights[:, :, None, :], (1, 1, self.num_experts, 1)
      )
      routing_weights = dispatch_mask * routing_weights
      return jnp.einsum('BTED,BTEK->BTD', stacked_outputs, routing_weights)

    # -------------------------------------------------------------
    # Distributed Megablox Execution Path
    # -------------------------------------------------------------
    ep_axis = self.shd_config.exp_weight_edf[0]  # Typically 'fsdp' or 'ep'

    # Map global partition specs to local shard_map specs
    shd_act = P(*self.shd_config.act_btd)
    # The routing arrays (weights & indices) have shape [B, T, K], lacking the D dim
    shd_routing = P(*self.shd_config.act_btd[:-1], None)
    shd_exp_edf = P(*self.shd_config.exp_weight_edf)
    shd_exp_efd = P(*self.shd_config.exp_weight_efd)

    global_gate_w = self.gate_proj.value.astype(self.dtype)
    global_up_w = self.up_proj.value.astype(self.dtype)
    global_down_w = self.down_proj.value.astype(self.dtype)

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            shd_act,
            shd_routing,
            shd_routing,
            shd_exp_edf,
            shd_exp_edf,
            shd_exp_efd,
        ),
        out_specs=shd_act,
        check_rep=False,
    )
    def sharded_megablox_moe(inputs, weights, indices, gate_w, up_w, down_w):
      tp_axis = self.shd_config.act_btd[-1]

      if tp_axis is not None and tp_axis in mesh.axis_names:
        inputs = jax.lax.all_gather(
            inputs, axis_name=tp_axis, tiled=True, axis=2
        )

      B, T, D_global = inputs.shape

      if ep_axis is not None and ep_axis in mesh.axis_names:
        num_ep = jax.lax.psum(1, axis_name=ep_axis)
        ep_shard_idx = jax.lax.axis_index(ep_axis)
      else:
        num_ep = 1
        ep_shard_idx = 0

      num_local_experts = self.num_experts // num_ep

      flat_repeated_inputs = jnp.repeat(
          inputs.reshape(B * T, D_global), self.experts_per_tok, axis=0
      )
      flat_selected_indices = indices.reshape(-1)

      sort_indices = jnp.argsort(flat_selected_indices)
      unsort_indices = jnp.argsort(sort_indices)

      sorted_inputs = _custom_permute(flat_repeated_inputs, sort_indices)
      sorted_expert_indices = flat_selected_indices[sort_indices]

      group_sizes = jnp.bincount(sorted_expert_indices, length=self.num_experts)

      if num_ep > 1:
        global_group_sizes = jax.lax.all_gather(
            group_sizes, axis_name=ep_axis, tiled=False, axis=0
        )
        global_send_sizes = jnp.sum(
            jnp.reshape(
                global_group_sizes, (num_ep, num_ep, num_local_experts)
            ),
            axis=-1,
            keepdims=False,
        )
        local_send_sizes = global_send_sizes[ep_shard_idx]
        local_recv_sizes = global_send_sizes[:, ep_shard_idx]

        global_in_offsets, global_out_offsets = get_global_input_output_offsets(
            global_send_sizes, num_ep
        )
        local_input_offsets = global_in_offsets[ep_shard_idx]
        local_output_offsets = global_out_offsets[ep_shard_idx]

        output_buffer_size = (
            min(self.experts_per_tok, num_local_experts) * B * T * num_ep
        )
        output_buffer = jax.lax.empty(
            shape=(output_buffer_size, D_global), dtype=inputs.dtype
        )

        sorted_inputs = jax.lax.ragged_all_to_all(
            sorted_inputs,
            output_buffer,
            local_input_offsets,
            local_send_sizes,
            local_output_offsets,
            local_recv_sizes,
            axis_name=ep_axis,
            axis_index_groups=None,
        )

        group_start_idx = ep_shard_idx * num_local_experts
        local_expert_group_sizes = jax.lax.dynamic_slice_in_dim(
            global_group_sizes, group_start_idx, num_local_experts, axis=1
        )
        local_group_sizes = jnp.sum(local_expert_group_sizes, axis=0)

        flat_local_expert_group_sizes = local_expert_group_sizes.reshape(-1)
        local_expert_indices = jnp.mod(
            jnp.arange(flat_local_expert_group_sizes.shape[0]),
            num_local_experts,
        )
        local_expert_indices = jnp.repeat(
            local_expert_indices,
            flat_local_expert_group_sizes,
            total_repeat_length=sorted_inputs.shape[0],
        )

        local_expert_sort_indices = jnp.argsort(local_expert_indices)
        sorted_inputs = _custom_permute(
            sorted_inputs, local_expert_sort_indices
        )
      else:
        local_group_sizes = group_sizes
        local_expert_sort_indices = jnp.arange(sorted_inputs.shape[0])

      m, k_dim = sorted_inputs.shape
      n = gate_w.shape[-1]
      ffn0_tiling = (
          round_up_to_base(min(128, m), base=8, threshold=8),
          round_up_to_base(min(128, k_dim), base=128, threshold=128),
          round_up_to_base(min(128, n), base=128, threshold=128),
      )
      ffn1_tiling = (ffn0_tiling[0], ffn0_tiling[2], ffn0_tiling[1])

      projected = megablox.gmm(
          sorted_inputs,
          gate_w,
          group_sizes=local_group_sizes,
          tiling=ffn0_tiling,
          preferred_element_type=inputs.dtype,
      )
      middle = jax.nn.silu(projected) * megablox.gmm(
          sorted_inputs,
          up_w,
          group_sizes=local_group_sizes,
          tiling=ffn0_tiling,
          preferred_element_type=inputs.dtype,
      )
      sorted_outputs = megablox.gmm(
          middle,
          down_w,
          group_sizes=local_group_sizes,
          tiling=ffn1_tiling,
          preferred_element_type=inputs.dtype,
      )

      if num_ep > 1:
        sorted_outputs = _custom_permute(
            sorted_outputs, jnp.argsort(local_expert_sort_indices)
        )

        global_in_offsets, global_out_offsets = get_global_input_output_offsets(
            global_send_sizes.T, num_ep  # pylint: disable=undefined-variable
        )
        local_send_sizes, local_recv_sizes = local_recv_sizes, local_send_sizes  # pylint: disable=undefined-variable

        output_buffer = jax.lax.empty(
            shape=(B * T * self.experts_per_tok, D_global), dtype=inputs.dtype
        )
        sorted_outputs = jax.lax.ragged_all_to_all(
            sorted_outputs,
            output_buffer,
            global_in_offsets[ep_shard_idx],
            local_send_sizes,
            global_out_offsets[ep_shard_idx],
            local_recv_sizes,
            axis_name=ep_axis,
            axis_index_groups=None,
        )

      outputs = _custom_permute(sorted_outputs, unsort_indices)
      outputs = outputs.reshape(B, T, self.experts_per_tok, D_global).transpose(
          0, 1, 3, 2
      )

      final_output = jnp.einsum('btdk,btk->btd', outputs, weights)

      if tp_axis is not None and tp_axis in mesh.axis_names:
        final_output = jax.lax.psum_scatter(
            final_output, axis_name=tp_axis, scatter_dimension=2, tiled=True
        )

      return final_output

    return sharded_megablox_moe(
        x,
        routing_weights,
        routing_idx,
        global_gate_w,
        global_up_w,
        global_down_w,
    )


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
    self.gate_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            self.shd_config.ffw_weight_df,
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.up_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(),
            self.shd_config.ffw_weight_df,
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.down_proj = nnx.Linear(
        in_features=config.hidden_dim,
        out_features=config.embed_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(), self.shd_config.ffw_weight_fd
        ),
        dtype=config.dtype,
        param_dtype=config.param_dtype,
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
    if (
        self.config.remat_config == RematConfig.BLOCK
        or self.config.remat_config == RematConfig.BLOCK.value
    ):
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
    self.config = config
    self.input_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.attn = Attention(
        config=config,
        rngs=rngs,
    )
    self.post_attention_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    if config.num_experts is None:
      self.mlp = MLP(
          config=config,
          rngs=rngs,
      )
    else:
      self.mlp = MoELayer(
          config=config,
          rngs=rngs,
      )

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    inputs_normalized = self.input_layernorm(x)
    cache, attn_output = self.attn(
        inputs_normalized,
        segment_pos,
        cache,
        attn_mask,
    )
    attn_output += x
    residual = attn_output
    attn_output = self.post_attention_layernorm(attn_output)
    outputs = self.mlp(attn_output)
    outputs = residual + outputs
    return cache, outputs

  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    if (
        self.config.remat_config == RematConfig.DECODER
        or self.config.remat_config == RematConfig.DECODER.value
    ):
      return nnx.remat(self.block.__func__)(self, x, segment_pos, cache, attn_mask)
    else:
      return self.block(x, segment_pos, cache, attn_mask)


class Qwen3(BackendMappingMixin, nnx.Module):
  """Qwen3 model."""

  BACKEND_PACKAGE_PATH = __name__

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.embedder = Embedder(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        rngs=rngs,
        shd_config=self.config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    self.layers = compat.ModuleList([
        DecoderLayer(config=config, rngs=rngs) for _ in range(config.num_layers)
    ])
    self.final_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        norm_eps=config.norm_eps,
        shd_config=self.config.shd_config,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
    )
    if not config.use_tied_embedding:
      self.lm_head = Einsum(
          einsum_str='BTD,DV->BTV',
          shape=(config.embed_dim, config.vocab_size),
          rngs=rngs,
          sharding=self.config.shd_config.emb_dv,
          dtype=config.dtype,
          param_dtype=config.param_dtype,
      )

  def init_cache(
      self, batch_size: int, cache_size: int, dtype: jnp.dtype
  ) -> Cache:
    """Initializes the cache for the model."""
    config = self.config
    shape = (batch_size, cache_size, config.num_kv_heads, config.head_dim)
    k = jnp.zeros(shape, dtype=config.dtype)
    v = jnp.zeros(shape, dtype=config.dtype)
    end_index = jnp.zeros((batch_size,), dtype=jnp.int32)
    # Jax array is immutable, so updates to each layer creates new arrays.
    return {
        f'layer_{i}': {'k': k, 'v': v, 'end_index': end_index}
        for i in range(config.num_layers)
    }

  def __call__(
      self,
      input_tokens: jaxtyping.Array,  # [B, L]
      positions: jaxtyping.Array,  # [B, L]
      cache: Cache | None,  # (sequence length L')
      attention_mask: jaxtyping.Array,  # [B, L, L']
      output_hidden_states: bool = False,
  ) -> tuple[jaxtyping.Array, Cache | None]:
    """Qwen3 model.

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
