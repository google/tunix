# Copyright 2026 Google LLC
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

"""Qwen3.5 hybrid (Gated DeltaNet + Output-Gated GQA) model."""

import dataclasses
import enum
import functools
from typing import Tuple

import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.interpreters import pxla
import jax.sharding as shd
from jax.sharding import PartitionSpec as P
import jaxtyping
from tunix.generate.mappings import BackendMappingMixin
from tunix.models.qwen3 import model as _qwen3_model
from tunix.utils import compat
from tunix.utils import env_utils

# `//third_party/py/jax/experimental:pallas_tpu_ops` has a package-group
# visibility allowlist in `//learning/brain/research/jax/lib/jax.bzl` that
# includes `//third_party/py/tunix/models/qwen3`. Re-export its splash
# attention modules here until `qwen3p5` is added to `pallas_tpu_ops_users`.
# TODO(tsbao): Import directly once added to `pallas_tpu_ops_users`.
splash = _qwen3_model.splash
mask_lib = _qwen3_model.mask_lib

env_utils.setup_sharding_environment()


K_MASK = -2.3819763e38

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


class RematConfig(enum.Enum):
  NONE = enum.auto()  # No remat, all activations will be stored in HBM.
  BLOCK = enum.auto()  # Remat the entire attn block.
  DECODER = enum.auto()  # Remat the entire decoder layer (attn + mlp).


class SplashAttentionImpl(enum.Enum):
  """Backend implementation to use for splash (flash) attention.

  JAX: `jax.experimental.pallas.ops.tpu.splash_attention`.
  TOKAMAX: `tokamax._src.ops.splash_attention`.
  """

  JAX = 'jax'
  TOKAMAX = 'tokamax'


def _tokamax_splash_libs():
  """Lazy import for Tokamax splash attention libraries."""
  try:
    from tokamax._src.ops.splash_attention import splash_attention_kernel as tokamax_splash_lib  # pylint: disable=g-import-not-at-top
    from tokamax._src.ops.splash_attention import splash_attention_mask as tokamax_splash_mask_lib  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    raise ImportError(
        "The Tokamax splash_attention backend requires the 'tokamax' package."
        ' Install it with `pip install tokamax` or keep the default'
        ' `splash_attention_impl=SplashAttentionImpl.JAX`'
    ) from e
  return tokamax_splash_lib, tokamax_splash_mask_lib


class LayerType(enum.Enum):
  LINEAR_ATTENTION = 'linear_attention'
  FULL_ATTENTION = 'full_attention'


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  """Sharding configuration for Qwen3.5 model."""

  emb_vd: Tuple[str | None, ...]
  emb_dv: Tuple[str | None, ...]
  q_weight_dnh: Tuple[str | None, ...]
  kv_weight_dkh: Tuple[str | None, ...]
  o_weight_nhd: Tuple[str | None, ...]
  ffw_weight_df: Tuple[str | None, ...]
  ffw_weight_fd: Tuple[str | None, ...]
  rms_norm_weight: Tuple[str | None, ...]
  act_btd: Tuple[str | None, ...]
  act_btf: Tuple[str | None, ...]
  act_btnh: Tuple[str | None, ...]
  exp_weight_edf: Tuple[str | None, ...]
  exp_weight_efd: Tuple[str | None, ...]
  linear_qkv_df: Tuple[str | None, ...]
  linear_z_df: Tuple[str | None, ...]
  linear_ba_dh: Tuple[str | None, ...]
  linear_conv_k1d: Tuple[str | None, ...]
  linear_head_h: Tuple[str | None, ...]
  linear_out_fd: Tuple[str | None, ...]
  head_norm_weight: Tuple[str | None, ...] = (None,)

  @staticmethod
  def get_default_sharding(is_sampling: bool = False) -> 'ShardingConfig':
    fsdp = 'fsdp' if not is_sampling else None

    return ShardingConfig(
        emb_vd=('tp', fsdp),
        emb_dv=(fsdp, 'tp'),
        q_weight_dnh=(fsdp, 'tp', None),
        kv_weight_dkh=(fsdp, 'tp', None),
        o_weight_nhd=('tp', None, fsdp),
        ffw_weight_df=(fsdp, 'tp'),
        ffw_weight_fd=('tp', fsdp),
        rms_norm_weight=(None,),
        act_btd=('fsdp', None, None),
        act_btf=('fsdp', None, 'tp'),
        act_btnh=('fsdp', None, 'tp', None),
        exp_weight_edf=(None, fsdp, 'tp'),
        exp_weight_efd=(None, 'tp', fsdp),
        linear_qkv_df=(fsdp, None),
        linear_z_df=(fsdp, 'tp'),
        linear_ba_dh=(fsdp, 'tp'),
        linear_conv_k1d=(None, None, None),
        linear_head_h=('tp',),
        linear_out_fd=('tp', fsdp),
        head_norm_weight=(None,),
    )


@dataclasses.dataclass(slots=True, kw_only=True)
class ModelConfig:
  """Configuration for Qwen3.5 hybrid models.

  Note: `GatedDeltaNet` uses a chunkwise parallel WY recurrence
  (`_chunk_gated_delta_rule`) with `@jax.checkpoint` per chunk when
  `seq_len > 1`. For training very long sequences under tight HBM budgets,
  `RematConfig.BLOCK` or `RematConfig.DECODER` can additionally recompute block
  or decoder activations during the backward pass.
  """

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
  use_norm_offset: bool = True
  attn_output_gate: bool = True
  partial_rotary_factor: float = 0.25
  full_attention_interval: int = 4
  layer_types: Tuple[LayerType, ...] | None = None
  linear_num_key_heads: int = 16
  linear_num_value_heads: int = 16
  linear_key_head_dim: int = 128
  linear_value_head_dim: int = 128
  linear_conv_kernel_dim: int = 4
  num_experts: int | None = None
  num_experts_per_tok: int | None = None
  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
  remat_config: RematConfig = RematConfig.NONE
  dtype: jax.typing.DTypeLike = jnp.bfloat16
  param_dtype: jax.typing.DTypeLike = jnp.bfloat16
  use_flash_attention: bool = False
  flash_attention_block_size: int = 1024
  splash_attention_impl: SplashAttentionImpl = SplashAttentionImpl.JAX

  def __post_init__(self):
    if self.layer_types is None:
      self.layer_types = tuple(
          LayerType.FULL_ATTENTION
          if (i + 1) % self.full_attention_interval == 0
          else LayerType.LINEAR_ATTENTION
          for i in range(self.num_layers)
      )
    if len(self.layer_types) != self.num_layers:
      raise ValueError(
          f'layer_types length ({len(self.layer_types)}) must match '
          f'num_layers ({self.num_layers}).'
      )

  @property
  def rotary_dim(self) -> int:
    return int(self.head_dim * self.partial_rotary_factor)

  @classmethod
  def qwen3p5_0p8b(cls) -> 'ModelConfig':  # Qwen/Qwen3.5-0.8B
    return cls(
        num_layers=24,
        vocab_size=248320,
        embed_dim=1024,
        hidden_dim=3584,
        num_heads=8,
        head_dim=256,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=10_000_000,
        use_tied_embedding=True,
        use_norm_offset=True,
        attn_output_gate=True,
        partial_rotary_factor=0.25,
        full_attention_interval=4,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )

  @classmethod
  def qwen3p5_2b(cls) -> 'ModelConfig':  # Qwen/Qwen3.5-2B
    return cls(
        num_layers=24,
        vocab_size=248320,
        embed_dim=2048,
        hidden_dim=6144,
        num_heads=8,
        head_dim=256,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=10_000_000,
        use_tied_embedding=True,
        use_norm_offset=True,
        attn_output_gate=True,
        partial_rotary_factor=0.25,
        full_attention_interval=4,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )

  @classmethod
  def qwen3p5_35b_a3b(cls) -> 'ModelConfig':  # Qwen/Qwen3.5-35B-A3B
    return cls(
        num_layers=40,
        vocab_size=248320,
        embed_dim=2048,
        hidden_dim=512,
        num_heads=16,
        head_dim=256,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=10_000_000,
        use_tied_embedding=False,
        use_norm_offset=True,
        attn_output_gate=True,
        partial_rotary_factor=0.25,
        full_attention_interval=4,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=256,
        num_experts_per_tok=8,
    )

  @classmethod
  def qwen3p5_397b_a17b(cls) -> 'ModelConfig':  # Qwen/Qwen3.5-397B-A17B
    return cls(
        num_layers=60,
        vocab_size=248320,
        embed_dim=4096,
        hidden_dim=1024,
        num_heads=32,
        head_dim=256,
        num_kv_heads=2,
        norm_eps=1e-06,
        rope_theta=10_000_000,
        use_tied_embedding=False,
        use_norm_offset=True,
        attn_output_gate=True,
        partial_rotary_factor=0.25,
        full_attention_interval=4,
        linear_num_key_heads=16,
        linear_num_value_heads=64,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=512,
        num_experts_per_tok=10,
    )


def shard(x: jnp.ndarray, s: Tuple[str | None, ...]):
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == 'cpu':
    return x
  return jax.lax.with_sharding_constraint(x, shd.NamedSharding(mesh, P(*s)))


class Embedder(nnx.Module):
  """Embedder module."""

  def __init__(
      self,
      vocab_size: int,
      embed_dim: int,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      param_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      dtype: jax.typing.DTypeLike = jnp.bfloat16,
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
    x = jnp.asarray(x, dtype=self.dtype)
    x = shard(x, self.shd_config.act_btd)
    return x

  @jax.named_scope('embedder_decode')
  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = jnp.asarray(x, dtype=self.dtype)
    return jnp.dot(x, self.input_embedding.value.T.astype(self.dtype))


class Einsum(nnx.Module):
  """Einsum is a convenience module for parameterized tensor multiplication."""

  def __init__(
      self,
      einsum_str: str,
      shape: flax.typing.Shape,
      *,
      rngs: nnx.Rngs,
      sharding: Tuple[str | None, ...],
      param_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      dtype: jax.typing.DTypeLike = jnp.bfloat16,
  ):
    self.einsum_str = einsum_str
    self.shape = shape
    self.dtype = dtype
    self.w = nnx.Param(
        nnx.initializers.normal(dtype=param_dtype)(rngs.params(), shape),
        sharding=sharding,
    )

  @jax.named_scope('einsum')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = jnp.asarray(x, dtype=self.dtype)
    return jnp.einsum(self.einsum_str, x, self.w.value.astype(self.dtype))


def apply_rope(
    inputs: jaxtyping.Array,  # [B, L, N, rotary_dim]
    positions: jaxtyping.Array,  # [B, L]
    head_dim: int,
    rope_theta: int = 10_000_000,
) -> jaxtyping.Array:
  """Applies Rotary Position Embeddings (RoPE) to the rotary slice."""
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
  """RMSNorm module supporting Qwen3.5 zero-centered (1 + w) offset."""

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float = 1e-06,
      use_norm_offset: bool = True,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      weight_sharding: Tuple[str | None, ...] | None = None,
      param_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      dtype: jax.typing.DTypeLike = jnp.bfloat16,
  ):
    init_fn = (
        nnx.initializers.zeros_init()
        if use_norm_offset
        else nnx.initializers.ones_init()
    )
    resolved_sharding = (
        shd_config.rms_norm_weight
        if weight_sharding is None
        else weight_sharding
    )
    self.w = nnx.Param(
        init_fn(rngs.params(), (dim,), param_dtype),
        sharding=resolved_sharding,
    )
    self.norm_eps = norm_eps
    self.use_norm_offset = use_norm_offset
    self.dtype = dtype

  @jax.named_scope('rms_norm')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    x_f32 = x.astype(jnp.float32)
    rms = jax.lax.rsqrt(
        jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + self.norm_eps
    )
    scale = self.w.value.astype(jnp.float32)
    if self.use_norm_offset:
      scale = 1.0 + scale
    return (x_f32 * rms * scale).astype(self.dtype)


class RMSNormGated(nnx.Module):
  """Gated RMSNorm module for Qwen3.5 Gated DeltaNet output normalization."""

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float = 1e-06,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      param_dtype: jax.typing.DTypeLike = jnp.bfloat16,
      dtype: jax.typing.DTypeLike = jnp.bfloat16,
  ):
    self.w = nnx.Param(
        nnx.initializers.ones_init()(rngs.params(), (dim,), param_dtype),
        sharding=shd_config.head_norm_weight,
    )
    self.norm_eps = norm_eps
    self.dtype = dtype

  @jax.named_scope('rms_norm_gated')
  def __call__(
      self, hidden_states: jaxtyping.Array, gate: jaxtyping.Array
  ) -> jaxtyping.Array:
    input_dtype = hidden_states.dtype
    x_f32 = hidden_states.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(
        jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + self.norm_eps
    )
    normed = (x_f32 * inv_rms).astype(input_dtype)
    normed = self.w.value.astype(input_dtype) * normed
    gated = normed.astype(jnp.float32) * jax.nn.silu(gate.astype(jnp.float32))
    return gated.astype(input_dtype)


def _l2norm(x: jaxtyping.Array, eps: float = 1e-6) -> jaxtyping.Array:
  """L2-normalizes x along the last dimension in float32."""
  x_f32 = x.astype(jnp.float32)
  inv_norm = jax.lax.rsqrt(
      jnp.sum(jnp.square(x_f32), axis=-1, keepdims=True) + eps
  )
  return x_f32 * inv_norm


def _extract_token_mask(
    attention_mask: jaxtyping.Array | None, seq_len: int
) -> jaxtyping.Array | None:
  """Extracts per-token validity mask (B, T) from causal attention_mask.

  For `seq_len > 1`, assumes initial prefill (`end_index == 0`) so that query
  position `t` corresponds to key column `t` along the diagonal of
  `attention_mask[:, :seq_len, :seq_len]`.

  Args:
    attention_mask: Boolean causal mask of shape `(B, T, S)` or `(B, 1, T, S)`.
    seq_len: Query sequence length `T`.

  Returns:
    Boolean validity mask of shape `(B, T)`, or `None` if `attention_mask` is
    `None`.
  """
  if attention_mask is None:
    return None
  mask = attention_mask
  if mask.ndim == 4:
    mask = mask[:, 0, :, :]
  if seq_len == 1:
    return jnp.any(mask, axis=-1)
  return jnp.diagonal(mask[:, :seq_len, :seq_len], axis1=1, axis2=2)


def _align_kv_heads_for_splash(
    key_proj: jaxtyping.Array,
    value_proj: jaxtyping.Array,
    num_q_heads: int,
    head_shards: int,
) -> tuple[jaxtyping.Array, jaxtyping.Array, int]:
  """Aligns KV heads with `head_shards` for Splash GQA head sharding.

  Splash Attention infers the GQA group size from per-shard local head counts
  (`local_q_heads // local_kv_heads`). When `head_shards > 1` does not divide
  `num_kv_heads` (for example `num_q_heads=8, num_kv_heads=2, head_shards=4`),
  replicating KV heads across shards would map query heads on shard `s > 0` to
  the wrong KV heads. Repeating KV heads to `num_q_heads` before `shard_map`
  ensures each shard receives its exact corresponding KV heads.

  Args:
    key_proj: Key tensor of shape `(B, K, T, D)`.
    value_proj: Value tensor of shape `(B, K, T, D)`.
    num_q_heads: Total number of query heads `N`.
    head_shards: Number of shards along the head dimension.

  Returns:
    Tuple of `(key_proj, value_proj, effective_num_kv_heads)`.
  """
  if num_q_heads % head_shards != 0:
    raise ValueError(
        f'num_heads ({num_q_heads}) must be divisible by head_shards'
        f' ({head_shards}).'
    )
  num_kv_heads = key_proj.shape[1]
  if head_shards > 1 and num_kv_heads % head_shards != 0:
    if num_q_heads % num_kv_heads != 0:
      raise ValueError(
          f'num_heads ({num_q_heads}) must be divisible by num_kv_heads'
          f' ({num_kv_heads}).'
      )
    repeat_factor = num_q_heads // num_kv_heads
    key_proj = jnp.repeat(key_proj, repeat_factor, axis=1)
    value_proj = jnp.repeat(value_proj, repeat_factor, axis=1)
    num_kv_heads = num_q_heads
  return key_proj, value_proj, num_kv_heads


def _invert_unit_lower_triangular(s_mat: jaxtyping.Array) -> jaxtyping.Array:
  """Inverts unit lower-triangular `(I + s_mat)` via 2x2 block recursion on MXU.

  For strictly lower-triangular `s_mat` of power-of-two size `C`, recursively
  inverts diagonal blocks of size `b = 1, 2, 4, ..., C // 2` using
  `[[A00, 0], [S10, A11]]^{-1} = [[A00^{-1}, 0], [-A11^{-1} S10 A00^{-1},
  A11^{-1}]]`. This avoids XLA's `TriangularSolve` unrolled sequential column
  loop for `C < 128`.

  Args:
    s_mat: Strictly lower-triangular float32 tensor of shape
      `(*batch_dims, C, C)` where `C` is a power of two.

  Returns:
    Unit lower-triangular inverse `(I + s_mat)^{-1}` of shape
    `(*batch_dims, C, C)`.
  """
  batch_dims = s_mat.shape[:-2]
  c_size = s_mat.shape[-1]
  if c_size == 1:
    return jnp.ones_like(s_mat)

  inv_blocks = jnp.ones((*batch_dims, c_size, 1, 1), dtype=s_mat.dtype)
  b = 1
  while b < c_size:
    num_pairs = c_size // (2 * b)
    inv_pairs = inv_blocks.reshape(*batch_dims, num_pairs, 2, b, b)
    inv_00 = inv_pairs[..., 0, :, :]
    inv_11 = inv_pairs[..., 1, :, :]

    s_pairs = s_mat.reshape(*batch_dims, num_pairs, 2, b, num_pairs, 2, b)
    s_sub = s_pairs[..., 1, :, :, 0, :]
    s_10 = jnp.moveaxis(
        jnp.diagonal(s_sub, axis1=-4, axis2=-2), -1, -3
    )

    if b == 1:
      inv_10 = -s_10
    else:
      inv_10 = -jnp.matmul(
          inv_11,
          jnp.matmul(s_10, inv_00, precision=jax.lax.Precision.HIGHEST),
          precision=jax.lax.Precision.HIGHEST,
      )

    zeros_01 = jnp.zeros_like(inv_00)
    top = jnp.concatenate([inv_00, zeros_01], axis=-1)
    bot = jnp.concatenate([inv_10, inv_11], axis=-1)
    inv_blocks = jnp.concatenate([top, bot], axis=-2)
    b *= 2

  return jnp.squeeze(inv_blocks, axis=-3)


def _chunk_gated_delta_rule(
    q: jaxtyping.Array,
    k: jaxtyping.Array,
    v: jaxtyping.Array,
    g: jaxtyping.Array,
    beta: jaxtyping.Array,
    initial_state: jaxtyping.Array,
    valid_seg: jaxtyping.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jaxtyping.Array, jaxtyping.Array]:
  """Chunkwise parallel Gated DeltaNet recurrence using the WY representation.

  Reformulates the token-by-token delta rule recurrence over chunks of size `C`
  so intra-chunk transitions are evaluated via matrix-matrix multiplications on
  the TPU MXU and `(I + L)^{-1}` triangular solves, while inter-chunk state
  propagation scans over `ceil(T / C)` chunks.

  Args:
    q: L2-normalized and scaled query tensor of shape `(B, T, H, D_k)` in
      float32.
    k: L2-normalized key tensor of shape `(B, T, H, D_k)` in float32 (zeroed on
      masked tokens).
    v: Value tensor of shape `(B, T, H, D_v)` in float32 (zeroed on masked
      tokens).
    g: Log-decay tensor of shape `(B, T, H)` in float32 (`<= 0`, zeroed on
      masked tokens).
    beta: Writing gate tensor of shape `(B, T, H)` in float32 (zeroed on masked
      tokens).
    initial_state: Initial recurrent state of shape `(B, H, D_k, D_v)` in
      float32.
    valid_seg: Optional integer segment IDs of shape `(B, T)` (`0` on padding).
      When provided, recurrent state is reset to `0` at segment boundaries and
      on padding tokens (`valid_seg == 0`).
    chunk_size: Chunk length `C` (power of two) for intra-chunk
      parallelization.

  Returns:
    Tuple of `(final_state, core_out)` where `final_state` has shape
    `(B, H, D_k, D_v)` and `core_out` has shape `(B, T, H, D_v)` in float32.
  """
  batch_size, seq_len, num_heads, head_k_dim = q.shape
  head_v_dim = v.shape[-1]
  target_c = min(chunk_size, seq_len)
  c_size = 1 << (target_c - 1).bit_length()
  pad_len = (c_size - (seq_len % c_size)) % c_size
  if pad_len > 0:
    q = jnp.pad(q, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
    k = jnp.pad(k, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
    v = jnp.pad(v, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
    g = jnp.pad(g, ((0, 0), (0, pad_len), (0, 0)))
    beta = jnp.pad(beta, ((0, 0), (0, pad_len), (0, 0)))

  padded_len = seq_len + pad_len
  num_chunks = padded_len // c_size

  q_c = q.reshape(
      batch_size, num_chunks, c_size, num_heads, head_k_dim
  ).transpose(0, 1, 3, 2, 4)
  k_c = k.reshape(
      batch_size, num_chunks, c_size, num_heads, head_k_dim
  ).transpose(0, 1, 3, 2, 4)
  v_c = v.reshape(
      batch_size, num_chunks, c_size, num_heads, head_v_dim
  ).transpose(0, 1, 3, 2, 4)
  g_c = g.reshape(batch_size, num_chunks, c_size, num_heads).transpose(
      0, 1, 3, 2
  )
  beta_c = beta.reshape(batch_size, num_chunks, c_size, num_heads).transpose(
      0, 1, 3, 2
  )

  g_cumsum = jnp.cumsum(g_c, axis=-1)  # (B, NC, H, C)
  g_diff = jnp.tril(g_cumsum[..., :, None] - g_cumsum[..., None, :])
  decay_matrix = jnp.exp(g_diff)  # (B, NC, H, C, C)

  mask_strict = jnp.tril(jnp.ones((c_size, c_size), dtype=jnp.bool_), k=-1)
  mask_causal = jnp.tril(jnp.ones((c_size, c_size), dtype=jnp.bool_), k=0)

  if valid_seg is not None:
    prev_seg = jnp.pad(valid_seg[:, :-1], ((0, 0), (1, 0)), constant_values=0)
    reset = (valid_seg != prev_seg) | (valid_seg == 0)  # (B, T)
    if pad_len > 0:
      reset = jnp.pad(reset, ((0, 0), (0, pad_len)), constant_values=False)
    run_id = jnp.cumsum(reset.astype(jnp.int32), axis=-1)
    run_id_c = run_id.reshape(batch_size, num_chunks, c_size)
    prev_chunk_run_id = jnp.concatenate(
        [jnp.zeros((batch_size, 1), dtype=jnp.int32), run_id_c[:, :-1, -1]],
        axis=1,
    )
    same_run = (
        run_id_c[:, :, None, :, None] == run_id_c[:, :, None, None, :]
    )  # (B, NC, 1, C, C)
    carry_in_mask = (
        run_id_c == prev_chunk_run_id[:, :, None]
    )[:, :, None, :, None]  # (B, NC, 1, C, 1)
    mask_strict_full = mask_strict & same_run
  else:
    run_id_c = None
    prev_chunk_run_id = None
    carry_in_mask = None
    mask_strict_full = mask_strict

  k_beta = k_c * beta_c[..., None]  # (B, NC, H, C, D_k)
  s_mat = jnp.matmul(
      k_beta, k_c.swapaxes(-1, -2), precision=jax.lax.Precision.HIGHEST
  )
  s_mat = jnp.where(mask_strict_full, s_mat * decay_matrix, 0.0)

  a_mat = _invert_unit_lower_triangular(s_mat)

  v_beta = v_c * beta_c[..., None]
  u_chunks = jnp.matmul(a_mat, v_beta, precision=jax.lax.Precision.HIGHEST)

  g_exp = jnp.exp(g_cumsum)[..., None]  # (B, NC, H, C, 1)
  if carry_in_mask is not None:
    g_exp = jnp.where(carry_in_mask, g_exp, 0.0)
  k_beta_g = k_beta * g_exp
  w_chunks = jnp.matmul(a_mat, k_beta_g, precision=jax.lax.Precision.HIGHEST)

  def _chunk_step_impl(h, scan_inputs):
    u_i, w_i, q_i, k_i, g_cs_i, run_id_i, prev_run_i = scan_inputs
    v_new = u_i - jnp.matmul(w_i, h, precision=jax.lax.Precision.HIGHEST)

    g_exp_i = jnp.exp(g_cs_i)[..., None]  # (B, H, C, 1)
    g_diff_i = jnp.tril(g_cs_i[..., :, None] - g_cs_i[..., None, :])
    decay_i = jnp.exp(g_diff_i)  # (B, H, C, C)
    g_last_exp_i = jnp.exp(g_cs_i[..., -1:])[..., None]  # (B, H, 1, 1)
    g_diff_last_i = jnp.exp(g_cs_i[..., -1:] - g_cs_i)[
        ..., None
    ]  # (B, H, C, 1)

    if run_id_i is not None and prev_run_i is not None:
      same_run_i = run_id_i[:, None, :, None] == run_id_i[:, None, None, :]
      carry_in_i = (run_id_i == prev_run_i[:, None])[:, None, :, None]
      carry_out_i = (run_id_i[:, -1:] == prev_run_i[:, None])[:, None, :, None]
      last_run_i = (run_id_i == run_id_i[:, -1:])[:, None, :, None]
      g_exp_i = jnp.where(carry_in_i, g_exp_i, 0.0)
      g_last_exp_i = jnp.where(carry_out_i, g_last_exp_i, 0.0)
      g_diff_last_i = jnp.where(last_run_i, g_diff_last_i, 0.0)
      causal_mask_i = mask_causal & same_run_i
    else:
      causal_mask_i = mask_causal

    q_g_i = q_i * g_exp_i
    o_inter = jnp.matmul(q_g_i, h, precision=jax.lax.Precision.HIGHEST)

    qk_i = jnp.matmul(
        q_i, k_i.swapaxes(-1, -2), precision=jax.lax.Precision.HIGHEST
    )
    attn_i = jnp.where(causal_mask_i, qk_i * decay_i, 0.0)
    o_intra = jnp.matmul(attn_i, v_new, precision=jax.lax.Precision.HIGHEST)
    o_i = o_inter + o_intra

    k_last_i = k_i * g_diff_last_i
    h_new = h * g_last_exp_i + jnp.matmul(
        k_last_i.swapaxes(-1, -2), v_new, precision=jax.lax.Precision.HIGHEST
    )
    return h_new, o_i

  if num_chunks == 1:
    final_state, o_single = _chunk_step_impl(
        initial_state,
        (
            u_chunks[:, 0],
            w_chunks[:, 0],
            q_c[:, 0],
            k_c[:, 0],
            g_cumsum[:, 0],
            run_id_c[:, 0] if run_id_c is not None else None,
            prev_chunk_run_id[:, 0] if prev_chunk_run_id is not None else None,
        ),
    )
    core_out = o_single.transpose(0, 2, 1, 3)
  else:
    scan_xs = (
        jnp.swapaxes(u_chunks, 0, 1),
        jnp.swapaxes(w_chunks, 0, 1),
        jnp.swapaxes(q_c, 0, 1),
        jnp.swapaxes(k_c, 0, 1),
        jnp.swapaxes(g_cumsum, 0, 1),
        jnp.swapaxes(run_id_c, 0, 1) if run_id_c is not None else None,
        jnp.swapaxes(prev_chunk_run_id, 0, 1)
        if prev_chunk_run_id is not None
        else None,
    )
    final_state, core_out_scan = jax.lax.scan(
        jax.checkpoint(_chunk_step_impl), initial_state, scan_xs
    )
    core_out = core_out_scan.transpose(1, 0, 3, 2, 4).reshape(
        batch_size, padded_len, num_heads, head_v_dim
    )

  if pad_len > 0:
    core_out = core_out[:, :seq_len, :, :]
  return final_state, core_out


class GatedDeltaNet(nnx.Module):
  """Qwen3.5 Gated DeltaNet linear attention module."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.shd_config = config.shd_config
    self.num_k_heads = config.linear_num_key_heads
    self.num_v_heads = config.linear_num_value_heads
    self.head_k_dim = config.linear_key_head_dim
    self.head_v_dim = config.linear_value_head_dim
    self.conv_kernel_size = config.linear_conv_kernel_dim
    self.key_dim = self.num_k_heads * self.head_k_dim
    self.value_dim = self.num_v_heads * self.head_v_dim
    self.conv_dim = self.key_dim * 2 + self.value_dim

    self.in_proj_qkv = nnx.Linear(
        config.embed_dim,
        self.conv_dim,
        use_bias=False,
        rngs=rngs,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.normal(), self.shd_config.linear_qkv_df
        ),
    )
    self.in_proj_z = nnx.Linear(
        config.embed_dim,
        self.value_dim,
        use_bias=False,
        rngs=rngs,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.normal(), self.shd_config.linear_z_df
        ),
    )
    self.in_proj_b = nnx.Linear(
        config.embed_dim,
        self.num_v_heads,
        use_bias=False,
        rngs=rngs,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.normal(), self.shd_config.linear_ba_dh
        ),
    )
    self.in_proj_a = nnx.Linear(
        config.embed_dim,
        self.num_v_heads,
        use_bias=False,
        rngs=rngs,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.normal(), self.shd_config.linear_ba_dh
        ),
    )
    # `self.conv1d` serves as a parameter container for the depthwise conv
    # kernel `(K, 1, conv_dim)`; `block()` applies the causal depthwise
    # convolution explicitly so `conv_state` caching and segment masking can be
    # controlled directly.
    self.conv1d = nnx.Conv(
        in_features=self.conv_dim,
        out_features=self.conv_dim,
        kernel_size=(self.conv_kernel_size,),
        feature_group_count=self.conv_dim,
        padding='CAUSAL',
        use_bias=False,
        rngs=rngs,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.normal(), self.shd_config.linear_conv_k1d
        ),
    )
    self.dt_bias = nnx.Param(
        nnx.initializers.ones_init()(
            rngs.params(), (self.num_v_heads,), config.param_dtype
        ),
        sharding=self.shd_config.linear_head_h,
    )
    self.A_log = nnx.Param(  # pylint: disable=invalid-name
        nnx.initializers.zeros_init()(
            rngs.params(), (self.num_v_heads,), config.param_dtype
        ),
        sharding=self.shd_config.linear_head_h,
    )
    self.norm = RMSNormGated(
        self.head_v_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=self.shd_config,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    self.out_proj = nnx.Linear(
        self.value_dim,
        config.embed_dim,
        use_bias=False,
        rngs=rngs,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.normal(), self.shd_config.linear_out_fd
        ),
    )

  def block(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attention_mask: jaxtyping.Array | None,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    batch_size, seq_len, _ = x.shape
    if segment_ids is not None and cache is not None and seq_len > 1:
      raise ValueError(
          'segment_ids with cache is not supported for seq_len > 1.'
      )
    x = jnp.asarray(x, dtype=self.config.dtype)

    token_mask = _extract_token_mask(attention_mask, seq_len)
    if segment_ids is not None:
      seg_token_mask = segment_ids != 0
      token_mask = (
          seg_token_mask
          if token_mask is None
          else (token_mask & seg_token_mask)
      )
    if token_mask is not None:
      x = jnp.where(token_mask[:, :, None], x, jnp.zeros_like(x))

    mixed_qkv = self.in_proj_qkv(x)  # (B, T, conv_dim)
    z = self.in_proj_z(x).reshape(
        batch_size, seq_len, self.num_v_heads, self.head_v_dim
    )
    b = self.in_proj_b(x)  # (B, T, num_v_heads)
    a = self.in_proj_a(x)  # (B, T, num_v_heads)

    state_len = self.conv_kernel_size - 1
    if cache is None:
      conv_prefix = jnp.zeros(
          (batch_size, state_len, self.conv_dim), dtype=mixed_qkv.dtype
      )
      recurrent_state = jnp.zeros(
          (batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim),
          dtype=jnp.float32,
      )
    else:
      conv_prefix = cache['conv_state'].astype(mixed_qkv.dtype)
      recurrent_state = cache['recurrent_state']

    conv_input = jnp.concatenate([conv_prefix, mixed_qkv], axis=1)

    conv_w = jnp.squeeze(self.conv1d.kernel.value, axis=1).astype(
        jnp.float32
    )  # (K, conv_dim)
    valid_seg = None
    padded_seg = None
    if segment_ids is not None and cache is None:
      valid_seg = (
          jnp.where(token_mask, segment_ids, 0)
          if token_mask is not None
          else segment_ids
      )
      padded_seg = jnp.pad(
          valid_seg, ((0, 0), (state_len, 0)), constant_values=0
      )
    conv_out = jnp.zeros(mixed_qkv.shape, dtype=jnp.float32)
    conv_input_f32 = conv_input.astype(jnp.float32)
    for k_idx in range(self.conv_kernel_size):
      step_input = conv_input_f32[:, k_idx : k_idx + seq_len, :]
      if padded_seg is not None and valid_seg is not None:
        same_seg = (padded_seg[:, k_idx : k_idx + seq_len] == valid_seg) & (
            valid_seg != 0
        )
        step_input = jnp.where(same_seg[:, :, None], step_input, 0.0)
      conv_out = conv_out + step_input * conv_w[k_idx, None, None, :]
    conv_out = jax.nn.silu(conv_out).astype(mixed_qkv.dtype)

    q, k, v = jnp.split(conv_out, [self.key_dim, self.key_dim * 2], axis=-1)
    q = q.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
    k = k.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
    v = v.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

    if self.num_v_heads > self.num_k_heads:
      repeat_ratio = self.num_v_heads // self.num_k_heads
      q = jnp.repeat(q, repeat_ratio, axis=2)
      k = jnp.repeat(k, repeat_ratio, axis=2)

    q = shard(q, self.shd_config.act_btnh)
    k = shard(k, self.shd_config.act_btnh)
    v = shard(v, self.shd_config.act_btnh)

    beta = jax.nn.sigmoid(b.astype(jnp.float32))
    g = -jnp.exp(self.A_log.value.astype(jnp.float32)) * jax.nn.softplus(
        a.astype(jnp.float32) + self.dt_bias.value.astype(jnp.float32)
    )
    if token_mask is not None:
      g = jnp.where(token_mask[:, :, None], g, 0.0)
      beta = jnp.where(token_mask[:, :, None], beta, 0.0)

    scale = self.head_k_dim**-0.5
    q_f32 = _l2norm(q, eps=1e-6) * scale
    k_f32 = _l2norm(k, eps=1e-6)
    v_f32 = v.astype(jnp.float32)
    if token_mask is not None:
      k_f32 = jnp.where(token_mask[:, :, None, None], k_f32, 0.0)
      v_f32 = jnp.where(token_mask[:, :, None, None], v_f32, 0.0)

    if seq_len == 1:
      g_step = jnp.exp(g[:, 0, :])[..., None, None]  # (B, H, 1, 1)
      beta_step = beta[:, 0, :][..., None]  # (B, H, 1)
      q_step = q_f32[:, 0, :, :]  # (B, H, D_k)
      k_step = k_f32[:, 0, :, :]  # (B, H, D_k)
      v_step = v_f32[:, 0, :, :]  # (B, H, D_v)

      state = recurrent_state * g_step
      kv_mem = jnp.sum(state * k_step[..., None], axis=-2)  # (B, H, D_v)
      delta = (v_step - kv_mem) * beta_step  # (B, H, D_v)
      new_recurrent_state = state + k_step[..., :, None] * delta[..., None, :]
      core_out = jnp.sum(new_recurrent_state * q_step[..., None], axis=-2)[
          :, None, :, :
      ]  # (B, 1, H, D_v)
    else:
      new_recurrent_state, core_out = _chunk_gated_delta_rule(
          q_f32,
          k_f32,
          v_f32,
          g,
          beta,
          recurrent_state,
          valid_seg=valid_seg,
      )

    core_out = self.norm(core_out.astype(self.config.dtype), z)
    core_out = core_out.reshape(batch_size, seq_len, self.value_dim)
    output = self.out_proj(core_out)
    output = shard(output, self.shd_config.act_btd)

    if cache is None:
      new_cache = None
    else:
      candidate_conv_state = conv_input[:, -state_len:, :].astype(
          cache['conv_state'].dtype
      )
      if seq_len == 1 and token_mask is not None:
        new_conv_state = jnp.where(
            token_mask[:, :, None], candidate_conv_state, cache['conv_state']
        )
      else:
        new_conv_state = candidate_conv_state
      new_cache = {
          'conv_state': new_conv_state,
          'recurrent_state': new_recurrent_state,
          'end_index': cache['end_index'] + seq_len,
      }
    return new_cache, output

  @jax.named_scope('gated_delta_net')
  def __call__(
      self,
      x: jaxtyping.Array,
      cache: LayerCache | None,
      attention_mask: jaxtyping.Array | None,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    if (
        self.config.remat_config == RematConfig.BLOCK
        or self.config.remat_config == RematConfig.BLOCK.value
    ) and cache is None:
      graphdef, state = nnx.split(self)

      def _checkpointed_block(state, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        return module.block(*args, **kwargs)

      return jax.checkpoint(_checkpointed_block)(
          state, x, cache, attention_mask, segment_ids=segment_ids
      )
    return self.block(x, cache, attention_mask, segment_ids=segment_ids)


class Attention(nnx.Module):
  """Qwen3.5 Output-Gated Full Attention module with Partial RoPE."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.shd_config = config.shd_config
    self.q_head_multiplier = 2 if config.attn_output_gate else 1
    self.q_proj = Einsum(
        einsum_str='BTD,DNH->BTNH',
        shape=(
            config.embed_dim,
            config.num_heads,
            config.head_dim * self.q_head_multiplier,
        ),
        rngs=rngs,
        sharding=self.shd_config.q_weight_dnh,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    self.k_proj = Einsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dkh,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    self.v_proj = Einsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dkh,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    self.o_proj = Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(config.num_heads, config.head_dim, config.embed_dim),
        rngs=rngs,
        sharding=self.shd_config.o_weight_nhd,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    self.q_norm = RMSNorm(
        config.head_dim,
        norm_eps=config.norm_eps,
        use_norm_offset=config.use_norm_offset,
        rngs=rngs,
        shd_config=self.shd_config,
        weight_sharding=self.shd_config.head_norm_weight,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    self.k_norm = RMSNorm(
        config.head_dim,
        norm_eps=config.norm_eps,
        use_norm_offset=config.use_norm_offset,
        rngs=rngs,
        shd_config=self.shd_config,
        weight_sharding=self.shd_config.head_norm_weight,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    self.n_rep = config.num_heads // config.num_kv_heads
    self.scale = self.head_dim**-0.5

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    seq_len = x.shape[1]

    q_proj_out = self.q_proj(x)
    if self.config.attn_output_gate:
      query_positions, gate = jnp.split(q_proj_out, 2, axis=-1)
    else:
      query_positions = q_proj_out
      gate = None

    query_positions = self.q_norm(query_positions)
    key_positions = self.k_norm(self.k_proj(x))
    value_positions = self.v_proj(x)

    query_positions = shard(query_positions, self.shd_config.act_btnh)
    key_positions = shard(key_positions, self.shd_config.act_btnh)
    value_positions = shard(value_positions, self.shd_config.act_btnh)

    rotary_dim = self.config.rotary_dim
    if rotary_dim < self.head_dim:
      q_rot = apply_rope(
          query_positions[..., :rotary_dim],
          segment_pos,
          head_dim=rotary_dim,
          rope_theta=self.config.rope_theta,
      )
      k_rot = apply_rope(
          key_positions[..., :rotary_dim],
          segment_pos,
          head_dim=rotary_dim,
          rope_theta=self.config.rope_theta,
      )
      query_positions = jnp.concatenate(
          [q_rot, query_positions[..., rotary_dim:]], axis=-1
      )
      key_positions = jnp.concatenate(
          [k_rot, key_positions[..., rotary_dim:]], axis=-1
      )
    else:
      query_positions = apply_rope(
          query_positions,
          segment_pos,
          head_dim=self.head_dim,
          rope_theta=self.config.rope_theta,
      )
      key_positions = apply_rope(
          key_positions,
          segment_pos,
          head_dim=self.head_dim,
          rope_theta=self.config.rope_theta,
      )

    if cache is not None:
      end_index = cache['end_index'][0]
      slice_indices = (0, end_index % cache['v'].shape[1], 0, 0)
      cache_value_positions = jax.lax.dynamic_update_slice(
          cache['v'],
          value_positions,
          slice_indices,
      )
      cache_key_positions = jax.lax.dynamic_update_slice(
          cache['k'], key_positions, slice_indices
      )
    else:
      cache_value_positions = value_positions
      cache_key_positions = key_positions

    b, t, qh, d = query_positions.shape
    _, _, kh, _ = key_positions.shape
    if segment_ids is not None and cache is not None and seq_len > 1:
      raise ValueError(
          'segment_ids with cache is not supported for seq_len > 1.'
      )

    # NB: flash attention doesn't work for decoding step. When `cache` is
    # provided with `seq_len > 1`, flash prefill assumes initial prefill
    # (`end_index == 0`).
    if self.config.use_flash_attention and seq_len > 1:
      query_proj = query_positions.transpose(0, 2, 1, 3) * self.scale
      key_proj = key_positions.transpose(0, 2, 1, 3)
      value_proj = value_positions.transpose(0, 2, 1, 3)

      effective_segment_ids = segment_ids
      if effective_segment_ids is None and attn_mask is not None:
        token_mask = _extract_token_mask(attn_mask, seq_len)
        if token_mask is not None:
          effective_segment_ids = token_mask.astype(jnp.int32)

      mesh = pxla.thread_resources.env.physical_mesh
      if mesh.empty:
        try:
          abs_mesh = jax.sharding.get_abstract_mesh()
          if abs_mesh is not None and not abs_mesh.empty:
            mesh = abs_mesh
          else:
            ctx_mesh = jax.sharding.get_mesh()
            if ctx_mesh is not None and not ctx_mesh.empty:
              mesh = ctx_mesh
        except Exception:  # pylint: disable=broad-exception-caught
          pass

      shd_b, shd_t, shd_n, shd_h = self.shd_config.act_btnh
      if mesh is None or not mesh.shape:
        shd_b = shd_t = shd_n = shd_h = None
      else:
        if shd_b is not None and (
            shd_b not in mesh.shape or b % mesh.shape[shd_b] != 0
        ):
          shd_b = None
        if shd_t is not None and shd_t not in mesh.shape:
          shd_t = None
        if shd_n is not None and shd_n not in mesh.shape:
          shd_n = None
        if shd_h is not None and shd_h not in mesh.shape:
          shd_h = None

      head_shards = (
          mesh.shape[shd_n]
          if mesh is not None and shd_n is not None and shd_n in mesh.shape
          else 1
      )
      q_seq_shards = (
          mesh.shape[shd_t]
          if mesh is not None and shd_t is not None and shd_t in mesh.shape
          else 1
      )
      key_proj, value_proj, kh = _align_kv_heads_for_splash(
          key_proj, value_proj, qh, head_shards
      )

      bs = self.config.flash_attention_block_size
      if self.config.splash_attention_impl == SplashAttentionImpl.TOKAMAX:
        tokamax_splash, tokamax_mask_lib = _tokamax_splash_libs()
        tokamax_mask = tokamax_mask_lib.make_causal_mask((seq_len, seq_len))
        splash_config = tokamax_splash.SplashConfig(
            block_q=bs,
            block_kv=bs,
            block_kv_compute=bs,
            block_q_dkv=bs,
            block_kv_dkv=bs,
            block_kv_dkv_compute=bs,
            use_base2_exp=False,
        )
        splash_attn_kernel = tokamax_splash.make_splash_mha(
            tokamax_mask,
            config=splash_config,
            q_seq_shards=q_seq_shards,
        )
        kernel_spec = (
            splash_attn_kernel.manual_sharding_spec(
                shd.NamedSharding(mesh, P(shd_t))
            )
            if mesh is not None
            else None
        )
        segment_ids_cls = tokamax_splash.SegmentIds
      else:
        causal_mask = mask_lib.CausalMask((seq_len, seq_len))
        multi_head_mask = mask_lib.MultiHeadMask(
            [causal_mask for _ in range(qh)]
        )
        block_sizes = splash.BlockSizes(
            block_q=bs,
            block_kv=bs,
            block_q_dkv=bs,
            block_kv_dkv=bs,
            block_kv_dkv_compute=bs,
            block_q_dq=bs,
            block_kv_dq=bs,
        )
        splash_attn_kernel = splash.make_splash_mha(
            multi_head_mask,
            block_sizes=block_sizes,
            head_shards=head_shards,
            q_seq_shards=q_seq_shards,
        )
        kernel_spec = (
            splash_attn_kernel.manual_sharding_spec(
                shd.NamedSharding(mesh, P(shd_n, shd_t))
            )
            if mesh is not None
            else None
        )
        segment_ids_cls = splash.SegmentIds

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

      if effective_segment_ids is not None:
        seg_spec = P(shd_b, shd_t)
        unsharded_seg_spec = P(shd_b, None)

        @functools.partial(
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
          seg_ids = segment_ids_cls(q=q_seg_block, kv=kv_seg_block)
          return jax.vmap(kernel)(
              q_block, k_block, v_block, segment_ids=seg_ids
          )

        attn_vec = sharded_splash_attn(
            splash_attn_kernel,
            query_proj,
            key_proj,
            value_proj,
            effective_segment_ids,
            effective_segment_ids,
        )
      else:

        @functools.partial(
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

        attn_vec = sharded_splash_attn(
            splash_attn_kernel, query_proj, key_proj, value_proj
        )
      attn_vec = attn_vec.transpose(0, 2, 1, 3)
    else:
      s = cache_key_positions.shape[1]
      g = qh // kh

      q = query_positions.reshape((b, t, kh, g, d))
      scores = (
          jnp.einsum(
              'btkgh,bskh->bkgts',
              q.astype(jnp.float32),
              cache_key_positions.astype(jnp.float32),
          )
          * self.scale
      )
      scores = scores.reshape((b, qh, t, s))

      if attn_mask is not None:
        if attn_mask.ndim == 3:
          attn_mask = jnp.expand_dims(attn_mask, axis=1)
        scores = jnp.where(attn_mask, scores, K_MASK)
      elif cache is None:
        causal_mask = jnp.tril(jnp.ones((t, s), dtype=jnp.bool_))
        scores = jnp.where(causal_mask[None, None, :, :], scores, K_MASK)

      if segment_ids is not None:
        seg_mask = segment_ids[:, :, None] == segment_ids[:, None, :]
        scores = jnp.where(seg_mask[:, None, :, :], scores, K_MASK)

      probs = jax.nn.softmax(scores, axis=-1).astype(
          cache_value_positions.dtype
      )
      probs = probs.reshape((b, kh, g, t, s))
      attn_vec = jnp.einsum('bkgts,bskh->btkgh', probs, cache_value_positions)
      attn_vec = attn_vec.reshape((b, t, qh, d))

    if gate is not None:
      attn_vec = (
          attn_vec.astype(jnp.float32)
          * jax.nn.sigmoid(gate.astype(jnp.float32))
      ).astype(self.config.dtype)

    attn_vec = shard(attn_vec, self.shd_config.act_btnh)
    output = self.o_proj(attn_vec)
    output = shard(output, self.shd_config.act_btd)

    if cache is not None:
      new_cache = {
          'v': cache_value_positions,
          'k': cache_key_positions,
          'end_index': cache['end_index'] + seq_len,
      }
    else:
      new_cache = None

    return new_cache, output

  @jax.named_scope('attention')
  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    if (
        self.config.remat_config == RematConfig.BLOCK
        or self.config.remat_config == RematConfig.BLOCK.value
    ) and cache is None:
      graphdef, state = nnx.split(self)

      def _checkpointed_block(state, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        return module.block(*args, **kwargs)

      return jax.checkpoint(_checkpointed_block)(
          state, x, segment_pos, cache, attn_mask, segment_ids=segment_ids
      )
    return self.block(x, segment_pos, cache, attn_mask, segment_ids=segment_ids)

  @property
  def head_dim(self) -> int:
    return self.o_proj.shape[1]

  @property
  def num_heads(self) -> int:
    return self.q_proj.shape[1]

  @property
  def num_kv_heads(self) -> int:
    return self.k_proj.shape[1]


class MLP(nnx.Module):
  """SwiGLU MLP module."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.shd_config = config.shd_config
    self.dtype = config.dtype
    kernel_init_fn = nnx.initializers.zeros_init()
    self.gate_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_df
        ),
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    self.up_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_df
        ),
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    self.down_proj = nnx.Linear(
        in_features=config.hidden_dim,
        out_features=config.embed_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_fd
        ),
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )

  @jax.named_scope('feed_forward')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    x = jnp.asarray(x, dtype=self.dtype)
    activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
    activations = shard(activations, self.shd_config.act_btf)
    outputs = self.down_proj(activations)
    return outputs


class DecoderLayer(nnx.Module):
  """Qwen3.5 hybrid decoder layer (GatedDeltaNet or Attention + MLP)."""

  def __init__(
      self,
      config: ModelConfig,
      layer_type: LayerType,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.layer_type = layer_type
    self.input_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        use_norm_offset=config.use_norm_offset,
        rngs=rngs,
        shd_config=config.shd_config,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    self.post_attention_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        use_norm_offset=config.use_norm_offset,
        rngs=rngs,
        shd_config=config.shd_config,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    if layer_type == LayerType.FULL_ATTENTION:
      self.attn = Attention(config=config, rngs=rngs)
    elif layer_type == LayerType.LINEAR_ATTENTION:
      self.linear_attn = GatedDeltaNet(config=config, rngs=rngs)
    else:
      raise ValueError(f'Unsupported layer_type: {layer_type}')

    if config.num_experts is not None:
      raise NotImplementedError(
          'MoE layers (num_experts != None) are not yet supported in native '
          'Qwen3p5; use ModelSource.MAXTEXT.'
      )
    self.mlp = MLP(config=config, rngs=rngs)

  def block(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    inputs_normalized = self.input_layernorm(x)
    if self.layer_type == LayerType.FULL_ATTENTION:
      cache, attn_output = self.attn(
          inputs_normalized,
          segment_pos,
          cache,
          attn_mask,
          segment_ids=segment_ids,
      )
    else:
      cache, attn_output = self.linear_attn(
          inputs_normalized,
          cache,
          attn_mask,
          segment_ids=segment_ids,
      )
    attn_output += x
    residual = attn_output
    attn_output = self.post_attention_layernorm(attn_output)
    outputs = self.mlp(attn_output)
    outputs = residual + outputs
    return cache, outputs

  @jax.named_scope('decoder_layer')
  def __call__(
      self,
      x: jaxtyping.Array,
      segment_pos: jaxtyping.Array,
      cache: LayerCache | None,
      attn_mask: jaxtyping.Array | None,
      segment_ids: jaxtyping.Array | None = None,
  ) -> tuple[LayerCache | None, jaxtyping.Array]:
    if (
        self.config.remat_config == RematConfig.DECODER
        or self.config.remat_config == RematConfig.DECODER.value
    ) and cache is None:
      graphdef, state = nnx.split(self)

      def _checkpointed_block(state, *args, **kwargs):
        module = nnx.merge(graphdef, state)
        return module.block(*args, **kwargs)

      return jax.checkpoint(_checkpointed_block)(
          state, x, segment_pos, cache, attn_mask, segment_ids=segment_ids
      )
    return self.block(x, segment_pos, cache, attn_mask, segment_ids=segment_ids)


class Qwen3p5(BackendMappingMixin, nnx.Module):
  """Qwen3.5 hybrid language model."""

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
        shd_config=config.shd_config,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    assert config.layer_types is not None
    self.layers = compat.ModuleList([
        DecoderLayer(
            config=config,
            layer_type=config.layer_types[i],
            rngs=rngs,
        )
        for i in range(config.num_layers)
    ])
    self.final_norm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        use_norm_offset=config.use_norm_offset,
        rngs=rngs,
        shd_config=config.shd_config,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    if not config.use_tied_embedding:
      self.lm_head = Einsum(
          einsum_str='BTD,DV->BTV',
          shape=(config.embed_dim, config.vocab_size),
          rngs=rngs,
          sharding=config.shd_config.emb_dv,
          param_dtype=config.param_dtype,
          dtype=config.dtype,
      )

  def init_cache(
      self, batch_size: int, cache_size: int, dtype: jnp.dtype
  ) -> Cache:
    """Initializes hybrid cache (KV for full_attention, state for GDN)."""
    config = self.config
    assert config.layer_types is not None
    kv_shape = (batch_size, cache_size, config.num_kv_heads, config.head_dim)
    conv_dim = (
        config.linear_num_key_heads * config.linear_key_head_dim * 2
        + config.linear_num_value_heads * config.linear_value_head_dim
    )
    conv_shape = (batch_size, config.linear_conv_kernel_dim - 1, conv_dim)
    recurrent_shape = (
        batch_size,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
    )
    cache: Cache = {}
    for i, layer_type in enumerate(config.layer_types):
      if layer_type == LayerType.FULL_ATTENTION:
        cache[f'layer_{i}'] = {
            'k': jnp.zeros(kv_shape, dtype=dtype),
            'v': jnp.zeros(kv_shape, dtype=dtype),
            'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
        }
      else:
        cache[f'layer_{i}'] = {
            'conv_state': jnp.zeros(conv_shape, dtype=dtype),
            'recurrent_state': jnp.zeros(recurrent_shape, dtype=jnp.float32),
            'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
        }
    return cache

  @jax.named_scope('qwen3p5')
  def __call__(
      self,
      input_tokens: jaxtyping.Array,  # [B, L]
      positions: jaxtyping.Array,  # [B, L]
      cache: Cache | None = None,
      attention_mask: jaxtyping.Array | None = None,
      output_hidden_states: bool = False,
      segment_ids: jaxtyping.Array | None = None,
      skip_lm_head: bool = False,
  ) -> tuple[jaxtyping.Array, Cache | None]:
    """Qwen3.5 model forward pass."""
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
          segment_ids=segment_ids,
      )
      if new_cache is not None and layer_cache is not None:
        new_cache[layer_name] = layer_cache

    x = self.final_norm(x)
    if output_hidden_states:
      self.sow(nnx.Intermediate, 'all_hidden_states', x)

    if skip_lm_head:
      return x, new_cache

    logits = self.compute_final_logits(x)

    return logits, new_cache

  def compute_final_logits(
      self,
      x: jaxtyping.Array,
  ) -> jaxtyping.Array:
    """Computes the final logits from the model output."""
    if self.config.use_tied_embedding:
      logits = self.embedder.decode(x)
    else:
      logits = self.lm_head(x)
    return jnp.astype(logits, jnp.float32)

  def get_model_input(self):
    """Returns a dummy model input for the transformer."""
    dummy_batch_size = 2
    dummy_seq_len = 128
    return {
        'input_tokens': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'positions': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'cache': None,
        'attention_mask': jnp.ones(
            (dummy_batch_size, dummy_seq_len, dummy_seq_len), dtype=jnp.bool
        ),
    }

  @property
  def num_embed(self) -> int:
    return self.config.vocab_size
