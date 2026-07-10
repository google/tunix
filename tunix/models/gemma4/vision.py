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

"""Vision encoder for Gemma4 using Flax NNX."""

import dataclasses
from typing import Any, Tuple
from flax import nnx
import jax
import jax.numpy as jnp
from tunix.utils import compat
from tunix.utils import sharding_utils


@dataclasses.dataclass(slots=True, frozen=True)
class VisionShardingConfig:
  """Sharding configuration for vision encoder."""

  emb_patch_kernel: Tuple[str | None, ...]
  emb_pos_kernel: Tuple[str | None, ...]

  attn_q_kernel: Tuple[str | None, ...]
  attn_kv_kernel: Tuple[str | None, ...]
  attn_out_kernel: Tuple[str | None, ...]

  ffw_gate_kernel: Tuple[str | None, ...]
  ffw_out_kernel: Tuple[str | None, ...]

  rms_norm_weight: Tuple[str | None, ...]

  # Activation sharding
  act_btd: Tuple[str | None, ...]
  act_btf: Tuple[str | None, ...]
  act_btnh: Tuple[str | None, ...]
  act_bskh: Tuple[str | None, ...]
  act_bkgts: Tuple[str | None, ...]

  @staticmethod
  def get_default_sharding(is_sampling: bool = False):
    fsdp = 'fsdp' if not is_sampling else None
    return VisionShardingConfig(
        emb_patch_kernel=(None, 'tp'),
        emb_pos_kernel=(None, None, 'tp'),
        attn_q_kernel=('tp', fsdp, None),
        attn_kv_kernel=(None, 'tp', fsdp, None),
        attn_out_kernel=('tp', None, fsdp),
        ffw_gate_kernel=(None, fsdp, 'tp'),
        ffw_out_kernel=(fsdp, 'tp'),
        rms_norm_weight=('tp',),
        act_btd=('fsdp', None, None if is_sampling else 'tp'),
        act_btf=('fsdp', None, 'tp'),
        act_btnh=('fsdp', None, 'tp', None),
        act_bskh=('fsdp', None, 'tp', None),
        act_bkgts=('fsdp', 'tp', None, None, None),
    )


POSITIONS_PAD_VALUE = -1
TOKEN_PLACEHOLDER = -2
K_MASK = -2.3819763e38


@dataclasses.dataclass(slots=True)
class VisionEncoderConfig:
  d_model: int = 768
  num_layers: int = 16
  num_heads: int = 12
  ffw_hidden: int = 3072
  patch_size: int = 16
  output_length: int | tuple[int, ...] = 1120
  pos_emb_shape_yx: tuple[int, int] = (10240, 2)
  pooling_kernel_size: int = 3
  use_clipped_linears: bool = False
  standardize_embeddings: bool = False

  @property
  def max_patches(self) -> int:
    output_len = self.output_length
    if isinstance(output_len, tuple):
      output_len = max(output_len)
    return int(output_len * self.pooling_kernel_size**2)

  @property
  def num_mm_tokens_per_image(self) -> int:
    output_len = self.output_length
    if isinstance(output_len, tuple):
      output_len = max(output_len)
    return int(output_len)

  @property
  def image_height(self) -> int:
    side = int(self.max_patches**0.5) * self.patch_size
    return side

  @property
  def image_width(self) -> int:
    side = int(self.max_patches**0.5) * self.patch_size
    return side


def apply_rope(
    inputs: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: int,
    scale_factor: float = 1.0,
    rope_proportion: float = 1.0,
) -> jax.Array:
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


def apply_multidimensional_rope(
    inputs: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: int,
    rotary_fraction: float | None = None,
    scale_factor: float = 1.0,
) -> jax.Array:
  if positions.ndim + 2 == inputs.ndim:
    return apply_rope(
        inputs=inputs,
        positions=positions,
        base_frequency=base_frequency,
        scale_factor=scale_factor,
        rope_proportion=rotary_fraction if rotary_fraction is not None else 1.0,
    )

  ndim = positions.shape[-1]
  num_input_channels = inputs.shape[-1]
  num_rotated_channels = num_input_channels
  if rotary_fraction is not None:
    num_rotated_channels = int(round(num_rotated_channels * rotary_fraction))
  num_rotated_channels_per_dim = 2 * (num_rotated_channels // (2 * ndim))

  assert (
      num_rotated_channels_per_dim > 0
  ), f'Requirement not satisfied: 2 * {ndim=} <= {num_input_channels=}.'

  split_points = [(k + 1) * num_rotated_channels_per_dim for k in range(ndim)]
  if rotary_fraction is None:
    split_points = split_points[:-1]
  x_parts = jnp.split(inputs, split_points, axis=-1)
  y_parts = [
      apply_rope(
          x_parts[k],
          positions=positions[..., k],
          base_frequency=base_frequency,
          scale_factor=scale_factor,
      )
      for k in range(ndim)
  ]

  if rotary_fraction is not None:
    y_parts.append(x_parts[-1])

  return jnp.concatenate(y_parts, axis=-1)


def factorized_posemb(posemb: jax.Array, positions_xy: jax.Array) -> jax.Array:
  """Compute factorized position embedding from (x, y) coordinates."""
  one_hot = jax.nn.one_hot(positions_xy, posemb.shape[0], dtype=posemb.dtype)
  nan = jnp.logical_not(one_hot.any(axis=-1, keepdims=True))
  nan = jnp.logical_and(nan, positions_xy[..., None] != -1)
  pos_oh = jnp.where(nan, jnp.nan, one_hot)
  pe_seq = jnp.einsum('blis,sid->ibld', pos_oh, posemb).astype(posemb.dtype)
  return jnp.sum(pe_seq, axis=0)


def avg_pool_by_positions(
    x: jax.Array,
    positions_xy: jax.Array,
    length: int,
) -> tuple[jax.Array, jax.Array]:
  k = int((x.shape[1] // length) ** 0.5)
  assert k * k * length == x.shape[1], f'Cannot pool {x.shape=} to {length=}'

  max_x = positions_xy[..., 0].max(axis=-1, keepdims=True) + 1
  kernel_idxs = jnp.floor_divide(positions_xy, k)
  flat_kernel_idx = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
  weights = jax.nn.one_hot(flat_kernel_idx, length) / k**2
  output = jnp.einsum('bLl,bLd->bld', weights, x)
  mask = jnp.logical_not((weights == 0).all(axis=1))
  return output, mask


class Einsum(nnx.Module):
  """Simple Einsum layer for vision models."""

  def __init__(
      self,
      shape: tuple[int, ...],
      *,
      rngs: nnx.Rngs,
      dtype: jnp.dtype = jnp.float32,
      param_dtype: jnp.dtype = jnp.float32,
      w_scale: float | None = None,
      initializer: Any = None,
      sharding: tuple[str | None, ...] | None = None,
  ):
    self.dtype = dtype
    self.w_scale = w_scale
    if initializer is None:
      initializer = nnx.initializers.normal()
    self.w = nnx.Param(
        initializer(rngs.params(), shape).astype(param_dtype),
        sharding=sharding,
    )

  def __call__(self, equation: str, x: jax.Array) -> jax.Array:
    w = self.w.value
    if self.w_scale is not None:
      w = w * self.w_scale
    x = jnp.astype(x, self.dtype)
    w = jnp.astype(w, self.dtype)
    return jnp.einsum(equation, x, w)


class ClippedEinsum(nnx.Module):
  """Einsum with input and output activation clamping."""

  def __init__(
      self,
      shape: tuple[int, ...],
      *,
      rngs: nnx.Rngs,
      dtype: jnp.dtype = jnp.float32,
      param_dtype: jnp.dtype = jnp.float32,
      w_scale: float | None = None,
      initializer: Any = None,
      sharding: tuple[str | None, ...] | None = None,
  ):
    self.dtype = dtype
    self.w_scale = w_scale
    if initializer is None:
      initializer = nnx.initializers.normal()
    self.w = nnx.Param(
        initializer(rngs.params(), shape).astype(param_dtype),
        sharding=sharding,
    )
    self.clip_input_min = nnx.Param(jnp.array(-float('inf'), dtype=param_dtype))
    self.clip_input_max = nnx.Param(jnp.array(float('inf'), dtype=param_dtype))
    self.clip_output_min = nnx.Param(
        jnp.array(-float('inf'), dtype=param_dtype)
    )
    self.clip_output_max = nnx.Param(jnp.array(float('inf'), dtype=param_dtype))

  def __call__(self, equation: str, x: jax.Array) -> jax.Array:
    w = self.w.value
    if self.w_scale is not None:
      w = w * self.w_scale
    x = jnp.clip(x, self.clip_input_min.value, self.clip_input_max.value)
    x = jnp.astype(x, self.dtype)
    w = jnp.astype(w, self.dtype)
    x = jnp.einsum(equation, x, w)
    x = jnp.clip(x, self.clip_output_min.value, self.clip_output_max.value)
    return x


class RMSNorm(nnx.Module):
  """RMSNorm layer helper for vision block norms."""

  def __init__(
      self,
      dim: int,
      *,
      rngs: nnx.Rngs,
      dtype: jnp.dtype = jnp.float32,
      param_dtype: jnp.dtype = jnp.float32,
      with_scale: bool = True,
      sharding: tuple[str | None, ...] | None = None,
  ):
    self.with_scale = with_scale
    if with_scale:
      self.scale = nnx.Param(
          jnp.zeros((dim,), dtype=param_dtype),
          sharding=sharding,
      )
    self.dtype = dtype

  def __call__(self, x: jax.Array) -> jax.Array:
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_inputs = x * jax.lax.rsqrt(var + 1e-06)
    if self.with_scale:
      scale = jnp.expand_dims(self.scale.value, axis=range(len(x.shape) - 1))
      normed_inputs = normed_inputs * scale
    return normed_inputs.astype(self.dtype)


class Standardize(nnx.Module):
  """Applies feature-wise standardization: x = (x - bias) * scale."""

  def __init__(
      self,
      dim: int,
      *,
      rngs: nnx.Rngs,
      param_dtype: jnp.dtype = jnp.float32,
      sharding: tuple[str | None, ...] | None = None,
  ):
    self.scale = nnx.Param(jnp.ones((dim,), dtype=param_dtype), sharding=sharding)
    self.bias = nnx.Param(jnp.zeros((dim,), dtype=param_dtype), sharding=sharding)

  def __call__(self, x: jax.Array) -> jax.Array:
    return (x - self.bias.value.astype(x.dtype)) * self.scale.value.astype(
        x.dtype
    )


class Attention(nnx.Module):
  """Vision core self-attention block."""

  def __init__(
      self,
      num_heads: int,
      num_kv_heads: int,
      features: int,
      head_dim: int,
      *,
      rngs: nnx.Rngs,
      rope_base_frequency: int = 10_000,
      rope_scale_factor: float = 1.0,
      use_qk_norm: bool = False,
      use_clipped_linears: bool = False,
      param_dtype: jnp.dtype = jnp.float32,
      shd_config: VisionShardingConfig | None = None,
  ):
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.features = features
    self.head_dim = head_dim
    self.rope_base_frequency = rope_base_frequency
    self.rope_scale_factor = rope_scale_factor
    self.use_qk_norm = use_qk_norm
    self.shd_config = shd_config

    linear_cls = ClippedEinsum if use_clipped_linears else Einsum
    self.attn_vec_einsum = linear_cls(
        shape=(num_heads, head_dim, features),
        rngs=rngs,
        param_dtype=param_dtype,
        sharding=shd_config.attn_out_kernel if shd_config else None,
    )
    self.q_einsum = linear_cls(
        shape=(num_heads, features, head_dim),
        rngs=rngs,
        param_dtype=param_dtype,
        sharding=shd_config.attn_q_kernel if shd_config else None,
    )
    self.kv_einsum = linear_cls(
        shape=(2, num_kv_heads, features, head_dim),
        rngs=rngs,
        param_dtype=param_dtype,
        sharding=shd_config.attn_kv_kernel if shd_config else None,
    )
    self.query_norm = RMSNorm(
        head_dim,
        rngs=rngs,
        param_dtype=param_dtype,
        sharding=shd_config.rms_norm_weight if shd_config else None,
    )
    self.key_norm = RMSNorm(
        head_dim,
        rngs=rngs,
        param_dtype=param_dtype,
        sharding=shd_config.rms_norm_weight if shd_config else None,
    )
    self.value_norm = RMSNorm(
        head_dim, rngs=rngs, param_dtype=param_dtype, with_scale=False
    )

  def __call__(
      self,
      x: jax.Array,
      segment_pos: jax.Array,
      attn_mask: jax.Array | None,
  ) -> jax.Array:
    query_proj = self.q_einsum('BTD,NDH->BTNH', x)
    key_proj, value_proj = self.kv_einsum('BSD,CKDH->CBSKH', x)

    if self.use_qk_norm:
      query_proj = self.query_norm(query_proj)
      key_proj = self.key_norm(key_proj)

    value_proj = self.value_norm(value_proj)

    if self.shd_config:
      query_proj = sharding_utils.shard(query_proj, self.shd_config.act_btnh)  # pyrefly: ignore[bad-argument-type]
      key_proj = sharding_utils.shard(key_proj, self.shd_config.act_bskh)  # pyrefly: ignore[bad-argument-type]
      value_proj = sharding_utils.shard(value_proj, self.shd_config.act_bskh)  # pyrefly: ignore[bad-argument-type]

    query_proj = apply_multidimensional_rope(
        query_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
    )
    key_proj = apply_multidimensional_rope(
        key_proj,
        segment_pos,
        base_frequency=self.rope_base_frequency,
        scale_factor=self.rope_scale_factor,
    )

    attn_vec = self._compute_attn_vec(
        query_proj, key_proj, value_proj, attn_mask
    )

    if self.shd_config:
      attn_vec = sharding_utils.shard(attn_vec, self.shd_config.act_btnh)  # pyrefly: ignore[bad-argument-type]

    attn_output = self.attn_vec_einsum('BTNH,NHD->BTD', attn_vec)
    if self.shd_config:
      attn_output = sharding_utils.shard(attn_output, self.shd_config.act_btd)  # pyrefly: ignore[bad-argument-type]
    return attn_output

  def _compute_attn_vec(
      self,
      q: jnp.ndarray,
      k: jnp.ndarray,
      v: jnp.ndarray,
      attn_mask: jnp.ndarray | None = None,
  ) -> jnp.ndarray:
    b, q_len, _, h = q.shape
    q = q.reshape(
        b, q_len, self.num_kv_heads, self.num_heads // self.num_kv_heads, h
    )
    out = self._qkv(q=q, k=k, v=v, attn_mask=attn_mask)
    return out

  def _qkv(
      self,
      q: jnp.ndarray,
      k: jnp.ndarray,
      v: jnp.ndarray,
      attn_mask: jnp.ndarray | None,
  ) -> jnp.ndarray:
    b, q_len, _, _, h = q.shape
    num_heads = q.shape[2] * q.shape[3]

    attn_logits = jnp.einsum('btkgh,bskh->bkgts', q, k)

    if attn_mask is not None:
      if attn_mask.dtype == jnp.bool_:
        attn_logits = jnp.where(
            attn_mask[:, None, None, :, :], attn_logits, K_MASK
        )
      else:
        attn_logits += attn_mask[:, None, None, :, :]

    if self.shd_config:
      attn_logits = sharding_utils.shard(attn_logits, self.shd_config.act_bkgts)  # pyrefly: ignore[bad-argument-type]

    attn_weights = jax.nn.softmax(attn_logits, axis=-1).astype(v.dtype)
    result = jnp.einsum('bkgts,bskh->btkgh', attn_weights, v)
    return result.reshape(b, q_len, num_heads, h)


class FeedForward(nnx.Module):
  """Vision MLP feedforward block."""

  def __init__(
      self,
      features: int,
      hidden_dim: int,
      *,
      rngs: nnx.Rngs,
      use_clipped_linears: bool = False,
      param_dtype: jnp.dtype = jnp.float32,
      shd_config: VisionShardingConfig | None = None,
  ):
    linear_cls = ClippedEinsum if use_clipped_linears else Einsum
    self.gating_einsum = linear_cls(
        shape=(2, hidden_dim, features),
        rngs=rngs,
        param_dtype=param_dtype,
        sharding=shd_config.ffw_gate_kernel if shd_config else None,
    )
    self.linear = linear_cls(
        shape=(hidden_dim, features),
        rngs=rngs,
        param_dtype=param_dtype,
        sharding=shd_config.ffw_out_kernel if shd_config else None,
    )
    self.shd_config = shd_config

  def __call__(self, x: jax.Array) -> jax.Array:
    gate = self.gating_einsum('btd,cfd->btcf', x)
    activations = nnx.gelu(gate[..., 0, :]) * gate[..., 1, :]
    if self.shd_config:
      activations = sharding_utils.shard(activations, self.shd_config.act_btf)  # pyrefly: ignore[bad-argument-type]
    out = self.linear('btf,fd->btd', activations)
    if self.shd_config:
      out = sharding_utils.shard(out, self.shd_config.act_btd)  # pyrefly: ignore[bad-argument-type]
    return out


class VisionBlock(nnx.Module):
  """Vision core transformer block."""

  def __init__(
      self,
      d_model: int,
      ffw_hidden: int,
      num_heads: int,
      num_kv_heads: int,
      key_size: int,
      *,
      rngs: nnx.Rngs,
      use_clipped_linears: bool = False,
      param_dtype: jnp.dtype = jnp.float32,
      shd_config: VisionShardingConfig | None = None,
  ):
    self.pre_attention_norm = RMSNorm(
        d_model,
        rngs=rngs,
        param_dtype=param_dtype,
        sharding=shd_config.rms_norm_weight if shd_config else None,
    )
    self.attn = Attention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        features=d_model,
        head_dim=key_size,
        rngs=rngs,
        rope_base_frequency=100,
        rope_scale_factor=1.0,
        use_qk_norm=True,
        use_clipped_linears=use_clipped_linears,
        param_dtype=param_dtype,
        shd_config=shd_config,
    )
    self.post_attention_norm = RMSNorm(
        d_model,
        rngs=rngs,
        param_dtype=param_dtype,
        sharding=shd_config.rms_norm_weight if shd_config else None,
    )
    self.pre_ffw_norm = RMSNorm(
        d_model,
        rngs=rngs,
        param_dtype=param_dtype,
        sharding=shd_config.rms_norm_weight if shd_config else None,
    )
    self.mlp = FeedForward(
        features=d_model,
        hidden_dim=ffw_hidden,
        rngs=rngs,
        use_clipped_linears=use_clipped_linears,
        param_dtype=param_dtype,
        shd_config=shd_config,
    )
    self.post_ffw_norm = RMSNorm(
        d_model,
        rngs=rngs,
        param_dtype=param_dtype,
        sharding=shd_config.rms_norm_weight if shd_config else None,
    )
    self.shd_config = shd_config

  def __call__(
      self,
      inputs: jax.Array,
      positions: jax.Array,
      attn_mask: jax.Array | None,
  ) -> jax.Array:
    if self.shd_config:
      inputs = sharding_utils.shard(inputs, self.shd_config.act_btd)  # pyrefly: ignore[bad-argument-type]
    normed_inputs = self.pre_attention_norm(inputs)
    if self.shd_config:
      normed_inputs = sharding_utils.shard(
          normed_inputs, self.shd_config.act_btd  # pyrefly: ignore[bad-argument-type]
      )
    attn_output = self.attn(
        x=normed_inputs,
        segment_pos=positions,
        attn_mask=attn_mask,
    )
    attn_output = self.post_attention_norm(attn_output)
    if self.shd_config:
      attn_output = sharding_utils.shard(attn_output, self.shd_config.act_btd)  # pyrefly: ignore[bad-argument-type]
    attn_output += inputs
    outputs = self.pre_ffw_norm(attn_output)
    if self.shd_config:
      outputs = sharding_utils.shard(outputs, self.shd_config.act_btd)  # pyrefly: ignore[bad-argument-type]
    outputs = self.mlp(outputs)
    outputs = self.post_ffw_norm(outputs)
    if self.shd_config:
      outputs = sharding_utils.shard(outputs, self.shd_config.act_btd)  # pyrefly: ignore[bad-argument-type]
    outputs += attn_output
    return outputs


class VisionEntry(nnx.Module):
  """Vision model entrance block."""

  def __init__(
      self,
      d_model: int,
      patch_size: int,
      pos_emb_shape_yx: tuple[int, int],
      *,
      rngs: nnx.Rngs,
      param_dtype: jnp.dtype = jnp.float32,
      shd_config: VisionShardingConfig | None = None,
  ):
    self.patch_size = patch_size
    self.d_model = d_model
    self.pos_emb_shape_yx = pos_emb_shape_yx
    self.shd_config = shd_config

    self.input_projection = Einsum(
        shape=(patch_size * patch_size * 3, d_model),
        rngs=rngs,
        param_dtype=param_dtype,
        sharding=shd_config.emb_patch_kernel if shd_config else None,
    )

    pos_emb_init = nnx.initializers.normal(stddev=0.02)
    self.pos_emb = nnx.Param(
        pos_emb_init(
            rngs.params(), (pos_emb_shape_yx[0], pos_emb_shape_yx[1], d_model)
        ).astype(param_dtype),
        sharding=shd_config.emb_pos_kernel if shd_config else None,
    )

  def __call__(
      self,
      patches: jax.Array,
      positions_xy: jax.Array,
  ) -> jax.Array:
    patches = 2.0 * (patches - 0.5)
    x = self.input_projection('btm,md->btd', patches)
    if self.shd_config:
      x = sharding_utils.shard(x, self.shd_config.act_btd)  # pyrefly: ignore[bad-argument-type]
    pos_embed = factorized_posemb(self.pos_emb.value, positions_xy).astype(
        x.dtype
    )
    if self.shd_config:
      pos_embed = sharding_utils.shard(pos_embed, self.shd_config.act_btd)  # pyrefly: ignore[bad-argument-type]
    out = x + pos_embed
    if self.shd_config:
      out = sharding_utils.shard(out, self.shd_config.act_btd)  # pyrefly: ignore[bad-argument-type]
    return out


class VisionExit(nnx.Module):
  """Vision exit scaling/pooling layer."""

  def __init__(
      self,
      d_model: int,
      output_length: int | tuple[int, ...],
      param_dtype: jnp.dtype = jnp.float32,
  ):
    self.d_model = d_model
    self.output_length = output_length
    self.param_dtype = param_dtype

  def _maybe_downsample(
      self,
      x: jax.Array,
      positions_xy: jax.Array,
      length: int,
  ) -> tuple[jax.Array, jax.Array]:
    cur_length = x.shape[1]
    if cur_length == length:
      mask = jnp.logical_not((positions_xy == POSITIONS_PAD_VALUE).all(axis=-1))
      return x, mask

    x_pooled, mask = avg_pool_by_positions(
        x, positions_xy=positions_xy, length=length
    )
    return x_pooled, mask

  def _single_call(
      self,
      x: jax.Array,
      positions_xy: jax.Array,
      length: int,
  ) -> tuple[jax.Array, jax.Array]:
    x, mask = self._maybe_downsample(
        x, positions_xy=positions_xy, length=length
    )
    x = x * jnp.sqrt(self.d_model)
    return x, mask

  def __call__(
      self,
      x: jax.Array,
      positions_xy: jax.Array,
  ) -> tuple[tuple[jax.Array, jax.Array], ...]:
    lens = self.output_length
    if isinstance(lens, int):
      lens = (lens,)
    finfo = jnp.finfo(x.dtype)
    x = jax.lax.reduce_precision(x, finfo.nexp, finfo.nmant)
    return tuple(
        self._single_call(x, positions_xy=positions_xy, length=length)
        for length in lens
        if length <= x.shape[1]
    )


class VisionEncoder(nnx.Module):
  """Vision encoder for Gemma4 implemented in Flax NNX."""

  def __init__(
      self,
      *,
      rngs: nnx.Rngs,
      config: VisionEncoderConfig,
      param_dtype: jnp.dtype = jnp.float32,
      shd_config: VisionShardingConfig | None = None,
  ):
    self.config = config
    self.shd_config = shd_config

    self.entry = VisionEntry(
        d_model=self.config.d_model,
        patch_size=self.config.patch_size,
        pos_emb_shape_yx=self.config.pos_emb_shape_yx,
        rngs=rngs,
        param_dtype=param_dtype,
        shd_config=shd_config,
    )

    key_size = self.config.d_model // self.config.num_heads
    self.layers = compat.ModuleList([
        VisionBlock(
            d_model=self.config.d_model,
            ffw_hidden=self.config.ffw_hidden,
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_heads,
            key_size=key_size,
            rngs=rngs,
            use_clipped_linears=self.config.use_clipped_linears,
            param_dtype=param_dtype,
            shd_config=shd_config,
        )
        for _ in range(self.config.num_layers)
    ])

    self.exit = VisionExit(
        d_model=self.config.d_model,
        output_length=self.config.output_length,
        param_dtype=param_dtype,
    )
    if self.config.standardize_embeddings:
      self.standardize = Standardize(
          dim=self.config.d_model,
          rngs=rngs,
          param_dtype=param_dtype,
          sharding=shd_config.rms_norm_weight if shd_config else None,
      )

  @property
  def max_patches(self) -> int:
    return self.config.max_patches

  @property
  def num_mm_tokens_per_image(self) -> int:
    return self.config.num_mm_tokens_per_image

  @property
  def image_height(self) -> int:
    return self.config.image_height

  @property
  def image_width(self) -> int:
    return self.config.image_width

  def __call__(
      self,
      patches: jax.Array,
      positions_xy: jax.Array,
  ) -> tuple[tuple[jax.Array, jax.Array | None], ...]:
    input_mask = jnp.logical_not(
        (positions_xy == POSITIONS_PAD_VALUE).all(axis=-1)
    )

    embeddings = self.entry(patches, positions_xy=positions_xy)

    x = embeddings
    attn_mask = input_mask[:, :, None] * input_mask[:, None, :]
    for layer in self.layers:
      x = layer(x, positions_xy, attn_mask)

    outputs = self.exit(
        x,
        positions_xy=positions_xy,
    )

    if self.config.standardize_embeddings:
      standardized_outputs = []
      for emb, mask in outputs:
        dtype = emb.dtype
        emb_std = self.standardize(emb.astype(jnp.float32)).astype(dtype)
        standardized_outputs.append((emb_std, mask))
      outputs = tuple(standardized_outputs)

    if self.shd_config:
      sharded_outputs = []
      for emb, mask in outputs:
        emb = sharding_utils.shard(emb, self.shd_config.act_btd)  # pyrefly: ignore[bad-argument-type]
        sharded_outputs.append((emb, mask))
      outputs = tuple(sharded_outputs)

    return outputs
