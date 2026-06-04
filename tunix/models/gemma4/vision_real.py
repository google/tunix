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

"""Faithful JAX/nnx port of the REAL Gemma 4 vision tower (`gemma4_vision`).

This is a from-source port of HuggingFace
`transformers.models.gemma4.modeling_gemma4` (the classes `Gemma4Vision*` and
`Gemma4MultimodalEmbedder`), NOT the SigLIP encoder borrowed from Gemma 3.

The real `gemma4_vision` tower (per `google/gemma-4-e2b-it`'s config.json) is a
rotary, gated-MLP, sandwich-norm ViT with:
  * a patch embedder that flattens 3*patch^2 pixels and adds a factored 2D
    learned position embedding (one-hot @ a [2, position_embedding_size, hidden]
    table);
  * multidimensional (2D) RoPE applied per spatial axis;
  * per-head q/k RMSNorm (scaled) and v RMSNorm (unscaled);
  * 4-norm sandwich encoder layers;
  * a spatial average pooler that reduces patches to soft tokens.

WIP STATUS (read before trusting this):
  * Wiring + shapes are smoke-tested on random init (see
    tests/models/gemma4/vision_real_test.py).
  * NUMERICS ARE NOT YET VALIDATED against the real checkpoint. Bit-exact
    parity requires loading the real safetensors weights into both this module
    and HF `Gemma4VisionModel` and comparing activations (needs torch + the
    checkpoint). See docs/gemma4_vision_port.md, "Stage 3: parity".
  * Batched padding handling in the pooler (HF does a dynamic boolean
    `hidden_states[pooler_mask]` gather) is intentionally left to the caller /
    a later stage; here we return a padded `[B, output_length, hidden]` tensor
    plus the validity mask, which is JAX-jit friendly.

Reference dimensions (Gemma4VisionConfig defaults / e2b):
  hidden_size=768, intermediate_size=3072, num_hidden_layers=16,
  num_attention_heads=12, num_key_value_heads=12, head_dim=64,
  patch_size=16, position_embedding_size=10240, pooling_kernel_size=3,
  rope_theta=100.0, hidden_activation="gelu_pytorch_tanh", rms_norm_eps=1e-6.
"""

from __future__ import annotations

import dataclasses

from flax import nnx
import jax
from jax import numpy as jnp
import jaxtyping
from tunix.utils import compat


@dataclasses.dataclass(frozen=True)
class Gemma4VisionConfig:
  """Mirror of HF `Gemma4VisionConfig` (only the fields the tower needs)."""

  hidden_size: int = 768
  intermediate_size: int = 3072
  num_hidden_layers: int = 16
  num_attention_heads: int = 12
  num_key_value_heads: int = 12
  head_dim: int = 64
  rms_norm_eps: float = 1e-6
  patch_size: int = 16
  position_embedding_size: int = 10 * 1024
  pooling_kernel_size: int = 3
  rope_theta: float = 100.0
  # e2b ships with use_clipped_linears=False, so the ClippableLinear input/
  # output clamp buffers are +/-inf and are no-ops. We keep the `.linear.`
  # nesting in the module tree so checkpoint keys map 1:1, but skip the clamp.
  use_clipped_linears: bool = False
  standardize: bool = False
  param_dtype: jnp.dtype = jnp.float32
  dtype: jnp.dtype = jnp.float32


def _gelu_tanh(x: jaxtyping.Array) -> jaxtyping.Array:
  # HF "gelu_pytorch_tanh" == jax.nn.gelu(approximate=True).
  return jax.nn.gelu(x, approximate=True)


class RMSNorm(nnx.Module):
  """HF `Gemma4RMSNorm`: x_fp32 * rsqrt(mean(x^2)+eps) * weight.

  NOTE: unlike Gemma 1/2/3 (which scale by ``1 + weight``), Gemma 4 scales by
  ``weight`` directly (weight is initialised to ones). ``with_scale=False``
  drops the parameter entirely (used by v_norm and the projector pre-norm).
  """

  def __init__(
      self,
      dim: int,
      *,
      eps: float,
      with_scale: bool,
      rngs: nnx.Rngs,
      param_dtype: jnp.dtype,
  ):
    self.eps = eps
    self.with_scale = with_scale
    if with_scale:
      self.scale = nnx.Param(jnp.ones((dim,), dtype=param_dtype))

  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    in_dtype = x.dtype
    x = x.astype(jnp.float32)
    normed = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
    if self.with_scale:
      normed = normed * self.scale.value.astype(jnp.float32)
    return normed.astype(in_dtype)


class ClippableLinear(nnx.Module):
  """HF `Gemma4ClippableLinear`: a bias-free Linear, optionally clamped.

  The module nests an inner ``linear`` so checkpoint keys
  (``...{proj}.linear.weight``) map cleanly onto ``...{proj}.linear.kernel``.
  For e2b (``use_clipped_linears=False``) the clamp is a no-op.
  """

  def __init__(
      self,
      in_features: int,
      out_features: int,
      *,
      rngs: nnx.Rngs,
      param_dtype: jnp.dtype,
      dtype: jnp.dtype,
  ):
    self.linear = nnx.Linear(
        in_features,
        out_features,
        use_bias=False,
        rngs=rngs,
        param_dtype=param_dtype,
        dtype=dtype,
    )

  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    # use_clipped_linears is False for e2b; clamp buffers are +/-inf no-ops and
    # are intentionally not ported here. Revisit if a clipped variant ships.
    return self.linear(x)


# ---------------------------------------------------------------------------
# Rotary embedding (multidimensional / 2D).
# ---------------------------------------------------------------------------
def _rotate_half(x: jaxtyping.Array) -> jaxtyping.Array:
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return jnp.concatenate((-x2, x1), axis=-1)


def _apply_rotary(x, cos, sin, unsqueeze_dim):
  cos = jnp.expand_dims(cos, unsqueeze_dim)
  sin = jnp.expand_dims(sin, unsqueeze_dim)
  return (x * cos) + (_rotate_half(x) * sin)


def apply_multidimensional_rope(x, cos, sin, position_ids, unsqueeze_dim=2):
  """Port of HF `apply_multidimensional_rope`.

  Splits the head_dim into ``ndim`` equal chunks (ndim == positions' last dim,
  i.e. 2 for images) and applies 1D rotary independently to each chunk.

  x: [B, L, N, head_dim]; cos/sin: [B, L, head_dim]; position_ids: [B, L, ndim].
  """
  ndim = position_ids.shape[-1]
  num_input_channels = x.shape[-1]
  per_dim = 2 * (num_input_channels // (2 * ndim))
  if per_dim <= 0:
    raise ValueError(
        f"num_rotated_channels_per_dim must be > 0 (got channels="
        f"{num_input_channels}, ndim={ndim})"
    )
  sizes = [per_dim] * ndim
  bounds = [sum(sizes[:i]) for i in range(1, ndim)]
  x_parts = jnp.split(x, bounds, axis=-1)
  cos_parts = jnp.split(cos, bounds, axis=-1)
  sin_parts = jnp.split(sin, bounds, axis=-1)
  y = [
      _apply_rotary(x_parts[k], cos_parts[k], sin_parts[k], unsqueeze_dim)
      for k in range(ndim)
  ]
  return jnp.concatenate(y, axis=-1)


class VisionRotaryEmbedding(nnx.Module):
  """Port of HF `Gemma4VisionRotaryEmbedding` (default rope).

  inv_freq is computed on ``spatial_dim = head_dim // 2`` and reused for both
  spatial axes; cos/sin for each axis are concatenated to width head_dim.
  """

  def __init__(self, config: Gemma4VisionConfig):
    # Keep only Python scalars so nothing lands in the param tree: inv_freq is a
    # derived constant (absent from the checkpoint) and is recomputed per call.
    self.head_dim = config.head_dim
    self.rope_theta = config.rope_theta

  def _inv_freq(self) -> jaxtyping.Array:
    spatial_dim = self.head_dim // 2
    # 1.0 / base ** (arange(0, spatial_dim, 2)/spatial_dim)
    exps = jnp.arange(0, spatial_dim, 2, dtype=jnp.float32) / spatial_dim
    return jnp.asarray(1.0 / (self.rope_theta**exps))  # [spatial_dim/2]

  def __call__(self, position_ids: jaxtyping.Array):
    # position_ids: [B, L, 2] -> cos/sin: [B, L, head_dim]
    all_cos, all_sin = [], []
    inv = self._inv_freq().astype(jnp.float32)  # [F]
    for i in range(2):
      pos = position_ids[:, :, i].astype(jnp.float32)  # [B, L]
      # outer product -> [B, L, F]
      freqs = pos[:, :, None] * inv[None, None, :]
      emb = jnp.concatenate((freqs, freqs), axis=-1)  # [B, L, 2F]
      all_cos.append(jnp.cos(emb))
      all_sin.append(jnp.sin(emb))
    cos = jnp.concatenate(all_cos, axis=-1)  # [B, L, head_dim]
    sin = jnp.concatenate(all_sin, axis=-1)
    return cos, sin


class VisionAttention(nnx.Module):
  """Port of HF `Gemma4VisionAttention` (bidirectional, scaling=1.0)."""

  def __init__(self, config: Gemma4VisionConfig, *, rngs: nnx.Rngs):
    self.config = config
    h = config.hidden_size
    hd = config.head_dim
    self.num_heads = config.num_attention_heads
    self.num_kv_heads = config.num_key_value_heads
    self.head_dim = hd
    kw = dict(rngs=rngs, param_dtype=config.param_dtype, dtype=config.dtype)
    self.q_proj = ClippableLinear(h, self.num_heads * hd, **kw)
    self.k_proj = ClippableLinear(h, self.num_kv_heads * hd, **kw)
    self.v_proj = ClippableLinear(h, self.num_kv_heads * hd, **kw)
    self.o_proj = ClippableLinear(self.num_heads * hd, h, **kw)
    nkw = dict(
        eps=config.rms_norm_eps, rngs=rngs, param_dtype=config.param_dtype
    )
    self.q_norm = RMSNorm(hd, with_scale=True, **nkw)
    self.k_norm = RMSNorm(hd, with_scale=True, **nkw)
    self.v_norm = RMSNorm(hd, with_scale=False, **nkw)

  def __call__(self, x, cos, sin, position_ids, attn_bias):
    b, l, _ = x.shape
    q = self.q_norm(self.q_proj(x).reshape(b, l, self.num_heads, self.head_dim))
    q = apply_multidimensional_rope(q, cos, sin, position_ids)
    k = self.k_norm(self.k_proj(x).reshape(b, l, self.num_kv_heads, self.head_dim))
    k = apply_multidimensional_rope(k, cos, sin, position_ids)
    v = self.v_norm(self.v_proj(x).reshape(b, l, self.num_kv_heads, self.head_dim))

    # [B, N, L, H]
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))
    n_rep = self.num_heads // self.num_kv_heads
    if n_rep > 1:
      k = jnp.repeat(k, n_rep, axis=1)
      v = jnp.repeat(v, n_rep, axis=1)

    scaling = 1.0  # HF sets self.scaling = 1.0 for the vision tower.
    scores = jnp.einsum("bnqh,bnkh->bnqk", q, k).astype(jnp.float32) * scaling
    if attn_bias is not None:
      scores = scores + attn_bias  # [B, 1, 1, L] additive mask
    weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    out = jnp.einsum("bnqk,bnkh->bnqh", weights, v)
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(b, l, -1)
    return self.o_proj(out)


class VisionMLP(nnx.Module):
  """Port of HF `Gemma4VisionMLP` (gated, gelu_tanh)."""

  def __init__(self, config: Gemma4VisionConfig, *, rngs: nnx.Rngs):
    kw = dict(rngs=rngs, param_dtype=config.param_dtype, dtype=config.dtype)
    self.gate_proj = ClippableLinear(
        config.hidden_size, config.intermediate_size, **kw
    )
    self.up_proj = ClippableLinear(
        config.hidden_size, config.intermediate_size, **kw
    )
    self.down_proj = ClippableLinear(
        config.intermediate_size, config.hidden_size, **kw
    )

  def __call__(self, x):
    return self.down_proj(_gelu_tanh(self.gate_proj(x)) * self.up_proj(x))


class VisionEncoderLayer(nnx.Module):
  """Port of HF `Gemma4VisionEncoderLayer` (4-norm sandwich)."""

  def __init__(self, config: Gemma4VisionConfig, *, rngs: nnx.Rngs):
    nkw = dict(
        eps=config.rms_norm_eps,
        with_scale=True,
        rngs=rngs,
        param_dtype=config.param_dtype,
    )
    self.input_layernorm = RMSNorm(config.hidden_size, **nkw)
    self.post_attention_layernorm = RMSNorm(config.hidden_size, **nkw)
    self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, **nkw)
    self.post_feedforward_layernorm = RMSNorm(config.hidden_size, **nkw)
    self.self_attn = VisionAttention(config, rngs=rngs)
    self.mlp = VisionMLP(config, rngs=rngs)

  def __call__(self, x, cos, sin, position_ids, attn_bias):
    residual = x
    x = self.input_layernorm(x)
    x = self.self_attn(x, cos, sin, position_ids, attn_bias)
    x = self.post_attention_layernorm(x)
    x = residual + x

    residual = x
    x = self.pre_feedforward_layernorm(x)
    x = self.mlp(x)
    x = self.post_feedforward_layernorm(x)
    return residual + x


class VisionPatchEmbedder(nnx.Module):
  """Port of HF `Gemma4VisionPatchEmbedder`.

  pixel_values are pre-flattened patches of shape [B, P, 3*patch^2].
  """

  def __init__(self, config: Gemma4VisionConfig, *, rngs: nnx.Rngs):
    self.config = config
    in_dim = 3 * config.patch_size**2
    self.input_proj = nnx.Linear(
        in_dim,
        config.hidden_size,
        use_bias=False,
        rngs=rngs,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )
    self.position_embedding_table = nnx.Param(
        jnp.ones(
            (2, config.position_embedding_size, config.hidden_size),
            dtype=config.param_dtype,
        )
    )

  def _position_embeddings(self, pixel_position_ids, padding_positions):
    # pixel_position_ids: [B, P, 2] -> [B, P, hidden]
    clamped = jnp.maximum(pixel_position_ids, 0)
    one_hot = jax.nn.one_hot(
        clamped, self.config.position_embedding_size, dtype=jnp.float32
    )  # [B, P, 2, pos_size]
    one_hot = jnp.transpose(one_hot, (0, 2, 1, 3))  # [B, 2, P, pos_size]
    table = self.position_embedding_table.value.astype(jnp.float32)  # [2,pos,h]
    pos = jnp.einsum("baps,ash->baph", one_hot, table)  # [B, 2, P, h]
    pos = jnp.sum(pos, axis=1)  # sum over the 2 spatial axes -> [B, P, h]
    pos = jnp.where(padding_positions[..., None], 0.0, pos)
    return pos

  def __call__(self, pixel_values, pixel_position_ids, padding_positions):
    pixel_values = 2.0 * (pixel_values - 0.5)
    hidden = self.input_proj(pixel_values.astype(self.input_proj.kernel.value.dtype))
    pos = self._position_embeddings(pixel_position_ids, padding_positions)
    return hidden + pos.astype(hidden.dtype)


class VisionEncoder(nnx.Module):
  """Port of HF `Gemma4VisionEncoder`."""

  def __init__(self, config: Gemma4VisionConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.rotary_emb = VisionRotaryEmbedding(config)
    self.layers = compat.ModuleList()
    for _ in range(config.num_hidden_layers):
      self.layers.append(VisionEncoderLayer(config, rngs=rngs))

  def __call__(self, inputs_embeds, valid_mask, pixel_position_ids):
    # valid_mask: [B, P] True for real patches. Build additive bidirectional
    # bias [B, 1, 1, P].
    attn_bias = jnp.where(
        valid_mask[:, None, None, :], 0.0, jnp.finfo(jnp.float32).min
    )
    cos, sin = self.rotary_emb(pixel_position_ids)
    x = inputs_embeds
    for layer in self.layers:
      x = layer(x, cos, sin, pixel_position_ids, attn_bias)
    return x


class VisionPooler(nnx.Module):
  """Port of HF `Gemma4VisionPooler` (2D average pool + sqrt(hidden) scale).

  Returns padded `[B, output_length, hidden]` plus a `[B, output_length]`
  validity mask. (HF additionally gathers valid rows into a ragged tensor; we
  defer that to the merge step to stay jit-friendly.)
  """

  def __init__(self, config: Gemma4VisionConfig):
    self.hidden_size = config.hidden_size
    self.root_hidden_size = config.hidden_size**0.5

  def _avg_pool_by_positions(self, hidden_states, pixel_position_ids, length):
    b, input_seq_len, h = hidden_states.shape
    k = int((input_seq_len // length) ** 0.5)
    k_squared = k**2
    if k_squared * length != input_seq_len:
      raise ValueError(
          f"Cannot pool seq_len={input_seq_len} to {length}: k={k}^2 * "
          f"{length} must equal {input_seq_len}."
      )
    clamped = jnp.maximum(pixel_position_ids, 0)
    max_x = jnp.max(clamped[..., 0], axis=-1, keepdims=True) + 1  # [B,1]
    kernel_idxs = clamped // k  # [B, P, 2]
    kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]  # [B,P]
    weights = jax.nn.one_hot(kernel_idxs, length, dtype=jnp.float32) / k_squared
    output = jnp.einsum("bpl,bph->blh", weights, hidden_states.astype(jnp.float32))
    mask = jnp.logical_not(jnp.all(weights == 0, axis=1))  # [B, length]
    return output.astype(hidden_states.dtype), mask

  def __call__(
      self, hidden_states, pixel_position_ids, padding_positions, output_length
  ):
    if output_length > hidden_states.shape[1]:
      raise ValueError(
          f"Requested {output_length} soft tokens > "
          f"{hidden_states.shape[1]} patches."
      )
    hidden_states = jnp.where(
        padding_positions[..., None], 0.0, hidden_states
    )
    if hidden_states.shape[1] != output_length:
      hidden_states, mask = self._avg_pool_by_positions(
          hidden_states, pixel_position_ids, output_length
      )
    else:
      mask = jnp.logical_not(padding_positions)
    hidden_states = hidden_states * self.root_hidden_size
    return hidden_states, mask


class Gemma4VisionModel(nnx.Module):
  """Port of HF `Gemma4VisionModel` (patch embed -> encoder -> pool)."""

  def __init__(self, config: Gemma4VisionConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.patch_embedder = VisionPatchEmbedder(config, rngs=rngs)
    self.encoder = VisionEncoder(config, rngs=rngs)
    self.pooler = VisionPooler(config)

  def __call__(self, pixel_values, pixel_position_ids):
    pk = self.config.pooling_kernel_size
    output_length = pixel_values.shape[-2] // (pk * pk)
    padding_positions = jnp.all(pixel_position_ids == -1, axis=-1)  # [B, P]
    x = self.patch_embedder(pixel_values, pixel_position_ids, padding_positions)
    x = self.encoder(x, jnp.logical_not(padding_positions), pixel_position_ids)
    hidden_states, mask = self.pooler(
        x, pixel_position_ids, padding_positions, output_length
    )
    return hidden_states, mask


class Gemma4MultimodalEmbedder(nnx.Module):
  """Port of HF `Gemma4MultimodalEmbedder` (the `embed_vision` projector)."""

  def __init__(
      self,
      vision_hidden_size: int,
      text_hidden_size: int,
      *,
      eps: float,
      rngs: nnx.Rngs,
      param_dtype: jnp.dtype = jnp.float32,
      dtype: jnp.dtype = jnp.float32,
  ):
    self.embedding_pre_projection_norm = RMSNorm(
        vision_hidden_size,
        eps=eps,
        with_scale=False,
        rngs=rngs,
        param_dtype=param_dtype,
    )
    self.embedding_projection = nnx.Linear(
        vision_hidden_size,
        text_hidden_size,
        use_bias=False,
        rngs=rngs,
        param_dtype=param_dtype,
        dtype=dtype,
    )

  def __call__(self, soft_tokens):
    return self.embedding_projection(
        self.embedding_pre_projection_norm(soft_tokens)
    )


class Gemma4VisionStack(nnx.Module):
  """Vision tower + ``embed_vision`` projector under the checkpoint's names.

  The submodule names (``vision_tower``, ``embed_vision``) match the real
  checkpoint prefixes, so loading is a plain ``model.`` strip + per-tensor
  transpose. This is also the shape Stage 4 plugs into ``Gemma4``.
  """

  def __init__(
      self,
      config: Gemma4VisionConfig,
      text_hidden_size: int,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.vision_tower = Gemma4VisionModel(config, rngs=rngs)
    self.embed_vision = Gemma4MultimodalEmbedder(
        config.hidden_size,
        text_hidden_size,
        eps=config.rms_norm_eps,
        rngs=rngs,
        param_dtype=config.param_dtype,
        dtype=config.dtype,
    )

  def __call__(self, pixel_values, pixel_position_ids):
    """Returns (soft_tokens_in_text_space [B, S, text_hidden], valid_mask)."""
    soft, mask = self.vision_tower(pixel_values, pixel_position_ids)
    return self.embed_vision(soft), mask
