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

"""Qwen3-VL model."""

import dataclasses
import enum
from typing import Tuple

import flax
from flax import nnx
import jax
from jax import numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping
import numpy as np
from tunix.generate.mappings import BackendMappingMixin
from tunix.models.qwen3vl.vision import VisionEmbeddings
from tunix.models.qwen3vl.vision import VisionGridData
from tunix.models.qwen3vl.vision import VisionModel
from tunix.models.qwen3vl.vision import VisionModelConfig
from tunix.utils import compat
from tunix.utils import env_utils

env_utils.setup_sharding_environment()


K_MASK = -2.3819763e38

LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


class RematConfig(enum.Enum):
  NONE = enum.auto()  # No remat, all activations will be stored in HBM.
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
  param_dtype: jnp.dtype = jnp.bfloat16
  vision_config: VisionModelConfig | None = None

  @classmethod
  def qwen3vl_4b(cls):  # qwen3-vl-4b
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
        vision_config=VisionModelConfig(
            hidden_size=1024,
            out_hidden_size=2560,
            depth=24,
            num_heads=16,
            intermediate_size=4096,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            window_size=32,
            in_channels=3,
            num_position_embeddings=2304,
            deepstack_visual_indexes=(5, 11, 17),
            mrope_section=(24, 20, 20),
            image_pad_id=151655,
        ),
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
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.einsum_str = einsum_str
    self.shape = shape
    self.w = nnx.Param(
        nnx.initializers.normal(dtype=param_dtype)(rngs.params(), shape),
        sharding=sharding,
    )

  @jax.named_scope('einsum')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    return jnp.einsum(self.einsum_str, x, self.w)


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
    x = shard(x, self.shd_config.act_btd)
    return x

  @jax.named_scope('embedder_decode')
  def decode(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    return jnp.dot(x, self.input_embedding.T)


def get_rope_index(
    input_ids: jax.Array,  # [B, L]
    image_grid_thw: jax.Array | None,  # [N_images, 3]
    video_grid_thw: jax.Array | None,  # [N_videos, 3]
    attention_mask: jax.Array | None,  # [B, L]
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
) -> tuple[jax.Array, jax.Array]:
  """
  Computes 3D M-RoPE position ids for Qwen3-VL.

  Returns:
      position_ids: [3, B, L]  (t, h, w axes)
      mrope_position_deltas: [B, 1]
  """
  # Video frames are treated individually with timestamps —
  # expand each video entry into t separate single-frame entries.
  if video_grid_thw is not None:
    video_grid_thw = np.repeat(
        np.array(video_grid_thw),
        np.array(video_grid_thw)[:, 0],
        axis=0,
    )
    video_grid_thw[:, 0] = 1

  input_ids_np = np.array(input_ids)
  attn_mask_np = (
      np.ones_like(input_ids_np)
      if attention_mask is None
      else np.array(attention_mask)
  )

  image_grid_np = (
      np.array(image_grid_thw) if image_grid_thw is not None else None
  )
  video_grid_np = (
      np.array(video_grid_thw) if video_grid_thw is not None else None
  )

  B, L = input_ids_np.shape
  position_ids = np.zeros((3, B, L), dtype=np.int32)
  mrope_position_deltas = []

  image_index, video_index = 0, 0

  for i in range(B):
    tokens = input_ids_np[i][attn_mask_np[i] == 1]  # [L_i]
    input_tokens = tokens.tolist()

    # Count images and videos in this sequence
    vision_start_indices = np.where(tokens == vision_start_token_id)[0]
    vision_tokens = tokens[vision_start_indices + 1]
    image_nums = int((vision_tokens == image_token_id).sum())
    video_nums = int((vision_tokens == video_token_id).sum())

    llm_pos_ids_list = []
    st = 0
    remain_images, remain_videos = image_nums, video_nums

    for _ in range(image_nums + video_nums):
      # Find next image token position
      try:
        ed_image = (
            input_tokens.index(image_token_id, st)
            if remain_images > 0
            else len(input_tokens) + 1
        )
      except ValueError:
        ed_image = len(input_tokens) + 1
      # Find next video token position
      try:
        ed_video = (
            input_tokens.index(video_token_id, st)
            if remain_videos > 0
            else len(input_tokens) + 1
        )
      except ValueError:
        ed_video = len(input_tokens) + 1

      if ed_image < ed_video:
        t, h, w = image_grid_np[image_index]
        image_index += 1
        remain_images -= 1
        ed = ed_image
      else:
        t, h, w = video_grid_np[video_index]
        video_index += 1
        remain_videos -= 1
        ed = ed_video

      llm_grid_t = int(t)
      llm_grid_h = int(h) // spatial_merge_size
      llm_grid_w = int(w) // spatial_merge_size

      # Text segment before this vision block
      text_len = ed - st
      st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0

      # Text positions: all 3 axes identical (degenerates to 1D RoPE)
      text_pos = np.arange(text_len)[np.newaxis, :] + st_idx  # [1, text_len]
      text_pos = np.broadcast_to(text_pos, (3, text_len))  # [3, text_len]
      llm_pos_ids_list.append(text_pos)

      # Vision positions: 3D meshgrid over (t, h, w)
      t_index = np.tile(
          np.arange(llm_grid_t)[:, np.newaxis], (1, llm_grid_h * llm_grid_w)
      ).flatten()
      h_index = np.tile(
          np.arange(llm_grid_h)[np.newaxis, :, np.newaxis],
          (llm_grid_t, 1, llm_grid_w),
      ).flatten()
      w_index = np.tile(
          np.arange(llm_grid_w)[np.newaxis, np.newaxis, :],
          (llm_grid_t, llm_grid_h, 1),
      ).flatten()

      vis_pos = (
          np.stack([t_index, h_index, w_index]) + text_len + st_idx
      )  # [3, t*h*w]
      llm_pos_ids_list.append(vis_pos)

      st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    # Trailing text after last vision block
    if st < len(input_tokens):
      st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
      text_len = len(input_tokens) - st
      text_pos = np.arange(text_len)[np.newaxis, :] + st_idx
      text_pos = np.broadcast_to(text_pos, (3, text_len))
      llm_pos_ids_list.append(text_pos)

    # Concatenate all segments: [3, L_i]
    llm_positions = np.concatenate(llm_pos_ids_list, axis=1)

    # Write into output, respecting the attention mask
    valid_positions = np.where(attn_mask_np[i] == 1)[0]
    position_ids[:, i, valid_positions] = llm_positions

    mrope_position_deltas.append(int(llm_positions.max()) + 1 - L)

  position_ids_jax = jnp.array(position_ids, dtype=jnp.int32)
  mrope_deltas_jax = jnp.array(mrope_position_deltas, dtype=jnp.int32)[
      :, np.newaxis
  ]

  return position_ids_jax, mrope_deltas_jax


def apply_rope(
    inputs: jaxtyping.Array,  # [B, L, N, H]
    positions: jaxtyping.Array,  # [3, B, L]
    head_dim: int,
    rope_theta: int = 1_000_000,
    mrope_section: tuple[int, ...] = (24, 20, 20),
) -> jaxtyping.Array:
  fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
  timescale = rope_theta**fraction  # [H/2]

  # [3, B, L, H/2]
  sinusoid_inp = (
      positions[:, :, :, jnp.newaxis].astype(jnp.float32)
      / timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
  )

  # Interleaved mRoPE (matches HF apply_interleaved_mrope):
  # T-axis is the base; H and W overwrite their interleaved slots.
  # With mrope_section=(24,20,20) and H/2=64:
  #   H occupies [1,4,...,58], W occupies [2,5,...,59], T keeps the rest.
  freq = sinusoid_inp[0]  # [B, L, H/2] — T axis as base
  h_idx = jnp.arange(1, mrope_section[1] * 3, 3)  # [1, 4, ..., 3*s_h-2]
  w_idx = jnp.arange(2, mrope_section[2] * 3, 3)  # [2, 5, ..., 3*s_w-1]
  freq = freq.at[..., h_idx].set(sinusoid_inp[1][..., h_idx])
  freq = freq.at[..., w_idx].set(sinusoid_inp[2][..., w_idx])

  # [B, L, H/2] -> [B, L, 1, H/2]
  freq = freq[:, :, jnp.newaxis, :]
  sin = jnp.sin(freq)
  cos = jnp.cos(freq)

  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  out = jnp.concatenate(
      [
          first_half * cos - second_half * sin,
          second_half * cos + first_half * sin,
      ],
      axis=-1,
  )
  return out.astype(inputs.dtype)


def make_causal_mask_from_positions(
    text_positions: jax.Array,  # [B, L]
    padding_mask: jax.Array | None = None,  # [B, L] - 1=real token, 0=padding
) -> jax.Array:
  """Creates a position-based causal attention mask for Qwen3-VL.

  Token i can attend to token j iff text_positions[b, j] <= text_positions[b, i].

  For pure text tokens (monotonically increasing positions) this is identical to
  the standard lower-triangular causal mask.  For vision tokens, which all share
  the same T-axis position id P, P <= P is always True, giving full bidirectional
  attention within the image — exactly matching HF create_causal_mask behaviour.

  Args:
    text_positions: Text/temporal axis of the 3D M-RoPE position ids, i.e.
      positions[0] from get_rope_index.  Shape [B, L].
    padding_mask: Optional 2D padding mask (1 = real token, 0 = padding).
      When supplied, padded key positions are blocked from being attended to.

  Returns:
    Boolean mask of shape [B, L, L].  True means the query can attend to the
    key; False means the key is masked out.
  """
  # [B, L, 1] and [B, 1, L] broadcast to [B, L, L]
  query_pos = text_positions[:, :, None]  # [B, L, 1]
  key_pos = text_positions[:, None, :]  # [B, 1, L]
  mask = key_pos <= query_pos  # [B, L, L]
  if padding_mask is not None:
    mask = mask & padding_mask[:, None, :].astype(jnp.bool_)
  return mask


class RMSNorm(nnx.Module):
  """RMSNorm layer."""

  def __init__(
      self,
      dim: int,
      *,
      norm_eps: float = 1e-06,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.w = nnx.Param(
        nnx.initializers.ones_init()(rngs.params(), dim).astype(param_dtype),
        sharding=shd_config.rms_norm_weight,
    )
    self.norm_eps = norm_eps

  @jax.named_scope('rms_norm')
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    dtype = x.dtype
    x_f32 = x.astype(jnp.float32)
    # Use rsqrt (single hardware op) to match PyTorch's
    #   hidden *= torch.rsqrt(variance + eps)
    # Using sqrt+divide gives IEEE-754 exact results but can differ from
    # rsqrt by 1 ULP in float32, which the MLP down_proj amplifies to ~64.
    rms_inv = jax.lax.rsqrt(
        jnp.mean(x_f32**2, axis=-1, keepdims=True) + self.norm_eps
    )
    return self.w * (x_f32 * rms_inv).astype(dtype)


class Attention(nnx.Module):
  """Attention module."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.config = config
    self.shd_config = config.shd_config
    self.q_proj = Einsum(
        einsum_str='BTD,DNH->BTNH',
        shape=(config.embed_dim, config.num_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.q_weight_dnh,
        param_dtype=param_dtype,
    )
    self.k_proj = Einsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
        param_dtype=param_dtype,
    )
    self.v_proj = Einsum(
        einsum_str='BSD,DKH->BSKH',
        shape=(config.embed_dim, config.num_kv_heads, config.head_dim),
        rngs=rngs,
        sharding=self.shd_config.kv_weight_dnh,
        param_dtype=param_dtype,
    )
    self.o_proj = Einsum(
        einsum_str='BTNH,NHD->BTD',
        shape=(config.num_heads, config.head_dim, config.embed_dim),
        rngs=rngs,
        sharding=self.shd_config.o_weight_nhd,
        param_dtype=param_dtype,
    )
    self.q_norm = RMSNorm(
        config.head_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=self.shd_config,
        param_dtype=param_dtype,
    )
    self.k_norm = RMSNorm(
        config.head_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=self.shd_config,
        param_dtype=param_dtype,
    )
    self.n_rep = config.num_heads // config.num_kv_heads
    self.scale = self.head_dim**-0.5

  def block(
      self,
      x: jaxtyping.Array,
      positions: jaxtyping.Array,
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

    mrope_section = None
    if self.config.vision_config is not None:
      mrope_section = self.config.vision_config.mrope_section
    query_proj = apply_rope(
        query_proj,
        positions,
        head_dim=self.head_dim,
        mrope_section=mrope_section,
    )
    key_proj = apply_rope(
        key_proj,
        positions,
        head_dim=self.head_dim,
        mrope_section=mrope_section,
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
    attn = attn.reshape((b, qh, t, s))

    if attn_mask is not None:
      attn = jnp.where((jnp.expand_dims(attn_mask, -3)), attn, K_MASK)

    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(
        key_proj.dtype
    )

    attn = attn.reshape((b, kh, qh // kh, t, s))
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
    if self.config.remat_config == RematConfig.BLOCK:
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


class MLP(nnx.Module):
  """MLP module."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      param_dtype: jnp.dtype = jnp.bfloat16,
  ):
    self.shd_config = config.shd_config
    kernel_init_fn = nnx.initializers.zeros_init()
    self.gate_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_df
        ),
        param_dtype=param_dtype,
    )
    self.up_proj = nnx.Linear(
        in_features=config.embed_dim,
        out_features=config.hidden_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_df
        ),
        param_dtype=param_dtype,
    )
    self.down_proj = nnx.Linear(
        in_features=config.hidden_dim,
        out_features=config.embed_dim,
        use_bias=False,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(
            kernel_init_fn, self.shd_config.ffw_weight_fd
        ),
        param_dtype=param_dtype,
    )

  @jax.named_scope('feed_forward')
  def __call__(self, x: jaxtyping.ArrayLike) -> jaxtyping.Array:
    activations = nnx.silu(self.gate_proj(x)) * self.up_proj(x)
    activations = shard(activations, self.shd_config.act_btf)
    outputs = self.down_proj(activations)
    return outputs


class DecoderLayer(nnx.Module):
  """DecoderLayer."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
      shd_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ):
    self.input_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=shd_config,
        param_dtype=config.param_dtype,
    )
    self.attn = Attention(
        config=config,
        rngs=rngs,
        param_dtype=config.param_dtype,
    )
    self.post_attention_layernorm = RMSNorm(
        config.embed_dim,
        norm_eps=config.norm_eps,
        rngs=rngs,
        shd_config=shd_config,
        param_dtype=config.param_dtype,
    )
    if config.num_experts is None:
      self.mlp = MLP(
          config=config,
          rngs=rngs,
          param_dtype=config.param_dtype,
      )
    else:
      self.mlp = MoELayer(
          config=config,
          rngs=rngs,
          param_dtype=config.param_dtype,
      )

  def __call__(
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


class Qwen3VL(BackendMappingMixin, nnx.Module):
  """Qwen3-VL model."""

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
        param_dtype=config.param_dtype,
    )
    self.layers = compat.ModuleList([
        DecoderLayer(config=config, rngs=rngs, shd_config=shd_config)
        for _ in range(config.num_layers)
    ])
    self.final_norm = RMSNorm(
        config.embed_dim,
        rngs=rngs,
        norm_eps=config.norm_eps,
        shd_config=shd_config,
        param_dtype=config.param_dtype,
    )
    if not config.use_tied_embedding:
      self.lm_head = Einsum(
          einsum_str='BTD,DV->BTV',
          shape=(config.embed_dim, config.vocab_size),
          rngs=rngs,
          sharding=shd_config.emb_dv,
          param_dtype=config.param_dtype,
      )
    self.visual = (
        VisionModel(self.config.vision_config, rngs=rngs)
        if self.config.vision_config
        else None
    )

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

  def __call__(
      self,
      input_tokens: jaxtyping.Array,  # [B, L]
      positions: jaxtyping.Array,  # [3, B, L] - 3D M-RoPE position ids from get_rope_index
      pixel_values: jaxtyping.Array | None,  # [N_tokens, patch_volume]
      vision_precomputed: VisionGridData | None,
      cache: Cache | None,  # (sequence length L')
      attention_mask: (
          jaxtyping.Array | None
      ),  # [B, L] padding mask (1=real, 0=pad) or None
      output_hidden_states: bool = False,
  ) -> tuple[jaxtyping.Array, Cache | None]:
    """Qwen3-VL model.

    Args:
      input_tokens: input sequence of tokens.
      positions: 3D M-RoPE position ids, shape [3, B, L].  Row 0 is the
        text/temporal axis used for causal masking; rows 1-2 are the spatial
        H/W axes used only for RoPE.  Obtained from get_rope_index().
      cache: Attention KV cache or None.
      attention_mask: Optional 2D padding mask of shape [B, L], where 1
        indicates a real token and 0 indicates padding.  A position-based
        causal mask is built internally from positions[0] and this mask.
      output_hidden_states: whether to output the hidden states.

    Returns:
      predicted_logits, new_cache

      predicted_logits: output logits predicted by the model
      new_cache: updated cache if the input cache is not None, None elsewhere.
    """
    new_cache = None if cache is None else {}
    x = self.embedder.encode(input_tokens)
    bsz = x.shape[0]

    # Build the causal attention mask from the text/temporal axis of M-RoPE
    # positions (positions[0]).  Vision tokens all share the same position id,
    # so they obtain full bidirectional attention among themselves while still
    # following causal order with respect to text tokens — matching HF
    # create_causal_mask(position_ids=text_position_ids).
    text_positions = positions[0]  # [B, L]
    causal_mask = make_causal_mask_from_positions(
        text_positions, attention_mask
    )

    vision_embeds = None
    if self.config.vision_config and pixel_values is not None:
      image_pad_id = self.config.vision_config.image_pad_id
      vision_embeds = self.encode_vision(pixel_values, vision_precomputed)
      vision_embeds = vision_embeds.cast(
          self.config.param_dtype
      ).with_batch_dim(bsz)
      # Inject vision tokens at <|image_pad|> positions
      visual_mask = input_tokens == jnp.int32(image_pad_id)

      def _inject(h, tok, vis):
        num_vis = vis.shape[0]
        pos = jnp.where(
            tok == jnp.int32(image_pad_id), size=num_vis, fill_value=-1
        )[0]
        valid = pos >= 0
        pos = jnp.where(valid, pos, 0)
        updates = jnp.where(valid[:, None], vis.astype(h.dtype), h[pos])
        return h.at[pos].set(updates)

      x = jax.vmap(_inject)(x, input_tokens, vision_embeds.tokens)

    deepstack = vision_embeds.deepstack if vision_embeds else ()
    for i, layer in enumerate(self.layers):
      layer_name = f'layer_{i}'
      layer_cache = cache[layer_name] if cache else None
      layer_cache, x = layer(
          x,
          positions,
          layer_cache,
          causal_mask,
      )
      if cache is not None:
        new_cache[layer_name] = layer_cache  # pytype: disable=container-type-mismatch
      if deepstack and i < len(deepstack) and visual_mask is not None:
        x = self._apply_deepstack(x, visual_mask, deepstack[i])

    x = self.final_norm(x)
    if output_hidden_states:
      self.sow(nnx.Intermediate, 'all_hidden_states', x)
    if self.config.use_tied_embedding:
      logits = self.embedder.decode(x)
    else:
      logits = self.lm_head(x)

    return logits, new_cache  # pytype: disable=bad-return-type

  def encode_vision(
      self, pixel_values: jax.Array, precomputed: VisionGridData
  ) -> VisionEmbeddings:
    if self.visual is None:
      raise ValueError('Vision backbone not configured')
    tokens, deepstack = self.visual(pixel_values, precomputed)
    return VisionEmbeddings(tokens=tokens, deepstack=tuple(deepstack))

  @staticmethod
  def _apply_deepstack(
      hidden: jax.Array, visual_mask: jax.Array | None, features: jax.Array
  ) -> jax.Array:
    """Add deepstack vision features to hidden states at vision token positions"""
    if visual_mask is None or features.size == 0:
      return hidden

    def _add(h, mask, feat):
      if feat.shape[0] == 0:
        return h
      idx = jnp.where(mask, size=feat.shape[0], fill_value=-1)[0]
      valid = idx >= 0
      idx = jnp.where(valid, idx, 0)
      updates = jnp.where(
          valid[:, None],
          feat.astype(h.dtype),
          jnp.zeros_like(feat, dtype=h.dtype),
      )
      return h.at[idx].add(updates)

    return jax.vmap(_add)(hidden, visual_mask.astype(bool), features)

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
            (3, dummy_batch_size, dummy_seq_len), dtype=jnp.int32
        ),
        'pixel_values': None,
        'vision_precomputed': None,
        'cache': None,
        'attention_mask': jnp.ones(
            (dummy_batch_size, dummy_seq_len), dtype=jnp.bool
        ),
    }
