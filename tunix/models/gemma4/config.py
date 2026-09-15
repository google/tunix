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

"""Gemma4 model configuration."""

import dataclasses
import enum
from typing import Any, Tuple
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P
import jaxtyping
from tunix.models import cache_utils
from tunix.models.gemma4 import audio
from tunix.models.gemma4 import vision
from tunix.utils import env_utils

IMAGE_SOFT_TOKEN_PLACEHOLDER = -2
AUDIO_SOFT_TOKEN_PLACEHOLDER = -4


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


@dataclasses.dataclass(frozen=True)
class PreprocessedAudioInput:
  """PyTree container for audio input.

  Attributes:
      audios: waveforms with shape (batch_size, num_clips, samples)
      sequence_lengths: shape (batch_size, num_clips)
  """

  audios: jax.Array
  sequence_lengths: jax.Array


jax.tree_util.register_dataclass(PreprocessedAudioInput)

env_utils.setup_sharding_environment()


LayerCache = dict[str, jaxtyping.Array]
Cache = dict[str, LayerCache]


class RematConfig(enum.Enum):
  NONE = enum.auto()
  BLOCK = enum.auto()
  DECODER = enum.auto()


class SplashAttentionImpl(enum.Enum):
  """Backend implementation to use for splash (flash) attention.

  JAX: `jax.experimental.pallas.ops.tpu.splash_attention`.
  TOKAMAX: `tokamax._src.ops.experimental.tpu.splash_attention`.
  """

  JAX = 'jax'
  TOKAMAX = 'tokamax'


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  """Sharding configuration for gemma transformer."""

  emb_vd: Tuple[str | None, ...] | P
  q_weight_ndh: Tuple[str | None, ...] | P
  kv_weight_cndh: Tuple[str | None, ...] | P
  qkv_weight_cndh: Tuple[str | None, ...] | P
  o_weight_nhd: Tuple[str | None, ...] | P
  ffw_weight_df: Tuple[str | None, ...] | P
  ffw_weight_fd: Tuple[str | None, ...] | P
  rms_norm_weight: Tuple[str | None, ...] | P
  act_btd: Tuple[str | None, ...] | P
  act_btf: Tuple[str | None, ...] | P
  act_btnh: Tuple[str | None, ...] | P
  vision_proj: Tuple[str | None, ...] | P
  vision_soft_emb_norm_weight: Tuple[str | None, ...] | P
  audio_proj: Tuple[str | None, ...] | P
  # MoE sharding
  exp_weight_edf: Tuple[str | None, ...] | P
  exp_weight_efd: Tuple[str | None, ...] | P
  # PLE sharding
  per_layer_model_projection: Tuple[str | None, ...] | P
  per_layer_input_gate: Tuple[str | None, ...] | P
  per_layer_projection: Tuple[str | None, ...] | P
  per_layer_input_embedding: Tuple[str | None, ...] | P
  vision_shd: vision.VisionShardingConfig | None = None
  # Critic score sharding
  score_weight_d1: Tuple[str | None, ...] | P | None = None

  @staticmethod
  def get_default_sharding(is_sampling: bool = False):
    fsdp = 'fsdp' if not is_sampling else None

    return ShardingConfig(
        emb_vd=P('tp', fsdp),
        q_weight_ndh=P('tp', fsdp, None),
        kv_weight_cndh=P(None, 'tp', fsdp, None),
        qkv_weight_cndh=P(None, 'tp', fsdp, None),
        o_weight_nhd=P('tp', None, fsdp),
        ffw_weight_df=P(fsdp, 'tp'),
        ffw_weight_fd=P('tp', fsdp),
        rms_norm_weight=P('tp'),
        act_btd=P('fsdp', None, None if is_sampling else 'tp'),
        act_btf=P('fsdp', None, 'tp'),
        act_btnh=P('fsdp', None, 'tp', None),
        score_weight_d1=P(fsdp, None),
        vision_proj=P(fsdp, 'tp'),
        vision_soft_emb_norm_weight=P('tp'),
        audio_proj=P(fsdp, 'tp'),  # TODO check if good!
        exp_weight_edf=P(fsdp, None, None, 'tp'),
        exp_weight_efd=P(fsdp, 'tp', None),
        per_layer_model_projection=P(fsdp, 'tp'),
        per_layer_input_gate=P(fsdp, 'tp'),
        per_layer_projection=P('tp', fsdp),
        per_layer_input_embedding=P('tp', fsdp),
        vision_shd=vision.VisionShardingConfig.get_default_sharding(
            is_sampling
        ),
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
  # Backend implementation for splash (flash) attention when
  # `use_flash_attention` is True.
  splash_attention_impl: SplashAttentionImpl = SplashAttentionImpl.JAX
  flash_attention_compute_block_size: int = 256
  # Backward needs more VMEM/tile than forward; prod uses 256 (SPLASH_BLOCK_SIZES in
  # //depot/GOOGLE_INTERNAL_PACKAGE_PATH/learning/gemini/prod/serving/jet_engine/gemma4/config_utils.py).
  flash_attention_bwd_block_size: int = 256
  use_sliding_window_kv_cache: bool = False

  # Remat checkpoint policy name from jax.checkpoint_policies; controls which
  # activations are saved in fwd vs recomputed in bwd. Default recomputes
  # everything (minimum HBM).
  remat_policy: str = 'nothing_saveable'

  # Prefix-length bucket ladder for chunked prefill. Each distinct bucketed
  # prefix_length is baked as a static partial arg, so it is a separate XLA
  # compilation; this ladder bounds the number of compiles. An empty tuple
  # disables bucketing (passthrough, accepts recompiles). Use pow2_buckets() /
  # linear_buckets() to build common ladders. Must be sorted and non-negative.
  prefix_bucket_boundaries: tuple[int, ...] = cache_utils.linear_buckets(
      step=512
  )

  # When True, the splash attention backward pass uses a single fused kernel
  # for dQ+dKV instead of two separate passes, reducing VMEM round-trips.
  # When enabled, block_q_dq and block_kv_dq are ignored (set to None).
  flash_attention_use_fused_bwd: bool = False

  # When True and flash attention is enabled, chunked prefill uses split
  # attention with logsumexp merge instead of concatenating prefix+suffix KV,
  # eliminating the jnp.concatenate copy. Requires save_residuals=True from JAX
  # splash attention.
  use_split_attention: bool = False

  # MoE config
  enable_moe: bool = False
  num_experts: int | None = None
  num_experts_per_tok: int | None = None
  expert_dim: int | None = None
  moe_dense_hidden_dim: int | None = None

  # Vision config
  vision_encoder: vision.VisionEncoderConfig | None = None
  use_bidirectional_attention: str | None = None

  # Audio config
  audio_encoder: audio.ConformerConfig | None = None

  def __post_init__(self):
    boundaries = self.prefix_bucket_boundaries
    if any(b < 0 for b in boundaries) or boundaries != tuple(
        sorted(set(boundaries))
    ):
      raise ValueError(
          'prefix_bucket_boundaries must be non-negative and sorted strictly '
          f'ascending with no duplicates; got {boundaries}'
      )
    if self.use_split_attention and not self.use_flash_attention:
      raise ValueError(
          'use_split_attention=True requires use_flash_attention=True: split '
          'attention optimization requires Splash flash attention kernels.'
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
        audio_encoder=audio.ConformerConfig(),
    )

  @classmethod
  def gemma4_e2b_it(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    return cls.gemma4_e2b(sharding_config=sharding_config)

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
        audio_encoder=audio.ConformerConfig(),
    )

  @classmethod
  def gemma4_e4b_it(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    return cls.gemma4_e4b(sharding_config=sharding_config)

  @classmethod
  def gemma4_12b(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    return cls(
        num_layers=48,
        num_embed=262144,
        embed_dim=3840,
        hidden_dim=3840 * 4,
        num_heads=16,
        head_dim=256,
        num_kv_heads=8,
        num_global_kv_heads=1,
        sliding_window_size=1024,
        shd_config=sharding_config,
        k_eq_v_global=True,
        attention_pattern=GEMMA4_ATTENTION_PATTERN,
    )

  @classmethod
  def gemma4_12b_it(
      cls,
      sharding_config: ShardingConfig = ShardingConfig.get_default_sharding(),
  ) -> 'ModelConfig':
    return cls.gemma4_12b(sharding_config=sharding_config)

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
      attn_type = attention_types[i]
      if (attn_type == AttentionType.GLOBAL and share_global) or (
          attn_type == AttentionType.LOCAL_SLIDING and share_local
      ):
        lender = None
        for j in reversed(range(num_unshared_layers)):
          if attention_types[j] == attn_type:
            lender = j
            break
        if lender is None:
          raise ValueError(
              f'Cannot share KV cache for layer {i} of type {attn_type}: no'
              ' unshared layer of the same type exists in layers'
              f' 0..{num_unshared_layers - 1}. Reduce frac_shared_layers or'
              ' adjust attention_types.'
          )
        kv_cache_sharing_patterns.append(lender)
      else:
        kv_cache_sharing_patterns.append(i)
  return kv_cache_sharing_patterns
