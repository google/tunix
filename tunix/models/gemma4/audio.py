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

"""Audio encoder implementation for Gemma4."""

import dataclasses
from typing import Optional

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.models.gemma4 import vision as vision_lib
from tunix.utils import compat

ClippedEinsum = vision_lib.ClippedEinsum
RMSNorm = vision_lib.RMSNorm


@dataclasses.dataclass(frozen=True, kw_only=True)
class AudioEncoderConfig:
  """Configuration for the AudioEncoder."""

  num_layers: int = 12
  model_dims: int = 1024
  lm_model_dims: int = 1536
  atten_num_heads: int = 8
  atten_left_context: int = 13
  atten_right_context: int = 0
  conv_kernel_size: int = 5
  gradient_clipping: float = 10_000_000_000.0
  conf_reduction_factor: int = 1
  sample_rate: int = 16000
  audio_seq_length: int = 750
  param_dtype: jnp.dtype = jnp.float32
  compute_dtype: Optional[jnp.dtype] = None


class GemaxMelFilterbank(nnx.Module):
  """Computes Mel-filterbanks from a raw audio waveform."""

  def __init__(
      self,
      sample_rate: int = 16000,
      win_length: int = 320,
      hop_length: int = 160,
      subframe_factor: int = 160,
      n_mels: int = 128,
      f_min: float = 0.0,
      f_max: float = 8000.0,
      num_mel_bins: float = 128.0,
      constant: float = 0.001,
  ):
    self.sample_rate = sample_rate
    self.win_length = win_length
    self.hop_length = hop_length
    self.subframe_factor = subframe_factor
    self.n_mels = n_mels
    self.f_min = f_min
    self.f_max = f_max
    self.num_mel_bins = num_mel_bins
    self.constant = constant

    assert self.win_length > self.hop_length
    self.n_fft = int(2 ** np.ceil(np.log2(self.win_length)))

    # Pre-compute window and mel basis
    self.window = self.hann_window(self.win_length, True, True)
    self.mel_basis = self.linear_to_mel_weight_matrix()[
        jnp.newaxis, :, :
    ].transpose(0, 2, 1)

  def hertz_to_mel(self, freq):
    return 2595.0 * np.log10(1.0 + (freq / 700.0))

  def mel_to_hertz(self, mels):
    return 700.0 * (np.power(10, mels / 2595.0) - 1.0)

  def _create_triangular_filter_bank(self, fft_freqs, filter_freqs):
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(0.0, np.minimum(down_slopes, up_slopes))

  def linear_to_mel_weight_matrix(self) -> jnp.ndarray:
    num_spectrogram_bins = int(self.n_fft / 2) + 1
    nyquist_hertz = self.sample_rate / 2.0
    linear_frequencies = np.linspace(
        0.0, nyquist_hertz, num_spectrogram_bins, dtype=np.float64
    )

    mel_min = self.hertz_to_mel(self.f_min)
    mel_max = self.hertz_to_mel(self.f_max)
    mel_freqs = np.linspace(mel_min, mel_max, int(self.num_mel_bins) + 2)
    filter_freqs = self.mel_to_hertz(mel_freqs)

    mel_weights_matrix = self._create_triangular_filter_bank(
        linear_frequencies, filter_freqs
    )
    return jnp.array(mel_weights_matrix.T.astype(np.float32))

  def hann_window(
      self, window_length: int, periodic: bool, nonzero: bool = False
  ) -> jnp.ndarray:
    if nonzero:
      arg = jnp.pi * 2.0 / window_length
      return 0.5 - (
          0.5
          * jnp.cos(arg * (jnp.arange(window_length, dtype=jnp.float32) + 0.5))
      )
    a = 0.5
    b = 1 - a
    even = 1 - window_length % 2
    n = jnp.asarray(window_length + int(periodic) * even - 1, dtype=jnp.float32)
    count = jnp.arange(window_length, dtype=jnp.float32)
    cos_arg = 2 * jnp.pi * count / n
    hann_values = a - b * jnp.cos(cos_arg)
    return hann_values

  def __call__(self, waveform: jax.Array) -> jax.Array:
    waveform = waveform.reshape(waveform.shape[0], 1, -1)
    assert len(waveform.shape) == 3, "Must be [batch, 1, seq_len]"
    assert waveform.shape[1] == 1, "Must be 1"

    frame_size_for_unfold = self.win_length + 1
    seq_len = waveform.shape[-1]
    num_frames = (seq_len - frame_size_for_unfold) // self.hop_length + 1

    start_indices = (jnp.arange(num_frames) * self.hop_length)[:, jnp.newaxis]
    window_indices = jnp.arange(frame_size_for_unfold)[jnp.newaxis, :]
    indices = start_indices + window_indices

    frames = waveform[:, 0, :][:, indices]
    frames = frames[..., :-1]

    windowed_frames = frames * self.window
    stft_spectrogram = jnp.fft.rfft(windowed_frames, n=self.n_fft)
    spectrogram = jnp.abs(stft_spectrogram)

    batch_size = spectrogram.shape[0]
    mel_basis = jnp.repeat(self.mel_basis, batch_size, axis=0)
    mel_spectrogram = spectrogram @ mel_basis
    mel_spectrogram += self.constant
    mel_spectrogram = jnp.log(mel_spectrogram)
    return mel_spectrogram


class SubSamplingBlock(nnx.Module):
  """Subsampling block (Conv+LN+ReLU) reducing temporal dimension."""

  def __init__(
      self,
      input_features: int,
      output_proj_dim: int,
      *,
      rngs: nnx.Rngs,
      dtype: jnp.dtype = jnp.float32,
  ):
    # conv0: [B, T, F, 1] -> [B, T/2, F/2, 128]
    self.conv0 = nnx.Conv(
        in_features=1,
        out_features=128,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=((1, 1), (1, 1)),
        use_bias=False,
        dtype=dtype,
        rngs=rngs,
    )
    self.norm0 = nnx.LayerNorm(
        num_features=128, use_bias=False, use_scale=True, rngs=rngs
    )

    # conv1: [B, T/2, F/2, 128] -> [B, T/4, F/4, 32]
    self.conv1 = nnx.Conv(
        in_features=128,
        out_features=32,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=((1, 1), (1, 1)),
        use_bias=False,
        dtype=dtype,
        rngs=rngs,
    )
    self.norm1 = nnx.LayerNorm(
        num_features=32, use_bias=False, use_scale=True, rngs=rngs
    )

    # Project collapsed features to output_proj_dim
    collapsed_features = (input_features // 4) * 32
    self.input_proj = nnx.Linear(
        in_features=collapsed_features,
        out_features=output_proj_dim,
        use_bias=False,
        dtype=dtype,
        rngs=rngs,
    )

  def __call__(
      self, x: jax.Array, mask: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    x = jnp.expand_dims(x, -1)  # [batch, time, features, 1]

    x = self.conv0(x)
    mask = mask[:, ::2][:, : x.shape[1]]
    x = self.norm0(x)
    x = jax.nn.relu(x)

    x = self.conv1(x)
    mask = mask[:, ::2][:, : x.shape[1]]
    x = self.norm1(x)
    x = jax.nn.relu(x)

    b, t, f, c = x.shape
    x = jnp.reshape(x, (b, t, f * c))
    x = self.input_proj(x)
    return x, mask


class FFNBlock(nnx.Module):
  """Residual FFN block with RMSNorm."""

  def __init__(
      self,
      config: AudioEncoderConfig,
      ffn_residual_weight: float = 0.5,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.ffn_residual_weight = ffn_residual_weight

    self.pre_layer_norm = RMSNorm(
        dim=config.model_dims,
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )
    self.ffn_layer1 = ClippedEinsum(
        shape=(config.model_dims, config.model_dims * 4),
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )
    self.ffn_layer2 = ClippedEinsum(
        shape=(config.model_dims * 4, config.model_dims),
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )
    self.post_layer_norm = RMSNorm(
        dim=config.model_dims,
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    residual = x
    x = jnp.clip(
        x, -self.config.gradient_clipping, self.config.gradient_clipping
    )

    y = self.pre_layer_norm(x)
    y = self.ffn_layer1("...D,DF->...F", y)
    y = jax.nn.swish(y)
    y = self.ffn_layer2("...D,DF->...F", y)
    y = jnp.clip(
        y, -self.config.gradient_clipping, self.config.gradient_clipping
    )
    y = self.post_layer_norm(y)
    return residual + y * self.ffn_residual_weight


class TransformerXLRelativePositionEmbedding(nnx.Module):
  """Relative position embedding logic from Transformer-XL."""

  def __init__(self, config: AudioEncoderConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.atten_num_heads = config.atten_num_heads
    self.units_per_head = config.model_dims // config.atten_num_heads
    self.model_dims = config.model_dims
    self.atten_left_context = config.atten_left_context
    self.atten_right_context = config.atten_right_context

    assert (
        self.atten_right_context == 0
    ), "Not yet implemented for right context"

    self.pos_proj = nnx.Linear(
        self.model_dims,
        self.model_dims,
        use_bias=False,
        kernel_init=nnx.initializers.glorot_uniform(),
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )

  @staticmethod
  def _get_timing_signal_1d_pos(
      position: jnp.ndarray,
      channels: int,
      min_timescale: float = 1.0,
      max_timescale: float = 1.0e4,
      dtype: jnp.dtype = jnp.float32,
  ) -> jnp.ndarray:
    position = jnp.asarray(position, jnp.float32)
    num_timescales = channels // 2
    log_timescale_increment = jnp.log(
        float(max_timescale) / float(min_timescale)
    ) / max(num_timescales - 1, 1)
    inv_timescales = min_timescale * jnp.exp(
        jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
    )
    scaled_time = (
        position[:, :, jnp.newaxis]
        * inv_timescales[jnp.newaxis, jnp.newaxis, :]
    )
    timing_signal = jnp.concatenate(
        [jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=2
    )
    timing_signal = jnp.pad(timing_signal, [[0, 0], [0, 0], [0, channels % 2]])
    return timing_signal.astype(dtype)

  def __call__(self, queries: jax.Array, keys: jax.Array) -> jax.Array:
    term_ac = jnp.einsum(
        "BuwNH,BucNH->BNuwc",
        queries,
        keys,
        precision="highest",
    )
    b = queries.shape[0]
    u = queries.shape[1]
    w = queries.shape[2]
    c = keys.shape[2]
    n = self.atten_num_heads
    l = max(0, self.atten_left_context - 1)
    r = self.atten_right_context
    assert c == w + l + r

    pos = jnp.arange(l, -r - 1, -1)[jnp.newaxis, :]
    sin_emb = self._get_timing_signal_1d_pos(
        pos,
        self.model_dims,
        min_timescale=1,
        max_timescale=10000,
        dtype=queries.dtype,
    )
    sin_emb = self.pos_proj(sin_emb)
    sin_emb = sin_emb.reshape(
        1, l + r + 1, self.atten_num_heads, self.units_per_head
    )
    sin_emb = jnp.squeeze(sin_emb, 0)

    term_bd = jnp.einsum(
        "BuwNH,FNH->BNuwF",
        queries,
        sin_emb,
        precision="float32",
    )
    term_bd = jnp.pad(
        term_bd,
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, (c + 1) - (l + r + 1)],
        ],
        constant_values=jnp.array(0, dtype=term_bd.dtype),
    )
    term_bd = jnp.reshape(
        term_bd,
        [b, n, u, w * (c + 1)],
    )
    term_bd = term_bd[:, :, :, : w * c]
    term_bd = jnp.reshape(
        term_bd,
        [b, n, u, w, c],
    )
    return term_ac + term_bd


class LocalDotProductAttention(nnx.Module):
  """Local dot-product self-attention with relative position embeddings."""

  block_size: int = 12

  def __init__(
      self,
      config: AudioEncoderConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.atten_num_heads = config.atten_num_heads
    self.units_per_head = config.model_dims // config.atten_num_heads
    self.model_dims = config.model_dims
    self.atten_left_context = config.atten_left_context
    self.atten_right_context = config.atten_right_context
    self.attention_logits_soft_capping = 50.0

    self.query = ClippedEinsum(
        shape=(self.model_dims, self.model_dims),
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )
    self.key = ClippedEinsum(
        shape=(self.model_dims, self.model_dims),
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )
    self.value = ClippedEinsum(
        shape=(self.model_dims, self.model_dims),
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )

    self.per_dim_scale = nnx.Param(
        jnp.ones(
            (self.units_per_head,), dtype=config.param_dtype or jnp.float32
        )
    )
    self.relative_position_embedding = TransformerXLRelativePositionEmbedding(
        config=config,
        rngs=rngs,
    )

  @staticmethod
  def _extract_block_context(
      x: jnp.ndarray,
      block_size: int,
      left_context: int,
      right_context: int,
      padding_val: float | jnp.bool_ = 0.0,
  ) -> jnp.ndarray:
    if block_size < 1:
      raise ValueError(f"{block_size=} must be at least 1.")
    paddings = [(0, 0)] * len(x.shape)
    paddings[1] = (left_context, right_context + block_size - 1)
    x = jnp.pad(x, paddings, constant_values=jnp.asarray(padding_val, x.dtype))

    frame_length = block_size + left_context + right_context
    frame_step = block_size
    num_frames = (x.shape[1] - frame_length) // frame_step + 1

    start_indices = jnp.arange(num_frames) * frame_step
    relative_indices = jnp.arange(frame_length)
    indices = start_indices[:, jnp.newaxis] + relative_indices[jnp.newaxis, :]
    return jnp.take(x, indices, axis=1)

  @staticmethod
  def _convert_to_block(
      x: jnp.ndarray, block_size: int, padding_val: float = 0.0
  ) -> jnp.ndarray:
    shape = x.shape
    b, t = shape[0], shape[1]
    if block_size < 1:
      raise ValueError(f"{block_size=} must be at least 1.")
    num_blocks = (t + block_size - 1) // block_size
    pad_length = num_blocks * block_size - t

    if pad_length > 0:
      paddings = [[0, 0]] * len(shape)
      paddings[1] = [0, pad_length]
      x = jnp.pad(x, paddings, constant_values=jnp.array(padding_val, x.dtype))
    reshaped = jnp.reshape(x, (b, num_blocks, block_size) + shape[2:])
    return reshaped

  @staticmethod
  def ones_matrix_band_part(
      rows: int,
      cols: int,
      num_lower: int,
      num_upper: int,
      out_dtype: jnp.dtype = jnp.float32,
      out_shape: Optional[tuple[int, ...]] = None,
  ) -> jnp.ndarray:
    m = jnp.arange(rows).reshape((rows, 1))
    n = jnp.arange(cols).reshape((1, cols))
    mask_lower = True
    if num_lower >= 0:
      mask_lower = (m - n) <= num_lower
    mask_upper = True
    if num_upper >= 0:
      mask_upper = (n - m) <= num_upper
    band = jnp.logical_and(mask_lower, mask_upper).astype(out_dtype)
    if out_shape:
      band = jnp.reshape(band, out_shape)
    return band

  def __call__(
      self, x: jax.Array, mask: jax.Array, causal_valid_mask: jax.Array
  ) -> jax.Array:
    batch_size, seq_len, _ = x.shape

    q = self.query("...D,DF->...F", x)
    k = self.key("...D,DF->...F", x)
    v = self.value("...D,DF->...F", x)

    q = q.reshape(
        batch_size, seq_len, self.atten_num_heads, self.units_per_head
    ).astype("float32")
    k = k.reshape(
        batch_size, seq_len, self.atten_num_heads, self.units_per_head
    ).astype("float32")
    v = v.reshape(
        batch_size, seq_len, self.atten_num_heads, self.units_per_head
    ).astype("float32")

    r_softplus_0 = 1.442695041
    query_scale = jnp.array(
        r_softplus_0 / jnp.sqrt(self.units_per_head), dtype=q.dtype
    )
    q *= query_scale * jax.nn.softplus(self.per_dim_scale.value.astype(q.dtype))

    key_scale = jnp.array(
        r_softplus_0 * jax.nn.softplus(jnp.ones(())), dtype=k.dtype
    )
    k *= key_scale

    q = q.astype("float32")
    k = k.astype("float32")

    original_query_time = q.shape[1]
    k_context = self._extract_block_context(
        k,
        self.block_size,
        max(0, self.atten_left_context - 1),
        self.atten_right_context,
    )
    q_blocked = self._convert_to_block(q, block_size=self.block_size)

    logits = self.relative_position_embedding(q_blocked, k_context)
    logits = self.attention_logits_soft_capping * jnp.tanh(
        logits / self.attention_logits_soft_capping
    )

    num_query_blocks = q_blocked.shape[1]
    valid_mask_blocked = self._extract_block_context(
        mask,
        self.block_size,
        max(0, self.atten_left_context - 1),
        self.atten_right_context,
        padding_val=jnp.bool_(False),
    )
    valid_mask_blocked = valid_mask_blocked[:, jnp.newaxis, :, jnp.newaxis, :]
    valid_mask_blocked = jnp.logical_and(
        valid_mask_blocked,
        causal_valid_mask[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :],
    )

    logits = jnp.where(
        valid_mask_blocked,
        logits,
        jnp.asarray(-1e9, dtype=logits.dtype),
    )
    probabilities = jax.nn.softmax(logits, axis=-1).astype("float32")

    values_blocks = self._extract_block_context(
        v,
        self.block_size,
        max(0, self.atten_left_context - 1),
        self.atten_right_context,
    )
    context_vectors = jnp.einsum(
        "BNuwc,BucNH->BuwNH",
        probabilities,
        values_blocks.astype("float32"),
        precision="float32",
    )
    context_vectors = jnp.reshape(
        context_vectors,
        [
            batch_size,
            num_query_blocks * self.block_size,
            self.atten_num_heads,
            self.units_per_head,
        ],
    )
    return context_vectors[:, :original_query_time]


class AttentionBlock(nnx.Module):
  """Attention block wrapping the local attention mechanism."""

  def __init__(self, config: AudioEncoderConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.pre_norm = RMSNorm(
        dim=config.model_dims,
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )
    self.self_atten = LocalDotProductAttention(config=config, rngs=rngs)
    self.post = ClippedEinsum(
        shape=(
            config.atten_num_heads,
            config.model_dims // config.atten_num_heads,
            config.model_dims,
        ),
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )
    self.post_norm = RMSNorm(
        dim=config.model_dims,
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )

  def __call__(
      self, x: jax.Array, mask: jax.Array, causal_valid_mask: jax.Array
  ) -> jax.Array:
    residual = x
    x = jnp.clip(
        x, -self.config.gradient_clipping, self.config.gradient_clipping
    )
    y = self.pre_norm(x)
    y = self.self_atten(y, mask, causal_valid_mask)
    y = self.post("...NH,NHD->...D", y)
    y = jnp.clip(
        y, -self.config.gradient_clipping, self.config.gradient_clipping
    )
    y = self.post_norm(y)
    return residual + y


class LightweightConvBlock(nnx.Module):
  """Residual lightweight 1D convolutional block."""

  def __init__(self, config: AudioEncoderConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.ln = RMSNorm(
        dim=config.model_dims,
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )
    self.linear_start = ClippedEinsum(
        shape=(config.model_dims, 2 * config.model_dims),
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )
    self.depthwise_conv1d = nnx.Conv(
        in_features=config.model_dims,
        out_features=config.model_dims,
        kernel_size=(config.conv_kernel_size,),
        strides=(1,),
        padding="CAUSAL",
        feature_group_count=config.model_dims,
        use_bias=False,
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )
    self.conv_norm = RMSNorm(
        dim=config.model_dims,
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )
    self.linear_end = ClippedEinsum(
        shape=(config.model_dims, config.model_dims),
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    residual = x
    y = self.ln(x)
    gated_input = self.linear_start("...D,DF->...F", y)
    y = jax.nn.glu(gated_input)
    y = self.depthwise_conv1d(y)
    y = jnp.clip(
        y, -self.config.gradient_clipping, self.config.gradient_clipping
    )
    y = self.conv_norm(y)
    y = jax.nn.swish(y)
    y = self.linear_end("...D,DF->...F", y)
    return residual + y


class ConformerLayer(nnx.Module):
  """A single layer of the Conformer model."""

  def __init__(
      self, config: AudioEncoderConfig, layer_idx: int = -1, *, rngs: nnx.Rngs
  ):
    self.config = config
    self.layer_idx = layer_idx
    self.fflayer_start = FFNBlock(config, ffn_residual_weight=0.5, rngs=rngs)
    self.trans_atten = AttentionBlock(config, rngs=rngs)
    self.lconv = LightweightConvBlock(config, rngs=rngs)
    self.fflayer_end = FFNBlock(config, ffn_residual_weight=0.5, rngs=rngs)
    self.final_ln = RMSNorm(
        dim=config.model_dims,
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )

  def __call__(
      self, x: jax.Array, mask: jax.Array, causal_valid_mask: jax.Array
  ) -> jax.Array:
    x = self.fflayer_start(x)
    x = self.trans_atten(x, mask, causal_valid_mask)
    validity_mask = mask[:, :, jnp.newaxis].astype(x.dtype)
    x = x * validity_mask
    x = self.lconv(x)
    x = self.fflayer_end(x)
    x = jnp.clip(
        x, -self.config.gradient_clipping, self.config.gradient_clipping
    )
    x = self.final_ln(x)
    return x


class AudioEncoder(nnx.Module):
  """Conformer-based Audio Encoder."""

  def __init__(self, config: AudioEncoderConfig, *, rngs: nnx.Rngs):
    self.config = config
    self.mel_filterbank = GemaxMelFilterbank(
        sample_rate=config.sample_rate,
        win_length=320,
        hop_length=160,
        subframe_factor=160,
        n_mels=128,
        f_min=0.0,
        f_max=8000.0,
        num_mel_bins=128.0,
        constant=0.001,
    )
    self.subsampling = SubSamplingBlock(
        input_features=128,
        output_proj_dim=config.model_dims,
        rngs=rngs,
        dtype=config.compute_dtype or jnp.float32,
    )
    self.blocks = compat.ModuleList([
        ConformerLayer(config, layer_idx=i, rngs=rngs)
        for i in range(config.num_layers)
    ])
    self.output_projection = nnx.Linear(
        config.model_dims,
        config.lm_model_dims,
        use_bias=True,
        dtype=config.compute_dtype or jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )

  def to_float32(self, audio_data: jnp.ndarray):
    if audio_data.dtype == jnp.int16:
      return audio_data.astype(jnp.float32) / 32768.0
    elif audio_data.dtype == jnp.int32:
      return audio_data.astype(jnp.float32) / 2147483648.0
    elif audio_data.dtype == jnp.uint8:
      return (audio_data.astype(jnp.float32) - 128.0) / 128.0
    elif audio_data.dtype in [jnp.float16, jnp.float32]:
      return audio_data.astype(jnp.float32)
    else:
      raise ValueError(f"Unsupported format: {audio_data.dtype}")

  def infer_mask(
      self, x: jnp.ndarray, sequence_lengths: jnp.ndarray, original_seq_len: int
  ) -> jnp.ndarray:
    compressed_seq_len = x.shape[1]
    compression_rate = original_seq_len / compressed_seq_len
    new_sequence_lengths = jnp.floor(
        sequence_lengths / compression_rate
    ).astype(jnp.int32)
    indices = jnp.arange(compressed_seq_len)[jnp.newaxis, :]
    mask = indices < new_sequence_lengths[:, jnp.newaxis]
    return mask

  @staticmethod
  def _compute_causal_valid_mask(config: AudioEncoderConfig):
    chunk_size = LocalDotProductAttention.block_size
    max_future_horizon = config.atten_right_context
    max_past_horizon = max(0, config.atten_left_context - 1)
    context_size = chunk_size + max_past_horizon + max_future_horizon
    upper_diagonal = max_past_horizon + max_future_horizon

    lower_causal_mask = LocalDotProductAttention.ones_matrix_band_part(
        context_size,
        chunk_size,
        num_lower=-1,
        num_upper=0,
        out_dtype=jnp.bool_,
    ).T
    upper_causal_mask = LocalDotProductAttention.ones_matrix_band_part(
        chunk_size,
        context_size,
        num_lower=-1,
        num_upper=upper_diagonal,
        out_dtype=jnp.bool_,
    )
    causal_valid_mask = lower_causal_mask & upper_causal_mask
    return causal_valid_mask

  def __call__(
      self, x: jax.Array, sequence_lengths: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    x = self.to_float32(x)
    original_seq_len = x.shape[-1]

    # Mel Filterbank
    x = self.mel_filterbank(x)

    # Infer mask and apply
    mask = self.infer_mask(x, sequence_lengths, original_seq_len)
    x = jnp.where(mask[:, :, jnp.newaxis], x, 0.0)

    # Subsampling
    x, mask = self.subsampling(x, mask)

    causal_valid_mask = self._compute_causal_valid_mask(self.config)

    # Conformer stack
    for block in self.blocks:
      x = block(x, mask, causal_valid_mask)

    if self.config.conf_reduction_factor > 1:
      x = x[:, :: self.config.conf_reduction_factor]
      mask = mask[:, :: self.config.conf_reduction_factor]

    # Final projection
    x = self.output_projection(x)

    x = jnp.where((~mask)[:, :, jnp.newaxis], 0.0, x)

    return x, mask
