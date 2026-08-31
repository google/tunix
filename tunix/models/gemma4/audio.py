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

"""Audio encoder for Gemma4 using Flax NNX.

Ported from the Flax Linen reference implementation in:
https://github.com/google-deepmind/gemma/tree/main/gemma/gm/nn/gemma4/audio.
"""

import dataclasses
import math
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.utils import compat


@dataclasses.dataclass(slots=True)
class ConformerConfig:
  """Configuration for the Conformer model."""

  num_layers: int = 12
  model_dims: int = 1024
  lm_model_dims: int = 1536
  atten_num_heads: int = 8
  atten_left_context: int = 13
  atten_right_context: int = 0
  atten_block_size: int = 12
  conv_kernel_size: int = 5
  gradient_clipping: float = 10_000_000_000.0
  conf_reduction_factor: int = 1
  param_dtype: jnp.dtype = jnp.float32
  compute_dtype: jnp.dtype | None = None


# GemaxMelFilterbank contains no traininable parameters.
# Hence, it's not an nnx.Module. It holds a few (static) arrays, which nnx would
# begin considering part of the model state if we _do_ make an nnx.Module.
class GemaxMelFilterbank:
  """Class to compute Mel-filterbanks from raw audio waveforms."""

  def __init__(
      self,
      sample_rate: int = 16000,
      win_length: int = 320,
      hop_length: int = 160,
      n_mels: int = 128,
      f_min: float = 0,
      f_max: float = 8000,
      num_mel_bins: float = 128,
      constant: float = 0.001,
  ):
    assert win_length > hop_length

    self.sample_rate = sample_rate
    self.win_length = win_length
    self.hop_length = hop_length
    self.n_mels = n_mels
    self.f_min = f_min
    self.f_max = f_max
    self.num_mel_bins = num_mel_bins
    self.constant = constant

    def next_power_of_2(x):
      return int(2 ** int(np.ceil(np.log2(x))))

    self.n_fft = next_power_of_2(self.win_length)
    self.window = self.hann_window(self.win_length, True, True)

    # Pre-compute the Mel filter-bank matrix.
    self.mel_basis = self.linear_to_mel_weight_matrix()[
        np.newaxis, :, :
    ].transpose(0, 2, 1)

  def hertz_to_mel(self, freq):
    """Mel scale used in Gemma 4: htk."""
    return 2595.0 * np.log10(1.0 + (freq / 700.0))

  def mel_to_hertz(self, mels):
    """Mel scale used in Gemma 4: htk."""
    return 700.0 * (np.power(10, mels / 2595.0) - 1.0)

  def _create_triangular_filter_bank(self, fft_freqs, filter_freqs):
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(0.0, np.minimum(down_slopes, up_slopes))

  def linear_to_mel_weight_matrix(self) -> np.ndarray:
    """Inspired from `tf.signal.linear_to_mel_weight_matrix` but allows a custom hertz_to_mel function.

    Returns:
      An array of shape `[num_spectrogram_bins, num_mel_bins]`.
    Raises:
      ValueError: If `num_mel_bins`/`num_spectrogram_bins`/`sample_rate` are
        not positive, `lower_edge_hertz` is negative, frequency edges are
        incorrectly ordered, `upper_edge_hertz` is larger than the Nyquist
        frequency.
    """

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

    return np.array(mel_weights_matrix.T.astype(np.float32))

  def hann_window(
      self,
      window_length: int,
      periodic: bool,
      nonzero: bool = False,
  ) -> np.ndarray:
    """Computes a raised cosine window ported from tf.signal.

    Not using jax.numpy.hanning because it is not periodic.

    Args:
      window_length: The length of the window.
      periodic: Whether the window is periodic.
      nonzero: If True, uses a +0.5 offset so the window never touches zero at
        endpoints (matches HF's hanning_nonzero).

    Returns:
      A `np.ndarray` containing the Hann window.
    """

    if nonzero:
      arg = np.pi * 2.0 / window_length
      return 0.5 - (
          0.5 * np.cos(arg * (np.arange(window_length, dtype=np.float32) + 0.5))
      )

    a = 0.5
    b = 1 - a
    even = 1 - window_length % 2
    n = np.asarray(window_length + int(periodic) * even - 1, dtype=np.float32)
    count = np.arange(window_length, dtype=np.float32)
    cos_arg = 2 * np.pi * count / n
    hann_values = a - b * np.cos(cos_arg)
    return hann_values

  def __call__(self, waveform: jax.Array) -> jax.Array:
    """Compute Mel-filterbank spectrogram from raw waveforms.

    Args:
        waveform: The input audio signal, shape: (batch, samples).

    Returns:
        The Mel-filterbank spectrogram, shape: (batch, frames, n_mels).
    """
    waveform = waveform.reshape(waveform.shape[0], 1, -1)
    assert len(waveform.shape) == 3, 'Must be [batch, 1, seq_len]'
    assert waveform.shape[1] == 1, 'Must be 1'

    frame_size_for_unfold = self.win_length + 1
    seq_len = waveform.shape[-1]
    num_frames = (seq_len - frame_size_for_unfold) // self.hop_length + 1

    start_indices = (jnp.arange(num_frames) * self.hop_length)[:, jnp.newaxis]
    window_indices = jnp.arange(frame_size_for_unfold)[jnp.newaxis, :]
    indices = start_indices + window_indices

    frames = waveform[:, 0, :][:, indices]
    frames = frames[..., :-1]

    # Apply the window function to each frame
    windowed_frames = frames * self.window

    # 3. Short-Time Fourier Transform (STFT)
    # We use rfft for real-valued inputs for efficiency
    stft_spectrogram = jnp.fft.rfft(windowed_frames, n=self.n_fft)

    # 4. Compute Spectrogram instead of power spectrogram
    spectrogram = jnp.abs(stft_spectrogram)

    # 5. Apply the Mel Filterbank to batched audio
    batch_size = spectrogram.shape[0]
    mel_basis = jnp.repeat(self.mel_basis, batch_size, axis=0)
    mel_spectrogram = spectrogram @ mel_basis

    # Adding a constant value
    mel_spectrogram += self.constant

    # Natural log instead of log10
    mel_spectrogram = jnp.log(mel_spectrogram)
    return mel_spectrogram


class SubSamplingBlock(nnx.Module):
  """Subsampling block for the Conformer model."""

  def __init__(
      self,
      *,
      rngs: nnx.Rngs,
      in_features: int = 128,
      out_features: int = 1024,
      dtype: jnp.dtype = jnp.float32,
      param_dtype: jnp.dtype = jnp.float32,
  ):

    self.dtype = dtype
    self.param_dtype = param_dtype

    self.subsampling_0 = nnx.Conv(
        rngs=rngs,
        in_features=1,
        out_features=128,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=((1, 1), (1, 1)),
        use_bias=False,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    self.norm_0 = nnx.LayerNorm(
        rngs=rngs,
        num_features=128,
        use_bias=False,
        use_scale=True,
        use_fast_variance=False,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    self.subsampling_1 = nnx.Conv(
        rngs=rngs,
        in_features=128,
        out_features=32,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding=((1, 1), (1, 1)),
        use_bias=False,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    self.norm_1 = nnx.LayerNorm(
        rngs=rngs,
        num_features=32,
        use_bias=False,
        use_scale=True,
        use_fast_variance=False,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    self.input_proj = nnx.LinearGeneral(
        rngs=rngs,
        in_features=(in_features // 4, 32),
        out_features=out_features,
        axis=(-2, -1),
        use_bias=False,
        dtype=dtype,
        param_dtype=param_dtype,
    )

  def __call__(
      self,
      x: jax.Array,
      mask: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    """Applies two conv+LayerNorm+ReLU subsampling stages and a projection.

    Each convolution stage reduces the time dimension by its stride factor.
    The validity mask is downsampled accordingly. Finally, the frequency and
    channel dimensions are projected to `output_proj_dim`.

    Args:
      x: Input features of shape [batch, time, in_features].
      mask: Boolean validity mask of shape [batch, time], True = valid.

    Returns:
      Tuple of (subsampled_features, subsampled_mask).
      subsampled_features of shape [batch, subsampled_time, out_features]
      subsampled_mask of shape [batch, subsampled_time]
    """

    x = jnp.expand_dims(x, -1)

    x = self.subsampling_0(x)
    mask = mask[:, ::2][:, : x.shape[1]]
    x = self.norm_0(x)
    x = nnx.relu(x)

    x = self.subsampling_1(x)
    mask = mask[:, ::2][:, : x.shape[1]]
    x = self.norm_1(x)
    x = nnx.relu(x)

    x = self.input_proj(x)

    return x, mask


# This is an almost-replica of vision.ClippedEinsum.
# Only difference is attrib name: self.kernel (here) vs self.w (in vision.py).
# Upstream gemma uses the name "kernel" in the AudioTokenizer.
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
    self.kernel = nnx.Param(
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
    w = self.kernel.value
    if self.w_scale is not None:
      w = w * self.w_scale
    x = jnp.clip(x, self.clip_input_min.value, self.clip_input_max.value)
    x = jnp.astype(x, self.dtype)
    w = jnp.astype(w, self.dtype)
    x = jnp.einsum(equation, x, w)
    x = jnp.clip(x, self.clip_output_min.value, self.clip_output_max.value)
    return x


class FFNBlock(nnx.Module):
  """A weighted-residual Feed-Forward Network block."""

  def __init__(
      self,
      num_features: int,
      *,
      rngs: nnx.Rngs,
      ffn_residual_weight: float = 0.5,
      gradient_clipping: float = 10_000_000_000.0,
      dtype: jnp.dtype | None = None,
      param_dtype: jnp.dtype = jnp.float32,
  ):
    self.ffn_residual_weight = ffn_residual_weight
    self.gradient_clipping = gradient_clipping
    self.pre_layer_norm = nnx.RMSNorm(
        rngs=rngs,
        num_features=num_features,
        use_scale=True,
        use_fast_variance=False,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.ffn_layer1 = ClippedEinsum(
        rngs=rngs,
        shape=(num_features, num_features * 4),
        dtype=dtype,  # pyrefly: ignore[bad-argument-type]
        param_dtype=param_dtype,
    )
    self.ffn_layer2 = ClippedEinsum(
        rngs=rngs,
        shape=(num_features * 4, num_features),
        dtype=dtype,  # pyrefly: ignore[bad-argument-type]
        param_dtype=param_dtype,
    )
    self.post_layer_norm = nnx.RMSNorm(
        rngs=rngs,
        num_features=num_features,
        use_scale=True,
        use_fast_variance=False,
        dtype=dtype,
        param_dtype=param_dtype,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    residual = x

    clip_value = self.gradient_clipping
    x = jnp.clip(x, -clip_value, clip_value)
    y = self.pre_layer_norm(x)
    y = self.ffn_layer1('...D,DF->...F', y)
    y = nnx.swish(y)
    y = self.ffn_layer2('...D,DF->...F', y)
    y = jnp.clip(y, -clip_value, clip_value)
    y = self.post_layer_norm(y)
    output = residual + y * self.ffn_residual_weight
    return output


class LightweightConvBlock(nnx.Module):
  """Residual lightweight convolutional block."""

  def __init__(
      self,
      num_features: int,
      kernel_size: int,
      *,
      rngs: nnx.Rngs,
      dtype: jnp.dtype | None = None,
      param_dtype: jnp.dtype = jnp.float32,
      gradient_clipping: float = 10_000_000_000.0,
  ):
    self.gradient_clipping = gradient_clipping
    self.ln = nnx.RMSNorm(
        rngs=rngs,
        num_features=num_features,
        use_fast_variance=False,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.linear_start = ClippedEinsum(
        rngs=rngs,
        shape=(num_features, 2 * num_features),  # feature expansion for GLU.
        dtype=dtype,  # pyrefly: ignore[bad-argument-type]
        param_dtype=param_dtype,
    )
    self.depthwise_conv1d = nnx.Conv(
        rngs=rngs,
        in_features=num_features,
        out_features=num_features,
        kernel_size=(kernel_size,),
        strides=(1,),
        padding='CAUSAL',
        feature_group_count=num_features,
        use_bias=False,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.conv_norm = nnx.RMSNorm(
        num_features=num_features,
        rngs=rngs,
        use_fast_variance=False,
        dtype=dtype,
        param_dtype=param_dtype,
    )
    self.linear_end = ClippedEinsum(
        rngs=rngs,
        shape=(num_features, num_features),
        dtype=dtype,  # pyrefly: ignore[bad-argument-type]
        param_dtype=param_dtype,
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    clip_value = self.gradient_clipping
    residual = x

    y = self.ln(x)
    gated_input = self.linear_start('...D,DF->...F', y)
    y = nnx.glu(gated_input)  # Squeezes input features by 2x.
    y = self.depthwise_conv1d(y)
    y = jnp.clip(y, -clip_value, clip_value)
    y = self.conv_norm(y)
    y = nnx.swish(y)
    y = self.linear_end('...D,DF->...F', y)
    return residual + y


class TransformerXLRelativePositionEmbedding(nnx.Module):
  """Relative position embedding from Transformer-XL."""

  def __init__(
      self,
      *,
      rngs: nnx.Rngs,
      atten_num_heads: int,
      units_per_head: int,
      atten_left_context: int,
      atten_right_context: int = 0,
      param_dtype: jnp.dtype = jnp.float32,
      dtype: jnp.dtype | None = None,
  ):
    assert atten_right_context == 0, 'Not yet implemented for right context'

    self.num_units = atten_num_heads * units_per_head
    self.atten_num_heads = atten_num_heads
    self.atten_left_context = atten_left_context
    self.atten_right_context = atten_right_context

    self.pos_proj = nnx.LinearGeneral(
        rngs=rngs,
        in_features=self.num_units,
        out_features=(atten_num_heads, units_per_head),
        param_dtype=param_dtype,
        dtype=dtype,
        use_bias=False,
        kernel_init=nnx.initializers.glorot_uniform(),
    )

  @staticmethod
  def _get_timing_signal_1d_pos(
      position: jnp.ndarray,
      channels: int,
      min_timescale: float = 1.0,
      max_timescale: float = 1.0e4,
      dtype: jnp.dtype = jnp.float32,
  ) -> jnp.ndarray:
    """Sinusoidal position embeddings with explicit positions."""
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
    """Calculates the relative positional attention scores.

    Args:
        queries: The query tensor, shaped `(batch_size, num_query_blocks,
          block_size, num_heads, units_per_head)`.
        keys: The key tensor, shaped `(batch_size, num_query_blocks,
          context_size, num_heads, units_per_head)`.

    Returns:
        The calculated relative positional attention scores. Shape:
        `(batch_size,
        num_heads, num_query_blocks, block_size, context_size)`.
    """

    # This module calculates the position-based logits (terms B from the paper)
    # to be added to the content-based logits in the attention mechanism.
    #
    # Term (B): Content-dependent position bias (Query @ Relative_Position)
    # Term (D): Position-dependent position bias
    #   (Global_Pos_Bias_v @ Relative_Position)
    #
    # The parent attention layer is responsible for calculating:
    # Term (A): Content-based logits (Query @ Key)
    # Term (C): Content-dependent position bias (Global_Pos_Bias_u @ Key)

    # Compute term_ac.
    term_ac = jnp.einsum(
        'BuwNH,BucNH->BNuwc',
        queries,
        keys,
        precision='highest',
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
    assert pos.shape[1] == l + r + 1

    # [1, F, position_bias_dim]
    sin_emb = self._get_timing_signal_1d_pos(
        pos,
        self.num_units,
        min_timescale=1,
        max_timescale=10000,
        dtype=queries.dtype,
    )
    # [1, F, N, H]
    sin_emb = self.pos_proj(sin_emb)
    # [F, N, H]
    sin_emb = jnp.squeeze(sin_emb, 0)

    # [B, N, U, W, F]
    term_bd = jnp.einsum(
        'BuwNH,FNH->BNuwF',
        queries,
        sin_emb,
        precision='float32',
    )
    # Perform relative shift in order to get [B, N, U, W, C]
    # Pads the input to [B, N, U, W, C + 1]
    term_bd = jnp.pad(
        term_bd, ((0, 0), (0, 0), (0, 0), (0, 0), (0, (c + 1) - (l + r + 1)))
    )
    term_bd = jnp.reshape(term_bd, [b, n, u, w * (c + 1)])
    term_bd = term_bd[:, :, :, : w * c]
    # Reshapes to [B, N, U, W, C]. Note the output last dim is 1-smaller
    # than the input, which "pushses" one element off to the next row for each
    # row. The accumulated effect is row_i is right-shifted i steps (i>=0).
    term_bd = jnp.reshape(term_bd, [b, n, u, w, c])

    return term_ac + term_bd


class LocalDotProductAttention(nnx.Module):
  """Local dot-product self-attention with Transformer-XL relative embeddings."""

  def __init__(
      self,
      model_dims: int,
      atten_num_heads: int,
      units_per_head: int,
      atten_left_context: int,
      *,
      rngs: nnx.Rngs,
      atten_right_context: int = 0,
      attention_logits_soft_capping: float = 50.0,
      block_size: int = 12,
      param_dtype: jnp.dtype = jnp.float32,
      dtype: jnp.dtype | None = None,
  ):

    self.atten_num_heads = atten_num_heads
    self.units_per_head = units_per_head
    self.model_dims = model_dims
    self.atten_left_context = atten_left_context
    self.atten_right_context = atten_right_context
    self.attention_logits_soft_capping = attention_logits_soft_capping
    self.block_size = block_size

    self.query = ClippedEinsum(
        rngs=rngs,
        shape=(model_dims, model_dims),
        param_dtype=param_dtype,
        dtype=dtype,  # pyrefly: ignore[bad-argument-type]
    )
    self.key = ClippedEinsum(
        rngs=rngs,
        shape=(model_dims, model_dims),
        param_dtype=param_dtype,
        dtype=dtype,  # pyrefly: ignore[bad-argument-type]
    )
    self.value = ClippedEinsum(
        rngs=rngs,
        shape=(model_dims, model_dims),
        param_dtype=param_dtype,
        dtype=dtype,  # pyrefly: ignore[bad-argument-type]
    )

    self.per_dim_scale = nnx.Param(jnp.ones(units_per_head, dtype=param_dtype))

    self.relative_position_embedding = TransformerXLRelativePositionEmbedding(
        rngs=rngs,
        atten_num_heads=self.atten_num_heads,
        units_per_head=self.units_per_head,
        atten_left_context=self.atten_left_context,
        param_dtype=param_dtype,
        dtype=dtype,
    )

  @staticmethod
  def _extract_block_context(
      x: jnp.ndarray,
      block_size: int,
      left_context: int,
      right_context: int,
      padding_val: float | jnp.bool_ = 0.0,
  ) -> jnp.ndarray:
    """Extracts temporal context for every block.

    Args:
      x: a tensor of [batch, time, ...].
      block_size: int. Number of time frames in a block.
      left_context: int. Left context size.
      right_context: int. Right context size.
      padding_val: float. value on the padded frames.

    Returns:
      A tensor of [batch, num_blocks, context_size, ...], with necessary
      paddings, where context_size = block_size + left_context + right_context
      and output[:, i, ...] are x[:, start-left_context:end+right_context, ..]
      start = i * block_size, end = (i + 1) * block_size.
    """
    if block_size < 1:
      raise ValueError(f'{block_size=} must be at least 1.')
    if left_context < 0:
      raise ValueError(f'{left_context=} must be >= 0.')
    if right_context < 0:
      raise ValueError(f'{right_context=} must be >= 0.')

    # Pad outside of signal.frame so that we get the desired left/right
    # context and padding behavior.
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
    """Turns a sequence to non overlapping blocks.

    Args:
      x: a tensor of [batch, time, ...].
      block_size: int. Number of time frames in a block.
      padding_val: float. value on the padded frames.

    Returns:
      A tensor of [batch, num_blocks, block_size, ..], with necessary paddings
      where output[:, i, ...] are x[:, i*block_size:(i+1)*block_size, ...].
    """
    shape = x.shape
    b, t = shape[0], shape[1]
    if block_size < 1:
      raise ValueError(f'{block_size=} must be at least 1.')
    # Pad it to be a multiple of w.
    num_blocks = (t + block_size - 1) // block_size
    pad_length = num_blocks * block_size - t

    if pad_length > 0:
      paddings = [[0, 0]] * len(shape)
      paddings[1] = [0, pad_length]
      x = jnp.pad(x, paddings, constant_values=jnp.array(padding_val, x.dtype))
    reshaped = jnp.reshape(x, (b, num_blocks, block_size) + shape[2:])
    return reshaped

  @staticmethod
  def _ones_matrix_band_part(
      rows: int,
      cols: int,
      num_lower: int,
      num_upper: int,
      out_dtype: jnp.dtype = jnp.float32,
      out_shape: tuple[int, ...] | None = None,
  ) -> jnp.ndarray:
    """Matrix band part of ones."""
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
      self,
      x: jax.Array,
      mask: jax.Array,
      causal_valid_mask: jax.Array,
  ) -> jax.Array:
    """Forward pass.

    Args:
      x: shape (batch, seq_len, in_features)
      mask: shape ???
      causal_valid_mask: shape ???

    Returns:
      shape (batch seq_len num_heads units_per_head)
    """

    batch_size, seq_len, _ = x.shape

    # --- 1. Input Projections for Q, K, V ---
    q = self.query('...D,DF->...F', x)
    k = self.key('...D,DF->...F', x)
    v = self.value('...D,DF->...F', x)

    # Reshape and transpose for multi-head computation
    shape = (batch_size, seq_len, self.atten_num_heads, self.units_per_head)
    q = q.reshape(*shape).astype('float32')
    k = k.reshape(*shape).astype('float32')
    v = v.reshape(*shape).astype('float32')

    # Scaling queries with learned scales
    per_dim_scale = self.per_dim_scale.value
    r_softplus_0 = 1.442695041
    query_scale = jnp.array(
        r_softplus_0 / jnp.sqrt(self.units_per_head), dtype=q.dtype
    )
    q *= query_scale * jax.nn.softplus(per_dim_scale.astype(q.dtype))

    key_scale = jnp.array(
        r_softplus_0 * jax.nn.softplus(jnp.ones(())), dtype=k.dtype
    )
    k *= key_scale

    q = q.astype('float32')
    k = k.astype('float32')

    batch_size, original_query_time = q.shape[:2]
    k = self._extract_block_context(
        k,
        self.block_size,
        max(0, self.atten_left_context - 1),
        self.atten_right_context,
    )
    q = self._convert_to_block(q, block_size=self.block_size)

    # Position-based scores (bias)
    logits = self.relative_position_embedding(q, k)

    # Logits capping
    logits = self.attention_logits_soft_capping * jax.nn.tanh(
        logits / self.attention_logits_soft_capping
    )

    num_query_blocks = q.shape[1]

    # Squeeze the heads and query time out to get [b, keys_time].
    # valid_mask_blocked is [b, num_blocks, context_size]
    valid_mask_blocked = self._extract_block_context(
        mask,
        self.block_size,
        max(0, self.atten_left_context - 1),
        self.atten_right_context,
        padding_val=jnp.bool_(False),
    )

    # Reshape to [b, h=1, num_blocks, block_size=1, context_size].
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
    probabilities = jax.nn.softmax(logits, axis=-1).astype('float32')

    # [B, U, C, N, H]
    values_blocks = self._extract_block_context(
        v,
        self.block_size,
        max(0, self.atten_left_context - 1),
        self.atten_right_context,
    )

    # Contract the context windows dimension (c) into per-head context
    # vectors across each local block:
    # [batch, num_query_blocks, block_size, num_heads, units_per_head].
    context_vectors = jnp.einsum(
        'BNuwc,BucNH->BuwNH',
        probabilities,
        values_blocks.astype('float32'),
        precision='float32',
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
  """Residual block wrapping local attention."""

  def __init__(
      self,
      model_dims: int,
      atten_num_heads: int,
      atten_left_context: int,
      *,
      rngs: nnx.Rngs,
      atten_right_context: int = 0,
      gradient_clipping: float = 10_000_000_000.0,
      dtype: jnp.dtype | None = None,
      param_dtype: jnp.dtype = jnp.float32,
  ):
    self.gradient_clipping = gradient_clipping

    self.pre_norm = nnx.RMSNorm(
        rngs=rngs,
        num_features=model_dims,
        use_fast_variance=False,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    units_per_head = model_dims // atten_num_heads
    self.self_atten = LocalDotProductAttention(
        model_dims=model_dims,
        atten_num_heads=atten_num_heads,
        units_per_head=units_per_head,
        atten_left_context=atten_left_context,
        atten_right_context=atten_right_context,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    self.post = ClippedEinsum(
        rngs=rngs,
        shape=(atten_num_heads, units_per_head, model_dims),
        dtype=dtype,  # pyrefly: ignore[bad-argument-type]
        param_dtype=param_dtype,
    )

    self.post_norm = nnx.RMSNorm(
        rngs=rngs,
        num_features=model_dims,
        use_fast_variance=False,
        dtype=dtype,
        param_dtype=param_dtype,
    )

  def __call__(
      self,
      x: jax.Array,
      mask: jax.Array,
      causal_valid_mask: jax.Array,
  ) -> jax.Array:
    residual = x

    clip_value = self.gradient_clipping
    x = jnp.clip(x, -clip_value, clip_value)

    y = self.pre_norm(x)
    y = self.self_atten(y, mask, causal_valid_mask)
    y = self.post('...NH,NHD->...D', y)

    y = jnp.clip(y, -clip_value, clip_value)
    y = self.post_norm(y)

    return residual + y


class ConformerLayer(nnx.Module):
  """A single layer of the Conformer model."""

  def __init__(
      self,
      model_dims: int,
      atten_num_heads: int,
      atten_left_context: int,
      conv_kernel_size: int,
      *,
      rngs: nnx.Rngs,
      atten_right_context: int = 0,
      gradient_clipping: float = 10_000_000_000.0,
      dtype: jnp.dtype | None = None,
      param_dtype: jnp.dtype = jnp.float32,
  ):
    self.gradient_clipping = gradient_clipping

    self.fflayer_start = FFNBlock(
        num_features=model_dims,
        ffn_residual_weight=0.5,
        gradient_clipping=gradient_clipping,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    self.trans_atten = AttentionBlock(
        model_dims=model_dims,
        atten_num_heads=atten_num_heads,
        atten_left_context=atten_left_context,
        atten_right_context=atten_right_context,
        gradient_clipping=gradient_clipping,
        rngs=rngs,
        param_dtype=param_dtype,
        dtype=dtype,
    )

    self.lconv = LightweightConvBlock(
        num_features=model_dims,
        kernel_size=conv_kernel_size,
        gradient_clipping=gradient_clipping,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    self.fflayer_end = FFNBlock(
        num_features=model_dims,
        ffn_residual_weight=0.5,
        gradient_clipping=gradient_clipping,
        rngs=rngs,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    self.final_ln = nnx.RMSNorm(
        rngs=rngs,
        num_features=model_dims,
        use_fast_variance=False,
        dtype=dtype,
        param_dtype=param_dtype,
    )

  def __call__(
      self,
      x: jax.Array,
      mask: jax.Array,
      causal_valid_mask: jax.Array,
  ) -> jax.Array:
    x = self.fflayer_start(x)
    x = self.trans_atten(x, mask, causal_valid_mask)

    # Apply validity mask
    validity_mask = mask[:, :, jnp.newaxis].astype(x.dtype)
    x = x * validity_mask

    x = self.lconv(x)
    x = self.fflayer_end(x)

    # Clipping
    clip_value = self.gradient_clipping
    x = jnp.clip(x, -clip_value, clip_value)

    x = self.final_ln(x)
    return x


class AudioTokenizer(nnx.Module):
  """Audio encoder for Gemma4 implemented in Flax NNX."""

  def __init__(
      self,
      config: ConformerConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.sample_rate = 16_000

    n_mels = 128
    self.to_mel = GemaxMelFilterbank(  # Not an nnx.Module
        sample_rate=self.sample_rate,
        n_mels=n_mels,
    )
    self.feature = SubSamplingBlock(
        in_features=n_mels,
        out_features=self.config.model_dims,
        rngs=rngs,
        param_dtype=config.param_dtype,
    )
    self.conformer_layers = compat.ModuleList([
        ConformerLayer(
            model_dims=config.model_dims,
            atten_num_heads=config.atten_num_heads,
            atten_left_context=config.atten_left_context,
            conv_kernel_size=config.conv_kernel_size,
            rngs=rngs,
            atten_right_context=config.atten_right_context,
            gradient_clipping=config.gradient_clipping,
            dtype=config.compute_dtype,
            param_dtype=config.param_dtype,
        )
        for _ in range(config.num_layers)
    ])
    self.output_projection = nnx.Linear(
        in_features=config.model_dims,
        out_features=config.lm_model_dims,
        rngs=rngs,
        dtype=config.compute_dtype,
        param_dtype=config.param_dtype,
    )

  @staticmethod
  def infer_mask(
      x: jnp.ndarray, sequence_lengths: jnp.ndarray, original_seq_len: int
  ) -> jnp.ndarray:
    """Infer boolean validity mask after temporal compression.

    Args:
      x: Tensor with compressed time dimension in axis 1.
      sequence_lengths: Original sequence lengths per batch element.
      original_seq_len: Original time dimension before compression.

    Returns:
      Boolean mask of shape [batch, compressed_time], True for valid positions.
    """
    compressed_seq_len = x.shape[1]
    compression_rate = original_seq_len / compressed_seq_len
    new_sequence_lengths = jnp.floor(
        sequence_lengths / compression_rate
    ).astype(jnp.int32)

    indices = jnp.arange(compressed_seq_len)[jnp.newaxis, :]
    mask = indices < new_sequence_lengths[:, jnp.newaxis]
    return mask

  @staticmethod
  def to_float32(x: jnp.ndarray):
    if x.dtype == jnp.int16:
      return x.astype(jnp.float32) / 32768.0
    elif x.dtype == jnp.int32:
      return x.astype(jnp.float32) / 2147483648.0
    elif x.dtype == jnp.uint8:
      return (x.astype(jnp.float32) - 128.0) / 128.0
    elif x.dtype in [jnp.float16, jnp.float32]:
      return x.astype(jnp.float32)
    else:
      raise ValueError(f'Unsupported format: {x.dtype}')

  @staticmethod
  def _compute_causal_valid_mask(config: ConformerConfig):
    """Computes the local causal validity mask for chunked attention."""
    chunk_size = config.atten_block_size
    max_future_horizon = config.atten_right_context
    max_past_horizon = max(0, config.atten_left_context - 1)
    context_size = chunk_size + max_past_horizon + max_future_horizon
    upper_diagonal = max_past_horizon + max_future_horizon

    lower_causal_mask = LocalDotProductAttention._ones_matrix_band_part(  # pylint: disable=protected-access
        context_size,
        chunk_size,
        num_lower=-1,
        num_upper=0,
        out_dtype=jnp.bool_,
    ).T
    upper_causal_mask = LocalDotProductAttention._ones_matrix_band_part(  # pylint: disable=protected-access
        chunk_size,
        context_size,
        num_lower=-1,
        num_upper=upper_diagonal,
        out_dtype=jnp.bool_,
    )
    causal_valid_mask = lower_causal_mask & upper_causal_mask
    return causal_valid_mask

  def __call__(
      self,
      x: jax.Array,
      sequence_lengths: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    """Computes audio embeddings from raw waveforms.

    Args:
      x: Input audio waveforms. shape: (batch, samples)
      sequence_lengths: Length of each sequence in the batch. shape: (batch)

    Returns:
      Tuple of (audio_embeddings, padding_mask) where padding_mask is a
      boolean array with True indicating padding positions.
    """
    x = self.to_float32(x)
    original_seq_len = x.shape[-1]

    x = self.to_mel(x)
    mask = self.infer_mask(x, sequence_lengths, original_seq_len)
    x = jnp.where(mask[:, :, jnp.newaxis], x, 0.0)

    x, mask = self.feature(x, mask)
    causal_valid_mask = self._compute_causal_valid_mask(self.config)

    for i in range(self.config.num_layers):
      x = self.conformer_layers[i](x, mask, causal_valid_mask)

    if self.config.conf_reduction_factor > 1:
      x = x[:, :: self.config.conf_reduction_factor]
      mask = mask[:, :: self.config.conf_reduction_factor]

    x = self.output_projection(x)

    padding_mask = ~mask
    x = jnp.where(padding_mask[:, :, jnp.newaxis], 0.0, x)

    return x, padding_mask

  def get_num_soft_tokens(self, num_samples: int) -> int:
    """Calculates the number of soft tokens produced for a given audio length.

    Args:
      num_samples: Number of audio samples (e.g., 16000 for 1 sec at 16kHz).

    Returns:
      The number of soft tokens.
    """
    # 1. Mel Filterbank (STFT frames calculation)
    win_length = self.to_mel.win_length
    hop_length = self.to_mel.hop_length
    frame_size_for_unfold = win_length + 1
    if num_samples < frame_size_for_unfold:
      return 0
    num_frames = (num_samples - frame_size_for_unfold) // hop_length + 1

    # 2. SubSamplingBlock (Convolution)
    # Pad = 1, Kernel = 3, Stride = 2 for both layers.
    # Stage 1
    t1 = ((num_frames + 2 - 3) // 2) + 1
    # Stage 2
    t2 = ((t1 + 2 - 3) // 2) + 1

    # 3. Conformer reduction factor slicing (x[:, ::reduction_factor])
    num_tokens = math.ceil(t2 / self.config.conf_reduction_factor)
    return num_tokens
