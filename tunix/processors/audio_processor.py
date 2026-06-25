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

"""Audio processing for Gemma4."""

from typing import Any
import jax.numpy as jnp
import numpy as np
from tunix.models.gemma4 import model as model_lib

POSITIONS_PAD_VALUE = -1


def compute_soft_token_count(
    length: int,
    sample_rate: int = 16000,
    max_audio_seq_len: int = 750,
) -> int:
  """Computes the number of soft tokens for a given audio length."""
  frame_length = int(round(sample_rate * 20.0 / 1000.0))
  hop_length = int(round(sample_rate * 10.0 / 1000.0))
  frame_size_for_unfold = frame_length + 1
  num_mel_frames = (length - frame_size_for_unfold) // hop_length + 1
  t = num_mel_frames
  for _ in range(2):
    t_padded = t + 2
    t = (t_padded - 3) // 2 + 1
  return min(t, max_audio_seq_len)


def add_variable_extra_tokens_for_audio(
    tokens: np.ndarray,
    *,
    soft_token_counts: list[int] | tuple[tuple[int, ...], ...],
    placeholder_token: int = 258881,  # <|audio|>
    start_token: int = 256000,  # <|audio>
    end_token: int = 258883,  # <audio|>
) -> np.ndarray:
  """Expand `AUDIO_PLACEHOLDER` with a variable number of placeholders."""
  batch_size = tokens.shape[0]
  results = []
  for b in range(batch_size):
    row = tokens[b].tolist()
    expanded = []
    audio_idx = 0

    if len(soft_token_counts) > 0 and isinstance(soft_token_counts[0], int):
      counts = soft_token_counts
    else:
      counts = soft_token_counts[b] if b < len(soft_token_counts) else ()

    for token in row:
      if token == placeholder_token and audio_idx < len(counts):
        count = counts[audio_idx]
        expanded.append(start_token)
        expanded.extend([model_lib.AUDIO_TOKEN_PLACEHOLDER] * count)
        expanded.append(end_token)
        audio_idx += 1
      else:
        expanded.append(token)
    results.append(expanded)

  max_len = max(len(r) for r in results)
  padded = np.zeros((batch_size, max_len), dtype=np.int32)
  for b, row in enumerate(results):
    padded[b, : len(row)] = row

  return padded


def process_gemma4_audio_inputs(
    audio: Any,
    tokens: list[np.ndarray],
    audio_encoder_config: Any,
    pad_id: int,
) -> tuple[model_lib.PreprocessedAudioInput, list[np.ndarray]]:
  """Processes audio and tokens for Gemma4 multimodal models."""

  # Normalize audio input to list[list[np.ndarray]] (batch, clips)
  if not isinstance(audio, list):
    audio = [[audio]]
  elif len(audio) > 0 and not isinstance(audio[0], list):
    audio = [[clip] for clip in audio]

  max_n_clips = max((len(batch) for batch in audio), default=0)

  batch_audio = []
  batch_lengths = []
  all_soft_token_counts = []
  max_samples = 0

  sample_rate = getattr(audio_encoder_config, "sample_rate", 16000)
  max_audio_seq_len = getattr(audio_encoder_config, "audio_seq_length", 750)

  for batch in audio:
    if not batch:
      batch_audio.append([])
      batch_lengths.append([])
      all_soft_token_counts.append(())
      continue

    clip_audios = []
    clip_lengths = []
    clip_soft_token_counts = []
    for clip in batch:
      clip = np.asarray(clip, dtype=np.float32)
      clip_audios.append(clip)
      clip_lengths.append(len(clip))
      clip_soft_token_counts.append(
          compute_soft_token_count(len(clip), sample_rate, max_audio_seq_len)
      )
      max_samples = max(max_samples, len(clip))

    batch_audio.append(clip_audios)
    batch_lengths.append(clip_lengths)
    all_soft_token_counts.append(tuple(clip_soft_token_counts))

  if max_samples == 0:
    max_samples = 16000

  final_audio = []
  final_lengths = []

  for b_idx in range(len(audio)):
    clips = batch_audio[b_idx]
    lengths = batch_lengths[b_idx]

    padded_clips = []
    padded_lengths = []

    for clip_idx in range(len(clips)):
      clip = clips[clip_idx]
      length = lengths[clip_idx]
      pad_len = max_samples - len(clip)
      if pad_len > 0:
        clip = np.pad(clip, (0, pad_len))
      padded_clips.append(clip)
      padded_lengths.append(length)

    n_pad = max_n_clips - len(clips)
    for _ in range(n_pad):
      padded_clips.append(np.zeros(max_samples, dtype=np.float32))
      padded_lengths.append(0)

    if padded_clips:
      final_audio.append(np.stack(padded_clips, axis=0))
      final_lengths.append(np.array(padded_lengths, dtype=np.int32))
    else:
      final_audio.append(np.zeros((max_n_clips, max_samples), dtype=np.float32))
      final_lengths.append(np.zeros(max_n_clips, dtype=np.int32))

  if final_audio:
    audio_tensor = jnp.stack(final_audio, axis=0)
    lengths_tensor = jnp.stack(final_lengths, axis=0)
  else:
    batches = len(audio)
    audio_tensor = jnp.zeros(
        (batches, max_n_clips, max_samples), dtype=jnp.float32
    )
    lengths_tensor = jnp.zeros((batches, max_n_clips), dtype=jnp.int32)

  processed_audio = model_lib.PreprocessedAudioInput(
      audio=audio_tensor,
      audio_lengths=lengths_tensor,
      soft_token_counts=tuple(all_soft_token_counts),
  )

  if all_soft_token_counts:
    max_len = max(len(t) for t in tokens)
    padded_tokens = np.array([
        np.pad(x, (0, max_len - len(x)), constant_values=pad_id) for x in tokens
    ])
    expanded_tokens = add_variable_extra_tokens_for_audio(
        padded_tokens,
        soft_token_counts=tuple(all_soft_token_counts),
    )
    tokens = [
        np.array([tid for tid in row if tid != pad_id])
        for row in expanded_tokens.tolist()
    ]

  return processed_audio, tokens
