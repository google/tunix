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

"""Audio processing for ALMs."""

from typing import Protocol
import numpy as np
import tunix.models.gemma4.model as gemma4_model_lib


class _Gemma4AudioTokenizerLike(Protocol):
  sample_rate: int

  def get_num_soft_tokens(self, num_samples: int) -> int:
    ...


def process_gemma4_inputs(
    audios: np.ndarray | list[np.ndarray | list[np.ndarray]],
    tokens: list[np.ndarray],
    audio_encoder: _Gemma4AudioTokenizerLike,
    max_audio_length: int | None = None,
    max_audio_clips: int | None = None,
) -> tuple[gemma4_model_lib.PreprocessedAudioInput | None, list[np.ndarray]]:
  """Process audio and tokens for Gemma4-E2B/E4B models.

  Args:
    audios: Raw audio waveforms. Can be a single array (batch_size=1), list of
      arrays (multiple samples in a batch, each sample with one clip), or a list
      of list of arrays (multiple clips for multiple samples in a batch). A mix
      of these is also allowed. E.g. [a1, [a2, a3], []] would mean the first
      sample has 1 audio clip (a1), the second sample has 2 audio clips (a2 and
      a3), and the third sample has 0 audio clips.
    tokens: List containing a batch of token sequences. Length of this list must
      match the length of `audios`. Additionally, the number of audio token
      placeholders (numeric value of 258881) in these sequences must match the
      number of audio clips provided in audios. These audio token placeholders
      will be expanded into appropriate numbers of soft-token placeholders.
    audio_encoder: An instance of Gemma4's AudioTokenizer.
    max_audio_length: Maximum length of audio waveforms. If specified, audio
      input to the model will be padded upto this length. Specify to avoid
      recompilation on different audio lengths across calls.
    max_audio_clips: Maximum number of audio clips in a sample. If specified,
      audio input to the model will be padded upto this count. Specify to avoid
      recompilation on different number of clips across calls.

  Returns:
    Processed audio input for the model, and the expanded list of token
    sequences.
  """
  # Gemma4 has limit of 30s per audio clip
  # pylint: disable-next=invalid-name
  MAX_CLIP_LENGTH = audio_encoder.sample_rate * 30

  if max_audio_length is not None:
    if max_audio_length > MAX_CLIP_LENGTH:
      raise ValueError(
          f'max_audio_length={max_audio_length} too big. Gemma4 supports'
          f' maximum audio clip length of {MAX_CLIP_LENGTH} samples.'
      )

  # Coerce audios into a list[list[np.ndarray]]
  # First dimension of this list corresponds to batch-index
  # Second dimension corresponds to clip-index
  if isinstance(audios, np.ndarray):
    assert audios.ndim == 1
    audios = [[audios]]
  elif isinstance(audios, list):
    clean_input = []
    for clips in audios:
      if isinstance(clips, np.ndarray):
        assert clips.ndim == 1
        clean_input.append([clips])
      elif isinstance(clips, list):
        assert all(isinstance(x, np.ndarray) for x in clips)
        assert all(x.ndim == 1 for x in clips)
        clean_input.append(clips)
      else:
        raise ValueError(f'Unsupported audio waveform type {type(clips)}')
    audios = clean_input
  else:
    raise ValueError(f'Unsupported audio waveform type {type(audios)}')

  batch_size = len(audios)
  if batch_size != len(tokens):
    raise ValueError(
        f'Batch size of tokens ({len(tokens)}) does not match '
        f'the batch size of audios ({len(audios)}).'
    )

  # Validate the clip counts and lengths
  num_audio_clips_list = [len(clips) for clips in audios]
  actual_max_audio_clips = max(num_audio_clips_list)
  if max_audio_clips is None:
    max_audio_clips = actual_max_audio_clips
  else:
    if actual_max_audio_clips > max_audio_clips:
      raise ValueError(
          'A batch entry has more clips than the specified "max_audio_clips". '
          f'{max_audio_clips=}, {actual_max_audio_clips=}.'
      )

  actual_max_audio_length = (
      max(len(clip) for clips in audios for clip in clips)
      if actual_max_audio_clips > 0
      else 0
  )
  if max_audio_length is None:
    max_audio_length = actual_max_audio_length
    if max_audio_length > MAX_CLIP_LENGTH:
      raise ValueError(
          f'Gemma4 supports maximum audio clip length of {MAX_CLIP_LENGTH}'
          f' samples. Got a clip with {max_audio_length} samples.'
      )
  else:
    if actual_max_audio_length > max_audio_length:
      raise ValueError(
          'An audio clip is longer than the specified "max_audio_length". '
          f'{max_audio_length=}, {actual_max_audio_length=}.'
      )

  # Expand <|audio|> tokens with appropriate amount of soft token placeholders.
  # pylint: disable=invalid-name
  SOFT_TOKEN_PLACEHOLDER = gemma4_model_lib.AUDIO_SOFT_TOKEN_PLACEHOLDER
  # Constants from upstream gemma4's tokenizer (gemma/text/_tokenizer.py).
  AUDIO_PLACEHOLDER = 258881  # <|audio|>
  START_OF_AUDIO = 256000  # <|audio> (BOA)
  END_OF_AUDIO = 258883  # <audio|> (EOA)
  # pylint: enable=invalid-name

  assert batch_size == len(tokens)
  expanded_token_batch = []
  for b in range(batch_size):
    sequence = tokens[b].tolist()

    # Ensure that we have exactly the right amount of clips.
    num_placeholders = sum(token == AUDIO_PLACEHOLDER for token in sequence)
    if num_audio_clips_list[b] != num_placeholders:
      raise ValueError(
          f'Input mismatch at batch index {b}. '
          f'Placeholders provided for {num_placeholders} clips, '
          f'but only {num_audio_clips_list[b]} provided.'
      )

    expanded_sequence = []
    clip_index = 0
    for token in tokens[b]:
      if token == AUDIO_PLACEHOLDER:
        audio_length = len(audios[b][clip_index])
        num_soft_tokens = audio_encoder.get_num_soft_tokens(audio_length)
        expanded_sequence.append(START_OF_AUDIO)
        expanded_sequence.extend([SOFT_TOKEN_PLACEHOLDER] * num_soft_tokens)
        expanded_sequence.append(END_OF_AUDIO)
        clip_index += 1
      else:
        expanded_sequence.append(token)
    expanded_token_batch.append(np.array(expanded_sequence))

  processed_audios = None
  if max_audio_clips > 0 and max_audio_length > 0:
    # Create model input.
    # Gemma4's AudioTokenizer takes a padded audio tensor, and sequence length
    # of all clips. Sequence lenghts are used by model internally for masking.
    padded_audios = np.zeros(
        shape=(batch_size, max_audio_clips, max_audio_length),
        dtype=np.float32,
    )
    padded_audio_lengths = np.zeros(
        shape=(batch_size, max_audio_clips),
        dtype=int,
    )
    for b, clips in enumerate(audios):
      for i, clip in enumerate(clips):
        if i < max_audio_clips:
          padded_audios[b, i, : len(clip)] = clip
          padded_audio_lengths[b, i] = len(clip)

    processed_audios = gemma4_model_lib.PreprocessedAudioInput(
        audios=padded_audios,  # pyrefly: ignore[bad-argument-type]
        sequence_lengths=padded_audio_lengths,  # pyrefly: ignore[bad-argument-type]
    )

  return processed_audios, expanded_token_batch
