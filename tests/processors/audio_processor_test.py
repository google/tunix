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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.processors import audio_processor


class Gemma4AudioProcessorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    class MockAudioTokenizer:
      sample_rate = 16_000

      def get_num_soft_tokens(self, num_samples: int) -> int:
        # Simplified logic for testing: 25 soft tokens per 16000 samples (1 sec)
        return max(1, 25 * num_samples // 16000)

    self.audio_encoder = MockAudioTokenizer()
    self.audio_placeholder = 258881
    self.boa = 256000
    self.eoa = 258883
    self.soft_placeholder = -4

  def test_process_inputs_single_ndarray(self):
    # 1 second of audio
    np.random.seed(42)
    audio = np.random.uniform(-1.0, 1.0, (16000,)).astype(np.float32)
    tokens = [np.array([1, 2, self.audio_placeholder, 3])]

    processed_audios, new_tokens = audio_processor.process_gemma4_inputs(
        audios=audio,
        tokens=tokens,
        audio_encoder=self.audio_encoder,
    )

    self.assertIsNotNone(processed_audios)
    self.assertEqual(processed_audios.audios.shape, (1, 1, 16000))
    self.assertEqual(processed_audios.sequence_lengths.shape, (1, 1))
    np.testing.assert_array_equal(processed_audios.sequence_lengths, [[16000]])
    np.testing.assert_array_equal(processed_audios.audios[0, 0, :], audio)

    expected_tokens = [
        np.array(
            [1, 2, self.boa] + [self.soft_placeholder] * 25 + [self.eoa, 3]
        )
    ]
    self.assertLen(new_tokens, 1)
    np.testing.assert_array_equal(new_tokens[0], expected_tokens[0])

  def test_process_inputs_batch_list_ndarray(self):
    np.random.seed(42)
    audio1 = np.random.uniform(-1.0, 1.0, (16000,)).astype(np.float32)
    audio2 = np.random.uniform(-1.0, 1.0, (8000,)).astype(np.float32)
    audios = [audio1, audio2]

    tokens = [
        np.array([self.audio_placeholder, 1]),
        np.array([2, self.audio_placeholder]),
    ]

    processed_audios, new_tokens = audio_processor.process_gemma4_inputs(
        audios=audios,  # pyrefly: ignore[bad-argument-type]
        tokens=tokens,
        audio_encoder=self.audio_encoder,
    )

    # Padded to max length in batch (16000)
    self.assertIsNotNone(processed_audios)
    self.assertEqual(processed_audios.audios.shape, (2, 1, 16000))
    np.testing.assert_array_equal(
        processed_audios.sequence_lengths, [[16000], [8000]]
    )

    # Verify content and padding
    np.testing.assert_array_equal(processed_audios.audios[0, 0, :], audio1)
    np.testing.assert_array_equal(processed_audios.audios[1, 0, :8000], audio2)
    np.testing.assert_array_equal(
        processed_audios.audios[1, 0, 8000:], np.zeros(8000)
    )

    # Clip 1: 16000 samples -> 25 soft tokens
    # Clip 2: 8000 samples -> 12 soft tokens
    self.assertLen(new_tokens, 2)
    np.testing.assert_array_equal(
        new_tokens[0],
        np.array([self.boa] + [self.soft_placeholder] * 25 + [self.eoa, 1]),
    )
    np.testing.assert_array_equal(
        new_tokens[1],
        np.array([2, self.boa] + [self.soft_placeholder] * 12 + [self.eoa]),
    )

  def test_process_inputs_batch_nested_list(self):
    np.random.seed(42)
    audio1_1 = np.random.uniform(-1.0, 1.0, (16000,)).astype(np.float32)
    audio1_2 = np.random.uniform(-1.0, 1.0, (32000,)).astype(np.float32)
    audio2_1 = np.random.uniform(-1.0, 1.0, (8000,)).astype(np.float32)

    audios = [[audio1_1, audio1_2], [audio2_1], []]

    tokens = [
        np.array([self.audio_placeholder, self.audio_placeholder]),
        np.array([self.audio_placeholder]),
        np.array([1, 2, 3]),
    ]

    proc_audios, new_tokens = audio_processor.process_gemma4_inputs(
        audios=audios,  # pyrefly: ignore[bad-argument-type]
        tokens=tokens,
        audio_encoder=self.audio_encoder,
    )

    # Max clips = 2, Max length = 32000, Batch size = 3
    self.assertIsNotNone(proc_audios)
    self.assertEqual(proc_audios.audios.shape, (3, 2, 32000))
    np.testing.assert_array_equal(
        proc_audios.sequence_lengths, [[16000, 32000], [8000, 0], [0, 0]]
    )

    # Verify content
    np.testing.assert_array_equal(
        proc_audios.audios[0, 0], np.concat((audio1_1, np.zeros(16000)))
    )
    np.testing.assert_array_equal(proc_audios.audios[0, 1], audio1_2)
    np.testing.assert_array_equal(
        proc_audios.audios[1, 0], np.concat((audio2_1, np.zeros(24000)))
    )
    np.testing.assert_array_equal(proc_audios.audios[1, 1], np.zeros(32000))
    np.testing.assert_array_equal(proc_audios.audios[2, 0], np.zeros(32000))
    np.testing.assert_array_equal(proc_audios.audios[2, 1], np.zeros(32000))

    # Verify expanded tokens
    self.assertLen(new_tokens, 3)

    expected_tokens_0 = (
        [self.boa]
        + [self.soft_placeholder] * 25
        + [self.eoa]
        + [self.boa]
        + [self.soft_placeholder] * 50
        + [self.eoa]
    )
    np.testing.assert_array_equal(new_tokens[0], np.array(expected_tokens_0))

    expected_tokens_1 = [self.boa] + [self.soft_placeholder] * 12 + [self.eoa]
    np.testing.assert_array_equal(new_tokens[1], np.array(expected_tokens_1))

    np.testing.assert_array_equal(new_tokens[2], np.array([1, 2, 3]))

  def test_process_inputs_no_audio_clips(self):
    audios = [[], []]
    tokens = [
        np.array([1, 2, 3]),
        np.array([4, 5]),
    ]

    processed_audios, new_tokens = audio_processor.process_gemma4_inputs(
        audios=audios,  # pyrefly: ignore[bad-argument-type]
        tokens=tokens,
        audio_encoder=self.audio_encoder,
    )

    self.assertIsNone(processed_audios)
    # Tokens should be unchanged
    self.assertLen(new_tokens, 2)
    np.testing.assert_array_equal(new_tokens[0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(new_tokens[1], np.array([4, 5]))

  def test_process_inputs_custom_padding(self):
    audio = np.zeros((8000,), dtype=np.float32)
    tokens = [np.array([self.audio_placeholder])]

    # Force padding to higher limits
    max_len = 20000
    max_clips = 3

    processed_audios, _ = audio_processor.process_gemma4_inputs(
        audios=audio,
        tokens=tokens,
        audio_encoder=self.audio_encoder,
        max_audio_length=max_len,
        max_audio_clips=max_clips,
    )

    self.assertIsNotNone(processed_audios)
    self.assertEqual(processed_audios.audios.shape, (1, max_clips, max_len))
    self.assertEqual(processed_audios.sequence_lengths.shape, (1, max_clips))
    self.assertEqual(processed_audios.sequence_lengths[0, 0], 8000)
    self.assertEqual(processed_audios.sequence_lengths[0, 1], 0)

  def test_max_clip_length_error(self):
    # 31 seconds of audio
    audio = np.zeros((16000 * 31,), dtype=np.float32)
    tokens = [np.array([self.audio_placeholder])]

    with self.assertRaisesRegex(
        ValueError, "Gemma4 supports maximum audio clip length"
    ):
      audio_processor.process_gemma4_inputs(
          audios=audio,
          tokens=tokens,
          audio_encoder=self.audio_encoder,
      )

  def test_batch_mismatch_error(self):
    audios = [np.zeros((1000,)), np.zeros((1000,))]
    tokens = [np.array([1])]  # Batch size 1

    with self.assertRaisesRegex(
        ValueError, "Batch size of tokens.*does not match"
    ):
      audio_processor.process_gemma4_inputs(
          audios=audios,  # pyrefly: ignore[bad-argument-type]
          tokens=tokens,
          audio_encoder=self.audio_encoder,
      )

  def test_placeholder_mismatch_error(self):
    audios = [np.zeros((1000,))]
    # 2 placeholders but only 1 audio clip provided
    tokens = [np.array([self.audio_placeholder, self.audio_placeholder])]

    with self.assertRaisesRegex(
        ValueError, "Placeholders provided for 2 clips, but only 1 provided"
    ):
      audio_processor.process_gemma4_inputs(
          audios=audios,  # pyrefly: ignore[bad-argument-type]
          tokens=tokens,
          audio_encoder=self.audio_encoder,
      )

  def test_max_audio_clips_exceeded_error(self):
    audio1 = np.zeros((1000,), dtype=np.float32)
    audio2 = np.zeros((1000,), dtype=np.float32)
    # 2 clips in this sample
    audios = [[audio1, audio2]]
    tokens = [np.array([self.audio_placeholder, self.audio_placeholder])]

    with self.assertRaisesRegex(
        ValueError, "A batch entry has more clips than the specified"
    ):
      audio_processor.process_gemma4_inputs(
          audios=audios,  # pyrefly: ignore[bad-argument-type]
          tokens=tokens,
          audio_encoder=self.audio_encoder,
          max_audio_clips=1,  # Limit to 1 clip
      )

  def test_max_audio_length_exceeded_error(self):
    # 2000 samples
    audio = np.zeros((2000,), dtype=np.float32)
    tokens = [np.array([self.audio_placeholder])]

    with self.assertRaisesRegex(
        ValueError, "An audio clip is longer than the specified"
    ):
      audio_processor.process_gemma4_inputs(
          audios=audio,
          tokens=tokens,
          audio_encoder=self.audio_encoder,
          max_audio_length=1000,  # Limit to 1000 samples
      )


if __name__ == "__main__":
  absltest.main()
