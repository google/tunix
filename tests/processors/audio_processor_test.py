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

"""Tests for audio_processor."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.processors import audio_processor


class AudioProcessorTest(parameterized.TestCase):

  def test_compute_soft_token_count(self):
    # 1 second of audio at 16kHz
    self.assertEqual(audio_processor.compute_soft_token_count(16000), 25)
    # 2 seconds of audio at 16kHz
    self.assertEqual(audio_processor.compute_soft_token_count(32000), 50)

  def test_add_variable_extra_tokens_for_audio(self):
    tokens = np.array([
        [1, 2, 258881, 3],
        [4, 258881, 258881, 5],
    ])
    soft_token_counts = ((2,), (1, 2))

    expanded = audio_processor.add_variable_extra_tokens_for_audio(
        tokens,
        soft_token_counts=soft_token_counts,
    )

    expected_row_0 = [1, 2, 256000, -4, -4, 258883, 3]
    expected_row_1 = [4, 256000, -4, 258883, 256000, -4, -4, 258883, 5]

    max_len = max(len(expected_row_0), len(expected_row_1))
    padded_expected = np.zeros((2, max_len), dtype=np.int32)
    padded_expected[0, : len(expected_row_0)] = expected_row_0
    padded_expected[1, : len(expected_row_1)] = expected_row_1

    np.testing.assert_array_equal(expanded, padded_expected)

  def test_process_gemma4_audio_inputs_batch(self):
    class _DummyAudioConfig:
      sample_rate = 16000
      audio_seq_length = 750

    config = _DummyAudioConfig()

    audio = [[np.zeros(16000), np.zeros(8000)], [np.zeros(32000)]]
    tokens = [
        np.array([1, 258881, 3, 258881]),
        np.array([258881, 4]),
    ]

    processed, new_tokens = audio_processor.process_gemma4_audio_inputs(
        audio=audio,
        tokens=tokens,
        audio_encoder_config=config,
        pad_id=0,
    )

    self.assertEqual(processed.audio.shape, (2, 2, 32000))
    self.assertEqual(processed.audio_lengths.shape, (2, 2))

    np.testing.assert_array_equal(processed.audio_lengths[0], [16000, 8000])
    np.testing.assert_array_equal(processed.audio_lengths[1], [32000, 0])

    self.assertEqual(processed.soft_token_counts, ((25, 12), (50,)))

    self.assertLen(new_tokens, 2)
    expected_new_tokens_0 = np.array(
        [1]
        + [256000]
        + [-4] * 25
        + [258883]
        + [3]
        + [256000]
        + [-4] * 12
        + [258883]
    )
    expected_new_tokens_1 = np.array([256000] + [-4] * 50 + [258883] + [4])
    np.testing.assert_array_equal(new_tokens[0], expected_new_tokens_0)
    np.testing.assert_array_equal(new_tokens[1], expected_new_tokens_1)


if __name__ == "__main__":
  absltest.main()
