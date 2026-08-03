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

"""Tests for batch_utils."""

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import batch_utils


class UtilsTest(absltest.TestCase):

  def test_run_chunked_matches_single_pass(self):
    prompt_tokens = np.ones((6, 2), dtype=np.int32)
    completion_tokens = np.ones((6, 3), dtype=np.int32)

    calls = []

    def _fn(p, _):
      calls.append(p.shape[0])
      return np.ones((p.shape[0],), dtype=np.float32)

    # Single pass
    single = batch_utils.apply_chunked(
        _fn, None, prompt_tokens, completion_tokens
    )
    self.assertEqual(calls, [6])
    self.assertEqual(single.shape, (6,))

    calls.clear()

    # Chunked
    chunked = batch_utils.apply_chunked(
        _fn, 2, prompt_tokens, completion_tokens
    )
    self.assertEqual(calls, [2, 2, 2])
    self.assertEqual(chunked.shape, (6,))
    np.testing.assert_array_equal(single, chunked)

  def test_apply_chunked_validates_1d_arrays(self):
    with self.assertRaisesRegex(
        ValueError, "Arrays must have at least 1 dimension"
    ):
      batch_utils.apply_chunked(
          lambda p, _: None,
          None,
          np.ones((), dtype=np.int32),
          np.ones((3, 3), dtype=np.int32),
      )

  def test_apply_chunked_validates_matching_batch_sizes(self):
    with self.assertRaisesRegex(ValueError, "Mismatched batch sizes"):
      batch_utils.apply_chunked(
          lambda p, _: None,
          None,
          np.ones((4, 2), dtype=np.int32),
          np.ones((5, 3), dtype=np.int32),
      )

  def test_apply_chunked_rejects_non_positive_chunk_size(self):
    with self.assertRaisesRegex(ValueError, "chunk_size must be positive"):
      batch_utils.apply_chunked(
          lambda p, _: None,
          0,
          np.ones((4, 2), dtype=np.int32),
          np.ones((4, 3), dtype=np.int32),
      )


if __name__ == "__main__":
  absltest.main()
