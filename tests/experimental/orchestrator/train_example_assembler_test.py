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

"""Tests for the pure trajectories_to_train_example assembler."""

import numpy as np

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import train_example_assembler as tea
from tunix.rl.agentic import utils as agentic_utils


_TOKENIZER = datatypes.TokenizerInfo(pad_id=0, eos_id=1)
_SHAPE = datatypes.ShapeConfig(max_prompt_length=4, max_response_tokens=5)


def _samples():
  return [
      tea.SampleTokens(
          prompt_tokens=np.array([7, 8]),
          completion_tokens=np.array([3, 4, 1]),
          completion_mask=np.array([1, 1, 1]),
          policy_version=2,
          old_logprobs=np.array([-0.1, -0.2, -0.3], dtype=np.float32),
      ),
      tea.SampleTokens(
          prompt_tokens=np.array([9]),
          completion_tokens=np.array([5, 6]),
          completion_mask=np.array([1, 1]),
          policy_version=3,
          old_logprobs=np.array([-0.4, -0.5], dtype=np.float32),
      ),
  ]


class TrainExampleAssemblerTest(absltest.TestCase):

  def test_padding_masks_and_carry_through(self):
    example = tea.trajectories_to_train_example(
        _samples(),
        advantages=np.array([0.5, -0.5]),
        tokenizer_info=_TOKENIZER,
        shape_config=_SHAPE,
        use_rollout_logps=True,
    )
    np.testing.assert_array_equal(
        example.prompt_ids, [[0, 0, 7, 8], [0, 0, 0, 9]]
    )
    np.testing.assert_array_equal(
        example.prompt_mask, [[0, 0, 1, 1], [0, 0, 0, 1]]
    )
    np.testing.assert_array_equal(
        example.completion_ids, [[3, 4, 1, 0, 0], [5, 6, 0, 0, 0]]
    )
    # loss_mask is the right-padded completion mask.
    np.testing.assert_array_equal(
        example.loss_mask, [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]
    )
    np.testing.assert_allclose(
        example.old_per_token_logps,
        [[-0.1, -0.2, -0.3, 0.0, 0.0], [-0.4, -0.5, 0.0, 0.0, 0.0]],
        rtol=1e-6,
    )
    np.testing.assert_array_equal(example.policy_version, [2, 3])
    np.testing.assert_allclose(example.advantages, [0.5, -0.5])
    self.assertIsNone(example.ref_per_token_logps)

  def test_old_logps_omitted_without_rollout_logps(self):
    example = tea.trajectories_to_train_example(
        _samples(),
        advantages=np.array([0.5, -0.5]),
        tokenizer_info=_TOKENIZER,
        shape_config=_SHAPE,
        use_rollout_logps=False,
    )
    self.assertIsNone(example.old_per_token_logps)

  def test_completion_longer_than_bucket_is_truncated(self):
    samples = [
        tea.SampleTokens(
            prompt_tokens=np.array([7]),
            completion_tokens=np.array([3, 4, 5, 6, 7, 8, 9]),  # 7 > 5
            completion_mask=np.array([1, 1, 1, 1, 1, 1, 1]),
            policy_version=1,
        ),
        tea.SampleTokens(
            prompt_tokens=np.array([7]),
            completion_tokens=np.array([3, 4]),
            completion_mask=np.array([1, 1]),
            policy_version=1,
        ),
    ]
    example = tea.trajectories_to_train_example(
        samples,
        advantages=np.array([0.0, 0.0]),
        tokenizer_info=_TOKENIZER,
        shape_config=_SHAPE,
    )
    self.assertEqual(example.completion_ids.shape, (2, 5))
    np.testing.assert_array_equal(example.completion_ids[0], [3, 4, 5, 6, 7])

  def test_matches_upstream_pad_helpers(self):
    # The assembler reuses the exact upstream padding helpers, so a caller that
    # delegates its pad/mask step here produces identical arrays.
    samples = _samples()
    example = tea.trajectories_to_train_example(
        samples,
        advantages=np.array([1.0, 2.0]),
        tokenizer_info=_TOKENIZER,
        shape_config=_SHAPE,
        use_rollout_logps=True,
    )
    for i, sample in enumerate(samples):
      exp_prompt, exp_completion, _ = agentic_utils.pad_prompt_and_completion(
          list(sample.prompt_tokens),
          list(sample.completion_tokens),
          _SHAPE.max_prompt_length,
          _SHAPE.max_response_tokens,
          _TOKENIZER.pad_id,
      )
      np.testing.assert_array_equal(example.prompt_ids[i], exp_prompt)
      np.testing.assert_array_equal(
          example.completion_ids[i], exp_completion[: _SHAPE.max_response_tokens]
      )

  def test_empty_samples_raises(self):
    with self.assertRaises(ValueError):
      tea.trajectories_to_train_example(
          [], advantages=np.array([]),
          tokenizer_info=_TOKENIZER, shape_config=_SHAPE,
      )


if __name__ == "__main__":
  absltest.main()
