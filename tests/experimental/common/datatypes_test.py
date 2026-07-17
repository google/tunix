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

"""Serialization-discipline tests for the common wire DTOs."""

from absl.testing import absltest
import cloudpickle
import numpy as np
from tunix.experimental.common import datatypes


def _sample_result() -> datatypes.RolloutResult:
  return datatypes.RolloutResult(
      request_id="req-1",
      prompt_id="prompt-1",
      status="SUCCEEDED",
      prompt_tokens=np.array([10, 11, 12], dtype=np.int32),
      segments=[
          datatypes.TokenSegment(
              source="assistant",
              tokens=np.array([20, 21], dtype=np.int32),
              loss_mask=np.array([1, 1], dtype=np.int32),
              logprobs=np.array([-0.5, -1.5], dtype=np.float32),
          ),
          datatypes.TokenSegment(
              source="env",
              tokens=np.array([30], dtype=np.int32),
              loss_mask=np.array([0], dtype=np.int32),
          ),
      ],
      env_reward=1.25,
      policy_version=7,
  )


class WireSerializationTest(absltest.TestCase):

  def test_trajectory_result_round_trips_through_cloudpickle(self):
    original = _sample_result()

    restored = cloudpickle.loads(cloudpickle.dumps(original))

    self.assertEqual(restored.request_id, original.request_id)
    self.assertEqual(restored.prompt_id, original.prompt_id)
    self.assertEqual(restored.status, original.status)
    self.assertEqual(restored.env_reward, original.env_reward)
    self.assertEqual(restored.policy_version, original.policy_version)
    self.assertIsNone(restored.error)
    np.testing.assert_array_equal(
        restored.prompt_tokens, original.prompt_tokens
    )
    self.assertLen(restored.segments, 2)
    np.testing.assert_array_equal(
        restored.segments[0].tokens, original.segments[0].tokens
    )
    np.testing.assert_array_equal(
        restored.segments[0].loss_mask, original.segments[0].loss_mask
    )
    np.testing.assert_allclose(
        restored.segments[0].logprobs, original.segments[0].logprobs
    )
    self.assertIsNone(restored.segments[1].logprobs)

  def test_error_result_round_trips(self):
    result = datatypes.RolloutResult(
        request_id="req-2",
        prompt_id="prompt-2",
        status="TIMEOUT",
        error=datatypes.ErrorInfo(
            error_type="TimeoutError",
            message="deadline exceeded",
            retryable=True,
        ),
    )

    restored = cloudpickle.loads(cloudpickle.dumps(result))

    self.assertEqual(restored.status, "TIMEOUT")
    self.assertEqual(restored.error.error_type, "TimeoutError")
    self.assertTrue(restored.error.retryable)
    self.assertEqual(restored.prompt_tokens.size, 0)
    self.assertEmpty(restored.segments)


  def test_token_segment_enforces_shapes(self):
    with self.assertRaisesRegex(
        ValueError, "loss_mask shape .* != tokens shape"
    ):
      datatypes.TokenSegment(
          source="env",
          tokens=np.array([1, 2]),
          loss_mask=np.array([1]),
      )

    with self.assertRaisesRegex(
        ValueError, "logprobs shape .* != tokens shape"
    ):
      datatypes.TokenSegment(
          source="assistant",
          tokens=np.array([1, 2]),
          loss_mask=np.array([1, 1]),
          logprobs=np.array([0.5]),
      )


if __name__ == "__main__":
  absltest.main()
