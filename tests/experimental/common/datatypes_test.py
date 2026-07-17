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

import cloudpickle
import numpy as np

from absl.testing import absltest
from tunix.experimental.common import datatypes


def _sample_result() -> datatypes.TrajectoryResult:
  return datatypes.TrajectoryResult(
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
    np.testing.assert_array_equal(restored.prompt_tokens, original.prompt_tokens)
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
    result = datatypes.TrajectoryResult(
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

  def test_validate_wire_safe_accepts_numpy_result(self):
    datatypes.validate_wire_safe(_sample_result())  # Should not raise.

  def test_validate_wire_safe_rejects_top_level_device_array(self):
    import jax.numpy as jnp

    result = _sample_result()
    result.prompt_tokens = jnp.asarray(result.prompt_tokens)

    with self.assertRaises(TypeError):
      datatypes.validate_wire_safe(result)

  def test_validate_wire_safe_rejects_device_array_in_segment(self):
    import jax.numpy as jnp

    result = _sample_result()
    result.segments[0].tokens = jnp.asarray(result.segments[0].tokens)

    with self.assertRaises(TypeError):
      datatypes.validate_wire_safe(result)


if __name__ == "__main__":
  absltest.main()
