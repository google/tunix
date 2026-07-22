"""Tests for rpc_utils."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.common import rpc_utils


def _sample_result() -> datatypes.RolloutResult:
  return datatypes.RolloutResult(
      request_id="req-1",
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


class RpcUtilsTest(absltest.TestCase):

  def test_validate_wire_safe_accepts_numpy_result(self):
    rpc_utils.validate_wire_safe(_sample_result())  # Should not raise.

  def test_validate_wire_safe_rejects_top_level_device_array(self):
    result = _sample_result()
    result.prompt_tokens = jnp.asarray(result.prompt_tokens)

    with self.assertRaises(TypeError):
      rpc_utils.validate_wire_safe(result)

  def test_validate_wire_safe_rejects_device_array_in_segment(self):
    result = _sample_result()
    result.segments[0].tokens = jnp.asarray(result.segments[0].tokens)

    with self.assertRaises(TypeError):
      rpc_utils.validate_wire_safe(result)

  def test_validate_wire_safe_catches_cycles(self):
    cyclic_list = []
    cyclic_list.append(cyclic_list)
    rpc_utils.validate_wire_safe(cyclic_list)  # Should not raise or hang

  def test_validate_wire_safe_checks_dict_keys(self):
    class MockDeviceArray:
      shape = (1,)
      dtype = np.float32

    bad_key = (MockDeviceArray(),)
    test_dict = {bad_key: "value"}
    with self.assertRaisesRegex(
        TypeError, "wire payload contains a non-numpy array"
    ):
      rpc_utils.validate_wire_safe(test_dict)

  def test_validate_wire_safe_accepts_sets(self):
    test_set = {1, 2, "three"}
    rpc_utils.validate_wire_safe(test_set)  # Should not raise


if __name__ == "__main__":
  absltest.main()
