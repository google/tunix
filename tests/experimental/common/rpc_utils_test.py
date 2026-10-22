"""Tests for rpc_utils."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.common import rpc_utils


def _sample_response() -> datatypes.RolloutResponse:
  item = datatypes.TrajectoryItem(
      prompt_id="req-rollout-42",
      group_index=1,
      start_step=0,
      traj={
          "reward": 1.25,
          "status": datatypes.TrajectoryStatus.SUCCEEDED,
          "prompt_tokens": np.array([10, 11, 12], dtype=np.int32),
          "completion_tokens": np.array([20, 21], dtype=np.int32),
          "action_mask": np.array([1, 1], dtype=np.float32),
          "policy_version": 7,
      },
  )
  return datatypes.RolloutResponse(
      request_id="req-1",
      status="SUCCEEDED",
      payload=item,
  )


class RpcUtilsTest(absltest.TestCase):

  def test_validate_wire_safe_accepts_numpy_result(self):
    rpc_utils.validate_wire_safe(_sample_response())  # Should not raise.

  def test_validate_wire_safe_rejects_top_level_device_array(self):
    with self.assertRaises(TypeError):
      rpc_utils.validate_wire_safe(jnp.asarray([10, 11, 12]))

  def test_validate_wire_safe_rejects_device_array_in_payload(self):
    result = _sample_response()
    result.payload.traj["completion_tokens"] = jnp.asarray(
        result.payload.completion_tokens
    )

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
