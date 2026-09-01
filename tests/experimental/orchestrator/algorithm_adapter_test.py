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
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.rl import algo_core


class AlgorithmAdapterTest(absltest.TestCase):

  def test_grpo_advantage_normalization(self):
    adapter = algorithm_adapter.GRPOAdapter(group_size=4)
    rewards = [1.0, 2.0, 3.0, 4.0]
    advs = adapter.compute_advantages(rewards, num_generations=4)

    self.assertEqual(len(advs), 4)
    # Mean should be 0.0
    self.assertAlmostEqual(float(np.mean(advs)), 0.0, places=4)
    # Std should be 1.0
    self.assertAlmostEqual(float(np.std(advs)), 1.0, places=4)

  def test_grpo_create_trainer_payloads(self):
    adapter = algorithm_adapter.GRPOAdapter(group_size=2)
    item1 = datatypes.TrajectoryItem(
        group_index=0,
        prompt_id="g1",
        start_step=0,
        traj=datatypes.Trajectory(reward=1.0),
    )
    item1.prompt_tokens = np.array([1, 2], dtype=np.int32)
    item1.completion_tokens = np.array([3, 4], dtype=np.int32)
    item1.action_mask = np.array([1, 1], dtype=np.float32)

    item2 = datatypes.TrajectoryItem(
        group_index=1,
        prompt_id="g1",
        start_step=0,
        traj=datatypes.Trajectory(reward=2.0),
    )
    item2.prompt_tokens = np.array([1, 2], dtype=np.int32)
    item2.completion_tokens = np.array([5, 6], dtype=np.int32)
    item2.action_mask = np.array([1, 1], dtype=np.float32)

    payloads = adapter.create_trainer_payloads([item1, item2], rewards=[1.0, 2.0])
    self.assertLen(payloads, 2)
    self.assertIsInstance(payloads[0], datatypes.RLTrainerPayload)
    self.assertLess(payloads[0].advantages[0], 0.0)
    self.assertGreater(payloads[1].advantages[0], 0.0)
    self.assertEqual(adapter.loss_fn(), algo_core.grpo_loss_fn)

  def test_ppo_advantages_and_trainer_payloads(self):
    adapter = algorithm_adapter.PPOAdapter(group_size=2, gamma=0.99, lam=0.95)
    item = datatypes.TrajectoryItem(
        group_index=0,
        prompt_id="g1",
        start_step=0,
        traj=datatypes.Trajectory(reward=1.0),
    )
    item.prompt_tokens = np.array([10], dtype=np.int32)
    item.completion_tokens = np.array([20], dtype=np.int32)

    payloads = adapter.create_trainer_payloads([item], rewards=[2.0], values=[1.0])
    self.assertLen(payloads, 1)
    self.assertAlmostEqual(payloads[0].advantages[0], 1.0)
    self.assertAlmostEqual(payloads[0].returns[0], 2.0)
    self.assertEqual(adapter.loss_fn(), algo_core.ppo_policy_loss_fn)


_ROUTING_LAYERS = 2
_ROUTING_TOP_K = 2


def _routing(length, fill):
  """`[length, num_layers, top_k]` routing where every slot holds `fill`."""
  shape = (length, _ROUTING_LAYERS, _ROUTING_TOP_K)
  return np.full(shape, fill, dtype=np.int32)


class RoutedExpertsForItemTest(absltest.TestCase):
  """`_routed_experts_for` must match the payload's sequence length exactly."""

  def _align(self, routed, seq_len=8):
    item = datatypes.TrajectoryItem(routed_experts=routed)
    return algorithm_adapter._routed_experts_for(item, seq_len)  # pylint: disable=protected-access

  def test_returns_none_without_capture(self):
    self.assertIsNone(self._align(None))

  def test_short_capture_is_padded_as_unset(self):
    """Missing tail rows must fall back to the gate, not replay expert 0."""
    out = self._align(_routing(5, 3))
    self.assertEqual(out.shape, (8, _ROUTING_LAYERS, _ROUTING_TOP_K))
    np.testing.assert_array_equal(out[:5], 3)
    np.testing.assert_array_equal(out[5:], datatypes.UNSET_ROUTED_EXPERT)

  def test_wrong_rank_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "length, num_layers, top_k"):
      self._align(np.zeros((8, _ROUTING_TOP_K), dtype=np.int32))


if __name__ == "__main__":
  absltest.main()
