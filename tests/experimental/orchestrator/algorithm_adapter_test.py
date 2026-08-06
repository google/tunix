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

from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.rl import algorithm_config


class AlgorithmAdapterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = algorithm_config.AlgorithmConfig(
        algo_variant="grpo",
        advantage_estimator="grpo",
    )
    self.adapter = algorithm_adapter.get_algorithm_adapter(self.config)

  def test_compute_advantages(self):
    rewards = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    advs = self.adapter.compute_advantages(rewards, num_generations=2)
    self.assertEqual(advs.shape, (4,))

  def test_assemble_train_example(self):
    prompt_ids = np.ones((2, 4), dtype=np.int32)
    prompt_mask = np.ones((2, 4), dtype=np.bool_)
    completion_ids = np.ones((2, 4), dtype=np.int32)
    completion_mask = np.ones((2, 4), dtype=np.bool_)
    advantages = np.array([0.5, -0.5], dtype=np.float32)

    ex = self.adapter.assemble_train_example(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        advantages=advantages,
        policy_version=5,
    )

    self.assertIsNotNone(ex)
    self.assertEqual(ex.prompt_ids.shape, (2, 4))
    self.assertEqual(ex.advantages.shape, (2,))
    np.testing.assert_array_equal(ex.policy_version, np.array([5, 5]))

  def test_get_metrics(self):
    rewards = np.array([1.0, 3.0], dtype=np.float32)
    advantages = np.array([-1.0, 1.0], dtype=np.float32)
    metrics = self.adapter.get_metrics(rewards, advantages)

    self.assertIn("rewards/mean", metrics)
    self.assertEqual(metrics["rewards/mean"][0], 2.0)
    self.assertEqual(metrics["rewards/advantage/mean"][0], 0.0)

  def test_factory_dispatch_ppo(self):
    ppo_config = algorithm_config.AlgorithmConfig(
        algo_variant="ppo",
        advantage_estimator="gae",
    )
    adapter = algorithm_adapter.get_algorithm_adapter(ppo_config)
    self.assertIsInstance(adapter, algorithm_adapter.PPOAdapter)

  def test_get_loss_fn(self):
    loss_fn = self.adapter.get_loss_fn()
    self.assertTrue(callable(loss_fn))


if __name__ == "__main__":
  absltest.main()
