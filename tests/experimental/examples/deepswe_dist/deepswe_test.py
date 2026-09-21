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

"""Tests for the distributed DeepSWE recipe wiring."""

from pathlib import Path

from absl.testing import absltest
from tunix.experimental.examples.deepswe_dist import deepswe
from tunix.experimental.examples.deepswe_dist import run_deepswe_dist


class DeepSWEDistTest(absltest.TestCase):

  def test_recipe_defaults_match_reference(self):
    args = run_deepswe_dist._parse_args([])  # pylint: disable=protected-access

    self.assertEqual(args.model_id, "Qwen/Qwen3-32B")
    self.assertEqual(args.batch_size, 8)
    self.assertEqual(args.mini_batch_size, 8)
    self.assertEqual(args.num_generations, 8)
    self.assertEqual(args.max_steps, 50)
    self.assertEqual(args.max_turns, 50)
    self.assertEqual(args.max_prompt_length, 4096)
    self.assertEqual(args.max_response_length, 8192)
    self.assertEqual(args.epsilon, 0.2)
    self.assertEqual(args.epsilon_high, 0.28)
    self.assertEqual(args.advantage_estimator, "rloo")
    self.assertEqual(args.loss_agg_mode, "sequence-mean-token-scale")
    self.assertFalse(args.use_rollout_logps)
    self.assertEqual(args.dataset_name, "R2E-Gym/R2E-Gym-Subset")
    self.assertEqual(args.weight_sync_mode.value, "raiden")

  def test_algorithm_uses_reference_recipe_config(self):
    args = run_deepswe_dist._parse_args([])  # pylint: disable=protected-access
    algo = run_deepswe_dist._build_algo(args)  # pylint: disable=protected-access

    self.assertEqual(algo.mini_batch_size, 8)
    self.assertEqual(algo.algo_config.advantage_estimator, "rloo")
    self.assertEqual(
        algo.algo_config.loss_agg_mode, "sequence-mean-token-scale"
    )
    self.assertEqual(algo.algo_config.epsilon_high, 0.28)

  def test_launchers_use_reference_training_defaults(self):
    example_dir = Path(deepswe.__file__).parent
    local_launcher = (example_dir / "launcher.sh").read_text(encoding="utf-8")
    k8s_launcher = (example_dir / "k8s_launcher.sh").read_text(
        encoding="utf-8"
    )

    for launcher in (local_launcher, k8s_launcher):
      self.assertIn("WEIGHT_SYNC_MODE=${WEIGHT_SYNC_MODE:-raiden}", launcher)
      self.assertIn("USE_ROLLOUT_LOGPS=${USE_ROLLOUT_LOGPS:-false}", launcher)
      self.assertIn("PARAM_DTYPE=${PARAM_DTYPE:-float32}", launcher)
      self.assertIn(
          "FLASH_ATTENTION_BLOCK_SIZE=${FLASH_ATTENTION_BLOCK_SIZE:-1024}",
          launcher,
      )


if __name__ == "__main__":
  absltest.main()
