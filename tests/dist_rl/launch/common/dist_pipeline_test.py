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

"""Unit tests for `dist_pipeline.DistPipeline`."""

from __future__ import annotations

from absl.testing import absltest
from tunix.cli import base_rl_pipeline
from tunix.dist_rl.launch.common import dist_pipeline
from tunix.dist_rl.launch.common import resolve_config
from tunix.rl import rl_cluster


class DistPipelineTest(absltest.TestCase):

  def test_is_base_pipeline_with_dist_defaults(self):
    hp = dist_pipeline.initialize(["batch_size=8"])
    self.assertIsInstance(hp, base_rl_pipeline.BasePipeline)
    # `finalize_dist_config` ran (mini_batch_size null -> batch_size).
    self.assertEqual(hp.config["rl_training_config"]["mini_batch_size"], 8)

  def test_matches_dist_hyperparameters(self):
    argv = ["batch_size=8", "cluster_config.run_id=fixed"]
    self.assertEqual(
        dist_pipeline.initialize(argv).config,
        resolve_config.initialize(argv).config,
    )

  def test_runs_dist_validation(self):
    with self.assertRaisesRegex(ValueError, "divisible"):
      dist_pipeline.initialize(
          ["batch_size=8", "rl_training_config.mini_batch_size=3"]
      )

  def test_builds_rl_training_config_with_actor_optimizer(self):
    hp = dist_pipeline.initialize([
        "rl_training_config.max_steps=7",
        "rl_training_config.mini_batch_size=2",
    ])
    training_config = hp.create_rl_training_config()
    self.assertIsInstance(training_config, rl_cluster.RLTrainingConfig)
    self.assertEqual(training_config.max_steps, 7)
    self.assertEqual(training_config.mini_batch_size, 2)
    self.assertIsNotNone(training_config.actor_optimizer)
    self.assertIsNotNone(training_config.checkpointing_options)

  def test_builds_rollout_config_and_tokenizer_free_parts(self):
    hp = dist_pipeline.initialize([
        "rollout_config.max_prompt_length=128",
        "rollout_config.total_generation_steps=64",
    ])
    rollout_config = hp.create_rollout_config()
    self.assertEqual(rollout_config.max_prompt_length, 128)
    self.assertEqual(rollout_config.max_tokens_to_generate, 64)

  def test_run_is_not_supported(self):
    hp = dist_pipeline.initialize([])
    with self.assertRaises(NotImplementedError):
      hp.run_trainer()


if __name__ == "__main__":
  absltest.main()
