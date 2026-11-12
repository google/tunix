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

"""Tests for the GRPO and rollout-replica flags of `run_deepswe_dist.py`."""

from __future__ import annotations

import unittest

from absl.testing import absltest

# `r2egym` is an optional dependency in OSS (`swe_agent` requires it at import
# time); match `deepswe_test.py` so OSS CPU CI skips cleanly when absent.
try:
  import r2egym  # pylint: disable=unused-import,g-import-not-at-top
except ImportError as exc:
  raise unittest.SkipTest("r2egym is not installed") from exc

# pylint: disable=g-import-not-at-top
from tunix.dist_rl.examples.deepswe_dist import run_deepswe_dist
from tunix.rl import algorithm_config

# pylint: enable=g-import-not-at-top


class RunDeepSWEDistTest(absltest.TestCase):

  def test_build_algo_defaults_match_grpo_config(self):
    algo = run_deepswe_dist._build_algo(
        run_deepswe_dist._parse_args(["--num_generations=2"])
    )
    default = algorithm_config.GRPOConfig(num_generations=2)
    self.assertEqual(algo.epsilon_high, default.epsilon_high)
    self.assertEqual(algo.advantage_estimator, default.advantage_estimator)
    self.assertEqual(algo.loss_agg_mode, default.loss_agg_mode)
    self.assertEqual(algo.sampler_is, default.sampler_is)
    self.assertEqual(algo.sampler_is_threshold, default.sampler_is_threshold)

  def test_build_algo_forwards_converging_recipe_flags(self):
    algo = run_deepswe_dist._build_algo(
        run_deepswe_dist._parse_args([
            "--num_generations=8",
            "--epsilon=0.2",
            "--epsilon_high=0.28",
            "--advantage_estimator=rloo",
            "--loss_agg_mode=sequence-mean-token-scale",
            "--sampler_is=token",
            "--sampler_is_threshold=2.0",
        ])
    )
    self.assertEqual(algo.num_generations, 8)
    self.assertEqual(algo.epsilon, 0.2)
    self.assertEqual(algo.epsilon_high, 0.28)
    self.assertEqual(algo.advantage_estimator, "rloo")
    self.assertEqual(algo.loss_agg_mode, "sequence-mean-token-scale")
    self.assertEqual(algo.sampler_is, "token")
    self.assertEqual(algo.sampler_is_threshold, 2.0)

  def test_none_disables_sampler_is_and_its_threshold(self):
    args = run_deepswe_dist._parse_args(
        ["--sampler_is=none", "--sampler_is_threshold=null"]
    )
    self.assertIsNone(args.sampler_is)
    self.assertIsNone(args.sampler_is_threshold)

  def test_rollout_replicas_must_be_positive(self):
    args = run_deepswe_dist._parse_args(["--rollout_replicas=0"])
    with self.assertRaisesRegex(ValueError, "rollout_replicas must be >= 1"):
      run_deepswe_dist._validate_args(args)


if __name__ == "__main__":
  absltest.main()
