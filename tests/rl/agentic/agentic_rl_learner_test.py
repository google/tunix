# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for agentic_rl_learner."""

import asyncio
from typing import Any
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import agentic_rl_learner
from tunix.rl.rollout import base_rollout


class DummyLearner(agentic_rl_learner.AgenticRLLearner):
  def _process_results(self, **kwargs):
    return []


class AgenticRLLearnerTest(parameterized.TestCase):

  def test_validate_rollout_config_mismatch_max_tokens(self):
    rl_cluster = mock.Mock()
    rl_cluster.cluster_config = mock.Mock()
    rl_cluster.cluster_config.rollout_engine = "generic"
    rollout_config = base_rollout.RolloutConfig(
        max_prompt_length=32,
        max_tokens_to_generate=10,
        return_logprobs=True,
    )
    rl_cluster.cluster_config.rollout_config = rollout_config

    algo_config = agentic_rl_learner.AgenticRLConfig(
        max_response_length=20,  # Mismatch: 10 != 20
        use_rollout_logps=True,
    )

    with self.assertRaisesRegex(
        ValueError, r"max_tokens_to_generate \(10\) must match AgenticRLConfig max_response_length \(20\)"
    ):
      DummyLearner(
          rl_cluster=rl_cluster,
          reward_fns=mock.Mock(),
          algo_config=algo_config,
      )

  def test_validate_rollout_config_missing_logprobs(self):
    rl_cluster = mock.Mock()
    rl_cluster.cluster_config = mock.Mock()
    rl_cluster.cluster_config.rollout_engine = "generic"
    rollout_config = base_rollout.RolloutConfig(
        max_prompt_length=32,
        max_tokens_to_generate=10,
        return_logprobs=False,  # Should be True
    )
    rl_cluster.cluster_config.rollout_config = rollout_config

    algo_config = agentic_rl_learner.AgenticRLConfig(
        max_response_length=10,
        use_rollout_logps=True,
    )

    with self.assertRaisesRegex(
        ValueError, r"must have return_logprobs=True"
    ):
      DummyLearner(
          rl_cluster=rl_cluster,
          reward_fns=mock.Mock(),
          algo_config=algo_config,
      )

  def test_validate_rollout_config_dict_mode(self):
    rl_cluster = mock.Mock()
    rl_cluster.cluster_config = mock.Mock()
    rl_cluster.cluster_config.rollout_engine = "generic"
    rollout_config_train = base_rollout.RolloutConfig(
        max_prompt_length=32,
        max_tokens_to_generate=10,
        return_logprobs=True,
    )
    rollout_config_eval = base_rollout.RolloutConfig(
        max_prompt_length=32,
        max_tokens_to_generate=10,
        return_logprobs=False,  # Mismatch in eval mode
    )
    rl_cluster.cluster_config.rollout_config = {
        "train": rollout_config_train,
        "eval": rollout_config_eval,
    }

    algo_config = agentic_rl_learner.AgenticRLConfig(
        max_response_length=10,
        use_rollout_logps=True,
    )

    with self.assertRaisesRegex(
        ValueError, r"RolloutConfig \(eval\) must have return_logprobs=True"
    ):
      DummyLearner(
          rl_cluster=rl_cluster,
          reward_fns=mock.Mock(),
          algo_config=algo_config,
      )

  def test_validate_rollout_config_vllm_missing_server_mode(self):
    rl_cluster = mock.Mock()
    rl_cluster.cluster_config = mock.Mock()
    rl_cluster.cluster_config.rollout_engine = "vllm"
    rollout_config = base_rollout.RolloutConfig(
        max_prompt_length=32,
        max_tokens_to_generate=10,
        return_logprobs=True,
        rollout_vllm_server_mode=False,  # Should be True for vLLM
    )
    rl_cluster.cluster_config.rollout_config = rollout_config

    algo_config = agentic_rl_learner.AgenticRLConfig(
        max_response_length=10,
        use_rollout_logps=True,
    )

    with self.assertRaisesRegex(
        ValueError,
        r"must have rollout_vllm_server_mode set to True for AgenticRLLearner"
        r" if using vLLM engine",
    ):
      DummyLearner(
          rl_cluster=rl_cluster,
          reward_fns=mock.Mock(),
          algo_config=algo_config,
      )

  def test_train_batch_size_mismatch_raises_error(self):
    with mock.patch.object(
        rl_utils, "is_sharing_weights", return_value=False
    ):
      rl_cluster = mock.Mock()
      rl_cluster.cluster_config = mock.Mock()
      rl_cluster.cluster_config.role_to_mesh = {
          rl_cluster_lib.Role.ACTOR: mock.Mock(),
          rl_cluster_lib.Role.ROLLOUT: mock.Mock(),
      }
      training_config = mock.Mock()
      training_config.compute_logps_micro_batch_size = 2
      training_config.train_micro_batch_size = 1
      training_config.mini_batch_size = None
      training_config.max_seq_token_per_tpu = None
      rl_cluster.cluster_config.training_config = training_config
      rl_cluster.cluster_config.rollout_config = base_rollout.RolloutConfig(
          max_tokens_to_generate=10, return_logprobs=True
      )
      rl_cluster.cluster_config.rollout_engine = 'generic'
      rl_cluster.actor_trainer = mock.Mock()
      rl_cluster.actor_trainer.restored_global_step.return_value = 0
      rl_cluster.actor_trainer.iter_steps = 0
      rl_cluster.rollout = mock.Mock()
      rl_cluster.tokenizer = mock.Mock()
      algo_config = agentic_rl_learner.AgenticRLConfig(max_response_length=10)
      learner = DummyLearner(
          rl_cluster=rl_cluster,
          reward_fns=mock.Mock(),
          algo_config=algo_config,
      )
      train_dataset = [{'prompt': ['p1']}]
      with self.assertRaisesRegex(
          ValueError,
          r'compute_logps_micro_batch_size \(2\) must be equal to'
          r' train_micro_batch_size \(1\)',
      ):
        learner.train(train_dataset)


if __name__ == "__main__":
  absltest.main()
