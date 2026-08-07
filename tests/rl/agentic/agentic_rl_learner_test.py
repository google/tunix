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
import jax.numpy as jnp
import numpy as np
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import agentic_rl_learner
from tunix.rl.rollout import base_rollout


class DummyLearner(agentic_rl_learner.AgenticRLLearner):
  def _process_results(self, **kwargs):
    return []


class AgenticRLLearnerTest(parameterized.TestCase):

  def test_p32_compact_d3b0_output_preserves_values_and_deletes_arrays(self):
    loss = jnp.asarray(1.25, jnp.float32)
    logps = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    entropy = jnp.ones((3, 4), jnp.float32)
    scratch = jnp.arange(17, dtype=jnp.float32)
    output = {
        "loss": loss,
        "per_token_logps": logps,
        "token_entropy": entropy,
        "loss_output": {"scratch": scratch},
        "reports": ({"group": 0},),
        "gradient_microbatches": 16,
    }

    compact, deleted_arrays, deleted_bytes = (
        agentic_rl_learner._p32_compact_d3b0_output(output)
    )

    self.assertTrue(np.array_equal(compact["loss"], np.asarray(1.25)))
    self.assertTrue(
        np.array_equal(
            compact["per_token_logps"],
            np.arange(12, dtype=np.float32).reshape(3, 4),
        )
    )
    self.assertEqual(compact["reports"], ({"group": 0},))
    self.assertEqual(compact["gradient_microbatches"], 16)
    self.assertEqual(deleted_arrays, 4)
    self.assertEqual(deleted_bytes, (1 + 12 + 12 + 17) * 4)
    self.assertTrue(all(value.is_deleted() for value in (
        loss, logps, entropy, scratch
    )))

  def test_p32_prompt_placement_accepts_four_intact_groups(self):
    prompts = np.repeat(np.arange(4, dtype=np.int32)[:, None], 8, axis=0)
    report = agentic_rl_learner._p32_prompt_placement(prompts)
    self.assertEqual(report["rank_counts"], [16, 16])
    self.assertEqual(report["rank_group_ids"], [[0, 1], [2, 3]])
    self.assertLen(report["group_hashes"], 4)

  def test_p32_prompt_placement_rejects_split_group(self):
    prompts = np.repeat(np.arange(4, dtype=np.int32)[:, None], 8, axis=0)
    prompts[[7, 8]] = prompts[[8, 7]]
    with self.assertRaisesRegex(ValueError, "prompt group 0 is not contiguous"):
      agentic_rl_learner._p32_prompt_placement(prompts)

  def test_p31_segmented_eval_uses_preupdate_step_exactly_once(self):
    self.assertEqual(
        agentic_rl_learner._eval_schedule_step(
            segmented_update=True,
            pre_update_train_step=0,
            current_train_step=1,
        ),
        0,
    )
    self.assertTrue(
        agentic_rl_learner._should_run_eval(
            prompt_count=100,
            schedule_step=0,
            eval_every_n_steps=25,
            last_eval_train_step=-1,
        )
    )
    self.assertFalse(
        agentic_rl_learner._should_run_eval(
            prompt_count=100,
            schedule_step=0,
            eval_every_n_steps=25,
            last_eval_train_step=0,
        )
    )
    self.assertEqual(
        agentic_rl_learner._eval_schedule_step(
            segmented_update=False,
            pre_update_train_step=0,
            current_train_step=1,
        ),
        1,
    )
    self.assertFalse(
        agentic_rl_learner._should_run_eval(
            prompt_count=100,
            schedule_step=1,
            eval_every_n_steps=25,
            last_eval_train_step=-1,
        )
    )

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
