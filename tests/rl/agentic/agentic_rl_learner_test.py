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
import queue
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

  def test_rollout_batch_watchdog_fails_waiting_for_first_group(self):
    class StalledOrchestrator:

      async def run_producers_from_stream(self, **kwargs):
        del kwargs
        await asyncio.Event().wait()

      async def yield_batches(self, batch_size):
        del batch_size
        await asyncio.Event().wait()
        if False:
          yield []

    async def run():
      learner = object.__new__(DummyLearner)
      learner.loop = asyncio.get_running_loop()
      learner.rl_cluster = mock.Mock(global_steps=0)
      learner._full_batch_size = 1
      learner._background_tasks = set()
      learner.algo_config = agentic_rl_learner.AgenticRLConfig(
          num_generations=2,
          rollout_batch_timeout=0.01,
      )
      stream = learner._orchestrator_producer(
          StalledOrchestrator(), iter(()), 2, "Token"
      )
      with self.assertRaisesRegex(
          TimeoutError, "completed_prompt_groups=0/1"
      ):
        await anext(stream)
      await stream.aclose()

    asyncio.run(run())

  def test_p38_diagnostic_consumer_covers_all_prompt_groups(self):
    self.assertEqual(
        agentic_rl_learner._p38_diagnostic_consumer_contract(
            enabled=True,
            full_batch_size=32,
            mini_batch_size=4,
            train_micro_batch_size=4,
            num_generations=8,
            process_in_consumer=True,
        ),
        (32, True, 8),
    )

  def test_p38_diagnostic_consumer_is_noop_when_disabled(self):
    self.assertEqual(
        agentic_rl_learner._p38_diagnostic_consumer_contract(
            enabled=False,
            full_batch_size=17,
            mini_batch_size=5,
            train_micro_batch_size=3,
            num_generations=2,
            process_in_consumer=False,
        ),
        (3, False, 0),
    )

  def test_p38_diagnostic_consumer_admits_onehost_rehearsal(self):
    self.assertEqual(
        agentic_rl_learner._p38_diagnostic_consumer_contract(
            enabled=True,
            full_batch_size=2,
            mini_batch_size=2,
            train_micro_batch_size=4,
            num_generations=2,
            process_in_consumer=True,
            onehost_rehearsal=True,
        ),
        (2, True, 1),
    )

  def test_p38_diagnostic_consumer_rejects_subset_geometry(self):
    with self.assertRaisesRegex(ValueError, "coverage geometry changed"):
      agentic_rl_learner._p38_diagnostic_consumer_contract(
          enabled=True,
          full_batch_size=32,
          mini_batch_size=5,
          train_micro_batch_size=5,
          num_generations=8,
          process_in_consumer=True,
      )

  def test_p38_diagnostic_consumer_rejects_partial_tail(self):
    learner = object.__new__(DummyLearner)
    data = queue.Queue()
    for value in range(5):
      data.put(value)
    data.put(None)
    batches = learner._data_consumer_batch_generator(
        data, 32, require_full_batch=True
    )
    with self.assertRaisesRegex(RuntimeError, "refusing subset alignment"):
      next(batches)

  def test_normal_consumer_keeps_legacy_partial_tail_behavior(self):
    learner = object.__new__(DummyLearner)
    data = queue.Queue()
    for value in range(5):
      data.put(value)
    data.put(None)
    batches = learner._data_consumer_batch_generator(data, 32)
    self.assertEqual(next(batches), list(range(5)))

  def test_model_call_wraps_one_conversation_as_a_prompt_batch(self):
    learner = object.__new__(DummyLearner)
    learner.chat_parser = None
    learner.rl_cluster = mock.Mock()
    learner.rl_cluster.generate.return_value = mock.sentinel.rollout
    conversation = [{"role": "user", "content": "hello"}]

    result = learner._model_call(conversation)

    self.assertIs(result, mock.sentinel.rollout)
    self.assertEqual(
        learner.rl_cluster.generate.call_args.kwargs["prompts"],
        [conversation],
    )

  def test_frozenlake_evaluation_metrics_are_finite_and_complete(self):
    metrics = agentic_rl_learner._frozenlake_evaluation_metrics(
        [0.0, 0.2, 1.0], wall_seconds=2.5, policy_step=10
    )
    self.assertEqual(metrics["n"], 3)
    self.assertAlmostEqual(metrics["solve"], 2.0 / 3.0)
    self.assertEqual(metrics["wall_seconds"], 2.5)
    self.assertEqual(metrics["policy_step"], 10)

  def test_frozenlake_evaluation_metrics_reject_nonfinite_rewards(self):
    with self.assertRaisesRegex(ValueError, "nonempty and finite"):
      agentic_rl_learner._frozenlake_evaluation_metrics(
          [0.0, float("nan")], wall_seconds=1.0, policy_step=0
      )

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

  def test_nonpositive_eval_cadence_disables_evaluation(self):
    for cadence in (0, -1):
      with self.subTest(cadence=cadence):
        self.assertFalse(
            agentic_rl_learner._should_run_eval(
                prompt_count=100,
                schedule_step=0,
                eval_every_n_steps=cadence,
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
