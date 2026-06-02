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

  def test_validate_rollout_config_rejects_offpolicy_in_colocate_mode(self):
    rl_cluster = mock.Mock()
    rl_cluster.cluster_config = mock.Mock()
    rl_cluster.cluster_config.rollout_engine = 'generic'
    rl_cluster.cluster_config.colocate_mode = True
    rl_cluster.cluster_config.rollout_config = base_rollout.RolloutConfig(
        max_prompt_length=32,
        max_tokens_to_generate=10,
        return_logprobs=True,
    )

    algo_config = agentic_rl_learner.AgenticRLConfig(
        max_response_length=10,
        use_rollout_logps=True,
        off_policy_steps=1,
    )

    with self.assertRaisesRegex(
        ValueError, r'colocate_mode requires off_policy_steps to be 0'
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

  def test_colocate_producer_waits_until_rollout_window_closes(self):
    with mock.patch.object(
        rl_utils, 'is_sharing_weights', return_value=False
    ):
      rl_cluster = mock.Mock()
      rl_cluster.cluster_config = mock.Mock()
      rl_cluster.cluster_config.role_to_mesh = {
          rl_cluster_lib.Role.ACTOR: mock.Mock(),
          rl_cluster_lib.Role.ROLLOUT: mock.Mock(),
      }
      rl_cluster.cluster_config.training_config = mock.Mock(
          compute_logps_micro_batch_size=1,
          train_micro_batch_size=1,
          mini_batch_size=None,
      )
      rl_cluster.cluster_config.rollout_config = base_rollout.RolloutConfig(
          max_tokens_to_generate=10, return_logprobs=True
      )
      rl_cluster.cluster_config.rollout_engine = 'generic'
      rl_cluster.cluster_config.colocate_mode = True
      rl_cluster.can_overlap_actor_and_rollout.return_value = False
      rl_cluster.is_colocate_mode_enabled.return_value = True
      rl_cluster.exit_trainer_window.side_effect = lambda: events.append('trainer-exit')
      rl_cluster.enter_trainer_window.side_effect = lambda: events.append('trainer-enter')
      rl_cluster.enter_colocate_rollout_window.side_effect = lambda: events.append('enter')
      rl_cluster.exit_colocate_rollout_window.side_effect = lambda: events.append('exit')
      rl_cluster.actor_trainer = mock.Mock()
      rl_cluster.actor_trainer.restored_global_step.return_value = 0
      rl_cluster.actor_trainer.iter_steps = 0
      rl_cluster.actor_trainer.model = mock.Mock()
      rl_cluster.rollout = mock.Mock()
      rl_cluster.rollout.model.return_value = mock.Mock()
      rl_cluster.tokenizer = mock.Mock()

      learner = DummyLearner(
          rl_cluster=rl_cluster,
          reward_fns=mock.Mock(),
          algo_config=agentic_rl_learner.AgenticRLConfig(
              max_response_length=10,
              off_policy_steps=0,
          ),
      )

      events = []
      learner._batch_to_train_example = lambda batch_results, mode: [
          f'{mode}-{batch_results[0]}'
      ]

            class FakeOrchestrator:

                def __init__(self):
                    self.queue = asyncio.Queue()
                    self.run_calls = 0

                async def run_producers_from_stream(
                        self, pairs_stream, group_size, group_key_fn, collect_mode
                ):
                    del group_size, group_key_fn, collect_mode
                    self.run_calls += 1
                    async for _agent, env in pairs_stream:
                        await self.queue.put([f'group-{env.extra_kwargs["group_id"] + 1}'])
                    await self.queue.put(None)

                async def yield_batches(self, batch_size):
                    del batch_size
                    while True:
                        batch = await self.queue.get()
                        if batch is None:
                            return
                        yield batch

      prompt_queue = queue.Queue()
            prompt_queue.put({'prompt': ['p1']})
            prompt_queue.put({'prompt': ['p2']})
      prompt_queue.put(None)
      train_data_queue = mock.Mock()
      train_data_queue.put.side_effect = lambda item: events.append(f'put:{item}')

            fake_orchestrator = FakeOrchestrator()

            asyncio.run(
                    learner._producer(fake_orchestrator, prompt_queue, train_data_queue)
            )

      self.assertEqual(
          events,
          [
              'trainer-exit',
              'enter',
              'exit',
              'trainer-enter',
              f'put:{rl_cluster_lib.Mode.TRAIN}-group-1',
                            'trainer-exit',
                            'enter',
                            'exit',
                            'trainer-enter',
              f'put:{rl_cluster_lib.Mode.TRAIN}-group-2',
              'put:None',
          ],
      )
            self.assertEqual(fake_orchestrator.run_calls, 1)

  def test_train_sets_process_in_consumer_in_colocate_mode(self):
    with (
        mock.patch.object(
            rl_utils, 'is_sharing_weights', return_value=False
        ),
        mock.patch.object(agentic_rl_learner.sft_utils, 'show_hbm_usage'),
    ):
      rl_cluster = mock.Mock()
      rl_cluster.cluster_config = mock.Mock()
      rl_cluster.cluster_config.role_to_mesh = {
          rl_cluster_lib.Role.ACTOR: mock.Mock(),
          rl_cluster_lib.Role.ROLLOUT: mock.Mock(),
      }
      rl_cluster.cluster_config.training_config = mock.Mock(
          compute_logps_micro_batch_size=1,
          train_micro_batch_size=1,
          mini_batch_size=None,
          max_seq_token_per_tpu=None,
          max_steps=1,
      )
      rl_cluster.cluster_config.rollout_config = base_rollout.RolloutConfig(
          max_tokens_to_generate=10, return_logprobs=True
      )
      rl_cluster.cluster_config.rollout_engine = 'generic'
      rl_cluster.cluster_config.colocate_mode = True
      rl_cluster.can_overlap_actor_and_rollout.return_value = False
      rl_cluster.is_colocate_mode_enabled.return_value = True
      rl_cluster.actor_trainer = mock.Mock()
      rl_cluster.actor_trainer.restored_global_step.return_value = 0
      rl_cluster.actor_trainer.iter_steps = 0
      rl_cluster.actor_trainer.model = mock.Mock()
      rl_cluster.actor_trainer.train_steps = 1
      rl_cluster.rollout = mock.Mock()
      rl_cluster.rollout.model.return_value = mock.Mock()
      rl_cluster.tokenizer = mock.Mock()
      rl_cluster.perf_v2 = mock.Mock()
      rl_cluster.perf_v2.export.return_value = {}
      rl_cluster.buffer_metrics.return_value = None
      rl_cluster.close.return_value = None

      learner = DummyLearner(
          rl_cluster=rl_cluster,
          reward_fns=mock.Mock(),
          algo_config=agentic_rl_learner.AgenticRLConfig(
              max_response_length=10,
              off_policy_steps=0,
          ),
      )

      seen_flags = []

      async def fake_producer(orchestrator, prompt_queue, train_data_queue):
        del orchestrator, prompt_queue
        seen_flags.append(learner._process_in_consumer)
        train_data_queue.put(None)

      learner._producer = fake_producer

      learner.train([{'prompt': ['p1']}])

      self.assertEqual(seen_flags, [True])


if __name__ == "__main__":
  absltest.main()
