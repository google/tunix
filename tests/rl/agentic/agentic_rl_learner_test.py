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
import dataclasses
import math
import types
from typing import Any
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import optax
from tunix.common import configs
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import agentic_rl_learner
from tunix.rl.rollout import base_rollout


class DummyLearner(agentic_rl_learner.AgenticRLLearner):

  def __init__(self, *args, eval_rewards_to_inject=None, **kwargs):
    super().__init__(*args, **kwargs)
    self.eval_rewards_to_inject = eval_rewards_to_inject

  def _process_results(
      self,
      trajectories=None,
      mode=rl_engine_lib.Mode.TRAIN,
      **kwargs,
  ):
    if mode == rl_engine_lib.Mode.EVAL and self.eval_rewards_to_inject is not None:
      with self._rewards_window_lock:
        self._eval_rewards_window.extend(self.eval_rewards_to_inject)
    return [
        agentic_rl_learner.TrainExample(
            prompt_ids=np.zeros((1, 5), dtype=np.int32),
            prompt_mask=np.ones((1, 5), dtype=np.bool_),
            completion_ids=np.zeros((1, 5), dtype=np.int32),
            completion_mask=np.ones((1, 5), dtype=np.bool_),
            advantages=np.zeros((1, 5), dtype=np.float32),
            ref_per_token_logps=None,
            old_per_token_logps=None,
        )
    ]


@dataclasses.dataclass(slots=True, kw_only=True)
class _CustomAlgoConfig(agentic_rl_learner.AgenticRLConfig):
  eval_num_generations: int = 4


class AgenticRLLearnerTest(parameterized.TestCase):

  def test_validate_rollout_config_mismatch_max_tokens(self):
    rl_engine = mock.Mock()
    rl_engine.cluster_config = mock.Mock()
    rl_engine.cluster_config.rollout_engine = "generic"
    rollout_config = base_rollout.RolloutConfig(
        max_prompt_length=32,
        max_tokens_to_generate=10,
        return_logprobs=True,
    )
    rl_engine.cluster_config.rollout_config = rollout_config

    algo_config = agentic_rl_learner.AgenticRLConfig(
        max_response_length=20,  # Mismatch: 10 != 20
        use_rollout_logps=True,
    )

    with self.assertRaisesRegex(
        ValueError, r"max_tokens_to_generate \(10\) must match AgenticRLConfig max_response_length \(20\)"
    ):
      DummyLearner(
          rl_engine=rl_engine,
          reward_fns=mock.Mock(),
          algo_config=algo_config,
      )

  def test_validate_rollout_config_missing_logprobs(self):
    rl_engine = mock.Mock()
    rl_engine.cluster_config = mock.Mock()
    rl_engine.cluster_config.rollout_engine = "generic"
    rollout_config = base_rollout.RolloutConfig(
        max_prompt_length=32,
        max_tokens_to_generate=10,
        return_logprobs=False,  # Should be True
    )
    rl_engine.cluster_config.rollout_config = rollout_config

    algo_config = agentic_rl_learner.AgenticRLConfig(
        max_response_length=10,
        use_rollout_logps=True,
    )

    with self.assertRaisesRegex(
        ValueError, r"must have return_logprobs=True"
    ):
      DummyLearner(
          rl_engine=rl_engine,
          reward_fns=mock.Mock(),
          algo_config=algo_config,
      )

  def test_validate_rollout_config_dict_mode(self):
    rl_engine = mock.Mock()
    rl_engine.cluster_config = mock.Mock()
    rl_engine.cluster_config.rollout_engine = "generic"
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
    rl_engine.cluster_config.rollout_config = {
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
          rl_engine=rl_engine,
          reward_fns=mock.Mock(),
          algo_config=algo_config,
      )

  def test_validate_rollout_config_vllm_missing_server_mode(self):
    rl_engine = mock.Mock()
    rl_engine.cluster_config = mock.Mock()
    rl_engine.cluster_config.rollout_engine = "vllm"
    rollout_config = base_rollout.RolloutConfig(
        max_prompt_length=32,
        max_tokens_to_generate=10,
        return_logprobs=True,
        rollout_vllm_server_mode=False,  # Should be True for vLLM
    )
    rl_engine.cluster_config.rollout_config = rollout_config

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
          rl_engine=rl_engine,
          reward_fns=mock.Mock(),
          algo_config=algo_config,
      )

  def test_train_batch_size_mismatch_raises_error(self):
    with mock.patch.object(
        rl_utils, "is_sharing_weights", return_value=False
    ):
      rl_engine = mock.Mock()
      rl_engine.cluster_config = mock.Mock()
      rl_engine.cluster_config.role_to_mesh = {
          rl_engine_lib.Role.ACTOR: mock.Mock(),
          rl_engine_lib.Role.ROLLOUT: mock.Mock(),
      }
      training_config = mock.Mock()
      training_config.compute_logps_micro_batch_size = 2
      training_config.train_micro_batch_size = 1
      training_config.mini_batch_size = None
      training_config.max_seq_token_per_tpu = None
      rl_engine.cluster_config.training_config = training_config
      rl_engine.cluster_config.rollout_config = base_rollout.RolloutConfig(
          max_tokens_to_generate=10, return_logprobs=True
      )
      rl_engine.cluster_config.rollout_engine = "generic"
      rl_engine.actor_trainer = mock.Mock()
      rl_engine.actor_trainer.restored_global_step.return_value = 0
      rl_engine.actor_trainer.iter_steps = 0
      rl_engine.rollout = mock.Mock()
      rl_engine.tokenizer = mock.Mock()
      algo_config = agentic_rl_learner.AgenticRLConfig(max_response_length=10)
      learner = DummyLearner(
          rl_engine=rl_engine,
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

  def test_train_with_packing_executes_end_to_end(self):
    with mock.patch.object(
        rl_utils, "is_sharing_weights", return_value=False
    ):
      rl_engine = mock.Mock()
      rl_engine.cluster_config = mock.Mock()
      mesh = mock.Mock()
      mesh.shape = {'fsdp': 1, 'dp': 1}
      rl_engine.cluster_config.role_to_mesh = {
          rl_engine_lib.Role.ACTOR: mesh,
          rl_engine_lib.Role.ROLLOUT: mesh,
      }
      training_config = mock.Mock()
      training_config.compute_logps_micro_batch_size = 1
      training_config.train_micro_batch_size = 1
      training_config.mini_batch_size = None
      training_config.max_seq_token_per_tpu = 16  # Enable packing
      training_config.max_steps = 100
      rl_engine.cluster_config.training_config = training_config
      rl_engine.cluster_config.rollout_config = base_rollout.RolloutConfig(
          max_tokens_to_generate=10, return_logprobs=True
      )
      rl_engine.cluster_config.rollout_engine = "generic"
      rl_engine.actor_trainer = mock.Mock()
      rl_engine.actor_trainer.restored_global_step.return_value = 0
      rl_engine.actor_trainer.iter_steps = 0
      rl_engine.global_steps = 0
      rl_engine.rollout = mock.Mock()
      rl_engine.tokenizer = mock.Mock()
      algo_config = agentic_rl_learner.AgenticRLConfig(max_response_length=10)
      learner = DummyLearner(
          rl_engine=rl_engine,
          reward_fns=mock.Mock(),
          algo_config=algo_config,
      )
      train_dataset = [{'prompt': ['p1']}]

      async def mock_producer(*args, **kwargs):
        if False:
          yield

      with mock.patch.object(learner, "_orchestrator_producer", side_effect=mock_producer):
        learner.train(train_dataset)

  def _create_mock_eval_learner(
      self,
      eval_every_n_steps=1,
      skip_first_n_steps_for_eval=0,
      eval_num_generations=None,
      algo_eval_num_generations=None,
      train_steps=0,
      eval_rewards_to_inject=None,
  ):
    rl_engine = mock.Mock()
    rl_engine.cluster_config = mock.Mock()
    mesh = mock.Mock()
    mesh.shape = {"fsdp": 1, "dp": 1}
    rl_engine.cluster_config.role_to_mesh = {
        rl_engine_lib.Role.ACTOR: mesh,
        rl_engine_lib.Role.ROLLOUT: mesh,
    }
    training_config = configs.RLTrainingConfig(
        actor_optimizer=optax.sgd(1e-3),
        mini_batch_size=1,
        train_micro_batch_size=1,
        eval_every_n_steps=eval_every_n_steps,
        skip_first_n_steps_for_eval=skip_first_n_steps_for_eval,
        eval_num_generations=eval_num_generations,
        max_steps=1,
    )
    rl_engine.cluster_config.training_config = training_config
    rl_engine.cluster_config.rollout_config = base_rollout.RolloutConfig(
        max_tokens_to_generate=10, return_logprobs=True
    )
    rl_engine.cluster_config.rollout_engine = "generic"
    rl_engine.actor_trainer = mock.Mock()
    rl_engine.actor_trainer.train_steps = train_steps
    rl_engine.actor_trainer.restored_global_step.return_value = 0
    rl_engine.actor_trainer.iter_steps = 0
    rl_engine.global_steps = 0
    rl_engine.rollout = mock.Mock()
    rl_engine.tokenizer = mock.Mock()
    rl_engine.buffer_metrics_async = mock.Mock()
    rl_engine.perf_v2 = mock.MagicMock()

    if algo_eval_num_generations is not None:
      algo_config = _CustomAlgoConfig(
          max_response_length=10,
          eval_num_generations=algo_eval_num_generations,
      )
    else:
      algo_config = agentic_rl_learner.AgenticRLConfig(max_response_length=10)

    learner = DummyLearner(
        rl_engine=rl_engine,
        reward_fns=mock.Mock(),
        algo_config=algo_config,
        eval_rewards_to_inject=eval_rewards_to_inject,
    )
    return learner, rl_engine

  def test_eval_skipped_when_train_step_less_than_skip_first_n_steps(self):
    with mock.patch.object(rl_utils, "is_sharing_weights", return_value=True):
      learner, rl_engine = self._create_mock_eval_learner(
          eval_every_n_steps=1,
          skip_first_n_steps_for_eval=5,
          train_steps=0,
      )
      producer_calls = []

      async def mock_producer(*args, **kwargs):
        producer_calls.append(kwargs.get("num_generations", 1))
        yield [types.SimpleNamespace(group_id=0)]

      with mock.patch.object(learner, "_orchestrator_producer", side_effect=mock_producer), \
           mock.patch.object(learner, "_build_orchestrator", return_value=mock.Mock()):
        learner.train(train_dataset=[{"prompt": ["p1"]}], eval_dataset=[{"prompt": ["e1"]}])

      # Only training producer should be called, eval producer is skipped.
      self.assertLen(producer_calls, 1)
      for call in rl_engine.buffer_metrics_async.call_args_list:
        _, kwargs = call
        self.assertNotEqual(kwargs.get("mode"), rl_engine_lib.Mode.EVAL)

  def test_eval_runs_when_train_step_reaches_skip_first_n_steps(self):
    with mock.patch.object(rl_utils, "is_sharing_weights", return_value=True):
      learner, rl_engine = self._create_mock_eval_learner(
          eval_every_n_steps=2,
          skip_first_n_steps_for_eval=2,
          train_steps=2,
          eval_rewards_to_inject=[1.0, 0.0, 1.0, 0.0],
      )
      producer_calls = []

      async def mock_producer(*args, **kwargs):
        producer_calls.append(kwargs.get("num_generations", 1))
        yield [types.SimpleNamespace(group_id=0)]

      with mock.patch.object(learner, "_orchestrator_producer", side_effect=mock_producer), \
           mock.patch.object(learner, "_build_orchestrator", return_value=mock.Mock()):
        learner.train(train_dataset=[{"prompt": ["p1"]}], eval_dataset=[{"prompt": ["e1"]}])

      # 1 training producer call + 1 evaluation producer call
      self.assertLen(producer_calls, 2)
      self.assertEqual(producer_calls[1], 4)  # Default eval_num_generations is 4

      eval_calls = [
          call for call in rl_engine.buffer_metrics_async.call_args_list
          if call[1].get("mode") == rl_engine_lib.Mode.EVAL
      ]
      self.assertNotEmpty(eval_calls)

  def test_eval_num_generations_precedence(self):
    with mock.patch.object(rl_utils, "is_sharing_weights", return_value=True):
      # 1. training_config.eval_num_generations takes precedence
      learner, _ = self._create_mock_eval_learner(
          eval_num_generations=8,
          algo_eval_num_generations=6,
      )
      producer_calls = []

      async def mock_producer(*args, **kwargs):
        producer_calls.append(kwargs.get("num_generations", 1))
        yield [types.SimpleNamespace(group_id=0)]

      with mock.patch.object(learner, "_orchestrator_producer", side_effect=mock_producer), \
           mock.patch.object(learner, "_build_orchestrator", return_value=mock.Mock()):
        learner.train(train_dataset=[{"prompt": ["p1"]}], eval_dataset=[{"prompt": ["e1"]}])

      self.assertEqual(producer_calls[1], 8)

      # 2. Fallback to algo_config.eval_num_generations
      learner, _ = self._create_mock_eval_learner(
          eval_num_generations=None,
          algo_eval_num_generations=6,
      )
      producer_calls.clear()
      with mock.patch.object(learner, "_orchestrator_producer", side_effect=mock_producer), \
           mock.patch.object(learner, "_build_orchestrator", return_value=mock.Mock()):
        learner.train(train_dataset=[{"prompt": ["p1"]}], eval_dataset=[{"prompt": ["e1"]}])

      self.assertEqual(producer_calls[1], 6)

      # 3. Default fallback to 4
      learner, _ = self._create_mock_eval_learner(
          eval_num_generations=None,
          algo_eval_num_generations=None,
      )
      producer_calls.clear()
      with mock.patch.object(learner, "_orchestrator_producer", side_effect=mock_producer), \
           mock.patch.object(learner, "_build_orchestrator", return_value=mock.Mock()):
        learner.train(train_dataset=[{"prompt": ["p1"]}], eval_dataset=[{"prompt": ["e1"]}])

      self.assertEqual(producer_calls[1], 4)

  def test_eval_metrics_pass_at_1_and_pass_at_4_n_equals_4(self):
    with mock.patch.object(rl_utils, "is_sharing_weights", return_value=True):
      # Subtest 1: 1 out of 4 successful
      learner, rl_engine = self._create_mock_eval_learner(
          eval_num_generations=4,
          eval_rewards_to_inject=[0.0, 0.0, 0.5, 0.0],
      )

      async def mock_producer(*args, **kwargs):
        yield [types.SimpleNamespace(group_id=0)]

      with mock.patch.object(learner, "_orchestrator_producer", side_effect=mock_producer), \
           mock.patch.object(learner, "_build_orchestrator", return_value=mock.Mock()):
        learner.train(train_dataset=[{"prompt": ["p1"]}], eval_dataset=[{"prompt": ["e1"]}])

      eval_metrics = None
      for call in rl_engine.buffer_metrics_async.call_args_list:
        args, kwargs = call
        if kwargs.get("mode") == rl_engine_lib.Mode.EVAL:
          eval_metrics = args[0]
          break

      self.assertIsNotNone(eval_metrics)
      self.assertAlmostEqual(eval_metrics["diagnostics/reward_mean"][0], 0.125)
      self.assertAlmostEqual(eval_metrics["diagnostics/pass_at_1"][0], 0.25)
      self.assertEqual(eval_metrics["diagnostics/reward_count"][0], 4.0)
      self.assertEqual(eval_metrics["diagnostics/pass_at_4"][0], 1.0)

      # Subtest 2: 0 out of 4 successful
      learner_zero, rl_engine_zero = self._create_mock_eval_learner(
          eval_num_generations=4,
          eval_rewards_to_inject=[0.0, 0.0, 0.0, 0.0],
      )
      with mock.patch.object(learner_zero, "_orchestrator_producer", side_effect=mock_producer), \
           mock.patch.object(learner_zero, "_build_orchestrator", return_value=mock.Mock()):
        learner_zero.train(train_dataset=[{"prompt": ["p1"]}], eval_dataset=[{"prompt": ["e1"]}])

      eval_metrics_zero = None
      for call in rl_engine_zero.buffer_metrics_async.call_args_list:
        args, kwargs = call
        if kwargs.get("mode") == rl_engine_lib.Mode.EVAL:
          eval_metrics_zero = args[0]
          break

      self.assertIsNotNone(eval_metrics_zero)
      self.assertAlmostEqual(eval_metrics_zero["diagnostics/pass_at_1"][0], 0.0)
      self.assertEqual(eval_metrics_zero["diagnostics/pass_at_4"][0], 0.0)

  def test_eval_metrics_pass_at_4_unbiased_estimator_n_greater_than_4(self):
    with mock.patch.object(rl_utils, "is_sharing_weights", return_value=True):
      learner, rl_engine = self._create_mock_eval_learner(
          eval_num_generations=6,
          eval_rewards_to_inject=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # n=6, c=1
      )

      async def mock_producer(*args, **kwargs):
        yield [types.SimpleNamespace(group_id=0)]

      with mock.patch.object(learner, "_orchestrator_producer", side_effect=mock_producer), \
           mock.patch.object(learner, "_build_orchestrator", return_value=mock.Mock()):
        learner.train(train_dataset=[{"prompt": ["p1"]}], eval_dataset=[{"prompt": ["e1"]}])

      eval_metrics = None
      for call in rl_engine.buffer_metrics_async.call_args_list:
        args, kwargs = call
        if kwargs.get("mode") == rl_engine_lib.Mode.EVAL:
          eval_metrics = args[0]
          break

      self.assertIsNotNone(eval_metrics)
      # 1.0 - comb(6 - 1, 4) / comb(6, 4) = 1.0 - 5 / 15 = 10 / 15 = 2/3
      expected_pass_at_4 = 1.0 - math.comb(5, 4) / math.comb(6, 4)
      self.assertAlmostEqual(eval_metrics["diagnostics/pass_at_4"][0], expected_pass_at_4)

      # Subtest when n - c < 4 (e.g. c=3, n=6 -> n - c = 3 < 4)
      learner_high_c, rl_engine_high_c = self._create_mock_eval_learner(
          eval_num_generations=6,
          eval_rewards_to_inject=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
      )
      with mock.patch.object(learner_high_c, "_orchestrator_producer", side_effect=mock_producer), \
           mock.patch.object(learner_high_c, "_build_orchestrator", return_value=mock.Mock()):
        learner_high_c.train(train_dataset=[{"prompt": ["p1"]}], eval_dataset=[{"prompt": ["e1"]}])

      eval_metrics_high_c = None
      for call in rl_engine_high_c.buffer_metrics_async.call_args_list:
        args, kwargs = call
        if kwargs.get("mode") == rl_engine_lib.Mode.EVAL:
          eval_metrics_high_c = args[0]
          break

      self.assertIsNotNone(eval_metrics_high_c)
      self.assertEqual(eval_metrics_high_c["diagnostics/pass_at_4"][0], 1.0)

  def test_eval_metrics_pass_at_4_omitted_when_num_generations_less_than_4(self):
    with mock.patch.object(rl_utils, "is_sharing_weights", return_value=True):
      learner, rl_engine = self._create_mock_eval_learner(
          eval_num_generations=2,
          eval_rewards_to_inject=[0.0, 1.0],
      )

      async def mock_producer(*args, **kwargs):
        yield [types.SimpleNamespace(group_id=0)]

      with mock.patch.object(learner, "_orchestrator_producer", side_effect=mock_producer), \
           mock.patch.object(learner, "_build_orchestrator", return_value=mock.Mock()):
        learner.train(train_dataset=[{"prompt": ["p1"]}], eval_dataset=[{"prompt": ["e1"]}])

      eval_metrics = None
      for call in rl_engine.buffer_metrics_async.call_args_list:
        args, kwargs = call
        if kwargs.get("mode") == rl_engine_lib.Mode.EVAL:
          eval_metrics = args[0]
          break

      self.assertIsNotNone(eval_metrics)
      self.assertIn("diagnostics/pass_at_1", eval_metrics)
      self.assertNotIn("diagnostics/pass_at_4", eval_metrics)

  def test_eval_metrics_pass_at_4_omitted_when_reward_count_less_than_num_generations(self):
    with mock.patch.object(rl_utils, "is_sharing_weights", return_value=True):
      learner, rl_engine = self._create_mock_eval_learner(
          eval_num_generations=4,
          eval_rewards_to_inject=[0.0, 1.0],  # size 2 < 4
      )

      async def mock_producer(*args, **kwargs):
        yield [types.SimpleNamespace(group_id=0)]

      with mock.patch.object(learner, "_orchestrator_producer", side_effect=mock_producer), \
           mock.patch.object(learner, "_build_orchestrator", return_value=mock.Mock()):
        learner.train(train_dataset=[{"prompt": ["p1"]}], eval_dataset=[{"prompt": ["e1"]}])

      eval_metrics = None
      for call in rl_engine.buffer_metrics_async.call_args_list:
        args, kwargs = call
        if kwargs.get("mode") == rl_engine_lib.Mode.EVAL:
          eval_metrics = args[0]
          break

      self.assertIsNotNone(eval_metrics)
      self.assertIn("diagnostics/pass_at_1", eval_metrics)
      self.assertNotIn("diagnostics/pass_at_4", eval_metrics)


if __name__ == "__main__":
  absltest.main()
