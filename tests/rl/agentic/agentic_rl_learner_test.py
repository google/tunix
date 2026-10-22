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
import types
from typing import Any
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import agentic_grpo_learner
from tunix.rl.agentic import agentic_rl_learner
from tunix.rl.rollout import base_rollout


class DummyLearner(agentic_rl_learner.AgenticRLLearner):

  def _process_results(self, **kwargs):
    return []


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
        ValueError,
        r"max_tokens_to_generate \(10\) must match AgenticRLConfig"
        r" max_response_length \(20\)",
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

    with self.assertRaisesRegex(ValueError, r"must have return_logprobs=True"):
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
    with mock.patch.object(rl_utils, "is_sharing_weights", return_value=False):
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
      train_dataset = [{"prompt": ["p1"]}]
      with self.assertRaisesRegex(
          ValueError,
          r"compute_logps_micro_batch_size \(2\) must be equal to"
          r" train_micro_batch_size \(1\)",
      ):
        learner.train(train_dataset)

  def test_train_with_packing_executes_end_to_end(self):
    with mock.patch.object(rl_utils, "is_sharing_weights", return_value=False):
      rl_engine = mock.Mock()
      rl_engine.cluster_config = mock.Mock()
      mesh = mock.Mock()
      mesh.shape = {"fsdp": 1, "dp": 1}
      rl_engine.cluster_config.role_to_mesh = {
          rl_engine_lib.Role.ACTOR: mesh,
          rl_engine_lib.Role.ROLLOUT: mesh,
      }
      training_config = mock.Mock()
      training_config.compute_logps_micro_batch_size = 1
      training_config.train_micro_batch_size = 3
      training_config.mini_batch_size = 4
      training_config.max_seq_token_per_tpu = 16  # Enable packing
      training_config.max_segments_per_packed_row = None
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
      self.assertTrue(learner._packing_enabled)
      train_dataset = [{"prompt": ["p1", "p2", "p3", "p4"]}]

      async def mock_producer(*args, **kwargs):
        if False:
          yield

      with (
          mock.patch.object(
              learner, "_orchestrator_producer", side_effect=mock_producer
          ),
          mock.patch.object(
              learner,
              "_data_consumer_batch_generator",
              wraps=learner._data_consumer_batch_generator,
          ) as spy_consumer_gen,
      ):
        learner.train(train_dataset)
        spy_consumer_gen.assert_called_once_with(mock.ANY, 4)


class ExactTokenContinuityConfigTest(absltest.TestCase):

  def test_default_exact_token_continuity_reaches_the_collector(self):
    config = base_rollout.RolloutConfig(
        max_tokens_to_generate=1024, return_logprobs=True
    )
    engine = mock.MagicMock()
    engine.cluster_config.rollout_engine = "generic"
    engine.cluster_config.rollout_config = config
    engine.cluster_config.training_config.max_seq_token_per_tpu = None
    with mock.patch.object(rl_utils, "is_sharing_weights", return_value=False):
      learner = DummyLearner(
          rl_engine=engine,
          reward_fns=mock.Mock(),
          algo_config=agentic_grpo_learner.GRPOConfig(),
          chat_parser=object(),
      )
    self.assertIs(learner.algo_config.exact_token_continuity, True)

    for enabled in (False, True):
      obj = types.SimpleNamespace(
          algo_config=agentic_rl_learner.AgenticRLConfig(
              exact_token_continuity=enabled
          ),
          _model_call=mock.Mock(),
          tokenizer=mock.Mock(),
          chat_parser=mock.Mock(),
          rl_engine=types.SimpleNamespace(perf_v2=mock.Mock()),
          _rollout_sync_lock=mock.Mock(),
      )
      with mock.patch.object(
          agentic_rl_learner.rollout_orchestrator, "RolloutOrchestrator"
      ) as factory:
        agentic_rl_learner.AgenticRLLearner._build_orchestrator(obj)
      self.assertIs(
          factory.call_args.kwargs["engine_kwargs"]["exact_token_continuity"],
          enabled,
      )

  def test_exact_mode_rejects_rollout_configs_it_cannot_honor(
      self,
  ):
    for option, value, max_seq_token, message in (
        ("return_logprobs", False, None, "sampled logprobs"),
        ("return_routed_experts", True, 1024, "expert routing"),
    ):
      config = base_rollout.RolloutConfig(
          max_tokens_to_generate=1024, return_logprobs=True
      )
      setattr(config, option, value)
      engine = types.SimpleNamespace(
          rollout=types.SimpleNamespace(),
          tokenizer=object(),
          cluster_config=types.SimpleNamespace(
              rollout_config=config,
              training_config=types.SimpleNamespace(
                  max_seq_token_per_tpu=max_seq_token
              ),
          ),
      )
      with self.assertRaisesRegex(ValueError, message):
        agentic_grpo_learner.GRPOLearner(
            engine,
            agentic_grpo_learner.GRPOConfig(
                exact_token_continuity=True, use_rollout_logps=False
            ),
            chat_parser=object(),
        )

  def test_model_call_forwards_token_ids_without_parsing(self):
    obj = types.SimpleNamespace(
        algo_config=agentic_rl_learner.AgenticRLConfig(
            exact_token_continuity=True
        ),
        chat_parser=types.SimpleNamespace(
            parse=mock.Mock(side_effect=AssertionError("history parse"))
        ),
        rl_engine=types.SimpleNamespace(generate=mock.Mock()),
        policy_version=7,
    )
    agentic_rl_learner.AgenticRLLearner._model_call(
        obj, None, prompt_token_ids=np.array([0, 3])
    )
    obj.chat_parser.parse.assert_not_called()
    sent = obj.rl_engine.generate.call_args.kwargs
    self.assertIsNone(sent["prompts"])
    self.assertFalse(sent["apply_chat_template"])
    np.testing.assert_array_equal(sent["prompt_token_ids"], [[0, 3]])

  def test_model_call_parses_messages_with_numpy_task_fields_as_strings(self):
    qwen_parser = agentic_rl_learner.agentic_utils.chat_template_parser.QwenChatTemplateParser(
        types.SimpleNamespace(bos_token="", eos_token="<|im_end|>")
    )
    # Micro-batching leaves each task field as a 1-element array, and the
    # default agent copies the "prompts" field into the user message as is.
    raw_messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": np.array(["What is 1 + 1?"])},
    ]
    # Without unwrapping before `parse()`, BaseChatTemplateParser stringifies
    # the 1-element array as `"['What is 1 + 1?']"` in the rendered prompt.
    self.assertEqual(
        qwen_parser.parse(
            messages=raw_messages,
            add_generation_prompt=True,
            is_first_msg=True,
        ),
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\n['What is 1 + 1?']<|im_end|>\n"
        "<|im_start|>assistant\n",
    )

    obj = types.SimpleNamespace(
        chat_parser=qwen_parser,
        rl_engine=types.SimpleNamespace(generate=mock.Mock()),
        policy_version=7,
    )
    agentic_rl_learner.AgenticRLLearner._model_call(obj, raw_messages)
    self.assertEqual(
        obj.rl_engine.generate.call_args.kwargs["prompts"],
        [
            "<|im_start|>system\n<|im_end|>\n"
            "<|im_start|>user\nWhat is 1 + 1?<|im_end|>\n"
            "<|im_start|>assistant\n"
        ],
    )

  def test_model_call_without_messages_or_token_ids_raises(self):
    obj = types.SimpleNamespace(
        chat_parser=types.SimpleNamespace(parse=mock.Mock()),
        rl_engine=types.SimpleNamespace(generate=mock.Mock()),
        policy_version=7,
    )
    with self.assertRaisesRegex(ValueError, "chat_lists is required"):
      agentic_rl_learner.AgenticRLLearner._model_call(obj, None)
    obj.chat_parser.parse.assert_not_called()
    obj.rl_engine.generate.assert_not_called()

  def test_model_call_rejects_message_fields_that_are_not_strings(self):
    for content, error in (
        (np.array([1]), "must be a string"),
        (np.array(["a", "b"]), "size 1"),
    ):
      with self.subTest(content=content):
        obj = types.SimpleNamespace(
            chat_parser=types.SimpleNamespace(parse=mock.Mock()),
            rl_engine=types.SimpleNamespace(generate=mock.Mock()),
            policy_version=7,
        )
        with self.assertRaisesRegex(ValueError, error):
          agentic_rl_learner.AgenticRLLearner._model_call(
              obj, [{"role": "user", "content": content}]
          )
        obj.chat_parser.parse.assert_not_called()
        obj.rl_engine.generate.assert_not_called()

  def test_model_call_without_chat_parser_sends_string_messages(self):
    obj = types.SimpleNamespace(
        chat_parser=None,
        rl_engine=types.SimpleNamespace(generate=mock.Mock()),
        policy_version=7,
    )
    agentic_rl_learner.AgenticRLLearner._model_call(
        obj, [{"role": "user", "content": np.array(["q"])}]
    )
    sent = obj.rl_engine.generate.call_args.kwargs
    self.assertTrue(sent["apply_chat_template"])
    self.assertEqual(sent["prompts"], [[{"role": "user", "content": "q"}]])
    self.assertEqual([type(m["content"]) for m in sent["prompts"][0]], [str])


if __name__ == "__main__":
  absltest.main()
