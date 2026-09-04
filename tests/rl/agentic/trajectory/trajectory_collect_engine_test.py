# Copyright 2025 Google LLC
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

import asyncio
import contextlib
import io
import os
from pathlib import Path
import tempfile
import time
from unittest import mock

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import tracer as perf_tracer_v2
from tunix.generate import base_sampler
from tunix.rl import utils as rl_utils
from tunix.rl.agentic import utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.rollout import base_rollout

RolloutOutput = base_rollout.RolloutOutput


class TrajectoryCollectEngineTest(absltest.TestCase):

  class _TestEnv(base_environment.BaseTaskEnv):
    """Dummy class to expose reward_fn to autospec."""

    reward_fn = None
    final_reward_fn = None

  def setUp(self):
    super().setUp()
    trajectory_collect_engine.token_continuity._reset_token_collection_for_test()
    self.mock_agent = mock.create_autospec(
        base_agent.ConversationAgentBase, instance=True
    )
    self.mock_env = mock.create_autospec(self._TestEnv, instance=True)

    self.mock_env.max_steps = 10

    self.mock_model_call = mock.Mock()
    self.mock_env.final_reward_fn = mock.Mock(return_value=0.5)
    self.mock_final_reward_fn = self.mock_env.final_reward_fn
    self.mock_tokenizer = mock.Mock()
    self.mock_tokenizer.encode.return_value = [1, 2, 3]
    self.mock_chat_parser = mock.Mock()
    self.mock_chat_parser.update_assistant_end_tokens.side_effect = (
        lambda tokens: (tokens, 0)
    )

    self.trajectory = agent_types.Trajectory()
    self.mock_agent.trajectory = self.trajectory

    self._chat_history = []
    self.mock_agent.chat_completions = self._chat_history

    self.current_step = None

    def _update_from_model(resp):
      self.current_step = agent_types.Step(
          model_response=resp, action=agent_types.Action(action=['action'])
      )
      self.trajectory.steps.append(self.current_step)
      self._chat_history.append({'role': 'assistant', 'content': resp})
      return self.current_step

    def _update_from_env(observation, reward, done, info):
      if self.current_step:
        self.current_step.observation = observation
        self.current_step.reward = reward
        self.current_step.done = done
        self.current_step.info = info
      self._chat_history.append({'role': 'user', 'content': observation})

    def _reset_agent():
      self.trajectory.steps.clear()
      self._chat_history.clear()  # Clear the local list
      self.current_step = None

    self.mock_agent.update_from_model.side_effect = _update_from_model
    self.mock_agent.update_from_env.side_effect = _update_from_env
    self.mock_agent.reset.side_effect = _reset_agent
    self.mock_agent.get_current_step.side_effect = lambda: self.current_step

    # Configure mock env
    self.mock_env.reset.return_value = ('initial_obs', {})
    self.mock_env.step.side_effect = [
        ('obs1', 1.0, False, {}),
        ('obs2', 2.0, True, {}),
    ]
    self.mock_env.task = {'some': 'task'}
    self.mock_env.extra_kwargs = {}
    self.trajectory.task = self.mock_env.task

    def _mock_rollout_output(text, tokens):
      return RolloutOutput(
          text=[text],
          logits=[jnp.zeros_like(tokens)],
          tokens=[tokens],
          left_padded_prompt_tokens=np.array([[0, 0, 101]]),
          logprobs=[np.ones_like(tokens)],
          prompt_lengths=np.array([1], dtype=np.int32),
      )

    # Configure mock model call
    self.mock_model_call.side_effect = [
        _mock_rollout_output('response1', np.array([201, 202])),
        _mock_rollout_output('response2', np.array([203, 204])),
        _mock_rollout_output('response3', np.array([205, 206])),
        _mock_rollout_output('response4', np.array([207, 208])),
        _mock_rollout_output('response5', np.array([209, 210])),
    ]

  async def _run_collect(self, engine, mode='Trajectory'):
    return await engine.collect(mode=mode)

  def test_get_perf_tags(self):
    self.mock_env.extra_kwargs = {
        'group_id': 'test_group',
        'pair_index': 42,
    }
    self.mock_env.task = {
        'policy_version': 'v1.0',
    }
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
    )
    tags = engine._get_perf_tags()
    expected_tags = {
        perf_constants.GROUP_ID: 'test_group',
        perf_constants.PAIR_INDEX: 42,
        perf_constants.STEP: 'v1.0',
    }
    self.assertEqual(tags, expected_tags)

  def test_get_perf_tags_missing_attributes(self):
    del self.mock_env.extra_kwargs
    del self.mock_env.task
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
    )
    tags = engine._get_perf_tags()
    self.assertEqual(tags, {})

  def test_perf_v2_and_noop_used_by_default(self):
    self.mock_env.max_steps = 1
    self.mock_env.step.return_value = ('obs1', 1.0, True, {})
    self.mock_env.extra_kwargs = {'group_id': 'test_group'}

    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
    )
    self.assertIsInstance(engine.perf_v2, perf_tracer_v2.NoopTracer)
    with mock.patch.object(engine.perf_v2, 'span', autospec=True) as mock_span:
      mock_span.return_value.__enter__.return_value = (
          perf_tracer_v2.AsyncWaitlist()
      )
      asyncio.run(self._run_collect(engine, mode='Trajectory'))
      mock_span.assert_called_once_with(
          perf_constants.ENVIRONMENT,
          tags={perf_constants.GROUP_ID: 'test_group'},
      )

  def test_collect_trajectory_mode(self):
    self.mock_env.max_steps = 5
    self.mock_env.reward_fn.return_value = 0.5
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        gamma=0.9,
    )
    result_traj = asyncio.run(self._run_collect(engine, mode='Trajectory'))

    self.assertLen(result_traj.steps, 2)
    self.assertEqual(self.mock_env.reset.call_count, 1)
    self.assertEqual(self.mock_env.step.call_count, 2)
    self.assertEqual(self.mock_model_call.call_count, 2)
    self.mock_env.final_reward_fn.assert_called_once_with()
    self.mock_env.close.assert_called_once()

    # Check rewards and returns
    # Step 2: reward = 2.0 (from env) + 0.5 (final) = 2.5
    # Step 1: reward = 1.0 (from env)
    self.assertEqual(result_traj.steps[0].reward, 1.0)
    self.assertEqual(result_traj.steps[1].reward, 2.5)

    # Check env_time (mocked thread_time delta)
    self.assertIsInstance(result_traj.env_time, dict)
    self.assertGreaterEqual(result_traj.env_time['step_latency'], 0.0)
    self.assertGreaterEqual(result_traj.env_time['reset_latency'], 0.0)
    self.assertIsInstance(result_traj.reward_time, dict)
    self.assertGreaterEqual(result_traj.reward_time['reward_latency'], 0.0)

    # Check returns (gamma=0.9)
    # G_2 = 2.5
    # G_1 = 1.0 + 0.9 * 2.5 = 1.0 + 2.25 = 3.25
    self.assertAlmostEqual(result_traj.steps[1].mc_return, 2.5)
    self.assertAlmostEqual(result_traj.steps[0].mc_return, 3.25)
    self.assertAlmostEqual(result_traj.reward, 3.5)  # 1.0 + 2.5

  def test_collect_with_list_logprobs(self):
    # Test that it works with logprobs as a list (which doesn't have .size)
    self.mock_env.max_steps = 1
    self.mock_env.step.side_effect = [
        ('obs1', 1.0, True, {}),
    ]

    def _mock_rollout_output_list_logprobs(text, tokens):
      return RolloutOutput(
          text=[text],
          logits=[jnp.zeros_like(tokens)],
          tokens=[tokens],
          left_padded_prompt_tokens=np.array([1]),
          logprobs=[[0.1] * len(tokens)],  # logprobs as a list
      )

    self.mock_model_call.side_effect = [
        _mock_rollout_output_list_logprobs('resp', np.array([1, 2]))
    ]

    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
    )
    # This should not raise AttributeError: 'list' object has no attribute
    # 'size'
    result_traj = asyncio.run(
        self._run_collect(engine, mode='Trajectory')
    )
    self.assertLen(result_traj.steps, 1)
    self.assertEqual(len(result_traj.steps[0].logprobs), 2)

  def test_collect_conversation_mode(self):
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        max_response_length=1024,
    )
    conversation = asyncio.run(self._run_collect(engine, mode='Conversation'))

    expected_conversation = [
        {'role': 'user', 'content': 'initial_obs'},
        {'role': 'assistant', 'content': 'response1'},
        {'role': 'user', 'content': 'obs1'},
        {'role': 'assistant', 'content': 'response2'},
        {'role': 'user', 'content': 'obs2'},
    ]
    self.assertEqual(conversation, expected_conversation)

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_collect_with_tokenization(self, mock_convert):
    mock_convert.side_effect = [
        ([101], [1]),  # prompt tokens
        ([301, 302], [1, 1]),  # env tokens 1
        ([303, 304], [1, 1]),  # env tokens 2
    ]
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=self.mock_chat_parser,
        max_response_length=1024,
    )
    token_data = asyncio.run(self._run_collect(engine, mode='Token'))
    expected_tokens = {
        'conversation_text': [
            {'role': 'user', 'content': 'initial_obs'},
            {'role': 'assistant', 'content': 'response1'},
            {'role': 'user', 'content': 'obs1'},
            {'role': 'assistant', 'content': 'response2'},
            {'role': 'user', 'content': 'obs2'},
        ],
        'prompt_tokens': np.array([0, 0, 101]),
        'prompt_length': 1,
        'conversation_tokens': np.array(
            [201, 202, 301, 302, 203, 204]
        ),
        'conversation_masks': np.array([1, 1, 1, 1, 1, 1]),
        'trajectory_reward': (
            3.5
        ),  # 1.0 + 2.0 + 0.5 (final reward from final_reward_fn)
        'env_time': {
            'reset_latency': 0.0,
            'step_latency': 0.0,
            'close_latency': 0.0,
        },
        'reward_time': {
            'reward_latency': 0.0,
        },
        'old_logprobs': np.array([1, 1, 0, 0, 1, 1]),
        'policy_version': None,
        'original_input': {'some': 'task'},
        'group_id': None,
        'status': 'SUCCEEDED',
    }

    for k, v in expected_tokens.items():
      if k in ['env_time', 'reward_time']:
        self.assertIsInstance(token_data[k], dict)
        for sub_k in v:
          self.assertGreaterEqual(token_data[k][sub_k], 0.0)
      elif isinstance(v, np.ndarray):
        np.testing.assert_array_equal(token_data[k], v)
      else:
        self.assertEqual(token_data[k], v, msg=f'Failed for key: {k}')

    # The function using the parser is mocked, so the parser itself is not
    # called. Instead, we check that the parser is passed as an argument.
    self.assertTrue(mock_convert.called)
    for call in mock_convert.call_args_list:
      self.assertIs(call.kwargs['parser'], self.mock_chat_parser)

    # Verify that the initial prompt tokenization in _reset is called with
    # contains_first_msg=True and contains_generation_msg=True.
    self.assertGreaterEqual(mock_convert.call_count, 2)
    self.assertTrue(
        mock_convert.call_args_list[0].kwargs['contains_first_msg'],
        'contains_first_msg should be True for initial prompt tokenization',
    )
    self.assertTrue(
        mock_convert.call_args_list[0].kwargs['contains_generation_msg'],
        'contains_generation_msg should be True for initial prompt'
        ' tokenization',
    )

    # Verify that tokenization for environment observations
    # has contains_generation_msg=True.
    self.assertEqual(mock_convert.call_count, 2)
    self.assertTrue(
        mock_convert.call_args_list[1].kwargs['contains_generation_msg']
    )

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_deepswe_continuation_reuses_exact_sampled_and_environment_tokens(
      self, mock_convert
  ):
    mock_convert.side_effect = [
        ([101], [0]),
        ([301, 302], [0, 0]),
    ]
    with mock.patch.object(
        trajectory_collect_engine.deepswe_debug,
        'deepswe_exact_token_continuity',
        return_value=True,
    ):
      engine = trajectory_collect_engine.TrajectoryCollectEngine(
          agent=self.mock_agent,
          env=self.mock_env,
          model_call=self.mock_model_call,
          tokenizer=self.mock_tokenizer,
          chat_parser=self.mock_chat_parser,
          max_response_length=1024,
      )
    asyncio.run(self._run_collect(engine, mode='Token'))

    first_call, second_call = self.mock_model_call.call_args_list
    self.assertNotIn('prompt_token_ids', first_call.kwargs)
    np.testing.assert_array_equal(
        second_call.kwargs['prompt_token_ids'],
        np.asarray([101, 201, 202, 301, 302], dtype=np.int32),
    )

  def test_deepswe_continuation_rejects_missing_environment_tokens(self):
    with mock.patch.object(
        trajectory_collect_engine.deepswe_debug,
        'deepswe_exact_token_continuity',
        return_value=True,
    ):
      engine = trajectory_collect_engine.TrajectoryCollectEngine(
          agent=self.mock_agent,
          env=self.mock_env,
          model_call=self.mock_model_call,
          tokenizer=self.mock_tokenizer,
          chat_parser=self.mock_chat_parser,
      )
    self.trajectory.prompt_tokens = np.asarray([0, 0, 101], dtype=np.int32)
    self.trajectory.prompt_length = 1
    self.trajectory.steps.append(
        agent_types.Step(
            assistant_tokens=np.asarray([201], dtype=np.int32), done=False
        )
    )
    engine._response_token_count = 1
    with self.assertRaisesRegex(ValueError, 'has no environment tokens'):
      engine._deepswe_continuation_prompt_token_ids()

  def test_deepswe_and_m15_exact_admissions_are_mutually_exclusive(self):
    with mock.patch.object(
        trajectory_collect_engine.deepswe_debug,
        'deepswe_exact_token_continuity',
        return_value=True,
    ), mock.patch.object(
        trajectory_collect_engine.token_continuity,
        'm15_token_continuity_mode',
        return_value='exact',
    ):
      with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
        trajectory_collect_engine.TrajectoryCollectEngine(
            agent=self.mock_agent,
            env=self.mock_env,
            model_call=self.mock_model_call,
            tokenizer=self.mock_tokenizer,
            chat_parser=self.mock_chat_parser,
        )

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_m15_verify_observes_drift_without_replacing_text_prompt(
      self, mock_convert
  ):
    mock_convert.side_effect = [
        ([101], [0]),
        ([301, 302], [0, 0]),
    ]
    with mock.patch.object(
        trajectory_collect_engine.token_continuity,
        'm15_token_continuity_mode',
        return_value='verify',
    ):
      engine = trajectory_collect_engine.TrajectoryCollectEngine(
          agent=self.mock_agent,
          env=self.mock_env,
          model_call=self.mock_model_call,
          tokenizer=self.mock_tokenizer,
          chat_parser=self.mock_chat_parser,
          max_response_length=1024,
      )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
      asyncio.run(self._run_collect(engine, mode='Token'))

    self.assertIn('verdict=TOKEN_STREAM_DIFFERENT', output.getvalue())
    self.assertIn('first_mismatch=1', output.getvalue())
    for model_call in self.mock_model_call.call_args_list:
      self.assertNotIn('prompt_token_ids', model_call.kwargs)

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_m15_verify_reports_exact_later_turn_prompt(self, mock_convert):
    mock_convert.side_effect = [
        ([101], [0]),
        ([301, 302], [0, 0]),
    ]

    def _rollout(text, tokens, prompt_tokens, prompt_length):
      return RolloutOutput(
          text=[text],
          logits=[jnp.zeros_like(tokens)],
          tokens=[tokens],
          left_padded_prompt_tokens=np.asarray([prompt_tokens]),
          logprobs=[np.ones_like(tokens)],
          prompt_lengths=np.asarray([prompt_length], dtype=np.int32),
      )

    self.mock_model_call.side_effect = [
        _rollout('response1', np.asarray([201, 202]), [0, 0, 101], 1),
        _rollout(
            'response2',
            np.asarray([203, 204]),
            [0, 101, 201, 202, 301, 302],
            5,
        ),
    ]
    with mock.patch.object(
        trajectory_collect_engine.token_continuity,
        'm15_token_continuity_mode',
        return_value='verify',
    ):
      engine = trajectory_collect_engine.TrajectoryCollectEngine(
          agent=self.mock_agent,
          env=self.mock_env,
          model_call=self.mock_model_call,
          tokenizer=self.mock_tokenizer,
          chat_parser=self.mock_chat_parser,
          max_response_length=1024,
      )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
      asyncio.run(self._run_collect(engine, mode='Token'))

    self.assertIn('verdict=TOKEN_STREAM_EQUAL', output.getvalue())
    self.assertIn('first_mismatch=-1', output.getvalue())
    for model_call in self.mock_model_call.call_args_list:
      self.assertNotIn('prompt_token_ids', model_call.kwargs)

  def test_m15_verify_rejects_caller_prompt_token_override(self):
    with mock.patch.object(
        trajectory_collect_engine.token_continuity,
        'm15_token_continuity_mode',
        return_value='verify',
    ), self.assertRaisesRegex(ValueError, 'owns prompt_token_ids'):
      trajectory_collect_engine.TrajectoryCollectEngine(
          agent=self.mock_agent,
          env=self.mock_env,
          model_call=self.mock_model_call,
          model_call_kwargs={'prompt_token_ids': np.asarray([101])},
      )

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_m15_exact_reuses_and_verifies_exact_turn_tokens(
      self, mock_convert
  ):
    mock_convert.side_effect = [
        ([101], [0]),
        ([301, 302], [0, 0]),
    ]

    def _rollout(text, tokens, prompt_tokens, prompt_length):
      return RolloutOutput(
          text=[text],
          logits=[jnp.zeros_like(tokens)],
          tokens=[tokens],
          left_padded_prompt_tokens=np.asarray([prompt_tokens]),
          logprobs=[np.ones_like(tokens)],
          prompt_lengths=np.asarray([prompt_length], dtype=np.int32),
      )

    self.mock_model_call.side_effect = [
        _rollout('response1', np.asarray([201, 202]), [0, 0, 101], 1),
        _rollout(
            'response2',
            np.asarray([203, 204]),
            [0, 101, 201, 202, 301, 302],
            5,
        ),
    ]
    with mock.patch.object(
        trajectory_collect_engine.token_continuity,
        'm15_token_continuity_mode',
        return_value='exact',
    ):
      engine = trajectory_collect_engine.TrajectoryCollectEngine(
          agent=self.mock_agent,
          env=self.mock_env,
          model_call=self.mock_model_call,
          tokenizer=self.mock_tokenizer,
          chat_parser=self.mock_chat_parser,
          max_response_length=1024,
      )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
      asyncio.run(self._run_collect(engine, mode='Token'))

    first_call, second_call = self.mock_model_call.call_args_list
    self.assertNotIn('prompt_token_ids', first_call.kwargs)
    np.testing.assert_array_equal(
        second_call.kwargs['prompt_token_ids'],
        np.asarray([101, 201, 202, 301, 302], dtype=np.int32),
    )
    self.assertIn('mode=exact', output.getvalue())
    self.assertIn('verdict=TOKEN_STREAM_EQUAL', output.getvalue())

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_p45_exact_reuses_and_verifies_exact_turn_tokens(
      self, mock_convert
  ):
    mock_convert.side_effect = [
        ([101], [0]),
        ([301, 302], [0, 0]),
    ]

    def _rollout(text, tokens, prompt_tokens, prompt_length):
      return RolloutOutput(
          text=[text],
          logits=[jnp.zeros_like(tokens)],
          tokens=[tokens],
          left_padded_prompt_tokens=np.asarray([prompt_tokens]),
          logprobs=[np.ones_like(tokens)],
          prompt_lengths=np.asarray([prompt_length], dtype=np.int32),
      )

    self.mock_model_call.side_effect = [
        _rollout('response1', np.asarray([201, 202]), [0, 0, 101], 1),
        _rollout(
            'response2',
            np.asarray([203, 204]),
            [0, 101, 201, 202, 301, 302],
            5,
        ),
    ]
    contract = trajectory_collect_engine.token_continuity.FrozenLakeTokenContinuity(
        workload='p45',
        mode='exact',
        selector=(
            trajectory_collect_engine.token_continuity.P57_TOKEN_CONTINUITY_ENV
        ),
    )
    with mock.patch.object(
        trajectory_collect_engine.token_continuity,
        'frozenlake_token_continuity',
        return_value=contract,
    ):
      engine = trajectory_collect_engine.TrajectoryCollectEngine(
          agent=self.mock_agent,
          env=self.mock_env,
          model_call=self.mock_model_call,
          tokenizer=self.mock_tokenizer,
          chat_parser=self.mock_chat_parser,
          max_response_length=1024,
      )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
      asyncio.run(self._run_collect(engine, mode='Token'))

    first_call, second_call = self.mock_model_call.call_args_list
    self.assertNotIn('prompt_token_ids', first_call.kwargs)
    np.testing.assert_array_equal(
        second_call.kwargs['prompt_token_ids'],
        np.asarray([101, 201, 202, 301, 302], dtype=np.int32),
    )
    self.assertIn(
        '[CANON_P57_TOKEN_CONTINUITY] workload=p45 mode=exact',
        output.getvalue(),
    )
    self.assertIn('verdict=TOKEN_STREAM_EQUAL', output.getvalue())
    self.assertIn(
        '[CANON_P57_TOKEN_CONTINUITY_SUMMARY] workload=p45',
        output.getvalue(),
    )
    self.assertIn('expected_later_turns=1 receipts=1 verdict=PASS', output.getvalue())

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_p57_m15_exact_reuses_and_verifies_exact_turn_tokens(
      self, mock_convert
  ):
    mock_convert.side_effect = [
        ([101], [0]),
        ([301, 302], [0, 0]),
    ]

    def _rollout(text, tokens, prompt_tokens, prompt_length):
      return RolloutOutput(
          text=[text],
          logits=[jnp.zeros_like(tokens)],
          tokens=[tokens],
          left_padded_prompt_tokens=np.asarray([prompt_tokens]),
          logprobs=[np.ones_like(tokens)],
          prompt_lengths=np.asarray([prompt_length], dtype=np.int32),
      )

    self.mock_model_call.side_effect = [
        _rollout('response1', np.asarray([201, 202]), [0, 0, 101], 1),
        _rollout(
            'response2',
            np.asarray([203, 204]),
            [0, 101, 201, 202, 301, 302],
            5,
        ),
    ]
    contract = (
        trajectory_collect_engine.token_continuity.FrozenLakeTokenContinuity(
            workload='m15',
            mode='exact',
            selector=(
                trajectory_collect_engine.token_continuity.P57_TOKEN_CONTINUITY_ENV
            ),
        )
    )
    with mock.patch.object(
        trajectory_collect_engine.token_continuity,
        'frozenlake_token_continuity',
        return_value=contract,
    ):
      engine = trajectory_collect_engine.TrajectoryCollectEngine(
          agent=self.mock_agent,
          env=self.mock_env,
          model_call=self.mock_model_call,
          tokenizer=self.mock_tokenizer,
          chat_parser=self.mock_chat_parser,
          max_response_length=1024,
      )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
      asyncio.run(self._run_collect(engine, mode='Token'))

    first_call, second_call = self.mock_model_call.call_args_list
    self.assertNotIn('prompt_token_ids', first_call.kwargs)
    np.testing.assert_array_equal(
        second_call.kwargs['prompt_token_ids'],
        np.asarray([101, 201, 202, 301, 302], dtype=np.int32),
    )
    self.assertIn(
        '[CANON_P57_TOKEN_CONTINUITY] workload=m15 mode=exact',
        output.getvalue(),
    )
    self.assertIn('verdict=TOKEN_STREAM_EQUAL', output.getvalue())
    self.assertIn(
        '[CANON_P57_TOKEN_CONTINUITY_SUMMARY] workload=m15',
        output.getvalue(),
    )
    self.assertIn('expected_later_turns=1 receipts=1 verdict=PASS', output.getvalue())

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_m15_exact_fails_if_serving_consumes_different_tokens(
      self, mock_convert
  ):
    mock_convert.side_effect = [
        ([101], [0]),
        ([301, 302], [0, 0]),
    ]
    with mock.patch.object(
        trajectory_collect_engine.token_continuity,
        'm15_token_continuity_mode',
        return_value='exact',
    ):
      engine = trajectory_collect_engine.TrajectoryCollectEngine(
          agent=self.mock_agent,
          env=self.mock_env,
          model_call=self.mock_model_call,
          tokenizer=self.mock_tokenizer,
          chat_parser=self.mock_chat_parser,
          max_response_length=1024,
      )
    with self.assertRaisesRegex(ValueError, 'differs from the serving prompt'):
      asyncio.run(self._run_collect(engine, mode='Token'))

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_p57_first_diff_debug_persists_reconstructable_capsule(
      self, mock_convert
  ):
    mock_convert.side_effect = [
        ([101], [0]),
        ([301, 302], [0, 0]),
    ]
    self.mock_env.extra_kwargs = {'pair_index': 7, 'group_id': 9}
    contract = (
        trajectory_collect_engine.token_continuity.FrozenLakeTokenContinuity(
            workload='p45',
            mode='exact',
            selector=(
                trajectory_collect_engine.token_continuity.P57_TOKEN_CONTINUITY_ENV
            ),
        )
    )
    with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
        os.environ,
        {
            'CANON_STATE': tmp,
            'CANON_EXPECT_COMMIT': 'a' * 40,
            'CANON_CLIENT_IMAGE': 'sha256:' + 'b' * 64,
        },
    ), mock.patch.object(
        trajectory_collect_engine.token_continuity,
        'frozenlake_token_continuity',
        return_value=contract,
    ), mock.patch.object(
        trajectory_collect_engine.token_continuity,
        'frozenlake_token_continuity_debug_mode',
        return_value=(
            trajectory_collect_engine.token_continuity
            .P57_TOKEN_CONTINUITY_DEBUG_FIRST_DIFF
        ),
    ):
      engine = trajectory_collect_engine.TrajectoryCollectEngine(
          agent=self.mock_agent,
          env=self.mock_env,
          model_call=self.mock_model_call,
          tokenizer=self.mock_tokenizer,
          chat_parser=self.mock_chat_parser,
          max_response_length=1024,
      )
      output = io.StringIO()
      with contextlib.redirect_stdout(output), self.assertRaisesRegex(
          ValueError, 'differs from the serving prompt'
      ):
        asyncio.run(self._run_collect(engine, mode='Token'))
      log = output.getvalue()
      self.assertIn('[CANON_P57_TOKEN_CONTINUITY_DEBUG] ', log)
      self.assertIn('[CANON_P57_TOKEN_CONTINUITY_DEBUG_JSON] ', log)
      self.assertIn(
          '[CANON_P57_TOKEN_CONTINUITY_DEBUG_CAPSULE] verdict=PASS', log
      )
      capsules = list(
          (Path(tmp) / 'token-continuity-first-diff').glob('*.json')
      )
      self.assertLen(capsules, 1)
      capsule = (
          trajectory_collect_engine.token_continuity.debug_capsule_from_receipts(
              log.splitlines()
          )
      )
      self.assertEqual(capsule['header']['pair_index'], '7')
      self.assertEqual(capsule['header']['group_id'], '9')
      self.assertEqual(capsule['actual']['tokens'], [101])
      self.assertEqual(
          [
              token
              for segment in capsule['expected_segments']
              for token in segment['tokens']
          ],
          [101, 201, 202, 301, 302],
      )
      self.mock_model_call.side_effect = [
          RolloutOutput(
              text=['response1'],
              logits=[jnp.zeros(2)],
              tokens=[np.asarray([201, 202])],
              left_padded_prompt_tokens=np.asarray([[0, 0, 101]]),
              logprobs=[np.ones(2)],
              prompt_lengths=np.asarray([1], dtype=np.int32),
          ),
          RolloutOutput(
              text=['response2'],
              logits=[jnp.zeros(2)],
              tokens=[np.asarray([203, 204])],
              left_padded_prompt_tokens=np.asarray([[0, 0, 101]]),
              logprobs=[np.ones(2)],
              prompt_lengths=np.asarray([1], dtype=np.int32),
          ),
      ]
      self.mock_env.step.side_effect = [
          ('obs1', 1.0, False, {}),
          ('obs2', 2.0, True, {}),
      ]
      mock_convert.side_effect = [
          ([101], [0]),
          ([301, 302], [0, 0]),
      ]
      repeated_output = io.StringIO()
      with contextlib.redirect_stdout(repeated_output), self.assertRaisesRegex(
          ValueError, 'differs from the serving prompt'
      ):
        asyncio.run(self._run_collect(engine, mode='Token'))
      self.assertNotIn(
          '[CANON_P57_TOKEN_CONTINUITY_DEBUG] ',
          repeated_output.getvalue(),
      )
      self.assertLen(
          list((Path(tmp) / 'token-continuity-first-diff').glob('*.json')),
          1,
      )

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_p57_collect_diff_stops_only_trajectory_and_masks_tokens(
      self, mock_convert
  ):
    mock_convert.side_effect = [
        ([101], [0]),
        ([301, 302], [0, 0]),
    ]
    self.mock_env.extra_kwargs = {'pair_index': 7, 'group_id': 9}
    contract = (
        trajectory_collect_engine.token_continuity.FrozenLakeTokenContinuity(
            workload='p45',
            mode='exact',
            selector=(
                trajectory_collect_engine.token_continuity
                .P57_TOKEN_CONTINUITY_ENV
            ),
        )
    )
    witness = lambda request_id: base_sampler.PromptTokenWitness(
        request_id=request_id,
        submitted_tokens=1,
        submitted_sha256='a' * 64,
        engine_echo_tokens=1,
        engine_echo_sha256='a' * 64,
    )
    self.mock_model_call.side_effect = [
        RolloutOutput(
            text=['response1'],
            logits=[jnp.zeros(2)],
            tokens=[np.asarray([201, 202])],
            left_padded_prompt_tokens=np.asarray([[0, 0, 101]]),
            logprobs=[np.ones(2)],
            prompt_lengths=np.asarray([1], dtype=np.int32),
            prompt_token_witnesses=[witness('request-0')],
        ),
        RolloutOutput(
            text=['response2'],
            logits=[jnp.zeros(2)],
            tokens=[np.asarray([203, 204])],
            left_padded_prompt_tokens=np.asarray([[0, 0, 101]]),
            logprobs=[np.ones(2)],
            prompt_lengths=np.asarray([1], dtype=np.int32),
            prompt_token_witnesses=[witness('request-1')],
        ),
    ]
    debug_mode = (
        trajectory_collect_engine.token_continuity
        .P57_TOKEN_CONTINUITY_DEBUG_COLLECT
    )
    with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
        os.environ,
        {
            'CANON_STATE': tmp,
            'CANON_EXPECT_COMMIT': 'a' * 40,
            'CANON_CLIENT_IMAGE': 'sha256:' + 'b' * 64,
        },
    ), mock.patch.object(
        trajectory_collect_engine.token_continuity,
        'frozenlake_token_continuity',
        return_value=contract,
    ), mock.patch.object(
        trajectory_collect_engine.token_continuity,
        'frozenlake_token_continuity_debug_mode',
        return_value=debug_mode,
    ):
      trajectory_collect_engine.token_continuity.begin_token_continuity_collection()
      engine = trajectory_collect_engine.TrajectoryCollectEngine(
          agent=self.mock_agent,
          env=self.mock_env,
          model_call=self.mock_model_call,
          tokenizer=self.mock_tokenizer,
          chat_parser=self.mock_chat_parser,
          max_response_length=1024,
      )
      output = io.StringIO()
      with contextlib.redirect_stdout(output):
        token_data = asyncio.run(self._run_collect(engine, mode='Token'))

      self.assertEqual(
          token_data['status'],
          agent_types.TrajectoryStatus.TOKEN_CONTINUITY_DIFFERENT.name,
      )
      np.testing.assert_array_equal(
          token_data['conversation_masks'], np.zeros(4, dtype=np.int32)
      )
      self.assertEqual(self.mock_env.step.call_count, 1)
      self.mock_final_reward_fn.assert_not_called()
      self.assertIn('verdict=DIFFERENT', output.getvalue())
      self.assertNotIn(
          '[CANON_P57_TOKEN_CONTINUITY_DEBUG_JSON] ', output.getvalue()
      )
      self.assertNotIn('"tokens":', output.getvalue())
      self.assertLen(
          list((Path(tmp) / 'token-continuity-first-diff').glob('*.json')),
          1,
      )
      self.assertLen(
          list((Path(tmp) / 'p57_tito_witness' / 'host').glob('*.json')),
          2,
      )
      snapshot = (
          trajectory_collect_engine.token_continuity
          .token_collection_snapshot()
      )
      self.assertEqual(snapshot['trajectories'], 1)
      self.assertEqual(snapshot['different_trajectories'], 1)
      self.assertEqual(snapshot['capsules_reserved'], 1)
      self.assertEqual(snapshot['capsules_emitted'], 1)
      self.assertEqual(snapshot['emission_failures'], 0)
      for call in self.mock_model_call.call_args_list:
        self.assertTrue(call.kwargs['return_prompt_token_witnesses'])

  def test_p57_collect_cap_is_process_wide_and_allocated_before_io(self):
    continuity = trajectory_collect_engine.token_continuity
    with mock.patch.object(
        continuity,
        'frozenlake_token_continuity_debug_mode',
        return_value=continuity.P57_TOKEN_CONTINUITY_DEBUG_COLLECT,
    ):
      continuity.begin_token_continuity_collection()
    slots = [
        continuity.reserve_token_difference_capsule()
        for _ in range(continuity.P57_TOKEN_CONTINUITY_COLLECT_LIMIT + 1)
    ]
    self.assertEqual(
        slots[:-1],
        list(range(1, continuity.P57_TOKEN_CONTINUITY_COLLECT_LIMIT + 1)),
    )
    self.assertIsNone(slots[-1])
    snapshot = continuity.token_collection_snapshot()
    self.assertEqual(
        snapshot['capsules_reserved'],
        continuity.P57_TOKEN_CONTINUITY_COLLECT_LIMIT,
    )
    self.assertEqual(snapshot['capsules_omitted'], 1)

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_p57_record_full_diff_preserves_training_row_and_request_join(
      self, mock_convert
  ):
    mock_convert.side_effect = [
        ([101], [0]),
        ([301, 302], [0, 0]),
    ]
    self.mock_env.extra_kwargs = {'pair_index': 7, 'group_id': 9}
    self.mock_env.task = {'prompts': 'initial', 'policy_version': 3}
    self.trajectory.task = self.mock_env.task
    contract = (
        trajectory_collect_engine.token_continuity.FrozenLakeTokenContinuity(
            workload='p45',
            mode='exact',
            selector=(
                trajectory_collect_engine.token_continuity
                .P57_TOKEN_CONTINUITY_ENV
            ),
        )
    )

    def witness(request_id, tokens, echoed=None):
      values = np.asarray(tokens, dtype=np.int32)
      echoed_values = np.asarray(
          tokens if echoed is None else echoed, dtype=np.int32
      )
      digest = trajectory_collect_engine.token_continuity._prompt_witness_digest(
          values
      )
      echo_digest = trajectory_collect_engine.token_continuity._prompt_witness_digest(
          echoed_values
      )
      return base_sampler.PromptTokenWitness(
          request_id=request_id,
          submitted_tokens=len(values),
          submitted_sha256=digest,
          engine_echo_tokens=len(echoed_values),
          engine_echo_sha256=echo_digest,
          submitted_token_ids=tuple(values.tolist()),
          engine_echo_token_ids=tuple(echoed_values.tolist()),
      )

    expected_later = [101, 201, 202, 301, 302]
    self.mock_model_call.side_effect = [
        RolloutOutput(
            text=['response1'],
            logits=[jnp.zeros(2)],
            tokens=[np.asarray([201, 202])],
            left_padded_prompt_tokens=np.asarray([[0, 0, 101]]),
            logprobs=[np.ones(2)],
            prompt_lengths=np.asarray([1], dtype=np.int32),
            prompt_token_witnesses=[witness('request-0', [101], [102])],
        ),
        RolloutOutput(
            text=['response2'],
            logits=[jnp.zeros(2)],
            tokens=[np.asarray([203, 204])],
            left_padded_prompt_tokens=np.asarray([[0, 0, 101]]),
            logprobs=[np.ones(2)],
            prompt_lengths=np.asarray([1], dtype=np.int32),
            prompt_token_witnesses=[witness('request-1', expected_later)],
        ),
    ]
    debug_mode = (
        trajectory_collect_engine.token_continuity
        .P57_TOKEN_CONTINUITY_DEBUG_RECORD_FULL
    )
    with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
        os.environ,
        {
            'CANON_STATE': tmp,
            'CANON_EXPECT_COMMIT': 'a' * 40,
            'CANON_CLIENT_IMAGE': 'sha256:' + 'b' * 64,
            'CANON_DP_SIZE': '8',
            'CANON_TP_SIZE': '8',
        },
    ), mock.patch.object(
        trajectory_collect_engine.token_continuity,
        'frozenlake_token_continuity',
        return_value=contract,
    ), mock.patch.object(
        trajectory_collect_engine.token_continuity,
        'frozenlake_token_continuity_debug_mode',
        return_value=debug_mode,
    ):
      trajectory_collect_engine.token_continuity.begin_token_continuity_collection()
      engine = trajectory_collect_engine.TrajectoryCollectEngine(
          agent=self.mock_agent,
          env=self.mock_env,
          model_call=self.mock_model_call,
          tokenizer=self.mock_tokenizer,
          chat_parser=self.mock_chat_parser,
          max_response_length=1024,
      )
      token_data = asyncio.run(self._run_collect(engine, mode='Token'))

      self.assertEqual(token_data['status'], 'SUCCEEDED')
      np.testing.assert_array_equal(
          token_data['conversation_masks'],
          np.asarray([1, 1, 0, 0, 1, 1]),
      )
      self.assertEqual(self.mock_env.step.call_count, 2)
      self.mock_final_reward_fn.assert_called_once()
      self.assertTrue(token_data['p57_token_continuity_different'])
      self.assertEqual(token_data['p57_token_continuity_later_turns'], 1)
      self.assertEqual(
          token_data['p57_token_continuity_request_ids'],
          ('request-0', 'request-1'),
      )
      capsules = list(
          (Path(tmp) / 'token-continuity-first-diff').glob('*.json')
      )
      self.assertLen(capsules, 1)
      snapshot = (
          trajectory_collect_engine.token_continuity
          .token_collection_snapshot()
      )
      self.assertEqual(snapshot['different_trajectories'], 1)
      self.assertEqual(snapshot['capsules_reserved'], 1)
      self.assertEqual(snapshot['engine_echo_differences'], 1)

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_collect_token_mode_empty_steps(self, mock_convert):
    mock_convert.side_effect = [
        ([101], [1]),  # prompt tokens
    ]
    self.mock_env.max_steps = 0  # No steps will be taken
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=self.mock_chat_parser,
        max_response_length=1024,
    )
    token_data = asyncio.run(self._run_collect(engine, mode='Token'))
    self.assertEmpty(self.mock_agent.trajectory.steps)
    np.testing.assert_array_equal(
        token_data['conversation_tokens'], np.array([], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        token_data['conversation_masks'], np.array([], dtype=np.int32)
    )
    self.assertIsNone(token_data['old_logprobs'])

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_collect_with_incomplete_tokenizer_config_skips_tokenization(
      self, mock_tokenize
  ):
    # Scenario 1: Tokenizer is missing, but chat parser is present.
    # Tokenization should be skipped as both are required.
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=None,
        chat_parser=self.mock_chat_parser,
    )
    asyncio.run(self._run_collect(engine))
    mock_tokenize.assert_not_called()

    # Reset mocks for the next scenario.
    self.setUp()
    mock_tokenize.reset_mock()

    # Scenario 2: Chat parser is missing, but tokenizer is present.
    # Tokenization should be skipped as both are required.
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=None,
    )
    asyncio.run(self._run_collect(engine))
    mock_tokenize.assert_not_called()

  async def _run_collect_multiple(self, engine_args, pairs):
    results = []
    async for (
        i,
        traj,
    ) in trajectory_collect_engine.TrajectoryCollectEngine.collect_multiple(
        pairs, **engine_args
    ):
      results.append((i, traj))
    return results

  def test_collect_multiple(self):
    # Helper to configure a new mock agent
    def configure_mock_agent(initial_obs):
      agent = mock.create_autospec(
          base_agent.ConversationAgentBase, instance=True
      )
      traj = agent_types.Trajectory()
      agent.trajectory = traj
      agent.chat_completions = []
      current_step = [None]

      def _update_from_model(resp):
        step = agent_types.Step(
            model_response=resp, action=agent_types.Action(action=['action'])
        )
        traj.steps.append(step)
        current_step[0] = step
        agent.chat_completions.append({'role': 'assistant', 'content': resp})
        return step

      def _update_from_env(observation, reward, done, info):
        if current_step[0]:
          current_step[0].observation = observation
          current_step[0].reward = reward
          current_step[0].done = done
          current_step[0].info = info
        agent.chat_completions.append({'role': 'user', 'content': observation})

      agent.update_from_model.side_effect = _update_from_model
      agent.update_from_env.side_effect = _update_from_env
      agent.get_current_step.side_effect = lambda: current_step[0]

      def _reset_agent():
        traj.steps.clear()
        agent.chat_completions.clear()

      agent.reset.side_effect = _reset_agent
      return agent

    agent1 = configure_mock_agent('initial1')
    env1 = mock.create_autospec(self._TestEnv, instance=True)
    env1.final_reward_fn = mock.Mock(return_value=0.5)
    env1.reset.return_value = ('initial1', {})
    env1.step.return_value = ('obs1', 1.0, True, {})
    env1.task = {}
    env1.extra_kwargs = {}
    env1.max_steps = 5

    agent2 = configure_mock_agent('initial2')
    env2 = mock.create_autospec(self._TestEnv, instance=True)
    env2.final_reward_fn = mock.Mock(return_value=0.5)
    env2.reset.return_value = ('initial2', {})
    env2.step.side_effect = [
        ('obs2a', 2.0, False, {}),
        ('obs2b', 2.1, True, {}),
    ]
    env2.task = {}
    env2.extra_kwargs = {}
    env2.max_steps = 5

    pairs = [(agent1, env1), (agent2, env2)]
    engine_args = {
        'model_call': self.mock_model_call,
        'mode': 'Conversation',
    }

    results = asyncio.run(self._run_collect_multiple(engine_args, pairs))

    self.assertLen(results, 2)
    results.sort(key=lambda x: x[0])
    # The default mode for collect() is "Conversation", so we check conversation
    # length.
    # Pair 1: reset_obs, model_resp, step_obs -> 3 messages
    self.assertLen(results[0][1], 3)
    # Pair 2: reset_obs, resp1, obs1, resp2, obs2 -> 5 messages
    self.assertLen(results[1][1], 5)

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_status_max_context_limit_reached(self, mock_convert):
    # 100 assistant + 100 env = 200 > 150. Should stop after 1 step.
    mock_convert.side_effect = [
        ([1] * 100, [1] * 100),  # prompt tokens
        ([1] * 100, [1] * 100),  # assistant tokens 1
        ([1] * 100, [1] * 100),  # env tokens 1
    ]
    # Setup specific for this test
    self.mock_model_call.side_effect = [
        RolloutOutput(
            text=['response1'],
            logits=[np.zeros((100,))],
            tokens=[np.array([1] * 100)],
            left_padded_prompt_tokens=np.array([1]),
            logprobs=[np.ones((100,))],
        )
    ]
    self.mock_env.max_steps = 5
    self.mock_chat_parser.parse.return_value = 'mock_parsed_text'

    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=self.mock_chat_parser,
        max_response_length=150,
    )

    result_traj = asyncio.run(self._run_collect(engine, mode='Trajectory'))

    # Verify status is MAX_CONTEXT_LIMIT_REACHED
    self.assertEqual(
        result_traj.status,
        agent_types.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED,
    )
    # 100 step = 100 > 150. Should stop after 1 step.
    self.assertLen(result_traj.steps, 1)

  def test_collect_max_steps_reached(self):
    self.mock_env.max_steps = 1
    self.mock_env.step.side_effect = [
        ('obs1', 1.0, True, {}),
    ]
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
    )
    result_traj = asyncio.run(self._run_collect(engine, mode='Trajectory'))

    self.assertEqual(result_traj.status, agent_types.TrajectoryStatus.SUCCEEDED)
    self.assertLen(result_traj.steps, 1)

  def test_collect_timeout(self):
    self.mock_env.max_steps = 10
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        max_response_length=1024,
        timeout=0.1,
    )
    with mock.patch.object(
        engine,
        '_remaining_time',
        side_effect=[1.0, 1.0, 1.0, -0.1, -0.1],
    ):
      result_traj = asyncio.run(self._run_collect(engine, mode='Trajectory'))

    self.assertTrue(result_traj.steps[-1].done)
    self.assertEqual(result_traj.status, agent_types.TrajectoryStatus.TIMEOUT)

  def test_model_timeout_aborts_turn_and_always_closes(self):
    def slow_model(*args, **kwargs):
      del args, kwargs
      time.sleep(0.05)
      return RolloutOutput(
          text=['late'],
          logits=None,
          tokens=[np.array([1])],
          left_padded_prompt_tokens=np.array([[1]]),
          logprobs=[np.array([0.0])],
      )

    self.mock_model_call.side_effect = slow_model
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        timeout=1.0,
        per_turn_timeout=0.01,
        cleanup_timeout=0.1,
    )
    result = asyncio.run(self._run_collect(engine, mode='Trajectory'))

    self.assertEqual(
        result.status, agent_types.TrajectoryStatus.MODEL_TIMEOUT
    )
    self.mock_env.step.assert_not_called()
    self.mock_env.close.assert_called_once()
    request_timeout = self.mock_model_call.call_args.kwargs[
        'request_timeout_s'
    ]
    self.assertGreater(request_timeout, 0)
    self.assertLess(request_timeout, 0.01)

  def test_shared_batch_deadline_reduces_late_collector_budget(self):
    def slow_model(*args, **kwargs):
      del args, kwargs
      time.sleep(0.05)
      return RolloutOutput(
          text=['late'],
          logits=None,
          tokens=[np.array([1])],
          left_padded_prompt_tokens=np.array([[1]]),
          logprobs=[np.array([0.0])],
      )

    self.mock_model_call.side_effect = slow_model
    batch_started = time.perf_counter() - 0.04
    self.mock_env.extra_kwargs = {
        '_trajectory_batch_started_monotonic': batch_started,
        '_trajectory_batch_started_unix': time.time() - 0.04,
    }
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        timeout=0.06,
        cleanup_timeout=0.1,
    )
    result = asyncio.run(self._run_collect(engine, mode='Trajectory'))

    self.assertEqual(result.status, agent_types.TrajectoryStatus.MODEL_TIMEOUT)
    self.assertGreaterEqual(
        result.trajectory_time['collector_start_skew_secs'], 0.03
    )
    self.assertGreaterEqual(
        result.trajectory_time['batch_elapsed_secs'], 0.05
    )
    self.assertLess(result.model_time['generation_latency'], 0.08)

  def test_reset_timeout_still_closes_environment(self):
    def slow_reset():
      time.sleep(0.03)
      return 'late', {}

    self.mock_env.reset.side_effect = slow_reset
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        timeout=0.01,
        cleanup_timeout=0.1,
    )
    result = asyncio.run(self._run_collect(engine, mode='Trajectory'))

    self.assertEqual(
        result.status, agent_types.TrajectoryStatus.ENV_TIMEOUT
    )
    self.assertEqual(result.timeout_stage, "environment_reset")
    self.mock_model_call.assert_not_called()
    self.mock_env.close.assert_called_once()

  def test_reset_timeout_token_preserves_environment_task(self):
    def slow_reset():
      time.sleep(0.03)
      return 'late', {}

    self.trajectory.task = None
    self.mock_env.task = {
        'prompts': ['original prompt'],
        'policy_version': 7,
    }
    self.mock_env.reset.side_effect = slow_reset
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        timeout=0.01,
        cleanup_timeout=0.1,
    )
    result = asyncio.run(self._run_collect(engine, mode='Token'))

    self.assertEqual(result['status'], 'ENV_TIMEOUT')
    self.assertEqual(result['original_input'], self.mock_env.task)
    self.assertEqual(result['policy_version'], 7)
    merged = rl_utils.merge_micro_batches([result['original_input']])
    self.assertEqual(merged['prompts'], ['original prompt'])
    np.testing.assert_array_equal(merged['policy_version'], np.array([7]))
    self.mock_model_call.assert_not_called()
    self.mock_env.close.assert_called_once()

  def test_token_prefers_policy_seeded_environment_task(self):
    self.trajectory.task = {
        'prompts': ['formatted observation'],
        'turn_local_field': 'must not leak',
    }
    self.mock_env.task = {
        'prompts': ['dataset prompt'],
        'policy_version': 7,
    }
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
    )

    self.assertEqual(engine._original_input(), self.mock_env.task)

  def test_token_merges_policy_version_into_frozenlake_prompt(self):
    self.trajectory.task = {'prompts': ['rendered FrozenLake observation']}
    self.mock_env.task = {
        'policy_version': 7,
        'environment_receipt': 'frozenlake',
    }
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
    )

    self.assertEqual(
        engine._original_input(),
        {
            'prompts': ['rendered FrozenLake observation'],
            'policy_version': 7,
            'environment_receipt': 'frozenlake',
        },
    )
    self.assertEqual(
        self.trajectory.task, {'prompts': ['rendered FrozenLake observation']}
    )
    self.assertEqual(
        self.mock_env.task,
        {'policy_version': 7, 'environment_receipt': 'frozenlake'},
    )

  def test_policy_seeded_original_input_missing_prompt_fails_closed(self):
    self.trajectory.task = None
    self.mock_env.task = {'policy_version': 7}
    self.mock_env.reset.side_effect = TimeoutError('reset failed')
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        timeout=1.0,
        cleanup_timeout=0.1,
    )

    with self.assertRaisesRegex(ValueError, "missing required key 'prompts'"):
      asyncio.run(self._run_collect(engine, mode='Token'))
    self.mock_env.close.assert_called_once()

  def test_token_missing_original_input_fails_closed(self):
    self.trajectory.task = None
    self.mock_env.task = None
    self.mock_env.reset.side_effect = TimeoutError('reset failed')
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        timeout=1.0,
        cleanup_timeout=0.1,
    )

    with self.assertRaisesRegex(TypeError, 'original_input must be a dict'):
      asyncio.run(self._run_collect(engine, mode='Token'))
    self.mock_env.close.assert_called_once()

  def test_reset_raised_timeout_is_env_timeout(self):
    self.mock_env.reset.side_effect = TimeoutError(
        "Kubernetes pod did not start within 1200s; phase=Pending "
        "conditions=PodScheduled:False:Unschedulable:0/1 nodes available: "
        "Insufficient cpu"
    )
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        timeout=1.0,
        cleanup_timeout=0.1,
    )
    result = asyncio.run(self._run_collect(engine, mode='Trajectory'))

    self.assertEqual(
        result.status, agent_types.TrajectoryStatus.ENV_TIMEOUT
    )
    self.assertEqual(result.timeout_stage, "sandbox_start")
    self.assertEqual(result.timeout_scheduler_reason, "unschedulable")
    self.assertEqual(result.timeout_resource, "cpu")
    self.mock_model_call.assert_not_called()
    self.mock_env.close.assert_called_once()

  def test_reset_scheduling_gate_is_distinct_env_timeout(self):
    self.mock_env.reset.side_effect = TimeoutError(
        "Kubernetes pod did not start within 1200s; phase=Pending "
        "conditions=PodScheduled:False:SchedulingGated:Scheduling is "
        "blocked due to non-empty scheduling gates"
    )
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        timeout=1.0,
        cleanup_timeout=0.1,
    )
    result = asyncio.run(self._run_collect(engine, mode='Trajectory'))

    self.assertEqual(
        result.status, agent_types.TrajectoryStatus.ENV_TIMEOUT
    )
    self.assertEqual(result.timeout_stage, "sandbox_start")
    self.assertEqual(result.timeout_scheduler_reason, "scheduling_gated")
    self.assertEqual(result.timeout_resource, "")
    self.mock_model_call.assert_not_called()
    self.mock_env.close.assert_called_once()

  def test_final_reward_timeout_is_recorded_and_closes(self):
    self.mock_env.max_steps = 1
    self.mock_env.step.side_effect = [('done', 0.0, True, {})]

    def slow_reward():
      time.sleep(0.05)
      return 1.0

    self.mock_env.final_reward_fn.side_effect = slow_reward
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        timeout=0.02,
        cleanup_timeout=0.1,
    )
    result = asyncio.run(self._run_collect(engine, mode='Trajectory'))

    self.assertEqual(
        result.status, agent_types.TrajectoryStatus.REWARD_TIMEOUT
    )
    self.mock_env.close.assert_called_once()

  def test_cleanup_timeout_is_a_hard_error(self):
    self.mock_env.max_steps = 1
    self.mock_env.step.side_effect = [('done', 0.0, True, {})]

    def slow_close():
      time.sleep(0.05)

    self.mock_env.close.side_effect = slow_close
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        timeout=1.0,
        cleanup_timeout=0.01,
    )
    with self.assertRaisesRegex(TimeoutError, 'environment cleanup exceeded'):
      asyncio.run(self._run_collect(engine, mode='Trajectory'))

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_overlong_filter_masks_out_and_skips_reward(self, mock_convert):
    # Setup for MAX_STEPS_REACHED
    self.mock_env.max_steps = 1
    self.mock_env.step.side_effect = [
        ('obs1', 1.0, False, {}),  # Not done, so it hits max_steps
    ]
    mock_convert.side_effect = [
        ([101], [1]),  # prompt tokens
        ([301], [1]),  # env tokens 1
    ]

    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=self.mock_chat_parser,
        overlong_filter=True,
    )

    token_data = asyncio.run(self._run_collect(engine, mode='Token'))

    # Verify status is MAX_STEPS_REACHED
    self.assertEqual(
        token_data['status'],
        agent_types.TrajectoryStatus.MAX_STEPS_REACHED.name,
    )

    # Verify final reward was NOT called
    self.mock_final_reward_fn.assert_not_called()

    # Verify masks are zeroed out
    # Assistant tokens (201, 202) and Env tokens (301) should have masks
    # [0, 0, 0]
    expected_masks = np.array([0, 0, 0])
    np.testing.assert_array_equal(
        token_data['conversation_masks'], expected_masks
    )

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_overlong_filter_disabled_does_not_mask_out(self, mock_convert):
    # Setup for MAX_STEPS_REACHED but with overlong_filter=False
    self.mock_env.max_steps = 1
    self.mock_env.step.side_effect = [
        ('obs1', 1.0, False, {}),
    ]
    mock_convert.side_effect = [
        ([101], [1]),  # prompt tokens
        ([301], [1]),  # env tokens 1
    ]

    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=self.mock_chat_parser,
        overlong_filter=False,
    )

    token_data = asyncio.run(self._run_collect(engine, mode='Token'))

    # Verify final reward WAS called
    self.mock_final_reward_fn.assert_called_once()

    # Verify masks are NOT zeroed out
    expected_masks = np.array([1, 1, 1])
    np.testing.assert_array_equal(
        token_data['conversation_masks'], expected_masks
    )

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_overlong_filter_does_not_mask_out_on_success(self, mock_convert):
    # Setup for SUCCEEDED
    self.mock_env.max_steps = 5
    self.mock_env.step.side_effect = [
        ('obs1', 1.0, True, {}),
    ]
    mock_convert.side_effect = [
        ([101], [1]),  # prompt tokens
        ([301], [1]),  # env tokens 1
    ]

    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=self.mock_chat_parser,
        overlong_filter=True,
    )

    token_data = asyncio.run(self._run_collect(engine, mode='Token'))

    # Verify status is SUCCEEDED
    self.assertEqual(
        token_data['status'], agent_types.TrajectoryStatus.SUCCEEDED.name
    )

    # Verify masks are NOT zeroed out.
    # Note: Terminal-step env tokens are not appended to the mask.
    # Therefore, we only get the assistant tokens masks (2 tokens, value 1).
    expected_masks = np.array([1, 1])
    np.testing.assert_array_equal(
        token_data['conversation_masks'], expected_masks
    )


if __name__ == '__main__':
  absltest.main()
