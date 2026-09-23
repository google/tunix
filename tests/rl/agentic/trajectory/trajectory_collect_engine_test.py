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
import time
from unittest import mock

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from tunix.experimental.trajectory import converter as converter_lib
from tunix.experimental.trajectory import in_memory_store
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import tracer as perf_tracer_v2
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
          left_padded_prompt_tokens=np.array([101]),
          logprobs=[np.ones_like(tokens)],
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
    self.assertIsInstance(result_traj.env_time['step_latency'], list)
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

  def test_collect_with_async_model_call(self):
    """Verifies that TrajectoryCollectEngine supports an async coroutine model_call callback."""
    self.mock_env.max_steps = 1
    self.mock_env.step.side_effect = [
        ('obs_async', 2.0, True, {}),
    ]

    async def _async_model_call(chat_completions, env=None, **kwargs):
      del env, kwargs
      await asyncio.sleep(0.001)
      return RolloutOutput(
          text=['async_resp'],
          logits=[jnp.zeros_like(np.array([10, 20]))],
          tokens=[np.array([10, 20])],
          left_padded_prompt_tokens=np.array([1]),
          logprobs=[[0.5, 0.5]],
      )

    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=_async_model_call,
    )
    result_traj = asyncio.run(
        self._run_collect(engine, mode='Trajectory')
    )
    self.assertLen(result_traj.steps, 1)
    self.assertEqual(result_traj.steps[0].reward, 2.5)

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
        'prompt_tokens': np.array([101]),
        'conversation_tokens': np.array(
            [201, 202, 301, 302, 203, 204]
        ),
        'conversation_masks': np.array([1, 1, 1, 1, 1, 1]),
        'trajectory_reward': (
            3.5
        ),  # 1.0 + 2.0 + 0.5 (final reward from final_reward_fn)
        'env_time': {
            'reset_latency': 0.0,
            'step_latency': [],
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
          val = token_data[k][sub_k]
          if isinstance(val, list):
            self.assertIsInstance(val, list)
          else:
            self.assertGreaterEqual(val, 0.0)
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
  def test_missing_step_logprobs_raises_error(self, mock_tokenize):
    mock_tokenize.side_effect = [
        ([101], [1]),  # prompt tokens
    ]
    self.mock_model_call.side_effect = [
        RolloutOutput(
            text=['resp1'],
            logits=None,
            tokens=[np.array([201, 202])],
            left_padded_prompt_tokens=np.array([[101]]),
            logprobs=None,
        )
    ]
    self.mock_env.step.side_effect = [('obs1', 1.0, True, {})]
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=self.mock_chat_parser,
    )
    with self.assertRaisesRegex(ValueError, 'missing step logprobs'):
      asyncio.run(self._run_collect(engine, mode='Token'))

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_missing_step_logprobs_empty_tokens_does_not_raise(
      self, mock_tokenize
  ):
    mock_tokenize.side_effect = [
        ([101], [1]),  # prompt tokens
    ]
    self.mock_model_call.side_effect = [
        RolloutOutput(
            text=[''],
            logits=None,
            tokens=[np.array([], dtype=np.int32)],
            left_padded_prompt_tokens=np.array([[101]]),
            logprobs=None,
        )
    ]
    self.mock_env.step.side_effect = [('obs1', 1.0, True, {})]
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=self.mock_chat_parser,
    )
    token_data = asyncio.run(self._run_collect(engine, mode='Token'))
    self.assertIsNotNone(token_data['old_logprobs'])

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
    with mock.patch.object(time, 'perf_counter') as mock_perf:
      # Reset: 3 calls
      # Step 1: 3 calls
      # Final reward: 2 calls
      # Close: 2 calls
      mock_perf.side_effect = [
          100.0,
          100.01,
          100.02,  # _reset
          100.03,
          100.04,
          100.2,  # _one_step: 100.2 - 100.02 = 0.18 > 0.1
          100.21,
          100.22,
          100.23,  # _append_final_reward
          100.24,
          100.25,  # _close
      ]

      engine = trajectory_collect_engine.TrajectoryCollectEngine(
          agent=self.mock_agent,
          env=self.mock_env,
          model_call=self.mock_model_call,
          max_response_length=1024,
          timeout=0.1,
      )
      result_traj = asyncio.run(self._run_collect(engine, mode='Trajectory'))

    self.assertTrue(result_traj.steps[-1].done)
    self.assertEqual(result_traj.status, agent_types.TrajectoryStatus.TIMEOUT)

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

  def test_env_time_metrics_close(self):
    self.mock_env.close = mock.Mock()
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
    )
    trajectory = asyncio.run(self._run_collect(engine, mode='Trajectory'))
    self.mock_env.close.assert_called_once()
    self.assertIn('reset_latency', trajectory.env_time)
    self.assertIn('step_latency', trajectory.env_time)
    self.assertIsInstance(trajectory.env_time['step_latency'], list)
    self.assertIn('close_latency', trajectory.env_time)

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_token_mode_with_routed_experts(self, mock_convert):
    mock_convert.side_effect = [
        ([101, 102], [1, 1]),  # prompt tokens
        ([301, 302, 303], [1, 1, 1]),  # env tokens 1
    ]
    num_layers, top_k = 4, 2
    prompt_tokens = np.array([101, 102], dtype=np.int32)
    step1_tokens = np.array([201, 202], dtype=np.int32)
    step2_tokens = np.array([203, 204], dtype=np.int32)

    # Step 1: prompt (len 2, fill 3) + generation (len 2, fill 5) -> len 4
    step1_routed = np.concatenate(
        [
            np.full((2, num_layers, top_k), 3, dtype=np.int32),
            np.full((2, num_layers, top_k), 5, dtype=np.int32),
        ],
        axis=0,
    )
    # Step 2: env tokens (len 3, fill 6) + generation (len 2, fill 7) -> len 5
    step2_routed = np.concatenate(
        [
            np.full((3, num_layers, top_k), 6, dtype=np.int32),
            np.full((2, num_layers, top_k), 7, dtype=np.int32),
        ],
        axis=0,
    )

    self.mock_model_call.side_effect = [
        RolloutOutput(
            text=['resp1'],
            logits=None,
            tokens=[step1_tokens],
            left_padded_prompt_tokens=np.array([prompt_tokens]),
            logprobs=[np.ones(2, dtype=np.float32)],
            routed_experts=[step1_routed],
        ),
        RolloutOutput(
            text=['resp2'],
            logits=None,
            tokens=[step2_tokens],
            left_padded_prompt_tokens=np.array([prompt_tokens]),
            logprobs=[np.ones(2, dtype=np.float32)],
            routed_experts=[step2_routed],
        ),
    ]

    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=self.mock_chat_parser,
    )

    token_data = asyncio.run(self._run_collect(engine, mode='Token'))

    # Verify routed_experts_prompt_start was passed with cumulative offset on turn 1
    self.assertEqual(len(self.mock_model_call.call_args_list), 2)
    turn0_kwargs = self.mock_model_call.call_args_list[0].kwargs
    self.assertNotIn('routed_experts_prompt_start', turn0_kwargs)
    turn1_kwargs = self.mock_model_call.call_args_list[1].kwargs
    self.assertEqual(turn1_kwargs.get('routed_experts_prompt_start'), 4)

    self.assertIn('routed_experts', token_data)
    routed = token_data['routed_experts']
    self.assertIsNotNone(routed)
    self.assertEqual(routed.dtype, np.int16)

    # 2 prompt + 2 asst1 + 3 env1 + 2 asst2 = 9 tokens total
    prompt_len = len(token_data['prompt_tokens'])
    conv_len = len(token_data['conversation_tokens'])
    self.assertEqual(prompt_len, 2)
    self.assertEqual(conv_len, 7)
    self.assertEqual(routed.shape, (9, num_layers, top_k))

    # Prompt tokens carry value 3
    np.testing.assert_array_equal(routed[:2], 3)
    # Step 1 assistant tokens carry value 5
    np.testing.assert_array_equal(routed[2:4], 5)
    # Step 1 env tokens carry value 6
    np.testing.assert_array_equal(routed[4:7], 6)
    # Step 2 assistant tokens carry value 7
    np.testing.assert_array_equal(routed[7:9], 7)

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_steps_mode_with_routed_experts(self, mock_convert):
    mock_convert.side_effect = [
        ([101, 102], [1, 1]),  # prompt tokens
        ([301, 302, 303], [1, 1, 1]),  # env tokens 1
    ]
    num_layers, top_k = 4, 2
    prompt_tokens = np.array([101, 102], dtype=np.int32)
    step1_tokens = np.array([201, 202], dtype=np.int32)
    step2_tokens = np.array([203, 204], dtype=np.int32)

    step1_routed = np.concatenate(
        [
            np.full((2, num_layers, top_k), 3, dtype=np.int32),
            np.full((2, num_layers, top_k), 5, dtype=np.int32),
        ],
        axis=0,
    )
    step2_routed = np.concatenate(
        [
            np.full((3, num_layers, top_k), 6, dtype=np.int32),
            np.full((2, num_layers, top_k), 7, dtype=np.int32),
        ],
        axis=0,
    )

    self.mock_model_call.side_effect = [
        RolloutOutput(
            text=['resp1'],
            logits=None,
            tokens=[step1_tokens],
            left_padded_prompt_tokens=np.array([prompt_tokens]),
            logprobs=[np.ones(2, dtype=np.float32)],
            routed_experts=[step1_routed],
        ),
        RolloutOutput(
            text=['resp2'],
            logits=None,
            tokens=[step2_tokens],
            left_padded_prompt_tokens=np.array([prompt_tokens]),
            logprobs=[np.ones(2, dtype=np.float32)],
            routed_experts=[step2_routed],
        ),
    ]

    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=self.mock_chat_parser,
    )

    steps_data = asyncio.run(self._run_collect(engine, mode='Steps'))
    self.assertEqual(len(steps_data), 2)
    self.assertEqual(
        steps_data[0]['assistant_routed_experts'].dtype, np.int16
    )
    self.assertEqual(
        steps_data[0]['assistant_routed_experts'].shape, (2, num_layers, top_k)
    )
    np.testing.assert_array_equal(steps_data[0]['assistant_routed_experts'], 5)
    self.assertEqual(steps_data[0]['env_routed_experts'].dtype, np.int16)
    self.assertEqual(
        steps_data[0]['env_routed_experts'].shape, (3, num_layers, top_k)
    )
    np.testing.assert_array_equal(steps_data[0]['env_routed_experts'], 6)
    self.assertEqual(
        steps_data[1]['assistant_routed_experts'].dtype, np.int16
    )
    self.assertEqual(
        steps_data[1]['assistant_routed_experts'].shape, (2, num_layers, top_k)
    )
    np.testing.assert_array_equal(steps_data[1]['assistant_routed_experts'], 7)
    self.assertIsNone(steps_data[1]['env_routed_experts'])

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_token_mode_with_routed_experts_and_end_tokens(self, mock_convert):
    mock_convert.side_effect = [
        ([101], [1]),  # prompt tokens
    ]
    num_layers, top_k = 3, 2
    prompt_tokens = np.array([101], dtype=np.int32)
    step_tokens = np.array([201, 202], dtype=np.int32)
    # Prompt (len 1, fill 3) + generation (len 2, fill 9) -> len 3
    step_routed = np.concatenate(
        [
            np.full((1, num_layers, top_k), 3, dtype=np.int32),
            np.full((2, num_layers, top_k), 9, dtype=np.int32),
        ],
        axis=0,
    )

    # Chat parser appends 1 end token
    self.mock_chat_parser.update_assistant_end_tokens.side_effect = (
        lambda tokens: (np.append(tokens, 999), 1)
    )

    self.mock_model_call.side_effect = [
        RolloutOutput(
            text=['resp1'],
            logits=None,
            tokens=[step_tokens],
            left_padded_prompt_tokens=np.array([prompt_tokens]),
            logprobs=[np.ones(2, dtype=np.float32)],
            routed_experts=[step_routed],
        ),
    ]
    # One step episode
    self.mock_env.step.side_effect = [('obs1', 1.0, True, {})]

    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=self.mock_chat_parser,
    )

    token_data = asyncio.run(self._run_collect(engine, mode='Token'))
    routed = token_data['routed_experts']
    self.assertIsNotNone(routed)

    # Prompt (len 1, 3), generated assistant tokens (len 2, 9), appended end token (len 1, -1)
    self.assertEqual(routed.shape, (4, num_layers, top_k))
    np.testing.assert_array_equal(routed[0], 3)
    np.testing.assert_array_equal(routed[1:3], 9)
    np.testing.assert_array_equal(routed[3], agent_types.UNSET_ROUTED_EXPERT)

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_env_routed_experts_length_mismatch_raises_error(self, mock_convert):
    mock_convert.side_effect = [
        ([101, 102], [1, 1]),  # prompt tokens
        ([301, 302, 303], [1, 1, 1]),  # env tokens len 3
    ]
    num_layers, top_k = 2, 2
    prompt_tokens = np.array([101, 102], dtype=np.int32)
    step1_tokens = np.array([201], dtype=np.int32)
    step2_tokens = np.array([202], dtype=np.int32)

    # Step 1: prompt (len 2) + gen (len 1) = len 3
    step1_routed = np.full((3, num_layers, top_k), 1, dtype=np.int32)
    # Step 2: only 2 env tokens instead of 3
    step2_routed = np.full((2, num_layers, top_k), 2, dtype=np.int32)

    self.mock_model_call.side_effect = [
        RolloutOutput(
            text=['resp1'],
            logits=None,
            tokens=[step1_tokens],
            left_padded_prompt_tokens=np.array([prompt_tokens]),
            logprobs=[np.ones(1, dtype=np.float32)],
            routed_experts=[step1_routed],
        ),
        RolloutOutput(
            text=['resp2'],
            logits=None,
            tokens=[step2_tokens],
            left_padded_prompt_tokens=np.array([prompt_tokens]),
            logprobs=[np.ones(1, dtype=np.float32)],
            routed_experts=[step2_routed],
        ),
    ]

    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=self.mock_chat_parser,
    )

    with self.assertRaisesRegex(
        ValueError, 'Mismatch between captured env_routed_experts length'
    ):
      asyncio.run(self._run_collect(engine, mode='Token'))

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_token_mode_with_vllm_autoregressive_routed_experts(
      self, mock_convert
  ):
    mock_convert.side_effect = [
        ([101, 102], [1, 1]),  # prompt tokens (len 2)
        ([301, 302, 303], [1, 1, 1]),  # env tokens (len 3)
    ]
    num_layers, top_k = 4, 2
    prompt_tokens = np.array([101, 102], dtype=np.int32)
    step1_tokens = np.array([201, 202], dtype=np.int32)
    step2_tokens = np.array([203, 204], dtype=np.int32)

    # Turn 0: prompt (len 2, fill 3) + generation (len 2 - 1 = 1, fill 5) -> len 3
    # Token 202 is sampled at end of decode and not routed in turn 0.
    step1_routed = np.concatenate(
        [
            np.full((2, num_layers, top_k), 3, dtype=np.uint8),
            np.full((1, num_layers, top_k), 5, dtype=np.uint8),
        ],
        axis=0,
    )
    # Turn 1: delayed assistant token (len 1, fill 5) + env tokens (len 3, fill 6)
    # + generation (len 2 - 1 = 1, fill 7) -> len 5 total
    step2_routed = np.concatenate(
        [
            np.full((1, num_layers, top_k), 5, dtype=np.uint8),
            np.full((3, num_layers, top_k), 6, dtype=np.uint8),
            np.full((1, num_layers, top_k), 7, dtype=np.uint8),
        ],
        axis=0,
    )

    self.mock_model_call.side_effect = [
        RolloutOutput(
            text=['resp1'],
            logits=None,
            tokens=[step1_tokens],
            left_padded_prompt_tokens=np.array([prompt_tokens]),
            logprobs=[np.ones(2, dtype=np.float32)],
            routed_experts=[step1_routed],
        ),
        RolloutOutput(
            text=['resp2'],
            logits=None,
            tokens=[step2_tokens],
            left_padded_prompt_tokens=np.array([prompt_tokens]),
            logprobs=[np.ones(2, dtype=np.float32)],
            routed_experts=[step2_routed],
        ),
    ]

    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=self.mock_chat_parser,
    )

    token_data = asyncio.run(self._run_collect(engine, mode='Token'))

    # Verify routed_experts_prompt_start was passed with cumulative offset 3 on turn 1
    self.assertEqual(len(self.mock_model_call.call_args_list), 2)
    turn0_kwargs = self.mock_model_call.call_args_list[0].kwargs
    self.assertNotIn('routed_experts_prompt_start', turn0_kwargs)
    turn1_kwargs = self.mock_model_call.call_args_list[1].kwargs
    self.assertEqual(turn1_kwargs.get('routed_experts_prompt_start'), 3)

    self.assertIn('routed_experts', token_data)
    routed = token_data['routed_experts']
    self.assertIsNotNone(routed)
    self.assertEqual(routed.dtype, np.int16)

    # 2 prompt + 2 asst1 + 3 env1 + 2 asst2 = 9 tokens total
    prompt_len = len(token_data['prompt_tokens'])
    conv_len = len(token_data['conversation_tokens'])
    self.assertEqual(prompt_len, 2)
    self.assertEqual(conv_len, 7)
    self.assertEqual(routed.shape, (9, num_layers, top_k))

    # Prompt tokens carry value 3
    np.testing.assert_array_equal(routed[:2], 3)
    # Step 1 assistant tokens carry value 5 (including stitched token from turn 1 prefill)
    np.testing.assert_array_equal(routed[2:4], 5)
    # Step 1 env tokens carry value 6
    np.testing.assert_array_equal(routed[4:7], 6)
    # Step 2 assistant token 0 carries value 7
    np.testing.assert_array_equal(routed[7], 7)
    # Step 2 terminal token padded with UNSET_ROUTED_EXPERT
    np.testing.assert_array_equal(routed[8], agent_types.UNSET_ROUTED_EXPERT)

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_token_mode_with_zero_generated_tokens_and_routed_experts(
      self, mock_convert
  ):
    mock_convert.side_effect = [
        ([101], [1]),  # prompt tokens
    ]
    num_layers, top_k = 2, 2
    prompt_tokens = np.array([101], dtype=np.int32)
    step_tokens = np.array([], dtype=np.int32)
    prompt_routing = np.full((1, num_layers, top_k), 3, dtype=np.int32)

    self.mock_chat_parser.update_assistant_end_tokens.side_effect = (
        lambda tokens: (tokens, 0)
    )

    self.mock_model_call.side_effect = [
        RolloutOutput(
            text=[''],
            logits=None,
            tokens=[step_tokens],
            left_padded_prompt_tokens=np.array([prompt_tokens]),
            logprobs=[np.zeros(0, dtype=np.float32)],
            routed_experts=[prompt_routing],
        ),
    ]
    self.mock_env.step.side_effect = [('obs1', 1.0, True, {})]

    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=self.mock_chat_parser,
    )

    token_data = asyncio.run(self._run_collect(engine, mode='Token'))
    routed = token_data['routed_experts']
    self.assertIsNotNone(routed)

    # Prompt (len 1, carrying prompt_routing 3) and conversation (len 0)
    self.assertEqual(routed.shape, (1, num_layers, top_k))
    np.testing.assert_array_equal(routed[0], 3)
  def test_on_rollout_output_callback(self):
    outputs = []

    def _on_rollout_output(output):
      outputs.append(output)

    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        on_rollout_output_callback=_on_rollout_output,
    )
    asyncio.run(self._run_collect(engine, mode='Trajectory'))
    self.assertLen(outputs, 2)
    self.assertEqual(outputs[0].text, ['response1'])
    self.assertEqual(outputs[1].text, ['response2'])

  def test_on_final_reward_callback(self):
    reward_steps = []

    def _on_final_reward(step):
      reward_steps.append(step)

    def _mock_final_reward():
      return 0.5

    self.mock_env.final_reward_fn = _mock_final_reward
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        on_final_reward_callback=_on_final_reward,
    )
    asyncio.run(self._run_collect(engine, mode='Trajectory'))
    self.assertLen(reward_steps, 1)
    self.assertEqual(reward_steps[0].reward, 2.5)  # initial 2.0 + final 0.5

  def test_trajectory_store_writes(self):
    store = in_memory_store.InMemoryTrajectoryStore()
    metadata = converter_lib.create_trajectory_metadata(
        traj_id='traj_test_123',
    )
    self.mock_agent.trajectory.task = {'prompts': ['Solve math']}
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        trajectory_store=store,
        metadata=metadata,
        policy_version=42,
    )
    with (
        mock.patch.object(store, 'flush', wraps=store.flush) as mock_flush,
        mock.patch.object(store, 'close', wraps=store.close) as mock_close,
    ):
      asyncio.run(self._run_collect(engine, mode='Trajectory'))
      mock_flush.assert_not_called()
      mock_close.assert_not_called()

    # Verify trajectory store contains the written steps via public API
    trajs = store.get_trajectories(['traj_test_123'])
    self.assertLen(trajs, 1)
    stored_traj = trajs[0]
    self.assertGreaterEqual(len(stored_traj.steps), 3)
    self.assertEqual(stored_traj.steps[0].step_id, 0)
    self.assertEqual(stored_traj.steps[0].message, 'Solve math')
    self.assertEqual(stored_traj.steps[1].mc_return, 3.5)
    self.assertEqual(stored_traj.steps[3].mc_return, 2.5)
    metas = store.get_trajectories_metadata()
    self.assertLen(metas, 1)
    self.assertEqual(metas[0].trajectory_id, 'traj_test_123')
    self.assertEqual(metas[0].status, 'SUCCEEDED')
    self.assertEqual(metas[0].target_policy_versions, [42])

  @mock.patch.object(utils, 'tokenize_and_generate_masks')
  def test_on_model_and_env_step_callbacks(self, mock_convert):
    mock_convert.side_effect = [
        ([101], [1]),  # prompt tokens
        ([301, 302], [1, 1]),  # env tokens 1
        ([303, 304], [1, 1]),  # env tokens 2
    ]
    model_steps = []
    env_steps = []

    def _on_model_step(step):
      model_steps.append(
          (
              np.copy(step.assistant_tokens)
              if step.assistant_tokens is not None
              else None,
              np.copy(step.assistant_masks)
              if step.assistant_masks is not None
              else None,
          )
      )

    def _on_env_step(step):
      env_steps.append(
          (
              np.copy(step.env_tokens) if step.env_tokens is not None else None,
              np.copy(step.env_masks) if step.env_masks is not None else None,
          )
      )

    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=self.mock_chat_parser,
        on_model_step_callback=_on_model_step,
        on_env_step_callback=_on_env_step,
    )
    asyncio.run(self._run_collect(engine, mode='Trajectory'))
    self.assertLen(model_steps, 2)
    for tokens, masks in model_steps:
      self.assertIsNotNone(tokens)
      self.assertIsNotNone(masks)
      self.assertEqual(len(tokens), len(masks))
    self.assertLen(env_steps, 2)
    self.assertIsNotNone(env_steps[0][0])
    self.assertIsNotNone(env_steps[0][1])



class _FreshTextTokenizer:
  """Only freshly formatted observations are allowed into this encoder."""

  def __init__(self, env_rows):
    self.rows = iter(env_rows)
    self.encoded = []

  def encode(self, text, **kwargs):
    assert text.startswith('fresh-env:'), 'history was re-encoded'
    self.encoded.append(text)
    return next(self.rows)

  def dedup_bos_ids(self, ids):
    return ids


class _FreshTextParser:

  def __init__(self, suffix):
    self.suffix = suffix

  def parse(self, messages, **kwargs):
    assert len(messages) == 1 and messages[0]['role'] in ('user', 'tool')
    return 'fresh-env:' + messages[0]['content']

  def update_assistant_end_tokens(self, tokens):
    return np.concatenate([tokens, np.array(self.suffix, np.int32)]), len(
        self.suffix
    )


class _ToolFixtureEnv(base_environment.BaseTaskEnv):

  def __init__(self):
    super().__init__(task={'policy_version': 7}, max_steps=5)
    self.close = mock.Mock()

  def _initial_observation(self):
    return 'Inspect the repository and fix the fixture bug.'

  def _step_impl(self, action):
    assert 'execute_bash' in action
    return base_environment.EnvStepResult(
        observation=f'tool stdout at turn {self.step_count}',
        reward=0.0,
        done=self.step_count >= 3,
        info={},
    )


class ExactTokenContinuityCollectTest(absltest.TestCase):
  """Later turns submit recorded IDs; the first turn keeps the text route."""

  def _frozenlake(self):
    import pathlib  # pylint: disable=g-import-not-at-top
    import sys  # pylint: disable=g-import-not-at-top

    here = pathlib.Path(__file__)
    repo_root = next(
        (
            base
            for base in (*here.parents, *here.resolve().parents)
            if (base / 'examples/frozenlake').is_dir()
            or (base / 'oss/examples/frozenlake').is_dir()
        ),
        None,
    )
    if repo_root is not None and str(repo_root) not in sys.path:
      sys.path.insert(0, str(repo_root))
    try:
      from examples.frozenlake.agent import FrozenLakeAgent  # pylint: disable=g-import-not-at-top
      from examples.frozenlake.env import FrozenLakeEnv  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      self.skipTest(f'requires tunix[frozenlake]: {e}')
    size = 5
    desc = ['SFFG' + 'F' * (size - 4)] + ['F' * size] * (size - 1)
    env = FrozenLakeEnv(
        {'size': size, 'seed': 0, 'p': 1.0},
        desc=desc,
        is_slippery=False,
        max_steps=5,
    )
    env.task['policy_version'] = 7
    env.close = mock.Mock(wraps=env.close)
    return FrozenLakeAgent(use_multistep_prompt=False), env

  def _collector(
      self, agent, env, *, tool=False, asynchronous=False, poison=None
  ):
    prompt = [200, 201] if tool else [5, 100, 101]
    samples = [[30, 31], [32], [33, 34]] if tool else [[10, 11], [12], [13, 14]]
    suffix = [91, 92] if tool else [90]
    env_rows = [[40, 41, 42], [43, 44]] if tool else [[20, 21], [22]]
    expected_inputs = (
        [
            [200, 201, 30, 31, 91, 92, 40, 41, 42],
            [200, 201, 30, 31, 91, 92, 40, 41, 42, 32, 91, 92, 43, 44],
        ]
        if tool
        else [
            [5, 100, 101, 10, 11, 90, 20, 21],
            [5, 100, 101, 10, 11, 90, 20, 21, 12, 90, 22],
        ]
    )
    calls = []

    def model_call(chat, environment, *, prompt_token_ids=None, **kwargs):
      del environment, kwargs
      index = len(calls)
      if index == 0:
        assert chat is not None and prompt_token_ids is None
        submitted = prompt
      else:
        assert chat is None
        np.testing.assert_array_equal(
            prompt_token_ids, expected_inputs[index - 1]
        )
        submitted = list(prompt_token_ids)
      calls.append(list(submitted))
      echoed = list(submitted)
      if poison == 'echo' and index == 1:
        echoed[-1] += 1
      text = (
          '<function=execute_bash><parameter=command>pwd</parameter></function>'
          if tool
          else '```Right```'
      )
      if poison == 'history' and index == 1:
        agent.chat_completions[0]['content'] += ' edited'
      return base_rollout.RolloutOutput(
          text=[text],
          logits=None,
          tokens=[np.array(samples[index], np.int32)],
          left_padded_prompt_tokens=np.array([[0, 0] + echoed], np.int32),
          prompt_lengths=np.array([len(echoed)], np.int32),
          logprobs=None
          if poison == 'logprobs'
          else [np.full(len(samples[index]), -0.5)],
      )

    async def async_model_call(*args, **kwargs):
      return model_call(*args, **kwargs)

    tokenizer = _FreshTextTokenizer(env_rows)
    parser = _FreshTextParser(suffix)
    if poison == 'suffix':
      parser.update_assistant_end_tokens = lambda tokens: (
          np.array([9, 9, 9]),
          1,
      )
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=agent,
        env=env,
        tokenizer=tokenizer,
        chat_parser=parser,
        model_call=async_model_call if asynchronous else model_call,
        max_response_length=64,
        exact_token_continuity=True,
    )
    return engine, calls, tokenizer

  def _assert_training_consumer(self, record):
    """Feeds the collector record through actual batch construction/packing."""
    import copy  # pylint: disable=g-import-not-at-top
    from types import SimpleNamespace  # pylint: disable=g-import-not-at-top
    from tunix.rl import rl_cluster  # pylint: disable=g-import-not-at-top
    from tunix.rl import utils as rl_utils  # pylint: disable=g-import-not-at-top
    from tunix.rl.agentic import agentic_grpo_learner  # pylint: disable=g-import-not-at-top

    learner = object.__new__(agentic_grpo_learner.GRPOLearner)
    learner.algo_config = agentic_grpo_learner.GRPOConfig(
        exact_token_continuity=True,
        max_response_length=24,
        beta=0.0,
        use_rollout_logps=True,
    )
    learner._trajectory_logger = None
    learner.metric_fns = []
    learner._compute_rewards = lambda **kw: np.array([0.0, 1.0])
    learner.rl_engine = SimpleNamespace(
        rollout=SimpleNamespace(pad_id=lambda: 0, eos_id=lambda: 255),
        r2m={rl_cluster.Role.ACTOR: None},
        perf_v2=perf_tracer_v2.NoopTracer(),
        buffer_metrics_async=mock.Mock(),
        cluster_config=SimpleNamespace(
            rollout_config=base_rollout.RolloutConfig(max_prompt_length=5),
            training_config=SimpleNamespace(
                max_seq_token_per_tpu=64, compute_logps_micro_batch_size=1
            ),
        ),
    )
    items = [
        agent_types.TrajectoryItem(traj=copy.deepcopy(record)) for _ in range(2)
    ]
    batch = learner._process_results(items)[0]
    for row in rl_utils.unpad_train_example(batch):
      np.testing.assert_array_equal(
          row['completion_ids'], record['conversation_tokens']
      )
      np.testing.assert_array_equal(
          row['completion_mask'], record['conversation_masks']
      )
      np.testing.assert_array_equal(
          row['old_per_token_logps'], record['old_logprobs']
      )
    packed = list(
        rl_utils.pack_sequences(
            iter([[batch]]), max_token_budget=64, sequences_per_update=2
        )
    )[0][0]
    expected_ids = list(
        record['prompt_tokens'][-record['prompt_length'] :]
    ) + list(record['conversation_tokens'])
    expected_mask = [0] * record['prompt_length'] + list(
        record['conversation_masks']
    )
    for segment in (1, 2):
      valid = np.asarray(packed.segment_ids[0]) == segment
      np.testing.assert_array_equal(
          packed.completion_ids[0][valid], expected_ids
      )
      np.testing.assert_array_equal(
          packed.completion_mask[0][valid], expected_mask
      )

  def test_frozenlake_three_turns_submit_recorded_ids(self):
    for asynchronous in (False, True):
      agent, env = self._frozenlake()
      engine, calls, tokenizer = self._collector(
          agent, env, asynchronous=asynchronous
      )
      result = asyncio.run(engine.collect(mode='Token'))
      self.assertEqual((len(calls), len(tokenizer.encoded)), (3, 2))
      self.assertEqual(
          (result['prompt_length'], result['policy_version']), (3, 7)
      )
      np.testing.assert_array_equal(result['prompt_tokens'][-3:], [5, 100, 101])
      np.testing.assert_array_equal(
          result['conversation_tokens'],
          [10, 11, 90, 20, 21, 12, 90, 22, 13, 14, 90],
      )
      np.testing.assert_array_equal(
          result['conversation_masks'], [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0]
      )
      np.testing.assert_array_equal(
          result['old_logprobs'],
          [-0.5, -0.5, 0, 0, 0, -0.5, 0, 0, -0.5, -0.5, 0],
      )
      self.assertEqual(
          engine._response_token_count, 8
      )  # suffixes are not samples
      env.close.assert_called_once()
      self._assert_training_consumer(result)

  def test_first_turn_terminal_keeps_suffix_without_env_encoding(self):
    agent, env = self._frozenlake()
    env.max_steps = 1
    engine, calls, tokenizer = self._collector(agent, env)
    result = asyncio.run(engine.collect(mode='Token'))
    self.assertEqual((len(calls), tokenizer.encoded), (1, []))
    np.testing.assert_array_equal(result['conversation_tokens'], [10, 11, 90])
    np.testing.assert_array_equal(result['conversation_masks'], [1, 1, 0])
    self.assertTrue(agent.trajectory.steps[0].done)

  def test_negatives_fail_closed(self):
    for poison, message in (
        ('echo', 'differs from recorded history'),
        ('history', 'rewrote previously recorded'),
        ('suffix', 'must only append'),
    ):
      agent, env = self._frozenlake()
      engine, _, _ = self._collector(agent, env, poison=poison)
      with self.assertRaisesRegex(ValueError, message):
        asyncio.run(engine.collect(mode='Token'))

  def test_swe_agent_tool_turns_submit_recorded_ids(self):
    import importlib.util  # pylint: disable=g-import-not-at-top
    import pathlib  # pylint: disable=g-import-not-at-top
    import sys  # pylint: disable=g-import-not-at-top
    import types  # pylint: disable=g-import-not-at-top

    # Only the unavailable R2E action parser is stubbed; the real SWEAgent
    # message/step code runs. This does not certify R2E or a sandbox.
    class FixtureAction:

      @classmethod
      def from_string(cls, text):
        obj = cls()
        obj.text = text
        return obj

      def to_xml_string(self):
        return self.text

    action_module = types.ModuleType('r2egym.agenthub.action')
    action_module.Action = FixtureAction
    fake_modules = {
        'r2egym': types.ModuleType('r2egym'),
        'r2egym.agenthub': types.ModuleType('r2egym.agenthub'),
        'r2egym.agenthub.action': action_module,
    }
    here = pathlib.Path(__file__)
    swe_path = next(
        p
        for base in (*here.parents, *here.resolve().parents)
        for p in (
            base / 'oss/examples/deepswe/swe_agent.py',
            base / 'examples/deepswe/swe_agent.py',
        )
        if p.exists()
    )
    with (
        mock.patch.dict(sys.modules, fake_modules),
        mock.patch.object(
            sys,
            'path',
            [
                str(swe_path.parent),
                str(swe_path.parent.parent.parent),
                *sys.path,
            ],
        ),
    ):
      spec = importlib.util.spec_from_file_location(
          'swe_agent_tito_fixture', swe_path
      )
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)
      agent = module.SWEAgent(format_model_response=True)
      env = _ToolFixtureEnv()
      engine, calls, tokenizer = self._collector(agent, env, tool=True)
      result = asyncio.run(engine.collect(mode='Token'))
    self.assertEqual((len(calls), len(tokenizer.encoded)), (3, 2))
    np.testing.assert_array_equal(
        result['conversation_tokens'],
        [30, 31, 91, 92, 40, 41, 42, 32, 91, 92, 43, 44, 33, 34, 91, 92],
    )
    np.testing.assert_array_equal(
        result['conversation_masks'],
        [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
    )
    env.close.assert_called_once()
    self._assert_training_consumer(result)


if __name__ == '__main__':
  absltest.main()
