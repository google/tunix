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
          left_padded_prompt_tokens=np.array([1]),
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
            'reset_cpu_time': 0.0,
            'step_latency': 0.0,
            'step_cpu_time': 0.0,
        },
        'reward_time': {
            'reward_latency': 0.0,
            'reward_cpu_time': 0.0,
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
    with mock.patch.object(time, 'perf_counter') as mock_perf:
      # Reset: 3 calls
      # Step 1: 3 calls
      # Final reward: 2 calls
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


if __name__ == '__main__':
  absltest.main()
