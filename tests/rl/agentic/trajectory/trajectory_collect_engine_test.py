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
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.rewards import reward_types
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.rl.agentic import utils


class TrajectoryCollectEngineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_agent = mock.create_autospec(
        base_agent.ConversationAgentBase, instance=True
    )
    self.mock_env = mock.create_autospec(
        base_environment.BaseTaskEnv, instance=True
    )
    self.mock_model_call = mock.Mock()
    self.mock_final_reward_fn = mock.Mock(
        return_value=reward_types.RewardOutput(reward=0.5)
    )
    self.mock_tokenizer = mock.Mock()
    self.mock_chat_parser = mock.Mock()

    # Configure mock agent
    self.trajectory = agent_types.Trajectory()
    self.mock_agent.trajectory = self.trajectory
    self.mock_agent.chat_completions = []
    self.current_step = None

    def _update_from_model(resp):
      self.current_step = agent_types.Step(
          model_response=resp, action=agent_types.Action(action=['action'])
      )
      self.trajectory.steps.append(self.current_step)
      self.mock_agent.chat_completions.append(
          {'role': 'assistant', 'content': resp}
      )
      return self.current_step

    def _update_from_env(observation, reward, done, info):
      if self.current_step:
        self.current_step.observation = observation
        self.current_step.reward = reward
        self.current_step.done = done
        self.current_step.info = info
      self.mock_agent.chat_completions.append(
          {'role': 'user', 'content': observation}
      )

    def _get_current_state():
      return self.current_step

    def _reset_agent():
      self.trajectory.steps.clear()
      self.mock_agent.chat_completions.clear()
      self.current_step = None

    self.mock_agent.update_from_model.side_effect = _update_from_model
    self.mock_agent.update_from_env.side_effect = _update_from_env
    self.mock_agent.get_current_state.side_effect = _get_current_state
    self.mock_agent.reset.side_effect = _reset_agent

    # Configure mock env
    self.mock_env.reset.return_value = ('initial_obs', {})
    # Let it run for 2 steps then done
    self.mock_env.step.side_effect = [
        ('obs1', 1.0, False, {}),
        ('obs2', 2.0, True, {}),
    ]
    self.mock_env.task = {'some': 'task'}

    # Configure mock model call
    self.mock_model_call.side_effect = ['response1', 'response2']

  async def _run_collect(self, engine, mode='Trajectory'):
    return await engine.collect(mode=mode)

  def test_collect_trajectory_mode(self):
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        final_reward_fn=self.mock_final_reward_fn,
        max_steps=5,
        gamma=0.9,
    )
    result_traj = asyncio.run(self._run_collect(engine, mode='Trajectory'))

    self.assertLen(result_traj.steps, 2)
    self.assertEqual(self.mock_env.reset.call_count, 1)
    self.assertEqual(self.mock_env.step.call_count, 2)
    self.assertEqual(self.mock_model_call.call_count, 2)
    self.mock_final_reward_fn.assert_called_once_with(
        self.mock_env.task, 'response2'
    )
    self.mock_env.close.assert_called_once()

    # Check rewards and returns
    # Step 2: reward = 2.0 (from env) + 0.5 (final) = 2.5
    # Step 1: reward = 1.0 (from env)
    self.assertEqual(result_traj.steps[0].reward, 1.0)
    self.assertEqual(result_traj.steps[1].reward, 2.5)

    # Check returns (gamma=0.9)
    # G_2 = 2.5
    # G_1 = 1.0 + 0.9 * 2.5 = 1.0 + 2.25 = 3.25
    self.assertAlmostEqual(result_traj.steps[1].mc_return, 2.5)
    self.assertAlmostEqual(result_traj.steps[0].mc_return, 3.25)
    self.assertAlmostEqual(result_traj.reward, 3.5)  # 1.0 + 2.5

  def test_collect_conversation_mode(self):
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        max_steps=5,
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
        ([201, 202], [1, 1]),  # assistant tokens 1
        ([301, 302], [1, 1]),  # env tokens 1
        ([203, 204], [1, 1]),  # assistant tokens 2
        ([303, 304], [1, 1]),  # env tokens 2
    ]
    engine = trajectory_collect_engine.TrajectoryCollectEngine(
        agent=self.mock_agent,
        env=self.mock_env,
        model_call=self.mock_model_call,
        tokenizer=self.mock_tokenizer,
        chat_parser=self.mock_chat_parser,
        max_steps=5,
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
        'prompt_tokens': [101],
        'conversation_tokens': [201, 202, 301, 302, 203, 204, 303, 304],
        'conversation_masks': [1, 1, 1, 1, 1, 1, 1, 1],
        'trajectory_reward': 3.0,  # 1.0 + 2.0
        'policy_version': None,
        'original_input': None,
        'group_id': None,
    }
    self.assertEqual(token_data, expected_tokens)

    # The function using the parser is mocked, so the parser itself is not
    # called. Instead, we check that the parser is passed as an argument.
    self.assertTrue(mock_convert.called)
    for call in mock_convert.call_args_list:
      self.assertIs(call.kwargs['parser'], self.mock_chat_parser)

    # Verify that the initial prompt tokenization in _reset is called with
    # contains_first_msg=True and contains_generation_msg=False.
    self.assertGreaterEqual(mock_convert.call_count, 1)
    self.assertTrue(
        mock_convert.call_args_list[0].kwargs['contains_first_msg'],
        'contains_first_msg should be True for initial prompt tokenization',
    )
    self.assertFalse(
        mock_convert.call_args_list[0].kwargs['contains_generation_msg'],
        'contains_generation_msg should be False for initial prompt'
        ' tokenization',
    )

    # Verify that tokenization for model responses and environment observations
    # has contains_generation_msg=False.
    self.assertEqual(mock_convert.call_count, 5)
    # Calls 1 and 3 are for model responses.
    self.assertFalse(
        mock_convert.call_args_list[1].kwargs['contains_generation_msg']
    )
    self.assertFalse(
        mock_convert.call_args_list[3].kwargs['contains_generation_msg']
    )
    # Calls 2 and 4 are for environment observations.
    self.assertFalse(
        mock_convert.call_args_list[2].kwargs['contains_generation_msg']
    )
    self.assertFalse(
        mock_convert.call_args_list[4].kwargs['contains_generation_msg']
    )

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

  def test_collect_timeout(self):
    with mock.patch.object(time, 'time') as mock_time:
      mock_time.side_effect = [
          100.0,  # start time in _reset
          100.05,  # time check in _one_step (1st call)
          100.11,  # time check in _one_step (2nd call) -> timeout
      ]
      engine = trajectory_collect_engine.TrajectoryCollectEngine(
          agent=self.mock_agent,
          env=self.mock_env,
          model_call=self.mock_model_call,
          max_steps=5,
          timeout=0.1,
      )
      result_traj = asyncio.run(self._run_collect(engine, mode='Trajectory'))

    # Should run for two steps, with the second one timing out and marked as
    # done
    self.assertLen(result_traj.steps, 2)
    self.assertFalse(result_traj.steps[0].done)
    self.assertTrue(result_traj.steps[1].done)
    self.assertEqual(self.mock_env.step.call_count, 2)

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
      agent.get_current_state.side_effect = lambda: current_step[0]

      def _reset_agent():
        traj.steps.clear()
        agent.chat_completions.clear()

      agent.reset.side_effect = _reset_agent
      return agent

    agent1 = configure_mock_agent('initial1')
    env1 = mock.create_autospec(base_environment.BaseTaskEnv, instance=True)
    env1.reset.return_value = ('initial1', {})
    env1.step.return_value = ('obs1', 1.0, True, {})
    env1.task = {}

    agent2 = configure_mock_agent('initial2')
    env2 = mock.create_autospec(base_environment.BaseTaskEnv, instance=True)
    env2.reset.return_value = ('initial2', {})
    env2.step.side_effect = [
        ('obs2a', 2.0, False, {}),
        ('obs2b', 2.1, True, {}),
    ]
    env2.task = {}

    pairs = [(agent1, env1), (agent2, env2)]
    mock_model_call = mock.Mock(side_effect=['resp1', 'resp2a', 'resp2b'])
    engine_args = {
        'model_call': mock_model_call,
        'max_steps': 5,
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


if __name__ == '__main__':
  absltest.main()
