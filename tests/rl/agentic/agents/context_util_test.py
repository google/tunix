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

import unittest
from unittest import mock
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import context_util


class ContextUtilTest(unittest.TestCase):

  def setUp(self):
    # Mock tokenizer returns length of string for simple deterministic testing
    self.mock_tokenizer = mock.Mock(side_effect=len)

  def test_safe_serialize_handles_complex_types(self):
    data = {"key": "value", "number": 123}

    result = context_util.safe_serialize(data)
    self.assertEqual(result, '{"key": "value", "number": 123}')

    self.assertEqual(context_util.safe_serialize(set([1])), '"{1}"')

  def test_count_multimodal_observation_with_media(self):
    # Test dictionary with a mix of "media" and "text"
    obs = {
        "image": "binary_blob_ignored",
        "text_field": "hello",
    }
    # Expected: 576 (default image cost) + 21 (length of '{"text_field": "hello"}')
    count = context_util.count_multimodal_observation(obs, self.mock_tokenizer)

    self.assertGreater(count, 576)
    # Verify the image blob itself wasn't passed to the tokenizer
    self.mock_tokenizer.assert_called_with('{"text_field": "hello"}')

  def test_count_step_tokens(self):
    step = agent_types.Step(
        thought="I should act",  # len = 12
        model_response="Action: search",  # len = 14
        action=agent_types.Action(
            action={"cmd": "search"}
        ),  # serialized len = 18
        observation="Found it",  # len = 8 (as a string, it's returned as-is)
    )

    # 12 + 14 + 18 + 8 = 52.
    # Let's adjust the expectation to match the actual serialized lengths:
    total = context_util.count_step_tokens(step, self.mock_tokenizer)

    expected_thought = 12
    expected_response = 14
    expected_action = len('{"cmd": "search"}')  # 18
    expected_obs = 8  # "Found it" (string returns as-is)

    self.assertEqual(
        total,
        expected_thought + expected_response + expected_action + expected_obs,
    )


def test_count_step_tokens_skips_duplicate_thought(self):
  # If model_response IS the thought, it shouldn't be counted twice
  step = agent_types.Step(thought="Thinking...", model_response="Thinking...")
  total = context_util.count_step_tokens(step, self.mock_tokenizer)
  self.assertEqual(total, 11)  # Only counted once


def test_calculate_trajectory_tokens(self):
  traj = agent_types.Trajectory(
      task="Find a cat",
      steps=[
          agent_types.Step(thought="Step 1"),
          agent_types.Step(thought="Step 2"),
      ],
  )

  # "Find a cat" (10) + "Step 1" (6) + "Step 2" (6)
  total = context_util.calculate_trajectory_tokens(traj, self.mock_tokenizer)
  self.assertEqual(total, 10 + 6 + 6)


def test_custom_media_costs(self):
  custom_costs = {"special_sensor": 1000}
  obs = {"special_sensor": "data", "other": "info"}

  count = context_util.count_multimodal_observation(
      obs, self.mock_tokenizer, media_costs=custom_costs
  )
  # 1000 + len('{"other": "info"}')
  self.assertEqual(count, 1017)


if __name__ == "__main__":
  unittest.main()
