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

import json

from absl.testing import absltest
from tunix.rl.agentic.environments import tool_environment

ToolEnvironment = tool_environment.ToolEnvironment


class ExtractLlmAnswerTest(absltest.TestCase):

  def test_plain_string_action(self):
    self.assertEqual(ToolEnvironment._extract_llm_answer("42"), "42")

  def test_finish_with_dict_arguments(self):
    action = [
        {"function": {"name": "finish", "arguments": {"response": "42"}}}
    ]
    self.assertEqual(ToolEnvironment._extract_llm_answer(action), "42")

  def test_finish_with_json_string_arguments(self):
    # ToolAgent.update_from_model serializes parsed tool-call arguments with
    # json.dumps, so an explicit finish tool call emitted by the model (e.g.
    # <tool_call>{"name": "finish", ...}</tool_call>) arrives here with
    # `arguments` as a JSON string rather than a dict.
    action = [{
        "function": {
            "name": "finish",
            "arguments": json.dumps({"response": "42"}),
        }
    }]
    self.assertEqual(ToolEnvironment._extract_llm_answer(action), "42")

  def test_finish_with_non_json_string_arguments(self):
    action = [{"function": {"name": "finish", "arguments": "plain text"}}]
    self.assertEqual(
        ToolEnvironment._extract_llm_answer(action), "plain text"
    )


class ToolEnvironmentStepTest(absltest.TestCase):

  def test_step_with_json_string_finish_computes_reward(self):
    env = ToolEnvironment(
        task={"question": "q"},
        tool_map={},
        reward_fn=lambda task, action: 1.0 if action == "42" else 0.0,
        max_steps=3,
    )
    env.reset()

    _, reward, done, _ = env.step([{
        "function": {
            "name": "finish",
            "arguments": json.dumps({"response": "42"}),
        }
    }])

    self.assertTrue(done)
    self.assertEqual(reward, 1.0)


if __name__ == "__main__":
  absltest.main()
