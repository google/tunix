# Copyright 2026 Google LLC
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

from absl.testing import absltest
from tunix.experimental.examples.math_gsm8k_dist import gsm8k
from tunix.experimental.rl.agentic import registry


class GSM8KTest(absltest.TestCase):

  def test_registered_components(self):
    self.assertIs(
        registry.ENV_REGISTRY.get(gsm8k.GSM8K_ENV_NAME), gsm8k.GSM8KEnv
    )
    self.assertIs(
        registry.AGENT_REGISTRY.get(gsm8k.GSM8K_AGENT_NAME), gsm8k.GSM8KAgent
    )

  def test_env_scores_answer_containing_gold_answer(self):
    env = gsm8k.GSM8KEnv(prompt="What is 2+2?", gold_answer="4")
    obs, _ = env.reset()
    self.assertEqual(obs, {"prompts": "What is 2+2?"})

    next_obs, reward, done, info = env.step("The answer is 4.")

    self.assertEqual(next_obs["gold_answer"], "4")
    self.assertEqual(reward, 1.0)
    self.assertTrue(done)
    self.assertTrue(info["correct"])

  def test_agent_forwards_model_response_as_action(self):
    agent = gsm8k.GSM8KAgent()
    action = agent.update_from_model("The answer is 4.")

    self.assertEqual(action.action, "The answer is 4.")
    self.assertEqual(agent.name, gsm8k.GSM8K_AGENT_NAME)
    self.assertLen(agent.trajectory.steps, 1)


if __name__ == "__main__":
  absltest.main()
