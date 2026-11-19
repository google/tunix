# Copyright 2026 Google LLC
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

"""Unit tests for GSM8KEnv and GSM8KAgent."""

import unittest

from examples.math_gsm8k import env as gsm8k_env
from examples.math_gsm8k import agent as gsm8k_agent


class GSM8KUtilitiesTest(unittest.TestCase):

  def test_extract_hash_answer(self):
    self.assertEqual(
        gsm8k_env.extract_hash_answer(
            "Natalia sold 48 clips in April. In May she sold half. In total 48 + 24 = 72. #### 72"
        ),
        "72",
    )
    self.assertEqual(
        gsm8k_env.extract_hash_answer("The cost is $1,250. #### 1,250"),
        "1,250",
    )
    self.assertEqual(gsm8k_env.extract_hash_answer("72"), "72")
    self.assertIsNone(gsm8k_env.extract_hash_answer(""))

  def test_extract_boxed_answer(self):
    # Standard format
    text1 = "<reasoning>Step by step</reasoning>\n<answer>\\boxed{72}</answer>"
    self.assertEqual(gsm8k_env.extract_boxed_answer(text1), "72")

    # Nested braces
    text2 = "<answer>\\boxed{\\frac{72}{1}}</answer>"
    self.assertEqual(gsm8k_env.extract_boxed_answer(text2), "\\frac{72}{1}")

    # Text markup
    text3 = "<answer>\\boxed{\\text{85}}</answer>"
    self.assertEqual(gsm8k_env.extract_boxed_answer(text3), "\\text{85}")

    # Unclosed brace fallback
    text4 = "The result is \\boxed{36"
    self.assertEqual(gsm8k_env.extract_boxed_answer(text4), "36")

    # Raw <answer> tag
    text5 = "<reasoning>Done</reasoning><answer>42</answer>"
    self.assertEqual(gsm8k_env.extract_boxed_answer(text5), "42")

    # Trailing numeric fallback
    text6 = "Therefore the final amount is 85 clips."
    self.assertEqual(gsm8k_env.extract_boxed_answer(text6), "85")

  def test_normalize_answer(self):
    self.assertEqual(gsm8k_env.normalize_answer("72,000"), "72000")
    self.assertEqual(gsm8k_env.normalize_answer("$85"), "85")
    self.assertEqual(gsm8k_env.normalize_answer("50%"), "50")
    self.assertEqual(gsm8k_env.normalize_answer("72.0"), "72")
    self.assertEqual(gsm8k_env.normalize_answer("\\text{36}"), "36")
    self.assertIsNone(gsm8k_env.normalize_answer(None))

  def test_answers_match(self):
    self.assertTrue(gsm8k_env.answers_match("72", "72"))
    self.assertTrue(gsm8k_env.answers_match("72.0", "72"))
    self.assertTrue(gsm8k_env.answers_match("1,000", "1000"))
    self.assertTrue(gsm8k_env.answers_match("$85", "85"))
    self.assertFalse(gsm8k_env.answers_match("72", "36"))
    self.assertFalse(gsm8k_env.answers_match(None, "72"))

  def test_is_format_correct(self):
    valid_vtc = (
        "<reasoning>First 48, then 24.</reasoning>\n"
        "<answer>\\boxed{72}</answer>"
    )
    self.assertTrue(gsm8k_env.is_format_correct(valid_vtc))

    valid_think = (
        "<think>First 48, then 24.</think>\n"
        "<answer>\\boxed{72}</answer>"
    )
    self.assertTrue(gsm8k_env.is_format_correct(valid_think))

    missing_reasoning = "<answer>\\boxed{72}</answer>"
    self.assertFalse(gsm8k_env.is_format_correct(missing_reasoning))

    missing_answer = "<reasoning>First 48, then 24.</reasoning> 72"
    self.assertFalse(gsm8k_env.is_format_correct(missing_answer))


class GSM8KEnvTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.sample_task = {
        "question": (
            "Natalia sold clips to 48 friends in April, and then she sold half"
            " as many clips in May. How many clips did Natalia sell altogether"
            " in April and May?"
        ),
        "answer": "In April 48. In May 24. Total = 72. #### 72",
    }

  def test_initial_observation(self):
    env = gsm8k_env.GSM8KEnv(entry=self.sample_task)
    obs, info = env.reset()

    self.assertIn("question", obs)
    self.assertEqual(obs["gold_answer"], "72")
    self.assertIn("Problem: Natalia sold clips", obs["prompts"])
    self.assertEqual(info, {})

  def test_step_correct_format_and_answer(self):
    env = gsm8k_env.GSM8KEnv(entry=self.sample_task)
    env.reset()

    action = (
        "<reasoning>\n"
        "Natalia sold 48 in April. In May: 48 / 2 = 24.\n"
        "Total = 48 + 24 = 72.\n"
        "</reasoning>\n"
        "<answer>\\boxed{72}</answer>"
    )
    obs, reward, done, info = env.step(action)

    self.assertEqual(reward, 1.0)
    self.assertTrue(done)
    self.assertTrue(info["format_correct"])
    self.assertTrue(info["answer_correct"])
    self.assertEqual(info["extracted_answer"], "72")
    self.assertTrue(env.success())

  def test_step_correct_format_wrong_answer(self):
    env = gsm8k_env.GSM8KEnv(entry=self.sample_task)
    env.reset()

    action = (
        "<reasoning>\n"
        "Natalia sold 48 in April. In May: 48 / 2 = 20.\n"
        "Total = 48 + 20 = 68.\n"
        "</reasoning>\n"
        "<answer>\\boxed{68}</answer>"
    )
    obs, reward, done, info = env.step(action)

    self.assertEqual(reward, 0.1)  # Format bonus
    self.assertTrue(done)
    self.assertTrue(info["format_correct"])
    self.assertFalse(info["answer_correct"])
    self.assertEqual(info["extracted_answer"], "68")
    self.assertFalse(env.success())

  def test_step_wrong_format_correct_answer(self):
    env = gsm8k_env.GSM8KEnv(entry=self.sample_task)
    env.reset()

    action = "She sold 48 + 24 = 72 clips altogether."
    obs, reward, done, info = env.step(action)

    self.assertEqual(reward, 0.5)  # Partial accuracy reward
    self.assertTrue(done)
    self.assertFalse(info["format_correct"])
    self.assertTrue(info["answer_correct"])
    self.assertEqual(info["extracted_answer"], "72")
    self.assertTrue(env.success())

  def test_step_both_wrong(self):
    env = gsm8k_env.GSM8KEnv(entry=self.sample_task)
    env.reset()

    action = "I am not sure how to solve this."
    obs, reward, done, info = env.step(action)

    self.assertEqual(reward, 0.0)
    self.assertTrue(done)
    self.assertFalse(info["format_correct"])
    self.assertFalse(info["answer_correct"])
    self.assertFalse(env.success())

  def test_from_dict_factory(self):
    env = gsm8k_env.GSM8KEnv.from_dict({
        "question": "What is 2 + 2?",
        "answer": "#### 4",
        "group_id": "test_group_1",
        "pair_index": 0,
    })
    self.assertEqual(env.question, "What is 2 + 2?")
    self.assertEqual(env.gold_answer, "4")
    self.assertEqual(env.group_id, "test_group_1")


class GSM8KAgentInteractionTest(unittest.TestCase):

  def test_agent_env_interaction(self):
    task = {
        "question": "Weng babysat for 3 hours at $12/hr. How much did she earn?",
        "answer": "3 * 12 = 36. #### 36",
    }
    env = gsm8k_env.GSM8KEnv(entry=task)
    agent = gsm8k_agent.GSM8KAgent()

    obs, _ = env.reset()
    agent._observation_to_messages(obs, reward=0.0, done=False)

    # Verify agent's chat completions have system prompt and user question
    completions = agent.chat_completions
    self.assertEqual(len(completions), 2)
    self.assertEqual(completions[0]["role"], "system")
    self.assertEqual(completions[1]["role"], "user")
    self.assertIn("Weng babysat", completions[1]["content"])

    # Simulate model response
    model_response = (
        "<reasoning>Weng works 3 hours at $12. 3 * 12 = 36.</reasoning>\n"
        "<answer>\\boxed{36}</answer>"
    )
    action = agent.update_from_model(model_response)

    # Step environment with agent's action
    next_obs, reward, done, info = env.step(action)
    self.assertEqual(reward, 1.0)
    self.assertTrue(done)
    self.assertTrue(env.success())
    self.assertEqual(len(agent.trajectory.steps), 1)


class GSM8KDataTest(unittest.TestCase):

  def test_smoke_test_dataset(self):
    from examples.math_gsm8k import data as gsm8k_data

    ds = gsm8k_data.create_smoke_test_dataset()
    self.assertEqual(len(ds), 4)
    self.assertEqual(ds[0]["gold_answer"], "72")
    self.assertIn("<reasoning>", ds[0]["prompts"])
    self.assertEqual(ds[1]["gold_answer"], "36")


if __name__ == "__main__":
  unittest.main()

