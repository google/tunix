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

import types
from unittest import mock
from absl.testing import absltest
import numpy as np
import tensorflow_datasets as tfds
from tunix.experimental.examples.math_gsm8k_dist import gsm8k
from tunix.experimental.rl.agentic import registry


class _FakeMapDataSource:

  def __init__(self, data):
    self._data = list(data)

  def __len__(self):
    return len(self._data)

  def __getitem__(self, idx):
    return self._data[idx]


class GSM8KTest(absltest.TestCase):

  def test_registered_components(self):
    self.assertIs(
        registry.ENV_REGISTRY.get(gsm8k.GSM8K_ENV_NAME), gsm8k.GSM8KEnv
    )
    self.assertIs(
        registry.AGENT_REGISTRY.get(gsm8k.GSM8K_AGENT_NAME), gsm8k.GSM8KAgent
    )

  def test_env_scores_formatted_correct_answer(self):
    env = gsm8k.GSM8KEnv(prompt="What is 2+2?", answer="4")
    obs, _ = env.reset()
    self.assertEqual(obs, {"prompts": "What is 2+2?"})
    next_obs, reward, done, info = env.step(
        "2+2=4.</reasoning><answer>\\boxed{4}</answer>"
    )
    self.assertEqual(next_obs["gold_answer"], "4")
    self.assertEqual(reward, 1.0)
    self.assertTrue(done)
    self.assertTrue(info["correct"])
    self.assertTrue(info["format_correct"])

  def test_scores_formatted_wrong_answer_with_format_reward(self):
    reward, info = gsm8k.score_gsm8k_completion(
        "2+2=5.</reasoning><answer>\\boxed{5}</answer>", "4"
    )
    self.assertEqual(reward, 0.1)
    self.assertTrue(info["format_correct"])
    self.assertFalse(info["answer_correct"])

  def test_build_prompt_and_hash_answer(self):
    self.assertEqual(gsm8k.extract_hash_answer("reasoning #### 72"), "72")
    self.assertIn(
        "<answer>\\boxed{}</answer>",
        gsm8k.build_prompt("How many clips?"),
    )
    self.assertIn(
        "immediately end your response and stop generating",
        gsm8k.build_prompt("How many clips?"),
    )

  def test_agent_forwards_model_response_as_action(self):
    agent = gsm8k.GSM8KAgent()
    action = agent.update_from_model("The answer is 4.")
    self.assertEqual(action.action, "The answer is 4.")
    self.assertEqual(agent.name, gsm8k.GSM8K_AGENT_NAME)
    self.assertLen(agent.trajectory.steps, 1)

  def test_normalize_example_value(self):
    self.assertEqual(
        gsm8k.normalize_example_value(np.array(["test"])),
        "test",
    )
    self.assertEqual(
        gsm8k.normalize_example_value(np.bytes_(b"bytes_val")),
        "bytes_val",
    )
    self.assertEqual(
        gsm8k.normalize_example_value(b"raw_bytes"),
        "raw_bytes",
    )
    self.assertEqual(
        gsm8k.normalize_example_value(np.array(["a", "b"])),
        ["a", "b"],
    )
    self.assertEqual(gsm8k.normalize_example_value("string"), "string")
    self.assertEqual(gsm8k.normalize_example_value(42), 42)

  def test_as_text(self):
    self.assertEqual(gsm8k.as_text(b"hello"), "hello")
    self.assertEqual(gsm8k.as_text(np.array(["world"])), "world")
    self.assertEqual(gsm8k.as_text(123), "123")

  def test_load_gsm8k_dataset(self):
    fake_records = [
        {
            "question": np.array([b"What is 10 + 5?"]),
            "answer": np.array([b"10 + 5 is 15 #### 15"]),
        },
        {
            "question": "What is 3 * 7?",
            "answer": "3 * 7 = 21 #### 21",
        },
    ]
    fake_source = _FakeMapDataSource(fake_records)

    with mock.patch.object(
        tfds, "data_source", return_value=fake_source
    ) as mock_ds:
      dataset = gsm8k.load_gsm8k_dataset(
          split="train",
          data_dir="/tmp/test_dir",
          shuffle=False,
          seed=123,
      )

    mock_ds.assert_called_once_with(
        "gsm8k",
        split="train",
        data_dir="/tmp/test_dir",
        builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
        download=True,
    )
    self.assertLen(dataset, 2)
    first_item = dataset[0]
    self.assertEqual(first_item["question"], "What is 10 + 5?")
    self.assertEqual(first_item["answer"], "15")
    self.assertIn("What is 10 + 5?", first_item["prompts"])
    self.assertIn("<answer>\\boxed{}</answer>", first_item["prompts"])

    second_item = dataset[1]
    self.assertEqual(second_item["question"], "What is 3 * 7?")
    self.assertEqual(second_item["answer"], "21")

  def test_make_gsm8k_reward_fn(self):
    reward_fn = gsm8k.make_gsm8k_reward_fn(debug=True)

    # 1. Format correct and answer correct -> reward 1.0
    item1 = types.SimpleNamespace(
        metadata={
            "text": "Reasoning step.</reasoning><answer>\\boxed{42}</answer>",
            "gold_answer": "42",
            "prompt_id": "p1",
        }
    )
    self.assertEqual(reward_fn(item1), 1.0)

    # 2. Format correct, wrong answer -> reward 0.1
    item2 = types.SimpleNamespace(
        metadata={
            "text": "Reasoning step.</reasoning><answer>\\boxed{100}</answer>",
            "gold_answer": "42",
            "prompt_id": "p2",
        }
    )
    self.assertEqual(reward_fn(item2), 0.1)

    # 3. Format incorrect, right answer -> reward 0.0 (strict formatting)
    item3 = types.SimpleNamespace(
        metadata={
            "text": "The answer is \\boxed{42}",
            "gold_answer": "42",
            "prompt_id": "p3",
        }
    )
    self.assertEqual(reward_fn(item3), 0.0)

    # 4. Format incorrect, wrong answer -> reward 0.0
    item4 = types.SimpleNamespace(
        metadata={
            "text": "No answer here",
            "gold_answer": "42",
            "prompt_id": "p4",
        }
    )
    self.assertEqual(reward_fn(item4), 0.0)

    # 5. Uses 'answer' key in metadata fallback
    item5 = types.SimpleNamespace(
        metadata={
            "text": "Reasoning.</reasoning><answer>\\boxed{15}</answer>",
            "answer": "15",
        }
    )
    self.assertEqual(reward_fn(item5), 1.0)


if __name__ == "__main__":
  absltest.main()
