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

"""Unit tests for OpenMathInstruct-2 dataset, environment, agent, and rewards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import pathlib
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import datasets
import transformers
from tunix.dist_rl.common import registry
from tunix.dist_rl.examples.openmath_dist import openmath
from tunix.dist_rl.examples.openmath_dist import openmath_rewards
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import orchestrator

import multiprocessing

_YAML_PATH = pathlib.Path(openmath.__file__).parent / "openmath.yaml"


class _FakeTokenizer:
  """Minimal offline tokenizer matching Qwen3 chat-template behavior."""

  def apply_chat_template(
      self,
      messages: Sequence[Mapping[str, str]],
      *,
      tokenize: bool = False,
      add_generation_prompt: bool = True,
  ) -> str:
    del tokenize
    body = "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    )
    if add_generation_prompt:
      body += "<|im_start|>assistant\n"
    return body

  def tokenize(self, text: str) -> list[str]:
    return text.split()


class OpenMathTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="nested_boxed",
          response=(
              "<reasoning>Step 1</reasoning>"
              "<answer>Therefore \\boxed{\\frac{1 + \\sqrt{5}}{2}}.</answer>"
          ),
          expected="\\frac{1 + \\sqrt{5}}{2}",
      ),
      dict(
          testcase_name="last_boxed_wins",
          response="<answer>First \\boxed{10}, correction \\boxed{42}</answer>",
          expected="42",
      ),
      dict(
          testcase_name="plain_answer_tag",
          response="<reasoning>ok</reasoning><answer> 128 </answer>",
          expected="128",
      ),
      dict(
          testcase_name="fallback_when_missing",
          response="no tags or boxed answer",
          expected=openmath.FALLBACK_ANSWER,
      ),
  )
  def test_extract_answer(self, response: str, expected: str):
    self.assertEqual(openmath.extract_answer(response), expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="format_and_symbolic_fraction_equivalence",
          completion=(
              "<reasoning>2 divided by 4 simplifies to 1/2.</reasoning>"
              "<answer>\\boxed{\\frac{2}{4}}</answer>"
          ),
          gold=json.dumps(["\\frac{1}{2}"]),
          expected_reward=1.1,
          expected_format=True,
          expected_correct=True,
      ),
      dict(
          testcase_name="nested_sqrt_and_rationalized_fraction_equivalence",
          completion=(
              "<reasoning>1/sqrt(2) = sqrt(2)/2.</reasoning>"
              "<answer>\\boxed{\\frac{1}{\\sqrt{2}}}</answer>"
          ),
          gold=json.dumps(["\\frac{\\sqrt{2}}{2}"]),
          expected_reward=1.1,
          expected_format=True,
          expected_correct=True,
      ),
      dict(
          testcase_name="nested_radicals_and_braced_exponents",
          completion=(
              "<reasoning>1 + sqrt(4) = 3 and 2^10 = 1024.</reasoning>"
              "<answer>\\boxed{\\sqrt{1 + \\sqrt{4}} \\cdot 2^{10}}</answer>"
          ),
          gold=json.dumps(["1024 \\sqrt{3}"]),
          expected_reward=1.1,
          expected_format=True,
          expected_correct=True,
      ),
      dict(
          testcase_name="correct_boxed_without_reasoning_tag",
          completion="<answer>\\boxed{42}</answer>",
          gold="42",
          expected_reward=1.0,
          expected_format=False,
          expected_correct=True,
      ),
      dict(
          testcase_name="format_only_wrong_answer",
          completion="<reasoning>guess</reasoning><answer>\\boxed{99}</answer>",
          gold="42",
          expected_reward=0.1,
          expected_format=True,
          expected_correct=False,
      ),
      dict(
          testcase_name="unclosed_repeated_tags_do_not_backtrack",
          completion=(
              "<reasoning>" + ("</reasoning><answer>" * 2000) + "\\boxed{42}"
          ),
          gold="42",
          expected_reward=1.0,
          expected_format=False,
          expected_correct=True,
      ),
  )
  def test_score_openmath_completion(
      self,
      completion: str,
      gold: str,
      expected_reward: float,
      expected_format: bool,
      expected_correct: bool,
  ):
    reward, info = openmath.score_openmath_completion(completion, gold)
    self.assertAlmostEqual(reward, expected_reward)
    self.assertEqual(info["format_correct"], expected_format)
    self.assertEqual(info["answer_correct"], expected_correct)

  def test_verifier_recovers_after_timeout(self):
    runner = openmath._MathVerifyRunner(max_workers=1)
    self.assertTrue(runner.verify_symbolic(["\\boxed{1/2}"], "\\boxed{0.5}"))
    first_slot = runner._slots.queue[0]
    assert first_slot is not None
    first_proc = first_slot[0]
    first_pid = first_proc.pid
    self.assertIsNotNone(first_pid)
    self.assertTrue(first_proc.is_alive())

    with mock.patch.object(
        openmath, "_wait_for_worker_reply", autospec=True, return_value=False
    ):
      self.assertFalse(
          runner.verify_symbolic(["\\boxed{x+1}"], "\\boxed{1+x}", timeout_s=1)
      )
    self.assertFalse(first_proc.is_alive())
    self.assertIsNone(runner._slots.queue[0])

    self.assertTrue(runner.verify_symbolic(["\\boxed{3/6}"], "\\boxed{1/2}"))
    second_slot = runner._slots.queue[0]
    assert second_slot is not None
    self.assertTrue(second_slot[0].is_alive())
    self.assertNotEqual(second_slot[0].pid, first_pid)

  def test_extract_question_and_answer_skips_none_values(self):
    q, a = openmath._extract_question_and_answer({
        "problem": None,
        "question": "Compute 1+1.",
        "expected_answer": None,
        "answer": "2",
    })
    self.assertEqual((q, a), ("Compute 1+1.", "2"))

  def test_process_mcq_accepts_both_letter_and_option_value(self):
    question = "What is 5+5? (A) 10 \\quad (B) 11 \\quad (C) 12"
    self.assertEqual(list(openmath.process_mcq(question, "A")), ["A", "10"])
    self.assertEqual(list(openmath.process_mcq(question, "10")), ["10", "A"])

  def test_create_dataset_and_orchestrator_prompt_items(self):
    tmpdir = self.create_tempdir().full_path
    parquet_path = str(pathlib.Path(tmpdir) / "train_1M-00000.parquet")
    rows = [
        {
            "problem": "Compute 2 + 3.",
            "expected_answer": "5",
            "question_type": "default",
        },
        {
            "problem": " ".join(["word"] * 200),
            "expected_answer": "overflow",
            "question_type": "default",
        },
        {
            "problem": "Pick ten: (A) 10 \\quad (B) 20",
            "expected_answer": "A",
            "question_type": "MCQ",
        },
    ]
    datasets.Dataset.from_list(rows).to_parquet(parquet_path)

    with mock.patch.object(
        transformers.AutoTokenizer,
        "from_pretrained",
        autospec=True,
        return_value=_FakeTokenizer(),
    ):
      hp = resolve_config.initialize([
          f"override_config_file={_YAML_PATH}",
          "actor_model_config.model_path=gs://test-bucket/qwen3-0.6b/0/items",
          f"data_config.data_files={parquet_path}",
          "data_config.max_prompt_length=120",
          "data_config.shuffle=false",
          "data_config.max_examples=10",
      ])
      _, rpc_timeout_s = orchestrator.resolve_weight_sync_timeouts(hp)
      self.assertGreaterEqual(rpc_timeout_s, 3600.0)
      extra_cfg = hp.config["dist_training_config"]["maxtext_extra_config"]
      self.assertEqual(extra_cfg["steps"], 20)
      self.assertEqual(extra_cfg["learning_rate_schedule_steps"], 20)
      self.assertEqual(extra_cfg["learning_rate_final_fraction"], 0.0)
      self.assertLessEqual(extra_cfg["sa_block_kv_dkv"], 512)
      self.assertEqual(hp.config["actor_model_config"]["dtype"], "bfloat16")
      self.assertEqual(hp.config["actor_model_config"]["load_dtype"], "float32")
      prompt_items = orchestrator.load_prompt_items(hp)
      first = prompt_items[0]
      second = prompt_items[1]
      # Repeats across epochs after the 2 kept examples (row 1 filtered out).
      third = prompt_items[2]

    assert first is not None and second is not None and third is not None
    first_meta = first["metadata"]
    second_meta = second["metadata"]
    third_meta = third["metadata"]
    assert isinstance(first_meta, Mapping)
    assert isinstance(second_meta, Mapping)
    assert isinstance(third_meta, Mapping)
    self.assertIn("<start_of_turn>user", str(first["prompt"]))
    self.assertEqual(first["task_name"], "openmath")
    self.assertEqual(first["max_turns"], 1)
    self.assertEqual(second_meta["answer"], json.dumps(["A", "10"]))
    self.assertEqual(first_meta["question"], third_meta["question"])

  def test_env_agent_and_orchestrator_reward_fn_agree(self):
    env_cls = registry.ENV_REGISTRY.get(openmath.OPENMATH_ENV_NAME)
    agent_cls = registry.AGENT_REGISTRY.get(openmath.OPENMATH_AGENT_NAME)
    env = env_cls(
        prompt="p",
        question="What is 6*7?",
        answer=json.dumps(["42"]),
    )
    agent = agent_cls()
    obs, _ = env.reset()
    self.assertEqual(obs, {"prompts": "p"})

    model_out = "<reasoning>6*7=42</reasoning><answer>\\boxed{42}</answer>"
    action = agent.update_from_model(model_out)
    _, reward, done, info = env.step(action)
    self.assertTrue(done)
    self.assertAlmostEqual(reward, 1.1)
    self.assertTrue(info["correct"])

    scores = openmath_rewards.reward_fn(
        prompts=["p"],
        completions=[model_out],
        answer=[json.dumps(["42"])],
    )
    self.assertLen(scores, 1)
    self.assertAlmostEqual(scores[0], 1.1)


if __name__ == "__main__":
  absltest.main()
