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

"""Tests for tunix.examples.data.math_dataset.apply_template."""

from __future__ import annotations

from absl.testing import absltest
from absl.testing import parameterized
from tunix.examples.data import math_dataset


class _FakeTokenizer:

  def __init__(self, template_suffix=""):  # suffix helps distinguish calls
    self.calls = []
    self.template_suffix = template_suffix

  def apply_chat_template(
      self, msgs, tokenize=False, add_generation_prompt=True
  ):
    # Capture call arguments for assertion
    self.calls.append({
        "msgs": msgs,
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
    })
    # Return a simple rendered string to check wiring
    user_msg = next(m["content"] for m in msgs if m["role"] == "user")
    return f"templated:{user_msg}{self.template_suffix}"


class _BaseDataset:

  def __init__(self, records):
    self._records = list(records)

  def map(self, fn):
    return _BaseDataset([fn(x) for x in self._records])

  def __iter__(self):
    return iter(self._records)

  def __getitem__(self, idx):
    return self._records[idx]


class ApplyTemplateTest(parameterized.TestCase):

  def test_with_tokenizer_uses_apply_chat_template_and_preserves_fields(self):
    ds = _BaseDataset([
        {"question": "Q1", "answer": "#### A1", "extra": 7},
    ])
    tok = _FakeTokenizer(template_suffix="!call")

    out = math_dataset.apply_template(
        ds, tokenizer=tok, tokenize=True, add_generation_prompt=False
    )
    result = out[0]

    # Prompts rendered via tokenizer
    self.assertEqual(result["prompts"], "templated:Q1!call")
    # Original fields preserved
    self.assertEqual(result["question"], "Q1")
    self.assertEqual(result["answer"], "A1")
    # Extra fields are dropped by apply_template
    self.assertNotIn("extra", result)

    # Tokenizer called once with expected messages and flags
    self.assertLen(tok.calls, 1)
    call = tok.calls[0]
    self.assertEqual(call["tokenize"], True)
    self.assertEqual(call["add_generation_prompt"], False)
    self.assertEqual(len(call["msgs"]), 2)
    self.assertEqual(call["msgs"][1]["content"], "Q1")
    self.assertIn(math_dataset.SYSTEM_PROMPT, call["msgs"][0]["content"])

  def test_without_tokenizer_falls_back_to_plain_template(self):
    ds = _BaseDataset([
        {"question": "Q2", "answer": "#### A2"},
    ])

    out = math_dataset.apply_template(
        ds, tokenizer=None, tokenize=False, add_generation_prompt=True
    )
    result = out[0]

    # Prompts rendered via fallback TEMPLATE
    self.assertIn("<start_of_turn>user", result["prompts"])
    self.assertIn("Q2", result["prompts"])
    self.assertIn(math_dataset.SYSTEM_PROMPT, result["prompts"])
    self.assertEqual(result["answer"], "A2")

  def test_no_valid_answer(self):
    ds = _BaseDataset([
        {"question": "Q2", "answer": "A2"},
    ])

    out = math_dataset.apply_template(
        ds, tokenizer=None, tokenize=False, add_generation_prompt=True
    )
    result = out[0]

    # Prompts rendered via fallback TEMPLATE
    self.assertIn("<start_of_turn>user", result["prompts"])
    self.assertIn("Q2", result["prompts"])
    self.assertIn(math_dataset.SYSTEM_PROMPT, result["prompts"])
    self.assertIsNone(result["answer"])

  @parameterized.parameters(True, False)
  def test_bytes_are_decoded_to_str(self, use_tokenizer):
    ds = _BaseDataset([
        {"question": b"byte-q", "answer": b"#### byte-a"},
    ])
    tok = _FakeTokenizer() if use_tokenizer else None

    out = math_dataset.apply_template(
        ds, tokenizer=tok, tokenize=False, add_generation_prompt=True
    )
    result = out[0]

    self.assertIsInstance(result["question"], str)
    self.assertIsInstance(result["answer"], str)
    self.assertEqual(result["question"], "byte-q")
    self.assertEqual(result["answer"], "byte-a")


if __name__ == "__main__":
  absltest.main()
