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

"""Tests for the shared GSM8K VTC recipe."""

import types

from absl.testing import absltest
import numpy as np
from tunix.utils import gsm8k_vtc


class Gsm8kVtcTest(absltest.TestCase):

  def test_prompt_opens_reasoning_block(self):
    prompt = gsm8k_vtc.build_prompt("How many clips?")
    self.assertIn("Problem: How many clips?", prompt)
    self.assertTrue(prompt.endswith("<reasoning>\n"))
    self.assertIn("<answer>\\boxed{}</answer>", prompt)

  def test_extract_hash_answer(self):
    self.assertEqual(gsm8k_vtc.extract_hash_answer("work #### 72"), "72")
    self.assertIsNone(gsm8k_vtc.extract_hash_answer("no delimiter"))

  def test_extract_boxed_answer(self):
    self.assertEqual(
        gsm8k_vtc.extract_boxed_answer("<answer>\\boxed{7}</answer>"), "7"
    )
    # Nested braces and the last block win.
    self.assertEqual(
        gsm8k_vtc.extract_boxed_answer(
            "<answer>\\boxed{1}</answer><answer>\\boxed{\\frac{3}{4}}</answer>"
        ),
        "\\frac{3}{4}",
    )
    # Loose \boxed outside any answer block still counts.
    self.assertEqual(gsm8k_vtc.extract_boxed_answer("so \\boxed 12 done"), "12")
    # An answer block without \boxed is not an answer.
    self.assertIsNone(gsm8k_vtc.extract_boxed_answer("<answer>42</answer>"))

  def test_format_requires_closing_tag_and_ordered_answer_only(self):
    # A completion that follows the prompt (closing tag only) is well-formed.
    self.assertTrue(
        gsm8k_vtc.is_vtc_format_correct(
            "2+2=4\n</reasoning>\n<answer>\\boxed{4}</answer>"
        )
    )
    self.assertTrue(
        gsm8k_vtc.is_vtc_format_correct(
            "<reasoning>work</reasoning><answer>\\boxed{4}</answer>"
        )
    )
    self.assertFalse(
        gsm8k_vtc.is_vtc_format_correct(
            "<answer>\\boxed{4}</answer></reasoning>"
        )
    )
    self.assertFalse(
        gsm8k_vtc.is_vtc_format_correct(
            "</reasoning></reasoning><answer>\\boxed{4}</answer>"
        )
    )
    self.assertFalse(gsm8k_vtc.is_vtc_format_correct("\\boxed{4}"))
    self.assertFalse(
        gsm8k_vtc.is_vtc_format_correct("<think>2+2=4</think>\\boxed{4}")
    )

  def test_completion_outcome_tiers(self):
    good = "2+2=4</reasoning><answer>\\boxed{4}</answer>"
    self.assertEqual(gsm8k_vtc.vtc_completion_outcome(good, "4")[0], 1.0)
    self.assertEqual(gsm8k_vtc.vtc_completion_outcome(good, "5")[0], 0.1)
    self.assertEqual(
        gsm8k_vtc.vtc_completion_outcome("\\boxed{4}", "4")[0], 0.5
    )
    self.assertEqual(gsm8k_vtc.vtc_completion_outcome("four", "4")[0], 0.0)
    # Gold answers arrive as numpy / bytes from TFDS and with thousands commas.
    self.assertEqual(
        gsm8k_vtc.vtc_completion_outcome(
            "</reasoning><answer>\\boxed{1,000}</answer>", np.bytes_(b"1000")
        )[0],
        1.0,
    )

  def test_env_reward_reads_action_and_gold_aliases(self):
    action = types.SimpleNamespace(
        action="</reasoning><answer>\\boxed{9}</answer>"
    )
    self.assertEqual(gsm8k_vtc.vtc_env_reward({"answer": "9"}, action), 1.0)
    self.assertEqual(
        gsm8k_vtc.vtc_env_reward({"gold_answer": "9"}, "\\boxed{9}"), 0.5
    )

  def test_metric_fn(self):
    metrics = gsm8k_vtc.vtc_metric_fn(
        None, None, [1.0, 0.5, 0.1, 0.0], None, None
    )
    self.assertEqual(metrics["rewards/solve_ratio"][0], 0.5)
    self.assertEqual(metrics["rewards/solve_all"][0], 0)
    self.assertEqual(metrics["rewards/solve_none"][0], 0)
    self.assertEqual(metrics["rewards/solve_partial"][0], 1)

  def test_raw_text_parser_joins_verbatim(self):
    parser = gsm8k_vtc.VTCRawTextParser()
    text = parser.parse(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": gsm8k_vtc.build_prompt("Q?")},
        ],
        add_generation_prompt=True,
        is_first_msg=True,
    )
    self.assertEqual(text, gsm8k_vtc.build_prompt("Q?"))
    self.assertTrue(text.endswith("<reasoning>\n"))
    tokens, extra = parser.update_assistant_end_tokens(np.array([1, 2]))
    self.assertEqual(extra, 0)
    np.testing.assert_array_equal(tokens, [1, 2])


if __name__ == "__main__":
  absltest.main()
