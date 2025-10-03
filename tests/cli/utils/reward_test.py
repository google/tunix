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
"""Unit tests for reward functions."""
from absl.testing import absltest
from absl.testing import parameterized
from tunix.cli.utils import reward


class RewardTest(parameterized.TestCase):

  def test_match_reward_exactly(self):
    completions = [
        (
            "<reasoning>This is why.</reasoning> <answer>This is the"
            " answer.</answer>"
        ),
        (
            "<reasoning>Another reason.</reasoning>\n<answer>Another"
            " answer.</answer>"
        ),
        "<answer>Only answer, no reasoning.</answer>",
        "<reasoning>Only reasoning, no answer.</reasoning>",
        "Some other text",
    ]
    rewards = reward.match_format_exactly(None, completions)
    self.assertEqual(rewards, [3.0, 3.0, 0.0, 0.0, 0.0])

  @parameterized.named_parameters(
      dict(
          testcase_name="perfect_match",
          completions=["<reasoning>foo</reasoning><answer>bar</answer>"],
          expected_scores=[2.0],
      ),
      dict(
          testcase_name="no_tags",
          completions=["This has no tags."],
          expected_scores=[-2.0],
      ),
      dict(
          testcase_name="missing_some_tags",
          completions=["<reasoning>foo</reasoning>"],
          expected_scores=[0.0],  # +0.5, +0.5, -0.5, -0.5
      ),
      dict(
          testcase_name="missing_other_tags",
          completions=["<answer>bar</answer>"],
          expected_scores=[0.0],  # -0.5, -0.5, +0.5, +0.5
      ),
      dict(
          testcase_name="duplicate_tags",
          completions=[
              "<reasoning>a</reasoning><reasoning>b</reasoning><answer>c</answer>"
          ],
          expected_scores=[0.0],
          # Corrected expected: -0.5 (rs>1), -0.5 (re>1), +0.5 (ss==1), +0.5 (se==1) = 0.0
      ),
      dict(
          testcase_name="duplicate_reasoning_start",
          completions=["<reasoning><reasoning>a</reasoning><answer>b</answer>"],
          expected_scores=[1.0],  # -0.5, +0.5, +0.5, +0.5
      ),
      dict(
          testcase_name="duplicate_reasoning_end",
          completions=[
              "<reasoning>a</reasoning></reasoning><answer>b</answer>"
          ],
          expected_scores=[1.0],  # +0.5, -0.5, +0.5, +0.5
      ),
      dict(
          testcase_name="duplicate_solution_start",
          completions=["<reasoning>a</reasoning><answer><answer>b</answer>"],
          expected_scores=[1.0],  # +0.5, +0.5, -0.5, +0.5
      ),
      dict(
          testcase_name="duplicate_solution_end",
          completions=["<reasoning>a</reasoning><answer>b</answer></answer>"],
          expected_scores=[1.0],  # +0.5, +0.5, +0.5, -0.5
      ),
      dict(
          testcase_name="all_duplicate",
          completions=[
              "<r><r></r></r><a><a></a></a>"
          ],  # Simplified for clarity
          expected_scores=[-2.0],
          # Test with actual tags
      ),
      dict(
          testcase_name="all_duplicate_actual_tags",
          completions=[
              "<reasoning><reasoning>a</reasoning></reasoning><answer><answer>b</answer></answer>"
          ],
          expected_scores=[-2.0],  # -0.5 * 4
      ),
      dict(
          testcase_name="empty_completion",
          completions=[""],
          expected_scores=[-2.0],
      ),
      dict(
          testcase_name="multiple_completions",
          completions=[
              "<reasoning>a</reasoning><answer>b</answer>",  # Score 2.0
              "no tags here",  # Score -2.0
              "<reasoning>c</reasoning>",  # Score 0.0
              "<reasoning><reasoning>d</reasoning><answer>e</answer>",  # Score 1.0
          ],
          expected_scores=[2.0, -2.0, 0.0, 1.0],
      ),
      dict(
          testcase_name="empty_list",
          completions=[],
          expected_scores=[],
      ),
  )
  def test_match_format_approximately(self, completions, expected_scores):
    scores = reward.match_format_approximately(
        prompts=None, completions=completions
    )
    self.assertEqual(scores, expected_scores)

  def _format_completion(self, answer_text):
    return f"<reasoning>Some reason.</reasoning><answer>{answer_text}</answer>"

  @parameterized.named_parameters(
      dict(
          testcase_name="exact_match",
          completions=["<reasoning>R1</reasoning><answer>Hello</answer>"],
          answers=["Hello"],
          expected_scores=[3.0],
      ),
      dict(
          testcase_name="no_match_format",
          completions=["<answer>Only Answer</answer>"],
          answers=["Only Answer"],
          expected_scores=[0.0],
      ),
      dict(
          testcase_name="strip_match",
          completions=["<reasoning>R1</reasoning><answer>  World  </answer>"],
          answers=["World"],
          expected_scores=[1.5],
      ),
      dict(
          testcase_name="numeric_close_10",
          completions=["<reasoning>R1</reasoning><answer>105</answer>"],
          answers=["100"],
          expected_scores=[0.5],
      ),
      dict(
          testcase_name="numeric_close_20",
          completions=["<reasoning>R1</reasoning><answer>81</answer>"],
          answers=["100"],
          expected_scores=[0.25],
      ),
      dict(
          testcase_name="numeric_not_close",
          completions=["<reasoning>R1</reasoning><answer>50</answer>"],
          answers=["100"],
          expected_scores=[-1.0],
      ),
      dict(
          testcase_name="numeric_guess_not_num",
          completions=["<reasoning>R1</reasoning><answer>abc</answer>"],
          answers=["100"],
          expected_scores=[-0.5],
      ),
      dict(
          testcase_name="numeric_true_not_num",
          completions=["<reasoning>R1</reasoning><answer>100</answer>"],
          answers=["abc"],
          expected_scores=[-0.5],
      ),
      dict(
          testcase_name="numeric_true_is_zero",
          completions=["<reasoning>R1</reasoning><answer>10</answer>"],
          answers=["0"],
          expected_scores=[-0.5],
      ),
      dict(
          testcase_name="numeric_both_are_zero",
          completions=["<reasoning>R1</reasoning><answer>0</answer>"],
          answers=["0"],
          expected_scores=[3.0],  # Exact match takes precedence
      ),
      dict(
          testcase_name="numeric_both_are_zero_float",
          completions=["<reasoning>R1</reasoning><answer>0.0</answer>"],
          answers=["0"],
          expected_scores=[3.0],  # Goes to numeric, true is zero, guess is zero
      ),
      dict(
          testcase_name="string_mismatch",
          completions=["<reasoning>R1</reasoning><answer>Apple</answer>"],
          answers=["Banana"],
          expected_scores=[-0.5],  # Fails to convert to float
      ),
      dict(
          testcase_name="multiple_items",
          completions=[
              "<reasoning>R1</reasoning><answer>100</answer>",
              "<reasoning>R2</reasoning><answer>Test</answer>",
              "No Answer Tag",
              "<reasoning>R3</reasoning><answer>  Match  </answer>",
              "<reasoning>R4</reasoning><answer>115</answer>",
          ],
          answers=["100", "Test", "Something", "Match", "100"],
          expected_scores=[3.0, 3.0, 0.0, 1.5, 0.25],
      ),
      dict(
          testcase_name="empty_inputs",
          completions=[],
          answers=[],
          expected_scores=[],
      ),
  )
  def test_check_answer(self, completions, answers, expected_scores):
    scores = reward.check_answer(
        prompts=None, completions=completions, answer=answers
    )
    self.assertEqual(scores, expected_scores)

  @parameterized.named_parameters(
      dict(
          testcase_name="exact_match",
          completions=["<answer>12.5</answer>"],
          answers=["12.5"],
          expected_scores=[1.5],
      ),
      dict(
          testcase_name="number_mismatch",
          completions=["<answer>12.5</answer>"],
          answers=["12.6"],
          expected_scores=[0.0],
      ),
      dict(
          testcase_name="no_extraction",
          completions=["Some text without answer tags 12.5"],
          answers=["12.5"],
          expected_scores=[0.0],
      ),
      dict(
          testcase_name="extraction_with_internal_text",
          completions=["<answer>The answer is 0.34</answer>"],
          answers=["0.34"],
          expected_scores=[1.5],
      ),
      dict(
          testcase_name="extraction_with_spaces_around_number",
          completions=["<answer>  25  </answer>"],
          answers=["25"],
          expected_scores=[1.5],
      ),
      dict(
          testcase_name="answer_with_spaces",
          completions=["<answer>25</answer>"],
          answers=["  25  "],
          expected_scores=[1.5],
      ),
      dict(
          testcase_name="value_error_in_guess_no_regex_match",
          completions=["<answer>not-a-number</answer>"],
          answers=["10"],
          expected_scores=[0.0],
      ),
      dict(
          testcase_name="value_error_in_guess_post_regex",
          completions=["<answer>1.2.3</answer>"],
          answers=["10"],
          expected_scores=[0.0],
      ),
      dict(
          testcase_name="value_error_in_answer",
          completions=["<answer>10</answer>"],
          answers=["not-a-number"],
          expected_scores=[0.0],
      ),
      dict(
          testcase_name="multiple_items",
          completions=[
              "<answer>1</answer>",
              "No tag here",
              "<answer>3.0</answer>",
              "<answer>four</answer>",
              "<answer>5.1</answer>",
          ],
          answers=["1.0", "2", "3", "4", "5.0"],
          expected_scores=[1.5, 0.0, 1.5, 0.0, 0.0],
      ),
      dict(
          testcase_name="empty_inputs",
          completions=[],
          answers=[],
          expected_scores=[],
      ),
      dict(
          testcase_name="regex_specifics",
          completions=[
              "<answer>0.34</answer>",
              "<answer>  0.34  </answer>",
              "<answer>Text before 12.5</answer>",
              "<answer>NoNumber</answer>",
              "<answer>1.2.3</answer>",
              f"<answer>12</answer><answer>34</answer>",
          ],
          answers=["0.34", "0.34", "12.5", "0", "0", "12"],
          expected_scores=[1.5, 1.5, 1.5, 0.0, 0.0, 1.5],
      ),
  )
  def test_check_numbers(self, completions, answers, expected_scores):
    """Tests check_numbers with various inputs."""
    scores = reward.check_numbers(
        prompts=[], completions=completions, answer=answers, question=["Q1"]
    )
    self.assertEqual(scores, expected_scores)


if __name__ == "__main__":
  absltest.main()
