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

"""Tests for tunix.utils.math_utils special handling."""

from absl.testing import absltest
from absl.testing import parameterized
from tunix.utils import math_utils


class MathUtilsSpecialHandlingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="recurring_decimal_overlap",
          given_answer="16.67",
          ground_truth=r"16.\overline{6}",
          expected=True,
      ),
      dict(
          testcase_name="recurring_decimal_all_single_digit_pattern",
          given_answer="2.33",
          ground_truth=r"2.\overline{3}",
          expected=True,
      ),
      dict(
          testcase_name="recurring_decimal_all_single_digit_pattern2",
          given_answer="2.3",
          ground_truth=r"2.\overline{3}",
          expected=True,
      ),
      dict(
          testcase_name="invalid_sqrt_cleanup_equivalent",
          given_answer=r"\frac{3\sqrt{3}}{2}",
          ground_truth=r"\frac{3\sqrt{}{3}}{2}",
          expected=True,
      ),
      dict(
          testcase_name="interval_union_equivalence",
          given_answer=r"$-5\lex\le1$or$3\lex\le9$",
          ground_truth=r"[-5,1]\cup[3,9]",
          expected=True,
      ),
      dict(
          testcase_name="partial_interval_not_tolerated",
          given_answer=r"$-5\lex\le1$or$3\lex\le9$",
          ground_truth=r"-5,1]\cup[3,9]",
          expected=False,
      ),
  )
  def test_grade_answer_special_handling(
      self, given_answer: str, ground_truth: str, expected: bool
  ):
    self.assertEqual(
        math_utils.grade_answer_special_handling(given_answer, ground_truth),
        expected,
    )


if __name__ == "__main__":
  absltest.main()
