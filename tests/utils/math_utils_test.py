# Copyright 2025 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
from absl.testing import parameterized
from tunix.utils import math_utils


class MathDNormalizeAnswerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("none_input", None, None),
      ("simple_number", "42", "42"),
      ("whitespace", "  42  ", "42"),
      ("text_wrapper", "\\text{hello}", "hello"),
      ("frac_shorthand", "\\frac12", "\\frac{1}{2}"),
      ("dfrac_to_frac", "\\dfrac{1}{2}", "\\frac{1}{2}"),
      ("tfrac_to_frac", "\\tfrac{1}{2}", "\\frac{1}{2}"),
      ("remove_left_right", "\\left(x\\right)", "(x)"),
      ("remove_circ", "90^{\\circ}", "90"),
      ("remove_dollar", "\\$100", "100"),
      ("remove_percent", "50\\%", "50"),
      ("leading_dot", ".5", "0.5"),
      ("slash_to_frac", "1/2", "\\frac{1}{2}"),
      ("half_decimal", "0.5", "\\frac{1}{2}"),
      ("sqrt_shorthand", "\\sqrt2", "\\sqrt{2}"),
      ("variable_equals", "x = 5", "5"),
  )
  def test_normalize(self, answer, expected):
    self.assertEqual(math_utils.mathd_normalize_answer(answer), expected)


class ExtractBoxedAnswerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("simple", "The answer is \\boxed{42}", "42"),
      ("nested_braces", "\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
      ("no_boxed", "The answer is 42", None),
      ("multiple_boxed", "\\boxed{1} and \\boxed{2}", "2"),
      ("fbox", "\\fbox{hello}", "hello"),
  )
  def test_extract_boxed_answer(self, solution, expected):
    self.assertEqual(math_utils.extract_boxed_answer(solution), expected)


class LastBoxedOnlyStringTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("simple", "\\boxed{5}", "\\boxed{5}"),
      ("no_match", "no boxed here", None),
      ("nested", "\\boxed{\\frac{3}{4}}", "\\boxed{\\frac{3}{4}}"),
      ("unclosed", "\\boxed{5", None),
      ("last_of_many", "\\boxed{1} then \\boxed{2}", "\\boxed{2}"),
  )
  def test_last_boxed_only_string(self, string, expected):
    self.assertEqual(math_utils.last_boxed_only_string(string), expected)


class RemoveBoxedTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("simple", "\\boxed{42}", "42"),
      ("expression", "\\boxed{x+1}", "x+1"),
  )
  def test_remove_boxed(self, s, expected):
    self.assertEqual(math_utils.remove_boxed(s), expected)


class ExtractAnswerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("with_boxed", "Solution: \\boxed{7}", "7"),
      ("no_boxed", "The answer is 7", None),
  )
  def test_extract_answer(self, passage, expected):
    self.assertEqual(math_utils.extract_answer(passage), expected)


class SplitTupleTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("empty", "", []),
      ("single", "5", ["5"]),
      ("parenthesized_pair", "(1, 2)", ["1", "2"]),
      ("bracketed_pair", "[3, 4]", ["3", "4"]),
      ("triple", "(1, 2, 3)", ["1", "2", "3"]),
      ("no_parens", "1, 2", ["1, 2"]),
      ("formatted_commas", "1,000", ["1000"]),
  )
  def test_split_tuple(self, expr, expected):
    self.assertEqual(math_utils.split_tuple(expr), expected)


class HelperFunctionsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("integer", "42", True),
      ("float", "3.14", True),
      ("negative", "-1.5", True),
      ("not_float", "abc", False),
      ("empty", "", False),
  )
  def test_is_float(self, num, expected):
    self.assertEqual(math_utils._is_float(num), expected)

  @parameterized.named_parameters(
      ("exact_int", 5.0, True),
      ("close_to_int", 5.0000000001, True),
      ("not_int", 5.5, False),
      ("negative_int", -3.0, True),
  )
  def test_is_int(self, x, expected):
    self.assertEqual(math_utils._is_int(x), expected)

  @parameterized.named_parameters(
      ("simple_frac", "1/2", True),
      ("negative_frac", "-3/4", True),
      ("not_frac", "abc", False),
      ("division_by_zero", "1/0", False),
      ("decimal", "1.5", False),
  )
  def test_is_frac(self, expr, expected):
    self.assertEqual(math_utils._is_frac(expr), expected)

  @parameterized.named_parameters(
      ("integer_str", "5", True),
      ("float_int_str", "5.0", True),
      ("not_int_str", "5.5", False),
      ("formatted", "1,000", True),
      ("text", "abc", False),
  )
  def test_str_is_int(self, x, expected):
    self.assertEqual(math_utils._str_is_int(x), expected)

  @parameterized.named_parameters(
      ("simple", "42", 42),
      ("with_commas", "1,000", 1000),
  )
  def test_str_to_int(self, x, expected):
    self.assertEqual(math_utils._str_to_int(x), expected)

  @parameterized.named_parameters(
      ("no_letters", "123+456", 0),
      ("one_var", "x+1", 1),
      ("sqrt_ignored", "sqrt(x)", 1),
      ("frac_ignored", "frac(a, b)", 2),
      ("many_vars", "a+b+c+d", 4),
  )
  def test_count_unknown_letters(self, expr, expected):
    self.assertEqual(math_utils.count_unknown_letters_in_expr(expr), expected)

  @parameterized.named_parameters(
      ("simple", "1+2", True),
      ("too_many_vars", "a+b+c+d", False),
      ("bad_caret_brace", "x^{2}", False),
      ("bad_caret_paren", "x^(2)", False),
      ("bad_multi_caret", "2^3^", False),
  )
  def test_should_allow_eval(self, expr, expected):
    self.assertEqual(math_utils.should_allow_eval(expr), expected)


class StripProperlyFormattedCommasTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("thousands", "1,000", "1000"),
      ("millions", "1,000,000", "1000000"),
      ("no_commas", "1234", "1234"),
      ("non_numeric_comma", "a,b", "a,b"),
  )
  def test_strip_commas(self, expr, expected):
    self.assertEqual(
        math_utils._strip_properly_formatted_commas(expr), expected
    )


class InjectImplicitMixedNumberTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("mixed_number", "7 3/4", "7+3/4"),
      ("no_mixed", "3/4", "3/4"),
      ("multiple_spaces", "1 2 3", "1+2+3"),
  )
  def test_inject_implicit(self, step, expected):
    self.assertEqual(
        math_utils._inject_implicit_mixed_number(step), expected
    )


class GradeAnswerSympyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("exact_match", "42", "42", True),
      ("equivalent_fraction", "0.5", "1/2", True),
      ("different_values", "3", "4", False),
      ("empty_given", "", "42", False),
      ("symbolic_equiv", "2*x", "x+x", True),
      ("tuple_match", "(1, 2)", "(1, 2)", True),
      ("tuple_mismatch_delim", "(1, 2)", "[1, 2]", False),
      ("tuple_length_mismatch", "(1, 2)", "(1, 2, 3)", False),
  )
  def test_grade_answer(self, given, ground_truth, expected):
    self.assertEqual(
        math_utils.grade_answer_sympy(given, ground_truth), expected
    )


class AreEqualUnderSympyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("same_value", "5", "5", True),
      ("equivalent_expr", "x+x", "2*x", True),
      ("not_equal", "3", "4", False),
  )
  def test_are_equal(self, gt, given, expected):
    self.assertEqual(
        math_utils.are_equal_under_sympy(gt, given), expected
    )


if __name__ == "__main__":
  absltest.main()
