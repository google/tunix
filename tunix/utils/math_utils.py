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

"""Math utils for evaluating on Math Dataset like Math500 and AIME2024."""

from decimal import Decimal, ROUND_HALF_UP
import re
from absl import logging
from pylatexenc import latex2text
import sympy
from sympy.parsing import sympy_parser


def mathd_normalize_answer(answer: str | None) -> str | None:
  if answer is None:
    return None
  answer = answer.strip()
  try:
    # Remove enclosing `\text{}`.
    m = re.search(r"^\\text\{(?P<text>.+?)\}", answer)
    if m is not None:
      answer = m.group("text").strip()
    return _strip_string(answer)
  except Exception:
    return answer


def _strip_string(string: str):
  def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
      substrs = substrs[1:]
      for substr in substrs:
        new_str += "\\frac"
        if substr[0] == "{":
          new_str += substr
        else:
          try:
            assert len(substr) >= 2
          except Exception:
            return string
          a = substr[0]
          b = substr[1]
          if b != "{":
            if len(substr) > 2:
              post_substr = substr[2:]
              new_str += "{" + a + "}{" + b + "}" + post_substr
            else:
              new_str += "{" + a + "}{" + b + "}"
          else:
            if len(substr) > 2:
              post_substr = substr[2:]
              new_str += "{" + a + "}" + b + post_substr
            else:
              new_str += "{" + a + "}" + b
    string = new_str
    return string

  def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
      return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
      a = int(a)
      b = int(b)
      assert string == "{}/{}".format(a, b)
      new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
      return new_string
    except Exception:
      return string

  def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
      splits = string.split("\\text{ ")
      assert len(splits) == 2
      return splits[0]
    else:
      return string

  def _fix_sqrt(string):
    if "\\sqrt" not in string:
      return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
      if split[0] != "{":
        a = split[0]
        new_substr = "\\sqrt{" + a + "}" + split[1:]
      else:
        new_substr = "\\sqrt" + split
      new_string += new_substr
    return new_string

  # linebreaks
  string = string.replace("\n", "")
  # print(string)

  # remove inverse spaces
  string = string.replace("\\!", "")
  # print(string)

  # replace \\ with \
  string = string.replace("\\\\", "\\")
  # print(string)

  # replace tfrac and dfrac with frac
  string = string.replace("tfrac", "frac")
  string = string.replace("dfrac", "frac")
  # print(string)

  # remove \left and \right
  string = string.replace("\\left", "")
  string = string.replace("\\right", "")
  # print(string)

  # Remove circ (degrees)
  string = string.replace("^{\\circ}", "")
  string = string.replace("^\\circ", "")

  # remove dollar signs
  string = string.replace("\\$", "")

  # remove units (on the right)
  string = _remove_right_units(string)

  # remove percentage
  string = string.replace("\\%", "")

  # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
  string = string.replace(" .", " 0.")
  string = string.replace("{.", "{0.")
  # if empty, return empty string
  if len(string) == 0:
    return string
  if string[0] == ".":
    string = "0" + string

  # to consider: get rid of e.g. "k = " or "q = " at beginning
  if len(string.split("=")) == 2:
    if len(string.split("=")[0]) <= 2:
      string = string.split("=")[1]

  # fix sqrt3 --> sqrt{3}
  string = _fix_sqrt(string)

  # remove spaces
  string = string.replace(" ", "")

  # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
  string = _fix_fracs(string)

  # manually change 0.5 --> \frac{1}{2}
  if string == "0.5":
    string = "\\frac{1}{2}"

  # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
  string = _fix_a_slash_b(string)

  return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
  """Parses an expression with sympy."""
  py_expr = expr.replace("^", "**")
  return sympy_parser.parse_expr(
      py_expr,
      transformations=(
          sympy_parser.standard_transformations
          + (sympy_parser.implicit_multiplication_application,)
      ),
  )


def _parse_latex(expr: str) -> str:
  """Attempts to parse latex to an expression sympy can read."""
  expr = expr.replace("\\tfrac", "\\frac")
  expr = expr.replace("\\dfrac", "\\frac")
  expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
  expr = latex2text.LatexNodes2Text().latex_to_text(expr)

  # Replace the specific characters that this parser uses.
  expr = expr.replace("√", "sqrt")
  expr = expr.replace("π", "pi")
  expr = expr.replace("∞", "inf")
  expr = expr.replace("∪", "U")
  expr = expr.replace("·", "*")
  expr = expr.replace("×", "*")

  return expr.strip()


def _is_float(num: str) -> bool:
  try:
    float(num)
    return True
  except ValueError:
    return False


def _is_int(x: float) -> bool:
  try:
    return abs(x - int(round(x))) <= 1e-7
  except Exception:
    return False


def _is_frac(expr: str) -> bool:
  return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
  try:
    x = _strip_properly_formatted_commas(x)
    x = float(x)
    return abs(x - int(round(x))) <= 1e-7
  except Exception:
    return False


def _str_to_int(x: str) -> int:
  x = x.replace(",", "")
  x = float(x)
  return int(x)


def _inject_implicit_mixed_number(step: str):
  """Automatically make a mixed number evalable e.g.

  7 3/4 => 7+3/4
  """
  p1 = re.compile("([0-9]) +([0-9])")
  step = p1.sub("\\1+\\2", step)  ## implicit mults
  return step


def _strip_properly_formatted_commas(expr: str):
  # We want to be careful because we don't want to strip tuple commas
  p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
  while True:
    next_expr = p1.sub("\\1\\3\\4", expr)
    if next_expr == expr:
      break
    expr = next_expr
  return next_expr


def _normalize(expr: str) -> str:
  """Normalize answer expressions."""
  if expr is None:
    return None

  # Remove enclosing `\text{}`.
  m = re.search(r"^\\text\{(?P<text>.+?)\}", expr)
  if m is not None:
    expr = m.group("text")

  expr = expr.rstrip("\\")  # remove trailing \\ since it indicates a new line break and will cause latex2text to stuck # pylint: disable=line-too-long
  expr = expr.replace("\\%", "%")
  expr = expr.replace("\\$", "$")
  expr = expr.replace("$", "")
  expr = expr.replace("%", "")
  expr = expr.replace(" or ", " , ")
  expr = expr.replace(" and ", " , ")

  expr = expr.replace("million", "*10^6")
  expr = expr.replace("billion", "*10^9")
  expr = expr.replace("trillion", "*10^12")

  for unit in [
      "degree",
      "cm",
      "centimeter",
      "meter",
      "mile",
      "second",
      "minute",
      "hour",
      "day",
      "week",
      "month",
      "year",
      "foot",
      "feet",
      "inch",
      "yard",
  ]:
    expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
  expr = re.sub(r"\^ *\\circ", "", expr)

  if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
    expr = expr[1:-1]

  expr = re.sub(",\\\\! *", "", expr)
  if _is_float(expr) and _is_int(float(expr)):
    expr = str(int(round(float(expr))))
  if "\\" in expr:
    try:
      expr = _parse_latex(expr)
    except Exception:
      pass

  # edge case with mixed numbers and negative signs
  expr = re.sub("- *", "-", expr)

  expr = _inject_implicit_mixed_number(expr)
  expr = expr.replace(" ", "")

  # if we somehow still have latex braces here, just drop them
  expr = expr.replace("{", "")
  expr = expr.replace("}", "")

  # don't be case sensitive for text answers
  expr = expr.lower()

  if _str_is_int(expr):
    expr = str(_str_to_int(expr))

  return expr


def count_unknown_letters_in_expr(expr: str):
  expr = expr.replace("sqrt", "")
  expr = expr.replace("frac", "")
  letters_in_expr = set([x for x in expr if x.isalpha()])
  return len(letters_in_expr)


def should_allow_eval(expr: str):
  # Avoid parsing unknown text or functions of more than two variables.
  if count_unknown_letters_in_expr(expr) > 2:
    return False

  for bad_string in BAD_SUBSTRINGS:
    if bad_string in expr:
      return False

  for bad_regex in BAD_REGEXES:
    if re.search(bad_regex, expr) is not None:
      return False

  return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
  are_equal = False
  try:
    expr = f"({ground_truth_normalized})-({given_normalized})"
    if should_allow_eval(expr):
      sympy_diff = _sympy_parse(expr)
      simplified = sympy.simplify(sympy_diff)
      if simplified == 0:
        are_equal = True
  except Exception:
    pass
  return are_equal


def split_tuple(expr: str):
  """Split the elements in a tuple/interval, while handling well-formatted commas in large numbers"""

  expr = _strip_properly_formatted_commas(expr)
  if len(expr) == 0:
    return []
  if (
      len(expr) > 2
      and expr[0] in TUPLE_CHARS
      and expr[-1] in TUPLE_CHARS
      and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
  ):
    elems = [elem.strip() for elem in expr[1:-1].split(",")]
  else:
    elems = [expr]
  return elems


def last_boxed_only_string(string):
  idx = string.rfind("\\boxed")
  if idx < 0:
    idx = string.rfind("\\fbox")
    if idx < 0:
      return None

  i = idx
  right_brace_idx = None
  num_left_braces_open = 0
  while i < len(string):
    if string[i] == "{":
      num_left_braces_open += 1
    if string[i] == "}":
      num_left_braces_open -= 1
      if num_left_braces_open == 0:
        right_brace_idx = i
        break
    i += 1

  if right_brace_idx is None:
    retval = None
  else:
    retval = string[idx : right_brace_idx + 1]

  return retval


def remove_boxed(s):
  left = "\\boxed{"
  try:
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]
  except AssertionError:
    return None


def extract_boxed_answer(solution: str):
  """Extract the answer from inside a LaTeX \\boxed{} command"""
  solution = last_boxed_only_string(solution)
  solution = remove_boxed(solution) if solution is not None else solution
  logging.vlog(4, f"{solution=} in extracted_boxed_answer")
  return solution


def _cleanup_invalid_empty_sqrt(expr: str) -> str:
  """Fix malformed latex like `\\sqrt{}{3}` -> `\\sqrt{3}`."""
  return re.sub(r"sqrt\{\}", r"sqrt", expr)


def _parse_special_decimal_interval(expr: str):
  """Parse known recurring-decimal special cases to numeric intervals."""
  expr = expr.replace("$", "").replace(" ", "")
  m = re.fullmatch(r"([+-]?\d+)\.([0-9]*)\\overline\{([0-9])\}", expr)
  if m is not None:
    int_part = m.group(1)
    non_repeating_decimals = m.group(2)
    recurring_digit = m.group(3)

    # Only support single-digit recurring blocks, e.g. `16.\overline{6}`.
    # Map to the interval formed by 1-decimal and 2-decimal rounded values,
    # so answers like `16.7` and `16.67` can both match.
    decimal_places = len(non_repeating_decimals)
    scale = Decimal(10) ** decimal_places
    value = (
        Decimal(int_part)
        + Decimal(non_repeating_decimals or "0") / scale
        + Decimal(recurring_digit) / (Decimal(9) * scale)
    )

    rounded_1 = float(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    rounded_2 = float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    return (min(rounded_1, rounded_2), max(rounded_1, rounded_2))

  try:
    value = float(expr)
    return (value, value)
  except Exception:
    return None


def _intervals_overlap(
    interval_a: tuple[float, float], interval_b: tuple[float, float]
):
  return not (interval_a[1] < interval_b[0] or interval_b[1] < interval_a[0])


def _parse_interval_set(expr: str):
  """Parse interval unions from either inequality or bracket notation."""
  expr = expr.lower().strip()
  expr = expr.replace("$", "")
  expr = expr.replace("≤", "\\le")
  expr = expr.replace("\\leq", "\\le")
  expr = expr.replace("<=", "\\le")
  expr = expr.replace("\\cup", "|")
  expr = expr.replace("∪", "|")
  expr = expr.replace("or", "|")
  expr = expr.replace(" ", "")

  if not expr:
    return None

  parts = [part for part in expr.split("|") if part]
  if not parts:
    return None

  # First try interval notation: [a,b], (a,b], etc.
  intervals = []
  all_interval_notation = True
  for part in parts:
    m = re.fullmatch(
        r"([\[(])([+-]?(?:\d+(?:\.\d+)?|\.\d+)),([+-]?(?:\d+(?:\.\d+)?|\.\d+))([\])])",
        part,
    )
    if m is None:
      all_interval_notation = False
      break
    left = float(m.group(2))
    right = float(m.group(3))
    left_closed = m.group(1) == "["
    right_closed = m.group(4) == "]"

    if left > right:
      left, right = right, left
      left_closed, right_closed = right_closed, left_closed
    intervals.append((left, right, left_closed, right_closed))

  if all_interval_notation:
    return sorted(intervals)

  # Then try inequalities: -5\lex\le1, -5\lex\le1, etc.
  intervals = []
  for part in parts:
    m = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\\le[a-z]?\\le([+-]?(?:\d+(?:\.\d+)?|\.\d+))",
        part,
    )
    if m is None:
      return None
    left = float(m.group(1))
    right = float(m.group(2))
    if left > right:
      left, right = right, left
    intervals.append((left, right, True, True))

  return sorted(intervals)


def _match_recurring_decimal_special_case(
    given_clean: str, ground_truth_clean: str
) -> bool:
  """Handle recurring decimal overlaps for single-digit overline forms."""
  if not (
      re.search(r"[0-9]+\.\s*\\overline\{[0-9]\}", given_clean)
      or re.search(r"[0-9]+\.\s*\\overline\{[0-9]\}", ground_truth_clean)
  ):
    return False

  given_interval = _parse_special_decimal_interval(given_clean)
  ground_truth_interval = _parse_special_decimal_interval(ground_truth_clean)
  return (
      given_interval is not None
      and ground_truth_interval is not None
      and _intervals_overlap(given_interval, ground_truth_interval)
  )


def _match_interval_union_special_case(
    given_clean: str, ground_truth_clean: str
) -> bool:
  """Handle inequality unions and interval unions as equivalent sets."""
  given_intervals = _parse_interval_set(given_clean)
  ground_truth_intervals = _parse_interval_set(ground_truth_clean)
  return (
      given_intervals is not None
      and ground_truth_intervals is not None
      and given_intervals == ground_truth_intervals
  )


def _match_invalid_sqrt_special_case(
    given_answer: str,
    ground_truth: str,
    given_clean: str,
    ground_truth_clean: str,
) -> bool:
  """Handle malformed `sqrt{}` cleanup equivalence checks."""
  if given_clean == given_answer and ground_truth_clean == ground_truth:
    return False

  given_normalized = _normalize(given_clean)
  ground_truth_normalized = _normalize(ground_truth_clean)
  if (
      given_normalized is not None
      and ground_truth_normalized is not None
      and given_normalized == ground_truth_normalized
  ):
    return True
  return (
      given_normalized is not None
      and ground_truth_normalized is not None
      and len(given_normalized) > 0
      and are_equal_under_sympy(ground_truth_normalized, given_normalized)
  )


def grade_answer_special_handling(given_answer: str, ground_truth: str) -> bool:
  if given_answer is None or ground_truth is None:
    return False
  # Only clean the ground truth for latex errors.
  ground_truth_clean = _cleanup_invalid_empty_sqrt(ground_truth)

  if given_answer == ground_truth_clean:
    return True

  # Case 1: recurring decimal overlap special handling.
  if _match_recurring_decimal_special_case(given_answer, ground_truth_clean):
    return True

  # Case 2: malformed sqrt{} cleanups should still evaluate as equivalent.
  if _match_invalid_sqrt_special_case(
      given_answer, ground_truth, given_answer, ground_truth_clean
  ):
    return True

  # Case 3: inequality union vs interval union equivalence.
  if _match_interval_union_special_case(given_answer, ground_truth_clean):
    return True

  return False


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
  """Grades a given answer against a ground truth using sympy for evaluation."""
  ground_truth_normalized = _normalize(ground_truth)
  given_normalized = _normalize(given_answer)

  if ground_truth_normalized is None:
    return False

  if ground_truth_normalized == given_normalized:
    return True

  if len(given_normalized) == 0:
    return False

  ground_truth_elems = split_tuple(ground_truth_normalized)
  given_elems = split_tuple(given_normalized)

  if len(ground_truth_elems) > 1 and (
      ground_truth_normalized[0] != given_normalized[0]
      or ground_truth_normalized[-1] != given_normalized[-1]
  ):
    is_correct = False
  elif len(ground_truth_elems) != len(given_elems):
    is_correct = False
  else:
    for ground_truth_elem, given_elem in zip(
        ground_truth_elems, given_elems, strict=False
    ):
      if _is_frac(ground_truth_elem) and _is_frac(given_elem):
        # if fractions aren't reduced, then shouldn't be marked as correct
        # so, we don't want to allow sympy.simplify in this case
        is_correct = ground_truth_elem == given_elem
      elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
        # if the ground truth answer is an integer, we require the given answer to be a strict match
        # (no sympy.simplify)
        is_correct = False
      else:
        is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
      if not is_correct:
        break

  return is_correct


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
  ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
  given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

  # be at least as lenient as mathd
  if ground_truth_normalized_mathd == given_answer_normalized_mathd:
    print(
        "mathd"
        f" {ground_truth_normalized_mathd=} {given_answer_normalized_mathd=} IS"
        " CORRECT"
    )
    return True
  print(
      "mathd"
      f" {ground_truth_normalized_mathd=} {given_answer_normalized_mathd=} IS"
      " NOT CORRECT"
  )
  return False


def extract_answer(passage: str):
  if "\\boxed" in passage:
    return extract_boxed_answer(passage)
  return None
