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
"""Reward functions for OpenMathInstruct-2 dataset."""

import re
import math
from typing import List, Dict, Any, Optional
import numpy as np
import sympy
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from absl import logging

# Try to import from Google3, fallback for other environments
try:
    from google3.nlp.bard.learning.evaluation import math_utils
    logging.info("Successfully imported math_utils from google3.")
except ImportError:
    logging.warning(
        "Failed to import math_utils from google3. "
        "Symbolic comparison will use more basic normalization."
    )
    math_utils = None

NO_SOLUTION_VARS = ["no solution", "no solutions", "nosolution", "no real solution", "no real solutions"]
INFINITY_VARS = ["infinity", "inf", "\\infty"]

class _OpenMathRewardScorer:
    """
    Internal class to handle reward logic for OpenMathInstruct-2.
    Extracts answers from LaTeX \\boxed{} using brace counting and uses sympy for comparison.
    """

    def __init__(self, correct_reward: float = 1.0, incorrect_reward: float = 0.0, tolerance: float = 1e-3, **kwargs):
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.tolerance = tolerance
        if sympy is not None and hasattr(sympy.parsing.latex, 'PARSE_LATEX_DEBUG'):
            sympy.parsing.latex.PARSE_LATEX_DEBUG = False
        self.transformations = standard_transformations + (implicit_multiplication_application,)

    def extract_boxed_answer(self, text: str) -> Optional[str]:
        """Extracts content from the last \\boxed{} by properly counting nested braces."""
        if not isinstance(text, str):
            return None

        box_str = "\\boxed{"
        idx = text.rfind(box_str)
        if idx == -1:
            return None

        start_idx = idx + len(box_str)
        brace_count = 1
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1

            if brace_count == 0:
                return text[start_idx:i].strip()

        logging.warning("Unbalanced braces found in \\boxed extraction for: %s", text[idx:])
        return None

    def basic_normalize(self, text: str) -> str:
        """Basic normalization for math expressions."""
        text = text.strip()
        text = re.sub(r"\\text\{.*?\}", "", text)
        text = text.replace("\\%", "").replace("%", "")
        text = text.replace("\\$", "").replace("$", "")
        text = text.replace("\\ ", "").replace("\\!", "").replace("\\,", "").replace("\\:", "")
        text = text.replace("\\left", "").replace("\\right", "")
        return text.lower()

    def normalize_answer(self, text: str) -> str:
        if math_utils:
            try:
                return math_utils.preprocess_latex(text)
            except Exception as e:
                logging.warning(f"math_utils.preprocess_latex failed: {e}")
        return self.basic_normalize(text)

    def is_text_answer_equivalent(self, ans1: str, ans2: str) -> Optional[bool]:
        ans1_lower = ans1.lower().replace("\\", "").replace("{", "").replace("}", "").strip()
        ans2_lower = ans2.lower().replace("\\", "").replace("{", "").replace("}", "").strip()
        if any(v in ans1_lower for v in NO_SOLUTION_VARS) and \
           any(v in ans2_lower for v in NO_SOLUTION_VARS):
            return True
        if any(v in ans1_lower for v in INFINITY_VARS) and \
           any(v in ans2_lower for v in INFINITY_VARS):
            return True
        return None

    def parse_expression(self, norm_ans: str):
        try:
            return parse_latex(norm_ans)
        except Exception as e_latex:
            logging.debug(f"parse_latex failed for '{norm_ans}': {e_latex}")
            try:
                fallback_ans = norm_ans.replace("\\pi", "pi").replace("\\sqrt", "sqrt")
                fallback_ans = fallback_ans.replace("\\cdot", "*").replace("\\times", "*")
                return parse_expr(fallback_ans, transformations=self.transformations, evaluate=False)
            except Exception as e_expr:
                logging.debug(f"parse_expr failed for '{fallback_ans}': {e_expr}")
                try:
                    return sympy.Float(norm_ans)
                except Exception as e_float:
                    logging.debug(f"sympy.Float failed for '{norm_ans}': {e_float}")
                    return norm_ans # Return normalized string if all parsing fails

    def symbolic_compare(self, ans1: str, ans2: str) -> bool:
        if not ans1 or not ans2:
            return False

        text_equiv = self.is_text_answer_equivalent(ans1, ans2)
        if text_equiv is not None:
            return text_equiv

        norm_ans1 = self.normalize_answer(ans1)
        norm_ans2 = self.normalize_answer(ans2)

        if norm_ans1 == norm_ans2:
            return True

        if sympy is None:
            return False

        try:
            sym_expr1 = self.parse_expression(norm_ans1)
            sym_expr2 = self.parse_expression(norm_ans2)

            if isinstance(sym_expr1, str) or isinstance(sym_expr2, str):
                return sym_expr1 == sym_expr2

            if sym_expr1.equals(sym_expr2): return True

            try:
                diff = sympy.simplify(sym_expr1 - sym_expr2)
                if diff == 0: return True
                if sympy.expand(diff) == 0: return True
                if diff.is_constant():
                    try:
                        eval_diff = complex(diff.evalf())
                        if abs(eval_diff) < self.tolerance: return True
                    except Exception: pass
            except Exception: pass

            try:
                val1 = sym_expr1.evalf(15, subs={sympy.pi: np.pi})
                val2 = sym_expr2.evalf(15, subs={sympy.pi: np.pi})
                if math.isclose(float(val1), float(val2), rel_tol=self.tolerance, abs_tol=self.tolerance):
                    return True
            except Exception: pass

            return False
        except Exception as e:
            logging.warning(f"Error in symbolic_compare: {e}, norm_ans1='{norm_ans1}', norm_ans2='{norm_ans2}'")
            return False

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        labels: List[str],
        **kwargs,
    ) -> np.ndarray:
        rewards = []
        for i in range(len(completions)):
            completion = completions[i] if isinstance(completions[i], str) else ""
            gt_answer_raw = labels[i]
            if isinstance(gt_answer_raw, bytes):
                 gt_answer_raw = gt_answer_raw.decode('utf-8', errors='ignore')
            gt_answer_raw = gt_answer_raw if isinstance(gt_answer_raw, str) else ""

            model_answer = self.extract_boxed_answer(completion)
            gt_answer = gt_answer_raw.strip()

            is_correct = False
            if model_answer is not None and gt_answer:
                is_correct = self.symbolic_compare(model_answer, gt_answer)

            rewards.append(self.correct_reward if is_correct else self.incorrect_reward)
            if kwargs.get("debug_logging", False): # Optional debug logging
                 logging.info("GT Raw: %s | Model Extracted: %s | GT: %s | Correct: %s", gt_answer_raw, model_answer, gt_answer, is_correct)
        return np.array(rewards, dtype=np.float32)

def reward_fn(prompts: List[str], completions: List[str], labels: List[str], **kwargs) -> np.ndarray:
    """Top-level reward function entry point for Tunix."""
    scorer = _OpenMathRewardScorer(**kwargs)
    return scorer(prompts=prompts, completions=completions, labels=labels, **kwargs)

