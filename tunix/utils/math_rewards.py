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

import math
import re
from typing import List

from tunix.utils import math_utils
from tunix.rl.agentic.rewards import reward
from tunix.rl.agentic.rewards import reward_types


THOUGHT_DELIMITER_END = "</think>"
RewardOutput = reward_types.RewardOutput

_NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _first_numeric_value(text: str) -> float | None:
  """Extracts the first numeric literal from text."""
  if not text:
    return None
  match = _NUMERIC_PATTERN.search(text)
  if not match:
    return None
  try:
    return float(match.group(0))
  except ValueError:
    return None


def _is_numeric_close(given_answer: str, ground_truth: str) -> bool:
  """Checks whether two answers are numerically close."""
  given_value = _first_numeric_value(given_answer)
  truth_value = _first_numeric_value(ground_truth)
  if given_value is None or truth_value is None:
    return False
  return math.isclose(given_value, truth_value, rel_tol=1e-2, abs_tol=1e-3)


@reward.register("deepscaler_math")
def math_reward(prompts: List[str], completions: List[str], answer: List[str], **kwargs):
  """
  A reward function for math tasks that implements the RewardFunction protocol.
  Args:
    task: The task dictionary containing data_source, ground_truth and other metadata
    action: The agent's response/solution

  Returns:
    float: The calculated reward value based on math evaluation
  """
  del prompts, kwargs
  rewards = []
  # Extract information from task_info
  for i, completion in enumerate(completions):
    model_response = completion

    # Handle None or empty response
    if model_response is None or model_response == "":
      rewards.append(0.0)
      continue

    # Extract solution.
    if THOUGHT_DELIMITER_END in model_response:
      model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
    else:
      model_solution = model_response

    reward_value = 0.0
    if "\\boxed" in model_solution:
      reward_value += 0.05

    model_answer = math_utils.extract_answer(model_solution)
    if model_answer is not None:
      reward_value += 0.05
    else:
      rewards.append(reward_value)
      continue

    # Process the ground truth(s)
    ground_truths = answer[i]
    if ground_truths is None:
      rewards.append(reward_value)
      continue

    # Convert single answer to list for uniform processing
    if isinstance(ground_truths, str | float | int):
      ground_truths = [ground_truths]

    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
      truth = str(truth)
      if "\\boxed" in truth:
        processed_truth = math_utils.extract_answer(truth)
        if processed_truth is not None:
          processed_ground_truths.append(processed_truth)
      else:
        processed_ground_truths.append(truth)

    if not processed_ground_truths:
      rewards.append(reward_value)
      continue

    # Check against all possible correct answers.
    found_exact_or_symbolic = False
    found_numeric_close = False
    for ground_truth in processed_ground_truths:
      if found_exact_or_symbolic:
        break
      is_exact_or_symbolic = math_utils.grade_answer_mathd(
          model_answer, ground_truth
      ) or math_utils.grade_answer_sympy(model_answer, ground_truth)
      if is_exact_or_symbolic:
        found_exact_or_symbolic = True
        break
      if _is_numeric_close(model_answer, ground_truth):
        found_numeric_close = True

    if found_exact_or_symbolic:
      reward_value = 1.0
    elif found_numeric_close:
      reward_value = max(reward_value, 0.3)  # Maximum fallback reward for numerically close answers.

    rewards.append(reward_value)
  return rewards
