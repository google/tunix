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
"""Reward functions for the RLHF pipeline."""

import os
import re
from typing import Callable, List
from absl import logging


# Define the expected signature with type hints
ExpectedSignature = Callable[..., List[float]]

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"
match_format = re.compile(
    rf"{re.escape(reasoning_start)}.+?{re.escape(reasoning_end)}.*?"
    rf"{re.escape(solution_start)}(.+?)(?:{re.escape(solution_end)}|$)",
    flags=re.MULTILINE | re.DOTALL,
)


# All reward functions must have this signature.
# range: [0, 3]
def match_format_exactly(prompts, completions, **kwargs):
  return [
      0 if match_format.search(response) is None else 3.0
      for response in completions
  ]


# range: [-2, 2]
def match_format_approximately(prompts, completions, **kwargs):
  scores = []

  for completion in completions:
    score = 0
    response = completion
    # Count how many keywords are seen - we penalize if too many!
    # If we see 1, then plus some points!
    score += 0.5 if response.count(reasoning_start) == 1 else -0.5
    score += 0.5 if response.count(reasoning_end) == 1 else -0.5
    score += 0.5 if response.count(solution_start) == 1 else -0.5
    score += 0.5 if response.count(solution_end) == 1 else -0.5
    scores.append(score)
  return scores


# range: [-1, 3]
def check_answer(prompts, completions, answer, **kwargs):
  """Checks if the extracted response matches the gold answer."""
  # Clean formatting (e.g. stripping spaces and commas).
  clean_answers = [str(a).strip().replace(",", "") for a in answer]

  log_rollout = os.environ.get("LOG_GRPO_ROLLOUT", "0") == "1"
  if log_rollout:
    logging.info("GRPO Rollout Prompts: %r", prompts)
    logging.info("GRPO Rollout Completions: %r", completions)
    logging.info("GRPO Rollout Gold Answers (Raw): %r", answer)
    logging.info("GRPO Rollout Gold Answers (Cleaned): %s", clean_answers)

  extracted_responses = [
      guess.group(1).strip().replace(",", "")
      if (guess := match_format.search(r)) is not None
      else None
      for r in completions
  ]

  scores = []
  for guess, true_answer in zip(extracted_responses, clean_answers):
    if guess is None:
      scores.append(0)
      continue

    # Correct answer gets 3 points!
    if guess == true_answer:
      scores.append(3.0)
      continue

    # If string comparison fails, try numerical equivalence.
    try:
      guess_float = float(guess)
      true_answer_float = float(true_answer)

      if guess_float == true_answer_float:
        scores.append(3.0)
      elif true_answer_float == 0:
        # Cannot compute ratio. Penalize if guess is not also 0.
        scores.append(-1.0 if guess_float != 0 else 0.0)
      else:
        ratio = guess_float / true_answer_float
        if 0.9 <= ratio <= 1.1:
          scores.append(0.5)
        elif 0.8 <= ratio <= 1.2:
          scores.append(0.25)
        else:
          scores.append(-1.0)  # Penalize wrong answers
    except (ValueError, ZeroDivisionError):
      # Penalize if float conversion or division fails.
      scores.append(-0.5)

  return scores


# range: [0, 1.5]
def check_numbers(prompts, completions, answer, **kwargs):
  match_numbers = re.compile(
      rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
  )
  responses = completions
  extracted_responses = [
      guess.group(1) if (guess := match_numbers.search(r)) is not None else None
      for r in responses
  ]

  # Clean formatting (e.g. stripping spaces and commas).
  clean_answers = [str(a).strip().replace(",", "") for a in answer]

  scores = []
  for guess, true_answer in zip(extracted_responses, clean_answers):
    if guess is None:
      scores.append(0)
      continue
    # Convert to numbers
    try:
      true_answer_val = float(true_answer.strip().replace(",", ""))
      guess_val = float(guess.strip().replace(",", ""))
      scores.append(1.5 if guess_val == true_answer_val else 0.0)
    except ValueError:
      scores.append(0)
      continue
  return scores
