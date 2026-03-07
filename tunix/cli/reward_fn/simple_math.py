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
"""simple format and answer binary reward functions."""

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
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)


# range: [0, 0.1]
def format(prompts, completions, r=0.1, **kwargs):
  return [
      0 if match_format.search(response) is None else r
      for response in completions
  ]


# range: [0, 1]
def answer(prompts, completions, answer, r=1, **kwargs):
  match_numbers = re.compile(
      rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
  )
  responses = completions
  extracted_responses = [
      guess.group(1) if (guess := match_numbers.search(r)) is not None else None
      for r in responses
  ]

  scores = []
  for guess, true_answer in zip(extracted_responses, answer):
    if guess is None:
      scores.append(0)
      continue
    # Convert to numbers
    try:
      true_answer = float(true_answer.strip())
      guess = float(guess.strip())
      # TODO (atwigg) add sympy for more complex answers
      scores.append(r if guess == true_answer else 0.0)
    except ValueError:
      scores.append(0)
      continue
  return scores
