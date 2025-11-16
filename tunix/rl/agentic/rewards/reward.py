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

"""Registry and utilities for reward functions used in agentic RL experiments.

This module provides a mechanism to register, retrieve, and combine various
reward functions. Reward functions take a task context and an agent's action
as input and return a `RewardOutput` containing a scalar reward and metadata.
"""

from typing import Any, Callable, Dict

from tunix.rl.agentic.rewards import reward_types

RewardOutput = reward_types.RewardOutput
_REGISTRY: Dict[str, Callable[[Dict[str, Any], str], RewardOutput]] = {}


def register(name: str):
  """Decorator for registering reward functions into the global registry.

  Enables reward functions to be discovered and instantiated by name,
  supporting configuration-driven reward selection in experimental settings.

  Args:
      name (str): Unique identifier for the reward function

  Returns:
      Callable: The decorated function, registered in the system

  Raises:
      ValueError: If a reward function with the given name already exists
  """

  def _wrap(fn):
    if name in _REGISTRY:
      raise ValueError(f"Reward {name} already registered.")
    _REGISTRY[name] = fn
    return fn

  return _wrap


def unregister(name: str) -> bool:
  """Remove a reward function from the registry.

  Enables cleanup of registered functions, particularly useful for
  unit testing to prevent state leakage between test cases.

  Args:
      name (str): Name of the reward function to remove

  Returns:
      bool: True if the function was removed, False if it wasn't registered
  """
  if name in _REGISTRY:
    del _REGISTRY[name]
    return True
  return False


def get_reward_fn(name: str):
  """Retrieve a registered reward function by name.

  Args:
      name (str): The registered name of the reward function

  Returns:
      Callable: The reward function implementation
  """
  return _REGISTRY[name]


@register("exact_match")
def exact_match(task: Dict[str, Any], action: str) -> RewardOutput:
  """Binary reward based on exact string matching with ground truth.

  Returns 1.0 for perfect matches after whitespace normalization,
  0.0 for any deviation. Suitable for deterministic answer tasks.

  Args:
      task (Dict[str, Any]): Task context containing 'ground_truth' field
      action (str): Agent's response to evaluate

  Returns:
      RewardOutput: Binary reward (1.0 or 0.0) with match status
  """
  truth = str(task.get("ground_truth", "")).strip()
  score = 1.0 if action.strip() == truth else 0.0
  return RewardOutput(score, {"exact_match": score})


def combine_rewards(
    weights: Dict[str, float],
) -> Callable[[Dict[str, Any], str], RewardOutput]:
  """Create a composite reward function from multiple registered functions.

  Performs weighted linear combination of multiple reward components,
  enabling complex reward engineering through composition.

  Args:
      weights (Dict[str, float]): Mapping from reward function names to weights

  Returns:
      Callable: Composite reward function that computes weighted sum

  Example:
      composite_fn = combine_rewards({"exact_match": 1.0, "zero": 0.0})
  """

  def _fn(task: Dict[str, Any], action: str):
    total, meta = 0.0, {}
    for name, w in weights.items():
      out = get_reward_fn(name)(task, action)
      total += w * out.reward
      meta.update(out.metadata)
    return RewardOutput(total, meta)

  return _fn


# -------- Example Reward Function --------
@register("is_two")
def is_two_reward(task: Dict[str, Any], action: str) -> RewardOutput:
  """Specialized reward function that checks if action represents the number 2.

  Attempts to parse the action as numeric value and returns 1.0 if it equals
  2.0,
  otherwise returns 0.0. Handles both string and numeric representations.

  Args:
      action (str): Agent's response to evaluate

  Returns:
      RewardOutput: Binary reward with parsing status in metadata
  """
  try:
    value = float(action.strip())
    score = 1.0 if value == 2.0 else 0.0
  except ValueError:
    score = 0.0
  return RewardOutput(score, {"is_two": score})


@register("dummy")
def dummy_reward(task: Dict[str, Any], action: str) -> RewardOutput:
  """A dummy reward function that always returns zero."""
  return RewardOutput(0.0, {})


@register("calculate")
def calculate_reward(task: Dict[str, Any], action: str) -> RewardOutput:
  """Calculates the reward for a math expression based on answer correctness.

  WARNING: Uses eval(), which is NOT SAFE for untrusted input. This is only for
  feature testing.

  Args:
    task: The task context containing the 'question' field.
    action: The model's answer as a string.

  Returns:
    RewardOutput: 1.0 if the model's answer matches the evaluated expression
      within a tolerance, 0.0 otherwise.
  """
  question_str = task.get("question", "")
  expression = question_str.replace("= ?", "").replace("=", "").strip()

  try:
    answer_str = (
        action.replace("The answer is ", "").strip().rstrip(".")
    )
    answer = float(answer_str)
    correct_value = eval(expression)
    tolerance = 1e-6
    if abs(correct_value - answer) < tolerance:
      score = 1.0
    else:
      score = 0.0

  except Exception:
    score = 0.0
  return RewardOutput(score, {"calculate_correct": score})
