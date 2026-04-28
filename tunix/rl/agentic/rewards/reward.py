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
as input and return a `reward_types.RewardOutput` containing a scalar reward and
metadata.
"""

import ast
import operator
from typing import Any, Callable, Dict

from tunix.rl.agentic.rewards import reward_types

_REGISTRY: Dict[
    str, Callable[[Dict[str, Any], str], reward_types.RewardOutput]
] = {}


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
def exact_match(task: Dict[str, Any], action: str) -> reward_types.RewardOutput:
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
  return reward_types.RewardOutput(score, {"exact_match": score})


def combine_rewards(
    weights: Dict[str, float],
) -> Callable[[Dict[str, Any], str], reward_types.RewardOutput]:
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
    return reward_types.RewardOutput(total, meta)

  return _fn


# -------- Example Reward Function --------
@register("is_two")
def is_two_reward(
    task: Dict[str, Any], action: str
) -> reward_types.RewardOutput:
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
  return reward_types.RewardOutput(score, {"is_two": score})


@register("dummy")
def dummy_reward(
    task: Dict[str, Any], action: str
) -> reward_types.RewardOutput:
  """A dummy reward function that always returns zero."""
  return reward_types.RewardOutput(0.0, {})


@register("calculate")
def calculate_reward(
    task: Dict[str, Any], action: str
) -> reward_types.RewardOutput:
  """Calculates the reward for a math expression based on answer correctness.

  Uses a safe AST-based math evaluator instead of eval() to prevent
  arbitrary code execution from untrusted input.

  Args:
    task: The task context containing the 'question' field.
    action: The model's answer as a string.

  Returns:
    RewardOutput: 1.0 if the model's answer matches the evaluated
    expression
      within a tolerance, 0.0 otherwise.
  """
  question_str = task.get("question", "")
  expression = question_str.replace("= ?", "").replace("=", "").strip()

  try:
    answer_str = action.replace("The answer is ", "").strip().rstrip(".")
    answer = float(answer_str)
    correct_value = _safe_eval_math(expression)
    tolerance = 1e-6
    if abs(correct_value - answer) < tolerance:
      score = 1.0
    else:
      score = 0.0

  except Exception:
    score = 0.0
  return reward_types.RewardOutput(score, {"calculate_correct": score})


# Allowed binary operators for safe math evaluation.
_SAFE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators for safe math evaluation.
_SAFE_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_math(expression: str) -> float:
  """Safely evaluate a mathematical expression using AST parsing.

  Only supports numeric literals and basic arithmetic operators
  (+, -, *, /, //, %, **). Raises ValueError for any non-arithmetic
  content, preventing code injection.

  Args:
    expression: A string containing a mathematical expression.

  Returns:
    The numeric result of the expression.

  Raises:
    ValueError: If the expression contains unsupported operations.
  """
  tree = ast.parse(expression, mode="eval")
  return float(_eval_node(tree.body))


def _eval_node(node: ast.AST) -> float:
  """Recursively evaluate an AST node for safe math expressions.

  Args:
    node: An AST node to evaluate.

  Returns:
    The numeric result of evaluating the node.

  Raises:
    ValueError: If the node type is not a supported numeric operation.
  """
  if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
    return float(node.value)
  if isinstance(node, ast.BinOp):
    op_fn = _SAFE_BINOPS.get(type(node.op))
    if op_fn is None:
      raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
    return op_fn(_eval_node(node.left), _eval_node(node.right))
  if isinstance(node, ast.UnaryOp):
    op_fn = _SAFE_UNARYOPS.get(type(node.op))
    if op_fn is None:
      raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    return op_fn(_eval_node(node.operand))
  raise ValueError(
      f"Unsupported expression node: {type(node).__name__}."
      " Only numeric literals and arithmetic operators are allowed."
  )
