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

"""Reinforcement learning environment for GSM8K mathematical reasoning tasks.

This module implements GSM8KEnv, a BaseTaskEnv subclass tailored for GSM8K
problem solving in the Tunix Agentic RL framework (compatible with GRPOLearner,
StandardRLProgram, and TrajectoryCollectEngine).

Similar to FrozenLakeEnv, it encapsulates:
  1. Task state and prompt templating.
  2. Initial observation rendering (presenting problem & reasoning instructions).
  3. Action processing (parsing thought chain & final boxed numerical answer).
  4. Format validation (checking <reasoning>...</reasoning> and <answer>\\boxed{}</answer> tags).
  5. Composite reward computation (dense format reward + exact numerical accuracy reward).
  6. Episode lifecycle and termination management.
"""

from __future__ import annotations

import copy
import logging
import re
from typing import Any, Callable, Dict

try:
  from tunix.rl.agentic.environments.base_environment import BaseTaskEnv, EnvStepResult
except ModuleNotFoundError:
  # Fallback when running in environments without full tunix / jax installed
  import importlib.util
  import os
  _BASE_ENV_PATH = os.path.abspath(
      os.path.join(
          os.path.dirname(__file__),
          "..",
          "..",
          "tunix",
          "rl",
          "agentic",
          "environments",
          "base_environment.py",
      )
  )
  if os.path.exists(_BASE_ENV_PATH):
    _spec = importlib.util.spec_from_file_location("base_environment", _BASE_ENV_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    BaseTaskEnv = _mod.BaseTaskEnv
    EnvStepResult = _mod.EnvStepResult
  else:
    raise


# ==============================================================================
# Prompt Templates and Formats
# ==============================================================================

DEFAULT_SYSTEM_PROMPT: str = (
    "You are a helpful assistant that solves math problems step by step. "
    "Put your detailed reasoning process between <reasoning> and </reasoning> tags. "
    "Then, put your final numerical answer inside <answer>\\boxed{}</answer> tags."
)

DEFAULT_PROMPT_TEMPLATE: str = (
    "<|im_start|>user\n"
    "Solve the following math problem step by step. "
    "Put your final numerical answer inside <answer>\\boxed{{}}</answer> tags.\n\n"
    "Problem: {question}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n"
)


# ==============================================================================
# Text and Answer Extraction Utilities
# ==============================================================================

def extract_hash_answer(text: str) -> str | None:
  """Extracts the target numerical answer following '####' in GSM8K solutions.

  Args:
    text: Raw ground-truth solution text from GSM8K.

  Returns:
    The clean stripped target answer string, or None if empty.
  """
  if not text:
    return None
  text_str = str(text)
  if "####" in text_str:
    return text_str.split("####")[-1].strip()
  return text_str.strip()


def extract_boxed_answer(text: str) -> str | None:
  """Extracts the final answer from boxed LaTeX or answer XML tags.

  Uses a robust bracket-matching parser to handle nested curly braces inside
  \\boxed{...}, with fallbacks to regex, <answer> tags, and trailing numbers.

  Args:
    text: Model generation output.

  Returns:
    Extracted answer string, or None if not found.
  """
  if not text:
    return None

  # Prefer content inside <answer>...</answer> tags if present
  answer_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
  content = answer_blocks[-1] if answer_blocks else text

  # 1. Stack-based extraction for balanced \boxed{...}
  boxed: list[str] = []
  stack: list[int] = []
  for i, ch in enumerate(content):
    if ch == "{":
      stack.append(i)
    elif ch == "}":
      if not stack:
        continue
      open_idx = stack.pop()
      if content[:open_idx].endswith(r"\boxed"):
        boxed.append(content[open_idx + 1 : i].strip())
  if boxed:
    return boxed[-1]

  # 2. Fallback regex for unclosed or malformed \boxed
  fallback = re.search(r"\\boxed\s*\{?\s*([a-zA-Z0-9\.,\-]+)\s*\}?", content)
  if fallback:
    return fallback.group(1).strip()

  # 3. Fallback to raw text inside <answer> tags
  if answer_blocks and answer_blocks[-1].strip():
    candidate = answer_blocks[-1].strip()
    # Strip any residual \boxed markup
    candidate = re.sub(r"^\\boxed\{?", "", candidate)
    candidate = re.sub(r"\}?$", "", candidate)
    return candidate.strip()

  # 4. Final numeric fallback
  numeric = re.findall(r"-?\d+(?:\.\d+)?", content)
  return numeric[-1].replace(",", "") if numeric else None


def normalize_answer(text: str | None) -> str | None:
  """Normalizes answer representation for robust mathematical equivalence.

  Strips commas, currency symbols, percentages, trailing float decimals, and
  LaTeX text annotations (e.g. \\text{...}).

  Args:
    text: Raw answer string.

  Returns:
    Canonicalized string representation.
  """
  if text is None:
    return None
  s = str(text).strip()
  s = s.replace(",", "").replace("$", "").replace("%", "")
  s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
  s = s.strip()
  try:
    f = float(s)
    if f.is_integer():
      return str(int(f))
    return str(f)
  except ValueError:
    return s


def is_format_correct(text: str) -> bool:
  """Checks if the completion satisfies required XML reasoning and answer tags.

  Accepts either (<reasoning> and </reasoning>) or </reasoning> boundary,
  or (<think> and </think>) or </think> boundary,
  and requires either (<answer> and </answer>) or \\boxed.

  Args:
    text: Model response string.

  Returns:
    True if reasoning and answer boundaries are both present.
  """
  if not text:
    return False
  has_reasoning = (
      ("<reasoning>" in text and "</reasoning>" in text)
      or ("</reasoning>" in text)
      or ("<think>" in text and "</think>" in text)
      or ("</think>" in text)
  )
  has_answer = (
      (r"\boxed" in text)
      or ("<answer>" in text and "</answer>" in text)
      or ("</answer>" in text)
  )
  return bool(has_reasoning and has_answer)


def answers_match(pred: str | None, gold: str | None) -> bool:
  """Compares predicted and gold numerical answers for equality."""
  p = normalize_answer(pred)
  g = normalize_answer(gold)
  if p is None or g is None:
    return False
  if p == g:
    return True
  try:
    return float(p) == float(g)
  except ValueError:
    return False


# ==============================================================================
# GSM8K Environment
# ==============================================================================

class GSM8KEnv(BaseTaskEnv):
  """Reinforcement learning environment for GSM8K mathematical reasoning tasks.

  Implements the BaseTaskEnv lifecycle:
    - _initial_observation(): Returns the structured task prompt observation.
    - _step_impl(action): Validates format, extracts boxed numerical answer,
      computes composite reward, and terminates the episode.

  Attributes:
    question: Raw GSM8K problem text.
    gold_answer: Normalized ground-truth numerical solution.
    prompt: Formatted prompt presented to the agent.
    format_reward: Reward awarded when reasoning and answer tags are correct.
    accuracy_reward: Reward awarded when final numerical answer is correct.
    partial_reward: Reward awarded when answer is correct but format is imperfect.
  """

  def __init__(
      self,
      entry: dict[str, Any] | None = None,
      *,
      task: dict[str, Any] | None = None,
      reward_fn: Callable[..., Any] | None = None,
      max_steps: int = 1,
      format_reward: float = 0.1,
      accuracy_reward: float = 1.0,
      partial_reward: float = 0.5,
      prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
      system_prompt: str = DEFAULT_SYSTEM_PROMPT,
      group_id: Any | None = None,
      pair_index: int | None = None,
      **kwargs,
  ):
    """Initializes the GSM8K environment.

    Args:
      entry: Dictionary containing task specification (question, answer, etc.).
      task: Alias for entry, accepted for BaseTaskEnv compatibility.
      reward_fn: Optional custom reward callable taking (task, action). If None,
        built-in composite format + accuracy scoring is used.
      max_steps: Maximum turns allowed (default 1 for standard single-turn CoT).
      format_reward: Reward bonus for correct tag formatting (default 0.1).
      accuracy_reward: Reward for correct numerical solution (default 1.0).
      partial_reward: Reward for correct answer with missing tags (default 0.5).
      prompt_template: Template used to format the problem observation.
      system_prompt: System instructions guiding step-by-step reasoning.
      group_id: Identifier grouping completions from the same prompt (for GRPO).
      pair_index: Index of completion within a prompt group.
      **kwargs: Extra parameters passed to BaseTaskEnv.
    """
    raw_task = entry if entry is not None else (task or {})
    self.raw_task = copy.deepcopy(raw_task)

    # Resolve question and gold answer
    self.question: str = str(
        self.raw_task.get("question") or self.raw_task.get("problem") or ""
    ).strip()

    raw_answer = self.raw_task.get("answer") or self.raw_task.get("gold_answer") or ""
    self.gold_answer: str | None = (
        self.raw_task.get("gold_answer")
        or extract_hash_answer(str(raw_answer))
    )
    if self.gold_answer is not None:
      self.gold_answer = normalize_answer(self.gold_answer)

    self.prompt_template = prompt_template
    self.system_prompt = system_prompt
    self.format_reward = float(format_reward)
    self.accuracy_reward = float(accuracy_reward)
    self.partial_reward = float(partial_reward)

    # Generate formatted prompt if not already explicitly provided
    explicit_prompt = self.raw_task.get("prompts") or self.raw_task.get("prompt")
    if explicit_prompt:
      self.prompt = str(explicit_prompt)
    elif self.question:
      self.prompt = self.prompt_template.format(question=self.question)
    else:
      self.prompt = ""

    # Task dict for BaseTaskEnv
    task_dict = {
        "question": self.question,
        "answer": str(raw_answer),
        "gold_answer": self.gold_answer,
        "prompts": self.prompt,
        "group_id": group_id,
        "pair_index": pair_index,
    }

    super().__init__(
        task=task_dict,
        reward_fn=reward_fn,
        max_steps=max_steps,
        **kwargs,
    )

    self.group_id = group_id
    self.pair_index = pair_index
    self.last_response: str | None = None
    self.last_extracted_answer: str | None = None
    self._success: bool = False

  def _initial_observation(self) -> dict[str, Any]:
    """Resets internal state and returns the initial observation dictionary."""
    self.last_response = None
    self.last_extracted_answer = None
    self._success = False
    return {
        "prompts": self.prompt,
        "question": self.question,
        "gold_answer": self.gold_answer,
    }

  def _step_impl(self, action: Any) -> EnvStepResult:
    """Executes a step by evaluating the agent's action/completion.

    Args:
      action: Model completion, string or object with an 'action' attribute.

    Returns:
      EnvStepResult with observation, reward, done flag, and diagnostic info.
    """
    response_text = action.action if hasattr(action, "action") else str(action)
    self.last_response = response_text

    format_ok = is_format_correct(response_text)
    pred_answer = extract_boxed_answer(response_text)
    self.last_extracted_answer = pred_answer
    answer_ok = answers_match(pred_answer, self.gold_answer)
    self._success = answer_ok

    # Calculate reward
    if self.reward_fn is not None:
      reward = float(self.reward_fn(task=self.task, action=response_text))
    else:
      if format_ok and answer_ok:
        reward = self.accuracy_reward
      elif format_ok and not answer_ok:
        reward = self.format_reward
      elif not format_ok and answer_ok:
        reward = self.partial_reward
      else:
        reward = 0.0

    # In single-turn tasks (max_steps=1), episode terminates immediately.
    # In multi-turn tasks (max_steps>1), finish upon correct answer or step limit.
    done = (self.max_steps <= 1) or answer_ok

    next_obs: dict[str, Any] = {}
    if not done:
      next_obs = {
          "prompts": (
              "Your previous answer was not correct or format was invalid. "
              "Please carefully re-read the problem, check your reasoning, "
              "and output your final answer inside <answer>\\boxed{}</answer>."
          ),
          "question": self.question,
      }

    info = {
        "format_correct": format_ok,
        "answer_correct": answer_ok,
        "extracted_answer": pred_answer,
        "gold_answer": self.gold_answer,
        "action_is_effective": bool(response_text.strip()),
        "response": response_text,
    }

    return EnvStepResult(
        observation=next_obs,
        reward=float(reward),
        done=done,
        info=info,
    )

  def success(self) -> bool:
    """Returns whether the agent has achieved the correct mathematical answer."""
    return self._success

  def render(self, mode: str = "text") -> str:
    """Renders the current environment state or interaction summary."""
    lines = [
        f"Problem: {self.question}",
        f"Gold Answer: {self.gold_answer}",
    ]
    if self.last_response is not None:
      lines.extend([
          "--- Last Completion ---",
          self.last_response,
          "-----------------------",
          f"Extracted Answer: {self.last_extracted_answer}",
          f"Success: {self._success}",
      ])
    return "\n".join(lines)

  @classmethod
  def from_dict(cls, env_info: dict[str, Any]) -> "GSM8KEnv":
    """Factory method to instantiate GSM8KEnv from a configuration dictionary.

    Args:
      env_info: Configuration dictionary containing task data and settings.

    Returns:
      A fully initialized GSM8KEnv instance.
    """
    info = dict(env_info)
    entry = info.pop("entry", info.pop("task", info))
    group_id = info.pop("group_id", None)
    pair_index = info.pop("pair_index", None)
    max_steps = info.pop("max_steps", 1)
    reward_fn = info.pop("reward_fn", None)
    return cls(
        entry=entry,
        reward_fn=reward_fn,
        group_id=group_id,
        pair_index=pair_index,
        max_steps=max_steps,
        **info,
    )
