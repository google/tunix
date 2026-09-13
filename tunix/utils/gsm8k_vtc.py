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

"""GSM8K "VTC" recipe: prompt template, reward, rollout metrics, raw parser.

Single source of truth for the GSM8K GRPO recipe, shared by
`examples/math_gsm8k/qwen3_grpo_demo.py` and the distributed experimental
example (`tunix/experimental/examples/math_gsm8k_dist/gsm8k.py`) so the two
cannot drift.

The recipe's contract, all in one place:

* The prompt (`VTC_PROMPT_TEMPLATE`) ends by opening `<reasoning>` for the
  model. It is meant to be fed verbatim (`VTCRawTextParser`), so the model
  continues the reasoning block rather than seeing the tag closed inside a
  chat-template user turn.
* Because the prompt opens the tag, `is_vtc_format_correct` checks the
  completion for exactly one `</reasoning>` and one ordered
  `<answer>..</answer>` pair -- it does NOT require the opening tag.
* The reward is graded: 1.0 format + correct answer, 0.1 format only,
  0.5 correct answer without the format, 0.0 otherwise.
"""

from __future__ import annotations

import re
from typing import Any, Tuple

from absl import logging
import numpy as np
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib

VTC_PROMPT_TEMPLATE = """Solve the following math problem.
First, put your detailed step-by-step reasoning process inside <reasoning>...</reasoning> tags.
Then, put your final numerical answer inside <answer>\\boxed{{}}</answer> tags. Do not put anything else in the answer tags.

Problem: {}
<reasoning>
"""


# ====== Dataset helpers ======


def normalize_example_value(value: Any) -> Any:
  """Normalizes numpy / bytes dataset values to Python primitives and str."""
  if isinstance(value, np.ndarray):
    flat = value.reshape(-1).tolist()
    if len(flat) == 1:
      return normalize_example_value(flat[0])
    return [normalize_example_value(v) for v in flat]
  if isinstance(value, np.bytes_):
    return value.tobytes().decode("utf-8")
  if isinstance(value, bytes):
    return value.decode("utf-8")
  return value


def normalize_single_example(example: dict[str, Any]) -> dict[str, Any]:
  return {key: normalize_example_value(value) for key, value in example.items()}


def as_text(value: Any) -> str:
  """Converts a dataset field value (str, bytes, numpy) to text."""
  normalized = normalize_example_value(value)
  return normalized if isinstance(normalized, str) else str(normalized)


def extract_hash_answer(text: str) -> str | None:
  """Extracts the canonical GSM8K final answer after the `####` delimiter."""
  if "####" not in text:
    return None
  return text.split("####", 1)[1].strip()


def build_prompt(question: str) -> str:
  return VTC_PROMPT_TEMPLATE.format(question)


# ====== Reward ======


def extract_boxed_answer(text: str) -> str | None:
  r"""Extracts the last `\boxed{...}` from the last `<answer>` block (or text).

  Args:
    text: The completion text to search.

  Returns:
    The extracted answer, or None if the text carries no `\boxed` answer.
  """
  answer_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
  content = answer_blocks[-1] if answer_blocks else text

  boxed = []
  stack = []
  for idx, char in enumerate(content):
    if char == "{":
      stack.append(idx)
    elif char == "}":
      if not stack:
        continue
      open_idx = stack.pop()
      if content[:open_idx].endswith(r"\boxed"):
        boxed.append(content[open_idx + 1 : idx].strip())
  if boxed:
    return boxed[-1]

  fallback = re.search(r"\\boxed\s*\{?\s*([a-zA-Z0-9\.,\-]+)\s*\}?", content)
  if fallback:
    return fallback.group(1).strip()
  return None


def is_vtc_format_correct(text: str) -> bool:
  """Exactly one `</reasoning>` and one `<answer>..</answer>`, in that order.

  The opening `<reasoning>` tag is not required: `VTC_PROMPT_TEMPLATE` opens
  it, so a completion that follows the prompt only carries the closing tag.

  Args:
    text: The completion text to check.

  Returns:
    True if the completion follows the recipe's format.
  """
  has_reasoning = text.count("</reasoning>") == 1
  has_answer = text.count("<answer>") == 1 and text.count("</answer>") == 1
  reasoning_end = text.find("</reasoning>")
  answer_open = text.find("<answer>")
  answer_close = text.find("</answer>")
  return (
      has_reasoning
      and has_answer
      and reasoning_end != -1
      and answer_open != -1
      and answer_close != -1
      and reasoning_end < answer_open < answer_close
  )


def normalize_answer(text: Any) -> str | None:
  if text is None:
    return None
  return str(text).replace(",", "").strip()


def vtc_completion_outcome(
    completion: str, gold: Any
) -> Tuple[float, bool, bool, bool]:
  """Scores one completion.

  Args:
    completion: The model's completion text.
    gold: The reference answer, as it comes out of the dataset (str, bytes or
      numpy scalar).

  Returns:
    (score, format_ok, answer_ok, extracted_ok) where score is 1.0 for
    format + correct answer, 0.1 for format only, 0.5 for a correct answer
    without the format, and 0.0 otherwise.
  """
  format_ok = is_vtc_format_correct(completion)
  pred = normalize_answer(extract_boxed_answer(completion))
  true = normalize_answer(normalize_example_value(gold))
  answer_ok = pred is not None and true is not None and pred == true
  extracted_ok = pred is not None
  if format_ok and answer_ok:
    score = 1.0
  elif format_ok and not answer_ok:
    score = 0.1
  elif not format_ok and answer_ok:
    score = 0.5
  else:
    score = 0.0
  return score, format_ok, answer_ok, extracted_ok


def vtc_env_reward(task: dict[str, Any], action: Any) -> float:
  """Environment reward: scores the action text against the task's answer."""
  gold = task.get("answer", task.get("gold_answer"))
  completion = action.action if hasattr(action, "action") else action
  score, _, _, _ = vtc_completion_outcome(str(completion), gold)
  return score


_metric_call_idx = 0


def vtc_metric_fn(prompts, completions, rewards, advantages, answer, **kwargs):
  """Rollout metrics: solve_all / solve_none / solve_partial / solve_ratio.

  "Solved" means reward > 0.1, i.e. the answer was correct with or without
  the format (the 0.1 tier is format-only).

  Args:
    prompts: Unused; part of the rollout metric callback signature.
    completions: Unused; part of the rollout metric callback signature.
    rewards: Per-completion rewards for the current rollout group.
    advantages: Unused; part of the rollout metric callback signature.
    answer: Unused; part of the rollout metric callback signature.
    **kwargs: Unused; part of the rollout metric callback signature.

  Returns:
    A mapping from metric name to a `(value, aggregation_fn)` pair.
  """
  del prompts, completions, advantages, answer, kwargs
  global _metric_call_idx
  _metric_call_idx += 1
  rewards = np.asarray(rewards, dtype=np.float32)
  solve_all = bool(np.all(rewards > 0.1))
  solve_none = bool(np.all(np.isclose(rewards, 0.0)))
  solve_partial = (not solve_all) and (not solve_none)
  solve_ratio = float(np.mean(rewards > 0.1))
  reward_mean = float(rewards.mean())
  reward_max = float(rewards.max())
  logging.info(
      "[rollout-metric] call=%d n=%d solve_ratio=%.3f reward_mean=%.3f"
      " reward_max=%.3f solve_all=%d solve_none=%d",
      _metric_call_idx,
      len(rewards),
      solve_ratio,
      reward_mean,
      reward_max,
      int(solve_all),
      int(solve_none),
  )
  return {
      "rewards/solve_all": (1 if solve_all else 0, np.mean),
      "rewards/solve_none": (1 if solve_none else 0, np.mean),
      "rewards/solve_partial": (1 if solve_partial else 0, np.mean),
      "rewards/solve_ratio": (solve_ratio, np.mean),
  }


# ====== Prompt parser ======


# The raw-text (no chat template) parser is a generic prompting mode, not part
# of this recipe; it lives with the other parsers. Kept as an alias so the
# recipe's name for it stays stable.
VTCRawTextParser = chat_parser_lib.RawTextParser
