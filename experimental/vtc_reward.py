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

"""VTC reward for the standard GRPO CLI path (recipe-alignment ablation).

Ports the exact scoring used by the agentic demo
(examples/math_gsm8k/qwen3_grpo_demo.py: extract_boxed_answer,
is_vtc_format_correct, normalize_answer, _vtc_completion_outcome) so the
standard `grpo_main` path trains on the SAME reward as the converging
agentic run. Helpers are underscore-prefixed on purpose: the CLI reward
loader (tunix/cli/config.py obtain_reward_fn) registers every PUBLIC
function in this file as a reward, so exactly one public function is
exposed. Scoring: format+answer=1.0, format-only=0.1, answer-only=0.5,
neither=0.0 (single scalar in [0,1]; NOT the 4-function sum of
tunix/cli/reward_fn/gsm8k.py).
"""

import re

import numpy as np


def _normalize_example_value(value):
  # Ported from qwen3_grpo_demo.py::_normalize_example_value.
  if isinstance(value, np.ndarray):
    flat = value.reshape(-1).tolist()
    if len(flat) == 1:
      return _normalize_example_value(flat[0])
    return [_normalize_example_value(v) for v in flat]
  if isinstance(value, np.bytes_):
    return value.tobytes().decode("utf-8")
  if isinstance(value, bytes):
    return value.decode("utf-8")
  return value


def _extract_boxed_answer(text: str) -> str | None:
  # Ported from qwen3_grpo_demo.py::extract_boxed_answer.
  answer_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
  content = answer_blocks[-1] if answer_blocks else text

  boxed = []
  stack = []
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

  fallback = re.search(r"\\boxed\s*\{?\s*([a-zA-Z0-9\.,\-]+)\s*\}?", content)
  if fallback:
    return fallback.group(1).strip()
  return None


def _is_vtc_format_correct(text: str) -> bool:
  # Ported from qwen3_grpo_demo.py::is_vtc_format_correct.
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


def _normalize_answer(text: str | None) -> str | None:
  # Ported from qwen3_grpo_demo.py::normalize_answer.
  if text is None:
    return None
  return str(text).replace(",", "").strip()


def _vtc_completion_outcome(completion: str, gold) -> float:
  # Ported from qwen3_grpo_demo.py::_vtc_completion_outcome (score only).
  format_ok = _is_vtc_format_correct(completion)
  pred = _normalize_answer(_extract_boxed_answer(completion))
  true = _normalize_answer(_normalize_example_value(gold))
  answer_ok = pred is not None and true is not None and pred == true

  if format_ok and answer_ok:
    score = 1.0
  elif format_ok and not answer_ok:
    score = 0.1
  elif not format_ok and answer_ok:
    score = 0.5
  else:
    score = 0.0
  return score


# The single public reward function the CLI loader registers.
def vtc_reward(prompts, completions, answer, **kwargs):
  """VTC reward: 1.0 / 0.1 / 0.5 / 0.0 per completion (see module doc)."""
  del prompts, kwargs
  return [
      _vtc_completion_outcome(completion, gold)
      for completion, gold in zip(completions, answer)
  ]
