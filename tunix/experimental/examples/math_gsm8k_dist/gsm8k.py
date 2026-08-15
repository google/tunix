# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GSM8K agentic components used by the distributed GRPO example."""

import re
from typing import Any

from tunix.experimental.rl.agentic import registry
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment

GSM8K_ENV_NAME = "gsm8kenv"
GSM8K_AGENT_NAME = "gsm8kagent"

PROMPT_TEMPLATE = """Solve the following math problem.
First, put your detailed step-by-step reasoning process inside <reasoning>...</reasoning> tags.
Then, put your final numerical answer inside <answer>\\boxed{{}}</answer> tags. Do not put anything else in the answer tags.

Problem: {question}
<reasoning>
"""


def build_prompt(question: str) -> str:
  """Builds the same single-turn GSM8K prompt style as the math GRPO demo."""
  return PROMPT_TEMPLATE.format(question=question)


def extract_hash_answer(text: str) -> str | None:
  """Extracts GSM8K's canonical answer from the dataset answer field."""
  if "####" not in text:
    return None
  return text.split("####", 1)[1].strip().replace(",", "")


def extract_boxed_answer(text: str) -> str | None:
  """Extracts the final boxed answer from a model completion."""
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


def is_math_gsm8k_format_correct(text: str) -> bool:
  """Checks the reasoning/answer tags used by the math GSM8K demo."""
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
  """Normalizes numeric GSM8K answers for exact string comparison."""
  if text is None:
    return None
  return str(text).replace(",", "").strip()


def score_completion(
    completion: str,
    gold_answer: Any,
    *,
    partial_credit: bool = True,
) -> tuple[float, bool, bool, bool]:
  """Scores a completion with the math GSM8K reward semantics.

  Returns:
    A tuple of (score, format_ok, answer_ok, extracted_ok).
  """
  format_ok = is_math_gsm8k_format_correct(completion)
  pred = normalize_answer(extract_boxed_answer(completion))
  true = normalize_answer(gold_answer)
  answer_ok = pred is not None and true is not None and pred == true
  extracted_ok = pred is not None

  if not partial_credit:
    return (1.0 if answer_ok else 0.0), format_ok, answer_ok, extracted_ok
  if format_ok and answer_ok:
    score = 1.0
  elif format_ok and not answer_ok:
    score = 0.1
  elif not format_ok and answer_ok:
    score = 0.5
  else:
    score = 0.0
  return score, format_ok, answer_ok, extracted_ok


@registry.register_env(GSM8K_ENV_NAME)
class GSM8KEnv(base_environment.BaseTaskEnv):
  """Single-step GSM8K environment for answer-only math rollouts."""

  def __init__(
      self,
      prompt: str = "",
      gold_answer: str = "",
      group_id: str = "",
      pair_index: int = 0,
      policy_version: int = 0,
      max_steps: int = 1,
      **kwargs: Any,
  ):
    super().__init__(
        task={
            "prompts": prompt,
            "gold_answer": gold_answer,
            "policy_version": policy_version,
        },
        max_steps=max_steps,
        group_id=group_id,
        pair_index=pair_index,
        **kwargs,
    )

  def _initial_observation(self) -> dict[str, str]:
    return {"prompts": self.task.get("prompts", "")}

  def _step_impl(self, action: Any) -> base_environment.EnvStepResult:
    completion = str(action)
    gold_answer = str(self.task.get("gold_answer", ""))
    score, format_ok, answer_ok, extracted_ok = score_completion(
        completion, gold_answer
    )
    return base_environment.EnvStepResult(
        observation={
            "answer": completion,
            "gold_answer": gold_answer,
        },
        reward=score,
        done=True,
        info={
            "format_ok": format_ok,
            "answer_ok": answer_ok,
            "extracted_ok": extracted_ok,
        },
    )


@registry.register_agent(GSM8K_AGENT_NAME)
class GSM8KAgent(base_agent.ConversationAgentBase):
  """Agent that forwards generated model text as the GSM8K environment action."""

  name = GSM8K_AGENT_NAME

  def __init__(self):
    super().__init__(
        "Solve the math problem. Return the final numeric answer clearly."
    )

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    del kwargs
    action = agent_types.Action(action=response)
    self.trajectory.steps.append(
        agent_types.Step(
            model_response=response,
            thought="",
            action=action,
        )
    )
    self.chat_completions.append({"role": "assistant", "content": response})
    return action
