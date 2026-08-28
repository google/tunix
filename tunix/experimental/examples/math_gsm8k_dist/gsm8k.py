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
"""GSM8K agentic components used by the distributed GRPO example."""

import re
from typing import Any
from tunix.experimental.rl.agentic import registry
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment

GSM8K_ENV_NAME = "gsm8kenv"
GSM8K_AGENT_NAME = "gsm8kagent"

GSM8K_PROMPT_TEMPLATE = (
    "Solve the following math problem.\n"
    "First, put your detailed step-by-step reasoning process inside "
    "<reasoning>...</reasoning> tags.\n"
    "Then, put your final numerical answer inside "
    "<answer>\\boxed{{}}</answer> tags.\n"
    "Do not put anything else in the answer tags.\n\n"
    "Problem: {}\n"
    "<reasoning>\n"
)


def extract_hash_answer(text: str) -> str | None:
  if "####" not in text:
    return None
  return text.split("####", 1)[1].strip()


def build_prompt(question: str) -> str:
  return GSM8K_PROMPT_TEMPLATE.format(question)


def extract_boxed_answer(text: str) -> str | None:
  """Extracts the final boxed answer from a GSM8K completion."""
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


def is_gsm8k_format_correct(text: str) -> bool:
  """Returns whether the completion follows the expected GSM8K response format."""
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


def score_gsm8k_completion(
    completion: str, gold: Any
) -> tuple[float, dict[str, Any]]:
  """Scores a GSM8K completion with the math example reward."""
  format_ok = is_gsm8k_format_correct(completion)
  prediction = normalize_answer(extract_boxed_answer(completion))
  target = normalize_answer(gold)
  answer_ok = (
      prediction is not None and target is not None and prediction == target
  )

  if format_ok and answer_ok:
    reward = 1.0
  elif format_ok and not answer_ok:
    reward = 0.1
  elif not format_ok and answer_ok:
    reward = 0.5
  else:
    reward = 0.0

  return reward, {
      "format_correct": format_ok,
      "answer_correct": answer_ok,
      "extracted_answer": prediction,
      "gold_answer": target,
  }


def gsm8k_env_reward(
    task: dict[str, Any], action: Any
) -> tuple[float, dict[str, Any]]:
  completion = action.action if hasattr(action, "action") else action
  return score_gsm8k_completion(str(completion), task.get("answer"))


@registry.register_env(GSM8K_ENV_NAME)
class GSM8KEnv(base_environment.BaseTaskEnv):
  """Single-step GSM8K environment using the math reward."""

  def __init__(
      self,
      prompt: str = "",
      prompts: str = "",
      question: str = "",
      answer: str = "",
      gold_answer: str = "",
      group_id: str = "",
      pair_index: int = 0,
      policy_version: int = 0,
      max_steps: int = 1,
      **kwargs: Any,
  ):
    prompt_text = prompts or prompt or (
        build_prompt(question) if question else ""
    )
    answer_text = answer or gold_answer
    super().__init__(
        task={
            "prompts": prompt_text,
            "question": question,
            "answer": answer_text,
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
    reward, info = gsm8k_env_reward(self.task, action)
    return base_environment.EnvStepResult(
        observation={
            "answer": str(action),
            "gold_answer": info.get("gold_answer"),
            "extracted_answer": info.get("extracted_answer"),
        },
        reward=reward,
        done=True,
        info=info,
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
