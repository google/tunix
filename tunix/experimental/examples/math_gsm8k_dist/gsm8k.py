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

import collections.abc
import logging
import re
from typing import Any

import grain
import numpy as np
import tensorflow_datasets as tfds
from tunix.experimental.rl.agentic import registry
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment

try:
  # For OSS usage
  import tensorflow_datasets.text.gsm8k  # pylint: disable=unused-import,g-import-not-at-top
except (ImportError, ModuleNotFoundError):
  pass

GSM8K_ENV_NAME = "gsm8kenv"
GSM8K_AGENT_NAME = "gsm8kagent"

GSM8K_PROMPT_TEMPLATE = (
    "Solve the following math problem.\n"
    "First, put your detailed step-by-step reasoning process inside "
    "<reasoning>...</reasoning> tags.\n"
    "Then, put your final numerical answer inside "
    "<answer>\\boxed{{}}</answer> tags. Do not put anything else in the "
    "answer tags.\n"
    "After the closing </answer> tag, immediately end your response and stop generating.\n\n"
    "Problem: {}\n"
    "<reasoning>\n"
)


def extract_hash_answer(text: str) -> str | None:
  """Extracts the canonical GSM8K final answer after the `####` delimiter."""
  if "####" not in text:
    return None
  return text.split("####", 1)[1].strip()


def build_prompt(question: str) -> str:
  return GSM8K_PROMPT_TEMPLATE.format(question)


def normalize_example_value(value: Any) -> Any:
  """Normalizes numpy/bytes values from dataset records to python primitives/strings."""
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


def as_text(value: Any) -> str:
  """Converts a dataset field value to text string."""
  normalized = normalize_example_value(value)
  return normalized if isinstance(normalized, str) else str(normalized)


def load_gsm8k_dataset(
    split: str = "train",
    data_dir: str = "/tmp/gsm8k_data",
    shuffle: bool = True,
    seed: int = 42,
) -> grain.MapDataset:
  """Loads the GSM8K split and maps examples to prompt/question/answer records."""
  logging.info(
      "Loading GSM8K TFDS split=%s data_dir=%s shuffle=%s seed=%d.",
      split,
      data_dir,
      shuffle,
      seed,
  )
  data = tfds.data_source(
      "gsm8k",
      split=split,
      data_dir=data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=True,
  )
  dataset = grain.MapDataset.source(data)
  if shuffle:
    dataset = dataset.shuffle(seed=seed)
  logging.info("GSM8K dataset loaded successfully: %d examples.", len(dataset))
  return dataset.map(
      lambda x: {
          "prompts": build_prompt(as_text(x["question"])),
          "question": as_text(x["question"]),
          "answer": extract_hash_answer(as_text(x["answer"])),
      }
  )


def extract_boxed_answer(text: str) -> str | None:
  """Extracts the final boxed answer from the VTC answer block."""
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


def is_gsm8k_format_correct(text: str) -> bool:
  """Checks the reasoning-then-boxed-answer format used by the GSM8K recipe."""
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
    completion: str, gold_answer: Any
) -> tuple[float, dict[str, Any]]:
  """Scores a GSM8K completion using the real VTC recipe reward shape."""
  format_correct = is_gsm8k_format_correct(completion)
  predicted = normalize_answer(extract_boxed_answer(completion))
  expected = normalize_answer(gold_answer)
  answer_correct = (
      predicted is not None and expected is not None and predicted == expected
  )

  if format_correct and answer_correct:
    reward = 1.0
  elif format_correct and not answer_correct:
    reward = 0.1
  else:
    reward = 0.0
  return reward, {
      "format_correct": format_correct,
      "answer_correct": answer_correct,
      "extracted_answer": predicted,
      "gold_answer": expected,
  }


def gsm8k_env_reward(
    task: dict[str, Any], action: Any
) -> tuple[float, dict[str, Any]]:
  completion = action.action if hasattr(action, "action") else str(action)
  gold_answer = task.get("answer", task.get("gold_answer"))
  return score_gsm8k_completion(str(completion), gold_answer)


def make_gsm8k_reward_fn(
    debug: bool = False,
) -> collections.abc.Callable[[Any], float]:
  """Creates an orchestrator-side reward function scoring completions against gold answers."""

  def reward_fn(item: Any) -> float:
    metadata = dict(getattr(item, "metadata", None) or {})
    text = str(metadata.get("text", ""))
    gold_answer = metadata.get("answer", metadata.get("gold_answer"))
    reward, _ = score_gsm8k_completion(text, gold_answer)
    if debug:
      prompt_id = metadata.get(
          "prompt_id",
          getattr(item, "group_id", getattr(item, "prompt_id", "unknown")),
      )
      logging.debug(
          "[Orchestrator] Sampler response for %s:\n"
          "[Sampled Response] ---\n%s\n--- [End Response] ---\n"
          "Gold Answer: %s, Extracted Answer: %s",
          prompt_id,
          text,
          gold_answer,
          extract_boxed_answer(text),
      )
    return reward

  return reward_fn


@registry.register_env(GSM8K_ENV_NAME)
class GSM8KEnv(base_environment.BaseTaskEnv):
  """Single-step GSM8K environment for VTC math rollouts."""

  def __init__(
      self,
      prompt: str = "",
      prompts: str = "",
      question: str = "",
      answer: str = "",
      gold_answer: str = "",
      prompt_id: str = "",
      group_index: int = 0,
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
            "gold_answer": answer_text,
            "policy_version": policy_version,
        },
        max_steps=max_steps,
        prompt_id=prompt_id,
        group_index=group_index,
        **kwargs,
    )

  def _initial_observation(self) -> dict[str, str]:
    return {"prompts": self.task.get("prompts", "")}

  def _step_impl(self, action: Any) -> base_environment.EnvStepResult:
    completion = action.action if hasattr(action, "action") else str(action)
    reward, info = gsm8k_env_reward(self.task, action)
    info["correct"] = bool(info["answer_correct"])
    return base_environment.EnvStepResult(
        observation={
            "answer": str(completion),
            "gold_answer": str(self.task.get("gold_answer", "")),
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
