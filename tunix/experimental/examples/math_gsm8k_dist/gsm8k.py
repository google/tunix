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

from typing import Any
from tunix.experimental.rl.agentic import registry
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment

GSM8K_ENV_NAME = "gsm8kenv"
GSM8K_AGENT_NAME = "gsm8kagent"


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
    answer = str(action)
    gold_answer = str(self.task.get("gold_answer", ""))
    is_correct = bool(gold_answer) and gold_answer in answer
    return base_environment.EnvStepResult(
        observation={"answer": answer, "gold_answer": gold_answer},
        reward=1.0 if is_correct else 0.0,
        done=True,
        info={"correct": is_correct},
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
