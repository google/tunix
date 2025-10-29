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

"""Agent implementation for single-turn interactions."""

import copy
import logging
from typing import Any

from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent

Trajectory = agent_types.Trajectory
Step = agent_types.Step
LLMBaseAgent = base_agent.LLMBaseAgent
Action = base_agent.Action

logger = logging.getLogger(__name__)


class ModelAgent(LLMBaseAgent):
  """Agent for single-turn interaction, responding directly to a task."""

  def __init__(self, system_prompt: str):
    self.system_prompt = system_prompt
    self.reset()

  @property
  def chat_completions(self) -> list[dict[str, str]]:
    return self._messages

  @property
  def trajectory(self) -> Trajectory:
    return self._trajectory

  def update_from_env(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: dict[str, Any],
      **kwargs,
  ):
    step = self.get_current_state()
    if step:
      step.observation = observation
      step.reward = reward
      step.done = done
      step.info = info or {}
    self._obs_cache = observation
    if isinstance(observation, dict) and "question" in observation:
      self._messages.append(
          {"role": "user", "content": observation["question"]}
      )
    elif isinstance(observation, str):
      self._messages.append({"role": "user", "content": observation})
    elif not observation:
      logger.info(
          "No observation returned, trajectory ended."
      )
    else:
      logger.warning("Unknown dict observation format: %s", observation)

  def update_from_model(self, response: str, **kwargs) -> Action:
    """Receive model response and return it as the final action."""
    self._messages.append({"role": "assistant", "content": response})
    step = Step(
        chat_completions=copy.deepcopy(self._messages),
        action=Action(action=response),
        observation=self._obs_cache,
        model_response=response,
    )
    self._trajectory.steps.append(step)
    # For single-turn, the response itself is the action to be evaluated.
    return Action(action=response)

  def reset(self):
    self._trajectory = Trajectory()
    self._obs_cache = None
    self._messages = [{"role": "system", "content": self.system_prompt}]
