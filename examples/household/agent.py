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

"""Household Task Exploration Agent for Multi-Turn Agentic RL."""

from collections.abc import Sequence
import re
from typing import Any

from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent

SYSTEM_PROMPT: str = """You are an intelligent agent exploring a household to accomplish interactive tasks.

Household Quick Guide:
Your goal is specified at the beginning of the episode. You can navigate rooms, inspect furniture, open and unlock containers, and pick up or place objects.

Valid Verbs and Actions:
- go <direction>               (e.g., go north, go south, go east, go west)
- look                         (describe current room, exits, and visible items)
- inventory                    (list items currently carried in your inventory)
- open <container>             (e.g., open refrigerator, open kitchen drawer)
- close <container>            (e.g., close refrigerator)
- unlock <container> with <key> (e.g., unlock wooden chest with silver key)
- take <item>                  (e.g., take apple, take silver key from drawer)
- put <item> in <container>    (e.g., put apple in refrigerator)
- examine <item or container>  (inspect properties and contents)

Rules & Strategy:
1. Always analyze the observation and plan your next move toward the goal.
2. If a container is locked, locate the appropriate key first.
3. Show your thinking process, then provide your exact action enclosed in triple backticks:
   ```<action>```.
4. Output ONLY ONE action per turn inside ``` ```.

Example:
Assistant: I am in the kitchen and see a kitchen drawer. I should open it to check if the key is inside.
Action: ```open kitchen drawer```
"""

MULTI_SHOT_SYSTEM_PROMPT: str = """You are an intelligent agent exploring a household to accomplish interactive tasks.

Household Quick Guide:
Your goal is specified at the beginning of the episode. You can navigate rooms, inspect furniture, open and unlock containers, and pick up or place objects.

Valid Verbs and Actions:
- go <direction>               (e.g., go north, go south, go east, go west)
- look                         (describe current room, exits, and visible items)
- inventory                    (list items currently carried in your inventory)
- open <container>             (e.g., open refrigerator, open kitchen drawer)
- close <container>            (e.g., close refrigerator)
- unlock <container> with <key> (e.g., unlock wooden chest with silver key)
- take <item>                  (e.g., take apple, take silver key from drawer)
- put <item> in <container>    (e.g., put apple in refrigerator)
- examine <item or container>  (inspect properties and contents)

Rules & Strategy:
1. Always analyze the observation and plan your next move toward the goal.
2. If a container is locked, locate the appropriate key first.
3. Show your thinking process, then provide your exact action enclosed in triple backticks:
   ```<action>```.
4. Output ONLY ONE action per turn inside ``` ```.

Example 1:
Observation:
Goal: Find the silver key in the kitchen drawer, unlock the wooden chest in the bedroom, and take the golden trophy.
You are in the living room. Exits: north to kitchen, east to bedroom.

Assistant: The goal mentions the silver key is in the kitchen drawer. I am in the living room and an exit to the kitchen is north. I will move north.
Action: ```go north```

Example 2:
Observation:
You are in the kitchen. You see: a kitchen drawer (closed).

Assistant: I need to check the drawer for the silver key. First, I need to open the drawer.
Action: ```open kitchen drawer```
"""


class HouseholdAgent(base_agent.ConversationAgentBase):
  """Agent interacting with HouseholdEnv in multi-turn RL episodes."""

  def __init__(self, system_prompt: str | None = None, multi_shot: bool = True):
    prompt = system_prompt or (
        MULTI_SHOT_SYSTEM_PROMPT if multi_shot else SYSTEM_PROMPT
    )
    super().__init__(system_prompt=prompt)
    self.last_observation: str | None = None
    self._last_raw_obs: str | None = None
    self.cur_step: agent_types.Step | None = None

  def update_from_env(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: dict[str, Any] | None = None,
      **kwargs,
  ) -> None:
    """Process feedback from HouseholdEnv and format next prompt."""
    obs_str = str(observation)
    new_obs_str = f"Observation:\n{obs_str}"
    if not done:
      new_obs_str += (
          "\nPlease think step-by-step and provide your next action in ``` ```."
      )

    # Loop detection warning based on raw observation stagnation
    if self._last_raw_obs is not None and self._last_raw_obs == obs_str:
      new_obs_str += (
          "\nWarning: Your position or state did not change. Please verify your"
          " command and ensure it follows valid verb syntax in ``` ```."
      )

    self._last_raw_obs = obs_str
    self.last_observation = new_obs_str
    super().update_from_env(new_obs_str, reward, done, info)
    self.cur_step = agent_types.Step(observation=new_obs_str)

  def _observation_to_messages(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: dict[str, Any],
  ) -> None:
    del reward, done, info
    self._messages.append({"role": "user", "content": str(observation)})

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    """Extract action command from model generation."""
    thought = response
    action_str = "look"

    matches = re.findall(r"```(.*?)```", response, re.DOTALL)
    if matches:
      last_match = matches[-1].strip()
      last_idx = response.rfind(f"```{last_match}```")
      if last_idx != -1:
        thought = response[:last_idx].strip()
      action_str = last_match.strip()
      if "\n" in action_str:
        first_line, rest = action_str.split("\n", 1)
        if first_line.strip().lower() in {"action", "text", "bash", "sh"}:
          action_str = rest.strip()

    # Append assistant's response
    self._messages.append({"role": "assistant", "content": response})

    action_obj = agent_types.Action(action=action_str)
    # Record step in trajectory
    if self.cur_step is not None:
      self.cur_step.thought = thought
      self.cur_step.action = action_obj
      self.cur_step.model_response = response
      self._trajectory.steps.append(self.cur_step)

    self.step += 1
    return action_obj

  def reset(self) -> None:
    """Reset agent conversation history and cached state."""
    super().reset()
    self.last_observation = None
    self._last_raw_obs = None
    self.cur_step = None
