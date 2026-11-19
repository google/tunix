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

"""Agent implementation for GSM8K mathematical reasoning tasks.

This module implements GSM8KAgent, a ConversationAgentBase subclass tailored for
math problem solving in the Tunix Agentic RL framework.

Similar to FrozenLakeAgent, it maintains conversation context, translates
environment observations into chat prompt messages, and converts LLM responses
into structured Actions.
"""

from __future__ import annotations

import copy
import sys
from typing import Any, Dict

from examples.math_gsm8k.env import DEFAULT_SYSTEM_PROMPT

try:
  from tunix.rl.agentic.agents import agent_types
  from tunix.rl.agentic.agents import base_agent
except ModuleNotFoundError:
  # Fallback when running in environments without full tunix / jax installed
  import importlib.util
  import os
  import types

  _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
  _AGENT_DIR = os.path.join(_REPO_ROOT, "tunix", "rl", "agentic", "agents")

  if "tunix" not in sys.modules:
    sys.modules["tunix"] = types.ModuleType("tunix")
    sys.modules["tunix.rl"] = types.ModuleType("tunix.rl")
    sys.modules["tunix.rl.agentic"] = types.ModuleType("tunix.rl.agentic")
    sys.modules["tunix.rl.agentic.agents"] = types.ModuleType("tunix.rl.agentic.agents")

  _types_spec = importlib.util.spec_from_file_location(
      "agent_types", os.path.join(_AGENT_DIR, "agent_types.py")
  )
  agent_types = importlib.util.module_from_spec(_types_spec)
  sys.modules["tunix.rl.agentic.agents.agent_types"] = agent_types
  _types_spec.loader.exec_module(agent_types)

  _base_spec = importlib.util.spec_from_file_location(
      "base_agent", os.path.join(_AGENT_DIR, "base_agent.py")
  )
  base_agent = importlib.util.module_from_spec(_base_spec)
  sys.modules["tunix.rl.agentic.agents.base_agent"] = base_agent
  _base_spec.loader.exec_module(base_agent)


class GSM8KAgent(base_agent.ConversationAgentBase):
  """Agent for GSM8K mathematical reasoning interactions.

  Manages dialogue turns, formats observations into user prompts, and preserves
  step-by-step reasoning trajectories for GRPO / RL training.
  """

  def __init__(
      self,
      system_prompt: str = DEFAULT_SYSTEM_PROMPT,
      **kwargs,
  ):
    """Initializes the GSM8K agent.

    Args:
      system_prompt: Guiding prompt instructing the model to emit reasoning
        and boxed answers.
      **kwargs: Extra parameters passed to ConversationAgentBase.
    """
    super().__init__(system_prompt=system_prompt)
    self.last_observation: Any = None

  def _init_messages(self, system_prompt: str) -> None:
    """Initializes conversation history with the system prompt."""
    self._messages = [{"role": "system", "content": system_prompt or ""}]

  def _observation_to_messages(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: Dict[str, Any] | None = None,
  ) -> None:
    """Converts an observation from GSM8KEnv into chat messages."""
    del reward, done, info
    if isinstance(observation, dict):
      content = observation.get("prompts") or observation.get("question") or ""
    elif isinstance(observation, str):
      content = observation
    else:
      content = str(observation)

    if content:
      self._messages.append({"role": "user", "content": content})

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    """Processes LLM response, updates trajectory, and returns Action."""
    # 1. Record assistant completion in conversation history
    self._messages.append({"role": "assistant", "content": response})

    # 2. Record trajectory step
    step = agent_types.Step(
        chat_completions=copy.deepcopy(self._messages),
        action=agent_types.Action(action=response),
        model_response=response,
    )
    self._trajectory.steps.append(step)
    self.step += 1

    return agent_types.Action(action=response)

  def reset(self) -> None:
    """Resets agent state for a new episode."""
    super().reset()
    self.last_observation = None
