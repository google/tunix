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

from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent


class ModelAgent(base_agent.ConversationAgentBase):
  """Agent for single-turn interaction, responding directly to a task."""

  def __init__(self, system_prompt: str):
    super().__init__(system_prompt=system_prompt)

  # If you want to handle observations in a special way, you can override
  # _observation_to_messages. Here, we stick to the default behavior of
  # ConversationAgentBase.

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    """Receive model response and return it as the final action."""
    # 1. Add the model's output to the conversation history.
    self.chat_completions.append({"role": "assistant", "content": response})

    # 2. Record the Step (observation uses the cache from the last env
    # feedback).
    step = agent_types.Step(
        chat_completions=copy.deepcopy(self.chat_completions),
        action=agent_types.Action(action=response),
        model_response=response,
    )
    self.trajectory.steps.append(step)

    # In a single-turn scenario, the response itself is the action to be
    # directly evaluated by the environment.
    return agent_types.Action(action=response)
