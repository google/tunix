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

"""Agent implementation that supports tool usage."""

import copy
import json
import logging
from typing import Any
import uuid

from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.parser.tool_parser import tool_parser_base
from tunix.rl.agentic.parser.tool_parser import tool_parser_registry
from tunix.rl.agentic.tools import base_tool
from tunix.rl.agentic.tools import tool_manager


ToolManager = tool_manager.ToolManager
BaseTool = base_tool.BaseTool
ToolParser = tool_parser_base.ToolParser
Trajectory = agent_types.Trajectory
Step = agent_types.Step
Action = agent_types.Action
ConversationAgentBase = base_agent.ConversationAgentBase
get_tool_parser = tool_parser_registry.get_tool_parser

logger = logging.getLogger(__name__)


class ToolAgent(ConversationAgentBase):
  """Agent implementation that supports tool usage within ConversationAgentBase.

  This agent extends the base conversation agent functionality to enable
  structured tool-calling capabilities. It manages a collection of tools
  through a ToolManager, parses LLM responses to extract tool calls using
  configurable parsers, and maintains conversation context across multi-turn
  interactions with both the environment and tool systems.
  """

  def __init__(
      self,
      system_prompt: str,
      tool_parser_name: str = "qwen",
      tool_map: dict[str, type[BaseTool]] | None = None,
  ):
    # Tool management system for routing and executing tool calls.
    self.tool_manager = ToolManager(tool_map=tool_map or {})

    # Parser component for converting LLM responses to structured tool calls.
    parser_cls: type[ToolParser] = get_tool_parser(tool_parser_name)
    self.tool_parser = parser_cls()

    # Generate tool prompt by injecting JSON Schema into parser template.
    self.tools_prompt = self.tool_parser.get_tool_prompt(
        self.tool_manager.get_tools()
    )

    # Initialize the ConversationAgentBase
    super().__init__(system_prompt=system_prompt)

  # ─────────────────────────────────────────────────────────────
  # ConversationAgentBase hooks
  # ─────────────────────────────────────────────────────────────

  def _init_messages(self, system_prompt: str) -> None:
    """Initialize conversation history with system prompt + tools prompt."""
    # Build a single string with tools prompt appended to system prompt.
    content = system_prompt + (self.tools_prompt or "")
    self._messages = [{"role": "system", "content": content}]

  def _observation_to_messages(self, observation: Any) -> None:
    """Convert environment observation into messages, including tool outputs."""
    if isinstance(observation, dict):
      if "tool_outputs" in observation:
        # Handle structured tool execution results.
        for call_id, output in observation["tool_outputs"].items():
          self._messages.append({
              "role": "tool",
              "tool_call_id": call_id,
              "content": "Tool returned result: " + output,
          })
      elif "question" in observation:
        self._messages.append({
            "role": "user",
            "content": observation["question"],
        })
      else:
        logger.warning("Unknown dict observation format: %s", observation)
    elif isinstance(observation, str):
      self._messages.append({"role": "user", "content": observation})

  # ─────────────────────────────────────────────────────────────
  # Interaction with Model
  # ─────────────────────────────────────────────────────────────

  def update_from_model(self, response: str, **kwargs) -> Action:
    """Parse LLM response to extract tool calls and create structured action.

    Attempts to parse the model response for tool calls using the configured
    parser. If parsing fails or no tools are detected, falls back to a
    'finish' function call with the raw response. Records the complete
    interaction step in the trajectory.

    Args:
      response: The raw string response from the LLM.
      **kwargs: Additional keyword arguments.

    Returns:
      An `Action` object containing a list of tool calls derived from the
      model's response.
    """
    # pylint: disable=broad-exception-caught
    try:
      tool_calls = self.tool_parser.parse(response)
    except Exception as e:
      logger.warning("ToolParser failed: %s", e)
      tool_calls = []

    # Fallback mechanism: if no tool calls detected, use finish function.
    if not tool_calls:
      tool_calls_dict = [{
          "id": str(uuid.uuid4()),
          "type": "function",
          "function": {"name": "finish", "arguments": {"response": response}},
      }]
    else:
      # Convert parsed tool calls to standard format.
      tool_calls_dict = []
      for tool_call in tool_calls:
        args = tool_call.arguments
        if isinstance(args, dict):
          args = json.dumps(args)
        tool_calls_dict.append({
            "id": str(uuid.uuid4()),
            "type": "function",
            "function": {"name": tool_call.name, "arguments": args},
        })

    # Add assistant's response to conversation history.
    self._messages.append({"role": "assistant", "content": response})

    # Record complete step with conversation context and parsed action.
    step = Step(
        chat_completions=copy.deepcopy(self._messages),
        action=Action(action=tool_calls_dict),
        observation=self._obs_cache,
        model_response=response,
    )
    self._trajectory.steps.append(step)

    return Action(action=tool_calls_dict)
