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
LLMBaseAgent = base_agent.LLMBaseAgent
Action = base_agent.Action
get_tool_parser = tool_parser_registry.get_tool_parser

logger = logging.getLogger(__name__)


class ToolAgent(LLMBaseAgent):
  """Agent implementation that supports tool usage within the LLMBaseAgent framework.

  This agent extends the base agent functionality to enable structured tool
  calling
  capabilities. It manages a collection of tools through a ToolManager, parses
  LLM
  responses to extract tool calls using configurable parsers, and maintains
  conversation
  context across multi-turn interactions with both the environment and tool
  systems.

  The agent automatically handles tool call formatting, response parsing, and
  fallback
  mechanisms when tool parsing fails. It supports dependency injection of custom
  tool sets and parser implementations for maximum flexibility.
  """

  def __init__(
      self,
      system_prompt: str,
      tool_parser_name: str = "qwen",
      tool_map: dict[str, type[BaseTool]] | None = None,
  ):
    """Initialize the ToolAgent with system prompt, parser, and tool configuration.

    Args:
        system_prompt (str): Base system prompt that defines the agent's
          behavior and instructions. Will be combined with auto-generated tool
          documentation.
        tool_parser_name (str): Name of the tool parser to use for extracting
          tool calls from LLM responses. Must be registered in the tool parser
          registry.
        tool_map (dict[str, type[BaseTool]] | None): Mapping of tool names to
          their implementation classes. If None, agent will operate without
          tools.
    """
    self.system_prompt = system_prompt

    # Tool management system for routing and executing tool calls
    self.tool_manager = ToolManager(tool_map=tool_map or {})

    # Parser component for converting LLM responses to structured tool calls
    parser_cls: type[ToolParser] = get_tool_parser(tool_parser_name)
    self.tool_parser = parser_cls()

    # Generate tool prompt by injecting JSON Schema into parser template
    self.tools_prompt = self.tool_parser.get_tool_prompt(
        self.tool_manager.get_tools()
    )

    # Internal state management
    self._trajectory = Trajectory()
    self._messages: list[dict[str, Any]] = []
    self._obs_cache = None  # Caches the last observation for step recording

    self.reset()

  # ─────────────────────────────────────────────────────────────
  # Property Interfaces
  # ─────────────────────────────────────────────────────────────

  @property
  def chat_completions(self) -> list[dict[str, str]]:
    """Get the current conversation context for LLM inference.

    Returns the complete message history including system prompt with tool
    documentation, user inputs, assistant responses, and tool call results.

    Returns:
        list[dict[str, str]]: Messages in OpenAI Chat Completions format
    """
    return self._messages

  @property
  def trajectory(self) -> Trajectory:
    """Get the complete trajectory for the current episode.

    Contains the full sequence of interaction steps including tool calls,
    environment responses, and performance metrics for analysis and storage.

    Returns:
        Trajectory: Complete episode trace with all steps and metadata
    """
    return self._trajectory

  # ─────────────────────────────────────────────────────────────
  # Interaction with Environment
  # ─────────────────────────────────────────────────────────────

  def update_from_env(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: dict[str, Any],
      **kwargs,
  ):
    """Process environment feedback and update conversation context.

    Handles different observation formats including structured tool outputs
    and plain text responses. Updates the current step with environment
    feedback and converts observations to appropriate message format.

    Args:
        observation (Any): Environment response - can be dict with tool_outputs,
          dict with question field, or plain string
        reward (float): Numerical reward signal from environment
        done (bool): Episode termination flag
        info (dict): Additional environment metadata
        **kwargs: Extended parameters for future compatibility
    """
    step = self.get_current_state()
    if step:
      step.observation = observation
      step.reward = reward
      step.done = done
      step.info = info or {}

    # Cache observation for inclusion in next step record
    self._obs_cache = observation

    # Convert observation to appropriate message format for conversation context
    if isinstance(observation, dict):
      if "tool_outputs" in observation:
        # Handle structured tool execution results
        for call_id, output in observation["tool_outputs"].items():
          self._messages.append({
              "role": "tool",
              "tool_call_id": call_id,
              "content": "Tool returned result: " + output,
          })
      elif "question" in observation:
        # Handle question-based task inputs
        self._messages.append({
            "role": "user",
            "content": observation["question"],
        })
      else:
        # Handle unexpected dict observation formats
        logger.warning("Unknown dict observation format: %s", observation)
    elif isinstance(observation, str):
      # Handle plain text observations
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
        response (str): Raw text output from the language model
        **kwargs: Additional model response metadata

    Returns:
        Action: Structured action containing tool calls ready for environment
        execution
    """
    # pylint: disable=broad-exception-caught
    try:
      tool_calls = self.tool_parser.parse(response)
    except Exception as e:
      logger.warning("ToolParser failed: %s", e)
      tool_calls = []

    # Fallback mechanism: if no tool calls detected, use finish function
    if not tool_calls:
      tool_calls_dict = [{
          "id": str(uuid.uuid4()),
          "type": "function",
          "function": {"name": "finish", "arguments": {"response": response}},
      }]
    else:
      # Convert parsed tool calls to standard format
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

    # Add assistant's response to conversation history
    self._messages.append({"role": "assistant", "content": response})

    # Record complete step with conversation context and parsed action
    step = Step(
        chat_completions=copy.deepcopy(self._messages),
        action=Action(action=tool_calls_dict),
        observation=self._obs_cache,
        model_response=response,
    )

    self._trajectory.steps.append(step)

    return Action(action=tool_calls_dict)

  # ─────────────────────────────────────────────────────────────
  # Lifecycle Control
  # ─────────────────────────────────────────────────────────────

  def reset(self):
    """Reset agent state for a new episode.

    Clears the trajectory history, message context, and observation cache.
    Reinitializes the conversation with the system prompt combined with
    tool documentation to prepare for fresh interaction sequence.
    """
    self._trajectory = Trajectory()
    self._obs_cache = None
    self._messages = [
        {"role": "system", "content": self.system_prompt + self.tools_prompt}
    ]
