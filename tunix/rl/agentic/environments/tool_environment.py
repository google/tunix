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

"""RL Environment for tool-based agent interactions.

This module defines the `ToolEnvironment`, a reinforcement learning environment
designed to interface with agents that interact with a set of tools. It handles
tool execution, reward calculation, and episode management within an RL
framework.
"""

import json
import logging
from typing import Any, Dict, List
import uuid

from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.rewards import reward
from tunix.rl.agentic.tools import base_tool
from tunix.rl.agentic.tools import tool_manager


BaseTaskEnv = base_environment.BaseTaskEnv
EnvStepResult = base_environment.EnvStepResult
BaseTool = base_tool.BaseTool
ToolManager = tool_manager.ToolManager
ToolCall = base_tool.ToolCall
Action = agent_types.Action
dummy_reward = reward.dummy_reward


class ToolEnvironment(BaseTaskEnv):
  """Reinforcement learning environment for tool-based agent interactions.

  This environment enables agents to execute external tools and receive
  structured feedback compatible with standard RL training pipelines.
  It manages tool execution, reward computation, episode termination,
  and maintains compatibility with the BaseEnv/BaseTaskEnv interface.

  The environment supports both single and multi-step tool interactions,
  automatic episode termination through 'finish' tool calls or reaching
  `max_steps`, and flexible reward function integration for various
  evaluation strategies.
  """

  def __init__(
      self,
      task: Dict[str, Any] | None = None,
      *,
      tool_map: Dict[str, type[BaseTool]],
      reward_fn=None,
      max_steps: int = 10,
      **kwargs,
  ):
    """Initialize the tool environment with task configuration and constraints.

    Args:
      task (Dict[str, Any] | None): Task specification containing problem
        description, ground truth, or other task-specific parameters. If None,
        defaults to empty dict.
      tool_map (Dict[str, type[BaseTool]]): Mapping of tool names to their
        implementation classes for tool discovery and execution.
      reward_fn: Reward function that takes (task, action) and returns
        RewardOutput with `.reward` and `.metadata` fields. If None, defaults to
        `dummy_reward` with a warning.
      max_steps (int): Maximum number of interaction steps before forced
        termination. Prevents infinite loops and controls episode length.
      **kwargs: Additional arguments reserved for future extensions.
    """
    if reward_fn is None:
      logging.warning("No reward_fn provided, defaulting to dummy_reward().")
      reward_fn = dummy_reward

    # Let BaseTaskEnv handle task, reward_fn, step_count, and max_steps.
    super().__init__(
        task=task,
        reward_fn=reward_fn,
        max_steps=max_steps,
        **kwargs,
    )

    # Tool execution system for managing available tools and their invocation.
    self.tool_manager = ToolManager(tool_map)

  def _initial_observation(self) -> Dict[str, Any]:
    """Reset the environment to initial state for a new episode.

    Clears step counter (in BaseTaskEnv) and returns the task specification as
    the initial observation. The task dictionary serves as the starting context
    for the agent to begin tool-based problem solving.

    Returns:
      Dict[str, Any]: The task specification as the initial observation.
    """
    return self.task

  def _step_impl(self, action: Any) -> EnvStepResult:
    """Execute one logical step of tool interaction based on agent's action.

    Processes the agent's action which can be a string response (indicating
    completion) or a list of tool calls to execute. Manages episode termination
    logic based on action type and 'finish' tool calls. The `max_steps` limit
    is enforced by the outer `BaseTaskEnv.step()`.

    Args:
      action (Any): Agent's action - can be string (final answer), list of tool
        call dicts, or None (treated as empty list).

    Returns:
      EnvStepResult: The outcome of this logical step.
    """
    if isinstance(action, Action):
      action = action.action

    # Handle None action as empty action list.
    if action is None:
      action = []

    # Normalize single tool call dict to list format.
    if isinstance(action, dict):
      action = [action]

    is_string = isinstance(action, str)
    done = is_string

    # Check for explicit finish tool call when action is a list.
    if isinstance(action, list):
      if any(
          call.get("function", {}).get("name") == "finish" for call in action
      ):
        done = True

    # Handle episode termination: compute final reward.
    if done:
      llm_answer = self._extract_llm_answer(action)
      r_out = self.reward_fn(task=self.task, action=llm_answer)
      return EnvStepResult(
          observation={},
          reward=r_out.reward,
          done=True,
          info={"response": action, "metadata": r_out.metadata},
      )

    # Handle continuing episode: execute tools and return intermediate results.
    tool_outputs = self._execute_tool_calls(action)
    obs = {"tool_outputs": tool_outputs}
    return EnvStepResult(
        observation=obs,
        reward=0.0,
        done=False,
        info={"response": action, "metadata": {}},
    )

  @staticmethod
  def _extract_llm_answer(action: Any) -> str:
    """Extract the final answer string from various action formats.

    Handles multiple action types including direct string responses,
    finish tool calls with response arguments, and fallback string
    conversion for other action types.

    Args:
      action (Any): Agent's action in various possible formats.

    Returns:
      str: Extracted answer string for reward computation.
    """
    if isinstance(action, str):
      return action
    if isinstance(action, list):
      # Look for finish tool call with response argument.
      for call in action:
        if call.get("function", {}).get("name") == "finish":
          args = call["function"].get("arguments", {})
          return args.get("response", "")
    # Fallback: convert action to string representation.
    return str(action)

  def _execute_tool_calls(
      self, tool_calls: List[Dict[str, Any]]
  ) -> Dict[str, str]:
    """Execute a list of tool calls and return their outputs.

    Converts raw tool call dictionaries to ToolCall objects, assigns
    unique IDs if missing, and delegates execution to the ToolManager
    with parallel execution enabled for performance.

    Args:
      tool_calls (List[Dict[str, Any]]): List of tool call specifications
        containing function name, arguments, and optional call ID.

    Returns:
      Dict[str, str]: Mapping from tool call IDs to their output strings.
    """
    call_objs = []
    for tc in tool_calls:
      name = tc["function"]["name"]
      args = json.loads(tc["function"]["arguments"])
      call_id = tc.get("id") or str(uuid.uuid4())

      # Create ToolCall object and attach ID for result tracking.
      call_obj = ToolCall(name=name, arguments=args)
      setattr(call_obj, "id", call_id)
      call_objs.append(call_obj)

    return self.tool_manager.execute_calls(call_objs, parallel=True)

  @classmethod
  def from_dict(cls, env_args: Dict[str, Any]) -> "ToolEnvironment":
    """Create ToolEnvironment instance from configuration dictionary.

    Factory method that extracts environment-specific parameters from
    the configuration dict and uses remaining entries as the task
    specification. This preserves the original behavior:

      * `tool_map` (required) specifies the available tools.
      * `reward_fn` (optional) overrides the default reward function.
      * `max_steps` (optional, defaults to 10) controls episode length.
      * Any other entries become part of the `task` specification.

    Args:
      env_args (Dict[str, Any]): Configuration dictionary.

    Returns:
      ToolEnvironment: Configured environment instance ready for use.
    """
    # Extract environment configuration parameters.
    tool_map = env_args.pop("tool_map", None)
    reward_fn = env_args.pop("reward_fn", None)
    max_steps = env_args.pop("max_steps", 10)

    # Remaining entries form the task specification.
    task = env_args

    return cls(
        task=task,
        tool_map=tool_map,
        reward_fn=reward_fn,
        max_steps=max_steps,
    )
