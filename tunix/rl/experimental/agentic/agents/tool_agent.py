import copy
import json
import logging
import uuid
from typing import Any

from tunix.rl.experimental.agentic.agents.base_agent import LLMBaseAgent, Step, Action, Trajectory
from tunix.rl.experimental.agentic.tools.base_tool import BaseTool
from tunix.rl.experimental.agentic.tools.tool_manager import ToolManager
from tunix.rl.experimental.agentic.parser.tool_parser.tool_parser_registry import get_tool_parser
from tunix.rl.experimental.agentic.parser.tool_parser.tool_parser_base import ToolParser

logger = logging.getLogger(__name__)


class ToolAgent(LLMBaseAgent):
    """
    An Agent implementation that supports tool usage, conforming to the LLMBaseAgent abstract interface.
    Supports injecting tool_map and parsing with a specified parser.
    """

    def __init__(
        self,
        system_prompt: str,
        parser_name: str = "qwen",
        tool_map: dict[str, type[BaseTool]] | None = None,
    ):
        self.system_prompt = system_prompt

        # Tool manager (Router)
        self.tool_manager = ToolManager(tool_map=tool_map or {})

        # Parser (converts LLM responses to tool calls)
        parser_cls: type[ToolParser] = get_tool_parser(parser_name)
        self.tool_parser = parser_cls()
        
        # Build tools prompt: inject JSON Schema
        tools_json = json.dumps(self.tool_manager.json, indent=2)
        self.tools_prompt = self.tool_parser.get_tool_prompt(tools_json)
        
        # Internal state
        self._trajectory = Trajectory()
        self._messages: list[dict[str, Any]] = []
        self._obs_cache = None  # Caches the last observation

        self.reset()

    # ─────────────────────────────────────────────────────────────
    # Property Interfaces
    # ─────────────────────────────────────────────────────────────

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return self._messages

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    # ─────────────────────────────────────────────────────────────
    # Interaction with Environment
    # ─────────────────────────────────────────────────────────────

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Write observation / reward and update context"""
        step = self.get_current_state()
        if step:
            step.observation = observation
            step.reward = reward
            step.done = done
            step.info = info or {}

        # Cache the observation for next round of generation
        self._obs_cache = observation

        # Convert to messages
        if isinstance(observation, dict):
            if "tool_outputs" in observation:
                for call_id, output in observation["tool_outputs"].items():
                    self._messages.append({
                        "role": "user",
                        "tool_call_id": call_id,
                        "content": "Tool returned result: " + output,
                    })
            elif "question" in observation:
                self._messages.append({
                    "role": "user",
                    "content": observation["question"],
                })
        elif isinstance(observation, str):
            self._messages.append({"role": "user", "content": observation})

    # ─────────────────────────────────────────────────────────────
    # Interaction with Model
    # ─────────────────────────────────────────────────────────────

    def update_from_model(self, response: str, **kwargs) -> Action:
        """Parse model output, construct Action, and record Step"""
        try:
            tool_calls = self.tool_parser.parse(response)
        except Exception as e:
            logger.warning(f"ToolParser failed: {e}")
            tool_calls = []

        # Fallback: no tool call → use finish function
        if not tool_calls:
            tool_calls_dict = [{
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": "finish",
                    "arguments": {
                        "response": response
                    }
                }
            }]
        else:
            tool_calls_dict = []
            for tool_call in tool_calls:
                args = tool_call.arguments
                if isinstance(args, dict):
                    args = json.dumps(args)
                tool_calls_dict.append({
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": args
                    }
                })

        # Append assistant's response
        self._messages.append({"role": "assistant", "content": response})

        # Record Step
        step = Step(
            chat_completions=copy.deepcopy(self._messages),
            model_response=response,
            action=tool_calls_dict,
            observation=self._obs_cache,
        )
        self._trajectory.steps.append(step)

        return Action(action=tool_calls_dict)

    # ─────────────────────────────────────────────────────────────
    # Lifecycle Control
    # ─────────────────────────────────────────────────────────────

    def reset(self):
        self._trajectory = Trajectory()
        self._obs_cache = None
        self._messages = [{
            "role": "system",
            "content": self.system_prompt + self.tools_prompt
        }]
