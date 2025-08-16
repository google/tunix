# tunix/rl/multi_turn/environments/tool_env.py
import json
import warnings
from typing import Any, Dict, List
import uuid
from tunix.rl.multi_turn.environments.base_environment import BaseEnv
from tunix.rl.multi_turn.rewards.reward import zero_reward
from tunix.rl.multi_turn.tools.tool_manager import ToolManager
from tunix.rl.multi_turn.tools.base_tool import ToolCall, BaseTool


class ToolEnvironment(BaseEnv):
    """
    Environment that lets an Agent call external tools and produces
    (observation, reward, done, info) tuples compatible with RL pipelines.
    """

    def __init__(
        self,
        task: Dict | None = None,
        *,
        tool_map: Dict[str, type[BaseTool]],
        reward_fn = None,
        max_steps: int = 10,
    ):
        super().__init__()

        self.tool_manager = ToolManager(tool_map)

        if reward_fn is None:
            warnings.warn("No reward_fn provided, defaulting to zero_reward().")
            reward_fn = zero_reward
        self.reward_fn = reward_fn

        self.task = task or {}
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self) -> tuple[Dict, Dict]:
        self.step_count = 0
        return self.task, {}

    def step(self, action: Any) -> tuple[Any, float, bool, Dict]:
        if action is None:
            action = []

        if isinstance(action, dict):
            action = [action]
        is_string = isinstance(action, str)
        self.step_count += 1

        done = is_string or self.step_count >= self.max_steps
        if isinstance(action, list):
            if any(call.get("function", {}).get("name") == "finish" for call in action):
                done = True

        # ───── done: compute reward and terminate ─────
        if done:
            llm_answer = self._extract_llm_answer(action)
            r_out = self.reward_fn(task=self.task, action=llm_answer)
            return {}, r_out.reward, True, {"response": action, "metadata": r_out.metadata}

        # ───── not done: execute tools ─────
        tool_outputs = self._execute_tool_calls(action)
        obs = {"tool_outputs": tool_outputs}
        return obs, 0.0, False, {"response": action, "metadata": {}}

    @staticmethod
    def _extract_llm_answer(action: Any) -> str:
        if isinstance(action, str):
            return action
        if isinstance(action, list):
            for call in action:
                if call.get("function", {}).get("name") == "finish":
                    args = call["function"].get("arguments", {})
                    return args.get("response", "")
        return str(action)

    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Convert raw tool call dicts → ToolCall instances → pass to ToolManager
        """
        call_objs = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            call_id = tc.get("id") or str(uuid.uuid4())
            call_obj = ToolCall(name=name, arguments=args)
            setattr(call_obj, "id", call_id)
            call_objs.append(call_obj)

        return self.tool_manager.execute_calls(call_objs, parallel=True)

    @staticmethod
    def from_dict(env_args: Dict) -> "ToolEnvironment":
        tool_map = env_args.pop("tool_map", None)
        reward_fn = env_args.pop("reward_fn", None)
        max_steps = env_args.pop("max_steps", 10)
        task = env_args
        return ToolEnvironment(
            task=task,
            tool_map=tool_map,
            reward_fn=reward_fn,
            max_steps=max_steps,
        )
