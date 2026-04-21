# Copyright 2025 Model AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import re
from typing import Any

from env import FrozenLakeEnv  # noqa: E402

from rllm.core import Action, BaseAgent, register_agent

logger = logging.getLogger(__name__)

# Prompting format inspired by the RAGEN project: https://github.com/RAGEN-AI/RAGEN
SYSTEM_PROMPT: str = """You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Action (separated by | ):
Up | Down | Left | Right

Rewards:
Fall into hole: 0
Reach goal: +1.0

You will be provided the current observation, please decide on the next Action.
You should show your thought process and then input the final action in ``` ```.
You should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```.
You should plan ahead and need to achieve it in minimum number of steps.
You should be aware that frozen tiles can be slippery, but the chance is small and you should not overthink it.

Please show your thinking process and put the final action in ``` ```. In every turn, the final action MUST be one of Up, Down, Left, Right.
"""

MULTI_SHOT_SYSTEM_PROMPT: str = """You are a helpful assistant. You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Action (separated by | ):
Up | Down | Left | Right

Rewards:
Fall into hole: 0
Reach goal: +1.0

You will be provided the current observation, please decide on the next Action.
You should show your thought process and then input the final action in ``` ```.
You should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```.
You should plan ahead and need to achieve it in minimum number of steps.

Below are examples for an interaction:
Example1:
User: Current Observation:
P   _   _   _   _
O   _   _   O   _
O   _   O   _   _
O   _   _   G   _
_   _   _   _   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: P is now at the top right corner. It should reach G at the bottom right corner. I should move it closer to it. I can move right or down but there is a hole in down position and I can not move diagonally. There is no hole in my next movement right so I can move to right. Action: ```Right```

Example2:
User: Current Observation:
_   _   _   _
_   _   _   O
_   O   _   P
O   _   _   G
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: P is now at the near G. It should reach G to its bottom. I should move to be on it. There is no hole in my next movement so I can move to down. Action: ```Down```

Example3:
User: Current Observation:
_   _   _   O   _
O   _   P   O   _
O   _   O   _   _
O   _   _   G   _
_   _   _   _   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: G is at the bottom right relative to P. I want to move closer so I should move right or down. But there is a hole at each position and I do not want to fall into holes. Up and left are both valid but left brings me closer. Action: ```Left```

Example4:
User: Current Observation:
_   _   _   _
_   _   _   O
_   O   _   O
O   G   P   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: P is now near G. But game has not finished. P is not at G and I should never output invalid action. I need to recheck my understanding. P is not actually on G yet because they are not overlapping, it needs reach G to its left. Action: ```Left```

Example5:
User: Current Observation:
_   _   _   O   _
O   _   P   _   _
O   _   O   O   O
O   _   O   G   _
O   _   _   _   _
You have not achieved the goal, P has not reached G yet. Please give the next action.

Assistant: G is at the bottom right corner of P. I can move left, right, or up. Move right will initially bring me closer but I can't reach G that way. Move up and left means I can still reach G. Move up will result in 9 steps in total while left is 7 steps. I need to move left. Action: ```Left```

Now it is your turn, please show your thinking process and put the final action in ``` ```. In every turn, the final action MUST be one of Up, Down, Left, Right.
"""


@register_agent("frozenlake")
class FrozenLakeAgent(BaseAgent):
    def __init__(self, use_multistep_prompt: bool | None = False):
        self.step: int = 0
        self.multistep_prompt: bool | None = use_multistep_prompt
        self.last_observation: Any = None
        self.reset()

    @property
    def system_prompt(self):
        return SYSTEM_PROMPT if not self.multistep_prompt else MULTI_SHOT_SYSTEM_PROMPT

    def process_observation(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        new_obs_str = str(observation)
        # Base message for the user
        new_obs_str = "Current Observation: \n" + new_obs_str
        if not done:
            new_obs_str += (
                "\n" + "You have not achieved the goal, P has not reached G yet. Please give the next action."
            )

        # Check if the observation is the same as the previous step's observation
        # This check only makes sense if we have completed at least one step (i.e., received a model response and acted)
        if self.last_observation and self.last_observation == new_obs_str:
            new_obs_str += "\nYour last response is invalid. Your position didn't change at all. You may need to recheck your thinking process, action outputted, and the format of response. Remember, you should only output the NEXT ACTION at each interation in the ``` ```. For example, if you want to move up, you should output ```Up```."
        self.last_observation = new_obs_str
        return new_obs_str

    def process_action(self, action: str):
        DIRECTION_MAP = {"left": 1, "down": 2, "right": 3, "up": 4}

        thought = action
        action_str = str(FrozenLakeEnv.INVALID_ACTION)

        matches = re.findall(r"```(.*?)```", action, re.DOTALL)

        if matches:
            last_match_content = matches[-1].strip()
            last_match_index = action.rfind(f"```{last_match_content}```")
            if last_match_index != -1:
                thought = action[:last_match_index].strip()

            extracted_text = last_match_content.lower()

            if extracted_text in DIRECTION_MAP:
                action_str = str(DIRECTION_MAP[extracted_text])
            elif extracted_text.isdigit() and int(extracted_text) in DIRECTION_MAP.values():
                action_str = str(int(extracted_text))

        return Action(thought=thought, action=action_str)

    def reset(self) -> None:
        self.step = 0
        self.last_observation = None