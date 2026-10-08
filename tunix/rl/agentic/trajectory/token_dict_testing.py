# Copyright 2026 Google LLC
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

"""Test utilities for Token-mode dicts backed by the Trajectory Store."""

from typing import Any

import numpy as np
from tunix.experimental.trajectory import store as store_lib
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.trajectory import token_dict


def make_trajectory(reward: float = 1.5) -> agent_types.Trajectory:
  """A finished two-turn trajectory that the Trajectory Store round-trips."""
  return agent_types.Trajectory(
      task={"prompts": ["q"], "answer": None, "ids": np.array([7, 8])},
      steps=[
          agent_types.Step(
              model_response="a1",
              observation="obs1",
              assistant_tokens=np.array([11, 12, 13], dtype=np.int32),
              assistant_masks=np.array([1, 1, 0], dtype=np.int32),
              logprobs=np.array([-0.1, -0.2, 0.0], dtype=np.float32),
              env_tokens=np.array([21, 22], dtype=np.int32),
              env_masks=np.array([0, 0], dtype=np.int32),
          ),
          agent_types.Step(
              model_response="a2",
              done=True,
              reward=reward,
              assistant_tokens=np.array([14, 15], dtype=np.int32),
              assistant_masks=np.array([1, 1], dtype=np.int32),
          ),
      ],
      reward=reward,
      status=agent_types.TrajectoryStatus.SUCCEEDED,
      env_time={"reset_latency": 0.25},
      reward_time={"reward_latency": 0.125},
      prompt_tokens=np.array([1, 2, 3], dtype=np.int32),
  )


def token_dict_context(group_id: int = 9) -> token_dict.TokenDictContext:
  """The collect context of `make_trajectory`.

  Every call builds a new context: `build_token_dict` puts its
  `chat_completions` list into the dict it returns, so dicts built from one
  shared context would alias that list.

  Args:
    group_id: The GRPO group the trajectory belongs to.

  Returns:
    The context.
  """
  return token_dict.TokenDictContext(
      chat_completions=[{"role": "user", "content": "q"}],
      policy_version=4,
      group_id=group_id,
      masked_out=False,
      exact_token_continuity=False,
  )


def stored_token_dict(
    store: store_lib.TrajectoryWriter,
    trajectory_id: str,
    reward: float = 1.5,
    *,
    group_id: int = 9,
) -> dict[str, Any]:
  """Does what TrajectoryCollectEngine does with a store configured.

  Args:
    store: The Trajectory Store to write the trajectory to.
    trajectory_id: The store id to write it under.
    reward: The trajectory's reward.
    group_id: The GRPO group the trajectory belongs to.

  Returns:
    The Token-mode dict of `make_trajectory(reward)`, tagged with
    `trajectory_id`.
  """
  trajectory = make_trajectory(reward)
  context = token_dict_context(group_id)
  token_dict.write_trajectory(
      store, trajectory, context, trajectory_id=trajectory_id
  )
  return token_dict.build_token_dict(
      trajectory, context, trajectory_id=trajectory_id
  )
