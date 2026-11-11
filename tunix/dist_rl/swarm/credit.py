# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Detached role-wise or episode-wise GRPO credit and native trainer payloads.

Matches DrMAS c2f724e7358e7dc5bf2ea792ab07a3b5d90290c9 outcome-advantage
semantics, including sample standard deviation and its singleton convention.
No new policy loss or optimizer implementation is introduced here.
"""

import collections
from collections.abc import Sequence
import dataclasses
import math
from typing import Literal

import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.swarm import trace


@dataclasses.dataclass(frozen=True)
class CreditBatch:
  payloads: tuple[datatypes.RLTrainerPayload, ...]
  rewards: tuple[float, ...]
  advantages: tuple[float, ...]
  roles: tuple[str, ...]


def build_credit(
    episodes: Sequence[trace.SwarmEpisode],
    *,
    normalization: Literal["role", "episode"] = "role",
    invalid_action_penalty: float = 0.1,
    epsilon: float = 1e-6,
) -> CreditBatch:
  """Broadcast one call advantage to that call's active generated tokens.

  Role mode groups call rewards by (task, role), including repeated Solver
  calls. Episode mode first averages call rewards within each attempt, computes
  statistics across attempts of a task, then assigns that attempt's advantage
  to every call. Final task reward is shared; format penalties are per call.
  """
  if normalization not in ("role", "episode"):
    raise ValueError("normalization must be 'role' or 'episode'.")
  if not math.isfinite(epsilon) or epsilon <= 0:
    raise ValueError("epsilon must be finite and positive.")
  if not math.isfinite(invalid_action_penalty) or invalid_action_penalty < 0:
    raise ValueError("invalid_action_penalty must be finite and nonnegative.")
  if not episodes or len({e.episode_id for e in episodes}) != len(episodes):
    raise ValueError("A credit batch needs episodes with unique IDs.")
  calls = [c for e in episodes for c in e.calls]
  if len({c.call_id for c in calls}) != len(calls):
    raise ValueError("Call IDs must be unique across the collection round.")
  if len({(c.policy_id, c.policy_version) for c in calls}) != 1:
    raise ValueError("V0 supports one shared policy and one version per round.")
  groups = collections.defaultdict(list)
  rows = []
  for episode in episodes:
    call_rewards = [
        episode.reward - invalid_action_penalty * (not c.valid_action)
        for c in episode.calls
    ]
    if normalization == "episode":
      # Accumulate in float64 before casting to the score dtype. Otherwise
      # identical rewards with different call counts can gain fake variance.
      episode_mean = float(np.float32(np.mean(call_rewards, dtype=np.float64)))
      groups[(episode.task_id,)].append(episode_mean)
    for call, reward in zip(episode.calls, call_rewards, strict=True):
      key = (
          (episode.task_id, call.role)
          if normalization == "role"
          else (episode.task_id,)
      )
      if normalization == "role":
        groups[key].append(reward)
      rows.append((
          episode,
          call,
          key,
          reward if normalization == "role" else episode_mean,
      ))
  stats = {}
  for key, values in groups.items():
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == 1:
      stats[key] = (0.0, 1.0)
    elif np.all(arr == arr[0]):
      # Upstream float32 reductions can produce nonzero advantages for a
      # constant decimal reward (e.g. eight -0.1 penalties). Preserve the
      # intended zero-variance semantics rather than amplifying roundoff/eps.
      stats[key] = (float(arr[0]), 0.0)
    else:
      stats[key] = (float(arr.mean()), float(arr.std(ddof=1)))
  payloads, rewards, advantages, roles = [], [], [], []
  for episode, call, key, reward in rows:
    mean, std = stats[key]
    advantage = float((np.float32(reward) - mean) / (std + epsilon))
    mask = np.asarray(call.loss_mask, dtype=np.float32)[None, :]
    payloads.append(
        datatypes.RLTrainerPayload(
            prompt_ids=np.asarray(call.prompt_ids, dtype=np.int32)[None, :],
            prompt_mask=np.ones((1, len(call.prompt_ids)), dtype=np.float32),
            completion_ids=np.asarray(call.completion_ids, dtype=np.int32)[
                None, :
            ],
            completion_mask=mask,
            completion_attention_mask=np.ones_like(mask),
            advantages=mask * advantage,
            old_per_token_logps=np.asarray(
                call.rollout_logprobs, dtype=np.float32
            )[None, :],
            metadata={
                "task_id": episode.task_id,
                "episode_id": episode.episode_id,
                "call_id": call.call_id,
                "traj_id": call.call_id,
                "role": call.role,
                "policy_id": call.policy_id,
                "policy_version": call.policy_version,
            },
        )
    )
    rewards.append(reward)
    advantages.append(advantage)
    roles.append(call.role)
  return CreditBatch(
      tuple(payloads), tuple(rewards), tuple(advantages), tuple(roles)
  )
