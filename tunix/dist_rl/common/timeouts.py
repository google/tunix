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

"""Shared timeout defaults and layer-to-layer derivation helpers.

Distributed RL coordinates timeouts across four layers:
  1. Per-episode budget (`episode_timeout_s`) inside `TrajectoryCollectEngine`.
  2. Rollout soft drain window (`drain_timeout_s` = `episode_timeout_s` +
     `DRAIN_SLACK_S`) inside `RolloutManager.pre_weight_sync` before remaining
     stragglers are paused.
  3. Coordinator hard pre-sync deadline (`PhaseTimeouts.pre` >=
     `drain_timeout_s` + `PRE_WEIGHT_SYNC_SLACK_S`) inside
     `WeightSyncCoordinator`.
  4. Worker gRPC deadline (`rpc_timeout_s` >= `PhaseTimeouts.max_rpc_phase_s()`)
     on remote `ActorHandle` calls.

Each layer must contain the one above it. With default values, in the worst
case where an episode starts right as `pre_weight_sync` begins draining:

    0s             600s  660s        900s                     1800s
    |--------------|     :           :                        :
    1. episode_timeout_s :           :                        :
    |--------------------|           :                        :
    2. drain_timeout_s = (1) + DRAIN_SLACK_S                  :
    |--------------------------------|                        :
    3. PhaseTimeouts.pre >= (2) + PRE_WEIGHT_SYNC_SLACK_S     :
    |---------------------------------------------------------|
    4. rpc_timeout_s >= PhaseTimeouts.max_rpc_phase_s()
"""

import math

# Default per-episode collection timeout (10 minutes).
DEFAULT_EPISODE_TIMEOUT_S: float = 600.0

# Best-effort headroom added on top of `episode_timeout_s` for `RolloutManager`
# soft drain. `env.reset`, `final_reward_fn`, and `env.close` run outside the
# per-step episode timer with their own budgets; because `RolloutManager` drain
# is soft, any straggler episodes still in reset/final_reward_fn/close when the
# drain window expires are paused (not lost) while weight sync proceeds.
DRAIN_SLACK_S: float = 60.0

# Headroom added on top of `drain_timeout_s` for `WeightSyncCoordinator`'s hard
# `pre_weight_sync` deadline (sampler flush, pause_all, and RPC serialization).
PRE_WEIGHT_SYNC_SLACK_S: float = 240.0

# Default soft drain timeout (`DEFAULT_EPISODE_TIMEOUT_S + DRAIN_SLACK_S`).
DEFAULT_DRAIN_TIMEOUT_S: float = DEFAULT_EPISODE_TIMEOUT_S + DRAIN_SLACK_S

# Default coordinator hard `pre_weight_sync` timeout (`DEFAULT_DRAIN_TIMEOUT_S +
# PRE_WEIGHT_SYNC_SLACK_S`).
DEFAULT_PRE_WEIGHT_SYNC_TIMEOUT_S: float = (
    DEFAULT_DRAIN_TIMEOUT_S + PRE_WEIGHT_SYNC_SLACK_S
)


def validate_positive_finite(value: float, name: str) -> float:
  """Validates that `value` is a positive finite int/float (excluding bool)."""
  if (
      isinstance(value, bool)
      or not isinstance(value, (int, float))
      or not math.isfinite(value)
      or value <= 0
  ):
    raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
  return float(value)


def drain_timeout_for(episode_timeout_s: float) -> float:
  """Derives the rollout soft drain timeout for a given episode timeout."""
  return (
      validate_positive_finite(episode_timeout_s, "episode_timeout_s")
      + DRAIN_SLACK_S
  )


def pre_weight_sync_timeout_for(drain_timeout_s: float) -> float:
  """Derives the coordinator hard `pre_weight_sync` timeout for a drain window."""
  return (
      validate_positive_finite(drain_timeout_s, "drain_timeout_s")
      + PRE_WEIGHT_SYNC_SLACK_S
  )
