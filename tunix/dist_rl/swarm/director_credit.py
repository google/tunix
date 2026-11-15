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

"""R²Orch Director credit, independently implemented from its specification.

Reference: beita6969/R2Orch, director_update.py at 12ceb75. Center rewards
within (task, scaffold condition), select low-signal episodes deterministically,
then whiten over active Director tokens with sample variance. Worker calls are
execution evidence, never training rows. This is not Dr. MAS role normalization.
"""

import collections
from collections.abc import Sequence
import dataclasses
import hashlib
import math

import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.swarm import credit
from tunix.dist_rl.swarm import policy
from tunix.dist_rl.swarm import trace


@dataclasses.dataclass(frozen=True, kw_only=True)
class DirectorCreditConfig:
  whiten_eps: float = 1e-8
  low_signal_abs_advantage: float = 0.05
  low_signal_keep_fraction: float = 0.15
  cost_lambda: float = 0.005
  reference_stride: int = 1

  def __post_init__(self):
    if not all(math.isfinite(x) for x in dataclasses.astuple(self)):
      raise ValueError("Director credit settings must be finite.")
    if (
        self.whiten_eps <= 0
        or self.low_signal_abs_advantage < 0
        or not 0 <= self.low_signal_keep_fraction <= 1
        or self.cost_lambda < 0
        or type(self.reference_stride) is not int
        or self.reference_stride < 1
    ):
      raise ValueError("Invalid Director credit settings.")


def build_director_credit(
    episodes: Sequence[trace.SwarmEpisode],
    *,
    plan: policy.LearningPlan,
    config: DirectorCreditConfig = DirectorCreditConfig(),
    reference_step: int = 0,
) -> credit.CreditBatch:
  """Build detached Director-only payloads from complete episode groups.

  The episode reward is raw task correctness; worker_cost is normalized by the
  recipe, not a raw token count. A singleton condition cannot provide a relative
  outcome and contributes no training row. Constant rewards yield exactly zero
  advantages. KL anchors may still be optimized by the program in that case.
  """
  plan.validate(episodes)
  grouped = collections.defaultdict(list)
  rewards = {}
  calls = {}
  for episode in episodes:
    grouped[(episode.task_id, episode.scaffold_condition)].append(episode)
    rewards[episode.episode_id] = (
        episode.reward - config.cost_lambda * episode.worker_cost
    )
    calls[episode.episode_id] = plan.select_calls(episode)
  centered = {}
  for group in grouped.values():
    if len(group) < 2:
      continue
    values = [rewards[e.episode_id] for e in group]
    mean = math.fsum(values) / len(values)
    for episode in group:
      centered[episode.episode_id] = rewards[episode.episode_id] - mean
  selected = []
  for episode_id, value in centered.items():
    digest = hashlib.sha256(f"{episode_id}|low_signal".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if (
        abs(value) >= config.low_signal_abs_advantage
        or fraction < config.low_signal_keep_fraction
    ):
      selected.append(episode_id)
  # Upstream retains all centered samples if the filter would empty the batch.
  selected = selected or list(centered)
  counts = {eid: sum(sum(c.loss_mask) for c in calls[eid]) for eid in selected}
  tokens = sum(counts.values())
  if tokens:
    mean = math.fsum(centered[eid] * counts[eid] for eid in selected) / tokens
    variance = math.fsum(
        (centered[eid] - mean) ** 2 * counts[eid] for eid in selected
    ) / max(tokens - 1, 1)
    scale = 1.0 / math.sqrt(variance + config.whiten_eps)
    advantages = {eid: (centered[eid] - mean) * scale for eid in selected}
  else:
    advantages = {}
  payloads, row_rewards, row_advantages, roles = [], [], [], []
  reference_seen = collections.Counter()
  for episode in episodes:
    eid = episode.episode_id
    if eid not in advantages:
      continue
    condition = episode.scaffold_condition
    kl_rank = reference_seen[condition] + reference_step
    reference_seen[condition] += 1
    kl_weight = float(
        config.reference_stride if kl_rank % config.reference_stride == 0 else 0
    )
    for call in calls[eid]:
      mask = np.asarray(call.loss_mask, dtype=np.float32)[None, :]
      advantage = advantages[eid]
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
              reference_kl_weights=mask * kl_weight,
              old_per_token_logps=np.asarray(
                  call.rollout_logprobs, dtype=np.float32
              )[None, :],
              action_support_ids=(
                  None
                  if call.action_support_ids is None
                  else np.asarray(call.action_support_ids, np.int32)[None, :]
              ),
              metadata={
                  "task_id": episode.task_id,
                  "episode_id": eid,
                  "call_id": call.call_id,
                  "traj_id": call.call_id,
                  "role": call.role,
                  "policy_id": call.policy_id,
                  "policy_version": call.policy_version,
                  "scaffold_condition": episode.scaffold_condition,
                  "centered_advantage": centered[eid],
                  "worker_cost": episode.worker_cost,
              },
          )
      )
      row_rewards.append(rewards[eid])
      row_advantages.append(advantage)
      roles.append(call.role)
  return credit.CreditBatch(
      tuple(payloads), tuple(row_rewards), tuple(row_advantages), tuple(roles)
  )
