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

"""Collective-return and eligibility hooks for shared-policy RLCR.

The Avalon-M1 paper, sections 2.2.1--2.2.2, specifies CISPO cap 5, dropping
zero-variance groups, flat collective advantages and optional penalties/masks.
It does not publish the advantage standard-deviation correction or epsilon.
Here they are explicitly sample standard deviation and an additive epsilon.
The native CISPO loss is registered separately by `tunix.rl.cispo`.
"""

import collections
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import math

import numpy as np
from tunix.dist_rl.swarm import credit
from tunix.dist_rl.swarm import trace


@dataclasses.dataclass(frozen=True, kw_only=True)
class RLCRConfig:
  """Environment-specific reward choices; validation defaults add no shaping."""

  parse_error_penalty: float = 0.0
  execution_error_penalty: float = 0.0
  underhire_penalty: float = 0.0
  mask_zero_environment_return: bool = False
  mask_repetitions: bool = False
  mask_misplaced_tool_calls: bool = False
  epsilon: float = 1e-6

  def __post_init__(self):
    for name in (
        "parse_error_penalty",
        "execution_error_penalty",
        "underhire_penalty",
    ):
      value = getattr(self, name)
      if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    if not math.isfinite(self.epsilon) or self.epsilon <= 0:
      raise ValueError("epsilon must be finite and positive.")
    for value in (
        self.mask_zero_environment_return,
        self.mask_repetitions,
        self.mask_misplaced_tool_calls,
    ):
      if type(value) is not bool:
        raise ValueError("Eligibility switches must be boolean.")

  @classmethod
  def avalon(cls, *, underhire_penalty: float, signed_zero_mask: bool = False):
    """Source-described hooks, with the unspecified environment weight supplied.

    `signed_zero_mask` interprets zero reward as zero net environment return.
    The source does not disambiguate cancellation of positive/negative events;
    use an explicit episode eligibility callback for a different convention.
    """
    return cls(
        parse_error_penalty=0.1,
        execution_error_penalty=0.05,
        underhire_penalty=underhire_penalty,
        mask_zero_environment_return=signed_zero_mask,
        mask_repetitions=True,
        mask_misplaced_tool_calls=True,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class CallDiagnostics:
  agent_id: str
  malformed_tool: bool = False
  execution_error: bool = False
  misplaced_tool: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True)
class EpisodeDiagnostics:
  calls: Mapping[str, CallDiagnostics] = dataclasses.field(default_factory=dict)
  # Specialist budget/count exclude the initial CEO. For a harness supporting
  # firing/rehiring, the environment must state how it measures utilisation.
  specialist_budget: int = 0
  specialists_hired: int = 0

  def __post_init__(self):
    for value in (self.specialist_budget, self.specialists_hired):
      if type(value) is not int or value < 0:
        raise ValueError("Organisation counts must be nonnegative integers.")


def has_tandem_repeat(tokens: Sequence[int]) -> bool:
  """Detect a <=32-token unit repeated >=20 times covering >=300 tokens.

  Exact contiguous repetitions are measured over sampled active tokens from one
  agent, excluding observations. At most 32 vector comparisons are required.
  """
  values = np.asarray(tokens, dtype=np.int64)
  for width in range(1, min(32, len(values)) + 1):
    matches = values[width:] == values[:-width]
    if not np.any(matches):
      continue
    padded = np.r_[False, matches, False].astype(np.int8)
    changes = np.flatnonzero(np.diff(padded))
    longest_run = int(np.max(changes[1::2] - changes[::2], initial=0))
    copies = (longest_run + width) // width
    if copies >= 20 and copies * width >= 300:
      return True
  return False


def collective_return(
    episode: trace.SwarmEpisode,
    diagnostics: EpisodeDiagnostics,
    config: RLCRConfig,
) -> float:
  """Add declared error-token and underhire penalties to environment return.

  Set diagnostics only on the affected assistant turn. Each penalty is counted
  once per sampled active token and summed into the team's undiscounted return;
  the resulting team advantage is broadcast to every eligible agent token.
  """
  result = episode.reward
  for call in episode.calls:
    info = diagnostics.calls.get(call.call_id)
    if info is None:
      continue
    count = sum(call.loss_mask)
    result -= count * (
        config.parse_error_penalty * info.malformed_tool
        + config.execution_error_penalty * info.execution_error
    )
  result -= config.underhire_penalty * max(
      0, diagnostics.specialist_budget - diagnostics.specialists_hired
  )
  if not math.isfinite(result):
    raise ValueError("The shaped collective return must be finite.")
  return float(result)


def build_rlcr_credit(
    episodes: Sequence[trace.SwarmEpisode],
    *,
    diagnostics: Mapping[str, EpisodeDiagnostics] | None = None,
    config: RLCRConfig = RLCRConfig(),
    token_eligibility: Callable[[trace.CallTrace], Sequence[int]] | None = None,
    episode_eligibility: Callable[[trace.SwarmEpisode], bool] | None = None,
) -> credit.CreditBatch:
  """Normalize complete task groups, then apply explicit loss eligibility.

  A masked attempt remains in the reward-normalization comparison population;
  it contributes no loss tokens. This convention is explicit because the paper
  does not specify whether its eligibility masks alter the comparison moments.
  Empty/constant groups produce no payloads or optimizer weight. Received
  messages and tool observations are prompt context, never action tokens.
  """
  if not episodes:
    return credit.CreditBatch((), (), (), ())
  diagnostics = diagnostics or {}
  original = {e.episode_id: e for e in episodes}
  if len(original) != len(episodes):
    raise ValueError("Episode IDs must be unique.")
  calls = [c for e in episodes for c in e.calls]
  if len({c.call_id for c in calls}) != len(calls):
    raise ValueError("Call IDs must be unique across the round.")
  if len({(c.policy_id, c.policy_version) for c in calls}) != 1:
    raise ValueError("RLCR requires one shared policy snapshot.")
  groups = collections.defaultdict(list)
  for episode in episodes:
    info = diagnostics.get(episode.episode_id, EpisodeDiagnostics())
    groups[episode.task_id].append(
        dataclasses.replace(
            episode, reward=collective_return(episode, info, config)
        )
    )
  kept = []
  for group in groups.values():
    rewards = np.asarray([e.reward for e in group], dtype=np.float32)
    if not np.all(np.isfinite(rewards)):
      raise ValueError("Collective rewards must be representable in float32.")
    if not np.all(rewards == rewards[0]):
      kept.extend(group)
  if not kept:
    return credit.CreditBatch((), (), (), ())
  batch = credit.build_credit(
      kept,
      normalization="episode",
      invalid_action_penalty=0,
      epsilon=config.epsilon,
  )
  eligibility = {}
  for episode in kept:
    info = diagnostics.get(episode.episode_id, EpisodeDiagnostics())
    active_by_agent = collections.defaultdict(list)
    if config.mask_repetitions:
      for call in episode.calls:
        if (
            call.call_id not in info.calls
            or not info.calls[call.call_id].agent_id
        ):
          raise ValueError("Repetition masking requires every call's agent ID.")
        active_by_agent[info.calls[call.call_id].agent_id].extend(
            token
            for token, active in zip(
                call.completion_ids, call.loss_mask, strict=True
            )
            if active
        )
    repeated = {
        a for a, tokens in active_by_agent.items() if has_tandem_repeat(tokens)
    }
    env_episode = original[episode.episode_id]
    episode_allowed = not (
        config.mask_zero_environment_return and env_episode.reward == 0
    ) and (episode_eligibility is None or episode_eligibility(env_episode))
    for call in episode.calls:
      mask = np.asarray(call.loss_mask, dtype=np.float32).copy()
      call_info = info.calls.get(call.call_id)
      if not episode_allowed or (
          call_info is not None
          and (
              call_info.agent_id in repeated
              or (config.mask_misplaced_tool_calls and call_info.misplaced_tool)
          )
      ):
        mask[:] = 0
      if token_eligibility is not None:
        extra = np.asarray(token_eligibility(call))
        if extra.shape != mask.shape or not np.all((extra == 0) | (extra == 1)):
          raise ValueError("Token eligibility must be an aligned binary mask.")
        mask *= extra
      eligibility[call.call_id] = mask[None, :]
  payloads, rewards, advantages, roles = [], [], [], []
  for payload, reward, advantage, role in zip(
      batch.payloads, batch.rewards, batch.advantages, batch.roles, strict=True
  ):
    mask = eligibility[payload.metadata["call_id"]]
    if not np.any(mask):
      continue
    # Keep every sampled token as attention context, including excluded tokens.
    payloads.append(
        dataclasses.replace(
            payload, completion_mask=mask, advantages=mask * advantage
        )
    )
    rewards.append(reward)
    advantages.append(advantage)
    roles.append(role)
  return credit.CreditBatch(
      tuple(payloads), tuple(rewards), tuple(advantages), tuple(roles)
  )
