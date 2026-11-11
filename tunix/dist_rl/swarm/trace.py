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

"""Immutable, exact per-call records; a role is not a policy identity."""

import dataclasses
import math


@dataclasses.dataclass(frozen=True, kw_only=True)
class CallTrace:
  """One sampled response with the exact context used to generate it.

  Tuples deliberately prevent mutation after collection. Tokens with loss_mask
  zero remain visible as attention context. Padding is introduced only later by
  the batch assembler; an EOS token equal to pad_id is still an active token.
  """

  call_id: str
  role: str
  policy_id: str
  policy_version: int
  prompt_ids: tuple[int, ...]
  completion_ids: tuple[int, ...]
  rollout_logprobs: tuple[float, ...]
  loss_mask: tuple[int, ...]
  text: str
  valid_action: bool = True

  def __post_init__(self):
    for name in (
        "prompt_ids",
        "completion_ids",
        "rollout_logprobs",
        "loss_mask",
    ):
      object.__setattr__(self, name, tuple(getattr(self, name)))
    if not self.call_id or not self.role or not self.policy_id:
      raise ValueError("A call needs a call_id, role and policy_id.")
    if (
        self.policy_version < 0
        or not self.prompt_ids
        or not self.completion_ids
    ):
      raise ValueError("A call needs nonempty token sequences and a version.")
    if any(x < 0 or int(x) != x for x in self.prompt_ids + self.completion_ids):
      raise ValueError("Token IDs must be nonnegative integers.")
    if not (
        len(self.completion_ids)
        == len(self.rollout_logprobs)
        == len(self.loss_mask)
    ):
      raise ValueError("Completion IDs, logprobs and loss mask must align.")
    if any(x not in (0, 1) for x in self.loss_mask) or not any(self.loss_mask):
      raise ValueError("Loss mask must be binary and contain an active token.")
    if any(not math.isfinite(x) or x > 1e-5 for x in self.rollout_logprobs):
      raise ValueError("Rollout logprobs must be finite log probabilities.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class SwarmEpisode:
  """One complete team attempt; task_id identifies its GRPO comparison group."""

  task_id: str
  episode_id: str
  calls: tuple[CallTrace, ...]
  reward: float
  final_text: str

  def __post_init__(self):
    object.__setattr__(self, "calls", tuple(self.calls))
    if not self.task_id or not self.episode_id or not self.calls:
      raise ValueError("An episode needs IDs and at least one recorded call.")
    if not math.isfinite(self.reward):
      raise ValueError("Episode reward must be finite.")
    if len({c.call_id for c in self.calls}) != len(self.calls):
      raise ValueError("Call IDs must be unique within an episode.")
    if len({c.policy_version for c in self.calls}) != 1:
      raise ValueError("A synchronous episode cannot mix policy versions.")
