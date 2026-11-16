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


def _validate_alignment(call: "CallTrace") -> None:
  """Validates loss mask and logprob invariants for trainable or frozen calls."""
  n_tokens = len(call.completion_ids)
  if len(call.loss_mask) != n_tokens:
    raise ValueError("Completion IDs and loss mask must align.")
  if call.trainable:
    if len(call.rollout_logprobs) != n_tokens:
      raise ValueError("Completion IDs, logprobs and loss mask must align.")
    if any(x not in (0, 1) for x in call.loss_mask) or not any(call.loss_mask):
      raise ValueError("Loss mask must be binary and contain an active token.")
  else:
    if any(x != 0 for x in call.loss_mask):
      raise ValueError("Frozen calls must use an all-zero loss mask.")
    if call.rollout_logprobs and len(call.rollout_logprobs) != n_tokens:
      raise ValueError("Recorded frozen logprobs must align with completion IDs.")
  if any(not math.isfinite(x) or x > 1e-5 for x in call.rollout_logprobs):
    raise ValueError("Rollout logprobs must be finite log probabilities.")


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
  trainable: bool = True

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
    _validate_alignment(self)


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
    versions_by_policy: dict[str, set[int]] = {}
    for call in self.calls:
      versions_by_policy.setdefault(call.policy_id, set()).add(
          call.policy_version
      )
    if any(len(vs) != 1 for vs in versions_by_policy.values()):
      raise ValueError("A synchronous episode cannot mix policy versions.")
