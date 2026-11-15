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

"""Policy identity and update ownership for one synchronous learner round."""

from collections.abc import Sequence
import dataclasses

from tunix.dist_rl.swarm import trace


@dataclasses.dataclass(frozen=True, kw_only=True)
class PolicyBinding:
  """A learner uses its published version; frozen policies pin their version.

  Model routing lives in the runtime. This contract records which exact policy
  may contribute training rows, independently of its logical role name.
  """

  policy_id: str
  trainable: bool
  version: int | None = None

  def __post_init__(self):
    if not self.policy_id:
      raise ValueError("A policy binding needs a policy_id.")
    if self.trainable and self.version is not None:
      raise ValueError("The learner version comes from its published snapshot.")
    if not self.trainable and (self.version is None or self.version < 0):
      raise ValueError("A frozen policy requires a pinned nonnegative version.")


@dataclasses.dataclass(frozen=True, kw_only=True)
class LearningPlan:
  """One trainable policy and any number of frozen policies per update.

  Separate learners can use separate plans. This deliberately does not claim
  atomic multi-optimizer publication or simultaneous multi-policy training.
  """

  bindings: tuple[PolicyBinding, ...]
  learner_policy_id: str

  def __post_init__(self):
    object.__setattr__(self, "bindings", tuple(self.bindings))
    if len({b.policy_id for b in self.bindings}) != len(self.bindings):
      raise ValueError("Policy bindings must have unique policy IDs.")
    learners = [b.policy_id for b in self.bindings if b.trainable]
    if learners != [self.learner_policy_id]:
      raise ValueError("A learning plan must identify exactly one learner.")

  def select_calls(
      self, episode: trace.SwarmEpisode
  ) -> tuple[trace.CallTrace, ...]:
    return tuple(
        c for c in episode.calls if c.policy_id == self.learner_policy_id
    )

  def validate(
      self,
      episodes: Sequence[trace.SwarmEpisode],
      learner_version: int | None = None,
  ) -> None:
    """Reject unknown, stale, mixed, duplicate, or untrainable traces."""
    if not episodes or len({e.episode_id for e in episodes}) != len(episodes):
      raise ValueError("A learning round needs unique episode IDs.")
    bindings = {b.policy_id: b for b in self.bindings}
    call_ids = set()
    versions = set()
    for episode in episodes:
      if not self.select_calls(episode):
        raise ValueError("Every episode must contain a learner call.")
      for call in episode.calls:
        if call.call_id in call_ids:
          raise ValueError("Call IDs must be unique across the round.")
        call_ids.add(call.call_id)
        if call.policy_id not in bindings:
          raise ValueError(f"Unknown policy: {call.policy_id}.")
        binding = bindings[call.policy_id]
        if binding.trainable:
          versions.add(call.policy_version)
          if (
              learner_version is not None
              and call.policy_version != learner_version
          ):
            raise ValueError(
                "Collection returned a stale learner policy version."
            )
        elif call.policy_version != binding.version:
          raise ValueError("Collection changed a frozen policy version.")
    if len(versions) != 1:
      raise ValueError("A learning round cannot mix learner policy versions.")
