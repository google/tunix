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

"""Algorithm-specific hooks for the RL orchestrator (the pluggable seam).

`RLOrchestrator` (the primitive API a learner loop is built on) is deliberately
algorithm-agnostic: generation, training, weight sync, and scoring are the same
regardless of algorithm. The pieces that genuinely differ between algorithms --
how rewards become advantages, and (later) how a group is postprocessed into a
train example and what the loss is -- live behind this adapter, so one
orchestrator and one loop can serve GRPO, PPO, and future algorithms by swapping
the adapter.
"""

from typing import Any, Protocol, runtime_checkable

from tunix.rl import function_registry


@runtime_checkable
class AlgorithmAdapter(Protocol):
  """The algorithm-specific bits an otherwise-generic RL loop needs.

  v0 covers advantage estimation; postprocess/assembly and loss-spec hooks are
  added here as the loop is promoted to Layer 2.
  """

  def compute_advantages(self, rewards: Any, *, num_generations: int) -> Any:
    """Turns per-completion rewards into per-completion advantages."""
    ...


class GRPOAdapter:
  """GRPO advantages, reusing the shared advantage-estimator registry.

  This does not reimplement the group-relative math; it dispatches to the same
  estimator the agentic GRPO learner uses, so the orchestrator and the legacy
  learner stay numerically identical.
  """

  def __init__(self, advantage_estimator: str = "grpo"):
    self._advantage_estimator = advantage_estimator

  def compute_advantages(self, rewards: Any, *, num_generations: int) -> Any:
    estimator = function_registry.get_advantage_estimator(
        self._advantage_estimator
    )
    return estimator(rewards=rewards, num_generations=num_generations)
