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

"""The RL primitive API a learner loop is built on ("thinker" layer).

`RLOrchestrator` sits above the cluster (`AbstractRLCluster`: in-process
`RLCluster` or worker-backed `OrchestratorRLCluster`) and below the learner loop.
It exposes a small, algorithm-agnostic set of primitives -- generate, train_step,
sync_weights, reference/actor scoring -- plus algorithm-parameterized advantage
estimation via an injected `AlgorithmAdapter`. A learner (Layer 3) composes these
primitives into a training loop and stays agnostic to:

  * where compute runs (single process vs distributed) -- that is the cluster's
    choice, injected here;
  * which algorithm is running (GRPO/PPO/...) -- that is the adapter's choice.

The compute primitives are thin pass-throughs to the cluster today; as the loop
is promoted, more algorithm-agnostic building blocks (train-example assembly,
grouping, metrics/checkpoint cadence) move onto this class so learner loops stay
thin and pluggable.
"""

from typing import Any, Mapping

from tunix.experimental.orchestrator import algorithm_adapter as algorithm_adapter_lib
from tunix.rl import rl_cluster as rl_cluster_lib


class RLOrchestrator:
  """Algorithm-agnostic RL primitives over a cluster + an algorithm adapter."""

  def __init__(
      self,
      cluster: Any,
      algorithm: algorithm_adapter_lib.AlgorithmAdapter,
  ):
    """Initializes the orchestrator.

    Args:
      cluster: An `AbstractRLCluster` (in-process `RLCluster` or worker-backed
        `OrchestratorRLCluster`). Determines where each primitive runs.
      algorithm: The `AlgorithmAdapter` carrying the algorithm-specific bits
        (advantage estimation today).
    """
    self._cluster = cluster
    self._algorithm = algorithm

  # --- Generation (rollout) -------------------------------------------------

  def generate(
      self,
      prompts: Any,
      apply_chat_template: bool = False,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
      micro_batch_size: int | None = None,
      trace_tags: Mapping[str, Any] | None = None,
      max_generation_steps: int | None = None,
  ) -> Any:
    return self._cluster.generate(
        prompts,
        apply_chat_template,
        mode,
        micro_batch_size,
        trace_tags,
        max_generation_steps,
    )

  # --- Training (train_step) ------------------------------------------------

  def train_step(
      self, batch: Any, eval_ds: Any = None, skip_jit: bool = False
  ) -> None:
    """Runs one actor trainer pass over the (chunked) micro-batch."""
    self._cluster.update_actor(batch, eval_ds, skip_jit)

  # --- Weight sync ----------------------------------------------------------

  def sync_weights(self) -> None:
    self._cluster.sync_weights()

  # --- Scoring --------------------------------------------------------------

  def reference_logps(
      self,
      prompt_tokens: Any,
      completion_tokens: Any,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
  ) -> Any:
    return self._cluster.get_ref_per_token_logps(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=pad_id,
        eos_id=eos_id,
        micro_batch_size=micro_batch_size,
    )

  def actor_logps(
      self,
      prompt_tokens: Any,
      completion_tokens: Any,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
  ) -> Any:
    return self._cluster.get_actor_per_token_logps(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=pad_id,
        eos_id=eos_id,
        micro_batch_size=micro_batch_size,
    )

  # --- Advantage estimation (algorithm-parameterized) -----------------------

  def compute_advantages(self, rewards: Any, *, num_generations: int) -> Any:
    return self._algorithm.compute_advantages(
        rewards, num_generations=num_generations
    )

  # --- Shared state ---------------------------------------------------------

  @property
  def global_steps(self) -> int:
    return self._cluster.global_steps

  @global_steps.setter
  def global_steps(self, value: int) -> None:
    self._cluster.global_steps = value

  @property
  def cluster(self) -> Any:
    return self._cluster

  @property
  def algorithm(self) -> algorithm_adapter_lib.AlgorithmAdapter:
    return self._algorithm
