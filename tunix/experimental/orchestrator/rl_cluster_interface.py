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

"""The primitive-API surface an RL learning loop drives.

`RLCluster` (in-process) and `OrchestratorRLCluster` (worker-backed) are two
implementations of the same surface: a learner (`AgenticRLLearner` and friends)
builds its loop entirely out of these calls and is agnostic to whether the work
runs in-process or is dispatched to remote workers. Swapping the implementation
is what turns a single-process run into a distributed one -- the loop code does
not change.

This Protocol documents the load-bearing subset of that surface -- the compute
primitives plus the shared bookkeeping the loop reads/writes. It is intentionally
structural: `RLCluster` already satisfies it without inheriting anything, and
`OrchestratorRLCluster` satisfies it by routing the primitives to workers and
delegating the rest. Concrete clusters expose more than this (e.g. `rollout`,
`actor_trainer`, `cluster_config`, `perf_v2`); those accessors are provided by
both implementations but are not part of the minimal contract enumerated here.
"""

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class AbstractRLCluster(Protocol):
  """Structural interface for the RL primitives a learning loop drives."""

  # --- Shared bookkeeping ---------------------------------------------------
  global_steps: int  # step counter / weight version, read and written by the loop.

  def buffer_metrics(self, metrics: Mapping[str, Any], **kwargs) -> None:
    """Buffers metrics to be flushed on the next step boundary."""
    ...

  # --- Generation (rollout) -------------------------------------------------
  def generate(
      self,
      prompts: Any,
      apply_chat_template: bool = False,
      mode: Any = None,
      micro_batch_size: int | None = None,
      trace_tags: Mapping[str, Any] | None = None,
      max_generation_steps: int | None = None,
  ) -> Any:
    """Generates completions for `prompts` (returns a `RolloutOutput`)."""
    ...

  # --- Training (train_step) ------------------------------------------------
  def update_actor(self, train_ds: Any, eval_ds: Any, skip_jit: bool = False) -> None:
    """Runs the actor trainer over the (chunked) micro-batch."""
    ...

  def update_critic(self, train_ds: Any, eval_ds: Any, skip_jit: bool = False) -> None:
    """Runs the critic trainer (PPO); a no-op for critic-free algorithms."""
    ...

  # --- Scoring (feeds advantage / IS math) ----------------------------------
  def get_ref_per_token_logps(
      self,
      prompt_tokens: Any,
      completion_tokens: Any,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
  ) -> Any:
    """Per-token logprobs under the frozen reference model."""
    ...

  def get_actor_per_token_logps(
      self,
      prompt_tokens: Any,
      completion_tokens: Any,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
  ) -> Any:
    """Per-token logprobs under the (trainable) actor model."""
    ...

  # --- Weight sync ----------------------------------------------------------
  def sync_weights(self) -> None:
    """Publishes the trainer's weights to the rollout/inference replicas."""
    ...
