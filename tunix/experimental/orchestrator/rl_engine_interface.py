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

"""The RL engine surface a learning loop drives.

An RL engine is the compute driver for an RL training loop. `RLEngine`
(in-process) and `OrchestratorRLEngine` (worker-backed) are two implementations
of this same `AbstractRLEngine` surface: a learner builds its loop out of these
calls and is agnostic to whether the work runs in-process or is dispatched to
workers. Swapping the implementation turns a single-process run into a
distributed one -- the loop code does not change.

This Protocol maps directly to the current `RLEngine` interface in
`tunix.rl.rl_cluster.RLEngine`, derived from the members the agentic learners
(`AgenticRLLearner` / `GRPOLearner`) actually touch. It is grouped into:

  * compute primitives (generate / train / sync / score),
  * shared bookkeeping (step counter, metrics, tokenizer, teardown),
  * config & topology (cluster_config, role->mesh, rollout config), and
  * sub-engine accessors (rollout, actor_trainer, critic_trainer).

The Protocol is structural: `RLEngine` satisfies it as-is, and
`OrchestratorRLEngine` satisfies it by routing the compute primitives to
workers and delegating the rest. Sub-engines are typed `Any` here (their own
surfaces -- `rollout.pad_id()/eos_id()/model()`,
`actor_trainer.with_loss_fn(...)`, etc. -- are large and
worker-implementation-specific); the members exercised by the loop are noted in
comments.
"""

from typing import Any, Mapping, Protocol, runtime_checkable
from jax.typing import ArrayLike
from tunix.experimental.common import datatypes


@runtime_checkable
class AbstractRLEngine(Protocol):
  """Structural interface for an RL engine (maps to current `RLEngine`)."""

  # --- Shared bookkeeping ---------------------------------------------------
  # Step counter / weight version, read and written by the loop.
  global_steps: int
  # Tokenizer adapter used for chat templating / (de)tokenization.
  tokenizer: Any

  def buffer_metrics(self, metrics: Mapping[str, Any], mode: Any = ...) -> None:
    """Buffers metrics, flushed on the next step boundary."""
    ...

  def buffer_metrics_async(
      self, metrics: Mapping[str, Any], mode: Any = ..., step: int = ...
  ) -> None:
    """Buffers metrics for a specific step (async producer path)."""
    ...

  def close(self) -> None:
    """Releases cluster resources at end of run."""
    ...

  # --- Generation (rollout) -------------------------------------------------
  def generate(
      self,
      prompts: list[str] | list[list[dict[str, str]]],
      apply_chat_template: bool = False,
      mode: Any = None,
      micro_batch_size: int | None = None,
      trace_tags: Mapping[str, Any] | None = None,
      max_generation_steps: int | None = None,
  ) -> Any:
    """Generates completions for `prompts` (returns a `RolloutOutput`)."""
    ...

  # --- Training (train_step) ------------------------------------------------
  def train(
      self,
      role: datatypes.Role,
      train_ds: Any,
      eval_ds: Any,
      skip_jit: bool = False,
  ) -> None:
    """Runs a training update for the specified role (e.g. Role.ACTOR, Role.CRITIC)."""
    ...

  def eval_actor(self, eval_ds: Any) -> Any:
    """Runs an explicit actor evaluation phase over eval micro-batches."""
    ...

  # --- Scoring (feeds advantage / IS math) ----------------------------------
  def per_token_logps(
      self,
      role: datatypes.Role,
      prompt_tokens: ArrayLike,
      completion_tokens: ArrayLike,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
      segment_ids: ArrayLike | None = None,
      **kwargs: Any,
  ) -> Any:
    """Per-token logprobs under the specified model role."""
    # TODO(noghabi): add per batch interface (replace or keep both). that is
    # the interface worker uses.
    ...

  # --- Weight sync ----------------------------------------------------------
  def sync_weights(self) -> None:
    """Publishes the trainer's weights to the rollout/inference replicas."""
    ...

  # --- Config & topology ----------------------------------------------------
  # ClusterConfig: training_config (+ micro-batch sizes, chunk sizes, metrics
  # options), rollout_config, rollout_engine, role_to_mesh.
  cluster_config: Any
  # Role -> Mesh mapping (aka cluster_config.role_to_mesh); `.devices`/`.empty`.
  r2m: Any

  def get_rollout_config(self, mode: Any) -> Any:
    """Returns the effective rollout config for the given mode."""
    ...

  # --- Sub-engine accessors (surfaces the loop reaches into) -----------------
  # rollout: `.pad_id()`, `.eos_id()`, `.model()`.
  rollout: Any
  # actor_trainer: `.with_loss_fn(...)`, `.with_gen_model_input_fn(...)`,
  # `.with_rl_metrics_to_log(...)`, `.is_managed_externally`,
  # `.restored_global_step()`, `.iter_steps`, `.train_steps`, `.model`.
  actor_trainer: Any
  # critic_trainer: present only for actor-critic algorithms (PPO). Guard with
  # hasattr(engine, "critic_trainer") before use.
