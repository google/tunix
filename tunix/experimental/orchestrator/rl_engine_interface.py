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
(in-process) and `DistributedRLEngine` (worker-backed) are two implementations
of this same `AbstractRLEngine` surface: a learner builds its loop out of these
calls and is agnostic to whether the work runs in-process or is dispatched to
workers. Swapping the implementation turns a single-process run into a
distributed one -- the loop code does not change.

This Protocol defines the core compute primitives (`generate`, `train`,
`train_step`, `per_token_logps`, `sync_weights`) required by learning algorithms
(`RLDriver`, `AgenticRLLearner`, `GRPOLearner`).

The Protocol is structural and purely stateless: `DistributedRLEngine` satisfies
it by routing compute calls across distributed workers without carrying
bookkeeping or topology state.
"""

from typing import Any, Mapping, Protocol, runtime_checkable
from jax.typing import ArrayLike
from tunix.experimental.common import datatypes


@runtime_checkable
class AbstractRLEngine(Protocol):
  """Structural interface for an RL engine (stateless compute primitives)."""

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
    """Runs a training update for the specified role (e.g.

    Role.ACTOR, Role.CRITIC).
    """
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
  def sync_weights(self) -> Any:
    """Publishes the trainer's weights to the rollout/inference replicas."""
    ...
