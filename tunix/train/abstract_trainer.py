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

"""Abstract step-level trainer API.

Defines the pure ML algorithmic core of a trainer. Implementations know
nothing about orchestrators, RPCs, or networking, and operate entirely on
local JAX/Flax data structures. Loop-level concerns (data iteration, eval
cadence, checkpoint policy, hooks, metrics logging, progress bars) belong to
a separate loop/driver layer built on top of this API.
"""

import abc
import dataclasses
from typing import Any, Dict, Optional

import flax
from flax import nnx
import jax


@flax.struct.dataclass(frozen=True)
class TrainerPayload:
  """A training batch for a single step.

  The trainer applies its configured input transform (e.g.
  `gen_model_input_fn`) inside the jitted step so the transform stays traced,
  and applies mesh sharding internally, since it owns the mesh.

  Attributes:
    inputs: The batch pytree; after the trainer's input transform it becomes
      the keyword arguments of the loss function.
    metadata: Optional step tags (e.g. for perf tracing). Not traced.
  """

  inputs: Any
  metadata: Optional[Dict[str, Any]] = flax.struct.field(
      pytree_node=False, default=None
  )


@dataclasses.dataclass(slots=True, kw_only=True)
class StepMetrics:
  """Per-step result.

  Values stay as device arrays; callers decide when to sync to host so that
  inflight computation overlap is preserved.

  Attributes:
    loss: The step loss.
    grad_norm: Global gradient norm. None for eval steps.
    aux: Auxiliary output of the loss function, if any.
    additional: Extra named metrics.
  """

  loss: jax.Array
  grad_norm: jax.Array | None = None
  aux: Any = None
  additional: Dict[str, jax.Array] = dataclasses.field(default_factory=dict)


class AbstractTrainer(abc.ABC):
  """The pure ML algorithmic core of a trainer.

  Step-level only: no training loops, no I/O policy, no orchestration.
  """

  @abc.abstractmethod
  def init_state(self) -> None:
    """Initializes optimizer state, sharding, and jitted step functions.

    Idempotent. Does NOT restore checkpoints; callers do that explicitly via
    `restore_checkpoint`.
    """

  @abc.abstractmethod
  def train_step(
      self,
      payload: TrainerPayload,
      *,
      apply_gradients: bool = True,
      **kwargs,
  ) -> StepMetrics:
    """Executes one forward/backward pass. Updates internal state.

    Gradient accumulation is caller-driven: when `apply_gradients` is False,
    gradients are accumulated internally and no optimizer update happens;
    when True, the accumulated (mean) gradients are applied and the
    accumulation buffer is reset. `global_step` increments only when
    gradients are applied.

    Args:
      payload: The model-ready batch.
      apply_gradients: Whether to apply (vs. accumulate) gradients.
      **kwargs: Implementation-specific options.

    Returns:
      Metrics for this step.
    """

  @abc.abstractmethod
  def eval_step(self, payload: TrainerPayload, **kwargs) -> StepMetrics:
    """Executes a forward-only evaluation step.

    Must not mutate any trainer state, including gradient accumulation
    buffers.

    Args:
      payload: The model-ready batch.
      **kwargs: Implementation-specific options.

    Returns:
      Metrics for this step (`grad_norm` is None).
    """

  @abc.abstractmethod
  def save_checkpoint(self, path: Optional[str] = None, **kwargs) -> str:
    """Serializes the current model and optimizer state now.

    Checkpoint cadence/policy is the caller's responsibility.

    Args:
      path: Destination. If None, the implementation's configured default
        location is used.
      **kwargs: Implementation-specific options.

    Returns:
      The path the checkpoint was written to.
    """

  @abc.abstractmethod
  def restore_checkpoint(self, path: str, **kwargs) -> int:
    """Restores model and optimizer state from disk.

    Args:
      path: The checkpoint to restore.
      **kwargs: Implementation-specific options.

    Returns:
      The restored global step.
    """

  @abc.abstractmethod
  def get_weights(
      self,
      *,
      gather: bool = False,
      full_params: bool = False,
      **kwargs,
  ) -> nnx.State:
    """Returns current model weights (e.g., for weight syncing).

    Args:
      gather: If True, fully gather weights; otherwise return them sharded
        with sharding intact.
      full_params: If True, include/merge base weights; otherwise return only
        trainable params (e.g., LoRA-only when LoRA is enabled).
      **kwargs: Implementation-specific options (e.g., output format).

    Returns:
      The model weights as an nnx.State.
    """

  @property
  @abc.abstractmethod
  def global_step(self) -> int:
    """Number of optimizer updates applied (the model's version number).

    Increments only when `train_step` applies gradients, not per micro-batch.
    """

  def close(self) -> None:
    """Releases resources held by the trainer. Default: no-op."""
