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
from typing import Any, Dict, List, Optional, Tuple

import flax
from flax import nnx
import jax
import jax.numpy as jnp


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


@flax.struct.dataclass
class WeightedMetrics:
  """A metric that requires weighted reduction.

  Attributes:
    unreduced_sum: The sum of the metric values. Should be a scalar ().
    denominator: The weight or count of valid tokens/examples. Should be a
      scalar ().
    eps: Optional epsilon added to denominator for numerical stability.
    min_denom: Optional minimum bound for the denominator.
  """

  unreduced_sum: jax.Array
  denominator: jax.Array
  eps: float | None = flax.struct.field(default=None, pytree_node=False)
  min_denom: float | None = flax.struct.field(default=None, pytree_node=False)

  def compute_scale(self) -> jax.Array:
    """Safely computes the scale factor (1 / denominator) with bounds."""
    denom = self.denominator
    if self.eps is not None:
      denom = denom + self.eps
    if self.min_denom is not None:
      denom = jnp.maximum(denom, self.min_denom)

    # JAX Safe Division: Prevent division-by-zero NaNs from poisoning gradients
    # We replace 0s with 1.0 *before* dividing.
    safe_denom = jnp.where(denom == 0, 1.0, denom)

    # Calculate scale, masking out pure zero denominators to 0.0
    scale = 1.0 / safe_denom
    return jnp.where(denom == 0, 0.0, scale)

  def compute(self) -> jax.Array:
    """Safely computes total / count with optional legacy equivalence bounds."""
    return self.unreduced_sum * self.compute_scale()


def as_weighted_metrics(loss: Any) -> WeightedMetrics:
  """Coerces a loss value into `WeightedMetrics`.

  Backward-compatibility adapter: loss functions returning a plain
  (already-reduced) scalar are wrapped with denominator 1, so weighted
  aggregation degenerates to the legacy equal-weight mean across
  micro-batches.

  Args:
    loss: A `WeightedMetrics` or a scalar loss array.

  Returns:
    The loss as `WeightedMetrics`.
  """
  if isinstance(loss, WeightedMetrics):
    return loss
  return WeightedMetrics(
      unreduced_sum=jnp.asarray(loss), denominator=jnp.asarray(1.0)
  )


@dataclasses.dataclass(slots=True, kw_only=True)
class StepMetrics:
  """Per-step result.

  Values stay as device arrays; callers decide when to sync to host so that
  inflight computation overlap is preserved.

  Attributes:
    loss: The step loss as a WeightedMetrics object.
    grad_norm: Global gradient norm. None for eval steps.
    aux: Auxiliary output of the loss function, if any.
    additional: Extra named metrics.
  """

  loss: WeightedMetrics
  grad_norm: jax.Array | None = None
  aux: Any = None
  additional: Dict[str, Any] = dataclasses.field(default_factory=dict)

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

  def forward_batch(self, payload: TrainerPayload, **kwargs) -> Any:
    """Executes a forward-only pass and returns model outputs.

    Unlike `eval_step` (which returns loss metrics), this returns the model
    outputs themselves (e.g., per-token log-probs recomputed with the
    trainer's exact parallelism, for RL log-prob recomputation). Must not
    mutate any trainer state.

    Optional capability: implementations that don't support it inherit this
    default, which raises NotImplementedError.

    Args:
      payload: The batch to run the forward pass on.
      **kwargs: Implementation-specific options.

    Returns:
      Model outputs as a pytree of arrays (gradients stopped).
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement forward_batch."
    )

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

  def update_weights(self, weights: nnx.State, **kwargs) -> None:
    """Updates model weights in place (counterpart of `get_weights`).

    Orchestrators use this to push externally produced weights into the
    trainer — e.g. resharded copies from another mesh, an anchor-policy
    snapshot, or weights broadcast by a colocated trainer — without reaching
    into the trainer's model. `weights` may be a partial state (e.g.
    LoRA-only); unmentioned variables are left untouched.

    Optional capability: implementations that don't support it inherit this
    default, which raises NotImplementedError.

    Args:
      weights: The weights to merge into the model, as an nnx.State (or a
        compatible partial state).
      **kwargs: Implementation-specific options.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement update_weights."
    )

  def offload(self, memory_kind: str = "pinned_host", **kwargs) -> None:
    """Moves trainer state off device to host memory, freeing HBM.

    Covers everything the trainer keeps alive on device between steps: model
    parameters, optimizer state, and internal step buffers (e.g. accumulated
    gradients). Sharding layout (mesh and partition specs) is preserved —
    only the memory kind changes — so `load` restores the exact previous
    placement and previously compiled step functions remain valid.

    Typical use: colocated RL, where the trainer's HBM is released while the
    rollout engine generates, then restored via `load` before the next
    optimizer step. Idempotent. Step methods must not be called while
    offloaded; call `load` first.

    Optional capability: implementations that don't support it inherit this
    default, which raises NotImplementedError.

    Args:
      memory_kind: Destination host memory kind ("pinned_host" or
        "unpinned_host").
      **kwargs: Implementation-specific options.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement offload."
    )

  def load(self, **kwargs) -> None:
    """Moves offloaded trainer state back to device memory.

    Inverse of `offload`. No-op if the trainer is not offloaded.

    Optional capability: implementations that don't support it inherit this
    default, which raises NotImplementedError.

    Args:
      **kwargs: Implementation-specific options.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement load."
    )

  @property
  def is_offloaded(self) -> bool:
    """Whether trainer state currently lives off device (see `offload`)."""
    return False

  @abc.abstractmethod
  def get_metrics(self) -> List[Tuple[StepMetrics, int, bool, bool]]:
    """Returns and clears the recently collected step metrics.

    Returns:
      A list of tuples: (step_metrics, step_id, is_eval, apply_gradients)
    """

  def close(self) -> None:
    """Releases resources held by the trainer. Default: no-op."""
