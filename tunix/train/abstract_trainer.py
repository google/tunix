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

NOTE: this interface is being implemented incrementally. Methods use
`NotImplementedError` defaults (rather than `abc.abstractmethod`) so that
implementations can adopt the API one method at a time; calling a
not-yet-implemented method fails loudly at the call site.

Implementation status in `PeftTrainer`:
  - fwd_bwd, update, with_loss_fn, compile: implemented (step 1).
  - eval_step, save_checkpoint, restore_checkpoint, get_weights,
    get_metrics: pending subsequent steps.
"""

import abc
from typing import Any, Callable, List, Optional


class AbstractTrainer(abc.ABC):
  """The pure ML algorithmic core of a trainer.

  Step-level only: no training loops, no I/O policy, no orchestration.
  """

  def with_loss_fn(
      self, loss_fn: Callable[..., Any], has_aux: bool = False
  ) -> "AbstractTrainer":
    """Sets the loss function used by `fwd_bwd` (and evaluation).

    Changing the loss function invalidates any compiled step functions;
    implementations must rebuild them (see `compile`).

    Args:
      loss_fn: Called as `loss_fn(model, **inputs)`; returns the loss, or
        `(loss, aux)` when `has_aux` is True.
      has_aux: Whether `loss_fn` returns auxiliary output alongside the loss.

    Returns:
      self, for chaining.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement with_loss_fn."
    )

  def compile(self) -> None:
    """Builds the step functions and shards optimizer state.

    Idempotent; safe to call multiple times. Under JAX jit semantics, XLA
    compilation itself still happens on the first call per input shape; this
    method constructs the jitted callables and applies optimizer sharding so
    the first step avoids double compilation. Does NOT restore checkpoints.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement compile."
    )

  def fwd_bwd(self, inputs: Any, **kwargs) -> Any:
    """Executes one forward/backward micro-step.

    Computes the loss and gradients for one micro-batch. How the gradients
    are handled is implementation-specific (e.g. staged for `update`, or
    accumulated/applied by an optimizer wrapper fused into this step).

    Args:
      inputs: A raw training batch.
      **kwargs: Implementation-specific options.

    Returns:
      Implementation-defined step outputs (e.g. loss, aux, grad norm) as
      device arrays.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement fwd_bwd."
    )

  def update(self, **kwargs) -> int:
    """Finalizes one optimizer update over the preceding `fwd_bwd` calls.

    Call once per gradient-accumulation window. Where the model weights are
    actually applied is implementation-specific: here, or fused into the
    accumulation-boundary `fwd_bwd` call (e.g. via `optax.MultiSteps`).
    Advances the train step count either way.

    Args:
      **kwargs: Implementation-specific options.

    Returns:
      The new train step count (number of optimizer updates applied).
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement update."
    )

  def eval_step(self, inputs: Any, **kwargs) -> Any:
    """Executes a forward-only evaluation step.

    Must not mutate any trainer state, including gradient accumulation
    buffers.

    Args:
      inputs: A raw batch.
      **kwargs: Implementation-specific options.

    Returns:
      Implementation-defined evaluation outputs.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement eval_step."
    )

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
    raise NotImplementedError(
        f"{type(self).__name__} does not implement save_checkpoint."
    )

  def restore_checkpoint(self, path: str, **kwargs) -> int:
    """Restores model and optimizer state from disk.

    Args:
      path: The checkpoint to restore.
      **kwargs: Implementation-specific options.

    Returns:
      The restored global step.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement restore_checkpoint."
    )

  def get_weights(self, **kwargs) -> Any:
    """Returns current model weights (e.g., for weight syncing)."""
    raise NotImplementedError(
        f"{type(self).__name__} does not implement get_weights."
    )

  def get_metrics(self) -> List[Any]:
    """Returns and clears the recently collected step metric records."""
    raise NotImplementedError(
        f"{type(self).__name__} does not implement get_metrics."
    )

  def close(self) -> None:
    """Releases resources held by the trainer. Default: no-op."""
