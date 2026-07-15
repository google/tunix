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

Defines the pure ML algorithmic core of a trainer.
"""

import abc
from typing import Any, Callable, List, Optional


class AbstractTrainer(abc.ABC):
  """The pure ML algorithmic core of a trainer.

  Step-level only: no training loops, no I/O policy, no orchestration.
  """

  @abc.abstractmethod
  def __init__(self, config: Any):
    """Initializes the trainer with the given config.

    Args:
      config: The trainer config.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement __init__."
    )

  @abc.abstractmethod
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

  @abc.abstractmethod
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

  @abc.abstractmethod
  def fwd_bwd(self, inputs: Any, **kwargs) -> Any:
    """Executes one forward/backward pass and accumulates gradients.

    Does NOT apply an optimizer update; gradients are accumulated internally
    until `update()` is called. Gradient accumulation is therefore
    caller-driven: one `update()` per N `fwd_bwd()` calls.
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

  @abc.abstractmethod
  def update(self, **kwargs) -> int:
    """Applies the accumulated (mean) gradients as one optimizer update.

    Must be preceded by at least one `fwd_bwd()` call since the last update.
    Args:
      **kwargs: Implementation-specific options.
    Returns:
      The new train step count (number of optimizer updates applied).
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement update."
    )

  @abc.abstractmethod
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
    raise NotImplementedError(
        f"{type(self).__name__} does not implement save_checkpoint."
    )

  @abc.abstractmethod
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

  @abc.abstractmethod
  def prepare_weight_sync(self, **kwargs) -> Any:
    """Stages weights for transfer and returns coordinates/metadata.

    Args:
      **kwargs: Implementation-specific options.
    Returns:
      Coordinates/metadata for Rollouts to pull.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement get_weights."
    )

  @abc.abstractmethod
  def get_metrics(self) -> List[Any]:
    """Returns and clears the recently collected step metric records."""
    raise NotImplementedError(
        f"{type(self).__name__} does not implement get_metrics."
    )

  @abc.abstractmethod
  def close(self) -> None:
    """Releases resources held by the trainer. Default: no-op."""
    pass
