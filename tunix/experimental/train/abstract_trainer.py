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

from tunix.experimental.metrics import metrics


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
  def compile(self, dummy_data: Any) -> None:
    """Triggers JAX compilation. `with_loss_fn` must be called first.

    Idempotent; safe to call multiple times. Under JAX jit semantics, XLA
    compilation itself still happens on the first call per input shape; this
    method constructs the jitted callables and applies optimizer sharding so
    the first step avoids double compilation. Does NOT restore checkpoints.
    Args:
      dummy_data: A dummy batch of data to trigger compilation.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement compile."
    )

  @abc.abstractmethod
  def fwd_bwd(self, payload: Any, **kwargs) -> None:
    """Executes forward and backward passes.

    Metrics are cached to overlap train steps.
    Does NOT apply an optimizer update; gradients are accumulated internally
    until `update()` is called. Gradient accumulation is therefore
    caller-driven: one `update()` per N `fwd_bwd()` calls.
    Args:
      payload: A packed micro-batch ready for gradient descent.
      **kwargs: Implementation-specific options.
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
  def eval_step(self, payload: Any, **kwargs) -> None:
    """Executes one evaluation step on the given payload.

    Must not mutate any trainer state, including gradient accumulation
    buffers.
    Args:
      payload: A packed micro-batch ready for evaluation.
      **kwargs: Implementation-specific options.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement eval_step."
    )

  @abc.abstractmethod
  def save_checkpoint(self, metadata: Any, **kwargs) -> None:
    """Force the trainer to serialize its state (model + optimizer).

    Checkpoint cadence/policy is the caller's responsibility.
    Args:
      metadata: The metadata pytree to save alongside the checkpoint.
      **kwargs: Implementation-specific options.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement save_checkpoint."
    )

  @abc.abstractmethod
  def restore_checkpoint(self, **kwargs) -> Any:
    """Restore state from latest checkpoint and return the metadata pytree.

    The metadata is the same as what is used on save_checkpoint.
    Args:
      **kwargs: Implementation-specific options.

    Returns:
      The restored metadata pytree (global_step, etc.).
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement restore_checkpoint."
    )

  @abc.abstractmethod
  def prepare_weight_sync(self, **kwargs) -> None:
    """Stages weights for transfer and returns coordinates/metadata for Rollouts to pull.

    For a Raiden based implementation, trigger the d2h weight transfer here.
    Args:
      **kwargs: Implementation-specific options.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement prepare_weight_sync."
    )

  @abc.abstractmethod
  def get_metrics(self) -> metrics.MetricsBuffer:
    """Returns and clears the recently collected step metric records."""
    raise NotImplementedError(
        f"{type(self).__name__} does not implement get_metrics."
    )

  @abc.abstractmethod
  def close(self) -> None:
    """Releases resources held by the trainer. Default: no-op."""
    pass
