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

import numpy as np
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
from tunix.experimental.common import datatypes
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
  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], dict[str, Any]]
  ) -> "AbstractTrainer":
    """Sets the last-mile adapter mapping a payload to the loss fn's kwargs.

    This adapter enables the trainer to accept arbitrary payloads (SFT, RL,
    etc.) by transforming them into kwargs for the loss function via
    `gen_model_input_fn(payload)`.
    Args:
      gen_model_input_fn: Maps a payload to a dict of loss-fn keyword arguments.

    Returns:
      self, for chaining.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement with_gen_model_input_fn."
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
  def fwd_bwd(self, payload: datatypes.TrainerPayload, **kwargs) -> None:
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
  def eval_step(self, payload: datatypes.TrainerPayload, **kwargs) -> None:
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
  def per_token_logps(
      self,
      *,
      prompt_tokens: ArrayLike,
      completion_tokens: ArrayLike,
      pad_id: int,
      eos_id: int,
      temperature: float | None = None,
      segment_ids: ArrayLike | None = None,
      segment_positions: ArrayLike | None = None,
      micro_batch_size: int | None = None,
  ) -> np.ndarray:
    """Scores per-token log-probabilities of completions under live weights.

    Unlike a frozen reference scorer, this uses the trainer's current (actor)
    parameters, so callers can measure sampler-vs-trainer agreement. Must not
    mutate trainer state (no gradient accumulation, no optimizer update).
    Args:
      prompt_tokens: [B, P] token ids, LEFT-padded (or [B, 0] in packed mode).
      completion_tokens: [B, C] token ids, RIGHT-padded; results align to these.
      pad_id: Pad token id.
      eos_id: End-of-sequence token id.
      temperature: Softmax temperature to score under; defaults to 1.0 when None.
      segment_ids: Optional packing segment ids (sequence packing).
      segment_positions: Optional packing local position indices.
      micro_batch_size: Optional row chunk size to bound peak memory.
    Returns:
      [B, C] per-token log-probabilities aligned to `completion_tokens`.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement per_token_logps."
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
  def restore_checkpoint(
      self, step: int | None = None, **kwargs
  ) -> Any:
    """Restore state from checkpoint and return the metadata pytree.

    The metadata is the same as what is used on save_checkpoint.
    Args:
      step: Checkpoint step to restore. If None, restores from the latest
        checkpoint.
      **kwargs: Implementation-specific options.

    Returns:
      The restored metadata pytree (global_step, etc.).
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement restore_checkpoint."
    )

  def set_target_state(self, target_state: Any) -> None:
    """Stores target state shape/dtype pytree for rollout parameter conversion.

    Default implementation is a no-op for trainers that do not require state
    conversion before weight synchronization.

    Args:
      target_state: The target state pytree from the rollout worker.
    """
    pass

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
