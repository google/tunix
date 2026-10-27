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
import dataclasses
from typing import Any, Callable, Protocol, TypeVar, cast

import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.metrics import metrics

_PayloadT = TypeVar("_PayloadT")


class _TrainerWithModel(Protocol):
  """Structural protocol for trainer engines exposing a `.model` attribute."""

  @property
  def model(self) -> Any:
    ...


class AbstractTrainer(abc.ABC):
  """The pure ML algorithmic core of a trainer.

  Step-level only: no training loops, no I/O policy, no orchestration.
  """

  model: Any

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
  def compile(
      self, dummy_data: datatypes.TrainerPayload | None = None
  ) -> None:
    """Triggers ahead-of-time JAX compilation. `with_loss_fn` must be called first.

    Idempotent; safe to call multiple times. When `dummy_data` is `None`,
    compilation is deferred (no-op) until a concrete payload is supplied.
    When `dummy_data` is provided, lowers and compiles the trainer step
    executables (`fwd_bwd`, `update`, and `eval`) ahead of time so the first
    real step hits the JIT cache without mutating model or optimizer state.
    Does NOT restore checkpoints.

    Args:
      dummy_data: Optional dummy batch of data to lower and compile step
        executables ahead of time. If `None`, compilation is deferred.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement compile."
    )

  @abc.abstractmethod
  def fwd_bwd(self, payload: datatypes.TrainerPayload) -> None:
    """Executes forward and backward passes.

    Metrics are cached to overlap train steps.
    Does NOT apply an optimizer update; gradients are accumulated internally
    until `update()` is called. Gradient accumulation is therefore
    caller-driven: one `update()` per N `fwd_bwd()` calls.
    Args:
      payload: A packed micro-batch ready for gradient descent.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement fwd_bwd."
    )

  @abc.abstractmethod
  def update(self) -> int:
    """Applies the accumulated (mean) gradients as one optimizer update.

    Must be preceded by at least one `fwd_bwd()` call since the last update.

    Returns:
      The new train step count (number of optimizer updates applied).
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement update."
    )

  @abc.abstractmethod
  def eval_step(self, payload: datatypes.TrainerPayload) -> None:
    """Executes one evaluation step on the given payload.

    Must not mutate any trainer state, including gradient accumulation
    buffers.
    Args:
      payload: A packed micro-batch ready for evaluation.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement eval_step."
    )

  @abc.abstractmethod
  def fwd_only(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Read-only.

    Runs a forward pass with the trainer model and caller-supplied function.

    The trainer owns model access and shards `args`/`kwargs` on its data axis;
    `fn` owns the computation and must not mutate the model.
    Args:
      fn: Called as `fn(model, *args, **kwargs)`.
      *args: Positional arguments forwarded to `fn` after sharding.
      **kwargs: Keyword arguments forwarded to `fn` after sharding.

    Returns:
      Whatever `fn` returns.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement fwd_only."
    )

  @abc.abstractmethod
  def save_checkpoint(
      self,
      metadata: Any = None,
      *,
      step: int | None = None,
      force: bool = False,
  ) -> None:
    """Force the trainer to serialize its state (model + optimizer).

    Checkpoint cadence/policy is the caller's responsibility.
    Args:
      metadata: The metadata pytree to save alongside the checkpoint, if any.
      step: The step to save under. If None, implementations use
        `metadata["step"]` when present, else their current train step.
      force: Save even if the checkpoint manager's save policy (e.g. a save
        interval) would skip this step.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement save_checkpoint."
    )

  @abc.abstractmethod
  def restore_checkpoint(self, *, step: int | None = None) -> Any:
    """Restore state from checkpoint and return the metadata pytree.

    The metadata is the same as what is used on save_checkpoint.
    Args:
      step: Checkpoint step to restore. If None, restores from the latest
        checkpoint.

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
  def prepare_weight_sync(self, *, sync_request: Any = None) -> Any:
    """Stages weights for transfer and returns coordinates/metadata for Rollouts to pull.

    For a Raiden based implementation, trigger the d2h weight transfer here.
    Args:
      sync_request: The round's request from `WeightSyncCoordinator`.

    Returns:
      The staged weights' transfer metadata (e.g. work-unit metadata).
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement prepare_weight_sync."
    )

  def release_weight_sync(self, *, sync_request: Any = None) -> Any:
    """Ends the staging hold taken by `prepare_weight_sync`.

    Called once the round's transfer is complete. Default implementation is a
    no-op for trainers that hold nothing between prepare and release.

    Args:
      sync_request: The same request passed to `prepare_weight_sync`.

    Returns:
      An implementation-defined acknowledgement.
    """
    del sync_request
    return None

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


def resolve_dummy_routed_experts(
    trainer: AbstractTrainer | _TrainerWithModel,
    data: _PayloadT,
) -> _PayloadT:
  """Expands `(B, S, 0, 0)` precompile `routed_experts` placeholders.

  During pre-rollout trainer compilation, `batch_assembly` builds a dummy
  `RLTrainerPayload` with a zero-sized `(B, S, 0, 0)` `routed_experts`
  placeholder and attaches `routed_experts_encode_fn` in `payload.metadata`.
  Expanding and encoding the placeholder against the trainer model's MoE
  dimensions (`num_layers` or MaxText's `num_decoder_layers`,
  `num_experts_per_tok`, and `num_experts`) ensures all trainer backends
  (`PeftTrainer` and `MaxTextTrainingEngine`) precompile with the exact
  encoded `routed_experts` shape and dtype.

  This function is idempotent: once expanded (`routed_experts.size != 0`),
  subsequent calls return `data` unchanged.

  Args:
    trainer: Initialized trainer or training engine exposing `.model.config` (or
      `.model.base.config` for `TunixMaxTextAdapter`-wrapped MaxText models).
    data: Optional dummy payload passed to `compile()`.

  Returns:
    `data` with any zero-sized `routed_experts` placeholder expanded (and
    encoded via `metadata.pop("routed_experts_encode_fn")` if present).
  """
  if not isinstance(data, datatypes.RLTrainerPayload):
    return data
  if data.routed_experts is None:
    return data
  routed_experts = np.asarray(data.routed_experts)
  if routed_experts.size != 0:
    return data
  if routed_experts.ndim < 2:
    raise ValueError(
        "Expected dummy routed_experts placeholder with rank >= 2, got shape"
        f" {routed_experts.shape}."
    )
  model = getattr(trainer, "model", None)
  # `maxtext.utils.model_creation_utils.from_pretrained(...,
  # wrap_with_tunix_adapter=True)` sets `adapter.config = None`, so treat a
  # `None` `.config` as absent and fall back to `.base.config` (or `_config`).
  cfg = (
      getattr(model, "config", None)
      or getattr(getattr(model, "base", None), "config", None)
      or getattr(trainer, "_config", None)
  )
  if cfg is None:
    raise ValueError(
        "Cannot resolve dummy routed_experts: trainer model does not expose a"
        " non-None `.config` or `.base.config`."
    )
  num_layers = getattr(cfg, "num_layers", None)
  if num_layers is None:
    num_layers = cfg.num_decoder_layers
  batch_size, seq_len = routed_experts.shape[:2]
  expanded = np.zeros(
      (batch_size, seq_len, num_layers, cfg.num_experts_per_tok),
      dtype=routed_experts.dtype,
  )
  metadata = dict(data.metadata)
  if "routed_experts_encode_fn" in metadata:
    encode_fn = metadata.pop("routed_experts_encode_fn")
    expanded = encode_fn(expanded, num_experts=cfg.num_experts)
  return cast(
      _PayloadT,
      dataclasses.replace(data, routed_experts=expanded, metadata=metadata),
  )
