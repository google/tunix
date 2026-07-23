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

"""Data types shared by diffusion training objectives."""

from collections.abc import Mapping
from typing import Any, TypeAlias

import flax
import jax
import jax.numpy as jnp
import numpy as np

Array: TypeAlias = jax.Array | np.ndarray
ModelInputs: TypeAlias = Mapping[str, Any]


def _shape_and_dtype(name: str, value: Any) -> tuple[tuple[int, ...], Any]:
  if not isinstance(value, (jax.Array, jax.core.Tracer, np.ndarray)):
    raise TypeError(f"{name} must be a JAX or NumPy array")
  return tuple(value.shape), value.dtype


def _is_loss_weight_dtype(dtype: Any) -> bool:
  return (
      jnp.issubdtype(dtype, jnp.bool_)
      or jnp.issubdtype(dtype, jnp.integer)
      or jnp.issubdtype(dtype, jnp.floating)
  )


def _concrete_numpy(value: Any) -> np.ndarray | None:
  """Returns eager values without performing host conversion while tracing."""

  if isinstance(value, jax.core.Tracer):
    return None
  if isinstance(value, jax.Array) and not value.is_fully_addressable:
    return None
  return np.asarray(value)


@flax.struct.dataclass(frozen=True)
class DiffusionTokenBatch:
  """Canonical target-aligned batch for diffusion training.

  Every array in ``model_inputs`` must be batch-major. Model-static values and
  non-batch tensors belong in the scoring callable so that Tunix can shard the
  complete batch pytree consistently along its leading dimension.

  Attributes:
    model_inputs: Model-specific, batch-major array pytree consumed by a
      ``DiffusionLogitsFn``.
    target_ids: Target token IDs with shape ``[batch, length]``.
    loss_weights: Per-target weights with the same shape as ``target_ids``.
  """

  model_inputs: ModelInputs
  target_ids: Array
  loss_weights: Array

  @classmethod
  def create(
      cls,
      *,
      model_inputs: ModelInputs,
      target_ids: Array,
      loss_weights: Array,
  ) -> "DiffusionTokenBatch":
    """Constructs and validates a diffusion batch."""

    return cls(
        model_inputs=model_inputs,
        target_ids=target_ids,
        loss_weights=loss_weights,
    ).validate()

  def validate(self) -> "DiffusionTokenBatch":
    """Validates the batch's static shape and dtype contract."""

    if not isinstance(self.model_inputs, Mapping):
      raise TypeError("model_inputs must be a mapping")

    target_shape, target_dtype = _shape_and_dtype("target_ids", self.target_ids)
    if len(target_shape) != 2:
      raise ValueError(
          f"target_ids must have shape [batch, length]; received {target_shape}"
      )
    if not jnp.issubdtype(target_dtype, jnp.integer):
      raise TypeError(
          f"target_ids must have an integer dtype; received {target_dtype}"
      )

    weight_shape, weight_dtype = _shape_and_dtype(
        "loss_weights", self.loss_weights
    )
    if weight_shape != target_shape:
      raise ValueError(
          "loss_weights must match target_ids shape; "
          f"received {weight_shape} and {target_shape}"
      )
    if not _is_loss_weight_dtype(weight_dtype):
      raise TypeError(
          "loss_weights must have a real numeric or boolean dtype; "
          f"received {weight_dtype}"
      )
    concrete_weights = _concrete_numpy(self.loss_weights)
    if concrete_weights is not None:
      if not np.all(np.isfinite(concrete_weights)):
        raise ValueError("loss_weights must contain only finite values")
      if np.any(concrete_weights < 0):
        raise ValueError("loss_weights must be nonnegative")
      concrete_targets = _concrete_numpy(self.target_ids)
      if concrete_targets is not None and np.any(
          concrete_targets[concrete_weights > 0] < 0
      ):
        raise ValueError("active target_ids must be nonnegative")

    model_input_leaves = jax.tree.leaves(self.model_inputs)
    if not model_input_leaves:
      raise ValueError("model_inputs must contain at least one array")
    batch_size = target_shape[0]
    for index, leaf in enumerate(model_input_leaves):
      leaf_shape, leaf_dtype = _shape_and_dtype(
          f"model_inputs leaf {index}", leaf
      )
      if not leaf_shape:
        raise ValueError(
            f"model_inputs leaf {index} must be batch-major, not scalar"
        )
      if leaf_shape[0] != batch_size:
        raise ValueError(
            f"model_inputs leaf {index} has batch size {leaf_shape[0]}; "
            f"expected {batch_size}"
        )
      if not _is_loss_weight_dtype(leaf_dtype):
        raise TypeError(
            f"model_inputs leaf {index} must have a real numeric or boolean "
            f"dtype; received {leaf_dtype}"
        )
    return self


def validate_diffusion_logits(
    batch: DiffusionTokenBatch, logits: Array
) -> Array:
  """Validates and returns target-aligned token logits.

  The checks use static shape and dtype metadata, so this function is safe to
  invoke while tracing a JAX computation.

  Args:
    batch: The batch whose targets the scores must align with.
    logits: Floating-point token logits with shape ``[batch, length, vocab]``.

  Returns:
    ``logits`` unchanged.

  Raises:
    TypeError: If ``logits`` is not an array or has a non-floating dtype.
    ValueError: If ``logits`` is not target-aligned, has an empty vocabulary,
      or an active target ID falls outside that vocabulary.
  """

  logit_shape, logit_dtype = _shape_and_dtype("logits", logits)
  if len(logit_shape) != 3:
    raise ValueError(
        f"logits must have shape [batch, length, vocab]; received {logit_shape}"
    )
  if logit_shape[:2] != tuple(batch.target_ids.shape):
    raise ValueError(
        "logits must align with target_ids on [batch, length]; "
        f"received {logit_shape[:2]} and {tuple(batch.target_ids.shape)}"
    )
  if logit_shape[2] < 1:
    raise ValueError("logits must have a non-empty vocabulary dimension")
  if not jnp.issubdtype(logit_dtype, jnp.floating):
    raise TypeError(
        f"logits must have a floating-point dtype; received {logit_dtype}"
    )
  concrete_targets = _concrete_numpy(batch.target_ids)
  concrete_weights = _concrete_numpy(batch.loss_weights)
  if concrete_targets is not None and concrete_weights is not None:
    active_targets = concrete_targets[concrete_weights > 0]
    if np.any(active_targets >= logit_shape[2]):
      raise ValueError(
          "active target_ids must be smaller than vocabulary size"
          f" {logit_shape[2]}"
      )
  return logits
