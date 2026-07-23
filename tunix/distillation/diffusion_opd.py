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

"""On-policy distillation over target-aligned diffusion scores."""

import math
import numbers
from typing import Any, TypeVar

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.diffusion import interfaces as diffusion_interfaces
from tunix.distillation import diffusion
from tunix.sft import peft_trainer
from tunix.sft import utils as sft_utils

RawBatchT = TypeVar("RawBatchT")
TrainerT = TypeVar("TrainerT", bound=peft_trainer.PeftTrainer)


def _validate_loss_configuration(
    temperature: float,
    soft_loss_weight: float,
    hard_loss_weight: float,
) -> None:
  """Validates static diffusion distillation loss settings."""

  values = {
      "temperature": temperature,
      "soft_loss_weight": soft_loss_weight,
      "hard_loss_weight": hard_loss_weight,
  }
  for name, value in values.items():
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
    ):
      raise ValueError(f"{name} must be a finite real scalar; received {value}")
  if temperature <= 0:
    raise ValueError(f"temperature must be positive; received {temperature}")
  if soft_loss_weight < 0 or hard_loss_weight < 0:
    raise ValueError(
        "soft_loss_weight and hard_loss_weight must be nonnegative"
    )
  if soft_loss_weight == 0 and hard_loss_weight == 0:
    raise ValueError("at least one loss weight must be positive")


def _validate_active_teacher_logits(
    batch: diffusion.DiffusionDistillationBatch,
) -> None:
  """Rejects invalid eager teacher logits without converting JAX tracers."""

  if isinstance(batch.teacher_logits, jax.core.Tracer) or isinstance(
      batch.student_batch.loss_weights, jax.core.Tracer
  ):
    return
  teacher_logits = np.asarray(batch.teacher_logits)
  active_targets = np.asarray(batch.student_batch.loss_weights) > 0
  if not np.all(np.isfinite(teacher_logits[active_targets])):
    raise ValueError("active teacher_logits must contain only finite values")


def diffusion_opd_loss_fn(
    student_model: nnx.Module,
    batch: diffusion.DiffusionDistillationBatch,
    student_logits_fn: diffusion_interfaces.DiffusionLogitsFn,
    *,
    temperature: float = 1.0,
    soft_loss_weight: float = 1.0,
    hard_loss_weight: float = 0.0,
) -> sft_utils.LossOutput:
  """Computes weighted on-policy distillation loss for diffusion scores.

  Soft loss is the forward KL ``KL(teacher || student)`` over the same
  target-aligned student rollout, with standard temperature-squared scaling.
  Hard loss is optional target-aligned cross-entropy against ``target_ids``.
  Teacher logits are always stop-gradient.

  Args:
    student_model: The trainable student model.
    batch: Target-aligned student inputs and external teacher logits.
    student_logits_fn: Callable that returns student logits with shape
      ``[batch, length, vocab]``.
    temperature: Positive softmax temperature for the KL term.
    soft_loss_weight: Nonnegative multiplier for forward KL.
    hard_loss_weight: Nonnegative multiplier for hard-target cross-entropy.

  Returns:
    An unreduced weighted loss and token-weighted component metrics.
  """

  _validate_loss_configuration(temperature, soft_loss_weight, hard_loss_weight)
  batch.validate()
  if soft_loss_weight:
    _validate_active_teacher_logits(batch)

  student_logits = diffusion_interfaces.compute_diffusion_logits(
      student_model, batch.student_batch, student_logits_fn
  )
  if tuple(student_logits.shape) != tuple(batch.teacher_logits.shape):
    raise ValueError(
        "student and teacher logits must have identical shapes; received "
        f"{tuple(student_logits.shape)} and {tuple(batch.teacher_logits.shape)}"
    )

  weights = jnp.asarray(batch.student_batch.loss_weights, dtype=jnp.float32)
  active_targets = weights != 0
  student_logits = jnp.asarray(student_logits, dtype=jnp.float32)
  teacher_logits = jax.lax.stop_gradient(
      jnp.asarray(batch.teacher_logits, dtype=jnp.float32)
  )
  student_logits = jnp.where(active_targets[..., None], student_logits, 0.0)
  teacher_logits = jnp.where(active_targets[..., None], teacher_logits, 0.0)

  if soft_loss_weight:
    student_log_probs = jax.nn.log_softmax(
        student_logits / temperature, axis=-1
    )
    teacher_probs = jax.nn.softmax(teacher_logits / temperature, axis=-1)
    soft_token_losses = optax.kl_divergence(
        student_log_probs, teacher_probs
    ) * (temperature**2)
  else:
    soft_token_losses = jnp.zeros_like(weights)

  if hard_loss_weight:
    targets = jnp.asarray(batch.student_batch.target_ids)
    targets = jnp.where(active_targets, targets, 0)
    hard_token_losses = optax.softmax_cross_entropy_with_integer_labels(
        logits=student_logits,
        labels=targets,
    )
  else:
    hard_token_losses = jnp.zeros_like(weights)

  soft_sum = jnp.sum(soft_token_losses * weights, dtype=jnp.float32)
  hard_sum = jnp.sum(hard_token_losses * weights, dtype=jnp.float32)
  total_sum = (
      jnp.asarray(soft_loss_weight, dtype=jnp.float32) * soft_sum
      + jnp.asarray(hard_loss_weight, dtype=jnp.float32) * hard_sum
  )
  weight_sum = jnp.sum(weights, dtype=jnp.float32)

  return sft_utils.LossOutput(
      primary_loss=sft_utils.WeightedMetric(total_sum, weight_sum),
      aux_metrics={
          "distill/soft_loss": sft_utils.WeightedMetric(soft_sum, weight_sum),
          "distill/hard_loss": sft_utils.WeightedMetric(hard_sum, weight_sum),
      },
  )


def configure_prepared_diffusion_opd(
    trainer: TrainerT,
    batch_adapter: diffusion.PreparedDiffusionDistillationBatchAdapter[
        RawBatchT
    ],
    student_logits_fn: diffusion_interfaces.DiffusionLogitsFn,
    *,
    temperature: float = 1.0,
    soft_loss_weight: float = 1.0,
    hard_loss_weight: float = 0.0,
) -> TrainerT:
  """Configures a ``PeftTrainer`` for prepared external-teacher diffusion OPD.

  The raw batch must contain a fresh rollout from the current student and
  teacher logits for that exact rollout. This function does not perform or
  verify on-policy generation; a model-aware integration must prepare the
  batch outside the jitted loss step. It only binds the prepared adapter and
  objective to Tunix's stable trainer extension points.
  """

  _validate_loss_configuration(temperature, soft_loss_weight, hard_loss_weight)

  def gen_model_input_fn(raw_batch: RawBatchT) -> dict[str, Any]:
    return {"batch": batch_adapter(raw_batch)}

  def loss_fn(
      student_model: nnx.Module,
      batch: diffusion.DiffusionDistillationBatch,
  ) -> sft_utils.LossOutput:
    return diffusion_opd_loss_fn(
        student_model,
        batch,
        student_logits_fn,
        temperature=temperature,
        soft_loss_weight=soft_loss_weight,
        hard_loss_weight=hard_loss_weight,
    )

  trainer.with_gen_model_input_fn(gen_model_input_fn)
  trainer.with_loss_fn(loss_fn)
  return trainer
