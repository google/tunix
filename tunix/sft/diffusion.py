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

"""Supervised fine-tuning loss for target-aligned diffusion scores."""

from typing import Any, TypeVar

from flax import nnx
import jax.numpy as jnp
import optax
from tunix.diffusion import interfaces as diffusion_interfaces
from tunix.diffusion import types as diffusion_types
from tunix.sft import peft_trainer
from tunix.sft import utils as sft_utils

RawBatchT = TypeVar("RawBatchT")
TrainerT = TypeVar("TrainerT", bound=peft_trainer.PeftTrainer)


def diffusion_loss_fn(
    model: nnx.Module,
    batch: diffusion_types.DiffusionTokenBatch,
    logits_fn: diffusion_interfaces.DiffusionLogitsFn,
) -> sft_utils.LossOutput:
  """Computes weighted cross-entropy for target-aligned diffusion scores.

  ``logits_fn`` must return logits whose first two dimensions already align with
  ``batch.target_ids``. No autoregressive one-token shift is applied. Whether
  the batch represents CFT or SFT is determined by the external batch adapter.

  Args:
    model: The model to score.
    batch: Canonical diffusion inputs, targets, and per-target weights.
    logits_fn: Model-specific target-aligned logits callable.

  Returns:
    A ``LossOutput`` containing the weighted loss sum and total weight.
  """

  logits = diffusion_interfaces.compute_diffusion_logits(
      model, batch, logits_fn
  )
  logits = jnp.asarray(logits, dtype=jnp.float32)
  targets = jnp.asarray(batch.target_ids)
  weights = jnp.asarray(batch.loss_weights, dtype=jnp.float32)

  active_targets = weights != 0
  logits = jnp.where(active_targets[..., None], logits, 0.0)
  targets = jnp.where(active_targets, targets, 0)
  token_losses = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits,
      labels=targets,
  )
  weighted_losses = token_losses * weights
  loss_sum = jnp.sum(weighted_losses, dtype=jnp.float32)
  weight_sum = jnp.sum(weights, dtype=jnp.float32)

  return sft_utils.LossOutput(
      primary_loss=sft_utils.WeightedMetric(loss_sum, weight_sum),
      aux_metrics={},
  )


def configure_diffusion_sft(
    trainer: TrainerT,
    batch_adapter: diffusion_interfaces.DiffusionBatchAdapter[RawBatchT],
    logits_fn: diffusion_interfaces.DiffusionLogitsFn,
) -> TrainerT:
  """Configures and returns a trainer for diffusion supervised fine-tuning."""

  def gen_model_input_fn(raw_batch: RawBatchT) -> Any:
    return {"batch": batch_adapter(raw_batch)}

  def loss_fn(
      model: nnx.Module,
      batch: diffusion_types.DiffusionTokenBatch,
  ) -> sft_utils.LossOutput:
    return diffusion_loss_fn(model, batch, logits_fn)

  trainer.with_gen_model_input_fn(gen_model_input_fn)
  trainer.with_loss_fn(loss_fn)
  return trainer
