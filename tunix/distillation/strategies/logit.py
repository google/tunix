# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements Logit Distillation Strategy."""

from typing import Callable

import jax
import jax.numpy as jnp
import optax
from tunix.distillation.strategies import base_strategy

ModelForwardCallable = base_strategy.ModelForwardCallable


class LogitStrategy(base_strategy.BaseStrategy):
  """Implements Logit Distillation.

  This strategy minimizes the KL divergence between the student and teacher
  logits. It combines this attention loss with a standard task loss (softmax
  cross-entropy) on the student's final logits.
  """

  def __init__(
      self,
      student_forward_fn: ModelForwardCallable[jax.Array],
      teacher_forward_fn: ModelForwardCallable[jax.Array],
      labels_fn: Callable[..., jax.Array],
      temperature: float = 2.0,
      alpha: float = 0.5,
  ):
    """Initializes the Logit strategy.

    Args:
        student_forward_fn: Function to compute student model outputs.
        teacher_forward_fn: Function to compute teacher model outputs.
        labels_fn: Function to compute labels from model inputs.
        temperature: Temperature for softening probabilities (> 0).
        alpha: Weight to balance distillation loss and task loss (0.0 to 1.0).
    """
    super().__init__(student_forward_fn, teacher_forward_fn, labels_fn)
    if temperature <= 0:
      raise ValueError(
          f"Temperature must be a positive number, got {temperature}"
      )
    if not (0.0 <= alpha <= 1.0):
      raise ValueError(
          f"Alpha must be a float between 0.0 and 1.0, got {alpha}"
      )

    self.temperature = float(temperature)
    self.alpha = alpha

  def compute_eval_loss(
      self,
      student_output: jax.Array,
      labels: jax.Array,
  ) -> jax.Array:
    """Computes the task loss.

    Args:
        student_output: The logits from the student model.
        labels: The ground truth labels for the examples.

    Returns:
        A JAX array representing the task loss, averaged over the batch.
    """
    # Calculate Task Loss (Cross-Entropy on original student logits)
    ce_loss = optax.softmax_cross_entropy(logits=student_output, labels=labels)
    task_loss = jnp.mean(ce_loss)

    return task_loss

  def compute_loss(
      self,
      student_output: jax.Array,
      teacher_output: jax.Array,
      labels: jax.Array,
  ) -> jax.Array:
    """Computes the loss for logit distillation.

    Args:
        student_output: The logits from the student model.
        teacher_output: The logits from the teacher model.
        labels: The ground truth labels for the examples.

    Returns:
        A JAX array representing the combined distillation and task loss,
        averaged over the batch.
    """
    # Calculate Distillation Loss (KL Divergence on softened targets)
    log_student_probs_temp = jax.nn.log_softmax(
        student_output / self.temperature, axis=-1
    )
    teacher_probs_temp = jax.nn.softmax(
        teacher_output / self.temperature, axis=-1
    )
    kl_loss = optax.kl_divergence(log_student_probs_temp, teacher_probs_temp)
    # Applying temperature scaling to KL loss as per
    # (https://arxiv.org/pdf/1503.02531)
    scaled_kl_loss = kl_loss * (self.temperature**2)
    distillation_loss = jnp.mean(scaled_kl_loss)

    # Calculate Task Loss (Cross-Entropy on original student logits)
    ce_loss = optax.softmax_cross_entropy(logits=student_output, labels=labels)
    task_loss = jnp.mean(ce_loss)

    # Combine the losses
    combined_loss = (self.alpha * distillation_loss) + (
        (1.0 - self.alpha) * task_loss
    )

    return combined_loss
