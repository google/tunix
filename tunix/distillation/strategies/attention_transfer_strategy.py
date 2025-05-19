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

"""Implements Attention Transfer Distillation Strategy."""

from typing import Callable
import jax
import jax.numpy as jnp
import optax
from tunix.distillation import feature_extraction
from tunix.distillation.strategies import base_strategy

ModelForwardCallable = base_strategy.ModelForwardCallable


def compute_cosine_distance_loss(
    student_attention: jax.Array, teacher_attention: jax.Array
) -> jax.Array:
  """Computes the cosine distance between two attention maps."""
  teacher_attention = feature_extraction.avg_pool_array_to_target_shape(
      teacher_attention, student_attention.shape
  )
  return jnp.mean(optax.cosine_distance(student_attention, teacher_attention))


class AttentionTransferStrategy(base_strategy.BaseStrategy):
  """Implements Attention Transfer distillation.

  This strategy minimizes the difference (typically MSE) between student and
  teacher attention maps from attention layers. It combines this attention loss
  with a standard task loss (softmax cross-entropy) on the student's final
  logits.
  """

  def __init__(
      self,
      student_forward_fn: ModelForwardCallable[dict[str, jax.Array]],
      teacher_forward_fn: ModelForwardCallable[jax.Array],
      labels_fn: Callable[..., jax.Array],
      alpha: float = 0.5,
      attention_loss_fn: (
          Callable[[jax.Array, jax.Array], jax.Array] | None
      ) = None,
  ):
    """Initializes the AttentionTransfer strategy.

    Args:
        student_forward_fn: Function to compute student model outputs.
        teacher_forward_fn: Function to compute teacher model outputs.
        labels_fn: Function to compute labels from model inputs.
        alpha: Weight to balance attention loss and task loss (0.0 to 1.0).
        attention_loss_fn: A function that takes two jax. Arrays (student_map,
          teacher_map) and returns a scalar loss. Defaults to Mean Squared
          Error.
    """
    super().__init__(student_forward_fn, teacher_forward_fn, labels_fn)
    if not (0.0 <= alpha <= 1.0):
      raise ValueError(
          f"Alpha must be a float between 0.0 and 1.0, got {alpha}"
      )

    self.alpha = alpha
    self.attention_loss_fn = attention_loss_fn or compute_cosine_distance_loss

  def compute_eval_loss(
      self,
      student_output: dict[str, jax.Array],
      labels: jax.Array,
  ) -> jax.Array:
    """Computes the task loss.

    Args:
        student_output: Dictionary from student model, must contain 'logits'.
        labels: The ground truth labels for the examples.

    Returns:
        A JAX array representing the task loss, averaged over the batch.
    """
    # Calculate Task Loss (Cross-Entropy on original student logits)
    ce_loss = optax.softmax_cross_entropy(
        logits=student_output["logits"], labels=labels
    )
    task_loss = jnp.mean(ce_loss)

    return task_loss

  def compute_loss(
      self,
      student_output: dict[str, jax.Array],
      teacher_output: jax.Array,
      labels: jax.Array,
  ) -> jax.Array:
    """Computes the combined attention transfer and task loss.

    Args:
        student_output: Dictionary from student model, must contain 'logits' and
          'attentions'.
        teacher_output: List of teacher attention arrays.
        labels: The ground truth labels (e.g., one-hot encoded).

    Returns:
        A JAX array representing the combined distillation and task loss,
        averaged over the batch.
    """
    if "logits" not in student_output or "attentions" not in student_output:
      raise TypeError(
          "student_outputs must be a dict containing 'logits' and 'attentions'"
      )

    student_logits = student_output["logits"]
    student_attentions = student_output["attentions"]

    attention_loss = self.attention_loss_fn(student_attentions, teacher_output)

    # Calculate Task Loss (Cross-Entropy)
    ce_loss_per_example = optax.softmax_cross_entropy(
        logits=student_logits, labels=labels
    )
    task_loss = jnp.mean(ce_loss_per_example)

    # Combine the losses
    combined_loss = (self.alpha * attention_loss) + (
        (1.0 - self.alpha) * task_loss
    )

    return combined_loss
