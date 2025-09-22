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

"""Implements Attention Based Distillation Strategy."""

from typing import Callable

from flax import nnx
import jax
from tunix.distillation.strategies import base_strategy
from tunix.distillation.strategies import feature_pooling
from tunix.distillation.strategies import feature_projection

ModelForwardCallable = base_strategy.ModelForwardCallable


class AttentionTransferStrategy(feature_pooling.FeaturePoolingStrategy):
  """Implements Attention Transfer distillation.

  This strategy minimizes the difference (typically Cosine Distance) between
  student and teacher attention maps from attention layers. It combines this
  attention loss with a standard task loss (softmax cross-entropy) on the
  student's final logits.
  """

  def __init__(
      self,
      student_forward_fn: ModelForwardCallable[jax.Array],
      teacher_forward_fn: ModelForwardCallable[jax.Array],
      labels_fn: Callable[..., jax.Array],
      attention_layer: type[nnx.Module],
      alpha: float = 0.75,
      attention_loss_fn: (
          Callable[[jax.Array, jax.Array], jax.Array] | None
      ) = None,
  ):
    """Initializes the AttentionTransfer strategy.

    Args:
        student_forward_fn: Function to compute student model outputs.
        teacher_forward_fn: Function to compute teacher model outputs.
        labels_fn: Function to compute labels from model inputs.
        attention_layer: The attention layer to use for distillation.
        alpha: Weight to balance attention loss and task loss (0.0 to 1.0).
        attention_loss_fn: A function that takes two jax. Arrays (student_map,
          teacher_map) and returns a scalar loss. Defaults to Cosine Distance.
          Error.
    """
    super().__init__(
        student_forward_fn,
        teacher_forward_fn,
        labels_fn,
        feature_layer=attention_layer,
        alpha=alpha,
        feature_loss_fn=attention_loss_fn,
    )


class AttentionProjectionStrategy(feature_projection.FeatureProjectionStrategy):
  """Implements Attention Projection distillation.

  This strategy minimizes the difference (typically MSE) between
  projected student and teacher attention maps from attention layers. It
  combines this attention loss with a standard task loss (softmax cross-entropy)
  on the student's final logits.
  """

  def __init__(
      self,
      student_forward_fn: ModelForwardCallable[jax.Array],
      teacher_forward_fn: ModelForwardCallable[jax.Array],
      labels_fn: Callable[..., jax.Array],
      attention_layer: type[nnx.Module],
      dummy_input: dict[str, jax.Array],
      rngs: nnx.Rngs,
      alpha: float = 0.75,
      attention_loss_fn: (
          Callable[[jax.Array, jax.Array], jax.Array] | None
      ) = None,
  ):
    """Initializes the AttentionProjection strategy.

    Args:
        student_forward_fn: Function to compute student model outputs.
        teacher_forward_fn: Function to compute teacher model outputs.
        labels_fn: Function to compute labels from model inputs.
        attention_layer: The attention layer to use for distillation.
        dummy_input: Dummy input to perform a forward pass on the models.
        rngs: Random number generator.
        alpha: Weight to balance attention loss and task loss (0.0 to 1.0).
        attention_loss_fn: A function that takes two jax. Arrays (student_map,
          teacher_map) and returns a scalar loss. Defaults to Mean Squared
          Error.
    """
    super().__init__(
        student_forward_fn,
        teacher_forward_fn,
        labels_fn,
        feature_layer=attention_layer,
        alpha=alpha,
        feature_loss_fn=attention_loss_fn,
        dummy_input=dummy_input,
        rngs=rngs,
    )
