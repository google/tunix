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

"""Implements Distillation Strategies."""

import abc
from typing import Any, Callable, Protocol, TypeVar

from flax import nnx
import jax

R = TypeVar("R")


class ModelForwardCallable(Protocol[R]):

  def __call__(self, model: nnx.Module, *args: Any, **kwargs: Any) -> R:
    ...


class BaseStrategy(abc.ABC):
  """Abstract Base Class for all distillation strategies.

  Defines the common interface for computing the distillation loss. Concrete
  strategies must implement the `compute_loss` method.
  """

  def __init__(
      self,
      student_forward_fn: ModelForwardCallable[Any],
      teacher_forward_fn: ModelForwardCallable[Any],
      labels_fn: Callable[..., jax.Array],
  ):
    """Initializes the BaseStrategy.

    Args:
      student_forward_fn: Function to compute student model outputs.
      teacher_forward_fn: Function to compute teacher model outputs.
      labels_fn: Function to compute labels from model inputs.
    """
    self._student_forward_fn = nnx.jit(student_forward_fn)
    self._teacher_forward_fn = nnx.jit(teacher_forward_fn)
    self._labels_fn = nnx.jit(labels_fn)

  @abc.abstractmethod
  def compute_loss(
      self, student_output: Any, teacher_output: Any, labels: jax.Array
  ) -> jax.Array:
    """Computes the distillation loss based on model outputs and labels."""

  @abc.abstractmethod
  def compute_eval_loss(
      self, student_output: Any, labels: jax.Array
  ) -> jax.Array:
    """Computes the distillation loss based on model outputs and labels."""

  def pre_process_models(
      self,
      student_model: nnx.Module,
      teacher_model: nnx.Module,
  ) -> tuple[nnx.Module, nnx.Module]:
    """Pre-processes the models to prepare for distillation.

    This method is called before the distillation process. It can be used to
    modify the models to prepare them for distillation.

    Args:
      student_model: Original student model.
      teacher_model: Original teacher model.

    Returns:
      Student model processed for distillation.
      Teacher model processed for distillation.
    """
    return student_model, teacher_model

  def post_process_models(
      self,
      student_model: nnx.Module,
      teacher_model: nnx.Module,
  ) -> tuple[nnx.Module, nnx.Module]:
    """Post-processes the models after distillation.

    This method usually reverts the changes made in `pre_process_models`.

    Args:
      student_model: Pre processed student model.
      teacher_model: Pre processed teacher model.

    Returns:
      Original student model.
      Original processed teacher model.
    """
    return student_model, teacher_model

  def get_teacher_outputs(
      self,
      teacher_model: nnx.Module,
      inputs: dict[str, jax.Array],
  ) -> Any:
    """Computes the teacher model outputs."""
    output = self._teacher_forward_fn(teacher_model, **inputs)
    output = jax.lax.stop_gradient(output)
    return output

  def get_student_outputs(
      self,
      student_model: nnx.Module,
      inputs: dict[str, jax.Array],
  ) -> Any:
    """Computes the student model outputs."""
    return self._student_forward_fn(student_model, **inputs)

  def get_train_loss(
      self,
      student_model: nnx.Module,
      teacher_output: Any,
      inputs: dict[str, jax.Array],
  ) -> jax.Array:
    """Computes the distillation loss."""
    student_output = self.get_student_outputs(student_model, inputs)
    labels = self._labels_fn(**inputs)
    return self.compute_loss(
        student_output=student_output,
        teacher_output=teacher_output,
        labels=labels,
    )

  def get_eval_loss(
      self,
      student_model: nnx.Module,
      inputs: dict[str, jax.Array],
  ) -> jax.Array:
    """Computes the task loss based on student model forward pass and labels."""
    student_output = self.get_student_outputs(student_model, inputs)
    student_output = jax.lax.stop_gradient(student_output)
    labels = self._labels_fn(**inputs)
    return self.compute_eval_loss(
        student_output=student_output,
        labels=labels,
    )
