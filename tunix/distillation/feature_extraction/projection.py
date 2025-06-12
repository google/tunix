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

"""Utilities for feature projection for distillation."""

from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.distillation.feature_extraction import sowed_module


class ModelWithFeatureProjection(nnx.Module):
  """Wraps a model with a projection layer for extracted features.

  This module takes a base model, extracts its intermediate features using
  `sowed_module`, and then applies a linear projection to these features.
  """

  def __init__(
      self,
      model: nnx.Module,
      feature_shape: int | tuple[int, ...],
      feature_target_shape: int | tuple[int, ...],
      *,
      rngs: nnx.Rngs,
  ):
    """Initializes a ModelWithFeatureProjection.

    Args:
      model: The base sowed model to wrap.
      feature_shape: The shape of the features that will be extracted.
      feature_target_shape: The target shape of the features to project.
      rngs: The random number generator.
    """
    if isinstance(feature_shape, int):
      feature_shape = (feature_shape,)
    self.model = model
    self.projection_layer = nnx.LinearGeneral(
        feature_shape,
        feature_target_shape,
        axis=np.arange(len(feature_shape)),
        rngs=rngs,
    )

  def __call__(self, *args, **kwargs):
    output = self.model(*args, **kwargs)
    model_features = sowed_module.pop_sowed_intermediate_outputs(self.model)
    model_features = jnp.stack(jax.tree.leaves(model_features))
    projected_features = self.projection_layer(model_features)
    return output, projected_features


def setup_models_with_feature_projection(
    student_model: nnx.Module,
    teacher_model: nnx.Module,
    student_layer_to_capture: type[nnx.Module],
    teacher_layer_to_capture: type[nnx.Module],
    dummy_student_input: dict[str, Any],
    dummy_teacher_input: dict[str, Any],
    *,
    rngs: nnx.Rngs,
) -> tuple[ModelWithFeatureProjection, nnx.Module]:
  """Builds a student model with feature projection.

  This function takes a student and a teacher model, wraps them with sowed
  modules to capture intermediate features, and then builds a
  `ModelWithFeatureProjection` to project the student's features to match the
  teacher's features.

  Args:
    student_model: The student model.
    teacher_model: The teacher model.
    student_layer_to_capture: The type of layer to capture in the student model.
    teacher_layer_to_capture: The type of layer to capture in the teacher model.
    dummy_student_input: A dummy input for the student model.
    dummy_teacher_input: A dummy input for the teacher model.
    rngs: The random number generator.

  Returns:
    A tuple containing:
      - The student model with feature projection.
      - The teacher model.
  """
  sowed_module.wrap_model_with_sowed_modules(
      student_model, [student_layer_to_capture]
  )
  sowed_module.wrap_model_with_sowed_modules(
      teacher_model, [teacher_layer_to_capture]
  )
  _ = student_model(**dummy_student_input)
  _ = teacher_model(**dummy_teacher_input)

  student_features = sowed_module.pop_sowed_intermediate_outputs(student_model)
  student_features = jnp.stack(jax.tree.leaves(student_features))

  teacher_features = sowed_module.pop_sowed_intermediate_outputs(teacher_model)
  teacher_features = jnp.stack(jax.tree.leaves(teacher_features))

  student_model_with_feature_projection = ModelWithFeatureProjection(
      student_model,
      student_features.shape,
      teacher_features.shape,
      rngs=rngs,
  )
  return student_model_with_feature_projection, teacher_model


def remove_feature_projection_from_models(
    student_model: nnx.Module, teacher_model: nnx.Module
) -> tuple[nnx.Module, nnx.Module]:
  """Returns original models from models modified with feature projection setup.

  Args:
    student_model: The student model with feature projection setup.
    teacher_model: The teacher model with feature projection setup.

  Returns:
    A tuple containing:
      - The original student model.
      - The original teacher model.
  """
  if isinstance(student_model, ModelWithFeatureProjection):
    student_model = student_model.model
  sowed_module.unwrap_sowed_modules(student_model)
  sowed_module.unwrap_sowed_modules(teacher_model)
  return student_model, teacher_model
