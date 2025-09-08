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

"""Unit tests for wrapping models with feature projection."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy.testing as npt
from tunix.distillation.feature_extraction import projection
from tunix.distillation.feature_extraction import sowed_module


class DummyModel(nnx.Module):

  def __init__(self, in_features=3, out_features=10):
    if isinstance(in_features, int):
      in_features = (in_features,)
    self.linear = nnx.LinearGeneral(
        in_features=in_features,
        out_features=out_features,
        axis=tuple(range(len(in_features))),
        rngs=nnx.Rngs(0),
    )

  def __call__(self, x):
    x = self.linear(x)
    return x


class TestModelWithFeatureProjection(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="1_to_1",
          student_features=1,
          teacher_features=1,
      ),
      dict(
          testcase_name="1_to_5",
          student_features=1,
          teacher_features=5,
      ),
      dict(
          testcase_name="10_to_10",
          student_features=10,
          teacher_features=10,
      ),
      dict(
          testcase_name="10_to_23",
          student_features=10,
          teacher_features=23,
      ),
      dict(
          testcase_name="10,12_to_23",
          student_features=(10, 12),
          teacher_features=23,
      ),
      dict(
          testcase_name="10,12_to_23,15,19",
          student_features=(10, 12),
          teacher_features=(23, 15, 19),
      ),
      dict(
          testcase_name="10,12,13_to_23,15",
          student_features=(10, 12, 13),
          teacher_features=(23, 15),
      ),
  )
  def test_setup_models_with_feature_projection(
      self, student_features, teacher_features
  ):
    student_model = DummyModel(
        in_features=student_features, out_features=student_features
    )
    teacher_model = DummyModel(
        in_features=teacher_features, out_features=teacher_features
    )
    dummy_input_student = {"x": jnp.ones(student_features)}
    dummy_input_teacher = {"x": jnp.ones(teacher_features)}
    original_student_model_linear = student_model.linear
    original_teacher_model_linear = teacher_model.linear
    student_logits = student_model(dummy_input_student["x"])

    student_model_with_feature_projection, teacher_model = (
        projection.setup_models_with_feature_projection(
            student_model,
            teacher_model,
            nnx.LinearGeneral,
            nnx.LinearGeneral,
            dummy_input_student,
            dummy_input_teacher,
            rngs=nnx.Rngs(0),
        )
    )
    new_student_logits, student_features = (
        student_model_with_feature_projection(dummy_input_student["x"])
    )
    _ = teacher_model(dummy_input_teacher["x"])
    teacher_features = jnp.stack(
        jax.tree.leaves(
            sowed_module.pop_sowed_intermediate_outputs(teacher_model)
        )
    )

    self.assertIsInstance(
        student_model_with_feature_projection,
        projection.ModelWithFeatureProjection,
    )
    self.assertIsInstance(teacher_model, DummyModel)
    self.assertIs(student_model_with_feature_projection.model, student_model)
    self.assertIs(
        student_model_with_feature_projection.model.linear.wrapped_model,
        original_student_model_linear,
    )
    self.assertIs(
        teacher_model.linear.wrapped_model, original_teacher_model_linear
    )
    self.assertEqual(student_features.shape, teacher_features.shape)
    npt.assert_allclose(new_student_logits, student_logits, rtol=1e-6)

  def test_remove_feature_projection_from_models(self):
    student_model = DummyModel()
    teacher_model = DummyModel(out_features=20)
    dummy_input = {"x": jnp.ones(3)}
    original_student_model_linear = student_model.linear
    original_teacher_model_linear = teacher_model.linear
    student_model_with_feature_projection, new_teacher_model = (
        projection.setup_models_with_feature_projection(
            student_model,
            teacher_model,
            nnx.LinearGeneral,
            nnx.LinearGeneral,
            dummy_input,
            dummy_input,
            rngs=nnx.Rngs(0),
        )
    )

    restored_student, restored_teacher = (
        projection.remove_feature_projection_from_models(
            student_model_with_feature_projection, new_teacher_model
        )
    )

    self.assertIs(restored_student, student_model)
    self.assertIs(restored_teacher, teacher_model)
    self.assertIs(
        restored_student.linear,
        original_student_model_linear,
    )
    self.assertIs(restored_teacher.linear, original_teacher_model_linear)


if __name__ == "__main__":
  absltest.main()
