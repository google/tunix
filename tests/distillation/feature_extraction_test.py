# Copyright 2025 Google LLC
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

"""Tests for distillation feature extraction helpers.

Covers:
- Avg-pooling array utilities (VALID vs SAME and pad-count behavior)
- Sowed module wrap/pop/unwrap behavior
- Feature projection setup/removal integration
"""

from __future__ import annotations
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from tunix.distillation import feature_extraction
from tunix.distillation.feature_extraction import pooling
from tunix.distillation.feature_extraction import sowed_module

class _FeatureLayer(nnx.Module):
    """A tiny feature block used in toy models for unit testing."""

    def __init__(self, in_dim: int, feat_dim: int, *, rngs: nnx.Rngs):
        self._proj = nnx.Linear(in_dim, feat_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.relu(self._proj(x))


class _ToyClassifier(nnx.Module):
    """Toy classifier: FeatureLayer -> Linear head."""
    def __init__(
        self,
        in_dim: int,
        feat_dim: int,
        num_classes: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.feature = _FeatureLayer(in_dim, feat_dim, rngs=rngs)
        self.head = nnx.Linear(feat_dim, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.head(self.feature(x))


class _ToyDeepClassifier(nnx.Module):
    """Toy classifier with two feature layers: FeatureLayer -> FeatureLayer -> head.
    Used to test multi-leaf sowing behavior.
    """
    def __init__(
        self,
        in_dim: int,
        feat_dim1: int,
        feat_dim2: int,
        num_classes: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.feature1 = _FeatureLayer(in_dim, feat_dim1, rngs=rngs)
        self.feature2 = _FeatureLayer(feat_dim1, feat_dim2, rngs=rngs)
        self.head = nnx.Linear(feat_dim2, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.feature1(x)
        x = self.feature2(x)
        return self.head(x)


def _pop_leaves(model: nnx.Module) -> list[jax.Array]:
    """Pop sowed outputs and return leaves as a Python list."""
    state = sowed_module.pop_sowed_intermediate_outputs(model)
    return list(jax.tree.leaves(state)) if state else []


class FeatureExtractionTest(parameterized.TestCase):
    """Unit tests for feature extraction and sowed module helpers."""
    def test_avg_pool_valid_1d_exact(self):
        # Shape (1, B, D) so pooling touches only the last axis.
        x = jnp.arange(1, 1 + 1 * 2 * 6, dtype=jnp.float32).reshape(1, 2, 6)
        y = feature_extraction.avg_pool_array_to_target_shape(
            x,
            target_shape=(1, 2, 2),
            padding_mode=pooling.PaddingMode.VALID,
        )
        self.assertEqual(y.shape, (1, 2, 2))

        # For D=6 -> target=2: stride=3, window=3 => chunk means.
        x_np = np.asarray(x)
        expected = np.stack(
            [
                x_np[:, :, 0:3].mean(axis=-1),
                x_np[:, :, 3:6].mean(axis=-1),
            ],
            axis=-1,
        )
        np.testing.assert_allclose(np.asarray(y), expected, atol=1e-6)

    def test_avg_pool_same_include_pad_changes_result(self):
        x = jnp.array([1, 2, 3, 4, 5], dtype=jnp.float32).reshape(1, 1, 5)

        y_exclude = feature_extraction.avg_pool_array_to_target_shape(
            x,
            target_shape=(1, 1, 2),
            padding_mode=pooling.PaddingMode.SAME,
            count_include_pad_for_same_padding=False,
        )
        y_include = feature_extraction.avg_pool_array_to_target_shape(
            x,
            target_shape=(1, 1, 2),
            padding_mode=pooling.PaddingMode.SAME,
            count_include_pad_for_same_padding=True,
        )

        self.assertEqual(y_exclude.shape, (1, 1, 2))
        self.assertEqual(y_include.shape, (1, 1, 2))

        # window1: [1,2,3] => 2
        # window2: [4,5,0]
        #   exclude pad => (4+5)/2 = 4.5
        #   include pad => (4+5+0)/3 = 3
        np.testing.assert_allclose(np.asarray(y_exclude)[0, 0, 0], 2.0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(y_exclude)[0, 0, 1], 4.5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(y_include)[0, 0, 0], 2.0, atol=1e-6)
        np.testing.assert_allclose(np.asarray(y_include)[0, 0, 1], 3.0, atol=1e-6)

    def test_avg_pool_rank_mismatch_raises(self):
        x = jnp.ones((2, 3), dtype=jnp.float32)
        with self.assertRaises(ValueError):
            feature_extraction.avg_pool_array_to_target_shape(x, target_shape=(2, 3, 1))

    def test_avg_pool_invalid_target_dim_raises(self):
        x = jnp.ones((2, 3), dtype=jnp.float32)
        with self.assertRaises(ValueError):
            feature_extraction.avg_pool_array_to_target_shape(x, target_shape=(2, 4))

    def test_wrap_pop_unwrap_sowed_modules(self):
        rngs = nnx.Rngs(0)
        model = _ToyClassifier(in_dim=4, feat_dim=8, num_classes=3, rngs=rngs)
        x = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)

        original_feature_module = model.feature
        sowed_module.wrap_model_with_sowed_modules(model, [_FeatureLayer])

        _ = model(x)

        captured = sowed_module.pop_sowed_intermediate_outputs(model)
        leaves = jax.tree.leaves(captured)
        self.assertLen(leaves, 1)

        captured_feat = leaves[0]
        expected_feat = original_feature_module(x)
        np.testing.assert_allclose(
            np.asarray(captured_feat),
            np.asarray(expected_feat),
            atol=1e-6,
        )

        sowed_module.unwrap_sowed_modules(model)
        self.assertIs(model.feature, original_feature_module)

    def test_wrap_pop_unwrap_sowed_modules_multiple_leaves(self):
        rngs = nnx.Rngs(0)
        model = _ToyDeepClassifier(
            in_dim=4, feat_dim1=6, feat_dim2=8, num_classes=3, rngs=rngs
        )
        x = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)

        # Baseline forward.
        baseline_logits = model(x)

        # Wrap both _FeatureLayer instances.
        sowed_module.wrap_model_with_sowed_modules(model, [_FeatureLayer])

        logits = model(x)
        np.testing.assert_allclose(
            np.asarray(logits), np.asarray(baseline_logits), atol=1e-6
        )

        leaves = _pop_leaves(model)
        self.assertLen(leaves, 2)

        expected1 = model.feature1(x)
        expected2 = model.feature2(expected1)

        leaves_np = [np.asarray(a) for a in leaves]
        exp_np = [np.asarray(expected1), np.asarray(expected2)]

        def _matches_any(arr, candidates):
            return any(
                (arr.shape == c.shape) and np.allclose(arr, c, atol=1e-6)
                for c in candidates
            )

        self.assertTrue(_matches_any(exp_np[0], leaves_np), "feature1 not captured")
        self.assertTrue(_matches_any(exp_np[1], leaves_np), "feature2 not captured")

        sowed_module.unwrap_sowed_modules(model)
        for _, m in model.iter_modules():
            self.assertNotIsInstance(m, sowed_module.SowedModule)

    def test_setup_and_remove_feature_projection(self):
        batch_size = 2
        in_dim = 4
        num_classes = 3

        student = _ToyClassifier(
            in_dim=in_dim,
            feat_dim=4,
            num_classes=num_classes,
            rngs=nnx.Rngs(0),
        )
        teacher = _ToyClassifier(
            in_dim=in_dim,
            feat_dim=8,
            num_classes=num_classes,
            rngs=nnx.Rngs(1),
        )

        dummy_x = jnp.ones((batch_size, in_dim), dtype=jnp.float32)

        student_wrapped, teacher_wrapped = (
            feature_extraction.setup_models_with_feature_projection(
                student_model=student,
                teacher_model=teacher,
                student_layer_to_capture=_FeatureLayer,
                teacher_layer_to_capture=_FeatureLayer,
                dummy_student_input={"x": dummy_x},
                dummy_teacher_input={"x": dummy_x},
                rngs=nnx.Rngs(42),
            )
        )

        self.assertIsInstance(
            student_wrapped, feature_extraction.ModelWithFeatureProjection
        )

        logits, projected = student_wrapped(dummy_x)
        self.assertEqual(logits.shape, (batch_size, num_classes))

        _ = teacher_wrapped(dummy_x)
        teacher_state = sowed_module.pop_sowed_intermediate_outputs(teacher_wrapped)
        teacher_feats = jnp.stack(jax.tree.leaves(teacher_state))
        self.assertEqual(projected.shape, teacher_feats.shape)

        student_orig, teacher_orig = (
            feature_extraction.remove_feature_projection_from_models(
                student_wrapped, teacher_wrapped
            )
        )
        self.assertIsInstance(student_orig, _ToyClassifier)
        self.assertIsInstance(teacher_orig, _ToyClassifier)

        for _, m in student_orig.iter_modules():
            self.assertNotIsInstance(m, sowed_module.SowedModule)
        for _, m in teacher_orig.iter_modules():
            self.assertNotIsInstance(m, sowed_module.SowedModule)


if __name__ == "__main__":
    absltest.main()