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

"""Tests for distillation strategies.

Smoke tests for:
- LogitStrategy
- FeaturePoolingStrategy
- FeatureProjectionStrategy
- ContrastiveRepresentationDistillationStrategy (CRD)
"""

from __future__ import annotations
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from tunix.distillation.feature_extraction import sowed_module
from tunix.distillation.strategies.crd_strategy import (
    ContrastiveRepresentationDistillationStrategy,
)
from tunix.distillation.strategies.feature_pooling import FeaturePoolingStrategy
from tunix.distillation.strategies.feature_projection import FeatureProjectionStrategy
from tunix.distillation.strategies.logit import LogitStrategy

# Tiny toy models for testing.
class _FeatureLayer(nnx.Module):
    def __init__(self, in_dim: int, feat_dim: int, *, rngs: nnx.Rngs):
        self.proj = nnx.Linear(in_dim, feat_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.tanh(self.proj(x))


class _ToyClassifier(nnx.Module):
    def __init__(self, in_dim: int, feat_dim: int, num_classes: int, *, rngs: nnx.Rngs):
        self.feature = _FeatureLayer(in_dim, feat_dim, rngs=rngs)
        self.head = nnx.Linear(feat_dim, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.feature(x)
        return self.head(h)


def _forward_logits(model: nnx.Module, x: jax.Array, labels: jax.Array) -> jax.Array:
    del labels
    return model(x)


def _forward_logits_and_features(model: nnx.Module, x: jax.Array, labels: jax.Array):
    del labels
    # For FeatureProjectionStrategy, processed student model returns:
    # (logits, projected_features)
    return model(x)


def _forward_logits_and_embedding(model: nnx.Module, x: jax.Array, labels: jax.Array):
    del labels
    # For CRD, processed student model returns:
    # (logits, z_student)
    return model(x)


def _labels_passthrough(x: jax.Array, labels: jax.Array) -> jax.Array:
    del x
    return labels


def _has_sowed_modules(model: nnx.Module) -> bool:
    return any(isinstance(m, sowed_module.SowedModule) for _, m in model.iter_modules())


def _assert_no_sowed_modules(testcase: absltest.TestCase, model: nnx.Module):
    for _, m in model.iter_modules():
        testcase.assertNotIsInstance(m, sowed_module.SowedModule)


def _assert_teacher_output_has_batch(
    testcase: absltest.TestCase,
    teacher_out: Any,
    batch_size: int,
):
    """Accept either stacked-array teacher outputs or PyTree sowed state."""
    # Case 1: Array-like output with .shape (e.g., (N, B, ...)).
    if hasattr(teacher_out, "shape"):
        shape = teacher_out.shape
        testcase.assertGreaterEqual(len(shape), 2)
        testcase.assertEqual(shape[1], batch_size)
        return

    # Case 2: PyTree output (e.g., sowed state). Leaves should be (B, ...).
    leaves = list(jax.tree.leaves(teacher_out))
    testcase.assertNotEmpty(leaves)
    for leaf in leaves:
        testcase.assertTrue(hasattr(leaf, "shape"))
        testcase.assertGreaterEqual(len(leaf.shape), 1)
        testcase.assertEqual(leaf.shape[0], batch_size)

class StrategiesTest(parameterized.TestCase):
    def test_logit_strategy_matches_manual_kl(self):
        # Sanity check: compare to a manual computation of the same formula.
        temperature = 2.0
        alpha = 0.7
        strat = LogitStrategy(
            _forward_logits,
            _forward_logits,
            _labels_passthrough,
            temperature=temperature,
            alpha=alpha,
        )

        student_logits = jnp.array([[1.0, 0.0, -1.0]], dtype=jnp.float32)
        teacher_logits = jnp.array([[0.5, 0.25, -0.75]], dtype=jnp.float32)
        labels = jax.nn.one_hot(jnp.array([0]), 3)

        loss = strat.compute_loss(student_logits, teacher_logits, labels)
        self.assertEqual(loss.shape, ())

        log_student_probs_temp = jax.nn.log_softmax(
            student_logits / temperature, axis=-1
        )
        teacher_probs_temp = jax.nn.softmax(teacher_logits / temperature, axis=-1)
        kl = optax.kl_divergence(log_student_probs_temp, teacher_probs_temp) * (
            temperature**2
        )
        distill_loss = jnp.mean(kl)
        task_loss = jnp.mean(optax.softmax_cross_entropy(student_logits, labels))
        expected = alpha * distill_loss + (1.0 - alpha) * task_loss

        np.testing.assert_allclose(np.asarray(loss), np.asarray(expected), atol=1e-6)

    def test_feature_pooling_strategy_smoke(self):
        batch_size = 2
        in_dim = 4
        num_classes = 3

        student = _ToyClassifier(
            in_dim, feat_dim=4, num_classes=num_classes, rngs=nnx.Rngs(0)
        )
        teacher = _ToyClassifier(
            in_dim, feat_dim=8, num_classes=num_classes, rngs=nnx.Rngs(1)
        )

        strat = FeaturePoolingStrategy(
            student_forward_fn=_forward_logits,
            teacher_forward_fn=_forward_logits,
            labels_fn=_labels_passthrough,
            feature_layer=_FeatureLayer,
            alpha=0.5,
        )

        student_p, teacher_p = strat.pre_process_models(student, teacher)
        self.assertTrue(_has_sowed_modules(student_p))
        self.assertTrue(_has_sowed_modules(teacher_p))

        x = (
            jnp.arange(batch_size * in_dim, dtype=jnp.float32).reshape(
                batch_size, in_dim
            )
            / 10.0
        )
        labels = jax.nn.one_hot(jnp.array([0, 2]), num_classes)

        teacher_out = strat.get_teacher_outputs(teacher_p, {"x": x, "labels": labels})
        _assert_teacher_output_has_batch(self, teacher_out, batch_size)

        loss = strat.get_train_loss(student_p, teacher_out, {"x": x, "labels": labels})
        self.assertEqual(loss.shape, ())

        eval_loss = strat.get_eval_loss(student_p, {"x": x, "labels": labels})
        self.assertEqual(eval_loss.shape, ())

        student_o, teacher_o = strat.post_process_models(student_p, teacher_p)
        _assert_no_sowed_modules(self, student_o)
        _assert_no_sowed_modules(self, teacher_o)

    def test_feature_projection_strategy_smoke(self):
        # Keep training batch == dummy batch in this test.
        batch_size = 2
        in_dim = 4
        num_classes = 3

        student = _ToyClassifier(
            in_dim, feat_dim=4, num_classes=num_classes, rngs=nnx.Rngs(0)
        )
        teacher = _ToyClassifier(
            in_dim, feat_dim=8, num_classes=num_classes, rngs=nnx.Rngs(1)
        )

        dummy_x = jnp.ones((batch_size, in_dim), dtype=jnp.float32)

        strat = FeatureProjectionStrategy(
            student_forward_fn=_forward_logits_and_features,
            teacher_forward_fn=_forward_logits,
            labels_fn=_labels_passthrough,
            feature_layer=_FeatureLayer,
            dummy_input={"x": dummy_x},
            rngs=nnx.Rngs(42),
            alpha=0.5,
        )

        student_p, teacher_p = strat.pre_process_models(student, teacher)
        self.assertTrue(_has_sowed_modules(student_p))
        self.assertTrue(_has_sowed_modules(teacher_p))

        x = (
            jnp.arange(batch_size * in_dim, dtype=jnp.float32).reshape(
                batch_size, in_dim
            )
            + 1.0
        ) / 10.0
        labels = jax.nn.one_hot(jnp.array([1, 0]), num_classes)

        teacher_out = strat.get_teacher_outputs(teacher_p, {"x": x, "labels": labels})
        _assert_teacher_output_has_batch(self, teacher_out, batch_size)

        loss = strat.get_train_loss(student_p, teacher_out, {"x": x, "labels": labels})
        self.assertEqual(loss.shape, ())

        student_o, teacher_o = strat.post_process_models(student_p, teacher_p)
        self.assertIsInstance(student_o, _ToyClassifier)
        self.assertIsInstance(teacher_o, _ToyClassifier)
        _assert_no_sowed_modules(self, student_o)
        _assert_no_sowed_modules(self, teacher_o)

    def test_contrastive_representation_distillation_strategy_smoke(self):
        batch_size = 2
        in_dim = 4
        num_classes = 3

        student = _ToyClassifier(
            in_dim, feat_dim=4, num_classes=num_classes, rngs=nnx.Rngs(0)
        )
        teacher = _ToyClassifier(
            in_dim, feat_dim=8, num_classes=num_classes, rngs=nnx.Rngs(1)
        )

        dummy_x = jnp.ones((batch_size, in_dim), dtype=jnp.float32)

        strat = ContrastiveRepresentationDistillationStrategy(
            student_forward_fn=_forward_logits_and_embedding,
            teacher_forward_fn=_forward_logits,
            labels_fn=_labels_passthrough,
            student_layer_to_capture=_FeatureLayer,
            teacher_layer_to_capture=_FeatureLayer,
            dummy_student_input={"x": dummy_x},
            dummy_teacher_input={"x": dummy_x},
            rngs=nnx.Rngs(42),
            embedding_dim=16,
            mlp_hidden_dim=32,
            temperature=0.2,
            alpha=0.5,
            symmetric=False,
        )

        student_p, teacher_p = strat.pre_process_models(student, teacher)
        self.assertTrue(_has_sowed_modules(student_p))
        self.assertTrue(_has_sowed_modules(teacher_p))

        x = (
            jnp.arange(batch_size * in_dim, dtype=jnp.float32).reshape(
                batch_size, in_dim
            )
            + 1.0
        ) / 10.0
        labels = jax.nn.one_hot(jnp.array([1, 0]), num_classes)

        teacher_out = strat.get_teacher_outputs(teacher_p, {"x": x, "labels": labels})
        _assert_teacher_output_has_batch(self, teacher_out, batch_size)

        loss = strat.get_train_loss(student_p, teacher_out, {"x": x, "labels": labels})
        self.assertEqual(loss.shape, ())

        eval_loss = strat.get_eval_loss(student_p, {"x": x, "labels": labels})
        self.assertEqual(eval_loss.shape, ())

        student_o, teacher_o = strat.post_process_models(student_p, teacher_p)
        self.assertIsInstance(student_o, _ToyClassifier)
        self.assertIsInstance(teacher_o, _ToyClassifier)
        _assert_no_sowed_modules(self, student_o)
        _assert_no_sowed_modules(self, teacher_o)

if __name__ == "__main__":
    absltest.main()