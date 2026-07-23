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

"""Tests for external-teacher diffusion distillation batches."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from tunix.diffusion import types as diffusion_types
from tunix.distillation import diffusion


def _student_batch() -> diffusion_types.DiffusionTokenBatch:
  return diffusion_types.DiffusionTokenBatch.create(
      model_inputs={"example_ids": jnp.array([0], dtype=jnp.int32)},
      target_ids=jnp.array([[0, 1]], dtype=jnp.int32),
      loss_weights=jnp.ones((1, 2), dtype=jnp.float32),
  )


class DiffusionDistillationBatchTest(absltest.TestCase):

  def test_rejects_non_diffusion_student_batch(self):
    with self.assertRaisesRegex(TypeError, "must be a DiffusionTokenBatch"):
      diffusion.DiffusionDistillationBatch.create(
          student_batch={"target_ids": jnp.array([[0, 1]])},
          teacher_logits=jnp.ones((1, 2, 2), dtype=jnp.float32),
      )

  def test_rejects_misaligned_teacher_logits(self):
    with self.assertRaisesRegex(ValueError, "align with target_ids"):
      diffusion.DiffusionDistillationBatch.create(
          student_batch=_student_batch(),
          teacher_logits=jnp.ones((1, 1, 2), dtype=jnp.float32),
      )

  def test_is_a_jax_pytree_with_aligned_teacher_logits(self):
    teacher_logits = jnp.array([[[1.0, -1.0], [-0.5, 0.5]]], dtype=jnp.float32)
    batch = diffusion.DiffusionDistillationBatch.create(
        student_batch=_student_batch(),
        teacher_logits=teacher_logits,
    )

    leaves, treedef = jax.tree.flatten(batch)
    restored = jax.tree.unflatten(treedef, leaves)

    self.assertLen(leaves, 4)
    np.testing.assert_array_equal(restored.teacher_logits, teacher_logits)
    np.testing.assert_array_equal(
        restored.student_batch.target_ids, batch.student_batch.target_ids
    )


if __name__ == "__main__":
  absltest.main()
