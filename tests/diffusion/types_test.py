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

"""Tests for framework-neutral diffusion contracts."""

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import jax.sharding as shd
import numpy as np
from tunix.diffusion import interfaces
from tunix.diffusion import types
from tunix.sft import sharding_utils


def _batch(
    *,
    model_inputs=None,
    target_ids=None,
    loss_weights=None,
) -> types.DiffusionTokenBatch:
  if model_inputs is None:
    model_inputs = {
        "input_tokens": jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32),
        "noise_levels": jnp.array([0.25, 0.75], dtype=jnp.float32),
    }
  if target_ids is None:
    target_ids = jnp.array([[2, 3, 4], [5, 6, 7]], dtype=jnp.int32)
  if loss_weights is None:
    loss_weights = jnp.ones((2, 3), dtype=jnp.float32)
  return types.DiffusionTokenBatch.create(
      model_inputs=model_inputs,
      target_ids=target_ids,
      loss_weights=loss_weights,
  )


class _ToyModel(nnx.Module):

  def __init__(self, vocab_size: int):
    self.bias = nnx.Param(jnp.arange(vocab_size, dtype=jnp.float32))

  def score(self, input_tokens: jax.Array) -> jax.Array:
    bias = self.bias[...]
    return jax.nn.one_hot(input_tokens, bias.shape[0]) + bias


def _score_fn(model: nnx.Module, model_inputs: types.ModelInputs) -> jax.Array:
  return model.score(model_inputs["input_tokens"])


class DiffusionTokenBatchTest(absltest.TestCase):

  def test_requires_rank_two_integer_targets(self):
    with self.assertRaisesRegex(ValueError, "shape \\[batch, length\\]"):
      _batch(target_ids=jnp.ones((2, 3, 1), dtype=jnp.int32))
    with self.assertRaisesRegex(TypeError, "integer dtype"):
      _batch(target_ids=jnp.ones((2, 3), dtype=jnp.float32))

  def test_requires_matching_real_loss_weights(self):
    with self.assertRaisesRegex(ValueError, "must match target_ids shape"):
      _batch(loss_weights=jnp.ones((2, 2), dtype=jnp.float32))
    with self.assertRaisesRegex(TypeError, "real numeric or boolean"):
      _batch(loss_weights=jnp.ones((2, 3), dtype=jnp.complex64))

  def test_accepts_boolean_and_integer_loss_weights(self):
    self.assertEqual(
        _batch(loss_weights=jnp.ones((2, 3), dtype=bool)).loss_weights.dtype,
        jnp.bool_,
    )
    self.assertEqual(
        _batch(
            loss_weights=jnp.ones((2, 3), dtype=jnp.int32)
        ).loss_weights.dtype,
        jnp.int32,
    )

  def test_rejects_invalid_loss_weight_values(self):
    with self.assertRaisesRegex(ValueError, "finite"):
      _batch(loss_weights=jnp.array([[jnp.nan, 1, 1], [1, 1, 1]]))
    with self.assertRaisesRegex(ValueError, "finite"):
      _batch(loss_weights=jnp.array([[jnp.inf, 1, 1], [1, 1, 1]]))
    with self.assertRaisesRegex(ValueError, "nonnegative"):
      _batch(loss_weights=jnp.array([[-1.0, 1, 1], [1, 1, 1]]))

  def test_rejects_negative_active_targets_but_allows_inactive_sentinels(self):
    target_ids = jnp.array([[-1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
    with self.assertRaisesRegex(ValueError, "active target_ids"):
      _batch(target_ids=target_ids)

    batch = _batch(
        target_ids=target_ids,
        loss_weights=jnp.array([[0, 1, 1], [1, 1, 1]], dtype=jnp.float32),
    )
    self.assertEqual(int(batch.target_ids[0, 0]), -1)

  def test_requires_nonempty_batch_major_array_inputs(self):
    with self.assertRaisesRegex(ValueError, "at least one array"):
      _batch(model_inputs={})
    with self.assertRaisesRegex(TypeError, "must be a JAX or NumPy array"):
      _batch(model_inputs={"tokens": [[1, 2, 3], [4, 5, 6]]})
    with self.assertRaisesRegex(ValueError, "not scalar"):
      _batch(model_inputs={"temperature": jnp.array(1.0)})
    with self.assertRaisesRegex(ValueError, "batch size 3; expected 2"):
      _batch(model_inputs={"tokens": jnp.ones((3, 3), dtype=jnp.int32)})
    with self.assertRaisesRegex(TypeError, "real numeric or boolean"):
      _batch(model_inputs={"tokens": np.array([["a"], ["b"]])})

  def test_is_a_jax_pytree_and_jittable(self):
    batch = _batch()
    leaves, tree_def = jax.tree.flatten(batch)

    self.assertLen(leaves, 4)
    restored = jax.tree.unflatten(tree_def, leaves)
    np.testing.assert_array_equal(restored.target_ids, batch.target_ids)

    @jax.jit
    def weighted_targets(value):
      return value.target_ids * value.loss_weights

    np.testing.assert_array_equal(
        weighted_targets(batch), batch.target_ids.astype(jnp.float32)
    )

  def test_array_leaves_are_compatible_with_data_sharding(self):
    batch = _batch()
    mesh = shd.Mesh(np.array(jax.devices()[:1]), ("data",))

    with mesh:
      sharded_batch = sharding_utils.shard_input(batch, ("data",))

    for leaf in jax.tree.leaves(sharded_batch):
      self.assertIsInstance(leaf, jax.Array)
      self.assertEqual(leaf.sharding.spec, shd.PartitionSpec("data"))


class DiffusionLogitsTest(absltest.TestCase):

  def test_scores_are_target_aligned_and_jittable(self):
    batch = _batch()
    model = _ToyModel(vocab_size=8)

    @nnx.jit
    def score(model, batch):
      return interfaces.compute_diffusion_logits(model, batch, _score_fn)

    scores = score(model, batch)
    self.assertEqual(scores.shape, (2, 3, 8))
    self.assertEqual(scores.dtype, jnp.float32)

  def test_rejects_non_target_aligned_scores(self):
    batch = _batch()
    with self.assertRaisesRegex(ValueError, "align with target_ids"):
      types.validate_diffusion_logits(
          batch, jnp.ones((2, 2, 8), dtype=jnp.float32)
      )
    with self.assertRaisesRegex(ValueError, "non-empty vocabulary"):
      types.validate_diffusion_logits(
          batch, jnp.ones((2, 3, 0), dtype=jnp.float32)
      )

  def test_rejects_invalid_score_rank_and_dtype(self):
    batch = _batch()
    with self.assertRaisesRegex(ValueError, "shape \\[batch, length, vocab\\]"):
      types.validate_diffusion_logits(
          batch, jnp.ones((2, 3), dtype=jnp.float32)
      )
    with self.assertRaisesRegex(TypeError, "floating-point dtype"):
      types.validate_diffusion_logits(
          batch, jnp.ones((2, 3, 8), dtype=jnp.int32)
      )

  def test_rejects_active_targets_outside_vocabulary(self):
    batch = _batch(
        target_ids=jnp.array([[2, 8, 4], [5, 6, 7]], dtype=jnp.int32)
    )
    with self.assertRaisesRegex(ValueError, "smaller than vocabulary size 8"):
      types.validate_diffusion_logits(
          batch, jnp.ones((2, 3, 8), dtype=jnp.float32)
      )

  def test_validated_scorer_rejects_misaligned_implementation(self):
    def misaligned_score_fn(model, model_inputs):
      del model
      batch_size = model_inputs["input_tokens"].shape[0]
      return jnp.ones((batch_size, 1, 8), dtype=jnp.float32)

    with self.assertRaisesRegex(ValueError, "align with target_ids"):
      interfaces.compute_diffusion_logits(
          _ToyModel(vocab_size=8), _batch(), misaligned_score_fn
      )


if __name__ == "__main__":
  absltest.main()
