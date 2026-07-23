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

"""Tests for prepared diffusion policy scoring."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.diffusion import types as diffusion_types
from tunix.rl import diffusion
from tunix.rl import rl_cluster
from tunix.rl.rollout import base_rollout


class _LogitModel(nnx.Module):

  def __init__(self, logits):
    self.logits = nnx.Param(logits)


def _score_fn(model, model_inputs):
  return model.logits[model_inputs["example_ids"]]


def _batch(*, target_ids, loss_weights):
  batch_size = len(target_ids)
  return diffusion_types.DiffusionTokenBatch.create(
      model_inputs={
          "example_ids": jnp.arange(batch_size, dtype=jnp.int32),
      },
      target_ids=jnp.asarray(target_ids, dtype=jnp.int32),
      loss_weights=jnp.asarray(loss_weights, dtype=jnp.float32),
  )


def _rollout_output(diffusion_batch=None):
  batch_size = (
      1 if diffusion_batch is None else diffusion_batch.target_ids.shape[0]
  )
  return base_rollout.RolloutOutput(
      text=["generated"] * batch_size,
      logits=None,
      tokens=[np.zeros((1,), dtype=np.int32)] * batch_size,
      left_padded_prompt_tokens=np.zeros((batch_size, 1), dtype=np.int32),
      logprobs=None,
      diffusion_batch=diffusion_batch,
  )


class DiffusionPolicyScoringTest(parameterized.TestCase):

  def test_returns_temperature_scaled_completion_aligned_scores_and_entropy(
      self,
  ):
    logits = jnp.array(
        [
            [[2.0, 0.0, -1.0], [0.0, 1.0, 3.0]],
            [[-2.0, 1.0, 0.0], [4.0, 0.0, -1.0]],
        ],
        dtype=jnp.float32,
    )
    batch = _batch(
        target_ids=[[0, 2], [1, 0]],
        loss_weights=[[1.0, 1.0], [1.0, 1.0]],
    )

    logps, entropy = diffusion.compute_diffusion_per_token_logps(
        _LogitModel(logits),
        batch,
        _score_fn,
        temperature=2.0,
        stop_gradient=False,
        return_entropy=True,
    )
    expected_log_probs = jax.nn.log_softmax(logits / 2.0, axis=-1)
    expected_logps = jnp.take_along_axis(
        expected_log_probs, batch.target_ids[..., None], axis=-1
    )[..., 0]
    expected_entropy = -jnp.sum(
        jnp.exp(expected_log_probs) * expected_log_probs, axis=-1
    )

    self.assertEqual(logps.shape, batch.target_ids.shape)
    self.assertEqual(entropy.shape, batch.target_ids.shape)
    np.testing.assert_allclose(logps, expected_logps, rtol=1e-6)
    np.testing.assert_allclose(entropy, expected_entropy, rtol=1e-6)

  def test_inactive_positions_sanitize_sentinel_targets_and_nonfinite_logits(
      self,
  ):
    logits = jnp.array(
        [[[2.0, -1.0], [jnp.nan, jnp.inf], [jnp.inf, -jnp.inf]]],
        dtype=jnp.float32,
    )
    batch = _batch(
        target_ids=[[0, -1, 99]],
        loss_weights=[[1.0, 0.0, 0.0]],
    )

    logps, entropy = diffusion.compute_diffusion_per_token_logps(
        _LogitModel(logits),
        batch,
        _score_fn,
        return_entropy=True,
    )

    self.assertTrue(bool(jnp.all(jnp.isfinite(logps))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(entropy))))
    np.testing.assert_array_equal(logps[0, 1:], jnp.zeros((2,)))
    np.testing.assert_array_equal(entropy[0, 1:], jnp.zeros((2,)))

  def test_entropy_remains_finite_when_active_logits_mask_a_token(self):
    logits = jnp.array([[[2.0, 0.0, -jnp.inf]]], dtype=jnp.float32)
    batch = _batch(target_ids=[[0]], loss_weights=[[1.0]])

    logps, entropy = diffusion.compute_diffusion_per_token_logps(
        _LogitModel(logits),
        batch,
        _score_fn,
        return_entropy=True,
    )

    self.assertTrue(bool(jnp.all(jnp.isfinite(logps))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(entropy))))

  def test_stop_gradient_controls_policy_gradient(self):
    batch = _batch(
        target_ids=[[0, 1]],
        loss_weights=[[1.0, 1.0]],
    )

    def score_sum(model, stop_gradient):
      return jnp.sum(
          diffusion.compute_diffusion_per_token_logps(
              model,
              batch,
              _score_fn,
              stop_gradient=stop_gradient,
          )
      )

    trainable_model = _LogitModel(
        jnp.array([[[1.0, -1.0], [-1.0, 1.0]]], dtype=jnp.float32)
    )
    detached_model = _LogitModel(jnp.copy(trainable_model.logits[...]))
    trainable_grads = nnx.grad(lambda model: score_sum(model, False))(
        trainable_model
    )
    detached_grads = nnx.grad(lambda model: score_sum(model, True))(
        detached_model
    )

    self.assertGreater(float(jnp.linalg.norm(trainable_grads.logits[...])), 0)
    np.testing.assert_array_equal(
        detached_grads.logits[...], jnp.zeros_like(detached_grads.logits[...])
    )

  @parameterized.parameters(0.0, -1.0, float("inf"), float("nan"), True, "1")
  def test_rejects_invalid_temperature(self, temperature):
    model = _LogitModel(jnp.zeros((1, 1, 2), dtype=jnp.float32))
    batch = _batch(target_ids=[[0]], loss_weights=[[1.0]])

    with self.assertRaisesRegex(ValueError, "temperature"):
      diffusion.compute_diffusion_per_token_logps(
          model, batch, _score_fn, temperature=temperature
      )


class DiffusionRolloutMergeTest(absltest.TestCase):

  def test_merges_prepared_batches_on_leading_dimension(self):
    first = _batch(target_ids=[[1, 2]], loss_weights=[[1.0, 1.0]])
    second = _batch(target_ids=[[3, 4]], loss_weights=[[1.0, 0.0]])

    merged = rl_cluster._merge_diffusion_batches(  # pylint: disable=protected-access
        [_rollout_output(first), _rollout_output(second)]
    )

    self.assertIsNotNone(merged)
    np.testing.assert_array_equal(merged.target_ids, [[1, 2], [3, 4]])
    np.testing.assert_array_equal(merged.loss_weights, [[1.0, 1.0], [1.0, 0.0]])
    np.testing.assert_array_equal(merged.model_inputs["example_ids"], [0, 0])

  def test_preserves_host_numpy_batches_until_role_mesh_sharding(self):
    def host_batch(target_id):
      return diffusion_types.DiffusionTokenBatch.create(
          model_inputs={
              "trace_tokens": np.array([[target_id, 99]], dtype=np.int32),
              "action_steps": np.array([[0, 1]], dtype=np.int32),
          },
          target_ids=np.array([[target_id, -1]], dtype=np.int32),
          loss_weights=np.array([[1.0, 0.0]], dtype=np.float32),
      )

    merged = rl_cluster._merge_diffusion_batches(  # pylint: disable=protected-access
        [_rollout_output(host_batch(1)), _rollout_output(host_batch(2))]
    )

    self.assertIsInstance(merged.target_ids, np.ndarray)
    self.assertIsInstance(merged.loss_weights, np.ndarray)
    self.assertIsInstance(merged.model_inputs["trace_tokens"], np.ndarray)
    self.assertIsInstance(merged.model_inputs["action_steps"], np.ndarray)
    np.testing.assert_array_equal(
        merged.model_inputs["trace_tokens"], [[1, 99], [2, 99]]
    )

  def test_allows_ar_microbatches_to_omit_diffusion_metadata(self):
    self.assertIsNone(
        rl_cluster._merge_diffusion_batches(  # pylint: disable=protected-access
            [_rollout_output(), _rollout_output()]
        )
    )

  def test_rejects_mixed_diffusion_metadata(self):
    batch = _batch(target_ids=[[1]], loss_weights=[[1.0]])

    with self.assertRaisesRegex(ValueError, "either all provide"):
      rl_cluster._merge_diffusion_batches(  # pylint: disable=protected-access
          [_rollout_output(batch), _rollout_output()]
      )

  def test_rejects_different_model_input_pytrees(self):
    first = _batch(target_ids=[[1]], loss_weights=[[1.0]])
    second = diffusion_types.DiffusionTokenBatch.create(
        model_inputs={"other_ids": jnp.array([0], dtype=jnp.int32)},
        target_ids=jnp.array([[2]], dtype=jnp.int32),
        loss_weights=jnp.array([[1.0]], dtype=jnp.float32),
    )

    with self.assertRaisesRegex(ValueError, "same pytree structure"):
      rl_cluster._merge_diffusion_batches(  # pylint: disable=protected-access
          [_rollout_output(first), _rollout_output(second)]
      )

  def test_rejects_batch_size_that_does_not_match_rollout(self):
    batch = _batch(
        target_ids=[[1], [2]],
        loss_weights=[[1.0], [1.0]],
    )
    output = _rollout_output(batch)
    output.text = ["only one generated example"]

    with self.assertRaisesRegex(ValueError, "batch size must match"):
      rl_cluster._merge_diffusion_batches(  # pylint: disable=protected-access
          [output]
      )


if __name__ == "__main__":
  absltest.main()
