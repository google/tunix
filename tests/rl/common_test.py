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

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import common
from tunix.tests import test_common as tc

jax.config.update("jax_threefry_partitionable", False)


class CommonTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          "kl",
          "kl",
          np.array([
              [
                  [0.38486493, 0.7469206, 0.3495195, 0.5621129],
                  [-0.4474684, -0.13095665, 0.46064317, -0.19887352],
              ],
              [
                  [0.43108296, 0.53564644, -0.25296474, 0.44137287],
                  [-0.06834459, -0.12115264, -0.61533415, 0.15468943],
              ],
          ]),
      ),
      (
          "mse_kl",
          "mse_kl",
          np.array([
              [
                  [0.07406051, 0.27894518, 0.06108194, 0.15798548],
                  [0.10011399, 0.00857482, 0.10609607, 0.01977534],
              ],
              [
                  [0.09291626, 0.14345856, 0.03199558, 0.09740501],
                  [0.00233549, 0.00733898, 0.18931806, 0.01196441],
              ],
          ]),
      ),
      (
          "low_var_kl",
          "low_var_kl",
          np.array([
              [
                  [0.0654075, 0.220744, 0.0545462, 0.1321163],
                  [0.1168784, 0.0089617, 0.0915209, 0.0211542],
              ],
              [
                  [0.080888, 0.1209372, 0.0348731, 0.0845257],
                  [0.0023897, 0.0076445, 0.2349406, 0.0113707],
              ],
          ]),
      ),
  )
  def test_compute_kl_divergence(self, method, expected_value):
    rng = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(rng)
    per_token_logps = jax.random.uniform(k1, shape=(2, 2, 4))
    ref_per_token_logps = jax.random.uniform(k2, shape=(2, 2, 4))
    kl_divergence = common.compute_kl_divergence(
        per_token_logps, ref_per_token_logps, method=method
    )
    np.testing.assert_allclose(
        kl_divergence, expected_value, atol=1e-5, rtol=1e-2
    )

  def test_selective_log_softmax(self):
    rng = jax.random.PRNGKey(0)
    logits = jax.random.uniform(rng, shape=(2, 4, 8))
    input_ids = jax.random.randint(rng, shape=(2, 4), minval=0, maxval=8)
    per_token_logps = common.selective_log_softmax(logits, input_ids)
    jitted_per_token_logps = jax.jit(common.selective_log_softmax)(
        logits, input_ids
    )
    expected_value = jnp.array([
        [-2.242679, -2.2733693, -2.1024966, -1.9994389],
        [-2.0603075, -2.4863663, -1.9176172, -2.0206313],
    ])
    np.testing.assert_allclose(
        per_token_logps, expected_value, rtol=1e-04, atol=1e-04
    )
    np.testing.assert_allclose(
        per_token_logps, jitted_per_token_logps, rtol=1e-05, atol=1e-05
    )

  def test_get_per_token_logps(self):
    rng = jax.random.PRNGKey(0)
    model = tc.ToyTransformer(rngs=nnx.Rngs(0))
    input_tokens = jax.random.randint(rng, shape=(2, 4), minval=0, maxval=8)
    positions = jnp.ones((2, 4))
    attn_mask = common.make_causal_attn_mask(positions)
    per_token_logps = common.get_per_token_logps(
        model, input_tokens, positions, attn_mask, logits_to_keep=2
    )
    np.testing.assert_allclose(
        per_token_logps,
        np.array([[-5.7448483, -5.937829], [-4.222273, -4.41953]]),
        rtol=1e-02,
        atol=1e-03,
    )

  def test_compute_per_token_logps(self):
    model = tc.ToyTransformer(rngs=nnx.Rngs(0))
    prompt_tokens = jnp.array([[1, 2, 3, 4], [0, 0, 1, 2], [0, 1, 2, 3]])
    completion_tokens = jnp.array(
        [[10, 11, -1, 12], [10, 11, 12, 13], [10, 11, 12, -1]]
    )
    per_token_logps = common.compute_per_token_logps(
        model,
        prompt_tokens,
        completion_tokens,
        pad_id=0,
        eos_id=-1,
        return_logits=False,
    )
    np.testing.assert_allclose(
        per_token_logps,
        np.array([
            [-5.876301, -8.700251, -5.046069, -5.788748],
            [-6.071025, -7.5328417, -5.9712567, -4.653783],
            [-6.039485, -8.264197, -6.2771187, -4.767109],
        ]),
        atol=1e-1,
        rtol=1e-2,
    )
    _, logits = common.compute_per_token_logps(
        model,
        prompt_tokens,
        completion_tokens,
        pad_id=0,
        eos_id=-1,
        return_logits=True,
    )
    np.testing.assert_equal(logits.shape, (3, 4, 256))

  def test_make_completion_mask(self):
    completion_ids = jnp.array([
        [1, 2, 3, 4],
        [1, 2, 3, 0],
        [1, 2, 0, 0],
        [1, 0, 0, 0],
    ])
    completion_mask = common.make_completion_mask(completion_ids)
    expected_value = jnp.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 0],
    ])
    np.testing.assert_allclose(completion_mask, expected_value)

  def test_pad_to_length(self):
    x = jnp.ones((2, 4))
    padded_x = common.pad_to_length(x, target_length=5)
    self.assertEqual(padded_x.shape, (5, 4))
    self.assertEqual(jnp.sum(padded_x), 8)
    padded_x = common.pad_to_length(x, target_length=5, axis=-1)
    self.assertEqual(padded_x.shape, (2, 5))
    self.assertEqual(jnp.sum(padded_x), 8)
    padded_x = common.pad_to_length(x, target_length=5, pad_value=1, axis=-1)
    self.assertEqual(padded_x.shape, (2, 5))
    self.assertEqual(jnp.sum(padded_x), 10)
    padded_x = common.pad_to_length(x, target_length=3, axis=-1)
    np.testing.assert_array_equal(padded_x, x)
    padded_x = common.pad_to_length(x, target_length=5, left=True, axis=-1)
    np.testing.assert_array_equal(
        padded_x, jnp.array([[0, 1, 1, 1, 1], [0, 1, 1, 1, 1]])
    )


if __name__ == "__main__":
  absltest.main()
