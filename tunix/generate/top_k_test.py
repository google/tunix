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

"""Tests for top_k_sampler."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from tunix.generate import top_k as top_k_lib

class TopKSamplerTest(parameterized.TestCase):

  def test_sample_top_k_basic(self):
    key = jax.random.PRNGKey(0)
    logits = jnp.array([[0.1, 0.2, 0.3, 0.4, 0.0]])  # Batch size 1, vocab size 5
    top_k_val = 3

    sampled_token = top_k_lib.sample_top_k(logits, key, top_k_val, temperature=1.0)

    self.assertEqual(sampled_token.shape, (1,))
    # With top_k=3, possible tokens are indices 1, 2, 3 (0-indexed)
    # Logits are [0.2, 0.3, 0.4] for these tokens
    self.assertIn(sampled_token[0], [1, 2, 3])

  def test_sample_top_k_temperature_low(self):
    key = jax.random.PRNGKey(1)
    # Last logit is highest
    logits = jnp.array([[1.0, 2.0, 3.0, 10.0]])
    top_k_val = 2
    # Low temperature should make it pick the highest logit among the top_k
    sampled_token = top_k_lib.sample_top_k(logits, key, top_k_val, temperature=0.01)

    self.assertEqual(sampled_token.shape, (1,))
    # Top 2 logits are 3.0 (index 2) and 10.0 (index 3)
    # With low temperature, it should strongly prefer index 3
    self.assertEqual(sampled_token[0], 3)

  def test_sample_top_k_temperature_high(self):
    key = jax.random.PRNGKey(2)
    logits = jnp.array([[0.1, 0.2, 0.3, 0.4, 0.0]])
    top_k_val = 3

    # High temperature should make probabilities more uniform
    sampled_token = top_k_lib.sample_top_k(logits, key, top_k_val, temperature=100.0)

    self.assertEqual(sampled_token.shape, (1,))
    self.assertIn(sampled_token[0], [1, 2, 3])
    # It's hard to assert specific distribution with high temp, just that it's one of the top_k

  def test_sample_top_k_k_equals_one(self):
    key = jax.random.PRNGKey(3)
    logits = jnp.array([[1.0, 2.0, 5.0, 3.0]]) # Max logit is 5.0 at index 2
    top_k_val = 1

    sampled_token = top_k_lib.sample_top_k(logits, key, top_k_val, temperature=1.0)

    self.assertEqual(sampled_token.shape, (1,))
    self.assertEqual(sampled_token[0], 2) # Should always pick the highest logit

  def test_sample_top_k_k_larger_than_vocab(self):
    key = jax.random.PRNGKey(4)
    logits = jnp.array([[0.1, 0.8, 0.3]]) # Vocab size 3
    top_k_val = 5 # Larger than vocab size

    sampled_token = top_k_lib.sample_top_k(logits, key, top_k_val, temperature=1.0)

    self.assertEqual(sampled_token.shape, (1,))
    # Should pick from any of the available tokens [0, 1, 2]
    self.assertIn(sampled_token[0], [0, 1, 2])

  def test_sample_top_k_batch_size_greater_than_one(self):
    key = jax.random.PRNGKey(5)
    logits = jnp.array([
        [0.1, 0.2, 0.3, 0.4, 0.0], # Batch 1
        [0.5, 0.1, 0.2, 0.0, 0.2]  # Batch 2
    ])
    top_k_val = 2

    sampled_tokens = top_k_lib.sample_top_k(logits, key, top_k_val, temperature=1.0)

    self.assertEqual(sampled_tokens.shape, (2,))
    # Batch 1: top_k are indices 2, 3 (logits 0.3, 0.4)
    self.assertIn(sampled_tokens[0], [2, 3])
    # Batch 2: top_k are indices 0, 2 (logits 0.5, 0.2) or 0, 4 (logits 0.5, 0.2)
    # depending on tie-breaking in jax.lax.top_k.
    # For key 5, logits [0.5, 0.1, 0.2, 0.0, 0.2], top_k_indices are [0, 2] or [0, 4]
    # (assuming stable sort for values, but JAX doesn't guarantee this for same values)
    # Probabilities for [0.5, 0.2] (indices 0, 2) are roughly [0.57, 0.43]
    # Probabilities for [0.5, 0.2] (indices 0, 4) are roughly [0.57, 0.43]
    # Let's check against the actual top_k indices for this seed
    # For batch 2, logits are [0.5, 0.1, 0.2, 0.0, 0.2]
    # top_k_logits, top_k_indices = jax.lax.top_k(logits[1], k=2)[1] -> top_k_indices = [0, 2] (or [0,4])
    # For seed 5, the specific outcome might be one or the other. A less brittle check:
    actual_top_indices_batch2 = jax.lax.top_k(logits[1], k=2)[1]
    self.assertIn(sampled_tokens[1], actual_top_indices_batch2)


  @parameterized.named_parameters(
      ("k_is_zero", 0),
      ("k_is_negative", -1),
  )
  def test_sample_top_k_invalid_k(self, k_val):
    key = jax.random.PRNGKey(0)
    logits = jnp.array([[0.1, 0.2, 0.3]])
    with self.assertRaises(ValueError):
      top_k_lib.sample_top_k(logits, key, k_val)

if __name__ == "__main__":
  absltest.main()
